import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import math
import numpy as np


# ============================================================
# REMAPStatsWrapper: Collect Removal Cost statistics
# ============================================================

class REMAPStatsWrapper(nn.Module):
    def __init__(
        self,
        original_block,
        layer_idx,
        replacement_mode: str = "nth",
        replacement_n: int = 1,
        replacement_average_m: int = 1,
    ):
        super().__init__()
        self.block = original_block
        self.layer_idx = layer_idx
        self.replacement_mode = replacement_mode
        self.replacement_n = max(1, int(replacement_n))
        self.replacement_average_m = max(1, int(replacement_average_m))
        if hasattr(original_block, 'n_routed_experts'):
             self.num_experts = original_block.n_routed_experts
        elif hasattr(original_block, 'config') and hasattr(original_block.config, 'n_routed_experts'):
             self.num_experts = original_block.config.n_routed_experts
        else:
             self.num_experts = original_block.num_experts 

        if hasattr(original_block, 'num_experts_per_tok'):
             self.top_k = original_block.num_experts_per_tok
        elif hasattr(original_block, 'config') and hasattr(original_block.config, 'num_experts_per_tok'):
             self.top_k = original_block.config.num_experts_per_tok
        else:
             self.top_k = original_block.top_k

        if hasattr(original_block, 'hidden_size'):
             self.hidden_dim = original_block.hidden_size
        elif hasattr(original_block, 'config') and hasattr(original_block.config, 'hidden_size'):
             self.hidden_dim = original_block.config.hidden_size
        else:
             self.hidden_dim = original_block.hidden_dim

        if hasattr(original_block, 'moe_intermediate_size'):
             self.ffn_dim = original_block.moe_intermediate_size
        elif hasattr(original_block, 'config') and hasattr(original_block.config, 'moe_intermediate_size'):
             self.ffn_dim = original_block.config.moe_intermediate_size
        else:
             self.ffn_dim = getattr(original_block, 'ffn_dim', 0)
        
        self._init_buffers()

    def _init_buffers(self):
        """Initialize statistics buffer"""
        self.expert_loss_sum = torch.zeros(self.num_experts, dtype=torch.float32, device='cpu')
        self.expert_count = torch.zeros(self.num_experts, dtype=torch.float32, device='cpu')
        # Dual-factor signals accumulation
        self.routing_entropy_sum = 0.0
        self.routing_token_count = 0
        self.moe_ctr_sum = 0.0   # MoE Contribution Ratio accumulation
        self.moe_ctr_count = 0
        # Diversity-Aware: expert output correlation matrix (cosine similarity accumulation)
        self.expert_sim_sum = torch.zeros(self.num_experts, self.num_experts, dtype=torch.float32, device='cpu')
        self.expert_sim_count = torch.zeros(self.num_experts, self.num_experts, dtype=torch.float32, device='cpu')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Collect MoE Contribution Ratio: input norm
        with torch.no_grad():
            inp_norm = hidden_states_flat.float().norm(dim=-1).mean().item()

        gate_output = self.block.gate(hidden_states_flat)
        # 1. Get Top K+fallback routing information
        router_logits = gate_output

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Collect dual-factor signals
        with torch.no_grad():
            # 1) Routing Entropy
            log_probs = torch.log(routing_weights.clamp_min(1e-10))
            token_entropy = -(routing_weights * log_probs).sum(dim=-1)
            self.routing_entropy_sum += token_entropy.sum().item()
            self.routing_token_count += token_entropy.numel()
        
        if self.replacement_mode == "nth":
            num_fallbacks = self.replacement_n
        elif self.replacement_mode == "mean":
            num_fallbacks = self.replacement_average_m
        else:
            raise ValueError(f"Unsupported replacement_mode: {self.replacement_mode}")

        topk_plus_extra = min(self.top_k + num_fallbacks, self.num_experts)
        ranked_weights, ranked_indices = torch.topk(routing_weights, topk_plus_extra, dim=-1)
        
        topk_indices = ranked_indices[:, :self.top_k]
        topk_weights = ranked_weights[:, :self.top_k]
        
        # Renormalized routing weights used in forward
        topk_sum = topk_weights.sum(dim=-1, keepdim=True)
        g_topk_norm = (topk_weights / topk_sum).to(hidden_states.dtype)
        
        # Original unnormalized weights used for Cost calculation (fully aligned with observer)
        g_topk_raw = topk_weights.to(hidden_states.dtype)
        
        fallback_count = max(0, topk_plus_extra - self.top_k)

        # 2. Extract hidden vectors
        num_tokens = hidden_states_flat.shape[0]
        expert_outputs = torch.zeros(
            (self.num_experts, num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        expert_mask = F.one_hot(ranked_indices, num_classes=self.num_experts).sum(dim=1).bool()
        
        for expert_idx in range(self.num_experts):
            sample_idx = torch.where(expert_mask[:, expert_idx])[0]
            if sample_idx.numel() == 0:
                continue
            
            expert_out = self.block.experts[expert_idx](hidden_states_flat[sample_idx])
            expert_outputs[expert_idx, sample_idx] = expert_out
            
        expert_outputs_permuted = expert_outputs.permute(1, 0, 2)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        f_topk = torch.gather(expert_outputs_permuted, 1, expanded_indices)
        fallback_indices = ranked_indices[:, self.top_k : topk_plus_extra]
        fallback_weights = ranked_weights[:, self.top_k : topk_plus_extra].to(hidden_states.dtype)
        fallback_expanded = fallback_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        fallback_outputs = (
            torch.gather(expert_outputs_permuted, 1, fallback_expanded)
            if fallback_count > 0
            else expert_outputs_permuted[:, :0, :]
        )
        replacement_slot = min(self.replacement_n - 1, fallback_count - 1) if fallback_count > 0 else 0
        average_m = min(self.replacement_average_m, fallback_count) if fallback_count > 0 else 0

        # 3. Compute real Y_out for forward (renormalized) and proxy Y_old_proxy for scoring (unnormalized)
        Y_out_real = (g_topk_norm.unsqueeze(-1) * f_topk).sum(dim=1)
        Y_old_proxy = (g_topk_raw.unsqueeze(-1) * f_topk).sum(dim=1)
        
        # 4. Compute Y_new_proxy and Cost 
        cost_sum = torch.zeros(self.num_experts, dtype=torch.float32, device='cpu')
        eps = max(1e-6, torch.finfo(routing_weights.dtype).eps)
        
        for i in range(self.num_experts):
            mask_in_topk = (topk_indices == i).any(dim=-1)
            active_indices = mask_in_topk.nonzero(as_tuple=False).squeeze(-1)
            
            if active_indices.numel() == 0:
                continue
                
            pos_in_k = (topk_indices[active_indices] == i).nonzero(as_tuple=True)[1]
            g_i_raw = g_topk_raw[active_indices, pos_in_k] # Unnormalized weight
            
            # 5. Calculate normalized Y_new_real for cost estimation
            f_topk_active = f_topk[active_indices]
            g_topk_active_raw = g_topk_raw[active_indices]
            
            mask_not_i = (topk_indices[active_indices] != i)
            g_others_raw = g_topk_active_raw * mask_not_i
            f_others = f_topk_active * mask_not_i.unsqueeze(-1)
            
            # Sum of the new top-k weights
            g_new_sum = g_others_raw.sum(dim=1)
            Y_new_unnorm = (g_others_raw.unsqueeze(-1) * f_others).sum(dim=1)
            
            if fallback_count > 0:
                if self.replacement_mode == "nth":
                    replacement_weight = fallback_weights[active_indices, replacement_slot]
                    replacement_output = fallback_outputs[active_indices, replacement_slot, :]
                else:
                    replacement_weight = fallback_weights[active_indices, :average_m].mean(dim=1)
                    replacement_output = fallback_outputs[active_indices, :average_m, :].mean(dim=1)

                g_new_sum = g_new_sum + replacement_weight
                Y_new_unnorm = Y_new_unnorm + replacement_weight.unsqueeze(-1) * replacement_output
            
            # Normalize to match actual MoE output distribution
            scale = (1.0 / g_new_sum.clamp_min(eps)).unsqueeze(-1)
            Y_new_real = Y_new_unnorm * scale
                
            diff = Y_out_real[active_indices] - Y_new_real
            cost_dist = (diff * diff).sum(dim=-1)
            
            cost_sum[i] = cost_dist.sum().to('cpu')
            
            # Update selected counts
            self.expert_count[i] += active_indices.numel()

        # 5. Update Buffer
        self.expert_loss_sum += cost_sum

        # 6. Accumulate expert output correlation matrix (Diversity-Aware)
        with torch.no_grad():
            # expert_outputs: [N_experts, N_tokens, hidden_dim]
            # For each token, compute the cosine similarity between top-k+1 expert outputs
            # Use expert_mask to determine which experts are activated
            # Vectorized accumulation: batch calculation utilizing expert_mask and normalized expert_outputs
            # expert_outputs: [N_experts, N_tokens, hidden_dim]
            # Apply L2 normalization for each expert (independent per token)
            eo_float = expert_outputs.float()  # [E, N, D]
            eo_norm = eo_float.norm(dim=2, keepdim=True).clamp_min(1e-8)  # [E, N, 1]
            eo_normed = eo_float / eo_norm  # [E, N, D]
            
            # expert_mask: [N, E] bool — which tokens activated which experts
            # Zero out non-active tokens (already zero since expert_outputs is initialized to zero)
            # Compute the sum of cosine similarities for each expert pair on co-activated tokens
            # sim_matrix[i,j] = sum_over_shared_tokens( cos(output_i, output_j) )
            # Using matrix multiplication: [E, N, D] x [E, N, D]^T -> but needs per-token alignment
            
            # Efficient scheme: batch dot product on co-activated tokens per expert pair
            active_per_expert = []
            for e_idx in range(self.num_experts):
                active_per_expert.append(expert_mask[:, e_idx])  # [N] bool
            
            for i in range(self.num_experts):
                mask_i = active_per_expert[i]  # [N]
                if mask_i.sum() == 0:
                    continue
                for j in range(i + 1, self.num_experts):
                    mask_j = active_per_expert[j]
                    shared = mask_i & mask_j  # [N] Co-activated tokens
                    n_shared = shared.sum().item()
                    if n_shared == 0:
                        continue
                    # Only compute cosine similarity on co-activated tokens
                    vi = eo_normed[i, shared]  # [n_shared, D]
                    vj = eo_normed[j, shared]  # [n_shared, D]
                    sim_batch = (vi * vj).sum(dim=-1)  # [n_shared]
                    sim_total = sim_batch.sum().item()
                    self.expert_sim_sum[i, j] += sim_total
                    self.expert_sim_sum[j, i] += sim_total
                    self.expert_sim_count[i, j] += n_shared
                    self.expert_sim_count[j, i] += n_shared

        # 6. Return real forward results
        final_hidden_states = Y_out_real.view(batch_size, sequence_length, hidden_dim)

        # Collect MoE Contribution Ratio: output norm
        with torch.no_grad():
            out_norm = final_hidden_states.float().view(-1, hidden_dim).norm(dim=-1).mean().item()
            self.moe_ctr_sum += out_norm / (inp_norm + 1e-10)
            self.moe_ctr_count += 1

        return final_hidden_states, router_logits

    def prune(self, experts_to_drop):
        """Physical pruning."""
        if not experts_to_drop:
            return

        all_experts = set(range(self.num_experts))
        drop_set = set(experts_to_drop)
        experts_to_reserve = sorted(list(all_experts - drop_set))
        
        old_gate = self.block.gate
        
        new_gate = nn.Linear(
            in_features=old_gate.in_features,
            out_features=len(experts_to_reserve),
            bias=False,
            device=old_gate.weight.device,
            dtype=old_gate.weight.dtype
        )
        new_gate.weight.data = old_gate.weight.data[experts_to_reserve]
        self.block.gate = new_gate

        new_experts = nn.ModuleList([self.block.experts[i] for i in experts_to_reserve])
        self.block.experts = new_experts

        self.block.num_experts = len(experts_to_reserve)
        self.num_experts = len(experts_to_reserve)


# ============================================================
# Next-K layer perturbation: top-k -> top-next-k layer sensitivity
# ============================================================

def _model_input_device(model):
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def _batch_to_model_inputs(batch, device):
    if isinstance(batch, dict):
        input_ids = batch["input_ids"].to(device)
        inputs = {"input_ids": input_ids, "use_cache": False}
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device)
        return inputs
    return {"input_ids": batch.to(device), "use_cache": False}





# ============================================================
# REMAP_Pruner: Oneshot Removal Cost Pruning
# ============================================================

class REMAP_Pruner:
    def __init__(self, model, dataloader, args):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        self.save_dir = getattr(args, 'save_path', args.output_path)
        self.log_file = os.path.join(self.save_dir, 'log.txt')
        self.r = args.r
        self.diversity_lambda = getattr(args, 'remap_diversity_lambda', 0.0)
        self.use_diversity = self.diversity_lambda > 0
        self.replacement_mode = getattr(args, 'remap_reroute_replacement_mode', 'nth')
        self.replacement_n = max(1, int(getattr(args, 'remap_reroute_replace_n', 1)))
        self.replacement_average_m = max(1, int(getattr(args, 'remap_reroute_average_m', 1)))
        
        tags = []
        if self.use_diversity: tags.append(f"Diversity(λ={self.diversity_lambda})")
        tags.append(
            f"Replace={self.replacement_mode}"
            f"{self.replacement_n if self.replacement_mode == 'nth' else self.replacement_average_m}"
        )
        tag_str = " | " + " + ".join(tags) if tags else ""
        print(f"REMAP Pruner Init | Keep r={self.r}{tag_str}")
        
    def _wrap_model(self):
        self.wrappers = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'block_sparse_moe'):
                original_moe = layer.block_sparse_moe
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                original_moe = layer.mlp
            else:
                continue 
            if hasattr(original_moe, 'model'): 
                original_moe = original_moe.model
            
            wrapper = REMAPStatsWrapper(
                original_moe,
                layer_idx=i,
                replacement_mode=self.replacement_mode,
                replacement_n=self.replacement_n,
                replacement_average_m=self.replacement_average_m,
            )
            layer.block_sparse_moe = wrapper
            self.wrappers.append(wrapper)
            
    def _unwrap_model(self):
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'block_sparse_moe') and isinstance(layer.block_sparse_moe, REMAPStatsWrapper):
                layer.block_sparse_moe = layer.block_sparse_moe.block
            elif hasattr(layer, 'mlp') and isinstance(layer.mlp, REMAPStatsWrapper):
                layer.mlp = layer.mlp.block
        self.wrappers = []

    def _run_inference(self):
        """Run calibration data once on the current model to collect statistics"""
        self.model.eval()
        with torch.inference_mode():
            for batch in tqdm(self.dataloader, desc="REMAP Inference"):
                if isinstance(batch, dict):
                    if 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(self.model.device)
                    else:
                        input_ids = batch.to(self.model.device)
                else:
                    input_ids = batch.to(self.model.device)
                self.model(input_ids)

    def _iter_moe_blocks(self):
        wrapper_idx = 0
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                yield wrapper_idx, layer_idx, layer, layer.block_sparse_moe
                wrapper_idx += 1
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                yield wrapper_idx, layer_idx, layer, layer.mlp
                wrapper_idx += 1

    @staticmethod
    def _set_layer_moe(layer, block):
        if hasattr(layer, 'block_sparse_moe'):
            layer.block_sparse_moe = block
        elif hasattr(layer, 'mlp'):
            layer.mlp = block

    @staticmethod
    def _snapshot_wrapper_stats(wrappers):
        snapshots = []
        for wrapper in wrappers:
            snapshots.append({
                "expert_loss_sum": wrapper.expert_loss_sum.clone(),
                "expert_count": wrapper.expert_count.clone(),
                "routing_entropy_sum": float(wrapper.routing_entropy_sum),
                "routing_token_count": int(wrapper.routing_token_count),
                "moe_ctr_sum": float(wrapper.moe_ctr_sum),
                "moe_ctr_count": int(wrapper.moe_ctr_count),
                "expert_sim_sum": wrapper.expert_sim_sum.clone(),
                "expert_sim_count": wrapper.expert_sim_count.clone(),
            })
        return snapshots

    @staticmethod
    def _restore_wrapper_stats(wrappers, snapshots):
        for wrapper, snapshot in zip(wrappers, snapshots):
            wrapper.expert_loss_sum.copy_(snapshot["expert_loss_sum"])
            wrapper.expert_count.copy_(snapshot["expert_count"])
    def _get_scores(self, wrapper):
        """Calculate expert importance."""
        loss_sum = wrapper.expert_loss_sum.float()
        count = wrapper.expert_count.float()
        avg_score = loss_sum / count.clamp_min(1e-6)
        return avg_score.numpy()

    def _get_sim_matrix(self, wrapper):
        """Get the normalized expert output correlation matrix"""
        sim_sum = wrapper.expert_sim_sum
        sim_count = wrapper.expert_sim_count
        # Average cosine similarity; pairs without co-activated tokens are set to 0
        sim_matrix = (sim_sum / (sim_count + 1e-6)).numpy()
        return sim_matrix

    def _select_diverse(self, scores, sim_matrix, r_layer, alpha):
        """
        Diversity-Aware expert selection (MMR greedy strategy).

        Each round selects: argmax_i [ alpha * norm_cost(i) - (1-alpha) * max_sim_to_selected(i) ]

        Args:
            scores: expert removal cost (higher = more important)
            sim_matrix: expert output cosine similarity matrix [n, n]
            r_layer: number of experts to keep
            alpha: cost vs diversity weight (1.0=pure cost, 0.0=pure diversity)
        Returns:
            (keep_list, drop_list)
        """
        n = len(scores)
        score_min, score_max = scores.min(), scores.max()
        if score_max - score_min > 1e-10:
            norm_scores = (scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones_like(scores)

        selected = []
        remaining = list(range(n))

        for _ in range(r_layer):
            if not remaining:
                break

            if not selected:
                best = max(remaining, key=lambda i: norm_scores[i])
            else:
                raw_sims = []
                for i in remaining:
                    max_sim = max(sim_matrix[i][j] for j in selected)
                    max_sim = max(0.0, float(max_sim))
                    raw_sims.append(max_sim)

                raw_sims = np.array(raw_sims)
                s_min, s_max = raw_sims.min(), raw_sims.max()
                if s_max - s_min > 1e-10:
                    norm_sims = (raw_sims - s_min) / (s_max - s_min)
                else:
                    norm_sims = raw_sims

                best_combined = -float('inf')
                best = remaining[0]
                for idx, i in enumerate(remaining):
                    combined = alpha * norm_scores[i] - (1 - alpha) * norm_sims[idx]
                    if combined > best_combined:
                        best_combined = combined
                        best = i

            selected.append(best)
            remaining.remove(best)

        return selected, remaining

    def _plan_layer_pruning(self, wrapper, keep_count):
        keep_count = int(max(wrapper.top_k, min(int(keep_count), int(wrapper.num_experts))))
        scores = self._get_scores(wrapper)

        if self.use_diversity:
            # diversity_lambda maps to (1 - alpha): lambda=0.25 -> alpha=0.75
            alpha = 1.0 - self.diversity_lambda
            sim_matrix = self._get_sim_matrix(wrapper)
            experts_to_keep, experts_to_drop = self._select_diverse(
                scores, sim_matrix, keep_count, alpha
            )
        else:
            sorted_indices = np.argsort(scores)[::-1]
            experts_to_keep = list(map(int, sorted_indices[:keep_count]))
            experts_to_drop = list(map(int, sorted_indices[keep_count:]))

        return scores, experts_to_keep, experts_to_drop

    def run(self):
        """Oneshot evaluation + Greedy Top-r pruning"""
        self._wrap_model()
        self._run_inference()
        
        log_lines = ["\n=== REMAP Oneshot Pruning ==="]
        
        print("\n>>> Oneshot: Analysis & Pruning...")
        for i, wrapper in enumerate(self.wrappers):
            r_layer = self.r
            scores, experts_to_keep, experts_to_drop = self._plan_layer_pruning(wrapper, r_layer)
            if self.use_diversity and i in [0, 4, 8]:
                sim_matrix = self._get_sim_matrix(wrapper)
                print(f"  [L{wrapper.layer_idx} Sim_Matrix Avg] : {sim_matrix.mean():.4f} | Max: {sim_matrix.max():.4f}")
            
            line = f"Layer {wrapper.layer_idx}: keep={r_layer} | Scores {scores.round(4).tolist()} | Kept {experts_to_keep} | Dropped {experts_to_drop}"
            print(line)
            log_lines.append(line)
            
            wrapper.prune(experts_to_drop)
        
        self._unwrap_model()
        
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write('\n'.join(log_lines))
            
        print(f"\nREMAP Pruning Completed. Log saved to {self.log_file}")
        return self.model


def remap_moe_pruning(model, dataloader, test_loader, args):
    """
    args.r: Number of experts kept per layer
    """
    pruner = REMAP_Pruner(model, dataloader, args)
    model = pruner.run()
    
    kept_counts = []
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
            kept_counts.append(len(layer.block_sparse_moe.experts))
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            kept_counts.append(len(layer.mlp.experts))
    if kept_counts:
        model.config.num_local_experts = kept_counts[0] if len(set(kept_counts)) == 1 else kept_counts
    
    return model
