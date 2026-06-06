"""
REMAP pruning for Qwen-MoE.

Qwen-MoE stores sparse MoE blocks at ``layer.mlp``.  Each sparse block has a
linear router, routed experts, and a shared expert path:
``shared_expert_gate(x) * shared_expert(x)``.  The REMAP score only ranks routed
experts; shared experts are kept untouched.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _decoder_layers(model: nn.Module):
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Could not find decoder layers on Qwen-MoE model.")


def _is_qwen_moe_layer(layer: nn.Module) -> bool:
    mlp = getattr(layer, "mlp", None)
    return (
        mlp is not None
        and hasattr(mlp, "experts")
        and hasattr(mlp, "gate")
        and hasattr(mlp, "shared_expert")
        and hasattr(mlp, "shared_expert_gate")
    )


def _num_experts(moe_block: nn.Module) -> int:
    experts = moe_block.experts
    if hasattr(experts, "num_experts"):
        return int(experts.num_experts)
    return len(experts)


def _top_k(moe_block: nn.Module) -> int:
    if hasattr(moe_block, "top_k"):
        return int(moe_block.top_k)
    if hasattr(moe_block, "gate") and hasattr(moe_block.gate, "top_k"):
        return int(moe_block.gate.top_k)
    if hasattr(moe_block, "num_experts_per_tok"):
        return int(moe_block.num_experts_per_tok)
    return 1


def _hidden_dim(moe_block: nn.Module) -> int:
    gate = moe_block.gate
    if hasattr(gate, "in_features"):
        return int(gate.in_features)
    if hasattr(gate, "hidden_dim"):
        return int(gate.hidden_dim)
    if hasattr(gate, "weight"):
        return int(gate.weight.shape[1])
    raise AttributeError(f"Could not infer Qwen-MoE hidden dim from gate type {type(gate)}")


def _gate_logits(moe_block: nn.Module, flat_hidden: torch.Tensor) -> torch.Tensor:
    output = moe_block.gate(flat_hidden)
    if isinstance(output, tuple):
        return output[0]
    return output


def _norm_topk_prob(moe_block: nn.Module) -> bool:
    if hasattr(moe_block, "norm_topk_prob"):
        return bool(moe_block.norm_topk_prob)
    if hasattr(moe_block, "gate") and not isinstance(moe_block.gate, nn.Linear):
        return True
    return False


def _expert_forward(moe_block: nn.Module, expert_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
    experts = moe_block.experts
    if isinstance(experts, nn.ModuleList):
        return experts[expert_idx](hidden_states)
    if all(hasattr(experts, name) for name in ("gate_up_proj", "down_proj", "act_fn")):
        gate, up = F.linear(hidden_states, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        return F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_idx])
    if hasattr(experts, "__getitem__"):
        return experts[expert_idx](hidden_states)
    raise TypeError(f"Unsupported Qwen-MoE experts container: {type(experts)}")


class REMAPStatsWrapperQwenMoe(nn.Module):
    """Collect per-routed-expert replacement cost for a Qwen-MoE block."""

    def __init__(
        self,
        original_block: nn.Module,
        layer_idx: int,
        ynew_scale: float = 1.0,
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
        self.num_experts = _num_experts(original_block)
        self.top_k = _top_k(original_block)
        self.hidden_dim = _hidden_dim(original_block)
        self.norm_topk_prob = _norm_topk_prob(original_block)
        self.ynew_scale = float(ynew_scale)
        self._init_buffers()

    def _init_buffers(self):
        self.expert_loss_sum = torch.zeros(self.num_experts, dtype=torch.float32, device="cpu")
        self.expert_count = torch.zeros(self.num_experts, dtype=torch.float32, device="cpu")
        self.routing_entropy_sum = 0.0
        self.routing_token_count = 0
        self.moe_ctr_sum = 0.0
        self.moe_ctr_count = 0
        self.expert_sim_sum = torch.zeros(
            self.num_experts, self.num_experts, dtype=torch.float32, device="cpu"
        )
        self.expert_sim_count = torch.zeros(
            self.num_experts, self.num_experts, dtype=torch.float32, device="cpu"
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_dim)

        with torch.no_grad():
            inp_norm = flat_hidden.float().norm(dim=-1).mean().item()

        router_logits = _gate_logits(self.block, flat_hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        with torch.no_grad():
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
        topk_indices = ranked_indices[:, : self.top_k]
        topk_weights = ranked_weights[:, : self.top_k]
        topk_raw = topk_weights.to(hidden_states.dtype)

        if self.norm_topk_prob:
            topk_forward = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)).to(
                hidden_states.dtype
            )
        else:
            topk_forward = topk_raw

        fallback_count = max(0, topk_plus_extra - self.top_k)

        num_tokens = flat_hidden.shape[0]
        slot_outputs = torch.zeros(
            (num_tokens, topk_plus_extra, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(ranked_indices == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = _expert_forward(self.block, expert_idx, flat_hidden[token_idx])
            slot_outputs[token_idx, slot_idx] = expert_out

        f_topk = slot_outputs[:, : self.top_k, :]
        fallback_outputs = slot_outputs[:, self.top_k : topk_plus_extra, :]
        fallback_weights = ranked_weights[:, self.top_k : topk_plus_extra].to(hidden_states.dtype)
        replacement_slot = min(self.replacement_n - 1, fallback_count - 1) if fallback_count > 0 else 0
        average_m = min(self.replacement_average_m, fallback_count) if fallback_count > 0 else 0

        routed_out = (topk_forward.unsqueeze(-1) * f_topk).sum(dim=1)
        cost_sum = torch.zeros(self.num_experts, dtype=torch.float32, device="cpu")
        eps = max(1e-6, torch.finfo(routing_weights.dtype).eps)

        for expert_idx in range(self.num_experts):
            active = (topk_indices == expert_idx).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue

            active_topk_indices = topk_indices[active]
            active_topk_raw = topk_raw[active]
            active_f_topk = f_topk[active]
            keep_mask = active_topk_indices != expert_idx

            new_weight_sum = (active_topk_raw * keep_mask).sum(dim=1)
            new_routed = (active_topk_raw.unsqueeze(-1) * active_f_topk * keep_mask.unsqueeze(-1)).sum(dim=1)

            if fallback_count > 0:
                if self.replacement_mode == "nth":
                    replacement_weight = fallback_weights[active, replacement_slot]
                    replacement_output = fallback_outputs[active, replacement_slot, :]
                else:
                    replacement_weight = fallback_weights[active, :average_m].mean(dim=1)
                    replacement_output = fallback_outputs[active, :average_m, :].mean(dim=1)
                new_weight_sum = new_weight_sum + replacement_weight
                new_routed = new_routed + replacement_weight.unsqueeze(-1) * replacement_output

            if self.norm_topk_prob:
                new_routed = new_routed / new_weight_sum.clamp_min(eps).unsqueeze(-1)
            else:
                pos_in_k = (active_topk_indices == expert_idx).nonzero(as_tuple=True)[1]
                removed_weight = active_topk_raw[torch.arange(active.numel(), device=active.device), pos_in_k]
                new_routed = new_routed / (1.0 - removed_weight).clamp_min(eps).unsqueeze(-1)

            new_routed = self.ynew_scale * new_routed

            diff = routed_out[active] - new_routed
            cost_sum[expert_idx] = (diff.float() * diff.float()).sum(dim=-1).sum().to("cpu")
            self.expert_count[expert_idx] += active.numel()

        self.expert_loss_sum += cost_sum

        with torch.no_grad():
            slot_float = slot_outputs.float()
            slot_norm = slot_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            slot_normed = slot_float / slot_norm

            for left_slot in range(topk_plus_extra):
                left_expert = ranked_indices[:, left_slot]
                left_output = slot_normed[:, left_slot, :]
                for right_slot in range(left_slot + 1, topk_plus_extra):
                    right_expert = ranked_indices[:, right_slot]
                    sim = (left_output * slot_normed[:, right_slot, :]).sum(dim=-1)
                    sim = sim.detach().cpu()
                    denom = torch.ones_like(sim, dtype=torch.float32)

                    left_cpu = left_expert.detach().cpu()
                    right_cpu = right_expert.detach().cpu()

                    self.expert_sim_sum.index_put_((left_cpu, right_cpu), sim, accumulate=True)
                    self.expert_sim_sum.index_put_((right_cpu, left_cpu), sim, accumulate=True)
                    self.expert_sim_count.index_put_((left_cpu, right_cpu), denom, accumulate=True)
                    self.expert_sim_count.index_put_((right_cpu, left_cpu), denom, accumulate=True)

        shared_out = self.block.shared_expert(flat_hidden)
        shared_gate = torch.sigmoid(self.block.shared_expert_gate(flat_hidden))
        final_hidden_states = routed_out + shared_gate * shared_out
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        with torch.no_grad():
            routed_norm = routed_out.float().norm(dim=-1).mean().item()
            self.moe_ctr_sum += routed_norm / (inp_norm + 1e-10)
            self.moe_ctr_count += 1

        return final_hidden_states, router_logits

    def prune(self, experts_to_drop):
        if not experts_to_drop:
            return

        retained = sorted(set(range(self.num_experts)) - set(map(int, experts_to_drop)))
        if not retained:
            raise ValueError(f"Layer {self.layer_idx}: cannot prune all routed experts.")

        old_gate = self.block.gate
        if isinstance(old_gate, nn.Linear):
            new_gate = nn.Linear(
                old_gate.in_features,
                len(retained),
                bias=old_gate.bias is not None,
                device=old_gate.weight.device,
                dtype=old_gate.weight.dtype,
            )
            new_gate.weight.data.copy_(old_gate.weight.data[retained])
            if old_gate.bias is not None:
                new_gate.bias.data.copy_(old_gate.bias.data[retained])
            self.block.gate = new_gate
        elif hasattr(old_gate, "weight"):
            old_gate.weight = nn.Parameter(old_gate.weight.data[retained].clone())
            if hasattr(old_gate, "num_experts"):
                old_gate.num_experts = len(retained)
        else:
            raise TypeError(f"Unsupported Qwen-MoE gate type: {type(old_gate)}")

        if isinstance(self.block.experts, nn.ModuleList):
            self.block.experts = nn.ModuleList([self.block.experts[idx] for idx in retained])
        elif all(hasattr(self.block.experts, name) for name in ("gate_up_proj", "down_proj")):
            self.block.experts.gate_up_proj = nn.Parameter(self.block.experts.gate_up_proj.data[retained].clone())
            self.block.experts.down_proj = nn.Parameter(self.block.experts.down_proj.data[retained].clone())
            if hasattr(self.block.experts, "num_experts"):
                self.block.experts.num_experts = len(retained)
        else:
            raise TypeError(f"Unsupported Qwen-MoE experts container: {type(self.block.experts)}")

        self.block.num_experts = len(retained)
        if hasattr(self.block, "top_k"):
            self.block.top_k = min(int(self.block.top_k), len(retained))
        if hasattr(self.block.gate, "top_k"):
            self.block.gate.top_k = min(int(self.block.gate.top_k), len(retained))
        self.num_experts = len(retained)
        self.top_k = _top_k(self.block)





class REMAP_Pruner_QwenMoe:
    def __init__(self, model, dataloader, args):
        self.model = model
        self.dataloader = dataloader
        self.args = args
        self.save_dir = getattr(args, "save_path", args.output_path)
        self.log_file = os.path.join(self.save_dir, "log.txt")
        self.r = int(args.r)
        self.sp_ratio = float(getattr(args, "sp_ratio", 0.0))
        self.diversity_lambda = getattr(args, "remap_diversity_lambda", 0.0)
        self.use_diversity = self.diversity_lambda > 0
        self.replacement_mode = getattr(args, "remap_reroute_replacement_mode", "nth")
        self.replacement_n = max(1, int(getattr(args, "remap_reroute_replace_n", 1)))
        self.replacement_average_m = max(1, int(getattr(args, "remap_reroute_average_m", 1)))
        self.qwen_ynew_scale = 1.0 / max(1.0 - self.sp_ratio, 1e-6)
        tags = [f"uniform r={self.r}"]
        tags.append(
            f"replace={self.replacement_mode}"
            f"{self.replacement_n if self.replacement_mode == 'nth' else self.replacement_average_m}"
        )
        if self.use_diversity:
            tags.append(f"diversity λ={self.diversity_lambda}")
        mode = " | ".join(tags)
        print(f"REMAP Qwen-MoE Pruner Init | {mode}")

    def _wrap_model(self):
        self.wrappers = []
        for layer_idx, layer in enumerate(_decoder_layers(self.model)):
            if not _is_qwen_moe_layer(layer):
                continue
            wrapper = REMAPStatsWrapperQwenMoe(
                layer.mlp,
                layer_idx,
                ynew_scale=self.qwen_ynew_scale,
                replacement_mode=self.replacement_mode,
                replacement_n=self.replacement_n,
                replacement_average_m=self.replacement_average_m,
            )
            layer.mlp = wrapper
            if hasattr(layer, "block_sparse_moe"):
                layer.block_sparse_moe = wrapper
            self.wrappers.append(wrapper)
        print(f"  Wrapped {len(self.wrappers)} Qwen-MoE sparse layers")

    def _unwrap_model(self):
        for layer in _decoder_layers(self.model):
            if isinstance(getattr(layer, "mlp", None), REMAPStatsWrapperQwenMoe):
                layer.mlp = layer.mlp.block
                if hasattr(layer, "block_sparse_moe"):
                    layer.block_sparse_moe = layer.mlp
        self.wrappers = []

    def _run_inference(self):
        self.model.eval()
        with torch.inference_mode():
            for batch in tqdm(self.dataloader, desc="REMAP Qwen-MoE Inference"):
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(self.model.device)
                else:
                    input_ids = batch.to(self.model.device)
                self.model(input_ids)

    def _get_scores(self, wrapper):
        loss_sum = wrapper.expert_loss_sum.float()
        count = wrapper.expert_count.float()
        avg_score = loss_sum / count.clamp_min(1e-6)
        return avg_score.numpy()

    @staticmethod
    def _get_sim_matrix(wrapper):
        return (wrapper.expert_sim_sum / (wrapper.expert_sim_count + 1e-6)).numpy()

    def run(self):
        self._wrap_model()
        self._run_inference()

        log_lines = [
            f"\n=== REMAP Qwen-MoE Uniform Pruning "
            f"(r={self.r}, ynew_scale={self.qwen_ynew_scale:.6g}) ==="
        ]
        layer_budgets = {wrapper.layer_idx: self.r for wrapper in self.wrappers}

        print("\n>>> Qwen-MoE pruning by Removal Cost...")
        for wrapper in self.wrappers:
            layer_idx = wrapper.layer_idx
            num_experts = wrapper.num_experts
            keep = max(1, min(int(layer_budgets[layer_idx]), num_experts))
            scores = self._get_scores(wrapper)
            sorted_indices = np.argsort(scores)[::-1]
            keep_indices = list(map(int, sorted_indices[:keep]))
            drop_indices = list(map(int, sorted_indices[keep:]))

            scores_str = " ".join(f"{score:.2f}" for score in scores)
            print(f"  L{layer_idx:2d} ({keep}/{num_experts}): [{scores_str}]")
            log_lines.append(
                f"Layer {layer_idx}: Keep {keep}/{num_experts} | Scores [{scores_str}] | "
                f"Kept {keep_indices} | Dropped {drop_indices}"
            )
            wrapper.prune(drop_indices)

        self._unwrap_model()

        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write("\n".join(log_lines))

        print(f"\nREMAP Qwen-MoE Pruning Completed. Log saved to {self.log_file}")
        return self.model


def remap_moe_pruning_qwen_moe(model, dataloader, test_loader, args):
    pruner = REMAP_Pruner_QwenMoe(model, dataloader, args)
    model = pruner.run()

    counts = [
        _num_experts(layer.mlp)
        for layer in _decoder_layers(model)
        if _is_qwen_moe_layer(layer)
    ]
    if counts and len(set(counts)) == 1:
        if hasattr(model.config, "num_experts"):
            model.config.num_experts = counts[0]
        if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "num_experts"):
            model.config.text_config.num_experts = counts[0]
        if hasattr(model, "num_experts"):
            model.num_experts = counts[0]

    return model
