from method.naee.naee_mixtral import PrunableMixtralSparseMoeBlockWrapper
from tqdm import tqdm
from argparse import Namespace
import logging
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import os
import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.mixtral import modeling_mixtral # MixtralForCausalLM#

from model.modeling_mixtral import MixtralForCausalLM
from method.remap.remap_mixtral import REMAPStatsWrapper




logger = logging.getLogger(__name__)


def _zero_similarity_matrices(model: MixtralForCausalLM):
    matrices = []
    for layer in model.model.layers:
        num_experts = layer.block_sparse_moe.num_experts
        matrices.append(np.zeros((num_experts, num_experts), dtype=np.float32))
    return matrices


def _wrap_dynamic_skip_for_inference(
    model: MixtralForCausalLM,
    similarity_matrices=None,
    similarity_gamma: float = 0.0,
):
    if not hasattr(model.config, 'betas'):
        logger.info('Dynamic skipping betas are missing; skip wrapper is not enabled.')
        return model

    if similarity_matrices is None:
        similarity_matrices = _zero_similarity_matrices(model)

    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock) or isinstance(layer.block_sparse_moe, modeling_mixtral.MixtralSparseMoeBlock):
            layer.block_sparse_moe = DynamicSkippingMixtralSparseMoeBlockWrapper(
                layer.block_sparse_moe,
                float(model.config.betas[str(i)]),
                similarity_matrices[i],
                similarity_gamma=similarity_gamma,
            )
    return model


def collect_remap_expert_similarity(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    wrappers = []
    for layer_idx, layer in enumerate(model.model.layers):
        wrapper = REMAPStatsWrapper(layer.block_sparse_moe, layer_idx=layer_idx)
        layer.block_sparse_moe = wrapper
        wrappers.append(wrapper)

    with torch.inference_mode():
        for batch in tqdm(calib_loader, desc='Collecting REMAP expert similarity...'):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    similarity_matrices = []
    for wrapper in wrappers:
        sim = wrapper.expert_sim_sum / (wrapper.expert_sim_count + 1e-6)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        similarity_matrices.append(sim.numpy().astype(np.float32))

    for layer in model.model.layers:
        layer.block_sparse_moe = layer.block_sparse_moe.block

    save_dir = getattr(args, 'save_path', args.output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, 'remap_expert_similarity_matrices.npz'),
            **{f'layer_{i}': mat for i, mat in enumerate(similarity_matrices)}
        )
        with open(os.path.join(save_dir, 'remap_expert_similarity_summary.txt'), 'w') as f:
            f.write(f"{'Layer':<6} | {'Mean Sim':<10} | {'P95 Sim':<10} | {'Observed Pairs':<14}\n")
            f.write("-" * 52 + "\n")
            for i, mat in enumerate(similarity_matrices):
                mask = ~np.eye(mat.shape[0], dtype=bool)
                vals = mat[mask]
                observed = vals[np.abs(vals) > 1e-12]
                mean = float(np.mean(observed)) if observed.size else 0.0
                p95 = float(np.percentile(observed, 95)) if observed.size else 0.0
                f.write(f"{i:<6d} | {mean:<10.4f} | {p95:<10.4f} | {observed.size:<14d}\n")

    return similarity_matrices


def dynamic_skipping(
    model: MixtralForCausalLM,
    calib_loader: DataLoader,
    args: Namespace,
    enable_inference_wrapper: bool = True,
    similarity_matrices=None,
    similarity_gamma: float = 0.0,
):
    
    assert isinstance(
        model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe)
        layer.block_sparse_moe.cache_logits = True
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    res_median = {}
    res_mean = {}
    
    # [Added] Container for plotting and statistics
    save_dir = getattr(args, 'save_path', args.save_path)
    all_layer_skip_rates = []     # (num_layers, num_experts)
    all_layer_total_stats = []    # (layer_idx, skipped, total, rate)

    for layer_idx in range(len(model.model.layers)):
        b = model.model.layers[layer_idx].block_sparse_moe
        b.cache_space.prepare_for_loader()
        dataloader = torch.utils.data.DataLoader(
            b.cache_space,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info(len(dataloader))

        ana_data = [] # [Modified] Store tuples list[(ratio, top2_expert_id)]
        
        for i, (router_logits, X, Z) in enumerate(dataloader):
            routing_weights = F.softmax(
                router_logits, dim=-1, dtype=torch.float).view(-1, b.model.num_experts)
            for j in range(len(routing_weights)):
                sorted_weights, sort_indices = torch.sort(
                    routing_weights[j], descending=True)
                
                # [Modified] Simultaneously record ratio and top2 expert IDs
                r = float(sorted_weights[1] / sorted_weights[0])
                eid = int(sort_indices[1])
                ana_data.append((r, eid))

        # Extract pure ratio to calculate statistics
        ratio_list = [x[0] for x in ana_data]
        median = np.median(ratio_list)
        mean = np.mean(ratio_list)
        logger.info(f'layer {layer_idx} | mean: {mean:.4f}, median: {median:.4f}')
        res_median[str(layer_idx)] = median
        res_mean[str(layer_idx)] = mean
        
        # === [Added] Calculate skipping rates ===
        num_experts = b.model.num_experts
        top2_counts = np.zeros(num_experts) # Total count of this expert acting as Top2
        skip_counts = np.zeros(num_experts) # Count of this expert acting as Top2 and being skipped
        
        for r, eid in ana_data:
            top2_counts[eid] += 1
            # The decision logic here matches inference: w2 < beta * w1 => w2/w1 < beta
            if r < median:
                skip_counts[eid] += 1
        
        # Calculate skip rate for each expert (Skipped / Total_as_Top2)
        expert_skip_rate = np.divide(skip_counts, top2_counts, out=np.zeros_like(skip_counts), where=top2_counts!=0)
        all_layer_skip_rates.append(expert_skip_rate)
        
        # Statistics for the total skipped count of this layer
        total_skipped = np.sum(skip_counts)
        total_sample = len(ana_data)
        layer_total_rate = total_skipped / (total_sample + 1e-6)
        all_layer_total_stats.append((layer_idx, total_skipped, total_sample, layer_total_rate))
        
        logger.info(f"Layer {layer_idx} Stats | Total Skipped: {int(total_skipped)}/{total_sample} ({layer_total_rate:.2%})")
        # =======================

   
    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.config.betas = res_median
    
    # === [Added] Plotting and saving statistics ===
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Save summary statistics text
        with open(os.path.join(save_dir, 'skipping_stats_summary.txt'), 'w') as f:
            f.write(f"{'Layer':<6} | {'Skipped Count':<15} | {'Total Count':<15} | {'Skip Rate':<10}\n")
            f.write("-" * 55 + "\n")
            for item in all_layer_total_stats:
                f.write(f"{item[0]:<6d} | {int(item[1]):<15d} | {item[2]:<15d} | {item[3]:.2%}\n")
        
        # 2. Plot heatmap (Layers x Experts)
        try:
            plt.figure(figsize=(10, 14))
            data_matrix = np.array(all_layer_skip_rates) # Shape: (32, 8)
            
            plt.imshow(data_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)
            cbar = plt.colorbar()
            cbar.set_label('Skip Rate (Skipped / Total as Top2)', rotation=270, labelpad=15)
            
            plt.title('Dynamic Skipping Rate per Expert (Threshold = Median)', fontsize=14)
            plt.xlabel('Expert ID', fontsize=12)
            plt.ylabel('Layer ID', fontsize=12)
            plt.xticks(np.arange(8))
            plt.yticks(np.arange(len(all_layer_skip_rates)))
            
            # Fill values in cells
            for i in range(data_matrix.shape[0]):
                for j in range(data_matrix.shape[1]):
                    val = data_matrix[i, j]
                    # Color of text (white or black) depends on the background depth
                    color = 'white' if val > 0.5 else 'black'
                    if val > 0.01: # Do not display values that are too small
                        plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

            plt.tight_layout()
            plot_path = os.path.join(save_dir, 'expert_skipping_rate.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Skipping analysis plot saved to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to plot skipping stats: {e}")

    if enable_inference_wrapper:
        model = _wrap_dynamic_skip_for_inference(
            model,
            similarity_matrices=similarity_matrices,
            similarity_gamma=similarity_gamma,
        )

    return model, (res_median, res_mean)


def dynamic_similarity_skipping(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    model, beta_stats = dynamic_skipping(
        model,
        calib_loader,
        args,
        enable_inference_wrapper=False,
    )
    similarity_matrices = collect_remap_expert_similarity(model, calib_loader, args)
    similarity_gamma = float(getattr(args, 'dynamic_similarity_gamma', 1.0))
    model = _wrap_dynamic_skip_for_inference(
        model,
        similarity_matrices=similarity_matrices,
        similarity_gamma=similarity_gamma,
    )
    logger.info(f'Enabled REMAP-similarity-aware dynamic skipping with gamma={similarity_gamma}.')
    return model, beta_stats


class DynamicSkippingMixtralSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model: MixtralSparseMoeBlock, beta: float, differences: list, similarity_gamma: float = 0.0):
        super().__init__()
        assert isinstance(model, MixtralSparseMoeBlock) or isinstance(model, modeling_mixtral.MixtralSparseMoeBlock)
        assert model.top_k == 2
        self.hidden_dim = model.hidden_dim
        self.ffn_dim = model.ffn_dim
        self.num_experts = model.num_experts
        self.top_k = model.top_k
        self.gate = model.gate
        self.experts = model.experts
        
        self.differences = torch.tensor(differences, dtype=torch.float32)
        self.beta = beta
        self.similarity_gamma = float(similarity_gamma)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)

        
        self.differences = self.differences.to(hidden_states.device)
        similarity = self.differences[selected_experts[:,0], selected_experts[:,1]].float().clamp(min=0.0)
        effective_beta = self.beta * (1.0 + self.similarity_gamma * similarity)
       
       
        #mask_top1 = (routing_weights[:, 1] < self.beta* unsimialrity *routing_weights[:, 0]) #
        mask_top1 = (routing_weights[:, 1] < effective_beta*routing_weights[:, 0]) #
        routing_weights[mask_top1, 1] = 0
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # (batch * sequence_length, self.top_k, n_experts)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts)
        
        expert_mask[mask_top1, 1, :] = 0
        expert_mask = expert_mask.permute(2, 1, 0)
        #print("dynamic skipping")
        # Loop over all available experts in the model and perform the computation on each expert

        for expert_idx in range(self.num_experts):
            
            expert_layer = self.experts[expert_idx]
            
            idx, top_x = torch.where(expert_mask[expert_idx])
            

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))


        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
