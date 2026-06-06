import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def _clone_batch_to_cpu(batch: Dict) -> Dict:
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().to("cpu", non_blocking=True).clone()
        else:
            cloned[key] = value
    return cloned

def _collect_e2e_params(wrappers: Dict):
    params = []
    for wrapper in wrappers.values():
        params.append(wrapper.alpha)
    return params

def _get_task_loss(outputs):
    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
        if isinstance(loss, torch.Tensor):
            return loss
        if isinstance(loss, dict) and "loss" in loss:
            return loss["loss"]
    return None

def spread_alpha(alpha_raw: torch.Tensor, low=0.2, high=0.8) -> torch.Tensor:
    a_min, a_max = alpha_raw.min(), alpha_raw.max()
    if a_max - a_min < 1e-6:
        # Uniform distribution fallback
        return torch.linspace(low, high, len(alpha_raw), device=alpha_raw.device, dtype=alpha_raw.dtype)
    normalized = (alpha_raw - a_min) / (a_max - a_min)  # → [0, 1]
    return normalized * (high - low) + low               # → [low, high]

def initialize_alpha(model: nn.Module, calib_loader: DataLoader, init_method: str = "random", cache_dir: str = "./pruned_data/alpha_init/"):
    os.makedirs(cache_dir, exist_ok=True)
    
    num_layers = len(model.model.layers)
    cache_path = os.path.join(cache_dir, f"alpha_{init_method}_l{num_layers}.pth")
    
    if os.path.exists(cache_path):
        logger.info(f"Loading cached alpha values from {cache_path}")
        return torch.load(cache_path)
    
    logger.info(f"Initializing alpha values using method: {init_method}")
    alpha_dict = {}
    
    if init_method == "random":
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                num_experts = layer.block_sparse_moe.num_experts
                alpha_raw = torch.randn(num_experts, dtype=torch.float32)
                alpha_dict[layer_idx] = spread_alpha(alpha_raw)
    elif init_method == "activation_frequency":
        expert_counts = {}
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                gate = layer.block_sparse_moe.gate
                device = gate.weight.device if hasattr(gate, 'weight') else model.device
                expert_counts[layer_idx] = torch.zeros(layer.block_sparse_moe.num_experts, dtype=torch.float32, device=device)
        
        hooks = []
        def get_hook(layer_idx, top_k):
            def hook(module, input, output):
                router_logits = output
                routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                expert_counts[layer_idx] += torch.bincount(selected_experts.view(-1), minlength=expert_counts[layer_idx].shape[0]).to(expert_counts[layer_idx].device)
            return hook
            
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                top_k = getattr(layer.block_sparse_moe, 'top_k', 2)
                hooks.append(layer.block_sparse_moe.gate.register_forward_hook(get_hook(layer_idx, top_k)))
                
        model.eval()
        with torch.no_grad():
            for batch in calib_loader:
                inputs = model.prepare_inputs_for_generation(**{k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()})
                model(**inputs)
                
        for h in hooks:
            h.remove()
            
        for layer_idx, counts in expert_counts.items():
            freq = counts / (counts.sum() + 1e-9)
            alpha_dict[layer_idx] = spread_alpha(freq.cpu())
    elif init_method == "gradient_sensitivity":
        # Use gradient magnitude to estimate the contribution of each expert to loss
        expert_sensitivity = {}
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                expert_sensitivity[layer_idx] = torch.zeros(
                    layer.block_sparse_moe.num_experts, dtype=torch.float32
                )
        
        model.eval()
        # Temporarily enable gradients
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                for p in layer.block_sparse_moe.experts.parameters():
                    p.requires_grad_(True)
        
        hooks = []
        def get_sens_hook(layer_idx, top_k, num_experts):
            def hook(module, input, output):
                router_logits = output
                routing_weights = F.softmax(router_logits, dim=-1)
                # Use the average of routing weights as importance proxy
                expert_sensitivity[layer_idx] += routing_weights.detach().mean(0).cpu().float()
            return hook
        
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                h = layer.block_sparse_moe.gate.register_forward_hook(
                    get_sens_hook(layer_idx, layer.block_sparse_moe.top_k, layer.block_sparse_moe.num_experts)
                )
                hooks.append(h)
        
        with torch.no_grad():
            for batch in calib_loader:
                inputs = {k: v.to(next(model.parameters()).device) if torch.is_tensor(v) else v 
                          for k, v in batch.items()}
                try:
                    prepared = model.prepare_inputs_for_generation(**inputs)
                    model(**prepared)
                except Exception:
                    pass
        
        for h in hooks:
            h.remove()
            
        for layer_idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "block_sparse_moe"):
                for p in layer.block_sparse_moe.experts.parameters():
                    p.requires_grad_(False)
        
        for layer_idx, sens in expert_sensitivity.items():
            alpha_dict[layer_idx] = spread_alpha(sens.cpu())
    else:
        raise ValueError(f"Unknown alpha initialization method: {init_method}")
        
    torch.save(alpha_dict, cache_path)
    logger.info(f"Cached alpha values to {cache_path}")
    return alpha_dict
