import torch
import torch.nn.functional as F
import itertools as I
from data import CacheDataset
from tqdm import tqdm
from argparse import Namespace
import logging
from typing import Optional
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from torch.utils.data import DataLoader
from transformers.models.mixtral import modeling_mixtral,MixtralForCausalLM # MixtralForCausalLM#





logger = logging.getLogger(__name__)


def naee_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    layer_devices = [
        next(layer.block_sparse_moe.parameters()).device
        for layer in model.model.layers
    ]

    logger.info('Keeping accelerate device_map placement for pruning...')
    torch.cuda.empty_cache()

    global_loss_history = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        b = layer.block_sparse_moe
        if not hasattr(b, 'cache_space'):
            continue
        b.to(layer_devices[l])
        loss_history = b.enumerate()
        global_loss_history[l] = loss_history
        b.prune()
        b.to(layer_devices[l])
        torch.cuda.empty_cache()

    logger.info('Merging & saving...')
    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )


def progressive_pruning(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM) or isinstance(model, modeling_mixtral.MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate(batch_size=getattr(args, 'batch_size', 4))
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = args.r
    model.config.num_local_experts = args.r

    return model, (global_loss_history, )


class PrunableMixtralSparseMoeBlockWrapper(torch.nn.Module):
    def __init__(self, model,
                 r: Optional[int] = None,
                 ):
        super().__init__()
        if isinstance(model, MixtralSparseMoeBlock) or isinstance(model, modeling_mixtral.MixtralSparseMoeBlock):
            self.model = model
        else:   
            self.model = model.model
        self.r = r

        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.experts_frequency = torch.zeros(8).float()
        self.prune_file='./logs/prune_experts.txt'

    # Forward uses topk
    def forward(self, hidden_states: torch.Tensor, Cache=True) -> torch.Tensor:
        """ """

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # try:
            #with torch.cuda.amp.autocast():

        router_logits = self.model.gate(hidden_states)



        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.model.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0) #selected_experts:[16384,2], expert_mask:[8,2,16384]

        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]

            idx, top_x = torch.where(expert_mask[expert_idx]) #output index that the element is not zero

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


            current_hidden_states = expert_layer(current_state)*routing_weights[top_x_list, idx_list, None]



            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))



        
        if self.experts_to_drop is None and Cache:
            self.cache_space.append(alpha=(router_logits if self.cache_logits else None), X=(hidden_states if self.cache_X else None), Z=(
            final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

    def enumerate(self, batch_size=4): 
        # Adjust batch_size based on GPU memory. Set to 4 or 2 if OOM occurs.
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        
        # 1. Pre-generate all dropped expert combinations and initialize loss dictionary.
        combinations = list(I.combinations(range(self.model.num_experts), self.model.num_experts - self.r))
        loss_history = {dropped: 0.0 for dropped in combinations}
        
        device = self.model.gate.weight.data.device

        with torch.inference_mode():
            total_samples = len(self.cache_space.Xs)
            
            # 2. Outer loop: Iterate over the dataset in batches.
            for i in range(0, total_samples, batch_size):
                # Get slices for current batch
                batch_Xs = self.cache_space.Xs[i : i + batch_size]
                batch_Zs = self.cache_space.Zs[i : i + batch_size]

                # Stack and transfer data to GPU at once to avoid repeated CPU->GPU (H2D) transfers.
                X_tensor = torch.stack(batch_Xs).to(device=device, non_blocking=True)
                # Use FP32 to avoid double precision performance bottlenecks on L40.
                Z_tensor = torch.stack(batch_Zs).to(device=device, dtype=torch.float32, non_blocking=True)

                # 3. Inner loop: Iterate over all expert combinations in GPU memory.
                for dropped in combinations:
                    self.experts_to_drop = dropped

                    # Batch forward computation
                    Z_e_tensor, _ = self.forward(X_tensor)

                    # Compute difference
                    diff = Z_tensor - Z_e_tensor.to(torch.float32)

                    # Compute Frobenius norm for each sample in the batch and accumulate
                    norms = torch.linalg.vector_norm(diff, dim=(1, 2))
                    loss_history[dropped] += norms.sum().item()

                # Clean up GPU memory fragments for the current batch to prevent OOM
                del X_tensor, Z_tensor, diff, Z_e_tensor, norms
        
        # Find the combination with the minimum loss
        self.experts_to_drop = min(loss_history, key=loss_history.get)
        
        # Critical fix: Return loss_history to be recorded by global_loss_history
        return loss_history
        
    @torch.no_grad()
    def prune(self):
        assert self.experts_to_drop is not None
        assert len(self.experts_to_drop) == self.model.num_experts - self.r
        del self.cache_space
        self.cache_X = False
        self.cache_Z = False

        experts_to_reserve = sorted(
            set(range(self.model.num_experts)) - set(self.experts_to_drop))

        gate_weight = self.model.gate.weight
        gate_new = torch.nn.Linear(in_features=self.model.gate.in_features,
                                   out_features=self.r, bias=False, device=gate_weight.device, dtype=gate_weight.dtype)
        gate_new.weight.data = self.model.gate.weight.data[list(
            experts_to_reserve)]
        self.model.gate = gate_new

        self.model.experts = torch.nn.ModuleList(
            [self.model.experts[i] for i in experts_to_reserve])
        self.model.num_experts = self.r
