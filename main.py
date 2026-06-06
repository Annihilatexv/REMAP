import os
import os.path as osp
import sys
import shutil
import json
import logging
import argparse
import torch
from datetime import datetime
from argparse import Namespace

# Third party
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers import AutoModelForCausalLM


def ensure_transformers_lm_eval_compat():
    try:
        from transformers.utils import import_utils

        if not hasattr(import_utils, "is_torch_fx_available"):
            def is_torch_fx_available():
                try:
                    import torch.fx  # noqa: F401
                    return True
                except Exception:
                    return False

            import_utils.is_torch_fx_available = is_torch_fx_available
    except Exception:
        pass

    try:
        from transformers import DynamicCache

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_usable_length"):
            def get_usable_length(self, new_seq_length=None, layer_idx=0):
                try:
                    return self.get_seq_length(layer_idx)
                except TypeError:
                    return self.get_seq_length()

            DynamicCache.get_usable_length = get_usable_length
    except Exception:
        pass

    try:
        from transformers import AutoModelForImageTextToText
    except Exception:
        AutoModelForImageTextToText = None

    if AutoModelForImageTextToText is not None:
        setattr(transformers, "AutoModelForVision2Seq", AutoModelForImageTextToText)

        try:
            import lm_eval.models.huggingface as lm_eval_hf

            setattr(lm_eval_hf.transformers, "AutoModelForVision2Seq", AutoModelForImageTextToText)
        except Exception:
            pass

    try:
        from transformers.models.mixtral import modeling_mixtral
    except Exception:
        return
    if not hasattr(modeling_mixtral, "MixtralBlockSparseTop2MLP"):
        modeling_mixtral.MixtralBlockSparseTop2MLP = getattr(
            modeling_mixtral,
            "MixtralSparseMoeBlock",
            object,
        )


ensure_transformers_lm_eval_compat()

# Project imports
from method import METHODS

DATASETS = {"c4": None, "math": None}

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def load_lm_eval():
    import lm_eval

    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import handle_non_serializable, make_table
    except ImportError:
        import numpy as np
        from lm_eval.utils import make_table
        from lm_eval.tasks import initialize_tasks

        initialize_tasks()

        def handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            if isinstance(o, set):
                return list(o)
            return str(o)

        from lm_eval.models.huggingface import HFLM

    return lm_eval, HFLM, handle_non_serializable, make_table


def get_decoder_layers(model):
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "layers"):
            return inner.layers
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
            return inner.language_model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Could not find decoder layers on the loaded model.")


def get_expert_count(experts):
    if hasattr(experts, "num_experts"):
        return int(experts.num_experts)
    return len(experts)


class Tee:
    """Standard dual-output stream for logging."""
    def __init__(self, stream, file): 
        self.stream, self.file = stream, file
    def write(self, data): 
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self): 
        self.stream.flush()
        self.file.flush()

def parse_args():
    parser = argparse.ArgumentParser()
    # General args
    parser.add_argument('--method', type=str, default="remap_pruning",
                        choices=list(METHODS.keys()),
                        help='Supported methods: ' + ' '.join(list(METHODS.keys())))
    parser.add_argument('--sp_ratio', type=float, default=0.0,
                        help='Sparsity ratio.')
    parser.add_argument('--calib_set', type=str, default='c4',
                        choices=list(DATASETS.keys()),
                        help=' '.join(['Supported calibration datasets:'] + list(DATASETS.keys())))
    parser.add_argument('--model_path', type=str, default="Mixtral-8x7B-v0.1",
                        help='Path to model to prune')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Output path (pruned model, pruning results, etc.)')
    parser.add_argument('--max_block_size', type=int, default=2048,
                        help='Maximal sequence length of each sample in calibration set')
    parser.add_argument('--n_blocks_for_stat', type=int, default=64,
                        help='Number of sequences in calibration set. If set to 0 or negative, the whole dataset will be used')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for calibration/training blocks')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproduction')
    parser.add_argument('--use_flash_attention_2', action='store_true',
                        help='If set, Flash Attention 2 will be used')
    parser.add_argument('--dynamic_similarity_gamma', type=float, default=1.0,
                        help='For dynamic_similarity_skipping: scale the REMAP expert-output similarity adjustment. '
                             '0.0 reduces to vanilla dynamic skipping.')
    parser.add_argument('--dynamic_only_stats', action='store_true',
                        help='For dynamic_skipping: only collect thresholds/statistics and do not enable inference-time skipping.')

    parser.add_argument('--eval_tasks', type=str, default="",
                        help='the evaluation tasks that are splited with comma, e.g.,mmlu,rte')
    parser.add_argument('--num_fewshot', type=int, default=0,
                        help='Number of few-shot examples for evaluation tasks')                 
    parser.add_argument('--eval_samples', type=int, default=None,
                        help='Number of samples to evaluate on each task. Default to use all samples.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size for lm-eval, defaults to "auto" if not set')

    parser.add_argument('--save_model', action='store_true',
                        help='If set, the pruned model will be saved.') 
    parser.add_argument('--custom_name', type=str, default='',
                        help='Customized name for the run (appended after timestamp).')                   

    # REMAP Pruning Arguments
    parser.add_argument('--remap_diversity_lambda', type=float, default=0.0,
                        help='Diversity regularization strength (λ in the paper). 0.0 disables diversity. Default: 0.0')
    parser.add_argument('--remap_reroute_replacement_mode', type=str, default='nth',
                        choices=['nth', 'mean'],
                        help='Replacement expert source for REMAP removal-cost reroute. nth uses the '
                             'top-k+N fallback expert; mean uses the average of the first M fallback experts.')
    parser.add_argument('--remap_reroute_replace_n', type=int, default=1,
                        help='N for --remap_reroute_replacement_mode nth. 1 matches the original top-k+1 fallback.')
    parser.add_argument('--remap_reroute_average_m', type=int, default=1,
                        help='M for --remap_reroute_replacement_mode mean.')
    return parser.parse_args()

def get_save_path(args):
    """Generates structured output path: ./output/Model-Method/YYMMDD/HHMM_Details..."""
    model_name = args.model_path.rstrip('/').split('/')[-1]
    date_str = datetime.now().strftime("%y%m%d")
    time_str = datetime.now().strftime("%H%M")
    
    # Level 1: Model & Method
    parent_dir = f"{model_name}-{args.method}"
    # Level 2: Date only
    date_dir = date_str
    
    custom_prefix = f"{args.custom_name}_" if args.custom_name else ""
    fatt_suffix = "fatt2" if args.use_flash_attention_2 else "no-fatt"
    
    # Evaluation flags
    eval_flags = ""
    if args.eval_tasks:
        num_tasks = len(args.eval_tasks.split(',')) if args.eval_tasks else 0
        eval_flags += f"_{args.num_fewshot}shot-{num_tasks}"
    if args.eval_samples is not None:
        eval_flags += f"_sample{args.eval_samples}"

    if args.method == "remap_pruning":
        ratio_str = f"sp{args.sp_ratio}"
        diversity_tag = f"_div{getattr(args, 'remap_diversity_lambda', 0.0)}" if getattr(args, 'remap_diversity_lambda', 0.0) > 0 else ""
        details = f"remap_{ratio_str}{diversity_tag}_{args.calib_set}{eval_flags}"
    elif args.method == "dynamic_similarity_skipping":
        details = f"dyn_remapsim_g{args.dynamic_similarity_gamma}_{args.calib_set}{eval_flags}"
    elif args.method == "dynamic_skipping":
        stats_tag = "_statsonly" if getattr(args, "dynamic_only_stats", False) else ""
        details = f"dynamic{stats_tag}_{args.calib_set}{eval_flags}"
    elif args.method in ["naee_pruning", "progressive_pruning"]:
        ratio_str = f"sp{args.sp_ratio}"
        details = f"naee_{ratio_str}_{args.calib_set}{eval_flags}"
    else:
        details = f"{args.calib_set}{eval_flags}"

    # Level 3: Time prefixing Details
    run_name = f"{time_str}_{custom_prefix}{details}_{fatt_suffix}"
    return osp.join(args.output_path, parent_dir, date_dir, run_name)

def load_model_and_tokenizer(args):
    """Loads tokenizer and model with custom memory allocation."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config_path = osp.join(args.model_path, "config.json")
    model_type = ""
    if osp.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "")
    
    # Auto-detect GPUs and set memory limits robustly
    num_gpus = torch.cuda.device_count()
    max_memory = {"cpu": "200GiB"}
    
    # Define buffers: pruning methods need room for calibration activations.
    buffer_gb = 7
    
    for i in range(num_gpus):
        total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        # Allocate (Total - Buffer), ensuring it doesn't go below a reasonable floor
        limit_gb = max(2, int(total_mem_gb - buffer_gb))
        max_memory[i] = f"{limit_gb}GiB"
        logger.info(f"GPU {i} ({torch.cuda.get_device_properties(i).name})\nTotal {total_mem_gb:.1f}GiB, Limit {limit_gb}GiB (Buffer {buffer_gb}GiB)")

    common_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_memory=max_memory,
        attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None,
        trust_remote_code=True,
    )

    try:
        if model_type == "qwen3_5_moe":
            logger.info(f"Loading model from {args.model_path} using AutoModelForImageTextToText...")
            try:
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_pretrained(args.model_path, **common_kwargs)
            except (ImportError, ValueError, KeyError):
                from transformers import Qwen3_5MoeForConditionalGeneration

                model = Qwen3_5MoeForConditionalGeneration.from_pretrained(args.model_path, **common_kwargs)
        else:
            logger.info(f"Loading model from {args.model_path} using AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(args.model_path, **common_kwargs)
    except (ImportError, ValueError, KeyError) as exc:
        if model_type == "qwen3_5_moe":
            raise RuntimeError(
                "This Qwen3.5-MoE checkpoint needs a transformers build with qwen3_5_moe support "
                "(the checkpoint config says transformers_version 4.57.1). Please install/update "
                "transformers in the active environment, then rerun."
            ) from exc
        raise
        
    # Standardize Layer Access: Alias MoE layer 'mlp' to 'block_sparse_moe' if needed
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for layer in get_decoder_layers(model):
        if not hasattr(layer, 'block_sparse_moe') and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
             layer.block_sparse_moe = layer.mlp

    return model, tokenizer


def run_evaluation(model, tokenizer, args, save_path):
    """Runs LM-Eval if requested and logs all to a single file."""
    results = None
    log_content = []

    # 2. LM Eval
    if args.eval_tasks:        
        logger.info(f"Starting LM-Eval task: {args.eval_tasks}")
        ensure_transformers_lm_eval_compat()
        lm_eval, HFLM, _, make_table = load_lm_eval()
        
        # Re-enable KV cache for generation (disabled during pruning to save VRAM)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        
        lm = HFLM(pretrained=model, tokenizer=tokenizer, dtype=torch.bfloat16, 
                 max_length=tokenizer.model_max_length, 
                 batch_size=args.eval_batch_size if args.eval_batch_size else "auto", 
                 trust_remote_code=True)
        results = lm_eval.simple_evaluate(model=lm, tasks=args.eval_tasks.split(','), limit=args.eval_samples,
                                         num_fewshot=args.num_fewshot)
        
        eval_table = make_table(results)
        print(eval_table)
        
        if "groups" in results: 
            log_content.append("\n" + make_table(results, "groups"))
        log_content.append("\n" + eval_table)
        log_content.append("\n" + str(model))

    # Single write operation if any results exist
    if log_content:
        log_file = osp.join(save_path, 'log.txt')
        with open(log_file, 'a') as f:
            f.write("\n".join(log_content) + "\n")
            
    return results


def finalize_save(model, tokenizer, save_path):
    """Saves model weights, config, and required modeling files."""
    if not os.path.exists(save_path): os.makedirs(save_path)

    def has_shared_expert_modules():
        for layer in get_decoder_layers(model):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
                experts = layer.block_sparse_moe.experts
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                experts = layer.mlp.experts
            else:
                continue
            if hasattr(experts, "num_experts") and not isinstance(experts, torch.nn.ModuleList):
                continue
            expert_ids = [id(expert) for expert in experts]
            if len(expert_ids) != len(set(expert_ids)):
                return True
        return False

    safe_serialization = not has_shared_expert_modules()
    if not safe_serialization:
        logger.info("Shared expert modules detected; saving with safe_serialization=False.")
    
    expert_counts = []
    for l in get_decoder_layers(model):
        if hasattr(l, 'block_sparse_moe'): # Mixtral
            expert_counts.append(get_expert_count(l.block_sparse_moe.experts))
        elif hasattr(l, 'mlp') and hasattr(l.mlp, 'experts'): # Generic MoE block (e.g. Qwen MoE)
            expert_counts.append(get_expert_count(l.mlp.experts))
            
    logger.info(f"Final expert counts: {expert_counts}")
    
    model.config.num_local_experts = expert_counts
    
    model.config.auto_map = {
        "AutoConfig": "configuration_mixtral.MixtralConfig",
        "AutoModelForCausalLM": "modeling_mixtral.MixtralForCausalLM"
    }
    
    model.save_pretrained(save_path, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(save_path)
    
    # Copy source files for AutoModel compatibility
    shutil.copy('model/modeling_mixtral.py', save_path)
    shutil.copy('model/configuration_mixtral.py', save_path)

    logger.info(f"Model and scripts saved to {save_path}")

def log_expert_sparsity(model, orig_counts, expert_id_map=None, save_path=None):
    """Calculates and logs layer-wise and global expert pruning sparsity.
       Also extracts and saves the replay configuration if context is provided.
    """
    try:
        kept_counts = []
        replay_config = None

        layer_kept_indices = {} # To store which specific experts were kept for visual
        if expert_id_map:
            kept_config = {}
            for layer_idx, layer in enumerate(get_decoder_layers(model)):
                experts = None
                if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
                    experts = layer.block_sparse_moe.experts
                elif hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                    experts = layer.mlp.experts

                id_to_original = expert_id_map.get(layer_idx, {})
                if experts is not None and id_to_original:
                    kept = [id_to_original.get(id(exp)) for exp in experts]
                    kept = [idx for idx in kept if idx is not None]
                    if kept:
                        kept_config[layer_idx] = sorted(kept)

            if kept_config:
                replay_config = {"format": "kept", "config": kept_config}
                kept_counts = [len(kept_config.get(i, list(range(o)))) for i, o in enumerate(orig_counts)]
                layer_kept_indices = kept_config

        # 2. Fallback: If config extraction skipped or failed, count manually
        if not kept_counts:
            kept_counts = []
            for l in get_decoder_layers(model):
                if hasattr(l, 'block_sparse_moe') and hasattr(l.block_sparse_moe, 'experts'):
                    kept_counts.append(get_expert_count(l.block_sparse_moe.experts))
                elif hasattr(l, 'mlp') and hasattr(l.mlp, 'experts'):
                    kept_counts.append(get_expert_count(l.mlp.experts))
                else:
                    kept_counts.append(0)
        
        # 3. Only count MoE layers (exclude Dense layers and shared experts)
        moe_orig = [(i, o, k) for i, (o, k) in enumerate(zip(orig_counts, kept_counts)) if o > 0]
        
        for i, o, k in moe_orig:
            if i in layer_kept_indices:
                kept_set = set(layer_kept_indices[i])
                visual = "".join(["■" if idx in kept_set else "□" for idx in range(o)])
            else:
                visual = "■" * k + "□" * (o - k)
            
            logger.info(f"Layer {i:2d}: [{visual}] ({k}/{o})")

        # Only count sparsity of MoE layers
        o_sum = sum(o for _, o, _ in moe_orig)
        k_sum = sum(k for _, _, k in moe_orig)
        logger.info(f"Global Routed Expert Sparsity: {(o_sum - k_sum)/o_sum*100:.2f}% ({o_sum-k_sum}/{o_sum} routed experts removed from {len(moe_orig)} MoE layers)")
        logger.info(f"*(Dense layers and shared experts are excluded from the above count)*")
        
        # 4. Save Config (if extracted)
        if replay_config and save_path:
            try:
                serializable = {
                    "format": replay_config["format"],
                    "config": {str(k): v for k, v in replay_config["config"].items()},
                }
                with open(osp.join(save_path, "pruning_config.json"), "w") as f:
                    json.dump(serializable, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save pruning config: {e}")
                
    except Exception as e:
        logger.error(f"Failed to log expert sparsity: {e}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Setup Path and Logging
    save_path = get_save_path(args)
    args.save_path = save_path
    os.makedirs(save_path, exist_ok=True)

    log_f = open(osp.join(save_path, 'console.log'), 'a', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    logging.getLogger().addHandler(logging.FileHandler(osp.join(save_path, 'console.log')))
    
    logger.info(f"Results will be saved to: {save_path}")
    logger.info(f"Arguments: {args}")

    # 2. Load Model & Data
    from data import build_calib_loader

    model, tokenizer = load_model_and_tokenizer(args)
    decoder_layers = get_decoder_layers(model)
    calib_loader, test_loader = build_calib_loader(
        args.calib_set, tokenizer, args.max_block_size, 
        args.n_blocks_for_stat, args.batch_size, args.num_workers, args.seed
    )

    # Capture original expert counts and IDs before pruning
    # Capture original expert counts and IDs before pruning
    if hasattr(model.config, "num_local_experts"):
        orig_val = model.config.num_local_experts
    elif hasattr(model.config, "n_routed_experts"):
        orig_val = model.config.n_routed_experts
    else:
        # Fallback: count from first layer
        orig_val = 0 # Should count later
        
    if isinstance(orig_val, (list, tuple)):
        orig_expert_counts = list(orig_val)
    elif isinstance(orig_val, int) and orig_val > 0:
        # Cannot simply use [orig_val]*layers — Some models may have dense layers
        orig_expert_counts = []
        for l in decoder_layers:
            if hasattr(l, 'block_sparse_moe') and hasattr(l.block_sparse_moe, 'experts'):
                orig_expert_counts.append(get_expert_count(l.block_sparse_moe.experts))
            elif hasattr(l, 'mlp') and hasattr(l.mlp, 'experts'):
                orig_expert_counts.append(get_expert_count(l.mlp.experts))
            else:
                orig_expert_counts.append(0)  # Dense layer
    else:
        # Manual count
        orig_expert_counts = []
        for l in decoder_layers:
            if hasattr(l, 'block_sparse_moe'):
                if hasattr(l.block_sparse_moe, 'experts'):
                     orig_expert_counts.append(get_expert_count(l.block_sparse_moe.experts))
                else:
                     # Wrapped block
                     if hasattr(l.block_sparse_moe, 'num_experts'):
                         orig_expert_counts.append(l.block_sparse_moe.num_experts)
                     else:
                         orig_expert_counts.append(0)
            elif hasattr(l, 'mlp') and hasattr(l.mlp, 'experts'):
                 orig_expert_counts.append(get_expert_count(l.mlp.experts))
            else:
                 orig_expert_counts.append(0) # Dense layer?

    # Calculate 'r' from sp_ratio
    max_experts = max(orig_expert_counts) if orig_expert_counts else 0
    if max_experts > 0:
        if args.sp_ratio > 0:
            # Sparsity ratio = fraction DROPPED.
            # Experts kept = Total * (1 - ratio)
            args.r = max(1, int(max_experts * (1 - args.sp_ratio)))
            logger.info(f"Calculated r={args.r} based on sp_ratio={args.sp_ratio} (Total experts: {max_experts})")
        else:
            args.r = max_experts
            logger.info(f"No pruning specified, setting r={args.r} (Keep all)")
    else:
        args.r = 0
        logger.warning("Could not determine total experts to calculate r from sp_ratio. Defaulting r=0.")


    
    # Build a map of {layer_idx: {id(expert_obj): index}} to track experts after pruning
    expert_id_map = {}
    for i, layer in enumerate(decoder_layers):
        if hasattr(layer, 'block_sparse_moe'):
            experts = layer.block_sparse_moe.experts
            if isinstance(experts, torch.nn.ModuleList):
                expert_id_map[i] = {id(exp): idx for idx, exp in enumerate(experts)}
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            experts = layer.mlp.experts
            if isinstance(experts, torch.nn.ModuleList):
                expert_id_map[i] = {id(exp): idx for idx, exp in enumerate(experts)}

    # 3. Pruning Dispatch
    logger.info(f"Starting method: {args.method}")
    start_t = datetime.now()
    
    if args.method == "remap_pruning":
        model = METHODS[args.method](model, calib_loader, test_loader, args)
    elif args.method == "dynamic_skipping":
        model, _ = METHODS[args.method](
            model,
            calib_loader,
            args,
            enable_inference_wrapper=not getattr(args, "dynamic_only_stats", False),
        )
    elif args.method == "dynamic_similarity_skipping":
        model, _ = METHODS[args.method](model, calib_loader, args)
    elif args.method == "naee_pruning":
        model, _ = METHODS[args.method](model, test_loader, args)
    elif args.method == "progressive_pruning":
        model, _ = METHODS[args.method](model, test_loader, args)
    
    pruning_elapsed = datetime.now() - start_t
    logger.info(f"Pruning finished in {pruning_elapsed.total_seconds()/3600:.4f} hours.")

    # 4. Final Logs & Evaluation
    # Now merges replay config extraction/saving into this step
    log_expert_sparsity(
        model,
        orig_expert_counts,
        expert_id_map=expert_id_map,
        save_path=save_path,
    )
    eval_start_t = datetime.now()
    results = run_evaluation(model, tokenizer, args, save_path)
    eval_elapsed = datetime.now() - eval_start_t
    logger.info(f"Evaluation finished in {eval_elapsed.total_seconds()/3600:.4f} hours.")

    # 5. Save Model
    if args.save_model and args.method.endswith('_pruning'):
        finalize_save(model, tokenizer, save_path)
    
    torch.cuda.empty_cache()
    logger.info("Execution finished.")

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", 
        level=logging.INFO, 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
