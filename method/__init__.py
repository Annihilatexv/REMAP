from importlib import import_module


def _lazy_method(module_name, function_name):
    def wrapper(*args, **kwargs):
        module = import_module(module_name, package=__package__)
        return getattr(module, function_name)(*args, **kwargs)

    return wrapper


def _model_type(model):
    return getattr(getattr(model, "config", None), "model_type", "")


def _is_qwen_moe_model(model):
    if _model_type(model) in {"qwen2_moe", "qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"}:
        return True
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    return getattr(text_config, "model_type", "") in {"qwen3_moe", "qwen3_5_moe_text"}


def _is_deepseek_moe_model(model):
    return any(
        hasattr(layer, "mlp") and hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate")
        for layer in model.model.layers
    )


def _naee_pruning_dispatch(model, calib_loader, args, **kwargs):
    if _is_deepseek_moe_model(model):
        raise NotImplementedError("NAEE pruning is not supported for DeepSeek models.")

    from .naee import naee_mixtral

    return naee_mixtral.naee_pruning(model, calib_loader, args, **kwargs)


def _remap_pruning_dispatch(model, calib_loader, test_loader, args):
    if _is_qwen_moe_model(model):
        from .remap import remap_qwen_moe

        return remap_qwen_moe.remap_moe_pruning_qwen_moe(model, calib_loader, test_loader, args)

    if _is_deepseek_moe_model(model):
        raise NotImplementedError("REMAP pruning is not supported for DeepSeek models.")

    from .remap import remap_mixtral

    return remap_mixtral.remap_moe_pruning(model, calib_loader, test_loader, args)


METHODS = {
    "remap_pruning": _remap_pruning_dispatch,
    "naee_pruning": _naee_pruning_dispatch,
    "progressive_pruning": _lazy_method(".naee.naee_mixtral", "progressive_pruning"),
    "dynamic_skipping": _lazy_method(".dynamic_skipping.dynamic_mixtral", "dynamic_skipping"),
    "dynamic_similarity_skipping": _lazy_method(".dynamic_skipping.dynamic_mixtral", "dynamic_similarity_skipping"),
}
