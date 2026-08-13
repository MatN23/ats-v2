"""Generates a DeepSpeed JSON config from an ATSConfig and initializes the
DeepSpeed engine. This is the only place ats calls deepspeed.initialize().
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch.nn as nn

from ats.config.schema import ATSConfig, ConfigError
from ats.parallelism.auto_parallel import resolve_strategy
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.parallelism.deepspeed_utils")

_ZERO_STAGE_BY_STRATEGY = {
    "deepspeed_zero0": 0,
    "deepspeed_zero1": 1,
    "deepspeed_zero2": 2,
    "deepspeed_zero3": 3,
    "deepspeed_moe": 2,
}


def get_param_groups(model: nn.Module, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Splits parameters into a weight-decayed group (matrix-shaped weights:
    attention/FFN/MoE projections, embeddings) and a non-decayed group
    (biases and normalization scale parameters). Weight decay on 1D
    parameters like RMSNorm weights or biases empirically hurts and is
    universally excluded in standard LLM training recipes (GPT-2/3, Llama,
    etc.) — decaying a norm's scale or a bias pulls it toward zero for no
    representational benefit."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D parameters (biases, RMSNorm/LayerNorm weights, and any
        # explicitly-named "embedding"/"norm" parameter) are excluded from
        # weight decay; everything else (2D+ projection matrices) is decayed.
        if param.dim() <= 1 or "norm" in name or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    if not groups:
        raise ConfigError(
            "get_param_groups found no trainable parameters on the model. "
            "Fix: check that the model has parameters with requires_grad=True."
        )
    return groups


def build_deepspeed_config(config: ATSConfig, micro_batch_size: int) -> Dict[str, Any]:
    """Programmatically build a DeepSpeed config dict from ATSConfig fields.
    No hardcoded values that ignore the user's YAML: every field here is
    derived from `config`.
    """
    strategy = resolve_strategy(config)
    if strategy not in _ZERO_STAGE_BY_STRATEGY and strategy != "fsdp":
        raise ConfigError(
            f"Unknown resolved parallelism strategy '{strategy}'. "
            f"Fix: parallelism.strategy must be one of "
            f"{sorted(_ZERO_STAGE_BY_STRATEGY)} or 'auto'."
        )
    if strategy == "fsdp":
        raise ConfigError(
            "parallelism.strategy=fsdp was requested, but ats-v2's training path uses "
            "DeepSpeed as its mandatory execution engine. Fix: choose a deepspeed_* "
            "strategy or 'auto'."
        )

    zero_stage = _ZERO_STAGE_BY_STRATEGY[strategy]

    ds_config: Dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": config.training.grad_accum_steps,
        "gradient_clipping": config.training.grad_clip_norm,
        "steps_per_print": config.logging.log_every,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "zero_allow_untested_optimizer": True,
    }

    if config.training.mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif config.training.mixed_precision == "fp16":
        ds_config["fp16"] = {"enabled": True, "auto_cast": True}
    # fp32 -> no precision block; DeepSpeed defaults to fp32.

    # NOTE: activation checkpointing is intentionally NOT configured here via
    # DeepSpeed's own "activation_checkpointing" JSON block. That block (and
    # its partition_activations / cpu_checkpointing / contiguous_memory_
    # optimization sub-options) only takes effect for code that explicitly
    # calls deepspeed.checkpointing.checkpoint(...); it has zero effect on
    # plain torch.utils.checkpoint.checkpoint(...), which is what
    # ats.model.transformer.ATSTransformer._run_layers actually uses
    # (controlled purely by model.checkpoint_every_n_layers). A previous
    # version of this function set this block whenever
    # config.model.checkpoint_every_n_layers was truthy, which did nothing
    # DeepSpeed-side (dead configuration) and, on top of that,
    # partition_activations specifically requires a model-parallel `mpu`
    # object (deepspeed.checkpointing.configure(mpu, ...)) to have any memory
    # effect at all -- ats-v2 has no tensor/model parallelism, so it would
    # have been a no-op even if the checkpointing calls did go through
    # DeepSpeed's API. If DeepSpeed-native CPU-offloaded checkpointing is
    # wanted in the future, ATSTransformer's checkpoint() calls need to be
    # switched to deepspeed.checkpointing.checkpoint (with an mpu configured
    # via deepspeed.checkpointing.configure) for this JSON block to matter.

    if config.optimizer.bits == 32:
        # Unchanged path: DeepSpeed builds torch.optim.AdamW itself from this
        # config block. For bits=8, no "optimizer" key is set here -- see
        # initialize_engine, which instead constructs a bitsandbytes
        # Adam8bit instance client-side and passes it directly to
        # deepspeed.initialize(optimizer=...).
        ds_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": config.training.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                # weight_decay is intentionally omitted here: it's set per
                # parameter-group instead (see get_param_groups), so biases and
                # norm weights get weight_decay=0.0 while projection matrices
                # get the real value.
            },
        }

    if strategy == "deepspeed_moe" or config.model.use_moe:
        ds_config["moe"] = {
            "enabled": True,
            "ep_size": max(1, config.parallelism.gpus * config.parallelism.nodes),
            "num_experts": config.model.num_experts,
            "top_k": config.model.moe_top_k,
            "capacity_factor": config.model.moe_capacity_factor,
            "min_capacity": 4,
            "moe_param_group": True,
        }

    return ds_config


def _build_bitsandbytes_optimizer(config: ATSConfig, param_groups: List[Dict[str, Any]]) -> Any:
    """Builds a bitsandbytes 8-bit Adam optimizer over `param_groups`. Raises
    ConfigError with an actionable message if bitsandbytes is not installed,
    rather than silently falling back to fp32 AdamW."""
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ConfigError(
            "optimizer.bits=8 requires the 'bitsandbytes' package, which is not "
            "installed. Fix: pip install 'ats-v2[8bit]' (or `pip install "
            "bitsandbytes` directly)."
        ) from exc

    return bnb.optim.Adam8bit(
        param_groups,
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def initialize_engine(
    model: nn.Module, config: ATSConfig, micro_batch_size: int,
) -> Tuple[Any, Any, Any, Any]:
    """Calls deepspeed.initialize() with a config generated from `config`.

    Returns (model_engine, optimizer, _, lr_scheduler) exactly as DeepSpeed does.
    Raises ConfigError with an actionable message if the deepspeed package is
    not installed, rather than silently falling back (parallelism.strategy is
    mandatory per ats-v2's design, unlike model-level MoE which has a
    single-process fallback).

    When config.optimizer.bits == 8, a bitsandbytes Adam8bit instance is
    constructed client-side over the same weight-decay/no-decay param groups
    used for the default fp32 path, and handed to deepspeed.initialize() as
    the `optimizer` kwarg -- DeepSpeed then wraps/shards that instance
    (ZeRO, mixed precision, etc.) exactly as it would its own AdamW, instead
    of building an optimizer from the "optimizer" key of the DeepSpeed JSON
    config (which build_deepspeed_config leaves unset for bits=8).
    """
    try:
        import deepspeed
    except ImportError as exc:
        raise ConfigError(
            "DeepSpeed is required to train with ats-v2 but is not installed. "
            "Fix: pip install deepspeed (see https://www.deepspeed.ai/tutorials/advanced-install/ "
            "for CUDA/build prerequisites)."
        ) from exc

    ds_config = build_deepspeed_config(config, micro_batch_size)
    logger.info("Initializing DeepSpeed engine with config: %s", ds_config)

    param_groups = get_param_groups(
        model, lr=config.training.learning_rate, weight_decay=config.training.weight_decay,
    )

    client_optimizer = (
        _build_bitsandbytes_optimizer(config, param_groups) if config.optimizer.bits == 8 else None
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups if client_optimizer is None else None,
        optimizer=client_optimizer,
        config=ds_config,
    )
    return model_engine, optimizer, _, lr_scheduler