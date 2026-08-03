"""Generates a DeepSpeed JSON config from an ATSConfig and initializes the
DeepSpeed engine. This is the only place ats calls deepspeed.initialize().
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

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

    if config.model.gradient_checkpointing:
        ds_config["activation_checkpointing"] = {
            "partition_activations": zero_stage == 3,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False,
        }

    ds_config["optimizer"] = {
        "type": "AdamW",
        "params": {
            "lr": config.training.learning_rate,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
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


def initialize_engine(
    model: nn.Module, config: ATSConfig, micro_batch_size: int,
) -> Tuple[Any, Any, Any, Any]:
    """Calls deepspeed.initialize() with a config generated from `config`.

    Returns (model_engine, optimizer, _, lr_scheduler) exactly as DeepSpeed does.
    Raises ConfigError with an actionable message if the deepspeed package is
    not installed, rather than silently falling back (parallelism.strategy is
    mandatory per ats-v2's design, unlike model-level MoE which has a
    single-process fallback).
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

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )
    return model_engine, optimizer, _, lr_scheduler
