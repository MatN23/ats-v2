"""Memory helpers: auto batch-size finder (binary search with real forward/backward
passes, catching only torch.cuda.OutOfMemoryError), GPU memory reporting, and a
pre-flight memory *estimator* that runs before any GPU memory is allocated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

import torch

from ats.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from ats.config.schema import ATSConfig

logger = get_logger("ats.utils.memory")

_BYTES_PER_PARAM = {"bf16": 2, "fp16": 2, "fp32": 4}
_BYTES_PER_GIB = 1024 ** 3
# Adam keeps two fp32 moment tensors (m, v) per parameter, plus (under mixed
# precision) an fp32 master-weight copy. This is the standard
# mixed-precision-Adam memory accounting used by DeepSpeed's own ZeRO memory
# calculator; see https://www.deepspeed.ai/tutorials/zero/ for the same
# 4 (master) + 4 + 4 (moments) = 12 bytes/param figure at ZeRO stage 0.
_ADAM_BYTES_PER_PARAM_FP32_STATES = 12
# Rough activation-memory-per-token-per-layer constant, following the
# standard transformer activation memory formula (Korthikanti et al. 2022 /
# Megatron-LM's activation recomputation paper): roughly
# ~34 * hidden_size bytes per token per layer without recomputation, in
# fp16/bf16. This is a heuristic, not an exact accounting of every buffer;
# it exists to give an order-of-magnitude pre-flight warning, not a
# guarantee.
_ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_PER_HIDDEN = 34


@dataclass
class MemoryReport:
    total_gb: float
    model_gb: float
    optimizer_gb: float
    activation_gb: float
    fits_on_single_gpu: bool
    suggested_batch_size: int
    suggested_grad_accum: int
    suggested_zero_stage: int
    available_gb: float


def _zero_stage_from_strategy(strategy: str) -> int:
    mapping = {
        "deepspeed_zero0": 0, "deepspeed_zero1": 1, "deepspeed_zero2": 2,
        "deepspeed_zero3": 3, "deepspeed_moe": 2, "fsdp": 3,
    }
    return mapping.get(strategy, 2)


def _detect_available_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device).total_memory / _BYTES_PER_GIB


def estimate_memory(config: "ATSConfig", target_batch_size: Optional[int] = None) -> MemoryReport:
    """Estimate peak per-GPU memory before training starts (no GPU memory is
    allocated by this function). Uses the same parameter-count arithmetic as
    ats.parallelism.auto_parallel.estimate_param_count, standard
    mixed-precision-Adam optimizer-state accounting, and a standard
    (heuristic) transformer activation-memory formula.

    This is an estimate for pre-flight warnings, not an exact simulator —
    real memory use depends on the specific CUDA allocator, fragmentation,
    and framework overhead. Treat the suggested_* fields as a starting point.
    """
    from ats.parallelism.auto_parallel import estimate_param_count, resolve_strategy

    if not config.model.is_resolved():
        raise ValueError(
            "estimate_memory requires a resolved ModelConfig (hidden_size/num_layers/"
            "etc must not be None). Call ats.config.defaults.apply_size_preset() first."
        )

    world_size = max(1, config.parallelism.gpus * config.parallelism.nodes)
    strategy = resolve_strategy(config)
    zero_stage = _zero_stage_from_strategy(strategy)

    num_params = estimate_param_count(config.model)
    bytes_per_param = _BYTES_PER_PARAM[config.training.mixed_precision]

    model_bytes = num_params * bytes_per_param
    optimizer_bytes = num_params * _ADAM_BYTES_PER_PARAM_FP32_STATES
    gradient_bytes = num_params * 4  # gradients accumulated in fp32

    # ZeRO sharding: stage 1 shards optimizer state, stage 2 additionally
    # shards gradients, stage 3 additionally shards parameters themselves.
    if zero_stage >= 1:
        optimizer_bytes = optimizer_bytes / world_size
    if zero_stage >= 2:
        gradient_bytes = gradient_bytes / world_size
    if zero_stage >= 3:
        model_bytes = model_bytes / world_size

    batch_size = target_batch_size if target_batch_size is not None else config.training.micro_batch_size
    seq_len = config.data.seq_length
    activation_bytes = (
        batch_size * seq_len * config.model.num_layers * config.model.hidden_size
        * _ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_PER_HIDDEN
    )
    if config.model.gradient_checkpointing:
        # This flag implements FULL (every-layer) activation checkpointing,
        # not a selective "checkpoint every sqrt(L)-th layer" scheme, so the
        # sqrt(num_layers) bound (Chen et al. 2016) does not apply here --
        # that bound is for the selective strategy specifically.
        #
        # The precise full-checkpointing formula is actually
        # num_layers * (per-layer checkpoint boundary tensor, small) +
        # (one layer's full recomputation activations, large), which is
        # dominated by different terms depending on num_layers and hidden
        # size in ways that are hard to bound tightly as a simple heuristic.
        # Rather than guess at a scaling law we can't validate without a
        # GPU, we use the commonly-cited practical figure reported for full
        # activation checkpointing in production LLM training (roughly a
        # constant 2-4x memory reduction, not a reduction that keeps
        # growing with depth) as a documented, conservative estimate.
        _CHECKPOINTING_REDUCTION_FACTOR = 3.0
        activation_bytes = activation_bytes / _CHECKPOINTING_REDUCTION_FACTOR

    model_gb = (model_bytes + gradient_bytes) / _BYTES_PER_GIB
    optimizer_gb = optimizer_bytes / _BYTES_PER_GIB
    activation_gb = activation_bytes / _BYTES_PER_GIB
    total_gb = model_gb + optimizer_gb + activation_gb

    available_gb = _detect_available_gpu_memory_gb()
    fits = (total_gb <= 0.8 * available_gb) if available_gb > 0 else True

    suggested_batch_size = batch_size
    suggested_grad_accum = config.training.grad_accum_steps
    if available_gb > 0 and not fits:
        non_activation_gb = model_gb + optimizer_gb
        remaining_gb = max(0.0, 0.8 * available_gb - non_activation_gb)
        per_sample_activation_gb = activation_gb / max(1, batch_size)
        if per_sample_activation_gb > 0:
            suggested_batch_size = max(1, int(remaining_gb / per_sample_activation_gb))
        else:
            suggested_batch_size = 1
        if suggested_batch_size < batch_size:
            ratio = max(1, batch_size // max(1, suggested_batch_size))
            suggested_grad_accum = config.training.grad_accum_steps * ratio

    suggested_zero_stage = zero_stage
    if available_gb > 0 and not fits and zero_stage < 3:
        suggested_zero_stage = zero_stage + 1

    return MemoryReport(
        total_gb=total_gb,
        model_gb=model_gb,
        optimizer_gb=optimizer_gb,
        activation_gb=activation_gb,
        fits_on_single_gpu=fits,
        suggested_batch_size=suggested_batch_size,
        suggested_grad_accum=suggested_grad_accum,
        suggested_zero_stage=suggested_zero_stage,
        available_gb=available_gb,
    )


def get_gpu_memory_info() -> Dict[str, float]:
    """Returns allocated/reserved GPU memory in GiB, or zeros if no CUDA device."""
    if not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0, "total_gib": 0.0}
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return {"allocated_gib": allocated, "reserved_gib": reserved, "total_gib": total}


def find_max_batch_size(
    try_batch_size_fn: Callable[[int], None],
    min_batch_size: int = 1,
    max_batch_size: int = 4096,
) -> int:
    """Binary search for the largest batch size for which `try_batch_size_fn`
    (a real forward+backward step) does not raise torch.cuda.OutOfMemoryError.

    `try_batch_size_fn` is responsible for its own cleanup (e.g. zeroing grads,
    calling torch.cuda.empty_cache()) between attempts.
    """
    if min_batch_size < 1:
        raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}.")
    if max_batch_size < min_batch_size:
        raise ValueError(
            f"max_batch_size ({max_batch_size}) must be >= min_batch_size ({min_batch_size})."
        )

    def _fits(batch_size: int) -> bool:
        try:
            try_batch_size_fn(batch_size)
            return True
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    if not _fits(min_batch_size):
        raise RuntimeError(
            f"Even the minimum batch size ({min_batch_size}) does not fit in memory. "
            f"Fix: reduce model size, enable gradient_checkpointing, or use a higher "
            f"ZeRO stage / more GPUs."
        )

    low, high = min_batch_size, min_batch_size
    while high <= max_batch_size and _fits(high):
        low = high
        high *= 2

    high = min(high, max_batch_size)
    while low < high:
        mid = (low + high + 1) // 2
        if _fits(mid):
            low = mid
        else:
            high = mid - 1

    logger.info("Auto batch-size search settled on batch_size=%d", low)
    return low
