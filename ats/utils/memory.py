"""Memory helpers: auto batch-size finder (binary search with real forward/backward
passes, catching only torch.cuda.OutOfMemoryError) and GPU memory reporting."""

from __future__ import annotations

from typing import Callable, Dict

import torch

from ats.utils.logging_utils import get_logger

logger = get_logger("ats.utils.memory")


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
