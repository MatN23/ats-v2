"""Sliding Window Attention (SWA) mask generation. Each query position i may
attend only to key positions j with j <= i and i - j < window_size — a
banded lower-triangular mask, not full causal. No custom CUDA kernels."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=8)
def generate_swa_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Returns a boolean attention mask of shape [seq_len, seq_len] where
    mask[i, j] is True iff position i is allowed to attend to position j:
    j <= i (causal) AND i - j < window_size (windowed).

    True == "attend", following torch.nn.functional.scaled_dot_product_attention's
    boolean-mask convention (True = keep, False = mask out).

    PERF: memoized on (seq_len, window_size, device) via lru_cache. Every
    SWA-enabled layer previously rebuilt this full [seq_len, seq_len]
    boolean tensor (16 MiB at seq_len=4096) from scratch on every forward
    pass, even though training uses a fixed seq_len/window_size for the
    entire run -- num_layers redundant O(seq_len^2) rebuilds per step for
    no reason. maxsize=8 caps memory if seq_len legitimately varies (e.g.
    padded final batches, or eval at a different length) rather than
    growing unbounded. NOTE: callers must not mutate the returned tensor
    in place (none currently do -- it's only read as an attn_mask/combined
    via `&`, which allocates a new tensor).
    """
    if seq_len <= 0:
        raise ValueError(f"generate_swa_mask requires seq_len > 0, got {seq_len}.")
    if window_size <= 0:
        raise ValueError(
            f"generate_swa_mask requires window_size > 0, got {window_size}."
        )

    positions = torch.arange(seq_len, device=device)
    i = positions.unsqueeze(1)  # [seq_len, 1]
    j = positions.unsqueeze(0)  # [1, seq_len]
    distance = i - j
    causal = distance >= 0
    windowed = distance < window_size
    mask = causal & windowed
    return mask


def is_full_attention_layer(layer_idx: int, full_attention_interval: int) -> bool:
    """Every `full_attention_interval`-th layer (1-indexed: layer 3, 7, 11, ...
    for interval=4) uses full causal attention instead of the sliding window,
    so long-range information can still propagate through the stack."""
    if full_attention_interval <= 0:
        raise ValueError(
            f"swa_full_attention_interval must be positive, got {full_attention_interval}."
        )
    return (layer_idx + 1) % full_attention_interval == 0
