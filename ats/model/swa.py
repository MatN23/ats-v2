"""Sliding Window Attention (SWA) mask generation. Each query position i may
attend only to key positions j with j <= i and i - j < window_size — a
banded lower-triangular mask, not full causal. No custom CUDA kernels."""

from __future__ import annotations

import torch


def generate_swa_mask(seq_len: int, window_size: int, device: torch.device) -> torch.Tensor:
    """Returns a boolean attention mask of shape [seq_len, seq_len] where
    mask[i, j] is True iff position i is allowed to attend to position j:
    j <= i (causal) AND i - j < window_size (windowed).

    True == "attend", following torch.nn.functional.scaled_dot_product_attention's
    boolean-mask convention (True = keep, False = mask out).
    """
    if seq_len <= 0:
        raise ValueError(f"generate_swa_mask requires seq_len > 0, got {seq_len}.")
    if window_size <= 0:
        raise ValueError(f"generate_swa_mask requires window_size > 0, got {window_size}.")

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
