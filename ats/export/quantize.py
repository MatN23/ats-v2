"""Post-training int8 weight quantization for HuggingFace export.

This is deliberately a *different* feature from ats.model.quantization's
int8 path: that module does quantization-*aware training* (fake-quant in
bf16/fp16, so it changes training numerics but saves no memory at all --
see PERF_AUDIT_REPORT.md and the README's Scale limitations table). This
module does *post-training* quantization: it runs once, after training is
finished, on an already-trained state_dict, and produces real int8 tensors
that are actually smaller on disk than the fp32/bf16/fp16 originals it
replaces.

Scheme: symmetric, per-output-channel (per-row) linear quantization.
For a weight tensor `w` of shape [out_features, in_features] (the standard
nn.Linear convention this codebase's Linear layers use throughout
ats/model/), each output row is quantized independently:

    scale[i]   = max(abs(w[i, :])) / 127
    int8_w[i]  = round(w[i, :] / scale[i]).clamp(-127, 127)
    dequant    = int8_w[i] * scale[i]  (approximately recovers w[i, :])

Per-channel (rather than one scale for the whole tensor) matters because
weight magnitude commonly varies a lot row-to-row; one global scale would
waste most of int8's 256 levels on the rows with the largest values and
under-resolve every other row.

Only 2D weight tensors are quantized, and only when their key doesn't look
like an embedding or normalization parameter (embedding tables and norm
gains/biases are far more sensitive to quantization noise per parameter
touched, relative to what they contribute to output quality, than the much
larger attention/FFN projection matrices this is actually worth it for; see
_should_quantize_key's docstring for the exact heuristic). Everything else
(biases, embeddings, norm weights, 1D tensors) passes through completely
unchanged, bit-for-bit.
"""

from __future__ import annotations

import torch

# Keys containing any of these substrings are left unquantized even if they
# are 2D float weight tensors, because they are not the "big matmul weight"
# case this scheme is designed for:
#   - embed / lm_head: embedding tables. Every row is looked up independently
#     (never averaged/summed with other rows the way a Linear's output is),
#     so per-row quantization noise shows up directly and undiluted in
#     whichever token happened to select that row, and vocab-sized tables
#     already dominate int8's per-row overhead differently than smaller
#     projection matrices do. Left unquantized rather than guessing at a
#     scheme that hasn't been checked here.
#   - norm: LayerNorm/RMSNorm gain (and bias, if present) parameters are
#     tiny (hidden_size-length, 1D in this codebase's norm.py, so they'd be
#     excluded by the 2D check alone anyway -- listed here for clarity, not
#     because it changes behavior).
_SKIP_SUBSTRINGS = ("embed", "lm_head", "norm")


def _should_quantize_key(key: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    if not torch.is_floating_point(tensor):
        return False
    lowered = key.lower()
    return not any(skip in lowered for skip in _SKIP_SUBSTRINGS)


def quantize_tensor_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes a single 2D [out_features, in_features] weight tensor.

    Returns (int8_weight, scale) where int8_weight has dtype torch.int8 and
    the same shape as `weight`, and scale is a 1D float32 tensor of shape
    [out_features] (one scale per output row). Raises ValueError if `weight`
    isn't 2D -- callers are expected to have already filtered via
    _should_quantize_key; this check exists so a caller that skips that
    filter fails loudly instead of silently misquantizing.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"quantize_tensor_int8 expects a 2D [out_features, in_features] "
            f"tensor, got shape {tuple(weight.shape)}."
        )
    weight = weight.detach().to(torch.float32)
    row_absmax = weight.abs().amax(dim=1)
    # A row of all zeros would divide by zero; its quantized values are all
    # zero regardless of scale, so any nonzero scale reconstructs it exactly
    # -- substitute 1.0 purely to avoid a NaN, not because it's meaningful.
    scale = torch.where(row_absmax > 0, row_absmax / 127.0, torch.ones_like(row_absmax))
    int8_weight = (
        torch.round(weight / scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    )
    return int8_weight, scale.to(torch.float32)


def dequantize_tensor_int8(
    int8_weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Inverse of quantize_tensor_int8: reconstructs an approximate float32
    weight tensor from its int8 representation and per-row scale."""
    if int8_weight.ndim != 2:
        raise ValueError(
            f"dequantize_tensor_int8 expects a 2D tensor, got shape "
            f"{tuple(int8_weight.shape)}."
        )
    if scale.shape != (int8_weight.shape[0],):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} doesn't match int8_weight's "
            f"output dimension {int8_weight.shape[0]}."
        )
    return int8_weight.to(torch.float32) * scale.unsqueeze(1)


def quantize_state_dict_int8(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Quantizes every eligible weight tensor in `state_dict` (see module
    docstring for the eligibility rule). Returns a NEW dict -- the input is
    not mutated -- where each quantized key `k` is replaced by an int8
    tensor and a new key `k + ".quant_scale"` holding its per-row fp32
    scale; every other key is carried over unchanged (same tensor object,
    not a copy, since it's untouched).

    Also returns the sorted list of keys that were quantized, so callers
    (export_to_huggingface) can record it in checkpoint metadata -- a loader
    needs to know which keys require dequantize_tensor_int8 before use, and
    guessing that from key names alone would be exactly the kind of
    unstated assumption this module's docstring argues against.
    """
    out: dict[str, torch.Tensor] = {}
    quantized_keys: list[str] = []
    for key, tensor in state_dict.items():
        if _should_quantize_key(key, tensor):
            int8_weight, scale = quantize_tensor_int8(tensor)
            out[key] = int8_weight
            out[f"{key}.quant_scale"] = scale
            quantized_keys.append(key)
        else:
            out[key] = tensor
    return out, sorted(quantized_keys)
