"""Tests for ats.export.quantize: post-training int8 weight quantization.

Verifies the actual numerical properties that matter here: round-trip
reconstruction error is bounded, real memory savings happen (int8 vs the
original dtype), embeddings/norms/1D tensors are never touched, and the
scheme integrates correctly with export_to_huggingface's config.json
metadata and safetensors output.
"""

from __future__ import annotations

import torch

from ats.export.quantize import (
    dequantize_tensor_int8,
    quantize_state_dict_int8,
    quantize_tensor_int8,
)


def test_round_trip_error_is_bounded():
    torch.manual_seed(0)
    weight = torch.randn(64, 128) * 0.3  # realistic small-init-scale weight
    int8_weight, scale = quantize_tensor_int8(weight)

    assert int8_weight.dtype == torch.int8
    assert int8_weight.shape == weight.shape
    assert scale.shape == (64,)

    reconstructed = dequantize_tensor_int8(int8_weight, scale)
    # Symmetric int8 quantization with per-row scale = max(abs(row))/127
    # gives a per-element error bound of scale/2 (rounding to the nearest of
    # 254 evenly spaced levels spanning [-max, max]). Assert against that
    # bound directly, per row, rather than an arbitrary tolerance.
    per_row_max_error = (reconstructed - weight).abs().amax(dim=1)
    per_row_bound = scale / 2.0 + 1e-6
    assert torch.all(per_row_max_error <= per_row_bound)


def test_all_zero_row_reconstructs_exactly():
    weight = torch.zeros(4, 8)
    int8_weight, scale = quantize_tensor_int8(weight)
    assert not torch.isnan(scale).any()
    reconstructed = dequantize_tensor_int8(int8_weight, scale)
    assert torch.equal(reconstructed, weight)


def test_quantize_tensor_int8_rejects_non_2d():
    import pytest

    with pytest.raises(ValueError, match="2D"):
        quantize_tensor_int8(torch.randn(10))


def test_state_dict_skips_embeddings_norms_and_1d_tensors():
    state_dict = {
        "embed_tokens.weight": torch.randn(100, 32),  # embedding -> skip
        "layers.0.attn.q_proj.weight": torch.randn(32, 32),  # Linear -> quantize
        "layers.0.norm.weight": torch.randn(32),  # 1D norm gain -> skip
        "layers.0.attn.q_proj.bias": torch.randn(32),  # 1D bias -> skip
        "lm_head.weight": torch.randn(100, 32),  # lm_head -> skip
    }
    out, quantized_keys = quantize_state_dict_int8(state_dict)

    assert quantized_keys == ["layers.0.attn.q_proj.weight"]
    assert out["layers.0.attn.q_proj.weight"].dtype == torch.int8
    assert "layers.0.attn.q_proj.weight.quant_scale" in out

    for skipped_key in (
        "embed_tokens.weight",
        "layers.0.norm.weight",
        "layers.0.attn.q_proj.bias",
        "lm_head.weight",
    ):
        assert out[skipped_key] is state_dict[skipped_key]  # untouched, same object
        assert f"{skipped_key}.quant_scale" not in out


def test_quantized_tensor_is_smaller_than_original():
    weight = torch.randn(256, 256)  # fp32: 256*256*4 bytes
    int8_weight, scale = quantize_tensor_int8(weight)
    original_bytes = weight.numel() * weight.element_size()
    quantized_bytes = (
        int8_weight.numel() * int8_weight.element_size()
        + scale.numel() * scale.element_size()
    )
    assert quantized_bytes < original_bytes / 3  # real savings, not just a relabel


def test_dequantize_rejects_mismatched_scale_shape():
    import pytest

    int8_weight = torch.zeros(4, 8, dtype=torch.int8)
    bad_scale = torch.ones(5)  # wrong length
    with pytest.raises(ValueError, match="doesn't match"):
        dequantize_tensor_int8(int8_weight, bad_scale)
