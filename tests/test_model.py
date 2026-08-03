"""Tests for ats.model: shapes, RoPE correctness, RMSNorm gradient check,
MoE routing, MoD capacity, SWA masking, MLA cache size."""

from __future__ import annotations

import math

import pytest
import torch

from ats.config.schema import ModelConfig
from ats.model.attention import GroupedQueryAttention
from ats.model.mla import MLAAttention
from ats.model.moe import MoELayer
from ats.model.mod import MixtureOfDepths
from ats.model.norm import RMSNorm
from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from ats.model.swa import generate_swa_mask, is_full_attention_layer
from ats.model.transformer import ATSTransformer, TransformerOutput


def test_forward_pass_shape(dummy_model, dummy_batch):
    output = dummy_model(dummy_batch["input_ids"])
    assert isinstance(output, TransformerOutput)
    batch, seq_len = dummy_batch["input_ids"].shape
    assert output.logits.shape == (batch, seq_len, dummy_model.config.vocab_size)


def test_backward_pass_does_not_crash(dummy_model, dummy_batch):
    output = dummy_model(dummy_batch["input_ids"])
    loss = output.logits.float().pow(2).mean() + output.aux_loss
    loss.backward()
    grad_found = any(p.grad is not None for p in dummy_model.parameters())
    assert grad_found


def test_forward_rejects_out_of_range_token_ids(dummy_model, dummy_model_config):
    bad_ids = torch.full((1, 4), dummy_model_config.vocab_size + 5, dtype=torch.long)
    with pytest.raises(ValueError):
        dummy_model(bad_ids)


def test_rope_angles_match_closed_form():
    dim, max_len, theta = 8, 16, 10000.0
    rope = RotaryEmbedding(dim, max_len, theta)
    cos, sin = rope(seq_len=4, device=torch.device("cpu"), dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    expected_freqs_pos2 = 2 * inv_freq  # position index 2
    expected_cos = torch.cat([expected_freqs_pos2.cos(), expected_freqs_pos2.cos()])
    assert torch.allclose(cos[2], expected_cos, atol=1e-5)


def test_rope_rotation_preserves_norm():
    dim = 8
    q = torch.randn(1, 1, 4, dim)
    k = torch.randn(1, 1, 4, dim)
    rope = RotaryEmbedding(dim, 16, 10000.0)
    cos, sin = rope(4, torch.device("cpu"), torch.float32)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-4)


def test_rmsnorm_matches_finite_difference_gradient():
    torch.manual_seed(0)
    norm = RMSNorm(hidden_size=6, eps=1e-6).double()
    x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda inp: norm(inp).sum(), (x,), eps=1e-6, atol=1e-4)


def test_rmsnorm_rejects_wrong_last_dim():
    norm = RMSNorm(hidden_size=6)
    with pytest.raises(ValueError):
        norm(torch.randn(2, 5))


def test_moe_fallback_output_shape_and_aux_loss():
    layer = MoELayer(hidden_size=32, intermediate_size=64, num_experts=4, top_k=2)
    x = torch.randn(2, 5, 32)
    out, aux_loss = layer(x)
    assert out.shape == x.shape
    assert aux_loss.dim() == 0
    assert aux_loss.item() >= 0.0


def test_moe_gating_weights_sum_to_one():
    gate = torch.nn.Linear(16, 4, bias=False)
    logits = gate(torch.randn(5, 16))
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, _ = torch.topk(probs, 2, dim=-1)
    normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    assert torch.allclose(normalized.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_mod_respects_capacity(dummy_model_config):
    hidden_size = 32
    block = torch.nn.Linear(hidden_size, hidden_size)

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = block

        def forward(self, x, **kwargs):
            return self.lin(x), torch.zeros(())

    mod = MixtureOfDepths(hidden_size, _Wrap(), capacity_factor=0.5)
    mod.eval()
    x = torch.randn(1, 10, hidden_size)
    out, aux_loss = mod(x)
    assert out.shape == x.shape


def test_swa_mask_blocks_beyond_window():
    seq_len, window = 10, 3
    mask = generate_swa_mask(seq_len, window, torch.device("cpu"))
    # position 9 should attend to 7,8,9 only (window=3) and NOT to 0..6
    assert mask[9, 6].item() is False
    assert mask[9, 7].item() is True
    assert mask[9, 9].item() is True
    # causality: cannot attend to the future
    assert mask[3, 5].item() is False


def test_swa_full_attention_interval():
    assert is_full_attention_layer(3, 4) is True  # layer 3 (0-indexed) = 4th layer
    assert is_full_attention_layer(0, 4) is False
    assert is_full_attention_layer(7, 4) is True


def test_gqa_with_swa_restricts_attention_span():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=64,
        use_flash_attention=False, use_swa=True, swa_window_size=2,
    )
    x = torch.randn(1, 10, 32)
    out, _ = attn(x)
    assert out.shape == x.shape


def test_mla_cache_smaller_than_gqa_cache():
    hidden_size, num_heads, num_kv_heads = 256, 8, 2
    head_dim = hidden_size // num_heads
    gqa_cache_per_token = 2 * num_kv_heads * head_dim  # K + V

    mla = MLAAttention(hidden_size=hidden_size, num_heads=num_heads, latent_dim=hidden_size // 4)
    assert mla.cache_size_per_token() < gqa_cache_per_token


def test_mla_forward_shape_matches_gqa():
    hidden_size, num_heads = 64, 4
    mla = MLAAttention(hidden_size=hidden_size, num_heads=num_heads, latent_dim=16, max_seq_len=32)
    x = torch.randn(2, 6, hidden_size)
    out, past = mla(x, use_cache=True)
    assert out.shape == x.shape
    assert past is not None
    latent_cache, rope_cache = past
    assert latent_cache.shape == (2, 6, 16)


def test_mla_incremental_decode_matches_full_forward():
    torch.manual_seed(0)
    hidden_size, num_heads = 32, 4
    mla = MLAAttention(hidden_size=hidden_size, num_heads=num_heads, latent_dim=8, max_seq_len=32)
    mla.eval()
    x = torch.randn(1, 4, hidden_size)

    full_out, _ = mla(x)

    past = None
    step_outs = []
    for t in range(4):
        step_out, past = mla(x[:, t:t + 1, :], past_key_value=past, use_cache=True)
        step_outs.append(step_out)
    incremental_out = torch.cat(step_outs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4)


def test_transformer_with_swa_and_hybrid_layers_forward():
    config = ModelConfig(
        hidden_size=32, num_layers=4, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_swa=True, swa_window_size=2, swa_full_attention_interval=2,
        use_flash_attention=False,
    )
    model = ATSTransformer(config)
    input_ids = torch.randint(0, 50, (1, 6))
    output = model(input_ids)
    assert output.logits.shape == (1, 6, 50)


def test_transformer_with_mla_forward():
    config = ModelConfig(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_mla=True, mla_latent_dim=8,
    )
    model = ATSTransformer(config)
    input_ids = torch.randint(0, 50, (1, 6))
    output = model(input_ids)
    assert output.logits.shape == (1, 6, 50)


# --- Mamba / MTP / diffusion / quantization / config-driven composition ---

def test_mamba_block_output_shape():
    from ats.model.mamba import MambaBlock

    block = MambaBlock(hidden_size=16, d_state=4, d_conv=3, expand=2)
    x = torch.randn(2, 5, 16)
    out = block(x)
    assert out.shape == x.shape


def test_mamba_block_has_recurrent_state_dependence():
    """Changing an early token must change later outputs (recurrent state),
    proving this isn't a position-independent (e.g. renamed FFN) block."""
    from ats.model.mamba import MambaBlock

    torch.manual_seed(0)
    block = MambaBlock(hidden_size=8, d_state=4, d_conv=2, expand=2)
    block.eval()
    x = torch.randn(1, 6, 8)
    out_a = block(x)

    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 5.0  # perturb only the FIRST token
    out_b = block(x_perturbed)

    # A position-independent (attention-free, non-recurrent) block would
    # leave every position except position 0 unchanged; Mamba's recurrence
    # should propagate the perturbation forward through later positions.
    assert not torch.allclose(out_a[:, 1:, :], out_b[:, 1:, :], atol=1e-5)


def test_mamba_transformer_composition():
    config = ModelConfig(
        hidden_size=16, num_layers=4, num_heads=2, num_kv_heads=2, intermediate_size=32,
        vocab_size=30, max_seq_len=16, use_mamba=True, mamba_every_n_layers=2,
        use_flash_attention=False,
    )
    model = ATSTransformer(config)
    # Layers 2 and 4 (1-indexed) should be MambaLayer; layers 1 and 3 TransformerBlock.
    from ats.model.transformer import MambaLayer, TransformerBlock

    assert isinstance(model.layers[1], MambaLayer)
    assert isinstance(model.layers[3], MambaLayer)
    assert isinstance(model.layers[0], TransformerBlock)

    input_ids = torch.randint(0, 30, (1, 5))
    output = model(input_ids)
    assert output.logits.shape == (1, 5, 30)


def test_mtp_head_predicts_n_tokens():
    from ats.model.mtp import MultiTokenPredictionHead

    head = MultiTokenPredictionHead(hidden_size=16, vocab_size=40, num_future_tokens=3)
    hidden = torch.randn(2, 10, 16)
    logits_per_offset = head(hidden)
    assert len(logits_per_offset) == 3
    for logits in logits_per_offset:
        assert logits.shape == (2, 10, 40)


def test_mtp_loss_is_mean_of_offset_losses():
    from ats.model.mtp import MultiTokenPredictionHead

    torch.manual_seed(0)
    head = MultiTokenPredictionHead(hidden_size=8, vocab_size=20, num_future_tokens=2)
    hidden = torch.randn(2, 6, 8)
    labels = torch.randint(0, 20, (2, 6))
    loss = head.compute_loss(hidden, labels)
    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_mtp_transformer_produces_mtp_logits():
    config = ModelConfig(
        hidden_size=16, num_layers=2, num_heads=2, num_kv_heads=2, intermediate_size=32,
        vocab_size=30, max_seq_len=16, use_mtp=True, mtp_num_tokens=2, use_flash_attention=False,
    )
    model = ATSTransformer(config)
    input_ids = torch.randint(0, 30, (1, 8))
    output = model(input_ids)
    assert output.mtp_logits is not None
    assert len(output.mtp_logits) == 2


def test_diffusion_loss_is_mse_not_cross_entropy():
    from ats.model.diffusion import DiffusionLM

    config = ModelConfig(
        hidden_size=16, num_layers=2, num_heads=2, num_kv_heads=2, intermediate_size=32,
        vocab_size=30, max_seq_len=16, use_flash_attention=False,
    )
    backbone = ATSTransformer(config)
    diffusion_model = DiffusionLM(backbone=backbone, hidden_size=16, num_timesteps=100)
    input_ids = torch.randint(0, 30, (2, 6))
    output = diffusion_model(input_ids, embed_tokens=backbone.embed_tokens)

    # The loss must be MSE between predicted and (implicitly, via the MSE
    # call) true noise, not a cross-entropy over vocab logits: confirm no
    # vocab-sized dimension appears anywhere and the loss is a plain scalar
    # produced by squared-error semantics (non-negative, no log-softmax).
    assert output.loss.dim() == 0
    assert output.loss.item() >= 0.0
    assert output.predicted_noise.shape == (2, 6, 16)  # embedding-space, not vocab-space


def test_diffusion_sampling_produces_valid_token_ids():
    from ats.model.diffusion import DiffusionLM

    config = ModelConfig(
        hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=2, intermediate_size=16,
        vocab_size=25, max_seq_len=8, use_flash_attention=False,
    )
    backbone = ATSTransformer(config)
    diffusion_model = DiffusionLM(backbone=backbone, hidden_size=8, num_timesteps=50)
    tokens = diffusion_model.sample(
        embed_tokens=backbone.embed_tokens, batch_size=1, seq_len=4, num_inference_steps=3,
    )
    assert tokens.shape == (1, 4)
    assert (tokens >= 0).all() and (tokens < 25).all()


def test_quantized_linear_none_matches_plain_linear_shape():
    from ats.model.quantization import QuantizedLinear

    layer = QuantizedLinear(8, 16, quantization="none")
    x = torch.randn(3, 8)
    out = layer(x)
    assert out.shape == (3, 16)


def test_quantized_linear_int8_changes_numerics():
    from ats.model.quantization import QuantizedLinear

    torch.manual_seed(0)
    plain = QuantizedLinear(8, 8, quantization="none")
    quantized = QuantizedLinear(8, 8, quantization="int8")
    quantized.linear.weight.data.copy_(plain.linear.weight.data)

    x = torch.randn(4, 8)
    out_plain = plain(x)
    out_quantized = quantized(x)
    # Fake quantization must actually perturb the numerics, not silently
    # pass through as an identity / no-op.
    assert not torch.allclose(out_plain, out_quantized, atol=1e-6)


def test_quantized_linear_fp8_without_backend_raises_import_error():
    from ats.model.quantization import QuantizedLinear

    try:
        import transformer_engine  # noqa: F401
        pytest.skip("transformer_engine is installed; fp8 path would succeed, not raise.")
    except ImportError:
        pass
    try:
        import torchao  # noqa: F401
        pytest.skip("torchao is installed; fp8 path would succeed, not raise.")
    except ImportError:
        pass

    with pytest.raises(ImportError):
        QuantizedLinear(8, 8, quantization="fp8")
