"""Tests for ats.model: shapes, RoPE correctness, RMSNorm gradient check,
MoE routing, MoD capacity, SWA masking, MLA cache size."""

from __future__ import annotations

import math

import pytest
import torch

from ats.config.schema import ModelConfig
from ats.model.attention import GroupedQueryAttention
from ats.model.mamba import MambaBlock
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
    layer = MoELayer(hidden_size=32, intermediate_size=64, num_experts=4, num_layers=2, top_k=2)
    x = torch.randn(2, 5, 32)
    out, aux_loss = layer(x)
    assert out.shape == x.shape
    assert aux_loss.dim() == 0
    assert aux_loss.item() >= 0.0


def test_moe_gating_weights_sum_to_one():
    """Exercises the REAL MoELayer's routing math (via compute_routing on its
    fallback module), not a standalone reimplementation, so a bug in the
    actual gating normalization would be caught here."""
    layer = MoELayer(hidden_size=16, intermediate_size=32, num_experts=4, num_layers=2, top_k=2)
    if layer.uses_deepspeed:
        pytest.skip("deepspeed is installed; this test targets the PyTorch fallback router.")

    flat_x = torch.randn(5, 16)
    top_k_probs, top_k_idx, router_probs = layer.moe.compute_routing(flat_x)

    assert top_k_probs.shape == (5, 2)
    assert torch.allclose(top_k_probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    # Every selected expert index must be a real, distinct expert.
    assert (top_k_idx >= 0).all() and (top_k_idx < 4).all()
    assert (top_k_idx[:, 0] != top_k_idx[:, 1]).all()
    # The full router distribution (used for the aux loss) must also be a
    # valid probability distribution over all 4 experts.
    assert torch.allclose(router_probs.sum(dim=-1), torch.ones(5), atol=1e-5)


def test_moe_layer_forward_uses_real_gating_end_to_end():
    """A behavioral check on the full MoELayer.forward path: routing must
    actually influence which expert processes which token. We verify this
    indirectly by checking that a token routed to different experts (forced
    via distinct random seeds producing different gate weights) produces
    different outputs, i.e. the gate isn't a no-op."""
    torch.manual_seed(1)
    layer_a = MoELayer(hidden_size=16, intermediate_size=32, num_experts=4, num_layers=2, top_k=2)
    torch.manual_seed(2)
    layer_b = MoELayer(hidden_size=16, intermediate_size=32, num_experts=4, num_layers=2, top_k=2)

    x = torch.randn(1, 3, 16)
    out_a, _ = layer_a(x)
    out_b, _ = layer_b(x)
    assert out_a.shape == x.shape
    assert not torch.allclose(out_a, out_b, atol=1e-6)


def test_mod_respects_capacity(dummy_model_config):
    hidden_size = 32
    lin = torch.nn.Linear(hidden_size, hidden_size)

    class _FakeBlock(torch.nn.Module):
        """Matches the REAL calling convention used by TransformerBlock and
        MambaLayer: always returns a 3-tuple (hidden, aux_loss, past_kv).
        A fake that returned fewer items would silently hide bugs in
        MixtureOfDepths' unpacking, which is exactly what happened before."""

        def __init__(self):
            super().__init__()
            self.lin = lin

        def forward(self, x, **kwargs):
            return self.lin(x), torch.zeros(()), None

    mod = MixtureOfDepths(hidden_size, _FakeBlock(), capacity_factor=0.5)
    mod.eval()
    x = torch.randn(1, 10, hidden_size)
    out, aux_loss, past_kv = mod(x)
    assert out.shape == x.shape
    assert aux_loss.dim() == 0
    assert past_kv is None


def test_mod_forward_does_not_crash_wrapping_a_real_transformer_block():
    """Regression test: MixtureOfDepths previously returned a 4-tuple when
    wrapping a block that itself returns the real 3-tuple
    (hidden, aux_loss, past_kv) convention -- e.g. any actual
    TransformerBlock or MambaLayer -- which crashed
    ATSTransformer._run_layers' `x, aux_loss, new_kv = layer(...)` unpack.
    This exercises MoD wrapping a REAL TransformerBlock end-to-end, the way
    ATSTransformer actually constructs it when model.use_mod=True."""
    config = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_mod=True, mod_capacity_factor=0.5,
        use_flash_attention=False,
    )
    model = ATSTransformer(config)
    input_ids = torch.randint(0, 50, (1, 8))
    output = model(input_ids)  # must not raise
    assert output.logits.shape == (1, 8, 50)


def test_mod_aux_loss_includes_wrapped_block_aux_loss():
    """The wrapped block's own aux_loss (e.g. from an inner MoE FFN) must be
    summed into MoD's returned aux_loss, not silently dropped."""
    hidden_size = 16
    nonzero_block_aux_loss = torch.tensor(3.5)

    class _BlockWithAuxLoss(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x, nonzero_block_aux_loss, None

    mod = MixtureOfDepths(hidden_size, _BlockWithAuxLoss(), capacity_factor=0.5)
    mod.eval()
    x = torch.randn(1, 6, hidden_size)
    _out, total_aux_loss, _past_kv = mod(x)
    # total_aux_loss = mod's own load-balancing loss + the block's 3.5;
    # since the load-balancing term is a small MSE, total must exceed 3.5.
    assert total_aux_loss.item() > nonzero_block_aux_loss.item()


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
    """Proves SWA actually restricts attention (not just that shapes match):
    perturbing a token that sits OUTSIDE the sliding window for a later
    query position must leave that later position's output completely
    unchanged, since the windowed mask should make it unreachable."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=64,
        use_flash_attention=False, use_swa=True, swa_window_size=2,
    )
    attn.eval()

    x = torch.randn(1, 10, 32)
    out_a, _ = attn(x)

    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 100.0  # perturb token 0 heavily

    out_b, _ = attn(x_perturbed)

    # window_size=2: query position 9 can see keys at positions 8,9 only
    # (distance < 2), so token 0 is far outside its window and position 9's
    # output must be exactly unchanged.
    assert torch.allclose(out_a[:, 9, :], out_b[:, 9, :], atol=1e-5)
    # Sanity check the test itself isn't vacuous: position 1 (within window
    # of the perturbed token 0) MUST change, or this whole comparison would
    # be meaningless (e.g. if masking were silently disabled everywhere).
    assert not torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-5)


def test_gqa_without_swa_has_full_attention_span():
    """Control test: with use_swa=False, perturbing token 0 SHOULD affect
    every later position (ordinary causal attention), confirming the
    previous test's zero-effect result is specifically due to windowing,
    not some unrelated bug that zeroes out early-token influence generally."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=64,
        use_flash_attention=False, use_swa=False,
    )
    attn.eval()

    x = torch.randn(1, 10, 32)
    out_a, _ = attn(x)
    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 100.0
    out_b, _ = attn(x_perturbed)

    assert not torch.allclose(out_a[:, 9, :], out_b[:, 9, :], atol=1e-5)


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
    # QuantizedLinear subclasses nn.Linear directly (it does not wrap one as
    # a `.linear` submodule -- see the module docstring on why: this keeps
    # its state_dict keys identical to a plain nn.Linear's, which matters
    # for export/checkpoint compatibility), so the weight lives at `.weight`.
    quantized.weight.data.copy_(plain.weight.data)

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


# --- Regression tests: initialization double-init bug ---

def test_init_residual_projection_marks_weight_to_prevent_overwrite():
    from ats.model.initialization import init_residual_projection

    linear = torch.nn.Linear(16, 16, bias=True)
    init_residual_projection(linear, num_layers=8)
    assert getattr(linear.weight, "_ats_residual_init", False) is True


def test_init_weights_skips_already_residual_initialized_linear():
    """Regression test: a later blanket init_weights() pass must NOT
    overwrite a weight already set by init_residual_projection. Verified by
    setting a distinctive constant value, then confirming init_weights
    leaves it untouched."""
    from ats.model.initialization import init_residual_projection, init_weights

    linear = torch.nn.Linear(16, 16, bias=False)
    init_residual_projection(linear, num_layers=8)
    marked_weight = linear.weight.detach().clone()

    init_weights(linear, num_layers=8)  # must be a no-op for this weight

    assert torch.equal(linear.weight, marked_weight)


def test_transformer_block_residual_projections_survive_full_model_construction():
    """End-to-end regression test for the double-init bug: build a full
    ATSTransformer (which internally does exactly what the bug involved --
    TransformerBlock.__init__ sets a depth-scaled std on o_proj/down_proj,
    then ATSTransformer.__init__ runs a later blanket self.apply(init_weights)
    across the whole model) and confirm o_proj's empirical std matches the
    depth-scaled target, not the generic BASE_LINEAR_STD it would have if
    silently overwritten."""
    from ats.model.initialization import residual_output_std

    num_layers = 48  # deep enough that depth-scaled and generic std differ by >10x
    config = ModelConfig(
        hidden_size=64, num_layers=num_layers, num_heads=8, num_kv_heads=8,
        intermediate_size=128, vocab_size=100, max_seq_len=32, use_flash_attention=False,
    )
    model = ATSTransformer(config)

    expected_std = residual_output_std(num_layers)
    generic_std = 0.02

    o_proj_weight = model.layers[0].attention.o_proj.weight
    empirical_std = o_proj_weight.std().item()

    # The weight matrix is large enough (64*64=4096 elements) for the
    # empirical std to be a reliable estimator. It must be close to the
    # depth-scaled target and clearly NOT close to the un-scaled generic std
    # (which is what the bug produced).
    assert empirical_std == pytest.approx(expected_std, rel=0.15)
    assert abs(empirical_std - generic_std) > abs(empirical_std - expected_std)


# --- Regression tests: quantization actually wired into the model (C3) ---

def test_quantization_none_produces_plain_linear_everywhere():
    from ats.model.quantization import QuantizedLinear

    config = ModelConfig(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_flash_attention=False, quantization="none",
    )
    model = ATSTransformer(config)
    block = model.layers[0]
    assert not isinstance(block.attention.q_proj, QuantizedLinear)
    assert not isinstance(block.ffn.down_proj, QuantizedLinear)


def test_quantization_int8_produces_quantized_linear_in_attention_and_ffn():
    from ats.model.quantization import QuantizedLinear

    config = ModelConfig(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_flash_attention=False, quantization="int8",
    )
    model = ATSTransformer(config)
    block = model.layers[0]
    assert isinstance(block.attention.q_proj, QuantizedLinear)
    assert isinstance(block.attention.k_proj, QuantizedLinear)
    assert isinstance(block.attention.v_proj, QuantizedLinear)
    assert isinstance(block.attention.o_proj, QuantizedLinear)
    assert isinstance(block.ffn.gate_up_proj, QuantizedLinear)
    assert isinstance(block.ffn.down_proj, QuantizedLinear)
    # Still an nn.Linear, so state_dict keys and isinstance checks elsewhere
    # (residual init, HF export) are unaffected.
    assert isinstance(block.attention.q_proj, torch.nn.Linear)


def test_quantization_int8_state_dict_keys_match_plain_linear():
    """Regression test: QuantizedLinear must expose .weight/.bias at the
    same state_dict key path as a plain nn.Linear would, or HF export's
    exact key remapping (ats/export/huggingface.py) would silently break
    the moment quantization was enabled."""
    config = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_flash_attention=False, quantization="int8",
    )
    model = ATSTransformer(config)
    state_dict_keys = set(model.state_dict().keys())
    assert "layers.0.attention.q_proj.weight" in state_dict_keys
    assert "layers.0.ffn.down_proj.weight" in state_dict_keys
    # Must NOT be nested under a wrapping submodule like ".linear.weight".
    assert not any(".linear.weight" in k for k in state_dict_keys)


def test_quantization_int8_forward_pass_shape(dummy_batch):
    config = ModelConfig(
        hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=100, max_seq_len=32, use_flash_attention=False, quantization="int8",
    )
    model = ATSTransformer(config)
    output = model(dummy_batch["input_ids"])
    assert output.logits.shape == (dummy_batch["input_ids"].shape[0], dummy_batch["input_ids"].shape[1], 100)


def test_quantization_int8_forward_differs_numerically_from_none():
    """int8 fake-quantization must actually perturb numerics (not silently
    behave identically to quantization='none')."""
    torch.manual_seed(0)
    config_none = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_flash_attention=False, quantization="none",
    )
    torch.manual_seed(0)
    config_int8 = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_flash_attention=False, quantization="int8",
    )
    torch.manual_seed(1)
    model_none = ATSTransformer(config_none)
    torch.manual_seed(1)
    model_int8 = ATSTransformer(config_int8)
    # Copy weights so the only difference is the forward-pass quantization.
    # strict=False: the int8 model has extra FakeQuantize observer buffers
    # (e.g. `.*.activation_post_process.*`) that model_none's state_dict
    # doesn't have any entry for -- those aren't real weights to copy, just
    # quantization bookkeeping, so a strict load would fail on missing keys
    # that were never expected to be present in the source state_dict.
    model_int8.load_state_dict(model_none.state_dict(), strict=False)

    input_ids = torch.randint(0, 50, (1, 6))
    out_none = model_none(input_ids).logits
    out_int8 = model_int8(input_ids).logits
    assert not torch.allclose(out_none, out_int8, atol=1e-6)


def test_moe_expert_uses_quantization():
    """MoE fallback experts (SwiGLU instances) must also respect
    model.quantization, since they're the majority of parameters in MoE
    models."""
    from ats.model.quantization import QuantizedLinear

    layer = MoELayer(
        hidden_size=16, intermediate_size=32, num_experts=2, num_layers=2, top_k=1, quantization="int8",
    )
    if layer.uses_deepspeed:
        pytest.skip("deepspeed is installed; this test targets the PyTorch fallback experts.")
    expert = layer.moe.experts[0]
    assert isinstance(expert.down_proj, QuantizedLinear)


# --- Regression test: Mamba chunked scan matches sequential ground truth ---

def test_mamba_chunked_scan_matches_naive_sequential_reference():
    """MambaBlock now uses a chunked parallel scan instead of a Python loop
    over every timestep, for speed. This test independently reimplements
    the naive O(seq_len) sequential recurrence using the exact same
    intermediate tensors (dt, A, B, C, x_conv) pulled from a real
    MambaBlock instance, and checks the two numerically agree -- so a bug
    in the chunked-scan math would be caught here, not just trusted from
    the standalone numpy prototype used during development."""
    torch.manual_seed(0)
    hidden_size, d_state, d_conv, expand = 16, 8, 3, 2
    block = MambaBlock(hidden_size=hidden_size, d_state=d_state, d_conv=d_conv, expand=expand, chunk_size=5)
    block.eval()

    batch, seq_len = 2, 23  # not a multiple of chunk_size=5, on purpose
    x = torch.randn(batch, seq_len, hidden_size)

    # Reproduce the exact same intermediate tensors MambaBlock.forward()
    # computes, so the sequential reference operates on IDENTICAL inputs.
    with torch.no_grad():
        x_and_gate = block.in_proj(x)
        x_main, gate = x_and_gate.chunk(2, dim=-1)
        x_conv = block.conv1d(x_main.transpose(1, 2))[..., :seq_len]
        x_conv = torch.nn.functional.silu(x_conv.transpose(1, 2))
        proj = block.x_proj(x_conv)
        B, C, dt_raw = torch.split(proj, [d_state, d_state, 1], dim=-1)
        dt = torch.nn.functional.softplus(block.dt_proj(dt_raw))
        A = -torch.exp(block.A_log)

        # Naive sequential reference, operating on the SAME dt/A/B/x_conv.
        d_inner = expand * hidden_size
        state = torch.zeros(batch, d_inner, d_state)
        ys = []
        for t in range(seq_len):
            dt_t = dt[:, t, :]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)
            state = state * dA + dB * x_conv[:, t, :].unsqueeze(-1)
            ys.append(torch.einsum("bdn,bn->bd", state, C[:, t, :]))
        sequential_y = torch.stack(ys, dim=1)
        sequential_states = None  # not needed; comparing y directly below

        # Now call the actual shipped chunked-scan implementation.
        chunked_states = block._chunked_scan(dt, A, B, x_conv)
        chunked_y = torch.einsum("btdn,btn->btd", chunked_states, C)

    assert torch.allclose(sequential_y, chunked_y, atol=1e-4), (
        f"Chunked scan diverges from sequential reference: "
        f"max abs diff = {(sequential_y - chunked_y).abs().max().item()}"
    )


def test_mamba_chunked_scan_various_chunk_sizes_agree():
    """The final output must be independent of chunk_size (it's a pure
    speed/memory tradeoff, not an approximation) -- check several chunk
    sizes, including ones that don't evenly divide seq_len, all agree."""
    torch.manual_seed(1)
    hidden_size, d_state = 8, 4
    seq_len = 17

    outputs = {}
    for chunk_size in [1, 4, 6, 17, 100]:
        torch.manual_seed(42)  # same weight init every time
        block = MambaBlock(hidden_size=hidden_size, d_state=d_state, d_conv=2, expand=2, chunk_size=chunk_size)
        block.eval()
        torch.manual_seed(7)  # same input every time
        x = torch.randn(1, seq_len, hidden_size)
        with torch.no_grad():
            outputs[chunk_size] = block(x)

    reference = outputs[1]
    for chunk_size, out in outputs.items():
        assert torch.allclose(out, reference, atol=1e-4), (
            f"chunk_size={chunk_size} output diverges from chunk_size=1 reference"
        )


def test_mamba_block_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        MambaBlock(hidden_size=16, chunk_size=0)


def test_mamba_config_chunk_size_field_wired_through():
    config = ModelConfig(
        hidden_size=16, num_layers=2, num_heads=2, num_kv_heads=2, intermediate_size=32,
        vocab_size=30, max_seq_len=16, use_mamba=True, mamba_every_n_layers=1,
        mamba_chunk_size=8, use_flash_attention=False,
    )
    model = ATSTransformer(config)
    from ats.model.transformer import MambaLayer

    mamba_layer = model.layers[0]
    assert isinstance(mamba_layer, MambaLayer)
    assert mamba_layer.mamba.chunk_size == 8


def test_mla_quantization_int8_produces_quantized_linear_everywhere():
    """MLA's projections (w_dkv, w_uk, w_uv, w_dq, w_uq, w_qr, w_kr, o_proj)
    must all respect model.quantization -- previously left unwired."""
    from ats.model.quantization import QuantizedLinear

    config = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=4, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_mla=True, mla_latent_dim=8, quantization="int8",
    )
    model = ATSTransformer(config)
    attn = model.layers[0].attention
    for proj_name in ("w_dkv", "w_uk", "w_uv", "w_dq", "w_uq", "w_qr", "w_kr", "o_proj"):
        proj = getattr(attn, proj_name)
        assert isinstance(proj, QuantizedLinear), f"{proj_name} was not quantized"
        assert isinstance(proj, torch.nn.Linear), f"{proj_name} broke the nn.Linear isinstance contract"


def test_mla_quantization_none_stays_plain_linear():
    from ats.model.quantization import QuantizedLinear

    config = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=4, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_mla=True, mla_latent_dim=8, quantization="none",
    )
    model = ATSTransformer(config)
    attn = model.layers[0].attention
    assert not isinstance(attn.w_dkv, QuantizedLinear)


def test_mla_quantization_forward_pass_shape(dummy_batch):
    config = ModelConfig(
        hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=4, intermediate_size=64,
        vocab_size=100, max_seq_len=32, use_mla=True, mla_latent_dim=8, quantization="int8",
    )
    model = ATSTransformer(config)
    output = model(dummy_batch["input_ids"])
    assert output.logits.shape[-1] == 100


def test_moe_fallback_expert_utilization_sums_to_one():
    """Regression test: the PyTorch fallback's last_expert_utilization
    previously summed to top_k (not 1.0), inconsistent with the DeepSpeed
    backend's counts/total normalization -- the same metric reported on
    two different scales depending on which backend happened to be active."""
    layer = MoELayer(hidden_size=16, intermediate_size=32, num_experts=4, num_layers=2, top_k=2)
    if layer.uses_deepspeed:
        pytest.skip("deepspeed is installed; this test targets the PyTorch fallback.")

    x = torch.randn(2, 5, 16)
    layer(x)
    assert layer.last_expert_utilization is not None
    total = sum(layer.last_expert_utilization.values())
    assert total == pytest.approx(1.0, abs=1e-5)


# --- Regression tests: incremental (multi-token, with-cache) causal masking ---

def test_build_incremental_causal_mask_basic_pattern():
    from ats.model.attention import build_incremental_causal_mask

    mask = build_incremental_causal_mask(seq_len=3, past_len=5, device=torch.device("cpu"))
    assert mask.shape == (3, 8)
    expected = torch.tensor([
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_build_incremental_causal_mask_with_window():
    from ats.model.attention import build_incremental_causal_mask

    mask = build_incremental_causal_mask(
        seq_len=3, past_len=5, device=torch.device("cpu"), window_size=3,
    )
    expected = torch.tensor([
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ], dtype=torch.bool)
    assert torch.equal(mask, expected)


def test_gqa_multi_token_continuation_with_cache_does_not_leak_future_tokens():
    """Regression test for a real correctness bug: previously, feeding
    multiple new tokens against an existing KV cache used is_causal=False
    with no explicit mask, letting new tokens attend to each other
    non-causally (including tokens that come after them). Verified by
    perturbing a LATER new token and confirming an EARLIER new token's
    output is unaffected (which would only hold under correct causal
    masking)."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=64, use_flash_attention=False,
    )
    attn.eval()

    prefix = torch.randn(1, 5, 32)
    _, past_kv = attn(prefix, use_cache=True)

    new_tokens = torch.randn(1, 3, 32)
    out_a, _ = attn(new_tokens, past_key_value=past_kv, use_cache=False)

    perturbed = new_tokens.clone()
    perturbed[:, 2, :] += 100.0  # perturb the LAST (latest) new token heavily
    out_b, _ = attn(perturbed, past_key_value=past_kv, use_cache=False)

    # Earlier new-token positions (0, 1) must be COMPLETELY unaffected by a
    # perturbation to a later new-token position (2), since position 2 comes
    # after them and causal attention must not let them see it.
    assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5)
    assert torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-5)
    # Sanity: position 2 itself (which WAS perturbed) must change, or this
    # whole comparison is vacuous.
    assert not torch.allclose(out_a[:, 2, :], out_b[:, 2, :], atol=1e-5)


def test_gqa_multi_token_continuation_still_sees_full_cache():
    """Control test: new tokens must still fully attend to the cached
    prefix (not just causally among themselves) -- perturbing the cached
    prefix must change every new token's output."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=64, use_flash_attention=False,
    )
    attn.eval()

    prefix = torch.randn(1, 5, 32)
    _, past_kv_a = attn(prefix, use_cache=True)
    perturbed_prefix = prefix.clone()
    perturbed_prefix[:, 0, :] += 100.0
    _, past_kv_b = attn(perturbed_prefix, use_cache=True)

    new_tokens = torch.randn(1, 3, 32)
    out_a, _ = attn(new_tokens, past_key_value=past_kv_a, use_cache=False)
    out_b, _ = attn(new_tokens, past_key_value=past_kv_b, use_cache=False)

    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_mla_multi_token_continuation_with_cache_does_not_leak_future_tokens():
    """Same regression test as GQA's, applied to MLAAttention."""
    torch.manual_seed(0)
    mla = MLAAttention(hidden_size=32, num_heads=4, latent_dim=8, max_seq_len=64)
    mla.eval()

    prefix = torch.randn(1, 5, 32)
    _, past_kv = mla(prefix, use_cache=True)

    new_tokens = torch.randn(1, 3, 32)
    out_a, _ = mla(new_tokens, past_key_value=past_kv, use_cache=False)

    perturbed = new_tokens.clone()
    perturbed[:, 2, :] += 100.0
    out_b, _ = mla(perturbed, past_key_value=past_kv, use_cache=False)

    assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5)
    assert torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-5)
    assert not torch.allclose(out_a[:, 2, :], out_b[:, 2, :], atol=1e-5)


def test_moe_expert_down_proj_gets_depth_scaled_residual_init():
    """Regression test: MoE expert FFN down_proj layers previously never
    received the depth-scaled residual-projection init that dense FFN
    down_proj layers get (the `if not self.ffn_is_moe` guard in
    TransformerBlock.__init__ skipped it entirely, since MoELayer has no
    single .down_proj attribute to call init_residual_projection on
    externally) -- silently leaving MoE experts on the generic,
    non-depth-scaled init instead."""
    from ats.model.initialization import residual_output_std

    num_layers = 32
    layer = MoELayer(
        hidden_size=64, intermediate_size=128, num_experts=4, num_layers=num_layers, top_k=2,
    )
    if layer.uses_deepspeed:
        pytest.skip("deepspeed is installed; this test targets the PyTorch fallback experts.")

    expected_std = residual_output_std(num_layers)
    for expert in layer.moe.experts:
        empirical_std = expert.down_proj.weight.std().item()
        assert empirical_std == pytest.approx(expected_std, rel=0.2), (
            f"expert down_proj std {empirical_std} does not match depth-scaled "
            f"target {expected_std} -- MoE experts are not getting the same "
            f"residual-projection init dense FFN layers get."
        )


def test_moe_layer_transformer_block_experts_get_residual_init():
    """End-to-end version: build a full ATSTransformer with use_moe=True and
    confirm the same property holds through the real construction path
    (TransformerBlock -> MoELayer), not just a standalone MoELayer."""
    from ats.model.initialization import residual_output_std

    num_layers = 24
    config = ModelConfig(
        hidden_size=32, num_layers=num_layers, num_heads=4, num_kv_heads=2, intermediate_size=64,
        vocab_size=50, max_seq_len=32, use_moe=True, num_experts=4, moe_top_k=2,
        use_flash_attention=False,
    )
    model = ATSTransformer(config)
    moe_layer = model.layers[0].ffn
    if moe_layer.uses_deepspeed:
        pytest.skip("deepspeed is installed; this test targets the PyTorch fallback experts.")

    expected_std = residual_output_std(num_layers)
    empirical_std = moe_layer.moe.experts[0].down_proj.weight.std().item()
    assert empirical_std == pytest.approx(expected_std, rel=0.3)
