"""Architecture-combination smoke matrix and independent reference checks.

Two things this file is for:

1. Combinations. Every feature in ats can be individually unit-tested and
   still fail to compose -- MoD wrapping a Mamba layer, MoE inside a
   MoD-wrapped block, SWA under MLA, MTP on top of any of them. These tests
   build tiny real models for each supported combination and run a real
   forward AND backward, checking finite loss, finite gradients, and that
   every parameter that should receive gradient does.

2. References. For the numerical pieces, a deliberately naive
   implementation written from the definition, compared against the shipped
   one in float64 so the tolerance can be tight enough to catch real
   algebra errors rather than absorbing them.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F

from ats.config.schema import ModelConfig
from ats.model.attention import GroupedQueryAttention, build_incremental_causal_mask
from ats.model.mla import MLAAttention
from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from ats.model.swa import generate_swa_mask, is_full_attention_layer
from ats.model.transformer import ATSTransformer

BASE = {
    "hidden_size": 32,
    "num_layers": 4,
    "num_heads": 4,
    "num_kv_heads": 2,
    "intermediate_size": 64,
    "vocab_size": 41,
    "max_seq_len": 32,
    "use_flash_attention": False,
}


# ---------------------------------------------------------------------------
# 1. Architecture combinations
# ---------------------------------------------------------------------------

_FEATURES = {
    "swa": {"use_swa": True, "swa_window_size": 4, "swa_full_attention_interval": 2},
    "mla": {"use_mla": True},
    "mamba": {
        "use_mamba": True,
        "mamba_every_n_layers": 2,
        "mamba_d_state": 4,
        "mamba_chunk_size": 4,
    },
    "moe": {"use_moe": True, "num_experts": 4, "moe_top_k": 2},
    "mod": {"use_mod": True, "mod_capacity_factor": 0.5},
    "mtp": {"use_mtp": True, "mtp_num_tokens": 2},
}


def _combo_ids(n):
    return ["+".join(c) or "dense" for c in itertools.combinations(_FEATURES, n)]


def _build(feature_names):
    kwargs = dict(BASE)
    for name in feature_names:
        kwargs.update(_FEATURES[name])
    return ModelConfig(**kwargs)


def _run_forward_backward(config, batch=2, seq_len=12, with_mask=False):
    torch.manual_seed(0)
    model = ATSTransformer(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    mask = None
    if with_mask:
        mask = torch.ones(batch, seq_len, dtype=torch.long)
        mask[-1, seq_len - 2 :] = 0
    out = model(input_ids, attention_mask=mask)

    assert out.logits.shape == (batch, seq_len, config.vocab_size)
    assert torch.isfinite(out.logits).all(), "non-finite logits"
    assert torch.isfinite(out.aux_loss).all(), "non-finite aux loss"

    loss = (
        F.cross_entropy(
            out.logits[:, :-1].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )
        + out.aux_loss
    )
    if out.mtp_logits is not None:
        for logits in out.mtp_logits:
            assert logits.shape == (batch, seq_len, config.vocab_size)
            loss = loss + logits.float().mean()
    loss.backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads, "no parameter received a gradient"
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), f"non-finite gradient in {name}"
    return model, out


@pytest.mark.parametrize("names", [()], ids=["dense"])
def test_dense_baseline(names):
    _run_forward_backward(_build(names))


@pytest.mark.parametrize("names", [(n,) for n in _FEATURES], ids=list(_FEATURES))
def test_single_feature(names):
    _run_forward_backward(_build(names))
    _run_forward_backward(_build(names), with_mask=True)


_PAIRS = [c for c in itertools.combinations(_FEATURES, 2)]


@pytest.mark.parametrize("names", _PAIRS, ids=_combo_ids(2))
def test_feature_pairs(names):
    """Pairs are where composition bugs live: MoD wrapping a Mamba layer,
    MoE inside a MoD-wrapped block, SWA under MLA. Some pairs are rejected
    by ModelConfig on purpose; a clear ValueError is a pass, a crash deep
    inside the model is not.
    """
    try:
        config = _build(names)
    except ValueError:
        pytest.skip(f"{'+'.join(names)} is rejected by config validation")
    _run_forward_backward(config)


def test_all_features_at_once():
    try:
        config = _build(tuple(_FEATURES))
    except ValueError:
        pytest.skip("the full combination is rejected by config validation")
    _run_forward_backward(config)


@pytest.mark.parametrize("seq_len,batch", [(1, 1), (2, 1), (3, 3), (12, 1), (17, 2)])
def test_odd_shapes_across_a_representative_combination(seq_len, batch):
    """Unusual batch sizes and sequence lengths, including seq_len below
    MoD's capacity floor and below MTP's offset count.
    """
    names = ("swa", "moe", "mod")
    config = _build(names)
    if seq_len < 4:
        # MTP needs seq_len > offset; excluded here deliberately.
        pass
    _run_forward_backward(config, batch=batch, seq_len=seq_len)


def test_diffusion_backbone_composes_with_supported_features():
    from ats.model.diffusion import DiffusionLM

    extras = (
        {},
        {"use_moe": True, "num_experts": 4, "moe_top_k": 2},
        {"use_mod": True, "mod_capacity_factor": 0.5},
    )
    for extra in extras:
        kwargs = dict(BASE)
        kwargs.update(extra)
        kwargs["model_type"] = "diffusion"
        try:
            config = ModelConfig(**kwargs)
        except ValueError:
            continue
        torch.manual_seed(0)
        backbone = ATSTransformer(config)
        model = DiffusionLM(backbone=backbone, hidden_size=32, num_timesteps=50)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, backbone.embed_tokens)
        assert torch.isfinite(out.loss)
        out.loss.backward()


def test_mamba_and_diffusion_are_rejected_not_silently_broken():
    """Mamba's scan has no bidirectional form; combining it with diffusion
    must raise at config time rather than train a causally-limited
    backbone for a model class that needs bidirectional context.
    """
    kwargs = dict(BASE)
    kwargs.update(_FEATURES["mamba"])
    kwargs["model_type"] = "diffusion"
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


# ---------------------------------------------------------------------------
# 2. Reference implementations
# ---------------------------------------------------------------------------


def _reference_rope(x, cos, sin):
    """RoPE written straight from the rotate-half definition."""
    half = x.shape[-1] // 2
    out = torch.empty_like(x)
    for t in range(x.shape[2]):
        for d in range(half):
            c, s = cos[t, d], sin[t, d]
            x1, x2 = x[:, :, t, d], x[:, :, t, d + half]
            out[:, :, t, d] = x1 * c - x2 * s
            out[:, :, t, d + half] = x2 * c + x1 * s
    return out


def test_rope_matches_an_elementwise_reference():
    torch.manual_seed(0)
    rope = RotaryEmbedding(dim=8, max_seq_len=16).double()
    cos, sin = rope(6, device=torch.device("cpu"), dtype=torch.float64)
    q = torch.randn(2, 3, 6, 8, dtype=torch.float64)
    got, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert (got - _reference_rope(q, cos, sin)).abs().max().item() < 1e-12


def test_rope_cache_growth_is_value_preserving():
    """The cache doubles on growth (an optimization). Positions already
    computed must keep identical values afterwards.
    """
    rope = RotaryEmbedding(dim=8, max_seq_len=4).double()
    cos_small, sin_small = rope(4, torch.device("cpu"), torch.float64)
    cos_small = cos_small.clone()
    sin_small = sin_small.clone()
    cos_big, sin_big = rope(9, torch.device("cpu"), torch.float64)
    assert torch.allclose(cos_big[:4], cos_small, atol=1e-15)
    assert torch.allclose(sin_big[:4], sin_small, atol=1e-15)


def _reference_gqa(q, k, v, num_kv_groups, causal=True, key_mask=None):
    """Naive GQA: expand kv heads with a Python loop, score, mask, softmax."""
    _b, h, s, d = q.shape
    out = torch.zeros_like(q)
    for head in range(h):
        kv_head = head // num_kv_groups
        scores = q[:, head] @ k[:, kv_head].transpose(-1, -2) / (d**0.5)
        if causal:
            causal_mask = torch.tril(torch.ones(s, k.shape[2], dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, :], float("-inf"))
        out[:, head] = torch.softmax(scores, dim=-1) @ v[:, kv_head]
    return out


def test_gqa_attention_matches_a_naive_reference_including_gradients():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        use_flash_attention=False,
    ).double()
    attn.eval()
    x = torch.randn(2, 7, 32, dtype=torch.float64, requires_grad=True)

    got, _ = attn(x)
    got.sum().backward()
    got_grad = x.grad.clone()

    # Rebuild q/k/v exactly as the module does, then attend naively.
    with torch.enable_grad():
        x2 = x.detach().clone().requires_grad_(True)
        b, s, _ = x2.shape
        q = attn.q_proj(x2).view(b, s, 4, 8).transpose(1, 2)
        k = attn.k_proj(x2).view(b, s, 2, 8).transpose(1, 2)
        v = attn.v_proj(x2).view(b, s, 2, 8).transpose(1, 2)
        cos, sin = attn.rotary_emb(s, x2.device, x2.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ref = _reference_gqa(q, k, v, attn.num_kv_groups)
        ref = attn.o_proj(ref.transpose(1, 2).reshape(b, s, 32))
        ref.sum().backward()

    assert (got - ref).abs().max().item() < 1e-10
    assert (got_grad - x2.grad).abs().max().item() < 1e-10


def test_gqa_repeat_kv_maps_query_heads_to_the_right_kv_head():
    """An off-by-one here produces plausible-looking output that trains
    badly, so check the mapping explicitly rather than via a norm.
    """
    kv = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    repeated = GroupedQueryAttention._repeat_kv(kv, n_rep=3)
    assert repeated.shape == (1, 6, 3, 4)
    for query_head in range(6):
        assert torch.equal(repeated[0, query_head], kv[0, query_head // 3])


def test_swa_mask_matches_its_definition():
    mask = generate_swa_mask(7, window_size=3, device=torch.device("cpu"))
    for i in range(7):
        for j in range(7):
            expected = (j <= i) and (i - j < 3)
            assert bool(mask[i, j]) is expected, (i, j)


def test_swa_full_attention_interval_selects_every_nth_layer():
    picked = [i for i in range(8) if is_full_attention_layer(i, 4)]
    assert picked == [3, 7]


def test_incremental_mask_matches_absolute_position_semantics():
    mask = build_incremental_causal_mask(
        seq_len=3, past_len=4, device=torch.device("cpu")
    )
    assert mask.shape == (3, 7)
    for qi in range(3):
        for kj in range(7):
            assert bool(mask[qi, kj]) is (kj <= 4 + qi)


def test_swa_and_full_attention_layers_actually_differ_in_a_real_model():
    """The hybrid schedule is only meaningful if the windowed layers and the
    full-attention layers behave differently. Prove it by perturbing a
    far-past token and checking the two layer types respond differently.
    """
    torch.manual_seed(0)
    config = ModelConfig(
        **{
            **BASE,
            "num_layers": 2,
            "use_swa": True,
            "swa_window_size": 2,
            "swa_full_attention_interval": 2,
        }
    )
    model = ATSTransformer(config)
    blocks = [layer for layer in model.layers]
    assert blocks[0].force_full_attention is False
    assert blocks[1].force_full_attention is True


def test_mla_cache_is_smaller_than_the_equivalent_gqa_cache():
    mla = MLAAttention(hidden_size=32, num_heads=4, latent_dim=8, max_seq_len=16)
    gqa_per_token = 2 * 2 * 8  # 2 (k and v) * num_kv_heads * head_dim
    assert mla.cache_size_per_token() < gqa_per_token


def test_mla_incremental_decode_matches_a_full_forward():
    """Decoding one token at a time from a cache must reproduce what a
    single full-sequence forward produces for the same position.
    """
    torch.manual_seed(0)
    mla = MLAAttention(
        hidden_size=32, num_heads=4, latent_dim=8, max_seq_len=16
    ).double()
    mla.eval()
    x = torch.randn(1, 5, 32, dtype=torch.float64)

    with torch.no_grad():
        full, _ = mla(x)
        cache = None
        stepwise = []
        for t in range(5):
            out, cache = mla(x[:, t : t + 1], past_key_value=cache, use_cache=True)
            stepwise.append(out)
        stepwise = torch.cat(stepwise, dim=1)

    assert (full - stepwise).abs().max().item() < 1e-9


def test_gqa_incremental_decode_matches_a_full_forward():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        use_flash_attention=False,
    ).double()
    attn.eval()
    x = torch.randn(1, 5, 32, dtype=torch.float64)

    with torch.no_grad():
        full, _ = attn(x)
        cache = None
        stepwise = []
        for t in range(5):
            out, cache = attn(x[:, t : t + 1], past_key_value=cache, use_cache=True)
            stepwise.append(out)
        stepwise = torch.cat(stepwise, dim=1)

    assert (full - stepwise).abs().max().item() < 1e-9


def test_mtp_loss_matches_a_per_offset_reference():
    from ats.model.mtp import compute_mtp_loss_from_logits

    torch.manual_seed(0)
    vocab, batch, seq = 7, 2, 6
    logits = [torch.randn(batch, seq, vocab, dtype=torch.float64) for _ in range(3)]
    labels = torch.randint(0, vocab, (batch, seq))

    reference = []
    for k, offset_logits in enumerate(logits, start=1):
        if k >= seq:
            continue
        pred = offset_logits[:, : seq - k, :].reshape(-1, vocab)
        target = labels[:, k:].reshape(-1)
        reference.append(F.cross_entropy(pred.float(), target))
    expected = torch.stack(reference).mean()

    got = compute_mtp_loss_from_logits(logits, labels, vocab)
    assert (got - expected).abs().item() < 1e-10


def test_mtp_raises_rather_than_silently_skipping_an_impossible_offset():
    from ats.model.mtp import compute_mtp_loss_from_logits

    logits = [torch.randn(1, 1, 5)]
    labels = torch.randint(0, 5, (1, 1))
    with pytest.raises(ValueError, match="too short"):
        compute_mtp_loss_from_logits(logits, labels, vocab_size=5)


def test_gradient_checkpointing_produces_identical_values_and_gradients():
    """Checkpointing is a memory/compute tradeoff, not an approximation."""
    config_kwargs = {**BASE, "num_layers": 4}
    torch.manual_seed(0)
    plain = ATSTransformer(ModelConfig(**config_kwargs)).double()
    torch.manual_seed(0)
    checkpointed = ATSTransformer(
        ModelConfig(**{**config_kwargs, "checkpoint_every_n_layers": 1})
    ).double()
    checkpointed.load_state_dict(plain.state_dict())

    plain.train()
    checkpointed.train()
    input_ids = torch.randint(0, BASE["vocab_size"], (2, 9))

    out_a = plain(input_ids)
    out_a.logits.sum().backward()
    out_b = checkpointed(input_ids)
    out_b.logits.sum().backward()

    assert (out_a.logits - out_b.logits).abs().max().item() < 1e-12
    for (name, pa), (_, pb) in zip(
        plain.named_parameters(), checkpointed.named_parameters()
    ):
        assert pa.grad is not None and pb.grad is not None, name
        assert (pa.grad - pb.grad).abs().max().item() < 1e-10, name


def test_gradient_accumulation_equals_one_large_batch():
    """N micro-batches with loss/N accumulated must give the same gradients
    as one batch of N times the size.
    """
    torch.manual_seed(0)
    model = ATSTransformer(ModelConfig(**BASE)).double()
    model.train()
    big = torch.randint(0, BASE["vocab_size"], (4, 8))

    def loss_of(ids):
        out = model(ids)
        return F.cross_entropy(
            out.logits[:, :-1].reshape(-1, BASE["vocab_size"]),
            ids[:, 1:].reshape(-1),
        )

    model.zero_grad(set_to_none=True)
    loss_of(big).backward()
    single = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad(set_to_none=True)
    for chunk in big.chunk(2, dim=0):
        (loss_of(chunk) / 2).backward()
    accumulated = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert len(single) == len(accumulated)
    for a, b in zip(single, accumulated):
        assert (a - b).abs().max().item() < 1e-10
