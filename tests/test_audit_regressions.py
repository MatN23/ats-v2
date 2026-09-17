"""Regression tests for the bugs found in the BUG-1xx audit pass.

Each test here is written to FAIL against the pre-fix implementation, not
merely to exercise the code. Where the pre-fix behaviour can be recreated
inside the test (by monkeypatching the specific predicate that was wrong),
the test asserts on the resulting numerical difference rather than on the
shape of the code.
"""

from __future__ import annotations

import contextlib
import gc
import unittest.mock

import pytest
import torch
import torch.nn.functional as F

from ats.config.schema import (
    ATSConfig,
    CheckpointConfig,
    DataConfig,
    DataSource,
    ModelConfig,
    TrainingConfig,
)
from ats.model.attention import GroupedQueryAttention, can_use_flash_attention
from ats.model.mamba import MambaBlock
from ats.model.mla import MLAAttention
from ats.model.mod import MixtureOfDepths
from ats.model.transformer import ATSTransformer
from ats.training.adaptive_controller import AdaptiveController
from ats.training.scheduler import WarmupCosineScheduler
from ats.training.trainer import Trainer


class _FakeEngineWrapper:
    """Stands in for a DeepSpeed model_engine in Trainer.train_step tests,
    without needing DeepSpeed: forward/backward run through a REAL
    ATSTransformer (so the MTP loss path is exercised authentically, not
    mocked), while step()/get_global_grad_norm()/optimizer are lightweight
    stand-ins for bookkeeping DeepSpeed would otherwise own.

    Duplicated from tests/test_bug_audit_fixes.py rather than imported from
    it: `tests/` has no __init__.py, so `from tests.test_bug_audit_fixes
    import ...` is not a reliable cross-environment import -- it depends on
    how pytest happens to insert rootdir onto sys.path, which varies by
    invocation. This is exactly what broke CI: `python -m pytest` (used
    while verifying this locally) implicitly adds the current directory to
    sys.path, so the dotted import worked there; plain `pytest` (what CI
    actually runs) does not, so it failed with `ModuleNotFoundError: No
    module named 'tests'`. See the identical note and duplication in
    test_bug_audit_fixes.py's own _TinyModelEngine.
    """

    def __init__(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> None:
        self.module = model
        self.optimizer = optimizer
        self.local_rank = torch.device("cpu")

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def parameters(self):
        return self.module.parameters()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def get_global_grad_norm(self):
        return None

    def eval(self) -> None:
        self.module.eval()

    def train(self) -> None:
        self.module.train()


def _make_mtp_trainer(tmp_path) -> Trainer:
    """Builds a real Trainer for train_step tests, bypassing __init__'s
    initialize_engine() call (the only piece that actually requires
    DeepSpeed) in favor of _FakeEngineWrapper around a real model.

    Duplicated from tests/test_bug_audit_fixes.py -- see
    _FakeEngineWrapper's docstring above for why.
    """
    config = ATSConfig(
        model=ModelConfig(
            hidden_size=16,
            num_layers=2,
            num_heads=2,
            num_kv_heads=2,
            intermediate_size=32,
            vocab_size=30,
            max_seq_len=16,
            use_mtp=True,
            mtp_num_tokens=2,
            use_flash_attention=False,
        ),
        training=TrainingConfig(
            max_steps=10, learning_rate=1e-3, warmup_steps=1, grad_accum_steps=1
        ),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path)),
    )
    model = ATSTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.model_engine = _FakeEngineWrapper(model, optimizer)
    trainer.optimizer = optimizer
    trainer.grad_accum_steps = 1
    trainer.scheduler = WarmupCosineScheduler(
        base_lr=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr_ratio=config.training.min_lr_ratio,
    )
    trainer.checkpoint_manager = None
    trainer.monitor = None
    trainer.adaptive_controller = AdaptiveController(config.adaptive)
    trainer._adaptive_lr_multiplier = 1.0
    trainer._max_adaptive_multiplier = config.adaptive.max_lr_multiplier
    trainer._min_adaptive_multiplier = config.adaptive.min_lr_multiplier
    trainer._adaptive_multiplier_decay = config.adaptive.lr_multiplier_decay
    trainer.global_step = 0
    trainer.epoch = 0
    trainer._accumulation_step = 0
    trainer._accumulated_tokens = 0
    return trainer


# ---------------------------------------------------------------------------
# BUG-101: the flash_attn dispatch condition ignored attention_mask, so a
# padded batch on CUDA/fp16 ran with causal=False and NO mask at all.
# ---------------------------------------------------------------------------


def _fake_flash_attn_func(q, k, v, dropout_p=0.0, causal=False, **kwargs):
    """Stand-in with flash_attn's real API surface: q/k/v are [b, s, h, d]
    and there is NO attn_mask parameter. That absence is the whole bug --
    anything the caller passes as attention_mask simply cannot reach the
    kernel.
    """
    assert "attn_mask" not in kwargs, "flash_attn_func has no attn_mask argument"
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize(
    "mask, past, expected",
    [
        (None, None, True),
        ("mask", None, False),  # BUG-101: this was True before the fix
        (None, "past", False),
        ("mask", "past", False),
    ],
)
def test_flash_dispatch_requires_no_mask_and_no_cache(mask, past, expected):
    assert (
        can_use_flash_attention(
            flash_enabled=True,
            is_cuda=True,
            dtype=torch.float16,
            attention_mask=mask,
            past_key_value=past,
        )
        is expected
    )


def test_flash_dispatch_rejects_fp32_and_cpu():
    assert not can_use_flash_attention(True, True, torch.float32, None, None)
    assert not can_use_flash_attention(True, False, torch.float16, None, None)
    assert not can_use_flash_attention(False, True, torch.float16, None, None)


def test_padded_batch_stays_causal_even_when_flash_is_available(monkeypatch):
    """End-to-end proof of BUG-101's impact.

    The flash branch is CUDA-only, so this test forces the branch by
    monkeypatching the dispatch predicate, then compares the two predicate
    behaviours directly:

      * the OLD predicate (ignores attention_mask) sends the call to the
        flash stub with causal=False and no mask -> position 0's output
        changes when a LATER token changes, i.e. attention leaked backwards
        in time.
      * the FIXED predicate refuses the flash path when a mask is present,
        so SDPA gets an explicit padding+causal mask and position 0 is
        unaffected by later tokens.
    """
    import ats.model.attention as attention_mod

    monkeypatch.setattr(attention_mod, "flash_attn_func", _fake_flash_attn_func)
    monkeypatch.setattr(attention_mod, "_FLASH_ATTN_AVAILABLE", True)

    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=16,
        use_flash_attention=True,
    )
    attn.eval()
    attn.use_flash_attention = True  # normally gated on the real package

    x = torch.randn(1, 6, 32)
    x_perturbed = x.clone()
    x_perturbed[0, -1] += 5.0
    mask = torch.ones(1, 6, dtype=torch.long)

    def first_position_delta():
        with torch.no_grad():
            a, _ = attn(x, attention_mask=mask)
            b, _ = attn(x_perturbed, attention_mask=mask)
        return (a[0, 0] - b[0, 0]).abs().max().item()

    # Pre-fix predicate: identical except it never looked at attention_mask.
    monkeypatch.setattr(
        attention_mod,
        "can_use_flash_attention",
        lambda flash_enabled, is_cuda, dtype, attention_mask, past_key_value: (
            flash_enabled and past_key_value is None
        ),
    )
    buggy_delta = first_position_delta()
    assert buggy_delta > 1e-4, (
        "test scaffolding is wrong: the recreated pre-fix path should have "
        "leaked future information into position 0"
    )

    # Restore the real predicate.
    monkeypatch.undo()
    monkeypatch.setattr(attention_mod, "flash_attn_func", _fake_flash_attn_func)
    monkeypatch.setattr(attention_mod, "_FLASH_ATTN_AVAILABLE", True)
    attn.use_flash_attention = True
    fixed_delta = first_position_delta()
    assert fixed_delta < 1e-6, (
        f"position 0 changed by {fixed_delta} when a LATER token changed -- "
        "attention is not causal on the masked/flash path (BUG-101)"
    )


# ---------------------------------------------------------------------------
# BUG-102: MLA dropped the sliding window during incremental decoding.
# ---------------------------------------------------------------------------


def test_mla_applies_sliding_window_during_incremental_decoding():
    """With swa_window_size=2, a decode step must not see cached positions
    further back than the window. Before the fix, MLA built its incremental
    mask with no window_size at all and attended to the whole cache.
    """
    torch.manual_seed(0)
    mla = MLAAttention(
        hidden_size=32,
        num_heads=4,
        latent_dim=8,
        max_seq_len=32,
        use_swa=True,
        swa_window_size=2,
    )
    mla.eval()

    prefix = torch.randn(1, 5, 32)
    new_token = torch.randn(1, 1, 32)

    def decode_with(prefix_tensor):
        with torch.no_grad():
            _, cache = mla(prefix_tensor, use_cache=True)
            out, _ = mla(new_token, past_key_value=cache, use_cache=True)
        return out

    baseline = decode_with(prefix)

    far_past = prefix.clone()
    far_past[0, 0] += 10.0  # distance 6 from the new token, window is 2
    out_far = decode_with(far_past)

    near_past = prefix.clone()
    near_past[0, -1] += 10.0  # distance 1, inside the window
    out_near = decode_with(near_past)

    far_delta = (baseline - out_far).abs().max().item()
    near_delta = (baseline - out_near).abs().max().item()

    assert near_delta > 1e-4, (
        "changing a cached position INSIDE the window had no effect -- the "
        "test is not exercising the attention path it thinks it is"
    )
    assert far_delta < 1e-6, (
        f"changing a cached position {far_delta} outside the sliding window "
        "changed the decode output; SWA is not applied during MLA decoding "
        "(BUG-102)"
    )


def test_gqa_and_mla_agree_on_windowed_decode_reachability():
    """The windowing contract must be the same for both attention backends:
    a cached position beyond swa_window_size is unreachable in either.
    """
    from ats.model.attention import build_incremental_causal_mask

    mask = build_incremental_causal_mask(
        seq_len=1, past_len=5, device=torch.device("cpu"), window_size=2
    )
    assert mask.shape == (1, 6)
    # Only the new token (index 5) and index 4 are within distance < 2.
    assert mask[0].tolist() == [False, False, False, False, True, True]


# ---------------------------------------------------------------------------
# BUG-103: Mixture-of-Depths ran the wrapped block on the full sequence.
# ---------------------------------------------------------------------------


class _RecordingBlock(torch.nn.Module):
    """Records the sequence length it is actually invoked with."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(hidden_size, hidden_size)
        self.seen_shapes: list[tuple[int, ...]] = []
        self.seen_masks: list[torch.Tensor | None] = []

    def forward(
        self, x, attention_mask=None, past_key_value=None, use_cache=False, causal=True
    ):
        self.seen_shapes.append(tuple(x.shape))
        self.seen_masks.append(attention_mask)
        return self.lin(x), torch.zeros(()), None


@pytest.mark.parametrize(
    "seq_len, capacity_factor, expected_capacity",
    [
        (16, 0.25, 4),  # capacity well below seq_len
        (16, 0.5, 8),
        (16, 1.0, 16),  # capacity == seq_len
        (1, 0.5, 1),  # degenerate: max(1, ...) floor
        (3, 0.5, 1),  # very short sequence
        (7, 0.75, 5),  # non-divisible
    ],
)
def test_mod_block_receives_exactly_capacity_tokens(
    seq_len, capacity_factor, expected_capacity
):
    """BUG-103: the block used to be called with the FULL seq_len every
    time, so Mixture-of-Depths saved no computation whatsoever.
    """
    hidden = 16
    block = _RecordingBlock(hidden)
    mod = MixtureOfDepths(hidden, block, capacity_factor=capacity_factor)
    mod.train()
    x = torch.randn(3, seq_len, hidden)
    out, _aux, _past = mod(x)

    assert out.shape == x.shape
    assert len(block.seen_shapes) == 1
    batch_seen, seq_seen, hidden_seen = block.seen_shapes[0]
    assert batch_seen == 3
    assert hidden_seen == hidden
    assert seq_seen == expected_capacity, (
        f"block ran on {seq_seen} positions but capacity is {expected_capacity} "
        f"(seq_len={seq_len}) -- MoD is not actually skipping computation"
    )


def test_mod_leaves_unselected_tokens_bitwise_unchanged():
    """Only the gathered positions may be modified; everything else must be
    the identity, not a blended approximation of it.
    """
    hidden = 8
    seq_len = 12

    class _Constant(torch.nn.Module):
        def forward(self, x, **kwargs):
            return torch.full_like(x, 99.0), torch.zeros(()), None

    torch.manual_seed(0)
    mod = MixtureOfDepths(hidden, _Constant(), capacity_factor=0.25)
    mod.eval()
    x = torch.randn(2, seq_len, hidden)
    with torch.no_grad():
        gate_probs = torch.sigmoid(mod.gate(x).squeeze(-1))
        capacity = max(1, int(0.25 * seq_len))
        selected = torch.topk(gate_probs, capacity, dim=1).indices
        out, _aux, _past = mod(x)

    for b in range(2):
        sel = set(selected[b].tolist())
        for t in range(seq_len):
            if t in sel:
                assert torch.allclose(out[b, t], torch.full((hidden,), 99.0))
            else:
                assert torch.equal(out[b, t], x[b, t]), (
                    f"unselected position {t} was modified; it should have "
                    "bypassed the block entirely"
                )


def test_mod_gathers_the_padding_mask_alongside_the_tokens():
    """If the mask were not gathered, position i of the compressed sequence
    would carry position i's ORIGINAL pad flag rather than the flag of the
    token that actually landed in that slot.
    """
    hidden = 8
    seq_len = 10
    block = _RecordingBlock(hidden)
    mod = MixtureOfDepths(hidden, block, capacity_factor=0.5)
    mod.train()
    x = torch.randn(1, seq_len, hidden)
    mask = torch.ones(1, seq_len, dtype=torch.long)
    mask[0, 7:] = 0
    mod(x, attention_mask=mask)

    seen = block.seen_masks[0]
    assert seen is not None
    assert seen.shape == (1, 5), "mask was not compressed alongside the tokens"

    with torch.no_grad():
        gate_probs = torch.sigmoid(mod.gate(x).squeeze(-1))
        selected, _ = torch.sort(torch.topk(gate_probs, 5, dim=1).indices, dim=1)
    assert torch.equal(seen, mask.gather(1, selected))


def test_mod_indices_are_sorted_so_causality_is_preserved():
    """The gathered subsequence must keep original relative order, or a
    token could attend to one that came after it.
    """
    hidden = 8
    captured = {}

    class _OrderProbe(torch.nn.Module):
        def forward(self, x, **kwargs):
            captured["x"] = x
            return x, torch.zeros(()), None

    torch.manual_seed(3)
    mod = MixtureOfDepths(hidden, _OrderProbe(), capacity_factor=0.5)
    mod.train()
    # Make each position uniquely identifiable by its first feature.
    x = torch.zeros(1, 10, hidden)
    x[0, :, 0] = torch.arange(10, dtype=torch.float32)
    x[0, :, 1:] = torch.randn(10, hidden - 1)
    mod(x)
    order = captured["x"][0, :, 0].tolist()
    assert order == sorted(order), f"gathered positions out of order: {order}"


def test_mod_gate_still_receives_gradient_through_selected_tokens():
    hidden = 8
    block = _RecordingBlock(hidden)
    mod = MixtureOfDepths(hidden, block, capacity_factor=0.5)
    mod.train()
    x = torch.randn(2, 10, hidden, requires_grad=True)
    out, aux, _ = mod(x)
    (out.sum() + aux).backward()

    assert mod.gate.weight.grad is not None
    assert torch.isfinite(mod.gate.weight.grad).all()
    assert mod.gate.weight.grad.abs().sum() > 0, "gate got no gradient signal"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Every position must get SOME input gradient: unselected ones through
    # the identity path, selected ones through the block as well.
    assert (x.grad.abs().sum(dim=-1) > 0).all()


def test_mod_falls_back_to_dense_for_cached_generation():
    """Documented, deliberate exception: a compressed subsequence cannot
    produce a continuable KV cache. Assert it is dense rather than silently
    returning a mis-indexed cache.
    """
    hidden = 8
    block = _RecordingBlock(hidden)
    mod = MixtureOfDepths(hidden, block, capacity_factor=0.25)
    mod.eval()
    x = torch.randn(1, 12, hidden)
    mod(x, use_cache=True)
    assert block.seen_shapes[0][1] == 12


def test_mod_end_to_end_through_transformer_with_padding_and_gradients():
    config = ModelConfig(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        vocab_size=50,
        max_seq_len=32,
        use_mod=True,
        mod_capacity_factor=0.5,
        use_flash_attention=False,
    )
    torch.manual_seed(0)
    model = ATSTransformer(config)
    model.train()
    input_ids = torch.randint(0, 50, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    mask[1, 9:] = 0
    out = model(input_ids, attention_mask=mask)
    (out.logits.sum() + out.aux_loss).backward()
    assert out.logits.shape == (2, 12, 50)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# ---------------------------------------------------------------------------
# BUG-104: the Mamba scan materialised every intermediate state and retained
# one [b, L, L, d_inner, d_state] tensor per chunk.
# ---------------------------------------------------------------------------


def _live_tensor_bytes() -> int:
    gc.collect()
    total = 0
    for obj in gc.get_objects():
        # gc.get_objects() can return objects mid-teardown (e.g. a tensor
        # whose storage was already freed), where isinstance/attribute
        # access on it can raise. This is a defensive best-effort memory
        # measurement, not correctness-critical code, so a stray object of
        # that kind is skipped rather than failing the test.
        if not torch.is_tensor(obj):
            continue
        try:
            total += obj.numel() * obj.element_size()
        except RuntimeError:  # pragma: no cover - freed/invalid storage
            continue
    return total


def test_mamba_scan_returns_contracted_output_not_the_full_state_tensor():
    block = MambaBlock(hidden_size=16, d_state=8, expand=2, chunk_size=4)
    b, s = 2, 9
    dt = torch.rand(b, s, block.d_inner) + 0.1
    B = torch.randn(b, s, 8)
    C = torch.randn(b, s, 8)
    x_conv = torch.randn(b, s, block.d_inner)
    A = -torch.exp(block.A_log)
    y, carry = block._chunked_scan(dt, A, B, C, x_conv)
    assert y.shape == (b, s, block.d_inner), (
        "scan still returns a per-state tensor; it must contract with C "
        "inside the chunk (BUG-104)"
    )
    assert carry.shape == (b, block.d_inner, 8)


def test_mamba_scan_matches_sequential_recurrence_in_float64():
    """Independent reference: the plain O(seq_len) recurrence. float64 so
    the tolerance can be tight enough to catch real algebra errors rather
    than absorbing them.
    """
    torch.manual_seed(0)
    d_state = 8
    block = MambaBlock(
        hidden_size=32, d_state=d_state, d_conv=3, expand=2, chunk_size=7
    ).double()
    b, s = 2, 33  # not a multiple of chunk_size, on purpose

    x = torch.randn(b, s, 32, dtype=torch.float64)
    x_main, _gate = block.in_proj(x).chunk(2, dim=-1)
    padded = F.pad(x_main.transpose(1, 2), (block.d_conv - 1, 0))
    x_conv = F.silu(block.conv1d(padded).transpose(1, 2))
    proj = block.x_proj(x_conv)
    B, C, dt_raw = torch.split(proj, [d_state, d_state, 1], dim=-1)
    dt = F.softplus(block.dt_proj(dt_raw))
    A = -torch.exp(block.A_log)

    state = torch.zeros(b, block.d_inner, d_state, dtype=torch.float64)
    ys = []
    for t in range(s):
        decay = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))
        state = decay * state + (dt[:, t, :] * x_conv[:, t, :]).unsqueeze(-1) * B[
            :, t, :
        ].unsqueeze(1)
        ys.append(torch.einsum("bdn,bn->bd", state, C[:, t, :]))
    reference = torch.stack(ys, dim=1)

    got_y, got_carry = block._chunked_scan(dt, A, B, C, x_conv)
    assert (reference - got_y).abs().max().item() < 1e-12
    assert (state - got_carry).abs().max().item() < 1e-12


def test_mamba_retained_activations_do_not_scale_with_chunk_count():
    """BUG-104's dominant term: the per-chunk decay-ratio tensor is
    [batch, L, L, d_inner, d_state] and einsum saved one of them for EVERY
    chunk, so retained memory was num_chunks x a single chunk's worth. With
    per-chunk recompute the retained total must stay within a small
    multiple of ONE chunk, independent of sequence length.
    """
    b, s, hidden, d_state, chunk = 2, 512, 128, 16, 32
    block = MambaBlock(hidden_size=hidden, d_state=d_state, expand=2, chunk_size=chunk)
    x = torch.randn(b, s, hidden, requires_grad=True)

    before = _live_tensor_bytes()
    y = block(x)
    after = _live_tensor_bytes()
    retained = after - before

    one_chunk_decay_ratio = b * chunk * chunk * block.d_inner * d_state * 4
    num_chunks = s // chunk
    pre_fix_estimate = one_chunk_decay_ratio * num_chunks

    assert retained < 4 * one_chunk_decay_ratio, (
        f"retained {retained / 1e6:.1f} MB; a single chunk's decay-ratio "
        f"tensor is {one_chunk_decay_ratio / 1e6:.1f} MB and the pre-fix "
        f"implementation retained ~{pre_fix_estimate / 1e6:.1f} MB"
    )
    # Sanity: the graph is still real and differentiable.
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mamba_gradients_survive_per_chunk_recompute():
    """Per-chunk torch.utils.checkpoint must not change gradients. Compare
    against the same block run with a single chunk covering the whole
    sequence (no cross-chunk carry, different code path through the loop).
    """
    torch.manual_seed(0)
    block = MambaBlock(hidden_size=16, d_state=8, expand=2, chunk_size=4).double()
    x = torch.randn(2, 16, 16, dtype=torch.float64, requires_grad=True)

    y_chunked = block(x)
    y_chunked.sum().backward()
    chunked_grad = x.grad.clone()
    chunked_param_grad = block.A_log.grad.clone()

    x.grad = None
    block.zero_grad(set_to_none=True)
    block.chunk_size = 64  # one chunk for the whole sequence
    y_single = block(x)
    y_single.sum().backward()

    assert (y_chunked - y_single).abs().max().item() < 1e-12
    assert (chunked_grad - x.grad).abs().max().item() < 1e-12
    assert (chunked_param_grad - block.A_log.grad).abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
# BUG-105: hardcoded torch.device(f"cuda:{local_rank}") made CPU/MPS
# training impossible.
# ---------------------------------------------------------------------------


class _EngineStub:
    """Mimics the device-reporting surface of a DeepSpeedEngine. Note the
    int local_rank: that is what DeepSpeed actually sets, including on a
    CPU-only run. The existing suite's fake engine set local_rank to a
    torch.device, which is exactly why this bug never showed up in CI.
    """

    def __init__(self, device=None, local_rank=None):
        if device is not None:
            self.device = device
        if local_rank is not None:
            self.local_rank = local_rank


def test_resolve_device_prefers_the_engines_own_device():
    from ats.utils.device import resolve_device

    engine = _EngineStub(device=torch.device("cpu"), local_rank=3)
    assert resolve_device(engine) == torch.device("cpu")
    assert resolve_device(_EngineStub(device="cpu")) == torch.device("cpu")


def test_resolve_device_does_not_invent_cuda_on_a_cpu_only_host():
    """The pre-fix expression produced torch.device("cuda:0") here, which
    constructs fine and then fails at the first .to(device) call.
    """
    from ats.utils.device import preferred_accelerator, resolve_device

    engine = _EngineStub(local_rank=0)  # int local_rank, no .device
    resolved = resolve_device(engine)
    assert resolved.type == preferred_accelerator()
    if not torch.cuda.is_available():
        assert resolved.type != "cuda", (
            "resolve_device returned a CUDA device on a host with no CUDA"
        )


def test_resolve_device_handles_a_torch_device_local_rank():
    from ats.utils.device import resolve_device

    assert resolve_device(_EngineStub(local_rank=torch.device("cpu"))) == torch.device(
        "cpu"
    )


def test_resolve_device_handles_an_engine_with_no_device_attributes():
    from ats.utils.device import preferred_accelerator, resolve_device

    assert resolve_device(_EngineStub()).type == preferred_accelerator()


def test_module_device_reports_parameter_placement():
    from ats.utils.device import module_device

    assert module_device(torch.nn.Linear(2, 2)) == torch.device("cpu")
    assert module_device(torch.nn.Identity()) == torch.device("cpu")


def test_trainer_train_step_runs_with_an_int_local_rank_engine(tmp_path):
    """End-to-end: a CPU engine that reports local_rank as an int (the real
    DeepSpeed shape) must train. Pre-fix this raised on the first
    .to(device) because device was torch.device("cuda:0").
    """
    trainer = _make_mtp_trainer(tmp_path)
    engine = trainer.model_engine
    assert isinstance(engine, _FakeEngineWrapper)
    engine.local_rank = 0  # int, as DeepSpeed reports it
    if hasattr(engine, "device"):
        del engine.device

    batch = {
        "input_ids": torch.randint(0, 30, (2, 8)),
        "labels": torch.randint(0, 30, (2, 8)),
    }
    metrics = trainer.train_step(batch)
    assert metrics is not None
    assert torch.isfinite(torch.tensor(metrics.loss))


# ---------------------------------------------------------------------------
# BUG-108: the fallback grad norm was measured after step() cleared grads.
# ---------------------------------------------------------------------------


def test_grad_norm_is_measured_before_gradients_are_cleared(tmp_path):
    """The engine stub deliberately returns None from
    get_global_grad_norm(), the case the fallback exists for. Pre-fix the
    fallback ran after step() had zeroed the gradients and therefore
    reported ~0.0 on every step forever -- which also meant
    AdaptiveController's gradient-explosion check could never fire.
    """
    trainer = _make_mtp_trainer(tmp_path)
    batch = {
        "input_ids": torch.randint(0, 30, (2, 8)),
        "labels": torch.randint(0, 30, (2, 8)),
    }
    # First step: the engine has not yet been observed to return None, so
    # the degraded value is expected and warned about.
    trainer.train_step(batch)
    trainer.global_step += 1
    metrics = trainer.train_step(batch)
    assert metrics is not None
    assert metrics.grad_norm > 1e-8, (
        f"grad_norm reported as {metrics.grad_norm}; gradients were measured "
        "after they had already been cleared (BUG-108)"
    )


def test_grad_norm_tracker_uses_the_engine_reported_norm_when_available():
    from ats.training.trainer import _GradNormTracker

    class _Reporting:
        def get_global_grad_norm(self):
            return 4.25

        def parameters(self):  # pragma: no cover - must never be reached
            raise AssertionError("manual norm computed despite engine reporting one")

    tracker = _GradNormTracker(_Reporting())
    assert tracker.pre_step() is None
    assert tracker.post_step(None) == 4.25
    # Still no manual pass on subsequent steps.
    assert tracker.pre_step() is None


# ---------------------------------------------------------------------------
# BUG-107: AdaptiveController inputs were rank-local.
# ---------------------------------------------------------------------------


def test_distributed_helpers_are_no_ops_on_a_single_process():
    from ats.training.trainer import _distributed_mean, _reduce_expert_utilization

    value = torch.tensor(3.5)
    assert _distributed_mean(value).item() == pytest.approx(3.5)
    util = {0: 0.25, 1: 0.75}
    assert _reduce_expert_utilization(util, torch.device("cpu")) == util
    assert _reduce_expert_utilization(None, torch.device("cpu")) is None


# ---------------------------------------------------------------------------
# BUG-118: the preprocessed-block reader was three Python passes per example.
# These tests pin the vectorized replacement to the exact output of the
# original list-based implementation, for both padding sides and the edge
# cases (fully valid block, fully padded block, single token).
# ---------------------------------------------------------------------------


IGNORE_INDEX = -100


def _reference_block_and_labels(block_row, valid_len, seq_length, padding_side):
    """The ORIGINAL implementation, kept verbatim as the reference."""
    block = list(block_row)
    labels = list(block)
    if padding_side == "right":
        for i in range(valid_len, seq_length):
            labels[i] = IGNORE_INDEX
    else:
        pad_len = seq_length - valid_len
        for i in range(pad_len):
            labels[i] = IGNORE_INDEX
    return block, labels


def _write_preprocessed(tmp_path, blocks, valid_lengths, padding_side):
    import json

    import numpy as np

    from ats.data.dataset import PREPROCESSED_TOKEN_DTYPE

    out = tmp_path / padding_side
    out.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(blocks, dtype=PREPROCESSED_TOKEN_DTYPE)
    arr.tofile(out / "tokens.bin")
    np.save(out / "valid_lengths.npy", np.asarray(valid_lengths, dtype=np.int64))
    meta = {
        "seq_length": arr.shape[1],
        "num_blocks": arr.shape[0],
        "padding_side": padding_side,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return out / "tokens.bin"


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_preprocessed_block_output_matches_the_original_implementation(
    tmp_path, padding_side
):
    from ats.data.dataset import _iter_preprocessed_examples

    seq_length = 6
    blocks = [
        [1, 2, 3, 4, 5, 6],  # fully valid
        [7, 8, 9, 0, 0, 0],  # partially padded
        [0, 0, 0, 0, 0, 11],  # almost entirely padded
        [12, 13, 14, 15, 16, 17],  # fully valid again
    ]
    valid_lengths = [6, 3, 1, 6]
    bin_path = _write_preprocessed(tmp_path, blocks, valid_lengths, padding_side)

    produced = list(_iter_preprocessed_examples(bin_path, seq_length))
    assert len(produced) == len(blocks)

    for example, row, valid_len in zip(produced, blocks, valid_lengths):
        ref_block, ref_labels = _reference_block_and_labels(
            row, valid_len, seq_length, padding_side
        )
        assert list(example["input_ids"]) == ref_block, (
            f"input_ids diverged from the original implementation "
            f"({padding_side} padding)"
        )
        assert list(example["labels"]) == ref_labels, (
            f"labels diverged from the original implementation ({padding_side} padding)"
        )
        # Tokens themselves are never masked -- only labels are.
        assert list(example["input_ids"]) == list(row)


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_preprocessed_block_edge_cases(tmp_path, padding_side):
    from ats.data.dataset import _iter_preprocessed_examples

    seq_length = 1
    blocks = [[5], [6]]
    valid_lengths = [1, 0]  # one fully valid, one fully padded
    bin_path = _write_preprocessed(tmp_path, blocks, valid_lengths, padding_side)
    produced = list(_iter_preprocessed_examples(bin_path, seq_length))

    for example, row, valid_len in zip(produced, blocks, valid_lengths):
        ref_block, ref_labels = _reference_block_and_labels(
            row, valid_len, seq_length, padding_side
        )
        assert list(example["input_ids"]) == ref_block
        assert list(example["labels"]) == ref_labels

    # A zero-valid-length block must have every label masked out.
    assert list(produced[1]["labels"]) == [IGNORE_INDEX]


def test_preprocessed_sharding_still_partitions_blocks_disjointly(tmp_path):
    from ats.data.dataset import _iter_preprocessed_examples

    seq_length = 4
    blocks = [[i, i, i, i] for i in range(10)]
    bin_path = _write_preprocessed(tmp_path, blocks, [4] * 10, "right")

    seen = []
    for shard_id in range(3):
        for example in _iter_preprocessed_examples(
            bin_path, seq_length, shard_id=shard_id, num_shards=3
        ):
            seen.append(next(iter(example["input_ids"])))
    assert sorted(seen) == list(range(10)), "sharding dropped or duplicated blocks"


def test_collate_accepts_both_array_and_list_examples():
    """The raw-text path still yields Python lists; the preprocessed path
    now yields numpy arrays. _collate must handle both identically.
    """
    import numpy as np

    from ats.data.dataloader import _collate

    as_lists = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, -100]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6]},
    ]
    as_arrays = [
        {
            "input_ids": np.asarray(ex["input_ids"], dtype=np.int64),
            "labels": np.asarray(ex["labels"], dtype=np.int64),
        }
        for ex in as_lists
    ]
    a = _collate(as_lists)
    b = _collate(as_arrays)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["labels"], b["labels"])
    assert a["input_ids"].dtype == torch.long
    assert b["labels"].dtype == torch.long


# ---------------------------------------------------------------------------
# BUG-123: optional heavy-dependency import guards (deepspeed's MoE layer,
# flash_attn, triton) only caught ImportError. Reproduced live during this
# audit: deepspeed 0.19.6 installed against torch 2.5.1 raised a bare
# ValueError from deep inside deepspeed's own import chain (its
# torch.library.custom_op registration), which propagated straight through
# ats/model/moe.py's `except ImportError` and crashed the import of every
# caller of that module -- not just MoE users. The fallback these guards
# exist for never got a chance to run.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _reloaded_with_broken_import(target_module, import_name, exc):
    """Simulates `import <import_name>` raising `exc` instead of succeeding,
    reloads target_module so its module-level try/except re-runs against
    the broken import, yields the reloaded module, and then restores the
    EXACT original module object afterwards -- not a second fresh import.

    This matters: ats.model.__init__ and ats.model.transformer import
    classes (e.g. MoELayer) FROM these modules at import time and keep
    their own reference. Popping a module from sys.modules and reimporting
    it creates a NEW, distinct class object; every isinstance() check
    elsewhere in the process that closed over the original class then
    silently starts failing for the rest of the test session. Restoring
    the original module object (rather than importing a fresh one) is what
    makes this test hermetic.
    """
    import builtins
    import importlib
    import sys

    # Ensure target_module is actually loaded before capturing "the
    # original" -- run in isolation (e.g. `pytest
    # tests/test_audit_regressions.py` on its own), a module like
    # ats.model.mla_triton may never have been imported by anything else
    # yet, so sys.modules would not have an entry to restore.
    if target_module not in sys.modules:
        importlib.import_module(target_module)
    original_module = sys.modules[target_module]
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == import_name or name.startswith(import_name + "."):
            raise exc
        return real_import(name, *args, **kwargs)

    try:
        with unittest.mock.patch("builtins.__import__", side_effect=fake_import):
            del sys.modules[target_module]
            reloaded = importlib.import_module(target_module)
            yield reloaded
    finally:
        sys.modules[target_module] = original_module


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("infer_schema(func): unsupported parameter type"),
        TypeError("unexpected keyword argument"),
        AttributeError("module has no attribute 'custom_op'"),
    ],
)
def test_moe_survives_a_non_import_error_from_deepspeed(exc):
    """This is the exact failure mode reproduced live: deepspeed installed,
    but its own import chain raises something other than ImportError.
    ats.model.moe must still import and fall back, not crash.
    """
    with _reloaded_with_broken_import("ats.model.moe", "deepspeed", exc) as module:
        assert module._DEEPSPEED_MOE_AVAILABLE is False
        assert module.DeepSpeedMoE is None
        # And the fallback actually works, not just the flag. Built via
        # this reloaded module's OWN MoELayer/_PyTorchMoEFallback classes
        # (not the ones imported at the top of this test file), since
        # those are now two distinct class objects for the duration of
        # this `with` block.
        layer = module.MoELayer(
            hidden_size=8,
            intermediate_size=16,
            num_experts=2,
            num_layers=1,
            top_k=1,
        )
        assert layer.uses_deepspeed is False
        out, aux = layer(torch.randn(1, 3, 8))
        assert out.shape == (1, 3, 8)
        assert torch.isfinite(aux)


def test_moe_still_falls_back_cleanly_on_plain_import_error():
    """The original, narrower behaviour must be unchanged: a plain
    ImportError (deepspeed simply not installed) still falls back exactly
    as before.
    """
    with _reloaded_with_broken_import(
        "ats.model.moe", "deepspeed", ImportError("no module")
    ) as module:
        assert module._DEEPSPEED_MOE_AVAILABLE is False
        assert module.DeepSpeedMoE is None


@pytest.mark.parametrize(
    "module_name,import_name",
    [
        ("ats.model.attention", "flash_attn"),
        ("ats.model.moe_triton", "triton"),
        ("ats.model.rope_triton", "triton"),
        ("ats.model.norm_triton", "triton"),
        ("ats.model.mla_triton", "triton"),
    ],
)
def test_optional_dependency_guards_survive_non_import_errors(module_name, import_name):
    """Same bug class, audited across every module with the same pattern:
    flash_attn and each Triton module. A version-mismatch failure inside
    the optional dependency's own import machinery must not propagate.
    """
    exc = RuntimeError("simulated incompatible native extension")
    with _reloaded_with_broken_import(module_name, import_name, exc) as module:
        flag_name = "_FLASH_ATTN_AVAILABLE" if "flash" in import_name else "HAS_TRITON"
        assert getattr(module, flag_name) is False
