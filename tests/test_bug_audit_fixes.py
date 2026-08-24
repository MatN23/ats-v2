"""Regression tests for the ATS-v2 bug-audit fixes (BUG-001 through BUG-009,
plus two bugs found during verification of the audit that it didn't
report: MLAAttention not handling attention_mask at all, in either the
plain or incremental-decoding case).

Each test is paired with a comment naming which audit bug it guards
against. Two of the audit's nine claims (BUG-002, BUG-005) did not
reproduce against this codebase as described; those are noted rather than
"fixed" with tests that would just pass trivially regardless.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ats.config.schema import (
    ATSConfig,
    CheckpointConfig,
    ConfigError,
    DataConfig,
    DataSource,
    ModelConfig,
    TrainingConfig,
)
from ats.model.attention import GroupedQueryAttention
from ats.model.mla import MLAAttention
from ats.model.transformer import ATSTransformer
from ats.training.adaptive_controller import AdaptiveController
from ats.training.checkpoint import CheckpointManager, _capture_rng_state, _restore_rng_state
from ats.training.scheduler import WarmupCosineScheduler
from ats.training.trainer import Trainer
from tests.test_training import _TinyModelEngine


# ---------------------------------------------------------------------------
# BUG-001: MTP loss crashed on a list .reshape() and used the wrong label
# shift per offset. Fixed via ats.model.mtp.compute_mtp_loss_from_logits,
# shared by MultiTokenPredictionHead.compute_loss and Trainer.train_step.
# ---------------------------------------------------------------------------


def test_compute_mtp_loss_from_logits_matches_per_offset_shift():
    from ats.model.mtp import compute_mtp_loss_from_logits

    torch.manual_seed(0)
    vocab_size = 20
    batch, seq_len = 2, 6
    logits_per_offset = [torch.randn(batch, seq_len, vocab_size) for _ in range(2)]
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    loss = compute_mtp_loss_from_logits(logits_per_offset, labels, vocab_size)

    loss_k1 = F.cross_entropy(
        logits_per_offset[0][:, :-1, :].reshape(-1, vocab_size),
        labels[:, 1:].reshape(-1),
    )
    loss_k2 = F.cross_entropy(
        logits_per_offset[1][:, :-2, :].reshape(-1, vocab_size),
        labels[:, 2:].reshape(-1),
    )
    expected = (loss_k1 + loss_k2) / 2
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_compute_mtp_loss_from_logits_accepts_a_plain_list():
    # The original bug called `.reshape()` directly on the list, which
    # raised AttributeError before ever reaching cross_entropy.
    from ats.model.mtp import compute_mtp_loss_from_logits

    logits_per_offset = [torch.randn(1, 5, 10)]
    assert isinstance(logits_per_offset, list)
    labels = torch.randint(0, 10, (1, 5))
    loss = compute_mtp_loss_from_logits(logits_per_offset, labels, vocab_size=10)
    assert torch.isfinite(loss)


def test_compute_mtp_loss_from_logits_raises_when_seq_len_too_short():
    from ats.model.mtp import compute_mtp_loss_from_logits

    logits_per_offset = [torch.randn(1, 1, 10)]
    labels = torch.randint(0, 10, (1, 1))
    with pytest.raises(ValueError, match="too short"):
        compute_mtp_loss_from_logits(logits_per_offset, labels, vocab_size=10)


class _FakeEngineWrapper:
    """Stands in for a DeepSpeed model_engine in Trainer.train_step tests,
    without needing DeepSpeed: forward/backward run through a REAL
    ATSTransformer (so the MTP loss path is exercised authentically, not
    mocked), while step()/get_global_grad_norm()/optimizer are lightweight
    stand-ins for bookkeeping DeepSpeed would otherwise own."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
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
    DeepSpeed) in favor of _FakeEngineWrapper around a real model."""
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


def test_trainer_train_step_with_mtp_does_not_crash(tmp_path):
    """Direct regression test for BUG-001 at the exact call site that was
    broken: Trainer.train_step with model.use_mtp=True, through a real
    model's forward/backward -- this used to raise
    'AttributeError: list object has no attribute reshape' immediately."""
    trainer = _make_mtp_trainer(tmp_path)
    batch = {
        "input_ids": torch.randint(0, 30, (2, 8)),
        "labels": torch.randint(0, 30, (2, 8)),
    }
    metrics = trainer.train_step(batch)
    assert metrics is not None
    assert math.isfinite(metrics.loss)


def test_trainer_train_step_mtp_gradients_are_finite(tmp_path):
    trainer = _make_mtp_trainer(tmp_path)
    batch = {
        "input_ids": torch.randint(0, 30, (2, 8)),
        "labels": torch.randint(0, 30, (2, 8)),
    }
    trainer.train_step(batch)
    for name, p in trainer.model_engine.module.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


# ---------------------------------------------------------------------------
# EXTRA BUG (found while verifying BUG-001, not itself in the audit):
# TrainingMetrics is a frozen dataclass, but Trainer.train_step (and
# DiffusionTrainer.train_step) did `metrics.tokens_this_step = actual_tokens`
# directly on an already-constructed instance -- frozen dataclasses reject
# any attribute assignment, so this raised FrozenInstanceError on every
# single successful optimizer step, for every training run, regardless of
# MTP. This means train_step could never actually complete before this fix
# -- more fundamental than BUG-001, and it was hiding directly behind it in
# the same method. Fixed by declaring tokens_this_step as a real field and
# using dataclasses.replace to produce a new instance instead of mutating.
# ---------------------------------------------------------------------------


def test_training_metrics_tokens_this_step_is_a_real_field():
    from ats.training.adaptive_controller import TrainingMetrics

    metrics = TrainingMetrics(step=0, loss=1.0, grad_norm=1.0, learning_rate=1e-3)
    assert metrics.tokens_this_step is None
    updated = dataclasses.replace(metrics, tokens_this_step=128)
    assert updated.tokens_this_step == 128
    # Still frozen -- this fix doesn't unfreeze the dataclass, it just makes
    # attaching the token count possible via the correct mechanism.
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.tokens_this_step = 999


# ---------------------------------------------------------------------------
# BUG-002 (audit claim): checkpoint resume crashes because DeepSpeed
# supposedly serializes client_state (including rng_state) through JSON,
# turning the numpy RNG state's ndarray into a list that np.random.set_state
# then can't consume.
#
# NOT REPRODUCED: DeepSpeed's real save_checkpoint()/load_checkpoint() path
# (CheckpointManager.save/load) passes client_state to
# model_engine.save_checkpoint(), which uses DeepSpeed's checkpoint_engine --
# torch.save/torch.load (pickle) by default, not JSON. Pickle preserves the
# numpy ndarray exactly; _restore_rng_state's `isinstance(x, list)` branches
# are never exercised by the real save/resume path. The only JSON written by
# CheckpointManager.save (training_state.json) contains
# global_step/epoch/config_hash only -- no RNG state.
#
# The defensive list-handling in _restore_rng_state is still made fully
# correct below (explicit dtype-aware np.array conversion) since it's cheap
# and harmless, but this is NOT the checkpoint-resume crash fix the audit
# describes -- there was no such crash to fix in this codebase.
# ---------------------------------------------------------------------------


def test_rng_state_json_roundtrip_restores_identical_stream():
    np.random.seed(123)
    state = _capture_rng_state()

    serialized = json.dumps(
        {
            "python": list(state["python"]),
            "numpy": [
                state["numpy"][0],
                state["numpy"][1].tolist(),
                state["numpy"][2],
                state["numpy"][3],
                state["numpy"][4],
            ],
            "torch": state["torch"],
        }
    )
    deserialized = json.loads(serialized)

    np.random.seed(123)
    expected_next = np.random.rand(5)

    np.random.seed(999)  # scramble
    _restore_rng_state(deserialized)
    restored_next = np.random.rand(5)
    assert np.allclose(expected_next, restored_next)


def test_rng_state_restore_produces_real_ndarray_not_list():
    np.random.seed(42)
    state = _capture_rng_state()
    serialized = json.loads(
        json.dumps(
            {
                "python": list(state["python"]),
                "numpy": [
                    state["numpy"][0],
                    state["numpy"][1].tolist(),
                    state["numpy"][2],
                    state["numpy"][3],
                    state["numpy"][4],
                ],
                "torch": state["torch"],
            }
        )
    )
    _restore_rng_state(serialized)
    restored_state = np.random.get_state()
    assert isinstance(restored_state[1], np.ndarray)
    assert restored_state[1].dtype == np.uint32


# ---------------------------------------------------------------------------
# BUG-003 (audit claim): dataloader modulo-based sharding requires every
# rank/worker to iterate 100% of the data and discard most of it.
#
# CONFIRMED, and already partially documented in ats/data/dataloader.py's
# own comments before this fix. For the RAW-TEXT path, true sharding isn't
# safely achievable without a larger redesign: chunk boundaries depend on a
# stochastic, weighted interleaving of sources plus a token-accumulation
# buffer, so which output chunk a given input line ends up in isn't
# knowable without running the mixing process. What IS fixed here: the
# PREPROCESSED (.bin/memmap) path, which preprocess.py's own docstring
# already recommends for production-scale training specifically because it
# skips on-the-fly tokenization -- that path now shards by direct
# block-index striding (O(1) per rank, no wasted memmap reads) instead of
# iterating every block on every rank.
# ---------------------------------------------------------------------------


def test_preprocessed_source_shards_by_direct_index_striding(tmp_path):
    from ats.data.dataset import _iter_preprocessed_examples

    num_blocks, seq_length = 20, 4
    meta = {"num_blocks": num_blocks, "seq_length": seq_length, "padding_side": "right"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    tokens = np.arange(num_blocks * seq_length, dtype=np.int32).reshape(
        num_blocks, seq_length
    )
    tokens.tofile(tmp_path / "tokens.bin")
    np.save(tmp_path / "valid_lengths.npy", np.full(num_blocks, seq_length))

    bin_path = tmp_path / "tokens.bin"

    world_size = 4
    all_first_tokens = set()
    for shard_id in range(world_size):
        examples = list(
            _iter_preprocessed_examples(
                bin_path, seq_length, shard_id=shard_id, num_shards=world_size
            )
        )
        assert len(examples) == num_blocks // world_size
        for ex in examples:
            all_first_tokens.add(ex["input_ids"][0])
    assert len(all_first_tokens) == num_blocks  # every block seen exactly once total


def test_preprocessed_source_default_is_unsharded(tmp_path):
    from ats.data.dataset import _iter_preprocessed_examples

    num_blocks, seq_length = 6, 4
    meta = {"num_blocks": num_blocks, "seq_length": seq_length, "padding_side": "right"}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    tokens = np.arange(num_blocks * seq_length, dtype=np.int32).reshape(
        num_blocks, seq_length
    )
    tokens.tofile(tmp_path / "tokens.bin")
    np.save(tmp_path / "valid_lengths.npy", np.full(num_blocks, seq_length))

    examples = list(_iter_preprocessed_examples(tmp_path / "tokens.bin", seq_length))
    assert len(examples) == num_blocks


# ---------------------------------------------------------------------------
# BUG-004: CheckpointManager.save called module.state_dict() (a required
# ZeRO-3 collective) on every rank and then unconditionally did .cpu() and
# built a full Python dict from it -- meaning every rank's HOST process
# retained a full desharded copy of the model in CPU RAM, not just rank 0.
# Fixed: every rank still calls the required collective, but only rank 0
# performs the .cpu() materialization and retains the result.
# ---------------------------------------------------------------------------


def test_checkpoint_save_only_rank_zero_writes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    config = ATSConfig(
        model=ModelConfig(
            hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=1,
            intermediate_size=16, vocab_size=20, use_flash_attention=False,
        ),
        training=TrainingConfig(max_steps=10, learning_rate=1e-3, warmup_steps=1),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path)),
    )
    model = torch.nn.Linear(8, 8)
    call_count = {"n": 0}
    real_state_dict = model.state_dict

    def counting_state_dict(*args, **kwargs):
        call_count["n"] += 1
        return real_state_dict(*args, **kwargs)

    model.state_dict = counting_state_dict

    manager = CheckpointManager(config)
    ckpt_dir = manager.save(_TinyModelEngine(model), global_step=1, epoch=0)

    # The collective call must still happen on every rank (required for
    # ZeRO-3's gather to complete) --
    assert call_count["n"] >= 1
    # -- but rank 1 must not write the artifacts CheckpointManager.save
    # itself is responsible for (training_state.json, model.safetensors).
    # DeepSpeed's own model_engine.save_checkpoint() call is intentionally
    # left rank-agnostic here (see the comment in checkpoint.py -- DeepSpeed
    # coordinates its own ranks for that call), so the fake engine's own
    # model.pt/client_state.json/rng_state.pt still appear regardless of
    # rank; that's the fake's behavior, not what this fix changes.
    assert not (ckpt_dir / "training_state.json").exists()
    assert not (ckpt_dir / "model.safetensors").exists()


def test_checkpoint_save_rank_zero_still_writes_safetensors(tmp_path):
    config = ATSConfig(
        model=ModelConfig(
            hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=1,
            intermediate_size=16, vocab_size=20, use_flash_attention=False,
        ),
        training=TrainingConfig(max_steps=10, learning_rate=1e-3, warmup_steps=1),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
        checkpoint=CheckpointConfig(output_dir=str(tmp_path)),
    )
    model = torch.nn.Linear(8, 8)

    manager = CheckpointManager(config)
    ckpt_dir = manager.save(_TinyModelEngine(model), global_step=1, epoch=0)
    assert (ckpt_dir / "model.safetensors").exists()


# ---------------------------------------------------------------------------
# BUG-005 (audit claim): "expert_output = self.expertsexpert_id" -- a
# mangled attribute access -- in _PyTorchMoEFallback.forward.
#
# NOT REPRODUCED: the installed source reads
# `expert_output = self.experts[expert_id](expert_input)` (correct) at the
# line the audit cites. Confirmed by reading the source directly and by
# running the fallback path end-to-end with deepspeed's import blocked --
# it worked without error. No fix applied; the regression test below just
# pins down that this keeps working.
# ---------------------------------------------------------------------------


def test_moe_pytorch_fallback_runs_without_deepspeed():
    import builtins

    from ats.model.moe import _PyTorchMoEFallback

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "deepspeed" or name.startswith("deepspeed."):
            raise ImportError("blocked for this test")
        return real_import(name, *args, **kwargs)

    fallback = _PyTorchMoEFallback(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        capacity_factor=1.25,
        load_balancing_weight=0.01,
        num_layers=2,
    )
    x = torch.randn(2, 6, 16)

    builtins.__import__ = blocking_import
    try:
        out, aux_loss = fallback(x)
    finally:
        builtins.__import__ = real_import

    assert out.shape == x.shape
    assert torch.isfinite(aux_loss)


# ---------------------------------------------------------------------------
# BUG-006: ATSTransformer.forward() validated token ids via
# input_ids.max().item() / .min().item() on every forward pass, forcing a
# synchronous GPU->CPU transfer in the training hot path. Fixed: the check
# now defaults to running only on CPU tensors (free there) and is skipped
# by default on CUDA tensors, with an explicit opt-in for anyone who wants
# it there anyway (e.g. one-off debugging of a tokenizer/vocab mismatch).
# ---------------------------------------------------------------------------


def test_forward_validates_input_ids_on_cpu_by_default():
    config = ModelConfig(
        hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=1,
        intermediate_size=16, vocab_size=20, use_flash_attention=False,
    )
    model = ATSTransformer(config)
    bad_input_ids = torch.tensor([[0, 1, 20]])  # 20 is out of range for vocab_size=20
    with pytest.raises(ValueError, match="outside"):
        model(bad_input_ids)


def test_forward_validate_input_ids_flag_can_be_forced_off():
    config = ModelConfig(
        hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=1,
        intermediate_size=16, vocab_size=20, use_flash_attention=False,
    )
    model = ATSTransformer(config)
    bad_input_ids = torch.tensor([[0, 1, 20]])
    # Explicitly disabling the check must skip straight past it -- the
    # friendly ValueError with "outside" in the message must NOT be what's
    # raised. The bad id still reaches nn.Embedding's own bounds check
    # (which raises a less-friendly IndexError) -- that's the correctness
    # trade-off of turning validation off, not a bug.
    with pytest.raises(Exception) as exc_info:
        model(bad_input_ids, validate_input_ids=False)
    assert "outside" not in str(exc_info.value)


def test_forward_validate_input_ids_false_does_not_affect_valid_input():
    config = ModelConfig(
        hidden_size=8, num_layers=1, num_heads=2, num_kv_heads=1,
        intermediate_size=16, vocab_size=20, use_flash_attention=False,
    )
    model = ATSTransformer(config)
    model.eval()
    good_input_ids = torch.tensor([[0, 1, 19]])
    out_validated = model(good_input_ids, validate_input_ids=True)
    out_unvalidated = model(good_input_ids, validate_input_ids=False)
    assert torch.equal(out_validated.logits, out_unvalidated.logits)


# ---------------------------------------------------------------------------
# BUG-007: Trainer.train (and DiffusionTrainer.train)'s dataloader-exhaustion
# handling caught the first StopIteration (end of epoch) but not a second
# one immediately after re-creating the iterator -- an empty dataloader
# crashed with an unhandled StopIteration instead of a clean halt.
# ---------------------------------------------------------------------------


def test_trainer_handles_empty_dataloader_without_crashing(tmp_path):
    trainer = _make_mtp_trainer(tmp_path)
    trainer.train_dataloader = []  # empty: iter([]) immediately raises StopIteration
    trainer.config.training.max_steps = 5

    with pytest.raises(RuntimeError, match="empty"):
        trainer.train()


# ---------------------------------------------------------------------------
# BUG-008: GroupedQueryAttention's incremental-decoding path only built a
# proper mask when attention_mask was None; when both past_key_value AND
# attention_mask were given (batched multi-token continuation with padding),
# it fell through to build_padding_causal_mask, which returns a
# [batch,1,seq_len,seq_len] mask -- but k/v were already total_len long, so
# SDPA raised a shape-mismatch RuntimeError. Fixed: the incremental path now
# always builds a [.., seq_len, total_len] mask, folding in the new-tokens'
# padding mask (with cached positions always treated as valid) when given.
#
# Same underlying issue existed in MLAAttention, actually more broadly: it
# used the raw, un-reshaped attention_mask directly as attn_mask in BOTH the
# plain (no cache) and incremental case, which SDPA rejects outright (wrong
# dtype/shape) rather than merely mis-computing -- this wasn't in the audit
# at all; found while verifying BUG-008. Fixed the same way GQA's plain-case
# already was, via build_padding_causal_mask, plus the same incremental-mask
# combination for the cached case.
# ---------------------------------------------------------------------------


def test_gqa_incremental_decoding_with_attention_mask():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=32, num_heads=4, num_kv_heads=2, max_seq_len=32,
        use_flash_attention=False,
    )
    x_prompt = torch.randn(2, 4, 32)
    _, past = attn(x_prompt, use_cache=True)

    x_new = torch.randn(2, 3, 32)  # multi-token continuation, seq_len > 1
    attention_mask = torch.ones(2, 3, dtype=torch.long)
    out, _ = attn(x_new, attention_mask=attention_mask, past_key_value=past, use_cache=True)
    assert out.shape == (2, 3, 32)
    assert torch.isfinite(out).all()


def test_gqa_incremental_mask_is_actually_causal_among_new_tokens():
    """Perturbing a later new token must not change an earlier new token's
    output -- if the causal component were dropped (the logical-leak half
    of BUG-008), it would."""
    torch.manual_seed(0)
    attn = GroupedQueryAttention(
        hidden_size=16, num_heads=2, num_kv_heads=2, max_seq_len=32,
        use_flash_attention=False,
    )
    attn.eval()
    x_prompt = torch.randn(1, 3, 16)
    _, past = attn(x_prompt, use_cache=True)

    x_new = torch.randn(1, 4, 16)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    out_a, _ = attn(x_new, attention_mask=attention_mask, past_key_value=past)

    x_new_perturbed = x_new.clone()
    x_new_perturbed[:, -1, :] += 10.0  # perturb only the LAST new token
    out_b, _ = attn(x_new_perturbed, attention_mask=attention_mask, past_key_value=past)

    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-5)
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :], atol=1e-5)


def test_mla_plain_attention_mask_no_longer_raises_dtype_error():
    torch.manual_seed(0)
    mla = MLAAttention(hidden_size=32, num_heads=4, latent_dim=16, max_seq_len=16)
    x = torch.randn(2, 8, 32)
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    out, _ = mla(x, attention_mask=attention_mask)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_mla_incremental_decoding_with_attention_mask():
    torch.manual_seed(0)
    mla = MLAAttention(hidden_size=32, num_heads=4, latent_dim=16, max_seq_len=32)
    x_prompt = torch.randn(2, 4, 32)
    _, past = mla(x_prompt, use_cache=True)

    x_new = torch.randn(2, 3, 32)
    attention_mask = torch.ones(2, 3, dtype=torch.long)
    out, _ = mla(x_new, attention_mask=attention_mask, past_key_value=past, use_cache=True)
    assert out.shape == (2, 3, 32)
    assert torch.isfinite(out).all()


def test_mla_incremental_mask_is_actually_causal_among_new_tokens():
    torch.manual_seed(0)
    mla = MLAAttention(hidden_size=16, num_heads=2, latent_dim=8, max_seq_len=32)
    mla.eval()
    x_prompt = torch.randn(1, 3, 16)
    _, past = mla(x_prompt, use_cache=True)

    x_new = torch.randn(1, 4, 16)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    out_a, _ = mla(x_new, attention_mask=attention_mask, past_key_value=past)

    x_new_perturbed = x_new.clone()
    x_new_perturbed[:, -1, :] += 10.0
    out_b, _ = mla(x_new_perturbed, attention_mask=attention_mask, past_key_value=past)

    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-5)
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :], atol=1e-5)


# ---------------------------------------------------------------------------
# BUG-009: DataConfig.seq_length allowed 1, which makes autoregressive
# shift-by-1 labels empty (shift_labels = labels[..., 1:] is a zero-length
# tensor). Empirically this doesn't raise the RuntimeError the audit
# describes -- F.cross_entropy on an empty batch silently returns NaN
# (0/0 in the mean reduction) rather than erroring -- arguably worse, since
# it could go unnoticed. Either way, seq_length=1 is not trainable; fixed by
# requiring >= 2 at the config layer with a clear error.
# ---------------------------------------------------------------------------


def test_config_rejects_seq_length_one():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least 2"):
        DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=1)


def test_config_accepts_seq_length_two():
    config = DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=2)
    assert config.seq_length == 2
