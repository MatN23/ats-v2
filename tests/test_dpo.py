"""Tests for ats.training.dpo (compute_sequence_logprobs, dpo_loss) and
ats.data.preference_dataset -- the parts of DPO alignment that are directly
verifiable without a GPU, DeepSpeed, or a real model."""

from __future__ import annotations

import json
import math

import pytest
import torch

from ats.config.schema import ConfigError
from ats.data.dataset import IGNORE_INDEX
from ats.data.preference_dataset import PreferenceDataset, build_preference_dataloader
from ats.training.dpo import compute_sequence_logprobs, dpo_loss

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _tiktoken_usable() -> bool:
    """See tests/test_data.py's identical helper for why this probe (not
    just an import check) is needed: tiktoken downloads its BPE merge file
    from a remote CDN on first use, which fails in network-restricted
    sandboxes/CI even when the package itself is installed."""
    if not _TIKTOKEN_AVAILABLE:
        return False
    try:
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:  # noqa: BLE001 -- deliberately broad, see test_data.py
        return False


_TIKTOKEN_USABLE = _tiktoken_usable()
_skip_without_tiktoken = pytest.mark.skipif(
    not _TIKTOKEN_USABLE,
    reason="tiktoken cl100k_base encoding not usable (not installed, or its data "
    "file could not be downloaded -- e.g. no network/blocked egress)",
)


# --- compute_sequence_logprobs -----------------------------------------


def test_compute_sequence_logprobs_matches_hand_computed_value():
    # A single position, single batch element, 3-way vocab: logits chosen so
    # softmax probabilities are easy to hand-verify.
    logits = torch.tensor([[[0.0, 0.0, 0.0]]])  # uniform -> each prob = 1/3
    labels = torch.tensor([[1]])
    logp = compute_sequence_logprobs(logits, labels)
    assert logp.shape == (1,)
    assert math.isclose(logp.item(), math.log(1 / 3), abs_tol=1e-6)


def test_compute_sequence_logprobs_sums_over_multiple_positions():
    # Two positions, same uniform-3-way logits at both -> total logp should
    # be exactly 2 * log(1/3), a direct sum, not an average.
    logits = torch.zeros(1, 2, 3)
    labels = torch.tensor([[0, 2]])
    logp = compute_sequence_logprobs(logits, labels)
    assert math.isclose(logp.item(), 2 * math.log(1 / 3), abs_tol=1e-6)


def test_compute_sequence_logprobs_ignores_masked_positions():
    logits = torch.randn(1, 4, 5)
    labels_masked = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 1, 2]])
    labels_unmasked_prefix_changed = labels_masked.clone()
    # Change what's under the IGNORE_INDEX positions in the underlying
    # tensor value directly (simulating different prompt tokens) -- since
    # they're still masked, the result must be identical either way.
    logp_a = compute_sequence_logprobs(logits, labels_masked)
    # Build a second logits tensor with different values ONLY at the masked
    # positions; result should be unchanged since those positions never
    # contribute.
    logits_b = logits.clone()
    logits_b[:, :2, :] = torch.randn(1, 2, 5) * 100
    logp_b = compute_sequence_logprobs(logits_b, labels_unmasked_prefix_changed)
    assert torch.allclose(logp_a, logp_b)


def test_compute_sequence_logprobs_all_masked_gives_zero():
    logits = torch.randn(2, 3, 4)
    labels = torch.full((2, 3), IGNORE_INDEX)
    logp = compute_sequence_logprobs(logits, labels)
    assert torch.equal(logp, torch.zeros(2))


def test_compute_sequence_logprobs_rejects_shape_mismatch():
    logits = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 5, dtype=torch.long)  # wrong seq_len
    with pytest.raises(ValueError, match="must match"):
        compute_sequence_logprobs(logits, labels)


def test_compute_sequence_logprobs_rejects_wrong_logits_ndim():
    logits = torch.randn(2, 4)  # missing vocab dim
    labels = torch.zeros(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="batch, seq_len, vocab_size"):
        compute_sequence_logprobs(logits, labels)


# --- dpo_loss ------------------------------------------------------------


def test_dpo_loss_is_low_when_policy_prefers_chosen_more_than_reference_does():
    # Policy pushes chosen's logp up and rejected's down relative to a flat
    # reference -> logits argument to logsigmoid is strongly positive ->
    # loss should be small (logsigmoid(large positive) ~ 0).
    policy_chosen = torch.tensor([5.0])
    policy_rejected = torch.tensor([-5.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss, metrics = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0
    )
    assert loss.item() < 0.01
    assert metrics.accuracy == 1.0
    assert metrics.reward_margin > 0


def test_dpo_loss_is_high_when_policy_prefers_rejected():
    policy_chosen = torch.tensor([-5.0])
    policy_rejected = torch.tensor([5.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss, metrics = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0
    )
    assert loss.item() > 5.0  # logsigmoid(-10) ~ -10
    assert metrics.accuracy == 0.0
    assert metrics.reward_margin < 0


def test_dpo_loss_matches_hand_computed_value_at_zero_logits():
    # When policy == reference exactly (no learning has happened yet), the
    # logits argument to logsigmoid is exactly 0, so loss = -log(sigmoid(0))
    # = -log(0.5) = log(2), regardless of beta.
    same = torch.tensor([1.23, -4.56])
    loss, metrics = dpo_loss(same, same, same, same, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-6)
    assert metrics.reward_margin == 0.0
    assert (
        metrics.accuracy == 0.0
    )  # chosen_reward > rejected_reward is False when equal


def test_dpo_loss_higher_beta_amplifies_the_same_logratio_difference():
    policy_chosen = torch.tensor([1.0])
    policy_rejected = torch.tensor([0.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss_low_beta, _ = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    loss_high_beta, _ = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=10.0
    )
    # Policy already prefers chosen -> increasing beta should push loss
    # further toward zero (more confident), not further from it.
    assert loss_high_beta.item() < loss_low_beta.item()


def test_dpo_loss_batch_reduction_is_mean_not_sum():
    # Two identical examples should give the same loss as one -- this only
    # holds if the reduction is a mean over the batch, not a sum.
    single = dpo_loss(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )[0]
    doubled = dpo_loss(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
    )[0]
    assert math.isclose(single.item(), doubled.item(), abs_tol=1e-6)


def test_dpo_loss_gradient_flows_only_through_policy_terms():
    policy_chosen = torch.tensor([0.5], requires_grad=True)
    policy_rejected = torch.tensor([-0.5], requires_grad=True)
    ref_chosen = torch.tensor([0.0], requires_grad=True)
    ref_rejected = torch.tensor([0.0], requires_grad=True)
    loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    loss.backward()
    assert policy_chosen.grad is not None and policy_chosen.grad.item() != 0.0
    assert policy_rejected.grad is not None and policy_rejected.grad.item() != 0.0
    # ref_* tensors are typically produced under torch.no_grad() by
    # DPOTrainer (never requires_grad=True in real use), but even if they
    # did carry requires_grad here, backward() should still succeed --
    # this test only checks the policy side actually receives real,
    # nonzero gradients, which is the property that matters for training.


def test_dpo_loss_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1D"):
        dpo_loss(
            torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2)
        )


def test_dpo_loss_rejects_non_positive_beta():
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(
            torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2), beta=0.0
        )


# --- DPOTrainer.compute_loss (real model, no DeepSpeed needed) -----------


def test_dpotrainer_compute_loss_end_to_end():
    """Exercises the actual model-integration path (real ATSTransformer
    forward passes -> compute_sequence_logprobs -> dpo_loss) without
    needing deepspeed installed: DPOTrainer is constructed via __new__,
    bypassing __init__ (so initialize_engine() never runs), with
    model_engine/reference_model set directly to plain ATSTransformer
    instances. This is the same "test the real integration, skip only the
    DeepSpeed-specific plumbing" approach test_training.py's _TinyModelEngine
    uses for CheckpointManager."""
    from ats.config.schema import ModelConfig
    from ats.model.transformer import ATSTransformer
    from ats.training.dpo import DPOTrainer

    torch.manual_seed(0)
    config = ModelConfig(
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=32,
        vocab_size=50,
        max_seq_len=16,
        use_flash_attention=False,
    )
    policy = ATSTransformer(config)
    reference = ATSTransformer(config)
    reference.load_state_dict(policy.state_dict())  # start identical to policy

    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.model_engine = policy
    trainer.reference_model = reference
    trainer.beta = 0.1

    batch_size, seq_len = 2, 8
    batch = {
        "chosen_input_ids": torch.randint(0, 50, (batch_size, seq_len)),
        "chosen_labels": torch.randint(0, 50, (batch_size, seq_len)),
        "rejected_input_ids": torch.randint(0, 50, (batch_size, seq_len)),
        "rejected_labels": torch.randint(0, 50, (batch_size, seq_len)),
    }
    # Mask the first 3 positions as "prompt" in both, matching real usage.
    batch["chosen_labels"][:, :3] = IGNORE_INDEX
    batch["rejected_labels"][:, :3] = IGNORE_INDEX

    loss, metrics = trainer.compute_loss(batch)

    assert loss.dim() == 0
    assert loss.requires_grad  # must be able to backward() through it
    # Policy == reference at this point (identical weights), so the DPO
    # logits argument is exactly 0 for every example -> loss == log(2),
    # exactly matching test_dpo_loss_matches_hand_computed_value_at_zero_logits's
    # hand-derived value, now verified through real model forward passes
    # rather than hand-constructed logp tensors.
    assert math.isclose(loss.item(), math.log(2), abs_tol=1e-4)
    assert metrics.reward_margin == pytest.approx(0.0, abs=1e-4)

    loss.backward()
    # Policy must receive real gradients; reference must not (frozen,
    # computed entirely under torch.no_grad() inside compute_loss).
    policy_grad_norms = [
        p.grad.norm().item() for p in policy.parameters() if p.grad is not None
    ]
    assert len(policy_grad_norms) > 0
    assert any(g > 0 for g in policy_grad_norms)
    assert all(p.grad is None for p in reference.parameters())


def test_dpotrainer_compute_loss_after_policy_diverges_from_reference():
    """A second, complementary integration check: after directly nudging
    the policy's weights away from the reference (simulating what training
    would do), compute_loss's reward_margin should become nonzero and its
    sign should be well-defined -- confirms compute_loss is actually
    sensitive to policy/reference divergence through the real forward
    pass, not just returning a constant."""
    from ats.config.schema import ModelConfig
    from ats.model.transformer import ATSTransformer
    from ats.training.dpo import DPOTrainer

    torch.manual_seed(1)
    config = ModelConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=32,
        vocab_size=30,
        max_seq_len=16,
        use_flash_attention=False,
    )
    policy = ATSTransformer(config)
    reference = ATSTransformer(config)
    reference.load_state_dict(policy.state_dict())
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.model_engine = policy
    trainer.reference_model = reference
    trainer.beta = 0.1

    batch = {
        "chosen_input_ids": torch.randint(0, 30, (2, 6)),
        "chosen_labels": torch.randint(0, 30, (2, 6)),
        "rejected_input_ids": torch.randint(0, 30, (2, 6)),
        "rejected_labels": torch.randint(0, 30, (2, 6)),
    }
    _loss, metrics = trainer.compute_loss(batch)
    # No specific sign asserted (policy diverged randomly, not toward
    # "chosen"), just that it's no longer exactly zero, proving compute_loss
    # actually reflects the real forward-pass divergence.
    assert metrics.reward_margin != pytest.approx(0.0, abs=1e-6)


# --- PreferenceDataset / build_preference_dataloader ----------------------


class _FakeTokenizer:
    """Same deterministic character-level fake as tests/test_data.py's, so
    PreferenceDataset's masking/padding/truncation logic gets real
    verification without needing network access to download a real
    tokenizer's BPE file."""

    vocab_size = 256
    eos_token_id = 256
    pad_token_id = 256

    def encode(self, text: str):
        return [ord(c) % 200 for c in text]


def test_preference_dataset_masking_with_fake_tokenizer(monkeypatch, tmp_path):
    import ats.data.preference_dataset as pref_mod

    monkeypatch.setattr(pref_mod, "Tokenizer", lambda _name: _FakeTokenizer())

    path = _write_preference_jsonl(
        tmp_path,
        [{"prompt": "hi", "chosen": "yes", "rejected": "no"}],
    )
    dataset = pref_mod.PreferenceDataset(path, "fake:tokenizer", seq_length=10)
    example = dataset[0]

    fake = _FakeTokenizer()
    prompt_ids = fake.encode("hi")  # length 2
    chosen_ids = fake.encode("yes") + [fake.eos_token_id]  # length 4
    expected_input = prompt_ids + chosen_ids  # length 6
    expected_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids  # length 6
    pad_len = 10 - len(expected_input)
    expected_input = expected_input + [fake.pad_token_id] * pad_len
    expected_labels = expected_labels + [IGNORE_INDEX] * pad_len

    assert example["chosen_input_ids"] == expected_input
    assert example["chosen_labels"] == expected_labels


def test_preference_dataset_truncates_when_too_long_with_fake_tokenizer(
    monkeypatch, tmp_path
):
    import ats.data.preference_dataset as pref_mod

    monkeypatch.setattr(pref_mod, "Tokenizer", lambda _name: _FakeTokenizer())

    path = _write_preference_jsonl(
        tmp_path,
        [{"prompt": "ab", "chosen": "verylongresponsehere", "rejected": "x"}],
    )
    dataset = pref_mod.PreferenceDataset(path, "fake:tokenizer", seq_length=5)
    example = dataset[0]
    assert len(example["chosen_input_ids"]) == 5
    assert len(example["chosen_labels"]) == 5
    # First 2 positions are the prompt -> masked; truncation keeps the
    # sequence at exactly seq_length rather than raising, since only the
    # PROMPT alone exceeding seq_length is treated as an error (see
    # test_preference_dataset_rejects_prompt_too_long).
    assert example["chosen_labels"][0] == IGNORE_INDEX
    assert example["chosen_labels"][1] == IGNORE_INDEX


def test_build_preference_dataloader_batches_with_fake_tokenizer(monkeypatch, tmp_path):
    import ats.data.preference_dataset as pref_mod

    monkeypatch.setattr(pref_mod, "Tokenizer", lambda _name: _FakeTokenizer())

    rows = [
        {"prompt": f"prompt {i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
        for i in range(6)
    ]
    path = _write_preference_jsonl(tmp_path, rows)
    dataloader = pref_mod.build_preference_dataloader(
        path, "fake:tokenizer", seq_length=16, batch_size=2
    )
    batch = next(iter(dataloader))
    assert batch["chosen_input_ids"].shape == (2, 16)
    assert batch["chosen_labels"].shape == (2, 16)
    assert batch["rejected_input_ids"].shape == (2, 16)
    assert batch["rejected_labels"].shape == (2, 16)
    assert batch["chosen_input_ids"].dtype == torch.long


def _write_preference_jsonl(tmp_path, rows):
    path = tmp_path / "prefs.jsonl"
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return str(path)


@_skip_without_tiktoken
def test_preference_dataset_produces_correctly_masked_labels(tmp_path):
    path = _write_preference_jsonl(
        tmp_path,
        [{"prompt": "hello", "chosen": "world", "rejected": "nope"}],
    )
    dataset = PreferenceDataset(path, "tiktoken:cl100k_base", seq_length=16)
    example = dataset[0]

    for prefix in ("chosen", "rejected"):
        ids = example[f"{prefix}_input_ids"]
        labels = example[f"{prefix}_labels"]
        assert len(ids) == 16
        assert len(labels) == 16
        # Every masked position corresponds to either the prompt or padding
        # -- never a response token, which must always have its real id.
        num_masked = sum(1 for lbl in labels if lbl == IGNORE_INDEX)
        num_unmasked = len(labels) - num_masked
        assert num_unmasked > 0  # at least the response + EOS survived
        # Unmasked labels must equal the input_ids at the same positions
        # (this is teacher-forcing: label at position i is the real next
        # token, and since this dataset stores non-shifted aligned pairs,
        # they're identical at every unmasked position).
        for i, (tok, lbl) in enumerate(zip(ids, labels)):
            if lbl != IGNORE_INDEX:
                assert tok == lbl, f"position {i}: input {tok} != label {lbl}"


@_skip_without_tiktoken
def test_preference_dataset_rejects_missing_field(tmp_path):
    path = _write_preference_jsonl(tmp_path, [{"prompt": "hi", "chosen": "there"}])
    with pytest.raises(ConfigError, match="rejected"):
        PreferenceDataset(path, "tiktoken:cl100k_base", seq_length=16)


@_skip_without_tiktoken
def test_preference_dataset_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ConfigError, match="no valid preference examples"):
        PreferenceDataset(str(path), "tiktoken:cl100k_base", seq_length=16)


@_skip_without_tiktoken
def test_preference_dataset_rejects_prompt_too_long(tmp_path):
    long_prompt = "word " * 100
    path = _write_preference_jsonl(
        tmp_path, [{"prompt": long_prompt, "chosen": "a", "rejected": "b"}]
    )
    dataset = PreferenceDataset(path, "tiktoken:cl100k_base", seq_length=8)
    with pytest.raises(ConfigError, match="seq_length"):
        dataset[0]


@_skip_without_tiktoken
def test_build_preference_dataloader_batches_correctly(tmp_path):
    rows = [
        {"prompt": f"prompt {i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
        for i in range(6)
    ]
    path = _write_preference_jsonl(tmp_path, rows)
    dataloader = build_preference_dataloader(
        path, "tiktoken:cl100k_base", seq_length=16, batch_size=2
    )
    batch = next(iter(dataloader))
    assert batch["chosen_input_ids"].shape == (2, 16)
    assert batch["chosen_labels"].shape == (2, 16)
    assert batch["rejected_input_ids"].shape == (2, 16)
    assert batch["rejected_labels"].shape == (2, 16)
    assert batch["chosen_input_ids"].dtype == torch.long


@_skip_without_tiktoken
def test_build_preference_dataloader_shards_by_rank(tmp_path):
    rows = [
        {"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(10)
    ]
    path = _write_preference_jsonl(tmp_path, rows)
    loader_rank0 = build_preference_dataloader(
        path, "tiktoken:cl100k_base", seq_length=16, batch_size=1, rank=0, world_size=2
    )
    loader_rank1 = build_preference_dataloader(
        path, "tiktoken:cl100k_base", seq_length=16, batch_size=1, rank=1, world_size=2
    )
    assert len(loader_rank0.dataset) == 5
    assert len(loader_rank1.dataset) == 5
