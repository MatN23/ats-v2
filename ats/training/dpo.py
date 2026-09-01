"""Direct Preference Optimization (DPO), per Rafailov et al. 2023
(https://arxiv.org/abs/2305.18290).

This implements DPO specifically, not the more general "RLHF" umbrella
--method rlhf in ats.cli.align still raises NotImplementedError, since
classic RLHF (PPO against a trained reward model) needs a reward-model
training pipeline and a generation/rollout loop that don't exist anywhere
in this codebase; bolting a reward model on top of ats-v2 in the same pass
as DPO would be a much larger, much less verifiable change. DPO was chosen
because it needs neither: it optimizes the policy directly against
preference pairs (prompt, chosen_response, rejected_response) relative to a
frozen reference model, using only the same forward-pass/cross-entropy
machinery every other trainer in this codebase already uses.

The two functions below are pure (no model, no I/O) and are the entire
mathematical core of DPO -- see tests/test_dpo.py for direct numerical
verification of both, independent of any model or DeepSpeed dependency.
DPOTrainer (below) is the part that actually calls a real model and needs
DeepSpeed to train at scale; its train_step is structured so the same pure
functions are exercised through a stubbed engine in tests, exactly like
Trainer/DiffusionTrainer's tests do (see tests/test_training.py's
_TinyModelEngine).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ats.config.schema import ATSConfig
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.checkpoint import CheckpointManager
from ats.training.monitor import Monitor
from ats.training.scheduler import WarmupCosineScheduler
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.training.dpo")

IGNORE_INDEX = -100  # matches ats.data.dataset.IGNORE_INDEX


def compute_sequence_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Sum of per-token log-probabilities of `labels` under `logits`,
    reduced over the sequence dimension to one scalar per batch element.
    Positions where labels == ignore_index (the prompt portion of a DPO
    example, and any padding) don't contribute to the sum at all -- not
    "contribute zero", genuinely excluded, so a longer or shorter masked
    region never changes the result for the tokens that do count.

    logits: [batch, seq_len, vocab_size] (already shifted so logits[:, t]
        predicts labels[:, t] -- callers pass in output.logits[..., :-1, :]
        and labels[..., 1:], matching every other cross-entropy call site in
        this codebase; this function does not do the shifting itself, since
        different callers may want different shift conventions and silently
        assuming one would be exactly the kind of unstated assumption this
        codebase's docstrings elsewhere argue against).
    labels: [batch, seq_len], int64, with ignore_index at masked positions.

    Returns: [batch] float tensor, one summed log-prob per sequence.
    """
    if logits.dim() != 3:
        raise ValueError(
            f"compute_sequence_logprobs expects logits of shape "
            f"[batch, seq_len, vocab_size], got {tuple(logits.shape)}."
        )
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match logits' "
            f"[batch, seq_len] = {tuple(logits.shape[:2])}."
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = labels != ignore_index
    # Clamp before gather: ignore_index (-100) is not a valid vocab index and
    # would make gather() raise; the gathered value at masked positions is
    # discarded by `mask` immediately after anyway, so the clamped value
    # (index 0) is never used for anything.
    safe_labels = labels.clamp(min=0)
    per_token_logp = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1))
    per_token_logp = per_token_logp.squeeze(-1) * mask
    return per_token_logp.sum(dim=1)


@dataclasses.dataclass
class DPOStepMetrics:
    loss: float
    chosen_reward: float  # mean beta * (policy - ref) log-ratio for chosen
    rejected_reward: float  # same, for rejected
    reward_margin: float  # chosen_reward - rejected_reward
    accuracy: float  # fraction of the batch where chosen_reward > rejected_reward


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, DPOStepMetrics]:
    """The DPO loss (Rafailov et al. 2023, eq. 7):

        L = -E[ log sigmoid( beta * ( (policy_chosen - ref_chosen)
                                     - (policy_rejected - ref_rejected) ) ) ]

    where each "chosen"/"rejected" term is that sequence's summed log-prob
    under the policy or (frozen) reference model. Intuition: the policy is
    rewarded for increasing chosen's log-prob and/or decreasing rejected's,
    relative to what the reference model already assigned -- not relative
    to some absolute scale, which is what keeps this from just maximizing
    chosen's raw likelihood regardless of quality (that's what plain
    supervised fine-tuning on `chosen` alone would do).

    All four input tensors are [batch], as returned by
    compute_sequence_logprobs. Returns (scalar loss, DPOStepMetrics with
    batch-mean diagnostics -- the "reward" terminology matches the DPO
    paper's framing of beta * log-ratio as an implicit reward, even though
    no explicit reward model is ever trained here).
    """
    for name, t in (
        ("policy_chosen_logps", policy_chosen_logps),
        ("policy_rejected_logps", policy_rejected_logps),
        ("ref_chosen_logps", ref_chosen_logps),
        ("ref_rejected_logps", ref_rejected_logps),
    ):
        if t.dim() != 1:
            raise ValueError(f"{name} must be 1D [batch], got shape {tuple(t.shape)}.")
    if beta <= 0:
        raise ValueError(f"dpo_loss beta must be positive, got {beta}.")

    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratio - ref_logratio)

    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
        accuracy = (chosen_reward > rejected_reward).float().mean()
        metrics = DPOStepMetrics(
            loss=loss.item(),
            chosen_reward=chosen_reward.mean().item(),
            rejected_reward=rejected_reward.mean().item(),
            reward_margin=(chosen_reward - rejected_reward).mean().item(),
            accuracy=accuracy.item(),
        )
    return loss, metrics


class DPOTrainer:
    """Trains a policy model against a frozen reference model on preference
    pairs. Structurally mirrors ats.training.trainer.Trainer (same
    DeepSpeed init, checkpoint manager, monitor, scheduler), but the
    optimization objective is dpo_loss instead of next-token cross-entropy,
    and every step needs two extra frozen-model forward passes (reference
    chosen/rejected) that never receive gradients.

    The reference model is a plain nn.Module run directly (NOT wrapped in
    DeepSpeed) in eval mode under torch.no_grad(): it never trains, so it
    doesn't need ZeRO sharding, an optimizer, or mixed-precision autocast
    management -- wrapping it in DeepSpeed would add all of that complexity
    for a component that's pure inference. This does mean the reference
    model needs to fit in memory alongside the (DeepSpeed-managed) policy
    model, unsharded; for models that don't fit twice, run reference
    log-probs as a separate precompute pass instead (not implemented here).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: ATSConfig,
        micro_batch_size: int,
        beta: float = 0.1,
    ) -> None:
        self.config = config
        self.beta = beta
        self.reference_model = reference_model
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.model_engine, self.optimizer, _, _ = initialize_engine(
            policy_model, config, micro_batch_size
        )
        self.scheduler = WarmupCosineScheduler(
            base_lr=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )
        self.monitor = Monitor(config.logging)
        self.checkpoint_manager = CheckpointManager(config)
        self.global_step = 0

    def _sequence_logps(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        output = model(input_ids)
        shift_logits = output.logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        return compute_sequence_logprobs(shift_logits, shift_labels)

    def compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, DPOStepMetrics]:
        """The DeepSpeed-independent core of a DPO step: four forward passes
        (policy/reference x chosen/rejected) plus dpo_loss. Deliberately
        split out from train_step so it's directly testable against a plain
        nn.Module (self.model_engine and self.reference_model are both just
        called via their forward() method here, nothing DeepSpeed-specific)
        without needing deepspeed installed at all -- see
        tests/test_dpo.py::test_dpotrainer_compute_loss_end_to_end, which
        constructs a DPOTrainer via __new__ (bypassing __init__, so
        initialize_engine() is never called) with model_engine and
        reference_model set directly to plain ATSTransformer instances."""
        policy_chosen_logps = self._sequence_logps(
            self.model_engine, batch["chosen_input_ids"], batch["chosen_labels"]
        )
        policy_rejected_logps = self._sequence_logps(
            self.model_engine, batch["rejected_input_ids"], batch["rejected_labels"]
        )
        with torch.no_grad():
            ref_chosen_logps = self._sequence_logps(
                self.reference_model, batch["chosen_input_ids"], batch["chosen_labels"]
            )
            ref_rejected_logps = self._sequence_logps(
                self.reference_model,
                batch["rejected_input_ids"],
                batch["rejected_labels"],
            )

        return dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> DPOStepMetrics:
        """batch must contain chosen_input_ids/chosen_labels and
        rejected_input_ids/rejected_labels (see ats.data.preference_dataset
        for the expected shapes/masking convention: labels use IGNORE_INDEX
        over the prompt portion, matching every other labels tensor in this
        codebase)."""
        self._set_lr(self.scheduler.get_lr(self.global_step))

        loss, metrics = self.compute_loss(batch)

        self.model_engine.backward(loss)
        self.model_engine.step()
        self.global_step += 1

        tokens_per_step = (
            batch["chosen_input_ids"].numel() + batch["rejected_input_ids"].numel()
        )
        self.monitor.log(
            self.global_step,
            {
                "loss": metrics.loss,
                "reward/chosen": metrics.chosen_reward,
                "reward/rejected": metrics.rejected_reward,
                "reward/margin": metrics.reward_margin,
                "reward/accuracy": metrics.accuracy,
                "lr": self.scheduler.get_lr(self.global_step),
            },
            tokens_per_step,
        )
        return metrics

    def _set_lr(self, lr: float) -> None:
        for param_group in self.model_engine.optimizer.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self) -> Any:
        return self.checkpoint_manager.save(
            self.model_engine, self.global_step, epoch=0
        )
