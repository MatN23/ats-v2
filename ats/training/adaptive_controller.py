"""AdaptiveController: an optional, synchronous training supervisor.

The trainer calls `controller.step(metrics)` explicitly once per optimizer
step and gets back an AdaptiveAction or None. There is no background thread,
no monkey-patching of the trainer, and no speculative "trajectory prediction"
or randomly-generated scores. Every decision is a deterministic function of
the recent metrics history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np

from ats.config.schema import AdaptiveConfig


@dataclass(frozen=True)
class TrainingMetrics:
    step: int
    loss: float
    grad_norm: float
    learning_rate: float
    expert_utilization: Optional[Dict[int, float]] = None


@dataclass(frozen=True)
class AdaptiveAction:
    type: str
    params: Dict[str, float]
    apply: bool


class TrainingHaltError(RuntimeError):
    """Raised by the trainer when the AdaptiveController issues a training_halt action."""


class AdaptiveController:
    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config
        self.enabled = config.enabled
        self.history: Deque[TrainingMetrics] = deque(maxlen=config.history_size)

        self._last_lr_adjust_step = -1_000_000
        self._consecutive_emergency_cuts = 0

    def step(self, metrics: TrainingMetrics) -> Optional[AdaptiveAction]:
        if not self.enabled:
            return None

        self.history.append(metrics)
        cfg = self.config

        # 1. Emergency: gradient explosion. No cooldown check needed for
        #    detection itself, but _make_action still enforces the cooldown
        #    so we don't cut the LR every single step while grads stay high.
        if metrics.grad_norm > cfg.grad_norm_threshold:
            action = self._make_action("emergency_lr_cut", factor=0.1, cooldown=100)
            if action is not None:
                return action

        # 2. Loss spike detection: compare the mean of the most recent
        #    spike_window losses against the mean of the spike_window before that.
        if len(self.history) >= 2 * cfg.spike_window:
            losses = [m.loss for m in self.history]
            recent = losses[-cfg.spike_window:]
            older = losses[-2 * cfg.spike_window:-cfg.spike_window]
            recent_avg = float(np.mean(recent))
            older_avg = float(np.mean(older))
            if older_avg > 0 and recent_avg > older_avg * cfg.spike_ratio:
                action = self._make_action("loss_spike_lr_cut", factor=0.5, cooldown=50)
                if action is not None:
                    return action

        # 3. Plateau detection: relative std of the loss over a longer window
        #    below a small threshold suggests LR could be boosted.
        if len(self.history) >= cfg.plateau_window:
            window = [m.loss for m in list(self.history)[-cfg.plateau_window:]]
            mean_loss = float(np.mean(window))
            if mean_loss > 0:
                rel_std = float(np.std(window)) / mean_loss
                if rel_std < cfg.plateau_rel_std:
                    action = self._make_action("plateau_lr_boost", factor=1.5, cooldown=200)
                    if action is not None:
                        return action

        # 4. MoE expert collapse: informational only, never auto-applied.
        if metrics.expert_utilization:
            usage = list(metrics.expert_utilization.values())
            if min(usage) < cfg.expert_collapse_threshold:
                return AdaptiveAction(
                    type="warn_expert_collapse",
                    params={"min_usage": min(usage), "max_usage": max(usage)},
                    apply=False,
                )

        return None

    def _make_action(self, action_type: str, factor: float, cooldown: int) -> Optional[AdaptiveAction]:
        """Enforces a cooldown between LR adjustments to prevent oscillation,
        and forces a training halt after 3 consecutive emergency cuts to
        prevent an LR death spiral."""
        step = self.history[-1].step if self.history else 0

        if step - self._last_lr_adjust_step < cooldown:
            return None

        self._last_lr_adjust_step = step

        if factor < 0.5:
            self._consecutive_emergency_cuts += 1
            if self._consecutive_emergency_cuts >= 3:
                return AdaptiveAction(
                    type="training_halt",
                    params={"reason_code": 1.0},
                    apply=True,
                )
        else:
            self._consecutive_emergency_cuts = 0

        return AdaptiveAction(
            type=action_type,
            params={"factor": factor, "min_lr": self.config.min_lr},
            apply=True,
        )
