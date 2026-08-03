"""Linear warmup followed by cosine decay to `min_lr_ratio * base_lr`."""

from __future__ import annotations

import math


class WarmupCosineScheduler:
    def __init__(
        self, base_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")
        if warmup_steps > max_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) cannot exceed max_steps ({max_steps})."
            )
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")

        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = base_lr * min_lr_ratio

    def get_lr(self, step: int) -> float:
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}.")
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if step >= self.max_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor
