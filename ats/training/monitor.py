"""Metrics logging: structured logger always, TensorBoard/WandB optionally.
No background threads — flush happens synchronously when the trainer calls log()."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ats.config.schema import LoggingConfig
from ats.utils.logging_utils import get_logger
from ats.utils.memory import get_gpu_memory_info

logger = get_logger("ats.training.monitor")


class Monitor:
    def __init__(self, config: LoggingConfig) -> None:
        self.config = config
        self._tokens_seen_at_last_log = 0
        self._time_at_last_log: Optional[float] = None

        self._tb_writer = None
        if config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "logging.use_tensorboard=True but tensorboard is not installed. "
                    "Fix: pip install tensorboard, or set logging.use_tensorboard: false."
                ) from exc
            self._tb_writer = SummaryWriter(log_dir=f"./tb_logs/{config.project_name}")

        self._wandb = None
        if config.use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "logging.use_wandb=True but wandb is not installed. "
                    "Fix: pip install wandb, or set logging.use_wandb: false."
                ) from exc
            wandb.init(project=config.project_name)
            self._wandb = wandb

    def log(self, step: int, metrics: Dict[str, float], tokens_per_step: int) -> None:
        now = time.monotonic()
        tokens_per_sec = 0.0
        if self._time_at_last_log is not None:
            elapsed = now - self._time_at_last_log
            if elapsed > 0:
                # log() is called every step (the log_every gate below only
                # controls whether this prints), so elapsed is already the
                # duration of a single step and tokens_per_step is already
                # that step's token count -- multiplying by log_every here
                # inflated the result by that factor.
                tokens_per_sec = tokens_per_step / elapsed
        self._time_at_last_log = now

        full_metrics = dict(metrics)
        full_metrics["tokens_per_sec"] = tokens_per_sec
        mem = get_gpu_memory_info()
        full_metrics["gpu_mem_allocated_gib"] = mem["allocated_gib"]
        full_metrics["gpu_mem_reserved_gib"] = mem["reserved_gib"]

        if step % self.config.log_every == 0:
            # "lr" is forced to scientific notation always: %.4g only
            # switches to scientific when the exponent is below -4, so a
            # value like the common 3e-4 target LR (exponent exactly -4)
            # would otherwise print as a plain decimal "0.0003" instead of
            # staying consistent with smaller values like "1.5e-07".
            parts = []
            for key, value in full_metrics.items():
                if key == "lr":
                    parts.append(f"{key}={value:.4e}")
                else:
                    parts.append(f"{key}={value:.4g}")
            formatted = " | ".join(parts)
            logger.info("step=%d | %s", step, formatted)

        if self._tb_writer is not None:
            for key, value in full_metrics.items():
                self._tb_writer.add_scalar(key, value, global_step=step)
        if self._wandb is not None:
            self._wandb.log(full_metrics, step=step)

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb is not None:
            self._wandb.finish()