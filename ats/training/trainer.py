"""Main training loop. Owns the DeepSpeed engine, the LR schedule, checkpointing,
metrics logging, and the (optional) AdaptiveController. Trainer handles the
standard autoregressive (cross-entropy) objective; DiffusionTrainer handles
model_type="diffusion" (MSE noise-prediction objective) using the same
scheduler/checkpoint/monitor/adaptive-controller infrastructure."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import torch
import torch.nn as nn

from ats.config.schema import ATSConfig
from ats.model.diffusion import DiffusionLM
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.adaptive_controller import AdaptiveController, TrainingMetrics
from ats.training.checkpoint import CheckpointManager, TrainingHaltError
from ats.training.monitor import Monitor
from ats.training.scheduler import WarmupCosineScheduler
from ats.utils.logging_utils import get_logger
from ats.utils.memory import estimate_memory

logger = get_logger("ats.training.trainer")


def _preflight_memory_check(config: ATSConfig, micro_batch_size: int) -> None:
    """Runs before any GPU memory is allocated for training. Logs a warning
    table (not a hard failure -- the estimate is a heuristic) if the
    estimated peak memory exceeds 80% of detected GPU memory."""
    try:
        report = estimate_memory(config, target_batch_size=micro_batch_size)
    except ValueError as exc:
        logger.warning("Skipping pre-flight memory estimate: %s", exc)
        return

    if report.available_gb <= 0:
        logger.info(
            "Pre-flight memory estimate: model=%.1fGB optimizer=%.1fGB "
            "activations=%.1fGB total=%.1fGB (no GPU detected to compare against)",
            report.model_gb, report.optimizer_gb, report.activation_gb, report.total_gb,
        )
        return

    logger.info(
        "Pre-flight memory estimate: model=%.1fGB optimizer=%.1fGB "
        "activations=%.1fGB total=%.1fGB / %.1fGB available",
        report.model_gb, report.optimizer_gb, report.activation_gb,
        report.total_gb, report.available_gb,
    )
    if not report.fits_on_single_gpu:
        logger.warning(
            "Estimated memory (%.1fGB) exceeds 80%% of available GPU memory (%.1fGB). "
            "Suggested fix: --micro-batch-size %d --grad-accum-steps %d, or "
            "--parallelism-strategy deepspeed_zero%d, or --gradient-checkpointing.",
            report.total_gb, report.available_gb,
            report.suggested_batch_size, report.suggested_grad_accum, report.suggested_zero_stage,
        )


def _log_oom_and_reraise(
    config: ATSConfig, micro_batch_size: int, grad_accum_steps: int, step: int, exc: Exception,
) -> None:
    """Logs an actionable OOM message (estimated memory breakdown + concrete
    suggested flags) and re-raises the original exception immediately --
    this is NOT a bare except that swallows the error."""
    try:
        report = estimate_memory(config, target_batch_size=micro_batch_size)
        model_gb, opt_gb, act_gb = report.model_gb, report.optimizer_gb, report.activation_gb
    except ValueError:
        model_gb = opt_gb = act_gb = float("nan")

    logger.error(
        "CUDA OOM at step %d.\n"
        "Model: ~%.1f GB | Optimizer: ~%.1f GB | Activations: ~%.1f GB\n"
        "Try: --micro-batch-size %d --grad-accum-steps %d\n"
        "Or:  --gradient-checkpointing\n"
        "Or:  --parallelism-strategy deepspeed_zero3",
        step, model_gb, opt_gb, act_gb,
        max(1, micro_batch_size // 2), grad_accum_steps * 2,
    )
    raise exc


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ATSConfig,
        train_dataloader: Iterator[Any],
        eval_dataloader: Optional[Iterator[Any]] = None,
        micro_batch_size: int = 1,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.micro_batch_size = micro_batch_size

        _preflight_memory_check(config, micro_batch_size)

        self.model_engine, self.optimizer, _, _ = initialize_engine(
            model, config, micro_batch_size
        )

        self.scheduler = WarmupCosineScheduler(
            base_lr=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )
        self.checkpoint_manager = CheckpointManager(config)
        self.monitor = Monitor(config.logging)
        self.adaptive_controller = AdaptiveController(config.adaptive)

        self.global_step = 0
        self.epoch = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _apply_adaptive_action(self, action) -> None:
        if action is None or not action.apply:
            return
        if action.type == "training_halt":
            raise TrainingHaltError(
                f"AdaptiveController halted training at step {self.global_step}: "
                f"3 consecutive emergency LR cuts were triggered. This usually means "
                f"the learning rate is far too high for this model/data combination. "
                f"Fix: lower training.learning_rate and resume from the last good checkpoint."
            )
        if action.type in ("emergency_lr_cut", "loss_spike_lr_cut", "plateau_lr_boost"):
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * action.params["factor"], action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e",
                action.type, self.global_step, current_lr, new_lr,
            )

    def train_step(self, batch: Any) -> TrainingMetrics:
        output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100,
        )
        total_loss = ce_loss + output.aux_loss

        self.model_engine.backward(total_loss)
        self.model_engine.step()

        grad_norm = float(self.model_engine.get_global_grad_norm() or 0.0)
        scheduled_lr = self.scheduler.get_lr(self.global_step)
        self._set_lr(scheduled_lr)

        metrics = TrainingMetrics(
            step=self.global_step,
            loss=float(ce_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)

        return metrics

    def train(self, max_steps: Optional[int] = None) -> None:
        target_steps = max_steps if max_steps is not None else self.config.training.max_steps
        train_iter = iter(self.train_dataloader)

        while self.global_step < target_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            try:
                metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError as exc:
                _log_oom_and_reraise(
                    self.config, self.micro_batch_size, self.config.training.grad_accum_steps,
                    self.global_step, exc,
                )
            tokens_per_step = int(batch["input_ids"].numel())
            self.monitor.log(
                self.global_step,
                {"loss": metrics.loss, "grad_norm": metrics.grad_norm, "lr": metrics.learning_rate},
                tokens_per_step,
            )

            self.global_step += 1

            if self.config.training.eval_every > 0 and self.global_step % self.config.training.eval_every == 0:
                if self.eval_dataloader is not None:
                    self.evaluate()

            if self.config.training.save_every > 0 and self.global_step % self.config.training.save_every == 0:
                self.checkpoint_manager.save(self.model_engine, self.global_step, self.epoch)

        self.monitor.close()

    def evaluate(self) -> float:
        if self.eval_dataloader is None:
            raise ValueError(
                "Trainer.evaluate() was called but no eval_dataloader was provided to Trainer(). "
                "Fix: pass eval_dataloader=... when constructing the Trainer."
            )
        self.model_engine.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                shift_logits = output.logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                    ignore_index=-100, reduction="sum",
                )
                num_valid = (shift_labels != -100).sum().item()
                total_loss += float(loss.item())
                total_tokens += int(num_valid)
        self.model_engine.train()

        if total_tokens == 0:
            raise ValueError(
                "Evaluation dataloader produced zero valid (non-ignored) label tokens. "
                "Fix: check that eval data and label masking are configured correctly."
            )
        avg_loss = total_loss / total_tokens
        perplexity = float(torch.exp(torch.tensor(avg_loss)))
        logger.info("Eval at step %d: loss=%.4f, perplexity=%.4f", self.global_step, avg_loss, perplexity)
        return perplexity


class DiffusionTrainer:
    """Training loop for model_type="diffusion". Wraps the ATSTransformer
    backbone in a DiffusionLM before handing it to DeepSpeed, and trains with
    the MSE noise-prediction objective instead of cross-entropy. Reuses the
    same scheduler, checkpoint manager, monitor, and adaptive controller as
    the autoregressive Trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: ATSConfig,
        train_dataloader: Iterator[Any],
        eval_dataloader: Optional[Iterator[Any]] = None,
        micro_batch_size: int = 1,
    ) -> None:
        if config.model.model_type != "diffusion":
            raise ValueError(
                f"DiffusionTrainer requires config.model.model_type == 'diffusion', "
                f"got '{config.model.model_type}'. Fix: use Trainer for autoregressive "
                f"models, or set model.model_type: diffusion."
            )
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.micro_batch_size = micro_batch_size

        _preflight_memory_check(config, micro_batch_size)

        # Keep a direct reference to the embedding table before wrapping in
        # DeepSpeed, so train_step can pass it into DiffusionLM.forward
        # without having to reach through model_engine.module each step.
        self._embed_tokens = model.embed_tokens

        diffusion_model = DiffusionLM(
            backbone=model,
            hidden_size=config.model.hidden_size,
            num_timesteps=config.model.diffusion_num_timesteps,
        )

        self.model_engine, self.optimizer, _, _ = initialize_engine(
            diffusion_model, config, micro_batch_size
        )

        self.scheduler = WarmupCosineScheduler(
            base_lr=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )
        self.checkpoint_manager = CheckpointManager(config)
        self.monitor = Monitor(config.logging)
        self.adaptive_controller = AdaptiveController(config.adaptive)

        self.global_step = 0
        self.epoch = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]

    def _set_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _apply_adaptive_action(self, action) -> None:
        if action is None or not action.apply:
            return
        if action.type == "training_halt":
            raise TrainingHaltError(
                f"AdaptiveController halted diffusion training at step {self.global_step}: "
                f"3 consecutive emergency LR cuts were triggered. "
                f"Fix: lower training.learning_rate and resume from the last good checkpoint."
            )
        if action.type in ("emergency_lr_cut", "loss_spike_lr_cut", "plateau_lr_boost"):
            current_lr = self.optimizer.param_groups[0]["lr"]
            new_lr = max(current_lr * action.params["factor"], action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e",
                action.type, self.global_step, current_lr, new_lr,
            )

    def train_step(self, batch: Any) -> TrainingMetrics:
        output = self.model_engine(batch["input_ids"], embed_tokens=self._embed_tokens)
        mse_loss = output.loss  # DiffusionOutput.loss, computed via MSE in DiffusionLM.forward

        self.model_engine.backward(mse_loss)
        self.model_engine.step()

        grad_norm = float(self.model_engine.get_global_grad_norm() or 0.0)
        scheduled_lr = self.scheduler.get_lr(self.global_step)
        self._set_lr(scheduled_lr)

        metrics = TrainingMetrics(
            step=self.global_step,
            loss=float(mse_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)

        return metrics

    def train(self, max_steps: Optional[int] = None) -> None:
        target_steps = max_steps if max_steps is not None else self.config.training.max_steps
        train_iter = iter(self.train_dataloader)

        while self.global_step < target_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            try:
                metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError as exc:
                _log_oom_and_reraise(
                    self.config, self.micro_batch_size, self.config.training.grad_accum_steps,
                    self.global_step, exc,
                )
            tokens_per_step = int(batch["input_ids"].numel())
            self.monitor.log(
                self.global_step,
                {"mse_loss": metrics.loss, "grad_norm": metrics.grad_norm, "lr": metrics.learning_rate},
                tokens_per_step,
            )

            self.global_step += 1

            if self.config.training.eval_every > 0 and self.global_step % self.config.training.eval_every == 0:
                if self.eval_dataloader is not None:
                    self.evaluate()

            if self.config.training.save_every > 0 and self.global_step % self.config.training.save_every == 0:
                self.checkpoint_manager.save(self.model_engine, self.global_step, self.epoch)

        self.monitor.close()

    def evaluate(self) -> float:
        if self.eval_dataloader is None:
            raise ValueError(
                "DiffusionTrainer.evaluate() was called but no eval_dataloader was provided. "
                "Fix: pass eval_dataloader=... when constructing DiffusionTrainer."
            )
        self.model_engine.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.eval_dataloader:
                output = self.model_engine(batch["input_ids"], embed_tokens=self._embed_tokens)
                total_loss += float(output.loss.item())
                num_batches += 1
        self.model_engine.train()

        if num_batches == 0:
            raise ValueError("Diffusion eval dataloader produced zero batches.")
        avg_loss = total_loss / num_batches
        logger.info("Diffusion eval at step %d: mse_loss=%.6f", self.global_step, avg_loss)
        return avg_loss
