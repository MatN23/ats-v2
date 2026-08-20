"""Main training loop. Owns the DeepSpeed engine, the LR schedule, checkpointing,
metrics logging, and the (optional) AdaptiveController. Trainer handles the
standard autoregressive (cross-entropy) objective; DiffusionTrainer handles
model_type="diffusion" (MSE noise-prediction objective) using the same
scheduler/checkpoint/monitor/adaptive-controller infrastructure."""

from __future__ import annotations

import math
import torch.distributed as dist
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

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
            "--parallelism-strategy deepspeed_zero%d, or --checkpoint-every-n-layers 1.",
            report.total_gb, report.available_gb,
            report.suggested_batch_size, report.suggested_grad_accum,
            report.suggested_zero_stage,
        )


def _log_oom_and_reraise(
    config: ATSConfig, micro_batch_size: int, grad_accum_steps: int,
    step: int, exc: Exception,
) -> None:
    try:
        report = estimate_memory(config, target_batch_size=micro_batch_size)
        model_gb, opt_gb, act_gb = report.model_gb, report.optimizer_gb, report.activation_gb
    except ValueError:
        model_gb = opt_gb = act_gb = float("nan")

    logger.error(
        "CUDA OOM at step %d.\nModel: ~%.1f GB | Optimizer: ~%.1f GB | Activations: ~%.1f GB\n"
        "Try: --micro-batch-size %d --grad-accum-steps %d\n"
        "Or:  --checkpoint-every-n-layers 1\nOr:  --parallelism-strategy deepspeed_zero3",
        step, model_gb, opt_gb, act_gb, max(1, micro_batch_size // 2), grad_accum_steps * 2,
    )
    raise exc


def _preflight_chinchilla_check(model: nn.Module, config: ATSConfig, micro_batch_size: int) -> None:
    num_params = sum(p.numel() for p in model.parameters())
    if num_params == 0:
        return

    total_tokens = (
        config.training.max_steps * config.training.grad_accum_steps
        * micro_batch_size * config.parallelism.gpus
        * config.parallelism.nodes * config.data.seq_length
    )
    chinchilla_optimal_tokens = 20 * num_params
    ratio = total_tokens / chinchilla_optimal_tokens

    logger.info(
        "Chinchilla check: %.1fM params, %.2fB configured training tokens "
        "(%.2fx the ~20 tok/param Chinchilla-optimal budget of %.2fB tokens).",
        num_params / 1e6, total_tokens / 1e9, ratio, chinchilla_optimal_tokens / 1e9,
    )
    if ratio < 0.5 or ratio > 2.0:
        logger.warning(
            "Configured token budget is %.2fx the Chinchilla-optimal ratio for this "
            "model's %.1fM parameters. %s Fix: adjust training.max_steps (or "
            "grad_accum_steps / micro_batch_size / parallelism.gpus) to change the "
            "token budget, if this wasn't intentional.",
            ratio, num_params / 1e6,
            "This significantly under-trains the model relative to its size."
            if ratio < 0.5
            else "This significantly over-trains the model relative to its size.",
        )


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _distributed_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _capture_controller_state(controller: AdaptiveController) -> dict[str, Any]:
    """Snapshots the AdaptiveController's mutable internal counters so they
    survive a checkpoint/resume cycle. Without this, a resumed run forgets
    how many consecutive emergency cuts or plateau boosts already fired and
    the cooldown clock (_last_lr_adjust_step) resets to "never adjusted",
    which can immediately re-trigger an action that had just been cooled
    down before the checkpoint was taken."""
    return {
        "controller_last_lr_adjust_step": controller._last_lr_adjust_step,
        "controller_consecutive_emergency_cuts": controller._consecutive_emergency_cuts,
        "controller_consecutive_plateau_boosts": controller._consecutive_plateau_boosts,
    }


def _restore_controller_state(controller: AdaptiveController, client_state: dict[str, Any]) -> None:
    controller._last_lr_adjust_step = client_state.get(
        "controller_last_lr_adjust_step", controller._last_lr_adjust_step
    )
    controller._consecutive_emergency_cuts = client_state.get(
        "controller_consecutive_emergency_cuts", controller._consecutive_emergency_cuts
    )
    controller._consecutive_plateau_boosts = client_state.get(
        "controller_consecutive_plateau_boosts", controller._consecutive_plateau_boosts
    )


class Trainer:
    def __init__(
        self, model: nn.Module, config: ATSConfig,
        train_dataloader: Iterable[Any], eval_dataloader: Iterable[Any] | None = None,
        micro_batch_size: int = 1,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = max(1, config.training.grad_accum_steps)

        _preflight_memory_check(config, micro_batch_size)
        _preflight_chinchilla_check(model, config, micro_batch_size)

        self.model_engine, self.optimizer, _, _ = initialize_engine(model, config, micro_batch_size)

        self.scheduler = WarmupCosineScheduler(
            base_lr=config.training.learning_rate,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )
        # Verify scheduler is purely functional (stateless). If it maintains internal
        # state, get_lr(global_step) will return stale values and the LR schedule
        # will silently freeze. This assertion catches that class of bug at init time.
        assert not hasattr(self.scheduler, "_step_count"), (
            "WarmupCosineScheduler appears stateful (_step_count attribute found). "
            "Trainer requires a purely functional scheduler where get_lr(step) "
            "depends only on the passed step argument. Fix: make WarmupCosineScheduler "
            "stateless, or switch to calling scheduler.step() + get_last_lr()."
        )

        self.checkpoint_manager = CheckpointManager(config)
        self.monitor = Monitor(config.logging)
        self.adaptive_controller = AdaptiveController(config.adaptive)

        self._adaptive_lr_multiplier = 1.0
        self._max_adaptive_multiplier = config.adaptive.max_lr_multiplier
        self._min_adaptive_multiplier = config.adaptive.min_lr_multiplier
        self._adaptive_multiplier_decay = config.adaptive.lr_multiplier_decay

        self.global_step = 0
        self.epoch = 0
        self._accumulation_step = 0
        self._accumulated_tokens = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]
        self._adaptive_lr_multiplier = client_state.get("adaptive_lr_multiplier", 1.0)
        # Clamp restored accumulation step to valid range to prevent off-by-one
        # when grad_accum_steps changes between runs
        raw_accum = client_state.get("accumulation_step", 0)
        self._accumulation_step = min(raw_accum, self.grad_accum_steps - 1)
        self._accumulated_tokens = client_state.get("accumulated_tokens", 0)
        _restore_controller_state(self.adaptive_controller, client_state)
        logger.info(
            "Resumed at step %d, epoch %d, adaptive_multiplier=%.4f, accum_step=%d/%d",
            self.global_step, self.epoch, self._adaptive_lr_multiplier,
            self._accumulation_step, self.grad_accum_steps,
        )

    def _set_lr(self, lr: float) -> None:
        for param_group in self.model_engine.optimizer.param_groups:
            param_group["lr"] = lr

    def _apply_adaptive_action(self, action) -> None:
        if action is not None and action.type == "warn_expert_collapse":
            logger.warning(
                "MoE expert collapse warning at step %d: min_usage=%.4f max_usage=%.4f",
                self.global_step, action.params["min_usage"], action.params["max_usage"],
            )
        if action is None or not action.apply:
            return
        if action.type == "training_halt":
            # Reset accumulation state so resume doesn't carry stale partial gradients
            self._accumulation_step = 0
            self._accumulated_tokens = 0
            # Clear any partial gradients from the incomplete accumulation
            # window so they can't leak into a subsequent resumed run.
            self.model_engine.zero_grad()
            raise TrainingHaltError(
                f"AdaptiveController halted training at step {self.global_step}: "
                f"3 consecutive emergency LR cuts were triggered. Fix: lower "
                f"training.learning_rate and resume from the last good checkpoint."
            )
        if action.type in ("emergency_lr_cut", "loss_spike_lr_cut", "plateau_lr_boost"):
            factor = action.params["factor"]
            prev_multiplier = self._adaptive_lr_multiplier
            new_multiplier = prev_multiplier * factor
            new_multiplier = min(new_multiplier, self._max_adaptive_multiplier)
            new_multiplier = max(new_multiplier, self._min_adaptive_multiplier)
            self._adaptive_lr_multiplier = new_multiplier

            scheduled_lr = self.scheduler.get_lr(self.global_step)
            old_lr = self.model_engine.optimizer.param_groups[0]["lr"]
            new_lr = max(scheduled_lr * new_multiplier, action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e (multiplier %.3f -> %.3f)",
                action.type, self.global_step, old_lr, new_lr, prev_multiplier, new_multiplier,
            )

    def train_step(self, batch: Any) -> TrainingMetrics | None:
        """Process one micro-batch. Returns metrics only on optimizer step boundary."""
        # Reset accumulation state at start to prevent stale gradients after exceptions
        device = self.model_engine.local_rank if isinstance(self.model_engine.local_rank, torch.device) \
            else torch.device(f"cuda:{self.model_engine.local_rank}")
        batch = _move_batch_to_device(batch, device)

        output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))

        shift_logits = output.logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100,
        )
        total_loss = ce_loss + output.aux_loss

        # Integrate MTP loss when multi-token prediction head is active
        if hasattr(output, "mtp_logits") and output.mtp_logits is not None:
            mtp_weight = getattr(self.config.model, "mtp_loss_weight", 1.0)
            mtp_loss = torch.nn.functional.cross_entropy(
                output.mtp_logits.reshape(-1, output.mtp_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100,
            )
            total_loss = total_loss + mtp_weight * mtp_loss

        scaled_loss = total_loss / self.grad_accum_steps
        self.model_engine.backward(scaled_loss)

        self._accumulation_step += 1
        self._accumulated_tokens += int(batch["input_ids"].numel())

        is_optimizer_step = self._accumulation_step == self.grad_accum_steps
        if not is_optimizer_step:
            return None

        # Measure grad norm before step() clears gradients
        pre_step_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), max_norm=float("inf"))
        )

        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )

        scheduled_lr = self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        self._set_lr(scheduled_lr)
        self.model_engine.step()

        # Capture actual accumulated tokens BEFORE resetting
        actual_tokens = self._accumulated_tokens
        self._accumulation_step = 0
        self._accumulated_tokens = 0

        grad_norm = self.model_engine.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = pre_step_grad_norm
        else:
            grad_norm = float(grad_norm)
            # get_global_grad_norm() reflects DeepSpeed's post-clip gradients,
            # while pre_step_grad_norm was measured before that clipping was
            # applied. Surface both when they diverge so a report of "grad
            # norm looks fine" doesn't hide that clipping is doing a lot of
            # work every step (a sign the LR or grad_clip_norm may be off).
            if not math.isclose(pre_step_grad_norm, grad_norm, rel_tol=1e-3):
                logger.info(
                    "Grad norm at step %d: pre-clip=%.4f post-clip=%.4f",
                    self.global_step, pre_step_grad_norm, grad_norm,
                )

        if self.global_step % self.config.logging.log_every == 0:
            fp16_opt = self.model_engine.optimizer
            cur_scale = getattr(fp16_opt, "cur_scale", getattr(fp16_opt, "loss_scale", None))
            overflow = getattr(fp16_opt, "overflow", None)
            if cur_scale is not None:
                logger.info(
                    "fp16 loss scale at step %d: cur_scale=%s overflow_this_step=%s",
                    self.global_step, cur_scale, overflow,
                )

        metrics = TrainingMetrics(
            step=self.global_step, loss=float(ce_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
            expert_utilization=output.expert_utilization,
        )

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)

        # Attach actual token count to metrics for accurate throughput logging
        metrics.tokens_this_step = actual_tokens  # type: ignore[attr-defined]
        return metrics

    def train(self, max_steps: int | None = None) -> None:
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
                _log_oom_and_reraise(self.config, self.micro_batch_size,
                                     self.grad_accum_steps, self.global_step, exc)
                continue  # unreachable (raises), satisfies static analysis
            except TrainingHaltError:
                raise  # propagate halt without catching

            if metrics is None:
                continue

            tokens_per_step = getattr(metrics, "tokens_this_step", self._accumulated_tokens)
            self.monitor.log(self.global_step, {
                "loss": metrics.loss, "grad_norm": metrics.grad_norm, "lr": metrics.learning_rate,
            }, tokens_per_step)

            self.global_step += 1

            if (self.config.training.eval_every > 0
                    and self.global_step % self.config.training.eval_every == 0
                    and self.eval_dataloader is not None):
                self.evaluate()

            if (self.config.training.save_every > 0
                    and self.global_step % self.config.training.save_every == 0):
                self.checkpoint_manager.save(
                    self.model_engine, self.global_step, self.epoch,
                    extra_client_state={
                        "adaptive_lr_multiplier": self._adaptive_lr_multiplier,
                        "accumulation_step": self._accumulation_step,
                        "accumulated_tokens": self._accumulated_tokens,
                        **_capture_controller_state(self.adaptive_controller),
                    },
                )

        self.monitor.close()

    def evaluate(self) -> float:
        if self.eval_dataloader is None:
            raise ValueError(
                "Trainer.evaluate() called without eval_dataloader. "
                "Fix: pass eval_dataloader=... when constructing Trainer."
            )
        self.model_engine.eval()
        device = self.model_engine.local_rank if isinstance(self.model_engine.local_rank, torch.device) \
            else torch.device(f"cuda:{self.model_engine.local_rank}")

        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, dtype=torch.long, device=device)

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = _move_batch_to_device(batch, device)
                output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                shift_logits = output.logits[..., :-1, :]
                shift_labels = batch["labels"][..., 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1), ignore_index=-100, reduction="sum",
                )
                num_valid = (shift_labels != -100).sum()
                total_loss += loss
                total_tokens += num_valid

        total_loss = _distributed_sum(total_loss)
        total_tokens = _distributed_sum(total_tokens)
        self.model_engine.train()

        if total_tokens.item() == 0:
            raise ValueError(
                "Eval dataloader produced zero valid label tokens. "
                "Fix: check eval data and label masking configuration."
            )
        avg_loss = (total_loss / total_tokens).item()
        perplexity = float(torch.exp(torch.tensor(avg_loss)))
        logger.info("Eval at step %d: loss=%.4f, perplexity=%.4f", self.global_step, avg_loss, perplexity)
        return perplexity


class DiffusionTrainer:
    def __init__(
        self, model: nn.Module, config: ATSConfig,
        train_dataloader: Iterable[Any], eval_dataloader: Iterable[Any] | None = None,
        micro_batch_size: int = 1,
    ) -> None:
        if config.model.model_type != "diffusion":
            raise ValueError(
                f"DiffusionTrainer requires model_type='diffusion', got '{config.model.model_type}'."
            )
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = max(1, config.training.grad_accum_steps)

        _preflight_memory_check(config, micro_batch_size)
        self._embed_tokens = model.embed_tokens

        assert config.model.hidden_size is not None
        diffusion_model = DiffusionLM(
            backbone=model, hidden_size=config.model.hidden_size,
            num_timesteps=config.model.diffusion_num_timesteps,
        )

        self.model_engine, self.optimizer, _, _ = initialize_engine(diffusion_model, config, micro_batch_size)

        self.scheduler = WarmupCosineScheduler(
            base_lr=config.training.learning_rate, warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps, min_lr_ratio=config.training.min_lr_ratio,
        )
        assert not hasattr(self.scheduler, "_step_count"), (
            "WarmupCosineScheduler appears stateful. DiffusionTrainer requires a purely "
            "functional scheduler. Fix: make it stateless or use scheduler.step()."
        )

        self.checkpoint_manager = CheckpointManager(config)
        self.monitor = Monitor(config.logging)
        self.adaptive_controller = AdaptiveController(config.adaptive)

        self._adaptive_lr_multiplier = 1.0
        self._max_adaptive_multiplier = config.adaptive.max_lr_multiplier
        self._min_adaptive_multiplier = config.adaptive.min_lr_multiplier
        self._adaptive_multiplier_decay = config.adaptive.lr_multiplier_decay

        self.global_step = 0
        self.epoch = 0
        self._accumulation_step = 0
        self._accumulated_tokens = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]
        self._adaptive_lr_multiplier = client_state.get("adaptive_lr_multiplier", 1.0)
        raw_accum = client_state.get("accumulation_step", 0)
        self._accumulation_step = min(raw_accum, self.grad_accum_steps - 1)
        self._accumulated_tokens = client_state.get("accumulated_tokens", 0)
        _restore_controller_state(self.adaptive_controller, client_state)

    def _set_lr(self, lr: float) -> None:
        for param_group in self.model_engine.optimizer.param_groups:
            param_group["lr"] = lr

    def _apply_adaptive_action(self, action) -> None:
        if action is not None and action.type == "warn_expert_collapse":
            logger.warning(
                "MoE expert collapse warning at step %d: min_usage=%.4f max_usage=%.4f",
                self.global_step, action.params["min_usage"], action.params["max_usage"],
            )
        if action is None or not action.apply:
            return
        if action.type == "training_halt":
            self._accumulation_step = 0
            self._accumulated_tokens = 0
            self.model_engine.zero_grad()
            raise TrainingHaltError(
                f"AdaptiveController halted diffusion training at step {self.global_step}. "
                f"Fix: lower training.learning_rate and resume from last good checkpoint."
            )
        if action.type in ("emergency_lr_cut", "loss_spike_lr_cut", "plateau_lr_boost"):
            factor = action.params["factor"]
            prev = self._adaptive_lr_multiplier
            new = min(max(prev * factor, self._min_adaptive_multiplier), self._max_adaptive_multiplier)
            self._adaptive_lr_multiplier = new
            scheduled_lr = self.scheduler.get_lr(self.global_step)
            old_lr = self.model_engine.optimizer.param_groups[0]["lr"]
            new_lr = max(scheduled_lr * new, action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e (multiplier %.3f -> %.3f)",
                action.type, self.global_step, old_lr, new_lr, prev, new,
            )

    def train_step(self, batch: Any) -> TrainingMetrics | None:
        device = self.model_engine.local_rank if isinstance(self.model_engine.local_rank, torch.device) \
            else torch.device(f"cuda:{self.model_engine.local_rank}")
        batch = _move_batch_to_device(batch, device)

        output = self.model_engine(
            batch["input_ids"], embed_tokens=self._embed_tokens,
            attention_mask=batch.get("attention_mask"),
        )
        mse_loss = output.loss

        scaled_loss = mse_loss / self.grad_accum_steps
        self.model_engine.backward(scaled_loss)

        self._accumulation_step += 1
        self._accumulated_tokens += int(batch["input_ids"].numel())

        if self._accumulation_step != self.grad_accum_steps:
            return None

        pre_step_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), max_norm=float("inf"))
        )

        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )
        scheduled_lr = self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        self._set_lr(scheduled_lr)
        self.model_engine.step()

        actual_tokens = self._accumulated_tokens
        self._accumulation_step = 0
        self._accumulated_tokens = 0

        grad_norm = self.model_engine.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = pre_step_grad_norm
        else:
            grad_norm = float(grad_norm)
            if not math.isclose(pre_step_grad_norm, grad_norm, rel_tol=1e-3):
                logger.info(
                    "Grad norm at step %d: pre-clip=%.4f post-clip=%.4f",
                    self.global_step, pre_step_grad_norm, grad_norm,
                )

        metrics = TrainingMetrics(
            step=self.global_step, loss=float(mse_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
        )
        metrics.tokens_this_step = actual_tokens  # type: ignore[attr-defined]

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)
        return metrics

    def train(self, max_steps: int | None = None) -> None:
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
                _log_oom_and_reraise(self.config, self.micro_batch_size,
                                     self.grad_accum_steps, self.global_step, exc)
                continue
            except TrainingHaltError:
                raise

            if metrics is None:
                continue

            tokens_per_step = getattr(metrics, "tokens_this_step", self._accumulated_tokens)
            self.monitor.log(self.global_step, {
                "mse_loss": metrics.loss, "grad_norm": metrics.grad_norm, "lr": metrics.learning_rate,
            }, tokens_per_step)

            self.global_step += 1

            if (self.config.training.eval_every > 0
                    and self.global_step % self.config.training.eval_every == 0
                    and self.eval_dataloader is not None):
                self.evaluate()

            if (self.config.training.save_every > 0
                    and self.global_step % self.config.training.save_every == 0):
                self.checkpoint_manager.save(
                    self.model_engine, self.global_step, self.epoch,
                    extra_client_state={
                        "adaptive_lr_multiplier": self._adaptive_lr_multiplier,
                        "accumulation_step": self._accumulation_step,
                        "accumulated_tokens": self._accumulated_tokens,
                        **_capture_controller_state(self.adaptive_controller),
                    },
                )

        self.monitor.close()

    def evaluate(self) -> float:
        if self.eval_dataloader is None:
            raise ValueError("DiffusionTrainer.evaluate() called without eval_dataloader.")
        self.model_engine.eval()
        device = self.model_engine.local_rank if isinstance(self.model_engine.local_rank, torch.device) \
            else torch.device(f"cuda:{self.model_engine.local_rank}")

        total_loss = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(0, dtype=torch.long, device=device)

        # Diffusion's MSE objective is already a mean over every element of
        # the noise tensor (see DiffusionLM.forward), so each batch's
        # output.loss is already a proper per-batch average -- combining
        # batches with a plain mean of those means is the correct
        # normalization here. Weighting by a token count (as the
        # cross-entropy Trainer.evaluate() does) would double-count the
        # hidden dimension baked into each batch's already-averaged MSE and
        # is not the right normalization for a continuous objective.
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = _move_batch_to_device(batch, device)
                output = self.model_engine(
                    batch["input_ids"], embed_tokens=self._embed_tokens,
                    attention_mask=batch.get("attention_mask"),
                )
                total_loss += output.loss
                num_batches += 1

        total_loss = _distributed_sum(total_loss)
        num_batches = _distributed_sum(num_batches)
        self.model_engine.train()

        if num_batches.item() == 0:
            raise ValueError("Diffusion eval produced zero batches.")
        avg_loss = (total_loss / num_batches).item()
        logger.info("Diffusion eval at step %d: mse_loss=%.6f", self.global_step, avg_loss)
        return avg_loss