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
            "--parallelism-strategy deepspeed_zero%d, or --checkpoint-every-n-layers 1.",
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
        "Or:  --checkpoint-every-n-layers 1\n"
        "Or:  --parallelism-strategy deepspeed_zero3",
        step, model_gb, opt_gb, act_gb,
        max(1, micro_batch_size // 2), grad_accum_steps * 2,
    )
    raise exc


def _preflight_chinchilla_check(model: nn.Module, config: ATSConfig, micro_batch_size: int) -> None:
    """Logs the configured total training-token budget against the
    Chinchilla-optimal ratio of ~20 tokens per parameter (Hoffmann et al.
    2022), warning if it's off by more than 2x in either direction. This is
    a compute-efficiency heuristic, not a hard requirement -- a smaller
    token budget trades final loss for cheaper training, and a larger one
    trades compute for a model that's cheaper to run at inference; either
    can be the right call depending on what the run is actually for."""
    num_params = sum(p.numel() for p in model.parameters())
    if num_params == 0:
        return

    total_tokens = (
        config.training.max_steps
        * config.training.grad_accum_steps
        * micro_batch_size
        * config.parallelism.gpus
        * config.parallelism.nodes
        * config.data.seq_length
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
            "model's %.1fM parameters. %s "
            "Fix: adjust training.max_steps (or grad_accum_steps / micro_batch_size / "
            "parallelism.gpus) to change the token budget, if this wasn't intentional.",
            ratio, num_params / 1e6,
            "This significantly under-trains the model relative to its size."
            if ratio < 0.5 else
            "This significantly over-trains the model relative to its size "
            "(diminishing returns on loss, though it can still be worthwhile for a "
            "smaller model that's cheaper to run at inference).",
        )


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    """Moves every tensor value in a batch dict onto the given device,
    leaving non-tensor values untouched. Without this, batches produced by
    a CPU dataloader stay on CPU while the model (wrapped in a DeepSpeed
    engine) lives on the GPU, and the first embedding lookup raises
    RuntimeError: Expected all tensors to be on the same device."""
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


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
        _preflight_chinchilla_check(model, config, micro_batch_size)

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

        # The AdaptiveController used to communicate purely via an absolute
        # LR value written directly into the optimizer. That value got
        # unconditionally overwritten by the next call to
        # scheduler.get_lr(global_step) on the very next step (get_lr is a
        # pure function of the step count -- it has no memory of adaptive
        # adjustments), so any boost/cut only ever affected a single
        # optimizer step before vanishing. Instead, the controller now
        # adjusts a persistent multiplier that's applied on top of the
        # schedule every step (effective_lr = scheduled_lr * multiplier),
        # decays back toward 1.0 over time, and is clamped (via
        # adaptive.max_lr_multiplier / min_lr_multiplier) so repeated
        # plateau boosts can't compound the LR arbitrarily high. See
        # _apply_adaptive_action and train_step.
        self._adaptive_lr_multiplier = 1.0
        self._max_adaptive_multiplier = config.adaptive.max_lr_multiplier
        self._min_adaptive_multiplier = config.adaptive.min_lr_multiplier
        self._adaptive_multiplier_decay = config.adaptive.lr_multiplier_decay

        self.global_step = 0
        self.epoch = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]

    def _set_lr(self, lr: float) -> None:
        # Write through model_engine.optimizer, not the separately-returned
        # self.optimizer reference from initialize_engine() -- under bf16,
        # DeepSpeed wraps the base optimizer, and there is no guarantee that
        # reference stays in sync with the object model_engine actually
        # calls .step() on internally. If it doesn't, LR changes here would
        # silently never reach the real optimizer step -- which would look
        # exactly like warmup never happening -- while the logged LR value
        # still reports the intended schedule. All LR reads elsewhere in
        # this class go through the same model_engine.optimizer reference
        # for the same reason.
        for param_group in self.model_engine.optimizer.param_groups:
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
            factor = action.params["factor"]
            prev_multiplier = self._adaptive_lr_multiplier
            new_multiplier = prev_multiplier * factor
            # Clamp so repeated plateau boosts can't compound the effective
            # LR arbitrarily far above the schedule, and so repeated cuts
            # can't collapse it to (near) zero either.
            new_multiplier = min(new_multiplier, self._max_adaptive_multiplier)
            new_multiplier = max(new_multiplier, self._min_adaptive_multiplier)
            self._adaptive_lr_multiplier = new_multiplier

            scheduled_lr = self.scheduler.get_lr(self.global_step)
            old_lr = self.model_engine.optimizer.param_groups[0]["lr"]
            new_lr = max(scheduled_lr * new_multiplier, action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e "
                "(multiplier %.3f -> %.3f)",
                action.type, self.global_step, old_lr, new_lr,
                prev_multiplier, new_multiplier,
            )

    def train_step(self, batch: Any) -> TrainingMetrics:
        batch = _move_batch_to_device(batch, self.model_engine.local_rank)
        output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
        # Bug 12 fix: .reshape() handles non-contiguous tensors itself (it
        # only copies when the view can't be expressed as a stride change),
        # so the explicit .contiguous() calls before .view() were an
        # unconditional copy that .reshape() makes unnecessary.
        shift_logits = output.logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
            ignore_index=-100,
        )
        total_loss = ce_loss + output.aux_loss

        self.model_engine.backward(total_loss)
        # Measure grad norm before step(): DeepSpeed clears/zeros gradients
        # internally as part of completing the optimizer update at ZeRO
        # stage 0, so anything measured after step() sees already-cleared
        # (or near-zero residual) grads instead of the real ones from
        # backward(). max_norm=inf means this only measures and never
        # clips -- clipping already happens inside model_engine.step() via
        # the gradient_clipping value in the DeepSpeed config.
        pre_step_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), max_norm=float("inf"))
        )

        # Let any active adaptive multiplier from a previous step relax a
        # little toward 1.0 before computing this step's LR, so a past
        # boost/cut fades out gradually instead of persisting forever.
        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )

        # Set the LR *before* stepping the optimizer, not after -- setting
        # it after step() means this step's update actually uses whatever
        # LR was left over from the end of the *previous* step (one-step
        # lag), and for adaptive actions specifically it meant the schedule
        # would immediately overwrite them again next step, before the
        # optimizer ever took a step at the intended value. See the
        # AdaptiveController-related comment in __init__.
        scheduled_lr = self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        self._set_lr(scheduled_lr)

        self.model_engine.step()

        # Prefer DeepSpeed's own accounting when it's actually populated
        # (e.g. under ZeRO stage 2/3 with a mixed-precision optimizer
        # wrapper); fall back to the value measured above otherwise.
        # Bug 7 fix: `get_global_grad_norm() or 0.0` treated a legitimate
        # zero gradient norm the same as "not populated", silently discarding
        # a real zero and substituting pre_step_grad_norm instead. Check
        # for None explicitly so an actual zero is trusted.
        grad_norm = self.model_engine.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = pre_step_grad_norm
        else:
            grad_norm = float(grad_norm)

        # fp16 (unlike bf16) uses dynamic loss scaling: DeepSpeed scales the
        # loss up before backward() and unscales before the real optimizer
        # update, and silently SKIPS the optimizer step (no parameter
        # update, just a scale-halving) whenever it detects an overflow.
        # That would look exactly like a plateau in the logged loss/step
        # curve without any error being raised. Surfacing loss scale and
        # overflow status here so a run of skipped steps is visible instead
        # of invisible. Attribute names vary across DeepSpeed versions, so
        # this degrades to a no-op rather than crashing if unavailable.
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
            step=self.global_step,
            loss=float(ce_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
            expert_utilization=output.expert_utilization,
        )

        # Runs *after* this step's optimizer update. Any action it returns
        # adjusts self._adaptive_lr_multiplier, which takes effect starting
        # with the LR computed at the top of the *next* train_step call --
        # it deliberately does not retroactively change the update that was
        # just applied.
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
                batch = _move_batch_to_device(batch, self.model_engine.local_rank)
                output = self.model_engine(batch["input_ids"], attention_mask=batch.get("attention_mask"))
                # Bug 12 fix: same .contiguous()+.view() -> .reshape() cleanup
                # as Trainer.train_step, applied here in evaluate() too.
                shift_logits = output.logits[..., :-1, :]
                shift_labels = batch["labels"][..., 1:]
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
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

        # See the matching comment in Trainer.__init__ -- same fix, applied
        # here so DiffusionTrainer's LR handling stays consistent with the
        # autoregressive Trainer instead of re-diverging.
        self._adaptive_lr_multiplier = 1.0
        self._max_adaptive_multiplier = config.adaptive.max_lr_multiplier
        self._min_adaptive_multiplier = config.adaptive.min_lr_multiplier
        self._adaptive_multiplier_decay = config.adaptive.lr_multiplier_decay

        self.global_step = 0
        self.epoch = 0

    def resume(self, checkpoint_dir: str) -> None:
        client_state = self.checkpoint_manager.load(self.model_engine, checkpoint_dir)
        self.global_step = client_state["global_step"]
        self.epoch = client_state["epoch"]

    def _set_lr(self, lr: float) -> None:
        for param_group in self.model_engine.optimizer.param_groups:
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
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e "
                "(multiplier %.3f -> %.3f)",
                action.type, self.global_step, old_lr, new_lr,
                prev_multiplier, new_multiplier,
            )

    def train_step(self, batch: Any) -> TrainingMetrics:
        batch = _move_batch_to_device(batch, self.model_engine.local_rank)
        output = self.model_engine(batch["input_ids"], embed_tokens=self._embed_tokens)
        mse_loss = output.loss  # DiffusionOutput.loss, computed via MSE in DiffusionLM.forward

        self.model_engine.backward(mse_loss)
        # See Trainer.train_step: measure before step() clears gradients.
        pre_step_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), max_norm=float("inf"))
        )

        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )

        # See Trainer.train_step: LR must be set before step(), not after,
        # so this step's update actually uses it.
        scheduled_lr = self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        self._set_lr(scheduled_lr)

        self.model_engine.step()

        # Bug 7 fix: see Trainer.train_step -- distinguish "not populated"
        # (None) from a legitimate zero grad norm instead of collapsing both
        # to 0.0 via `or`.
        grad_norm = self.model_engine.get_global_grad_norm()
        if grad_norm is None:
            grad_norm = pre_step_grad_norm
        else:
            grad_norm = float(grad_norm)

        metrics = TrainingMetrics(
            step=self.global_step,
            loss=float(mse_loss.detach().item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
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
                batch = _move_batch_to_device(batch, self.model_engine.local_rank)
                output = self.model_engine(batch["input_ids"], embed_tokens=self._embed_tokens)
                total_loss += float(output.loss.item())
                num_batches += 1
        self.model_engine.train()

        if num_batches == 0:
            raise ValueError("Diffusion eval dataloader produced zero batches.")
        avg_loss = total_loss / num_batches
        logger.info("Diffusion eval at step %d: mse_loss=%.6f", self.global_step, avg_loss)
        return avg_loss