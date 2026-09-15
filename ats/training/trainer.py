"""Main training loop. Owns the DeepSpeed engine, the LR schedule, checkpointing,
metrics logging, and the (optional) AdaptiveController. Trainer handles the
standard autoregressive (cross-entropy) objective; DiffusionTrainer handles
model_type="diffusion" (MSE noise-prediction objective) using the same
scheduler/checkpoint/monitor/adaptive-controller infrastructure."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from ats.config.schema import ATSConfig
from ats.model.diffusion import DiffusionLM
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.adaptive_controller import AdaptiveController, TrainingMetrics
from ats.training.checkpoint import CheckpointManager, TrainingHaltError
from ats.training.monitor import Monitor
from ats.training.scheduler import WarmupCosineScheduler
from ats.utils.device import resolve_device
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
            report.model_gb,
            report.optimizer_gb,
            report.activation_gb,
            report.total_gb,
        )
        return

    logger.info(
        "Pre-flight memory estimate: model=%.1fGB optimizer=%.1fGB "
        "activations=%.1fGB total=%.1fGB / %.1fGB available",
        report.model_gb,
        report.optimizer_gb,
        report.activation_gb,
        report.total_gb,
        report.available_gb,
    )
    if not report.fits_on_single_gpu:
        logger.warning(
            "Estimated memory (%.1fGB) exceeds 80%% of available GPU memory (%.1fGB). "
            "Suggested fix: --micro-batch-size %d --grad-accum-steps %d, or "
            "--parallelism-strategy deepspeed_zero%d, or --checkpoint-every-n-layers 1.",
            report.total_gb,
            report.available_gb,
            report.suggested_batch_size,
            report.suggested_grad_accum,
            report.suggested_zero_stage,
        )


def _log_oom_and_reraise(
    config: ATSConfig,
    micro_batch_size: int,
    grad_accum_steps: int,
    step: int,
    exc: Exception,
) -> None:
    try:
        report = estimate_memory(config, target_batch_size=micro_batch_size)
        model_gb, opt_gb, act_gb = (
            report.model_gb,
            report.optimizer_gb,
            report.activation_gb,
        )
    except ValueError:
        model_gb = opt_gb = act_gb = float("nan")

    logger.error(
        "CUDA OOM at step %d.\nModel: ~%.1f GB | Optimizer: ~%.1f GB | Activations: ~%.1f GB\n"
        "Try: --micro-batch-size %d --grad-accum-steps %d\n"
        "Or:  --checkpoint-every-n-layers 1\nOr:  --parallelism-strategy deepspeed_zero3",
        step,
        model_gb,
        opt_gb,
        act_gb,
        max(1, micro_batch_size // 2),
        grad_accum_steps * 2,
    )
    raise exc


def _preflight_chinchilla_check(
    model: nn.Module, config: ATSConfig, micro_batch_size: int
) -> None:
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
        num_params / 1e6,
        total_tokens / 1e9,
        ratio,
        chinchilla_optimal_tokens / 1e9,
    )
    if ratio < 0.5 or ratio > 2.0:
        logger.warning(
            "Configured token budget is %.2fx the Chinchilla-optimal ratio for this "
            "model's %.1fM parameters. %s Fix: adjust training.max_steps (or "
            "grad_accum_steps / micro_batch_size / parallelism.gpus) to change the "
            "token budget, if this wasn't intentional.",
            ratio,
            num_params / 1e6,
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


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _distributed_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Averages a scalar across the data-parallel group, in place.

    BUG-107: AdaptiveController's inputs must be IDENTICAL on every rank.
    Its decisions (emergency LR cut, spike cut, plateau boost, and the
    training_halt that raises TrainingHaltError) change the LR the
    optimizer uses and whether the process keeps running. Fed a rank-local
    micro-batch loss, rank 0 could cut the LR on a spike its own shard
    happened to see while rank 1 did not -- leaving ranks training the same
    all-reduced gradients with DIFFERENT learning rates, which silently
    stops being synchronous SGD. Worse, a rank-local `training_halt` raises
    on one rank only; the remaining ranks then block forever on the next
    collective. Averaging first makes every rank compute the same action
    from the same numbers.

    Cost: one all_reduce of a single scalar per optimizer step, which is
    immaterial next to the full-model gradient reduction on the same step.
    """
    if _is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / dist.get_world_size()
    return tensor


def _reduce_expert_utilization(
    utilization: Mapping[int, float] | None, device: torch.device
) -> Mapping[int, float] | None:
    """Averages per-expert utilization across ranks, for the same reason as
    _distributed_mean: expert-collapse warnings should describe the whole
    job, not one rank's shard, and must not differ between ranks. With
    expert parallelism a single rank only hosts a subset of experts, so a
    rank-local view of "expert 3 is unused" is close to meaningless.
    """
    if utilization is None or not _is_distributed():
        return utilization
    keys = sorted(utilization)
    values = torch.tensor(
        [utilization[k] for k in keys], dtype=torch.float32, device=device
    )
    values = _distributed_mean(values)
    return dict(zip(keys, values.tolist()))


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


def _restore_controller_state(
    controller: AdaptiveController, client_state: dict[str, Any]
) -> None:
    controller._last_lr_adjust_step = client_state.get(
        "controller_last_lr_adjust_step", controller._last_lr_adjust_step
    )
    controller._consecutive_emergency_cuts = client_state.get(
        "controller_consecutive_emergency_cuts", controller._consecutive_emergency_cuts
    )
    controller._consecutive_plateau_boosts = client_state.get(
        "controller_consecutive_plateau_boosts", controller._consecutive_plateau_boosts
    )


class _GradNormTracker:
    """Reports the global gradient norm for an optimizer step.

    BUG-108: the previous fallback ran clip_grad_norm_(max_norm=inf) AFTER
    model_engine.step() had already cleared the gradients, so on any engine
    that does not expose get_global_grad_norm() (a plain non-DeepSpeed
    engine, or DeepSpeed with gradient_clipping disabled) the reported norm
    was ~0.0 on every single step -- forever. That number is not just
    logged: it is fed to AdaptiveController.step() as `grad_norm`, and the
    emergency gradient-explosion check is `grad_norm > threshold`. A
    permanent 0.0 means that safety check could never fire on those
    engines, silently, while the logs showed a plausible-looking zero.

    The fix keeps the fast path intact -- when the engine reports its own
    (already globally reduced) norm, nothing extra is computed -- and only
    falls back to a manual pass for engines that have proven they do not
    report one, computing it BEFORE step() where the gradients still exist.
    The first step on such an engine still reports the degraded value,
    because whether the engine reports a norm is not knowable until it has
    been asked once; that one step is called out in the warning.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        # None = not yet known whether this engine reports its own norm.
        self._engine_reports: bool | None = (
            None if hasattr(engine, "get_global_grad_norm") else False
        )

    @property
    def needs_pre_step_norm(self) -> bool:
        return self._engine_reports is False

    def pre_step(self) -> float | None:
        """Call immediately BEFORE engine.step(). Returns a manually
        computed norm when this engine is known not to report one."""
        if not self.needs_pre_step_norm:
            return None
        return self._manual_norm()

    def post_step(self, pre_step_norm: float | None) -> float:
        """Call immediately AFTER engine.step()."""
        getter = getattr(self.engine, "get_global_grad_norm", None)
        if getter is not None:
            reported = getter()
            if reported is not None:
                self._engine_reports = True
                return float(reported)
        if pre_step_norm is not None:
            return pre_step_norm
        if self._engine_reports is None:
            self._engine_reports = False
            logger.warning(
                "This engine does not report a global gradient norm; ats will "
                "compute it directly before each optimizer step from now on. "
                "The norm reported for this first step is measured after the "
                "step cleared gradients and is therefore not meaningful."
            )
        return self._manual_norm()

    def _manual_norm(self) -> float:
        return float(
            torch.nn.utils.clip_grad_norm_(
                self.engine.parameters(), max_norm=float("inf")
            )
        )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ATSConfig,
        train_dataloader: Iterable[Any],
        eval_dataloader: Iterable[Any] | None = None,
        micro_batch_size: int = 1,
    ) -> None:
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = max(1, config.training.grad_accum_steps)

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

    @property
    def _grad_norm_tracker(self) -> _GradNormTracker:
        """Lazily bound to whatever engine this trainer currently holds.

        A property rather than an __init__ assignment so the tracker is
        always consistent with self.model_engine even when a trainer is
        assembled without running __init__ (the pattern the test suite uses
        to exercise train_step against a stub engine without standing up
        DeepSpeed), and so swapping the engine cannot leave a tracker
        holding a stale reference and reporting another engine's norms.
        """
        tracker = self.__dict__.get("_grad_norm_tracker_impl")
        if tracker is None or tracker.engine is not self.model_engine:
            tracker = _GradNormTracker(self.model_engine)
            self.__dict__["_grad_norm_tracker_impl"] = tracker
        return tracker

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
            self.global_step,
            self.epoch,
            self._adaptive_lr_multiplier,
            self._accumulation_step,
            self.grad_accum_steps,
        )

    def _set_lr(self, lr: float) -> None:
        for param_group in self.model_engine.optimizer.param_groups:
            param_group["lr"] = lr

    def _apply_adaptive_action(self, action) -> None:
        if action is not None and action.type == "warn_expert_collapse":
            logger.warning(
                "MoE expert collapse warning at step %d: min_usage=%.4f max_usage=%.4f",
                self.global_step,
                action.params["min_usage"],
                action.params["max_usage"],
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
                action.type,
                self.global_step,
                old_lr,
                new_lr,
                prev_multiplier,
                new_multiplier,
            )

    def train_step(self, batch: Any) -> TrainingMetrics | None:
        """Process one micro-batch. Returns metrics only on optimizer step boundary."""
        # Reset accumulation state at start to prevent stale gradients after exceptions
        # BUG-105: this used to hardcode torch.device(f"cuda:{local_rank}"),
        # making CPU and MPS training impossible. See ats.utils.device.
        device = resolve_device(self.model_engine)
        batch = _move_batch_to_device(batch, device)

        output = self.model_engine(
            batch["input_ids"], attention_mask=batch.get("attention_mask")
        )

        shift_logits = output.logits[..., :-1, :]
        shift_labels = batch["labels"][..., 1:]
        # BUG-122: the comment that used to sit here claimed transposing the
        # class dimension into position (instead of reshaping) avoided "a
        # full contiguous copy of the ENTIRE logits tensor". That claim is
        # false in two ways. First, the .float() upcast immediately below it
        # already materialises a full contiguous fp32 copy of the whole
        # tensor, so the copy it claimed to avoid happens regardless --
        # there is no memory saving at all. Second, cross_entropy is
        # measurably SLOWER on the transposed (channels-second,
        # non-contiguous) layout: benchmarked on CPU at batch=4,
        # seq_len=1024, vocab=32000, the transpose form took 2120 ms per
        # call against 989 ms for the flattened form -- a 2.1x pessimisation
        # in the hot path of every training step. Both forms produce
        # numerically identical losses. Reverted to the flattened form.
        # (Timings are CPU-only; this environment has no GPU, so the
        # relative cost on CUDA is reasoned from the layout, not measured.)
        #
        # .float(): cross_entropy's softmax reduction sums vocab_size exp()
        # terms per token. Each is ~1 near a max-subtracted logit, so the
        # running sum lands around vocab_size regardless of prediction
        # quality -- which alone exceeds fp16's 65504 maximum and overflows
        # the loss to inf on every step. DeepSpeed's plain fp16 mode casts
        # this reduction to fp16 with no per-op exception, unlike
        # torch.cuda.amp.autocast, which forces cross_entropy to fp32 for
        # exactly this reason. This upcast is load-bearing; do not remove it.
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.float().reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        total_loss = ce_loss + output.aux_loss

        # Integrate MTP loss when multi-token prediction head is active.
        # output.mtp_logits is a list[torch.Tensor] (one per predicted
        # offset), NOT a single tensor -- and each offset k needs labels
        # shifted by k, not the single 1-shift used for the main next-token
        # CE loss above. See ats.model.mtp.compute_mtp_loss_from_logits.
        if output.mtp_logits is not None:
            from ats.model.mtp import compute_mtp_loss_from_logits

            mtp_weight = self.config.model.mtp_loss_weight
            mtp_loss = compute_mtp_loss_from_logits(
                output.mtp_logits, batch["labels"], self.config.model.vocab_size
            )
            total_loss = total_loss + mtp_weight * mtp_loss

        scaled_loss = total_loss / self.grad_accum_steps
        self.model_engine.backward(scaled_loss)

        self._accumulation_step += 1
        self._accumulated_tokens += int(batch["input_ids"].numel())

        is_optimizer_step = self._accumulation_step == self.grad_accum_steps
        if not is_optimizer_step:
            return None

        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )

        scheduled_lr = (
            self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        )
        self._set_lr(scheduled_lr)
        # DeepSpeed's model_engine.step() already computes (and applies)
        # gradient clipping internally using config.training.grad_clip_norm
        # (see build_deepspeed_config's "gradient_clipping" key), and the
        # resulting global grad norm is retrievable afterward via
        # get_global_grad_norm(). The extra clip_grad_norm_(max_norm=inf)
        # call that used to run here recomputed the exact same global norm a
        # second time (a full reduction over every parameter's grad, i.e. a
        # second all-reduce + norm pass across the entire model on every
        # single optimizer step) purely to log it -- doubling the per-step
        # gradient-norm cost for no behavioral benefit, since max_norm=inf
        # never actually clips anything.
        # BUG-108: gradients must still exist to measure them. See
        # _GradNormTracker -- this is a no-op on engines that report their
        # own (already globally reduced) norm.
        pre_step_norm = self._grad_norm_tracker.pre_step()
        self.model_engine.step()

        # Capture actual accumulated tokens BEFORE resetting
        actual_tokens = self._accumulated_tokens
        self._accumulation_step = 0
        self._accumulated_tokens = 0

        grad_norm = self._grad_norm_tracker.post_step(pre_step_norm)

        if self.global_step % self.config.logging.log_every == 0:
            fp16_opt = self.model_engine.optimizer
            cur_scale = getattr(
                fp16_opt, "cur_scale", getattr(fp16_opt, "loss_scale", None)
            )
            overflow = getattr(fp16_opt, "overflow", None)
            if cur_scale is not None:
                logger.info(
                    "fp16 loss scale at step %d: cur_scale=%s overflow_this_step=%s",
                    self.global_step,
                    cur_scale,
                    overflow,
                )

        # BUG-107: the AdaptiveController must see the SAME numbers on every
        # rank or ranks diverge in learning rate (and a rank-local
        # training_halt deadlocks the others). Reduce before constructing
        # the metrics, not after. On a single process both helpers are
        # no-ops, so this costs nothing off the distributed path.
        # grad_norm is left alone: DeepSpeed's get_global_grad_norm() is
        # already reduced over the whole job.
        reduced_loss = _distributed_mean(ce_loss.detach().float())
        metrics = TrainingMetrics(
            step=self.global_step,
            loss=float(reduced_loss.item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
            expert_utilization=_reduce_expert_utilization(
                output.expert_utilization, device
            ),
        )

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)

        # Attach actual token count to metrics for accurate throughput
        # logging. metrics is a frozen dataclass (see TrainingMetrics --
        # immutable by design), so this must produce a NEW instance via
        # dataclasses.replace rather than assigning to metrics directly
        # (direct assignment raises FrozenInstanceError -- see
        # TrainingMetrics.tokens_this_step's docstring for how this was
        # found).
        metrics = dataclasses.replace(metrics, tokens_this_step=actual_tokens)
        return metrics

    def train(self, max_steps: int | None = None) -> None:
        target_steps = (
            max_steps if max_steps is not None else self.config.training.max_steps
        )
        train_iter = iter(self.train_dataloader)

        while self.global_step < target_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    raise RuntimeError(
                        "train_dataloader is empty: iterating it produced no batches "
                        "at all, even immediately after being freshly re-created. Fix: "
                        "check that data.sources point at real, non-empty data, and "
                        "that batch_size/seq_length aren't larger than the available "
                        "data."
                    ) from None

            try:
                metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError as exc:
                _log_oom_and_reraise(
                    self.config,
                    self.micro_batch_size,
                    self.grad_accum_steps,
                    self.global_step,
                    exc,
                )
                continue  # unreachable (raises), satisfies static analysis
            except TrainingHaltError:
                raise  # propagate halt without catching

            if metrics is None:
                continue

            tokens_per_step = getattr(
                metrics, "tokens_this_step", self._accumulated_tokens
            )
            self.monitor.log(
                self.global_step,
                {
                    "loss": metrics.loss,
                    "grad_norm": metrics.grad_norm,
                    "lr": metrics.learning_rate,
                },
                tokens_per_step,
            )

            self.global_step += 1

            if (
                self.config.training.eval_every > 0
                and self.global_step % self.config.training.eval_every == 0
                and self.eval_dataloader is not None
            ):
                self.evaluate()

            if (
                self.config.training.save_every > 0
                and self.global_step % self.config.training.save_every == 0
            ):
                self.checkpoint_manager.save(
                    self.model_engine,
                    self.global_step,
                    self.epoch,
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
        # BUG-105: this used to hardcode torch.device(f"cuda:{local_rank}"),
        # making CPU and MPS training impossible. See ats.utils.device.
        device = resolve_device(self.model_engine)

        total_loss = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, dtype=torch.long, device=device)

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = _move_batch_to_device(batch, device)
                output = self.model_engine(
                    batch["input_ids"], attention_mask=batch.get("attention_mask")
                )
                shift_logits = output.logits[..., :-1, :]
                shift_labels = batch["labels"][..., 1:]
                # See BUG-122 in train_step: the transpose form is slower
                # and saves no memory (.float() materialises the copy
                # either way). .float() itself is load-bearing -- it avoids
                # the same vocab-size-driven fp16 overflow in the per-token
                # logsumexp reduction, which reduction="sum" does not
                # prevent since the overflow happens per token, before the
                # sum.
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.float().reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
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
        logger.info(
            "Eval at step %d: loss=%.4f, perplexity=%.4f",
            self.global_step,
            avg_loss,
            perplexity,
        )
        return perplexity


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: ATSConfig,
        train_dataloader: Iterable[Any],
        eval_dataloader: Iterable[Any] | None = None,
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

    @property
    def _grad_norm_tracker(self) -> _GradNormTracker:
        """Lazily bound to whatever engine this trainer currently holds.

        A property rather than an __init__ assignment so the tracker is
        always consistent with self.model_engine even when a trainer is
        assembled without running __init__ (the pattern the test suite uses
        to exercise train_step against a stub engine without standing up
        DeepSpeed), and so swapping the engine cannot leave a tracker
        holding a stale reference and reporting another engine's norms.
        """
        tracker = self.__dict__.get("_grad_norm_tracker_impl")
        if tracker is None or tracker.engine is not self.model_engine:
            tracker = _GradNormTracker(self.model_engine)
            self.__dict__["_grad_norm_tracker_impl"] = tracker
        return tracker

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
                self.global_step,
                action.params["min_usage"],
                action.params["max_usage"],
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
            new = min(
                max(prev * factor, self._min_adaptive_multiplier),
                self._max_adaptive_multiplier,
            )
            self._adaptive_lr_multiplier = new
            scheduled_lr = self.scheduler.get_lr(self.global_step)
            old_lr = self.model_engine.optimizer.param_groups[0]["lr"]
            new_lr = max(scheduled_lr * new, action.params["min_lr"])
            self._set_lr(new_lr)
            logger.warning(
                "AdaptiveController applied %s at step %d: lr %.3e -> %.3e (multiplier %.3f -> %.3f)",
                action.type,
                self.global_step,
                old_lr,
                new_lr,
                prev,
                new,
            )

    def train_step(self, batch: Any) -> TrainingMetrics | None:
        # BUG-105: this used to hardcode torch.device(f"cuda:{local_rank}"),
        # making CPU and MPS training impossible. See ats.utils.device.
        device = resolve_device(self.model_engine)
        batch = _move_batch_to_device(batch, device)

        output = self.model_engine(
            batch["input_ids"],
            embed_tokens=self._embed_tokens,
            attention_mask=batch.get("attention_mask"),
        )
        mse_loss = output.loss

        scaled_loss = mse_loss / self.grad_accum_steps
        self.model_engine.backward(scaled_loss)

        self._accumulation_step += 1
        self._accumulated_tokens += int(batch["input_ids"].numel())

        if self._accumulation_step != self.grad_accum_steps:
            return None

        self._adaptive_lr_multiplier = (
            1.0 + (self._adaptive_lr_multiplier - 1.0) * self._adaptive_multiplier_decay
        )
        scheduled_lr = (
            self.scheduler.get_lr(self.global_step) * self._adaptive_lr_multiplier
        )
        self._set_lr(scheduled_lr)
        # See Trainer.train_step for why the pre-step clip_grad_norm_(inf)
        # call that used to run here was removed: DeepSpeed's step() already
        # computes the global grad norm internally, so recomputing it here
        # was a second full norm pass over every parameter's gradient on
        # every optimizer step, purely for logging.
        # BUG-108: see Trainer.train_step / _GradNormTracker.
        pre_step_norm = self._grad_norm_tracker.pre_step()
        self.model_engine.step()

        actual_tokens = self._accumulated_tokens
        self._accumulation_step = 0
        self._accumulated_tokens = 0

        grad_norm = self._grad_norm_tracker.post_step(pre_step_norm)

        # BUG-107: same cross-rank reduction as Trainer.train_step -- the
        # controller's LR actions and halt decision must be identical on
        # every rank.
        reduced_loss = _distributed_mean(mse_loss.detach().float())
        metrics = TrainingMetrics(
            step=self.global_step,
            loss=float(reduced_loss.item()),
            grad_norm=grad_norm,
            learning_rate=self.model_engine.optimizer.param_groups[0]["lr"],
        )
        # Attach actual token count to metrics for accurate throughput
        # logging. metrics is a frozen dataclass (see TrainingMetrics --
        # immutable by design), so this must produce a NEW instance via
        # dataclasses.replace rather than assigning to metrics directly
        # (direct assignment raises FrozenInstanceError on every step --
        # see TrainingMetrics.tokens_this_step's docstring for how this
        # was found).
        metrics = dataclasses.replace(metrics, tokens_this_step=actual_tokens)

        action = self.adaptive_controller.step(metrics)
        self._apply_adaptive_action(action)
        return metrics

    def train(self, max_steps: int | None = None) -> None:
        target_steps = (
            max_steps if max_steps is not None else self.config.training.max_steps
        )
        train_iter = iter(self.train_dataloader)

        while self.global_step < target_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                try:
                    batch = next(train_iter)
                except StopIteration:
                    raise RuntimeError(
                        "train_dataloader is empty: iterating it produced no batches "
                        "at all, even immediately after being freshly re-created. Fix: "
                        "check that data.sources point at real, non-empty data, and "
                        "that batch_size/seq_length aren't larger than the available "
                        "data."
                    ) from None

            try:
                metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError as exc:
                _log_oom_and_reraise(
                    self.config,
                    self.micro_batch_size,
                    self.grad_accum_steps,
                    self.global_step,
                    exc,
                )
                continue
            except TrainingHaltError:
                raise

            if metrics is None:
                continue

            tokens_per_step = getattr(
                metrics, "tokens_this_step", self._accumulated_tokens
            )
            self.monitor.log(
                self.global_step,
                {
                    "mse_loss": metrics.loss,
                    "grad_norm": metrics.grad_norm,
                    "lr": metrics.learning_rate,
                },
                tokens_per_step,
            )

            self.global_step += 1

            if (
                self.config.training.eval_every > 0
                and self.global_step % self.config.training.eval_every == 0
                and self.eval_dataloader is not None
            ):
                self.evaluate()

            if (
                self.config.training.save_every > 0
                and self.global_step % self.config.training.save_every == 0
            ):
                self.checkpoint_manager.save(
                    self.model_engine,
                    self.global_step,
                    self.epoch,
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
                "DiffusionTrainer.evaluate() called without eval_dataloader."
            )
        self.model_engine.eval()
        # BUG-105: this used to hardcode torch.device(f"cuda:{local_rank}"),
        # making CPU and MPS training impossible. See ats.utils.device.
        device = resolve_device(self.model_engine)

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
                    batch["input_ids"],
                    embed_tokens=self._embed_tokens,
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
        logger.info(
            "Diffusion eval at step %d: mse_loss=%.6f", self.global_step, avg_loss
        )
        return avg_loss