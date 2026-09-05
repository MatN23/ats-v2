#!/usr/bin/env python
"""Entry point: python -m ats.cli.test_modes [--steps 50] [--mode moe]

Aggressive architecture *training* smoke test. This is NOT a replacement for
the existing pytest suite (tests/test_model.py, tests/test_training.py,
etc. already unit-test individual components -- attention variants, MoE
routing, Mamba blocks, ... -- in isolation). This script asks a different,
narrower question that nothing else in the repo currently answers directly:

    "Can this whole architecture configuration actually train end-to-end?"

For each mode below, it builds a real ~1M-parameter ATSTransformer with
that mode's flags actually enabled (not guessed from the README -- see
ats/cli/train.py::_apply_architecture_preset, which this mirrors), runs 50
real optimizer steps against synthetic random token data (real forward
pass, real loss, real backward, real AdamW step -- no DeepSpeed, no mocks),
and checks: finite loss every step, finite gradients every step, and that
parameters actually moved by the end. It also sanity-checks that the
feature the mode claims to test is actually present in the constructed
module graph (e.g. "mamba" asserts a real MambaLayer exists in
model.layers), so a config flag that silently failed to reach the model
would be caught here, not just a config-validation test elsewhere.

Modes: dense, swa, mla, mamba, moe, mod, mtp, all. "all" enables every one
of the 6 architecture flags simultaneously -- exactly what
--architecture=all does on ats-train, not a hand-picked subset.

No GPU, no dataset download, no DeepSpeed: plain torch on CPU (or CUDA if
available) against ATSTransformer directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# This script only cares about "did training work", not compiled-kernel
# performance -- silence third-party diagnostic noise that has nothing to
# do with correctness: DeepSpeed's own startup accelerator-detection log,
# and torch dynamo's graph-break chatter when DeepSpeed's real MoE kernel
# (ats.model.moe's primary, non-fallback path) hits an op dynamo can't
# trace. Actual errors from either library still surface normally --
# these are diagnostics, not exceptions, so nothing is being swallowed.
logging.getLogger("DeepSpeed").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from ats.config.schema import ConfigError, ModelConfig
from ats.model.mla import MLAAttention
from ats.model.mod import MixtureOfDepths
from ats.model.transformer import ATSTransformer, MambaLayer, TransformerBlock

IGNORE_INDEX = -100
DEFAULT_STEPS = 50
SEQ_LEN = 32
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
# Minimum fraction of trainable parameters that must have moved by the end
# of training for a mode to PASS. Not 100%: e.g. an MoE expert that random
# routing never selects across 50 steps legitimately never gets a gradient
# -- that's expert-capacity behavior, not a training bug. A near-zero
# fraction, on the other hand, is exactly what "loss never actually
# backpropped" or "optimizer never actually stepped" looks like.
MIN_CHANGED_FRACTION = 0.5

MODES = ("dense", "swa", "mla", "mamba", "moe", "mod", "mtp", "all")


@dataclass
class ModeResult:
    mode: str
    passed: bool
    steps_completed: int
    total_steps: int = DEFAULT_STEPS
    last_loss: float | None = None
    error: str | None = None
    param_count: int | None = None


def _base_dims(mode: str) -> dict:
    """Hand-tuned per-mode architecture dims, each landing close to ~1M
    total parameters (see module docstring: MoE/Mamba/MLA/MTP all change
    the per-layer parameter footprint enough that one fixed hidden_size
    can't hit ~1M for every mode at once)."""
    if mode == "moe":
        return {
            "hidden_size": 120,
            "num_layers": 4,
            "num_heads": 6,
            "num_kv_heads": 2,
            "intermediate_size": 240,
            "vocab_size": 384,
            "max_seq_len": 64,
        }
    if mode == "all":
        return {
            "hidden_size": 136,
            "num_layers": 4,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 272,
            "vocab_size": 384,
            "max_seq_len": 64,
        }
    # dense, swa, mla, mamba, mod, mtp
    return {
        "hidden_size": 144,
        "num_layers": 5,
        "num_heads": 6,
        "num_kv_heads": 3,
        "intermediate_size": 300,
        "vocab_size": 512,
        "max_seq_len": 64,
    }


def build_model_config(mode: str) -> ModelConfig:
    """Builds a ModelConfig with exactly the flags needed to exercise
    `mode`, mirroring ats.cli.train's own --architecture preset semantics
    (ats/cli/train.py::_apply_architecture_preset) rather than guessing from
    the README: "all" turns on every one of the 6 architecture flags at
    once, "dense" turns on none, and every other mode turns on exactly its
    own flag."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    dims = _base_dims(mode)
    flags: dict = {
        "use_swa": False,
        "use_mla": False,
        "use_mamba": False,
        "use_moe": False,
        "use_mod": False,
        "use_mtp": False,
    }

    def enable(*names: str) -> None:
        for name in names:
            flags[f"use_{name}"] = True

    if mode == "dense":
        pass
    elif mode == "all":
        enable("swa", "mla", "mamba", "moe", "mod", "mtp")
    else:
        enable(mode)

    extra: dict = {}
    if flags["use_swa"]:
        extra.update(swa_window_size=8, swa_full_attention_interval=2)
    if flags["use_mla"]:
        extra.update(mla_compression_ratio=0.5)
    if flags["use_mamba"]:
        extra.update(
            mamba_every_n_layers=2,
            mamba_d_state=8,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_chunk_size=16,
        )
    if flags["use_moe"]:
        extra.update(num_experts=2, moe_top_k=1, moe_capacity_factor=1.25)
    if flags["use_mod"]:
        extra.update(mod_capacity_factor=0.5)
    if flags["use_mtp"]:
        extra.update(mtp_num_tokens=2)

    return ModelConfig(**dims, **flags, **extra, use_flash_attention=False)


def _unwrap_mod(layer: torch.nn.Module) -> torch.nn.Module:
    """MixtureOfDepths wraps another layer 1:1; unwrap it so feature checks
    below can inspect the actual attention/FFN-bearing layer underneath
    regardless of whether use_mod is also on."""
    return layer.block if isinstance(layer, MixtureOfDepths) else layer


def assert_feature_active(
    mode: str, model: ATSTransformer, config: ModelConfig
) -> None:
    """Fails loudly if the feature `mode` claims to test isn't actually
    present in the constructed module graph -- catches a config flag that
    silently didn't reach the model, which a passing config-validation test
    elsewhere would never notice."""
    transformer_blocks = [
        _unwrap_mod(layer)
        for layer in model.layers
        if isinstance(_unwrap_mod(layer), TransformerBlock)
    ]

    if mode in ("mamba", "all"):
        mamba_layers = [
            layer
            for layer in model.layers
            if isinstance(_unwrap_mod(layer), MambaLayer)
        ]
        if not mamba_layers:
            raise AssertionError(
                "expected at least one MambaLayer in model.layers, found none"
            )

    if mode in ("mla", "all"):
        mla_blocks = [
            b for b in transformer_blocks if isinstance(b.attention, MLAAttention)
        ]
        if not mla_blocks:
            raise AssertionError(
                "expected at least one TransformerBlock using MLAAttention, found none"
            )

    if mode in ("moe", "all"):
        moe_blocks = [b for b in transformer_blocks if b.ffn_is_moe]
        if not moe_blocks:
            raise AssertionError(
                "expected at least one TransformerBlock with an MoE FFN, found none"
            )

    if mode in ("mod", "all"):
        mod_layers = [
            layer for layer in model.layers if isinstance(layer, MixtureOfDepths)
        ]
        if not mod_layers:
            raise AssertionError(
                "expected at least one MixtureOfDepths-wrapped layer, found none"
            )

    if mode in ("swa", "all"):
        if not config.use_swa:
            raise AssertionError("config.use_swa is False")
        # SWA only has an effect on layers NOT forced to full attention --
        # if every block ended up forced full-attention, SWA would be
        # silently inert despite the flag being set.
        non_full = [b for b in transformer_blocks if not b.force_full_attention]
        if not non_full:
            raise AssertionError(
                "every TransformerBlock is forced to full attention; SWA "
                "would have no effect (check swa_full_attention_interval)"
            )

    if mode in ("mtp", "all") and not (model.uses_mtp and hasattr(model, "mtp_head")):
        raise AssertionError("expected model.uses_mtp=True and model.mtp_head to exist")


def _mtp_loss(
    logits_per_offset: list[torch.Tensor], labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """Same per-offset shift-and-cross-entropy math as
    ats.model.mtp.MultiTokenPredictionHead.compute_loss, operating on the
    already-computed logits list from TransformerOutput.mtp_logits (a plain
    Python list, NOT a tensor -- see the module docstring's note on the
    similarly-named code in ats.training.trainer.Trainer.train_step, which
    calls `.reshape()` directly on this list and would raise AttributeError
    if MTP loss ever actually ran through it)."""
    _batch, seq_len = labels.shape
    losses = []
    for k, logits in enumerate(logits_per_offset, start=1):
        if k >= seq_len:
            continue
        pred = logits[:, : seq_len - k, :].contiguous()
        target = labels[:, k:].contiguous()
        losses.append(
            F.cross_entropy(
                pred.reshape(-1, vocab_size),
                target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        )
    if not losses:
        raise RuntimeError(
            f"seq_len ({seq_len}) is too short for any of the "
            f"{len(logits_per_offset)} MTP prediction offsets to have a valid target"
        )
    return torch.stack(losses).mean()


def run_mode(
    mode: str,
    steps: int = DEFAULT_STEPS,
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
) -> ModeResult:
    torch.manual_seed(seed)
    device = "cpu"

    try:
        config = build_model_config(mode)
    except (ConfigError, ValueError) as exc:
        return ModeResult(
            mode=mode,
            passed=False,
            steps_completed=0,
            total_steps=steps,
            error=f"config build failed: {exc}",
        )

    try:
        with warnings.catch_warnings():
            # flash_attn isn't installed in a plain dev/CI environment;
            # ats.model.attention already falls back correctly and warns --
            # that fallback warning is expected noise for this script, not
            # a bug to report.
            warnings.filterwarnings("ignore", message=".*flash_attn.*")
            model = ATSTransformer(config)
    except Exception as exc:  # noqa: BLE001 - report any construction failure as a FAIL, not a crash
        return ModeResult(
            mode=mode,
            passed=False,
            steps_completed=0,
            total_steps=steps,
            error=f"model construction failed: {type(exc).__name__}: {exc}",
        )

    param_count = sum(p.numel() for p in model.parameters())

    try:
        assert_feature_active(mode, model, config)
    except AssertionError as exc:
        return ModeResult(
            mode=mode,
            passed=False,
            steps_completed=0,
            total_steps=steps,
            param_count=param_count,
            error=f"feature-active check failed: {exc}",
        )

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    initial_params = [p.detach().clone() for p in model.parameters()]

    last_loss: float | None = None
    for step in range(1, steps + 1):
        try:
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=device
            )
            labels = input_ids.clone()
            # Match what the real dataloader always sends (_collate in
            # ats/data/dataloader.py always produces an all-ones
            # attention_mask, even with no real padding): earlier versions
            # of this script never passed attention_mask at all, which
            # meant it couldn't have caught the real MLAAttention bug found
            # while verifying the bug-audit report (MLA didn't handle
            # attention_mask correctly in ANY case, so any real training run
            # with use_mla=True crashed on the first forward pass the
            # moment attention_mask was supplied -- see CHANGES.md and
            # tests/test_bug_audit_fixes.py). Passing it here now closes
            # that blind spot for future architecture changes.
            attention_mask = torch.ones(
                batch_size, seq_len, dtype=torch.long, device=device
            )

            output = model(input_ids, attention_mask=attention_mask)

            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            total_loss = ce_loss + output.aux_loss

            if output.mtp_logits is not None:
                mtp_loss = _mtp_loss(output.mtp_logits, labels, config.vocab_size)
                total_loss = total_loss + config.mtp_loss_weight * mtp_loss

            if not torch.isfinite(total_loss):
                # Single .item() call reused for both fields below (this branch
                # previously called .item() twice for the same tensor value).
                loss_value = float(total_loss.item())
                return ModeResult(
                    mode=mode,
                    passed=False,
                    steps_completed=step - 1,
                    total_steps=steps,
                    param_count=param_count,
                    last_loss=loss_value,
                    error=f"non-finite loss at step {step}: {loss_value}",
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return ModeResult(
                        mode=mode,
                        passed=False,
                        steps_completed=step - 1,
                        total_steps=steps,
                        param_count=param_count,
                        last_loss=float(total_loss.item()),
                        error=f"non-finite gradient in {name!r} at step {step}",
                    )

            optimizer.step()
            last_loss = float(total_loss.item())

        except Exception as exc:  # noqa: BLE001 - any exception mid-training is a FAIL, not a crash
            return ModeResult(
                mode=mode,
                passed=False,
                steps_completed=step - 1,
                total_steps=steps,
                param_count=param_count,
                last_loss=last_loss,
                error=f"{type(exc).__name__}: {exc}",
            )

    num_changed = sum(
        0 if torch.equal(p.detach(), before) else 1
        for p, before in zip(model.parameters(), initial_params)
    )
    changed_fraction = num_changed / max(1, len(initial_params))
    if changed_fraction < MIN_CHANGED_FRACTION:
        return ModeResult(
            mode=mode,
            passed=False,
            steps_completed=steps,
            total_steps=steps,
            param_count=param_count,
            last_loss=last_loss,
            error=(
                f"only {num_changed}/{len(initial_params)} parameter tensors changed "
                f"after {steps} steps ({changed_fraction:.0%} < {MIN_CHANGED_FRACTION:.0%} "
                f"threshold) -- training does not appear to be updating the model"
            ),
        )

    return ModeResult(
        mode=mode,
        passed=True,
        steps_completed=steps,
        total_steps=steps,
        param_count=param_count,
        last_loss=last_loss,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggressively smoke-test that every ATS-v2 architecture mode can "
        "actually train end-to-end (real forward/loss/backward/optimizer step, "
        "synthetic data, no GPU required).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="Run only this one mode instead of all 8 (default: run all 8).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Optimizer steps per mode (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Torch RNG seed (default: 0)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    modes_to_run = (args.mode,) if args.mode else MODES

    print("ATS-v2 Architecture Training Test")
    print("=" * 34)

    results: list[ModeResult] = []
    for mode in modes_to_run:
        start = time.monotonic()
        result = run_mode(mode, steps=args.steps, seed=args.seed)
        elapsed = time.monotonic() - start
        results.append(result)

        if result.passed:
            print(
                f"[PASS] {mode:<6} {result.steps_completed}/{result.total_steps}  ({elapsed:.1f}s)"
            )
        else:
            print(f"[FAIL] {mode}")
            print(f"  step: {result.steps_completed}/{result.total_steps}")
            print(
                f"  loss: {result.last_loss if result.last_loss is not None else 'n/a'}"
            )
            print(f"  error: {result.error}")

    num_passed = sum(1 for r in results if r.passed)
    print(f"{num_passed}/{len(results)} PASSED")
    return 0 if num_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
