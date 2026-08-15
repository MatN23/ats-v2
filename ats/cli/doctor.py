#!/usr/bin/env python
"""Entry point: python -m ats.cli.doctor [--config configs/7b.yaml]

Prints a real environment diagnostic: Python version, PyTorch/CUDA,
DeepSpeed, Flash Attention, Triton, GPU count/memory, and (if --config is
given) an estimated memory report for that config. Every line is produced
by actually importing/inspecting the relevant package or device — nothing
here is a hardcoded string.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys

CHECK = "\u2713"  # ✓
CROSS = "\u2717"  # ✗
WARN = "!"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_torch() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"
    cuda_str = (
        f"+ CUDA {torch.version.cuda}"
        if torch.cuda.is_available()
        else "(CPU only, no CUDA detected)"
    )
    return True, f"PyTorch {torch.__version__} {cuda_str}"


def check_deepspeed() -> tuple[bool, str]:
    version = _package_version("deepspeed")
    if version is None:
        return False, "DeepSpeed not installed"
    return True, f"DeepSpeed {version}"


def check_flash_attention() -> tuple[bool, str]:
    version = _package_version("flash-attn") or _package_version("flash_attn")
    if version is None:
        return False, "Flash Attention not installed (SDPA fallback will be used)"
    return True, f"Flash Attention {version}"


def check_triton() -> tuple[bool, str]:
    version = _package_version("triton")
    if version is None:
        return False, "Triton not installed (Triton kernels will use PyTorch fallback)"
    return True, f"Triton {version}"


def check_gpus() -> list[tuple[bool, str]]:
    try:
        import torch
    except ImportError:
        return [(False, "Cannot check GPUs: PyTorch not installed")]
    if not torch.cuda.is_available():
        return [(False, "No CUDA GPUs detected")]

    results = []
    count = torch.cuda.device_count()
    total_gb = 0.0
    names = set()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb += props.total_memory / (1024**3)
        names.add(props.name)
    name_str = names.pop() if len(names) == 1 else "/".join(sorted(names))
    results.append((True, f"{count}\u00d7 {name_str} detected"))
    results.append((True, f"{total_gb:.0f}GB total GPU memory"))
    return results


def check_config_memory(config_path: str) -> tuple[bool, str]:
    try:
        from ats.config.loader import load_config
        from ats.config.schema import ConfigError
        from ats.utils.memory import estimate_memory
    except ImportError as exc:
        return False, (
            f"Cannot estimate memory for {config_path}: a required package is not "
            f"installed ({exc}). Fix: pip install -e . to install pydantic/torch/etc, "
            f"then re-run ats-doctor --config."
        )

    try:
        config = load_config(config_path)
        report = estimate_memory(config)
    except ConfigError as exc:
        return False, f"Could not load {config_path}: {exc}"

    if report.available_gb <= 0:
        return True, (
            f"{config_path} estimated at {report.total_gb:.0f}GB "
            f"(no GPU detected to compare against)"
        )
    if report.fits_on_single_gpu:
        return (
            True,
            f"{config_path} estimated at {report.total_gb:.0f}GB, fits within {report.available_gb:.0f}GB budget",
        )
    return False, (
        f"{config_path} estimated at {report.total_gb:.0f}GB, exceeds 80% of "
        f"{report.available_gb:.0f}GB available. Consider ZeRO-{report.suggested_zero_stage} "
        f"or --micro-batch-size {report.suggested_batch_size} --grad-accum-steps {report.suggested_grad_accum}."
    )


def _print_line(ok: bool, message: str, is_warning: bool = False) -> None:
    symbol = WARN if is_warning else (CHECK if ok else CROSS)
    print(f"{symbol} {message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the ats-v2 environment.")
    parser.add_argument(
        "--config", default=None, help="Optional config to estimate memory for."
    )
    args = parser.parse_args(argv)

    any_hard_failure = False

    ok, msg = check_python()
    _print_line(ok, msg)
    any_hard_failure = any_hard_failure or not ok

    ok, msg = check_torch()
    _print_line(ok, msg)
    any_hard_failure = any_hard_failure or not ok

    ok, msg = check_deepspeed()
    _print_line(ok, msg)
    any_hard_failure = any_hard_failure or not ok

    ok, msg = check_flash_attention()
    _print_line(
        ok, msg, is_warning=not ok
    )  # missing flash-attn is a soft warning, not fatal

    ok, msg = check_triton()
    _print_line(
        ok, msg, is_warning=not ok
    )  # missing triton is a soft warning, not fatal

    for ok, msg in check_gpus():
        _print_line(ok, msg, is_warning=not ok)

    if args.config is not None:
        ok, msg = check_config_memory(args.config)
        _print_line(ok, msg, is_warning=not ok)

    if any_hard_failure:
        print(
            "\nSome required components are missing. See the ✗ lines above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
