#!/usr/bin/env python
"""python scripts/verify.py

Verification script for ats-v2. Does three things, in order, stopping at
the first failure:
  1. Imports every module under ats/ (catches import-time errors, circular
     imports, syntax errors that somehow slipped past review).
  2. Instantiates the major classes with small dummy inputs and runs a
     forward + backward pass, exercising the actual model code path rather
     than just checking that classes exist.
  3. Runs `pytest tests/` as a subprocess.

Prints "ALL CHECKS PASSED" and exits 0 if everything succeeds, or
"FAILURES DETECTED" with the specific failures and exits 1 otherwise. This
script does not swallow errors to make itself look green -- every step's
real exception (or pytest's real exit code) determines the result.
"""

from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Running `python scripts/verify.py` only puts scripts/ on sys.path, not
    # the repo root, so `import ats` would fail here even in a repo where
    # `pip install -e .` hasn't been run yet. Insert it explicitly so this
    # script works both before and after installation.
    sys.path.insert(0, str(REPO_ROOT))


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_imports() -> List[str]:
    """Imports every module under ats/. Returns a list of failure messages
    (empty if everything imported cleanly)."""
    _print_header("1. Importing every module under ats/")
    failures = []
    try:
        import ats
    except ImportError as exc:
        return [f"Failed to import top-level 'ats' package: {exc}"]

    package_path = Path(ats.__file__).parent
    for module_info in pkgutil.walk_packages([str(package_path)], prefix="ats."):
        module_name = module_info.name
        try:
            importlib.import_module(module_name)
            print(f"  OK   {module_name}")
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: this
            # check's entire purpose is to surface every possible import-time
            # failure across the whole package, not to handle a specific
            # expected exception type. Each failure is captured and reported
            # individually below, never silently swallowed.
            print(f"  FAIL {module_name}: {exc}")
            failures.append(f"{module_name}: {exc}")

    return failures


def check_model_instantiation() -> List[str]:
    """Instantiates ATSTransformer with a tiny config and runs a real
    forward + backward pass. Returns a list of failure messages."""
    _print_header("2. Instantiating models and running forward/backward passes")
    failures = []

    try:
        import torch

        from ats.config.schema import ModelConfig
        from ats.model.transformer import ATSTransformer
    except ImportError as exc:
        return [f"Cannot run model checks: {exc}. Run `pip install -e .` first."]

    checks: List[Tuple[str, ModelConfig]] = [
        ("dense", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
        )),
        ("swa", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            use_swa=True, swa_window_size=4,
        )),
        ("mla", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32,
            use_mla=True, mla_latent_dim=8,
        )),
        ("moe", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            use_moe=True, num_experts=4, moe_top_k=2,
        )),
        ("mod", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            use_mod=True,
        )),
        ("mamba", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            use_mamba=True, mamba_every_n_layers=1,
        )),
        ("mtp", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            use_mtp=True, mtp_num_tokens=2,
        )),
        ("int8_quantization", ModelConfig(
            hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=64, vocab_size=50, max_seq_len=32, use_flash_attention=False,
            quantization="int8",
        )),
    ]

    for name, config in checks:
        try:
            model = ATSTransformer(config)
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            output = model(input_ids)
            loss = output.logits.float().pow(2).mean() + output.aux_loss
            loss.backward()
            grad_found = any(p.grad is not None for p in model.parameters())
            if not grad_found:
                raise RuntimeError("backward() ran but no parameter received a gradient")
            print(f"  OK   {name}: forward+backward pass succeeded, logits shape {tuple(output.logits.shape)}")
        except Exception as exc:  # noqa: BLE001 -- same rationale as check_imports:
            # this check exists specifically to surface any failure across
            # every architecture combination, captured and reported below.
            print(f"  FAIL {name}: {exc}")
            failures.append(f"model instantiation ({name}): {exc}\n{traceback.format_exc()}")

    return failures


def check_pytest() -> List[str]:
    """Runs `pytest tests/` as a subprocess. Returns a list with one entry
    describing the failure if pytest's exit code is non-zero."""
    _print_header("3. Running pytest tests/")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            cwd=str(REPO_ROOT), check=False,
        )
    except FileNotFoundError as exc:
        return [f"Could not run pytest: {exc}. Is pytest installed?"]

    if result.returncode != 0:
        return [f"pytest exited with code {result.returncode}"]
    return []


def main() -> int:
    all_failures: List[str] = []
    all_failures.extend(check_imports())
    all_failures.extend(check_model_instantiation())
    all_failures.extend(check_pytest())

    _print_header("Summary")
    if all_failures:
        print(f"FAILURES DETECTED ({len(all_failures)}):\n")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
