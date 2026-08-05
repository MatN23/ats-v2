#!/usr/bin/env python
"""Placeholder entry point: python -m ats.cli.finetune --config ... --base-checkpoint ...

Instruction fine-tuning (SFT on a checkpoint produced by ats-train) is not
implemented in this revision. This file exists so the directory structure
and `ats-finetune`-style console script are in place for a future revision,
but it does not silently no-op: running it tells you exactly what's missing
and exits non-zero, rather than hanging or pretending to train.
"""

from __future__ import annotations

import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="[NOT YET IMPLEMENTED] Instruction fine-tune an ats-v2 checkpoint.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--base-checkpoint", required=True, help="Checkpoint to fine-tune from.")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    parser.parse_args(argv)
    print(
        "ats.cli.finetune is not implemented in this revision. "
        "It is reserved structure for a future release covering instruction "
        "fine-tuning on top of an ats-train checkpoint. "
        "Use `ats-train --resume <checkpoint>` with a fine-tuning-sized "
        "training.max_steps/learning_rate config as a manual workaround today.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
