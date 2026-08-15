#!/usr/bin/env python
"""Placeholder entry point: python -m ats.cli.align --config ... --base-checkpoint ...

RLHF/DPO-style alignment training is not implemented in this revision. This
file exists so the directory structure and `ats-align`-style console script
are in place for a future revision, but it does not silently no-op: running
it tells you exactly what's missing and exits non-zero.
"""

from __future__ import annotations

import argparse
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="[NOT YET IMPLEMENTED] Align an ats-v2 checkpoint (DPO/RLHF-style).",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--base-checkpoint", required=True, help="Checkpoint to align from."
    )
    parser.add_argument(
        "--method",
        choices=["dpo", "rlhf"],
        default="dpo",
        help="Alignment method (not yet implemented for either value).",
    )
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    parser.parse_args(argv)
    print(
        "ats.cli.align is not implemented in this revision. It is reserved "
        "structure for a future release covering DPO/RLHF-style alignment on "
        "top of an ats-train (optionally ats-finetune) checkpoint.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
