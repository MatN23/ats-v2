"""Command-line entry points for ats-v2: train, evaluate, export, doctor.

finetune.py and align.py are reserved structure for future subcommands
(instruction fine-tuning and RLHF/DPO-style alignment) — not implemented in
this revision. They exit with a clear, informative message rather than
silently doing nothing, so running them tells you what's missing instead of
looking like a hang or a no-op.
"""

__all__ = ["train", "evaluate", "export", "doctor", "finetune", "align"]
