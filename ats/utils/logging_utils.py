"""Structured logging for ats. Every module gets a logger named ats.<module>,
so log verbosity can be controlled per-component with the standard logging config."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call multiple times (e.g. once in train.py, once in
    a test's conftest.py) without duplicating handlers."""
    global _CONFIGURED
    root = logging.getLogger("ats")
    root.setLevel(level)
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
