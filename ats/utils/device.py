"""Device resolution for the training/eval paths.

BUG-105: every device decision in ats/training used to be written as

    device = (
        engine.local_rank
        if isinstance(engine.local_rank, torch.device)
        else torch.device(f"cuda:{engine.local_rank}")
    )

duplicated verbatim in five places. That hardcodes CUDA. On a CPU-only box
`torch.device("cuda:0")` constructs fine and only blows up later, at the
first `.to(device)`, with a message about no CUDA driver rather than
anything pointing at the configuration. On Apple Silicon it fails the same
way even though MPS is present and usable. So CPU and MPS training were
both impossible, and the README's own "trains a tiny debug model on CPU"
quickstart could not have worked through Trainer.

The fix is to ask the engine what device it is on (DeepSpeedEngine exposes
`.device`, which is authoritative and already accounts for its own
accelerator abstraction) and only fall back to building a device from
local_rank when the engine does not expose one -- and in that fallback,
pick the accelerator that actually exists rather than assuming CUDA.

This is deliberately not a try/except around the old expression: the point
is to select the intended device, not to swallow the failure.
"""

from __future__ import annotations

from typing import Any

import torch


def preferred_accelerator() -> str:
    """The accelerator backend this process should use: "cuda", "mps" or
    "cpu". Checked in capability order; MPS is only reported when it is
    both built and available, since a torch built without MPS still
    exposes torch.backends.mps.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and mps.is_built():
        return "mps"
    return "cpu"


def resolve_device(engine: Any) -> torch.device:
    """The device an engine's parameters live on.

    Order of preference:
      1. `engine.device` -- what DeepSpeedEngine actually reports, and what
         any test double should report too.
      2. `engine.local_rank`, when it is already a torch.device.
      3. A device built from `local_rank` on the preferred accelerator.
         Only indexed for CUDA: "mps:0" and "cpu:0" are not meaningful
         placements, and a multi-process CPU run puts every rank on "cpu".
      4. The preferred accelerator with no index.
    """
    device = getattr(engine, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device:
        return torch.device(device)

    local_rank = getattr(engine, "local_rank", None)
    if isinstance(local_rank, torch.device):
        return local_rank

    backend = preferred_accelerator()
    if backend == "cuda" and isinstance(local_rank, int) and local_rank >= 0:
        return torch.device(f"cuda:{local_rank}")
    return torch.device(backend)


def module_device(module: torch.nn.Module) -> torch.device:
    """The device of a module's first parameter (or buffer), falling back to
    CPU for a parameterless module. Used where there is no engine to ask.
    """
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")
