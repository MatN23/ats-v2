"""adaptive-training-system-v2 (ats-v2)

A config-driven LLM training framework built on top of PyTorch and DeepSpeed.
"""

__version__ = "2.0.0"

from ats.config.schema import ATSConfig

__all__ = ["ATSConfig", "__version__"]
