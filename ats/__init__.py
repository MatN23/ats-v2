"""adaptive-training-system-v2 (ats-v2)

A config-driven LLM training framework built on top of PyTorch and DeepSpeed.
"""

__version__ = "2.0.0"

__all__ = ["ATSConfig", "__version__"]


def __getattr__(name: str):
    # Lazy import (PEP 562): `import ats` alone must not require pydantic to
    # be installed. This matters specifically for `ats.cli.doctor`, whose
    # entire purpose is to diagnose a broken/incomplete environment -- it
    # would be self-defeating if diagnosing a missing pydantic install
    # required pydantic to already be installed. `from ats import ATSConfig`
    # still works normally for everyone who does have pydantic.
    if name == "ATSConfig":
        from ats.config.schema import ATSConfig

        return ATSConfig
    raise AttributeError(f"module 'ats' has no attribute {name!r}")
