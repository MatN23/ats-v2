"""Quantization-aware training support.

- "none": behaves as a plain nn.Linear, no quantization involved.
- "int8": fake-quantizes weights and input activations during the forward
  pass using torch.ao.quantization's fake-quant primitives, so gradients
  still flow through a straight-through estimator. This actually changes
  numerics during training, it does not silently no-op.
- "fp8": requires transformer_engine or torchao. If neither is importable,
  this raises ImportError immediately rather than silently falling back to
  bf16/fp16, per the design brief.

QuantizedLinear subclasses nn.Linear (rather than wrapping one as a
submodule) specifically so it's a drop-in replacement everywhere an
nn.Linear is currently constructed: isinstance(module, nn.Linear) checks
(e.g. ats.model.initialization's init routines) keep working, and --
critically -- the parameter is still registered as `self.weight` /
`self.bias` directly, not nested under `self.linear.weight`, so
state_dict() key paths are byte-for-byte identical to a plain nn.Linear's.
This matters because ats.export.huggingface remaps exact state_dict key
strings (e.g. "attention.q_proj.weight"); a wrapping design would have
silently changed those keys to "attention.q_proj.linear.weight" the moment
quantization was enabled, breaking export in a way that wouldn't be obvious
until someone actually tried to export a quantized checkpoint.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

QuantizationMode = Literal["none", "int8", "fp8"]


class QuantizedLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, quantization: QuantizationMode = "none",
        bias: bool = False,
    ) -> None:
        if quantization not in ("none", "int8", "fp8"):
            raise ValueError(
                f"Unknown quantization mode '{quantization}'. "
                f"Fix: use one of 'none', 'int8', 'fp8'."
            )
        super().__init__(in_features, out_features, bias=bias)
        self.quantization = quantization
        self._torchao_converted = False

        if quantization == "int8":
            try:
                from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver
            except ImportError as exc:
                raise ImportError(
                    "int8 quantization-aware training requires torch.ao.quantization "
                    "(available in torch>=1.13). Fix: upgrade torch, or set "
                    "model.quantization: none."
                ) from exc
            self._weight_fake_quant = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
            )()
            self._act_fake_quant = FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_affine,
            )()

        elif quantization == "fp8":
            self._fp8_backend = self._resolve_fp8_backend()

    @staticmethod
    def _resolve_fp8_backend() -> str:
        try:
            import transformer_engine.pytorch  # noqa: F401
            return "transformer_engine"
        except ImportError:
            pass
        try:
            import torchao  # noqa: F401
            return "torchao"
        except ImportError:
            pass
        raise ImportError(
            "FP8 training requires transformer-engine or torchao. Install with: "
            "pip install transformer-engine[pytorch]"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantization == "none":
            return F.linear(x, self.weight, self.bias)

        if self.quantization == "int8":
            quantized_weight = self._weight_fake_quant(self.weight)
            quantized_x = self._act_fake_quant(x)
            return F.linear(quantized_x, quantized_weight, self.bias)

        # fp8 path: dispatch to whichever backend was resolved at construction
        # time. We intentionally do not reimplement fp8 numerics ourselves —
        # that would be exactly the kind of custom low-level kernel work this
        # project avoids — and instead defer to the backend's own primitives.
        if self._fp8_backend == "transformer_engine":
            import transformer_engine.pytorch as te

            with te.fp8_autocast(enabled=True):
                return F.linear(x, self.weight, self.bias)

        # torchao path: convert this module's weight to fp8 lazily on first
        # use (convert_to_float8_training operates on an nn.Linear in place;
        # since QuantizedLinear IS an nn.Linear, we can pass `self` directly).
        if not self._torchao_converted:
            from torchao.float8 import convert_to_float8_training

            convert_to_float8_training(self)
            self._torchao_converted = True
        return F.linear(x, self.weight, self.bias)


def make_linear(
    in_features: int, out_features: int, quantization: QuantizationMode = "none",
    bias: bool = False,
) -> nn.Linear:
    """Factory used throughout ats.model to construct a Linear layer:
    returns a plain nn.Linear when quantization is "none" (the overwhelming
    common case, with zero overhead), or a QuantizedLinear otherwise. This
    is the single place model code should go through instead of calling
    `nn.Linear(...)` directly, so `model.quantization` in the config
    actually has an effect on the model rather than being a config field
    that's silently ignored.
    """
    if quantization == "none":
        return nn.Linear(in_features, out_features, bias=bias)
    return QuantizedLinear(in_features, out_features, quantization=quantization, bias=bias)
