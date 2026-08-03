"""Quantization-aware training support.

- "none": returns a plain nn.Linear, no quantization involved.
- "int8": fake-quantizes weights and input activations during the forward
  pass using torch.ao.quantization's fake-quant primitives, so gradients
  still flow through a straight-through estimator. This actually changes
  numerics during training, it does not silently no-op.
- "fp8": requires transformer_engine or torchao. If neither is importable,
  this raises ImportError immediately rather than silently falling back to
  bf16/fp16, per the design brief.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

QuantizationMode = Literal["none", "int8", "fp8"]


class QuantizedLinear(nn.Module):
    """Wraps a linear layer with the requested quantization scheme applied
    to its weight (and, for int8, its input activations) during forward."""

    def __init__(
        self, in_features: int, out_features: int, quantization: QuantizationMode = "none",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if quantization not in ("none", "int8", "fp8"):
            raise ValueError(
                f"Unknown quantization mode '{quantization}'. "
                f"Fix: use one of 'none', 'int8', 'fp8'."
            )
        self.quantization = quantization
        self.linear = nn.Linear(in_features, out_features, bias=bias)
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
            return self.linear(x)

        if self.quantization == "int8":
            quantized_weight = self._weight_fake_quant(self.linear.weight)
            quantized_x = self._act_fake_quant(x)
            return torch.nn.functional.linear(quantized_x, quantized_weight, self.linear.bias)

        # fp8 path: dispatch to whichever backend was resolved at construction
        # time. We intentionally do not reimplement fp8 numerics ourselves —
        # that would be exactly the kind of custom low-level kernel work this
        # project avoids — and instead defer to the backend's own primitives.
        if self._fp8_backend == "transformer_engine":
            import transformer_engine.pytorch as te

            with te.fp8_autocast(enabled=True):
                return self.linear(x)

        # torchao path: convert the wrapped linear's weight to fp8 lazily on
        # first use, then reuse the converted module on subsequent calls.
        if not self._torchao_converted:
            from torchao.float8 import convert_to_float8_training

            convert_to_float8_training(self.linear)
            self._torchao_converted = True
        return self.linear(x)
