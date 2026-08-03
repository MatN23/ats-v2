"""Mixture-of-Depths (MoD): a learned per-token gate decides whether a token
is processed by the wrapped block or passed through unchanged. During
training this uses a straight-through estimator so gradients flow through the
hard decision; at inference the gate is thresholded directly (no STE needed
since there is no backward pass)."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn


class MixtureOfDepths(nn.Module):
    def __init__(self, hidden_size: int, block: nn.Module, capacity_factor: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < capacity_factor <= 1.0:
            raise ValueError(
                f"MixtureOfDepths capacity_factor must be in (0.0, 1.0], got {capacity_factor}."
            )
        self.hidden_size = hidden_size
        self.block = block
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor, **block_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MixtureOfDepths expected hidden_size={self.hidden_size}, got {hidden_size}."
            )

        gate_logits = self.gate(x).squeeze(-1)  # [batch, seq_len]
        gate_probs = torch.sigmoid(gate_logits)

        capacity = max(1, int(self.capacity_factor * seq_len))

        if self.training:
            # Straight-through: select top-`capacity` tokens per sequence by gate
            # probability, run the block only on those, blend using a hard 0/1
            # mask in the forward pass but the soft gate_probs gradient in the
            # backward pass.
            topk = torch.topk(gate_probs, capacity, dim=1)
            hard_mask = torch.zeros_like(gate_probs)
            hard_mask.scatter_(1, topk.indices, 1.0)
            ste_mask = hard_mask + (gate_probs - gate_probs.detach())
        else:
            threshold = torch.quantile(gate_probs, 1.0 - self.capacity_factor, dim=1, keepdim=True)
            hard_mask = (gate_probs >= threshold).float()
            ste_mask = hard_mask

        block_out = self.block(x, **block_kwargs)
        if isinstance(block_out, tuple):
            block_out, extra = block_out[0], block_out[1:]
        else:
            extra = ()

        mask = ste_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        output = mask * block_out + (1.0 - mask) * x

        # Load-balancing aux loss: encourage the mean gate probability to sit
        # near the target capacity_factor, so routing doesn't collapse to
        # always-on or always-off.
        target = torch.full_like(gate_probs.mean(dim=1), self.capacity_factor)
        aux_loss = torch.nn.functional.mse_loss(gate_probs.mean(dim=1), target)

        if extra:
            return (output, aux_loss) + extra
        return output, aux_loss
