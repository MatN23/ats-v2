"""Mixture-of-Depths (MoD): a learned per-token gate decides whether a token
is processed by the wrapped block or passed through unchanged. During
training this uses a straight-through estimator so gradients flow through the
hard decision; at inference the gate is thresholded directly (no STE needed
since there is no backward pass).

The wrapped block (a TransformerBlock or MambaLayer, per
ats.model.transformer) always returns a 3-tuple
(hidden_states, aux_loss, past_key_value), matching the calling convention
ATSTransformer._run_layers uses uniformly for every layer, MoD-wrapped or
not. MixtureOfDepths.forward must therefore also always return exactly that
3-tuple shape -- summing the wrapped block's own aux_loss (e.g. from an
inner MoE FFN) into MoD's load-balancing aux_loss rather than dropping it,
and passing the wrapped block's past_key_value through unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

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
        # Bug 10 fix: default Linear init gives ~50% selection regardless of
        # capacity_factor. For capacity_factor < 0.5, bias the gate at init
        # so the initial selection rate roughly matches the target capacity.
        if self.capacity_factor < 0.5:
            nn.init.constant_(
                self.gate.bias, math.log(self.capacity_factor / (1 - self.capacity_factor))
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[object]]:
        # Bug 1 fix: torch.utils.checkpoint.checkpoint in transformer.py calls
        # layer(x, attention_mask, past_kv, use_cache) positionally, so this
        # signature must accept those as named positional args, not just
        # **block_kwargs (which only captures keyword arguments).
        block_kwargs = {
            "attention_mask": attention_mask,
            "past_key_value": past_key_value,
            "use_cache": use_cache,
        }
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
            # Bug 6 fix: use topk (same as training) instead of quantile, so
            # inference selects the identical token set training would have
            # selected on ties, instead of potentially diverging.
            topk = torch.topk(gate_probs, capacity, dim=1)
            hard_mask = torch.zeros_like(gate_probs).scatter_(1, topk.indices, 1.0)
            ste_mask = hard_mask

        # The wrapped block is called exactly once. It always returns
        # (hidden_states, aux_loss, past_key_value) -- the same 3-tuple
        # convention every layer in ATSTransformer._run_layers uses.
        block_hidden, block_aux_loss, new_past_key_value = self.block(x, **block_kwargs)

        mask = ste_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        output = mask * block_hidden + (1.0 - mask) * x

        # Load-balancing aux loss: encourage the mean gate probability to sit
        # near the target capacity_factor, so routing doesn't collapse to
        # always-on or always-off. Added to (not replacing) the wrapped
        # block's own aux_loss, e.g. from an inner MoE FFN's routing loss.
        target = torch.full_like(gate_probs.mean(dim=1), self.capacity_factor)
        mod_aux_loss = torch.nn.functional.mse_loss(gate_probs.mean(dim=1), target)
        total_aux_loss = mod_aux_loss + block_aux_loss

        return output, total_aux_loss, new_past_key_value
