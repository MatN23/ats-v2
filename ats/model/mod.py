"""Mixture-of-Depths (MoD): a learned per-token gate decides whether a token
is processed by the wrapped block or passed through unchanged. During
training this uses a straight-through estimator so gradients flow through the
hard decision; at inference the gate is thresholded directly (no STE needed
since there is no backward pass).

The top-`capacity` tokens per sequence are GATHERED into a shorter
[batch, capacity, hidden] tensor and the wrapped block runs only on that,
with the results scattered back into the residual stream. This is where the
compute saving comes from, and it is real: the block sees
`capacity_factor * seq_len` positions, not `seq_len`. (An earlier version
ran the block on the full sequence and merely masked the output, which cost
exactly as much as not using MoD at all -- see the BUG-103 comment in
forward().)

One stated exception: when `use_cache=True` (or a `past_key_value` is
supplied), the block runs densely on the full sequence. A block invoked on
a compressed subsequence returns a KV cache of length `capacity` with no
record of which absolute positions those entries belong to, so a gathered
cache cannot be continued correctly on the next decode step. Cached
autoregressive generation therefore gets no MoD compute saving; training
and non-cached evaluation do.

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

import torch
from torch import nn


class MixtureOfDepths(nn.Module):
    def __init__(
        self, hidden_size: int, block: nn.Module, capacity_factor: float = 0.5
    ) -> None:
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
                self.gate.bias,
                math.log(self.capacity_factor / (1 - self.capacity_factor)),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: object | None = None,
        use_cache: bool = False,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, object | None]:
        # Bug 1 fix: torch.utils.checkpoint.checkpoint in transformer.py calls
        # layer(x, attention_mask, past_kv, use_cache) positionally, so this
        # signature must accept those as named positional args, not just
        # **block_kwargs (which only captures keyword arguments).
        block_kwargs = {
            "attention_mask": attention_mask,
            "past_key_value": past_key_value,
            "use_cache": use_cache,
            # BUG FIX (found alongside the diffusion causal-masking bug --
            # see ats.model.attention.GroupedQueryAttention.forward and
            # CHANGES.md): without this, a MoD-wrapped layer silently never
            # received causal=False at all, since it wasn't in block_kwargs
            # -- the wrapped block would fall back to its own default
            # (causal=True), quietly reintroducing causal masking for any
            # MoD-wrapped layer even after the rest of the diffusion fix.
            "causal": causal,
        }
        _batch, seq_len, hidden_size = x.shape
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

        # BUG FIX (BUG-103): this used to call the wrapped block on the FULL
        # [batch, seq_len, hidden] tensor and then blend the result with a
        # 0/1 mask. That computes every token through the block and throws
        # away the unselected results -- so Mixture-of-Depths cost exactly
        # as much FLOPs and activation memory as not using it at all, plus
        # the gate. The entire premise of MoD (skip the block for
        # (1 - capacity_factor) of tokens) was not implemented; only its
        # output was simulated. Below, the selected tokens are gathered into
        # a [batch, capacity, hidden] tensor, the block runs on that, and
        # the results are scattered back.
        #
        # `selected_idx` is sorted ascending so the gathered subsequence
        # keeps its original relative order: the wrapped block's attention
        # (and MambaLayer's scan) is causal over the positions it is handed,
        # so an unsorted gather would let a token attend to one that came
        # after it in the original sequence.
        if use_cache or past_key_value is not None:
            # KV caching is not compatible with a compressed subsequence:
            # the cache the block returns would be `capacity` long rather
            # than seq_len, and the next decode step has no way to know
            # which absolute positions those entries correspond to. Rather
            # than return a silently mis-indexed cache, run the block
            # densely for cached generation. This is stated, not silent --
            # see the class docstring. Training and non-cached evaluation
            # (where the compute actually matters) take the gather path.
            block_hidden, block_aux_loss, new_past_key_value = self.block(
                x, **block_kwargs
            )
            mask = ste_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            output = mask * block_hidden + (1.0 - mask) * x
        else:
            selected_idx, _ = torch.sort(topk.indices, dim=1)  # [batch, capacity]
            gather_idx = selected_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
            x_selected = torch.gather(x, 1, gather_idx)  # [batch, capacity, hidden]

            if attention_mask is not None:
                # The padding mask has to be gathered the same way, or the
                # block would apply position i's pad flag to whatever token
                # happens to land in slot i of the compressed sequence.
                block_kwargs["attention_mask"] = torch.gather(
                    attention_mask, 1, selected_idx
                )

            block_out, block_aux_loss, new_past_key_value = self.block(
                x_selected, **block_kwargs
            )

            # Straight-through gate applied only to the tokens that were
            # actually processed: forward value 1, backward gradient
            # d/d gate_prob. Unselected tokens pass through untouched (they
            # were never computed), and their gate logits are trained by
            # the load-balancing aux loss below rather than by a gradient
            # through a block output that does not exist for them.
            gate_selected = torch.gather(ste_mask, 1, selected_idx).unsqueeze(-1)
            update = gate_selected * (block_out - x_selected)
            output = x.scatter_add(1, gather_idx, update)

        # Load-balancing aux loss: encourage the mean gate probability to sit
        # near the target capacity_factor, so routing doesn't collapse to
        # always-on or always-off. Added to (not replacing) the wrapped
        # block's own aux_loss, e.g. from an inner MoE FFN's routing loss.
        target = torch.full_like(gate_probs.mean(dim=1), self.capacity_factor)
        mod_aux_loss = torch.nn.functional.mse_loss(gate_probs.mean(dim=1), target)
        total_aux_loss = mod_aux_loss + block_aux_loss

        return output, total_aux_loss, new_past_key_value
