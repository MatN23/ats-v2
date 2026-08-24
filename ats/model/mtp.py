"""Multi-Token Prediction (MTP): predicts `num_future_tokens` future tokens
in parallel from the final hidden state, each via its own linear head.
Training loss is the mean cross-entropy across all predicted offsets.
Inference uses only the first (t+1) head unless the caller explicitly
requests the others (e.g. for speculative decoding)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100


def compute_mtp_loss_from_logits(
    logits_per_offset: list[torch.Tensor], labels: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """The per-offset shift-and-cross-entropy math shared by
    MultiTokenPredictionHead.compute_loss (which computes logits_per_offset
    itself via forward()) and any caller that already has logits_per_offset
    from a prior forward pass on the same input -- e.g. Trainer.train_step,
    which gets this list directly from TransformerOutput.mtp_logits and
    must NOT call forward() a second time to get it again.

    labels: [batch, seq_len] token ids, unshifted (the same shape as the
    hidden_states/input_ids that produced logits_per_offset -- NOT
    pre-shifted by 1 the way the main next-token CE loss's labels are).
    For each offset k in [1..len(logits_per_offset)], offset k's logits
    predict labels shifted by k, with positions that run past the end of
    the sequence excluded rather than padded, since there is no valid
    target for them.
    """
    if labels.dim() != 2:
        raise ValueError(
            f"compute_mtp_loss_from_logits expected labels of shape "
            f"[batch, seq_len], got shape {tuple(labels.shape)}."
        )
    _batch, seq_len = labels.shape
    losses = []
    for k, logits in enumerate(logits_per_offset, start=1):
        if k >= seq_len:
            continue
        pred = logits[:, : seq_len - k, :].contiguous()
        target = labels[:, k:].contiguous()
        loss_k = F.cross_entropy(
            pred.reshape(-1, vocab_size),
            target.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        losses.append(loss_k)

    if not losses:
        raise ValueError(
            f"seq_len ({seq_len}) is too short for any of the "
            f"{len(logits_per_offset)} MTP prediction offset(s) to have a valid "
            f"target (offset k needs seq_len > k). Fix: use a longer sequence, "
            f"or reduce model.mtp_num_tokens."
        )
    return torch.stack(losses).mean()


class MultiTokenPredictionHead(nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, num_future_tokens: int = 2
    ) -> None:
        super().__init__()
        if num_future_tokens < 1:
            raise ValueError(
                f"MultiTokenPredictionHead requires num_future_tokens >= 1, got {num_future_tokens}."
            )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_future_tokens = num_future_tokens
        # One projection per future offset (t+1, t+2, ..., t+N). Offsets
        # beyond the first share a lightweight per-offset transform of the
        # base hidden state before their own unembedding, so each head can
        # specialize slightly rather than all reusing identical logits.
        self.offset_transforms = nn.ModuleList(
            [nn.Identity()]
            + [
                nn.Linear(hidden_size, hidden_size, bias=False)
                for _ in range(num_future_tokens - 1)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_future_tokens)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"MultiTokenPredictionHead expected last dim {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}."
            )
        logits_per_offset = []
        for transform, head in zip(self.offset_transforms, self.heads):
            logits_per_offset.append(head(transform(hidden_states)))
        return logits_per_offset

    def compute_loss(
        self, hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """labels: [batch, seq_len] token ids. For each offset k in
        [1..num_future_tokens], head k predicts labels shifted by k, with
        positions that run past the end of the sequence masked out."""
        if labels.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"MultiTokenPredictionHead.compute_loss: labels shape {tuple(labels.shape)} "
                f"does not match hidden_states shape {tuple(hidden_states.shape)} on the "
                f"batch/seq dimensions."
            )
        logits_per_offset = self.forward(hidden_states)
        return compute_mtp_loss_from_logits(logits_per_offset, labels, self.vocab_size)
