"""GPT-NeoX / DeepSeek-style weight initialization.

Embeddings and most linear layers use N(0, 0.02^2). Output projections that
sit at the residual-stream exit of a block (attention o_proj, FFN down_proj)
use a depth-scaled std to keep the residual stream's variance roughly
constant as num_layers grows, following the standard "scaled residual init"
practice (e.g. GPT-2/GPT-NeoX): std = 0.02 / sqrt(2 * num_layers).
No arbitrary per-layer crushing factors (no 0.8 / sqrt(layer_idx) terms).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

EMBEDDING_STD = 0.02
BASE_LINEAR_STD = 0.02


def init_weights(module: nn.Module, num_layers: int) -> None:
    """Recursively initialize all parameters of `module`.

    Skips any nn.Linear whose weight was already set by
    init_residual_projection (marked via the _ats_residual_init attribute
    below) -- without this check, ATSTransformer's later blanket
    `self.apply(init_weights)` pass would silently overwrite the
    depth-scaled residual-projection init (o_proj, down_proj) that
    TransformerBlock.__init__ sets during construction, replacing it with
    the generic un-scaled std. The marker makes this correct regardless of
    which init call happens to run first or last.

    Call once, after the full model has been constructed, via
    `model.apply(lambda m: init_weights(m, num_layers=cfg.num_layers))`;
    see ats.model.transformer.ATSTransformer.__init__ for the actual call
    site, which closes over num_layers.
    """
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=EMBEDDING_STD)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)
    elif isinstance(module, nn.Linear):
        if getattr(module.weight, "_ats_residual_init", False):
            return  # already correctly initialized by init_residual_projection
        nn.init.normal_(module.weight, mean=0.0, std=BASE_LINEAR_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def residual_output_std(num_layers: int) -> float:
    """Std for linear layers that write directly into the residual stream
    (attention o_proj, FFN down_proj), scaled down as depth grows."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    return BASE_LINEAR_STD / math.sqrt(2 * num_layers)


def init_residual_projection(module: nn.Linear, num_layers: int) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"init_residual_projection expects an nn.Linear, got {type(module).__name__}."
        )
    nn.init.normal_(module.weight, mean=0.0, std=residual_output_std(num_layers))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    # Mark this weight tensor so a later blanket init_weights() pass (e.g.
    # ATSTransformer's `self.apply(...)`) skips it instead of overwriting
    # this depth-scaled init with the generic one.
    module.weight._ats_residual_init = True
