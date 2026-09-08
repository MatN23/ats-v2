"""Held-out perplexity computation, shared by `ats.cli.evaluate`'s
perplexity mode and the PBT breeding orchestrator (`ats.pbt`), which uses it
as the fitness function for ranking population members. Kept as one
function so both call sites can't silently drift into computing loss
differently.
"""

from __future__ import annotations

from typing import Any

import torch

from ats.config.schema import ATSConfig
from ats.data.dataloader import build_dataloader
from ats.model.transformer import ATSTransformer
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.checkpoint import CheckpointManager
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.training.perplexity")


def compute_perplexity(
    config: ATSConfig,
    checkpoint_dir: str,
    micro_batch_size: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Loads `checkpoint_dir` under `config` and computes perplexity over
    `config.data.sources` (rank 0, world_size 1, no distributed setup).
    Returns (perplexity, client_state) where client_state is whatever
    CheckpointManager.load() returned (global_step, epoch, ...).

    Raises ConfigError if the checkpoint's config_hash doesn't match `config`
    (via CheckpointManager.load(); see its docstring) or if the eval
    dataloader produces zero valid tokens.
    """
    resolved_micro_batch_size = (
        micro_batch_size
        if micro_batch_size is not None
        else config.training.micro_batch_size
    )

    model = ATSTransformer(
        config.model, ep_size=max(1, config.parallelism.gpus * config.parallelism.nodes)
    )
    model_engine, _optimizer, _, _ = initialize_engine(
        model, config, resolved_micro_batch_size
    )

    checkpoint_manager = CheckpointManager(config)
    client_state = checkpoint_manager.load(model_engine, checkpoint_dir)
    logger.info(
        "Loaded checkpoint from step %d (epoch %d) for perplexity evaluation.",
        client_state["global_step"],
        client_state["epoch"],
    )
    model_engine.eval()

    eval_dataloader = build_dataloader(
        config.data,
        batch_size=resolved_micro_batch_size,
        rank=0,
        world_size=1,
        seed=config.training.seed,
        num_workers=config.data.num_workers,
    )
    device = (
        model_engine.local_rank
        if isinstance(model_engine.local_rank, torch.device)
        else torch.device(f"cuda:{model_engine.local_rank}")
    )
    # Accumulate on-device across the whole eval set and only call .item()
    # once, after the loop -- calling .item() per batch (as this used to)
    # forces a host-device sync every iteration, serializing what should be
    # an async eval pass. Perplexity eval commonly runs over a large held-out
    # set, so this isn't a one-off cost.
    total_loss_t = torch.zeros((), dtype=torch.float64, device=device)
    total_tokens_t = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in eval_dataloader:
            output = model_engine(
                batch["input_ids"], attention_mask=batch.get("attention_mask")
            )
            # Transpose (metadata-only) instead of an explicit .contiguous()
            # + .view() flatten, which forces a full copy of the whole
            # logits tensor -- see the matching fix in trainer.py. .float():
            # see trainer.py's identical fix -- the per-token logsumexp
            # reduction over vocab_size classes overflows fp16 (sums to
            # >65504 just from vocab_size alone, independent of prediction
            # quality) unless logits are upcast first.
            shift_logits = output.logits[..., :-1, :]
            shift_labels = batch["labels"][..., 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.float().transpose(1, 2),
                shift_labels,
                ignore_index=-100,
                reduction="sum",
            )
            num_valid = (shift_labels != -100).sum()
            total_loss_t += loss.detach().double()
            total_tokens_t += num_valid

    total_loss = float(total_loss_t.item())
    total_tokens = int(total_tokens_t.item())

    from ats.config.schema import ConfigError

    if total_tokens == 0:
        raise ConfigError(
            "Eval dataloader produced zero valid tokens; perplexity is undefined. "
            "Fix: check config.data.sources points at real, non-empty held-out data."
        )
    perplexity = float(torch.exp(torch.tensor(total_loss / total_tokens)))
    return perplexity, client_state