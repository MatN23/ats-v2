"""Builds a torch DataLoader over MixedDataset with a distributed-aware
collator. Padding/attention-mask/labels handling lives entirely here."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ats.config.schema import ConfigError, DataConfig
from ats.data.dataset import MixedDataset
from ats.data.tokenizer import Tokenizer


class _TorchMixedDataset(IterableDataset):
    """Thin torch.utils.data.IterableDataset adapter around MixedDataset,
    sharding by (rank, world_size) so each distributed process sees a
    disjoint stream.

    When used with DataLoader(num_workers > 0), PyTorch forks/copies this
    dataset object into each worker process, so every worker would otherwise
    iterate the exact same underlying MixedDataset stream (same seed, same
    file read order) and yield duplicate data. __iter__ additionally
    sub-shards by the DataLoader worker's (worker_id, num_workers) via
    torch.utils.data.get_worker_info(), combined with the outer (rank,
    world_size) sharding, so each (process, worker) pair gets a disjoint
    slice of the same deterministic stream."""

    def __init__(
        self, mixed_dataset: MixedDataset, rank: int = 0, world_size: int = 1,
    ) -> None:
        if world_size < 1:
            raise ConfigError(f"world_size must be >= 1, got {world_size}.")
        if not 0 <= rank < world_size:
            raise ConfigError(f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}.")
        self.mixed_dataset = mixed_dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # Combine outer (rank, world_size) sharding with inner (worker_id,
        # num_workers) sharding into one effective shard index/count.
        effective_id = self.rank * num_workers + worker_id
        effective_total = self.world_size * num_workers

        for i, example in enumerate(self.mixed_dataset):
            if i % effective_total == effective_id:
                yield example


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ConfigError("collate_fn received an empty batch.")
    seq_len = len(batch[0]["input_ids"])
    for example in batch:
        if len(example["input_ids"]) != seq_len:
            raise ConfigError(
                f"Batch has inconsistent sequence lengths: expected {seq_len}, "
                f"got {len(example['input_ids'])}. All examples from MixedDataset "
                f"should already be fixed-length; this indicates a bug upstream."
            )
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_dataloader(
    data_config: DataConfig,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ConfigError(f"batch_size must be positive, got {batch_size}.")

    tokenizer = Tokenizer(data_config.tokenizer_name)
    # IMPORTANT: seed is NOT varied by rank here. _TorchMixedDataset.__iter__
    # shards by filtering a deterministic stream via `i % effective_total ==
    # effective_id`, which requires every rank's MixedDataset to produce the
    # SAME underlying stream (same seed) so the modulo filter partitions it
    # correctly. Varying the seed by rank here used to compound with that
    # filtering: each rank would get its own already-different stream, THEN
    # modulo-filter that down further, discarding most of it unnecessarily
    # (confirmed: at world_size=8, only 12.5% of each rank's already-unique
    # stream survived, for an effective 87.5% data loss). Use the same seed
    # for every rank and let the modulo-based sharding do all the work.
    mixed_dataset = MixedDataset(
        sources=data_config.sources, tokenizer=tokenizer,
        seq_length=data_config.seq_length, seed=seed,
    )
    torch_dataset = _TorchMixedDataset(mixed_dataset, rank=rank, world_size=world_size)

    return DataLoader(
        torch_dataset, batch_size=batch_size, collate_fn=_collate, num_workers=num_workers,
    )
