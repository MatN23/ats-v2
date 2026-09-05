"""Loads a JSONL preference-pairs file for DPO training. Each line is:

    {"prompt": "...", "chosen": "...", "rejected": "..."}

Produces, for both "chosen" and "rejected", a fixed-length
[prompt_tokens + response_tokens + EOS] sequence padded/truncated to
seq_length, with `labels` using IGNORE_INDEX (matching ats.data.dataset's
convention exactly) over the prompt portion and any padding -- so
compute_sequence_logprobs only ever sums log-probs over the response
tokens actually being judged, never the (identical, for a given prompt)
prompt tokens or padding.

Loads the whole file into memory rather than streaming (unlike
ats.data.dataset.MixedDataset): preference-pair datasets for DPO are
overwhelmingly smaller than pretraining corpora (typically thousands to
low tens of thousands of examples, not the many-GB streaming case
MixedDataset is built for), and an in-memory torch.utils.data.Dataset is a
much simpler, more directly correct implementation for that scale rather
than reusing MixedDataset's shard-streaming machinery for a case it wasn't
designed for.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Dataset

from ats.config.schema import ConfigError
from ats.data.dataset import IGNORE_INDEX
from ats.data.tokenizer import Tokenizer
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.data.preference_dataset")


def _encode_example(
    tokenizer: Tokenizer, prompt: str, response: str, seq_length: int
) -> tuple[list[int], list[int]]:
    """Returns (input_ids, labels), both length seq_length. Prompt tokens
    and padding are IGNORE_INDEX in labels; response tokens (plus a
    trailing EOS, if it fits) are their real token ids in labels, exactly
    matching MixedDataset's masking convention so compute_sequence_logprobs
    behaves identically to how the rest of this codebase computes
    per-token losses."""
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]

    if len(prompt_ids) >= seq_length:
        raise ConfigError(
            f"A preference example's prompt alone ({len(prompt_ids)} tokens) is >= "
            f"seq_length ({seq_length}), leaving no room for any response tokens. "
            f"Fix: shorten the prompt, or increase data.seq_length."
        )

    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)

    if len(input_ids) > seq_length:
        input_ids = input_ids[:seq_length]
        labels = labels[:seq_length]
    else:
        pad_len = seq_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

    return input_ids, labels


class PreferenceDataset(Dataset):
    def __init__(self, path: str, tokenizer_name: str, seq_length: int) -> None:
        self.tokenizer = Tokenizer(tokenizer_name)
        self.seq_length = seq_length
        self.examples: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(
                        f"{path}:{line_num}: invalid JSON ({exc}). Fix: each line must "
                        f'be a JSON object like {{"prompt": ..., "chosen": ..., '
                        f'"rejected": ...}}.'
                    ) from exc
                for key in ("prompt", "chosen", "rejected"):
                    if key not in row or not isinstance(row[key], str):
                        raise ConfigError(
                            f"{path}:{line_num}: missing or non-string '{key}' field. "
                            f'Fix: each line must be {{"prompt": ..., "chosen": ..., '
                            f'"rejected": ...}}, all string values.'
                        )
                self.examples.append(row)

        if not self.examples:
            raise ConfigError(
                f"{path} contains no valid preference examples. "
                f"Fix: add at least one line of "
                f'{{"prompt": ..., "chosen": ..., "rejected": ...}}.'
            )
        logger.info("Loaded %d preference pairs from %s", len(self.examples), path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        row = self.examples[idx]
        chosen_ids, chosen_labels = _encode_example(
            self.tokenizer, row["prompt"], row["chosen"], self.seq_length
        )
        rejected_ids, rejected_labels = _encode_example(
            self.tokenizer, row["prompt"], row["rejected"], self.seq_length
        )
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        key: torch.tensor([ex[key] for ex in batch], dtype=torch.long)
        for key in (
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        )
    }


def build_preference_dataloader(
    path: str,
    tokenizer_name: str,
    seq_length: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    num_workers: int = 4,
) -> DataLoader:
    """rank/world_size shard the dataset by simple index slicing
    (dataset[rank::world_size]) -- adequate for preference-pair datasets'
    scale (see module docstring); MixedDataset's more elaborate
    shard-file-based distribution isn't needed at this scale, and this
    dataset doesn't stream from sharded files the way MixedDataset does."""
    full_dataset: Dataset = PreferenceDataset(path, tokenizer_name, seq_length)
    if world_size > 1:
        # Dataset's type stub doesn't guarantee __len__ (a Dataset only has
        # to implement __getitem__), even though every concrete Dataset used
        # here (PreferenceDataset, and Subset below) does define it.
        indices = list(range(rank, len(full_dataset), world_size))  # type: ignore[arg-type]
        if not indices:
            raise ConfigError(
                f"Preference dataset at {path} has {len(full_dataset)} examples, too "  # type: ignore[arg-type]
                f"few to shard across world_size={world_size} at rank={rank} (this "
                f"rank would get zero examples). Fix: use fewer ranks or a larger "
                f"preference dataset."
            )
        full_dataset = torch.utils.data.Subset(full_dataset, indices)

    generator = torch.Generator().manual_seed(seed + rank)
    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
