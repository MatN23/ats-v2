"""Loads a JSONL supervised fine-tuning file for chat-style instruction
tuning. Each line is:

    {"prompt": "...", "response": "..."}

Produces a fixed-length [prompt_tokens + response_tokens + EOS] sequence
padded/truncated to seq_length, with `labels` using IGNORE_INDEX (matching
ats.data.dataset's convention exactly, and mirroring
ats.data.preference_dataset's masking pattern) over the prompt portion and
any padding -- so loss is only ever computed on the assistant's response
tokens, never the prompt or padding.

This is what ats.cli.finetune should use for real instruction-tuning
instead of the generic MixedDataset/build_dataloader pretraining path: that
path applies plain next-token loss over every token, with no distinction
between prompt and response, so it never teaches turn structure no matter
how it's formatted. This module is the piece that was missing.

Deliberately mirrors ats/data/preference_dataset.py's implementation almost
line for line (in-memory list, simple rank::world_size sharding, no
streaming) for the same reason: SFT datasets are typically thousands to
low tens of thousands of examples, far below the scale MixedDataset's
shard-streaming machinery is built for.
"""

from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader, Dataset

from ats.config.schema import ConfigError
from ats.data.dataset import IGNORE_INDEX
from ats.data.tokenizer import Tokenizer
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.data.sft_dataset")


def _encode_example(
    tokenizer: Tokenizer, prompt: str, response: str, seq_length: int
) -> tuple[list[int], list[int]]:
    """Returns (input_ids, labels), both length seq_length. Prompt tokens
    and padding are IGNORE_INDEX in labels; response tokens (plus a
    trailing EOS, if it fits) are their real token ids -- identical
    masking convention to ats.data.preference_dataset._encode_example."""
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]

    if len(prompt_ids) >= seq_length:
        raise ConfigError(
            f"An SFT example's prompt alone ({len(prompt_ids)} tokens) is >= "
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


class SFTDataset(Dataset):
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
                        f'be a JSON object like {{"prompt": ..., "response": ...}}.'
                    ) from exc
                for key in ("prompt", "response"):
                    if key not in row or not isinstance(row[key], str):
                        raise ConfigError(
                            f"{path}:{line_num}: missing or non-string '{key}' field. "
                            f'Fix: each line must be {{"prompt": ..., "response": ...}}, '
                            f"both string values."
                        )
                self.examples.append(row)

        if not self.examples:
            raise ConfigError(
                f"{path} contains no valid SFT examples. "
                f'Fix: add at least one line of {{"prompt": ..., "response": ...}}.'
            )
        logger.info("Loaded %d SFT examples from %s", len(self.examples), path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        row = self.examples[idx]
        input_ids, labels = _encode_example(
            self.tokenizer, row["prompt"], row["response"], self.seq_length
        )
        return {"input_ids": input_ids, "labels": labels}


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    # Keys deliberately match what ats.training.trainer.Trainer.train_step
    # reads from a batch (batch["input_ids"], batch["labels"]) -- this
    # dataloader is a drop-in replacement for the pretraining one from
    # Trainer's point of view. No attention_mask: padding is always at the
    # end (right-padded) and attention is causal, so trailing pad tokens
    # can never be attended to by any real token regardless -- the same
    # simplification ats.data.preference_dataset already relies on.
    return {
        key: torch.tensor([ex[key] for ex in batch], dtype=torch.long)
        for key in ("input_ids", "labels")
    }


def build_sft_dataloader(
    path: str,
    tokenizer_name: str,
    seq_length: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    num_workers: int = 4,
) -> DataLoader:
    """rank/world_size shard the dataset by simple index slicing, identical
    to ats.data.preference_dataset.build_preference_dataloader -- adequate
    at SFT-dataset scale; MixedDataset's shard-file distribution isn't
    needed here."""
    full_dataset: Dataset = SFTDataset(path, tokenizer_name, seq_length)
    if world_size > 1:
        indices = list(range(rank, len(full_dataset), world_size))  # type: ignore[arg-type]
        if not indices:
            raise ConfigError(
                f"SFT dataset at {path} has {len(full_dataset)} examples, too few to "  # type: ignore[arg-type]
                f"shard across world_size={world_size} at rank={rank} (this rank "
                f"would get zero examples). Fix: use fewer ranks or a larger SFT "
                f"dataset."
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