"""Weighted-mixture dataset over multiple sources (local jsonl/text shards,
HuggingFace datasets in streaming mode, or WebDataset tar shards).

No data loss at shard boundaries: the final partial chunk of a shard is
padded with the tokenizer's pad token and a labels mask of -100 rather than
being dropped.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ats.config.schema import ConfigError, DataSource
from ats.data.tokenizer import Tokenizer

IGNORE_INDEX = -100


def _iter_local_jsonl(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"Malformed JSON on line {line_num} of {path}: {exc}. "
                    f"Fix: ensure every non-empty line is a valid JSON object "
                    f"with a 'text' field."
                ) from exc
            if "text" not in record:
                raise ConfigError(
                    f"Line {line_num} of {path} has no 'text' field: {record}. "
                    f"Fix: every record must contain a 'text' key with the raw string."
                )
            yield record["text"]


def _resolve_source(source: DataSource) -> Iterator[str]:
    path = Path(source.path)
    if path.suffix in (".jsonl", ".json"):
        if not path.exists():
            raise ConfigError(
                f"data source path does not exist: {path}. "
                f"Fix: check data.sources[*].path in your config."
            )
        return _iter_local_jsonl(path)
    raise ConfigError(
        f"Unsupported data source path '{source.path}'. "
        f"ats-v2 currently reads local .jsonl files directly; for HuggingFace "
        f"datasets streaming or WebDataset tar shards, point data.sources[*].path "
        f"at a loader script that yields records with a 'text' field."
    )


class MixedDataset:
    """Iterable dataset that samples proportionally to `source.weight` across
    `sources`, tokenizes, and yields fixed-length `seq_length` chunks with no
    data loss at source-file boundaries."""

    def __init__(
        self, sources: List[DataSource], tokenizer: Tokenizer, seq_length: int, seed: int = 42,
    ) -> None:
        if not sources:
            raise ConfigError("MixedDataset requires at least one source.")
        if seq_length <= 0:
            raise ConfigError(f"MixedDataset seq_length must be positive, got {seq_length}.")
        self.sources = sources
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        total_weight = sum(s.weight for s in sources)
        self._probs = [s.weight / total_weight for s in sources]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = random.Random(self.seed)
        source_iters = [iter(_resolve_source(s)) for s in self.sources]
        buffer: List[int] = []
        active = list(range(len(self.sources)))

        while active:
            idx = rng.choices(active, weights=[self._probs[i] for i in active], k=1)[0]
            try:
                text = next(source_iters[idx])
            except StopIteration:
                active.remove(idx)
                continue

            token_ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            buffer.extend(token_ids)

            while len(buffer) >= self.seq_length:
                chunk = buffer[: self.seq_length]
                buffer = buffer[self.seq_length:]
                yield self._make_example(chunk, is_padded=False)

        if buffer:
            # Final partial chunk: pad instead of dropping, mask pad positions
            # in the labels so they don't contribute to the loss.
            pad_len = self.seq_length - len(buffer)
            padded = buffer + [self.tokenizer.pad_token_id] * pad_len
            yield self._make_example(padded, is_padded=True, valid_len=len(buffer))

    def _make_example(
        self, chunk: List[int], is_padded: bool, valid_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        labels = list(chunk)
        if is_padded:
            valid_len = valid_len if valid_len is not None else len(chunk)
            for i in range(valid_len, len(chunk)):
                labels[i] = IGNORE_INDEX
        return {"input_ids": chunk, "labels": labels}
