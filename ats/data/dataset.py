"""Weighted-mixture dataset over multiple sources.

Two source kinds are supported:
  - Raw text (.jsonl): tokenized on the fly, exactly as before.
  - Preprocessed (.bin + a sibling meta.json, written by preprocess.py):
    pre-tokenized fixed-length blocks read via numpy memmap, with no
    on-the-fly tokenization. This is what preprocess.py's --packing output
    is meant to be read by.

No data loss at raw-text shard boundaries: the final partial chunk of a
shard is padded with the tokenizer's pad token and a labels mask of -100
rather than being dropped. Preprocessed sources apply the same rule at
preprocessing time (see preprocess.py), storing per-block valid lengths.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np

from ats.config.schema import ConfigError, DataSource
from ats.data.tokenizer import Tokenizer

IGNORE_INDEX = -100
PREPROCESSED_META_FILENAME = "meta.json"
PREPROCESSED_TOKEN_DTYPE = np.int32

SourceKind = Literal["text", "preprocessed"]


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


def _load_preprocessed_meta(bin_path: Path) -> Dict[str, Any]:
    meta_path = bin_path.parent / PREPROCESSED_META_FILENAME
    if not meta_path.exists():
        raise ConfigError(
            f"Preprocessed source {bin_path} has no sibling {PREPROCESSED_META_FILENAME} "
            f"in {bin_path.parent}. Fix: point data.sources[*].path at a directory produced "
            f"by preprocess.py (containing tokens.bin, valid_lengths.npy, and meta.json)."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_preprocessed_examples(bin_path: Path, expected_seq_length: int) -> Iterator[Dict[str, Any]]:
    meta = _load_preprocessed_meta(bin_path)
    if meta["seq_length"] != expected_seq_length:
        raise ConfigError(
            f"Preprocessed source {bin_path} was built with seq_length="
            f"{meta['seq_length']}, but data.seq_length={expected_seq_length}. "
            f"Fix: re-run preprocess.py with --seq-length {expected_seq_length}, or "
            f"update data.seq_length to match the preprocessed files."
        )
    num_blocks = meta["num_blocks"]
    seq_length = meta["seq_length"]

    tokens = np.memmap(bin_path, dtype=PREPROCESSED_TOKEN_DTYPE, mode="r", shape=(num_blocks, seq_length))
    valid_lengths_path = bin_path.parent / "valid_lengths.npy"
    if not valid_lengths_path.exists():
        raise ConfigError(f"Preprocessed source {bin_path} is missing valid_lengths.npy.")
    valid_lengths = np.load(valid_lengths_path)
    if len(valid_lengths) != num_blocks:
        raise ConfigError(
            f"valid_lengths.npy for {bin_path} has {len(valid_lengths)} entries, "
            f"expected {num_blocks} (one per block)."
        )

    for block_idx in range(num_blocks):
        block = tokens[block_idx].tolist()
        valid_len = int(valid_lengths[block_idx])
        labels = list(block)
        for i in range(valid_len, seq_length):
            labels[i] = IGNORE_INDEX
        yield {"input_ids": block, "labels": labels}


def _resolve_source(source: DataSource) -> Tuple[SourceKind, Iterator[Any]]:
    path = Path(source.path)
    if path.suffix == ".bin":
        if not path.exists():
            raise ConfigError(
                f"data source path does not exist: {path}. "
                f"Fix: check data.sources[*].path in your config, or run preprocess.py first."
            )
        return "preprocessed", path  # actual iterator built later once seq_length is known
    if path.suffix in (".jsonl", ".json"):
        if not path.exists():
            raise ConfigError(
                f"data source path does not exist: {path}. "
                f"Fix: check data.sources[*].path in your config."
            )
        return "text", _iter_local_jsonl(path)
    raise ConfigError(
        f"Unsupported data source path '{source.path}'. "
        f"ats-v2 currently reads local .jsonl files (raw text, tokenized on the fly) "
        f"or .bin files produced by preprocess.py (pre-tokenized blocks). "
        f"HuggingFace datasets streaming and WebDataset tar-shard support are not "
        f"implemented yet. Fix: point data.sources[*].path at a local .jsonl or .bin file."
    )


class MixedDataset:
    """Iterable dataset that samples proportionally to `source.weight` across
    `sources`. Raw-text (.jsonl) sources are tokenized and packed into
    fixed-length `seq_length` chunks on the fly, with no data loss at
    source-file boundaries. Preprocessed (.bin) sources are read directly as
    already-fixed-length blocks via numpy memmap."""

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
        kinds: List[SourceKind] = []
        source_iters: List[Iterator[Any]] = []
        for s in self.sources:
            kind, payload = _resolve_source(s)
            kinds.append(kind)
            if kind == "preprocessed":
                source_iters.append(_iter_preprocessed_examples(payload, self.seq_length))
            else:
                source_iters.append(payload)

        buffer: List[int] = []
        active = list(range(len(self.sources)))

        while active:
            idx = rng.choices(active, weights=[self._probs[i] for i in active], k=1)[0]

            if kinds[idx] == "preprocessed":
                try:
                    yield next(source_iters[idx])
                except StopIteration:
                    active.remove(idx)
                continue

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
