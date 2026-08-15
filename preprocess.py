#!/usr/bin/env python
"""Entry point: python preprocess.py --input data.jsonl --output-dir ./preprocessed \
    --tokenizer cl100k_base --seq-length 4096 [--packing]

Tokenizes a .jsonl dataset offline and streams it directly to disk as a
memory-mapped-readable .bin file of fixed-length int32 token blocks, a
valid_lengths.npy array (how many of each block's tokens are real content
vs. padding), and a meta.json describing the layout. Peak memory is O(one
block), not O(corpus size): each block's bytes are written to disk and
discarded immediately, rather than accumulating the whole tokenized corpus
in memory first. ats/data/dataset.py's MixedDataset reads the result
directly via np.memmap, with no on-the-fly tokenization.

Without --packing, each input document becomes its own block (truncated if
too long, padded if too short) -- simple, but wastes space on padding for
short documents. With --packing, documents are concatenated back-to-back
(separated by an EOS token) and sliced into seq_length blocks, so a corpus
of many short documents fills blocks almost completely instead of mostly
padding. Example: if the average document is 800 tokens and seq_length is
4096, packing fits ~5 documents per block instead of leaving ~3300 tokens
of padding per single-document block.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ats.data.tokenizer import Tokenizer
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.preprocess")

TOKEN_DTYPE = np.int32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-tokenize a dataset for ats-v2."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .jsonl file (records with a 'text' field).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write tokens.bin/valid_lengths.npy/meta.json to.",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="tiktoken encoding name (e.g. cl100k_base) or 'hf:<model_id>' for a HuggingFace tokenizer.",
    )
    parser.add_argument(
        "--seq-length", type=int, required=True, help="Fixed block length in tokens."
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Pack multiple short documents into each block instead of one document per block.",
    )
    return parser


def _resolve_tokenizer_name(tokenizer_arg: str) -> str:
    if tokenizer_arg.startswith(("hf:", "tiktoken:")):
        return tokenizer_arg
    return f"tiktoken:{tokenizer_arg}"


def _iter_documents(input_path: Path) -> Iterator[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_num} of {input_path}: {exc}"
                ) from exc
            if "text" not in record:
                raise ValueError(
                    f"Line {line_num} of {input_path} has no 'text' field: {record}"
                )
            yield record["text"]


def _tokenize_unpacked(
    documents: Iterator[str],
    tokenizer: Tokenizer,
    seq_length: int,
) -> Iterator[list[int]]:
    """One block per document: truncate long documents, pad short ones."""
    for text in documents:
        ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
        if len(ids) > seq_length:
            ids = tokenizer.truncate(ids, max_length=seq_length, strategy="right")
        yield ids  # caller pads to seq_length and records the true (unpadded) length


def _tokenize_packed(
    documents: Iterator[str],
    tokenizer: Tokenizer,
    seq_length: int,
) -> Iterator[list[int]]:
    """Concatenate documents (EOS-delimited) and slice into full seq_length
    blocks; only the very last block of the whole corpus may be partial."""
    buffer: list[int] = []
    for text in documents:
        buffer.extend(tokenizer.encode(text) + [tokenizer.eos_token_id])
        while len(buffer) >= seq_length:
            yield buffer[:seq_length]
            buffer = buffer[seq_length:]
    if buffer:
        yield buffer  # caller pads the final partial block


def preprocess(
    input_path: str,
    output_dir: str,
    tokenizer_name: str,
    seq_length: int,
    packing: bool,
) -> int:
    """Runs the full offline preprocessing pipeline. Returns the number of
    blocks written.

    Streams blocks directly to disk as they're produced, rather than
    accumulating the whole tokenized corpus in a Python list first: peak
    memory is O(one block), not O(corpus size). Verified this produces a
    byte-identical file to writing via np.memmap in one shot (see
    CHANGES.md for the verification), so ats/data/dataset.py's memmap
    reader needs no changes.
    """
    if seq_length <= 0:
        raise ValueError(f"--seq-length must be positive, got {seq_length}.")

    tokenizer = Tokenizer(_resolve_tokenizer_name(tokenizer_name))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    documents = _iter_documents(Path(input_path))
    block_generator = (
        _tokenize_packed(documents, tokenizer, seq_length)
        if packing
        else _tokenize_unpacked(documents, tokenizer, seq_length)
    )

    bin_path = out_path / "tokens.bin"
    valid_lengths: list[int] = []
    num_blocks = 0

    with open(bin_path, "wb") as bin_file:
        for block in block_generator:
            valid_len = len(block)
            if valid_len > seq_length:
                raise ValueError(
                    f"Internal error: produced a block of length {valid_len} > "
                    f"seq_length {seq_length}."
                )
            padded = block + [tokenizer.pad_token_id] * (seq_length - valid_len)
            # Write this block's raw bytes immediately and discard it --
            # only the current block is held in memory, not the whole corpus.
            bin_file.write(np.asarray(padded, dtype=TOKEN_DTYPE).tobytes())
            valid_lengths.append(valid_len)
            num_blocks += 1

    if num_blocks == 0:
        bin_path.unlink()  # clean up the empty file rather than leaving a stray 0-byte artifact
        raise ValueError(
            f"No documents were tokenized from {input_path}; is the file empty?"
        )

    np.save(out_path / "valid_lengths.npy", np.array(valid_lengths, dtype=np.int32))

    meta = {
        "seq_length": seq_length,
        "num_blocks": num_blocks,
        "tokenizer_name": tokenizer_name,
        "packing": packing,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
    }
    with open(out_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_tokens = int(sum(valid_lengths))
    total_capacity = num_blocks * seq_length
    padding_fraction = 1.0 - (total_tokens / total_capacity) if total_capacity else 0.0
    logger.info(
        "Wrote %d blocks (%d real tokens, %.1f%% padding) to %s",
        num_blocks,
        total_tokens,
        padding_fraction * 100,
        out_path,
    )
    return num_blocks


def main(argv=None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        preprocess(
            args.input, args.output_dir, args.tokenizer, args.seq_length, args.packing
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
