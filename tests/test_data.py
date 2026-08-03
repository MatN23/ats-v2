"""Tests for ats.data: tokenizer round-trip, dataset chunking with no data
loss at boundaries, and dataloader collation shapes."""

from __future__ import annotations

import json

import pytest
import torch

from ats.config.schema import ConfigError, DataSource
from ats.data.dataloader import _collate
from ats.data.dataset import MixedDataset

try:
    import tiktoken  # noqa: F401
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class _FakeTokenizer:
    """Deterministic character-level fake tokenizer so tests don't require
    tiktoken/transformers to be installed to check MixedDataset's chunking
    and padding logic in isolation."""

    vocab_size = 256
    eos_token_id = 256
    pad_token_id = 256

    def encode(self, text: str):
        return [ord(c) % 200 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i < 200)


def test_fake_tokenizer_round_trip():
    tok = _FakeTokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


@pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed in this environment")
def test_real_tokenizer_round_trip():
    from ats.data.tokenizer import Tokenizer

    tok = Tokenizer("tiktoken:cl100k_base")
    text = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_dataset_yields_fixed_length_chunks(tmp_path):
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        for i in range(5):
            f.write(json.dumps({"text": "abcdefghij"}) + "\n")

    tok = _FakeTokenizer()
    dataset = MixedDataset(
        sources=[DataSource(path=str(data_path), weight=1.0)], tokenizer=tok, seq_length=7, seed=0,
    )
    examples = list(dataset)
    assert len(examples) > 0
    for ex in examples:
        assert len(ex["input_ids"]) == 7
        assert len(ex["labels"]) == 7


def test_dataset_no_data_loss_at_final_boundary(tmp_path):
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        f.write(json.dumps({"text": "abc"}) + "\n")  # 3 chars + eos = 4 tokens, seq_length=10

    tok = _FakeTokenizer()
    dataset = MixedDataset(
        sources=[DataSource(path=str(data_path), weight=1.0)], tokenizer=tok, seq_length=10, seed=0,
    )
    examples = list(dataset)
    assert len(examples) == 1
    example = examples[0]
    # First 4 tokens are real content (3 chars + eos); rest is padding masked
    # out of the loss via label=-100.
    assert example["input_ids"][:4] == tok.encode("abc") + [tok.eos_token_id]
    assert example["labels"][4:] == [-100] * 6


def test_dataset_rejects_missing_text_field(tmp_path):
    data_path = tmp_path / "bad.jsonl"
    with open(data_path, "w") as f:
        f.write(json.dumps({"not_text": "abc"}) + "\n")

    tok = _FakeTokenizer()
    dataset = MixedDataset(
        sources=[DataSource(path=str(data_path), weight=1.0)], tokenizer=tok, seq_length=4, seed=0,
    )
    with pytest.raises(ConfigError):
        list(dataset)


def test_collate_produces_correct_shapes():
    batch = [
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},
    ]
    collated = _collate(batch)
    assert collated["input_ids"].shape == (2, 4)
    assert collated["labels"].shape == (2, 4)
    assert collated["attention_mask"].shape == (2, 4)
    assert torch.equal(collated["attention_mask"], torch.ones(2, 4, dtype=torch.long))


def test_collate_rejects_inconsistent_lengths():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [1, 2], "labels": [1, 2]},
    ]
    with pytest.raises(ConfigError):
        _collate(batch)
