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


# --- Regression tests for reported bugs ---

@pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed in this environment")
def test_tiktoken_pad_and_eos_ids_are_in_range():
    """Regression test: pad_token_id/eos_token_id must be < vocab_size, or
    nn.Embedding(vocab_size, hidden_size) raises IndexError the moment a
    padded batch reaches the model. vocab_size must reserve the +1 slot."""
    from ats.data.tokenizer import Tokenizer

    tok = Tokenizer("tiktoken:cl100k_base")
    assert tok.eos_token_id < tok.vocab_size
    assert tok.pad_token_id < tok.vocab_size
    assert tok.eos_token_id >= 0
    assert tok.pad_token_id >= 0


@pytest.mark.skipif(not _TIKTOKEN_AVAILABLE, reason="tiktoken not installed in this environment")
def test_tiktoken_middle_truncation_marker_is_in_range():
    """'middle' truncation inserts self.eos_token_id as a marker; this must
    also be a valid embedding index."""
    from ats.data.tokenizer import Tokenizer

    tok = Tokenizer("tiktoken:cl100k_base")
    long_text = "word " * 100
    ids = tok.encode(long_text)
    truncated = tok.truncate(ids, max_length=10, strategy="middle")
    assert all(0 <= t < tok.vocab_size for t in truncated)


def test_dataloader_worker_sharding_produces_no_duplicates():
    """Regression test for the num_workers>0 duplicate-data bug: two
    DataLoader workers must each see a disjoint slice of the stream, not
    identical copies of the whole thing. Simulated directly against
    _TorchMixedDataset.__iter__ by monkeypatching get_worker_info, since
    spinning up real subprocess workers is unnecessary to prove the sharding
    math is correct."""
    import ats.data.dataloader as dataloader_module

    class _FakeWorkerInfo:
        def __init__(self, id_, num_workers):
            self.id = id_
            self.num_workers = num_workers

    class _FakeMixedDataset:
        def __iter__(self):
            return iter(range(20))

    fake_dataset = dataloader_module._TorchMixedDataset(_FakeMixedDataset(), rank=0, world_size=1)

    seen_by_worker = {}
    for worker_id in (0, 1):
        original = dataloader_module.get_worker_info
        dataloader_module.get_worker_info = lambda: _FakeWorkerInfo(worker_id, 2)
        try:
            seen_by_worker[worker_id] = list(fake_dataset)
        finally:
            dataloader_module.get_worker_info = original

    worker_0_items = set(seen_by_worker[0])
    worker_1_items = set(seen_by_worker[1])
    assert worker_0_items.isdisjoint(worker_1_items)
    assert worker_0_items | worker_1_items == set(range(20))


# --- preprocess.py / preprocessed-source reading tests ---

def test_preprocess_packed_output_round_trips(tmp_path):
    """Exercises the real preprocess.preprocess() pipeline end-to-end using
    the fake character-level tokenizer's package-free logic pattern, but
    via the actual module this time (tiktoken not required: uses a
    monkeypatched Tokenizer)."""
    import preprocess as preprocess_module

    input_path = tmp_path / "docs.jsonl"
    with open(input_path, "w") as f:
        for text in ["hello world", "a short doc", "another one here", "final doc"]:
            f.write(json.dumps({"text": text}) + "\n")

    class _FakeTok:
        vocab_size = 300
        eos_token_id = 299
        pad_token_id = 299

        def __init__(self, _name):
            pass  # intentional no-op: test double, ignores the tokenizer name arg

        def encode(self, text):
            return [ord(c) % 250 for c in text]

        def truncate(self, ids, max_length, strategy="right"):
            return ids[:max_length]

    preprocess_module.Tokenizer = _FakeTok  # avoid requiring tiktoken/transformers

    output_dir = tmp_path / "preprocessed"
    num_blocks = preprocess_module.preprocess(
        str(input_path), str(output_dir), "cl100k_base", seq_length=10, packing=True,
    )
    assert num_blocks > 0
    assert (output_dir / "tokens.bin").exists()
    assert (output_dir / "valid_lengths.npy").exists()
    assert (output_dir / "meta.json").exists()

    with open(output_dir / "meta.json") as f:
        meta = json.load(f)
    assert meta["seq_length"] == 10
    assert meta["num_blocks"] == num_blocks
    assert meta["packing"] is True


def test_preprocessed_source_read_by_mixed_dataset(tmp_path):
    """MixedDataset must read a preprocess.py-shaped .bin source directly
    via memmap, with no on-the-fly tokenization, and respect valid_lengths
    for label masking on the padded final block."""
    import numpy as np

    from ats.data.dataset import MixedDataset

    seq_length = 6
    blocks = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]]  # second block: 3 valid + 3 pad
    valid_lengths = [6, 3]

    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    arr = np.array(blocks, dtype=np.int32)
    mm = np.memmap(out_dir / "tokens.bin", dtype=np.int32, mode="w+", shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    np.save(out_dir / "valid_lengths.npy", np.array(valid_lengths, dtype=np.int32))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"seq_length": seq_length, "num_blocks": 2}, f)

    dataset = MixedDataset(
        sources=[DataSource(path=str(out_dir / "tokens.bin"), weight=1.0)],
        tokenizer=_FakeTokenizer(), seq_length=seq_length, seed=0,
    )
    examples = list(dataset)
    assert len(examples) == 2
    input_ids_seen = {tuple(ex["input_ids"]) for ex in examples}
    assert input_ids_seen == {tuple(blocks[0]), tuple(blocks[1])}

    padded_example = next(ex for ex in examples if ex["input_ids"] == blocks[1])
    assert padded_example["labels"] == [7, 8, 9, -100, -100, -100]


def test_preprocessed_source_seq_length_mismatch_raises(tmp_path):
    import numpy as np

    from ats.data.dataset import MixedDataset

    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    arr = np.array([[1, 2, 3, 4]], dtype=np.int32)
    mm = np.memmap(out_dir / "tokens.bin", dtype=np.int32, mode="w+", shape=arr.shape)
    mm[:] = arr[:]
    mm.flush()
    np.save(out_dir / "valid_lengths.npy", np.array([4], dtype=np.int32))
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"seq_length": 4, "num_blocks": 1}, f)

    dataset = MixedDataset(
        sources=[DataSource(path=str(out_dir / "tokens.bin"), weight=1.0)],
        tokenizer=_FakeTokenizer(), seq_length=8, seed=0,  # mismatched on purpose
    )
    with pytest.raises(ConfigError):
        list(dataset)
