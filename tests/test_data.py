"""Tests for ats.data: tokenizer round-trip, dataset chunking with no data
loss at boundaries, and dataloader collation shapes."""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

from ats.config.schema import ConfigError, DataSource
from ats.data.dataloader import _collate
from ats.data.dataset import MixedDataset

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _load_preprocess_module() -> types.ModuleType:
    """Load the repo-root preprocess.py script as a module, by absolute
    path rather than `import preprocess`.

    preprocess.py is a top-level script, not part of the installed `ats`
    package, so it's only importable by name when the repo root happens to
    be on sys.path. That's true under `python -m pytest` (which always
    prepends cwd to sys.path) but NOT under a bare `pytest` invocation
    with the package installed via `pip install -e .` -- a PEP 660
    editable install only maps the `ats` package itself (see
    __editable___ats_v2_*_finder.py's MAPPING dict), so a bare `pytest`
    run (exactly what CI's `run: pytest -v --cov=ats ...` does) raises
    ModuleNotFoundError here. Loading by absolute file path sidesteps
    sys.path entirely, so this works the same way regardless of cwd,
    invocation mode, or installation method."""
    preprocess_path = Path(__file__).resolve().parent.parent / "preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiktoken_usable() -> bool:
    """tiktoken being importable doesn't mean it's usable: on first use it
    downloads its encoding's BPE merge file from a remote CDN
    (openaipublic.blob.core.windows.net), which fails in any
    network-restricted environment (many CI runners and sandboxes block
    egress to arbitrary hosts) even though the package itself is installed.
    Actually attempting the fetch here means the four tests below correctly
    skip with a clear reason in that case, instead of failing with a raw
    requests.exceptions.HTTPError that looks like a code bug."""
    if not _TIKTOKEN_AVAILABLE:
        return False
    try:
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:  # noqa: BLE001 -- deliberately broad: this is an availability
        # probe, and tiktoken.get_encoding can raise several different network/HTTP
        # exception types depending on environment; any failure here means "unusable".
        return False


_TIKTOKEN_USABLE = _tiktoken_usable()


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


@pytest.mark.skipif(
    not _TIKTOKEN_USABLE,
    reason="tiktoken cl100k_base encoding not usable (not installed, or its data file could not be downloaded -- e.g. no network/blocked egress)",
)
def test_real_tokenizer_round_trip():
    from ats.data.tokenizer import Tokenizer

    tok = Tokenizer("tiktoken:cl100k_base")
    text = "The quick brown fox jumps over the lazy dog."
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_dataset_yields_fixed_length_chunks(tmp_path):
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        f.writelines(json.dumps({"text": "abcdefghij"}) + "\n" for i in range(5))

    tok = _FakeTokenizer()
    dataset = MixedDataset(
        sources=[DataSource(path=str(data_path), weight=1.0)],
        tokenizer=tok,
        seq_length=7,
        seed=0,
    )
    examples = list(dataset)
    assert len(examples) > 0
    for ex in examples:
        assert len(ex["input_ids"]) == 7
        assert len(ex["labels"]) == 7


def test_dataset_no_data_loss_at_final_boundary(tmp_path):
    data_path = tmp_path / "data.jsonl"
    with open(data_path, "w") as f:
        f.write(
            json.dumps({"text": "abc"}) + "\n"
        )  # 3 chars + eos = 4 tokens, seq_length=10

    tok = _FakeTokenizer()
    dataset = MixedDataset(
        sources=[DataSource(path=str(data_path), weight=1.0)],
        tokenizer=tok,
        seq_length=10,
        seed=0,
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
        sources=[DataSource(path=str(data_path), weight=1.0)],
        tokenizer=tok,
        seq_length=4,
        seed=0,
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


@pytest.mark.skipif(
    not _TIKTOKEN_USABLE,
    reason="tiktoken cl100k_base encoding not usable (not installed, or its data file could not be downloaded -- e.g. no network/blocked egress)",
)
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


@pytest.mark.skipif(
    not _TIKTOKEN_USABLE,
    reason="tiktoken cl100k_base encoding not usable (not installed, or its data file could not be downloaded -- e.g. no network/blocked egress)",
)
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

    fake_dataset = dataloader_module._TorchMixedDataset(
        _FakeMixedDataset(), rank=0, world_size=1
    )

    seen_by_worker = {}
    for worker_id in (0, 1):
        original = dataloader_module.get_worker_info
        dataloader_module.get_worker_info = lambda wid=worker_id: _FakeWorkerInfo(
            wid, 2
        )
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
    preprocess_module = _load_preprocess_module()

    input_path = tmp_path / "docs.jsonl"
    with open(input_path, "w") as f:
        f.writelines(
            json.dumps({"text": text}) + "\n"
            for text in ["hello world", "a short doc", "another one here", "final doc"]
        )

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
        str(input_path),
        str(output_dir),
        "cl100k_base",
        seq_length=10,
        packing=True,
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
        tokenizer=_FakeTokenizer(),
        seq_length=seq_length,
        seed=0,
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
        tokenizer=_FakeTokenizer(),
        seq_length=8,
        seed=0,  # mismatched on purpose
    )
    with pytest.raises(ConfigError):
        list(dataset)


def test_build_dataloader_does_not_vary_seed_by_rank():
    """Regression test for a real correctness/efficiency bug: build_dataloader
    previously passed seed=seed+rank to MixedDataset, giving each rank an
    independently different random stream -- which then got ADDITIONALLY
    modulo-sharded by _TorchMixedDataset, compounding into each rank
    discarding most of its own already-unique stream (confirmed: at
    world_size=8, only 12.5% throughput). Every rank must now construct
    MixedDataset with the SAME seed, so the modulo-based sharding in
    _TorchMixedDataset.__iter__ can correctly partition one shared
    deterministic stream instead of over-shrinking N different ones."""
    import inspect

    from ats.data.dataloader import build_dataloader

    source = inspect.getsource(build_dataloader)
    assert "seed=seed + rank" not in source and "seed=seed+rank" not in source, (
        "build_dataloader must not vary MixedDataset's seed by rank -- this "
        "breaks the modulo-based sharding in _TorchMixedDataset, which "
        "requires every rank to see the same underlying stream."
    )
    assert "seed=seed," in source or "seed=seed)" in source


def test_dataloader_rank_sharding_gives_full_coverage_no_duplicates():
    """End-to-end version of the fix: build _TorchMixedDataset directly
    (bypassing tokenization) with the SAME underlying stream for every
    rank, confirm the union across all ranks covers every example exactly
    once -- no duplication, no silently-discarded data."""
    import ats.data.dataloader as dataloader_module

    class _FixedStream:
        def __iter__(self):
            return iter(range(200))

    world_size = 4
    all_seen = []
    for rank in range(world_size):
        ds = dataloader_module._TorchMixedDataset(
            _FixedStream(), rank=rank, world_size=world_size
        )
        all_seen.extend(list(ds))

    assert len(all_seen) == 200
    assert len(set(all_seen)) == 200
    assert sorted(all_seen) == list(range(200))


@pytest.mark.skipif(
    not _TIKTOKEN_USABLE,
    reason="tiktoken cl100k_base encoding not usable (not installed, or its data file could not be downloaded -- e.g. no network/blocked egress)",
)
def test_tokenizer_decode_filters_negative_ids():
    """Regression test: decode() previously only filtered ids >= vocab_size,
    not negative ids -- so passing a labels array (which legitimately
    contains -100 at masked/padded positions) would crash the underlying
    decoder instead of gracefully skipping those positions."""
    from ats.data.tokenizer import Tokenizer

    tok = Tokenizer("tiktoken:cl100k_base")
    text = "hello world"
    ids = tok.encode(text)

    # Simulate a labels array with some positions masked out (-100), as
    # ats.data.dataset.MixedDataset actually produces for padded blocks.
    labels_with_masking = ids[:3] + [-100, -100] + ids[3:]
    decoded = tok.decode(labels_with_masking)  # must not raise
    assert decoded == text  # masked positions contribute nothing to the output
