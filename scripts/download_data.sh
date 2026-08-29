#!/usr/bin/env bash
# download_data.sh — download exactly as much training data as a config
# actually needs, no more, no less.
#
# "Perfect amount" here means: the exact token count the trainer will
# consume over a full run of `training.max_steps` optimizer steps, i.e.
#
#     tokens_needed = max_steps
#                    * training.grad_accum_steps
#                    * training.micro_batch_size
#                    * data.seq_length
#                    * parallelism.gpus * parallelism.nodes
#
# (micro_batch_size and grad_accum_steps are PER GPU; each of
# gpus*nodes data-parallel ranks pulls its own stream, so the world
# multiplies in.) A small safety margin (default 5%) is added on top
# to cover tokenizer overhead (EOS tokens, docs that don't tokenize
# to nice round chunk boundaries) so you don't end up a few thousand
# tokens short on the very last step.
#
# Streams text from a HuggingFace dataset (default: fineweb-edu, whose
# `text` field already matches the {"text": ...} jsonl format
# ats/data/dataset.py expects), tokenizes with the EXACT tiktoken
# encoding named in data.tokenizer_name, and stops the instant the
# token budget is hit — so download time/disk usage scales with what
# the config will actually train on, not a guess.
#
# Usage:
#   ./scripts/download_data.sh --config configs/125m.yaml
#   ./scripts/download_data.sh --all
#   ./scripts/download_data.sh --all --dry-run
#   ./scripts/download_data.sh --config configs/7b.yaml --margin 1.10 --force
#
# Options:
#   --config PATH     Path to one config YAML. Repeatable.
#   --all              Process every configs/*.yaml (skips ones with no
#                       data.sources, e.g. purely architectural configs).
#   --dataset NAME      HF dataset to stream from. Default: HuggingFaceFW/fineweb-edu
#   --dataset-config C  HF dataset config/subset name. Default: sample-10BT
#   --split NAME         HF dataset split. Default: train
#   --text-field NAME     Field in the HF dataset containing raw text. Default: text
#   --transfer MODE        auto (default) | shards | stream. 'shards' downloads whole
#                           parquet files via huggingface_hub.hf_hub_download (uses HF's
#                           accelerated Xet/hf_transfer transfer path, and lets a second
#                           run skip shards it already has cached locally) and only reads
#                           as many shards as needed to hit the budget. 'stream' is the
#                           old row-by-row datasets streaming path -- much slower for
#                           Xet-backed repos (small ranged HTTP reads instead of bulk
#                           transfer), but works for datasets 'shards' can't resolve a
#                           file list for. 'auto' tries shards, falls back to stream.
#   --file-glob PATTERN     Override the glob used to find this dataset's parquet shards
#                           in 'shards'/'auto' mode. Default guess: dataset-config values
#                           of the form 'sample-10BT' -> 'sample/10BT/*.parquet' (fineweb's
#                           convention); anything else falls back to 'stream' unless you
#                           pass this explicitly.
#   --margin FLOAT      Multiplier applied to the computed token budget. Default: 1.05
#   --out PATH             Override the output .jsonl path (only valid with a single --config).
#   --batch-size N          Docs tokenized per tiktoken encode_batch() call. Default: 512.
#                           Higher = fewer, bigger batches (faster, more RAM); this is the
#                           main speed knob -- tokenizing one doc at a time is dramatically
#                           slower than batching, since tiktoken's batch API releases the
#                           GIL and parallelizes across threads.
#   --force               Re-download even if the destination already has enough tokens.
#   --dry-run              Print the computed budget table and exit; no download.
#   --install-deps         pip install the (few) missing Python deps and continue.
#   -h, --help              Show this help.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIGS=()
ALL=0
DATASET="HuggingFaceFW/fineweb-edu"
DATASET_CONFIG="sample-10BT"
SPLIT="train"
TEXT_FIELD="text"
TRANSFER="auto"
FILE_GLOB=""
MARGIN="1.05"
OUT_OVERRIDE=""
BATCH_SIZE=512
FORCE=0
DRY_RUN=0
INSTALL_DEPS=0

usage() {
    # Print the leading '#'-comment block (lines 2 through the first
    # non-comment line), dynamically -- avoids a hardcoded line range going
    # stale every time this header comment is edited.
    awk 'NR==1{next} /^#/{sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIGS+=("$2"); shift 2 ;;
        --all)              ALL=1; shift ;;
        --dataset)           DATASET="$2"; shift 2 ;;
        --dataset-config)    DATASET_CONFIG="$2"; shift 2 ;;
        --split)              SPLIT="$2"; shift 2 ;;
        --text-field)          TEXT_FIELD="$2"; shift 2 ;;
        --transfer)             TRANSFER="$2"; shift 2 ;;
        --file-glob)             FILE_GLOB="$2"; shift 2 ;;
        --margin)               MARGIN="$2"; shift 2 ;;
        --out)                   OUT_OVERRIDE="$2"; shift 2 ;;
        --batch-size)             BATCH_SIZE="$2"; shift 2 ;;
        --force)                  FORCE=1; shift ;;
        --dry-run)                  DRY_RUN=1; shift ;;
        --install-deps)              INSTALL_DEPS=1; shift ;;
        -h|--help)                    usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${ALL}" -eq 1 && ${#CONFIGS[@]} -gt 0 ]]; then
    echo "error: --all and --config are mutually exclusive" >&2
    exit 1
fi
if [[ "${ALL}" -eq 0 && ${#CONFIGS[@]} -eq 0 ]]; then
    echo "error: pass --config PATH (repeatable) or --all" >&2
    usage
    exit 1
fi
if [[ -n "${OUT_OVERRIDE}" && ${#CONFIGS[@]} -ne 1 ]]; then
    echo "error: --out only makes sense with exactly one --config" >&2
    exit 1
fi
case "${TRANSFER}" in
    auto|shards|stream) ;;
    *) echo "error: --transfer must be one of: auto, shards, stream (got '${TRANSFER}')" >&2; exit 1 ;;
esac

if [[ "${ALL}" -eq 1 ]]; then
    while IFS= read -r -d '' f; do
        CONFIGS+=("${f}")
    done < <(find "${REPO_ROOT}/configs" -maxdepth 1 -name '*.yaml' -print0 | sort -z)
fi

command -v python3 >/dev/null 2>&1 || { echo "error: python3 is required" >&2; exit 1; }

missing_deps() {
    python3 - "${DRY_RUN}" "${TRANSFER}" <<'PYEOF'
import importlib
import sys
dry_run = sys.argv[1] == "1"
transfer = sys.argv[2]
needed = [("yaml", "pyyaml")]
if not dry_run:
    needed += [("tiktoken", "tiktoken")]
    if transfer in ("auto", "shards"):
        needed += [("huggingface_hub", "huggingface_hub"), ("pyarrow", "pyarrow")]
    if transfer in ("auto", "stream"):
        needed += [("datasets", "datasets")]
missing = []
for mod, pip_name in needed:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(pip_name)
print(" ".join(missing))
PYEOF
}

MISSING="$(missing_deps)"
if [[ -n "${MISSING}" ]]; then
    if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
        echo "Installing missing deps: ${MISSING}" >&2
        pip install --break-system-packages -q ${MISSING}
    else
        echo "error: missing Python deps: ${MISSING}" >&2
        echo "  re-run with --install-deps, or: pip install ${MISSING}" >&2
        exit 1
    fi
fi

mkdir -p "${REPO_ROOT}/data"

for cfg in "${CONFIGS[@]}"; do
    if [[ ! -f "${cfg}" ]]; then
        echo "error: config not found: ${cfg}" >&2
        exit 1
    fi
done

# All the real work — YAML parsing, budget math, streaming download,
# tokenization-aware stopping, resume/skip logic — happens in one Python
# process per config so the token count used to decide "enough" is
# computed with the exact same tokenizer the trainer will use.
python3 - "${REPO_ROOT}" "${DRY_RUN}" "${FORCE}" "${MARGIN}" "${DATASET}" "${DATASET_CONFIG}" "${SPLIT}" "${TEXT_FIELD}" "${OUT_OVERRIDE}" "${BATCH_SIZE}" "${TRANSFER}" "${FILE_GLOB}" "${CONFIGS[@]}" <<'PYEOF'
import json
import os
import sys
from pathlib import Path

import yaml

repo_root = Path(sys.argv[1])
dry_run = sys.argv[2] == "1"
force = sys.argv[3] == "1"
margin = float(sys.argv[4])
dataset_name = sys.argv[5]
dataset_config = sys.argv[6]
split = sys.argv[7]
text_field = sys.argv[8]
out_override = sys.argv[9] or None
batch_size = int(sys.argv[10])
transfer_mode = sys.argv[11]
file_glob_override = sys.argv[12] or None
config_paths = sys.argv[13:]


def load_budget(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    training = cfg.get("training", {})
    data = cfg.get("data", {})
    parallelism = cfg.get("parallelism", {})
    sources = data.get("sources")
    if not sources:
        return None  # nothing to download for this config

    max_steps = training.get("max_steps")
    grad_accum = training.get("grad_accum_steps", 1)
    micro_batch = training.get("micro_batch_size", 1)
    seq_len = data.get("seq_length")
    gpus = parallelism.get("gpus", 1)
    nodes = parallelism.get("nodes", 1)
    tokenizer_name = data.get("tokenizer_name", "tiktoken:cl100k_base")
    out_path = sources[0]["path"]

    if max_steps is None or seq_len is None:
        return None

    world_size = max(1, gpus) * max(1, nodes)
    raw_tokens = max_steps * grad_accum * micro_batch * seq_len * world_size
    budget = int(raw_tokens * margin)
    return {
        "cfg_path": cfg_path,
        "size": cfg.get("model", {}).get("size", "?"),
        "max_steps": max_steps,
        "grad_accum": grad_accum,
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "world_size": world_size,
        "raw_tokens": raw_tokens,
        "budget": budget,
        "tokenizer_name": tokenizer_name,
        "out_path": out_path,
    }


def human(n: int) -> str:
    n = float(n)
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}P"


plans = []
for cfg_path in config_paths:
    plan = load_budget(cfg_path)
    if plan is None:
        print(f"skip  {cfg_path}: no data.sources / max_steps, nothing to compute")
        continue
    plans.append(plan)

if not plans:
    print("Nothing to do.")
    sys.exit(0)

print()
print(f"{'config':<22} {'size':>6} {'steps':>8} {'world':>6} {'seq_len':>8} {'tokens needed':>16}")
for p in plans:
    print(
        f"{Path(p['cfg_path']).name:<22} {p['size']:>6} {p['max_steps']:>8}"
        f" {p['world_size']:>6} {p['seq_len']:>8} {human(p['budget']):>16}"
    )
print()

if dry_run:
    print("(dry run — nothing downloaded)")
    sys.exit(0)

# Tokenizer import is always needed once we're actually downloading;
# huggingface_hub/pyarrow (shard mode) and datasets (stream mode) are
# imported lazily below, only for the transfer path actually used.
import tiktoken


def get_encoding(tokenizer_name: str):
    if not tokenizer_name.startswith("tiktoken:"):
        raise SystemExit(
            f"This script only knows how to count tokens for 'tiktoken:*' "
            f"tokenizer_name specs; got {tokenizer_name!r}. Fix: either use a "
            f"tiktoken encoding, or adapt this script's get_encoding()."
        )
    encoding_name = tokenizer_name.split(":", 1)[1]
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # network/proxy errors, unknown encoding, etc.
        raise SystemExit(
            f"Failed to load tiktoken encoding {encoding_name!r}: {exc}\n"
            f"This is usually a network/proxy issue (tiktoken downloads its "
            f"BPE file from openaipublic.blob.core.windows.net on first use "
            f"and caches it locally after that) -- check that host is "
            f"reachable, or pre-warm the tiktoken cache on a machine that "
            f"can reach it and copy ~/.cache/tiktoken over."
        ) from exc


def existing_token_count(path: Path, enc, batch_size: int) -> int:
    if not path.exists():
        return 0
    total = 0
    texts = []

    def flush():
        nonlocal total
        if not texts:
            return
        for ids in enc.encode_batch(texts, num_threads=os.cpu_count() or 4):
            total += len(ids) + 1
        texts.clear()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if len(texts) >= batch_size:
                flush()
    flush()
    return total


def flush_batch(texts, out_f, enc, budget: int, tokens_so_far: int):
    """Batch-tokenize `texts` in one tiktoken call (releases the GIL and uses
    multiple threads internally -- an order of magnitude faster than calling
    enc.encode() once per document in a Python loop), then write+count them
    one at a time so we can still stop at the EXACT token budget rather than
    overshooting by a whole batch."""
    if not texts:
        return tokens_so_far, 0, False
    encoded = enc.encode_batch(texts, num_threads=os.cpu_count() or 4)
    written = 0
    for text, ids in zip(texts, encoded):
        out_f.write(json.dumps({"text": text}) + "\n")
        tokens_so_far += len(ids) + 1  # +1 for the EOS the trainer appends
        written += 1
        if tokens_so_far >= budget:
            return tokens_so_far, written, True
    return tokens_so_far, written, False


def guess_file_glob(dataset_config: str) -> str | None:
    # Fineweb-family convention: config "sample-10BT" lives under the repo
    # path "sample/10BT/*.parquet". Nothing more general is safe to guess.
    if dataset_config.startswith("sample-"):
        return f"sample/{dataset_config[len('sample-'):]}/*.parquet"
    return None


def resolve_shard_files(dataset_name: str, file_glob: str):
    """Lists this dataset repo's files (a single, cheap API call -- no data
    transferred) and returns the ones matching file_glob, in a stable sorted
    order (matters for resuming: later reruns should look at the same shards
    first)."""
    import fnmatch

    from huggingface_hub import HfApi

    all_files = HfApi().list_repo_files(repo_id=dataset_name, repo_type="dataset")
    matches = sorted(f for f in all_files if fnmatch.fnmatch(f, file_glob))
    return matches


def process_shard_file(local_path, text_field, enc, batch_size, out_f, budget, tokens_so_far):
    """Reads one already-downloaded parquet shard with column projection (only
    the text column's pages are materialized, not the other ~8 metadata
    columns fineweb-family datasets carry), then batch-tokenizes/writes/counts
    exactly like the streaming path's flush_batch, stopping the instant the
    budget is hit (possibly partway through this shard)."""
    import pyarrow.parquet as pq

    table = pq.read_table(local_path, columns=[text_field])
    texts_all = table.column(text_field).to_pylist()
    del table

    docs_written = 0
    hit_budget = False
    for i in range(0, len(texts_all), batch_size):
        chunk = [t for t in texts_all[i : i + batch_size] if t]
        tokens_so_far, n, hit_budget = flush_batch(chunk, out_f, enc, budget, tokens_so_far)
        docs_written += n
        if hit_budget:
            break
    return tokens_so_far, docs_written, hit_budget


def download_via_shards(dataset_name, dataset_config, text_field, file_glob, enc,
                         batch_size, out_f, budget, tag):
    """Whole-file downloads via huggingface_hub.hf_hub_download instead of
    datasets' row-by-row streaming. This matters specifically because
    Xet/LFS-backed repos (fineweb-edu's parquet shards are Xet-backed) are
    only served through HF's accelerated chunked/deduplicated transfer path
    (and respect HF_HUB_ENABLE_HF_TRANSFER) when fetched as whole files --
    datasets' streaming=True mode reads via small ranged HTTP requests over
    fsspec instead, which doesn't use that fast path. Bonus: hf_hub_download
    caches shards locally, so a second run (or a different config that reuses
    the same dataset) skips the network entirely for shards already on disk.
    Returns (tokens_written, docs_written, hit_budget) or raises on failure
    so the caller can fall back to streaming mode.
    """
    from huggingface_hub import hf_hub_download

    files = resolve_shard_files(dataset_name, file_glob)
    if not files:
        raise RuntimeError(
            f"no files in {dataset_name} matched glob {file_glob!r} "
            f"(dataset repo layout may differ from the fineweb convention "
            f"this guess is based on -- pass --file-glob explicitly)"
        )
    print(f"[{tag}] shard mode: {len(files)} candidate shard(s) matching {file_glob!r}")

    tokens_written = 0
    docs_written = 0
    for i, filename in enumerate(files):
        print(f"[{tag}] fetching shard {i + 1}/{len(files)}: {filename}")
        local_path = hf_hub_download(
            repo_id=dataset_name, filename=filename, repo_type="dataset"
        )
        tokens_written, n, hit_budget = process_shard_file(
            local_path, text_field, enc, batch_size, out_f, budget, tokens_written
        )
        docs_written += n
        print(
            f"[{tag}]   ... {human(tokens_written)}/{human(budget)} tokens "
            f"({docs_written} docs so far)"
        )
        if hit_budget:
            return tokens_written, docs_written, True
    return tokens_written, docs_written, False


def download_via_stream(dataset_name, dataset_config, split, text_field, enc,
                         batch_size, out_f, budget, tag):
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    except Exception as exc:
        raise SystemExit(
            f"Failed to open dataset {dataset_name}/{dataset_config} "
            f"(split={split}): {exc}\nCheck --dataset/--dataset-config/--split, "
            f"network access to huggingface.co, and (for gated datasets) that "
            f"`huggingface-cli login` has been run."
        ) from exc

    tokens_written = 0
    docs_written = 0
    hit_budget = False
    buf = []
    for example in ds:
        text = example.get(text_field)
        if not text:
            continue
        buf.append(text)
        if len(buf) < batch_size:
            continue
        tokens_written, n, hit_budget = flush_batch(buf, out_f, enc, budget, tokens_written)
        docs_written += n
        buf = []
        print(
            f"[{tag}]   ... {human(tokens_written)}/{human(budget)} tokens "
            f"({docs_written} docs)",
            end="\r",
        )
        if hit_budget:
            break
    if not hit_budget and buf:
        tokens_written, n, _ = flush_batch(buf, out_f, enc, budget, tokens_written)
        docs_written += n
    print()
    return tokens_written, docs_written, hit_budget


for p in plans:
    out_path = Path(out_override) if out_override else repo_root / p["out_path"]
    out_path = out_path if out_path.is_absolute() else repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = get_encoding(p["tokenizer_name"])
    tag = Path(p["cfg_path"]).name

    if not force:
        have = existing_token_count(out_path, enc, batch_size)
        if have >= p["budget"]:
            print(
                f"[{tag}] {out_path} already has {human(have)} tokens "
                f"(need {human(p['budget'])}) — skipping. Use --force to "
                f"re-download."
            )
            continue

    print(
        f"[{tag}] downloading -> {out_path} (target {human(p['budget'])} "
        f"tokens from {dataset_name}/{dataset_config})"
    )

    file_glob = file_glob_override or guess_file_glob(dataset_config)
    use_shards = transfer_mode == "shards" or (transfer_mode == "auto" and file_glob)
    if transfer_mode == "shards" and not file_glob:
        raise SystemExit(
            "--transfer shards requires a resolvable shard glob; pass "
            "--file-glob explicitly for this dataset/config."
        )

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as out_f:
        if use_shards:
            try:
                tokens_written, docs_written, hit_budget = download_via_shards(
                    dataset_name, dataset_config, text_field, file_glob, enc,
                    batch_size, out_f, p["budget"], tag,
                )
            except Exception as exc:
                if transfer_mode == "shards":
                    raise SystemExit(
                        f"[{tag}] shard download failed: {exc}\n"
                        f"(--transfer shards was explicit, so not falling "
                        f"back to streaming -- rerun with --transfer auto "
                        f"to allow that, or fix --file-glob/--dataset-config.)"
                    ) from exc
                print(
                    f"[{tag}] shard mode failed ({exc}); falling back to "
                    f"row-by-row streaming."
                )
                out_f.seek(0)
                out_f.truncate()
                tokens_written, docs_written, hit_budget = download_via_stream(
                    dataset_name, dataset_config, split, text_field, enc,
                    batch_size, out_f, p["budget"], tag,
                )
        else:
            tokens_written, docs_written, hit_budget = download_via_stream(
                dataset_name, dataset_config, split, text_field, enc,
                batch_size, out_f, p["budget"], tag,
            )

    tmp_path.replace(out_path)
    print(
        f"[{tag}] done: {docs_written} docs, {human(tokens_written)} tokens "
        f"-> {out_path}"
    )
    if tokens_written < p["budget"]:
        print(
            f"  WARNING: {dataset_name}/{dataset_config}/{split} was exhausted "
            f"before reaching the budget ({human(tokens_written)} < "
            f"{human(p['budget'])}). Pick a larger --dataset-config or "
            f"--dataset, or let the trainer loop over this data for extra "
            f"epochs."
        )
PYEOF
