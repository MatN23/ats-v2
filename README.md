# ats-v2

A config-driven LLM training framework built on PyTorch + DeepSpeed. One YAML
file controls model size, architecture (dense / SWA / MLA / MoE / MoD),
parallelism strategy, and training hyperparameters — no Python edits required
for standard runs.

> **Status note:** this repository was written and reviewed by eye in a
> sandboxed environment without network access, so most of it has **not**
> been executed end-to-end here (no `pytest`, no real training run, no
> `pip install -e .` — the sandbox can't reach PyPI to install torch,
> deepspeed, or pydantic). A few pieces *were* actually run and verified in
> this sandbox specifically because they don't require those packages: the
> `ats-doctor` command was executed directly and correctly detected this
> sandbox's real (missing) PyTorch/DeepSpeed/Triton/GPU state; the core
> sequence-packing + memmap read/write logic used by `preprocess.py` and the
> preprocessed-data reader was run standalone and round-tripped correctly.
> Everything else — training, the Triton kernels in particular — is
> unverified. Run the verification commands below yourself before relying
> on this.

## Installation

```bash
pip install -e .
```

Installs ats-v2 (via `pyproject.toml`) and its dependencies (torch,
deepspeed, pydantic, tiktoken, transformers, safetensors, etc — see
`requirements.txt` for exact pins), plus five console scripts:
`ats-train`, `ats-eval`, `ats-export`, `ats-doctor`, and the not-yet-
implemented `ats-finetune`/`ats-align` placeholders. Optional extras:
`pip install -e ".[eval]"` for lm-evaluation-harness,
`pip install -e ".[triton]"` for the Triton kernels (GPU only).

Check your environment before training:

```bash
ats-doctor
ats-doctor --config configs/7b.yaml   # also estimates memory for that config
```

## Quickstart: train a tiny debug model in 3 commands

```bash
mkdir -p data
python -c "
import json
with open('data/debug.jsonl', 'w') as f:
    for i in range(200):
        f.write(json.dumps({'text': 'the quick brown fox jumps over the lazy dog ' * 5}) + chr(10))
"
python -m ats.cli.train --config configs/debug.yaml
```

This runs 100 steps of a ~14M parameter model on CPU (ZeRO-0, single
process) and writes checkpoints to `./checkpoints/debug`.

## Train a real-sized model

```bash
python -m ats.cli.train --config configs/1b.yaml
python -m ats.cli.train --config configs/7b.yaml
```

Architecture size (hidden_size, num_layers, num_heads, ...) is auto-filled
from `model.size` in the YAML via published-recipe presets in
`ats/config/defaults.py`. There is **one config per size**; every file ships
dense by default. All optional architecture features are enabled from the
command line, not by hand-writing more YAML files:

```bash
python -m ats.cli.train --config configs/7b.yaml                     # dense
python -m ats.cli.train --config configs/7b.yaml --use-swa            # sliding window attention
python -m ats.cli.train --config configs/7b.yaml --use-mla            # multi-head latent attention
python -m ats.cli.train --config configs/7b.yaml --use-moe --use-mod  # MoE + Mixture-of-Depths
python -m ats.cli.train --config configs/7b.yaml --architecture all   # every compatible feature at once
python -m ats.cli.train --config configs/debug.yaml --use-mamba --mamba-every-n-layers 2
python -m ats.cli.train --config configs/debug.yaml --model-type diffusion
```

`--architecture {dense,swa,mla,mamba,moe,mod,mtp,all}` is a convenience
preset that flips several `--use-x` flags at once; any individual
`--use-x`/`--no-use-x` you also pass on the same command line overrides the
preset for that one flag. Every flag actually mutates the loaded config
before the model is constructed (see `apply_cli_overrides` in `train.py`),
and the merged result is re-validated through the same Pydantic schema used
for YAML — so an invalid combination (e.g. `--num-heads 5` against a
`num_kv_heads` that doesn't divide it, or `--use-mtp --model-type diffusion`)
fails loudly with the same actionable error messages as a bad YAML file.

Numeric architecture fields, model-size fields, training hyperparameters,
data settings, and parallelism settings are all separately overridable; run
`python -m ats.cli.train --help` for the full flag list.

## Multi-GPU example with DeepSpeed

```bash
deepspeed --num_gpus 8 -m ats.cli.train --config configs/7b.yaml
```

`parallelism.strategy: auto` in the config resolves to a ZeRO stage based on
GPU count and estimated parameter count (see `ats/parallelism/auto_parallel.py`);
override explicitly with `parallelism.strategy: deepspeed_zero3` if needed.

For multi-node runs, `scripts/launch.sh` wraps `torchrun` with the right
rendezvous flags, and `scripts/slurm_submit.sh` is a SLURM template that
calls it via `srun`:

```bash
NUM_NODES=1 GPUS_PER_NODE=8 scripts/launch.sh --config configs/7b.yaml --use-moe
# or, on a SLURM cluster:
sbatch scripts/slurm_submit.sh
```

## Offline preprocessing

For large corpora, tokenize once and read via memory-mapped files instead of
tokenizing on the fly every epoch:

```bash
python preprocess.py --input data.jsonl --output-dir ./preprocessed \
  --tokenizer cl100k_base --seq-length 4096 --packing
```

`--packing` concatenates documents (EOS-delimited) into full `seq_length`
blocks instead of one block per document, eliminating most padding waste for
corpora of short documents. Point `data.sources[*].path` at the resulting
`preprocessed/tokens.bin` in your config; `MixedDataset` detects `.bin`
sources automatically and reads them via `numpy.memmap`, with no
on-the-fly tokenization.

## MoE training example

```bash
python -m ats.cli.train --config configs/7b.yaml --use-moe --moe-num-experts 8 --moe-top-k 2
```

## Checkpoint resume example

```bash
python -m ats.cli.train --config configs/1b.yaml --resume checkpoints/1b/step_5000
```

Resuming verifies the checkpoint's config hash matches the current config and
restores RNG state, optimizer state, and global step.

## Evaluate

Standard benchmarks (MMLU, HellaSwag, ARC, ...) are delegated to
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
not reimplemented here. `ats.cli.evaluate` auto-exports the checkpoint to
HuggingFace format first (reusing the export path, cached under
`<checkpoint>/hf_exported/` so it only happens once), then shells out to
`python -m lm_eval`:

```bash
python -m ats.cli.evaluate --checkpoint checkpoints/1b/step_5000 --tasks mmlu,hellaswag,arc_easy
```

This mode requires `pip install lm-eval` (or the `[eval]` extra) and only
works for dense/SWA autoregressive checkpoints, since only those export to
HuggingFace format at all.

For perplexity on your own held-out data (`data.sources` in a config) instead
of a standard benchmark — including for MoE/MoD/MLA/Mamba/diffusion
checkpoints, which can't be exported — pass `--config` instead of `--tasks`:

```bash
python -m ats.cli.evaluate --config configs/1b.yaml --checkpoint checkpoints/1b/step_5000
```

## Export to HuggingFace

```bash
python -m ats.cli.export --checkpoint checkpoints/1b/step_5000 --output_dir ./exported --config configs/1b.yaml
```

Dense and SWA models export to a `LlamaForCausalLM`-compatible checkpoint
(SWA models set HF's `sliding_window` field, matching Mistral's convention).
MoE, MoD, and MLA models raise a clear error instead of producing a
checkpoint that would silently load wrong — those architectures have no
HuggingFace `Llama` equivalent.

## Running tests

```bash
pytest tests/
```

## Mamba / MTP / diffusion / quantization

```bash
# Replace every 4th block with a Mamba selective-SSM block (pure PyTorch, no custom CUDA):
python -m ats.cli.train --config configs/7b.yaml --use-mamba --mamba-every-n-layers 4

# Predict 3 future tokens in parallel instead of 1:
python -m ats.cli.train --config configs/7b.yaml --use-mtp --mtp-num-tokens 3

# Train a diffusion LM (cosine noise schedule, MSE noise-prediction objective,
# DDIM sampling) instead of an autoregressive one:
python -m ats.cli.train --config configs/debug.yaml --model-type diffusion

# int8 quantization-aware training via torch.ao fake-quantization:
python -m ats.cli.train --config configs/7b.yaml --quantization int8
```

`--quantization fp8` requires `transformer-engine` or `torchao` to be
installed; if neither is present it raises `ImportError` immediately rather
than silently training in bf16, per this project's design principles.
`ats/model/quantization.py::QuantizedLinear` is exposed as a building block
but is not yet automatically substituted for every `nn.Linear` in the
backbone — wiring that through every module (attention, FFN, MoE experts) is
a larger change than this revision includes; today it's available for
callers to use directly.

## Known limitations

- **Mamba layers do not support KV-cache-based incremental decoding** in
  this reference implementation — the sequential scan recomputes over the
  full sequence each call. Fine for training; not yet wired for
  autoregressive generation with caching.
- **MoE/MoD/MLA/Mamba/diffusion models cannot be exported to HuggingFace
  format** — `ats/export/huggingface.py` raises a clear `ConfigError` for
  each rather than emitting a checkpoint that would silently load with the
  wrong architecture. Only dense and SWA models (both Llama/Mistral-family
  compatible) export today, which also means `ats-eval`'s lm-eval-harness
  path only works for those architectures; use `--config` (perplexity mode)
  for the others.
- **Triton kernels (`ats/model/*_triton.py`) are unverified on real
  hardware.** They were written without access to a GPU or a Triton
  installation to compile, run, or benchmark them. Each one is gated behind
  `HAS_TRITON` and falls back to a plain PyTorch implementation that *is*
  tested, so a missing/broken Triton install never crashes anything — but
  the Triton code paths themselves have not been proven correct by
  execution, only by careful review. Two of the four (MoE routing dispatch,
  MLA KV decompression) are also only *partially* fused, by design — see the
  docstring in each file for exactly what is and isn't fused, rather than
  taking "Triton kernel" to mean the whole pipeline is.
- `ats/cli/finetune.py` and `ats/cli/align.py` are placeholder structure —
  they parse arguments and print a clear "not implemented" message, they do
  not train anything.
- The `preprocess.py` implementation tokenizes the input file fully into
  memory before writing, which won't scale to very large corpora; a
  streaming two-pass (count blocks, then write) version would be needed for
  that.
- This repository was written and reviewed by eye in a sandboxed environment
  without network access, so most of it **has not been executed here** — no
  `pytest`, no real training run, no `pip install -e .` (the sandbox can't
  reach PyPI). A few package-free pieces *were* actually run and verified —
  see the status note at the top of this file for exactly which ones. Run
  the full verification commands below yourself before relying on this.
