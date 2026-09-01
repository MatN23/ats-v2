# ats-v2

A config-driven LLM training framework built on PyTorch + DeepSpeed. One YAML
file controls model size, architecture (dense / SWA / MLA / MoE / MoD),
parallelism strategy, and training hyperparameters — no Python edits required
for standard runs.

This is a research project for small teams. It is not an alternative to
LLM Foundry, NeMo, or Megatron-LM, and does not target their scale or
hardware-fleet regime (see [Scale limitations](#scale-limitations-what-this-framework-does-and-doesnt-do-for-memory)).

## Status

This is the single source of truth for what has and hasn't actually been run.
Every other mention of a feature's status elsewhere in this file points back
here rather than repeating it.

**CI:** green on `main` — `ruff check`, `ruff format --check`, `mypy`, and
`pytest` (Python 3.10–3.12) all pass.

| Feature | Status |
|---|---|
| Dense, SWA, MLA, MTP, diffusion (`--model-type diffusion`) | **Trained end-to-end on Colab GPU runtimes** — each has completed a real training run |
| MoE, MoD, Mamba | Passes code review and unit tests. **Not yet trained/verified on any GPU.** Treat as less trustworthy than the row above until someone runs it |
| Population Based Training (`ats-breed`) | Covered by 35 passing unit tests (`tests/test_pbt.py`, `tests/test_cli_breeding.py`). No end-to-end GPU run recorded either way |
| Triton kernels (`ats/model/*_triton.py`) | **Status unconfirmed.** A Colab run used a GPU with the `[triton]` extra installed, but there's no Triton-specific log (compile/autotune output) confirming the kernels actually executed rather than silently falling back to the tested PyTorch path — which every kernel here does automatically, with no warning, if Triton isn't available or errors. Don't take "trained successfully with Triton installed" as proof the Triton code ran; see [Known limitations](#known-limitations) for how to check yourself |
| 8-bit Adam (`--optimizer-bits 8`) | CLI/config plumbing and the DeepSpeed client-optimizer wiring verified by code review; `bitsandbytes` itself not installed in the environment this was reviewed in, so the actual 8-bit optimizer math hasn't been run |
| Selective activation checkpointing (`--checkpoint-every-n-layers`) | Verified directly: fires exactly on layers where `layer_idx % n == 0`, and not at all when disabled or during incremental decoding (`use_cache=True`) |
| Mamba's chunked parallel scan | Verified numerically against a sequential-loop reference (exact match at float64; ~1e-7 relative error in float32 at realistic scale) — a correctness property checkable through arithmetic alone, independent of the "not yet trained" status above |
| `preprocess.py`'s streaming-to-disk | Verified with a 20,000-document scale test showing flat peak memory regardless of corpus size |
| `ats/cli/finetune.py` (LoRA) | Run end-to-end against a tiny model with a stubbed DeepSpeed engine: checkpoint load → LoRA injection → training loop → adapter save → merge → HuggingFace export. **Not** run against a real multi-GPU DeepSpeed engine or a non-trivial model size |
| `ats/cli/align.py` | Placeholder — parses arguments and prints "not implemented"; does not train anything |

The "verified" rows above (8-bit Adam plumbing, selective checkpointing, the
Mamba scan math, `preprocess.py` streaming) were checked by code review
and/or standalone logic tests without a GPU, `deepspeed`, or PyPI access —
narrower than an end-to-end `ats-train` run, which is why each is called out
individually instead of folded into a blanket "everything works" claim.

## Installation

```bash
pip install -e .
```

Installs ats-v2 (via `pyproject.toml`) and its dependencies (torch,
deepspeed, pydantic, tiktoken, transformers, safetensors, etc — see
`requirements.txt` for exact pins), plus seven console scripts:
`ats-train`, `ats-eval`, `ats-export`, `ats-doctor`, `ats-finetune`,
`ats-breed` (Population Based Training — see
[Population Based Training (`ats-breed`)](#population-based-training-ats-breed)), and the
not-yet-implemented `ats-align` placeholder. Optional extras:
`pip install -e ".[eval]"` for lm-evaluation-harness,
`pip install -e ".[triton]"` for the Triton kernels (GPU only),
`pip install -e ".[8bit]"` for bitsandbytes 8-bit Adam (`--optimizer-bits 8`),
`pip install -e ".[finetune]"` for `peft` (required by `ats-finetune`).

Check your environment before training:

```bash
ats-doctor
ats-doctor --config configs/7b.yaml   # also estimates memory for that config
```

## CLI reference

Every file under `ats/cli/`, what it does, and where to read more. Purposes
below are drawn directly from each file's own module docstring, not
inferred from its name.

| Module | Console script | Purpose |
|---|---|---|
| `ats.cli.train` | `ats-train` | Trains a model from a YAML config; every architecture/hyperparameter flag is described in [Training](#training) |
| `ats.cli.evaluate` | `ats-eval` | Benchmark tasks via lm-evaluation-harness (`--tasks`), or perplexity on your own data (`--config`) — see [Evaluate](#evaluate) |
| `ats.cli.export` | `ats-export` | Exports a checkpoint to HuggingFace format — see [Export to HuggingFace](#export-to-huggingface) |
| `ats.cli.doctor` | `ats-doctor` | Environment diagnostic: Python/PyTorch/CUDA/DeepSpeed/Flash-Attention/Triton versions and GPU count/memory, plus (with `--config`) an estimated memory report for that config. Every line comes from actually importing/inspecting the relevant package or device, not a hardcoded string |
| `ats.cli.finetune` | `ats-finetune` | LoRA fine-tunes a checkpoint via `peft` — see [LoRA fine-tuning](#lora-fine-tuning) |
| `ats.cli.breed` | `ats-breed` | Population Based Training — see [Population Based Training (`ats-breed`)](#population-based-training-ats-breed) |
| `ats.cli.align` | `ats-align` | Placeholder for RLHF/DPO-style alignment. **Not implemented** — running it prints exactly what's missing and exits non-zero, rather than silently no-op'ing |
| `ats.cli.test_modes` | *(none — run via `python -m ats.cli.test_modes [--steps 50] [--mode moe]`)* | Architecture *training* smoke test, distinct from `pytest tests/` (which unit-tests components like attention variants or MoE routing in isolation). For each of `dense, swa, mla, mamba, moe, mod, mtp, all`, builds a real ~1M-parameter `ATSTransformer` with that mode's flags actually enabled, runs 50 real optimizer steps (real forward/backward/AdamW step, synthetic random token data, no DeepSpeed, no mocks), and checks: finite loss every step, finite gradients every step, parameters actually moved by the end, and that the claimed feature is actually present in the constructed module graph. Answers "can this whole architecture configuration train end-to-end", which nothing else in the repo answers directly |

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

## Downloading training data

`scripts/download_data.sh` downloads exactly the amount of data a config
actually needs — no more, no less. It reads a config's `training.max_steps`,
`grad_accum_steps`, `micro_batch_size`, `data.seq_length`, and
`parallelism.gpus`/`nodes`, computes the exact token count that run will
consume, then fetches from a HuggingFace dataset (default: `fineweb-edu`),
tokenizing with the config's own `tiktoken` encoding (batched via
`encode_batch` for speed), and stops the instant that budget is hit:

```bash
./scripts/download_data.sh --all --dry-run       # print the token budget for every config, download nothing
./scripts/download_data.sh --config configs/125m.yaml
./scripts/download_data.sh --all --install-deps  # every configs/*.yaml, auto-installing missing deps
```

By default it downloads whole parquet shards (`--transfer shards`, via
`huggingface_hub.hf_hub_download`) rather than row-by-row streaming —
important for Xet/LFS-backed repos like fineweb-edu, where whole-file
downloads get HF's accelerated transfer path (and `HF_HUB_ENABLE_HF_TRANSFER`,
if you `pip install hf_transfer`) and small ranged HTTP reads don't. It only
fetches as many shards as needed to hit the budget, and skips shards it
already has cached locally on a rerun. Pass `--transfer stream` to force the
old row-by-row `datasets` streaming path (needed for datasets whose file
layout `--file-glob` can't guess), or `--transfer auto` (the default) to try
shards first and fall back to streaming automatically.

It writes straight to each config's `data.sources[0].path` in the
`{"text": ...}` jsonl format `ats/data/dataset.py` expects, skips
re-downloading if the destination already has enough tokens (checked with
the same tokenizer — pass `--force` to override), and warns if the source
dataset runs out before the budget is reached. Run
`./scripts/download_data.sh --help` for the full flag list (`--dataset`,
`--dataset-config`, `--split`, `--text-field`, `--transfer`, `--file-glob`,
`--batch-size`, `--margin`, `--out`).

## Scripts reference

Everything under `scripts/`, verified against each file's own header
comment/docstring:

| Script | Purpose |
|---|---|
| `scripts/download_data.sh` | Downloads exactly the token budget a config needs from a HuggingFace dataset — see [Downloading training data](#downloading-training-data) above |
| `scripts/launch.sh` | Thin wrapper around `torchrun` for single- or multi-node launches of `ats.cli.train`. Reads `NUM_NODES`, `GPUS_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`, `JOB_ID` env vars (all optional; default to single-node/single-process values) to build the `torchrun` rendezvous flags, then passes every other argument straight through to `ats.cli.train` — see [Multi-GPU with DeepSpeed](#multi-gpu-with-deepspeed) below |
| `scripts/slurm_submit.sh` | SLURM batch-job template — submit with `sbatch scripts/slurm_submit.sh`. Ships with `#SBATCH` directives for 2 nodes × 8 GPUs × 24h (edit for your job), derives `MASTER_ADDR`/`NUM_NODES`/`JOB_ID` from SLURM's own environment variables, and calls `scripts/launch.sh` via `srun`. The `ats-train` arguments hardcoded at the bottom of the file (`--config configs/7b.yaml --use-moe --use-mla` by default) are meant to be edited per job, not used as-is |
| `scripts/verify.py` | Run via `python scripts/verify.py`. A three-stage sanity check, in order, stopping at the first failure: (1) imports every module under `ats/`, catching import-time errors, circular imports, or syntax errors that slipped past review; (2) instantiates the major model classes with small dummy inputs and runs a real forward + backward pass, exercising actual model code rather than just checking classes exist; (3) runs `pytest tests/` as a subprocess. Prints `ALL CHECKS PASSED` and exits 0 if everything succeeds, or `FAILURES DETECTED` with the specific failures and exits 1 otherwise — every step's real exception (or pytest's real exit code) determines the result, nothing is swallowed to look green |

## Training

### A real-sized model

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

See [Status](#status) above for which of these have actually completed a
real training run.

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

### Multi-GPU with DeepSpeed

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

### Memory-saving flags: 8-bit optimizer and selective checkpointing

```bash
# bitsandbytes 8-bit Adam instead of fp32 AdamW: ~4x less optimizer-state
# memory, at a small numerical precision cost. Requires `pip install
# bitsandbytes` (or the `[8bit]` extra). See Status above for verification level.
python -m ats.cli.train --config configs/7b.yaml --optimizer-bits 8

# Activation checkpointing every Nth layer instead of every layer: trades
# less memory savings for less recompute. 1 = every layer (the strongest
# memory saving); omit/0 disables checkpointing entirely.
python -m ats.cli.train --config configs/7b.yaml --checkpoint-every-n-layers 1
python -m ats.cli.train --config configs/7b.yaml --checkpoint-every-n-layers 3
```

Both are also settable directly in a config's `optimizer.bits` and
`model.checkpoint_every_n_layers` fields. `--checkpoint-every-n-layers`
replaces the old boolean `--gradient-checkpointing` flag (still accepted as a
deprecated alias: `true`/unset maps to `1`/disabled). `ats-doctor --config`'s
memory estimate reflects both: 8-bit Adam roughly quarters the reported
optimizer-state memory, and the activation-memory reduction from
checkpointing follows the `reduction_factor = 1 + 2 / checkpoint_every_n_layers`
heuristic in `ats/utils/memory.py` — see
[Scale limitations](#scale-limitations-what-this-framework-does-and-doesnt-do-for-memory)
for exactly what that formula does and doesn't guarantee.

### MoE training example

```bash
python -m ats.cli.train --config configs/7b.yaml --use-moe --moe-num-experts 8 --moe-top-k 2
```

See [Status](#status) — MoE has not yet been trained end-to-end on real
hardware; treat a real run through this as the first verification of it.

### Checkpoint resume example

```bash
python -m ats.cli.train --config configs/1b.yaml --resume checkpoints/1b/step_5000
```

Resuming verifies the checkpoint's config hash matches the current config and
restores RNG state, optimizer state, and global step (`ats/training/checkpoint.py`).

### Population Based Training (`ats-breed`)

```bash
python -m ats.cli.breed --config configs/debug.yaml \
    --population-size 10 --generations 5 --steps-per-generation 100 \
    --cull-fraction 0.5 --output-dir ./pbt_runs
```

Trains `population_size` independent copies of the model side by side.
After every `steps_per_generation` steps, each member is evaluated on its
own held-out data (`config.data.sources`), the bottom `cull_fraction` are
culled, and each culled member's weights + a perturbed copy of a surviving
winner's hyperparameters take its place. By default this perturbs
`training.learning_rate`, `training.weight_decay`, and `model.dropout` —
not architecture fields, since perturbing those would break the
weights-only transplant between population members (mismatched parameter
shapes). See `ats/pbt/orchestrator.py`'s module docstring for the full
mechanics.

**Cost warning:** total compute is roughly `population_size` times a single
run of the same step count — only point this at small configs
(`debug.yaml`, `125m.yaml`, `350m.yaml`, ...) unless you deliberately want
to multiply an already-large job by `population_size`. This CLI does not
support resuming an interrupted breeding run across process restarts (a
fresh call always starts a new population at generation 0); each member's
own per-generation training does still checkpoint normally under
`--output-dir`. See [Status](#status) for verification level.

### Mamba / MTP / diffusion / quantization

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

See [Status](#status) for which of these have completed a real training run.

`--quantization fp8` requires `torchao` (`pip install torchao`) — it's the
only backend `ats/model/quantization.py::QuantizedLinear` actually
integrates with. If `transformer_engine` is installed instead, it raises a
clear `ImportError` rather than silently training at full precision while
appearing to use fp8: `transformer_engine`'s `fp8_autocast()` only affects
`transformer_engine.pytorch`'s own modules, not a plain `nn.Linear`, so
wrapping this class's forward pass in it would do nothing. `QuantizedLinear`
is exposed as a standalone building block but is **not** automatically
substituted for every `nn.Linear` in the backbone (attention, FFN, MoE
experts) — that wiring is a larger change than this revision includes;
today it's available for callers to use directly.

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
HuggingFace format at all (see [Export to HuggingFace](#export-to-huggingface)).

For perplexity on your own held-out data (`data.sources` in a config) instead
of a standard benchmark — including for MoE/MoD/MLA/Mamba/diffusion
checkpoints, which can't be exported — pass `--config` instead of `--tasks`:

```bash
python -m ats.cli.evaluate --config configs/1b.yaml --checkpoint checkpoints/1b/step_5000
```

## LoRA fine-tuning

```bash
python -m ats.cli.finetune --config configs/7b.yaml \
    --checkpoint checkpoints/7b/step_50000 \
    --lora-r 16 --lora-alpha 32 --target-modules q_proj,v_proj,o_proj \
    --output-dir ./lora-run
```

Requires `pip install peft` (or the `[finetune]` extra). `ats-finetune` loads
the base checkpoint's weights (freezing them), injects LoRA adapters via
`peft.LoraConfig`/`get_peft_model`, and reuses the same `Trainer` and
dataloader as `ats-train` — only the LoRA adapter parameters end up with
`requires_grad=True`, so the optimizer only ever updates those. It writes two
outputs under `--output-dir`: `lora_adapter/` (just the adapter weights, via
`peft`'s own `save_pretrained`) and `merged/` (the adapter merged back into
the base weights and exported through the same HuggingFace export path as
`ats-export`, so it's a standard, adapter-free checkpoint). Like
`ats-export`, only dense and SWA autoregressive checkpoints are supported —
MoE/MoD, MLA, Mamba, and diffusion checkpoints have no merged-export path and
are rejected immediately with a clear error rather than after a full run.

Defaults come from a config's `peft:` block (`enabled`, `lora_r`,
`lora_alpha`, `lora_dropout`, `target_modules`); the CLI flags above override
it the same way `ats-train`'s `--use-moe`-style flags override `model:`.

One rough edge, worked around rather than fixed at the root: `peft`'s
`merge_and_unload()` runs a tied-embeddings check that expects `model.config`
to be a dict-like HuggingFace `PretrainedConfig` (`model_config.get(...)`),
which ats-v2's own `ModelConfig` (a Pydantic model, no `.get()`) doesn't
satisfy. `ats/cli/finetune.py` temporarily swaps in a two-key dict shim
around that one call and restores the real config immediately after, rather
than changing `ATSTransformer.config`'s type everywhere else it's used.

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

## Scale limitations: what this framework does and doesn't do for memory

**ats-v2 targets dense/MoE models up to roughly 14B parameters on ZeRO-3
alone.** Several features that sound like they should reduce *training*
memory actually don't, and it's worth being explicit about which is which
rather than letting the feature names imply more than they deliver:

| Technique | In ats-v2? | Training memory impact | Why |
|---|---|---|---|
| ZeRO-3 | Yes | High | Shards params + optimizer + gradients across GPUs |
| Gradient checkpointing | Yes | High (see formula below) | Real, but see the caveat below |
| Flash Attention | Yes (falls back to SDPA) | Medium | Saves activation memory vs. standard attention |
| Sequence packing | Yes | Low-Medium | Only for preprocessed `.bin` data |
| 8-bit Adam (bitsandbytes) | Yes (`--optimizer-bits 8`) | High (~4x less optimizer-state memory) | See [Status](#status) for verification level |
| **Mixture-of-Depths (MoD)** | Yes | **None** | The gate is applied *after* the block computes on every token — see below |
| **Sliding Window Attention (SWA)** | Yes | **None** | Full Q/K/V are still materialized for the whole sequence during training; SWA only shrinks the *inference* KV cache |
| **Int8 quantization** | Yes | **None** | `torch.ao`'s fake-quantization keeps weights in bf16/fp16 throughout; it simulates QAT numerics, it doesn't reduce memory |
| **FP8 quantization** | `QuantizedLinear` exists (torchao backend only) | **None, as shipped** | Not wired into attention/FFN/MoE/MLA by default — see [Quantization](#mamba--mtp--diffusion--quantization) above |
| Mamba (chunked scan) | Yes | N/A (speed, not memory) | O(seq_len/chunk_size) sequential steps, not O(seq_len) — see below |
| Tensor Parallelism | **No** | Critical for 70B | Not implemented — see below |
| Pipeline Parallelism | **No** | Critical for 70B | Not implemented — see below |
| ZeRO-Offload (CPU offload) | **No** | High | Not implemented |

**MoD in detail:** `ats/model/mod.py`'s gate decides which tokens' outputs
get *used*, but `self.block(x, ...)` still runs on the full sequence first —
the mask is applied to the result, not used to skip computation. This makes
MoD here a regularizer (via its load-balancing aux loss) and, if you build
inference-time gather/scatter around it yourself, a decode-time speedup —
but it is not a training-time compute or memory optimization as currently
implemented. Doing that properly means gathering only the selected tokens
*before* running the block and scattering the result back, which interacts
non-trivially with gradient checkpointing and DeepSpeed's ZeRO sharding;
that rewrite isn't attempted here rather than risk an under-tested version
of it.

**Gradient checkpointing formula:** `ats/utils/memory.py`'s pre-flight
estimator uses `reduction_factor = 1 + 2 / checkpoint_every_n_layers` for
activation memory: `checkpoint_every_n_layers=1` (checkpoint every layer)
gives a 3x reduction, based on commonly-reported practical figures for
full (every-layer) checkpointing — not a precise theoretical bound.
`checkpoint_every_n_layers > 1` (checkpoint every Nth layer) scales that
reduction down toward 1x (no savings) as `n` grows, since
DeepSpeed/`torch.utils.checkpoint` only trades recompute for memory on the
layers actually checkpointed. This is still a simple heuristic, not the
theoretical O(sqrt(num_layers)) bound from Chen et al. 2016 (that bound
assumes checkpointing exactly every sqrt(num_layers)-th layer specifically,
not an arbitrary N). Treat the estimator's numbers as a rough pre-flight
warning, not an exact prediction.

**No Tensor or Pipeline Parallelism:** the only parallelism strategies here
are ZeRO-0 through ZeRO-3 (data-parallel-with-sharding) and DeepSpeed's MoE
expert parallelism. For genuinely large (~70B+) dense models, ZeRO-3 alone
means every forward pass all-gathers the full parameter set across every
GPU in the job — at that scale the communication volume becomes the
bottleneck, which is exactly why frameworks built for that regime (Megatron-
LM, NeMo) combine tensor and pipeline parallelism with data parallelism.
**This is a deliberate scope boundary, not an oversight:** ats-v2 is meant
for the sub-~14B regime where ZeRO-3 is sufficient on its own. Models larger
than that are intended to be handled by a separate wrapper (planned, not
part of this repository) that would plug into ats-v2's config/checkpoint/
data interfaces rather than ats-v2 reimplementing Megatron-style 3D
parallelism itself. Unlike the Mamba scan or the gradient-checkpointing
formula above — both correctness properties that could be verified through
careful numerical reasoning without a GPU — tensor/pipeline parallelism's
correctness fundamentally depends on real multi-GPU collective
communication (NCCL all-reduce/all-gather/scatter across process groups,
pipeline bubble scheduling). There's no way to establish confidence in that
kind of implementation through arithmetic verification the way the fixes
above were checked; attempting it without hardware to actually run it on
would trade a disclosed gap for undisclosed, hard-to-detect correctness bugs
in distributed training, which is a worse outcome.

**Mamba uses a chunked parallel scan, not a Python loop over every
timestep:** `ats/model/mamba.py`'s selective scan solves the recurrence in
chunks of `mamba_chunk_size` (default 32) positions via a batched matmul
against a log-space lower-triangular decay matrix, dropping sequential
Python-level steps from O(seq_len) to O(seq_len / chunk_size). This is
mathematically exact (not an approximation) — see [Status](#status) for how
it was verified. `chunk_size` trades memory for speed: the per-chunk decay
tensor is `[batch, chunk_size, chunk_size, d_inner, d_state]`, so larger
chunks mean fewer sequential steps but quadratically more peak memory per
chunk — reduce `mamba_chunk_size` if you hit OOM specifically on this
tensor. Mamba layers still don't support KV-cache-based incremental
decoding (see [Known limitations](#known-limitations)) — that's a separate,
unrelated limitation from the scan algorithm.

## Known limitations

- **Mamba layers do not support KV-cache-based incremental decoding** in
  this reference implementation — the chunked scan recomputes over the
  full sequence each call. Fine for training; not yet wired for
  autoregressive generation with caching.
- **MoE/MoD/MLA/Mamba/diffusion models cannot be exported to HuggingFace
  format** — `ats/export/huggingface.py` raises a clear `ConfigError` for
  each rather than emitting a checkpoint that would silently load with the
  wrong architecture. Only dense and SWA models export today (see
  [Export to HuggingFace](#export-to-huggingface)).
- **Triton kernels status is unconfirmed** (see [Status](#status)). Each
  kernel is gated behind `HAS_TRITON` and falls back silently (no warning)
  to a plain PyTorch implementation that *is* tested — a real safety net,
  but one that also means a training run completing successfully on a GPU
  with Triton installed proves nothing about whether the Triton code
  actually ran. To check on your own GPU: temporarily force the
  `HAS_TRITON` gate to skip the fallback (so a broken kernel raises instead
  of silently substituting) and confirm training still runs, or add a
  print/log inside the Triton branch and check it fires. Two of the four
  (MoE routing dispatch, MLA KV decompression) are also only *partially*
  fused, by design — see the docstring in each file for exactly what is
  and isn't fused, rather than taking "Triton kernel" to mean the whole
  pipeline is.
- A separate, genuinely memory-reducing feature — post-training
  quantization for *inference* (storing real int8 weights in an exported
  checkpoint, no training involved) — is not implemented. This would be a
  different feature from what `model.quantization`'s int8 QAT path
  currently does (see [Quantization](#mamba--mtp--diffusion--quantization)
  above); it's a reasonable, lower-risk addition if useful, not an
  unfinished version of the existing path.
