# ats-v2

A config-driven LLM training framework built on PyTorch + DeepSpeed. One YAML
file controls model size, architecture (dense / SWA / MLA / MoE / MoD),
parallelism strategy, and training hyperparameters — no Python edits required
for standard runs.

This is a research project for small teams. It is not an alternative to
LLM Foundry, NeMo, or Megatron-LM, and does not target their scale or
hardware-fleet regime (see "Scale limitations" below).

## Status

- **CI:** green on `main` — `ruff check`, `ruff format --check`, `mypy`, and
  `pytest` (Python 3.10–3.12) all pass.
- **Trained end-to-end on Colab GPU runtimes:** dense, SWA, MLA, MTP, and
  diffusion (`--model-type diffusion`) have each completed successful
  training runs.
- **Not yet trained/verified on any GPU:** MoE, MoD, and Mamba. The code
  passes review and unit tests, but nobody has run an actual training loop
  through these architectures yet. Treat them as less trustworthy than the
  five modes above until someone does.
- **Triton kernels (`ats/model/*_triton.py`): status unconfirmed.** A Colab
  run used a GPU with the `[triton]` extra installed, but there's no
  Triton-specific log (compile/autotune output) confirming the kernels
  actually executed rather than silently falling back to the tested
  PyTorch path — which every kernel here does automatically, with no
  warning, if Triton isn't available or errors. Don't take "trained
  successfully on a GPU with Triton installed" as proof the Triton code
  itself ran; see [Known limitations](#known-limitations) for how to check.

## Installation

```bash
pip install -e .
```

Installs ats-v2 (via `pyproject.toml`) and its dependencies (torch,
deepspeed, pydantic, tiktoken, transformers, safetensors, etc — see
`requirements.txt` for exact pins), plus five console scripts:
`ats-train`, `ats-eval`, `ats-export`, `ats-doctor`, `ats-finetune`, and the
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

## Memory-saving flags: 8-bit optimizer and selective checkpointing

```bash
# bitsandbytes 8-bit Adam instead of fp32 AdamW: ~4x less optimizer-state
# memory, at a small numerical precision cost. Requires `pip install
# bitsandbytes` (or the `[8bit]` extra).
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
checkpointing scales down from ~3x at `checkpoint_every_n_layers=1` toward 1x
(no reduction) as `n` grows, since fewer layers get recomputed.

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

> MoE has not yet been trained end-to-end on real hardware (see
> [Status](#status)) — the CLI/config path works and is unit-tested, but
> treat a real run through this as the first verification of it, not a
> repeat of one that's already happened.

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
# NOTE: Mamba has not yet completed a real training run (see Status above) —
# the scan math is verified numerically against a sequential reference, but
# that's not the same as this path having actually been trained.
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

## Scale limitations: what this framework does and doesn't do for memory

**ats-v2 targets dense/MoE models up to roughly 14B parameters on ZeRO-3
alone.** Several features that sound like they should reduce *training*
memory actually don't, and it's worth being explicit about which is which
rather than letting the feature names imply more than they deliver:

| Technique | In ats-v2? | Training memory impact | Why |
|---|---|---|---|
| ZeRO-3 | Yes | High | Shards params + optimizer + gradients across GPUs |
| Gradient checkpointing | Yes | High (~2-4x) | Real, but see the caveat below |
| Flash Attention | Yes (falls back to SDPA) | Medium | Saves activation memory vs. standard attention |
| Sequence packing | Yes | Low-Medium | Only for preprocessed `.bin` data |
| **Mixture-of-Depths (MoD)** | Yes | **None** | The gate is applied *after* the block computes on every token — see below |
| **Sliding Window Attention (SWA)** | Yes | **None** | Full Q/K/V are still materialized for the whole sequence during training; SWA only shrinks the *inference* KV cache |
| **Int8 quantization** | Yes | **None** | `torch.ao`'s fake-quantization keeps weights in bf16/fp16 throughout; it simulates QAT numerics, it doesn't reduce memory |
| **FP8 quantization** | Yes | High, if used | `QuantizedLinear` is wired into attention/FFN/MoE-expert/MLA projections (see `model.quantization` in configs) but requires `transformer-engine` or `torchao` installed |
| Mamba (chunked scan) | Yes | N/A (speed, not memory) | O(seq_len/chunk_size) sequential steps, not O(seq_len) — see below |
| Tensor Parallelism | **No** | Critical for 70B | Not implemented — see below |
| Pipeline Parallelism | **No** | Critical for 70B | Not implemented — see below |
| 8-bit optimizers (bitsandbytes) | **No** | High | Not implemented |
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

**Selective checkpointing formula:** `ats/utils/memory.py`'s pre-flight
estimator uses a `reduction_factor = 1 + 2 / checkpoint_every_n_layers`
heuristic for activation memory: `checkpoint_every_n_layers=1` (checkpoint
every layer) gives the same constant ~3x reduction the old boolean
`gradient_checkpointing=True` flag used, based on commonly-reported
practical figures for full (every-layer) checkpointing — not a precise
theoretical bound. `checkpoint_every_n_layers > 1` (checkpoint every Nth
layer) scales that reduction down toward 1x (no savings) as `n` grows, since
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
parallelism itself. Unlike the Mamba scan or the memory-formula fix above —
both correctness properties that could be verified through careful numerical
reasoning without a GPU — tensor/pipeline parallelism's correctness
fundamentally depends on real multi-GPU collective communication (NCCL
all-reduce/all-gather/scatter across process groups, pipeline bubble
scheduling). There's no way to establish confidence in that kind of
implementation through arithmetic verification the way the fixes above were
checked; attempting it without hardware to actually run it on would trade a
disclosed gap for undisclosed, hard-to-detect correctness bugs in
distributed training, which is a worse outcome. Given you've already said
you're building this as a separate Megatron-based wrapper, that's also the
right place for it.

**Int8 "quantization-aware training" not saving training memory is by
design, not an unfinished fix:** `QuantizedLinear`'s int8 path
(`torch.ao.quantization.FakeQuantize`) exists specifically to simulate int8
rounding numerics during training via a straight-through estimator, while
keeping weights in bf16/fp16 so gradients can flow — that's what QAT means.
Making int8 training actually reduce memory would mean a different
technique entirely (storing and updating genuinely low-precision weights
with specialized gradient handling, e.g. what dedicated 8-bit-optimizer
libraries implement), not a bug fix to the QAT path that's already here. A
separate, genuinely memory-reducing feature — post-training quantization for
*inference* (storing real int8 weights in an exported checkpoint, no
training involved) — is not implemented and would be a reasonable, lower-risk
addition if useful; it's a different feature from what `model.quantization`
currently does.

**Mamba uses a chunked parallel scan, not a Python loop over every
timestep:** `ats/model/mamba.py`'s selective scan solves the recurrence in
chunks of `mamba_chunk_size` (default 32) positions via a batched matmul
against a log-space lower-triangular decay matrix, dropping sequential
Python-level steps from O(seq_len) to O(seq_len / chunk_size). This is
mathematically exact (not an approximation) — verified numerically against
a plain sequential-loop reference at both small scale (exact match to
float64 precision) and realistic scale (seq_len=4096, extreme decay-rate
range, ~1e-7 relative error in float32) before being written, and the
shipped code has its own regression test comparing against a sequential
reference built from the same intermediate tensors. `chunk_size` trades
memory for speed: the per-chunk decay tensor is
`[batch, chunk_size, chunk_size, d_inner, d_state]`, so larger chunks mean
fewer sequential steps but quadratically more peak memory per chunk —
reduce `mamba_chunk_size` if you hit OOM specifically on this tensor.
Mamba layers still don't support KV-cache-based incremental decoding (see
Known limitations below) — that's a separate, unrelated limitation from the
scan algorithm.

**`preprocess.py` streams directly to disk** (writes and discards each
block as it's produced) rather than accumulating the tokenized corpus in
memory — verified with a 20,000-document scale test showing flat peak
memory regardless of corpus size. It still tokenizes with a single Python
process, so very large corpora will be throughput-bound by that, but won't
run out of RAM.

## Known limitations

- **Mamba layers do not support KV-cache-based incremental decoding** in
  this reference implementation — the chunked scan recomputes over the
  full sequence each call. Fine for training; not yet wired for
  autoregressive generation with caching.
- **MoE/MoD/MLA/Mamba/diffusion models cannot be exported to HuggingFace
  format** — `ats/export/huggingface.py` raises a clear `ConfigError` for
  each rather than emitting a checkpoint that would silently load with the
  wrong architecture. Only dense and SWA models (both Llama/Mistral-family
  compatible) export today, which also means `ats-eval`'s lm-eval-harness
  path only works for those architectures; use `--config` (perplexity mode)
  for the others.
- **Triton kernels (`ats/model/*_triton.py`) status is unconfirmed, not
  confirmed-working.** Each one is gated behind `HAS_TRITON` and falls back
  silently (no warning) to a plain PyTorch implementation that *is* tested —
  a real safety net, but one that also means a training run completing
  successfully on a GPU with Triton installed proves nothing about whether
  the Triton code actually ran. If you want to check on your own GPU:
  temporarily force the `HAS_TRITON` gate to skip the fallback (so a broken
  kernel raises instead of silently substituting) and confirm training
  still runs, or add a print/log inside the Triton branch and check it fires.
  Two of the four (MoE routing dispatch, MLA KV decompression) are also only
  *partially* fused, by design — see the docstring in each file for exactly
  what is and isn't fused, rather than taking "Triton kernel" to mean the
  whole pipeline is.
- `ats/cli/align.py` is placeholder structure — it parses arguments and
  prints a clear "not implemented" message, it does not train anything.
  `ats/cli/finetune.py` (LoRA fine-tuning via `peft`) is implemented and was
  run end-to-end in this sandbox against a tiny model with a stubbed-out
  DeepSpeed engine (real `deepspeed` isn't installable here either): base
  checkpoint load → LoRA injection → a short training loop → adapter save →
  merge → HuggingFace export, producing a valid `LlamaForCausalLM.
  from_pretrained`-loadable checkpoint. It has **not** been run against a
  real multi-GPU DeepSpeed engine or a non-trivial model size. One rough
  edge found via that testing and worked around: `peft`'s
  `merge_and_unload()` runs a tied-embeddings check that expects
  `model.config` to be a dict-like HuggingFace `PretrainedConfig`
  (`model_config.get("tie_word_embeddings")`), which `ats-v2`'s own
  `ModelConfig` (a Pydantic model, no `.get()`) doesn't satisfy;
  `ats/cli/finetune.py` temporarily swaps in a two-key dict shim around that
  one call and restores the real config immediately after, rather than
  changing `ATSTransformer.config`'s type everywhere else it's used.
- 8-bit Adam (`--optimizer-bits 8`) was verified for CLI/config plumbing and
  the DeepSpeed client-optimizer wiring (`initialize_engine` passing a
  constructed `bitsandbytes.optim.Adam8bit` via `deepspeed.initialize
  (optimizer=...)` instead of the JSON `optimizer` block); the `bitsandbytes`
  package itself isn't installed in this sandbox, so the actual 8-bit
  optimizer math has not been run.
- Selective activation checkpointing (`checkpoint_every_n_layers`) was
  verified directly: `torch.utils.checkpoint.checkpoint()` fires exactly on
  layers where `layer_idx % n == 0` during training, and not at all when
  disabled or when `use_cache=True` (incremental decoding).
- The bullets above describing specific things as "verified" (8-bit Adam
  plumbing, selective checkpointing, the Mamba chunked-scan math,
  `preprocess.py` streaming) were checked by code review and/or standalone
  logic tests in a sandbox without a GPU, `deepspeed`, or PyPI access — not
  by running `ats-train` itself. That's a narrower claim than an end-to-end
  training run and is called out per-bullet above rather than implied by a
  blanket statement. See [Status](#status) at the top of this file for
  which architectures have actually completed real training runs, and which
  (MoE, MoD, Mamba, and the Triton kernels) have not yet.