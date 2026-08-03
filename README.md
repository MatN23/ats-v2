# ats-v2

A config-driven LLM training framework built on PyTorch + DeepSpeed. One YAML
file controls model size, architecture (dense / SWA / MLA / MoE / MoD),
parallelism strategy, and training hyperparameters — no Python edits required
for standard runs.

> **Status note:** this repository was written and reviewed by eye in a
> sandboxed environment without network access, so it has **not** been
> executed end-to-end here. Before relying on it, run the verification
> commands below yourself in an environment with `pip` access and (for the
> DeepSpeed/multi-GPU paths) real GPUs.

## Installation

```bash
pip install -e .
```

This installs ats-v2 and everything pinned in `requirements.txt` (torch,
deepspeed, pydantic, tiktoken, transformers, safetensors, etc).

## Quickstart: train a tiny debug model in 3 commands

```bash
mkdir -p data
python -c "
import json
with open('data/debug.jsonl', 'w') as f:
    for i in range(200):
        f.write(json.dumps({'text': 'the quick brown fox jumps over the lazy dog ' * 5}) + chr(10))
"
python train.py --config configs/debug.yaml
```

This runs 100 steps of a ~14M parameter model on CPU (ZeRO-0, single
process) and writes checkpoints to `./checkpoints/debug`.

## Train a real-sized model

```bash
python train.py --config configs/1b.yaml
python train.py --config configs/7b.yaml
```

Architecture size (hidden_size, num_layers, num_heads, ...) is auto-filled
from `model.size` in the YAML via published-recipe presets in
`ats/config/defaults.py`. There is **one config per size**; every file ships
dense by default. All optional architecture features are enabled from the
command line, not by hand-writing more YAML files:

```bash
python train.py --config configs/7b.yaml                     # dense
python train.py --config configs/7b.yaml --use-swa            # sliding window attention
python train.py --config configs/7b.yaml --use-mla            # multi-head latent attention
python train.py --config configs/7b.yaml --use-moe --use-mod  # MoE + Mixture-of-Depths
python train.py --config configs/7b.yaml --architecture all   # every compatible feature at once
python train.py --config configs/debug.yaml --use-mamba --mamba-every-n-layers 2
python train.py --config configs/debug.yaml --model-type diffusion
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
`python train.py --help` for the full flag list.

## Multi-GPU example with DeepSpeed

```bash
deepspeed --num_gpus 8 train.py --config configs/7b.yaml
```

`parallelism.strategy: auto` in the config resolves to a ZeRO stage based on
GPU count and estimated parameter count (see `ats/parallelism/auto_parallel.py`);
override explicitly with `parallelism.strategy: deepspeed_zero3` if needed.

## MoE training example

```bash
python train.py --config configs/7b.yaml --use-moe --moe-num-experts 8 --moe-top-k 2
```

## Checkpoint resume example

```bash
python train.py --config configs/1b.yaml --resume checkpoints/1b/step_5000
```

Resuming verifies the checkpoint's config hash matches the current config and
restores RNG state, optimizer state, and global step.

## Evaluate

```bash
python evaluate.py --config configs/1b.yaml --checkpoint checkpoints/1b/step_5000 --tasks mmlu
```

Omit `--tasks` to compute perplexity on `data.sources` instead.

## Export to HuggingFace

```bash
python export.py --checkpoint checkpoints/1b/step_5000 --output_dir ./exported --config configs/1b.yaml
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
python train.py --config configs/7b.yaml --use-mamba --mamba-every-n-layers 4

# Predict 3 future tokens in parallel instead of 1:
python train.py --config configs/7b.yaml --use-mtp --mtp-num-tokens 3

# Train a diffusion LM (cosine noise schedule, MSE noise-prediction objective,
# DDIM sampling) instead of an autoregressive one:
python train.py --config configs/debug.yaml --model-type diffusion

# int8 quantization-aware training via torch.ao fake-quantization:
python train.py --config configs/7b.yaml --quantization int8
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
  compatible) export today.
- This repository was written and reviewed by eye in a sandboxed environment
  without network access, so **none of it has been executed here** — no
  `pytest`, no real training run, no `pip install -e .`. Run the
  verification commands below yourself before relying on this.
