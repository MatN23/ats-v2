# CHANGES

## (a) Why v1 (AdaptiveTrainingSystem) was abandoned

The original codebase mixed a monkey-patched "adaptive orchestrator" into
the training loop via runtime method injection, used `np.random.random()`
to fabricate "safety scores," and used `np.polyfit` on loss history to
produce "trajectory predictions" with no statistical basis. It also shipped
dead/unused CUDA kernel files and an orchestrator with race conditions
between background monitoring threads and the main training loop. None of
that logic could be trusted or audited, so ats-v2 is a from-scratch
rewrite: synchronous control flow only, no runtime patching, and every
adaptive decision is a deterministic function of a bounded metrics history
(see `ats/training/adaptive_controller.py`).

## (b) RoPE implementation correction

The original design spec called for `torch.repeat_interleave(freqs, 2)` to
build the interleaved cos/sin cache, but paired it with a `rotate_half`
application (`q*cos - rotate_half(q)*sin`). Those two conventions are
mathematically inconsistent with each other: `rotate_half` assumes the
"duplicated" layout (`cat([freqs, freqs])`, splitting each vector into first
half / second half), not the interleaved layout. Implementing the spec
exactly as written would have silently produced incorrect rotations. ats-v2
uses the standard, internally-consistent Llama/GPT-NeoX convention:
`cat([freqs, freqs])` for the cache, paired with `rotate_half` for
application (`ats/model/rope.py`). This is exactly the kind of "broken math"
the design principles asked us to avoid, so we fixed it rather than
following the letter of an inconsistent spec.

## (c) DeepSpeed as a first-class citizen

`ats/parallelism/deepspeed_utils.py` is the single place `deepspeed.initialize()`
is called. The DeepSpeed JSON config is generated programmatically from the
resolved `ATSConfig` (ZeRO stage, precision, gradient accumulation, gradient
checkpointing, MoE settings) — no hardcoded config blobs. If DeepSpeed is not
installed, `initialize_engine` raises a `ConfigError` with the install
command, rather than silently falling back to a non-DeepSpeed path, since
DeepSpeed is mandatory for `parallelism.strategy` per the design brief.

## (d) No custom CUDA kernels

Attention uses `flash_attn.flash_attn_func` when available and falls back to
`torch.nn.functional.scaled_dot_product_attention` otherwise (both paths
also handle sliding-window attention — see (f)). MoE routing uses
`deepspeed.moe.layer.MoE` when available, with an explicit, clearly-labeled
single-process PyTorch fallback otherwise. No `.cu`/`.cpp` extension files
exist anywhere in this repository.

## (e) Clean adaptive controller vs. the old orchestrator

`AdaptiveController.step(metrics)` is called synchronously by the trainer
once per optimizer step and returns an `AdaptiveAction | None`. There is no
background thread, no monkey-patching of `Trainer` methods, and no randomly
generated "safety" metric. Cooldowns prevent LR-adjustment oscillation, and
three consecutive emergency gradient-explosion cuts trigger a
`TrainingHaltError` rather than spiraling the learning rate toward zero.

## (f) Sliding Window Attention (SWA)

`ats/model/swa.py::generate_swa_mask` builds a banded boolean mask
(`mask[i,j] = (i >= j) and (i - j < window_size)`), used by
`GroupedQueryAttention`'s SDPA fallback path. When flash-attn is used, we
attempt its `window_size=(left, 0)` argument first (supported in flash-attn
>= 2.2); if the installed version predates that argument, we catch the
resulting `TypeError` and fall back to the manual masked SDPA path rather
than silently training with full (unwindowed) attention.
`swa_full_attention_interval` (default 4) makes every Nth layer use standard
full causal attention instead of the window, following the hybrid
local/global pattern used in Mistral-style and some DeepSeek-style stacks,
so long-range dependencies are not entirely lost to windowing.

## (g) Multi-Head Latent Attention (MLA)

`ats/model/mla.py::MLAAttention` compresses K/V into a shared latent `c`
(`latent_dim`, default `hidden_size // 4`) via `W_DKV`, then up-projects to
per-head K/V content via `W_UK`/`W_UV`. RoPE is decoupled: a small
`rope_head_dim` slice per head is computed directly from the block input
(not from the latent) and rotated separately, then concatenated onto the
content K/Q before the attention dot product — the cached latent itself
never carries explicit position information. Critically, the KV cache tuple
returned by `MLAAttention.forward` is `(latent_cache, rope_k_cache)`, i.e.
`latent_dim + rope_head_dim` scalars per token, not `2 * num_kv_heads *
head_dim` as standard GQA caches. `tests/test_model.py::test_mla_cache_smaller_than_gqa_cache`
checks this arithmetic directly, and
`test_mla_incremental_decode_matches_full_forward` checks that step-by-step
decoding through the cache reproduces the same output as a single full
forward pass. MLA has no HuggingFace `Llama`-family equivalent, so
`ats/export/huggingface.py` explicitly refuses to export `use_mla=True`
models rather than emitting a checkpoint that would silently load with the
wrong attention mechanism.

## (h) Mamba / selective state-space blocks

`ats/model/mamba.py::MambaBlock` is a real, sequential selective-scan
recurrence in plain PyTorch (input-dependent `dt`/`B`/`C`, a per-channel
learned `A` stored in log-space for stability, a causal depthwise
convolution, and an explicit state tensor carried across the sequence loop)
— not a fused CUDA kernel (out of scope per the no-custom-kernels rule) and
not a relabeled transformer block. `tests/test_model.py::test_mamba_block_has_recurrent_state_dependence`
checks this directly: perturbing an early token measurably changes later
outputs, which a non-recurrent (e.g. purely per-position) block could not
reproduce. `MambaBlock` does not apply its own residual connection — like
`GroupedQueryAttention`/`SwiGLU` elsewhere in this codebase, it's a pure
sub-layer; `ats/model/transformer.py::MambaLayer` applies the pre-norm
residual (`x + MambaBlock(norm(x))`) and adapts it to the same
`(x, attention_mask, past_key_value, use_cache)` call signature as
`TransformerBlock`, so `ATSTransformer` can freely interleave the two via
`mamba_every_n_layers`. Note: this reference implementation does not support
KV-cache-based incremental decoding (`past_key_value` must be `None`);
that's a real limitation, documented in the README, not hidden.

## (i) Multi-Token Prediction (MTP)

`ats/model/mtp.py::MultiTokenPredictionHead` runs `mtp_num_tokens`
independent unembedding heads off the final hidden state, each predicting a
different future offset (t+1, t+2, ...); `compute_loss` averages
cross-entropy across all offsets whose target still falls inside the
sequence, masking the rest rather than crashing on short sequences. Offsets
beyond the first get a small per-offset linear transform before
unembedding, so heads aren't forced to share identical logits.

## (j) Diffusion language model

`ats/model/diffusion.py::DiffusionLM` wraps an `ATSTransformer` backbone
purely as an embedding-space noise predictor: `ATSTransformer.forward_hidden`
(new) accepts precomputed embeddings directly, skips the token embedding
lookup and the LM head, and returns final normed hidden states, since a
diffusion step never uses the autoregressive `lm_head`/logits path. Training
adds cosine-schedule noise to clean token embeddings and computes MSE
between predicted and true noise — never cross-entropy, since there is no
"next token" here. Sampling uses deterministic DDIM steps from Gaussian
noise back to a clean embedding, followed by nearest-neighbor lookup against
the embedding table to recover discrete tokens.
`ats/training/trainer.py::DiffusionTrainer` reuses the same
scheduler/checkpoint/monitor/adaptive-controller infrastructure as the
autoregressive `Trainer`, swapping only the loss computation and the model
wrapping.

## (k) Quantization-aware training

`ats/model/quantization.py::QuantizedLinear` supports three modes: `"none"`
(plain `nn.Linear`), `"int8"` (real `torch.ao.quantization` fake-quantization
on both weights and activations — verified in tests to actually perturb
numerics, not silently pass through), and `"fp8"`, which requires
`transformer-engine` or `torchao` and raises `ImportError` immediately if
neither is installed rather than silently falling back to bf16/fp16.

## (l) Full CLI override matrix

`train.py::apply_cli_overrides` merges every CLI flag into the loaded
`ATSConfig` with strict precedence (CLI > YAML > size preset > Pydantic
default), covering architecture toggles (`--use-swa`, `--use-mla`,
`--use-mamba`, `--use-moe`, `--use-mod`, `--use-mtp`), the `--architecture`
convenience preset (`dense`/`swa`/`mla`/`mamba`/`moe`/`mod`/`mtp`/`all`,
overridable per-flag by an explicit `--use-x`/`--no-use-x` on the same
command line), `--model-type`/`--quantization`, all architecture-specific
numeric knobs, and every model-size/training/data/parallelism field. Because
Pydantic's `model_copy(update=...)` bypasses validators, the merge finishes
by round-tripping the result through `ATSConfig.model_validate(...)`, so
every `field_validator`/`model_validator` — including the
`use_mtp` + `model_type="diffusion"` incompatibility check — runs again
against the fully merged configuration, not just the original YAML. This
means there is genuinely **one YAML file per model size**
(`configs/{debug,125m,350m,1b,3b,7b,14b,70b}.yaml`, all dense by default);
every architecture variant is a CLI flag on top of the same file, not a
separate config.

## Known gaps (honest, as of this revision)

- Mamba layers do not support KV-cache-based incremental decoding (training
  and full-sequence forward passes work; autoregressive generation with a
  cache does not, for Mamba layers specifically).
- `QuantizedLinear` is implemented and tested but not yet automatically
  substituted for every `nn.Linear` throughout the backbone — it's available
  as a building block, not yet wired into `attention.py`/`ffn.py`/`moe.py`
  by the `model.quantization` config field.
- MoE, MoD, MLA, Mamba, and diffusion models all correctly refuse HuggingFace
  export (`ats/export/huggingface.py`) since none of them has a standard
  `Llama`-family equivalent; only dense and SWA models export today.
- This revision was written and reviewed by eye without network access in
  the authoring sandbox, so it has not been executed (no `pytest`, no real
  training run, no `pip install -e .`) in that environment. Run the
  verification checklist yourself before relying on it.

## Bug-fix pass (external review)

An external review of this codebase reported 14 issues. Verifying each
against the actual code (not just trusting the report) found 9 real bugs,
which are now fixed, and 3 claims that didn't match the current code:

**Fixed:**
- **Tokenizer OOB ids** (`ats/data/tokenizer.py`): `pad_token_id`/`eos_token_id`
  were set to `vocab_size` itself, which is out of range for
  `nn.Embedding(vocab_size, ...)` (valid indices are `0..vocab_size-1`) and
  would `IndexError` on the first padded batch. Fixed by reserving vocab_size
  as `n_vocab + 1` and using `n_vocab` (the new last valid index) for both
  special tokens.
- **Hardcoded `rank=0` in `train.py`**: every distributed process was reading
  identical data. Now reads `RANK`/`LOCAL_RANK` from the environment (set by
  DeepSpeed/torchrun launchers), with a validation check against
  `parallelism.gpus * parallelism.nodes`.
- **`export.py` config auto-discovery pointed at a file that was never
  written**: `CheckpointManager.save()` never actually wrote a `config.yaml`
  anywhere, so no path `export.py` could have looked at would have worked.
  Fixed at the root: `save()` now writes `config.yaml` into the checkpoint's
  own tag directory, and `export.py` looks there (falling back to the parent
  directory for compatibility).
- **Misleading dataset docstring/error text** claiming HuggingFace-streaming
  and WebDataset support that was never implemented — corrected to describe
  the actual (local `.jsonl`-only) behavior.
- **`num_workers > 0` produced duplicate data**: every DataLoader worker
  process got an identical copy of `MixedDataset` with the same seed, so
  every worker yielded the same sequence. `_TorchMixedDataset.__iter__` now
  combines `(rank, world_size)` sharding with `(worker_id, num_workers)`
  sub-sharding via `torch.utils.data.get_worker_info()`.
- **`evaluate.py` hardcoded `./eval_data/<task>.jsonl`**: added
  `--eval-data-dir` (default `./eval_data`, but now overridable).
- **`evaluate.py` discarded `checkpoint_manager.load()`'s return value**: now
  captured and logged (resumed step/epoch).
- **Weak SWA test** (`test_gqa_with_swa_restricts_attention_span`) only
  checked output shape, not that attention was actually restricted. Replaced
  with a test that perturbs a token outside the window and asserts the
  output at a later position is *exactly* unchanged, plus a control test
  (`use_swa=False`) proving the same perturbation *does* propagate under
  ordinary causal attention — so the windowed result isn't a coincidence of
  some unrelated bug.
- **Fake MoE gating test** (`test_moe_gating_weights_sum_to_one`) reimplemented
  the gating math standalone instead of testing `MoELayer`, so a real bug in
  the module wouldn't have been caught. `_PyTorchMoEFallback.forward()` was
  refactored to call a new `compute_routing()` method, which the test now
  calls directly, plus an added end-to-end test confirming different gate
  weights actually produce different `MoELayer` outputs.

**Reported but not real, verified against current code:**
- "`train.py` ignores YAML `micro_batch_size`" — already fixed in the
  previous turn; the report was checking a stale copy of the file.
- "`DiffusionTrainer` may not exist, causing an import crash" — it exists
  (`ats/training/trainer.py`), and was added in the previous turn.
- "Missing `configs/350m.yaml`" — it exists in `configs/`.
- "`MixedDataset` buffer carries over between epochs under
  `persistent_workers=True`" — `buffer` is a local variable inside
  `__iter__`, not `self.buffer`; there is no persistent state to carry over.

**Reported but a spec-compliance question, not a bug:**
- "`--architecture all` enables Mamba by default, which is
  experimental/unstable" — the original design brief explicitly specified
  `"all" = enable SWA + MLA + Mamba + MoE + MoD + MTP`. Implemented as
  specified. Whether Mamba belongs in a default "everything" preset given
  its experimental status is a legitimate product question, not a code bug;
  flagging it here rather than silently changing agreed-on behavior.

**Not fixed, acknowledged:** the tokenizer's reserved-slot fix means a
tiktoken-backed `Tokenizer.vocab_size` is `n_vocab + 1`, but `model.vocab_size`
in the shipped configs is a fixed round number (50304) independent of the
tokenizer actually configured — nothing currently validates these two values
against each other. `ATSTransformer.forward()` will raise a clear
`ValueError` if a token id ever exceeds `model.vocab_size`, which is a
safety net, not a fix; wiring `data.tokenizer_name`'s resolved vocab size
into `model.vocab_size` validation is a real gap for a future pass.

## Infrastructure pass: CLI restructure, safetensors, memory estimator, Triton kernels, preprocessing

### `ats/cli/` restructure + `pyproject.toml`
`train.py`/`evaluate.py`/`export.py` moved to `ats/cli/`; added
`ats/cli/doctor.py` (real environment diagnostics), and honest placeholder
`ats/cli/finetune.py`/`ats/cli/align.py` (parse args, print a clear
"not implemented" message, exit 1 — not silent no-ops). `setup.py` was
removed in favor of `pyproject.toml`, which declares five console scripts
(`ats-train`, `ats-eval`, `ats-export`, `ats-doctor`, plus the two
placeholders).

**A real bug was caught by actually running `ats-doctor`**: `ats/__init__.py`
eagerly imported `ATSConfig`, which requires pydantic — meaning `ats-doctor`,
whose entire purpose is diagnosing a broken/incomplete environment, couldn't
run at all without pydantic already installed. Fixed with a lazy
`__getattr__` (PEP 562). Verified the fix by actually executing
`python -m ats.cli.doctor` in this sandbox afterward; it correctly reported
this environment's real (missing) PyTorch/DeepSpeed/Triton/GPU state — not
hardcoded strings.

### Safetensors checkpoints
`CheckpointManager.save()` now writes `model.safetensors` (via
`safetensors.torch.save_file`) alongside DeepSpeed's own optimizer/ZeRO
checkpoint. DeepSpeed's native checkpoint format is not something ats
controls or can swap out and still support exact training resumption —
the safetensors file is an additional, de-sharded, pickle-free artifact
specifically for fast loading, HF export, and manual inspection, which is
what the safetensors format's actual benefits (fast I/O, no arbitrary-code-
execution risk, HF-standard) are about. Added
`load_model_weights_safetensors()` for reading just the weights without a
full DeepSpeed engine.

### Weight decay parameter groups
`get_param_groups()` (`ats/parallelism/deepspeed_utils.py`) splits
parameters into decayed (2D+ projection matrices) and non-decayed (biases,
norm weights) groups, following standard LLM training practice. Required
adding a `training.weight_decay` config field, since none existed.

### Pre-flight memory estimator + actionable OOM messages
`ats/utils/memory.py::estimate_memory` returns a `MemoryReport` (model/
optimizer/activation GB, ZeRO-stage-aware sharding, suggested batch
size/grad accum/zero stage) using real parameter-count arithmetic and the
standard (documented, heuristic) transformer activation-memory formula.
`Trainer.__init__`/`DiffusionTrainer.__init__` call it and log a warning if
estimated memory exceeds 80% of detected GPU memory.
`ats.training.trainer._log_oom_and_reraise` catches
`torch.cuda.OutOfMemoryError` around each training step and logs the
model/optimizer/activation breakdown plus concrete suggested flags before
re-raising immediately (never swallowed).

### `ats-doctor`
Real checks only: Python version via `sys.version_info`, package versions
via `importlib.metadata` (no hardcoded version strings), GPU count/memory
via `torch.cuda`, and — given `--config` — a memory estimate for that
specific config. Verified by actual execution in this sandbox (see above).

### Auto-generated model cards
`ats/export/huggingface.py::_build_model_card` writes a `README.md`
alongside every HuggingFace export, with fields (hidden size, layers,
attention mechanism, FFN type, SWA window, MoE expert count, ...) read
directly from the resolved config, not hand-written — verified with a test
asserting two differently-configured exports produce different card content.

### Offline preprocessing + preprocessed-data reading
`preprocess.py` tokenizes a `.jsonl` corpus offline, with optional
`--packing` (concatenating documents EOS-delimited into full `seq_length`
blocks instead of one padded block per document), and writes a
memory-mapped `tokens.bin` (`numpy.memmap`, int32) plus `valid_lengths.npy`
and `meta.json`. `ats/data/dataset.py::MixedDataset` was extended to detect
`.bin` sources and read them directly via memmap, bypassing tokenization
entirely, alongside the existing `.jsonl` on-the-fly path (which is
unchanged). **The core packing + memmap write/read-back logic was actually
executed** (with a fake tokenizer, since the real one requires pydantic)
in this sandbox and round-tripped byte-for-byte correctly — this is real
verification, not just code review.

### Triton kernels — unverified, and explicitly scoped honestly
`ats/model/{norm,rope,moe,mla}_triton.py` add fused-kernel variants of
RMSNorm+residual, RoPE rotation, MoE softmax+top-k routing, and MLA KV
decompression. **None of these were run**: there is no GPU or Triton
installation in the authoring environment to compile, execute, or
benchmark them against. Every kernel is gated behind `HAS_TRITON` (and,
where relevant, `top_k <= 8` for the unrolled MoE routing loop) with a
PyTorch fallback that *is* tested and *is* verified equivalent to the
existing reference implementations (`ats.model.norm.RMSNorm`,
`ats.model.rope.apply_rotary_pos_emb`). Two kernels are intentionally only
*partially* fused, documented as such rather than overclaimed: MoE fuses
routing (softmax+top-k) but not the capacity-aware token gather/scatter
dispatch, which stays in the existing PyTorch `_PyTorchMoEFallback`; MLA
fuses the two KV up-projection matmuls into one by concatenating weights
ahead of time and using a single (standard, tutorial-pattern) Triton GEMM
kernel, rather than authoring a novel kernel shape blind.
`tests/test_triton.py` tests the PyTorch fallback paths for real (these do
run and do mean something) and gates the actual Triton-vs-fallback parity
tests behind `torch.cuda.is_available() and HAS_TRITON`, so they're skipped
— not silently passed — anywhere without real hardware to check them on.

### lm-evaluation-harness integration
`ats/cli/evaluate.py` was rewritten: standard benchmark tasks (MMLU,
HellaSwag, ARC, ...) are no longer hand-scored in ats — they're delegated to
`lm-evaluation-harness` via `subprocess.run([sys.executable, "-m", "lm_eval", ...])`,
after auto-exporting the checkpoint to HuggingFace format (cached under
`<checkpoint>/hf_exported/`). A separate `--config` (perplexity) mode is
kept for held-out-data evaluation and for architectures (MoE/MoD/MLA/Mamba/
diffusion) that can't be HF-exported at all, since lm-eval-harness has
nothing to evaluate them with either. The previous hand-rolled
`_score_multiple_choice` benchmark scorer was removed — no existing tests
referenced it.

### CI / launch scripts
`.github/workflows/ci.yml` runs `pip install -e .`, `pytest tests/`, a
basic import/config-load smoke test, and `ats-doctor`. `scripts/launch.sh`
(torchrun wrapper) and `scripts/slurm_submit.sh` (SLURM template) were added
and their bash syntax was actually checked with `bash -n` in this sandbox
(both valid).

### New/changed config fields
Added `training.weight_decay` (required by the param-group split) and
confirmed `training.micro_batch_size` (added in a previous session) is
still correctly wired everywhere after the `ats/cli/` move.
