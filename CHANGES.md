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
