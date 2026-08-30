# ats-v2 Performance Audit — 5,500 tok/s → target 20k–50k tok/s

Baseline: ~5,500 tokens/sec on the 125M config, with OOMs at reasonable batch
sizes. All fixes below are applied on top of the repo at HEAD; the git log has
one commit with the full diff.

## Phase 1: Mandatory fixes

### Fix 1 — Spatial cross-entropy loss (eliminates the 6+ GiB OOM)
**Files:** `ats/training/trainer.py` (`Trainer.train_step`, `Trainer.evaluate`),
`ats/training/perplexity.py`

`shift_logits = output.logits[..., :-1, :]` slices off the *second-to-last*
dimension, not the last one, so the result is **not contiguous** (its
per-batch stride still reflects the original, un-sliced `seq_len`). The old
code then called `.reshape(-1, vocab_size)` (or `.contiguous().view(...)` in
`perplexity.py`), which silently materializes a brand-new, fully contiguous
`[batch * (seq_len-1), vocab_size]` tensor. At this config's shapes
(`seq_len=4096`, `vocab_size=100352`, fp16) that's **~6.5 GiB** for a single
micro-batch of 8 — and it has to stay alive through backward.

Fix: `F.cross_entropy` natively accepts `(N, C, d1, ...)` inputs with
`(N, d1, ...)` targets (the same convention it uses for segmentation losses).
Transposing the class dimension into position 1 is metadata-only (0 bytes),
so no copy is ever made:

```python
shift_logits = output.logits[..., :-1, :]
shift_labels = batch["labels"][..., 1:]
ce_loss = torch.nn.functional.cross_entropy(
    shift_logits.transpose(1, 2),
    shift_labels,
    ignore_index=-100,
)
```

Applied in `train_step` (both `Trainer` and the eval path), `evaluate()`
(`reduction="sum"` variant), and `perplexity.py`'s eval loop.

### Fix 2 — Double gradient-norm computation
**File:** `ats/training/trainer.py`, `Trainer.train_step` / `DiffusionTrainer.train_step`

Removed the pre-step `torch.nn.utils.clip_grad_norm_(model_engine.parameters(),
max_norm=float("inf"))` call. `max_norm=inf` never clips anything — it was
only ever there to read back the norm — but DeepSpeed's `model_engine.step()`
already computes gradient clipping (and the resulting global grad norm)
internally per `gradient_clipping` in the DeepSpeed config. That call was a
second full all-reduce + norm pass over every parameter's gradient, every
single optimizer step, purely for logging. Now:

```python
self.model_engine.step()
grad_norm = self.model_engine.get_global_grad_norm()
if grad_norm is not None:
    grad_norm = float(grad_norm)
else:
    # rare fallback only (e.g. gradient_clipping disabled, or a
    # non-DeepSpeed test double) — not on the hot path for real runs
    grad_norm = float(
+        torch.nn.utils.clip_grad_norm_(
+            self.model_engine.parameters(), max_norm=float("inf")
+        )
+    )
```

### Fix 3 — Memory profiling gated behind `log_every`
**File:** `ats/training/monitor.py`, `Monitor.log`

`get_gpu_memory_info()` moved inside `if step % self.config.log_every == 0:`.
It was previously called unconditionally on every step even though
`log_every=50` clearly signals "report roughly every 50 steps" — 49/50 calls
were pure waste.

**Bonus find in the same function** (not in the original 5, see Phase 2 below
for the writeup): TensorBoard/W&B logging was *also* outside this gate and
has now been moved inside it too.

### Fix 4 — SDPA fallback `is_causal`
**File:** `ats/model/attention.py`

On inspection, `GroupedQueryAttention.forward`'s non-flash SDPA branch
(the final `else:` block) was **already correct**: when `attention_mask is
None` and no SWA/incremental-cache path applies, it computes
`is_causal = True` and calls
`F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)`,
which is exactly the form PyTorch dispatches to its fused
FlashAttention/mem-efficient kernels for. The flash-attn branch (when
`flash_attn` is installed) was likewise already passing `causal=is_causal`
correctly.

The reason the fast path never actually fired in practice traces to the data
pipeline, not `attention.py`: `ats/data/dataloader.py::_collate` was building
an all-ones `attention_mask` on *every* batch and handing it to every
forward pass. A non-`None` attention_mask unconditionally forces
`is_causal = False` in `attention.py` (by design — masks and
`is_causal=True` can't be combined safely), so the optimized causal kernel
path was dead code for every real training step. Fixed at the source
(`_collate` no longer manufactures this mask — see Phase 2 for the full
writeup, since this doubles as an independent finding); `attention.py`
itself required no code change.

### Fix 5 — Config defaults
**File:** `configs/125m.yaml`

```yaml
training:
  grad_accum_steps: 2
  micro_batch_size: 4
```

(Effective batch = 4 × 2 = 8 sequences × 4096 tokens per optimizer step —
smaller than a throughput-maximizing setting would be, but chosen per this
round's brief as a conservative default that leaves headroom against the
6+ GiB logits allocation this audit just eliminated, at `vocab_size=100352`.
Once Fix 1 lands, most GPUs have room to push `micro_batch_size` well past 4;
this is a safe floor, not a ceiling — see the memory numbers below.)

---

## Phase 2: Additional bugs found

### 2.1 — Collator manufacturing a useless, fast-path-killing attention_mask
**File:** `ats/data/dataloader.py::_collate`, line ~107 (pre-fix)

`attention_mask = torch.ones_like(input_ids, dtype=torch.long)` was built and
returned on *every* batch. Every example in this pipeline is already a
fixed-length `seq_length` block (asserted a few lines above); the only
padding that ever occurs (the final partial chunk of a stream, or a
preprocessed block's tail — see `MixedDataset._make_example` /
`_iter_preprocessed_shard`) is encoded entirely via `IGNORE_INDEX` in
`labels`, never via `input_ids` or a mask. So this tensor was **always
literally all ones** — it carried zero information — while costing:
  1. an extra full-size `[batch, seq_len]` allocation every batch, and
  2. (far more costly) permanently disabling `GroupedQueryAttention`'s
     `is_causal=True` fast path on every layer of every step, as described
     under Fix 4 above.

**Fix:** stopped building it; `_collate` now returns only `input_ids` and
`labels`. Every call site already used `batch.get("attention_mask")`, so
downstream code sees `None` and takes the fast path automatically — verified
by smoke test and confirmed via `test_data.py::test_collate_produces_correct_shapes`
(updated to assert the key is absent).

*(Verified safe: audited every attention variant — `GroupedQueryAttention`,
`MLAAttention`, `mod.py`'s `MixtureOfDepths` wrapper, `mamba.py` — and
`attention_mask=None` was already a fully supported, well-tested input to
all of them; it's the default value in every signature. The historical bug
CHANGES.md documents about MLA + `attention_mask` was the *opposite* case —
MLA used to crash when a mask *was* supplied — which this change doesn't
touch or reintroduce.)*

### 2.2 — TensorBoard/W&B logging bypassing `log_every`
**File:** `ats/training/monitor.py::Monitor.log`

The `self._tb_writer.add_scalar(...)` / `self._wandb.log(...)` calls lived
*outside* the `if step % self.config.log_every == 0:` block — every metric
was written to TensorBoard/W&B on **every single step**, not every 50th, even
though `log_every=50` is set explicitly in `configs/125m.yaml` and
`use_tensorboard: true`. Each `add_scalar` call does real Python-side
work (dict → proto encode, event-file write); W&B's `.log()` can additionally
involve network I/O. This was 50x the intended write volume.

**Fix:** moved both calls inside the `log_every` gate (see Fix 3's code).

### 2.3 — Per-expert `.item()` loop in MoE utilization tracking
**File:** `ats/model/moe.py`, ~line 160 (pure-PyTorch fallback) and ~line 257
(DeepSpeed backend)

```python
self.last_expert_utilization = {
    i: float(normalized_utilization[i].item()) for i in range(self.num_experts)
}
```

Each `.item()` call forces an independent GPU→CPU synchronization; this ran
`num_experts` times per forward pass, on every MoE layer. Not on the hot path
for the 125M dense config (`use_moe: false`), but real for any MoE run and
squarely in the ".item() in a loop" category this audit was asked to hunt
for.

**Fix:** replaced with a single `.tolist()` call (one sync for the whole
vector) in both backends:
```python
self.last_expert_utilization = dict(enumerate(normalized_utilization.tolist()))
```

### 2.4 — SWA mask rebuilt from scratch on every forward pass
**File:** `ats/model/swa.py::generate_swa_mask`

For SWA-enabled models, this rebuilds a full `[seq_len, seq_len]` boolean
tensor (16 MiB at `seq_len=4096`) via `torch.arange` + broadcasting compares
on *every* SWA layer's *every* forward call, even though `seq_len` and
`window_size` are fixed for the duration of a training run. `use_swa: false`
in the 125M default config, so this doesn't affect the target model, but it's
a real, avoidable O(seq_len²) redundant computation for any SWA run.

**Fix:** memoized with `functools.lru_cache(maxsize=8)` keyed on
`(seq_len, window_size, device)`. Safe because the mask is only ever read
(combined via `&`, which allocates a new tensor) and never mutated in place;
`maxsize=8` bounds memory if `seq_len` legitimately varies (padded final
batch, a different eval length) instead of growing unbounded.

### 2.5 — `gradient_checkpointing: true` in `configs/125m.yaml` (flagged, not changed)
Maps to `checkpoint_every_n_layers=1` — activation checkpointing on *every*
layer. This trades ~20–30% extra compute (recomputing the forward pass
during backward) for activation memory that a 125M model, at any reasonable
micro-batch size, doesn't need. **Recommended to disable** for this config
size now that Fix 1 has removed the actual memory pressure, but left as-is
here since it's a training-quality/memory-margin tradeoff outside this
round's 5 mandated changes — flagging per "be ruthless" instruction rather
than silently changing a default that wasn't asked for.

### 2.6 — No `num_workers` knob anywhere (flagged, not changed)
`build_dataloader()` (`ats/data/dataloader.py`) has a `num_workers: int = 0`
parameter, but `ats/cli/train.py` never passes it and `DataConfig` has no
corresponding field — there is **no way to enable multi-worker loading at
all** short of editing library code. For `.jsonl` (raw-text) sources this
means synchronous, main-thread tokenization blocking the GPU between every
batch. Not applied here because it needs schema (`DataConfig.num_workers`)
and CLI plumbing across `train.py`/`finetune.py`/`perplexity.py`, which is
larger in scope than this round's fixes and untestable end-to-end without a
real multi-process run. Recommended follow-up.

### 2.7 — Missing `torch.compile` (flagged, not applied)
No call site wraps the model in `torch.compile`, despite `ffn.py`'s own
docstring noting it would fuse `SwiGLU`'s elementwise ops. Not applied here:
`ATSTransformer` is used under DeepSpeed (ZeRO wraps parameters), with
optional gradient checkpointing, MoE/MoD routing, and Mamba/Triton kernels —
several of these interact poorly with `torch.compile`'s graph-capture without
per-feature validation on real hardware, which isn't feasible to verify in
this environment (no GPU/compile backend available here). Recommended as a
config-gated (`training.use_torch_compile`, default `False`) follow-up,
applied only to the base dense-attention path first.

### 2.8 — Not a bug: ZeRO stage / grad-accum, checked and correct
- `ats/parallelism/auto_parallel.py::resolve_strategy` already selects
  `deepspeed_zero0` for `gpus==1, nodes==1` (i.e. `parallelism.strategy: auto`
  with a single GPU correctly avoids ZeRO-2/3's sharding overhead, which
  only pays off with multiple ranks) and escalates to `deepspeed_zero2`
  (≤8 GPUs, ≤13B params) or `deepspeed_zero3` beyond that for multi-GPU.
  No change needed.
- `build_deepspeed_config` deliberately sets DeepSpeed's own
  `gradient_accumulation_steps: 1` (not `config.training.grad_accum_steps`)
  with a detailed comment explaining why: `Trainer`/`DiffusionTrainer` own
  gradient accumulation manually (dividing the loss and gating `step()`
  themselves so `AdaptiveController` sees one call per real optimizer step).
  Setting both would double-apply the 1/N loss scaling. Already correct,
  not a bug.

---

## Phase 3: Test results

```
226 passed, 8 skipped in ~3s   (tests/)
```

No failures. The 8 skips are pre-existing (Triton/DeepSpeed-hardware-gated
tests, `@pytest.mark.skipif`), unrelated to this audit. Ran targeted verbose
passes on `test_bug_audit_fixes.py`, `test_training.py`, `test_model.py`,
and `test_data.py` specifically to confirm:
- `Trainer`/`DiffusionTrainer`'s MTP and gradient-finiteness tests still pass
  with the new cross-entropy shape and grad-norm logic.
- `test_collate_produces_correct_shapes` (updated) confirms `_collate` no
  longer emits `attention_mask`.
- MoE utilization tests still pass with the `.tolist()`-based collection.
- All SWA tests pass with the memoized mask.

**Smoke test** (CPU, since no GPU/DeepSpeed available in this environment):
loaded `configs/125m.yaml` via the real config loader, shrank it to a tiny
architecture, ran `_collate` → confirmed no `attention_mask` key → ran a real
forward pass (hit the SDPA fallback, confirmed via log line it took the
no-flash-attn path) → computed the spatial cross-entropy loss → called
`.backward()` → confirmed gradients populated. All succeeded.

---

## Phase 4: Expected performance after all changes

Directional estimates (no GPU available in this sandbox to benchmark
directly — these follow from the nature of each fix):

- **Throughput:** the single largest lever is Fix 4/2.1 (attention_mask
  forcing off the fused-kernel path) — the difference between an explicit
  O(seq_len²) masked SDPA call and a fused FlashAttention-dispatching kernel
  is typically **2–4x** at `seq_len=4096` on its own. Combined with Fix 2
  (removing a full second grad-norm pass every step) and Fix 5 (raising
  micro_batch_size off of 1, cutting kernel-launch-overhead-dominated
  execution), a return to the previously-observed **20k–50k tok/s** range is
  plausible; the exact number depends on GPU, `flash_attn` availability, and
  final micro_batch_size chosen at deploy time.
- **Memory:** Fix 1 alone removes a **~6.5 GiB** transient allocation per
  micro-batch at this config's shapes (`seq_len=4096`, `vocab=100352`,
  fp16, `micro_batch_size=4`) — this was very plausibly the direct cause of
  the reported OOMs at "reasonable" batch sizes, independent of throughput.
  With that removed, `micro_batch_size` can likely go well above the `4`
  set in Fix 5 on typical 24–80 GiB training GPUs; `4` was chosen as a safe
  floor per this round's brief, not a ceiling.

---

## Zip location

`ats-v2-optimized.zip`, at the repository root.
