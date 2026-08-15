# CI Fixes (commit 0bb468a)

This documents the work to get `ruff check`, `ruff format --check`, `mypy
ats --ignore-missing-imports`, and `pytest tests/` all passing cleanly at
commit `0bb468a` ("Bugfix: address multiple model/CLI/trainer issues").

## Which jobs were failing

- ❌ Lint & format check (`ruff check .`, `ruff format --check .`)
- ❌ Type check (`mypy ats --ignore-missing-imports`)
- ❌ Test (py3.10, py3.11, py3.12) — `pytest`
- ⏭️ Validate example configs — skipped because `test` never got to run
  (the `config-validate` job `needs: test`)
- ✅ CodeQL — unaffected by any of this, still passes

## Important context: most of this predates commit 0bb468a

Before touching anything, I diffed the CI state against the immediately
prior commit (`b63048d`, "Added github actions" — the commit that
introduced the workflow file itself):

- `ruff check .` on `b63048d`: **304 errors**
- `ruff check .` on `0bb468a`: **307 errors** (307 - 304 = 3 new ones,
  from the bugfix commit's own added code)
- `mypy ats --ignore-missing-imports` on `b63048d`: **65 errors**
- `mypy ats --ignore-missing-imports` on `0bb468a`: **65 errors** (no
  change — the bugfix commit's added code happened not to introduce any
  *new* mypy errors, it just didn't fix any of the pre-existing ones
  either)

In other words: CI was red from the moment the workflow was added, on
essentially the whole repository, for reasons unrelated to the bugfix
commit's 14 targeted changes. The task framing ("commit 0bb468a broke
CI") isn't quite accurate — the honest summary is "CI was never green,
and this pass makes it green." I fixed everything regardless, since the
task's goal (`ruff`/`mypy`/`pytest` all pass cleanly) doesn't depend on
whose commit something originated in.

## Root causes and fixes, by job

### `ruff check .` (307 errors → 0)

All pre-existing repo-wide style debt, not specific to any of the 14
bugfix files. Categories, roughly in order of frequency:

- **`EXE001`** (8 files) — `.py` files with a `#!/usr/bin/env python`
  shebang line that weren't marked executable (`ats/cli/{align,doctor,
  evaluate,export,finetune,train}.py`, `preprocess.py`,
  `scripts/verify.py`). Fixed with `chmod +x`.
- **`C408`** (~10 occurrences) — `dict(...)` calls that should be `{...}`
  literals, mostly in test helper functions. Auto-fixed via
  `ruff check . --fix`.
- **`RUF059` / `F841`** (~8 occurrences) — unpacked-but-unused variables
  (`batch, seq_len, hidden_size = x.shape` where only `hidden_size` is
  used, etc.) and assigned-but-unused locals. Auto-fixed (prefixed with
  `_` or removed).
- **`I001`** — unsorted/unformatted import blocks in a few test files
  and (later, from my own mypy edits) `ats/model/moe.py`. Auto-fixed.
- **`PIE810`** — one `str.startswith(a) or str.startswith(b)` in
  `preprocess.py`, mergeable into one `startswith((a, b))` call.
  Auto-fixed.
- **`PLW1508`** (3 occurrences, `ats/cli/train.py`, `ats/cli/finetune.py`,
  `ats/training/checkpoint.py`) — `os.environ.get("LOCAL_RANK", 0)`
  passes an `int` default to `os.environ.get`, which only accepts `str`
  or `None`. Fixed by changing the default to `"0"`.
- **`SIM102`** (4 occurrences, `ats/config/schema.py`,
  `ats/training/trainer.py` x2) — nested `if` statements that ruff
  flags as combinable with `and`. Combined manually (the trainer.py
  ones also fold in a `self.eval_dataloader is not None` check that
  was on a separate nested `if`).
- **`BLE001`** — one deliberately-broad `except Exception:` in
  `tests/test_data.py` used as a tiktoken-availability probe. Left the
  broad catch (it's correct here — many different exception types can
  come from `tiktoken.get_encoding`'s network path, and the correct
  response to any of them is "unusable") and suppressed with
  `# noqa: BLE001` plus a comment explaining why.
- **`B023`** — a lambda inside a `for worker_id in (0, 1): ...` loop in
  `tests/test_data.py` that captured `worker_id` by reference (classic
  late-binding closure bug — both lambdas would have seen the final
  loop value). Fixed by binding via a default argument:
  `lambda wid=worker_id: ...`.

After `ruff check . --fix` plus these manual fixes: **0 errors**.

### `ruff format --check .` (50 files → 0)

Also pre-existing debt (confirmed also present on `b63048d`). Ran
`ruff format .` once; no manual intervention needed since none of it
conflicted with anything else.

### `mypy ats --ignore-missing-imports` (65 errors → 0)

One thing to flag up front: **the task's suggested fixes for the "likely
mypy failures" mostly weren't the actual problem.** `MixtureOfDepths.
forward`'s new signature, `MLAAttention.__init__`'s new params, and
`ep_size: int = 1` were all already correctly typed by the previous
patch round — none of those specifically triggered mypy errors. The
real errors clustered into a few genuinely distinct root causes:

**1. `ModelConfig`'s `int | None` architecture fields, read after
`is_resolved()` (the large majority of the 65 errors).** `ModelConfig.
hidden_size` / `num_layers` / `num_heads` / `num_kv_heads` /
`intermediate_size` are typed `int | None` (they start `None` until
`apply_size_preset()` fills them in from a `model.size` preset).
Runtime code correctly guards every use with `if not config.
is_resolved(): raise ...`, but `ModelConfig` is a mutable (non-frozen)
pydantic `BaseModel`, so mypy can't narrow `config.hidden_size` from
`int | None` to `int` across the rest of a function just because a
`bool`-returning method was called — narrowing only applies to local
variables, not attribute reads that could theoretically change between
accesses. Fixed by binding local `assert x is not None` + `x: int =
config.x` immediately after each `is_resolved()` check, in:
`TransformerBlock.__init__`, `ATSTransformer.__init__`, `MambaLayer.
__init__` (all in `ats/model/transformer.py`), `estimate_memory` (`ats/
utils/memory.py`), `DiffusionTrainer.__init__` (`ats/training/
trainer.py`), and `export_to_huggingface` (`ats/export/huggingface.py`).

**2. `quantization: str = "none"` instead of the `Literal["none",
"int8", "fp8"]` type (`QuantizationMode`, already defined in `ats/
model/quantization.py`).** `SwiGLU.__init__`, `GroupedQueryAttention.
__init__`, `MLAAttention.__init__`, and both `MoELayer` constructors
declared their `quantization` parameter as plain `str`, then passed it
into `make_linear(..., quantization: QuantizationMode, ...)`, which
mypy correctly flags as a type mismatch. Fixed by importing
`QuantizationMode` and using it as the parameter type in `ats/model/
ffn.py`, `ats/model/attention.py`, `ats/model/mla.py`, `ats/model/
moe.py`.

**3. `nn.Module`-typed attributes losing their concrete type.**
`TransformerBlock.attention` and `TransformerBlock.ffn` are declared
`self.attention: nn.Module = ...` / `self.ffn: nn.Module = ...`
(deliberately, since they can hold different concrete classes
depending on config). Reading `.o_proj` / `.down_proj` off them then
falls through torch's stub `nn.Module.__getattr__`, which returns
`Tensor | Module` — not what `init_residual_projection(module: nn.
Linear, ...)` expects. Both concrete types actually construct these via
`ats.model.quantization.make_linear()`, which returns `nn.Linear`, so
the mismatch is a typing gap, not a real bug. Fixed with `cast(nn.
Linear, ...)` at the two call sites in `TransformerBlock.__init__`. The
same pattern hit `MoELayer.__init__`'s `for expert in self.experts:
init_residual_projection(expert.down_proj, ...)` (iterating an `nn.
ModuleList` is typed as yielding plain `nn.Module`) — fixed with an
`isinstance(expert, SwiGLU)` assert instead, and
`ATSTransformer._collect_expert_utilization`'s `block.ffn.
last_expert_utilization` — fixed by adding `isinstance(block.ffn,
MoELayer)` to the existing `isinstance(block, TransformerBlock)` check.

**4. `checkpoint_every_n_layers: int | None` used after a `bool(n)`
proxy check.** `ATSTransformer._run_layers` computed `use_checkpointing
= self.training and not use_cache and bool(n)`, then later used `n`
directly in `layer_idx % n`. mypy doesn't track that `use_checkpointing`
being true implies `n is not None` — that's a relationship between two
different variables, not something narrowing follows. Fixed by adding
an (redundant at runtime, but mypy-visible) `n is not None` check
directly into the same `if` expression.

**5. Miscellaneous, one-off issues, each in its own file:**
- `ats/model/rope.py` — `self.inv_freq` (registered via
  `register_buffer`, no type annotation) fell through
  `nn.Module.__getattr__` as `Tensor | Module`; added an explicit
  `self.inv_freq: torch.Tensor` declaration.
- `ats/model/initialization.py` — `module.weight._ats_residual_init =
  True` deliberately monkey-patches a marker attribute onto a `Tensor`
  (torch allows this at runtime; the stubs don't model it). Added a
  scoped `# type: ignore[attr-defined]` with a comment.
- `ats/model/diffusion.py` — `DiffusionLM.backbone` was typed plain
  `nn.Module`, but the class calls `self.backbone.forward_hidden(...)`,
  which isn't a real `nn.Module` method (it's specific to
  `ATSTransformer`). Rather than import the concrete `ATSTransformer`
  class (which would tie `DiffusionLM` to one backbone implementation),
  added a small structural `_HiddenStateBackbone` `Protocol` declaring
  just the `forward_hidden` signature `DiffusionLM` actually depends on,
  and typed `self.backbone` against that (with a scoped
  `# type: ignore[assignment]` on the initial assignment, since
  `nn.Module` doesn't structurally satisfy the Protocol statically even
  though every real backbone passed in does at runtime).
- `ats/model/mla.py` — `attn_mask` was assigned `build_incremental_
  causal_mask(...)` (a `Tensor`) in the `if` branch and `attention_mask`
  (`Tensor | None`) in the `else` branch; mypy inferred the variable's
  type from the first (narrower) assignment and flagged the second.
  Fixed with an explicit `attn_mask: torch.Tensor | None` annotation
  before the `if`.
- `ats/utils/memory.py` — `model_bytes` / `optimizer_bytes` /
  `gradient_bytes` / `activation_bytes` were all inferred as `int` from
  their first assignment (products of `int`s), then reassigned a
  `float` a few lines later (`x / world_size` or `x / reduction_factor`,
  and Python division always produces `float`). Fixed by declaring each
  as `: float` at first assignment.
- `ats/cli/doctor.py` — `check_flash_attention() -> tuple[bool, str |
  None]` declared a possible `None` message, but neither actual return
  statement returns `None` for it. Narrowed the return type to
  `tuple[bool, str]` to match reality.
- `ats/data/dataset.py` — `_resolve_source` returns `(kind, payload)`
  where `payload` is a `Path` for `kind == "preprocessed"` (the
  iterator is built later, once `seq_length` is known) but an
  `Iterator[Any]` for `kind == "text"` — the return type was declared
  as always `Iterator[Any]`, which is simply wrong for the preprocessed
  case. Fixed the return type to `tuple[SourceKind, Path |
  Iterator[Any]]`, and narrowed at the one call site
  (`MixedDataset.__iter__`) using `isinstance(payload, Path)` (mypy
  can't correlate the separately-unpacked `kind` tag with `payload`'s
  type, so narrowing has to go through `payload` directly).
- `ats/training/trainer.py` — `Trainer.__init__` / `DiffusionTrainer.
  __init__` declared `train_dataloader: Iterator[Any]` /
  `eval_dataloader: Iterator[Any] | None`, but both are only ever
  consumed via `iter(self.train_dataloader)` or
  `for batch in self.eval_dataloader`, and callers pass an actual
  `torch.utils.data.DataLoader` (which is `Iterable`, not itself an
  `Iterator` — it has no `__next__`). The annotation was simply too
  narrow for what the code (and its real caller) actually needs.
  Changed both to `Iterable[Any]`.
- `ats/cli/train.py` — `trainer = DiffusionTrainer(...)` in one branch
  and `trainer = Trainer(...)` in the other; mypy inferred `trainer`'s
  type from the first (`if`) branch and flagged the second. Added an
  explicit `trainer: Trainer | DiffusionTrainer` annotation before the
  `if`.
- `ats/cli/finetune.py` — `base_module.config = {"tie_word_embeddings":
  ...}` deliberately and temporarily swaps `ATSTransformer.config`
  (normally a `ModelConfig`) for a dict-like shim, because `peft`'s
  `merge_and_unload()` expects `model.config.get(...)`-style access.
  This is a real, intentional type violation for the duration of one
  call (restored in a `finally` block right after), not something to
  "properly type" — added a scoped `# type: ignore[assignment]` with a
  comment explaining why.

### `pytest tests/` (was failing as a knock-on effect of the above)

No test logic changes were needed beyond the mamba-conv-padding one
already made in the prior patch round (`tests/test_model.py`'s
`test_mamba_chunked_scan_matches_naive_sequential_reference`, updated
to reproduce the new left-only-padding conv). The tests were "failing"
in the sense that the earlier `ruff --fix` pass and mypy narrowing
edits both had to leave runtime behavior unchanged for them to keep
passing — verified: **157 passed, 8 skipped** (skips are the
Triton/CUDA-only tests in `tests/test_triton.py`, expected in a
CPU-only environment) before, during, and after every fix in this
round.

### `config-validate` (was skipped, now would run and pass)

Ran the job's actual validation script locally against every file under
`configs/`: all 8 (`debug.yaml`, `125m.yaml`, `350m.yaml`, `1b.yaml`,
`3b.yaml`, `7b.yaml`, `14b.yaml`, `70b.yaml`) load and validate
successfully via `ATSConfig(**yaml.safe_load(...))`.

## Final verification

```
$ ruff check .
All checks passed!

$ ruff format --check .
63 files already formatted

$ mypy ats --ignore-missing-imports
Success: no issues found in 49 source files

$ pytest tests/ --tb=short -q
157 passed, 8 skipped in 1.50s
```
