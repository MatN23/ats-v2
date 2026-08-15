# CI Fixes Round 2 (commit 57ef7b98)

Starting point: commit `57ef7b98` ("CI fixes: lint, types, formatting, CLI
tweaks"), whose own `CI_FIXES.md` claims `ruff`/`ruff format`/`mypy`/`pytest`
all pass cleanly (157 passed, 8 skipped). This round verifies that claim
against the actual CI workflow rather than trusting it, since the task
states the GitHub Actions runners are still red.

I don't have direct access to the live GitHub Actions run logs from this
environment (no repo dashboard / API access to check-run output), so I
could not paste the literal CI log text. What I did instead: read
`.github/workflows/ci.yml` to get the **exact** commands, tool-install
method, and Python versions CI uses, and reproduced them as faithfully as
possible locally, byte-for-byte matching the workflow file rather than
re-running the commands from the previous `CI_FIXES.md` (which, as it
turns out, weren't quite the real CI commands — see below).

## What was actually different from the previous verification

Three environment/invocation gaps between how the previous round verified
things and what CI actually does:

1. **`ruff` and `mypy` are installed unpinned** (`pip install ruff`,
   `pip install mypy` — no version pin anywhere in the workflow or in
   `requirements.txt`/`requirements-dev.txt`). CI always gets whatever is
   latest on PyPI *at the time the workflow runs*. The previous round's
   mypy was pinned to whatever had been `pip install`ed earlier in that
   session (2.3.0); I upgraded to the actual current latest (2.3.1) before
   touching anything.
2. **The `test` job runs `pytest -v --cov=ats --cov-report=xml
   --cov-report=term-missing` from the repo root — not `pytest tests/`.**
   The previous round's `CI_FIXES.md` verification used `pytest tests/`
   (and, more importantly, `python3 -m pytest`, not the bare `pytest`
   console-script CI actually invokes). Both of these turned out to
   matter — see the real bug below.
3. **`pip install -e .` in a genuinely clean environment pulls in the
   *full* dependency tree** (`torch`, `deepspeed`, `transformers`,
   `datasets`, etc. — all built successfully this round, unlike earlier
   sessions where some of these silently failed to build and were
   skipped). This exposed one additional latent mypy error that only
   shows up once `transformers` is actually installed (see below).

## Root cause #1 (real, previously undetected): `pytest` fails, `python -m pytest` doesn't

**The actual CI command (`pytest -v --cov=ats ...`, run via the bare
`pytest` console-script) failed with:**

```
tests/test_data.py::test_preprocess_packed_output_round_trips ______________

    def test_preprocess_packed_output_round_trips(tmp_path):
        ...
>       import preprocess as preprocess_module
E       ModuleNotFoundError: No module named 'preprocess'

tests/test_data.py:244: ModuleNotFoundError
=========== 1 failed, 151 passed, 13 skipped, 42 warnings in 42.27s ==========
```

**Why the previous round's local verification missed this:** every local
check in the prior round used `python3 -m pytest tests/ ...`. Running
Python with `-m` always prepends the current working directory to
`sys.path` — a standard CPython behavior, unrelated to pytest itself. That
silently made the repo root (and therefore `preprocess.py`, a top-level
script) importable, masking the bug. CI's workflow step is
`run: pytest -v --cov=ats ...` — the bare console-script entry point, which
does **not** get that same automatic cwd insertion.

**Why it fails specifically with `pip install -e .` present:** confirmed
by inspecting the actual installed editable-install shim:

```python
# /usr/local/lib/python3.12/dist-packages/__editable___ats_v2_2_0_0_finder.py
MAPPING: dict[str, str] = {"ats": "/home/claude/ats-v2-round2/ats"}
```

Modern `pip install -e .` (PEP 660, via setuptools' finder-based editable
installs — this is now the default, not the old `.egg-link` /
`.pth`-file-that-appends-the-whole-repo-root style) only maps the
**declared package** (`ats`) into an import hook. It does not add the
repo root itself to `sys.path`. So `import ats...` now works (via the
hook) but `import preprocess` — a script that lives at the repo root and
isn't part of the `ats` package — never resolves, regardless of whether
pytest's own import-mode machinery would otherwise help (it doesn't
here either: `tests/conftest.py` exists but `tests/` itself, not the repo
root, is what pytest inserts on `sys.path` for test-file imports, since
insertion is based on walking up from each test file until an
`__init__.py`-less directory is found — that stops at `tests/`, not one
level higher).

**Confirmed the mechanism directly, both before and after the fix:**

```
$ pytest tests/test_data.py::test_preprocess_packed_output_round_trips -v
FAILED ... ModuleNotFoundError: No module named 'preprocess'

$ python3 -m pytest tests/test_data.py::test_preprocess_packed_output_round_trips -v
1 passed
```

**Fix (`tests/test_data.py`):** stopped relying on `sys.path` state
entirely. Added a small helper, `_load_preprocess_module()`, that loads
`preprocess.py` via `importlib.util.spec_from_file_location` using an
absolute path computed from `Path(__file__)` — this works identically
regardless of cwd, invocation mode (`pytest` vs `python -m pytest`), or
installation method, since it never touches `sys.path`:

```python
def _load_preprocess_module() -> types.ModuleType:
    preprocess_path = Path(__file__).resolve().parent.parent / "preprocess.py"
    spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

and replaced the one call site (`import preprocess as preprocess_module`)
with `preprocess_module = _load_preprocess_module()`. Verified
`preprocess.py` has a proper `if __name__ == "__main__":` guard, so
`exec_module` triggers no CLI side effects. The test's existing
`preprocess_module.Tokenizer = _FakeTok` monkeypatch still works exactly
as before, since it patches the same live module object `preprocess()`
executes against, regardless of how that module object was constructed.

**Re-verified both ways after the fix:**
```
$ pytest tests/test_data.py::test_preprocess_packed_output_round_trips -v
1 passed

$ python3 -m pytest tests/test_data.py::test_preprocess_packed_output_round_trips -v
1 passed
```

## Root cause #2 (real, previously undetected): a genuinely-installed `transformers` exposes a latent mypy error

With the fully successful `pip install -e .` this round (transformers,
deepspeed, etc. all actually installed, unlike earlier sessions where
some of these silently failed to build), `mypy ats --ignore-missing-imports`
reported:

```
ats/data/tokenizer.py:84: error: Incompatible return value type
(got "str | list[str]", expected "str")  [return-value]
```

**Root cause:** `Tokenizer.decode()` is declared `-> str`, and its HF
backend branch was `return self._hf_tok.decode(real_ids,
skip_special_tokens=True)`. `self._hf_tok` is a `transformers`
`PreTrainedTokenizerBase`-family object; its `decode()` method is typed
with a batch-decode overload that can return `str | list[str]` depending
on input shape. When `transformers` isn't installed at all,
`--ignore-missing-imports` makes mypy treat `self._hf_tok` as `Any`,
silently permitting the mismatch — that's exactly what happened in every
earlier verification, since `transformers` never successfully built in
those sessions. With it actually present, mypy sees the real signature
and correctly flags it.

**Fix (`ats/data/tokenizer.py`):** `real_ids` here is always a flat
`list[int]` (never a nested/batched input), so the runtime result is
always a plain `str` — the mismatch is a real typing gap, not a real bug.
Rather than a blind `# type: ignore`, added a runtime `assert
isinstance(decoded, str)` before returning: this satisfies mypy through
normal narrowing (no ignore comment needed at all) *and* adds a genuine
runtime safety net if that assumption ever stops holding (e.g., a future
change accidentally passes a nested list, which would now raise a clear
`AssertionError` instead of silently returning the wrong type to a
caller expecting `str`).

## Everything else: re-verified, not re-broken

Went through the previous round's specific concerns one at a time:

- **The mamba conv-padding test** (`test_mamba_chunked_scan_matches_naive_sequential_reference`)
  — already fixed correctly in the prior round; re-ran it individually,
  still passes.
- **`MoELayer.__init__` default `ep_size=1`** — confirmed present
  (`ats/model/moe.py`); tests constructing `MoELayer(...)` directly
  without `ep_size` pass.
- **MoD-forward mock/patch signature tests** — pass; no change needed.
- **Circular imports from the new `QuantizationMode` import** in
  `ats/model/{ffn,attention,mla,moe}.py` — none found; `python -c "import
  ats.model.transformer"` and the full test suite both succeed, and mypy
  (which would surface an import cycle as an error) is clean.
- **Missing test dependency in `requirements.txt`** — the one genuinely
  missing thing wasn't a *package* dependency, it was the `sys.path`
  issue above (root cause #1).
- **The 5 additional test skips this round** (`test_moe_gating_weights_sum_to_one`,
  `test_moe_expert_uses_quantization`, `test_moe_fallback_expert_utilization_sums_to_one`,
  `test_moe_expert_down_proj_gets_depth_scaled_residual_init`,
  `test_moe_layer_transformer_block_experts_get_residual_init`) are not a
  regression — each is self-documenting (`SKIPPED (deepspeed is
  installed; this test targets the PyTorch fallback ...)`). They're
  designed to exercise the non-deepspeed fallback code path and correctly
  skip themselves when deepspeed is actually present, which it now is
  (unlike in earlier sessions where deepspeed's build silently failed).
  Total test count is unchanged (165 collected either way): 152 passed +
  13 skipped this round vs. 157 passed + 8 skipped in a deepspeed-less
  environment — the same 165 tests, just a different pass/skip split
  depending on what's actually installed. The `config-validate` job
  (`needs: test`) would now run and pass — verified locally by running
  its exact validation script; all 8 configs under `configs/` validate.

## Why the previous fix commit's `CI_FIXES.md` didn't hold up in CI

Not because its actual code fixes were wrong (they weren't — mypy and
ruff genuinely were clean, and 157 real tests genuinely did pass) but
because its *verification method* didn't match what CI actually runs in
two specific ways: `python -m pytest` instead of bare `pytest` (masking
root cause #1), and a local environment where `transformers`/`deepspeed`
hadn't actually finished installing (masking root cause #2, and changing
which subset of tests ran at all). Both gaps were silent — nothing in
that round's output looked wrong, because from the *inside* of that
environment, nothing was.

## Final verification (exact CI commands, bare console-script binaries)

```
$ which pytest ruff mypy
/usr/local/bin/pytest
/usr/local/bin/ruff
/usr/local/bin/mypy

$ ruff check .
All checks passed!

$ ruff format --check .
64 files already formatted

$ mypy ats --ignore-missing-imports
Success: no issues found in 49 source files

$ pytest -v --cov=ats --cov-report=xml --cov-report=term-missing
...
================= 152 passed, 13 skipped, 42 warnings in 8.29s =================
```

Also re-ran the `config-validate` job's exact validation script: all 8
config files under `configs/` load and validate successfully.

I also attempted a fully from-scratch virtualenv (`python -m venv` with
no `--system-site-packages`) per Step 3, but the sandbox ran out of disk
space partway through re-downloading `torch` a second time (a ~2GB+
package). After freeing space, I used `--system-site-packages` to reuse
the already-verified-fresh `torch`/`transformers`/`deepspeed` install
while still confirming tool versions and the package's own editable
install resolve correctly. The one thing that check *couldn't* isolate
(a separate `pytest` console-script binary specifically) doesn't matter
for root cause #1's fix, since that fix (loading `preprocess.py` by
absolute path) is invocation-mode-independent by construction — verified
directly under both `pytest` and `python -m pytest` above.
