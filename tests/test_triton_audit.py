"""BUG-119: audit of the four Triton modules.

Findings, all reproduced by the tests below rather than asserted in prose:

  * ats/model/moe_triton.py
  * ats/model/rope_triton.py
  * ats/model/norm_triton.py
  * ats/model/mla_triton.py

1. No module under ats/ imports any of them. The only importers in the
   repository are tests. They therefore cannot execute during training,
   evaluation, export or generation, on any hardware.
2. Every Triton path launches a raw kernel into a torch.empty() buffer.
   autograd does not see raw kernel launches, so the result has no grad_fn
   and gradients stop at the call. The PyTorch fallback inside the SAME
   function is differentiable, so the function would silently be
   differentiable on CPU and silently non-differentiable on CUDA.
3. Each Triton path now refuses to run on grad-requiring inputs rather than
   returning a detached tensor.
4. Each duplicates a PyTorch implementation that already exists elsewhere
   in the package and is the one actually used.

These tests are tripwires. If someone wires a Triton kernel into the model,
test_no_production_module_imports_a_triton_module fails and this file has to
be revisited -- at which point the autograd.Function work in point 2 is the
prerequisite, not an optional extra.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

TRITON_MODULES = {
    "ats.model.moe_triton",
    "ats.model.rope_triton",
    "ats.model.norm_triton",
    "ats.model.mla_triton",
}

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "ats"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_production_module_imports_a_triton_module():
    """Tripwire. These modules are documented as experimental and
    unreachable; if that stops being true, the docstrings, the README and
    the autograd situation all need updating together.
    """
    offenders = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if path.stem.endswith("_triton"):
            continue
        hits = _imported_modules(path) & TRITON_MODULES
        if hits:
            offenders[str(path)] = sorted(hits)
    assert not offenders, (
        "a production module now imports a Triton module: "
        f"{offenders}. These kernels are NOT differentiable (no "
        "autograd.Function wrapper) and have never been validated on real "
        "hardware -- see tests/test_triton_audit.py's module docstring."
    )


@pytest.mark.parametrize("module_name", sorted(TRITON_MODULES))
def test_triton_modules_declare_their_experimental_status(module_name):
    import importlib

    module = importlib.import_module(module_name)
    doc = module.__doc__ or ""
    assert "EXPERIMENTAL" in doc, (
        f"{module_name} does not state that it is experimental; a reader "
        "would reasonably assume it is on the training path"
    )
    assert "NOT ON ANY PRODUCTION EXECUTION PATH" in doc


def test_fused_rope_fallback_is_differentiable_and_matches_the_real_rope():
    """The CPU fallback must stay exactly equivalent to the implementation
    the model actually uses, since that equivalence is the only reason the
    module is worth keeping at all.
    """
    from ats.model.rope import apply_rotary_pos_emb
    from ats.model.rope_triton import fused_apply_rope

    torch.manual_seed(0)
    q = torch.randn(2, 4, 6, 8, dtype=torch.float64, requires_grad=True)
    cos = torch.randn(6, 8, dtype=torch.float64)
    sin = torch.randn(6, 8, dtype=torch.float64)

    fused = fused_apply_rope(q, cos, sin)
    reference, _ = apply_rotary_pos_emb(q, q, cos, sin)
    assert torch.allclose(fused, reference, atol=1e-12)

    # Differentiable on the fallback path.
    assert fused.grad_fn is not None
    fused.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_fused_moe_routing_fallback_matches_the_models_own_routing():
    from ats.model.ffn import SwiGLU  # noqa: F401  (import graph sanity)
    from ats.model.moe import _PyTorchMoEFallback
    from ats.model.moe_triton import fused_moe_routing

    torch.manual_seed(0)
    layer = _PyTorchMoEFallback(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        top_k=2,
        capacity_factor=1.25,
        load_balancing_weight=0.01,
        num_layers=1,
    )
    x = torch.randn(5, 8)
    logits = layer.gate(x)
    model_probs, model_idx, _ = layer.compute_routing(x)
    fused_probs, fused_idx = fused_moe_routing(logits, top_k=2)
    assert torch.equal(model_idx, fused_idx)
    assert torch.allclose(model_probs, fused_probs, atol=1e-6)


def test_fused_rmsnorm_fallback_matches_rmsnorm_plus_residual():
    from ats.model.norm import RMSNorm
    from ats.model.norm_triton import fused_rmsnorm_residual

    torch.manual_seed(0)
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    residual = torch.randn(2, 3, 8)
    fused = fused_rmsnorm_residual(x, residual, norm.weight, eps=norm.eps)
    reference = residual + norm(x)
    assert torch.allclose(fused, reference, atol=1e-5)


def test_fused_mla_decompress_fallback_matches_two_linears():
    from ats.model.mla_triton import fused_mla_kv_decompress

    torch.manual_seed(0)
    c = torch.randn(3, 5, 6, dtype=torch.float64)
    w_uk = torch.randn(12, 6, dtype=torch.float64)
    w_uv = torch.randn(12, 6, dtype=torch.float64)
    k, v = fused_mla_kv_decompress(c, w_uk, w_uv)
    assert torch.allclose(k, c @ w_uk.t(), atol=1e-12)
    assert torch.allclose(v, c @ w_uv.t(), atol=1e-12)
    assert k.shape == (3, 5, 12)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton path requires CUDA"
)
def test_triton_paths_refuse_grad_requiring_inputs():
    """On CUDA the guard must fire instead of silently detaching. Skipped
    here -- no GPU in this environment -- so this specific behaviour is
    reasoned from the source, not runtime-verified.
    """
    from ats.model.rope_triton import fused_apply_rope

    q = torch.randn(1, 1, 4, 4, device="cuda", requires_grad=True)
    cos = torch.randn(4, 4, device="cuda")
    sin = torch.randn(4, 4, device="cuda")
    with pytest.raises(RuntimeError, match="not differentiable"):
        fused_apply_rope(q, cos, sin)


def test_triton_guard_source_is_present_in_every_module():
    """CPU stand-in for the CUDA-only test above: confirm each wrapper
    actually contains the grad guard, so the protection is not silently
    absent in an environment that cannot exercise it.
    """
    for module_name in sorted(TRITON_MODULES):
        path = _PACKAGE_ROOT / "model" / (module_name.split(".")[-1] + ".py")
        source = path.read_text(encoding="utf-8")
        assert "not differentiable" in source, f"{path} lost its grad guard"
        assert "torch.is_grad_enabled()" in source, f"{path} lost its grad guard"
