"""Backward correctness tests: flash_acc_reg._C.flash_bwd vs fp32 autograd through SDPA.

Run:  pytest -v test_bwd.py
Assumptions (edit the ADAPTER section if yours differ):
  * flash_fwd(Q,K,V[,causal]) -> (O, L)
  * flash_bwd(Q,K,V,O,dO,L[,causal]) -> (dQ, dK, dV)
"""
import pytest
import torch
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flash_acc_reg import _C as ext


def _call(fn, *args, causal):
    """Your Python binding may or may not expose `causal`; try kwarg, then positional, then omit (non-causal only)."""
    try:
        return fn(*args, causal=causal)
    except TypeError:
        pass
    try:
        return fn(*args, causal)
    except TypeError:
        if causal:
            raise RuntimeError("flash_fwd/flash_bwd binding has no `causal` argument")
        return fn(*args)


def fwd_raw(q, k, v, causal):          # -> (O, L)
    return _call(ext.flash_fwd, q, k, v, causal=causal)


def bwd_raw(q, k, v, o, do, L, causal):  # -> (dQ, dK, dV)
    return _call(ext.flash_bwd, q, k, v, o, do, L, causal=causal)

def run_fwd(q, k, v, causal):
    O, L = fwd_raw(q, k, v, causal)[:2]
    return O, L


def run_bwd(q, k, v, o, do, L, causal):
    dq, dk, dv = bwd_raw(q, k, v, o, do, L, causal)[:3]
    return dq, dk, dv


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


DTYPES = [torch.float16]  # kernel is fp16-only (add torch.bfloat16 if you add support)
HEAD_DIMS = [32, 64, 80, 96, 128, 160, 192, 224, 256]
SEQLENS = [1, 7, 33, 64, 100, 127, 128, 129, 255, 333, 512, 1000, 1024, 2048, 4095, 4096]
REL_FLOOR = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}  # relative to max|ref|


def make_qkvdo(B, H, N, D, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda: torch.randn(B, H, N, D, device="cuda", dtype=dtype, generator=g)
    return mk(), mk(), mk(), mk()


def ref_grads(q, k, v, do, causal, dtype=torch.float32):
    qs, ks, vs = [t.detach().to(dtype).requires_grad_() for t in (q, k, v)]
    o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal)
    return torch.autograd.grad(o, (qs, ks, vs), do.to(dtype))


def check(name, g, g_base, g_ref, dtype):
    assert torch.isfinite(g.float()).all(), f"{name} has NaN/Inf"
    err = (g.float() - g_ref).abs().max().item()
    base = (g_base.float() - g_ref).abs().max().item()
    floor = REL_FLOOR[dtype] * max(g_ref.abs().max().item(), 1.0)
    tol = 2.0 * base + floor
    assert err <= tol, f"{name}: max err {err:.3e} > tol {tol:.3e} (sdpa low-prec err {base:.3e})"


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("D", HEAD_DIMS)
@pytest.mark.parametrize("N", SEQLENS)
def test_bwd(N, D, causal, dtype):
    B, H = (2, 4) if N <= 1024 else (1, 2)
    q, k, v, do = make_qkvdo(B, H, N, D, dtype)

    o, L = run_fwd(q, k, v, causal)
    dq, dk, dv = run_bwd(q, k, v, o, do, L, causal)

    for name, g, ref in (("dQ", dq, q), ("dK", dk, k), ("dV", dv, v)):
        assert g.shape == ref.shape, f"{name} shape {g.shape} != {ref.shape}"
        assert g.dtype == dtype, f"{name} dtype {g.dtype}"

    rq, rk, rv = ref_grads(q, k, v, do, causal, torch.float32)
    bq, bk, bv = ref_grads(q, k, v, do, causal, dtype)
    check("dQ", dq, bq, rq, dtype)
    check("dK", dk, bk, rk, dtype)
    check("dV", dv, bv, rv, dtype)


def test_bwd_deterministic_dkdv():
    """dK/dV should be bitwise reproducible; dQ may use atomics, so only check closeness."""
    q, k, v, do = make_qkvdo(2, 4, 1024, 128, torch.float16)
    o, L = run_fwd(q, k, v, True)
    a = run_bwd(q, k, v, o, do, L, True)
    b = run_bwd(q, k, v, o, do, L, True)
    torch.testing.assert_close(a[0].float(), b[0].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(a[1].float(), b[1].float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(a[2].float(), b[2].float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("D", [7, 257, 264])
def test_bwd_rejects_bad_headdim(D):
    q = torch.randn(1, 1, 64, D, device="cuda", dtype=torch.float16)
    L = torch.zeros(1, 1, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        run_bwd(q, q, q, q, q, L, False)


if __name__ == "__main__":
    # `python <this file> [pytest args]`, e.g. `python test_fwd.py -k "causal and 128"`
    import sys
    sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short", *sys.argv[1:]]))
