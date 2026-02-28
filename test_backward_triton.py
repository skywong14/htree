"""End-to-end validation: compare Triton backward kernels against naive backward."""
import sys
import time
import logging
import torch

logging.getLogger("src.parallel").setLevel(logging.WARNING)

from src.naive import htree_forward_naive_autograd
from src.backward import htree_forward_triton_autograd

CR = 16
TK = 128
MTN = CR * TK
K = 32
V = 32
H = 4
H_kv = 2


def _make_inputs(B, T, device="cuda", seed=42, requires_grad=True):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32, requires_grad=requires_grad)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float32, requires_grad=requires_grad)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float32, requires_grad=requires_grad)
    return q, k, v


def test_forward_match():
    B, T = 1, 64
    q, k, v = _make_inputs(B, T, requires_grad=False)
    out_naive = htree_forward_naive_autograd(q.double(), k.double(), v.double(), CR, MTN, TK)
    out_triton = htree_forward_triton_autograd(q, k, v, CR, MTN, TK)
    diff = (out_naive.float() - out_triton.float()).abs()
    max_diff = diff.max().item()
    ok = max_diff < 5e-3
    print(f"[forward match] max diff = {max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_bottom_only():
    B, T = 1, 64
    q, k, v = _make_inputs(B, T, seed=7)

    q64 = q.detach().double().requires_grad_(True)
    k64 = k.detach().double().requires_grad_(True)
    v64 = v.detach().double().requires_grad_(True)
    out_naive = htree_forward_naive_autograd(q64, k64, v64, CR, MTN, TK)
    out_naive.sum().backward()
    dq_ref = q64.grad.float()
    dk_ref = k64.grad.float()
    dv_ref = v64.grad.float()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_triton = htree_forward_triton_autograd(q2, k2, v2, CR, MTN, TK)
    out_triton.sum().backward()
    dq_tri, dk_tri, dv_tri = q2.grad, k2.grad, v2.grad

    dq_diff = (dq_ref - dq_tri).abs().max().item()
    dk_diff = (dk_ref - dk_tri).abs().max().item()
    dv_diff = (dv_ref - dv_tri).abs().max().item()

    tol = 5e-2
    ok = dq_diff < tol and dk_diff < tol and dv_diff < tol
    print(f"[backward bottom-only] dq={dq_diff:.2e} dk={dk_diff:.2e} dv={dv_diff:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward_2layer():
    B, T = 1, 2064
    q, k, v = _make_inputs(B, T, seed=0)

    print("  computing naive backward ...", flush=True)
    t0 = time.time()
    q64 = q.detach().double().requires_grad_(True)
    k64 = k.detach().double().requires_grad_(True)
    v64 = v.detach().double().requires_grad_(True)
    out_naive = htree_forward_naive_autograd(q64, k64, v64, CR, MTN, TK)
    out_naive.sum().backward()
    dq_ref = q64.grad.float()
    dk_ref = k64.grad.float()
    dv_ref = v64.grad.float()
    print(f"  naive done in {time.time()-t0:.1f}s", flush=True)

    print("  computing Triton backward ...", flush=True)
    t0 = time.time()
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    out_triton = htree_forward_triton_autograd(q2, k2, v2, CR, MTN, TK)
    out_triton.sum().backward()
    dq_tri, dk_tri, dv_tri = q2.grad, k2.grad, v2.grad
    print(f"  Triton done in {time.time()-t0:.1f}s", flush=True)

    dq_diff = (dq_ref - dq_tri).abs().max().item()
    dk_diff = (dk_ref - dk_tri).abs().max().item()
    dv_diff = (dv_ref - dv_tri).abs().max().item()

    tol = 5e-2
    ok = dq_diff < tol and dk_diff < tol and dv_diff < tol
    print(f"[backward 2-layer] dq={dq_diff:.2e} dk={dk_diff:.2e} dv={dv_diff:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("forward_match", test_forward_match()))
    results.append(("backward_bottom_only", test_backward_bottom_only()))
    results.append(("backward_2layer", test_backward_2layer()))

    print("\n===== Summary =====")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    sys.exit(0 if all_ok else 1)
