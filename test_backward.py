"""
Backward correctness test for HTree naive autograd implementation.
Uses torch.autograd.gradcheck to verify analytical gradients against
numerical finite-difference gradients.
"""
import sys
import time
import torch

from src.naive import htree_forward_naive, htree_forward_naive_autograd


def test_forward_consistency():
    """Verify autograd wrapper produces same output as original forward."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, H_kv, K, V = 1, 32, 2, 1, 8, 8
    CR, TK = 4, 4
    MTN = CR * TK

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float64)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float64)

    out_ref = htree_forward_naive(q, k, v, CR, MTN, TK)
    out_ag = htree_forward_naive_autograd(q, k, v, CR, MTN, TK)

    diff = (out_ref.double() - out_ag.double()).abs()
    max_diff = diff.max().item()
    print(f"[forward consistency] max diff = {max_diff:.2e}", end="  ")
    # Original forward uses float32 internally; autograd version uses input dtype (float64).
    # ~1e-3 gap is expected from float32 vs float64.
    ok = max_diff < 2e-3
    print("PASS" if ok else "FAIL")
    return ok


def test_gradcheck_2layer():
    """gradcheck with 2-layer tree (T=32, CR=4, TOP_K=4, max_top=16)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, H_kv, K, V = 1, 32, 2, 1, 8, 8
    CR, TK = 4, 4
    MTN = CR * TK

    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float64, requires_grad=True)

    def fn(q_, k_, v_):
        return htree_forward_naive_autograd(q_, k_, v_, CR, MTN, TK)

    print("[gradcheck 2-layer] running ... ", end="", flush=True)
    t0 = time.time()
    ok = torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3,
                                  raise_exception=False)
    dt = time.time() - t0
    print(f"{'PASS' if ok else 'FAIL'}  ({dt:.1f}s)")

    if not ok:
        print("  Re-running with raise_exception=True for diagnostics:")
        try:
            torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3,
                                     raise_exception=True)
        except Exception as e:
            print(f"  {e}")
    return ok


def test_gradcheck_bottom_only():
    """gradcheck with 1-layer tree (T<=max_top, no Top-K selection)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, H_kv, K, V = 1, 16, 2, 1, 8, 8
    CR, TK = 4, 4
    MTN = CR * TK  # 16, so T=16 <= 16 => 1 layer

    torch.manual_seed(7)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float64, requires_grad=True)

    def fn(q_, k_, v_):
        return htree_forward_naive_autograd(q_, k_, v_, CR, MTN, TK)

    print("[gradcheck bottom-only] running ... ", end="", flush=True)
    t0 = time.time()
    ok = torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3,
                                  raise_exception=False)
    dt = time.time() - t0
    print(f"{'PASS' if ok else 'FAIL'}  ({dt:.1f}s)")

    if not ok:
        print("  Re-running with raise_exception=True for diagnostics:")
        try:
            torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3,
                                     raise_exception=True)
        except Exception as e:
            print(f"  {e}")
    return ok


def test_manual_grad_check():
    """Quick manual gradient verification: compare analytical vs numerical for a few elements."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, H_kv, K, V = 1, 32, 2, 1, 8, 8
    CR, TK = 4, 4
    MTN = CR * TK

    torch.manual_seed(1)
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float64, requires_grad=True)

    out = htree_forward_naive_autograd(q, k, v, CR, MTN, TK)
    loss = out.sum()
    loss.backward()

    eps = 1e-5
    max_rel = 0.0
    n_checked = 0
    for param, name in [(q, "q"), (k, "k"), (v, "v")]:
        ag = param.grad.clone()
        flat = param.data.view(-1)
        indices = torch.randperm(flat.numel())[:min(20, flat.numel())]
        for idx in indices:
            orig = flat[idx].item()
            flat[idx] = orig + eps
            out_p = htree_forward_naive_autograd(q, k, v, CR, MTN, TK).sum().item()
            flat[idx] = orig - eps
            out_m = htree_forward_naive_autograd(q, k, v, CR, MTN, TK).sum().item()
            flat[idx] = orig
            num_grad = (out_p - out_m) / (2 * eps)
            ana_grad = ag.view(-1)[idx].item()
            denom = max(abs(num_grad), abs(ana_grad), 1e-12)
            rel = abs(num_grad - ana_grad) / denom
            max_rel = max(max_rel, rel)
            n_checked += 1

    print(f"[manual grad check] {n_checked} elements, max rel error = {max_rel:.2e}", end="  ")
    ok = max_rel < 1e-3
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("forward_consistency", test_forward_consistency()))
    results.append(("gradcheck_bottom_only", test_gradcheck_bottom_only()))
    results.append(("gradcheck_2layer", test_gradcheck_2layer()))
    results.append(("manual_grad_check", test_manual_grad_check()))

    print("\n===== Summary =====")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)
