#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: Bitonic Sort Top-K  vs  Radix Select Top-K  vs  Triton Built-in Top-K

Configurations:
  Algorithms : bitonic, radix, triton (tl.topk / tl.sort)
  Strategies : direct (top512 in 8192), streaming (16× top512 in 1024)
  Payloads   : 0, 1, 8  int32 values carried alongside each element

Usage:
    python topk_test/benchmark.py
"""

import math
import time
import torch
import triton

from bitonic_topk import (
    bitonic_topk_direct_kernel,
    bitonic_topk_streaming_kernel,
)
from radix_topk import (
    radix_topk_direct_kernel,
    radix_topk_streaming_v2_kernel,
)
from triton_topk import (
    triton_topk_direct_kernel,
    triton_topk_streaming_kernel,
)

# ── Parameters ──────────────────────────────────────────────
N = 8192
K = 512
NEG_INF = -1.0e10
BATCH = 512         # number of independent top-K problems (grid size)
WARMUP = 50
REPS = 200

N_DIMS_N = int(math.log2(N))       # 13
N_DIMS_K = int(math.log2(K))       # 9
N_DIMS_2K = int(math.log2(2 * K))  # 10
LOG_N = N_DIMS_N                    # bits needed for index in [0, N)
NUM_BATCHES = N // K                # 16
TWO_K = 2 * K


def make_data(batch, n, n_pay, device="cuda"):
    """Create random input data + payloads."""
    inp = torch.randn(batch, n, device=device, dtype=torch.float32)
    pay1 = torch.randint(0, 100000, (batch, n), device=device, dtype=torch.int32) if n_pay >= 1 else torch.empty(1, device=device, dtype=torch.int32)
    pay8 = torch.randint(0, 100000, (batch, n, 8), device=device, dtype=torch.int32) if n_pay >= 8 else torch.empty(1, device=device, dtype=torch.int32)
    return inp, pay1, pay8


def alloc_outputs(batch, k, n_pay, device="cuda"):
    out_val = torch.empty(batch, k, device=device, dtype=torch.float32)
    out_idx = torch.empty(batch, k, device=device, dtype=torch.int32)
    out_pay1 = torch.empty(batch, k, device=device, dtype=torch.int32) if n_pay >= 1 else torch.empty(1, device=device, dtype=torch.int32)
    out_pay8 = torch.empty(batch, k, 8, device=device, dtype=torch.int32) if n_pay >= 8 else torch.empty(1, device=device, dtype=torch.int32)
    return out_val, out_idx, out_pay1, out_pay8


def reference_topk(inp, k):
    """PyTorch reference top-K (returns values, indices)."""
    return torch.topk(inp, k, dim=-1, largest=True, sorted=True)


# ── Wrapper functions ───────────────────────────────────────

def run_bitonic_direct(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    grid = (batch,)
    bitonic_topk_direct_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        N=N, K=K, N_DIMS=N_DIMS_N, LOG_N=LOG_N, N_PAY=n_pay, NEG_INF=NEG_INF,
    )
    return out_val, out_idx


def run_bitonic_streaming(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    grid = (batch,)
    bitonic_topk_streaming_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        N=N, K=K, N_DIMS_K=N_DIMS_K, N_DIMS_2K=N_DIMS_2K,
        LOG_N=LOG_N, N_PAY=n_pay, NEG_INF=NEG_INF, NUM_BATCHES=NUM_BATCHES,
    )
    return out_val, out_idx


def run_radix_direct(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    grid = (batch,)
    radix_topk_direct_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        N=N, K=K, N_PAY=n_pay,
    )
    return out_val, out_idx


def run_radix_streaming(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    scratch_val = torch.empty(batch, K, device=inp.device, dtype=torch.float32)
    scratch_idx = torch.empty(batch, K, device=inp.device, dtype=torch.int32)
    scratch_pay1 = torch.empty(batch, K, device=inp.device, dtype=torch.int32) if n_pay >= 1 else torch.empty(1, device=inp.device, dtype=torch.int32)
    grid = (batch,)
    radix_topk_streaming_v2_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        scratch_val, scratch_idx, scratch_pay1,
        N=N, K=K, N_PAY=n_pay, NEG_INF=NEG_INF,
        NUM_BATCHES=NUM_BATCHES, TWO_K=TWO_K,
    )
    return out_val, out_idx


def run_triton_direct(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    grid = (batch,)
    triton_topk_direct_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        N=N, K=K, LOG_N=LOG_N, N_PAY=n_pay, NEG_INF=NEG_INF,
    )
    return out_val, out_idx


def run_triton_streaming(inp, pay1, pay8, n_pay):
    batch = inp.shape[0]
    out_val, out_idx, out_pay1, out_pay8 = alloc_outputs(batch, K, n_pay)
    grid = (batch,)
    triton_topk_streaming_kernel[grid](
        inp, pay1, pay8, out_val, out_idx, out_pay1, out_pay8,
        N=N, K=K, N_DIMS_2K=N_DIMS_2K,
        LOG_N=LOG_N, N_PAY=n_pay, NEG_INF=NEG_INF, NUM_BATCHES=NUM_BATCHES,
    )
    return out_val, out_idx


# ── Correctness check ──────────────────────────────────────

def check_correctness(name, run_fn, inp, pay1, pay8, n_pay, ref_vals, ref_indices):
    """Verify that the kernel produces the same top-K set as PyTorch."""
    out_val, out_idx = run_fn(inp, pay1, pay8, n_pay)
    torch.cuda.synchronize()

    # Compare sets (order may differ)
    ref_set = set()
    test_set = set()
    # Check first few batch entries
    n_check = min(4, inp.shape[0])
    all_ok = True
    for b in range(n_check):
        ref_sorted, _ = torch.sort(ref_vals[b], descending=True)
        test_sorted, _ = torch.sort(out_val[b], descending=True)
        if not torch.allclose(ref_sorted, test_sorted, atol=1e-5, rtol=1e-5):
            # Allow small differences due to bit-packing truncation
            ref_top = set(ref_indices[b].cpu().tolist())
            test_top = set(out_idx[b].cpu().tolist())
            overlap = len(ref_top & test_top)
            pct = overlap / K * 100
            if pct < 99.0:
                print(f"  ⚠  {name} batch {b}: {pct:.1f}% overlap with reference")
                all_ok = False
    if all_ok:
        print(f"  ✓  {name}: correctness OK")
    return all_ok


# ── Benchmark ──────────────────────────────────────────────

def benchmark_fn(name, run_fn, inp, pay1, pay8, n_pay):
    """Benchmark a kernel and return median latency in ms."""
    # Warmup (also triggers JIT compilation)
    print(f"  Compiling {name} ...", end="", flush=True)
    t0 = time.time()
    for _ in range(2):
        run_fn(inp, pay1, pay8, n_pay)
        torch.cuda.synchronize()
    compile_time = time.time() - t0
    print(f" ({compile_time:.1f}s)")

    # Additional warmup
    for _ in range(WARMUP):
        run_fn(inp, pay1, pay8, n_pay)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(REPS):
        start.record()
        run_fn(inp, pay1, pay8, n_pay)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    return median, p10, p90


# ── Main ───────────────────────────────────────────────────

def main():
    device = "cuda"
    print("=" * 72)
    print(f"Top-K Benchmark: K={K} from N={N},  BATCH={BATCH}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)

    kernels = [
        ("bitonic_direct",    run_bitonic_direct),
        ("bitonic_streaming", run_bitonic_streaming),
        ("radix_direct",      run_radix_direct),
        ("radix_streaming",   run_radix_streaming),
        ("triton_direct",     run_triton_direct),
        ("triton_streaming",  run_triton_streaming),
    ]

    payload_configs = [0, 1, 8]

    results = {}

    for n_pay in payload_configs:
        print(f"\n{'─'*72}")
        print(f"Payload = {n_pay} × int32")
        print(f"{'─'*72}")

        inp, pay1, pay8 = make_data(BATCH, N, n_pay, device)

        # Reference
        ref_vals, ref_indices = reference_topk(inp, K)

        for name, run_fn in kernels:
            key = f"{name}_pay{n_pay}"
            try:
                # Correctness
                check_correctness(name, run_fn, inp, pay1, pay8, n_pay, ref_vals, ref_indices)

                # Benchmark
                median, p10, p90 = benchmark_fn(name, run_fn, inp, pay1, pay8, n_pay)
                results[key] = (median, p10, p90)
                print(f"  ▸ {name:25s}  median={median:.3f} ms  (p10={p10:.3f}, p90={p90:.3f})")
            except Exception as e:
                print(f"  ✗  {name}: FAILED – {e}")
                results[key] = None

    # ── Summary table ────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY (median latency in ms)")
    print(f"{'='*72}")
    header = f"{'Kernel':25s}"
    for np in payload_configs:
        header += f" | pay={np:1d}"
    print(header)
    print("-" * len(header))
    for name, _ in kernels:
        row = f"{name:25s}"
        for np in payload_configs:
            key = f"{name}_pay{np}"
            r = results.get(key)
            if r is not None:
                row += f" | {r[0]:7.3f}"
            else:
                row += f" |    FAIL"
        print(row)
    print("=" * 72)


if __name__ == "__main__":
    main()
