"""
Speed benchmark: HTree Triton backward vs NSA backward.

Measures average backward-pass latency using CUDA events.
Both methods run the full forward+backward pipeline; only the backward
portion is timed (forward is included in warmup to populate caches).

Usage examples:
    # Default (both htree and nsa, T=12000, K=V=32)
    python speed_test_bwd.py

    # Specific GPU, larger dims
    python speed_test_bwd.py --gpu 7 --k-dim 64 --v-dim 64

    # Only HTree, custom tree params
    python speed_test_bwd.py --only htree --htree-top-k 128 --seq-len 8000

    # Profile mode (use with ncu)
    ncu --profile-from-start off -o bwd_profile python speed_test_bwd.py --profile

Requirements: CUDA + Triton, torch, and `fla` (for NSA helpers).
"""

import argparse
import math
import os
import sys
from typing import Callable, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
#  Terminal colours
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _header(t: str) -> str:   return _c(t, "1;36")
def _ok(t: str) -> str:       return _c(t, "1;32")
def _warn(t: str) -> str:     return _c(t, "1;33")
def _dim(t: str) -> str:      return _c(t, "2")
def _err(t: str) -> str:      return _c(t, "1;31")

def _sync() -> None:
    torch.cuda.synchronize()

# ---------------------------------------------------------------------------
#  Input construction
# ---------------------------------------------------------------------------

def _build_causal_blocks(B: int, T: int, H_kv: int, S: int,
                         block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = math.ceil(T / block_size)
    block_ids = torch.arange(T, device=device) // block_size
    offsets = torch.arange(-(S - 1), 1, device=device)
    indices = block_ids[:, None] + offsets[None, :]
    indices = torch.clamp(indices, min=0, max=num_blocks - 1)
    return indices.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, S).contiguous()


def _make_inputs(
    B: int, T: int, HQ: int, H_kv: int, K: int, V: int,
    S: int, block_size: int, dtype: torch.dtype, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H_kv, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H_kv, V, device=device, dtype=dtype, requires_grad=True)
    block_indices = _build_causal_blocks(B, T, H_kv, S, block_size, device)
    return q, k, v, block_indices

# ---------------------------------------------------------------------------
#  Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_bwd(
    forward_fn: Callable[..., torch.Tensor],
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    """Return (fwd_avg_ms, bwd_avg_ms) measured with CUDA events."""
    dO_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])

    # --- warmup (compile kernels, fill caches) ---
    for _ in range(max(warmup, 1)):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out = forward_fn(q, k, v)
        dO = torch.randn_like(out)
        out.backward(dO)
    _sync()

    # --- timed iterations ---
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end   = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end   = torch.cuda.Event(enable_timing=True)

    total_fwd_ms = 0.0
    total_bwd_ms = 0.0
    for _ in range(iters):
        q.grad = k.grad = v.grad = None
        dO = torch.randn(*dO_shape, device=q.device, dtype=q.dtype)

        fwd_start.record()
        out = forward_fn(q, k, v)
        fwd_end.record()

        bwd_start.record()
        out.backward(dO)
        bwd_end.record()

        _sync()
        total_fwd_ms += fwd_start.elapsed_time(fwd_end)
        total_bwd_ms += bwd_start.elapsed_time(bwd_end)

    return total_fwd_ms / iters, total_bwd_ms / iters

# ---------------------------------------------------------------------------
#  Profile mode
# ---------------------------------------------------------------------------

def _run_profile_mode(targets, q, k, v, warmup: int) -> None:
    print(_header("\n>>> PROFILE mode (backward) <<<"))
    print(_dim("Warmup — JIT + cache warm ..."))
    for name, fn in targets:
        print(f"  warming up {name} ({warmup} iters) ...")
        for _ in range(max(warmup, 1)):
            q.grad = k.grad = v.grad = None
            out = fn(q, k, v)
            out.backward(torch.randn_like(out))
        _sync()
        print(f"  {name} warmup done.")

    print(_ok("Starting CUDA profiler — 1 backward iteration ..."))
    torch.cuda.cudart().cudaProfilerStart()
    for _name, fn in targets:
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v)
        out.backward(torch.randn_like(out))
    _sync()
    torch.cuda.cudart().cudaProfilerStop()
    print(_ok("CUDA profiler stopped."))
    print(_header("========================================\n"))

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "CUDA required"
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid --gpu {args.gpu} "
                         f"(visible devices: {torch.cuda.device_count()})")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True

    q, k, v, block_indices = _make_inputs(
        B=args.batch, T=args.seq_len, HQ=args.hq, H_kv=args.h_kv,
        K=args.k_dim, V=args.v_dim, S=args.s_blocks,
        block_size=args.block_size, dtype=dtype, device=device,
    )
    scale = args.k_dim ** -0.5

    # ---- build forward callables (with grad) ----
    run_htree_flag = args.only in ("all", "htree")
    run_nsa_flag   = args.only in ("all", "nsa")

    htree_fn: Optional[Callable] = None
    nsa_fn:   Optional[Callable] = None

    if run_htree_flag:
        from src.backward import htree_forward_triton_autograd
        htree_fn = lambda q_, k_, v_: htree_forward_triton_autograd(
            q_, k_, v_,
            compression_rate=args.htree_compression_rate,
            max_top_nodes=args.htree_max_top_nodes,
            top_k_per_layer=args.htree_top_k,
            scale=scale,
        )

    if run_nsa_flag:
        from ref_nsa.nsa_parallel import parallel_nsa
        gate_cmp = torch.zeros(args.batch, args.seq_len, args.hq,
                               device=device, dtype=dtype)

        nsa_fn = lambda q_, k_, v_: parallel_nsa(
            q_, k_, v_,
            g_cmp=gate_cmp,
            block_indices=None,
            block_size=args.block_size,
            scale=scale,
            block_counts=args.s_blocks,
        )

    # ---- header ----
    dev_name = torch.cuda.get_device_name(args.gpu)
    print("\n" + _header("====== Speed Benchmark (backward) ======"))
    print(_dim(f"Device : {device} ({dev_name})"))
    print(_dim(f"dtype  : {dtype}"))
    print(_dim(f"B={args.batch}, T={args.seq_len}, HQ={args.hq}, "
               f"H_kv={args.h_kv}, K={args.k_dim}, V={args.v_dim}"))
    print(_dim(f"warmup={args.warmup}, iters={args.iters}"))
    if run_htree_flag:
        print(_dim(f"htree  : CR={args.htree_compression_rate}, "
                   f"TK={args.htree_top_k}, MTN={args.htree_max_top_nodes}"))
    if run_nsa_flag:
        print(_dim(f"NSA    : block_size={args.block_size}, "
                   f"S={args.s_blocks}"))
    print(_warn("Timing: forward + backward per iter; fwd/bwd split reported."))
    print()

    # ---- profile mode ----
    if args.profile:
        targets = []
        if htree_fn: targets.append(("htree", htree_fn))
        if nsa_fn:   targets.append(("nsa", nsa_fn))
        _run_profile_mode(targets, q, k, v, args.warmup)
        return

    # ---- benchmark ----
    results = {}

    if htree_fn is not None:
        print(f"  Benchmarking htree backward ...", end="", flush=True)
        try:
            fwd_ms, bwd_ms = benchmark_bwd(htree_fn, q, k, v,
                                           warmup=args.warmup, iters=args.iters)
            results["htree"] = (fwd_ms, bwd_ms)
            print(f"  {_ok('done')}")
        except Exception as e:
            print(f"  {_err('FAILED')}: {e}")

    if nsa_fn is not None:
        print(f"  Benchmarking NSA backward   ...", end="", flush=True)
        try:
            fwd_ms, bwd_ms = benchmark_bwd(nsa_fn, q, k, v,
                                           warmup=args.warmup, iters=args.iters)
            results["nsa"] = (fwd_ms, bwd_ms)
            print(f"  {_ok('done')}")
        except Exception as e:
            print(f"  {_err('FAILED')}: {e}")

    # ---- results ----
    if not results:
        print(_err("\nNo successful benchmarks."))
        return

    print()
    print(_header("Results (avg ms / iter):"))

    col_name = 12
    col_num  = 12
    hdr = f"  {'Method':<{col_name}} {'Forward':>{col_num}} {'Backward':>{col_num}} {'Total':>{col_num}} {'Bwd/Fwd':>{col_num}}"
    print(_dim(hdr))
    print(_dim("  " + "-" * (col_name + col_num * 4 + 3)))

    for name, (fwd_ms, bwd_ms) in results.items():
        total = fwd_ms + bwd_ms
        ratio = bwd_ms / fwd_ms if fwd_ms > 0 else float("inf")
        line = (f"  {name:<{col_name}} "
                f"{fwd_ms:>{col_num}.3f} "
                f"{_ok(f'{bwd_ms:>{col_num}.3f}')} "
                f"{total:>{col_num}.3f} "
                f"{ratio:>{col_num}.2f}x")
        print(line)

    if len(results) == 2:
        htree_bwd = results["htree"][1]
        nsa_bwd   = results["nsa"][1]
        if nsa_bwd > 0:
            speedup = nsa_bwd / htree_bwd
            faster = "htree" if speedup > 1 else "NSA"
            print(f"\n  Backward speedup: {_ok(f'{faster} is {max(speedup, 1/speedup):.2f}x faster')}")

    # ---- param summary ----
    print()
    print(_header("Params:"))
    print(f"  B={args.batch}, T={args.seq_len}, HQ={args.hq}, "
          f"H_kv={args.h_kv}, K={args.k_dim}, V={args.v_dim}, dtype={args.dtype}")
    if run_htree_flag:
        print(f"  htree: CR={args.htree_compression_rate}, "
              f"TK={args.htree_top_k}, MTN={args.htree_max_top_nodes}")
    if run_nsa_flag:
        print(f"  NSA  : block_size={args.block_size}, S={args.s_blocks}, "
              "pipeline=pooling+compression+selection+execution")
    print(_header("========================================\n"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speed test: HTree Triton backward vs NSA backward")
    p.add_argument("--gpu",   type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=12000,
                   help="Sequence length (default 12000; smaller than fwd-only "
                        "because backward uses more memory)")
    p.add_argument("--hq",    type=int, default=64, help="#query heads")
    p.add_argument("--h-kv",  type=int, default=4,  help="#KV heads")
    p.add_argument("--k-dim", type=int, default=32)
    p.add_argument("--v-dim", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64, help="NSA block size")
    p.add_argument("--s-blocks",   type=int, default=16, help="NSA selected blocks")
    p.add_argument("--htree-top-k", type=int, default=512)
    p.add_argument("--htree-compression-rate", type=int, default=16)
    p.add_argument("--htree-max-top-nodes",    type=int, default=8192)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters",  type=int, default=3)
    p.add_argument("--dtype",  choices=["float16", "bfloat16", "float32"],
                   default="float32")
    p.add_argument("--only",   choices=["all", "htree", "nsa"], default="all",
                   help="Run only a subset (default: all)")
    p.add_argument("--profile", action="store_true",
                   help="Profile mode: warmup then capture 1 iter with "
                        "cudaProfilerStart/Stop")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
