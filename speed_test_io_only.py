"""
Speed test: Compare parallel.py (full computation) vs parallel_io_only.py (IO bandwidth test)

This script measures:
1. Full htree kernel with all computations
2. IO-only htree kernel with simplified computations
3. Speedup from removing computation overhead

The IO-only version should show the upper bound of IO bandwidth.
"""
import argparse
import sys
import os
import torch

from src.parallel import htree_forward_v2
from src.parallel_io_only import htree_forward_io_only

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _hdr(text: str) -> str:
    return _c(text, "1;36")  # bold cyan

def _ok(text: str) -> str:
    return _c(text, "1;32")  # bold green

def _warn(text: str) -> str:
    return _c(text, "1;33")  # bold yellow

def _dim(text: str) -> str:
    return _c(text, "2")  # dim

def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(fn, warmup: int = 3, iters: int = 10):
    """Return avg latency in ms."""
    _cuda_sync()
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    _cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(iters):
            fn()
    end.record()
    _cuda_sync()
    total_ms = start.elapsed_time(end)
    return total_ms / iters


def main():
    parser = argparse.ArgumentParser(description="Speed test: full computation vs IO-only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=120000)
    parser.add_argument("--hq", type=int, default=16, help="#query heads")
    parser.add_argument("--h-kv", type=int, default=1, help="#key/value heads")
    parser.add_argument("--k-dim", type=int, default=16)
    parser.add_argument("--v-dim", type=int, default=16)
    parser.add_argument("--compression-rate", type=int, default=16)
    parser.add_argument("--max-top-nodes", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU {args.gpu}")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    torch.backends.cuda.matmul.allow_tf32 = True

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(args.batch, args.seq_len, args.hq, args.k_dim, device=device, dtype=dtype).contiguous()
    k = torch.randn(args.batch, args.seq_len, args.h_kv, args.k_dim, device=device, dtype=dtype).contiguous()
    v = torch.randn(args.batch, args.seq_len, args.h_kv, args.v_dim, device=device, dtype=dtype).contiguous()
    scale = args.k_dim ** -0.5

    def run_full():
        return htree_forward_v2(
            q, k, v,
            compression_rate=args.compression_rate,
            max_top_nodes=args.max_top_nodes,
            top_k_per_layer=args.top_k,
            scale=scale,
        )

    def run_io_only():
        return htree_forward_io_only(
            q, k, v,
            compression_rate=args.compression_rate,
            max_top_nodes=args.max_top_nodes,
            top_k_per_layer=args.top_k,
            scale=scale,
        )

    print("\n" + _hdr("=" * 60))
    print(_hdr("Speed Test: Full Computation vs IO-Only"))
    print(_hdr("=" * 60))
    dev_name = torch.cuda.get_device_name(args.gpu)
    print(_dim(f"Device: {device} ({dev_name}), dtype: {dtype}"))
    print(_dim(f"B={args.batch}, T={args.seq_len}, HQ={args.hq}, H_kv={args.h_kv}, K={args.k_dim}, V={args.v_dim}"))
    print(_dim(f"compression_rate={args.compression_rate}, top_k={args.top_k}, max_top_nodes={args.max_top_nodes}"))
    print(_dim(f"warmup={args.warmup}, iters={args.iters}"))
    print()

    print(_warn("Note: First run includes Triton kernel compilation time..."))
    print()

    print(_hdr("Running Full Computation Version..."))
    full_ms = benchmark(run_full, warmup=args.warmup, iters=args.iters)
    
    print(_hdr("Running IO-Only Version..."))
    io_ms = benchmark(run_io_only, warmup=args.warmup, iters=args.iters)

    print()
    print(_hdr("=" * 60))
    print(_hdr("Results (avg ms / iter):"))
    print(_hdr("=" * 60))
    print(f"  Full Computation: {_ok(f'{full_ms:.3f} ms')}")
    print(f"  IO-Only        : {_ok(f'{io_ms:.3f} ms')}")
    
    if io_ms > 0:
        speedup = full_ms / io_ms
        overhead_pct = ((full_ms - io_ms) / full_ms) * 100
        print()
        print(_hdr("Analysis:"))
        print(f"  Speedup (Full/IO): {_ok(f'{speedup:.2f}x')}")
        print(f"  Computation overhead: {_warn(f'{overhead_pct:.1f}%')} of total time")
        print(f"  IO time (approx): {_ok(f'{io_ms:.3f} ms')} ({100-overhead_pct:.1f}% of total)")
    
    print()
    print(_dim("Interpretation:"))
    print(_dim("  - IO-Only time represents the upper bound of memory bandwidth"))
    print(_dim("  - Computation overhead = Full - IO-Only"))
    print(_dim("  - Lower overhead % means the kernel is more IO-bound"))
    print(_dim("  - Higher overhead % means more optimization potential in computation"))
    print(_hdr("=" * 60))
    print()


if __name__ == "__main__":
    main()
