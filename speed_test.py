"""
Speed benchmark: HTree vs NSA kernels (forward-only).

Measures average forward-pass latency in milliseconds using CUDA events.
Supports selective benchmarking (`--only htree`), profiling with
`ncu --profile-from-start off`, and NSA stage-level breakdown.

Requirements: CUDA + Triton, torch, and `fla` (for NSA helpers).
"""
import argparse
import math
import os
import sys
from typing import Callable, Tuple

import torch

from src.parallel import htree_forward

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
def _colorize(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"
def _fmt_header(text: str) -> str:
    return _colorize(text, "1;36")  # bold cyan
def _fmt_success(text: str) -> str:
    return _colorize(text, "1;32")  # bold green
def _fmt_warning(text: str) -> str:
    return _colorize(text, "1;33")  # bold yellow
def _fmt_dim(text: str) -> str:
    return _colorize(text, "2")  # dim
def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def build_causal_blocks(
    B: int,
    T: int,
    H_kv: int,
    S: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build causal block indices [B, T, H_kv, S]."""
    num_blocks = math.ceil(T / block_size)
    block_ids = torch.arange(T, device=device) // block_size  # [T]
    offsets = torch.arange(-(S - 1), 1, device=device)  # [S]
    indices = block_ids[:, None] + offsets[None, :]
    indices = torch.clamp(indices, min=0, max=num_blocks - 1)  # [T, S]
    indices = indices.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, S)
    return indices.contiguous()

def make_inputs(
    B: int,
    T: int,
    HQ: int,
    H_kv: int,
    K: int,
    V: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, T, H_kv, K, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, T, H_kv, V, device=device, dtype=dtype).contiguous()
    block_indices = build_causal_blocks(B, T, H_kv, S, block_size, device)
    return q, k, v, block_indices

def benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> float:
    """Return avg latency in ms for *fn* (synchronized, CUDA events)."""
    _cuda_sync()
    with torch.no_grad():
        for _ in range(max(warmup, 0)):
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
    return total_ms / max(iters, 1)

def _run_profile_mode(targets, warmup: int) -> None:
    """Warm up kernels, then capture a single profiled iteration."""
    print(_fmt_header("\n>>> PROFILE mode enabled <<<"))
    print(_fmt_dim("Warmup runs (profiler OFF) — Triton JIT + cache warm ..."))

    for name, fn in targets:
        print(f"  warming up {name} ({warmup} iters) ...")
        _cuda_sync()
        with torch.no_grad():
            for _ in range(max(warmup, 1)):
                fn()
        _cuda_sync()
        print(f"  {name} warmup done.")

    print(_fmt_success("Starting CUDA profiler — profiling 1 iteration ..."))
    torch.cuda.cudart().cudaProfilerStart()
    with torch.no_grad():
        for _name, fn in targets:
            fn()
    torch.cuda.cudart().cudaProfilerStop()
    _cuda_sync()
    print(_fmt_success("CUDA profiler stopped. Profile data captured."))
    print(_fmt_header("========================================\n"))


def _print_results(args: argparse.Namespace, htree_ms, nsa_ms) -> None:
    """Print benchmark results and parameter summary."""
    print(_fmt_header("Results (avg ms / iter):"))
    if htree_ms is not None:
        print(f"  htree: {_fmt_success(f'{htree_ms:.3f} ms')}")
    if nsa_ms is not None:
        print(f"  NSA  : {_fmt_success(f'{nsa_ms:.3f} ms')}")
    print(_fmt_header("Params summary:"))
    print(
        "  experiment: "
        f"B={args.batch}, "
        f"T={args.seq_len}, "
        f"HQ={args.hq}, "
        f"H_kv={args.h_kv}, "
        f"K={args.k_dim}, "
        f"V={args.v_dim}, "
        f"dtype={args.dtype}, "
        f"block_size={args.block_size}, "
        f"S(block_counts)={args.s_blocks}"
    )
    print(
        "  htree: "
        f"compression_rate={args.htree_compression_rate}, "
        f"top_k_per_layer={args.htree_top_k}, "
        f"max_top_nodes={args.htree_max_top_nodes}"
    )
    print(
        "  NSA  : "
        f"block_size={args.block_size}, "
        f"block_counts(S)={args.s_blocks}, "
        "pipeline=pooling+compression+selection+execution"
    )
    print(_fmt_header("========================================\n"))

def _run_nsa_breakdown(args: argparse.Namespace, q, k, v, scale: float) -> None:
    """Profile NSA by stage: pooling, compression, selection, execution."""
    print(_fmt_header("Profiling NSA breakdown..."))

    from fla.ops.utils.pooling import mean_pooling  # type: ignore[import-not-found]
    from ref_nsa.nsa_compression import parallel_nsa_compression
    from ref_nsa.nsa_parallel import parallel_nsa_topk, parallel_nsa_fwd

    # Warmup
    for _ in range(3):
        k_cmp = mean_pooling(k, args.block_size, None)
        v_cmp = mean_pooling(v, args.block_size, None)
        o_cmp, lse_cmp = parallel_nsa_compression(q, k_cmp, v_cmp, args.block_size, scale, None)
        block_indices = parallel_nsa_topk(q, k_cmp, lse_cmp, args.s_blocks, args.block_size, scale, None)
        parallel_nsa_fwd(q, k, v, block_indices, args.s_blocks, args.block_size, scale, None)

    _cuda_sync()

    start = torch.cuda.Event(enable_timing=True)
    event_pool = torch.cuda.Event(enable_timing=True)
    event_compress = torch.cuda.Event(enable_timing=True)
    event_select = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    elapsed_pool_ms = 0.0
    elapsed_compress_ms = 0.0
    elapsed_select_ms = 0.0
    elapsed_execute_ms = 0.0

    for _ in range(args.iters):
        start.record()
        k_cmp = mean_pooling(k, args.block_size, None)
        v_cmp = mean_pooling(v, args.block_size, None)
        event_pool.record()

        o_cmp, lse_cmp = parallel_nsa_compression(q, k_cmp, v_cmp, args.block_size, scale, None)
        event_compress.record()

        block_indices = parallel_nsa_topk(q, k_cmp, lse_cmp, args.s_blocks, args.block_size, scale, None)
        event_select.record()

        parallel_nsa_fwd(q, k, v, block_indices, args.s_blocks, args.block_size, scale, None)
        end.record()

        _cuda_sync()
        elapsed_pool_ms += start.elapsed_time(event_pool)
        elapsed_compress_ms += event_pool.elapsed_time(event_compress)
        elapsed_select_ms += event_compress.elapsed_time(event_select)
        elapsed_execute_ms += event_select.elapsed_time(end)

    n = args.iters
    total_ms = elapsed_pool_ms + elapsed_compress_ms + elapsed_select_ms + elapsed_execute_ms
    print(f"NSA Breakdown (avg over {n} iters):")
    print(f"  Pooling    : {elapsed_pool_ms / n:.3f} ms")
    print(f"  Compression: {elapsed_compress_ms / n:.3f} ms")
    print(f"  Selection  : {elapsed_select_ms / n:.3f} ms")
    print(f"  Execution  : {elapsed_execute_ms / n:.3f} ms")
    print(f"  Total      : {total_ms / n:.3f} ms")
    print(_fmt_header("========================================\n"))


# ========================================
#  Main entry point
# ========================================

def run_benchmarks(args: argparse.Namespace) -> None:
    """Orchestrate device setup, input creation, and benchmark execution."""
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    if args.gpu < 0:
        raise ValueError(f"--gpu must be >= 0, got {args.gpu}")
    if torch.cuda.device_count() <= args.gpu:
        raise ValueError(
            f"Requested --gpu {args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    torch.backends.cuda.matmul.allow_tf32 = True

    q, k, v, block_indices = make_inputs(
        B=args.batch, T=args.seq_len, HQ=args.hq, H_kv=args.h_kv,
        K=args.k_dim, V=args.v_dim, S=args.s_blocks,
        block_size=args.block_size, dtype=dtype, device=device,
    )
    scale = args.k_dim ** -0.5

    # Dummy gate to trigger full NSA pipeline (Compression + Selection + Execution)
    gate_compress = torch.zeros(args.batch, args.seq_len, args.hq, device=device, dtype=dtype)

    def run_htree():
        return htree_forward(
            q, k, v,
            compression_rate=args.htree_compression_rate,
            max_top_nodes=args.htree_max_top_nodes,
            top_k_per_layer=args.htree_top_k,
            scale=scale,
        )

    def run_nsa():
        from ref_nsa.nsa_parallel import parallel_nsa
        return parallel_nsa(
            q, k, v,
            g_cmp=gate_compress,
            block_indices=None,
            block_size=args.block_size,
            scale=scale,
            block_counts=args.s_blocks,
        )

    # Print header
    print("\n" + _fmt_header("====== Speed Benchmark (forward only) ======"))
    dev_name = torch.cuda.get_device_name(args.gpu)
    print(_fmt_dim(f"Device: {device} ({dev_name}), dtype: {dtype}"))
    print(_fmt_dim(f"B={args.batch}, T={args.seq_len}, HQ={args.hq}, H_kv={args.h_kv}, K={args.k_dim}, V={args.v_dim}"))
    print(_fmt_dim(f"block_size={args.block_size}, S={args.s_blocks}"))
    print(_fmt_dim(
        f"htree: compression_rate={args.htree_compression_rate}, top_k_per_layer={args.htree_top_k}, max_top_nodes={args.htree_max_top_nodes}"
    ))
    print(_fmt_dim(f"warmup={args.warmup}, iters={args.iters}"))
    print(_fmt_warning("Note: NSA runs FULL pipeline (pooling + compression + selection + execution)"))

    run_htree_flag = args.only in ("all", "htree")
    run_nsa_flag = args.only in ("all", "nsa")

    # Profile mode
    if args.profile:
        targets = []
        if run_htree_flag:
            targets.append(("htree", run_htree))
        if run_nsa_flag:
            targets.append(("nsa", run_nsa))
        _run_profile_mode(targets, args.warmup)
        return

    # Normal benchmark mode
    htree_ms = benchmark(run_htree, warmup=args.warmup, iters=args.iters) if run_htree_flag else None
    nsa_ms = benchmark(run_nsa, warmup=args.warmup, iters=args.iters) if run_nsa_flag else None

    _print_results(args, htree_ms, nsa_ms)

    # NSA stage-level breakdown
    if run_nsa_flag:
        _run_nsa_breakdown(args, q, k, v, scale)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed test: NSA vs htree kernels (forward only)")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use (after CUDA_VISIBLE_DEVICES remapping)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=120000)
    parser.add_argument("--hq", type=int, default=64, help="#query heads (HQ)")
    parser.add_argument("--h-kv", type=int, default=4, help="#key/value heads (H)")
    parser.add_argument("--k-dim", type=int, default=32)
    parser.add_argument("--v-dim", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64, help="NSA block size")
    parser.add_argument("--s-blocks", type=int, default=16, help="NSA blocks per token (S)")
    parser.add_argument("--htree-top-k", type=int, default=512)
    parser.add_argument("--htree-compression-rate", type=int, default=16)
    parser.add_argument("--htree-max-top-nodes", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--only", choices=["all", "htree", "nsa"], default="all",
                        help="Run only a subset of benchmarks (default: all)")
    parser.add_argument("--profile", action="store_true",
                        help="Profile mode: warmup without profiler, then capture a single "
                             "iteration with cudaProfilerStart/Stop. Use with: "
                             "ncu --profile-from-start off -o <output> python speed_test.py --profile")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmarks(parse_args())
