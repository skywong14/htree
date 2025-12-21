"""
Speed benchmark: compare NSA kernel vs htree kernel.

- Uses forward-only runs (no correctness check beyond successful execution).
- Defaults mimic the scale used in test_all_positions, but with GQA-friendly
  ratios that satisfy NSA's HQ/H requirement (HQ / H is a power of 2 and >= 16).

Requirements: CUDA + Triton, torch, and `fla` (for NSA helpers).
"""
import argparse
import math
import os
import sys
from typing import Callable, Tuple

import torch

from src.parallel import htree_forward_v2
from ref_nsa.nsa_parallel import parallel_nsa

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

def build_causal_blocks(
    B: int,
    T: int,
    H_kv: int,
    S: int,
    block_size: int,
    device: torch.device,
) -> torch.LongTensor:
    """Build causal block indices [B, T, H_kv, S]."""
    num_blocks = math.ceil(T / block_size)
    block_ids = torch.arange(T, device=device) // block_size  # [T]
    offsets = torch.arange(-(S - 1), 1, device=device)  # [S]
    indices = block_ids[:, None] + offsets[None, :]
    indices = torch.clamp(indices, min=0, max=num_blocks - 1)  # [T, S]
    indices = indices.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, S)
    return indices.contiguous()


def benchmark(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> float:
    """Return avg latency in ms for fn (synchronized, CUDA events)."""
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    torch.manual_seed(42)
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, T, H_kv, K, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, T, H_kv, V, device=device, dtype=dtype).contiguous()
    block_indices = build_causal_blocks(B, T, H_kv, S, block_size, device)
    return q, k, v, block_indices


def run_benchmarks(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "CUDA is required for this benchmark"
    if args.gpu < 0:
        raise ValueError(f"--gpu must be >= 0, got {args.gpu}")
    if torch.cuda.device_count() <= args.gpu:
        raise ValueError(
            f"Requested --gpu {args.gpu}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    torch.backends.cuda.matmul.allow_tf32 = True

    q, k, v, block_indices = make_inputs(
        B=args.batch,
        T=args.seq_len,
        HQ=args.hq,
        H_kv=args.h_kv,
        K=args.k_dim,
        V=args.v_dim,
        S=args.s_blocks,
        block_size=args.block_size,
        dtype=dtype,
        device=device,
    )
    scale = args.k_dim ** -0.5

    # Dummy gate for compressed attention to trigger full NSA pipeline (Compression + Selection + Execution)
    g_cmp = torch.zeros(args.batch, args.seq_len, args.hq, device=device, dtype=dtype)

    def run_htree():
        return htree_forward_v2(
            q, k, v,
            compression_rate=args.htree_compression_rate,
            max_top_nodes=args.htree_max_top_nodes,
            top_k_per_layer=args.htree_top_k,
            scale=scale,
        )

    def run_nsa():
        # Passing g_cmp triggers the full pipeline:
        # 1. mean_pooling (Compression)
        # 2. parallel_nsa_compression + parallel_nsa_topk (Selection)
        # 3. parallel_nsa_fwd (Execution)
        return parallel_nsa(
            q,
            k,
            v,
            g_cmp=g_cmp,
            block_indices=None, # Let NSA compute indices dynamically
            block_size=args.block_size,
            scale=scale,
            block_counts=args.s_blocks,
        )

    print("\n" + _hdr("====== Speed Benchmark (forward only) ======"))
    dev_name = torch.cuda.get_device_name(args.gpu)
    print(_dim(f"Device: {device} ({dev_name}), dtype: {dtype}"))
    print(_dim(f"B={args.batch}, T={args.seq_len}, HQ={args.hq}, H_kv={args.h_kv}, K={args.k_dim}, V={args.v_dim}"))
    print(_dim(f"block_size={args.block_size}, S={args.s_blocks}"))
    print(_dim(
        f"htree: compression_rate={args.htree_compression_rate}, top_k_per_layer={args.htree_top_k}, max_top_nodes={args.htree_max_top_nodes}"
    ))
    print(_dim(f"warmup={args.warmup}, iters={args.iters}"))
    print(_warn("Note: NSA runs FULL pipeline (pooling + compression + selection + execution)"))

    htree_ms = benchmark(run_htree, warmup=args.warmup, iters=args.iters)
    nsa_ms = benchmark(run_nsa, warmup=args.warmup, iters=args.iters)

    print(_hdr("Results (avg ms / iter):"))
    print(f"  htree: {_ok(f'{htree_ms:.3f} ms')}")
    print(f"  NSA  : {_ok(f'{nsa_ms:.3f} ms')}")
    print(_hdr("Params summary:"))
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
    print(_hdr("========================================\n"))

    # --- Profile NSA Breakdown ---
    print(_hdr("Profiling NSA breakdown..."))
    # Imports
    from fla.ops.utils.pooling import mean_pooling  # type: ignore[import-not-found]
    from ref_nsa.nsa_compression import parallel_nsa_compression
    from ref_nsa.nsa_parallel import parallel_nsa_topk, parallel_nsa_fwd

    # Warmup
    for _ in range(3):
        k_cmp, v_cmp = mean_pooling(k, args.block_size, None), mean_pooling(v, args.block_size, None)
        o_cmp, lse_cmp = parallel_nsa_compression(q, k_cmp, v_cmp, args.block_size, scale, None)
        block_indices = parallel_nsa_topk(q, k_cmp, lse_cmp, args.s_blocks, args.block_size, scale, None)
        parallel_nsa_fwd(q, k, v, block_indices, args.s_blocks, args.block_size, scale, None)
    
    _cuda_sync()
    
    start = torch.cuda.Event(enable_timing=True)
    e_pool = torch.cuda.Event(enable_timing=True)
    e_comp = torch.cuda.Event(enable_timing=True)
    e_sel = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    t_pool = 0.0
    t_comp = 0.0
    t_sel = 0.0
    t_exec = 0.0
    
    for _ in range(args.iters):
        start.record()
        k_cmp, v_cmp = mean_pooling(k, args.block_size, None), mean_pooling(v, args.block_size, None)
        e_pool.record()
        
        o_cmp, lse_cmp = parallel_nsa_compression(q, k_cmp, v_cmp, args.block_size, scale, None)
        e_comp.record()
        
        block_indices = parallel_nsa_topk(q, k_cmp, lse_cmp, args.s_blocks, args.block_size, scale, None)
        e_sel.record()
        
        parallel_nsa_fwd(q, k, v, block_indices, args.s_blocks, args.block_size, scale, None)
        end.record()
        
        _cuda_sync()
        t_pool += start.elapsed_time(e_pool)
        t_comp += e_pool.elapsed_time(e_comp)
        t_sel += e_comp.elapsed_time(e_sel)
        t_exec += e_sel.elapsed_time(end)
        
    print(f"NSA Breakdown (avg over {args.iters} iters):")
    print(f"  Pooling    : {t_pool/args.iters:.3f} ms")
    print(f"  Compression: {t_comp/args.iters:.3f} ms")
    print(f"  Selection  : {t_sel/args.iters:.3f} ms")
    print(f"  Execution  : {t_exec/args.iters:.3f} ms")
    print(f"  Total      : {(t_pool+t_comp+t_sel+t_exec)/args.iters:.3f} ms")
    print(_hdr("========================================\n"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed test: NSA vs htree kernels (forward only)")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index to use (after CUDA_VISIBLE_DEVICES remapping)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=40000)
    parser.add_argument("--hq", type=int, default=16, help="#query heads (HQ)")
    parser.add_argument("--h-kv", type=int, default=1, help="#key/value heads (H)")
    parser.add_argument("--k-dim", type=int, default=16)
    parser.add_argument("--v-dim", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64, help="NSA block size")
    parser.add_argument("--s-blocks", type=int, default=16, help="NSA blocks per token (S)")
    parser.add_argument("--htree-top-k", type=int, default=512)
    parser.add_argument("--htree-compression-rate", type=int, default=16)
    parser.add_argument("--htree-max-top-nodes", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmarks(parse_args())
