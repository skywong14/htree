"""
Long-sequence multi-seed correctness test for HTree Triton backward.

Judgement philosophy (aligned with correctness_test.py for forward):
  - TopK selection divergence between naive and Triton is EXPECTED; it causes
    sparse large outliers in output/gradients while the bulk stays accurate.
  - HARD criterion:  p99 of |ref - tri| must be below threshold (bulk correctness).
  - SOFT alarm:      max of |ref - tri| triggers a warning (selection divergence).
  - Three-level status: PASS / PASS_WITH_ALARM / FAIL.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

from src.backward import htree_forward_triton_autograd
from src.naive import htree_forward_naive_autograd


logging.getLogger("src.parallel").setLevel(logging.WARNING)
logging.getLogger("src.backward").setLevel(logging.WARNING)

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _fmt_header(text: str) -> str:
    return _c(text, "1;36")


def _fmt_ok(text: str) -> str:
    return _c(text, "1;32")


def _fmt_warn(text: str) -> str:
    return _c(text, "1;33")


def _fmt_error(text: str) -> str:
    return _c(text, "1;31")


def _fmt_dim(text: str) -> str:
    return _c(text, "2")


def _to_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _parse_seeds(spec: str) -> List[int]:
    seeds = [int(p.strip()) for p in spec.split(",") if p.strip()]
    if not seeds:
        raise ValueError("seeds cannot be empty")
    return seeds


def _validate_args(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hq % args.h_kv != 0:
        raise ValueError("hq must be divisible by h-kv (GQA)")
    if args.k_dim < 32 or (args.k_dim % 2) != 0:
        raise ValueError("k-dim must be even and >= 32")
    if (args.k_dim & (args.k_dim - 1)) != 0:
        raise ValueError("k-dim must be a power of 2")
    if args.v_dim <= 0 or (args.v_dim & (args.v_dim - 1)) != 0:
        raise ValueError("v-dim must be a power of 2")
    cr = args.compression_rate
    if cr <= 0 or (cr & (cr - 1)) != 0:
        raise ValueError("compression-rate must be a power of 2")
    tk = args.top_k_per_layer
    if tk <= 0 or (tk & (tk - 1)) != 0:
        raise ValueError("top-k-per-layer must be a power of 2")
    if tk % cr != 0:
        raise ValueError("top-k-per-layer must be divisible by compression-rate")
    if tk % 64 != 0:
        raise ValueError("top-k-per-layer must be divisible by 64 (kernel DROP_BLOCK constraint)")
    if args.max_top_nodes != cr * tk:
        raise ValueError("max-top-nodes must equal compression-rate * top-k-per-layer")
    if args.seq_len <= args.max_top_nodes:
        raise ValueError(
            "This script is for long-sequence stress only: require seq-len > max-top-nodes "
            "(to ensure multi-layer tree path is exercised)."
        )


# ---------------------------------------------------------------------------
#  Diff statistics
# ---------------------------------------------------------------------------

def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a.float() - b.float()).abs()
    flat = diff.flatten()
    n = flat.numel()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "p99_abs": torch.quantile(flat, 0.99).item(),
        "p999_abs": torch.quantile(flat, 0.999).item(),
        "outlier_frac": (flat > flat.quantile(0.99) * 10).float().mean().item(),
    }


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# ---------------------------------------------------------------------------
#  Per-tensor judgement
# ---------------------------------------------------------------------------

def _judge_tensor(
    stats: Dict[str, float],
    p99_atol: float,
    alarm_max: float,
) -> Tuple[str, List[str]]:
    """Return (status, messages) where status is 'pass' / 'alarm' / 'fail'."""
    msgs: List[str] = []
    if stats["p99_abs"] > p99_atol:
        msgs.append(f"p99={stats['p99_abs']:.3e} > threshold {p99_atol:.1e}")
        return "fail", msgs
    status = "pass"
    if stats["max_abs"] > alarm_max:
        msgs.append(
            f"max={stats['max_abs']:.3e} > alarm {alarm_max:.1e} "
            f"(p99={stats['p99_abs']:.3e} ok, likely TopK selection divergence)"
        )
        status = "alarm"
    return status, msgs


# ---------------------------------------------------------------------------
#  Run one seed
# ---------------------------------------------------------------------------

def _run_one_seed(
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    B, T = args.batch, args.seq_len
    q0 = torch.randn(B, T, args.hq, args.k_dim, device=device, dtype=dtype).contiguous()
    k0 = torch.randn(B, T, args.h_kv, args.k_dim, device=device, dtype=dtype).contiguous()
    v0 = torch.randn(B, T, args.h_kv, args.v_dim, device=device, dtype=dtype).contiguous()
    grad_out = torch.randn(B, T, args.hq, args.v_dim, device=device, dtype=dtype).contiguous()
    scale = args.k_dim ** -0.5

    fwd_kwargs = dict(
        compression_rate=args.compression_rate,
        max_top_nodes=args.max_top_nodes,
        top_k_per_layer=args.top_k_per_layer,
        scale=scale,
        rope_base=args.rope_base,
    )

    # --- naive reference ---
    q_ref = q0.detach().clone().requires_grad_(True)
    k_ref = k0.detach().clone().requires_grad_(True)
    v_ref = v0.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.time()
    out_ref = htree_forward_naive_autograd(q_ref, k_ref, v_ref, **fwd_kwargs)
    out_ref.backward(grad_out)
    torch.cuda.synchronize()
    ref_sec = time.time() - t0
    dq_ref, dk_ref, dv_ref = q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()

    # --- Triton ---
    q_tri = q0.detach().clone().requires_grad_(True)
    k_tri = k0.detach().clone().requires_grad_(True)
    v_tri = v0.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    t1 = time.time()
    out_tri = htree_forward_triton_autograd(q_tri, k_tri, v_tri, **fwd_kwargs)
    out_tri.backward(grad_out)
    torch.cuda.synchronize()
    tri_sec = time.time() - t1
    dq_tri, dk_tri, dv_tri = q_tri.grad.detach(), k_tri.grad.detach(), v_tri.grad.detach()

    # --- finiteness ---
    finite_ok = all(
        _is_finite(t) for t in [out_ref, out_tri, dq_ref, dq_tri, dk_ref, dk_tri, dv_ref, dv_tri]
    )

    # --- diff stats ---
    tensor_results = {}
    for name, ref, tri in [
        ("out", out_ref, out_tri),
        ("dq", dq_ref, dq_tri),
        ("dk", dk_ref, dk_tri),
        ("dv", dv_ref, dv_tri),
    ]:
        stats = _diff_stats(ref, tri)
        is_grad = name != "out"
        p99_atol = args.grad_p99_atol if is_grad else args.out_p99_atol
        alarm_max = args.grad_alarm_max if is_grad else args.out_alarm_max
        status, msgs = _judge_tensor(stats, p99_atol, alarm_max)
        tensor_results[name] = {**stats, "status": status, "msgs": msgs}

    has_fail = not finite_ok or any(r["status"] == "fail" for r in tensor_results.values())
    has_alarm = any(r["status"] == "alarm" for r in tensor_results.values())

    if has_fail:
        seed_status = "fail"
    elif has_alarm:
        seed_status = "alarm"
    else:
        seed_status = "pass"

    return {
        "seed": seed,
        "status": seed_status,
        "finite_ok": finite_ok,
        "ref_sec": ref_sec,
        "tri_sec": tri_sec,
        **tensor_results,
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HTree backward correctness stress test (long sequence + multi seed)"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help='comma-separated seeds, e.g. "0,1,2,3"')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=12000)
    parser.add_argument("--hq", type=int, default=16, help="query heads")
    parser.add_argument("--h-kv", type=int, default=4, help="kv heads")
    parser.add_argument("--k-dim", type=int, default=32)
    parser.add_argument("--v-dim", type=int, default=32)
    parser.add_argument("--compression-rate", type=int, default=16)
    parser.add_argument("--top-k-per-layer", type=int, default=512)
    parser.add_argument("--max-top-nodes", type=int, default=8192)
    parser.add_argument("--rope-base", type=float, default=10000.0)

    parser.add_argument("--out-p99-atol", type=float, default=5e-3,
                        help="HARD threshold: output p99 abs diff (default 5e-3)")
    parser.add_argument("--out-alarm-max", type=float, default=5e-2,
                        help="SOFT alarm: output max abs diff (default 5e-2)")
    parser.add_argument("--grad-p99-atol", type=float, default=1e-2,
                        help="HARD threshold: gradient p99 abs diff (default 1e-2)")
    parser.add_argument("--grad-alarm-max", type=float, default=1e-1,
                        help="SOFT alarm: gradient max abs diff (default 1e-1)")

    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--fail-on-alarm", action="store_true",
                        help="Treat PASS_WITH_ALARM as FAIL (exit code 1)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

_STATUS_FMT = {"pass": _fmt_ok, "alarm": _fmt_warn, "fail": _fmt_error}
_STATUS_LABEL = {"pass": "PASS", "alarm": "PASS_WITH_ALARM", "fail": "FAIL"}
_TENSOR_NAMES = ["out", "dq", "dk", "dv"]


def main() -> None:
    args = parse_args()
    _validate_args(args)
    seeds = _parse_seeds(args.seeds)

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = _to_dtype(args.dtype)

    print(_fmt_header("========== HTree Backward Correctness Stress Report =========="))
    print(_fmt_dim(f"device=cuda:{args.gpu}, dtype={args.dtype}"))
    print(_fmt_dim(
        f"config: B={args.batch}, T={args.seq_len}, H={args.hq}, H_kv={args.h_kv}, "
        f"K={args.k_dim}, V={args.v_dim}, CR={args.compression_rate}, "
        f"TK={args.top_k_per_layer}, MTN={args.max_top_nodes}"
    ))
    print(_fmt_dim(f"seeds: {seeds}"))
    print(_fmt_dim(
        f"hard threshold (p99): out<{args.out_p99_atol:.1e}, grad<{args.grad_p99_atol:.1e}"
    ))
    print(_fmt_dim(
        f"soft alarm    (max):  out>{args.out_alarm_max:.1e}, grad>{args.grad_alarm_max:.1e}"
    ))

    all_results: List[Dict] = []
    for idx, seed in enumerate(seeds):
        print(_fmt_header(f"\n--- [{idx + 1}/{len(seeds)}] seed={seed} ---"))
        result = _run_one_seed(seed=seed, device=device, dtype=dtype, args=args)
        all_results.append(result)

        fmt = _STATUS_FMT[result["status"]]

        timing = f"ref={result['ref_sec']:.1f}s tri={result['tri_sec']:.1f}s"
        stats_parts = []
        for tn in _TENSOR_NAMES:
            r = result[tn]
            stats_parts.append(f"{tn}:p99={r['p99_abs']:.2e},max={r['max_abs']:.2e}")
        print(fmt(f"[{_STATUS_LABEL[result['status']]}] {timing} | {' '.join(stats_parts)}"))

        if not result["finite_ok"]:
            print(_fmt_error("  !! non-finite values detected (NaN/Inf)"))
        for tn in _TENSOR_NAMES:
            for msg in result[tn]["msgs"]:
                level_fmt = _fmt_error if result[tn]["status"] == "fail" else _fmt_warn
                print(level_fmt(f"  {tn}: {msg}"))

        if args.fail_fast and result["status"] == "fail":
            break

    # --- summary ---
    worst = {tn: {} for tn in _TENSOR_NAMES}
    for tn in _TENSOR_NAMES:
        worst[tn]["max_abs"] = max(r[tn]["max_abs"] for r in all_results)
        worst[tn]["p99_abs"] = max(r[tn]["p99_abs"] for r in all_results)

    total_ref = sum(r["ref_sec"] for r in all_results)
    total_tri = sum(r["tri_sec"] for r in all_results)

    has_any_fail = any(r["status"] == "fail" for r in all_results)
    has_any_alarm = any(r["status"] == "alarm" for r in all_results)

    if has_any_fail:
        overall = "fail"
    elif has_any_alarm:
        overall = "alarm"
    else:
        overall = "pass"

    print(_fmt_header("\n-------------------- Summary --------------------"))
    print(f"seeds tested: {len(all_results)}/{len(seeds)}")
    print(f"time: naive={total_ref:.1f}s, triton={total_tri:.1f}s")

    for tn in _TENSOR_NAMES:
        w = worst[tn]
        line = f"  {tn:>3s}  worst p99={w['p99_abs']:.3e}  worst max={w['max_abs']:.3e}"
        print(_fmt_dim(line))

    failed_seeds = [r["seed"] for r in all_results if r["status"] == "fail"]
    alarm_seeds = [r["seed"] for r in all_results if r["status"] == "alarm"]

    overall_fmt = _STATUS_FMT[overall]
    status_line = f"overall status: {_STATUS_LABEL[overall]}"
    if failed_seeds:
        status_line += f"  failed seeds: {failed_seeds}"
    if alarm_seeds:
        status_line += f"  alarm seeds: {alarm_seeds}"
    print(overall_fmt(status_line))

    print(_fmt_header("================================================="))

    if overall == "fail":
        raise SystemExit(1)
    if overall == "alarm" and args.fail_on_alarm:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
