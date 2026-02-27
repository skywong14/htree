import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import triton

from src.naive import LayerTrace, htree_forward_naive
from src.parallel import (
    HTREE_SCORE_NEG_INF,
    HTREE_SCORE_VALID_THRESHOLD,
    htree_bottom_accumulate_gqa_kernel,
    htree_build_kernel,
    htree_final_normalize_kernel,
    htree_merge_to_global_kernel,
    htree_select_accumulate_gqa_kernel,
)


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _colorize(text: str, code: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _fmt_header(text: str) -> str:
    return _colorize(text, "1;36")  # bold cyan


def _fmt_ok(text: str) -> str:
    return _colorize(text, "1;32")  # bold green


def _fmt_warn(text: str) -> str:
    return _colorize(text, "1;33")  # bold yellow


def _fmt_error(text: str) -> str:
    return _colorize(text, "1;31")  # bold red


def _fmt_dim(text: str) -> str:
    return _colorize(text, "2")


def _sorted_hist_str(hist: Dict[int, int]) -> str:
    if not hist:
        return "{}"
    items = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda x: x[0]))
    return "{" + items + "}"


def _format_node_rank_list(items: List[Dict[str, int]]) -> str:
    if not items:
        return "[]"
    parts = []
    for x in items:
        node = x["node"]
        rank = x["rank"]
        rank_exact = x.get("rank_exact", -1)
        if rank_exact > 0:
            parts.append(f"{node}(top{rank}, exact{rank_exact})")
        else:
            parts.append(f"{node}(top{rank})")
    return "[" + ", ".join(parts) + "]"


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    y_lo = x_lo * cos - x_hi * sin
    y_hi = x_lo * sin + x_hi * cos
    return torch.cat([y_lo, y_hi], dim=-1)


def _build_init_prev_selected_parents(
    bsz: int,
    seq_len: int,
    h_kv: int,
    top_k_per_layer: int,
    compression_rate: int,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device)[None, :].expand(seq_len, -1)
    valid_mask = parent_candidates < num_virtual_parents[:, None]
    prev_selected_parents = torch.where(
        valid_mask,
        parent_candidates,
        torch.full_like(parent_candidates, -1),
    )
    return (
        prev_selected_parents[None, :, None, :]
        .expand(bsz, seq_len, h_kv, top_k_per_layer)
        .contiguous()
    )


def _build_tree_parallel(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int,
    max_top_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    bsz, seq_len, h_kv, k_dim = k.shape
    v_dim = v.shape[-1]

    layers_k = [k]
    layers_v = [v]
    current_k = k
    current_v = v
    current_len = seq_len

    while current_len > max_top_nodes:
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(bsz, next_len, h_kv, k_dim, dtype=k.dtype, device=k.device)
        next_v = torch.empty(bsz, next_len, h_kv, v_dim, dtype=v.dtype, device=v.device)

        block_size = 8
        grid = (triton.cdiv(next_len, block_size), bsz * h_kv)
        htree_build_kernel[grid](
            current_k,
            current_v,
            next_k,
            next_v,
            N_child=current_len,
            N_parent=next_len,
            B=bsz,
            H_kv=h_kv,
            K=k_dim,
            V=v_dim,
            COMPRESSION_RATE=compression_rate,
            BLOCK_SIZE=block_size,
        )

        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k = next_k
        current_v = next_v
        current_len = next_len

    return layers_k, layers_v


def _build_rope_cache(
    k_dim: int,
    max_top_nodes: int,
    rope_base: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_size = max_top_nodes + 1024
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, k_dim, 2, dtype=torch.float32, device=device) / k_dim))
    positions = torch.arange(cache_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def run_parallel_with_trace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int,
    max_top_nodes: int,
    top_k_per_layer: int,
    scale: Optional[float],
    rope_base: float,
) -> Tuple[torch.Tensor, Dict]:
    bsz, seq_len, h, k_dim = q.shape
    h_kv = k.shape[2]
    v_dim = v.shape[-1]
    num_groups = h // h_kv

    if scale is None:
        scale = k_dim ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    layers_k, layers_v = _build_tree_parallel(
        k=k,
        v=v,
        compression_rate=compression_rate,
        max_top_nodes=max_top_nodes,
    )
    num_layers = len(layers_k)

    cos_cache, sin_cache = _build_rope_cache(k_dim, max_top_nodes, rope_base, q.device)

    global_max = torch.full([bsz, seq_len, h], HTREE_SCORE_NEG_INF, dtype=torch.float32, device=q.device)
    global_sum = torch.zeros([bsz, seq_len, h], dtype=torch.float32, device=q.device)
    global_output = torch.zeros([bsz, seq_len, h, v_dim], dtype=torch.float32, device=q.device)

    layer_max = torch.empty([bsz, seq_len, h], dtype=torch.float32, device=q.device)
    layer_sum = torch.empty([bsz, seq_len, h], dtype=torch.float32, device=q.device)
    layer_output = torch.empty([bsz, seq_len, h, v_dim], dtype=torch.float32, device=q.device)

    max_candidates = max_top_nodes
    log_n = int(math.log2(max_candidates))
    n_dims_topk = int(math.log2(top_k_per_layer))
    drop_block = 64
    tile_p = 4
    g_pad = max(triton.next_power_of_2(num_groups), 16)

    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(seq_len, dtype=torch.int32, device=q.device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=q.device)[None, :].expand(seq_len, -1)
    valid_mask = parent_candidates < num_virtual_parents[:, None]
    prev_selected_parents = torch.where(
        valid_mask,
        parent_candidates,
        torch.full_like(parent_candidates, -1),
    )
    prev_selected_parents = (
        prev_selected_parents[None, :, None, :]
        .expand(bsz, seq_len, h_kv, top_k_per_layer)
        .contiguous()
    )
    next_selected_parents = torch.empty_like(prev_selected_parents)

    traces: Dict[int, LayerTrace] = {}

    for layer_idx in range(num_layers - 1, -1, -1):
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        n_layer = k_layer.shape[1]
        is_bottom = layer_idx == 0
        layer_power = compression_rate ** layer_idx

        grid_kv = (seq_len, bsz * h_kv)
        if is_bottom:
            htree_bottom_accumulate_gqa_kernel[grid_kv](
                q,
                k_layer,
                v_layer,
                prev_selected_parents,
                cos_cache,
                sin_cache,
                layer_max,
                layer_sum,
                layer_output,
                layer_power=layer_power,
                B=bsz,
                T=seq_len,
                H=h,
                H_kv=h_kv,
                NUM_GROUPS=num_groups,
                K=k_dim,
                V=v_dim,
                N_layer=n_layer,
                COMPRESSION_RATE=compression_rate,
                TOP_K=top_k_per_layer,
                NEG_INF=HTREE_SCORE_NEG_INF,
                G_PAD=g_pad,
                TILE_P=tile_p,
                scale=scale,
            )
            selected_parents = None
        else:
            htree_select_accumulate_gqa_kernel[grid_kv](
                q,
                k_layer,
                v_layer,
                prev_selected_parents,
                cos_cache,
                sin_cache,
                layer_max,
                layer_sum,
                layer_output,
                next_selected_parents,
                layer_power=layer_power,
                B=bsz,
                T=seq_len,
                H=h,
                H_kv=h_kv,
                NUM_GROUPS=num_groups,
                K=k_dim,
                V=v_dim,
                N_layer=n_layer,
                COMPRESSION_RATE=compression_rate,
                TOP_K=top_k_per_layer,
                LOG_N=log_n,
                N_DIMS_TOPK=n_dims_topk,
                SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
                NEG_INF=HTREE_SCORE_NEG_INF,
                DROP_BLOCK=drop_block,
                G_PAD=g_pad,
                TILE_P=tile_p,
                scale=scale,
            )
            selected_parents = next_selected_parents.clone()

        traces[layer_idx] = LayerTrace(
            layer_idx=layer_idx,
            selected_parents=selected_parents,
            layer_max=layer_max.clone(),
            layer_sum=layer_sum.clone(),
            layer_output=layer_output.clone(),
        )

        grid = (seq_len, bsz * h)
        htree_merge_to_global_kernel[grid](
            layer_max,
            layer_sum,
            layer_output,
            global_max,
            global_sum,
            global_output,
            B=bsz,
            T=seq_len,
            H=h,
            V=v_dim,
        )

        if not is_bottom:
            prev_selected_parents, next_selected_parents = next_selected_parents, prev_selected_parents

    output = torch.empty(bsz, seq_len, h, v_dim, dtype=q.dtype, device=q.device)
    grid = (seq_len, bsz * h)
    htree_final_normalize_kernel[grid](
        global_output,
        global_sum,
        output,
        B=bsz,
        T=seq_len,
        H=h,
        V=v_dim,
    )

    trace = {
        "num_layers": num_layers,
        "num_groups": num_groups,
        "layers": traces,
    }
    return output, trace


def _selection_set_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    a_set = set(int(x) for x in a.tolist() if x >= 0)
    b_set = set(int(x) for x in b.tolist() if x >= 0)
    return len(a_set.symmetric_difference(b_set))


def _selection_set_diff_details(a: torch.Tensor, b: torch.Tensor) -> Tuple[int, List[Dict[str, int]], List[Dict[str, int]]]:
    a_list = [int(x) for x in a.tolist() if x >= 0]
    b_list = [int(x) for x in b.tolist() if x >= 0]

    rank_a: Dict[int, int] = {}
    rank_b: Dict[int, int] = {}
    for idx, node in enumerate(a_list):
        if node not in rank_a:
            rank_a[node] = idx + 1
    for idx, node in enumerate(b_list):
        if node not in rank_b:
            rank_b[node] = idx + 1

    a_set = set(rank_a.keys())
    b_set = set(rank_b.keys())
    a_only_nodes = sorted(a_set - b_set, key=lambda x: rank_a[x])
    b_only_nodes = sorted(b_set - a_set, key=lambda x: rank_b[x])

    a_only = [{"node": n, "rank": rank_a[n]} for n in a_only_nodes]
    b_only = [{"node": n, "rank": rank_b[n]} for n in b_only_nodes]
    diff_count = len(a_set.symmetric_difference(b_set))
    return diff_count, a_only, b_only


def _layer_group_output(
    layer_trace: LayerTrace,
    b: int,
    t: int,
    kv_h: int,
    num_groups: int,
) -> torch.Tensor:
    h0 = kv_h * num_groups
    h1 = h0 + num_groups
    layer_sum = layer_trace.layer_sum[b, t, h0:h1]  # [G]
    layer_output = layer_trace.layer_output[b, t, h0:h1]  # [G, V]
    return layer_output / layer_sum.clamp_min(1e-30)[:, None]


def compare_traces(
    parallel_trace: Dict,
    naive_trace: Dict,
    *,
    topk_alarm_diff: int,
    layer_out_alarm_abs: float,
    max_report: int,
) -> Dict:
    num_layers = int(parallel_trace["num_layers"])
    num_groups = int(parallel_trace["num_groups"])
    layers_p: Dict[int, LayerTrace] = parallel_trace["layers"]
    layers_n: Dict[int, LayerTrace] = naive_trace["layers"]

    sample_layer = layers_p[num_layers - 1]
    bsz, seq_len, h_kv, _ = sample_layer.selected_parents.shape  # top layer is non-bottom
    prev_layer_exact_equal = torch.ones([bsz, seq_len, h_kv], dtype=torch.bool)

    topk_alarms = []
    layer_output_alarms = []
    mismatch_counter: Dict[int, int] = {}

    for layer_idx in range(num_layers - 1, 0, -1):
        p_sel = layers_p[layer_idx].selected_parents
        n_sel = layers_n[layer_idx].selected_parents
        cur_layer_exact_equal = torch.zeros([bsz, seq_len, h_kv], dtype=torch.bool)

        for b in range(bsz):
            for t in range(seq_len):
                for kv_h in range(h_kv):
                    diff_cnt, p_only, n_only = _selection_set_diff_details(
                        p_sel[b, t, kv_h],
                        n_sel[b, t, kv_h],
                    )
                    mismatch_counter[diff_cnt] = mismatch_counter.get(diff_cnt, 0) + 1
                    exact_equal = diff_cnt == 0
                    cur_layer_exact_equal[b, t, kv_h] = exact_equal

                    if bool(prev_layer_exact_equal[b, t, kv_h]) and diff_cnt > topk_alarm_diff:
                        if len(topk_alarms) < max_report:
                            topk_alarms.append(
                                {
                                    "layer": layer_idx,
                                    "b": b,
                                    "t": t,
                                    "kv_h": kv_h,
                                    "diff_count": diff_cnt,
                                    "parallel_only": p_only,
                                    "naive_only": n_only,
                                }
                            )

                    if exact_equal:
                        p_group_out = _layer_group_output(layers_p[layer_idx], b, t, kv_h, num_groups)
                        n_group_out = _layer_group_output(layers_n[layer_idx], b, t, kv_h, num_groups)
                        max_abs = (p_group_out - n_group_out).abs().max().item()
                        if max_abs > layer_out_alarm_abs and len(layer_output_alarms) < max_report:
                            layer_output_alarms.append(
                                {
                                    "layer": layer_idx,
                                    "b": b,
                                    "t": t,
                                    "kv_h": kv_h,
                                    "max_abs": max_abs,
                                }
                            )

        prev_layer_exact_equal = cur_layer_exact_equal

    return {
        "topk_alarms": topk_alarms,
        "layer_output_alarms": layer_output_alarms,
        "mismatch_counter": mismatch_counter,
    }


def _compute_exact_rank_map_for_alarm(
    alarm_item: Dict,
    *,
    q: torch.Tensor,
    layers_k: List[torch.Tensor],
    trace_parallel: Dict,
    init_prev_selected_parents: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    compression_rate: int,
) -> Dict[int, int]:
    layer_idx = int(alarm_item["layer"])
    b = int(alarm_item["b"])
    t = int(alarm_item["t"])
    kv_h = int(alarm_item["kv_h"])

    num_layers = len(layers_k)
    if layer_idx <= 0 or layer_idx >= num_layers:
        return {}

    if layer_idx == num_layers - 1:
        prev_selected = init_prev_selected_parents[b, t, kv_h]
    else:
        prev_selected = trace_parallel["layers"][layer_idx + 1].selected_parents[b, t, kv_h]

    valid_parents = prev_selected[prev_selected >= 0].to(torch.long)
    if valid_parents.numel() <= 0:
        return {}

    layer_k = layers_k[layer_idx]
    n_layer = layer_k.shape[1]
    h_kv = layer_k.shape[2]
    _, _, h, _ = q.shape
    num_groups = h // h_kv

    layer_power = compression_rate ** layer_idx
    rightmost_idx = t // layer_power
    rightmost_child_idx = rightmost_idx % compression_rate

    num_valid_parents = int(valid_parents.numel())
    num_candidates = (num_valid_parents - 1) * compression_rate + rightmost_child_idx + 1
    if num_candidates <= 0:
        return {}

    cand_pos = torch.arange(num_candidates, device=q.device, dtype=torch.long)
    parent_off = torch.div(cand_pos, compression_rate, rounding_mode="floor")
    child_slot = cand_pos - parent_off * compression_rate
    parent_idx = valid_parents[parent_off].to(torch.long)
    node_indices = parent_idx * compression_rate + child_slot
    if bool((node_indices < 0).any()) or bool((node_indices >= n_layer).any()):
        return {}

    k_nodes = layer_k[b, node_indices, kv_h].to(torch.float32)  # [N, K]
    cos_k = cos_cache[cand_pos]
    sin_k = sin_cache[cand_pos]
    k_rope = _apply_rope(k_nodes, cos_k, sin_k)

    h0 = kv_h * num_groups
    h1 = h0 + num_groups
    q_group = q[b, t, h0:h1].to(torch.float32)  # [G, K]
    rightmost_pos = num_candidates - 1
    cos_q = cos_cache[rightmost_pos]
    sin_q = sin_cache[rightmost_pos]
    q_rope = _apply_rope(q_group, cos_q, sin_q) * float(q.shape[-1] ** -0.5)
    scores = q_rope @ k_rope.transpose(0, 1)  # [G, N]

    # rank_exact uses raw importance (without force-select override).
    local_m = scores.max()
    importance = torch.exp(scores - local_m).sum(dim=0)
    _, sorted_idx = torch.sort(importance, descending=True, stable=True)

    rank_exact: Dict[int, int] = {}
    for i, pos in enumerate(sorted_idx.tolist()):
        node = int(node_indices[pos].item())
        if node not in rank_exact:
            rank_exact[node] = i + 1
    return rank_exact


def _attach_rank_exact_to_topk_alarms(
    trace_report: Dict,
    trace_parallel: Dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    compression_rate: int,
    top_k_per_layer: int,
    max_top_nodes: int,
    rope_base: float,
) -> None:
    if not trace_report["topk_alarms"]:
        return

    bsz, seq_len, _, _ = q.shape
    h_kv = k.shape[2]
    num_layers = int(trace_parallel["num_layers"])

    layers_k, _ = _build_tree_parallel(
        k=k,
        v=v,
        compression_rate=compression_rate,
        max_top_nodes=max_top_nodes,
    )
    cos_cache, sin_cache = _build_rope_cache(q.shape[-1], max_top_nodes, rope_base, q.device)
    init_prev_selected_parents = _build_init_prev_selected_parents(
        bsz=bsz,
        seq_len=seq_len,
        h_kv=h_kv,
        top_k_per_layer=top_k_per_layer,
        compression_rate=compression_rate,
        num_layers=num_layers,
        device=q.device,
    )

    for item in trace_report["topk_alarms"]:
        rank_exact = _compute_exact_rank_map_for_alarm(
            item,
            q=q,
            layers_k=layers_k,
            trace_parallel=trace_parallel,
            init_prev_selected_parents=init_prev_selected_parents,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            compression_rate=compression_rate,
        )
        for entry in item["parallel_only"]:
            entry["rank_exact"] = rank_exact.get(entry["node"], -1)
        for entry in item["naive_only"]:
            entry["rank_exact"] = rank_exact.get(entry["node"], -1)


def _to_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTree parallel vs naive correctness check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
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
    parser.add_argument(
        "--selection-mode",
        choices=["parallel_importance", "exp_sum"],
        default="parallel_importance",
    )
    parser.add_argument("--topk-alarm-diff", type=int, default=2)
    parser.add_argument("--layer-out-alarm-abs", type=float, default=0.2)
    parser.add_argument("--final-atol", type=float, default=3e-2)
    parser.add_argument("--final-rtol", type=float, default=3e-2)
    parser.add_argument("--max-report", type=int, default=20)
    parser.add_argument("--fail-on-alarm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hq % args.h_kv != 0:
        raise ValueError("hq must be divisible by h-kv (GQA)")
    if args.k_dim < 32 or (args.k_dim % 2) != 0:
        raise ValueError("k-dim must be even and >= 32 for current Triton dot kernel")
    if (args.k_dim & (args.k_dim - 1)) != 0:
        raise ValueError("k-dim must be a power of 2 (kernel reshape constraint)")
    if args.v_dim <= 0 or (args.v_dim & (args.v_dim - 1)) != 0:
        raise ValueError("v-dim must be a power of 2 (kernel reshape constraint)")
    if args.compression_rate <= 0 or (args.compression_rate & (args.compression_rate - 1)) != 0:
        raise ValueError("compression-rate must be a power of 2")
    if args.top_k_per_layer <= 0 or (args.top_k_per_layer & (args.top_k_per_layer - 1)) != 0:
        raise ValueError("top-k-per-layer must be a power of 2")
    if args.top_k_per_layer % args.compression_rate != 0:
        raise ValueError("top-k-per-layer must be divisible by compression-rate")
    if args.top_k_per_layer % 64 != 0:
        raise ValueError("top-k-per-layer must be divisible by 64 (kernel DROP_BLOCK constraint)")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = _to_dtype(args.dtype)
    max_top_nodes = args.max_top_nodes
    if max_top_nodes is None:
        max_top_nodes = args.compression_rate * args.top_k_per_layer
    if max_top_nodes != args.compression_rate * args.top_k_per_layer:
        raise ValueError("max-top-nodes must equal compression-rate * top-k-per-layer")
    if args.seq_len <= max_top_nodes:
        print(_fmt_warn(
            f"[warning] seq-len ({args.seq_len}) <= max-top-nodes ({max_top_nodes}), "
            "tree has only bottom layer and no TopK selection."
        ))

    torch.manual_seed(args.seed)
    q = torch.randn(args.batch, args.seq_len, args.hq, args.k_dim, device=device, dtype=dtype).contiguous()
    k = torch.randn(args.batch, args.seq_len, args.h_kv, args.k_dim, device=device, dtype=dtype).contiguous()
    v = torch.randn(args.batch, args.seq_len, args.h_kv, args.v_dim, device=device, dtype=dtype).contiguous()
    scale = args.k_dim ** -0.5

    with torch.no_grad():
        out_parallel, trace_parallel = run_parallel_with_trace(
            q=q,
            k=k,
            v=v,
            compression_rate=args.compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=args.top_k_per_layer,
            scale=scale,
            rope_base=args.rope_base,
        )
        out_naive, trace_naive = htree_forward_naive(
            q=q,
            k=k,
            v=v,
            compression_rate=args.compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=args.top_k_per_layer,
            scale=scale,
            rope_base=args.rope_base,
            selection_mode=args.selection_mode,
            return_trace=True,
        )

    final_diff = (out_parallel.float() - out_naive.float()).abs()
    final_max_abs = final_diff.max().item()
    final_mean_abs = final_diff.mean().item()
    final_p99_abs = torch.quantile(final_diff.flatten(), 0.99).item()
    final_allclose = torch.allclose(
        out_parallel.float(),
        out_naive.float(),
        atol=args.final_atol,
        rtol=args.final_rtol,
    )

    trace_report = compare_traces(
        trace_parallel,
        trace_naive,
        topk_alarm_diff=args.topk_alarm_diff,
        layer_out_alarm_abs=args.layer_out_alarm_abs,
        max_report=args.max_report,
    )
    _attach_rank_exact_to_topk_alarms(
        trace_report,
        trace_parallel,
        q,
        k,
        v,
        compression_rate=args.compression_rate,
        top_k_per_layer=args.top_k_per_layer,
        max_top_nodes=max_top_nodes,
        rope_base=args.rope_base,
    )

    has_alarm = bool(trace_report["topk_alarms"] or trace_report["layer_output_alarms"])
    if final_allclose and not has_alarm:
        status = _fmt_ok("PASS")
    elif final_allclose and has_alarm:
        status = _fmt_warn("PASS_WITH_ALARM")
    else:
        status = _fmt_error("FAIL")

    print(_fmt_header("========== HTree Correctness Report =========="))
    print(_fmt_dim(f"overall status: {status}"))
    print(
        f"config: B={args.batch}, T={args.seq_len}, H={args.hq}, H_kv={args.h_kv}, "
        f"K={args.k_dim}, V={args.v_dim}, dtype={args.dtype}"
    )
    print(
        f"tree: compression_rate={args.compression_rate}, top_k_per_layer={args.top_k_per_layer}, "
        f"max_top_nodes={max_top_nodes}, selection_mode={args.selection_mode}"
    )
    diff_line = (
        f"final output diff: max_abs={final_max_abs:.6e}, mean_abs={final_mean_abs:.6e}, "
        f"p99_abs={final_p99_abs:.6e}, allclose={final_allclose}"
    )
    print(_fmt_ok(diff_line) if final_allclose else _fmt_error(diff_line))

    print(_fmt_header("--- Selection Stats ---"))
    print(
        "topk mismatch histogram (set symmetric diff size, ignore order): "
        + _sorted_hist_str(trace_report["mismatch_counter"])
    )
    topk_alarm_cnt = len(trace_report["topk_alarms"])
    out_alarm_cnt = len(trace_report["layer_output_alarms"])
    print((_fmt_ok if topk_alarm_cnt == 0 else _fmt_warn)(f"topk alarms: {topk_alarm_cnt}"))
    print((_fmt_ok if out_alarm_cnt == 0 else _fmt_warn)(f"layer-output alarms: {out_alarm_cnt}"))

    if trace_report["topk_alarms"]:
        print(_fmt_warn("\n--- topk alarms (truncated) ---"))
        for item in trace_report["topk_alarms"]:
            print(_fmt_warn(
                f"layer={item['layer']} b={item['b']} t={item['t']} kv_h={item['kv_h']} "
                f"diff_count(set,ignore-order)={item['diff_count']}"
            ))
            print(_fmt_dim(f"  parallel_only={_format_node_rank_list(item['parallel_only'])}"))
            print(_fmt_dim(f"  naive_only   ={_format_node_rank_list(item['naive_only'])}"))

    if trace_report["layer_output_alarms"]:
        print(_fmt_warn("\n--- layer-output alarms (truncated) ---"))
        for item in trace_report["layer_output_alarms"]:
            print(_fmt_warn(
                f"layer={item['layer']} b={item['b']} t={item['t']} kv_h={item['kv_h']} "
                f"max_abs={item['max_abs']:.6e}"
            ))

    print(_fmt_header("============================================="))
    if args.fail_on_alarm and (has_alarm or not final_allclose):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
