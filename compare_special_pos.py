#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Debug: compare shared Top-K (GQA) between naive_gqa and parallel kernels.

Goal
  - Compare Top-K buffer positions selected by:
      * parallel.py: Kernel 2.1a (scores) + Kernel 2.1b (shared Top-K)
      * naive reference: Python reconstruction of the same candidate list + importance +
        stable Top-K with bit-packing (reusing naive_gqa helpers)

Design constraints
  - DO NOT modify src/parallel.py
  - To avoid huge all_scores for large H, we run per KV head by slicing:
      H_sub = NUM_GROUPS (= H/H_kv), H_kv_sub = 1
    This keeps memory proportional to group size.

Output
    - Prints full 512 Top-K lists to console (with colors).
    - No file output by default.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch

from src import parallel as par
from src import naive_gqa


@dataclass
class RunConfig:
    B: int
    T: int
    H: int
    H_kv: int
    K: int
    V: int
    compression_rate: int
    max_top_nodes: int
    top_k: int
    rope_base: float
    seed: int
    device: str

class _Color:
    END = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _use_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"{code}{text}{_Color.END}"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log_info(msg: str) -> None:
    print(f"{_c(_ts(), _Color.DIM)} {_c('[INFO]', _Color.CYAN)} {msg}")


def log_warn(msg: str) -> None:
    print(f"{_c(_ts(), _Color.DIM)} {_c('[WARN]', _Color.YELLOW)} {msg}")


def log_error(msg: str) -> None:
    print(f"{_c(_ts(), _Color.DIM)} {_c('[ERR ]', _Color.RED)} {msg}")


def _ensure_cuda(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("This debug script requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    return dev


def _estimate_all_scores_bytes(B: int, T: int, H_sub: int, max_candidates: int = 8192) -> int:
    # float32
    return B * T * H_sub * max_candidates * 4


def _format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def _build_tree_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int,
    max_top_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build tree using par.htree_build_kernel_v2 (same as parallel forward)."""
    assert k.ndim == 4 and v.ndim == 4
    B, T, H_kv, K = k.shape
    V = v.shape[-1]
    layers_k: List[torch.Tensor] = [k]
    layers_v: List[torch.Tensor] = [v]

    # Determine num_layers
    num_layers = 1
    temp_len = T
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1

    current_k, current_v = k, v
    current_len = T
    for _layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty((B, next_len, H_kv, K), device=k.device, dtype=k.dtype)
        next_v = torch.empty((B, next_len, H_kv, V), device=v.device, dtype=v.dtype)

        grid = (B * H_kv,)
        par.htree_build_kernel_v2[grid](
            current_k,
            current_v,
            next_k,
            next_v,
            N_child=current_len,
            N_parent=next_len,
            B=B,
            H_kv=H_kv,
            K=K,
            V=V,
            COMPRESSION_RATE=compression_rate,
            BLOCK_SIZE=128,
        )
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len

    return layers_k, layers_v


def _precompute_rope_cache(
    K: int,
    cache_size: int,
    rope_base: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
    positions = torch.arange(cache_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _init_prev_selected_parents_top_layer(
    B: int,
    T: int,
    top_k: int,
    compression_rate: int,
    top_layer_power: int,
    device: torch.device,
) -> torch.Tensor:
    """Same initialization as htree_forward_v2, but returns [B,T,1,TOP_K] for a single KV head."""
    t_indices = torch.arange(T, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_candidates = torch.arange(top_k, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected = torch.where(
        valid_mask,
        parent_candidates,
        torch.tensor(-1, dtype=torch.int32, device=device),
    )
    prev_selected = prev_selected.unsqueeze(0).unsqueeze(2).expand(B, T, 1, top_k).contiguous()
    return prev_selected


def _python_reference_shared_topk_positions(
    q_group: torch.Tensor,  # [B,T,G,K]
    layer_k_kv: torch.Tensor,  # [B,N,1,K]
    prev_selected_parents: torch.Tensor,  # [B,T,1,TOP_K]
    cos_cache: torch.Tensor,  # [cache, K/2]
    sin_cache: torch.Tensor,  # [cache, K/2]
    layer_idx: int,
    layer_power: int,
    compression_rate: int,
    top_k: int,
    t_pos: int,
    scale: float,
) -> torch.Tensor:
    """Pure PyTorch reference that mirrors parallel Kernel 2.1a+2.1b for one (b=0,t=t_pos,kv=0).

    Returns: topk buffer positions [TOP_K] int32 (padded with -1)
    """
    assert q_group.shape[0] == 1, "This reference currently assumes B=1 for simplicity"
    device = q_group.device
    B, T, G, K = q_group.shape
    assert t_pos >= 0 and t_pos < T
    assert layer_k_kv.shape[0] == 1 and layer_k_kv.shape[2] == 1
    N_layer = layer_k_kv.shape[1]

    # Rightmost info (same as kernel)
    rightmost_idx = t_pos // layer_power
    rightmost_parent_idx = rightmost_idx // compression_rate
    rightmost_child_idx = rightmost_idx % compression_rate

    # Parent list and n_cand
    parent_list = prev_selected_parents[0, t_pos, 0].to(torch.int32)  # [TOP_K]
    valid = parent_list >= 0
    num_valid_parents = int(valid.sum().item())
    if num_valid_parents <= 0:
        return torch.full((top_k,), -1, dtype=torch.int32, device=device)
    n_cand = (num_valid_parents - 1) * compression_rate + rightmost_child_idx + 1
    n_cand = int(n_cand)
    if n_cand <= 0:
        return torch.full((top_k,), -1, dtype=torch.int32, device=device)

    # Build candidate node indices in buffer order (batch->parent_slot->child_slot)
    parents_per_batch = 32
    BC = parents_per_batch * compression_rate  # 512
    num_batches = (num_valid_parents + parents_per_batch - 1) // parents_per_batch

    cand_node_idx = torch.empty((n_cand,), dtype=torch.int64, device=device)
    write = 0
    for b_id in range(num_batches):
        start = b_id * parents_per_batch
        end = min(start + parents_per_batch, num_valid_parents)
        batch_parents = parent_list[:num_valid_parents][start:end]  # [<=32]
        for p_slot in range(batch_parents.numel()):
            p_idx = int(batch_parents[p_slot].item())
            if p_idx < 0:
                continue
            # children count
            if p_idx == rightmost_parent_idx:
                n_child = rightmost_child_idx + 1
            else:
                n_child = compression_rate
            # Emit
            for c in range(n_child):
                if write >= n_cand:
                    break
                node = p_idx * compression_rate + c
                if node >= N_layer:
                    # padding nodes may go out of range; kernel would boundary-check and
                    # treat invalid loads as masked. Here we clamp to 0 but will mask by range.
                    node = 0
                cand_node_idx[write] = node
                write += 1
        if write >= n_cand:
            break

    # Rope positions are exactly buffer positions 0..n_cand-1
    rope_pos_k = torch.arange(n_cand, device=device, dtype=torch.int64)
    rope_pos_q = n_cand - 1

    # Gather K and apply RoPE
    k_raw = layer_k_kv[0, cand_node_idx, 0].to(torch.float32)  # [n_cand, K]
    q_raw = q_group[0, t_pos].to(torch.float32)  # [G, K]

    cos_k = cos_cache[rope_pos_k]  # [n_cand, K/2]
    sin_k = sin_cache[rope_pos_k]
    k_rope = naive_gqa.apply_rotary_emb(k_raw, cos_k, sin_k)  # [n_cand, K]

    cos_q = cos_cache[rope_pos_q]
    sin_q = sin_cache[rope_pos_q]
    q_rope = naive_gqa.apply_rotary_emb(q_raw, cos_q, sin_q)  # [G, K]

    scores = (q_rope * float(scale)) @ k_rope.transpose(0, 1)  # [G, n_cand]

    # importance_i = sum_g exp(score_{g,i} - lse_g)
    lse = torch.logsumexp(scores, dim=1)  # [G]
    imp = torch.exp(scores - lse[:, None]).sum(dim=0)  # [n_cand]

    # Force rightmost selected (same as Kernel 2.1b)
    imp = imp.to(torch.float32)
    imp[rightmost_pos := (n_cand - 1)] = 1e3

    # Stable Top-K positions using bit-packing (same semantics)
    topk_buf = naive_gqa.stable_topk_positions_with_bitpacking(
        imp,
        k=top_k,
        force_select_pos=rightmost_pos,
        log_n=13,
    )
    return topk_buf


def _python_topk_from_triton_scores(
    all_scores: torch.Tensor,  # [B,T,G,MAX_CAND]
    num_candidates: torch.Tensor,  # [B,T,1]
    top_k: int,
    t_pos: int,
) -> torch.Tensor:
    """(Deprecated) Pure PyTorch version of parallel Kernel 2.1b, consuming Kernel 2.1a outputs.

    Kept only for reference; the script now calls Triton Kernel 2.1b directly.
    """
    assert all_scores.shape[0] == 1, "This helper assumes B=1"
    device = all_scores.device
    n_cand = int(num_candidates[0, t_pos, 0].item())
    if n_cand <= 0:
        return torch.full((top_k,), -1, dtype=torch.int32, device=device)
    scores = all_scores[0, t_pos, :, :n_cand].to(torch.float32)  # [G,n_cand]
    lse = torch.logsumexp(scores, dim=1)  # [G]
    imp = torch.exp(scores - lse[:, None]).sum(dim=0).to(torch.float32)  # [n_cand]
    rightmost_pos = n_cand - 1
    imp[rightmost_pos] = 1e3
    return naive_gqa.stable_topk_positions_with_bitpacking(
        imp,
        k=top_k,
        force_select_pos=rightmost_pos,
        log_n=13,
    )


def _map_topk_positions_to_node_indices(
    prev_selected_parents_cpu: torch.Tensor,  # [B,T,1,TOP_K] on CPU
    topk_positions_cpu: torch.Tensor,  # [TOP_K] int32 on CPU
    compression_rate: int,
    t_pos: int,
) -> torch.Tensor:
    """Map buffer positions (0..MAX_CAND-1) to node indices (parent*CR + child).

    Mirrors `htree_compute_next_parents_kernel` mapping; returns [TOP_K] int32, -1 padded.
    """
    topk_pos = topk_positions_cpu.to(torch.int64)
    valid = topk_pos >= 0
    parents_per_batch = 32
    BC = parents_per_batch * compression_rate

    batch_id = torch.where(valid, topk_pos // BC, torch.zeros_like(topk_pos))
    within_batch = torch.where(valid, topk_pos - batch_id * BC, torch.zeros_like(topk_pos))
    parent_slot = torch.where(valid, within_batch // compression_rate, torch.zeros_like(within_batch))
    child_slot = torch.where(valid, within_batch - parent_slot * compression_rate, torch.zeros_like(within_batch))

    parent_list = prev_selected_parents_cpu[0, t_pos, 0].to(torch.int64)  # [TOP_K]
    gather_idx = (batch_id * parents_per_batch + parent_slot).clamp(min=0, max=parent_list.numel() - 1)
    parent_idx = torch.where(valid, parent_list.gather(0, gather_idx), -1)
    node_idx = torch.where(valid & (parent_idx >= 0), parent_idx * compression_rate + child_slot, -1)
    return node_idx.to(torch.int32)


def _print_topk_console(
    title: str,
    topk_positions_cpu: torch.Tensor,
    node_indices_cpu: torch.Tensor,
    color_code: str,
) -> None:
    print(_c(f"\n[{title}]", color_code))
    print(_c("positions:", _Color.DIM), " ".join(map(str, topk_positions_cpu.tolist())))
    print(_c("nodes:    ", _Color.DIM), " ".join(map(str, node_indices_cpu.tolist())))


def _print_set_diff(
    a_name: str,
    a_nodes_cpu: torch.Tensor,
    b_name: str,
    b_nodes_cpu: torch.Tensor,
    limit: int = 256,
) -> None:
    """Compare Top-K as sets (order-insensitive) and highlight symmetric differences."""
    a_set = set(int(x) for x in a_nodes_cpu.tolist() if x >= 0)
    b_set = set(int(x) for x in b_nodes_cpu.tolist() if x >= 0)
    only_a = sorted(a_set - b_set)
    only_b = sorted(b_set - a_set)

    if not only_a and not only_b:
        log_info(_c("Top-K SET MATCH (order ignored)", _Color.GREEN))
        return

    log_warn(_c("Top-K SET MISMATCH (order ignored)", _Color.YELLOW))
    print(
        _c(f"only_in_{a_name} nodes (count={len(only_a)}):", _Color.RED),
        " ".join(map(str, only_a[:limit])) + (" ..." if len(only_a) > limit else ""),
    )
    print(
        _c(f"only_in_{b_name} nodes (count={len(only_b)}):", _Color.RED),
        " ".join(map(str, only_b[:limit])) + (" ..." if len(only_b) > limit else ""),
    )


def _compare_final_output(
    cfg: RunConfig,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t_pos: int,
) -> None:
    """Run full forward (parallel + naive) and compare output at a single position."""
    if cfg.K != cfg.V:
        raise ValueError(
            "For final-output comparison, this script assumes K==V so that naive_gqa.forward_kernel "
            "returns output with the same last-dim size as parallel."
        )

    log_info(_c("Running full parallel forward (htree_forward_v2)...", _Color.CYAN))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out_par = par.htree_forward_v2(
            q,
            k,
            v,
            compression_rate=cfg.compression_rate,
            max_top_nodes=cfg.max_top_nodes,
            top_k_per_layer=cfg.top_k,
            scale=float(cfg.K ** -0.5),
            rope_base=cfg.rope_base,
        )

    log_info(_c("Running naive forward at single position...", _Color.CYAN))
    query_positions = torch.tensor([[t_pos]], device=q.device, dtype=torch.int64)
    out_naive_full = naive_gqa.forward_kernel(
        q,
        k,
        v,
        query_positions=query_positions,
        compression_rate=cfg.compression_rate,
        max_top_nodes=cfg.max_top_nodes,
        top_k_per_layer=cfg.top_k,
        scale=float(cfg.K ** -0.5),
    )

    out_par_pos = out_par[0, t_pos].detach().cpu().to(torch.float32)  # [H,V]
    out_naive_pos = out_naive_full[0, t_pos].detach().cpu().to(torch.float32)  # [H,V]
    diff = out_par_pos - out_naive_pos
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()

    print(_c(f"\n[final_output@t={t_pos}]", _Color.BOLD))
    print(_c("parallel:", _Color.MAGENTA))
    for h in range(cfg.H):
        print(_c(f"  h{h}: ", _Color.DIM) + " ".join(f"{x:.6e}" for x in out_par_pos[h].tolist()))

    print(_c("naive:", _Color.BLUE))
    for h in range(cfg.H):
        print(_c(f"  h{h}: ", _Color.DIM) + " ".join(f"{x:.6e}" for x in out_naive_pos[h].tolist()))

    print(_c("diff (parallel - naive):", _Color.YELLOW))
    for h in range(cfg.H):
        print(_c(f"  h{h}: ", _Color.DIM) + " ".join(f"{x:.6e}" for x in diff[h].tolist()))

    log_info(
        _c(
            f"final_output_diff: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}",
            _Color.GREEN if max_abs < 1e-3 else _Color.YELLOW,
        )
    )


def _run_one_kv_head(
    cfg: RunConfig,
    q: torch.Tensor,
    layers_k: List[torch.Tensor],
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_head: int,
    target_layer: int,
    t_pos: int,
    check_scores: bool,
) -> str:
    """Run decomposed kernels for a single KV head and compare Top-K at target_layer / t_pos."""
    assert cfg.top_k == 512, "parallel kernels currently assume TOP_K=512"
    device = q.device

    G = cfg.H // cfg.H_kv
    # Slice query heads for this KV head
    q_group = q[:, :, kv_head * G : (kv_head + 1) * G, :].contiguous()

    # Init parents at top layer
    num_layers = len(layers_k)
    top_layer = num_layers - 1
    top_layer_power = cfg.compression_rate ** top_layer
    prev_selected = _init_prev_selected_parents_top_layer(
        B=cfg.B,
        T=cfg.T,
        top_k=cfg.top_k,
        compression_rate=cfg.compression_rate,
        top_layer_power=top_layer_power,
        device=device,
    )  # [B,T,1,TOP_K]

    # Walk top -> target_layer
    captured_prev_selected = None
    captured_layer_k = None
    captured_layer_power = None
    captured_topk_positions = None
    captured_all_scores = None
    captured_num_candidates = None

    MAX_CANDIDATES = 8192
    scale = float(cfg.K ** -0.5)

    log_info(
        f"kv_head={kv_head} (G={G}) build path: layer {top_layer} -> {target_layer}, t_pos={t_pos}"
    )

    for layer_idx in range(top_layer, target_layer - 1, -1):
        layer_k_full = layers_k[layer_idx]
        layer_k = layer_k_full[:, :, kv_head : kv_head + 1, :].contiguous()  # [B,N,1,K]
        N_layer = layer_k.shape[1]
        layer_power = cfg.compression_rate ** layer_idx

        # Allocate buffers for this KV head group (H_sub=G, H_kv_sub=1)
        all_scores = torch.empty((cfg.B, cfg.T, G, MAX_CANDIDATES), device=device, dtype=torch.float32)
        num_candidates = torch.empty((cfg.B, cfg.T, 1), device=device, dtype=torch.int32)
        topk_positions = torch.empty((cfg.B, cfg.T, 1, cfg.top_k), device=device, dtype=torch.int32)

        log_info(f"Kernel 2.1a (scores) @ layer={layer_idx} N={N_layer}")

        grid_scores = (cfg.T, cfg.B * G)
        par.htree_compute_scores_kernel[grid_scores](
            q_group,
            layer_k,
            prev_selected,
            cos_cache,
            sin_cache,
            all_scores,
            num_candidates,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=cfg.B,
            T=cfg.T,
            H=G,
            H_kv=1,
            NUM_GROUPS=G,
            K=cfg.K,
            V=cfg.V,
            N_layer=N_layer,
            COMPRESSION_RATE=cfg.compression_rate,
            TOP_K=cfg.top_k,
            MAX_CANDIDATES=MAX_CANDIDATES,
            SCORE_VALID_THRESHOLD=par.HTREE_SCORE_VALID_THRESHOLD,
            scale=scale,
            num_warps=4,
        )

        # Kernel 2.1b: Shared Top-K selection (KV head granularity, non-bottom layers only)
        grid_topk = (cfg.T, cfg.B * 1)
        par.htree_select_topk_shared_gqa_kernel[grid_topk](
            all_scores,
            num_candidates,
            topk_positions,
            layer_idx=layer_idx,
            B=cfg.B,
            T=cfg.T,
            H=G,
            H_kv=1,
            NUM_GROUPS=G,
            TOP_K=cfg.top_k,
            MAX_CANDIDATES=MAX_CANDIDATES,
            COMPRESSION_RATE=cfg.compression_rate,
            SCORE_VALID_THRESHOLD=par.HTREE_SCORE_VALID_THRESHOLD,
            num_warps=4,
        )

        if layer_idx == target_layer:
            captured_prev_selected = prev_selected
            captured_layer_k = layer_k
            captured_layer_power = layer_power
            captured_topk_positions = topk_positions
            captured_all_scores = all_scores
            captured_num_candidates = num_candidates
            break

        # Prepare next layer parents
        next_selected = torch.empty_like(prev_selected)
        par.htree_compute_next_parents_kernel[grid_topk](
            prev_selected,
            topk_positions,
            next_selected,
            B=cfg.B,
            T=cfg.T,
            H_kv=1,
            COMPRESSION_RATE=cfg.compression_rate,
            TOP_K=cfg.top_k,
            num_warps=4,
        )
        prev_selected = next_selected

    assert captured_prev_selected is not None
    assert captured_layer_k is not None
    assert captured_layer_power is not None
    assert captured_topk_positions is not None

    torch.cuda.synchronize()

    # "Parallel" selected Top-K positions (computed from Kernel 2.1a outputs with Python 2.1b-equivalent selection)
    par_topk_pos = captured_topk_positions[0, t_pos, 0].detach().cpu().to(torch.int32)

    # Python reference selected Top-K positions
    ref_topk_pos = _python_reference_shared_topk_positions(
        q_group=q_group,
        layer_k_kv=captured_layer_k,
        prev_selected_parents=captured_prev_selected,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        layer_idx=target_layer,
        layer_power=captured_layer_power,
        compression_rate=cfg.compression_rate,
        top_k=cfg.top_k,
        t_pos=t_pos,
        scale=scale,
    ).detach().cpu().to(torch.int32)

    prev_sel_cpu = captured_prev_selected.detach().cpu()
    par_nodes = _map_topk_positions_to_node_indices(prev_sel_cpu, par_topk_pos, cfg.compression_rate, t_pos)
    ref_nodes = _map_topk_positions_to_node_indices(prev_sel_cpu, ref_topk_pos, cfg.compression_rate, t_pos)

    _print_topk_console("parallel_topk", par_topk_pos, par_nodes, _Color.MAGENTA)
    _print_topk_console("naive_topk", ref_topk_pos, ref_nodes, _Color.BLUE)
    _print_set_diff("parallel", par_nodes, "naive", ref_nodes)

    # Optional score sanity check: compare some entries between kernel scores and python recompute
    score_report = ""
    if check_scores:
        # Compare per-head scores for first min(n_cand, 256) candidates
        n_cand = int(captured_num_candidates[0, t_pos, 0].item())
        n_chk = min(n_cand, 256)
        # Recompute python scores for the same candidates (using the same reference path)
        # We'll compute max abs diff across heads/candidates for this prefix.
        device2 = q_group.device
        G = q_group.shape[2]
        # Reconstruct candidates as in reference
        parent_list = captured_prev_selected[0, t_pos, 0].to(torch.int32)
        valid = parent_list >= 0
        num_valid_parents = int(valid.sum().item())
        rightmost_idx = t_pos // captured_layer_power
        rightmost_parent_idx = rightmost_idx // cfg.compression_rate
        rightmost_child_idx = rightmost_idx % cfg.compression_rate
        n_cand2 = (num_valid_parents - 1) * cfg.compression_rate + rightmost_child_idx + 1
        n_cand2 = int(n_cand2)
        parents_per_batch = 32
        num_batches = (num_valid_parents + parents_per_batch - 1) // parents_per_batch
        cand_node_idx = torch.empty((n_cand2,), dtype=torch.int64, device=device2)
        write = 0
        for b_id in range(num_batches):
            start = b_id * parents_per_batch
            end = min(start + parents_per_batch, num_valid_parents)
            batch_parents = parent_list[:num_valid_parents][start:end]
            for p_slot in range(batch_parents.numel()):
                p_idx = int(batch_parents[p_slot].item())
                if p_idx < 0:
                    continue
                if p_idx == rightmost_parent_idx:
                    n_child = rightmost_child_idx + 1
                else:
                    n_child = cfg.compression_rate
                for c in range(n_child):
                    if write >= n_cand2:
                        break
                    node = p_idx * cfg.compression_rate + c
                    if node >= captured_layer_k.shape[1]:
                        node = 0
                    cand_node_idx[write] = node
                    write += 1
            if write >= n_cand2:
                break
        cand_node_idx = cand_node_idx[:n_chk]
        rope_pos_k = torch.arange(n_chk, device=device2, dtype=torch.int64)
        rope_pos_q = n_cand2 - 1

        k_raw = captured_layer_k[0, cand_node_idx, 0].to(torch.float32)
        q_raw = q_group[0, t_pos].to(torch.float32)
        k_rope = naive_gqa.apply_rotary_emb(k_raw, cos_cache[rope_pos_k], sin_cache[rope_pos_k])
        q_rope = naive_gqa.apply_rotary_emb(q_raw, cos_cache[rope_pos_q], sin_cache[rope_pos_q])
        scores_py = (q_rope * scale) @ k_rope.transpose(0, 1)  # [G,n_chk]

        scores_tr = captured_all_scores[0, t_pos, :, :n_chk].to(torch.float32)
        max_abs = (scores_tr - scores_py).abs().max().item() if n_chk > 0 else 0.0
        score_report = f"score_check: n_cand={n_cand} n_chk={n_chk} max_abs_diff={max_abs:.6e}"
        log_info(score_report)

    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7, help="CUDA device index (default: 7)")
    args = ap.parse_args()

    # Hard-coded test config (edit here if needed)
    B, T, H, H_kv, K, V = 1, 20000, 4, 1, 4, 4
    compression_rate = 16
    max_top_nodes = 8192
    top_k = 512
    rope_base = 10000.0
    seed = 42
    kv_head = 0
    # Pick a single query position to inspect (must be < T)
    t_pos = 9584
    # Target a non-bottom layer so Kernel 2.1b actually runs.
    # With T=10000 and max_top_nodes=8192, there are 2 layers: layer 1 (top) and layer 0 (bottom).
    target_layer = 1

    dev = _ensure_cuda(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)

    if H % H_kv != 0:
        raise ValueError(f"H ({H}) must be divisible by H_kv ({H_kv})")
    if top_k != 512:
        raise ValueError("This debug script (and parallel kernels) currently require TOP_K=512")
    if not (0 <= kv_head < H_kv):
        raise ValueError(f"kv_head must be in [0, {H_kv})")
    if not (0 <= t_pos < T):
        raise ValueError("t_pos out of range")

    cfg = RunConfig(
        B=B,
        T=T,
        H=H,
        H_kv=H_kv,
        K=K,
        V=V,
        compression_rate=compression_rate,
        max_top_nodes=max_top_nodes,
        top_k=top_k,
        rope_base=rope_base,
        seed=seed,
        device=str(dev),
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Inputs (FP32)
    q = torch.randn((cfg.B, cfg.T, cfg.H, cfg.K), device=dev, dtype=torch.float32).contiguous()
    k = torch.randn((cfg.B, cfg.T, cfg.H_kv, cfg.K), device=dev, dtype=torch.float32).contiguous()
    v = torch.randn((cfg.B, cfg.T, cfg.H_kv, cfg.V), device=dev, dtype=torch.float32).contiguous()

    # Tree
    log_info(
        f"Config: B={cfg.B} T={cfg.T} H={cfg.H} H_kv={cfg.H_kv} K={cfg.K} V={cfg.V} "
        f"compression_rate={cfg.compression_rate} top_k={cfg.top_k} (inputs=fp32)"
    )
    layers_k, _layers_v = _build_tree_triton(k, v, cfg.compression_rate, cfg.max_top_nodes)
    num_layers = len(layers_k)
    if not (0 <= target_layer < num_layers):
        raise ValueError(f"target_layer must be in [0,{num_layers})")

    # RoPE cache: kernel uses positions up to MAX_CANDIDATES (8192) within a layer.
    # We mirror htree_forward_v2 cache sizing.
    cache_size = cfg.max_top_nodes + 1024
    cos_cache, sin_cache = _precompute_rope_cache(cfg.K, cache_size, cfg.rope_base, dev)

    # Memory safety check (per KV head slice)
    G = cfg.H // cfg.H_kv
    bytes_needed = _estimate_all_scores_bytes(cfg.B, cfg.T, G, 8192)
    if bytes_needed > 6 * 1024**3:
        raise RuntimeError(
            "Estimated all_scores allocation is too large for a single KV head slice. "
            f"Need ~{_format_bytes(bytes_needed)} for all_scores alone (float32). "
            "Reduce T or H/H_kv (so group size G shrinks)."
        )

    _run_one_kv_head(
        cfg=cfg,
        q=q,
        layers_k=layers_k,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        kv_head=kv_head,
        target_layer=target_layer,
        t_pos=t_pos,
        check_scores=True,
    )

    # Continue to run the full forward and compare final output at t_pos.
    _compare_final_output(cfg=cfg, q=q, k=k, v=v, t_pos=t_pos)


if __name__ == "__main__":
    main()
