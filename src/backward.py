# -*- coding: utf-8 -*-
"""
HTree Backward Triton Kernels

Kernel architecture (FlashAttention-2 style split dQ / dKV):

  K0:  htree_bwd_preprocess_delta       D = rowsum(dO * O)
  K1a: htree_bwd_dq_bottom              Q-stationary dQ for bottom layer
  K1b: htree_bwd_dq_upper               Q-stationary dQ for upper layers (dropped only)
  K2:  htree_bwd_build_parent_mask      dense reverse index (reused for bottom & upper)
  K3:  htree_bwd_dkv_bottom             K-stationary dKV for bottom layer (no atomics)
  K3u: htree_bwd_dkv_upper              K-stationary dKV for upper layers (no atomics)
  K4:  htree_bwd_tree_backward          mean-pool gradient propagation
"""

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from src.parallel import (
    _cuda_timer,
    htree_forward,
    htree_build_kernel,
    htree_recompute_selection_kernel,
    HTREE_SCORE_NEG_INF,
    HTREE_SCORE_VALID_THRESHOLD,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================
#  K0: Preprocess Delta  –  D[b,t,h] = sum_v(dO[b,t,h,v] * O[b,t,h,v])
# ============================================================

@triton.jit
def htree_bwd_preprocess_delta_kernel(
    o_ptr,       # [B, T, H, V]
    do_ptr,      # [B, T, H, V]
    delta_ptr,   # [B, T, H]        (output)
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """Grid: (T, B*H)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    T_i64 = T.to(tl.int64)
    base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V + i_h.to(tl.int64) * V
    o_v = tl.arange(0, V).to(tl.int64)
    o_vals = tl.load(o_ptr + base + o_v).to(tl.float32)
    do_vals = tl.load(do_ptr + base + o_v).to(tl.float32)
    d = tl.sum(o_vals * do_vals)

    d_off = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H + i_h.to(tl.int64)
    tl.store(delta_ptr + d_off, d)


# ============================================================
#  K2: Build Parent Mask (reverse index for bottom-layer K3)
# ============================================================

@triton.jit
def htree_bwd_build_parent_mask_kernel(
    selected_parents,   # [B, T, H_kv, TOP_K]  (bottom layer parents)
    parent_mask,        # [B, H_kv, num_groups, T]  (output, bool packed as uint8)
    T,
    H_kv: tl.constexpr,
    TOP_K: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_TK: tl.constexpr,
):
    """For each (b, t, kv_h), mark which parent groups are attended.
    Grid: (T, B * H_kv).
    """
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T.to(tl.int64)
    sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )

    for tk_off in range(0, TOP_K, BLOCK_TK):
        offs = tl.arange(0, BLOCK_TK)
        parents = tl.load(selected_parents + sel_base + tk_off + offs, mask=(tk_off + offs) < TOP_K, other=-1)
        valid = parents >= 0
        safe_g = tl.where(valid, parents, 0).to(tl.int64)

        mask_base = (
            i_b.to(tl.int64) * H_kv * num_groups * T_i64
            + i_h_kv.to(tl.int64) * num_groups * T_i64
            + i_t.to(tl.int64)
        )
        mask_ptrs = parent_mask + mask_base + safe_g * T_i64
        tl.store(mask_ptrs, tl.full([BLOCK_TK], 1, dtype=tl.uint8), mask=valid)


# ============================================================
#  K1-bottom: dQ for bottom layer  (Q-stationary, load-add-store to dq)
# ============================================================

@triton.jit
def htree_bwd_dq_bottom_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,    # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,     # [cache_size, K//2]
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    dq,                       # [B, T, H, K]  (accumulated output)
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
):
    """Grid: (T, B * H_kv). Mirrors forward K1a."""
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T.to(tl.int64)

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    if num_valid_parents <= 0:
        return

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # Load Q heads for this kv group
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)

    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    # Load dO and D
    do_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
    o_v = tl.arange(0, V).to(tl.int64)
    do_block = tl.load(do + do_base + head_ids[:, None] * V + o_v[None, :],
                       mask=g_valid[:, None], other=0.0).to(tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    D_vals = tl.load(delta + state_base + head_ids, mask=g_valid, other=0.0)
    gm = tl.load(global_max + state_base + head_ids, mask=g_valid, other=0.0)
    gs = tl.load(global_sum + state_base + head_ids, mask=g_valid, other=1.0)

    # dQ accumulators (in RoPE'd space)
    dq_tilde_lo = tl.zeros([G_PAD, K // 2], dtype=tl.float32)
    dq_tilde_hi = tl.zeros([G_PAD, K // 2], dtype=tl.float32)

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    for tile_idx in range(num_tiles):
        tile_base = (tile_idx * TILE_P).to(tl.int32)
        p_offs = tile_base + parent_in_tile

        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(prev_selected_parents + prev_sel_base + safe_p_offs).to(tl.int32)

        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)

        # Gather K, apply RoPE
        k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        rope_pos = tile_base.to(tl.int64) * COMPRESSION_RATE + o_n.to(tl.int64)
        rope_pos = tl.maximum(rope_pos, 0)
        cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
        sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
        k_rope_lo = k_lo * cos_k - k_hi * sin_k
        k_rope_hi = k_lo * sin_k + k_hi * cos_k

        # Scores: [G_PAD, TILE_N]
        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, 0.0)

        # Reconstruct P = exp(S - global_max) / global_sum
        P = tl.exp(scores - gm[:, None]) / gs[:, None]
        P = tl.where(valid[None, :] & g_valid[:, None], P, 0.0)

        # Gather V tile
        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

        # dP = dO @ V^T : [G_PAD, TILE_N]
        dP = tl.dot(do_block, tl.trans(v_tile))
        # dS = P * (dP - D)
        dS = P * (dP - D_vals[:, None])

        # Accumulate dQ_tilde += dS @ K_rope : [G_PAD, K//2]
        dq_tilde_lo += tl.dot(dS.to(k_rope_lo.dtype), k_rope_lo)
        dq_tilde_hi += tl.dot(dS.to(k_rope_hi.dtype), k_rope_hi)

    # Inverse RoPE on accumulated dQ_tilde, then multiply by scale
    # dq_pre = scale * R^{-1}(dq_tilde)
    # But dq_tilde already has scale baked in via q_rope (which was scaled).
    # Actually: dS @ K_rope gives dQ_tilde in the RoPE'd scaled space.
    # d(loss)/d(q_lo) = d(loss)/d(q_rope_lo) * d(q_rope_lo)/d(q_lo) + d(loss)/d(q_rope_hi) * d(q_rope_hi)/d(q_lo)
    # q_rope_lo = (q_lo * cos - q_hi * sin) * scale
    # q_rope_hi = (q_lo * sin + q_hi * cos) * scale
    # => dq_lo = scale * (dq_tilde_lo * cos + dq_tilde_hi * sin)
    # => dq_hi = scale * (-dq_tilde_lo * sin + dq_tilde_hi * cos)
    dq_lo_result = scale * (dq_tilde_lo * cos_q[None, :] + dq_tilde_hi * sin_q[None, :])
    dq_hi_result = scale * (-dq_tilde_lo * sin_q[None, :] + dq_tilde_hi * cos_q[None, :])

    # Load-add-store to dq (no contention: unique (b,t,h) per program)
    dq_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    dq_ptrs = dq + dq_bt_base + head_ids[:, None] * K
    old_dq_lo = tl.load(dq_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    old_dq_hi = tl.load(dq_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    tl.store(dq_ptrs + o_k_half[None, :], old_dq_lo + dq_lo_result, mask=g_valid[:, None])
    tl.store(dq_ptrs + (K // 2) + o_k_half[None, :], old_dq_hi + dq_hi_result, mask=g_valid[:, None])


# ============================================================
#  K1b-dQ: dQ for upper (non-bottom) layers  (Q-stationary)
# ============================================================

@triton.jit
def htree_bwd_dq_upper_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,    # [B, T, H_kv, TOP_K]  (this layer's parents)
    next_layer_parents,       # [B, T, H_kv, TOP_K]  (child layer selected_parents, for merge mask)
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    dq,                       # [B, T, H, K]     (accumulated)
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    DROP_BLOCK: tl.constexpr,
    G_PAD: tl.constexpr,
    scale,
    SEL_CHUNK: tl.constexpr,
):
    """Grid: (T, B * H_kv). Q-stationary dQ for upper (non-bottom) layers.

    Only *dropped* candidates (not selected by child layer) contribute dQ.
    dKV is handled separately by the K-stationary htree_bwd_dkv_upper_kernel.
    """
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T.to(tl.int64)

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    if num_valid_parents <= 0:
        return

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    next_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)

    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    do_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
    o_v = tl.arange(0, V).to(tl.int64)
    do_block = tl.load(do + do_base + head_ids[:, None] * V + o_v[None, :],
                       mask=g_valid[:, None], other=0.0).to(tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    D_vals = tl.load(delta + state_base + head_ids, mask=g_valid, other=0.0)
    gm = tl.load(global_max + state_base + head_ids, mask=g_valid, other=0.0)
    gs = tl.load(global_sum + state_base + head_ids, mask=g_valid, other=1.0)

    dq_tilde_lo = tl.zeros([G_PAD, K // 2], dtype=tl.float32)
    dq_tilde_hi = tl.zeros([G_PAD, K // 2], dtype=tl.float32)

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, DROP_BLOCK).to(tl.int32)

    num_drop_tiles = (num_candidates + DROP_BLOCK - 1) // DROP_BLOCK
    for d_tile in range(num_drop_tiles):
        flat_pos = (d_tile * DROP_BLOCK).to(tl.int32) + o_n
        pos_valid = flat_pos < num_candidates

        parent_off = (flat_pos // COMPRESSION_RATE).to(tl.int32)
        child_slot = flat_pos - parent_off * COMPRESSION_RATE

        safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
        parent_idx = tl.load(prev_selected_parents + prev_sel_base + safe_poff).to(tl.int32)

        is_rm = parent_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_slot <= rightmost_child_idx)
        node_idx = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_slot
        cand_valid = pos_valid & (parent_off < num_valid_parents) & (parent_idx >= 0) & ch_ok & (node_idx >= 0) & (node_idx < N_layer)

        is_selected = tl.zeros([DROP_BLOCK], dtype=tl.int32)
        for chunk_start in range(0, TOP_K, SEL_CHUNK):
            sel_chunk = tl.load(
                next_layer_parents + next_sel_base + chunk_start + tl.arange(0, SEL_CHUNK)
            ).to(tl.int32)
            matches = (node_idx[:, None] == sel_chunk[None, :]).to(tl.int32)
            is_selected += tl.sum(matches, axis=1)

        is_dropped = (is_selected == 0) & cand_valid
        if tl.sum(is_dropped.to(tl.int32)) > 0:
            safe_node = tl.minimum(tl.maximum(node_idx, 0), N_layer - 1).to(tl.int64)

            k_row = k_base + safe_node[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = tl.maximum(flat_pos.to(tl.int64), 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            scores = tl.where(is_dropped[None, :] & g_valid[:, None], scores, 0.0)
            P = tl.exp(scores - gm[:, None]) / gs[:, None]
            P = tl.where(is_dropped[None, :] & g_valid[:, None], P, 0.0)

            v_row = v_base + safe_node[:, None] * (H_kv * V)
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

            dP = tl.dot(do_block, tl.trans(v_tile))
            dS = P * (dP - D_vals[:, None])

            dq_tilde_lo += tl.dot(dS.to(k_rope_lo.dtype), k_rope_lo)
            dq_tilde_hi += tl.dot(dS.to(k_rope_hi.dtype), k_rope_hi)

    dq_lo_result = scale * (dq_tilde_lo * cos_q[None, :] + dq_tilde_hi * sin_q[None, :])
    dq_hi_result = scale * (-dq_tilde_lo * sin_q[None, :] + dq_tilde_hi * cos_q[None, :])

    dq_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    dq_ptrs = dq + dq_bt_base + head_ids[:, None] * K
    old_dq_lo = tl.load(dq_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    old_dq_hi = tl.load(dq_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    tl.store(dq_ptrs + o_k_half[None, :], old_dq_lo + dq_lo_result, mask=g_valid[:, None])
    tl.store(dq_ptrs + (K // 2) + o_k_half[None, :], old_dq_hi + dq_hi_result, mask=g_valid[:, None])


# ============================================================
#  K1b-dKV: dKV for upper (non-bottom) layers  (K-stationary)
# ============================================================

@triton.jit
def htree_bwd_dkv_upper_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,    # [B, T, H_kv, TOP_K] (this layer's parents – group indices)
    next_layer_parents,       # [B, T, H_kv, TOP_K] (child layer selected_parents – node indices)
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    parent_mask,              # [B, H_kv, num_parent_groups, T] uint8
    dk, dv,                   # [B, N_layer, H_kv, K/V] (output)
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    num_parent_groups: tl.constexpr,
    G_PAD: tl.constexpr,
    SEL_CHUNK: tl.constexpr,
    scale,
):
    """K-stationary dKV for upper (non-bottom) layers.
    Grid: (num_parent_groups, B * H_kv).

    Each program handles one parent group g (CR nodes), iterating over all
    queries that attend to it (using parent_mask).  For each query, only
    DROPPED children (not in next_layer_parents) contribute dKV.
    """
    i_g = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T.to(tl.int64)

    o_cr = tl.arange(0, COMPRESSION_RATE).to(tl.int64)
    node_start = i_g.to(tl.int64) * COMPRESSION_RATE
    node_ids = node_start + o_cr
    node_valid = node_ids < N_layer
    node_ids_i32 = node_ids.to(tl.int32)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    k_group_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_group_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    k_lo = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_group_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
                      mask=node_valid[:, None], other=0.0).to(tl.float32)

    dk_lo_acc = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dk_hi_acc = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dv_acc = tl.zeros([COMPRESSION_RATE, V], dtype=tl.float32)

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    mask_base = (
        i_b.to(tl.int64) * H_kv * num_parent_groups * T_i64
        + i_h_kv.to(tl.int64) * num_parent_groups * T_i64
        + i_g.to(tl.int64) * T_i64
    )

    for i_t in range(T):
        m_val = tl.load(parent_mask + mask_base + i_t.to(tl.int64))
        if m_val != 0:
            rightmost_idx_t = i_t // layer_power
            rightmost_child_idx_t = rightmost_idx_t % COMPRESSION_RATE

            sel_base_t = (
                i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                + i_t.to(tl.int64) * H_kv * TOP_K
                + i_h_kv.to(tl.int64) * TOP_K
            )
            parent_list_t = tl.load(prev_selected_parents + sel_base_t + tl.arange(0, TOP_K))
            num_valid_t = tl.sum((parent_list_t >= 0).to(tl.int32))

            match_mask = (parent_list_t == i_g.to(tl.int32)) & (parent_list_t >= 0)
            match_pos = tl.arange(0, TOP_K).to(tl.int32)
            pos_in_list = tl.min(tl.where(match_mask, match_pos, TOP_K))

            if pos_in_list < TOP_K:
                num_candidates_t = ((num_valid_t - 1) * COMPRESSION_RATE + rightmost_child_idx_t + 1).to(tl.int32)
                rope_pos_q_t = num_candidates_t - 1

                k_flat_base = pos_in_list.to(tl.int64) * COMPRESSION_RATE
                k_flat_pos = k_flat_base + o_cr

                is_rightmost_group = (i_g == (rightmost_idx_t // COMPRESSION_RATE))
                k_valid = node_valid & tl.where(
                    is_rightmost_group,
                    o_cr.to(tl.int32) <= rightmost_child_idx_t,
                    True
                )

                # Dropped check: children NOT in next_layer_parents
                next_sel_base_t = (
                    i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                    + i_t.to(tl.int64) * H_kv * TOP_K
                    + i_h_kv.to(tl.int64) * TOP_K
                )
                is_selected = tl.zeros([COMPRESSION_RATE], dtype=tl.int32)
                for chunk_start in range(0, TOP_K, SEL_CHUNK):
                    sel_chunk = tl.load(
                        next_layer_parents + next_sel_base_t + chunk_start + tl.arange(0, SEL_CHUNK)
                    ).to(tl.int32)
                    matches = (node_ids_i32[:, None] == sel_chunk[None, :]).to(tl.int32)
                    is_selected += tl.sum(matches, axis=1)

                is_dropped = (is_selected == 0) & k_valid
                num_dropped = tl.sum(is_dropped.to(tl.int32))

                if num_dropped > 0:
                    cos_k_t = tl.load(cos_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                      mask=k_valid[:, None], other=0.0)
                    sin_k_t = tl.load(sin_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                      mask=k_valid[:, None], other=0.0)
                    k_rope_lo = k_lo * cos_k_t - k_hi * sin_k_t
                    k_rope_hi = k_lo * sin_k_t + k_hi * cos_k_t

                    q_bt_base_t = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
                    q_ptrs_t = q + q_bt_base_t + head_ids[:, None] * K
                    q_lo_t = tl.load(q_ptrs_t + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
                    q_hi_t = tl.load(q_ptrs_t + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

                    cos_q_t = tl.load(cos_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                    sin_q_t = tl.load(sin_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                    q_rope_lo_t = (q_lo_t * cos_q_t[None, :] - q_hi_t * sin_q_t[None, :]) * scale
                    q_rope_hi_t = (q_lo_t * sin_q_t[None, :] + q_hi_t * cos_q_t[None, :]) * scale

                    S = tl.dot(q_rope_lo_t, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi_t, tl.trans(k_rope_hi))
                    S = tl.where(is_dropped[None, :] & g_valid[:, None], S, 0.0)

                    state_base_t = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
                    gm_t = tl.load(global_max + state_base_t + head_ids, mask=g_valid, other=0.0)
                    gs_t = tl.load(global_sum + state_base_t + head_ids, mask=g_valid, other=1.0)
                    P = tl.exp(S - gm_t[:, None]) / gs_t[:, None]
                    P = tl.where(is_dropped[None, :] & g_valid[:, None], P, 0.0)

                    do_base_t = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
                    do_t = tl.load(do + do_base_t + head_ids[:, None] * V + o_v_dim[None, :],
                                   mask=g_valid[:, None], other=0.0).to(tl.float32)
                    D_t = tl.load(delta + state_base_t + head_ids, mask=g_valid, other=0.0)

                    dP = tl.dot(do_t, tl.trans(v_block))
                    dS = P * (dP - D_t[:, None])

                    dk_tilde_lo = tl.dot(tl.trans(dS).to(q_rope_lo_t.dtype), q_rope_lo_t)
                    dk_tilde_hi = tl.dot(tl.trans(dS).to(q_rope_hi_t.dtype), q_rope_hi_t)
                    dk_lo_acc += dk_tilde_lo * cos_k_t + dk_tilde_hi * sin_k_t
                    dk_hi_acc += -dk_tilde_lo * sin_k_t + dk_tilde_hi * cos_k_t

                    dv_acc += tl.dot(tl.trans(P).to(do_t.dtype), do_t)

    dk_out_base = dk + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    dv_out_base = dv + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
             dk_lo_acc, mask=node_valid[:, None])
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
             dk_hi_acc, mask=node_valid[:, None])
    tl.store(dv_out_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
             dv_acc, mask=node_valid[:, None])


# ============================================================
#  K3: dKV bottom  (K-stationary, no atomics)
# ============================================================

@triton.jit
def htree_bwd_dkv_bottom_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    selected_parents,         # [B, T, H_kv, TOP_K] (bottom layer)
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    parent_mask,              # [B, H_kv, num_groups, T] uint8
    dk, dv,                   # [B, N_layer, H_kv, K/V] (output)
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    num_parent_groups: tl.constexpr,
    G_PAD: tl.constexpr,
    scale,
):
    """K-stationary dKV for bottom layer. Grid: (num_parent_groups, B * H_kv).
    Each program handles one parent group g (CR nodes), iterating over all
    queries that attend to it (using parent_mask). Flat per-query loop.
    """
    i_g = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T.to(tl.int64)

    # Load K/V for this parent group (CR nodes)
    o_cr = tl.arange(0, COMPRESSION_RATE).to(tl.int64)
    node_start = i_g.to(tl.int64) * COMPRESSION_RATE
    node_ids = node_start + o_cr
    node_valid = node_ids < N_layer

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    k_group_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_group_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    k_lo = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_group_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
                      mask=node_valid[:, None], other=0.0).to(tl.float32)

    # dK/dV accumulators
    dk_lo_acc = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dk_hi_acc = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dv_acc = tl.zeros([COMPRESSION_RATE, V], dtype=tl.float32)

    # GQA head setup
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    # Parent mask base for this (b, kv_h, g)
    mask_base = (
        i_b.to(tl.int64) * H_kv * num_parent_groups * T_i64
        + i_h_kv.to(tl.int64) * num_parent_groups * T_i64
        + i_g.to(tl.int64) * T_i64
    )

    # Flat per-query iteration (no `continue` — use nested `if` for Triton compat)
    for i_t in range(T):
        m_val = tl.load(parent_mask + mask_base + i_t.to(tl.int64))
        if m_val != 0:
            rightmost_idx_t = i_t // layer_power
            rightmost_child_idx_t = rightmost_idx_t % COMPRESSION_RATE

            sel_base_t = (
                i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                + i_t.to(tl.int64) * H_kv * TOP_K
                + i_h_kv.to(tl.int64) * TOP_K
            )
            parent_list_t = tl.load(selected_parents + sel_base_t + tl.arange(0, TOP_K))
            num_valid_t = tl.sum((parent_list_t >= 0).to(tl.int32))

            match_mask = (parent_list_t == i_g.to(tl.int32)) & (parent_list_t >= 0)
            match_pos = tl.arange(0, TOP_K).to(tl.int32)
            pos_in_list = tl.min(tl.where(match_mask, match_pos, TOP_K))

            if pos_in_list < TOP_K:
                num_candidates_t = ((num_valid_t - 1) * COMPRESSION_RATE + rightmost_child_idx_t + 1).to(tl.int32)
                rope_pos_q_t = num_candidates_t - 1

                k_flat_base = pos_in_list.to(tl.int64) * COMPRESSION_RATE
                k_flat_pos = k_flat_base + o_cr

                is_rightmost_group = (i_g == (rightmost_idx_t // COMPRESSION_RATE))
                k_valid = node_valid & tl.where(
                    is_rightmost_group,
                    o_cr.to(tl.int32) <= rightmost_child_idx_t,
                    True
                )

                cos_k_t = tl.load(cos_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                  mask=k_valid[:, None], other=0.0)
                sin_k_t = tl.load(sin_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                  mask=k_valid[:, None], other=0.0)
                k_rope_lo = k_lo * cos_k_t - k_hi * sin_k_t
                k_rope_hi = k_lo * sin_k_t + k_hi * cos_k_t

                q_bt_base_t = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
                q_ptrs_t = q + q_bt_base_t + head_ids[:, None] * K
                q_lo_t = tl.load(q_ptrs_t + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
                q_hi_t = tl.load(q_ptrs_t + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

                cos_q_t = tl.load(cos_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                sin_q_t = tl.load(sin_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                q_rope_lo_t = (q_lo_t * cos_q_t[None, :] - q_hi_t * sin_q_t[None, :]) * scale
                q_rope_hi_t = (q_lo_t * sin_q_t[None, :] + q_hi_t * cos_q_t[None, :]) * scale

                S = tl.dot(q_rope_lo_t, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi_t, tl.trans(k_rope_hi))
                S = tl.where(k_valid[None, :] & g_valid[:, None], S, 0.0)

                state_base_t = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
                gm_t = tl.load(global_max + state_base_t + head_ids, mask=g_valid, other=0.0)
                gs_t = tl.load(global_sum + state_base_t + head_ids, mask=g_valid, other=1.0)
                P = tl.exp(S - gm_t[:, None]) / gs_t[:, None]
                P = tl.where(k_valid[None, :] & g_valid[:, None], P, 0.0)

                do_base_t = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
                do_t = tl.load(do + do_base_t + head_ids[:, None] * V + o_v_dim[None, :],
                               mask=g_valid[:, None], other=0.0).to(tl.float32)
                D_t = tl.load(delta + state_base_t + head_ids, mask=g_valid, other=0.0)

                dP = tl.dot(do_t, tl.trans(v_block))
                dS = P * (dP - D_t[:, None])

                dk_tilde_lo = tl.dot(tl.trans(dS).to(q_rope_lo_t.dtype), q_rope_lo_t)
                dk_tilde_hi = tl.dot(tl.trans(dS).to(q_rope_hi_t.dtype), q_rope_hi_t)
                dk_lo_acc += dk_tilde_lo * cos_k_t + dk_tilde_hi * sin_k_t
                dk_hi_acc += -dk_tilde_lo * sin_k_t + dk_tilde_hi * cos_k_t

                dv_acc += tl.dot(tl.trans(P).to(do_t.dtype), do_t)

    # Store dK, dV
    dk_out_base = dk + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    dv_out_base = dv + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
             dk_lo_acc, mask=node_valid[:, None])
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
             dk_hi_acc, mask=node_valid[:, None])
    tl.store(dv_out_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
             dv_acc, mask=node_valid[:, None])


# ============================================================
#  K4: Tree Backward  (mean-pool gradient propagation)
# ============================================================

@triton.jit
def htree_bwd_tree_backward_kernel(
    dk_parent,   # [B, N_parent, H_kv, K]
    dv_parent,   # [B, N_parent, H_kv, V]
    dk_child,    # [B, N_child, H_kv, K]  (accumulated)
    dv_child,    # [B, N_child, H_kv, V]  (accumulated)
    N_parent: tl.constexpr,
    N_child: tl.constexpr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Propagate dK/dV from parent layer down to child layer through mean-pooling.
    Grid: (ceil(N_parent / BLOCK_P), B * H_kv).
    """
    i_block = tl.program_id(0)
    pid = tl.program_id(1)
    B_Hkv = tl.num_programs(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    p_start = i_block * BLOCK_P
    p_ids = p_start + tl.arange(0, BLOCK_P)
    o_k = tl.arange(0, K).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    for p_local in range(BLOCK_P):
        p_idx = p_start + p_local
        if p_idx < N_parent:
            child_start = p_idx * COMPRESSION_RATE
            child_end = tl.minimum(child_start + COMPRESSION_RATE, N_child)
            count = child_end - child_start
            if count > 0:
                pk_base = dk_parent + (i_b.to(tl.int64) * N_parent * H_kv * K
                                       + p_idx * H_kv * K + i_h_kv.to(tl.int64) * K)
                pv_base = dv_parent + (i_b.to(tl.int64) * N_parent * H_kv * V
                                       + p_idx * H_kv * V + i_h_kv.to(tl.int64) * V)

                dk_p = tl.load(pk_base + o_k).to(tl.float32) / count.to(tl.float32)
                dv_p = tl.load(pv_base + o_v_dim).to(tl.float32) / count.to(tl.float32)

                for c_off in range(COMPRESSION_RATE):
                    c_idx = child_start + c_off
                    if c_idx < N_child:
                        ck_base = dk_child + (i_b.to(tl.int64) * N_child * H_kv * K
                                              + c_idx * H_kv * K + i_h_kv.to(tl.int64) * K)
                        cv_base = dv_child + (i_b.to(tl.int64) * N_child * H_kv * V
                                              + c_idx * H_kv * V + i_h_kv.to(tl.int64) * V)

                        old_dk = tl.load(ck_base + o_k).to(tl.float32)
                        old_dv = tl.load(cv_base + o_v_dim).to(tl.float32)
                        tl.store(ck_base + o_k, old_dk + dk_p)
                        tl.store(cv_base + o_v_dim, old_dv + dv_p)


# ============================================================
#  Recompute helpers (gradient checkpointing for parents)
# ============================================================

def _recompute_tree_build(k, v, num_layers, compression_rate):
    """Rebuild upper-layer tree K/V from bottom-layer K/V via mean-pooling."""
    B = k.shape[0]
    T = k.shape[1]
    H_kv = k.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    device, dtype = k.device, k.dtype

    layers_k, layers_v = [k], [v]
    current_k, current_v, current_len = k, v, T

    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H_kv, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H_kv, V, dtype=dtype, device=device)

        BLOCK_SIZE = 8
        grid = (triton.cdiv(next_len, BLOCK_SIZE), B * H_kv)
        htree_build_kernel[grid](
            current_k, current_v, next_k, next_v,
            N_child=current_len, N_parent=next_len,
            B=B, H_kv=H_kv, K=K, V=V,
            COMPRESSION_RATE=compression_rate, BLOCK_SIZE=BLOCK_SIZE,
        )
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len

    return layers_k, layers_v


def _recompute_parents(q, layers_k, cos_cache, sin_cache,
                       num_layers, compression_rate, top_k_per_layer,
                       max_top_nodes, scale, NUM_GROUPS, G_PAD, TILE_P):
    """Replay forward selection loop to reconstruct per_layer_parents."""
    B, T, H, K = q.shape
    H_kv = layers_k[0].shape[2]
    device = q.device

    MAX_CANDIDATES = max_top_nodes
    LOG_N = int(math.log2(MAX_CANDIDATES))
    N_DIMS_TOPK = int(math.log2(top_k_per_layer))

    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(T, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_cands = torch.arange(top_k_per_layer, dtype=torch.int32, device=device)
    parent_cands = parent_cands.unsqueeze(0).expand(T, -1)
    valid_mask = parent_cands < num_virtual_parents.unsqueeze(1)
    prev_sp = torch.where(valid_mask, parent_cands,
                          torch.tensor(-1, dtype=torch.int32, device=device))
    prev_sp = prev_sp.unsqueeze(0).unsqueeze(2).expand(
        B, T, H_kv, top_k_per_layer).contiguous()
    next_sp = torch.empty_like(prev_sp)

    per_layer_parents = {}
    for layer_idx in range(num_layers - 1, -1, -1):
        per_layer_parents[layer_idx] = prev_sp

        if layer_idx > 0:
            k_layer = layers_k[layer_idx]
            N_layer = k_layer.shape[1]
            layer_power = compression_rate ** layer_idx
            grid_kv = (T, B * H_kv)

            # Allocate a fresh output buffer each iteration so that the tensor
            # already stored in per_layer_parents[layer_idx] (= current prev_sp)
            # is never overwritten by a later selection kernel.  Reusing next_sp
            # via a swap would alias per_layer_parents[num_layers-1] with the
            # buffer written in the next pass, corrupting the stored parents when
            # num_layers >= 3 (i.e. T > max_top_nodes * compression_rate).
            next_sp = torch.empty_like(prev_sp)
            htree_recompute_selection_kernel[grid_kv](
                q, k_layer, prev_sp,
                cos_cache, sin_cache,
                next_sp,
                layer_power=layer_power,
                B=B, T=T, H=H, H_kv=H_kv,
                NUM_GROUPS=NUM_GROUPS, K=K,
                N_layer=N_layer,
                COMPRESSION_RATE=compression_rate,
                TOP_K=top_k_per_layer,
                LOG_N=LOG_N, N_DIMS_TOPK=N_DIMS_TOPK,
                SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
                NEG_INF=HTREE_SCORE_NEG_INF,
                G_PAD=G_PAD, TILE_P=TILE_P,
                scale=scale,
            )
            prev_sp = next_sp

    return per_layer_parents


# ============================================================
#  Orchestration: htree_backward
# ============================================================

def htree_backward(dO, q, output, bwd_ctx):
    """Dispatch all backward kernels and return (dq, dk, dv)."""
    B, T, H, K = q.shape
    V = dO.shape[-1]
    device = q.device
    dtype = torch.float32

    global_max = bwd_ctx['global_max']
    global_sum = bwd_ctx['global_sum']
    per_layer_parents = bwd_ctx['per_layer_parents']
    layers_k = bwd_ctx['layers_k']
    layers_v = bwd_ctx['layers_v']
    cos_cache = bwd_ctx['cos_cache']
    sin_cache = bwd_ctx['sin_cache']
    num_layers = bwd_ctx['num_layers']
    compression_rate = bwd_ctx['compression_rate']
    top_k_per_layer = bwd_ctx['top_k_per_layer']
    scale = bwd_ctx['scale']
    NUM_GROUPS = bwd_ctx['NUM_GROUPS']
    G_PAD = bwd_ctx['G_PAD']
    TILE_P = bwd_ctx['TILE_P']
    DROP_BLOCK = bwd_ctx['DROP_BLOCK']

    H_kv = layers_k[0].shape[2]

    dO = dO.contiguous().to(dtype)
    output_f32 = output.contiguous().to(dtype)

    logger.info("htree backward pass started")
    # K0: preprocess delta
    delta = torch.empty(B, T, H, dtype=dtype, device=device)
    grid_k0 = (T, B * H)
    with _cuda_timer("  Kernel K0 (preprocess delta)"):
        htree_bwd_preprocess_delta_kernel[grid_k0](
            output_f32, dO, delta, T=T, H=H, V=V,
        )

    # Allocate gradient buffers
    dq = torch.zeros(B, T, H, K, dtype=dtype, device=device)
    dk_layers = [torch.zeros_like(lk, dtype=dtype) for lk in layers_k]
    dv_layers = [torch.zeros_like(lv, dtype=dtype) for lv in layers_v]

    # Process layers top → bottom (same order as forward)
    for layer_idx in range(num_layers - 1, -1, -1):
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        is_bottom = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        psp = per_layer_parents[layer_idx]

        grid_kv = (T, B * H_kv)
        logger.info("  -> Processing layer %d (N=%d, bottom=%s)...", layer_idx, N_layer, is_bottom)

        if is_bottom:
            # K1-bottom: dQ only (dKV handled by K3)
            with _cuda_timer("    Kernel K1a (dQ bottom)"):
                htree_bwd_dq_bottom_kernel[grid_kv](
                    q, k_layer, v_layer,
                    psp, cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    dq,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    G_PAD=G_PAD, TILE_P=TILE_P,
                    scale=scale,
                )

            # K2: build parent mask
            num_parent_groups = (N_layer + compression_rate - 1) // compression_rate
            parent_mask = torch.zeros(B, H_kv, num_parent_groups, T, dtype=torch.uint8, device=device)
            grid_k2 = (T, B * H_kv)
            BLOCK_TK = min(top_k_per_layer, 64)
            with _cuda_timer("    Kernel K2 (build parent mask)"):
                htree_bwd_build_parent_mask_kernel[grid_k2](
                    psp, parent_mask,
                    T=T, H_kv=H_kv, TOP_K=top_k_per_layer,
                    num_groups=num_parent_groups, BLOCK_TK=BLOCK_TK,
                )

            # K3: K-stationary dKV for bottom
            grid_k3 = (num_parent_groups, B * H_kv)
            with _cuda_timer("    Kernel K3 (dKV bottom)"):
                htree_bwd_dkv_bottom_kernel[grid_k3](
                    q, k_layer, v_layer,
                    psp, cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    parent_mask,
                    dk_layers[0], dv_layers[0],
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    num_parent_groups=num_parent_groups,
                    G_PAD=G_PAD, scale=scale,
                )
        else:
            next_layer_parents = per_layer_parents[layer_idx - 1]
            SEL_CHUNK = min(top_k_per_layer, 64)

            # K1b-dQ: Q-stationary dQ for upper layer
            with _cuda_timer("    Kernel K1b-dQ (dQ upper)"):
                htree_bwd_dq_upper_kernel[grid_kv](
                    q, k_layer, v_layer,
                    psp, next_layer_parents,
                    cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    dq,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    DROP_BLOCK=DROP_BLOCK,
                    G_PAD=G_PAD,
                    scale=scale,
                    SEL_CHUNK=SEL_CHUNK,
                )

            # K2-upper: build parent mask (reuse bottom-layer K2 kernel)
            num_upper_parent_groups = (N_layer + compression_rate - 1) // compression_rate
            upper_parent_mask = torch.zeros(B, H_kv, num_upper_parent_groups, T,
                                            dtype=torch.uint8, device=device)
            BLOCK_TK = min(top_k_per_layer, 64)
            with _cuda_timer("    Kernel K2u (build upper parent mask)"):
                htree_bwd_build_parent_mask_kernel[grid_kv](
                    psp, upper_parent_mask,
                    T=T, H_kv=H_kv, TOP_K=top_k_per_layer,
                    num_groups=num_upper_parent_groups, BLOCK_TK=BLOCK_TK,
                )

            # K3-upper: K-stationary dKV for upper layer (no atomics)
            grid_upper_kv = (num_upper_parent_groups, B * H_kv)
            with _cuda_timer("    Kernel K3u (dKV upper)"):
                htree_bwd_dkv_upper_kernel[grid_upper_kv](
                    q, k_layer, v_layer,
                    psp, next_layer_parents,
                    cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    upper_parent_mask,
                    dk_layers[layer_idx], dv_layers[layer_idx],
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    num_parent_groups=num_upper_parent_groups,
                    G_PAD=G_PAD,
                    SEL_CHUNK=SEL_CHUNK,
                    scale=scale,
                )

    # K4: tree backward – propagate gradients down through mean-pooling
    BLOCK_P = 8
    for layer_idx in range(num_layers - 1, 0, -1):
        N_parent = layers_k[layer_idx].shape[1]
        N_child = layers_k[layer_idx - 1].shape[1]
        num_blocks = (N_parent + BLOCK_P - 1) // BLOCK_P
        grid_k4 = (num_blocks, B * H_kv)
        with _cuda_timer(f"  Kernel K4 (tree backward, {N_parent}->{N_child})"):
            htree_bwd_tree_backward_kernel[grid_k4](
                dk_layers[layer_idx], dv_layers[layer_idx],
                dk_layers[layer_idx - 1], dv_layers[layer_idx - 1],
                N_parent=N_parent, N_child=N_child,
                H_kv=H_kv, K=K, V=V,
                COMPRESSION_RATE=compression_rate,
                BLOCK_P=BLOCK_P,
            )

    logger.info("htree backward pass completed!")
    return dq, dk_layers[0], dv_layers[0]


# ============================================================
#  Autograd Function
# ============================================================

class HTreeTritonFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base):
        output, bwd_ctx = htree_forward(
            q, k, v,
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale,
            rope_base=rope_base,
            _save_for_backward=True,
            _save_parents=False,
        )

        ctx.save_for_backward(
            q, k, v, output,
            bwd_ctx['global_max'], bwd_ctx['global_sum'],
            bwd_ctx['cos_cache'], bwd_ctx['sin_cache'],
        )

        ctx.num_layers = bwd_ctx['num_layers']
        ctx.compression_rate = compression_rate
        ctx.top_k_per_layer = top_k_per_layer
        ctx.max_top_nodes = max_top_nodes
        ctx.scale = scale
        ctx.rope_base = rope_base
        ctx.NUM_GROUPS = bwd_ctx['NUM_GROUPS']
        ctx.G_PAD = bwd_ctx['G_PAD']
        ctx.TILE_P = bwd_ctx['TILE_P']
        ctx.DROP_BLOCK = bwd_ctx['DROP_BLOCK']
        return output

    @staticmethod
    def backward(ctx, dO):
        q, k, v, output, global_max, global_sum, cos_cache, sin_cache = ctx.saved_tensors

        with _cuda_timer("  Recompute tree build"):
            layers_k, layers_v = _recompute_tree_build(
                k, v, ctx.num_layers, ctx.compression_rate,
            )

        with _cuda_timer("  Recompute parents (selection-only)"):
            per_layer_parents = _recompute_parents(
                q, layers_k, cos_cache, sin_cache,
                ctx.num_layers, ctx.compression_rate, ctx.top_k_per_layer,
                ctx.max_top_nodes, ctx.scale,
                ctx.NUM_GROUPS, ctx.G_PAD, ctx.TILE_P,
            )

        bwd_ctx = dict(
            global_max=global_max, global_sum=global_sum,
            per_layer_parents=per_layer_parents,
            layers_k=layers_k, layers_v=layers_v,
            cos_cache=cos_cache, sin_cache=sin_cache,
            num_layers=ctx.num_layers,
            compression_rate=ctx.compression_rate,
            top_k_per_layer=ctx.top_k_per_layer,
            max_top_nodes=ctx.max_top_nodes,
            scale=ctx.scale,
            NUM_GROUPS=ctx.NUM_GROUPS,
            G_PAD=ctx.G_PAD,
            TILE_P=ctx.TILE_P,
            DROP_BLOCK=ctx.DROP_BLOCK,
        )

        dq, dk, dv = htree_backward(dO, q, output, bwd_ctx)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None


def htree_forward_triton_autograd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    compression_rate: int = 16, max_top_nodes: int = 8192,
    top_k_per_layer: int = 512, scale: Optional[float] = None,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Differentiable HTree forward using Triton kernels for both fwd and bwd."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return HTreeTritonFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base,
    )


__all__ = [
    'htree_backward',
    'htree_forward_triton_autograd',
    'HTreeTritonFunction',
]
