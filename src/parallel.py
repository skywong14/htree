# -*- coding: utf-8 -*-
"""
HTree Triton Kernels

Pipeline
  Phase 1: Tree Building
        - Kernel: `htree_build_kernel`
            - Bottom-up mean-pooling of (K, V) to build `(layers_k, layers_v)`.

  Phase 1.5: RoPE Cache
    - Host-side precomputation of `cos_cache / sin_cache` (shared by all layers).

  Phase 2: Init Global State & Workspaces
    - Zero-init online-softmax accumulators: `global_max / global_sum / global_output`.
    - Allocate per-layer reusable workspaces:
      - `prev_selected_parents`: [B, T, H_kv, TOP_K] -- parent list for next-layer expansion (ascending).
      - `next_selected_parents`: [B, T, H_kv, TOP_K] -- current-layer shared Top-K node indices (ascending, -1 padded).

  Phase 3: Layer-by-layer Forward (top -> bottom)
        - Kernel 1a `htree_bottom_accumulate_gqa_kernel`
      - Bottom layer: skip Top-K, accumulate all candidates via online-softmax.
      - Outputs `(layer_max, layer_sum, layer_output)`.

        - Kernel 1b `htree_select_accumulate_gqa_kernel`
      - Non-bottom layers: streaming Top-K via group-shared importance
        (merge 2*TOP_K with bitonic sort each iteration), immediately
        accumulating dropped candidates via online-softmax (avoids storing
        all scores and re-reading V).
      - Outputs `(layer_max, layer_sum, layer_output)` and `next_selected_parents`.

        - Kernel 2 `htree_merge_to_global_kernel`
      - Online-softmax merge: fold `(layer_max, layer_sum, layer_output)` into global state.

  Phase 4: Final Normalize
        - Kernel 3 `htree_final_normalize_kernel`
      - `output = global_output / global_sum`.
"""

import logging
import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@contextmanager
def _cuda_timer(label: str):
    """Context manager that times enclosed CUDA ops and logs the result."""
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    yield
    t1.record()
    torch.cuda.synchronize()
    logger.info("%s: %.2f ms", label, t0.elapsed_time(t1))

# ========================================
#  Global Constants
# ========================================

# Unified sentinel score for masked / invalid positions.
# All masked entries must have score <= HTREE_SCORE_NEG_INF.
HTREE_SCORE_NEG_INF: float = -1.0e10

# Threshold for distinguishing valid scores from masked ones.
# Must be strictly greater than HTREE_SCORE_NEG_INF so that
# `score > THRESHOLD` filters out masked entries (-1e10).
# Empirical value from the original implementation: -0.9e10.
HTREE_SCORE_VALID_THRESHOLD: float = HTREE_SCORE_NEG_INF * 0.9

# ========================================
#  Kernel (build): Tree Building
# ========================================

@triton.jit
def htree_build_kernel(
    child_k,  # [B, N_child, H_kv, K]
    child_v,  # [B, N_child, H_kv, V]
    parent_k,  # [B, N_parent, H_kv, K]
    parent_v,  # [B, N_parent, H_kv, V]
    N_child: tl.constexpr,
    N_parent: tl.constexpr,
    B: tl.constexpr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tree building kernel: mean pooling from child nodes to parent nodes
    Grid: (N_parent_blocks, B * H_kv)
    """
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    i_b = pid_bh // H_kv
    i_h = pid_bh % H_kv
    
    parent_start = pid_seq * BLOCK_SIZE
    # Early check, although grid should be tight
    if parent_start >= N_parent:
        return

    child_start = parent_start * COMPRESSION_RATE
    
    # load K
    k_base = child_k + i_b.to(tl.int64) * N_child * H_kv * K + i_h.to(tl.int64) * K
    k_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_child, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, 0),
        block_shape=(BLOCK_SIZE * COMPRESSION_RATE, K),
        order=(1, 0)
    )
    k_vals = tl.load(k_block_ptrs, boundary_check=(0, 1))
    
    # load V
    v_base = child_v + i_b.to(tl.int64) * N_child * H_kv * V + i_h.to(tl.int64) * V
    v_block_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(N_child, V),
        strides=(H_kv * V, 1),
        offsets=(child_start, 0),
        block_shape=(BLOCK_SIZE * COMPRESSION_RATE, V),
        order=(1, 0)
    )
    v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))
    
    # Reshape + Mean Pooling
    parent_global_idx = parent_start + tl.arange(0, BLOCK_SIZE)
    child_global_idx = parent_global_idx[:, None] * COMPRESSION_RATE + tl.arange(0, COMPRESSION_RATE)[None, :]
    child_valid = (parent_global_idx[:, None] < N_parent) & (child_global_idx < N_child)
    
    k_vals_reshaped = tl.reshape(k_vals, [BLOCK_SIZE, COMPRESSION_RATE, K])
    k_masked = tl.where(child_valid[:, :, None], k_vals_reshaped, 0.0)
    k_sum = tl.sum(k_masked, axis=1)
    num_valid_children = tl.sum(child_valid.to(tl.int32), axis=1)
    num_valid_children_safe = tl.maximum(num_valid_children, 1)
    k_mean = k_sum / num_valid_children_safe[:, None]
    
    v_vals_reshaped = tl.reshape(v_vals, [BLOCK_SIZE, COMPRESSION_RATE, V])
    v_masked = tl.where(child_valid[:, :, None], v_vals_reshaped, 0.0)
    v_sum = tl.sum(v_masked, axis=1)
    v_mean = v_sum / num_valid_children_safe[:, None]
    
    # store
    parent_k_base = parent_k + i_b.to(tl.int64) * N_parent * H_kv * K + i_h.to(tl.int64) * K
    parent_k_block_ptrs = tl.make_block_ptr(
        base=parent_k_base,
        shape=(N_parent, K),
        strides=(H_kv * K, 1),
        offsets=(parent_start, 0),
        block_shape=(BLOCK_SIZE, K),
        order=(1, 0)
    )
    tl.store(parent_k_block_ptrs, k_mean.to(parent_k.dtype.element_ty), boundary_check=(0, 1))
    
    parent_v_base = parent_v + i_b.to(tl.int64) * N_parent * H_kv * V + i_h.to(tl.int64) * V
    parent_v_block_ptrs = tl.make_block_ptr(
        base=parent_v_base,
        shape=(N_parent, V),
        strides=(H_kv * V, 1),
        offsets=(parent_start, 0),
        block_shape=(BLOCK_SIZE, V),
        order=(1, 0)
    )
    tl.store(parent_v_block_ptrs, v_mean.to(parent_v.dtype.element_ty), boundary_check=(0, 1))


# ========================================
#  Helper Functions
# ========================================

@triton.jit
def load_k_with_rope(
    layer_k,
    cos_cache,
    sin_cache,
    i_b,
    i_h_kv,
    child_start,
    rope_position_start,
    N_layer: tl.constexpr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    num_valid_children,
):
    """Load COMPRESSION_RATE tokens of K and apply RoPE."""
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    
    # First half of head dim
    k_lo_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k_lo = tl.load(k_lo_ptrs, boundary_check=(0, 1))
    
    # Second half of head dim
    k_hi_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, K // 2),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k_hi = tl.load(k_hi_ptrs, boundary_check=(0, 1))
    
    # Apply RoPE rotation
    rope_positions = rope_position_start + tl.arange(0, COMPRESSION_RATE)
    o_k_half = tl.arange(0, K // 2)
    
    valid_rows = tl.arange(0, COMPRESSION_RATE)[:, None] < num_valid_children
    cos_ptrs = cos_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    sin_ptrs = sin_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    cos_k = tl.load(cos_ptrs, mask=valid_rows, other=0.0)
    sin_k = tl.load(sin_ptrs, mask=valid_rows, other=0.0)
    
    k_rope_lo = k_lo * cos_k - k_hi * sin_k
    k_rope_hi = k_lo * sin_k + k_hi * cos_k
    
    return k_rope_lo, k_rope_hi


# ========================================
#  Kernel 2: Merge to Global  /  Kernel 3: Final Normalize
# ========================================

@triton.jit
def htree_merge_to_global_kernel(
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    global_max,  # [B, T, H]
    global_sum,  # [B, T, H]
    global_output,  # [B, T, H, V]
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """Kernel 2: merge layer state into global online-softmax accumulators. Grid: (T, B*H)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    T_i64 = T.to(tl.int64)
    state_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    
    cur_max = tl.load(layer_max + state_offset)
    cur_sum = tl.load(layer_sum + state_offset)
    
    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    cur_output = tl.load(layer_output + output_offset + o_v, mask=o_v < V, other=0.0)
    
    g_max = tl.load(global_max + state_offset)
    g_sum = tl.load(global_sum + state_offset)
    g_output_vals = tl.load(global_output + output_offset + o_v, mask=o_v < V, other=0.0)
    
    cur_has_contribution = cur_sum > 1e-10
    g_has_accumulation = g_sum > 1e-10
    
    new_max = tl.maximum(g_max, cur_max)
    scale_g = tl.where(g_has_accumulation, tl.exp(g_max - new_max), 0.0)
    scale_c = tl.where(cur_has_contribution, tl.exp(cur_max - new_max), 0.0)
    
    merged_sum = g_sum * scale_g + cur_sum * scale_c
    merged_output = g_output_vals * scale_g + cur_output * scale_c
    merged_max = new_max
    
    g_max = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_max, cur_max),
        g_max
    )
    g_sum = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_sum, cur_sum),
        g_sum
    )
    g_output_vals = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_output, cur_output),
        g_output_vals
    )
    
    tl.store(global_max + state_offset, g_max)
    tl.store(global_sum + state_offset, g_sum)
    tl.store(global_output + output_offset + o_v, g_output_vals, mask=o_v < V)


@triton.jit
def htree_final_normalize_kernel(
    global_output,  # [B, T, H, V]
    global_sum,  # [B, T, H]
    output,  # [B, T, H, V]
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """
    htree_final_normalize_kernel: output = global_output / global_sum
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # load sum
    T_i64 = T.to(tl.int64)
    sum_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    sum_val = tl.load(global_sum + sum_offset)
    
    # load and normalize output
    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    output_ptrs = global_output + output_offset + o_v
    output_vals = tl.load(output_ptrs, mask=o_v < V, other=0.0)
    
    normalized = output_vals / sum_val
    
    # write back
    out_ptrs = output + output_offset + o_v
    tl.store(out_ptrs, normalized, mask=o_v < V)

# ========================================
#  Kernel 1a: Bottom Accumulate (GQA)
# ========================================

@triton.jit
def htree_bottom_accumulate_gqa_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # outputs
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    # params / constexpr
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
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
):
    """Bottom layer forward: batch-gather K/V tiles + tl.dot (tensor core).

    Processes TILE_P parents (= TILE_P * CR tokens) per iteration instead of
    one parent at a time, using tl.dot for QK^T and PV accumulation.

    Grid: (T, B*H_kv). One program per (b, t, h_kv).
    """
    i_t = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

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

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # ---- Load Q for G_PAD heads (zero-pad when G_PAD > NUM_GROUPS) ----
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale  # [G_PAD, K//2]
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale  # [G_PAD, K//2]

    # ---- Online-softmax state (G_PAD-wide) ----
    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    # ---- Precompute base pointers & offsets ----
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE   # [TILE_N] which parent (0..TILE_P-1)
    child_in_tile = o_n % COMPRESSION_RATE     # [TILE_N] which child  (0..CR-1)
    o_v = tl.arange(0, V).to(tl.int64)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V

    # ---- Main loop: TILE_P parents per iteration (flat TILE_N indexing) ----
    #  All loads are UNCONDITIONAL (indices clamped to valid ranges) to avoid
    #  Triton layout conflicts between 1-D masked loads and 2-D tl.dot results.
    #  Validity is enforced purely by masking scores to NEG_INF / probs to 0.
    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    for tile_idx in range(num_tiles):
        tile_base = (tile_idx * TILE_P).to(tl.int32)

        # Global parent offset for each of the TILE_N positions
        p_offs = tile_base + parent_in_tile                               # [TILE_N]

        # Unconditional load — clamp index to [0, TOP_K-1]
        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(
            prev_selected_parents + prev_sel_base + safe_p_offs
        ).to(tl.int32)                                                    # [TILE_N]

        # Compute validity mask (pure 1-D, no layout interaction with tl.dot)
        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)

        # Clamp row indices for unconditional K / V loads
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)

        # ---- Gather K (two halves, unconditional) ----
        k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)           # [TILE_N, K//2]
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        # ---- RoPE on K (flat position = tile_base * CR + o_n, contiguous) ----
        rope_pos = tile_base.to(tl.int64) * COMPRESSION_RATE + o_n.to(tl.int64)
        rope_pos = tl.maximum(rope_pos, 0)
        cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
        sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
        k_rope_lo = k_lo * cos_k - k_hi * sin_k                                          # [TILE_N, K//2]
        k_rope_hi = k_lo * sin_k + k_hi * cos_k                                          # [TILE_N, K//2]

        # ---- Scores via split tl.dot ----
        # [G_PAD, K//2] × [K//2, TILE_N]  +  same  →  [G_PAD, TILE_N]
        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG_INF)

        # ---- Gather V (unconditional) ----
        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)                # [TILE_N, V]

        # ---- Online softmax ----
        tile_m = tl.max(scores, axis=1)                                   # [G_PAD]
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

        # ---- PV via tl.dot  [G_PAD, TILE_N] × [TILE_N, V] → [G_PAD, V] ----
        cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    # ---- Store (first NUM_GROUPS heads only) ----
    tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, cur_output, mask=g_valid[:, None])


# ========================================
#  Kernel 1b: Non-Bottom Select + Accumulate (GQA)
# ========================================

@triton.jit
def htree_select_accumulate_gqa_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # outputs
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    next_selected_parents,  # [B, T, H_kv, TOP_K]  (sorted asc, -1 padded)
    # params / constexpr
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
    LOG_N: tl.constexpr,
    N_DIMS_TOPK: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    DROP_BLOCK: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
):
    """Non-bottom layer forward: streaming Top-K + accumulate dropped.

    Uses tl.dot (tensor core) for QK^T and PV instead of element-wise broadcast.

    Grid: (T, B*H_kv). One program per (b, t, h_kv).
    """
    i_t = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

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

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # ---- Load Q for G_PAD heads (zero-pad when G_PAD > NUM_GROUPS) ----
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale  # [G_PAD, K//2]
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale  # [G_PAD, K//2]

    # ---- Online-softmax state (G_PAD-wide) ----
    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V

    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    idx_mask = (1 << LOG_N) - 1
    rightmost_pos = (num_candidates - 1).to(tl.int32)
    topk_running_max = tl.full([1], NEG_INF, dtype=tl.float32)

    init_idx = tl.arange(0, TOP_K).to(tl.int32)
    init_imp = tl.full([TOP_K], NEG_INF, dtype=tl.float32)
    init_int = init_imp.to(tl.int32, bitcast=True)
    init_encoded = (init_int & ~idx_mask) | (init_idx & idx_mask)
    running_topk_encoded = init_encoded.to(tl.float32, bitcast=True)

    o_v = tl.arange(0, V).to(tl.int64)
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    N_DROP_BLOCKS: tl.constexpr = TOP_K // DROP_BLOCK
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
    tile_rows = tl.arange(0, TILE_P).to(tl.int32)
    row_idx = tl.arange(0, 2)[:, None]
    block_ids = tl.arange(0, N_DROP_BLOCKS)[:, None]

    for i_batch in range(num_batches):
        # ---- Phase 1: TILE_P parents per tl.dot tile ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        # Per-batch running max used to stabilize exp(scores - m).
        # NOT use topk_running_max here (it starts at NEG_INF), otherwise exp overflows and later rescale (≈0) produces NaNs (inf * 0).
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff
            ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k   # [TILE_N, K//2]
            k_rope_hi = k_lo * sin_k + k_hi * cos_k   # [TILE_N, K//2]

            # Split tl.dot: [G_PAD, K//2] × [K//2, TILE_N] → [G_PAD, TILE_N]
            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            # Update per-batch running max and rescale previously written importance.
            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            # Importance: sum_g exp(score_g - batch_m). This stays finite (<= G_PAD).
            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)  # [TILE_N]
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            # Write tiled importance back to [PARENTS_PER_BATCH, CR]
            for i_r in tl.static_range(TILE_P):
                row_off = (tile_base + i_r).to(tl.int32)
                row_mask_in_tile = (tile_rows == i_r)[:, None]
                row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                is_row = (batch_rows == row_off)[:, None]
                importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        # ---- Phase 2: Top-K merge sort (unchanged) ----
        importance_flat = tl.reshape(importance_matrix, [BC])
        buf_off = (i_batch * BC).to(tl.int32)
        pos = buf_off + tl.arange(0, BC).to(tl.int32)
        valid_pos = pos < num_candidates
        importance_flat = tl.where(valid_pos, importance_flat, NEG_INF)

        # Rescale (running_topk, batch_importance) into a common max reference sel_m_new.
        # - running_topk is already in the previous reference topk_running_max
        # - current batch importance is in reference batch_m
        sel_m_new = tl.maximum(topk_running_max, batch_m)
        rescale_run = tl.exp(topk_running_max - sel_m_new)
        rescale_batch = tl.exp(batch_m - sel_m_new)

        run_int = running_topk_encoded.to(tl.int32, bitcast=True)
        run_raw_idx = run_int & idx_mask
        run_clean_int = run_int & ~idx_mask
        run_clean_imp = run_clean_int.to(tl.float32, bitcast=True)
        run_rescaled_imp = tl.where(run_clean_imp >= 0, run_clean_imp * rescale_run, run_clean_imp)
        run_rescaled_int = run_rescaled_imp.to(tl.int32, bitcast=True)
        run_rescaled_encoded = (run_rescaled_int & ~idx_mask) | run_raw_idx
        running_topk_encoded = run_rescaled_encoded.to(tl.float32, bitcast=True)

        importance_flat = tl.where(importance_flat >= 0, importance_flat * rescale_batch, importance_flat)
        importance_flat = tl.where(pos == rightmost_pos, 1e6, importance_flat)
        topk_running_max = sel_m_new

        importance_int = importance_flat.to(tl.int32, bitcast=True)
        encoded_idx = tl.where(importance_flat >= 0, ~pos, pos) & idx_mask
        encoded_int = (importance_int & ~idx_mask) | encoded_idx
        batch_encoded = encoded_int.to(tl.float32, bitcast=True)

        running_b = tl.broadcast_to(running_topk_encoded[None, :], [2, TOP_K])
        batch_b = tl.broadcast_to(batch_encoded[None, :], [2, TOP_K])
        merged_2d = tl.where(row_idx == 0, running_b, batch_b)
        merged_input = tl.reshape(merged_2d, [2 * TOP_K])

        N_DIMS_MERGE: tl.constexpr = N_DIMS_TOPK + 1
        sorted_merged = tl.sort(merged_input, descending=True)
        merged_sorted_2d = tl.reshape(sorted_merged, [2, TOP_K])

        running_topk_encoded = tl.sum(tl.where(row_idx == 0, merged_sorted_2d, 0.0), axis=0)
        dropped_encoded = tl.sum(tl.where(row_idx == 1, merged_sorted_2d, 0.0), axis=0)

        drop_int = dropped_encoded.to(tl.int32, bitcast=True)
        raw_idx = drop_int & idx_mask
        clean_int = drop_int & ~idx_mask
        clean_imp = clean_int.to(tl.float32, bitcast=True)
        drop_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx)
        drop_pos = (drop_pos & idx_mask).to(tl.int32)
        drop_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (drop_pos < num_candidates)

        # ---- Phase 3: accumulate dropped via tl.dot ----
        drop_pos_2d = tl.reshape(drop_pos, [N_DROP_BLOCKS, DROP_BLOCK]).to(tl.int32)
        drop_valid_2d_i32 = tl.reshape(drop_valid.to(tl.int32), [N_DROP_BLOCKS, DROP_BLOCK])

        for drop_block_idx in range(N_DROP_BLOCKS):
            row_mask = block_ids == drop_block_idx
            pos_blk = tl.sum(tl.where(row_mask, drop_pos_2d, 0), axis=0).to(tl.int32)
            valid_blk_i32 = tl.sum(tl.where(row_mask, drop_valid_2d_i32, 0), axis=0).to(tl.int32)
            valid_blk = valid_blk_i32 > 0

            parent_off_d = (pos_blk // COMPRESSION_RATE).to(tl.int32)
            child_slot = (pos_blk - parent_off_d * COMPRESSION_RATE).to(tl.int32)
            safe_poff_d = tl.minimum(tl.maximum(parent_off_d, 0), TOP_K - 1).to(tl.int64)
            parent_idx_d = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff_d
            ).to(tl.int32)

            is_rm_d = parent_idx_d == rightmost_parent_idx
            child_ok_d = (~is_rm_d) | (child_slot <= rightmost_child_idx)
            node_idx = tl.maximum(parent_idx_d, 0) * COMPRESSION_RATE + child_slot
            node_valid = valid_blk & (parent_idx_d >= 0) & child_ok_d & (node_idx >= 0) & (node_idx < N_layer)

            safe_node = tl.minimum(tl.maximum(node_idx, 0), N_layer - 1).to(tl.int64)
            safe_pos_d = tl.maximum(pos_blk, 0).to(tl.int64)

            k_row_d = k_base + safe_node[:, None] * (H_kv * K)
            k_lo_drop = tl.load(k_row_d + o_k_half[None, :]).to(tl.float32)
            k_hi_drop = tl.load(k_row_d + (K // 2) + o_k_half[None, :]).to(tl.float32)

            cos_k_drop = tl.load(cos_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            sin_k_drop = tl.load(sin_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo_drop = k_lo_drop * cos_k_drop - k_hi_drop * sin_k_drop   # [DROP_BLOCK, K//2]
            k_rope_hi_drop = k_lo_drop * sin_k_drop + k_hi_drop * cos_k_drop   # [DROP_BLOCK, K//2]

            # Split tl.dot: [G_PAD, K//2] × [K//2, DROP_BLOCK] → [G_PAD, DROP_BLOCK]
            scores_d = tl.dot(q_rope_lo, tl.trans(k_rope_lo_drop)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi_drop))
            scores_d = tl.where(node_valid[None, :] & g_valid[:, None], scores_d, NEG_INF)

            v_row_d = v_base + safe_node[:, None] * (H_kv * V)
            v_tile = tl.load(v_row_d + o_v[None, :]).to(tl.float32)  # [DROP_BLOCK, V]

            tile_m = tl.max(scores_d, axis=1)  # [G_PAD]
            new_max = tl.maximum(cur_max, tile_m)
            alpha = tl.exp(cur_max - new_max)
            cur_sum = cur_sum * alpha
            cur_output = cur_output * alpha[:, None]

            attn_probs = tl.exp(scores_d - new_max[:, None])
            attn_probs = tl.where(node_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

            # PV via tl.dot: [G_PAD, DROP_BLOCK] × [DROP_BLOCK, V] → [G_PAD, V]
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
            cur_max = new_max

    # ---- Store layer states (first NUM_GROUPS heads only) ----
    tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, cur_output, mask=g_valid[:, None])

    # ---- Decode final Top-K → next_selected_parents ----
    topk_int = running_topk_encoded.to(tl.int32, bitcast=True)
    raw_idx = topk_int & idx_mask
    clean_int = topk_int & ~idx_mask
    clean_imp = clean_int.to(tl.float32, bitcast=True)
    topk_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx)
    topk_pos = (topk_pos & idx_mask).to(tl.int32)
    topk_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (topk_pos < num_candidates)
    topk_pos = tl.where(topk_valid, topk_pos, -1).to(tl.int32)

    parent_off_f = tl.where(topk_pos >= 0, topk_pos // COMPRESSION_RATE, 0).to(tl.int32)
    child_slot_f = tl.where(topk_pos >= 0, topk_pos - parent_off_f * COMPRESSION_RATE, 0).to(tl.int32)
    parent_idx_f = tl.load(
        prev_selected_parents + prev_sel_base + parent_off_f.to(tl.int64),
        mask=topk_pos >= 0,
        other=-1,
    ).to(tl.int32)
    selected_node_indices = tl.where(
        topk_pos >= 0,
        parent_idx_f * COMPRESSION_RATE + child_slot_f,
        -1,
    ).to(tl.int32)

    MAX_IDX: tl.constexpr = 2147483647
    selected_sorted = tl.where(selected_node_indices >= 0, selected_node_indices, MAX_IDX)
    selected_sorted = tl.sort(selected_sorted, descending=False)
    selected_sorted = tl.where(selected_sorted < MAX_IDX, selected_sorted, -1)

    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(next_selected_parents + out_par_base + tl.arange(0, TOP_K), selected_sorted.to(tl.int32))


# ========================================
#  Main Forward Function
# ========================================

def htree_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """
    htree Forward
    
    Args:
        q: [B, T, H, K] - Query (H heads).
        k: [B, T, H_kv, K] - Key (H_kv heads, H_kv <= H, H % H_kv == 0).
        v: [B, T, H_kv, V] - Value (H_kv heads).
        compression_rate: 16
        max_top_nodes: 8192
        top_k_per_layer: 512
        scale: K^-0.5
        rope_base: 10000.0
    
    Returns:
        output: [B, T, H, V]
    """
    B, T, H, K = q.shape
    H_kv = k.shape[2]  # number of KV heads
    V = v.shape[-1]
    
    # Validate GQA configuration
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
    NUM_GROUPS = H // H_kv
    assert k.shape[2] == v.shape[2], f"K and V must have same number of heads"
    assert (top_k_per_layer & (top_k_per_layer - 1)) == 0, "top_k_per_layer (TOP_K) must be a power of 2"

    # Parameter regime constraints for this Triton implementation.
    # - Candidate buffer assumes MAX_CANDIDATES == TOP_K * COMPRESSION_RATE.
    # - Bitonic sorts assume power-of-2 sizes (TOP_K and MAX_CANDIDATES).
    # - Fused streaming Top-K assumes TOP_K is a multiple of COMPRESSION_RATE:
    #     PARENTS_PER_BATCH = TOP_K / COMPRESSION_RATE, per-batch candidates == TOP_K.
    assert max_top_nodes == top_k_per_layer * compression_rate, (
        f"max_top_nodes ({max_top_nodes}) must equal top_k_per_layer*compression_rate "
        f"({top_k_per_layer}*{compression_rate}={top_k_per_layer * compression_rate})"
    )
    assert (compression_rate & (compression_rate - 1)) == 0, "compression_rate must be a power of 2"
    assert (max_top_nodes & (max_top_nodes - 1)) == 0, "max_top_nodes must be a power of 2"
    assert top_k_per_layer % compression_rate == 0, (
        f"top_k_per_layer ({top_k_per_layer}) must be divisible by compression_rate ({compression_rate})"
    )
    
    if scale is None:
        scale = K ** -0.5
    
    device = q.device
    dtype = q.dtype
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # --- Phase 1: Build Tree ---
    nvtx.range_push("Phase1_TreeBuilding")
    logger.info("Phase 1: Building tree structure...")
    
    num_layers = 1
    temp_len = T
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    
    logger.info(f"  Tree has {num_layers} layers (H={H}, H_kv={H_kv}, num_groups={NUM_GROUPS})")
    
    layers_k = [k]
    layers_v = [v]
    
    current_k, current_v = k, v
    current_len = T
    
    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H_kv, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H_kv, V, dtype=dtype, device=device)
        
        BLOCK_SIZE = 8
        grid = (triton.cdiv(next_len, BLOCK_SIZE), B * H_kv)
        
        with _cuda_timer(f"  Built layer {layer_idx}: {(current_len + compression_rate - 1) // compression_rate} nodes"):
            htree_build_kernel[grid](
                current_k, current_v, next_k, next_v,
                N_child=current_len,
                N_parent=next_len,
                B=B, H_kv=H_kv, K=K, V=V,
                COMPRESSION_RATE=compression_rate,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len
    nvtx.range_pop()

    # --- Phase 1.5: RoPE Cache ---
    nvtx.range_push("Phase1.5_RoPE_Cache")
    logger.info("Phase 1.5: Precomputing RoPE cache...")
    
    cache_size = max_top_nodes + 1024
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
    positions = torch.arange(cache_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    nvtx.range_pop()

    # --- Phase 2: Init Global State ---
    nvtx.range_push("Phase2_Init_GlobalState_and_Buffers")
    logger.info("Phase 2: Initializing global states and buffers...")
    
    global_max = torch.full([B, T, H], -1e10, dtype=torch.float32, device=device)
    global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
    global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)
    
    nvtx.range_pop()

    # --- Phase 3: Layer-by-layer Forward ---
    nvtx.range_push("Phase3_LayerByLayer_Forward")
    logger.info("Phase 3: Layer-by-layer forward pass...")
    
    layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)

    # per-layer reusable workspaces
    # MAX_CANDIDATES is the physical maximum candidate length (TOP_K * compression_rate).
    MAX_CANDIDATES = max_top_nodes

    # constexprs for kernels that use bitonic sort / bit-packing
    LOG_N = int(math.log2(MAX_CANDIDATES))
    N_DIMS_TOPK = int(math.log2(top_k_per_layer))

    # Dropped candidates are accumulated in blocks for better throughput.
    DROP_BLOCK = 64
    assert top_k_per_layer % DROP_BLOCK == 0, f"top_k_per_layer ({top_k_per_layer}) must be divisible by DROP_BLOCK ({DROP_BLOCK})"

    # Bottom kernel: batch TILE_P parents per iteration; pad G to ≥16 for tl.dot
    TILE_P = 4
    G_PAD = max(triton.next_power_of_2(NUM_GROUPS), 16)
    
    # Initialize prev_selected_parents for the topmost layer
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)  # [T]
    rightmost_indices = t_indices // top_layer_power  # [T] rightmost node index at top layer
    # Number of virtual parents (each expands to CR top-level nodes)
    num_virtual_parents = rightmost_indices // compression_rate + 1  # [T]
    
    # [T, TOP_K]
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
    prev_selected_parents = prev_selected_parents.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, top_k_per_layer).contiguous()
    next_selected_parents = torch.empty_like(prev_selected_parents)
    
    for layer_idx in range(num_layers - 1, -1, -1):
        nvtx.range_push(f"Forward_Layer_{layer_idx}")
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        
        is_bottom_layer = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        
        logger.info(f"  -> Processing layer {layer_idx} (N={N_layer}, power={layer_power}, bottom={is_bottom_layer})...")
        
        grid = (T, B * H)

        # Kernel 1: Select+Accumulate (dispatch bottom vs non-bottom)
        nvtx.range_push("K1_SelectAccumulate")
        grid_kv = (T, B * H_kv)
        kernel_label = "Kernel 1a" if is_bottom_layer else "Kernel 1b"
        with _cuda_timer(f"      {kernel_label}"):
            if is_bottom_layer:
                logger.info("    Running bottom accumulate kernel (Kernel 1a)...")
                htree_bottom_accumulate_gqa_kernel[grid_kv](
                    q, k_layer, v_layer,
                    prev_selected_parents,
                    cos_cache, sin_cache,
                    layer_max, layer_sum, layer_output,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    NEG_INF=HTREE_SCORE_NEG_INF,
                    G_PAD=G_PAD,
                    TILE_P=TILE_P,
                    scale=scale,
                )
            else:
                logger.info("    Running select+accumulate kernel (Kernel 1b)...")
                htree_select_accumulate_gqa_kernel[grid_kv](
                    q, k_layer, v_layer,
                    prev_selected_parents,
                    cos_cache, sin_cache,
                    layer_max, layer_sum, layer_output,
                    next_selected_parents,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    LOG_N=LOG_N,
                    N_DIMS_TOPK=N_DIMS_TOPK,
                    SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
                    NEG_INF=HTREE_SCORE_NEG_INF,
                    DROP_BLOCK=DROP_BLOCK,
                    G_PAD=G_PAD,
                    TILE_P=TILE_P,
                    scale=scale,
                )
        nvtx.range_pop()
        
        # Kernel 2: Merge to Global State
        nvtx.range_push("K2_MergeToGlobal")
        logger.info("    Running merge to global kernel (Kernel 2)...")
        with _cuda_timer("      Kernel 2"):
            htree_merge_to_global_kernel[grid](
                layer_max, layer_sum, layer_output,
                global_max, global_sum, global_output,
                B=B, T=T, H=H, V=V,
            )
        nvtx.range_pop()

        # Swap parent indices: next becomes prev for the layer below.
        if not is_bottom_layer:
            prev_selected_parents, next_selected_parents = next_selected_parents, prev_selected_parents
        
        nvtx.range_pop()
    nvtx.range_pop()

    # --- Phase 4: Final Normalize ---
    nvtx.range_push("Phase4_Final_Normalize")
    logger.info("Phase 4: Final normalization...")
    
    output = torch.empty(B, T, H, V, dtype=dtype, device=device)
    grid = (T, B * H)
    
    with _cuda_timer("  Kernel 3 (final normalize)"):
        htree_final_normalize_kernel[grid](
            global_output, global_sum, output,
            B=B, T=T, H=H, V=V,
        )
    nvtx.range_pop()

    logger.info("htree forward pass completed!")
    
    return output


__all__ = [
    'htree_forward',
    'htree_build_kernel',
    'htree_bottom_accumulate_gqa_kernel',
    'htree_select_accumulate_gqa_kernel',
    'htree_merge_to_global_kernel',
    'htree_final_normalize_kernel',
]
