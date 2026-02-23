# -*- coding: utf-8 -*-
"""
Triton built-in tl.sort / tl.topk based Top-K kernels.

Uses tl.topk(x, K) for partial sort (returns top-K values sorted descending).
For index tracking (needed for payloads), uses the same bit-packing approach
as bitonic kernels: pack score + index into float32, sort, then unpack.

Direct:    tl.topk(N, K) on 8192 elements → top 512.
Streaming: 16× tl.sort(2K) on merged 1024 → keep top 512.
           (tl.topk cannot be used for streaming since we need to merge
            two K-sized arrays; we sort the full 2K and take top K.)
"""

import triton
import triton.language as tl


@triton.jit
def _pack_score_index(score, idx, idx_mask: tl.constexpr):
    s_int = score.to(tl.int32, bitcast=True)
    enc_idx = tl.where(score >= 0, ~idx, idx) & idx_mask
    packed = (s_int & ~idx_mask) | enc_idx
    return packed.to(tl.float32, bitcast=True)


@triton.jit
def _unpack_index(packed, idx_mask: tl.constexpr):
    p_int = packed.to(tl.int32, bitcast=True)
    raw = p_int & idx_mask
    clean = p_int & ~idx_mask
    score = clean.to(tl.float32, bitcast=True)
    idx = tl.where(score >= 0, ~raw, raw) & idx_mask
    return idx.to(tl.int32)


# ============================================================
# Kernel: Triton Built-in Direct  (tl.topk on N → K)
# ============================================================

@triton.jit
def triton_topk_direct_kernel(
    inp_ptr,
    pay1_ptr,
    pay8_ptr,
    out_val_ptr,
    out_idx_ptr,
    out_pay1_ptr,
    out_pay8_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    LOG_N: tl.constexpr,
    N_PAY: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N
    offs = tl.arange(0, N)
    vals = tl.load(inp_ptr + base + offs)

    idx_mask: tl.constexpr = (1 << LOG_N) - 1
    packed = _pack_score_index(vals, offs.to(tl.int32), idx_mask)

    # tl.topk returns top-K values sorted descending — single tensor [K]
    top_packed = tl.topk(packed, K)

    top_idx = _unpack_index(top_packed, idx_mask)
    top_val = tl.load(inp_ptr + base + top_idx.to(tl.int64))

    k_offs = tl.arange(0, K)
    out_base = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base + k_offs, top_val)
    tl.store(out_idx_ptr + out_base + k_offs, top_idx)

    if N_PAY >= 1:
        p1 = tl.load(pay1_ptr + base + top_idx.to(tl.int64))
        tl.store(out_pay1_ptr + out_base + k_offs, p1)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(pay8_ptr + pid.to(tl.int64) * N * 8 + top_idx.to(tl.int64) * 8 + pi)
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + k_offs.to(tl.int64) * 8 + pi, p8)


# ============================================================
# Kernel: Triton Built-in Streaming  (16× tl.sort on 2K)
# ============================================================

@triton.jit
def triton_topk_streaming_kernel(
    inp_ptr,
    pay1_ptr,
    pay8_ptr,
    out_val_ptr,
    out_idx_ptr,
    out_pay1_ptr,
    out_pay8_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_DIMS_2K: tl.constexpr,
    LOG_N: tl.constexpr,
    N_PAY: tl.constexpr,
    NEG_INF: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N
    idx_mask: tl.constexpr = (1 << LOG_N) - 1

    # Init running top-K with NEG_INF
    init_offs = tl.arange(0, K)
    init_vals = tl.full([K], NEG_INF, dtype=tl.float32)
    running = _pack_score_index(init_vals, init_offs.to(tl.int32), idx_mask)

    for i_batch in range(NUM_BATCHES):
        batch_start = i_batch * K
        b_offs = tl.arange(0, K)
        b_vals = tl.load(inp_ptr + base + batch_start + b_offs)
        global_idx = (batch_start + b_offs).to(tl.int32)
        batch_packed = _pack_score_index(b_vals, global_idx, idx_mask)

        # Merge: [2, K] → [2K]
        running_b = tl.broadcast_to(running[None, :], [2, K])
        batch_b = tl.broadcast_to(batch_packed[None, :], [2, K])
        row_idx = tl.arange(0, 2)[:, None]
        merged_2d = tl.where(row_idx == 0, running_b, batch_b)
        TWO_K: tl.constexpr = 2 * K
        merged = tl.reshape(merged_2d, [TWO_K])

        # Use tl.sort (built-in) instead of manual bitonic sort
        sorted_merged = tl.sort(merged, descending=True)

        # Keep top K (row 0 of [2, K])
        sorted_2d = tl.reshape(sorted_merged, [2, K])
        running = tl.sum(tl.where(row_idx == 0, sorted_2d, 0.0), axis=0)

    # Decode final top-K
    k_offs = tl.arange(0, K)
    top_idx = _unpack_index(running, idx_mask)
    top_val = tl.load(inp_ptr + base + top_idx.to(tl.int64))

    out_base = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base + k_offs, top_val)
    tl.store(out_idx_ptr + out_base + k_offs, top_idx)

    if N_PAY >= 1:
        p1 = tl.load(pay1_ptr + base + top_idx.to(tl.int64))
        tl.store(out_pay1_ptr + out_base + k_offs, p1)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(pay8_ptr + pid.to(tl.int64) * N * 8 + top_idx.to(tl.int64) * 8 + pi)
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + k_offs.to(tl.int64) * 8 + pi, p8)
