# -*- coding: utf-8 -*-
"""
Bitonic Sort based Top-K kernels (reused from parallel.py).

Direct:    sort full N=8192 descending, take top K=512.
Streaming: 16× merge running top-512 with new 512 → sort 1024 → keep top 512.
"""

import triton
import triton.language as tl

# ============================================================
# Core bitonic primitives (copied from parallel.py)
# ============================================================

@triton.jit
def _compare_and_swap_single(x, flip, i: tl.constexpr, n_dims: tl.constexpr, n_outer: tl.constexpr):
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    return ret.to(x.dtype, bitcast=True)


@triton.jit
def _bitonic_merge_single(x, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr, n_outer: tl.constexpr):
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x = _compare_and_swap_single(x, flip, i + (n_dims - stage), n_dims, n_outer)
    return x


@triton.jit
def sort_single(x, n_dims: tl.constexpr, n_outer: tl.constexpr, descending: tl.constexpr = tl.core.CONSTEXPR_0):
    for i in tl.static_range(1, n_dims + 1):
        x = _bitonic_merge_single(x, i, 2 if i < n_dims else descending, n_dims, n_outer)
    return x


# ============================================================
# Bit-packing helpers
# ============================================================

@triton.jit
def _pack_score_index(score, idx, idx_mask: tl.constexpr):
    """Pack score (float32) + index (int32) into float32 via bit manipulation."""
    s_int = score.to(tl.int32, bitcast=True)
    enc_idx = tl.where(score >= 0, ~idx, idx) & idx_mask
    packed = (s_int & ~idx_mask) | enc_idx
    return packed.to(tl.float32, bitcast=True)


@triton.jit
def _unpack_index(packed, idx_mask: tl.constexpr):
    """Unpack original index from packed float32."""
    p_int = packed.to(tl.int32, bitcast=True)
    raw = p_int & idx_mask
    clean = p_int & ~idx_mask
    score = clean.to(tl.float32, bitcast=True)
    idx = tl.where(score >= 0, ~raw, raw) & idx_mask
    return idx.to(tl.int32)


# ============================================================
# Kernel: Bitonic Direct  (sort N, take top K)
# ============================================================

@triton.jit
def bitonic_topk_direct_kernel(
    inp_ptr,          # [BATCH, N]  float32
    pay1_ptr,         # [BATCH, N]  int32   (payload, may be unused)
    pay8_ptr,         # [BATCH, N, 8] int32 (payload, may be unused)
    out_val_ptr,      # [BATCH, K]  float32
    out_idx_ptr,      # [BATCH, K]  int32
    out_pay1_ptr,     # [BATCH, K]  int32
    out_pay8_ptr,     # [BATCH, K, 8] int32
    N: tl.constexpr,          # 8192
    K: tl.constexpr,          # 512
    N_DIMS: tl.constexpr,     # log2(N) = 13
    LOG_N: tl.constexpr,      # = N_DIMS (bits for index)
    N_PAY: tl.constexpr,      # 0, 1, or 8
    NEG_INF: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N
    offs = tl.arange(0, N)
    vals = tl.load(inp_ptr + base + offs)

    idx_mask: tl.constexpr = (1 << LOG_N) - 1
    packed = _pack_score_index(vals, offs.to(tl.int32), idx_mask)

    # Full bitonic sort descending
    sorted_packed = sort_single(packed, N_DIMS, 1, descending=True)

    # Extract top K: reshape [N] → [N/K, K], take row 0
    KR: tl.constexpr = N // K
    reshaped = tl.reshape(sorted_packed, [KR, K])
    row_id = tl.arange(0, KR)[:, None]
    top_k_packed = tl.sum(tl.where(row_id == 0, reshaped, 0.0), axis=0)  # [K]

    top_idx = _unpack_index(top_k_packed, idx_mask)  # [K]

    # Reload exact values using indices (bit-packing truncates low mantissa bits)
    top_val_exact = tl.load(inp_ptr + base + top_idx.to(tl.int64))

    k_offs = tl.arange(0, K)
    out_base_k = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base_k + k_offs, top_val_exact)
    tl.store(out_idx_ptr + out_base_k + k_offs, top_idx)

    # Payloads
    if N_PAY >= 1:
        p1 = tl.load(pay1_ptr + base + top_idx.to(tl.int64))
        tl.store(out_pay1_ptr + out_base_k + k_offs, p1)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(pay8_ptr + pid.to(tl.int64) * N * 8 + top_idx.to(tl.int64) * 8 + pi)
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + k_offs.to(tl.int64) * 8 + pi, p8)


# ============================================================
# Kernel: Bitonic Streaming  (16× merge top-K with batch 512)
# ============================================================

@triton.jit
def bitonic_topk_streaming_kernel(
    inp_ptr,
    pay1_ptr,
    pay8_ptr,
    out_val_ptr,
    out_idx_ptr,
    out_pay1_ptr,
    out_pay8_ptr,
    N: tl.constexpr,          # 8192
    K: tl.constexpr,          # 512
    N_DIMS_K: tl.constexpr,   # log2(K) = 9
    N_DIMS_2K: tl.constexpr,  # log2(2K) = 10
    LOG_N: tl.constexpr,      # bits for global index in [0, N)
    N_PAY: tl.constexpr,
    NEG_INF: tl.constexpr,
    NUM_BATCHES: tl.constexpr, # N // K = 16
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

        # Merge: [2, K] -> [2K]
        running_b = tl.broadcast_to(running[None, :], [2, K])
        batch_b = tl.broadcast_to(batch_packed[None, :], [2, K])
        row_idx = tl.arange(0, 2)[:, None]
        merged_2d = tl.where(row_idx == 0, running_b, batch_b)
        merged = tl.reshape(merged_2d, [2 * K])

        sorted_merged = sort_single(merged, N_DIMS_2K, 1, descending=True)

        # Keep top K (row 0 of [2, K])
        sorted_2d = tl.reshape(sorted_merged, [2, K])
        running = tl.sum(tl.where(row_idx == 0, sorted_2d, 0.0), axis=0)

    # Decode final top-K
    k_offs = tl.arange(0, K)
    top_idx = _unpack_index(running, idx_mask)
    top_val = tl.load(inp_ptr + base + top_idx.to(tl.int64))

    out_base_k = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base_k + k_offs, top_val)
    tl.store(out_idx_ptr + out_base_k + k_offs, top_idx)

    if N_PAY >= 1:
        p1 = tl.load(pay1_ptr + base + top_idx.to(tl.int64))
        tl.store(out_pay1_ptr + out_base_k + k_offs, p1)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(pay8_ptr + pid.to(tl.int64) * N * 8 + top_idx.to(tl.int64) * 8 + pi)
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + k_offs.to(tl.int64) * 8 + pi, p8)
