# -*- coding: utf-8 -*-
"""
Radix Select based Top-K kernels for Triton.

Algorithm: bit-by-bit radix selection (MSB → LSB) on order-preserving
int32 representation of float32 values.  O(32·N) work per select,
minimal register pressure (no sort network reshapes).

Direct:    load N=8192, radix-select top K=512, compact output.
Streaming: 16× merge running top-512 with new 512 → radix-select top 512
           from 1024, compact via global scratch.
"""

import triton
import triton.language as tl


@triton.jit
def _float_to_ord(x):
    """float32 → signed int32 that preserves order.

    float a < float b  ⟺  ord(a) < ord(b)  (signed int32 comparison).

    IEEE 754 float32 layout:  [sign(1) | exponent(8) | mantissa(23)]
    - Positive floats (sign=0): int32 representation is already correctly
      ordered among themselves AND is non-negative → above all negative ints.
    - Negative floats (sign=1): int32 representation has reversed order
      among themselves (more negative float → larger magnitude int).
      Fix: XOR with 0x7FFFFFFF (flip all bits except sign) to reverse
      the value bits while keeping the sign bit = 1 (negative int32).
    """
    xi = x.to(tl.int32, bitcast=True)
    is_neg = xi < 0
    # neg: xi ^ 0x7FFFFFFF  (flip exponent+mantissa, keep sign=1 → negative, reversed)
    # pos: unchanged  (already ordered, non-negative)
    return tl.where(is_neg, xi ^ 0x7FFFFFFF, xi)


@triton.jit
def _radix_select(x_ord, N_VEC: tl.constexpr, K_VAL):
    """Core radix selection on order-preserving int32 values.

    Expects values where a < b (float) ⟺ a_ord < b_ord (signed int32).
    Internally flips bit 31 to get unsigned-equivalent ordering so that
    the MSB-first radix scan correctly associates bit=1 with "larger".

    Returns (above, select_at, remaining_k):
      - above:     [N_VEC] int32  – 1 if element is strictly above threshold
      - select_at: [N_VEC] int32  – 1 if element is at threshold and selected
      - remaining_k: scalar int32 – how many threshold-level elements to take
    """
    # Convert signed ordering → unsigned-equivalent: flip sign bit so that
    # positive int32 (originally large floats) get bit 31 = 1 (= "large").
    x_u = x_ord ^ tl.full([N_VEC], -2147483648, dtype=tl.int32)  # XOR 0x80000000

    active = tl.full([N_VEC], 1, dtype=tl.int32)
    remaining_k = K_VAL

    for bit_pos in tl.static_range(31, -1, -1):
        bit_val = tl.full([N_VEC], 1, dtype=tl.int32) << bit_pos
        has_bit = ((x_u & bit_val) != 0).to(tl.int32)
        count = tl.sum(has_bit * active)

        ge_k = (count >= remaining_k).to(tl.int32)

        # ge_k=1: threshold has this bit=1 → keep only active elements with bit=1
        # ge_k=0: elements with bit=1 are guaranteed top-K → continue with bit=0
        active = active * (has_bit * ge_k + (1 - has_bit) * (1 - ge_k))
        remaining_k = remaining_k - count * (1 - ge_k)

    # Recover threshold in original signed ordering
    # threshold_u is in unsigned space; convert back by flipping bit 31
    MIN_INT: tl.constexpr = -2147483648
    threshold_u = tl.max(tl.where(active > 0, x_u, tl.full([N_VEC], MIN_INT, dtype=tl.int32)))
    threshold_val = threshold_u ^ tl.full([1], -2147483648, dtype=tl.int32)

    above = (x_ord > threshold_val).to(tl.int32)
    at = (x_ord == threshold_val).to(tl.int32)
    cumsum_at = tl.cumsum(at, axis=0)
    select_at = at * (cumsum_at <= remaining_k).to(tl.int32)

    return above, select_at, remaining_k


# ============================================================
# Kernel: Radix Direct  (select top K from N)
# ============================================================

@triton.jit
def radix_topk_direct_kernel(
    inp_ptr,          # [BATCH, N]  float32
    pay1_ptr,         # [BATCH, N]  int32
    pay8_ptr,         # [BATCH, N, 8] int32
    out_val_ptr,      # [BATCH, K]  float32
    out_idx_ptr,      # [BATCH, K]  int32
    out_pay1_ptr,     # [BATCH, K]  int32
    out_pay8_ptr,     # [BATCH, K, 8] int32
    N: tl.constexpr,
    K: tl.constexpr,
    N_PAY: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * N
    offs = tl.arange(0, N)
    x = tl.load(inp_ptr + base + offs)

    x_ord = _float_to_ord(x)  # [N] int32, order-preserving

    above, select_at, remaining_k = _radix_select(x_ord, N, K)
    selected = above + select_at  # 1 = selected

    # Compact output via cumsum
    out_pos = tl.cumsum(selected, axis=0) - 1  # 0-indexed
    ok = (selected > 0) & (out_pos < K)

    out_base = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base + out_pos.to(tl.int64), x, mask=ok)
    tl.store(out_idx_ptr + out_base + out_pos.to(tl.int64), offs.to(tl.int32), mask=ok)

    if N_PAY >= 1:
        p1 = tl.load(pay1_ptr + base + offs)
        tl.store(out_pay1_ptr + out_base + out_pos.to(tl.int64), p1, mask=ok)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(pay8_ptr + pid.to(tl.int64) * N * 8 + offs.to(tl.int64) * 8 + pi)
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + out_pos.to(tl.int64) * 8 + pi, p8, mask=ok)


# ============================================================
# Kernel: Radix Streaming v2 (with global scratch for compaction)
# ============================================================

@triton.jit
def radix_topk_streaming_v2_kernel(
    inp_ptr,          # [BATCH, N] float32
    pay1_ptr,         # [BATCH, N] int32
    pay8_ptr,         # [BATCH, N, 8] int32
    out_val_ptr,      # [BATCH, K] float32
    out_idx_ptr,      # [BATCH, K] int32
    out_pay1_ptr,     # [BATCH, K] int32
    out_pay8_ptr,     # [BATCH, K, 8] int32
    scratch_val_ptr,  # [BATCH, K] float32
    scratch_idx_ptr,  # [BATCH, K] int32
    scratch_pay1_ptr, # [BATCH, K] int32
    N: tl.constexpr,
    K: tl.constexpr,
    N_PAY: tl.constexpr,
    NEG_INF: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    TWO_K: tl.constexpr,
):
    pid = tl.program_id(0)
    in_base = pid.to(tl.int64) * N
    sc_base = pid.to(tl.int64) * K
    k_offs = tl.arange(0, K).to(tl.int64)

    # Initialize scratch with NEG_INF
    tl.store(scratch_val_ptr + sc_base + k_offs, tl.full([K], NEG_INF, dtype=tl.float32))
    tl.store(scratch_idx_ptr + sc_base + k_offs, tl.full([K], -1, dtype=tl.int32))
    if N_PAY >= 1:
        tl.store(scratch_pay1_ptr + sc_base + k_offs, tl.zeros([K], dtype=tl.int32))

    for i_batch in range(NUM_BATCHES):
        batch_start = (i_batch * K).to(tl.int64)

        # Load running top-K from scratch
        run_vals = tl.load(scratch_val_ptr + sc_base + k_offs)
        run_idx = tl.load(scratch_idx_ptr + sc_base + k_offs)

        # Load new batch
        b_vals = tl.load(inp_ptr + in_base + batch_start + k_offs)
        b_idx = (batch_start + k_offs).to(tl.int32)

        # Merge [2K]: running in [0,K), batch in [K,2K)
        run_2 = tl.broadcast_to(run_vals[None, :], [2, K])
        bat_2 = tl.broadcast_to(b_vals[None, :], [2, K])
        row = tl.arange(0, 2)[:, None]
        m_vals = tl.reshape(tl.where(row == 0, run_2, bat_2), [TWO_K])

        ri2 = tl.broadcast_to(run_idx[None, :], [2, K])
        bi2 = tl.broadcast_to(b_idx[None, :], [2, K])
        m_idx = tl.reshape(tl.where(row == 0, ri2, bi2), [TWO_K]).to(tl.int32)

        if N_PAY >= 1:
            run_p1 = tl.load(scratch_pay1_ptr + sc_base + k_offs)
            b_p1 = tl.load(pay1_ptr + in_base + batch_start + k_offs)
            rp12 = tl.broadcast_to(run_p1[None, :], [2, K])
            bp12 = tl.broadcast_to(b_p1[None, :], [2, K])
            m_pay1 = tl.reshape(tl.where(row == 0, rp12, bp12), [TWO_K]).to(tl.int32)

        # Radix select top K from 2K
        x_ord = _float_to_ord(m_vals)
        above, select_at, remaining_k = _radix_select(x_ord, TWO_K, K)
        selected = above + select_at

        # Compact via scatter to scratch
        out_pos = tl.cumsum(selected, axis=0) - 1
        ok = (selected > 0) & (out_pos < K)

        tl.store(scratch_val_ptr + sc_base + out_pos.to(tl.int64), m_vals, mask=ok)
        tl.store(scratch_idx_ptr + sc_base + out_pos.to(tl.int64), m_idx, mask=ok)
        if N_PAY >= 1:
            tl.store(scratch_pay1_ptr + sc_base + out_pos.to(tl.int64), m_pay1, mask=ok)

    # Final: copy scratch → output
    final_vals = tl.load(scratch_val_ptr + sc_base + k_offs)
    final_idx = tl.load(scratch_idx_ptr + sc_base + k_offs)
    out_base = pid.to(tl.int64) * K
    tl.store(out_val_ptr + out_base + k_offs, final_vals)
    tl.store(out_idx_ptr + out_base + k_offs, final_idx)

    if N_PAY >= 1:
        fp1 = tl.load(scratch_pay1_ptr + sc_base + k_offs)
        tl.store(out_pay1_ptr + out_base + k_offs, fp1)
    if N_PAY >= 8:
        for pi in tl.static_range(8):
            p8 = tl.load(
                pay8_ptr + pid.to(tl.int64) * N * 8 + final_idx.to(tl.int64) * 8 + pi,
                mask=final_idx >= 0, other=0
            )
            tl.store(out_pay8_ptr + pid.to(tl.int64) * K * 8 + k_offs * 8 + pi, p8)
