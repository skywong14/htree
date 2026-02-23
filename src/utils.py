# -*- coding: utf-8 -*-
import triton
import triton.language as tl

# ==========================================
# 排序辅助函数 (Bit-Packing Top-K)
# ==========================================

@triton.jit
def _compare_and_swap_single(
    x,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """单数组 compare-and-swap, 不追踪索引 (索引已编码在值的低位)"""
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
def _bitonic_merge_single(
    x,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """单数组 bitonic merge"""
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x = _compare_and_swap_single(x, flip, i + (n_dims - stage), n_dims, n_outer)
    return x


@triton.jit
def sort_single(
    x,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
    descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    """单数组 bitonic sort (不返回索引, 用于 Bit-Packing Top-K)"""
    for i in tl.static_range(1, n_dims + 1):
        x = _bitonic_merge_single(x, i, 2 if i < n_dims else descending, n_dims, n_outer)
    return x
