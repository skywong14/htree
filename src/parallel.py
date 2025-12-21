# -*- coding: utf-8 -*-
"""
htree Triton Kernel - stable Top-K 版本（parallel2_stable_topk.py）


Pipeline
  Phase 1: Tree Building (htree_build_kernel_v2)
    - 逐层 mean pooling 构建树

  Phase 2: Forward (top → bottom)
    - Kernel 2.1 compute_scores_and_select:
        · 32 父 × 16 子批次遍历，RoPE 后写入固定 8192 buffer (scores & indices)
        · 非底层：将最右节点分数置 1e3，Bit-Packing 编码 buffer 位置并做单数组 bitonic 排序 → 稳定 Top-K
        · 底层跳过排序，topk_positions/selected_indices 置 -1
    - Kernel 2.2 accumulate_non_topk:
        · 按原批次顺序流式加载 V，基于 topk_positions 构建 mask，仅累积非 Top-K；底层累积全部
    - Kernel 2.3 merge_to_global_v2:
        · 在线 Softmax 合并当前层累计到全局状态

  Phase 3: Final Normalize (htree_final_normalize_kernel_v2)
    - output = global_output / global_sum
"""

from typing import Optional, Dict, Tuple
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# ==========================================
# Kernel 1: Tree Building
# ==========================================

@triton.jit
def htree_build_kernel_v2(
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
    Grid: (B * H_kv,)
    """
    i_bh = tl.program_id(0)
    i_b = i_bh // H_kv
    i_h = i_bh % H_kv
    
    num_iterations = tl.cdiv(N_parent, BLOCK_SIZE)
    
    for iter_idx in range(num_iterations):
        parent_start = iter_idx * BLOCK_SIZE
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


# ==========================================
# 辅助函数
# ==========================================

@triton.jit
def load_k_with_rope_v2(
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
    """加载 16 个 token 的 K 并应用 RoPE 编码"""
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    
    # 前半部分
    k1_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k1 = tl.load(k1_block_ptrs, boundary_check=(0, 1))
    
    # 后半部分
    k2_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, K // 2),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k2 = tl.load(k2_block_ptrs, boundary_check=(0, 1))
    
    # 应用 RoPE
    rope_positions = rope_position_start + tl.arange(0, COMPRESSION_RATE)
    o_k_half = tl.arange(0, K // 2)
    
    valid_rows = tl.arange(0, COMPRESSION_RATE)[:, None] < num_valid_children
    cos_ptrs = cos_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    sin_ptrs = sin_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    cos_k = tl.load(cos_ptrs, mask=valid_rows, other=0.0)
    sin_k = tl.load(sin_ptrs, mask=valid_rows, other=0.0)
    
    k_rope_1 = k1 * cos_k - k2 * sin_k
    k_rope_2 = k1 * sin_k + k2 * cos_k
    
    return k_rope_1, k_rope_2


@triton.jit
def load_v_v2(
    layer_v,
    i_b,
    i_h_kv,
    child_start,
    N_layer: tl.constexpr,
    H_kv: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """加载 16 个 token 的 V"""
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V
    
    v_block_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(N_layer, V),
        strides=(H_kv * V, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, V),
        order=(1, 0)
    )
    v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))
    
    return v_vals


# ==========================================
# Kernel 2.1: Compute Scores & Select Top-K
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=4, num_stages=2),
#         triton.Config({}, num_warps=8, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=3),
#         triton.Config({}, num_warps=8, num_stages=3),
#     ],
#     key=['B', 'H', 'K']
# )
@triton.jit
def htree_compute_scores_and_select_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    prev_selected_parents,  # [B, T, H, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # 输出 buffer
    all_scores,  # [B, T, H, MAX_CANDIDATES]
    num_candidates,  # [B, T, H]
    # 输出 Top-K
    topk_positions,  # [B, T, H, TOP_K]
    # 参数
    layer_idx: tl.constexpr,
    layer_power: tl.constexpr,
    B: tl.constexpr, T, H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    scale,
):
    """
    Kernel 2.1: 计算所有候选节点的 attention scores，并选出 Top-K
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    # GQA: 映射 Query 头索引到 KV 头索引
    i_h_kv = i_h // NUM_GROUPS
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    BC: tl.constexpr = TOP_K  # 512
    PARENTS_PER_BATCH: tl.constexpr = 32
    
    # ========================================
    # 阶段 1: 确定候选节点范围
    # ========================================
    
    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE
    
    T_i64 = T.to(tl.int64)
    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H * TOP_K
        + i_t.to(tl.int64) * H * TOP_K
        + i_h.to(tl.int64) * TOP_K
    )
    o_parent = tl.arange(0, TOP_K)
    parent_list = tl.load(prev_selected_parents + prev_sel_base + o_parent)
    
    valid_mask = parent_list >= 0
    num_valid_parents = tl.sum(valid_mask.to(tl.int32))
    num_batches = (num_valid_parents + 31) // 32
    
    # 计算实际候选节点数量
    if num_valid_parents > 0:
        n_cand = (num_valid_parents - 1) * COMPRESSION_RATE + (rightmost_idx % COMPRESSION_RATE) + 1
    else:
        n_cand = 0
    
    # 存储 num_candidates
    num_cand_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    tl.store(num_candidates + num_cand_offset, n_cand)
    
    # ========================================
    # 阶段 2: 应用 RoPE 到 Query
    # ========================================
    
    rope_pos_q = tl.maximum(n_cand - 1, 0)
    
    q_offset = (
        i_b.to(tl.int64) * T_i64 * H * K
        + i_t.to(tl.int64) * H * K
        + i_h.to(tl.int64) * K
    )
    o_k = tl.arange(0, K // 2)
    
    q1 = tl.load(q + q_offset + o_k)
    q2 = tl.load(q + q_offset + (K // 2) + o_k)
    
    rope_pos_q_i64 = rope_pos_q.to(tl.int64)
    cos_q = tl.load(cos_cache + rope_pos_q_i64 * (K // 2) + o_k)
    sin_q = tl.load(sin_cache + rope_pos_q_i64 * (K // 2) + o_k)
    
    q_rope_1 = (q1 * cos_q - q2 * sin_q) * scale
    q_rope_2 = (q1 * sin_q + q2 * cos_q) * scale
    
    # ========================================
    # 阶段 3: 批次遍历，加载 K 并计算 scores，同时维护 Streaming Top-K
    # ========================================

    # NOTE: all_scores is [B, T, H, MAX_CANDIDATES]. When T*H*MAX_CANDIDATES
    # exceeds 2^31-1 (e.g., T=20000, H=16, MAX_CANDIDATES=8192), int32 offset math overflows and
    # can produce negative pointers, causing illegal memory access. Promote to int64 explicitly.
    buffer_base = (
        i_b.to(tl.int64) * T_i64 * H * MAX_CANDIDATES
        + i_t.to(tl.int64) * H * MAX_CANDIDATES
        + i_h.to(tl.int64) * MAX_CANDIDATES
    )
    
    # 3.1 初始化 Streaming Top-K 容器 (encoded scores)
    # 我们维护一个大小为 TOP_K 的已排序(encoded)数组
    # 初始化为极小值 (encoded padding)
    running_topk_encoded = tl.full([TOP_K], -1e10, dtype=tl.float32)
    
    # Bit-Packing 常量
    LOG_N: tl.constexpr = 13  # log2(8192) = 13
    idx_mask = (1 << LOG_N) - 1  # 0x1FFF
    
    for i_batch in range(num_batches):
        # 加载 32 个父节点索引
        o_parent_local = tl.arange(0, PARENTS_PER_BATCH)
        parent_indices = tl.load(
            prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + o_parent_local,
            mask=o_parent_local < PARENTS_PER_BATCH,
            other=-1
        )  # [32]
        
        valid_parent_mask = parent_indices >= 0
        child_starts = parent_indices * COMPRESSION_RATE
        child_starts = tl.where(valid_parent_mask, child_starts, 0)
        
        # 判断最右侧父节点
        is_rightmost_parent = (child_starts // COMPRESSION_RATE == rightmost_idx // COMPRESSION_RATE)
        num_valid_children_per_parent = tl.where(
            is_rightmost_parent,
            (rightmost_idx % COMPRESSION_RATE) + 1,
            COMPRESSION_RATE
        )
        
        # 构建 2D mask [32, 16]
        o_child_offset = tl.arange(0, COMPRESSION_RATE)[None, :]
        valid_child_mask_2d = (
            valid_parent_mask[:, None] &
            (o_child_offset < num_valid_children_per_parent[:, None])
        )
        
        # rightmost mask（用于强制选中最右节点）
        rightmost_mask_2d = (
            (parent_indices[:, None] == rightmost_parent_idx)
            & (o_child_offset == rightmost_child_idx)
            & valid_child_mask_2d
        )

        # 初始化当前批次的结果
        scores_2d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], -1e10, dtype=tl.float32)
        
        # 遍历 32 个父节点
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                rope_pos_start = i_batch * BC + i_p * COMPRESSION_RATE

                # 当前父节点的有效子节点数
                is_rightmost_parent_scalar = parent_idx == rightmost_parent_idx
                num_valid_children = tl.where(is_rightmost_parent_scalar, rightmost_child_idx + 1, COMPRESSION_RATE)
                
                # 加载 K 并应用 RoPE (使用 GQA 映射的 i_h_kv)
                k_rope_1, k_rope_2 = load_k_with_rope_v2(
                    layer_k, cos_cache, sin_cache,
                    i_b, i_h_kv, child_start, rope_pos_start,
                    N_layer, H_kv, K, COMPRESSION_RATE, num_valid_children
                )
                
                # 计算 scores
                scores_16 = tl.sum(q_rope_1[None, :] * k_rope_1, axis=1) + \
                           tl.sum(q_rope_2[None, :] * k_rope_2, axis=1)
                
                # 填充到 2D 结果
                is_current_parent = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)
                scores_2d = tl.where(
                    is_current_parent & valid_child_mask_2d,
                    scores_16[None, :],
                    scores_2d
                )
        
        # Flatten 到 [BC=512]
        batch_scores = tl.reshape(scores_2d, [BC])
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        rightmost_mask_flat = tl.reshape(rightmost_mask_2d, [BC])
        
        # 应用 mask
        batch_scores = tl.where(valid_child_mask_flat, batch_scores, -1e10)
        
        # 3.2 存储到 8192 buffer (供 Kernel 2.2 使用)
        buffer_offset = i_batch * BC
        o_bc = tl.arange(0, BC)
        store_mask = (buffer_offset + o_bc) < MAX_CANDIDATES
        
        tl.store(all_scores + buffer_base + buffer_offset + o_bc, batch_scores, mask=store_mask)
        
        # 3.3 Streaming Top-K Update (非底层)
        if not is_bottom_layer:
            # 3.3.1 预处理当前批次分数
            # 右侧节点强制选中
            batch_scores_k = tl.where(rightmost_mask_flat, 1e3, batch_scores)
            
            # Padding for invalid nodes
            batch_scores_k = tl.where(valid_child_mask_flat, batch_scores_k, -1e10)
            
            # 3.3.2 Bit-Packing 当前批次
            # 计算当前批次在 buffer 中的位置索引 (0..8191)
            batch_buffer_indices = buffer_offset + o_bc  # [512]
            
            batch_scores_int = batch_scores_k.to(tl.int32, bitcast=True)
            encoded_idx = tl.where(batch_scores_k >= 0, ~batch_buffer_indices, batch_buffer_indices)
            encoded_idx = encoded_idx & idx_mask
            
            batch_scores_encoded_int = (batch_scores_int & ~idx_mask) | encoded_idx
            batch_scores_encoded = batch_scores_encoded_int.to(tl.float32, bitcast=True)
            
            # 3.3.3 对当前批次进行排序 (Stable Top-K on 512)
            n_dims_batch: tl.constexpr = 9  # log2(512) = 9
            sorted_batch = sort_single(
                batch_scores_encoded, n_dims_batch, 1, descending=True
            )
            
            # 3.3.4 与累积的 Top-K 合并
            if i_batch == 0:
                running_topk_encoded = sorted_batch
            else:
                # Merge 两个有序数组 (512 + 512 -> 1024)
                # 构造 [2, 512]
                running_b = tl.broadcast_to(running_topk_encoded[None, :], [2, BC])
                batch_b = tl.broadcast_to(sorted_batch[None, :], [2, BC])
                
                row_idx = tl.arange(0, 2)[:, None]
                merged_2d = tl.where(row_idx == 0, running_b, batch_b)
                
                # Flatten to [1024]
                merged_input = tl.reshape(merged_2d, [2 * BC])
                
                # 对 [1024] 进行排序
                n_dims_merge: tl.constexpr = 10  # log2(1024) = 10
                sorted_merged = sort_single(
                    merged_input, n_dims_merge, 1, descending=True
                )
                
                # 截取前 512: reshape to [2, 512] and take first row using sum+mask
                reshaped_merged = tl.reshape(sorted_merged, [2, BC])
                mask_row0 = (tl.arange(0, 2)[:, None] == 0)
                running_topk_encoded = tl.sum(tl.where(mask_row0, reshaped_merged, 0.0), axis=0)

    # ========================================
    # 阶段 4: Top-K 提取与存储（非底层）
    # ========================================
    
    if not is_bottom_layer:
        # running_topk_encoded 包含了全局 Top-K 的编码分数
        
        # 4.1 Bit-Packing 解码
        sorted_int = running_topk_encoded.to(tl.int32, bitcast=True)
        raw_idx = sorted_int & idx_mask
        clean_int = sorted_int & ~idx_mask
        clean_scores = clean_int.to(tl.float32, bitcast=True)
        
        # 还原索引
        topk_buffer_positions = tl.where(clean_scores >= 0, ~raw_idx, raw_idx)
        topk_buffer_positions = (topk_buffer_positions & idx_mask).to(tl.int32)
        
        # 过滤 Padding 值 (score <= -0.9e10)
        is_valid_score = clean_scores > -0.9e10
        topk_buffer_positions = tl.where(is_valid_score, topk_buffer_positions, -1)
        
        # 4.3 存储结果
        topk_offset = (
            i_b.to(tl.int64) * T_i64 * H * TOP_K
            + i_t.to(tl.int64) * H * TOP_K
            + i_h.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        
        tl.store(topk_positions + topk_offset + o_topk, topk_buffer_positions)
    
    else:
        # 底层：跳过排序，输出 -1
        topk_offset = (
            i_b.to(tl.int64) * T_i64 * H * TOP_K
            + i_t.to(tl.int64) * H * TOP_K
            + i_h.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        
        tl.store(topk_positions + topk_offset + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))


# ==========================================
# Post Kernel: Compute Next-Layer Parents
# ==========================================

@triton.jit
def htree_compute_next_parents_kernel(
    prev_selected_parents,  # [B, T, H, TOP_K]
    topk_positions,  # [B, T, H, TOP_K]  (buffer positions from Kernel 2.1)
    next_selected_parents,  # [B, T, H, TOP_K]
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Compute next layer's prev_selected_parents from (prev_selected_parents, topk_positions).

    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    BC: tl.constexpr = TOP_K  # 512
    PARENTS_PER_BATCH: tl.constexpr = 32

    T_i64 = T.to(tl.int64)
    base = (
        i_b.to(tl.int64) * T_i64 * H * TOP_K
        + i_t.to(tl.int64) * H * TOP_K
        + i_h.to(tl.int64) * TOP_K
    )

    o_topk = tl.arange(0, TOP_K)
    pos_i32 = tl.load(topk_positions + base + o_topk)
    gather_mask = pos_i32 >= 0

    pos_i64 = pos_i32.to(tl.int64)
    batch_id = (pos_i64 // BC).to(tl.int64)
    within_batch = (pos_i64 - batch_id * BC).to(tl.int64)
    parent_slot = (within_batch // COMPRESSION_RATE).to(tl.int64)
    child_slot = (within_batch - parent_slot * COMPRESSION_RATE).to(tl.int64)

    parent_idx = tl.load(
        prev_selected_parents + base + batch_id * PARENTS_PER_BATCH + parent_slot,
        mask=gather_mask,
        other=-1,
    )
    selected_node_indices = tl.where(
        gather_mask & (parent_idx >= 0),
        (parent_idx.to(tl.int64) * COMPRESSION_RATE + child_slot).to(tl.int32),
        -1,
    )

    # Sort ascending for next layer
    MAX_IDX: tl.constexpr = 2147483647
    selected_sorted = tl.where(selected_node_indices >= 0, selected_node_indices, MAX_IDX)

    n_dims_topk: tl.constexpr = 9  # log2(512)
    dummy = tl.arange(0, TOP_K).to(tl.int32)
    selected_sorted, _ = argsort_v2(selected_sorted, dummy, n_dims_topk, 1, descending=False)
    selected_sorted = tl.where(selected_sorted < MAX_IDX, selected_sorted, -1)

    tl.store(next_selected_parents + base + o_topk, selected_sorted.to(tl.int32))


# ==========================================
# Kernel 2.2: Accumulate Non-TopK (批次加载 V)
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=4, num_stages=2),
#         triton.Config({}, num_warps=8, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=3),
#         triton.Config({}, num_warps=8, num_stages=3),
#     ],
#     key=['B', 'H', 'K']
# )
@triton.jit
def htree_accumulate_non_topk_kernel(
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H, TOP_K]
    # Kernel 2.1 的输出
    all_scores,  # [B, T, H, MAX_CANDIDATES]
    num_candidates,  # [B, T, H]
    topk_positions,  # [B, T, H, TOP_K]
    # 输出
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    # 参数
    layer_idx: tl.constexpr,
    layer_power: tl.constexpr,
    B: tl.constexpr, T, H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
):
    """
    Kernel 2.2: 流式加载 V，累积非 Top-K 节点（Stream Accumulate）
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    # GQA: 映射 Query 头索引到 KV 头索引
    i_h_kv = i_h // NUM_GROUPS
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    BC: tl.constexpr = TOP_K  # 512
    PARENTS_PER_BATCH: tl.constexpr = 32
    
    rightmost_idx = i_t // layer_power
    
    # ========================================
    # 阶段 1: 加载元数据
    # ========================================
    
    T_i64 = T.to(tl.int64)
    num_cand_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    n_cand = tl.load(num_candidates + num_cand_offset)
    
    topk_offset = (
        i_b.to(tl.int64) * T_i64 * H * TOP_K
        + i_t.to(tl.int64) * H * TOP_K
        + i_h.to(tl.int64) * TOP_K
    )
    o_topk = tl.arange(0, TOP_K)
    topk_pos_vals = tl.load(topk_positions + topk_offset + o_topk) # topk_pos_vals [TOP_K] 内容为 buffer 中的下标位置（乱序）
    
    # Same overflow issue as Kernel 2.1: promote linear indexing to int64.
    buffer_base = (
        i_b.to(tl.int64) * T_i64 * H * MAX_CANDIDATES
        + i_t.to(tl.int64) * H * MAX_CANDIDATES
        + i_h.to(tl.int64) * MAX_CANDIDATES
    )
    
    # ========================================
    # 阶段 3: 流式加载 V 并累积（Stream Accumulate）
    # ========================================
    
    # 逐父节点在线累计，避免物化全量 values_2d [32, 16, V]
    cur_max = tl.full((), -1e10, dtype=tl.float32)
    cur_sum = tl.zeros((), dtype=tl.float32)
    cur_output = tl.zeros([V], dtype=tl.float32)
    
    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H * TOP_K
        + i_t.to(tl.int64) * H * TOP_K
        + i_h.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    valid_parent_list_mask = parent_list >= 0
    num_valid_parents = tl.sum(valid_parent_list_mask.to(tl.int32))
    num_batches = (num_valid_parents + 31) // 32

    for i_batch in range(num_batches):
        # ========================================
        # Stream Accumulate: 逐父节点流式累积
        # ========================================
        # 遍历当前批次的 32 个父节点，逐个处理
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            # 加载当前父节点索引
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            # 检查父节点有效性
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                
                # 加载当前父节点的 V values [16, V] (使用 GQA 映射的 i_h_kv)
                v_vals = load_v_v2(
                    layer_v, i_b, i_h_kv, child_start,
                    N_layer, H_kv, V, COMPRESSION_RATE
                )
                
                # 计算当前父节点的有效子节点数
                is_rightmost = (parent_idx == rightmost_idx // COMPRESSION_RATE)
                num_valid_children = tl.where(
                    is_rightmost,
                    (rightmost_idx % COMPRESSION_RATE) + 1,
                    COMPRESSION_RATE
                )
                
                # 构建子节点有效性 mask [16]
                o_child = tl.arange(0, COMPRESSION_RATE)
                child_valid_mask = o_child < num_valid_children
                
                # 计算全局位置（在 buffer 中的位置）
                global_pos_base = i_batch * BC + i_p * COMPRESSION_RATE
                global_pos = global_pos_base + o_child
                
                # 候选有效性（不超过实际候选数且子节点有效）
                child_candidate_mask = child_valid_mask & (global_pos < n_cand)
                
                # 非底层则剔除 top-k
                # TODO one of the bottlenecks
                if not is_bottom_layer:
                    # 检测当前子节点位置是否命中 Top-K
                    topk_hit = tl.sum(global_pos[:, None] == topk_pos_vals[None, :], axis=1).to(tl.int1)
                    final_child_mask = child_candidate_mask & (~topk_hit)
                else:
                    final_child_mask = child_candidate_mask
                
                # 加载当前子节点的 scores [16]
                child_scores = tl.load(
                    all_scores + buffer_base + global_pos,
                    mask=o_child < COMPRESSION_RATE,
                    other=-1e10
                )
                
                # 应用 mask
                masked_scores = tl.where(final_child_mask, child_scores, -1e10)
                masked_values = tl.where(final_child_mask[:, None], v_vals, 0.0)
                
                # 计算当前父节点的 softmax 分子
                parent_max = tl.max(masked_scores)
                exp_scores = tl.exp(masked_scores - parent_max)
                exp_scores = tl.where(final_child_mask, exp_scores, 0.0)
                parent_sum = tl.sum(exp_scores)
                parent_out = tl.sum(exp_scores[:, None] * masked_values, axis=0)
                
                # 与累计状态合并（online softmax）
                has_contribution = parent_sum > 0
                if has_contribution:
                    new_max = tl.maximum(cur_max, parent_max)
                    scale_cur = tl.where(cur_sum > 0, tl.exp(cur_max - new_max), 0.0)
                    scale_parent = tl.exp(parent_max - new_max)
                    
                    cur_sum = cur_sum * scale_cur + parent_sum * scale_parent
                    cur_output = cur_output * scale_cur + parent_out * scale_parent
                    cur_max = new_max

    # ========================================
    # 阶段 4: 存储结果
    # ========================================
    
    state_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    tl.store(layer_max + state_offset, cur_max)
    tl.store(layer_sum + state_offset, cur_sum)
    
    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    tl.store(layer_output + output_offset + o_v, cur_output, mask=o_v < V)


# ==========================================
# Kernel 2.3 & Final Normalization
# ==========================================

@triton.jit
def htree_merge_to_global_kernel_v2(
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
    """Kernel 2.3: 全局状态合并, Grid: (T, B*H)"""
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
def htree_final_normalize_kernel_v2(
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


# ==========================================
# 排序辅助函数 (Bit-Packing Top-K)
# ==========================================

# -------------------- 单数组排序 (用于 Bit-Packing) --------------------

@triton.jit
def _compare_and_swap_single(
    x,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """单数组 compare-and-swap，不追踪索引（索引已编码在值的低位）"""
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
    """单数组 bitonic sort（不返回索引，用于 Bit-Packing Top-K）"""
    for i in tl.static_range(1, n_dims + 1):
        x = _bitonic_merge_single(x, i, 2 if i < n_dims else descending, n_dims, n_outer)
    return x


# -------------------- 双数组排序 (用于 parents 升序排序) --------------------

@triton.jit
def _compare_and_swap_v2(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge_v2(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap_v2(x, ids, flip, i + (n_dims - stage), n_dims, n_outer)
    return x, ids


@triton.jit
def argsort_v2(
    x,
    ids,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
    descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge_v2(x, ids, i, 2 if i < n_dims else descending, n_dims, n_outer)
    return x, ids


# ==========================================
# Main Forward Function
# ==========================================

def htree_forward_v2(
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
    htree 前向传播 V2 (支持 Group Query Attention)
    
    Args:
        q: [B, T, H, K] - Query，H 个头
        k: [B, T, H_kv, K] - Key，H_kv 个头 (H_kv <= H, H % H_kv == 0)
        v: [B, T, H_kv, V] - Value，H_kv 个头
        compression_rate: 16
        max_top_nodes: 8192
        top_k_per_layer: 512
        scale: K^-0.5
        rope_base: 10000.0
    
    Returns:
        output: [B, T, H, V]
    
    Note:
        - 当 H_kv == H 时，等价于 Multi-Head Attention (MHA)
        - 当 H_kv == 1 时，等价于 Multi-Query Attention (MQA)
        - 当 1 < H_kv < H 时，为 Group Query Attention (GQA)
    """
    B, T, H, K = q.shape
    H_kv = k.shape[2]  # KV 头数量
    V = v.shape[-1]
    
    # 验证 GQA 配置
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
    assert k.shape[2] == v.shape[2], f"K and V must have same number of heads"
    
    if scale is None:
        scale = K ** -0.5
    
    device = q.device
    dtype = q.dtype
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # ========== Phase 1: Build Tree ==========
    nvtx.range_push("Phase1_TreeBuilding")
    print("Phase 1: Building tree structure...")
    
    num_layers = 1
    temp_len = T
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    
    print(f"  Tree has {num_layers} layers (H={H}, H_kv={H_kv}, num_groups={H // H_kv})")
    
    layers_k = [k]
    layers_v = [v]
    
    current_k, current_v = k, v
    current_len = T
    
    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H_kv, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H_kv, V, dtype=dtype, device=device)
        
        BLOCK_SIZE = 128
        grid = (B * H_kv,)
        
        torch.cuda.synchronize()
        start_build = torch.cuda.Event(enable_timing=True)
        end_build = torch.cuda.Event(enable_timing=True)
        start_build.record()

        htree_build_kernel_v2[grid](
            current_k, current_v, next_k, next_v,
            N_child=current_len,
            N_parent=next_len,
            B=B, H_kv=H_kv, K=K, V=V,
            COMPRESSION_RATE=compression_rate,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        end_build.record()
        torch.cuda.synchronize()
        time_build = start_build.elapsed_time(end_build)
        
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len
        
        print(f"  Built layer {layer_idx}: {current_len} nodes, time: {time_build:.2f} ms")
    nvtx.range_pop()

    # ========== Phase 1.5: Precompute RoPE cache ==========
    nvtx.range_push("Phase1.5_RoPE_Cache")
    print("Phase 1.5: Precomputing RoPE cache...")
    
    cache_size = max_top_nodes + 1024
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
    positions = torch.arange(cache_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    nvtx.range_pop()

    # ========== Phase 2: Initialize Global State & Buffers ==========
    nvtx.range_push("Phase2_Init_GlobalState_and_Buffers")
    print("Phase 2: Initializing global states and buffers...")
    
    global_max = torch.full([B, T, H], -1e10, dtype=torch.float32, device=device)
    global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
    global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)
    
    nvtx.range_pop()

    # ========== Phase 3: Layer-by-layer Forward ==========
    nvtx.range_push("Phase3_LayerByLayer_Forward")
    print("Phase 3: Layer-by-layer forward pass...")
    
    layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)

    # per-layer reusable workspaces (aligned with parallel.py)
    MAX_CANDIDATES = 8192
    all_scores = torch.empty([B, T, H, MAX_CANDIDATES], dtype=torch.float32, device=device)
    num_candidates = torch.empty([B, T, H], dtype=torch.int32, device=device)
    topk_positions = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    
     # 为最顶层初始化 prev_selected_parents
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)  # [T]
    rightmost_indices = t_indices // top_layer_power  # [T] 顶层最右侧节点索引
    # 虚拟父节点数量：每个虚拟父节点展开成16个顶层节点
    num_virtual_parents = rightmost_indices // compression_rate + 1  # [T]
    
    # [T, TOP_K]
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
    prev_selected_parents = prev_selected_parents.unsqueeze(0).unsqueeze(2).expand(B, T, H, top_k_per_layer).contiguous()
    
    for layer_idx in range(num_layers - 1, -1, -1):
        nvtx.range_push(f"Forward_Layer_{layer_idx}")
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        
        is_bottom_layer = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        
        print(f"  -> Processing layer {layer_idx} (N={N_layer}, power={layer_power}, bottom={is_bottom_layer})...")
        
        grid = (T, B * H)
        
        # Kernel 2.1: Compute Scores & Select Top-K
        nvtx.range_push("K2.1_ComputeScoresAndSelect")
        print("    Running compute scores and select kernel...")
        
        torch.cuda.synchronize()
        start_k21 = torch.cuda.Event(enable_timing=True)
        end_k21 = torch.cuda.Event(enable_timing=True)
        start_k21.record()
        
        htree_compute_scores_and_select_kernel[grid](
            q, k_layer,
            prev_selected_parents,
            cos_cache, sin_cache,
            all_scores, num_candidates,
            topk_positions,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, H_kv=H_kv,
            NUM_GROUPS=H // H_kv,
            K=K, V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
            scale=scale,
        )
        end_k21.record()
        torch.cuda.synchronize()
        time_k21 = start_k21.elapsed_time(end_k21)
        print(f"      Kernel 2.1 time: {time_k21:.2f} ms")
        
        nvtx.range_pop()
        
        # Kernel 2.2: Accumulate Non-TopK
        nvtx.range_push("K2.2_AccumulateNonTopK")
        print("    Running accumulate non-topk kernel...")
        
        torch.cuda.synchronize()
        start_k22 = torch.cuda.Event(enable_timing=True)
        end_k22 = torch.cuda.Event(enable_timing=True)
        start_k22.record()
        
        htree_accumulate_non_topk_kernel[grid](
            v_layer,
            prev_selected_parents,
            all_scores, num_candidates,
            topk_positions,
            layer_max, layer_sum, layer_output,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, H_kv=H_kv,
            NUM_GROUPS=H // H_kv,
            V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
        )
        end_k22.record()
        torch.cuda.synchronize()
        time_k22 = start_k22.elapsed_time(end_k22)
        print(f"      Kernel 2.2 time: {time_k22:.2f} ms")
        
        nvtx.range_pop()
        
        # Kernel 2.3: Merge to Global State
        nvtx.range_push("K2.3_MergeToGlobal")
        print("    Running merge to global kernel...")
        
        torch.cuda.synchronize()
        start_k23 = torch.cuda.Event(enable_timing=True)
        end_k23 = torch.cuda.Event(enable_timing=True)
        start_k23.record()
        
        htree_merge_to_global_kernel_v2[grid](
            layer_max, layer_sum, layer_output,
            global_max, global_sum, global_output,
            B=B, T=T, H=H, V=V,
        )
        end_k23.record()
        torch.cuda.synchronize()
        time_k23 = start_k23.elapsed_time(end_k23)
        print(f"      Kernel 2.3 time: {time_k23:.2f} ms")
        
        nvtx.range_pop()

        # 更新 parent indices（用当前层 prev_selected_parents + topk_positions 计算下一层 parents）
        if not is_bottom_layer:
            nvtx.range_push("Post_ComputeNextParents")

            print("    Running compute next parents kernel...")
            torch.cuda.synchronize()
            start_k24 = torch.cuda.Event(enable_timing=True)
            end_k24 = torch.cuda.Event(enable_timing=True)
            start_k24.record()

            htree_compute_next_parents_kernel[grid](
                prev_selected_parents,
                topk_positions,
                topk_positions,
                B=B, T=T, H=H,
                COMPRESSION_RATE=compression_rate,
                TOP_K=top_k_per_layer,
            )

            end_k24.record()
            torch.cuda.synchronize()
            time_k24 = start_k24.elapsed_time(end_k24)
            print(f"      Kernel 2.4 time: {time_k24:.2f} ms")

            nvtx.range_pop()

            # Swap buffers: topk_positions now holds next prev_selected_parents.
            # Reuse old prev_selected_parents buffer as the next iteration's topk_positions output.
            prev_selected_parents, topk_positions = topk_positions, prev_selected_parents
        
        nvtx.range_pop()
    nvtx.range_pop()

    # ========== Phase 4: Final Normalization ==========
    nvtx.range_push("Phase4_Final_Normalize")
    print("Phase 4: Final normalization...")
    
    output = torch.empty(B, T, H, V, dtype=dtype, device=device)
    grid = (T, B * H)
    
    torch.cuda.synchronize()
    start_k4 = torch.cuda.Event(enable_timing=True)
    end_k4 = torch.cuda.Event(enable_timing=True)
    start_k4.record()
    
    htree_final_normalize_kernel_v2[grid](
        global_output, global_sum, output,
        B=B, T=T, H=H, V=V,
    )
    end_k4.record()
    torch.cuda.synchronize()
    time_k4 = start_k4.elapsed_time(end_k4)
    print(f"  Final Normalization Kernel time: {time_k4:.2f} ms")
    
    nvtx.range_pop()

    print("htree forward pass V2 completed!")
    
    return output


__all__ = [
    'htree_forward_v2',
    'htree_build_kernel_v2',
    'htree_compute_scores_and_select_kernel',
    'htree_accumulate_non_topk_kernel',
    'htree_merge_to_global_kernel_v2',
    'htree_final_normalize_kernel_v2',
]
