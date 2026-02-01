# -*- coding: utf-8 -*-
"""
htree Triton Kernel - IO Only Version

This version preserves all memory IO operations but simplifies computation:
- Attention score computation: simplified to random-like patterns
- TopK selection: simplified to pseudo-random selection (not perfectly sequential)
- Goal: measure IO bandwidth upper bound without complex computation overhead
"""

from typing import Optional
import math
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# ==========================================
# 全局常量 (数值约定)
# ==========================================

HTREE_SCORE_NEG_INF: float = -1.0e10
HTREE_SCORE_VALID_THRESHOLD: float = HTREE_SCORE_NEG_INF * 0.9

# ==========================================
# Kernel 1: Tree Building (保留完整 IO)
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
# 辅助函数：简化的 K 加载（移除 RoPE）
# ==========================================

@triton.jit
def load_k_simple(
    layer_k,
    i_b,
    i_h_kv,
    child_start,
    N_layer: tl.constexpr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    num_valid_children,
):
    """加载 K 但不应用 RoPE（简化版本）"""
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
    
    return k1, k2


# ==========================================
# Kernel 2.1a: Compute Scores (简化计算)
# ==========================================

@triton.jit
def htree_compute_scores_kernel_io_only(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    prev_selected_parents,  # [B, T_blocks, H_kv, TOP_K]
    # 输出 buffer
    all_scores,  # [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    num_candidates,  # [B, T, H_kv]
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
    SCORE_VALID_THRESHOLD: tl.constexpr,
    scale,
):
    """
    Kernel 2.1 (scores only, IO版本): 加载 Q 和 K，计算简化的 scores
    """
    i_t = tl.program_id(0)
    i_bhk = tl.program_id(1)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv
    
    PARENTS_PER_BATCH: tl.constexpr = 32
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    
    # ========================================
    # 阶段 1: 确定候选节点范围
    # ========================================
    
    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE
    
    T_i64 = T.to(tl.int64)
    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    o_parent = tl.arange(0, TOP_K)
    parent_list = tl.load(prev_selected_parents + prev_sel_base + o_parent)
    
    valid_mask = parent_list >= 0
    num_valid_parents = tl.sum(valid_mask.to(tl.int32))
    num_batches = (num_valid_parents + 31) // 32
    
    # 计算实际候选节点数量
    n_cand_pos = (num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1
    n_cand = tl.where(num_valid_parents > 0, n_cand_pos, 0).to(tl.int32)
    
    # 存储 num_candidates
    num_cand_offset = (
        i_b.to(tl.int64) * T_i64 * H_kv
        + i_t.to(tl.int64) * H_kv
        + i_h_kv.to(tl.int64)
    )
    tl.store(num_candidates + num_cand_offset, n_cand)
    
    # ========================================
    # 阶段 2: 加载 Query（简化：不应用 RoPE）
    # ========================================
    
    head_ids = (i_h_kv * NUM_GROUPS + tl.arange(0, NUM_GROUPS)).to(tl.int64)  # [G]
    q_bt_base = (
        i_b.to(tl.int64) * T_i64 * H * K
        + i_t.to(tl.int64) * H * K
    )
    o_k = tl.arange(0, K // 2).to(tl.int64)[None, :]  # [1, K/2]
    q_head_base = q + q_bt_base + head_ids[:, None] * K  # [G, 1] pointer
    q1 = tl.load(q_head_base + o_k)  # [G, K/2]
    q2 = tl.load(q_head_base + (K // 2) + o_k)  # [G, K/2]
    
    # 简化：直接使用 q1, q2，乘以 scale
    q_scaled_1 = q1 * scale
    q_scaled_2 = q2 * scale
    
    # ========================================
    # 阶段 3: 批次遍历, 加载 K 并计算简化的 scores
    # ========================================

    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )
    
    for i_batch in range(num_batches):
        # 加载 32 个父节点索引
        o_parent_local = tl.arange(0, PARENTS_PER_BATCH)
        idx_in_topk = (i_batch * PARENTS_PER_BATCH + o_parent_local).to(tl.int32)
        parent_indices = tl.load(
            prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + o_parent_local,
            mask=idx_in_topk < TOP_K,
            other=-1
        )  # [32]
        
        valid_parent_mask = parent_indices >= 0
        child_starts = tl.where(valid_parent_mask, parent_indices * COMPRESSION_RATE, 0)
        
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

        # 初始化当前批次的结果: [parents, children, groups]
        scores_3d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE, NUM_GROUPS], -1e10, dtype=tl.float32)
        
        # 遍历 32 个父节点
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_off = (i_batch * PARENTS_PER_BATCH + i_p).to(tl.int32)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + parent_off.to(tl.int64),
                mask=parent_off < TOP_K,
                other=-1,
            )
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE

                # 当前父节点的有效子节点数
                is_rightmost_parent_scalar = parent_idx == rightmost_parent_idx
                num_valid_children = tl.where(is_rightmost_parent_scalar, rightmost_child_idx + 1, COMPRESSION_RATE)
                
                # 加载 K（简化版本：不应用 RoPE）
                k1, k2 = load_k_simple(
                    layer_k, i_b, i_h_kv, child_start,
                    N_layer, H_kv, K, COMPRESSION_RATE, num_valid_children
                )
                
                # 简化的 score 计算：使用伪随机模式
                # 目标：不让访问模式太规则，但计算简单
                # 方法：用位置信息和简单的哈希函数生成伪随机分数
                child_positions = child_start + tl.arange(0, COMPRESSION_RATE)
                # 简单哈希：(pos * 1103515245 + 12345) % 2^31
                hash_vals = (child_positions * 1103515245 + 12345 + i_h_kv * 7 + i_t * 13) & 0x7FFFFFFF
                # 归一化到 [0, 1] 范围
                pseudo_rand = (hash_vals.to(tl.float32) / 2147483647.0)  # [CR]
                
                # 叠加一些 K 值的简单特征（保持 IO）
                k_feature = tl.sum(k1, axis=1) + tl.sum(k2, axis=1)  # [CR]
                k_feature_scaled = k_feature * 0.01  # 弱信号
                
                # 组合：伪随机为主，K特征为辅
                base_scores = pseudo_rand * 10.0 - 5.0 + k_feature_scaled  # [CR]
                
                # 广播到所有 query heads: [CR, G]
                scores_crg = tl.broadcast_to(base_scores[:, None], [COMPRESSION_RATE, NUM_GROUPS])
                
                # 添加每个 head 的微小偏移（使用 head_id 作为扰动）
                head_offsets = tl.arange(0, NUM_GROUPS).to(tl.float32) * 0.001
                scores_crg = scores_crg + head_offsets[None, :]
                
                # 填充到 3D 结果
                is_current_parent = (tl.arange(0, PARENTS_PER_BATCH) == i_p)[:, None, None]  # [P,1,1]
                scores_3d = tl.where(
                    is_current_parent & valid_child_mask_2d[:, :, None],
                    scores_crg[None, :, :],
                    scores_3d
                )
        
        # Flatten 到 [BC=512, G]
        batch_scores = tl.reshape(scores_3d, [BC, NUM_GROUPS])
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        
        # 应用 mask
        batch_scores = tl.where(valid_child_mask_flat[:, None], batch_scores, -1e10)
        
        # 存储到 buffer
        buffer_offset = i_batch * BC
        o_bc = tl.arange(0, BC)
        pos = buffer_offset + o_bc
        store_mask = pos < MAX_CANDIDATES
        
        pos_i64 = pos.to(tl.int64)
        g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)
        ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
        tl.store(ptrs, batch_scores, mask=store_mask[:, None])


# ==========================================
# Kernel 2.1b: TopK Selection (简化为伪随机选择)
# ==========================================

@triton.jit
def htree_select_topk_simple_kernel(
    all_scores,  # [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    num_candidates,  # [B, T, H_kv]
    topk_positions,  # [B, T, H_kv, TOP_K] (buffer positions)
    layer_idx: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """
    简化版 TopK 选择：使用伪随机但不完全连续的选择策略
    - 强制选中 rightmost
    - 其他位置用伪随机选择（基于位置哈希）
    """
    i_t = tl.program_id(0)
    i_bhk = tl.program_id(1)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv

    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    if is_bottom_layer:
        # Bottom layer: no need to select for next layer
        T_i64 = T.to(tl.int64)
        base_out = (
            i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
            + i_t.to(tl.int64) * H_kv * TOP_K
            + i_h_kv.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        tl.store(topk_positions + base_out + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))
        return

    # Load n_cand
    T_i64 = T.to(tl.int64)
    num_cand_off = (
        i_b.to(tl.int64) * T_i64 * H_kv
        + i_t.to(tl.int64) * H_kv
        + i_h_kv.to(tl.int64)
    )
    n_cand = tl.load(num_candidates + num_cand_off).to(tl.int32)

    if n_cand <= 0:
        base_out = (
            i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
            + i_t.to(tl.int64) * H_kv * TOP_K
            + i_h_kv.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        tl.store(topk_positions + base_out + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))
        return

    # 简化选择策略：
    # 1. 位置 n_cand-1（rightmost）必选
    # 2. 其他位置用伪随机间隔选择
    
    rightmost_pos = (n_cand - 1).to(tl.int32)
    actual_k = tl.minimum(TOP_K, n_cand)
    
    o_k = tl.arange(0, TOP_K)
    
    # 伪随机选择：用哈希函数生成不太规则的索引
    # 使用 (k + i_t*7 + i_h_kv*13) 的哈希
    seed_base = i_t * 7 + i_h_kv * 13 + i_b * 31
    hash_vals = ((o_k + seed_base) * 1103515245 + 12345) & 0x7FFFFFFF
    # 映射到 [0, n_cand) 范围
    pseudo_positions = (hash_vals % n_cand).to(tl.int32)
    
    # 强制最后一个选 rightmost
    is_last_k = (o_k == (actual_k - 1))
    selected_pos = tl.where(is_last_k, rightmost_pos, pseudo_positions)
    
    # 超出 actual_k 的标记为 -1
    selected_pos = tl.where(o_k < actual_k, selected_pos, -1)
    
    # Store
    base_out = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(topk_positions + base_out + o_k.to(tl.int64), selected_pos.to(tl.int32))


# ==========================================
# Kernel 2.1.2: Mask Top-K Scores
# ==========================================

@triton.jit
def htree_mask_topk_scores_kernel(
    all_scores,  # [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    topk_positions,  # [B, T, H_kv, TOP_K]
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    """Mask Top-K scores by overwriting them with NEG_INF."""
    i_t = tl.program_id(0)
    i_bhk = tl.program_id(1)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv

    T_i64 = T.to(tl.int64)
    topk_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    o_topk = tl.arange(0, TOP_K)
    pos_i32 = tl.load(topk_positions + topk_base + o_topk)
    valid = (pos_i32 >= 0) & (pos_i32 < MAX_CANDIDATES)

    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )
    pos_i64 = pos_i32.to(tl.int64)
    g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)
    ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
    tl.store(ptrs, tl.full([TOP_K, NUM_GROUPS], NEG_INF, dtype=tl.float32), mask=valid[:, None])


# ==========================================
# Kernel 2.2: Accumulate (保留完整 IO)
# ==========================================

@triton.jit
def htree_accumulate_non_topk_kernel_io_only(
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    all_scores,  # [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    num_candidates,  # [B, T, H_kv]
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
    BV: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
):
    """
    Kernel 2.2 (IO版本): 加载 V 和 scores, 执行 online-softmax accumulate
    """
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bhk = tl.program_id(2)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    PARENTS_PER_BATCH: tl.constexpr = 32
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE

    rightmost_idx = i_t // layer_power

    # ========================================
    # 阶段 1: 加载元数据
    # ========================================

    T_i64 = T.to(tl.int64)
    num_cand_offset = (
        i_b.to(tl.int64) * T_i64 * H_kv
        + i_t.to(tl.int64) * H_kv
        + i_h_kv.to(tl.int64)
    )
    n_cand = tl.load(num_candidates + num_cand_offset).to(tl.int32)

    g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)  # [G]
    head_ids = (i_h_kv * NUM_GROUPS + g_ids).to(tl.int64)  # [G]

    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )

    v_start = (i_v * BV).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    v_cols = v_start + o_v  # [BV]
    v_mask = v_cols < V

    # ========================================
    # 阶段 2: 流式加载 V 并累积
    # ========================================

    cur_max = tl.full([NUM_GROUPS], -1e10, dtype=tl.float32)
    cur_sum = tl.zeros([NUM_GROUPS], dtype=tl.float32)
    cur_output = tl.zeros([NUM_GROUPS, BV], dtype=tl.float32)

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    valid_parent_list_mask = parent_list >= 0
    num_valid_parents = tl.sum(valid_parent_list_mask.to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    for i_batch in range(num_batches):
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_off = (i_batch * PARENTS_PER_BATCH + i_p).to(tl.int32)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + parent_off.to(tl.int64),
                mask=parent_off < TOP_K,
                other=-1,
            ).to(tl.int32)

            valid_parent = parent_idx >= 0

            child_start = (parent_idx * COMPRESSION_RATE).to(tl.int32)
            child_rows_i32 = child_start + tl.arange(0, COMPRESSION_RATE).to(tl.int32)  # [CR]
            row_mask = child_rows_i32 < N_layer

            # Load V slice [COMPRESSION_RATE, BV]
            v_base = (
                layer_v
                + i_b.to(tl.int64) * N_layer * H_kv * V
                + i_h_kv.to(tl.int64) * V
            )
            v_ptrs = v_base + child_rows_i32.to(tl.int64)[:, None] * (H_kv * V) + v_cols[None, :]
            v_vals = tl.load(
                v_ptrs,
                mask=valid_parent & row_mask[:, None] & v_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [CR, BV]

            is_rightmost = valid_parent & (parent_idx == (rightmost_idx // COMPRESSION_RATE))
            num_valid_children = tl.where(
                is_rightmost,
                (rightmost_idx % COMPRESSION_RATE) + 1,
                COMPRESSION_RATE,
            ).to(tl.int32)
            child_valid_mask = row_mask & (tl.arange(0, COMPRESSION_RATE) < num_valid_children)  # [CR]

            global_pos_base = i_batch * BC + i_p * COMPRESSION_RATE
            global_pos = global_pos_base.to(tl.int32) + tl.arange(0, COMPRESSION_RATE).to(tl.int32)  # [CR]
            child_candidate_mask = valid_parent & child_valid_mask & (global_pos < n_cand)  # [CR]

            # Load scores [CR, G]
            pos_i64 = global_pos.to(tl.int64)  # [CR]
            s_ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
            scores = tl.load(
                s_ptrs,
                mask=child_candidate_mask[:, None],
                other=-1e10,
            ).to(tl.float32)  # [CR, G]

            if not is_bottom_layer:
                score_valid = scores > SCORE_VALID_THRESHOLD
                final_mask = child_candidate_mask[:, None] & score_valid
            else:
                final_mask = child_candidate_mask[:, None]

            # Online softmax accumulation
            masked_for_max = tl.where(final_mask, scores, -1e10)
            batch_max = tl.max(masked_for_max, axis=0)  # [G]

            new_max = tl.maximum(cur_max, batch_max)  # [G]
            scale = tl.exp(cur_max - new_max)         # [G]

            cur_sum = cur_sum * scale
            cur_output = cur_output * scale[:, None]

            scores_for_exp = tl.where(final_mask, scores, -1e10)
            p = tl.exp(scores_for_exp - new_max[None, :])  # [CR, G]
            p = tl.where(final_mask, p, 0.0)
            dp = tl.sum(p[:, :, None] * v_vals[:, None, :], axis=0)  # [G, BV]

            cur_sum = cur_sum + tl.sum(p, axis=0)
            cur_output = cur_output + dp
            cur_max = new_max

    # ========================================
    # 阶段 3: 存储结果
    # ========================================

    if i_v == 0:
        state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
        tl.store(layer_max + state_base + head_ids, cur_max)
        tl.store(layer_sum + state_base + head_ids, cur_sum)

    out_base = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
    )
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + v_cols[None, :]
    tl.store(out_ptrs, cur_output, mask=v_mask[None, :])


# ==========================================
# Kernel 2.3 & Final Normalization (保留完整 IO)
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
    """全局状态合并, Grid: (T, B*H)"""
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
    Final normalization: output = global_output / global_sum
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    T_i64 = T.to(tl.int64)
    sum_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    sum_val = tl.load(global_sum + sum_offset)
    
    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    output_ptrs = global_output + output_offset + o_v
    output_vals = tl.load(output_ptrs, mask=o_v < V, other=0.0)
    
    normalized = output_vals / sum_val
    
    out_ptrs = output + output_offset + o_v
    tl.store(out_ptrs, normalized, mask=o_v < V)


# ==========================================
# Kernel 2.4: Compute Next Parents (简化)
# ==========================================

@triton.jit
def htree_compute_next_parents_simple_kernel(
    topk_positions,  # [B, T, H_kv, TOP_K] (in: buffer positions from topk kernel)
    next_selected_parents,  # [B, T, H_kv, TOP_K] (out: node indices for next layer)
    B: tl.constexpr,
    T,
    H_kv: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """
    简化版本：直接将 topk_positions 作为下一层的 parents（不做复杂映射）
    实际应用中，topk_positions 是 buffer 位置，需要映射回节点索引，
    但在 IO-only 版本中，我们简化这一步。
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H_kv
    i_h_kv = i_bh % H_kv

    T_i64 = T.to(tl.int64)
    base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )

    o_topk = tl.arange(0, TOP_K)
    # 直接读取并写入（简化版本）
    pos = tl.load(topk_positions + base + o_topk)
    tl.store(next_selected_parents + base + o_topk, pos)


# ==========================================
# Main Forward Function
# ==========================================

def htree_forward_io_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    htree 前向传播 - IO Only 版本
    
    保留所有 IO 操作，简化计算逻辑：
    - Score 计算简化为伪随机 + 简单特征
    - TopK 选择简化为伪随机选择
    
    Args:
        q: [B, T, H, K]
        k: [B, T, H_kv, K]
        v: [B, T, H_kv, V]
        compression_rate: 16
        max_top_nodes: 8192
        top_k_per_layer: 512
        scale: K^-0.5
    
    Returns:
        output: [B, T, H, V]
    """
    B, T, H, K = q.shape
    H_kv = k.shape[2]
    V = v.shape[-1]
    
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
    NUM_GROUPS = H // H_kv
    assert k.shape[2] == v.shape[2]
    assert (top_k_per_layer & (top_k_per_layer - 1)) == 0
    assert max_top_nodes == top_k_per_layer * compression_rate
    assert (compression_rate & (compression_rate - 1)) == 0
    assert (max_top_nodes & (max_top_nodes - 1)) == 0
    assert top_k_per_layer <= 32 * compression_rate
    
    if scale is None:
        scale = K ** -0.5
    
    device = q.device
    dtype = q.dtype
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # ========== Phase 1: Build Tree ==========
    nvtx.range_push("Phase1_TreeBuilding_IO")
    print("Phase 1: Building tree structure (IO only)...")
    
    num_layers = 1
    temp_len = T
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    
    print(f"  Tree has {num_layers} layers (H={H}, H_kv={H_kv}, num_groups={NUM_GROUPS})")
    
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

    # ========== Phase 2: Initialize Global State & Buffers ==========
    nvtx.range_push("Phase2_Init_GlobalState_and_Buffers_IO")
    print("Phase 2: Initializing global states and buffers...")
    
    global_max = torch.full([B, T, H], -1e10, dtype=torch.float32, device=device)
    global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
    global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)
    
    nvtx.range_pop()

    # ========== Phase 3: Layer-by-layer Forward ==========
    nvtx.range_push("Phase3_LayerByLayer_Forward_IO")
    print("Phase 3: Layer-by-layer forward pass (IO only)...")
    
    layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)

    MAX_CANDIDATES = max_top_nodes
    all_scores = torch.empty([B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS], dtype=torch.float32, device=device)
    num_candidates = torch.empty([B, T, H_kv], dtype=torch.int32, device=device)
    topk_positions = torch.empty([B, T, H_kv, top_k_per_layer], dtype=torch.int32, device=device)

    # 为最顶层初始化 prev_selected_parents
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
    prev_selected_parents = prev_selected_parents.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, top_k_per_layer).contiguous()
    
    for layer_idx in range(num_layers - 1, -1, -1):
        nvtx.range_push(f"Forward_Layer_{layer_idx}_IO")
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        
        is_bottom_layer = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        
        print(f"  -> Processing layer {layer_idx} (N={N_layer}, power={layer_power}, bottom={is_bottom_layer})...")
        
        # Kernel 2.1a: Compute Scores (IO only)
        nvtx.range_push("K2.1a_ComputeScores_IO")
        print("    Running compute scores kernel (IO only)...")
        
        torch.cuda.synchronize()
        start_k21 = torch.cuda.Event(enable_timing=True)
        end_k21 = torch.cuda.Event(enable_timing=True)
        start_k21.record()
        
        grid_kv = (T, B * H_kv)
        htree_compute_scores_kernel_io_only[grid_kv](
            q, k_layer,
            prev_selected_parents,
            all_scores, num_candidates,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, H_kv=H_kv,
            NUM_GROUPS=NUM_GROUPS,
            K=K, V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
            SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
            scale=scale,
        )
        end_k21.record()
        torch.cuda.synchronize()
        time_k21 = start_k21.elapsed_time(end_k21)
        print(f"      Kernel 2.1a time: {time_k21:.2f} ms")
        
        nvtx.range_pop()

        # Kernel 2.1b: Simple Top-K selection
        if not is_bottom_layer:
            nvtx.range_push("K2.1b_SelectTopKSimple_IO")
            print("    Running simple topk selection kernel...")
            torch.cuda.synchronize()
            start_k21b = torch.cuda.Event(enable_timing=True)
            end_k21b = torch.cuda.Event(enable_timing=True)
            start_k21b.record()
            grid_kv = (T, B * H_kv)
            htree_select_topk_simple_kernel[grid_kv](
                all_scores,
                num_candidates,
                topk_positions,
                layer_idx=layer_idx,
                B=B, T=T, H=H, H_kv=H_kv,
                NUM_GROUPS=NUM_GROUPS,
                TOP_K=top_k_per_layer,
                MAX_CANDIDATES=MAX_CANDIDATES,
                COMPRESSION_RATE=compression_rate,
            )
            end_k21b.record()
            torch.cuda.synchronize()
            time_k21b = start_k21b.elapsed_time(end_k21b)
            print(f"      Kernel 2.1b time: {time_k21b:.2f} ms")
            nvtx.range_pop()

        # Kernel 2.1.2: Mask Top-K scores
        if not is_bottom_layer:
            nvtx.range_push("K2.1.2_MaskTopKScores_IO")
            print("    Running mask topk scores kernel...")

            torch.cuda.synchronize()
            start_k212 = torch.cuda.Event(enable_timing=True)
            end_k212 = torch.cuda.Event(enable_timing=True)
            start_k212.record()

            grid_kv = (T, B * H_kv)
            htree_mask_topk_scores_kernel[grid_kv](
                all_scores,
                topk_positions,
                B=B, T=T, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS,
                TOP_K=top_k_per_layer,
                MAX_CANDIDATES=MAX_CANDIDATES,
                NEG_INF=HTREE_SCORE_NEG_INF,
            )

            end_k212.record()
            torch.cuda.synchronize()
            time_k212 = start_k212.elapsed_time(end_k212)
            print(f"      Kernel 2.1.2 time: {time_k212:.2f} ms")
            nvtx.range_pop()
        
        # Kernel 2.2: Accumulate (IO only)
        nvtx.range_push("K2.2_AccumulateNonTopK_IO")
        print("    Running accumulate kernel (IO only)...")
        
        torch.cuda.synchronize()
        start_k22 = torch.cuda.Event(enable_timing=True)
        end_k22 = torch.cuda.Event(enable_timing=True)
        start_k22.record()

        BV = min(128, max(16, triton.next_power_of_2(V)))
        grid_k22 = (T, triton.cdiv(V, BV), B * H_kv)

        htree_accumulate_non_topk_kernel_io_only[grid_k22](
            v_layer,
            prev_selected_parents,
            all_scores, num_candidates,
            layer_max, layer_sum, layer_output,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, H_kv=H_kv,
            NUM_GROUPS=NUM_GROUPS,
            V=V, BV=BV, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
            SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
        )
        end_k22.record()
        torch.cuda.synchronize()
        time_k22 = start_k22.elapsed_time(end_k22)
        print(f"      Kernel 2.2 time: {time_k22:.2f} ms")
        
        nvtx.range_pop()
        
        # Kernel 2.3: Merge to Global
        nvtx.range_push("K2.3_MergeToGlobal_IO")
        print("    Running merge to global kernel...")
        
        torch.cuda.synchronize()
        start_k23 = torch.cuda.Event(enable_timing=True)
        end_k23 = torch.cuda.Event(enable_timing=True)
        start_k23.record()
        
        grid = (T, B * H)
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

        # Compute next parents (简化版本)
        if not is_bottom_layer:
            nvtx.range_push("Post_ComputeNextParentsSimple_IO")

            print("    Running compute next parents kernel (simplified)...")
            torch.cuda.synchronize()
            start_k24 = torch.cuda.Event(enable_timing=True)
            end_k24 = torch.cuda.Event(enable_timing=True)
            start_k24.record()

            htree_compute_next_parents_simple_kernel[(T, B * H_kv)](
                topk_positions,
                topk_positions,
                B=B, T=T, H_kv=H_kv,
                TOP_K=top_k_per_layer,
            )

            end_k24.record()
            torch.cuda.synchronize()
            time_k24 = start_k24.elapsed_time(end_k24)
            print(f"      Kernel 2.4 time: {time_k24:.2f} ms")

            nvtx.range_pop()

            prev_selected_parents, topk_positions = topk_positions, prev_selected_parents
        
        nvtx.range_pop()
    nvtx.range_pop()

    # ========== Phase 4: Final Normalization ==========
    nvtx.range_push("Phase4_Final_Normalize_IO")
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

    print("htree forward pass (IO only) completed!")
    
    return output


__all__ = [
    'htree_forward_io_only',
]
