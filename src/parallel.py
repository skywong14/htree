# -*- coding: utf-8 -*-
"""
htree (Hierarchical Tree for KV Cache and Sparse Attention) 的 Triton Kernel 实现

基于逐层累积架构，使用在线 Softmax 算法实现高效的分层稀疏注意力。
"""

from typing import Optional, Tuple, List
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# ==========================================
# Kernel 1: Tree Building (逐层构建)
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8]
#     ],
#     key=['K', 'V', 'COMPRESSION_RATE'],
# )
@triton.jit
def htree_build_kernel(
    child_k,  # [B, N_child, H, K]
    child_v,  # [B, N_child, H, V]
    parent_k,  # [B, N_parent, H, K]
    parent_v,  # [B, N_parent, H, V]
    N_child: tl.constexpr,
    N_parent: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """
    Tree building kernel: mean pooling from child nodes to parent nodes
    Grid: (N_parent, B * H)
    Each block processes one parent node
    """
    i_parent = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # child range of parent node
    child_start = i_parent * COMPRESSION_RATE
    child_end = tl.minimum(child_start + COMPRESSION_RATE, N_child)
    num_valid_children = child_end - child_start
    
    # init
    k_sum = tl.zeros([K], dtype=tl.float32)
    v_sum = tl.zeros([V], dtype=tl.float32)
    
    # accumulate K and V (mean-pooling)
    for i_child in range(num_valid_children):
        child_idx = child_start + i_child
        
        # load child_k: [K]
        k_offset = i_b * N_child * H * K + child_idx * H * K + i_h * K
        k_ptrs = child_k + k_offset + tl.arange(0, K)
        k_vals = tl.load(k_ptrs, mask=tl.arange(0, K) < K, other=0.0)
        k_sum += k_vals
        
        # load child_v: [V]
        v_offset = i_b * N_child * H * V + child_idx * H * V + i_h * V
        v_ptrs = child_v + v_offset + tl.arange(0, V)
        v_vals = tl.load(v_ptrs, mask=tl.arange(0, V) < V, other=0.0)
        v_sum += v_vals
    
    k_mean = k_sum / num_valid_children
    v_mean = v_sum / num_valid_children
    
    # store K and V
    parent_k_offset = i_b * N_parent * H * K + i_parent * H * K + i_h * K
    parent_k_ptrs = parent_k + parent_k_offset + tl.arange(0, K)
    tl.store(parent_k_ptrs, k_mean.to(parent_k_ptrs.dtype.element_ty), mask=tl.arange(0, K) < K)
    
    parent_v_offset = i_b * N_parent * H * V + i_parent * H * V + i_h * V
    parent_v_ptrs = parent_v + parent_v_offset + tl.arange(0, V)
    tl.store(parent_v_ptrs, v_mean.to(parent_v_ptrs.dtype.element_ty), mask=tl.arange(0, V) < V)

# ==========================================
# Kernel 2: Layer-by-Layer Forward Pass
# ==========================================

# 辅助函数：加载 K 并应用 RoPE
@triton.jit
def load_k_with_rope(
    layer_k,
    cos_cache,
    sin_cache,
    i_b: tl.constexpr,
    i_h: tl.constexpr,
    child_start,
    rope_position_start,
    N_layer: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    num_valid_children,
):
    """
    加载 16 个 token 的 K 并应用 RoPE 编码
    
    返回:
        k_rope_1: [COMPRESSION_RATE, K//2]
        k_rope_2: [COMPRESSION_RATE, K//2]
    """
    k_base = layer_k + i_b * N_layer * H * K + i_h * K
    
    # 前半部分
    k1_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H * K, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k1 = tl.load(k1_block_ptrs, boundary_check=(0, 1))  # [16, K//2]
    
    # 后半部分
    k2_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H * K, 1),
        offsets=(child_start, K // 2),
        block_shape=(COMPRESSION_RATE, K // 2),
        order=(1, 0)
    )
    k2 = tl.load(k2_block_ptrs, boundary_check=(0, 1))  # [16, K//2]
    
    # 应用 RoPE
    rope_positions = rope_position_start + tl.arange(0, COMPRESSION_RATE)
    o_k_half = tl.arange(0, K // 2)
    
    # 加载 cos/sin cache [16, K//2]
    # mask: 前 num_valid_children 有效
    valid_rows = tl.arange(0, COMPRESSION_RATE)[:, None] < num_valid_children
    cos_ptrs = cos_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    sin_ptrs = sin_cache + rope_positions[:, None] * (K // 2) + o_k_half[None, :]
    cos_k = tl.load(cos_ptrs, mask=valid_rows, other=0.0)
    sin_k = tl.load(sin_ptrs, mask=valid_rows, other=0.0)
    
    k_rope_1 = k1 * cos_k - k2 * sin_k  # [16, K//2]
    k_rope_2 = k1 * sin_k + k2 * cos_k  # [16, K//2]
    
    return k_rope_1, k_rope_2


# 辅助函数：加载 V
@triton.jit
def load_v(
    layer_v,
    i_b: tl.constexpr,
    i_h: tl.constexpr,
    child_start,
    N_layer: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """
    加载 16 个 token 的 V
    
    返回:
        v_vals: [COMPRESSION_RATE, V]
    """
    v_base = layer_v + i_b * N_layer * H * V + i_h * V
    
    v_block_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(N_layer, V),
        strides=(H * V, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, V),
        order=(1, 0)
    )
    v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))  # [16, V]
    
    return v_vals


# ==========================================
# Kernel 2.1: 候选节点遍历 + 在线 Softmax + Top-K 选择
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [2, 4, 8]
#     ],
#     key=['K', 'V'],
# )
@triton.jit
def htree_compute_and_select_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H, K]
    layer_v,  # [B, N_layer, H, V]
    prev_selected_parents,  # [B, T, H, TOP_K]，-1 表示 padding
    cos_cache,  # [cache_size, K//2]
    sin_cache,  # [cache_size, K//2]
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    topk_indices,  # [B, T, H, TOP_K]
    topk_scores,  # [B, T, H, TOP_K]
    layer_idx: tl.constexpr,
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    scale: tl.constexpr,
):
    """
    Kernel 2.1: 候选节点遍历 + 在线 Softmax + Top-K 选择
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    BC: tl.constexpr = TOP_K
    
    # ======================================
    # 阶段 1: 确定候选节点范围与 Padding
    # ======================================
    
    rightmost_idx = i_t // layer_power
    
    prev_sel_base = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
    o_parent = tl.arange(0, TOP_K)
    
    parent_list = tl.load(prev_selected_parents + prev_sel_base + o_parent)
    
    valid_mask1 = parent_list >= 0
    num_valid_parents = tl.sum(valid_mask1.to(tl.int32))
    
    num_batches = (num_valid_parents + 31) // 32
    
    # ======================================
    # 阶段 2: 应用 RoPE 到 Query
    # ======================================
    
    if num_valid_parents > 0:
        num_candidates = (num_valid_parents - 1) * COMPRESSION_RATE + (rightmost_idx % COMPRESSION_RATE) + 1
    else:
        num_candidates = 0
    
    rope_pos_q = tl.maximum(num_candidates - 1, 0)
    
    q_offset = i_b * T * H * K + i_t * H * K + i_h * K
    o_k = tl.arange(0, K // 2)
    
    q1_ptrs = q + q_offset + o_k
    q2_ptrs = q + q_offset + (K // 2) + o_k
    q1 = tl.load(q1_ptrs)
    q2 = tl.load(q2_ptrs)
    
    cos_ptrs = cos_cache + rope_pos_q * (K // 2) + o_k
    sin_ptrs = sin_cache + rope_pos_q * (K // 2) + o_k
    cos_q = tl.load(cos_ptrs)
    sin_q = tl.load(sin_ptrs)
    
    q_rope_1 = (q1 * cos_q - q2 * sin_q) * scale
    q_rope_2 = (q1 * sin_q + q2 * cos_q) * scale
    
    # ======================================
    # 阶段 3: 批次遍历候选节点 (BC=512，32x16 加载)
    # ======================================
    
    # init
    cur_max = float('-inf')
    cur_sum = 0.0
    cur_output = tl.zeros([V], dtype=tl.float32)
    if not is_bottom_layer:
        b_i = tl.zeros([BC], dtype=tl.float32)
        o_i = tl.full([BC], -1, dtype=tl.int32)
        b_score = tl.full([BC], float('-inf'), dtype=tl.float32)
        m_i = tl.arange(0, BC) < BC // 2
    
    for i_batch in range(num_batches):
        PARENTS_PER_BATCH: tl.constexpr = 32
        
        # 加载 32 个父节点索引
        o_parent_local = tl.arange(0, PARENTS_PER_BATCH)
        parent_indices = tl.load(
            prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + o_parent_local,
            mask=o_parent_local < PARENTS_PER_BATCH,
            other=-1
        )  # [32]
        
        # mask
        valid_parent_mask = parent_indices >= 0  # [32]
        
        # 子节点起始位置
        child_starts = parent_indices * COMPRESSION_RATE  # [32]
        child_starts = tl.where(valid_parent_mask, child_starts, 0)  # 无效位置用0占位
        
        # 判断最右侧父节点
        is_rightmost_parent = (child_starts // COMPRESSION_RATE == rightmost_idx // COMPRESSION_RATE)  # [32]
        num_valid_children_per_parent = tl.where(
            is_rightmost_parent,
            (rightmost_idx % COMPRESSION_RATE) + 1,
            COMPRESSION_RATE
        )  # [32]
        
        # 构建 2D 索引 [32, 16]
        o_parent_2d = tl.arange(0, PARENTS_PER_BATCH)[:, None]  # [32, 1]
        o_child_offset = tl.arange(0, COMPRESSION_RATE)[None, :]  # [1, 16]
        
        # 子节点全局索引 [32, 16]
        child_indices_2d = child_starts[:, None] + o_child_offset  # [32, 16]
        
        # mask [32, 16]
        valid_child_mask_2d = (
            valid_parent_mask[:, None] &  # 父节点有效
            (o_child_offset < num_valid_children_per_parent[:, None])  # 子节点在范围内
        )
        
        scores_2d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], float('-inf'), dtype=tl.float32)
        values_2d = tl.zeros([PARENTS_PER_BATCH, COMPRESSION_RATE, V], dtype=tl.float32)
        # range or tl.static_range?
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                
                is_rightmost = (child_start // COMPRESSION_RATE == rightmost_idx // COMPRESSION_RATE)
                num_valid_children = tl.where(
                    is_rightmost,
                    (rightmost_idx % COMPRESSION_RATE) + 1,
                    COMPRESSION_RATE
                )
                
                rope_pos_start = i_batch * BC + i_p * COMPRESSION_RATE
                
                # 加载 Key 并应用 RoPE
                k_rope_1, k_rope_2 = load_k_with_rope(
                    layer_k, cos_cache, sin_cache,
                    i_b, i_h, child_start, rope_pos_start,
                    N_layer, H, K, COMPRESSION_RATE, num_valid_children
                )
                
                scores_16 = tl.sum(q_rope_1[None, :] * k_rope_1, axis=1) + \
                           tl.sum(q_rope_2[None, :] * k_rope_2, axis=1)
                v_vals = load_v(
                    layer_v, i_b, i_h, child_start,
                    N_layer, H, V, COMPRESSION_RATE
                )
                
                scores_2d = tl.where(
                    (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p) & (tl.arange(0, COMPRESSION_RATE)[None, :] < COMPRESSION_RATE),
                    scores_16[None, :],
                    scores_2d
                )
                values_2d = tl.where(
                    (tl.arange(0, PARENTS_PER_BATCH)[:, None, None] == i_p) & 
                    (tl.arange(0, COMPRESSION_RATE)[None, :, None] < COMPRESSION_RATE) &
                    (tl.arange(0, V)[None, None, :] < V),
                    v_vals[None, :, :],
                    values_2d
                )
        
        # Flatten 到 [BC] = [512]
        batch_scores = tl.reshape(scores_2d, [BC])
        batch_values = tl.reshape(values_2d, [BC, V])
        
        # 应用 mask
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        batch_scores = tl.where(valid_child_mask_flat, batch_scores, float('-inf'))
        
        # 子节点索引
        batch_node_indices = tl.reshape(child_indices_2d, [BC])
        batch_node_indices = tl.where(valid_child_mask_flat, batch_node_indices, -1)
        
        # Online Softmax
        new_max = tl.maximum(cur_max, tl.max(batch_scores))
        scale_old = tl.exp(cur_max - new_max)
        scale_batch = tl.exp(batch_scores - new_max)
        
        cur_sum = cur_sum * scale_old + tl.sum(scale_batch)
        cur_output = cur_output * scale_old + tl.sum(scale_batch[:, None] * batch_values, axis=0)
        cur_max = new_max
        
        # Top-K 选择 (非底层)
        if not is_bottom_layer:
            is_rightmost = (batch_node_indices == rightmost_idx)
            is_valid = (batch_scores > float('-inf'))
            
            importance = tl.where(
                is_rightmost,
                1.0,
                tl.where(
                    is_valid,
                    tl.exp(batch_scores - cur_max),
                    0.0
                )
            )
            
            b_i_prev = b_i
            o_i_prev = o_i
            
            batch_i = importance
            batch_o_i = batch_node_indices
          
            n_dims: tl.constexpr = 9   # n_dims = 9 (TOP_K = 2^9)
            n_outer: tl.constexpr = 1
            for i_sort in tl.static_range(1, n_dims):
                batch_i, batch_o_i = _bitonic_merge(batch_i, batch_o_i, i_sort, 2, n_dims, n_outer)
            
            if i_batch != 0:
                batch_i, batch_o_i = _bitonic_merge(batch_i, batch_o_i, n_dims, False, n_dims, n_outer)
                
                b_i_merged = b_i_prev * m_i + batch_i * (1 - m_i)
                o_i_merged = o_i_prev * m_i + batch_o_i * (1 - m_i)
                
                b_i, o_i = _bitonic_merge(b_i_merged, o_i_merged, n_dims, True, n_dims, n_outer)
            else:
                b_i, o_i = _bitonic_merge(batch_i, batch_o_i, n_dims, True, n_dims, n_outer)
            
            # 当节点索引 >= 0 且相等时才认为匹配
            match_matrix = (o_i[:, None] == batch_node_indices[None, :]) & (o_i[:, None] >= 0)
            
            matched_scores = tl.sum(tl.where(match_matrix, batch_scores[None, :], 0.0), axis=1)
            is_in_current_batch = tl.sum(match_matrix.to(tl.int32), axis=1) > 0
            
            if i_batch != 0:
                b_score = tl.where(is_in_current_batch, matched_scores, b_score)
            else:
                b_score = tl.where(is_in_current_batch, matched_scores, float('-inf'))
    
    # 存储当前层状态
    state_offset = i_b * T * H + i_t * H + i_h
    tl.store(layer_max + state_offset, cur_max)
    tl.store(layer_sum + state_offset, cur_sum)
    
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
    o_v = tl.arange(0, V)
    output_ptrs = layer_output + output_offset + o_v
    tl.store(output_ptrs, cur_output, mask=o_v < V)
    
    # 存储 Top-K
    if not is_bottom_layer:
        topk_offset = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
        o_topk = tl.arange(0, TOP_K)
        
        topk_indices_ptrs = topk_indices + topk_offset + o_topk
        tl.store(topk_indices_ptrs, o_i.to(topk_indices_ptrs.dtype.element_ty))
        
        # 无效索引对应的 b_score 为 -inf
        # TODO 可能有误
        b_score_corrected = tl.where(o_i >= 0, b_score, float('-inf'))
        
        topk_scores_ptrs = topk_scores + topk_offset + o_topk
        tl.store(topk_scores_ptrs, b_score_corrected)


# ==========================================
# Kernel 2.2: Top-K 贡献移除
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [2, 4, 8]
#     ],
#     key=['V'],
# )
@triton.jit
def htree_subtract_topk_kernel(
    layer_v,  # [B, N_layer, H, V]
    # Top-K (unsorted)
    topk_indices,  # [B, T, H, TOP_K]
    topk_scores,  # [B, T, H, TOP_K]
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    selected_indices,  # [B, T, H, TOP_K]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """
    Kernel 2.2: Top-K 贡献移除
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # 当前层
    state_offset = i_b * T * H + i_t * H + i_h
    cur_max = tl.load(layer_max + state_offset)
    cur_sum = tl.load(layer_sum + state_offset)
    
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
    o_v = tl.arange(0, V)
    output_ptrs = layer_output + output_offset + o_v
    cur_output = tl.load(output_ptrs, mask=o_v < V, other=0.0)
    
    # 加载 Top-K
    topk_offset = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
    o_topk = tl.arange(0, TOP_K)
    
    topk_indices_ptrs = topk_indices + topk_offset + o_topk
    topk_indices_unsorted = tl.load(topk_indices_ptrs)
    
    topk_scores_ptrs = topk_scores + topk_offset + o_topk
    topk_scores_vals = tl.load(topk_scores_ptrs)
    
    # ======================================
    # 步骤 1: 对 Top-K 索引排序
    # ======================================
    
    topk_indices_sorted = topk_indices_unsorted
    indices_positions = tl.arange(0, TOP_K).to(tl.int32)
    
    n_dims: tl.constexpr = 9
    n_outer: tl.constexpr = 1
    
    MAX_IDX = 2147483647
    topk_indices_for_sort = tl.where(topk_indices_sorted >= 0, topk_indices_sorted, MAX_IDX)
    
    for i_sort in tl.static_range(1, n_dims + 1):
        topk_indices_for_sort, indices_positions = _bitonic_merge(
            topk_indices_for_sort, indices_positions, i_sort, False, n_dims, n_outer
        )
    
    topk_indices_sorted = tl.where(topk_indices_for_sort < MAX_IDX, topk_indices_for_sort, -1)
    
    # 存储排序后的索引
    sel_ptrs = selected_indices + topk_offset + o_topk
    tl.store(sel_ptrs, topk_indices_sorted.to(sel_ptrs.dtype.element_ty))
    
    # ======================================
    # 步骤 2: 汇总 Top-K 节点的贡献
    # ======================================
    
    # [TOP_K]
    # 使用索引判断有效性
    is_valid_topk = (topk_indices_unsorted >= 0)
    
    # 对于有效节点，使用实际分数；对于无效节点，使用 -inf
    safe_scores = tl.where(is_valid_topk, topk_scores_vals, float('-inf'))
    
    # 全局最大值
    topk_max = tl.max(safe_scores)
    
    # 计算所有节点的指数权重
    # [TOP_K]
    # 先将无效节点的分数设为一个安全值（比如 topk_max - 100），避免 exp(-inf)
    # 这样 exp(topk_max - 100 - topk_max) = exp(-100) ≈ 0，而不会产生 NaN
    safe_scores_for_exp = tl.where(is_valid_topk, topk_scores_vals, topk_max - 100.0)
    topk_exp = tl.where(is_valid_topk, tl.exp(safe_scores_for_exp - topk_max), 0.0)
    topk_sum = tl.sum(topk_exp)
    
    # 加载所有 Top-K 节点的 Value 到临时数组，再加权求和
    topk_values = tl.zeros([TOP_K, V], dtype=tl.float32)
    
    # 循环加载所有 Top-K 节点的 Value
    # TODO: 编译时间过长，效率过低
    all_positions = tl.arange(0, TOP_K)  # [512]
    
    # TODO: range or tl.static_range?
    for i_node in range(TOP_K):
        # 第 i_node 个节点的索引
        node_mask = (all_positions == i_node)
        node_idx = tl.sum(tl.where(node_mask, topk_indices_unsorted, 0))
        
        is_valid_node = node_idx >= 0
        
        # 加载 Value [V]
        # 无效节点使用索引 0，后续会被 mask 掉
        safe_idx = tl.maximum(node_idx, 0)
        v_offset = i_b * N_layer * H * V + safe_idx * H * V + i_h * V
        v_ptrs = layer_v + v_offset + o_v
        v_node = tl.load(v_ptrs, mask=o_v < V, other=0.0)
        
        # 存到 topk_values 的第 i_node 行
        row_mask = (all_positions[:, None] == i_node) & (tl.arange(0, V)[None, :] < V)
        # 无效节点的 value 用 0 替代
        v_node_masked = tl.where(is_valid_node, v_node, 0.0)
        topk_values = tl.where(row_mask, v_node_masked[None, :], topk_values)
    
    # topk_output = sum(topk_exp[:, None] * topk_values, axis=0)
    # [TOP_K, 1] * [TOP_K, V] -> [TOP_K, V] -> sum(axis=0) -> [V]
    topk_output = tl.sum(topk_exp[:, None] * topk_values, axis=0)  # [V]
    
    # ======================================
    # 步骤 3: 从当前累积中移除 Top-K 贡献
    # ======================================
    
    if topk_sum > 0:
        topk_scale = tl.exp(topk_max - cur_max)
        topk_sum_scaled = topk_sum * topk_scale
        topk_output_scaled = topk_output * topk_scale
        
        cur_sum = cur_sum - topk_sum_scaled
        cur_output = cur_output - topk_output_scaled
    
    tl.store(layer_max + state_offset, cur_max)
    tl.store(layer_sum + state_offset, cur_sum)
    tl.store(output_ptrs, cur_output, mask=o_v < V)


# ==========================================
# Kernel 2.3: 全局状态合并
# ==========================================

@triton.jit
def htree_merge_to_global_kernel(
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    global_max,  # [B, T, H]
    global_sum,  # [B, T, H]
    global_output,  # [B, T, H, V]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """
    Kernel 2.3: 全局状态合并
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    state_offset = i_b * T * H + i_t * H + i_h
    
    # load current layer state
    cur_max = tl.load(layer_max + state_offset)
    cur_sum = tl.load(layer_sum + state_offset)
    
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
    o_v = tl.arange(0, V)
    layer_output_ptrs = layer_output + output_offset + o_v
    cur_output = tl.load(layer_output_ptrs, mask=o_v < V, other=0.0)
    
    # load global state
    g_max = tl.load(global_max + state_offset)
    g_sum = tl.load(global_sum + state_offset)
    
    global_output_ptrs = global_output + output_offset + o_v
    g_output_vals = tl.load(global_output_ptrs, mask=o_v < V, other=0.0)
    
    # Online Softmax
    new_max = tl.maximum(g_max, cur_max)
    scale_g = tl.exp(g_max - new_max)
    scale_c = tl.exp(cur_max - new_max)
    
    g_sum = g_sum * scale_g + cur_sum * scale_c
    g_output_vals = g_output_vals * scale_g + cur_output * scale_c
    g_max = new_max
    
    # write back
    tl.store(global_max + state_offset, g_max)
    tl.store(global_sum + state_offset, g_sum)
    tl.store(global_output_ptrs, g_output_vals, mask=o_v < V)


# ==========================================
# Kernel 3: Final Normalization
# ==========================================

@triton.jit
def htree_final_normalize_kernel(
    global_output,  # [B, T, H, V]
    global_sum,  # [B, T, H]
    output,  # [B, T, H, V]
    B: tl.constexpr,
    T: tl.constexpr,
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
    sum_offset = i_b * T * H + i_t * H + i_h
    sum_val = tl.load(global_sum + sum_offset)
    
    # load and normalize output
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
    o_v = tl.arange(0, V)
    output_ptrs = global_output + output_offset + o_v
    output_vals = tl.load(output_ptrs, mask=o_v < V, other=0.0)
    
    normalized = output_vals / sum_val
    
    # write back
    out_ptrs = output + output_offset + o_v
    tl.store(out_ptrs, normalized, mask=o_v < V)


# ==========================================
# Main Forward Function
# ==========================================

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
    htree 前向传播主函数
    
    Args:
        q: [B, T, H, K] Query tensor
        k: [B, T, H, K] Key tensor
        v: [B, T, H, V] Value tensor
        compression_rate: 16
        max_top_nodes: 8192
        top_k_per_layer: 512
        scale: K^-0.5
        rope_base: 10000.0
    
    Returns:
        output: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    device = q.device
    dtype = q.dtype
    
    # ensure contiguous
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
    
    print(f"  Tree has {num_layers} layers")
    
    # store K and V of each layer
    layers_k = [k]  # layer 0: 底层 token
    layers_v = [v]
    
    current_k, current_v = k, v
    current_len = T
    
    # build tree layer by layer
    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H, V, dtype=dtype, device=device)
        
        grid = (next_len, B * H)
        htree_build_kernel[grid](
            current_k, current_v, next_k, next_v,
            N_child=current_len,
            N_parent=next_len,
            B=B, H=H, K=K, V=V,
            COMPRESSION_RATE=compression_rate,
        )
        
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len
        
        print(f"  Built layer {layer_idx}: {current_len} nodes")
    nvtx.range_pop()

    # ========== Phase 1.5: Precompute RoPE cache ==========
    nvtx.range_push("Phase1.5_RoPE_Cache")
    print("Phase 1.5: Precomputing RoPE cache...")
    
    cache_size = max_top_nodes + 1024
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
    positions = torch.arange(cache_size, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)  # [cache_size, K//2]
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    nvtx.range_pop()

    # ========== Phase 2: Initialize Global State ==========
    nvtx.range_push("Phase2_Init_GlobalState")
    print("Phase 2: Initializing global states...")
    
    global_max = torch.full([B, T, H], float('-inf'), dtype=torch.float32, device=device)
    global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
    global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)
    nvtx.range_pop()

    # ========== Phase 3: Layer-by-layer Forward ==========
    nvtx.range_push("Phase3_LayerByLayer_Forward")
    print("Phase 3: Layer-by-layer forward pass...")
    
    # 分配中间状态张量（层内复用）
    layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)
    
    # Top-K 信息（非底层）
    topk_indices = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    topk_scores = torch.empty([B, T, H, top_k_per_layer], dtype=torch.float32, device=device)
    
    # 选中节点索引缓存 (层间复用)
    selected_indices = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    
    # 为最顶层初始化 prev_selected_parents（虚拟父节点）
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)  # [T]
    rightmost_indices = t_indices // top_layer_power  # [T] 顶层最右侧节点
    num_virtual_parents = rightmost_indices // compression_rate + 1  # [T] 虚拟父节点数量
    
    # [T, TOP_K]
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
    
    # [T, TOP_K] -> [B, T, H, TOP_K]
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
        
        # ======================================
        # Kernel 2.1: Candidate Node + Online Softmax + Top-K Selection
        # ======================================
        nvtx.range_push("K2.1_ComputeAndSelect")
        print("    Running compute and select kernel...")
        htree_compute_and_select_kernel[grid](
            q, k_layer, v_layer,
            prev_selected_parents,
            cos_cache, sin_cache,
            layer_max, layer_sum, layer_output,
            topk_indices, topk_scores,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, K=K, V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            scale=scale,
        )
        nvtx.range_pop()
        
        # ======================================
        # Kernel 2.2: Remove Top-K Contribution (Non-bottom layer)
        # ======================================
        if not is_bottom_layer:
            nvtx.range_push("K2.2_SubtractTopK")
            print("    Running subtract top-k kernel...")
            htree_subtract_topk_kernel[grid](
                v_layer,
                topk_indices, topk_scores,
                layer_max, layer_sum, layer_output,
                selected_indices,
                B=B, T=T, H=H, V=V, N_layer=N_layer,
                TOP_K=top_k_per_layer,
            )
            nvtx.range_pop()
        
        # ======================================
        # Kernel 2.3: Merge to Global State
        # ======================================
        nvtx.range_push("K2.3_MergeToGlobal")
        print("    Running merge to global kernel...")
        htree_merge_to_global_kernel[grid](
            layer_max, layer_sum, layer_output,
            global_max, global_sum, global_output,
            B=B, T=T, H=H, V=V,
        )
        nvtx.range_pop()
        
        # update parent indices (pass to next layer)
        if not is_bottom_layer:
            prev_selected_parents = selected_indices
        
        nvtx.range_pop()
    nvtx.range_pop()

    # ========== Phase 4: Final Normalization ==========
    nvtx.range_push("Phase4_Final_Normalize")
    print("Phase 4: Final normalization...")
    
    output = torch.empty(B, T, H, V, dtype=dtype, device=device)
    grid = (T, B * H)
    htree_final_normalize_kernel[grid](
        global_output, global_sum, output,
        B=B, T=T, H=H, V=V,
    )
    nvtx.range_pop()

    print("htree forward pass completed!")
    return output

__all__ = [
    'htree_forward',
    'htree_build_kernel',
    'htree_compute_and_select_kernel',
    'htree_subtract_topk_kernel',
    'htree_merge_to_global_kernel',
    'htree_final_normalize_kernel',
]


@triton.jit
def _compare_and_swap(
    x,
    ids,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True) # 不兼容 Triton 2.3.1
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x,
    ids,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """
    Bitonic merge operation for sorting.
    """
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids