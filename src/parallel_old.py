# -*- coding: utf-8 -*-
"""
htree (Hierarchical Tree for KV Cache and Sparse Attention) 的 Triton Kernel 实现

基于逐层累积架构，使用在线 Softmax 算法实现高效的分层稀疏注意力。
"""

from typing import Optional
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# 常量定义：用于标记无效/未初始化的分数
# 选择 -1e10 而非 -inf 避免 NaN 传播
INVALID_SCORE = -1e10

# ==========================================
# Kernel 1: Tree Building (逐层构建)
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [2, 4, 8]
#     ],
#     key=['K', 'V'],
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
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tree building kernel: mean pooling from child nodes to parent nodes
    Grid: (B * H,)
    Each block processes one (batch, head) combination, handling all parent nodes in that layer
    """
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H
    
    # 每个 block 处理多个父节点，分批次处理
    num_iterations = tl.cdiv(N_parent, BLOCK_SIZE)
    
    for iter_idx in range(num_iterations):
        # 当前批次父节点和子节点的起始位置
        parent_start = iter_idx * BLOCK_SIZE
        child_start = parent_start * COMPRESSION_RATE
        
        # load K: [num_children_in_batch, K]
        k_base = child_k + i_b * N_child * H * K + i_h * K
        k_block_ptrs = tl.make_block_ptr(
            base=k_base,
            shape=(N_child, K),
            strides=(H * K, 1),
            offsets=(child_start, 0),
            block_shape=(BLOCK_SIZE * COMPRESSION_RATE, K),
            order=(1, 0)
        )
        # load [BLOCK_SIZE * COMPRESSION_RATE, K]
        k_vals = tl.load(k_block_ptrs, boundary_check=(0, 1))  # [BLOCK_SIZE*16, K]
        
        # load V: [num_children_in_batch, V]
        v_base = child_v + i_b * N_child * H * V + i_h * V
        v_block_ptrs = tl.make_block_ptr(
            base=v_base,
            shape=(N_child, V),
            strides=(H * V, 1),
            offsets=(child_start, 0),
            block_shape=(BLOCK_SIZE * COMPRESSION_RATE, V),
            order=(1, 0)
        )
        v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))  # [BLOCK_SIZE*16, V]
        
        # Reshape + Mean Pooling
        
        # 每个子节点的全局索引: [BLOCK_SIZE, 16]
        parent_global_idx = parent_start + tl.arange(0, BLOCK_SIZE)  # [BLOCK_SIZE]
        child_global_idx = parent_global_idx[:, None] * COMPRESSION_RATE + tl.arange(0, COMPRESSION_RATE)[None, :]
        child_valid = (parent_global_idx[:, None] < N_parent) & (child_global_idx < N_child)  # [BLOCK_SIZE, 16]
        
        # Reshape k_vals: [BLOCK_SIZE*16, K] -> [BLOCK_SIZE, 16, K]
        k_vals_reshaped = tl.reshape(k_vals, [BLOCK_SIZE, COMPRESSION_RATE, K])
        
        # 应用 mask 并求和
        # child_valid: [BLOCK_SIZE, 16] -> [BLOCK_SIZE, 16, 1]
        k_masked = tl.where(child_valid[:, :, None], k_vals_reshaped, 0.0)
        k_sum = tl.sum(k_masked, axis=1)  # [BLOCK_SIZE, K]
        num_valid_children = tl.sum(child_valid.to(tl.int32), axis=1)  # [BLOCK_SIZE]
        
        # 计算 mean
        num_valid_children_safe = tl.maximum(num_valid_children, 1)
        k_mean = k_sum / num_valid_children_safe[:, None]  # [BLOCK_SIZE, K]
        v_vals_reshaped = tl.reshape(v_vals, [BLOCK_SIZE, COMPRESSION_RATE, V])
        v_masked = tl.where(child_valid[:, :, None], v_vals_reshaped, 0.0)
        v_sum = tl.sum(v_masked, axis=1)  # [BLOCK_SIZE, V]
        v_mean = v_sum / num_valid_children_safe[:, None]  # [BLOCK_SIZE, V]
        
        # store
        # parent_k: [B, N_parent, H, K]
        parent_k_base = parent_k + i_b * N_parent * H * K + i_h * K
        parent_k_block_ptrs = tl.make_block_ptr(
            base=parent_k_base,
            shape=(N_parent, K),
            strides=(H * K, 1),
            offsets=(parent_start, 0),
            block_shape=(BLOCK_SIZE, K),
            order=(1, 0)
        )
        tl.store(parent_k_block_ptrs, k_mean.to(parent_k.dtype.element_ty), boundary_check=(0, 1))
        # parent_v: [B, N_parent, H, V]
        parent_v_base = parent_v + i_b * N_parent * H * V + i_h * V
        parent_v_block_ptrs = tl.make_block_ptr(
            base=parent_v_base,
            shape=(N_parent, V),
            strides=(H * V, 1),
            offsets=(parent_start, 0),
            block_shape=(BLOCK_SIZE, V),
            order=(1, 0)
        )
        tl.store(parent_v_block_ptrs, v_mean.to(parent_v.dtype.element_ty), boundary_check=(0, 1))

# ==========================================
# Kernel 2: Layer-by-Layer Forward Pass
# ==========================================

# 辅助函数：加载 K 并应用 RoPE
@triton.jit
def load_k_with_rope(
    layer_k,
    cos_cache,
    sin_cache,
    i_b,
    i_h,
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
    i_b,
    i_h,
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
    max_score_excluding_topk_out,  # [B, T, H] 存储被排除节点的最大分数
    layer_idx: tl.constexpr,
    layer_power: tl.constexpr,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    scale,
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
    cur_max = -1e10
    cur_sum = 0.0
    cur_output = tl.zeros([V], dtype=tl.float32)
    if not is_bottom_layer:
        b_i = tl.zeros([BC], dtype=tl.float32)
        o_i = tl.full([BC], -1, dtype=tl.int32)
        b_score = tl.full([BC], -1e10, dtype=tl.float32)
        max_score_excluding_topk = -1e10
    
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
        
        scores_2d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], -1e10, dtype=tl.float32)
        values_2d = tl.zeros([PARENTS_PER_BATCH, COMPRESSION_RATE, V], dtype=tl.float32)
        
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                rope_pos_start = i_batch * BC + i_p * COMPRESSION_RATE
                
                # 使用 mask 从 valid_child_mask_2d 中提取当前父节点的子节点 mask
                # valid_child_mask_2d: [32, 16]
                # 创建选择第 i_p 行的 mask
                row_selector = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)  # [32, 1]
                # 提取第 i_p 行: 将非目标行置0，然后沿 axis=0 求和得到 [16]
                current_parent_mask = tl.sum(
                    tl.where(row_selector, valid_child_mask_2d.to(tl.int32), 0),
                    axis=0
                ).to(tl.int1)  # [16]
                num_valid_children = tl.sum(current_parent_mask.to(tl.int32))
                
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
                
                # 使用外层计算的统一 mask 填充 scores_2d 和 values_2d
                is_current_parent = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)
                scores_2d = tl.where(
                    is_current_parent & valid_child_mask_2d,
                    scores_16[None, :],
                    scores_2d
                )
                values_2d = tl.where(
                    is_current_parent[:, :, None] & 
                    valid_child_mask_2d[:, :, None] &
                    (tl.arange(0, V)[None, None, :] < V),
                    v_vals[None, :, :],
                    values_2d
                )
        
        # Flatten 到 [BC] = [512]
        batch_scores = tl.reshape(scores_2d, [BC])
        batch_values = tl.reshape(values_2d, [BC, V])
        
        # 应用 mask
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        batch_scores = tl.where(valid_child_mask_flat, batch_scores, -1e10)
        
        # 将无效节点的 Value 置零
        batch_values = tl.where(valid_child_mask_flat[:, None], batch_values, 0.0)

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
            
            importance = tl.where(
                is_rightmost,
                1.0,
                tl.where(
                    batch_node_indices >= 0,
                    tl.exp(batch_scores - cur_max),
                    0.0
                )
            )
            
            batch_i = importance
            batch_o_i = batch_node_indices
            batch_s = batch_scores  # 保持 score 与 node_id 的对应关系
          
            n_dims: tl.constexpr = 9   # n_dims = 9 (TOP_K = 2^9 = 512)
            n_outer: tl.constexpr = 1
            
            # 三路排序 (importance, node_id, score)
            for i_sort in tl.static_range(1, n_dims + 1):
                batch_i, batch_o_i, batch_s = _bitonic_merge_triple(
                    batch_i, batch_o_i, batch_s, i_sort, 
                    2 if i_sort < n_dims else True, 
                    n_dims, n_outer
                )
            
            if i_batch != 0:
                # fix：用新的 cur_max 重新计算历史节点的 importance
                # 否则历史节点和当前节点的 importance 基准不同，导致排序错误
                is_rightmost_historical = (o_i == rightmost_idx)
                b_i = tl.where(
                    is_rightmost_historical,
                    1.0,
                    tl.where(
                        o_i >= 0,  # 只重新计算有效节点
                        tl.exp(b_score - cur_max),
                        0.0
                    )
                )
                
                # 精确 Top-K：将历史512和当前512合并为1024，排序后取前512
                # 使用三路排序同时维护 (importance, node_id, score) 三元组
                merged_size: tl.constexpr = BC * 2  # 1024
                merged_n_dims: tl.constexpr = 10    # 2^10 = 1024
                merged_n_outer: tl.constexpr = 1
                
                # 1. 创建 [2, 512] 的 2D 数组
                # 2. 第0行填充历史 Top-K，第1行填充当前批次
                # 3. reshape 成 [1024]
                o_row = tl.arange(0, 2)[:, None]     # [2, 1]
                o_col = tl.arange(0, BC)[None, :]    # [1, 512]
                
                # 创建 [2, 512] 数组：第0行=历史，第1行=当前批次
                merged_2d_i = tl.where(
                    o_row == 0,
                    b_i[None, :],      # importance
                    batch_i[None, :]
                )
                merged_2d_o_i = tl.where(
                    o_row == 0,
                    o_i[None, :],      # node_id
                    batch_o_i[None, :]
                )
                merged_2d_score = tl.where(
                    o_row == 0,
                    b_score[None, :],  # score
                    batch_s[None, :]   # 使用排序后的 batch_s
                )
                
                # Reshape 成 [1024]
                merged_i = tl.reshape(merged_2d_i, [merged_size])
                merged_o_i = tl.reshape(merged_2d_o_i, [merged_size])
                merged_score = tl.reshape(merged_2d_score, [merged_size])
                
                # 对1024个元素排序 (降序) - 使用三路 bitonic merge
                # 排序基于 merged_i (importance)，merged_o_i 和 merged_score 跟着重排
                for i_sort in tl.static_range(1, merged_n_dims + 1):
                    merged_i, merged_o_i, merged_score = _bitonic_merge_triple(
                        merged_i, merged_o_i, merged_score, i_sort, 
                        2 if i_sort < merged_n_dims else True, 
                        merged_n_dims, merged_n_outer
                    )
                
                # 取前512个作为新的 Top-K
                # 将 [1024] reshape 成 [2, 512]，取第0行
                merged_2d_sorted_i = tl.reshape(merged_i, [2, BC])
                merged_2d_sorted_o_i = tl.reshape(merged_o_i, [2, BC])
                merged_2d_sorted_score = tl.reshape(merged_score, [2, BC])
                
                # 提取第0行 (前512个元素)
                o_row_extract = tl.arange(0, 2)[:, None]
                mask_first_row = (o_row_extract == 0)
                
                # 保留第 0 行，其他行置 0，然后沿 axis=0 求和 [512]
                b_i = tl.sum(tl.where(mask_first_row, merged_2d_sorted_i, 0.0), axis=0)
                o_i = tl.sum(tl.where(mask_first_row, merged_2d_sorted_o_i, 0), axis=0)
                b_score = tl.sum(tl.where(mask_first_row, merged_2d_sorted_score, 0.0), axis=0)
                
                # 提取第513个元素 (Top K+1) 作为被排除节点的最大分数
                # merged_2d_sorted_score[1, 0] 即第二行第一个元素
                mask_second_row_first = (o_row_extract == 1) & (tl.arange(0, BC)[None, :] == 0)
                top_kplus1_score = tl.sum(tl.where(mask_second_row_first, merged_2d_sorted_score, 0.0))
                max_score_excluding_topk = tl.maximum(max_score_excluding_topk, top_kplus1_score)
            else:
                # 第一批：直接使用排序后的当前批次
                b_i = batch_i
                o_i = batch_o_i
                b_score = batch_s  # 使用排序后的 batch_s
            
            # 注意：后续不再需要 match_matrix 来初始化 b_score
            # 因为 b_score 已经通过三路排序正确维护了
            
            # 更新 b_score：只需要更新在当前批次中的节点
            # 不在当前批次的节点，其 b_score 已经在三路排序中正确维护
            if i_batch != 0:
                # 当节点索引 >= 0 且相等时才认为匹配
                match_matrix = (o_i[:, None] == batch_node_indices[None, :]) & (o_i[:, None] >= 0)
                matched_scores = tl.sum(tl.where(match_matrix, batch_scores[None, :], 0.0), axis=1)
                is_in_current_batch = tl.sum(match_matrix.to(tl.int32), axis=1) > 0
                b_score = tl.where(is_in_current_batch, matched_scores, b_score)
    
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
        
        # 无效索引对应的 b_score 为 -1e10
        b_score_corrected = tl.where(o_i >= 0, b_score, -1e10)
        
        topk_scores_ptrs = topk_scores + topk_offset + o_topk
        tl.store(topk_scores_ptrs, b_score_corrected)
        
        # 存储被排除节点的最大分数
        tl.store(max_score_excluding_topk_out + state_offset, max_score_excluding_topk)


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
    max_score_excluding_topk_in,  # [B, T, H] 排除TopK节点的最大分数
    B: tl.constexpr,
    T,
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
    
    # 准备排序数据：将无效索引 -1 替换为 MAX_IDX
    MAX_IDX = 2147483647
    topk_indices_for_sort = tl.where(topk_indices_unsorted >= 0, topk_indices_unsorted, MAX_IDX)
    indices_positions = tl.arange(0, TOP_K).to(tl.int32)
    
    # 升序排序
    # TOP_K = 512, n_dims = log2(512) = 9, n_outer = 1 (1D数组)
    topk_n_dims: tl.constexpr = 9  # log2(TOP_K) where TOP_K = 512
    topk_indices_for_sort, indices_positions = argsort(
        topk_indices_for_sort,
        indices_positions,
        n_dims=topk_n_dims,
        n_outer=1,
        descending=False
    )
    
    topk_indices_sorted = tl.where(topk_indices_for_sort < MAX_IDX, topk_indices_for_sort, -1)
    
    # 存储排序后的索引
    sel_ptrs = selected_indices + topk_offset + o_topk
    tl.store(sel_ptrs, topk_indices_sorted.to(sel_ptrs.dtype.element_ty))
    
    # ======================================
    # 步骤 2: 汇总 Top-K 节点的贡献
    # ======================================
    
    # [TOP_K]
    is_valid_topk = (topk_indices_unsorted >= 0)
    
    # 对于有效节点，使用实际分数；对于无效节点，使用 -1e10
    safe_scores = tl.where(is_valid_topk, topk_scores_vals, -1e10)
    
    # 全局最大值
    topk_max = tl.max(safe_scores)
    
    # 计算所有节点的指数权重
    # [TOP_K]
    # 先将无效节点的分数设为一个安全值 (比如 topk_max - 100)，避免 exp(-inf)
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
    # 步骤 3: 确定新的 cur_max (被排除节点的最大分数)
    # ======================================
    
    # 读取被排除节点的最大分数
    max_excluded = tl.load(max_score_excluding_topk_in + state_offset)
    
    # 保存旧的 cur_max
    old_cur_max = cur_max
    
    # ======================================
    # 步骤 4: 先将 cur_sum 和 cur_output 重新缩放到 new_cur_max 基准
    # ======================================
    
    # 新的 cur_max 是被排除节点的最大分数
    new_cur_max = max_excluded
    
    # 将当前累积从 old_cur_max 缩放到 new_cur_max
    # scale_factor = exp(old_cur_max - new_cur_max)
    scale_factor = tl.exp(old_cur_max - new_cur_max)
    cur_sum = cur_sum * scale_factor
    cur_output = cur_output * scale_factor
    
    # ======================================
    # 步骤 5: 从重新缩放后的累积中移除 Top-K 贡献
    # ======================================
    
    # 现在 Top-K 的贡献也需要基于 new_cur_max 计算
    if topk_sum > 0:
        topk_scale = tl.exp(topk_max - new_cur_max)
        topk_sum_scaled = topk_sum * topk_scale
        topk_output_scaled = topk_output * topk_scale
        
        cur_sum = cur_sum - topk_sum_scaled
        cur_output = cur_output - topk_output_scaled
    
    # 当 cur_sum <= 0 时，表示所有节点都被移除，设置 cur_max 为标记值
    cur_max = tl.where(cur_sum <= 1e-10, -1e10, new_cur_max)
    
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
    T,
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
    
    # Online Softmax 合并 (处理无效值 -1e10 的特殊情况，避免 nan)
    # 情况1: 当前层无贡献 (cur_sum <= 0)→ 保持全局状态
    # 情况2: 全局无累积 (g_sum <= 0)→ 使用当前层状态
    # 情况3: 两者都有贡献 → 正常在线 Softmax 合并
    
    cur_has_contribution = cur_sum > 1e-10
    g_has_accumulation = g_sum > 1e-10
    
    # 计算合并结果，使用安全的 scale 避免 -inf - (-inf) = nan
    new_max = tl.maximum(g_max, cur_max)
    # 如果对应的状态没有贡献，scale 设为 0 (避免 exp(-inf - x) 的计算)
    scale_g = tl.where(g_has_accumulation, tl.exp(g_max - new_max), 0.0)
    scale_c = tl.where(cur_has_contribution, tl.exp(cur_max - new_max), 0.0)
    
    merged_sum = g_sum * scale_g + cur_sum * scale_c
    merged_output = g_output_vals * scale_g + cur_output * scale_c
    merged_max = new_max
    
    # 根据不同情况选择最终结果
    g_max = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_max, cur_max),  # 两者都有贡献用合并值，否则用当前值
        g_max  # 当前层无贡献，保持全局不变
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
        
        # 新设计: Grid = (B * H,)，每个 block 处理一层
        # BLOCK_SIZE 控制每次处理的父节点数量
        BLOCK_SIZE = 128  # 可以根据 K, V, next_len 调整
        grid = (B * H,)
        htree_build_kernel[grid](
            current_k, current_v, next_k, next_v,
            N_child=current_len,
            N_parent=next_len,
            B=B, H=H, K=K, V=V,
            COMPRESSION_RATE=compression_rate,
            BLOCK_SIZE=BLOCK_SIZE,
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
    
    global_max = torch.full([B, T, H], -1e10, dtype=torch.float32, device=device)
    global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
    global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)
    nvtx.range_pop()

    # ========== Phase 3: Layer-by-layer Forward ==========
    nvtx.range_push("Phase3_LayerByLayer_Forward")
    print("Phase 3: Layer-by-layer forward pass...")
    
    # 分配中间状态张量 (层内复用)
    layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
    layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)
    
    # Top-K 信息 (非底层)
    topk_indices = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    topk_scores = torch.empty([B, T, H, top_k_per_layer], dtype=torch.float32, device=device)
    
    # 被排除节点的最大分数 (非底层)
    max_score_excluding_topk = torch.empty([B, T, H], dtype=torch.float32, device=device)
    
    # 选中节点索引缓存 (层间复用)
    selected_indices = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    
    # 为最顶层初始化 prev_selected_parents (虚拟父节点)
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)  # [T]
    rightmost_indices = t_indices // top_layer_power  # [T] 顶层最右侧节点索引
    # 虚拟父节点数量：每个虚拟父节点展开成16个顶层节点
    num_virtual_parents = rightmost_indices // compression_rate + 1  # [T]
    
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
            max_score_excluding_topk,
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
                max_score_excluding_topk,
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
            prev_selected_parents, selected_indices = selected_indices, prev_selected_parents
        
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
    n_outer: tl.constexpr,
):
    # n_outer: tl.constexpr = x.numel >> n_dims
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
def _compare_and_swap_triple(
    x,
    ids,
    aux,
    flip,
    i: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """
    Compare-and-swap for three arrays: (x, ids, aux)
    Sorting is based on x, but ids and aux follow the same permutation.
    """
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    
    # Process x (the sorting key)
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    
    # Process ids
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    
    # Process aux
    y_aux = tl.reshape(aux, shape)
    left_aux = tl.broadcast_to(tl.sum(y_aux * (1 - mask), 1)[:, None, :], shape).to(y_aux.dtype)
    right_aux = tl.broadcast_to(tl.sum(y_aux * mask, 1)[:, None, :], shape).to(y_aux.dtype)
    left_aux = tl.reshape(left_aux, x.shape)
    right_aux = tl.reshape(right_aux, x.shape)
    
    # Determine swap condition based on x
    cond = (left > right) != flip
    
    # Swap x using bitcast trick
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    ret_x = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    
    # Swap ids
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    
    # Swap aux using bitcast trick
    idtype_aux = tl.core.get_int_dtype(bitwidth=aux.dtype.primitive_bitwidth, signed=True)
    ileft_aux = left_aux.to(idtype_aux, bitcast=True)
    iright_aux = right_aux.to(idtype_aux, bitcast=True)
    iaux = aux.to(idtype_aux, bitcast=True)
    ret_aux = iaux ^ tl.where(cond, ileft_aux ^ iright_aux, tl.zeros_like(iaux))
    
    return ret_x.to(x.dtype, bitcast=True), new_ids, ret_aux.to(aux.dtype, bitcast=True)


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
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims, n_outer)
    return x, ids


@triton.jit
def _bitonic_merge_triple(
    x,
    ids,
    aux,
    stage: tl.constexpr,
    order: tl.constexpr,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
):
    """
    Bitonic merge operation for sorting three arrays simultaneously.
    Sorting is based on x, ids and aux follow the same permutation.
    """
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids, aux = _compare_and_swap_triple(x, ids, aux, flip, i + (n_dims - stage), n_dims, n_outer)
    return x, ids, aux

@triton.jit
def argsort(
    x,
    ids,
    n_dims: tl.constexpr,
    n_outer: tl.constexpr,
    descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    # iteratively run bitonic merge-sort steps
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims, n_outer)
    return x, ids