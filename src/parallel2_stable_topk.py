# -*- coding: utf-8 -*-
"""
htree (Hierarchical Tree for KV Cache and Sparse Attention) 的 Triton Kernel 实现 V2

与 parallel.py 的主要区别：
- parallel.py:  先累积所有候选节点 → 再减去 Top-K 贡献 
- parallel2.py: 先选出 Top-K 节点 → 只累积非 Top-K 节点 (与 naive 一致)

===============================================================================

Phase 1: Tree Building (与 parallel.py 相同)
  - 逐层 mean pooling 构建层次结构

Phase 2: Forward Pass (逐层处理，从顶层到底层)

  【Kernel 2.1: Compute Scores & Select Top-K】
  输入:  q, layer_k, prev_selected_parents, cos_cache, sin_cache
  处理:
    1. 批次遍历候选节点（32父×16子=512/批）
    2. 计算 attention scores（Q @ K^T with RoPE）
    3. 存储到固定 8192 buffer: all_scores[8192], all_node_indices[8192]
    4. 非底层: 给 rightmost 节点 score 置为 1e3，按 score 降序排序选 Top-512
    5. Top-512 的全局索引升序排序后传给下一层
    6. 底层: 跳过排序
  输出:  all_scores, all_node_indices, topk_positions, selected_indices

    【Kernel 2.2: Accumulate Non-TopK】
    输入:  layer_v, all_scores, num_candidates, topk_positions
  处理:
    1. 按原批次顺序加载 V（每批 16 个连续）
    2. 创建 Top-K mask（标记 buffer 中 Top-512 的位置）
    3. 最终 mask = valid_mask & (~topk_mask)
    4. 对非 Top-K 候选节点进行 online softmax 累积
    5. 底层: 累积所有候选节点（无 Top-K mask）
  输出:  layer_max, layer_sum, layer_output

  【Kernel 2.3: Merge to Global】(与 parallel.py 相同)
    - 在线 Softmax 合并: (global_state, layer_state) → global_state

Phase 3: Final Normalization
  - output = global_output / global_sum

===============================================================================
"""

from typing import Optional
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# ==========================================
# Kernel 1: Tree Building (复用 parallel.py)
# ==========================================

@triton.jit
def htree_build_kernel_v2(
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
    """
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H
    
    num_iterations = tl.cdiv(N_parent, BLOCK_SIZE)
    
    for iter_idx in range(num_iterations):
        parent_start = iter_idx * BLOCK_SIZE
        child_start = parent_start * COMPRESSION_RATE
        
        # load K
        k_base = child_k + i_b * N_child * H * K + i_h * K
        k_block_ptrs = tl.make_block_ptr(
            base=k_base,
            shape=(N_child, K),
            strides=(H * K, 1),
            offsets=(child_start, 0),
            block_shape=(BLOCK_SIZE * COMPRESSION_RATE, K),
            order=(1, 0)
        )
        k_vals = tl.load(k_block_ptrs, boundary_check=(0, 1))
        
        # load V
        v_base = child_v + i_b * N_child * H * V + i_h * V
        v_block_ptrs = tl.make_block_ptr(
            base=v_base,
            shape=(N_child, V),
            strides=(H * V, 1),
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
# 辅助函数 (复用 parallel.py)
# ==========================================

@triton.jit
def load_k_with_rope_v2(
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
    """加载 16 个 token 的 K 并应用 RoPE 编码"""
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
    k1 = tl.load(k1_block_ptrs, boundary_check=(0, 1))
    
    # 后半部分
    k2_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_layer, K),
        strides=(H * K, 1),
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
    i_h,
    child_start,
    N_layer: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """加载 16 个 token 的 V"""
    v_base = layer_v + i_b * N_layer * H * V + i_h * V
    
    v_block_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(N_layer, V),
        strides=(H * V, 1),
        offsets=(child_start, 0),
        block_shape=(COMPRESSION_RATE, V),
        order=(1, 0)
    )
    v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))
    
    return v_vals


# ==========================================
# Kernel 2.1: Compute Scores & Select Top-K
# ==========================================

@triton.jit
def htree_compute_scores_and_select_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H, K]
    prev_selected_parents,  # [B, T, H, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # 输出 buffer
    all_scores,  # [B, T, H, MAX_CANDIDATES]
    all_node_indices,  # [B, T, H, MAX_CANDIDATES]
    num_candidates,  # [B, T, H]
    # 输出 Top-K
    topk_positions,  # [B, T, H, TOP_K]
    selected_indices,  # [B, T, H, TOP_K]
    # 参数
    layer_idx: tl.constexpr,
    layer_power: tl.constexpr,
    B: tl.constexpr, T, H: tl.constexpr,
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
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    BC: tl.constexpr = TOP_K  # 512
    PARENTS_PER_BATCH: tl.constexpr = 32
    
    # ========================================
    # 阶段 1: 确定候选节点范围
    # ========================================
    
    rightmost_idx = i_t // layer_power
    
    prev_sel_base = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
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
    num_cand_offset = i_b * T * H + i_t * H + i_h
    tl.store(num_candidates + num_cand_offset, n_cand)
    
    # ========================================
    # 阶段 2: 应用 RoPE 到 Query
    # ========================================
    
    rope_pos_q = tl.maximum(n_cand - 1, 0)
    
    q_offset = i_b * T * H * K + i_t * H * K + i_h * K
    o_k = tl.arange(0, K // 2)
    
    q1 = tl.load(q + q_offset + o_k)
    q2 = tl.load(q + q_offset + (K // 2) + o_k)
    
    cos_q = tl.load(cos_cache + rope_pos_q * (K // 2) + o_k)
    sin_q = tl.load(sin_cache + rope_pos_q * (K // 2) + o_k)
    
    q_rope_1 = (q1 * cos_q - q2 * sin_q) * scale
    q_rope_2 = (q1 * sin_q + q2 * cos_q) * scale
    
    # ========================================
    # 阶段 3: 批次遍历，加载 K 并计算 scores
    # ========================================
    
    buffer_base = i_b * T * H * MAX_CANDIDATES + i_t * H * MAX_CANDIDATES + i_h * MAX_CANDIDATES
    
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
        
        # 初始化当前批次的结果
        scores_2d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], -1e10, dtype=tl.float32)
        node_indices_2d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], -1, dtype=tl.int32)
        
        # 遍历 32 个父节点
        for i_p in range(PARENTS_PER_BATCH):
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                rope_pos_start = i_batch * BC + i_p * COMPRESSION_RATE
                
                # 提取当前父节点的 valid mask
                row_selector = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)
                current_parent_mask = tl.sum(
                    tl.where(row_selector, valid_child_mask_2d.to(tl.int32), 0),
                    axis=0
                ).to(tl.int1)
                num_valid_children = tl.sum(current_parent_mask.to(tl.int32))
                
                # 加载 K 并应用 RoPE
                k_rope_1, k_rope_2 = load_k_with_rope_v2(
                    layer_k, cos_cache, sin_cache,
                    i_b, i_h, child_start, rope_pos_start,
                    N_layer, H, K, COMPRESSION_RATE, num_valid_children
                )
                
                # 计算 scores
                scores_16 = tl.sum(q_rope_1[None, :] * k_rope_1, axis=1) + \
                           tl.sum(q_rope_2[None, :] * k_rope_2, axis=1)
                
                # 计算全局节点索引
                node_indices_16 = child_start + tl.arange(0, COMPRESSION_RATE)
                
                # 填充到 2D 结果
                is_current_parent = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)
                scores_2d = tl.where(
                    is_current_parent & valid_child_mask_2d,
                    scores_16[None, :],
                    scores_2d
                )
                node_indices_2d = tl.where(
                    is_current_parent & valid_child_mask_2d,
                    node_indices_16[None, :],
                    node_indices_2d
                )
        
        # Flatten 到 [BC=512]
        batch_scores = tl.reshape(scores_2d, [BC])
        batch_node_indices = tl.reshape(node_indices_2d, [BC])
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        
        # 应用 mask
        batch_scores = tl.where(valid_child_mask_flat, batch_scores, -1e10)
        batch_node_indices = tl.where(valid_child_mask_flat, batch_node_indices, -1)
        
        # 存储到 8192 buffer
        buffer_offset = i_batch * BC
        o_bc = tl.arange(0, BC)
        store_mask = (buffer_offset + o_bc) < MAX_CANDIDATES
        
        tl.store(all_scores + buffer_base + buffer_offset + o_bc, batch_scores, mask=store_mask)
        tl.store(all_node_indices + buffer_base + buffer_offset + o_bc, batch_node_indices, mask=store_mask)
    
    # ========================================
    # 阶段 4: Top-K 选择（非底层）- Bit-Packing 优化版
    # ========================================
    
    if not is_bottom_layer:
        # 4.1 加载 buffer 中的所有 scores 和 node_indices
        o_buf = tl.arange(0, MAX_CANDIDATES)
        buffer_scores = tl.load(all_scores + buffer_base + o_buf)
        buffer_node_indices = tl.load(all_node_indices + buffer_base + o_buf)

        # 4.1.1 仅保留当前批次真正写入且有效的节点
        written_limit = num_batches * BC
        written_mask = o_buf < written_limit
        node_valid_mask = buffer_node_indices >= 0
        valid_buffer_mask = written_mask & node_valid_mask
        padding_value = -1e10  # finite padding to avoid NaNs inside sort
        buffer_scores = tl.where(valid_buffer_mask, buffer_scores, padding_value)
        buffer_node_indices = tl.where(valid_buffer_mask, buffer_node_indices, -1)

        # 4.2 给 rightmost 节点赋值 1e3（确保它一定被选中）
        is_rightmost = (buffer_node_indices == rightmost_idx)
        buffer_scores = tl.where(is_rightmost, 1e3, buffer_scores)
        
        # ================================================================
        # 4.3 Bit-Packing 编码：将 buffer 位置索引编码到 float32 低位
        # ================================================================
        # MAX_CANDIDATES = 8192，需要 13 bits 存储索引
        LOG_N: tl.constexpr = 13  # log2(8192) = 13
        idx_mask = (1 << LOG_N) - 1  # 0x1FFF
        
        # 将 float32 视为 int32 进行位操作
        buffer_scores_int = buffer_scores.to(tl.int32, bitcast=True)

        # 手动清空尾数的第14位（bit 13，从0开始计数）
        # bit_13_mask = ~(1 << 13)  # 创建一个只有 bit 13 为 0 的掩码
        # buffer_scores_int = buffer_scores_int & bit_13_mask
        
        # 编码索引：
        # - 正数：取反索引 (~idx)，使小索引在数值相同时排在前面（稳定排序）
        # - 负数：保持索引原值
        encoded_idx = tl.where(buffer_scores >= 0, ~o_buf, o_buf)
        encoded_idx = encoded_idx & idx_mask
        
        # 清除 float32 低 13 位，填入编码后的索引
        buffer_scores_encoded_int = (buffer_scores_int & ~idx_mask) | encoded_idx
        buffer_scores_encoded = buffer_scores_encoded_int.to(tl.float32, bitcast=True)
        
        # ================================================================
        # 4.3.1 回写编码后的分数到 all_scores（用于调试）
        # ================================================================
        # tl.store(all_scores + buffer_base + o_buf, buffer_scores_encoded)
        
        # ================================================================
        # 4.4 排序：使用单数组 bitonic sort（索引已编码在值中）
        # ================================================================
        n_dims: tl.constexpr = 13  # log2(8192) = 13
        n_outer: tl.constexpr = 1
        
        # 降序排序编码后的值
        sorted_scores_encoded = sort_single(
            buffer_scores_encoded, n_dims, n_outer, descending=True
        )
        
        # ================================================================
        # 4.5 Bit-Packing 解码：从排序结果中提取 buffer 位置和恢复分数
        # ================================================================
        # 转回 int32
        sorted_int = sorted_scores_encoded.to(tl.int32, bitcast=True)
        
        # 提取编码的索引（低 13 位）
        raw_idx = sorted_int & idx_mask
        
        # 清除低位恢复分数（用于判断正负）
        clean_int = sorted_int & ~idx_mask
        clean_scores = clean_int.to(tl.float32, bitcast=True)
        
        # 还原索引：正数时索引被取反了，需要再次取反
        topk_sort_positions = tl.where(clean_scores >= 0, ~raw_idx, raw_idx)
        topk_sort_positions = (topk_sort_positions & idx_mask).to(tl.int32)
        
        # ================================================================
        # 4.6 提取前 TOP_K 个位置
        # ================================================================
        # topk_sort_positions: [8192] -> reshape成 [16, 512] -> 取第一行 [512]
        topk_sort_positions_2d = tl.reshape(topk_sort_positions, [MAX_CANDIDATES // TOP_K, TOP_K])
        row_mask = (tl.arange(0, MAX_CANDIDATES // TOP_K)[:, None] == 0)
        topk_buffer_positions = tl.sum(tl.where(row_mask, topk_sort_positions_2d, 0), axis=0).to(tl.int32)  # [TOP_K]
        
        # ================================================================
        # 4.7 根据 topk_buffer_positions 提取全局节点索引
        # ================================================================
        gather_mask = topk_buffer_positions >= 0
        selected_node_indices = tl.load(
            all_node_indices + buffer_base + topk_buffer_positions,
            mask=gather_mask,
            other=-1,
        )

        # ================================================================
        # 4.8 对 selected_node_indices 进行升序排序（传给下一层）
        # ================================================================
        # 将 -1 替换为 MAX_IDX
        MAX_IDX: tl.constexpr = 2147483647
        selected_node_indices_sorted = tl.where(selected_node_indices >= 0, selected_node_indices, MAX_IDX)

        # 升序排序（这里仍用 argsort_v2，因为需要对 int32 排序）
        n_dims_topk: tl.constexpr = 9  # log2(512) = 9
        dummy_indices = tl.arange(0, TOP_K).to(tl.int32)
        selected_node_indices_sorted, _ = argsort_v2(
            selected_node_indices_sorted, dummy_indices, 
            n_dims_topk, 1, 
            descending=False
        )

        # 恢复 -1
        selected_node_indices_sorted = tl.where(
            selected_node_indices_sorted < MAX_IDX,
            selected_node_indices_sorted, -1
        )

        # ================================================================
        # 4.9 存储结果
        # ================================================================
        topk_offset = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
        o_topk = tl.arange(0, TOP_K)
        
        # topk_positions: 在 buffer 中的位置（乱序，用于创建 mask）
        tl.store(topk_positions + topk_offset + o_topk, topk_buffer_positions)
        # selected_indices: 全局节点索引（升序排序后）
        tl.store(selected_indices + topk_offset + o_topk, selected_node_indices_sorted.to(tl.int32))
    
    else:
        # 底层：跳过排序，输出 -1
        topk_offset = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
        o_topk = tl.arange(0, TOP_K)
        
        tl.store(topk_positions + topk_offset + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))
        tl.store(selected_indices + topk_offset + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))


# ==========================================
# Kernel 2.2: Accumulate Non-TopK (批次加载 V)
# ==========================================

@triton.jit
def htree_accumulate_non_topk_kernel(
    layer_v,  # [B, N_layer, H, V]
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
    V: tl.constexpr,
    N_layer: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
):
    """
    Kernel 2.2: 批次加载 V，累积非 Top-K 节点
    Grid: (T, B*H)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    BC: tl.constexpr = TOP_K  # 512
    PARENTS_PER_BATCH: tl.constexpr = 32
    
    rightmost_idx = i_t // layer_power
    
    # ========================================
    # 阶段 1: 加载元数据
    # ========================================
    
    num_cand_offset = i_b * T * H + i_t * H + i_h
    n_cand = tl.load(num_candidates + num_cand_offset)
    
    topk_offset = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
    o_topk = tl.arange(0, TOP_K)
    topk_pos_vals = tl.load(topk_positions + topk_offset + o_topk) # topk_pos_vals [TOP_K] 内容为 buffer 中的下标位置（乱序）
    
    buffer_base = i_b * T * H * MAX_CANDIDATES + i_t * H * MAX_CANDIDATES + i_h * MAX_CANDIDATES
    
    # ========================================
    # 阶段 2: 创建 Top-K mask（在 8192 buffer 中）
    # ========================================
    
    # 根据 topk_pos_vals 创建 mask，标记 buffer 中 Top-K 的位置
    o_buf = tl.arange(0, MAX_CANDIDATES)
    
    topk_mask = tl.zeros([MAX_CANDIDATES], dtype=tl.int1)
    TOPK_CHUNK: tl.constexpr = 128
    NUM_TOPK_CHUNKS: tl.constexpr = (TOP_K + TOPK_CHUNK - 1) // TOPK_CHUNK

    for chunk_idx in range(NUM_TOPK_CHUNKS):
        chunk_start = chunk_idx * TOPK_CHUNK
        o_chunk = chunk_start + tl.arange(0, TOPK_CHUNK)
        chunk_pos = tl.load(
            topk_positions + topk_offset + o_chunk,
            mask=o_chunk < TOP_K,
            other=-1,
        )
        # 块级广播比较，生成命中标记
        chunk_match = (o_buf[:, None] == chunk_pos[None, :])
        chunk_hit = tl.sum(chunk_match, axis=1).to(tl.int1)
        topk_mask = topk_mask | chunk_hit
    
    # ========================================
    # 阶段 3: 批次加载 V（按原顺序，每批 16 个）
    # ========================================
    
    # 创建 all_values buffer
    all_values = tl.zeros([MAX_CANDIDATES, V], dtype=tl.float32)
    
    prev_sel_base = i_b * T * H * TOP_K + i_t * H * TOP_K + i_h * TOP_K
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    valid_mask = parent_list >= 0
    num_valid_parents = tl.sum(valid_mask.to(tl.int32))
    num_batches = (num_valid_parents + 31) // 32

    for i_batch in range(num_batches):
        # 加载 32 个父节点索引
        o_parent_local = tl.arange(0, PARENTS_PER_BATCH)
        parent_indices = tl.load(
            prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + o_parent_local,
            mask=o_parent_local < PARENTS_PER_BATCH,
            other=-1
        )
        
        valid_parent_mask = parent_indices >= 0
        child_starts = parent_indices * COMPRESSION_RATE
        child_starts = tl.where(valid_parent_mask, child_starts, 0)
        
        # 判断最右侧父节点（用于 padding）
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
        
        # 初始化当前批次的 values
        values_2d = tl.zeros([PARENTS_PER_BATCH, COMPRESSION_RATE, V], dtype=tl.float32)
        
        # 遍历 32 个父节点，加载 V
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_idx = tl.load(prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p)
            
            if parent_idx >= 0:
                child_start = parent_idx * COMPRESSION_RATE
                
                # 加载 V [16, V]
                v_vals = load_v_v2(
                    layer_v, i_b, i_h, child_start,
                    N_layer, H, V, COMPRESSION_RATE
                )
                
                # 填充到 2D 结果
                is_current_parent = (tl.arange(0, PARENTS_PER_BATCH)[:, None] == i_p)
                values_2d = tl.where(
                    is_current_parent[:, :, None] & 
                    valid_child_mask_2d[:, :, None] &
                    (tl.arange(0, V)[None, None, :] < V),
                    v_vals[None, :, :],
                    values_2d
                )
        
        # Flatten 到 [BC, V]
        batch_values = tl.reshape(values_2d, [BC, V])
        valid_child_mask_flat = tl.reshape(valid_child_mask_2d, [BC])
        
        # 应用 mask
        batch_values = tl.where(valid_child_mask_flat[:, None], batch_values, 0.0)
        
        # 存储到 all_values buffer
        buffer_offset = i_batch * BC
        
        for i_bc in range(BC):
            store_pos = buffer_offset + i_bc
            if store_pos < MAX_CANDIDATES:
                # 提取第 i_bc 行
                row_mask = (tl.arange(0, BC)[:, None] == i_bc)
                selected_row = tl.sum(tl.where(row_mask, batch_values, 0.0), axis=0)
                
                # 存储到 all_values[store_pos, :]
                store_mask_2d = (o_buf[:, None] == store_pos) & (tl.arange(0, V)[None, :] < V)
                all_values = tl.where(store_mask_2d, selected_row[None, :], all_values)

    # ========================================
    # 阶段 4: 加载 scores 并应用 mask
    # ========================================
    
    buffer_scores = tl.load(all_scores + buffer_base + o_buf)
    
    # 有效候选节点 mask
    valid_mask = o_buf < n_cand
    
    # 最终 mask：有效 且 非 top-k
    if not is_bottom_layer:
        final_mask = valid_mask & (~topk_mask)
    else:
        final_mask = valid_mask
    
    masked_scores = tl.where(final_mask, buffer_scores, -1e10)
    masked_values = tl.where(final_mask[:, None], all_values, 0.0)
    
    # ========================================
    # 阶段 5: Online Softmax 累积
    # ========================================
    
    cur_max = tl.max(masked_scores)
    
    exp_scores = tl.exp(masked_scores - cur_max)
    exp_scores = tl.where(final_mask, exp_scores, 0.0)
    
    cur_sum = tl.sum(exp_scores)
    cur_output = tl.sum(exp_scores[:, None] * masked_values, axis=0)
    
    # ========================================
    # 阶段 6: 存储结果
    # ========================================
    
    state_offset = i_b * T * H + i_t * H + i_h
    tl.store(layer_max + state_offset, cur_max)
    tl.store(layer_sum + state_offset, cur_sum)
    
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
    o_v = tl.arange(0, V)
    tl.store(layer_output + output_offset + o_v, cur_output, mask=o_v < V)


# ==========================================
# Kernel 2.3 & Final Normalization (复用 parallel.py)
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
    
    state_offset = i_b * T * H + i_t * H + i_h
    
    cur_max = tl.load(layer_max + state_offset)
    cur_sum = tl.load(layer_sum + state_offset)
    
    output_offset = i_b * T * H * V + i_t * H * V + i_h * V
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
# 排序辅助函数 (Bit-Packing Top-K 优化版)
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


# -------------------- 双数组排序 (保留用于 selected_indices 升序排序) --------------------

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
    htree 前向传播 V2 (基于 naive.py 的思路)
    
    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
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
    
    layers_k = [k]
    layers_v = [v]
    
    current_k, current_v = k, v
    current_len = T
    
    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H, V, dtype=dtype, device=device)
        
        BLOCK_SIZE = 128
        grid = (B * H,)
        htree_build_kernel_v2[grid](
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
    all_node_indices = torch.empty([B, T, H, MAX_CANDIDATES], dtype=torch.int32, device=device)
    num_candidates = torch.empty([B, T, H], dtype=torch.int32, device=device)
    topk_positions = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    selected_indices = torch.empty([B, T, H, top_k_per_layer], dtype=torch.int32, device=device)
    
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
        htree_compute_scores_and_select_kernel[grid](
            q, k_layer,
            prev_selected_parents,
            cos_cache, sin_cache,
            all_scores, all_node_indices, num_candidates,
            topk_positions, selected_indices,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, K=K, V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
            scale=scale,
        )
        nvtx.range_pop()
        
        # Kernel 2.2: Accumulate Non-TopK
        nvtx.range_push("K2.2_AccumulateNonTopK")
        print("    Running accumulate non-topk kernel...")
        htree_accumulate_non_topk_kernel[grid](
            v_layer,
            prev_selected_parents,
            all_scores, num_candidates,
            topk_positions,
            layer_max, layer_sum, layer_output,
            layer_idx=layer_idx,
            layer_power=layer_power,
            B=B, T=T, H=H, V=V, N_layer=N_layer,
            COMPRESSION_RATE=compression_rate,
            TOP_K=top_k_per_layer,
            MAX_CANDIDATES=MAX_CANDIDATES,
        )
        nvtx.range_pop()
        
        # Kernel 2.3: Merge to Global State
        nvtx.range_push("K2.3_MergeToGlobal")
        print("    Running merge to global kernel...")
        htree_merge_to_global_kernel_v2[grid](
            layer_max, layer_sum, layer_output,
            global_max, global_sum, global_output,
            B=B, T=T, H=H, V=V,
        )
        nvtx.range_pop()

        # 更新 parent indices
        if not is_bottom_layer:
            prev_selected_parents, selected_indices = selected_indices, prev_selected_parents
        
        nvtx.range_pop()
    nvtx.range_pop()

    # ========== Phase 4: Final Normalization ==========
    nvtx.range_push("Phase4_Final_Normalize")
    print("Phase 4: Final normalization...")
    
    output = torch.empty(B, T, H, V, dtype=dtype, device=device)
    grid = (T, B * H)
    htree_final_normalize_kernel_v2[grid](
        global_output, global_sum, output,
        B=B, T=T, H=H, V=V,
    )
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

