# -*- coding: utf-8 -*-
"""
htree Triton Kernel

Pipeline
  Phase 1: Tree Building
    - Kernel 1: `htree_build_kernel_v2`
      · 对 (K,V) 逐层 mean pooling, 构建从底层到顶层的 (layers_k, layers_v)

  Phase 1.5: RoPE Cache
    - 在 host 侧预计算 `cos_cache/sin_cache` (供所有层复用)

  Phase 2: Init Global State & Workspaces
    - 初始化全局 online-softmax 状态: `global_max/global_sum/global_output`
    - 分配每层复用的 workspace: 
      · `all_scores`: [B, T, H, MAX_CANDIDATES(=8192)] —— 候选 buffer (仅存 score)
      · `num_candidates`: [B, T, H_kv] —— KV head 粒度共享的候选数量 n_cand
      · `prev_selected_parents`: [B, T, H_kv, TOP_K] —— 下一层展开所需的 parent 列表 (升序)

  Phase 3: Layer-by-layer Forward (top → bottom)
    - Kernel 2.1a `htree_compute_scores_kernel` (scores only)
      · 对每个 (b,t,h) 计算其候选集合的 score, 写入 `all_scores[b,t,h,0:n_cand]`
      · 候选集合由 `prev_selected_parents[b,t,h_kv,:]` 给出 (KV head 粒度共享)
      · 采用固定 tile: 每批 32 parents * COMPRESSION_RATE(通常 16) children = 512

    - Kernel 2.1b `htree_select_topk_shared_gqa_kernel` (non-bottom layers only)
      · 对每个 (b,t,h_kv) 在同组 NUM_GROUPS 个 query heads 上做 NSA-style 聚合: 
                importance_i = Σ_g exp(score_{g,i} - lse_g)
      · 强制选中 “rightmost” 候选 (pos=n_cand-1)以保证因果边界一致
      · stable Top-K: 对 importance 做 bit-packing (把 buffer pos 编进低位), 再用 bitonic sort 选 Top-K
      · 输出 `topk_positions[b,t,h_kv,:]` (buffer position, -1 padding)

    - Kernel 2.1.2 `htree_mask_topk_scores_kernel` (non-bottom layers only)
      · 将 Top-K 对应 buffer 位置的 score 覆写为 `HTREE_SCORE_NEG_INF` (对同组所有 query heads 生效)
      · 目的: Kernel 2.2 只需用阈值判断 `score_valid`, 避免额外的 topk_hit 广播比较

    - Kernel 2.2 `htree_accumulate_non_topk_kernel`
      · 流式加载 V, 累积非 Top-K 节点 (底层累积全部候选)

    - Kernel 2.3 `htree_merge_to_global_kernel_v2`
      · online-softmax 合并: 把当前层 `(layer_max, layer_sum, layer_output)` 合并到全局状态

    - Kernel 2.4 `htree_compute_next_parents_kernel` (non-bottom layers only)
      · 把 `topk_positions` (buffer pos)映射回下一层的 node indices, 并做升序排序
      · 得到下一层的 `prev_selected_parents`

  Phase 4: Final Normalize
    - Kernel `htree_final_normalize_kernel_v2`
      · `output = global_output / global_sum`
"""

from typing import Optional
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

# ==========================================
# 全局常量 (数值约定)
# ==========================================

# 作为“负无穷/被屏蔽”的统一 sentinel 分数。
# 关系: 所有被屏蔽/无效位置的 score 必须 <= HTREE_SCORE_NEG_INF。
HTREE_SCORE_NEG_INF: float = -1.0e10

# 判定 score 是否“有效”的阈值。
# 关系: HTREE_SCORE_VALID_THRESHOLD 必须严格大于 HTREE_SCORE_NEG_INF, 
# 从而能用 `score > HTREE_SCORE_VALID_THRESHOLD` 区分出被屏蔽(-1e10)的项。
# 这里沿用原实现的经验值: -0.9e10。
HTREE_SCORE_VALID_THRESHOLD: float = HTREE_SCORE_NEG_INF * 0.9

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


# ==========================================
# Kernel 2.1: Compute Scores & Select Top-K
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=2, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=3),
#         triton.Config({}, num_warps=8, num_stages=2),
#         triton.Config({}, num_warps=8, num_stages=3),
#     ],
#     # Key by head grouping + rope/compression structure + bottom-vs-non-bottom.
#     # Avoid keying on T to prevent excessive cache fragmentation.
#     key=['K', 'NUM_GROUPS', 'COMPRESSION_RATE', 'TOP_K', 'layer_idx'],
# )
@triton.jit
def htree_compute_scores_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
     prev_selected_parents,  # [B, T_blocks, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
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
    Kernel 2.1 (scores only): 计算所有候选节点的 attention scores, 写入 all_scores。
    - parents / num_candidates 以 KV head 粒度共享 (H_kv), 同组 query heads 共享候选集合。
    - all_scores layout: all_scores[b, t, h_kv, pos, g]
    Grid: (T, B*H_kv)
    """
    i_t = tl.program_id(0)
    i_bhk = tl.program_id(1)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv
    
    PARENTS_PER_BATCH: tl.constexpr = 32
    # Per-batch candidates: 32 parents * COMPRESSION_RATE children.
    # This is 512 when COMPRESSION_RATE=16.
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
    # KV-granularity: one program owns (b, t, h_kv), so no write conflict.
    tl.store(num_candidates + num_cand_offset, n_cand)
    
    # ========================================
    # 阶段 2: 应用 RoPE 到 Query
    # ========================================
    
    rope_pos_q = tl.maximum(n_cand - 1, 0)
    
    # Vectorize over the NUM_GROUPS query heads in this KV group.
    head_ids = (i_h_kv * NUM_GROUPS + tl.arange(0, NUM_GROUPS)).to(tl.int64)  # [G]
    q_bt_base = (
        i_b.to(tl.int64) * T_i64 * H * K
        + i_t.to(tl.int64) * H * K
    )
    o_k = tl.arange(0, K // 2).to(tl.int64)[None, :]  # [1, K/2]
    q_head_base = q + q_bt_base + head_ids[:, None] * K  # [G, 1] pointer
    q1 = tl.load(q_head_base + o_k)  # [G, K/2]
    q2 = tl.load(q_head_base + (K // 2) + o_k)  # [G, K/2]
    
    rope_pos_q_i64 = rope_pos_q.to(tl.int64)
    cos_q = tl.load(cos_cache + rope_pos_q_i64 * (K // 2) + tl.arange(0, K // 2))  # [K/2]
    sin_q = tl.load(sin_cache + rope_pos_q_i64 * (K // 2) + tl.arange(0, K // 2))  # [K/2]
    
    q_rope_1 = (q1 * cos_q[None, :] - q2 * sin_q[None, :]) * scale  # [G, K/2]
    q_rope_2 = (q1 * sin_q[None, :] + q2 * cos_q[None, :]) * scale  # [G, K/2]
    
    # ========================================
    # 阶段 3: 批次遍历, 加载 K 并计算 scores, 同时维护 Streaming Top-K
    # ========================================

    # NOTE: all_scores is [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]. When
    # T*H_kv*MAX_CANDIDATES*NUM_GROUPS exceeds 2^31-1, int32 offset math overflows.
    # Promote to int64 explicitly.
    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )
    
    # 3.1 初始化 Streaming Top-K 容器 (encoded scores)
    # NOTE: Top-K selection is handled by a separate shared-topk kernel (Kernel 2.1b).
    
    for i_batch in range(num_batches):
        # 加载 32 个父节点索引
        o_parent_local = tl.arange(0, PARENTS_PER_BATCH)
        parent_indices = tl.load(
            prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + o_parent_local,
            mask=o_parent_local < PARENTS_PER_BATCH,
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
        
        # rightmost mask (用于强制选中最右节点)
        rightmost_mask_2d = (
            (parent_indices[:, None] == rightmost_parent_idx)
            & (o_child_offset == rightmost_child_idx)
            & valid_child_mask_2d
        )

        # 初始化当前批次的结果: [parents, children, groups]
        scores_3d = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE, NUM_GROUPS], -1e10, dtype=tl.float32)
        
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
                
                # 计算 scores for all NUM_GROUPS heads: [G, CR]
                scores_gcr = tl.sum(q_rope_1[:, None, :] * k_rope_1[None, :, :], axis=2) + \
                             tl.sum(q_rope_2[:, None, :] * k_rope_2[None, :, :], axis=2)
                # transpose to [CR, G] for convenient broadcasting into [P, CR, G]
                scores_crg = tl.trans(scores_gcr)
                
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
        
        # 3.2 存储到 8192 buffer (供 Kernel 2.2 使用)
        buffer_offset = i_batch * BC
        o_bc = tl.arange(0, BC)
        pos = buffer_offset + o_bc
        store_mask = pos < MAX_CANDIDATES
        
        pos_i64 = pos.to(tl.int64)
        g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)
        ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
        tl.store(ptrs, batch_scores, mask=store_mask[:, None])
    # No Top-K output here.


# ==========================================
# Kernel 2.1b: Shared Top-K Selection (NSA-style, per KV head)
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=2, num_stages=1),
#         triton.Config({}, num_warps=2, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=1),
#         triton.Config({}, num_warps=4, num_stages=2),
#         triton.Config({}, num_warps=8, num_stages=1),
#         triton.Config({}, num_warps=8, num_stages=2),
#     ],
#     key=['NUM_GROUPS', 'COMPRESSION_RATE', 'TOP_K', 'layer_idx'],
# )
@triton.jit
def htree_select_topk_shared_gqa_kernel(
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
    SCORE_VALID_THRESHOLD: tl.constexpr,
):
    """Kernel 2.1b: NSA-style shared Top-K selection per KV head.

    For a (b, t, h_kv), we aggregate across the G=NUM_GROUPS query heads:
      lse_g = log(sum_i exp(score_{g,i}))
      importance_i = sum_g exp(score_{g,i} - lse_g)
    Then pick Top-K (stable) over i in [0, n_cand).

    Output is `topk_positions[b,t,h_kv,:]` (buffer positions in [0, MAX_CANDIDATES), -1 padded).

    Grid: (T, B*H_kv)
    """
    i_t = tl.program_id(0)
    i_bhk = tl.program_id(1)
    i_b = i_bhk // H_kv      # Batch Index
    i_h_kv = i_bhk % H_kv    # KV Head Index in Batch

    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    if is_bottom_layer:
        # Bottom layer: no need to select for next layer.
        T_i64 = T.to(tl.int64)
        base_out = (
            i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
            + i_t.to(tl.int64) * H_kv * TOP_K
            + i_h_kv.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        tl.store(topk_positions + base_out + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))
        return

    # Load n_cand (KV-head granularity)
    T_i64 = T.to(tl.int64)
    num_cand_off = (
        i_b.to(tl.int64) * T_i64 * H_kv
        + i_t.to(tl.int64) * H_kv
        + i_h_kv.to(tl.int64)
    )
    n_cand = tl.load(num_candidates + num_cand_off).to(tl.int32)

    # Handle empty candidate list defensively
    if n_cand <= 0:
        base_out = (
            i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
            + i_t.to(tl.int64) * H_kv * TOP_K
            + i_h_kv.to(tl.int64) * TOP_K
        )
        o_topk = tl.arange(0, TOP_K)
        tl.store(topk_positions + base_out + o_topk, tl.full([TOP_K], -1, dtype=tl.int32))
        return

    # This implementation assumes a fixed per-batch candidate tile of 32*COMPRESSION_RATE (512 when rate=16).
    PARENTS_PER_BATCH: tl.constexpr = 32
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE

    # Base for all_scores: [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )
    g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)  # [G]

    # ------------------------------------------------
    # Pass 1: compute lse_g for each head in the group
    # ------------------------------------------------
    m = tl.full([NUM_GROUPS], float('-inf'), dtype=tl.float32)
    acc = tl.zeros([NUM_GROUPS], dtype=tl.float32)

    num_batches = (n_cand + BC - 1) // BC
    for i_batch in range(num_batches):
        buf_off = i_batch * BC
        o_bc = tl.arange(0, BC)
        pos = buf_off + o_bc
        valid_pos = pos < n_cand
        pos_i64 = pos.to(tl.int64)
        s_ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
        b_s = tl.load(s_ptrs, mask=valid_pos[:, None], other=float('-inf')).to(tl.float32)  # [BC, G]

        b_m = tl.max(b_s, axis=0)  # [G]
        m_new = tl.maximum(m, b_m)
        r = tl.exp(m - m_new)

        p = tl.exp(b_s - m_new[None, :])
        p = tl.where(valid_pos[:, None], p, 0.0)
        acc = acc * r + tl.sum(p, axis=0)
        m = m_new

    lse = m + tl.log(acc)  # [G]

    # ------------------------------------------------
    # Pass 2: compute importance per position and pick Top-K (stable)
    # ------------------------------------------------
    running_topk_encoded = tl.full([BC], float('-inf'), dtype=tl.float32)

    LOG_N: tl.constexpr = 13  # log2(8192)
    idx_mask = (1 << LOG_N) - 1

    rightmost_pos = (n_cand - 1).to(tl.int32)

    for i_batch in range(num_batches):
        buf_off = i_batch * BC
        o_bc = tl.arange(0, BC)
        pos = (buf_off + o_bc).to(tl.int32)
        valid_pos = pos < n_cand
        pos_i64 = pos.to(tl.int64)
        s_ptrs = all_scores + scores_base + pos_i64[:, None] * NUM_GROUPS + g_ids[None, :]
        b_s = tl.load(s_ptrs, mask=valid_pos[:, None], other=float('-inf')).to(tl.float32)  # [BC, G]

        b_p = tl.exp(b_s - lse[None, :])  # [BC, G]
        b_p = tl.where(valid_pos[:, None], b_p, 0.0)
        imp = tl.sum(b_p, axis=1)  # [BC]
        imp = tl.where(valid_pos, imp, float('-inf'))
        # force rightmost selected
        imp = tl.where(pos == rightmost_pos, 1e3, imp)

        # Bit-pack buffer position into imp for stability (ties -> smaller idx first).
        buf_idx = pos  # 0..MAX_CANDIDATES-1
        imp_int = imp.to(tl.int32, bitcast=True)
        encoded_idx = tl.where(imp >= 0, ~buf_idx, buf_idx) & idx_mask
        encoded_int = (imp_int & ~idx_mask) | encoded_idx
        imp_encoded = encoded_int.to(tl.float32, bitcast=True)

        # Sort per-batch (512) then merge into running topk (512) via sort on 1024
        n_dims_batch: tl.constexpr = 9  # log2(512)
        sorted_batch = sort_single(imp_encoded, n_dims_batch, 1, descending=True)

        if i_batch == 0:
            running_topk_encoded = sorted_batch
        else:
            running_b = tl.broadcast_to(running_topk_encoded[None, :], [2, BC])
            batch_b = tl.broadcast_to(sorted_batch[None, :], [2, BC])
            row_idx = tl.arange(0, 2)[:, None]
            merged_2d = tl.where(row_idx == 0, running_b, batch_b)
            merged_input = tl.reshape(merged_2d, [2 * BC])

            n_dims_merge: tl.constexpr = 10  # log2(1024)
            sorted_merged = sort_single(merged_input, n_dims_merge, 1, descending=True)
            reshaped_merged = tl.reshape(sorted_merged, [2, BC])
            mask_row0 = (tl.arange(0, 2)[:, None] == 0)
            running_topk_encoded = tl.sum(tl.where(mask_row0, reshaped_merged, 0.0), axis=0)

    # Decode indices and store TOP_K
    sorted_int = running_topk_encoded.to(tl.int32, bitcast=True)
    raw_idx = sorted_int & idx_mask
    clean_int = sorted_int & ~idx_mask
    clean_scores = clean_int.to(tl.float32, bitcast=True)

    topk_buf_pos = tl.where(clean_scores >= 0, ~raw_idx, raw_idx)
    topk_buf_pos = (topk_buf_pos & idx_mask).to(tl.int32)
    is_valid = clean_scores > SCORE_VALID_THRESHOLD
    topk_buf_pos = tl.where(is_valid, topk_buf_pos, -1)

    base_out = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    o_topk = tl.arange(0, TOP_K)
    tl.store(topk_positions + base_out + o_topk, topk_buf_pos.to(tl.int32))


# ==========================================
# Post Kernel: Compute Next-Layer Parents
# ==========================================

@triton.jit
def htree_compute_next_parents_kernel(
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    topk_positions,  # [B, T, H_kv, TOP_K]  (buffer positions)
    next_selected_parents,  # [B, T, H_kv, TOP_K]
    B: tl.constexpr,
    T,
    H_kv: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
):
    """Compute next layer's prev_selected_parents (KV head granularity) from (prev_selected_parents, topk_positions).

    Grid: (T, B*H_kv)
    """
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H_kv
    i_h_kv = i_bh % H_kv

    PARENTS_PER_BATCH: tl.constexpr = 32
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE  # 512 when COMPRESSION_RATE=16

    T_i64 = T.to(tl.int64)
    base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
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

    n_dims_topk: tl.constexpr = 9  # log2(512); TOP_K must be power-of-2 and currently assumed 512
    dummy = tl.arange(0, TOP_K).to(tl.int32)
    selected_sorted, _ = argsort_v2(selected_sorted, dummy, n_dims_topk, 1, descending=False)
    selected_sorted = tl.where(selected_sorted < MAX_IDX, selected_sorted, -1)

    tl.store(next_selected_parents + base + o_topk, selected_sorted.to(tl.int32))


# ==========================================
# Kernel 2.1.2: Mask Top-K Scores in Buffer
# ==========================================

@triton.jit
def htree_mask_topk_scores_kernel(
    all_scores,  # [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    topk_positions,  # [B, T, H_kv, TOP_K] (buffer positions, shared per KV head)
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    """Mask Top-K scores by overwriting them with NEG_INF.

    Grid: (T, B*H_kv)
    """
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

    # Spread the shared mask to all query heads in this KV group.
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
# Kernel 2.2: Accumulate Non-TopK (批次加载 V)
# ==========================================

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=4, num_stages=2),
#         triton.Config({}, num_warps=8, num_stages=2),
#         triton.Config({}, num_warps=4, num_stages=3),
#         triton.Config({}, num_warps=8, num_stages=3),
#     ],
#     # Separate autotune decisions by value dim, group size, and bottom-vs-non-bottom.
#     key=['V', 'NUM_GROUPS', 'layer_idx'],
# )
@triton.jit
def htree_accumulate_non_topk_kernel(
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    # Kernel 2.1 的输出
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
    Kernel 2.2 (GQA, KV-granularity, V-blocked): stream-load V and accumulate non-TopK nodes.

    - Grid: (T, nVBlocks, B*H_kv)
      where nVBlocks = ceil_div(V, BV).
    - Each program handles one (b, t, h_kv) and one V slice [i_v*BV:(i_v+1)*BV),
      and processes NUM_GROUPS query heads in the GQA group in a vectorized manner.
    - Only i_v == 0 writes `layer_max/layer_sum` to avoid write conflicts; all i_v write `layer_output`.
    """
    i_t = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bhk = tl.program_id(2)
    i_b = i_bhk // H_kv
    i_h_kv = i_bhk % H_kv
    
    is_bottom_layer: tl.constexpr = (layer_idx == 0)
    PARENTS_PER_BATCH: tl.constexpr = 32
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE  # 512 when COMPRESSION_RATE=16

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

    # Head ids in this GQA group (query-head granularity)
    g_ids = tl.arange(0, NUM_GROUPS).to(tl.int64)  # [G]
    head_ids = (i_h_kv * NUM_GROUPS + g_ids).to(tl.int64)  # [G]

    # Base for all_scores: [B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS]
    scores_base = (
        (i_b.to(tl.int64) * T_i64 * H_kv + i_t.to(tl.int64) * H_kv + i_h_kv.to(tl.int64))
        * MAX_CANDIDATES
        * NUM_GROUPS
    )

    # V-slice for this program
    v_start = (i_v * BV).to(tl.int64)
    o_v = tl.arange(0, BV).to(tl.int64)
    v_cols = v_start + o_v  # [BV]
    v_mask = v_cols < V

    # ========================================
    # 阶段 2: 流式加载 V 并累积 (Stream Accumulate, vectorized over G)
    # ========================================

    # Online-softmax state per query head in this GQA group.
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

    # use a runtime loop for i_batch to avoid compile-time unrolling explosion.
    for i_batch in range(num_batches):
        # Iterate over 32 parents in this batch (fixed tile).
        for i_p in tl.static_range(PARENTS_PER_BATCH):
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + i_batch * PARENTS_PER_BATCH + i_p
            ).to(tl.int32)

            valid_parent = parent_idx >= 0

            # Child range in this parent
            child_start = (parent_idx * COMPRESSION_RATE).to(tl.int32)
            child_rows_i32 = child_start + tl.arange(0, COMPRESSION_RATE).to(tl.int32)  # [CR]
            row_mask = child_rows_i32 < N_layer

            # TODO use make_block_ptr to load V
            # Load V slice [COMPRESSION_RATE, BV] once (shared across G query heads).
            # layer_v is [B, N_layer, H_kv, V] contiguous.
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

            # Number of valid children for the rightmost parent.
            is_rightmost = valid_parent & (parent_idx == (rightmost_idx // COMPRESSION_RATE))
            num_valid_children = tl.where(
                is_rightmost,
                (rightmost_idx % COMPRESSION_RATE) + 1,
                COMPRESSION_RATE,
            ).to(tl.int32)
            child_valid_mask = row_mask & (tl.arange(0, COMPRESSION_RATE) < num_valid_children)  # [CR]

            # Global buffer positions for these children.
            global_pos_base = i_batch * BC + i_p * COMPRESSION_RATE
            global_pos = global_pos_base.to(tl.int32) + tl.arange(0, COMPRESSION_RATE).to(tl.int32)  # [CR]
            child_candidate_mask = valid_parent & child_valid_mask & (global_pos < n_cand)  # [CR]

            # Load scores for all G query heads from the re-laid out buffer: [CR, G]
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

            # block_max over CR children (per query head in the group)
            masked_for_max = tl.where(final_mask, scores, -1e10)
            batch_max = tl.max(masked_for_max, axis=0)  # [G]

            new_max = tl.maximum(cur_max, batch_max)  # [G]
            scale = tl.exp(cur_max - new_max)         # [G]

            # rescale historical accumulation to the new reference max
            cur_sum = cur_sum * scale
            cur_output = cur_output * scale[:, None]

            # p = exp(scores - new_max) (masked), accumulate directly
            scores_for_exp = tl.where(final_mask, scores, -1e10)
            p = tl.exp(scores_for_exp - new_max[None, :])  # [CR, G], invalid -> 0
            p = tl.where(final_mask, p, 0.0) # avoid exp(0)=1
            dp = tl.sum(p[:, :, None] * v_vals[:, None, :], axis=0)  # [G, BV]

            cur_sum = cur_sum + tl.sum(p, axis=0)
            cur_output = cur_output + dp
            cur_max = new_max

    # ========================================
    # 阶段 3: 存储结果
    # ========================================

    # Only one V-slice program writes scalar states to avoid write conflicts.
    if i_v == 0:
        state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
        tl.store(layer_max + state_base + head_ids, cur_max)
        tl.store(layer_sum + state_base + head_ids, cur_sum)

    # Store output slice for each query head in the group.
    out_base = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t.to(tl.int64) * H * V
    )
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + v_cols[None, :]
    tl.store(out_ptrs, cur_output, mask=v_mask[None, :])


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
        q: [B, T, H, K] - Query, H 个头
        k: [B, T, H_kv, K] - Key, H_kv 个头 (H_kv <= H, H % H_kv == 0)
        v: [B, T, H_kv, V] - Value, H_kv 个头
        compression_rate: 16
        max_top_nodes: 8192
        top_k_per_layer: 512
        scale: K^-0.5
        rope_base: 10000.0
    
    Returns:
        output: [B, T, H, V]
    
    Note:
    - 当 H_kv == H 时, 等价于 Multi-Head Attention (MHA)
    - 当 H_kv == 1 时, 等价于 Multi-Query Attention (MQA)
    - 当 1 < H_kv < H 时, 为 Group Query Attention (GQA)
    """
    B, T, H, K = q.shape
    H_kv = k.shape[2]  # KV 头数量
    V = v.shape[-1]
    
    # 验证 GQA 配置
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
    NUM_GROUPS = H // H_kv
    assert k.shape[2] == v.shape[2], f"K and V must have same number of heads"
    assert (top_k_per_layer & (top_k_per_layer - 1)) == 0, "top_k_per_layer (TOP_K) must be a power of 2"

    # This implementation uses a fixed per-batch candidate tile of 32*compression_rate.
    # top_k_per_layer controls how many nodes are expanded to the next layer.
    assert top_k_per_layer <= 32 * compression_rate, (
        f"top_k_per_layer ({top_k_per_layer}) must be <= 32*compression_rate ({32 * compression_rate})"
    )
    
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
    # Re-laid out to make GQA-group dimension contiguous for score reads:
    # all_scores[b, t, h_kv, pos, g] where g in [0, NUM_GROUPS).
    all_scores = torch.empty([B, T, H_kv, MAX_CANDIDATES, NUM_GROUPS], dtype=torch.float32, device=device)
    num_candidates = torch.empty([B, T, H_kv], dtype=torch.int32, device=device)
    topk_positions = torch.empty([B, T, H_kv, top_k_per_layer], dtype=torch.int32, device=device)
    
     # 为最顶层初始化 prev_selected_parents
    top_layer_power = compression_rate ** (num_layers - 1)
    
    t_indices = torch.arange(T, dtype=torch.int32, device=device)  # [T]
    rightmost_indices = t_indices // top_layer_power  # [T] 顶层最右侧节点索引
    # 虚拟父节点数量: 每个虚拟父节点展开成16个顶层节点
    num_virtual_parents = rightmost_indices // compression_rate + 1  # [T]
    
    # [T, TOP_K]
    parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
    valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
    prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
    prev_selected_parents = prev_selected_parents.unsqueeze(0).unsqueeze(2).expand(B, T, H_kv, top_k_per_layer).contiguous()
    
    for layer_idx in range(num_layers - 1, -1, -1):
        nvtx.range_push(f"Forward_Layer_{layer_idx}")
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        
        is_bottom_layer = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        
        print(f"  -> Processing layer {layer_idx} (N={N_layer}, power={layer_power}, bottom={is_bottom_layer})...")
        
        grid = (T, B * H)
        
        # Kernel 2.1a: Compute Scores
        nvtx.range_push("K2.1a_ComputeScores")
        print("    Running compute scores kernel...")
        
        torch.cuda.synchronize()
        start_k21 = torch.cuda.Event(enable_timing=True)
        end_k21 = torch.cuda.Event(enable_timing=True)
        start_k21.record()
        
        # Kernel 2.1a is KV-granularity: one program per (b, t, h_kv), computing NUM_GROUPS heads.
        grid_kv = (T, B * H_kv)
        htree_compute_scores_kernel[grid_kv](
            q, k_layer,
            prev_selected_parents,
            cos_cache, sin_cache,
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

        # Kernel 2.1b: Shared Top-K selection (KV head granularity, non-bottom layers only)
        if not is_bottom_layer:
            nvtx.range_push("K2.1b_SelectTopKSharedGQA")
            print("    Running shared topk selection kernel...")
            torch.cuda.synchronize()
            start_k21b = torch.cuda.Event(enable_timing=True)
            end_k21b = torch.cuda.Event(enable_timing=True)
            start_k21b.record()
            grid_kv = (T, B * H_kv)
            htree_select_topk_shared_gqa_kernel[grid_kv](
                all_scores,
                num_candidates,
                topk_positions,
                layer_idx=layer_idx,
                B=B, T=T, H=H, H_kv=H_kv,
                NUM_GROUPS=NUM_GROUPS,
                TOP_K=top_k_per_layer,
                MAX_CANDIDATES=MAX_CANDIDATES,
                COMPRESSION_RATE=compression_rate,
                SCORE_VALID_THRESHOLD=HTREE_SCORE_VALID_THRESHOLD,
            )
            end_k21b.record()
            torch.cuda.synchronize()
            time_k21b = start_k21b.elapsed_time(end_k21b)
            print(f"      Kernel 2.1b time: {time_k21b:.2f} ms")
            nvtx.range_pop()

        # Kernel 2.1.2: Mask Top-K scores in all_scores (non-bottom layers only)
        if not is_bottom_layer:
            nvtx.range_push("K2.1.2_MaskTopKScores")
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
        
        # Kernel 2.2: Accumulate Non-TopK
        nvtx.range_push("K2.2_AccumulateNonTopK")
        print("    Running accumulate non-topk kernel...")
        
        torch.cuda.synchronize()
        start_k22 = torch.cuda.Event(enable_timing=True)
        end_k22 = torch.cuda.Event(enable_timing=True)
        start_k22.record()

        # GQA-efficient K2.2: KV-granularity + V-blocked grid.
        BV = min(128, max(16, triton.next_power_of_2(V)))
        grid_k22 = (T, triton.cdiv(V, BV), B * H_kv)

        htree_accumulate_non_topk_kernel[grid_k22](
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

        # 更新 parent indices (用当前层 prev_selected_parents + topk_positions 计算下一层 parents)
        if not is_bottom_layer:
            nvtx.range_push("Post_ComputeNextParents")

            print("    Running compute next parents kernel...")
            torch.cuda.synchronize()
            start_k24 = torch.cuda.Event(enable_timing=True)
            end_k24 = torch.cuda.Event(enable_timing=True)
            start_k24.record()

            # Next parents are KV-head granularity; launch on (T, B*H_kv)
            htree_compute_next_parents_kernel[(T, B * H_kv)](
                prev_selected_parents,
                topk_positions,
                topk_positions,
                B=B, T=T, H_kv=H_kv,
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
    'htree_compute_scores_kernel',
    'htree_compute_next_parents_kernel',
    'htree_mask_topk_scores_kernel',
    'htree_accumulate_non_topk_kernel',
    'htree_merge_to_global_kernel_v2',
    'htree_final_normalize_kernel_v2',
]
