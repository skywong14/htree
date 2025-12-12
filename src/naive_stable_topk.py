"""
Naive implementation of htree (Hierarchical Tree for KV Cache and Sparse Attention).
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F


class Rotary(torch.nn.Module):
    """
    Implementation of RoPE with pre-computed cache.
    
    Optimized version that pre-computes all RoPE values during initialization.
    """
    
    def __init__(self, dim: int, base: float = 10000.0, cache_size: int = 8192):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.dim = dim
        self.cache_size = cache_size
        
        # 延迟创建缓存，在第一次调用时根据设备创建
        self._cos_cache = None
        self._sin_cache = None
        self._cache_device = None
    
    def forward(self, positions: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [seq_len] Position indices. Must be < cache_size + 1024.
            device: Target device (for compatibility, buffer auto-migrates).
        Returns:
            cos, sin: [seq_len, dim//2] Cosine and sine values.
        
        Raises:
            ValueError: If any position exceeds the pre-computed cache range.
        """
        # 延迟创建缓存，确保在正确的设备上
        if self._cos_cache is None or self._cache_device != device:
            extended_size = self.cache_size + 1024
            positions_tensor = torch.arange(extended_size, dtype=torch.float32, device=device)
            freqs = torch.outer(positions_tensor, self.inv_freq.to(device))
            
            self._cos_cache = freqs.cos()
            self._sin_cache = freqs.sin()
            self._cache_device = device
        
        # 检查是否在预计算范围内
        if positions.numel() > 0:
            max_pos = positions.max().item()
            if max_pos >= len(self._cos_cache):
                raise ValueError(
                    f"Position {max_pos} exceeds pre-computed cache range "
                    f"[0, {len(self._cos_cache)}). Consider increasing cache_size."
                )
        
        return self._cos_cache[positions], self._sin_cache[positions]

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position encoding.
    
    Args:
        x: [..., dim] Input tensor.
        cos, sin: [seq_len, dim//2] or broadcastable shape.
    Returns:
        [..., dim] Tensor after RoPE.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # (x1 + ix2) * e^(iθ)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ==========================================
# Bit-Packing Top-K 辅助函数
# ==========================================

def bitpack_encode(scores: torch.Tensor, indices: torch.Tensor, log_n: int = 13) -> torch.Tensor:
    """
    将 buffer 位置索引编码到 float32 分数的低位。
    
    与 Triton 实现一致的 Bit-Packing 编码：
    - 正数：取反索引 (~idx)，使小索引在数值相同时排在前面（稳定排序）
    - 负数：保持索引原值
    
    Args:
        scores: [N] float32 分数
        indices: [N] int32 buffer 位置索引 (0 到 N-1)
        log_n: 索引位数 (13 for N=8192)
    
    Returns:
        encoded_scores: [N] float32 编码后的分数
    """
    idx_mask = (1 << log_n) - 1  # 0x1FFF for log_n=13
    
    # 将 float32 视为 int32
    scores_int = scores.view(torch.int32)
    
    # TODO
    # 手动清空尾数的第14位（bit 13，从0开始计数）
    # bit_13_mask = ~(1 << 13)  # 创建一个只有 bit 13 为 0 的掩码
    # scores_int = scores_int & bit_13_mask
    
    # 编码索引：正数取反，负数保持
    encoded_idx = torch.where(
        scores >= 0,
        ~indices.to(torch.int32),
        indices.to(torch.int32)
    )
    encoded_idx = encoded_idx & idx_mask
    
    # 清除 float32 低位，填入编码索引
    scores_encoded_int = (scores_int & ~idx_mask) | encoded_idx
    
    return scores_encoded_int.view(torch.float32)


def bitpack_decode(sorted_encoded_scores: torch.Tensor, log_n: int = 13) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从编码后的排序结果中解码出原始 buffer 位置和清理后的分数。
    
    Args:
        sorted_encoded_scores: [N] float32 排序后的编码分数
        log_n: 索引位数 (13 for N=8192)
    
    Returns:
        buffer_positions: [N] int32 原始 buffer 位置
        clean_scores: [N] float32 清理后的分数（低位置零）
    """
    idx_mask = (1 << log_n) - 1  # 0x1FFF for log_n=13
    
    # 转为 int32
    sorted_int = sorted_encoded_scores.view(torch.int32)
    
    # 提取编码的索引（低 log_n 位）
    raw_idx = sorted_int & idx_mask
    
    # 清除低位恢复分数
    clean_int = sorted_int & ~idx_mask
    clean_scores = clean_int.view(torch.float32)
    
    # 还原索引：正数时索引被取反了，需要再次取反
    buffer_positions = torch.where(
        clean_scores >= 0,
        ~raw_idx,
        raw_idx
    )
    buffer_positions = (buffer_positions & idx_mask).to(torch.int32)
    
    return buffer_positions, clean_scores


def stable_topk_with_bitpacking(
    scores: torch.Tensor,
    k: int,
    rightmost_idx: int,
    candidate_indices: torch.Tensor,
    log_n: int = 13
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 Bit-Packing 技术进行稳定的 Top-K 选择。
    
    与 Triton 实现完全一致的流程：
    1. 给 rightmost 节点设置高分 (1e3) 确保被选中
    2. Bit-Packing 编码
    3. 降序排序
    4. 解码提取 Top-K 的 buffer 位置
    5. 根据 buffer 位置获取全局节点索引
    
    Args:
        scores: [num_cand] float32 候选节点分数
        k: Top-K 数量
        rightmost_idx: 最右侧节点的全局索引
        candidate_indices: [num_cand] 候选节点的全局索引
        log_n: 索引位数
    
    Returns:
        topk_buffer_positions: [k] Top-K 在 buffer 中的位置（用于创建 mask）
        selected_node_indices_sorted: [k] Top-K 的全局节点索引（升序排序，传给下一层）
    """
    # 由于 bit-packing 需要直接 view int32，这里统一使用 float32 进行编码/排序，避免 FP16/BF16 的 view 报错。
    scores_fp32 = scores.float()

    num_cand = scores_fp32.numel()
    device = scores.device
    actual_k = min(k, num_cand)
    
    # 1. 给 rightmost 节点设置高分
    is_rightmost = (candidate_indices == rightmost_idx)
    scores_modified = torch.where(
        is_rightmost,
        torch.tensor(1e3, device=device, dtype=torch.float32),
        scores_fp32
    )
    
    # 2. Bit-Packing 编码
    buffer_indices = torch.arange(num_cand, device=device, dtype=torch.int32)
    encoded_scores = bitpack_encode(scores_modified, buffer_indices, log_n)
    
    # 3. 降序排序
    sorted_encoded_scores, _ = torch.sort(encoded_scores, descending=True)
    
    # 4. 解码
    buffer_positions, _ = bitpack_decode(sorted_encoded_scores, log_n)
    
    # 5. 提取前 K 个的 buffer 位置
    topk_buffer_positions = buffer_positions[:actual_k]
    
    # 6. 根据 buffer 位置获取全局节点索引
    selected_node_indices = candidate_indices[topk_buffer_positions.long()]
    
    # 7. 对全局节点索引升序排序（传给下一层）
    selected_node_indices_sorted, _ = torch.sort(selected_node_indices)
    
    # Padding to k if needed
    if actual_k < k:
        padding = torch.full([k - actual_k], -1, dtype=torch.int32, device=device)
        topk_buffer_positions = torch.cat([topk_buffer_positions, padding])
        selected_node_indices_sorted = torch.cat([selected_node_indices_sorted, padding])
    
    return topk_buffer_positions, selected_node_indices_sorted


def build_tree(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build hierarchical structure of htree.

    Args:
        k: [B, T, H, D]
        v: [B, T, H, D]
        compression_rate: Compression rate, 16.
        max_top_nodes: Maximum number of top nodes, 8192.

    Returns:
        layers: Hierarchical list, each element is (K_layer, V_layer, node_ranges)
            - K_layer, V_layer: [B, N_nodes, H, D]
            - node_ranges: [N_nodes, 2] Each node covers the token range [start, end)
    """
    nvtx.range_push("BuildTree")
    B, T, H, D = k.shape
    layers = []
    
    # TODO node_ranges也许没有必要，因为整个树的结构是唯一确定的
    # 层 0: 底层 token
    node_ranges = torch.stack([
        torch.arange(T, device=k.device),
        torch.arange(1, T + 1, device=k.device)
    ], dim=1)  # [T, 2]
    layers.append((k, v, node_ranges))
    
    current_k, current_v = k, v
    current_len = T
    
    # 逐层向上构建，直到节点数 <= max_top_nodes
    while current_len > max_top_nodes:
        # 下一层的节点数
        next_len = (current_len + compression_rate - 1) // compression_rate
        
        # mean pooling
        # padding
        pad_len = next_len * compression_rate - current_len
        if pad_len > 0:
            current_k = F.pad(current_k, (0, 0, 0, 0, 0, pad_len))  # [B, next_len*compression_rate, H, D]
            current_v = F.pad(current_v, (0, 0, 0, 0, 0, pad_len))
        
        # Reshape & mean
        # [B, next_len*compression_rate, H, D] -> [B, next_len, compression_rate, H, D] -> [B, next_len, H, D]
        next_k = current_k.view(B, next_len, compression_rate, H, D).mean(dim=2)
        next_v = current_v.view(B, next_len, compression_rate, H, D).mean(dim=2)
        
        # 节点范围: 每个父节点覆盖 compression_rate 个子节点的范围
        prev_ranges = layers[-1][2]  # [current_len, 2]
        # padding
        if pad_len > 0:
            last_end = prev_ranges[-1, 1]
            pad_ranges = last_end.expand(pad_len, 2)  # [pad_len, 2]，两列都是 last_end
            prev_ranges = torch.cat([prev_ranges, pad_ranges], dim=0)
        
        # 合并
        prev_ranges = prev_ranges.view(next_len, compression_rate, 2)
        node_ranges = torch.stack([
            prev_ranges[:, 0, 0],   # 第一个子节点的 start
            prev_ranges[:, -1, 1]   # 最后一个子节点的 end
        ], dim=1)  # [next_len, 2]
        
        layers.append((next_k, next_v, node_ranges))
        
        current_k, current_v = next_k, next_v
        current_len = next_len

    nvtx.range_pop()
    return layers


def online_softmax_merge(
    max_score: torch.Tensor,
    sum_exp: torch.Tensor,
    weighted_output: torch.Tensor,
    new_scores: torch.Tensor,
    new_values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge new scores and values into existing softmax accumulation state using online algorithm.
    
    Args:
        max_score: [1] Current maximum score
        sum_exp: [1] Current sum of exponentials
        weighted_output: [1, 1, D] Current weighted output (single head)
        new_scores: [1, num_new] New attention scores to merge
        new_values: [1, num_new, 1, D] New values to merge (single head)
    
    Returns:
        Updated (max_score, sum_exp, weighted_output)
    """
    if new_scores.numel() == 0:
        return max_score, sum_exp, weighted_output
    
    # Flatten batch dimension (always 1 in our case)
    new_scores = new_scores.squeeze(0)  # [num_new]
    new_values = new_values.squeeze(0).squeeze(1)  # [num_new, D]
    
    # Compute new max
    new_max = new_scores.max()
    old_max = max_score.item()
    updated_max = max(old_max, new_max.item())
    
    # Rescale existing accumulation if max changed
    if old_max > -float('inf'):
        scale_old = torch.exp(torch.tensor(old_max - updated_max, device=max_score.device))
        sum_exp = sum_exp * scale_old
        weighted_output = weighted_output * scale_old
    
    # Add new contributions
    exp_new = torch.exp(new_scores - updated_max)  # [num_new]
    sum_exp = sum_exp + exp_new.sum()
    weighted_output = weighted_output + torch.einsum('n,nd->d', exp_new, new_values).unsqueeze(0).unsqueeze(0)
    
    max_score = torch.tensor([updated_max], device=max_score.device)
    
    return max_score, sum_exp, weighted_output


def compute_select_and_merge(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    node_ranges: torch.Tensor,
    query: torch.Tensor,
    query_pos: int,
    prev_selected_parent_indices: Optional[torch.Tensor],
    compression_rate: int,
    layer_idx: int,
    rotary: Rotary,
    max_score: torch.Tensor,
    sum_exp: torch.Tensor,
    weighted_output: torch.Tensor,
    top_k: int = 512,
    scale: Optional[float] = None,
    return_debug_info: bool = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor],
]:
    """
    Compute attention scores, select Top-K nodes, and merge non-selected nodes into softmax state.
    
    Args:
        layer_k, layer_v: [1, N, 1, D] Single head data
        node_ranges: [N, 2] Each node covers the token range [start, end)
        query: [1, 1, D] Single head query
        query_pos: query position m
        prev_selected_parent_indices: [1, num_selected] or None (top layer)
        compression_rate: compression rate
        layer_idx: current layer index (0: bottom layer)
        rotary: Rotary embedding
        max_score: [1] Current maximum score in softmax accumulation
        sum_exp: [1] Current sum of exponentials
        weighted_output: [1, 1, D] Current weighted output (single head)
        top_k: number of selected nodes, 512
        scale: attention scale
    
    Returns:
        max_score: [1] Updated maximum score
        sum_exp: [1] Updated sum of exponentials
        weighted_output: [1, 1, D] Updated weighted output (single head)
        selected_indices: [1, top_k] Selected node indices for next layer (None for bottom layer)
        
        If return_debug_info=True, also returns:
        candidate_indices: [num_cand] All candidate node indices
        scores_pooled: [num_cand] All candidate scores
    """
    B, N, H, D = layer_k.shape
    device = layer_k.device
    
    assert B == 1 and H == 1, "This function processes single head data"
    
    if scale is None:
        scale = D ** -0.5
    
    # 确定候选节点范围
    if prev_selected_parent_indices is None:
        # 顶层：考虑所有左边界 ≤ m 的节点
        candidates_mask = node_ranges[:, 0] <= query_pos
    else:
        # 非顶层：只考虑上一层选中节点的子节点
        parents = prev_selected_parent_indices.reshape(-1)  # [M]
        child_offsets = torch.arange(compression_rate, device=device)
        children = (parents[:, None] * compression_rate + child_offsets).reshape(-1)
        children = children[children < N]
        
        # 过滤：只保留左边界 ≤ m 的节点
        valid_mask = node_ranges[:, 0] <= query_pos
        valid_children = children[valid_mask[children]]
        
        candidates_mask = torch.zeros(N, dtype=torch.bool, device=device)
        candidates_mask[valid_children] = True
    
    # 最右侧节点（包含 query 位置 m 的节点）一定在候选集中
    rightmost_idx = query_pos // (compression_rate ** layer_idx)
    if rightmost_idx < N:
        candidates_mask[rightmost_idx] = True
    
    # 提取候选节点
    candidate_indices = torch.where(candidates_mask)[0]  # [num_cand]
    candidate_indices = torch.sort(candidate_indices)[0]  # 确保有序
    num_candidates = candidate_indices.numel()
    
    if num_candidates == 0:
        raise RuntimeError(f"No candidates available at layer {layer_idx}, query_pos {query_pos}")
    
    candidate_k = layer_k[:, candidates_mask]  # [1, num_cand, 1, D]
    candidate_v = layer_v[:, candidates_mask]  # [1, num_cand, 1, D]
    
    # 应用 RoPE - 需要先初始化缓存
    positions = torch.arange(num_candidates, device=device)
    cos, sin = rotary.forward(positions, device)
    candidate_k_rope = apply_rotary_emb(candidate_k, cos[None, :, None, :], sin[None, :, None, :])
    
    # Query 使用最后一个位置（与最右侧候选节点相同）
    cos_q = cos[num_candidates - 1]
    sin_q = sin[num_candidates - 1]
    query_rope = apply_rotary_emb(query, cos_q, sin_q)
    
    # 计算注意力分数 (single head, no pooling)
    scores = torch.einsum('bhd,bnhd->bhn', query_rope * scale, candidate_k_rope)  # [1, 1, num_cand]
    scores_pooled = scores.squeeze(0).squeeze(0)  # [num_cand]

    # 确定参与在线Softmax的节点和选中的节点
    is_bottom_layer = (layer_idx == 0)
    if is_bottom_layer:
        # 底层：所有候选节点参与计算
        merge_scores = scores_pooled.unsqueeze(0)  # [1, num_cand]
        merge_values = candidate_v  # [1, num_cand, 1, D]
        selected_indices = None
        topk_buffer_positions = None
        # 底层不需要编码，scores_pooled_encoded 与 scores_pooled 相同
        scores_pooled_encoded = scores_pooled
    else:
        # ================================================================
        # 非底层：使用 Bit-Packing Top-K 选择（与 Triton 实现一致）
        # ================================================================
        
        # 将 position_id 编码到 scores_pooled 中（用于调试和可视化）
        log_n = 13  # MAX_CANDIDATES = 8192, log2(8192) = 13
        buffer_indices = torch.arange(num_candidates, device=device, dtype=torch.int32)
        
        # 给 rightmost 节点设置高分（与 Triton 一致）
        is_rightmost = (candidate_indices == rightmost_idx)
        scores_modified = torch.where(
            is_rightmost,
            torch.tensor(1e3, device=device, dtype=torch.float32),
            scores_pooled
        )
        
        # Bit-Packing 编码
        scores_pooled_encoded = bitpack_encode(scores_modified, buffer_indices, log_n)

        # 使用 Bit-Packing 稳定 Top-K（注意：这里传入编码后的分数）
        topk_buffer_positions, selected_node_indices_sorted = stable_topk_with_bitpacking(
            scores_pooled,  # 内部会重新编码，所以这里还是传原始分数
            top_k,
            rightmost_idx,
            candidate_indices,
            log_n=log_n
        )
        
        # 被选中的节点掩码（基于 buffer 位置）
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        valid_positions = topk_buffer_positions[topk_buffer_positions >= 0]
        selected_mask[valid_positions.long()] = True
        
        # 未被选中的节点，参与计算（使用原始 score）
        merge_mask = ~selected_mask
        merge_scores = scores_pooled[merge_mask].unsqueeze(0)  # [1, num_merge]
        merge_values = candidate_v[:, merge_mask]  # [1, num_merge, 1, D]
        
        # 选中的节点索引（升序排序，传给下一层）
        selected_indices = selected_node_indices_sorted.unsqueeze(0)  # [1, top_k]
    
    # Online softmax merge
    max_score, sum_exp, weighted_output = online_softmax_merge(
        max_score, sum_exp, weighted_output, merge_scores, merge_values
    )

    if return_debug_info:
        # 返回编码后的分数（包含 position_id），用于调试和可视化
        return max_score, sum_exp, weighted_output, selected_indices, candidate_indices, scores_pooled_encoded
    else:
        return max_score, sum_exp, weighted_output, selected_indices


def compute_and_select_only(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    node_ranges: torch.Tensor,
    query: torch.Tensor,
    query_pos: int,
    prev_selected_parent_indices: Optional[torch.Tensor],
    compression_rate: int,
    layer_idx: int,
    rotary: Rotary,
    max_score: torch.Tensor,
    sum_exp: torch.Tensor,
    weighted_output: torch.Tensor,
    top_k: int = 512,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Compute attention scores, select Top-K nodes, and merge ALL candidates into softmax state.
    This simulates Kernel 2.1 behavior (before subtract_topk).
    
    Args:
        Same as compute_select_and_merge
    
    Returns:
        max_score: [1] Updated maximum score (after merging ALL candidates)
        sum_exp: [1] Updated sum of exponentials (after merging ALL candidates)
        weighted_output: [1, 1, D] Updated weighted output (after merging ALL candidates)
        selected_indices: [1, top_k] Selected node indices for next layer (None for bottom layer)
        topk_indices_unsorted: [top_k] Top-K indices in importance order (padded with -1)
        topk_scores_unsorted: [top_k] Top-K scores in importance order
    """
    B, N, H, D = layer_k.shape
    device = layer_k.device
    
    assert B == 1 and H == 1, "This function processes single head data"
    
    if scale is None:
        scale = D ** -0.5
    
    # 确定候选节点范围
    if prev_selected_parent_indices is None:
        # 顶层：考虑所有左边界 ≤ m 的节点
        candidates_mask = node_ranges[:, 0] <= query_pos
    else:
        # 非顶层：只考虑上一层选中节点的子节点
        parents = prev_selected_parent_indices.reshape(-1)  # [M]
        child_offsets = torch.arange(compression_rate, device=device)
        children = (parents[:, None] * compression_rate + child_offsets).reshape(-1)
        children = children[children < N]
        
        # 过滤：只保留左边界 ≤ m 的节点
        valid_mask = node_ranges[:, 0] <= query_pos
        valid_children = children[valid_mask[children]]
        
        candidates_mask = torch.zeros(N, dtype=torch.bool, device=device)
        candidates_mask[valid_children] = True
    
    # 最右侧节点（包含 query 位置 m 的节点）一定在候选集中
    rightmost_idx = query_pos // (compression_rate ** layer_idx)
    if rightmost_idx < N:
        candidates_mask[rightmost_idx] = True
    
    # 提取候选节点
    candidate_indices = torch.where(candidates_mask)[0]  # [num_cand]
    candidate_indices = torch.sort(candidate_indices)[0]  # 确保有序
    num_candidates = candidate_indices.numel()
    
    if num_candidates == 0:
        raise RuntimeError(f"No candidates available at layer {layer_idx}, query_pos {query_pos}")
    
    candidate_k = layer_k[:, candidates_mask]  # [1, num_cand, 1, D]
    candidate_v = layer_v[:, candidates_mask]  # [1, num_cand, 1, D]
    
    # 应用 RoPE - 需要先初始化缓存
    positions = torch.arange(num_candidates, device=device)
    cos, sin = rotary.forward(positions, device)
    candidate_k_rope = apply_rotary_emb(candidate_k, cos[None, :, None, :], sin[None, :, None, :])
    
    # Query 使用最后一个位置（与最右侧候选节点相同）
    cos_q = cos[num_candidates - 1]
    sin_q = sin[num_candidates - 1]
    query_rope = apply_rotary_emb(query, cos_q, sin_q)
    
    # 计算注意力分数 (single head, no pooling)
    scores = torch.einsum('bhd,bnhd->bhn', query_rope * scale, candidate_k_rope)  # [1, 1, num_cand]
    scores_pooled = scores.squeeze(0).squeeze(0)  # [num_cand]
    
    # Merge ALL candidates into softmax state (Kernel 2.1 behavior)
    merge_scores = scores_pooled.unsqueeze(0)  # [1, num_cand]
    merge_values = candidate_v  # [1, num_cand, 1, D]
    
    max_score, sum_exp, weighted_output = online_softmax_merge(
        max_score, sum_exp, weighted_output, merge_scores, merge_values
    )
    
    # Select Top-K for next layer (but don't subtract them yet)
    is_bottom_layer = (layer_idx == 0)
    if is_bottom_layer:
        selected_indices = None
        topk_indices_unsorted = torch.full([top_k], -1, dtype=torch.int32, device=device)
        topk_scores_unsorted = torch.zeros([top_k], dtype=torch.float32, device=device)
    else:
        # 使用 importance 选择 Top-K（与 compute_select_and_merge 一致）
        # 计算当前最大分数（merge 后的 max_score）
        cur_max = max_score.item()
        
        # 计算 importance：最右侧节点 = 1.0，其他节点 = exp(score - cur_max)
        is_rightmost = (candidate_indices == rightmost_idx)
        importance = torch.where(
            is_rightmost,
            torch.tensor(1.0, device=device, dtype=scores_pooled.dtype),
            torch.exp(scores_pooled - cur_max)
        )
        
        # 基于 importance 选择 Top-K
        actual_top_k = min(top_k, num_candidates)
        topk_importance, topk_local_idx = torch.topk(importance, actual_top_k, dim=0)  # [actual_top_k]
        
        # 选中的节点索引
        selected_indices = candidate_indices[topk_local_idx].unsqueeze(0)  # [1, actual_top_k]
        
        # 返回未排序的 Top-K 信息（按 importance 顺序，padded with -1）
        # 注意：这里返回的是原始 score，不是 importance
        topk_indices_unsorted = torch.full([top_k], -1, dtype=torch.int32, device=device)
        topk_scores_unsorted = torch.zeros([top_k], dtype=torch.float32, device=device)
        
        topk_indices_unsorted[:actual_top_k] = candidate_indices[topk_local_idx].to(torch.int32)
        topk_scores_unsorted[:actual_top_k] = scores_pooled[topk_local_idx]
    
    return max_score, sum_exp, weighted_output, selected_indices, topk_indices_unsorted, topk_scores_unsorted


def forward_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: Optional[torch.Tensor] = None,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    forward kernel with online softmax algorithm

    Args:
        q: [B, T, H, D]
        k: [B, T, H, D]
        v: [B, T, H, D]
        query_positions: [B, num_positions] or None
            If None, compute attention for all positions.
            If provided, only compute attention for specified positions per batch.
        compression_rate: compression rate, 16
        max_top_nodes: maximum number of top nodes, 8192
        top_k_per_layer: number of selected nodes per layer, 512
        scale: attention scale

    Returns:
        output: [B, T, H, D]
    """
    nvtx.range_push("Naive_Forward")
    B, T, H, D = q.shape
    device = q.device
    if scale is None:
        scale = D ** -0.5
    rotary = Rotary(dim=D).to(device)
    
    # 1. Build tree
    nvtx.range_push("Phase1_BuildTree")
    layers = build_tree(k, v, compression_rate, max_top_nodes)
    num_layers = len(layers)
    nvtx.range_pop()
    
    # 2. Compute attention for each query position using online softmax
    # Each head is processed independently
    nvtx.range_push("Phase2_Attention")
    output = torch.zeros_like(q)
    
    for b in range(B):
        # Determine positions to compute for this batch element
        if query_positions is None:
            positions_to_compute = range(0, T)
        else:
            batch_positions = query_positions[b].tolist()
            if isinstance(batch_positions, int):
                batch_positions = [batch_positions]
            positions_to_compute = [p for p in batch_positions if p >= 0]
        
        for h in range(H):
            for t in positions_to_compute:
                current_query = q[b:b+1, t, h:h+1]  # [1, 1, D]
                
                # Initialize online softmax state (per head)
                max_score = torch.tensor([-float('inf')], device=q.device)
                sum_exp = torch.tensor([0.0], device=q.device)
                weighted_output = torch.zeros(1, 1, D, device=q.device, dtype=q.dtype)
                
                # 3. Iterate from top to bottom layer with online softmax merging
                prev_selected_indices = None
                
                for layer_idx in range(num_layers - 1, -1, -1):
                    layer_k, layer_v, node_ranges = layers[layer_idx]
                    
                    max_score, sum_exp, weighted_output, selected_indices = compute_select_and_merge(
                        layer_k[b:b+1, :, h:h+1],  # [1, N, 1, D] - single head
                        layer_v[b:b+1, :, h:h+1],  # [1, N, 1, D] - single head
                        node_ranges,
                        current_query,
                        t,
                        prev_selected_indices,
                        compression_rate,
                        layer_idx,
                        rotary,
                        max_score,
                        sum_exp,
                        weighted_output,
                        top_k_per_layer,
                        scale
                    )
                    
                    prev_selected_indices = selected_indices
                
                # 4. Normalize to get final output
                output[b:b+1, t, h:h+1] = (weighted_output / sum_exp.unsqueeze(-1)).squeeze(0)
    nvtx.range_pop()  # Phase2_Attention

    nvtx.range_pop()  # Naive_Forward
    return output