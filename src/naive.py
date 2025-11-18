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
    else:
        # 非底层：使用 importance 选择 Top-K
        # 计算当前最大分数（用于计算 importance）
        cur_max = max(max_score.item(), scores_pooled.max().item())
        
        # 计算 importance：最右侧节点 = 1.0，其他节点 = exp(score - cur_max)
        is_rightmost = (candidate_indices == rightmost_idx)
        importance = torch.where(
            is_rightmost,
            torch.tensor(1.0, device=device, dtype=scores_pooled.dtype),
            torch.exp(scores_pooled - cur_max)
        )
        
        # 基于 importance 选择 Top-K
        actual_top_k = min(top_k, num_candidates)
        _, topk_local_idx = torch.topk(importance, actual_top_k, dim=0)  # [actual_top_k]
        
        # 被选中的节点掩码
        selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        selected_mask[topk_local_idx] = True
        
        # 未被选中的节点，参与计算（使用原始 score，不是 importance）
        merge_mask = ~selected_mask
        merge_scores = scores_pooled[merge_mask].unsqueeze(0)  # [1, num_merge]
        merge_values = candidate_v[:, merge_mask]  # [1, num_merge, 1, D]
        
        # 选中的节点索引
        selected_indices = candidate_indices[topk_local_idx].unsqueeze(0)  # [1, actual_top_k]
    
    # Online softmax merge
    max_score, sum_exp, weighted_output = online_softmax_merge(
        max_score, sum_exp, weighted_output, merge_scores, merge_values
    )
    
    if return_debug_info:
        return max_score, sum_exp, weighted_output, selected_indices, candidate_indices, scores_pooled
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