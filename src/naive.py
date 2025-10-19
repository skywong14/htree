"""
Naive implementation of htree (Hierarchical Tree for KV Cache and Sparse Attention).
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


class Rotary(torch.nn.Module):
    """
    Implementation of RoPE with pre-computed cache.
    
    Optimized version that pre-computes all RoPE values during initialization.
    """
    
    def __init__(self, dim: int, base: float = 10000.0, cache_size: int = 8192):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.dim = dim
        self.cache_size = cache_size
        
        # 预计算 RoPE 值，额外留出 1024 个位置以处理边界情况
        # (如 query_pos = cache_size 的情况)
        extended_size = cache_size + 1024
        positions = torch.arange(extended_size, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        
        # 注册为 buffer，自动处理设备迁移
        self.register_buffer('_cos_cache', freqs.cos(), persistent=False)
        self.register_buffer('_sin_cache', freqs.sin(), persistent=False)
    
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
        # 检查是否在预计算范围内
        if positions.numel() > 0:
            max_pos = positions.max().item()
            if max_pos >= len(self._cos_cache):
                raise ValueError(
                    f"Position {max_pos} exceeds pre-computed cache range "
                    f"[0, {len(self._cos_cache)}). Consider increasing cache_size."
                )
        
        # 直接索引预计算的 cache（零开销）
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
    
    return layers


def compute_and_select(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    node_ranges: torch.Tensor,
    query: torch.Tensor,
    query_pos: int,
    prev_selected_parent_indices: Optional[torch.Tensor],
    compression_rate: int,
    layer_idx: int,
    rotary: Rotary,
    top_k: int = 512,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    compute and select Top-K nodes in the current layer
    
    Simplified strategy: instead of treating rightmost node separately,
    we give it a large score bonus to ensure it's always selected.
    
    Args:
        layer_k, layer_v: [B, N, H, D]
        node_ranges: [N, 2] Each node covers the token range [start, end)
        query: [B, H, D]
        query_pos: query position m
        prev_selected_parent_indices: [B, num_selected] or None(top layer)
        compression_rate: compression rate
        layer_idx: current layer index (0: bottom layer)
        rotary: Rotary
        top_k: number of selected nodes, 512
        scale: attention scale
    
    Returns:
        candidate_indices: [num_cand] all candidate node indices
        candidate_k: [B, num_cand, H, D] all candidate keys
        candidate_v: [B, num_cand, H, D] all candidate values
        candidate_scores: [B, num_cand] attention scores for all candidates
        selected_indices: [B, top_k] selected node indices (for next layer expansion)
        selected_k: [B, num_selected, H, D] selected keys
        selected_v: [B, num_selected, H, D] selected values
    """
    B, N, H, D = layer_k.shape
    device = layer_k.device
    
    if scale is None:
        scale = D ** -0.5
    
    # 确定候选节点范围
    if prev_selected_parent_indices is None:
        # 顶层：考虑所有左边界 ≤ m 的节点
        candidates_mask = node_ranges[:, 0] <= query_pos
    else:
        # 非顶层：只考虑上一层选中节点的子节点
        parents = prev_selected_parent_indices.reshape(-1)  # [B*M]
        child_offsets = torch.arange(compression_rate, device=device)
        children = (parents[:, None] * compression_rate + child_offsets).reshape(-1)
        children = children[children < N]
        
        # 过滤：只保留左边界 ≤ m 的节点
        valid_mask = node_ranges[:, 0] <= query_pos
        valid_children = children[valid_mask[children]]
        
        candidates_mask = torch.zeros(N, dtype=torch.bool, device=device)
        candidates_mask[valid_children] = True
    
    # 最右侧节点（包含 query 位置 m 的节点）一定在候选集中
    # 包含 query_pos 的节点索引
    rightmost_idx = query_pos // (compression_rate ** layer_idx)
    if rightmost_idx < N:
        candidates_mask[rightmost_idx] = True
    
    # 提取候选节点
    candidate_indices = torch.where(candidates_mask)[0]  # [num_cand]
    candidate_indices = torch.sort(candidate_indices)[0]  # 确保有序
    num_candidates = candidate_indices.numel()
    
    if num_candidates == 0:
        # 无候选节点
        raise RuntimeError(f"No candidates available at layer {layer_idx}, query_pos {query_pos}")
    
    candidate_k = layer_k[:, candidates_mask]  # [B, num_cand, H, D]
    candidate_v = layer_v[:, candidates_mask]  # [B, num_cand, H, D]
    
    # 应用 RoPE（优化：直接使用预计算的 cache，避免创建新 tensor）
    # 候选节点使用位置 0, 1, 2, ..., num_candidates-1
    cos = rotary._cos_cache[:num_candidates]
    sin = rotary._sin_cache[:num_candidates]
    candidate_k_rope = apply_rotary_emb(candidate_k, cos[None, :, None, :], sin[None, :, None, :])
    
    # Query 使用位置 num_candidates
    cos_q = rotary._cos_cache[num_candidates]
    sin_q = rotary._sin_cache[num_candidates]
    query_rope = apply_rotary_emb(query, cos_q, sin_q)
    
    # 计算注意力分数
    scores = torch.einsum('bhd,bnhd->bhn', query_rope * scale, candidate_k_rope)  # [B, H, num_cand]
    scores_pooled = scores.mean(dim=1)  # [B, num_cand]
    
    # 给包含 query 位置的最右侧节点加分数加成（确保一定被选中）
    # 注意：仅在非最底层加分数加成，因为最底层不需要向下拓展
    if layer_idx > 0:
        rightmost_local_idx = (candidate_indices == rightmost_idx).nonzero(as_tuple=True)[0]
        if rightmost_local_idx.numel() > 0:
            scores_pooled[:, rightmost_local_idx] += 1e9
    
    # Top-K
    actual_top_k = min(top_k, num_candidates)
    _, topk_idx = torch.topk(scores_pooled, actual_top_k, dim=1)  # [B, actual_top_k]
    
    selected_k = torch.gather(
        candidate_k, 1,
        topk_idx[:, :, None, None].expand(-1, -1, H, D)
    )  # [B, actual_top_k, H, D]
    selected_v = torch.gather(
        candidate_v, 1,
        topk_idx[:, :, None, None].expand(-1, -1, H, D)
    )  # [B, actual_top_k, H, D]
    selected_indices = torch.gather(
        candidate_indices[None, :].expand(B, -1), 1, topk_idx
    )  # [B, actual_top_k]
    
    # 返回所有候选节点信息 + 选中的节点信息
    return candidate_indices, candidate_k, candidate_v, scores_pooled, selected_indices, selected_k, selected_v


def final_compute(
    all_layer_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    query: torch.Tensor
) -> torch.Tensor:
    """
    compute final attention output
    
    Directly reuse attention scores computed in compute_and_select.
    Collect participating nodes from each layer and perform softmax over their scores.
    
    Args:
        all_layer_info: list of layer information
            each element is (candidate_indices, candidate_v, candidate_scores, selected_indices), shape:
            - candidate_indices: [num_candidates] all candidate node indices
            - candidate_v: [B, num_candidates, H, D] all candidate values
            - candidate_scores: [B, num_candidates] attention scores
            - selected_indices: [B, num_selected] selected node indices (for filtering)
        query: [B, H, D]
    
    Returns:
        output: [B, H, D] attention output
    """
    B, H, D = query.shape
    device = query.device
    num_layers = len(all_layer_info)
    output = torch.zeros(B, H, D, device=device, dtype=query.dtype)
    
    # Process each batch element
    for b in range(B):
        # Collect participating nodes from each layer
        all_scores = []
        all_values = []
        
        for layer_idx, (candidate_indices, candidate_v, candidate_scores, selected_indices) in enumerate(all_layer_info):
            # Determine participating nodes
            is_bottom_layer = (layer_idx == num_layers - 1)
            
            if is_bottom_layer:
                # Bottom layer: all candidates participate
                participating_mask = torch.ones(len(candidate_indices), dtype=torch.bool, device=device)
            else:
                # Non-bottom layer: exclude selected nodes
                is_selected = torch.isin(candidate_indices, selected_indices[b])
                participating_mask = ~is_selected
            
            # Extract scores and values for participating nodes
            if participating_mask.any():
                scores = candidate_scores[b, participating_mask]  # [num_participating]
                values = candidate_v[b, participating_mask]  # [num_participating, H, D]
                
                all_scores.append(scores)
                all_values.append(values)
        
        if len(all_scores) == 0:
            continue
        
        # Concatenate all scores and values
        all_scores = torch.cat(all_scores, dim=0)  # [total_participating]
        all_values = torch.cat(all_values, dim=0)  # [total_participating, H, D]
        
        # Compute softmax and weighted sum
        # For numerical stability, subtract max score before exp
        max_score = all_scores.max()
        exp_scores = torch.exp(all_scores - max_score)  # [total_participating]
        weights = exp_scores / exp_scores.sum()  # [total_participating]
        
        # Weighted sum: [total_participating] @ [total_participating, H, D] -> [H, D]
        output[b] = torch.einsum('n,nhd->hd', weights, all_values)
    
    return output  # [B, H, D]


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
    forward kernel
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
    B, T, H, D = q.shape
    if scale is None:
        scale = D ** -0.5
    rotary = Rotary(dim=D)
    
    # 1. build tree
    layers = build_tree(k, v, compression_rate, max_top_nodes)
    num_layers = len(layers)
    
    # 2. compute and select for each query position
    output = torch.zeros_like(q)
    
    for b in range(B):
        # Determine positions to compute for this batch element
        if query_positions is None:
            positions_to_compute = range(0, T)
        else:
            # query_positions is [B, num_positions]
            batch_positions = query_positions[b].tolist()
            if isinstance(batch_positions, int):
                batch_positions = [batch_positions]
            positions_to_compute = [p for p in batch_positions if p >= 0]
        
        for t in positions_to_compute:
            
            current_query = q[b:b+1, t]  # [1, H, D]
            
            # 3. compute and select
            layer_info = []
            prev_selected_indices = None
            
            for layer_idx in range(num_layers - 1, -1, -1):
                layer_k, layer_v, node_ranges = layers[layer_idx]
                
                candidate_idx, candidate_k, candidate_v, candidate_scores, selected_idx, selected_k, selected_v = compute_and_select(
                    layer_k[b:b+1],
                    layer_v[b:b+1],
                    node_ranges,
                    current_query,
                    t,
                    prev_selected_indices,
                    compression_rate,
                    layer_idx,
                    rotary,
                    top_k_per_layer,
                    scale
                )
                
                # record candidate and selected results
                layer_info.append((candidate_idx, candidate_v, candidate_scores, selected_idx))
                prev_selected_indices = selected_idx
            
            # 4. final_compute
            output[b:b+1, t] = final_compute(
                layer_info,
                current_query
            )
    
    return output