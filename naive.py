"""
Naive implementation of htree (Hierarchical Tree for KV Cache and Sparse Attention).
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


class Rotary(torch.nn.Module):
    """
    Implementation of RoPE.
    """
    
    def __init__(self, dim: int, base: float = 10000.0, cache_size: int = 8192):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.dim = dim
        self.cache_size = cache_size
        
        # 缓存：device -> (cos_cache, sin_cache)
        self._cache = {}
    
    def _build_cache(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build cache table for specified device.
        
        Args:
            device: Target device.
        Returns:
            cos_cache, sin_cache: Cache table of shape [cache_size, dim//2].
        """
        positions = torch.arange(self.cache_size, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        cos_cache = freqs.cos()  # [cache_size, dim//2]
        sin_cache = freqs.sin()  # [cache_size, dim//2]
        
        self._cache[device] = (cos_cache, sin_cache)
        return cos_cache, sin_cache
    
    def forward(self, positions: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: [seq_len] Position indices.
            device: Target device.
        Returns:
            cos, sin: [seq_len, dim//2] Cosine and sine values.
        """
        # 检查是否在缓存范围内
        max_pos = positions.max().item() if positions.numel() > 0 else -1
        
        if max_pos < self.cache_size:
            if device not in self._cache:
                self._build_cache(device)
            
            cos_cache, sin_cache = self._cache[device]
            return cos_cache[positions], sin_cache[positions]
        
        # 超出缓存范围
        freqs = torch.outer(positions.float(), self.inv_freq.to(device))
        return freqs.cos(), freqs.sin()

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        selected_indices: [B, top_k] selected node indices
        selected_k: [B, num_selected, H, D] selected keys
        selected_v: [B, num_selected, H, D] selected values
    """
    B, N, H, D = layer_k.shape
    device = layer_k.device
    
    if scale is None:
        scale = D ** -0.5
    
    # 确定候选节点范围
    if prev_selected_parent_indices is None:
        # 顶层：考虑所有不包含未来信息的节点
        candidates_mask = node_ranges[:, 1] <= query_pos
    else:
        # 非顶层：只考虑上一层选中节点的子节点
        parents = prev_selected_parent_indices.reshape(-1)  # [B*M]
        child_offsets = torch.arange(compression_rate, device=device)
        children = (parents[:, None] * compression_rate + child_offsets).reshape(-1)
        children = children[children < N]
        
        # 过滤未来信息
        valid_mask = node_ranges[:, 1] <= query_pos
        valid_children = children[valid_mask[children]]
        
        candidates_mask = torch.zeros(N, dtype=torch.bool, device=device)
        candidates_mask[valid_children] = True
    
    # 最右侧节点（保留，即使可能包含未来信息）
    rightmost_idx = (query_pos - 1) // (compression_rate ** layer_idx)
    candidates_mask[rightmost_idx] = True
    
    # 提取候选节点
    candidate_indices = torch.where(candidates_mask)[0]  # [num_cand]
    num_candidates = candidate_indices.numel()
    
    if num_candidates == 0:
        # 无候选节点
        return (torch.zeros(B, 0, dtype=torch.long, device=device),
                torch.zeros(B, 0, H, D, device=device, dtype=layer_k.dtype),
                torch.zeros(B, 0, H, D, device=device, dtype=layer_v.dtype))
    
    candidate_k = layer_k[:, candidates_mask]  # [B, num_cand, H, D]
    candidate_v = layer_v[:, candidates_mask]  # [B, num_cand, H, D]
    
    # 应用 RoPE
    positions = torch.arange(num_candidates, device=device)
    cos, sin = rotary.forward(positions, device)
    candidate_k_rope = apply_rotary_emb(candidate_k, cos[None, :, None, :], sin[None, :, None, :])
    
    query_pos_rope = torch.tensor([num_candidates], device=device)
    cos_q, sin_q = rotary.forward(query_pos_rope, device)
    query_rope = apply_rotary_emb(query, cos_q[None, None, :], sin_q[None, None, :])  # [B, H, D]
    
    # 计算注意力分数
    scores = torch.einsum('bhd,bnhd->bhn', query_rope * scale, candidate_k_rope)  # [B, H, num_cand]
    scores_pooled = scores.mean(dim=1)  # [B, num_cand]
    
    # 给最右侧节点加分数加成（确保一定被选中）
    rightmost_local_idx = (candidate_indices == rightmost_idx).nonzero(as_tuple=True)[0]
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
    
    return selected_indices, selected_k, selected_v


def final_compute(
    all_layer_selections: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    query: torch.Tensor,
    rotary: Rotary,
    compression_rate: int = 16,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    compute final attention output
    
    1. init: filter nodes and assign RoPE positions
    2. parallel: each layer computes online attention statistics independently
    3. reduce: use logsumexp to merge results from each layer
    
    Args:
        all_layer_selections: list of selected nodes in each layer
            each element is (indices, k, v, ranges), shape:
            - indices: [B, num_selected]
            - k, v: [B, num_selected, H, D]
            - ranges: [num_selected, 2] node coverage range
        query: [B, H, D]
        rotary: Rotary object, used to calculate RoPE encoding
        compression_rate: compression rate
        scale: attention scale
    
    Returns:
        output: [B, H, D] attention output
    """
    B, H, D = query.shape
    device = query.device
    
    if scale is None:
        scale = D ** -0.5
    
    num_layers = len(all_layer_selections)
    output = torch.zeros(B, H, D, device=device, dtype=query.dtype)
    
    # TODO 实现batch维度的并行
    for b in range(B):
        # ========== init：确定各层的有效节点及其RoPE位置 ==========
        
        # 1. 构建所有选中节点的集合
        # 收集所有层的索引和范围信息
        all_indices_list = []
        all_ranges_list = []
        layer_info_list = []  # 存储 (list_idx, start_offset, num_nodes)
        
        offset = 0
        for list_idx, (indices, k, v, ranges) in enumerate(all_layer_selections):
            num_nodes = indices.shape[1]
            all_indices_list.append(indices[b])  # [num_nodes]
            all_ranges_list.append(ranges)  # [num_nodes, 2]
            layer_info_list.append((list_idx, offset, num_nodes))
            offset += num_nodes
        
        # 拼接所有节点
        all_node_indices = torch.cat(all_indices_list, dim=0)  # [total_nodes]
        all_ranges = torch.cat(all_ranges_list, dim=0)  # [total_nodes, 2]
        all_start_pos = all_ranges[:, 0]  # [total_nodes]
        
        # 2. 判断子节点
        valid_mask = torch.ones(offset, dtype=torch.bool, device=device)
        
        for list_idx, layer_offset, num_nodes in layer_info_list[:-1]:  # 只过滤非最底层的节点
            # 当前层的节点索引
            layer_node_indices = all_node_indices[layer_offset:layer_offset + num_nodes]
            
            # 计算所有可能的子节点索引 [num_nodes, compression_rate]
            child_indices = layer_node_indices[:, None] * compression_rate + torch.arange(compression_rate, device=device)
            
            # 下一层的节点索引
            next_list_idx, next_layer_offset, next_num_nodes = layer_info_list[list_idx + 1]
            next_layer_indices = all_node_indices[next_layer_offset:next_layer_offset + next_num_nodes]
            
            # 检查子节点是否被选中 [num_nodes, compression_rate]
            # child_indices [num_nodes, compression_rate, 1] vs next_layer_indices [1, 1, next_num_nodes]
            has_child = (child_indices[:, :, None] == next_layer_indices[None, None, :]).any(dim=(1, 2))
            
            # 标记有子节点的父节点为无效
            valid_mask[layer_offset:layer_offset + num_nodes] = ~has_child
        
        # 3. 提取有效节点
        valid_indices = torch.where(valid_mask)[0]  # [num_valid]
        if valid_indices.numel() == 0:
            continue
        
        valid_start_pos = all_start_pos[valid_indices]
        
        # 4. 按原始序列位置排序，分配RoPE位置索引
        sorted_order = torch.argsort(valid_start_pos)  # [num_valid]
        sorted_valid_indices = valid_indices[sorted_order]  # [num_valid]
        
        # 构建每层的RoPE位置映射: list_idx -> (local_indices, rope_positions)
        layer_rope_mapping = []
        for list_idx, layer_offset, num_nodes in layer_info_list:
            # 找到属于当前层的有效节点
            mask = (sorted_valid_indices >= layer_offset) & (sorted_valid_indices < layer_offset + num_nodes)
            layer_valid_indices = sorted_valid_indices[mask]
            rope_positions = torch.where(mask)[0]  # 在排序后序列中的位置
            
            # 转换为 local_idx（在当前层内的索引）
            local_indices = layer_valid_indices - layer_offset
            layer_rope_mapping.append((local_indices, rope_positions))
        
        num_final = sorted_valid_indices.numel()
        
        # 应用 RoPE 到 query（num_final）
        cos_q, sin_q = rotary.forward(torch.tensor([num_final], device=device), device)
        query_rope = apply_rotary_emb(query[b], cos_q[None, :], sin_q[None, :])  # [H, D]
        
        # ========== parallel：各层独立计算在线注意力统计量 ==========
        
        layer_stats = []  # 每层的统计量: {'max_score': [H, 1], 'weighted_sum': [H, D], 'exp_sum': [H, 1]}
        
        for list_idx, (indices, k, v, ranges) in enumerate(all_layer_selections):
            # 提取该层的有效节点
            local_indices, rope_positions = layer_rope_mapping[list_idx]
            
            if local_indices.numel() == 0:
                continue
            
            # 提取 K, V
            layer_k = k[b, local_indices]  # [num_valid, H, D]
            layer_v = v[b, local_indices]  # [num_valid, H, D]
            
            # 应用 RoPE 到 K
            cos, sin = rotary.forward(rope_positions, device)
            layer_k_rope = apply_rotary_emb(layer_k, cos[:, None, :], sin[:, None, :])  # [num_valid, H, D]
            
            # 计算注意力分数: [H, D] @ [num_valid, H, D]^T -> [H, num_valid]
            scores = torch.einsum('hd,nhd->hn', query_rope * scale, layer_k_rope)  # [H, num_valid]
            
            # 计算在线注意力统计量
            max_score = scores.max(dim=-1, keepdim=True)[0]  # [H, 1]
            exp_scores = torch.exp(scores - max_score)  # [H, num_valid]
            
            # weighted_sum: sum(exp(scores - max) * V)
            weighted_sum = torch.einsum('hn,nhd->hd', exp_scores, layer_v)  # [H, D]
            exp_sum = exp_scores.sum(dim=-1, keepdim=True)  # [H, 1]
            
            layer_stats.append({
                'max_score': max_score,       # [H, 1]
                'weighted_sum': weighted_sum, # [H, D]
                'exp_sum': exp_sum            # [H, 1]
            })
        
        if len(layer_stats) == 0:
            continue
        
        # ========== reduce：用 logsumexp 合并各层的在线注意力结果 ==========
        
        # 全局最大分数
        global_max = torch.stack([stat['max_score'] for stat in layer_stats], dim=0).max(dim=0)[0]  # [H, 1]
        
        total_weighted_sum = torch.zeros(H, D, device=device, dtype=query.dtype)
        total_exp_sum = torch.zeros(H, 1, device=device, dtype=query.dtype)
        
        for stat in layer_stats:
            adjustment = torch.exp(stat['max_score'] - global_max)  # [H, 1]
            total_weighted_sum += stat['weighted_sum'] * adjustment
            total_exp_sum += stat['exp_sum'] * adjustment
        
        # output
        output[b] = total_weighted_sum / total_exp_sum  # [H, D]
    
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
        query_positions: [B, T] or None(each position to be computed)
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
    if query_positions is None:
        positions_to_compute = range(1, T)
    else:
        positions_to_compute = [p for p in query_positions.tolist() if p > 0]
    
    for b in range(B):
        for t in positions_to_compute:
            
            current_query = q[b:b+1, t]  # [1, H, D]
            
            # 3. compute and select
            layer_selections = []
            prev_selected_indices = None
            
            for layer_idx in range(num_layers - 1, -1, -1):
                layer_k, layer_v, node_ranges = layers[layer_idx]
                
                selected_idx, selected_k, selected_v = compute_and_select(
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
                
                # record selected results, update prev_selected_indices
                # selected_ranges: [num_selected, 2]
                selected_ranges = node_ranges[selected_idx[0]]
                layer_selections.append((selected_idx, selected_k, selected_v, selected_ranges))
                prev_selected_indices = selected_idx
            
            # 4. final_compute
            output[b:b+1, t] = final_compute(
                layer_selections,
                current_query,
                rotary,
                compression_rate,
                scale
            )
    
    return output