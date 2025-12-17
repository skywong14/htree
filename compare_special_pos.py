"""
统一的调试脚本：先运行 naive，然后运行 parallel2 (triton V2)，最后对比
针对指定 query_pos 的调试版本
"""

import sys
import torch
import struct

# ANSI 颜色代码
class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def parse_fp32_for_stable_topk(value):
    """
    解析 float32，按 stable_topk 的位布局显示：
    - Sign bit (1 bit)
    - Exponent (8 bits)
    - Mantissa high 10 bits (前部分)
    - Mantissa low 13 bits (后部分，position id 所在位置)
    
    FP32 格式：
    [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
     1    8              23 bits
    
    其中 Mantissa 23 bits 分为：
    - High 10 bits: M[22:13]  (前部分)
    - Low 13 bits:  M[12:0]   (position id 部分)
    """
    if value is None or (isinstance(value, float) and (value != value)):  # None or NaN
        return "N/A", "N/A", "N/A", "N/A"
    
    # 将 float32 转换为 32-bit 整数
    bytes_val = struct.pack('f', value)
    int_val = struct.unpack('I', bytes_val)[0]
    
    # 提取各部分
    sign = (int_val >> 31) & 0x1
    exponent = (int_val >> 23) & 0xFF
    mantissa = int_val & 0x7FFFFF  # 23 bits
    
    # 分离尾数的高10位和低13位
    mantissa_high = (mantissa >> 13) & 0x3FF  # bits[22:13], 10 bits
    mantissa_low = mantissa & 0x1FFF  # bits[12:0], 13 bits
    
    return sign, exponent, mantissa_high, mantissa_low

def format_fp32_stable_topk(value):
    """格式化显示 fp32 的 stable_topk 位布局"""
    sign, exp, m_high, m_low = parse_fp32_for_stable_topk(value)
    
    if sign == "N/A":
        return "N/A"
    
    # 解码 position_id
    position_id = decode_position_id(value)
    pos_str = f"pos={position_id}" if position_id is not None else "pos=N/A"
    
    # 格式：S=0 Exp=127 M_high=0x000 M_low=0x0000 (pos=123) (val=1.000000)
    sign_str = f"S={sign}"
    exp_str = f"Exp={exp:3d}"
    m_high_str = f"M_high=0x{m_high:03X}"
    m_low_str = f"M_low=0x{m_low:04X}"
    val_str = f"(val={value:.9f})"
    
    return f"{sign_str} {exp_str} {m_high_str} {m_low_str} ({pos_str}) {val_str}"

from src.naive_stable_topk import forward_kernel as naive_forward, build_tree, Rotary, apply_rotary_emb, compute_select_and_merge, compute_and_select_only
from src.parallel import htree_forward_v2

print(f"{Color.HEADER}{'='*100}")
print(f"步骤 1: 运行 Naive 实现，记录中间状态")
print(f"{'='*100}{Color.END}")

torch.manual_seed(42)
device = 'cuda'

# 测试配置
B, T, H, K, V = 1, 20000, 1, 8, 8
compression_rate = 16
max_top_nodes = 8192
top_k_per_layer = 512
scale = K ** -0.5
query_pos = 18101

q = torch.randn(B, T, H, K, device=device, dtype=torch.float32).contiguous()
k = torch.randn(B, T, H, K, device=device, dtype=torch.float32).contiguous()
v = torch.randn(B, T, H, V, device=device, dtype=torch.float32).contiguous()

# Naive/Triton 共用同一份 float32 输入；如需半精度，可另行创建副本
q_f32 = q
k_f32 = k
v_f32 = v

# ===== Naive 实现并记录中间状态 =====
print(f"\n配置: B={B}, T={T}, H={H}, K={K}, V={V}")
print(f"compression_rate={compression_rate}, top_k_per_layer={top_k_per_layer}")
print(f"query_pos={query_pos}, scale={scale}")

# 1. Build tree
print("\n构建树结构...")
layers = build_tree(k_f32, v_f32, compression_rate, max_top_nodes)
num_layers = len(layers)
print(f"树有 {num_layers} 层")

# 2. 对单个 query position 运行 naive 并记录中间状态
rotary = Rotary(dim=K).to(device)
b, h, t = 0, 0, query_pos

current_query = q_f32[b:b+1, t, h:h+1]  # [1, 1, D]

# 初始化在线 softmax 状态
max_score = torch.tensor([-float('inf')], device=device)
sum_exp = torch.tensor([0.0], device=device)
weighted_output = torch.zeros(1, 1, V, device=device, dtype=torch.float32)

naive_layer_info = []
prev_selected_indices = None

print(f"\n开始逐层处理 query_pos={query_pos}...")
for layer_idx in range(num_layers - 1, -1, -1):
    layer_k, layer_v, node_ranges = layers[layer_idx]
    
    print(f"\n--- Layer {layer_idx} (N={layer_k.shape[1]}) ---")
    
    # 保存进入该层前的状态
    max_score_before = max_score.item()
    sum_exp_before = sum_exp.item()
    weighted_output_before = weighted_output.clone()
    
    # 调用 compute_select_and_merge (with debug info)
    max_score_before_tensor = torch.tensor([max_score_before], device=device)
    sum_exp_before_tensor = torch.tensor([sum_exp_before], device=device)
    
    result = compute_select_and_merge(
        layer_k[b:b+1, :, h:h+1],
        layer_v[b:b+1, :, h:h+1],
        node_ranges,
        current_query,
        t,
        prev_selected_indices,
        compression_rate,
        layer_idx,
        rotary,
        max_score_before_tensor,
        sum_exp_before_tensor,
        weighted_output_before,
        top_k_per_layer,
        scale,
        return_debug_info=True
    )
    
    max_score, sum_exp, weighted_output, selected_indices, candidate_indices, scores_pooled = result
    
    # 记录该层信息
    layer_info = {
        'layer_idx': layer_idx,
        'N': layer_k.shape[1],
        'max_score_before': max_score_before,
        'sum_exp_before': sum_exp_before,
        'max_score_after': max_score.item(),
        'sum_exp_after': sum_exp.item(),
        'weighted_output_after': weighted_output.squeeze(),
        'selected_indices': selected_indices.squeeze() if selected_indices is not None else None,
        'candidate_indices': candidate_indices,  # Debug info
        'scores_pooled': scores_pooled,  # Debug info
    }
    naive_layer_info.append(layer_info)
    
    print(f"  进入前: max_score={max_score_before:.6f}, sum_exp={sum_exp_before:.6f}")
    print(f"  离开后: max_score={layer_info['max_score_after']:.6f}, sum_exp={layer_info['sum_exp_after']:.6f}")
    if selected_indices is not None:
        num_valid = (selected_indices >= 0).sum().item()
        print(f"  选中节点数: {num_valid}")
    
    prev_selected_indices = selected_indices

# 最终归一化
naive_output = (weighted_output / sum_exp.unsqueeze(-1)).squeeze(0)
print(f"\nNaive 最终输出 (pos={query_pos}): {naive_output.tolist()}")

print(f"\n{Color.HEADER}{'='*100}")
print(f"步骤 2: 运行 Triton V2 实现，记录中间状态")
print(f"{'='*100}{Color.END}")


# 运行完整的 Triton V2 forward
triton_output_full = htree_forward_v2(
    q, k, v,
    compression_rate=compression_rate,
    max_top_nodes=max_top_nodes,
    top_k_per_layer=top_k_per_layer,
    scale=scale,
)

triton_output = triton_output_full[0, query_pos, 0, :]
print(f"\nTriton V2 最终输出 (pos={query_pos}): {triton_output.float().tolist()}")

# 为了获取中间状态，重新运行并插桩记录
print("\n重新运行 Triton V2 实现以获取中间状态...")

# 构建树结构
from src.parallel import htree_build_kernel_v2
import triton

layers_k_triton = [k]
layers_v_triton = [v]
current_k_t, current_v_t = k, v
current_len_t = T

for layer_idx in range(1, num_layers):
    next_len = (current_len_t + compression_rate - 1) // compression_rate
    # 跟 htree_forward_v2 保持一致，使用输入的 dtype
    # TODO 这里换成 dtype=torch.float16 就会一致，但为什么？
    #   Top-K 集合对比:    仅在 Naive 中: 1 个节点      节点: [1080] 1080        0.024033
    #                     仅在 Triton 中: 1 个节点      节点: [186] 186          0.354884 感觉这个计算错了
    # Lyaer 1 差异 max_diff=0.000002, sum_diff=0.000977, output_diff=0.543751
    # 导致 Layer 0————max_score: Naive 3.547955    Triton V2 2.962736      diff: 0.585219
    next_k_t = torch.empty(B, next_len, H, K, dtype=k.dtype, device=device)
    next_v_t = torch.empty(B, next_len, H, V, dtype=v.dtype, device=device)
    
    BLOCK_SIZE = 128
    grid = (B * H,)
    htree_build_kernel_v2[grid](
        current_k_t, current_v_t, next_k_t, next_v_t,
        N_child=current_len_t,
        N_parent=next_len,
        B=B, H=H, K=K, V=V,
        COMPRESSION_RATE=compression_rate,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    layers_k_triton.append(next_k_t)
    layers_v_triton.append(next_v_t)
    current_k_t, current_v_t = next_k_t, next_v_t
    current_len_t = next_len

# 预计算 RoPE cache
cache_size = max_top_nodes + 1024
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
positions = torch.arange(cache_size, dtype=torch.float32, device=device)
freqs = torch.outer(positions, inv_freq)
cos_cache = freqs.cos()
sin_cache = freqs.sin()

# 初始化全局状态
global_max = torch.full([B, T, H], float('-inf'), dtype=torch.float32, device=device)
global_sum = torch.zeros([B, T, H], dtype=torch.float32, device=device)
global_output = torch.zeros([B, T, H, V], dtype=torch.float32, device=device)

# 分配中间状态张量
layer_max = torch.empty([B, T, H], dtype=torch.float32, device=device)
layer_sum = torch.empty([B, T, H], dtype=torch.float32, device=device)
layer_output = torch.empty([B, T, H, V], dtype=torch.float32, device=device)

# 分配 8192 buffer
MAX_CANDIDATES = 8192
all_scores = torch.full([B, T, H, MAX_CANDIDATES], float('-inf'), dtype=torch.float32, device=device)
all_node_indices = torch.full([B, T, H, MAX_CANDIDATES], -1, dtype=torch.int32, device=device)
num_candidates = torch.zeros([B, T, H], dtype=torch.int32, device=device)

topk_positions = torch.full([B, T, H, top_k_per_layer], -1, dtype=torch.int32, device=device)
selected_indices = torch.full([B, T, H, top_k_per_layer], -1, dtype=torch.int32, device=device)

# 初始化 prev_selected_parents
top_layer_power = compression_rate ** (num_layers - 1)
t_indices = torch.arange(T, dtype=torch.int32, device=device)
rightmost_indices = t_indices // top_layer_power
num_virtual_parents = rightmost_indices // compression_rate + 1

parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(T, -1)
valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
prev_selected_parents = torch.where(valid_mask, parent_candidates, torch.tensor(-1, dtype=torch.int32, device=device))
prev_selected_parents = prev_selected_parents.unsqueeze(0).unsqueeze(2).expand(B, T, H, top_k_per_layer).contiguous()

# 逐层运行并记录中间状态
from src.parallel import htree_compute_scores_and_select_kernel, htree_accumulate_non_topk_kernel, htree_merge_to_global_kernel_v2

triton_layer_info = []

print(f"\n开始逐层处理 (Triton V2)...")
for layer_idx in range(num_layers - 1, -1, -1):
    k_layer = layers_k_triton[layer_idx]
    v_layer = layers_v_triton[layer_idx]
    N_layer = k_layer.shape[1]
    
    is_bottom_layer = (layer_idx == 0)
    layer_power = compression_rate ** layer_idx
    
    print(f"\n--- Layer {layer_idx} (N={N_layer}) ---")
    
    grid = (T, B * H)

    # print prev_selected_parents for debug
    print(f"  prev_selected_parents (pos={query_pos}):")
    print(prev_selected_parents[0, query_pos, 0, :].tolist())
    
    # Kernel 2.1: Compute Scores & Select Top-K
    htree_compute_scores_and_select_kernel[grid](
        q, k_layer,  # parallel2 的 Kernel 2.1 不需要 v_layer
        prev_selected_parents,
        cos_cache, sin_cache,
        # 输出 buffer
        all_scores, all_node_indices, num_candidates,
        # 输出 Top-K
        topk_positions, selected_indices,
        layer_idx=layer_idx,
        layer_power=layer_power,
        B=B, T=T, H=H, K=K, V=V, N_layer=N_layer,
        COMPRESSION_RATE=compression_rate,
        TOP_K=top_k_per_layer,
        MAX_CANDIDATES=MAX_CANDIDATES,
        scale=scale,
    )

    # 记录 2.1 之后的 buffer 信息
    n_cand = num_candidates[0, query_pos, 0].item()
    
    # 打印输出 all_scores， all_node_indices
    if not is_bottom_layer:
        print(f"  all_scores (pos={query_pos}, n_cand={n_cand}):")
        print(all_scores[0, query_pos, 0, :n_cand].tolist())
        print(f"  all_node_indices (pos={query_pos}):")
        print(all_node_indices[0, query_pos, 0, :n_cand].tolist())
    
    
    # 输出 2.1 的结果（修改前）
    if not is_bottom_layer:
        print(f"  topk_positions (pos={query_pos}) - 修改前:")
        print(topk_positions[0, query_pos, 0, :512].tolist())
        print(f"  selected_indices (pos={query_pos}) - 修改前:")
        print(selected_indices[0, query_pos, 0, :512].tolist())


    # Special Test
    # 现在我们专注于测试 pos=8225 的情况，在该情况下 Naive 和 Triton 在 Layer1 的 Top-512 选择有一个节点不同，Naive的第512个节点选择了 233 而 Triton 选择了 400
    # 我希望知道这个差异是否导致 Layer0 两种实现的差异（最终输出向量的差异）超过阈值
    # 所以我要尝试手动将 Triton实现中 Top-K 的选择（即selected_indices[511]和topk_positions[511]）从 400 修改到 233
    # 然后检查这样修改后是否差异显著减小
    if layer_idx == 1 and query_pos == 8225:
        print(f"\n  {Color.WARNING}{'='*80}")
        print(f"  【Special Test】手动修改 Layer1 的 Top-K 选择")
        print(f"  {'='*80}{Color.END}")
        
        # 记录修改前的值
        original_selected_idx = selected_indices[0, query_pos, 0, 511].item()
        original_topk_pos = topk_positions[0, query_pos, 0, 511].item()
        
        print(f"  修改前:")
        print(f"    selected_indices[511] = {original_selected_idx}")
        print(f"    topk_positions[511] = {original_topk_pos}")
        
        # 找到节点 233 在 all_node_indices buffer 中的位置
        target_node = 233
        buffer_indices = all_node_indices[0, query_pos, 0, :].clone()
        
        # 查找目标节点在 buffer 中的位置
        target_positions = (buffer_indices == target_node).nonzero(as_tuple=True)[0]
        
        if target_positions.numel() > 0:
            target_buffer_pos = target_positions[0].item()
            
            # 修改 selected_indices 和 topk_positions
            selected_indices[0, query_pos, 0, 511] = target_node
            topk_positions[0, query_pos, 0, 511] = target_buffer_pos
            
            print(f"  修改后:")
            print(f"    selected_indices[511] = {target_node}")
            print(f"    topk_positions[511] = {target_buffer_pos}")
            print(f"  {Color.OKGREEN}✓ 成功修改为与 Naive 一致{Color.END}")
        else:
            print(f"  {Color.FAIL}✗ 错误: 节点 {target_node} 不在候选集中！{Color.END}")
        
        print(f"  {'='*80}\n")

    
    # 记录 2.1 之后的 Top-K 信息（在修改之后，调用 2.2 之前）
    # 这样记录的是修改后的值，用于后续对比
    if not is_bottom_layer:
        topk_pos_after_21 = topk_positions[0, query_pos, 0, :].clone()
        selected_indices_after_21 = selected_indices[0, query_pos, 0, :].clone()
    else:
        topk_pos_after_21 = None
        selected_indices_after_21 = None

    

    # Kernel 2.2: Accumulate Non-TopK
    htree_accumulate_non_topk_kernel[grid](
        v_layer,
        prev_selected_parents,
        # Kernel 2.1 的输出
        all_scores, num_candidates, topk_positions,
        # 输出
        layer_max, layer_sum, layer_output,
        layer_idx=layer_idx,
        layer_power=layer_power,
        B=B, T=T, H=H, V=V, N_layer=N_layer,
        COMPRESSION_RATE=compression_rate,
        TOP_K=top_k_per_layer,
        MAX_CANDIDATES=MAX_CANDIDATES,
    )

    if is_bottom_layer:
        # debug Print all_scores after 2.2
        print(f"  all_scores after 2.2 (pos={query_pos}, n_cand={n_cand}):")
        print(all_scores[0, query_pos, 0, :20].tolist())
        # debug Print layer_output after 2.2
        print(f"  layer_output after 2.2 (pos={query_pos}):")
        print(layer_output[0, query_pos, 0, :].tolist())

    
    # 记录 2.2 之后的状态
    layer_max_after_22 = layer_max[0, query_pos, 0].item()
    layer_sum_after_22 = layer_sum[0, query_pos, 0].item()
    layer_output_after_22 = layer_output[0, query_pos, 0, :].clone()
    
    # Kernel 2.3: Merge to Global
    htree_merge_to_global_kernel_v2[grid](
        layer_max, layer_sum, layer_output,
        global_max, global_sum, global_output,
        B=B, T=T, H=H, V=V,
    )
    
    # 记录 2.3 之后的状态
    global_max_after = global_max[0, query_pos, 0].item()
    global_sum_after = global_sum[0, query_pos, 0].item()
    global_output_after = global_output[0, query_pos, 0, :].clone()
    
    # 记录该层信息（包括该层的 all_scores 和 all_node_indices）
    layer_info = {
        'layer_idx': layer_idx,
        'N': N_layer,
        'n_cand': n_cand,
        'layer_max_after_22': layer_max_after_22,
        'layer_sum_after_22': layer_sum_after_22,
        'layer_output_after_22': layer_output_after_22,
        'global_max': global_max_after,
        'global_sum': global_sum_after,
        'global_output': global_output_after,
        'selected_indices': selected_indices_after_21,  # Kernel 2.1 的输出
        'topk_positions': topk_pos_after_21,  # Kernel 2.1 的输出
        # 保存该层的 all_scores 和 all_node_indices（克隆以避免被后续层覆盖）
        'all_scores': all_scores[0, query_pos, 0, :].clone(),
        'all_node_indices': all_node_indices[0, query_pos, 0, :].clone(),
    }
    triton_layer_info.append(layer_info)
    
    print(f"  候选节点数: {n_cand}")
    print(f"  2.2 后: layer_max={layer_max_after_22:.6f}, layer_sum={layer_sum_after_22:.6f}")
    print(f"  2.3 后: global_max={global_max_after:.6f}, global_sum={global_sum_after:.6f}")
    if not is_bottom_layer:
        num_valid_sel = (selected_indices_after_21 >= 0).sum().item()
        print(f"  选中节点数: {num_valid_sel}")
    
    # 升序排序 selected_indices (注意 -1 的padding放到最后)
    # 将 -1 临时替换为一个很大的值，使其在排序后位于末尾
    # INT_MAX = 2147483647
    # indices_for_sort = torch.where(
    #     selected_indices == -1,
    #     torch.tensor(INT_MAX, dtype=torch.int32, device=device),
    #     selected_indices
    # )
    # sorted_indices, _ = torch.sort(indices_for_sort, dim=-1, stable=True)
    # # 将大值替换回 -1
    # selected_indices = torch.where(
    #     sorted_indices == INT_MAX,
    #     torch.tensor(-1, dtype=torch.int32, device=device),
    #     sorted_indices
    # )
    
    # 更新 prev_selected_parents（交换缓冲区即可复用内存）
    if not is_bottom_layer:
        prev_selected_parents, selected_indices = selected_indices, prev_selected_parents

print(f"\n{Color.HEADER}{'='*100}")
print(f"步骤 3: 详细对比每一层的中间状态")
print(f"{'='*100}{Color.END}")

for idx in range(len(naive_layer_info)):
    layer_idx = naive_layer_info[idx]['layer_idx']
    
    # 对比 Layer 1
    if layer_idx not in [0,1]:
        continue
    
    print(f"\n{Color.HEADER}{'='*100}")
    print(f"【Layer {layer_idx}】 节点数: {naive_layer_info[idx]['N']}")
    print(f"{'='*100}{Color.END}")
    
    naive_info = naive_layer_info[idx]
    triton_info = triton_layer_info[idx]
    
    # 验证层索引一致
    if naive_info['layer_idx'] != triton_info['layer_idx']:
        print(f"⚠️ 警告：层索引不匹配！")
        continue
    
    # ========================================
    # 0. Layer 自身的输出状态（合并前）
    # ========================================
    print(f"\n{Color.OKBLUE}【0. Layer {layer_idx} 自身的输出状态（Kernel 2.2 后，合并到全局前）】{Color.END}")
    print(f"  {'-'*80}")
    print(f"  {'参数':<20} {'Naive':<20} {'Triton V2':<20} {'差异':<15}")
    print(f"  {'-'*80}")
    
    naive_max_before_merge = naive_info['max_score_after']
    naive_sum_before_merge = naive_info['sum_exp_after']
    naive_wo_before_merge = naive_info['weighted_output_after'].float()
    
    triton_max_before_merge = triton_info['layer_max_after_22']
    triton_sum_before_merge = triton_info['layer_sum_after_22']
    triton_wo_before_merge = triton_info['layer_output_after_22'].float()
    
    diff_max_bm = abs(naive_max_before_merge - triton_max_before_merge)
    diff_sum_bm = abs(naive_sum_before_merge - triton_sum_before_merge)
    diff_wo_bm = (naive_wo_before_merge - triton_wo_before_merge).abs()
    
    color_max_bm = Color.OKGREEN if diff_max_bm < 1e-4 else Color.WARNING if diff_max_bm < 0.1 else Color.FAIL
    color_sum_bm = Color.OKGREEN if (diff_sum_bm / naive_sum_before_merge if naive_sum_before_merge > 0 else diff_sum_bm) < 0.01 else Color.WARNING if (diff_sum_bm / naive_sum_before_merge if naive_sum_before_merge > 0 else diff_sum_bm) < 0.1 else Color.FAIL
    
    print(f"  {'max_score':<20} {naive_max_before_merge:>19.6f} {triton_max_before_merge:>19.6f} {color_max_bm}{diff_max_bm:>14.6f}{Color.END}")
    print(f"  {'sum_exp':<20} {naive_sum_before_merge:>19.6f} {triton_sum_before_merge:>19.6f} {color_sum_bm}{diff_sum_bm:>14.6f}{Color.END}")
    if naive_sum_before_merge > 0:
        print(f"  {'sum_exp (相对)':<20} {'':<20} {'':<20} {color_sum_bm}{diff_sum_bm/naive_sum_before_merge*100:>13.2f}%{Color.END}")
    print(f"  {'output (max)':<20} {naive_wo_before_merge.max().item():>19.6f} {triton_wo_before_merge.max().item():>19.6f} {diff_wo_bm.max().item():>14.6f}")
    print(f"  {'output (mean)':<20} {naive_wo_before_merge.mean().item():>19.6f} {triton_wo_before_merge.mean().item():>19.6f} {diff_wo_bm.mean().item():>14.6f}")
    print(f"  {'output (norm)':<20} {naive_wo_before_merge.norm().item():>19.6f} {triton_wo_before_merge.norm().item():>19.6f} {(naive_wo_before_merge.norm()-triton_wo_before_merge.norm()).abs().item():>14.6f}")
    
    print(f"\n  【详细输出向量 (weighted_output) - 合并前】")
    print(f"  Naive:     {naive_wo_before_merge.tolist()}")
    print(f"  Triton V2: {triton_wo_before_merge.tolist()}")
    print(f"  差异:      {diff_wo_bm.tolist()}")
    print(f"  最大差异:  {diff_wo_bm.max().item():.6f}")
    print(f"  平均差异:  {diff_wo_bm.mean().item():.6f}")
    
    # 归一化后的输出（合并前）
    if naive_sum_before_merge > 0 and triton_sum_before_merge > 0:
        naive_normalized_bm = naive_wo_before_merge / naive_sum_before_merge
        triton_normalized_bm = triton_wo_before_merge / triton_sum_before_merge
        diff_normalized_bm = (naive_normalized_bm - triton_normalized_bm).abs()
        
        print(f"\n  【归一化后的输出向量 (output = weighted_output / sum_exp) - 合并前】")
        print(f"  Naive:     {naive_normalized_bm.tolist()}")
        print(f"  Triton V2: {triton_normalized_bm.tolist()}")
        print(f"  差异:      {diff_normalized_bm.tolist()}")
        print(f"  最大差异:  {diff_normalized_bm.max().item():.6f}")
        print(f"  平均差异:  {diff_normalized_bm.mean().item():.6f}")
    
    # ========================================
    # 1. Top-K 节点详细对比（按 Kernel 返回顺序）
    # ========================================
    print(f"\n{Color.OKBLUE}【1. Top-K 节点对比 - 按 Kernel 返回顺序（前20项）】{Color.END}")
    
    # 获取 Naive 和 Triton 的 Top-K 节点索引
    naive_sel = naive_info['selected_indices']
    triton_sel = triton_info['selected_indices']
    triton_topk_pos = triton_info['topk_positions']
    
    # Layer 0 是底层，selected_indices 为 None
    if naive_sel is None or triton_sel is None:
        print(f"  Layer {layer_idx} 是底层，不需要选择 Top-K 节点，跳过此部分对比")
    else:
        # 准备 Naive 数据
        naive_candidate_indices = naive_info['candidate_indices'].cpu()
        naive_scores = naive_info['scores_pooled'].cpu()
        naive_valid = naive_sel[naive_sel >= 0].cpu()
        
        # 准备 Triton 数据（使用该层保存的 all_scores 和 all_node_indices）
        triton_all_scores = triton_info['all_scores'].cpu()
        triton_all_indices = triton_info['all_node_indices'].cpu()
        triton_valid = triton_sel[triton_sel >= 0].cpu()
        triton_topk_pos_valid = triton_topk_pos[triton_topk_pos >= 0].cpu() if triton_topk_pos is not None else None
        
        print(f"  Naive Top-K 数量:  {len(naive_valid)}")
        print(f"  Triton Top-K 数量: {len(triton_valid)}")
        
        # 构建 Naive Top-K 列表（节点，分数）
        naive_topk_list = []
        for node_idx in naive_valid:
            pos = (naive_candidate_indices == node_idx).nonzero(as_tuple=True)[0]
            score = naive_scores[pos[0]].item() if pos.numel() > 0 else float('nan')
            naive_topk_list.append((node_idx.item(), score))
        
        # 构建 Triton Top-K 列表（节点，分数）
        # 使用节点索引从 all_scores 中提取分数（topk_positions 与 selected_indices 次序不同）
        triton_topk_list = []
        # 先构建节点->score 映射，确保分数与节点匹配
        triton_score_map = {}
        for node_idx in triton_valid:
            matches = (triton_all_indices == node_idx).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                triton_score_map[node_idx.item()] = triton_all_scores[matches[0]].item()
        for node_idx in triton_valid:
            triton_topk_list.append((node_idx.item(), triton_score_map.get(node_idx.item(), float('nan'))))
        if triton_topk_pos_valid is None:
            print(f"  {Color.FAIL}⚠️  提示：缺少 topk_positions 数据，已改为按节点查分数{Color.END}")
        
        # TODO DEBUG 特别关注 Layer1 中节点 1080 和 186 的分数
        if layer_idx == 1:
            print(f"\n  {Color.BOLD}{Color.OKCYAN}【Layer 1 特殊节点分数】{Color.END}")
            print(f"  {Color.OKCYAN}注意：这里显示的分数是【编码后的分数】{Color.END}")
            print(f"  {Color.OKCYAN}M_low (低13位) 包含编码的 position_id（buffer位置）{Color.END}")
            print(f"  {Color.OKCYAN}对于正数分数，position_id 被取反编码；对于负数分数，position_id 保持原值{Color.END}\n")
            target_nodes = [1080, 186]
            for target in target_nodes:
                naive_val = None
                triton_val = None
                
                naive_pos = (naive_candidate_indices == target).nonzero(as_tuple=True)[0]
                if naive_pos.numel() > 0:
                    naive_val = naive_scores[naive_pos[0]].item()
                
                triton_pos = (triton_all_indices == target).nonzero(as_tuple=True)[0]
                if triton_pos.numel() > 0:
                    triton_val = triton_all_scores[triton_pos[0]].item()
                
                if naive_val is not None and triton_val is not None:
                    diff = abs(naive_val - triton_val)
                    highlight = Color.OKGREEN if diff < 1e-4 else Color.WARNING if diff < 0.1 else Color.FAIL
                else:
                    highlight = Color.WARNING
                
                # 使用 stable_topk 位布局格式显示
                naive_str = format_fp32_stable_topk(naive_val)
                triton_str = format_fp32_stable_topk(triton_val)
                
                print(f"  {highlight}节点 {target}:{Color.END}")
                print(f"    Naive:     {naive_str}")
                print(f"    Triton V2: {triton_str}")

        # 打印对比表格（前20项）
        print(f"\n  {'位置':<6} {'Naive节点':<12} {'Naive分数':<15} {'Triton节点':<12} {'Triton分数':<15} {'节点匹配':<10}")
        print(f"  {'-'*80}")
        
        for i in range(min(512, max(len(naive_topk_list), len(triton_topk_list)))):
            naive_node = naive_topk_list[i][0] if i < len(naive_topk_list) else -1
            naive_score = naive_topk_list[i][1] if i < len(naive_topk_list) else float('nan')
            
            triton_node = triton_topk_list[i][0] if i < len(triton_topk_list) else -1
            triton_score = triton_topk_list[i][1] if i < len(triton_topk_list) else float('nan')
            
            match = "✓" if naive_node == triton_node else "✗"
            match_color = Color.OKGREEN if naive_node == triton_node else Color.FAIL
            
            print(f"  {i+1:<6} {naive_node:<12} {naive_score:<15.6f} {triton_node:<12} {triton_score:<15.6f} {match_color}{match:<10}{Color.END}")
        
        # 统计位置匹配
        # matches = sum(1 for i in range(min(len(naive_topk_list), len(triton_topk_list))) 
        #               if naive_topk_list[i][0] == triton_topk_list[i][0])
        # total = min(len(naive_topk_list), len(triton_topk_list))
        
        # if total > 0:
        #     print(f"\n  位置匹配统计: {matches}/{total} ({100*matches/total:.1f}% 在相同位置选择了相同节点)")
        
        # 检查 Top-K 集合是否一致（不考虑顺序）
        naive_set = set(n for n, _ in naive_topk_list)
        triton_set = set(n for n, _ in triton_topk_list)
        
        common = naive_set & triton_set
        only_naive = naive_set - triton_set
        only_triton = triton_set - naive_set
        
        print(f"\n  Top-K 集合对比:")
        print(f"    共同节点数: {len(common)}/{len(naive_set)}")
        if only_naive:
            print(f"    仅在 Naive 中: {len(only_naive)} 个节点")
            if len(only_naive) <= 10:
                print(f"      节点: {sorted(only_naive)}")
        if only_triton:
            print(f"    仅在 Triton 中: {len(only_triton)} 个节点")
            if len(only_triton) <= 10:
                print(f"      节点: {sorted(only_triton)}")
        
        if len(common) == len(naive_set) == len(triton_set):
            print(f"  {Color.OKGREEN}✓ Top-K 集合完全一致{Color.END}")
        else:
            print(f"  {Color.WARNING}⚠️  Top-K 集合不一致！{Color.END}")
    
    
    # ========================================
    # 2. 合并到全局后的状态对比
    # ========================================
    print(f"\n{Color.OKBLUE}【2. 合并到全局后的状态对比】{Color.END}")
    print(f"  {'-'*80}")
    print(f"  {'参数':<20} {'Naive':<20} {'Triton V2':<20} {'差异':<15}")
    print(f"  {'-'*80}")
    
    # Naive 在处理完该层后的累积状态
    naive_global_max = naive_info['max_score_after']
    naive_global_sum = naive_info['sum_exp_after']
    naive_global_wo = naive_info['weighted_output_after'].float()
    
    # Triton 合并到全局后的状态
    triton_global_max = triton_info['global_max']
    triton_global_sum = triton_info['global_sum']
    triton_global_wo = triton_info['global_output'].float()
    
    diff_global_max = abs(naive_global_max - triton_global_max)
    diff_global_sum = abs(naive_global_sum - triton_global_sum)
    diff_global_wo = (naive_global_wo - triton_global_wo).abs()
    
    # 颜色标记
    color_global_max = Color.OKGREEN if diff_global_max < 1e-4 else Color.WARNING if diff_global_max < 0.1 else Color.FAIL
    color_global_sum = Color.OKGREEN if diff_global_sum / max(naive_global_sum, 1e-10) < 0.01 else Color.WARNING if diff_global_sum / max(naive_global_sum, 1e-10) < 0.1 else Color.FAIL
    
    print(f"  {'global_max':<20} {naive_global_max:>19.6f} {triton_global_max:>19.6f} {color_global_max}{diff_global_max:>14.6f}{Color.END}")
    print(f"  {'global_sum':<20} {naive_global_sum:>19.6f} {triton_global_sum:>19.6f} {color_global_sum}{diff_global_sum:>14.6f}{Color.END}")
    if naive_global_sum > 1e-10:
        print(f"  {'global_sum (相对)':<20} {'':<20} {'':<20} {color_global_sum}{diff_global_sum/naive_global_sum*100:>13.2f}%{Color.END}")
    print(f"  {'global_output (max)':<20} {naive_global_wo.max().item():>19.6f} {triton_global_wo.max().item():>19.6f} {diff_global_wo.max().item():>14.6f}")
    print(f"  {'global_output (mean)':<20} {naive_global_wo.mean().item():>19.6f} {triton_global_wo.mean().item():>19.6f} {diff_global_wo.mean().item():>14.6f}")
    print(f"  {'global_output (norm)':<20} {naive_global_wo.norm().item():>19.6f} {triton_global_wo.norm().item():>19.6f} {(naive_global_wo.norm()-triton_global_wo.norm()).abs().item():>14.6f}")
    
    # 详细的输出向量对比
    print(f"\n  【详细输出向量 (global_output) - 合并后】")
    print(f"  Naive:     {naive_global_wo.tolist()}")
    print(f"  Triton V2: {triton_global_wo.tolist()}")
    print(f"  差异:      {diff_global_wo.tolist()}")
    
    # 归一化后的输出
    naive_normalized_global = naive_global_wo / max(naive_global_sum, 1e-10)
    triton_normalized_global = triton_global_wo / max(triton_global_sum, 1e-10)
    diff_normalized_global = (naive_normalized_global - triton_normalized_global).abs()
    
    print(f"\n  【归一化后的输出向量 (output = global_output / global_sum)】")
    print(f"  Naive:     {naive_normalized_global.tolist()}")
    print(f"  Triton V2: {triton_normalized_global.tolist()}")
    print(f"  差异:      {diff_normalized_global.tolist()}")
    print(f"  最大差异:  {diff_normalized_global.max().item():.6f}")
    print(f"  平均差异:  {diff_normalized_global.mean().item():.6f}")
    
    # 检查差异是否在可接受范围
    print(f"\n  {'='*80}")
    if diff_global_max > 0.1 or diff_global_sum > 0.1 or diff_global_wo.max().item() > 0.1:
        print(f"  {Color.FAIL}⚠️  警告：Layer {layer_idx} 发现明显差异！{Color.END}")
        print(f"     max_diff={diff_global_max:.6f}, sum_diff={diff_global_sum:.6f}, output_diff={diff_global_wo.max().item():.6f}")
    elif diff_normalized_global.max().item() > 0.01:
        print(f"  {Color.WARNING}⚠️  警告：Layer {layer_idx} 归一化后输出差异较大 ({diff_normalized_global.max().item():.6f}){Color.END}")
    else:
        print(f"  {Color.OKGREEN}✓ Layer {layer_idx} 的差异在可接受范围内{Color.END}")

print(f"\n{Color.HEADER}{'='*100}")
print(f"最终输出对比:")
print(f"{'='*100}{Color.END}")
print(f"\nNaive 输出 (pos={query_pos}):")
print(f"  {naive_output.tolist()}")
print(f"\nTriton V2 输出 (pos={query_pos}):")
print(f"  {triton_output.float().tolist()}")

diff = (naive_output.float() - triton_output.float()).abs()
print(f"\n差异:")
print(f"  {diff.tolist()}")
print(f"  最大差异: {diff.max().item():.6f}")
print(f"  平均差异: {diff.mean().item():.6f}")

if diff.max().item() < 0.01:
    print(f"\n{Color.OKGREEN}{Color.BOLD}✓ 测试通过: 差异在可接受范围内{Color.END}")
else:
    print(f"\n{Color.FAIL}{Color.BOLD}✗ 测试失败: 差异过大{Color.END}")

import time
end_time = time.localtime()
print(f"\n测试完成时间————{end_time.tm_hour}:{end_time.tm_min}")
