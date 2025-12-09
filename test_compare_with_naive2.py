"""
测试端到端 forward kernel V2 - 与 naive 实现对比

对比 parallel2.py (基于 naive 思路的 Triton 实现) 与 naive.py 的输出
"""

import sys
import torch
from src.parallel2 import htree_forward_v2
from src.naive import forward_kernel as naive_forward


def test_compare_with_naive():
    """与 naive 实现对比"""
    # 设置随机种子保证可重现
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试数据
    B, T, H, K, V = 1, 20000, 1, 8, 8
    
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float16).contiguous()
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float16).contiguous()
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float16).contiguous()

    compression_rate = 16
    max_top_nodes = 8192
    top_k_per_layer = 512
    scale = K ** -0.5
    
    print("\n" + "="*80 + "\n")
    print(f"\n对比测试配置: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"compression_rate={compression_rate}, top_k_per_layer={top_k_per_layer}")
    print(f"scale={scale}")
    print("="*80 + "\n")
    
    # 只测试几个 query 位置
    test_positions = [10000]
    query_positions = torch.tensor([test_positions], device=device).expand(B, -1)
    
    # Naive 实现
    print("\n运行 naive 实现...")
    try:
        output_naive = naive_forward(
            q.float(), k.float(), v.float(),
            query_positions=query_positions,
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale
        )
        print(f"✓ Naive forward 完成: output shape {output_naive.shape}")
    except Exception as e:
        print(f"✗ Naive forward 失败: {e}")
        raise
    
    # Triton V2 实现
    print("\n运行 Triton V2 实现...")
    try:
        output_triton_v2 = htree_forward_v2(
            q, k, v,
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale,
        )
        print(f"✓ Triton V2 forward 完成: output shape {output_triton_v2.shape}")
    except Exception as e:
        print(f"✗ Triton V2 forward 失败: {e}")
        raise
    
    # 只比较测试的位置
    print("\n" + "="*80)
    print("误差分析:")
    print("="*80)
    
    all_pass = True
    for pos in test_positions:
        # 比较实现结果
        naive_out = output_naive[:, pos, :, :].float()
        triton_out = output_triton_v2[:, pos, :, :].float()
        
        # 计算误差
        abs_diff = torch.abs(naive_out - triton_out)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        
        rel_diff = abs_diff / (torch.abs(naive_out) + 1e-6)
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        # 检查是否通过
        pass_abs = max_abs_diff < 0.0003
        pass_mean = mean_abs_diff < 0.00003
        status = "✓" if (pass_abs and pass_mean) else "✗"
        
        print(f"\n{status} 位置 {pos:3d}: "
              f"abs_diff=[max={max_abs_diff:.6f}, mean={mean_abs_diff:.6f}], "
              f"rel_diff=[max={max_rel_diff:.6f}, mean={mean_rel_diff:.6f}]")
        
        if not (pass_abs and pass_mean):
            all_pass = False
            
        # 打印更详细的信息用于调试
        print(f"\n  详细数值对比:")
        print(f"    Naive output:     {naive_out[0, 0, :]}")
        print(f"    Triton V2 output: {triton_out[0, 0, :]}")
        print(f"    Difference:       {abs_diff[0, 0, :]}")
    
    print("="*80)
    if all_pass:
        print("✓ 所有位置测试通过!")
    else:
        print("✗ 部分位置测试失败!")
    print("="*80)
    
    return all_pass


if __name__ == "__main__":
    try:
        success = test_compare_with_naive()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

