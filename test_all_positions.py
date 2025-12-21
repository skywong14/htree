"""
测试全部位置的 naive_stable_topk 和 parallel2_stable_topk 实现对比
(支持 Group Query Attention)

两个实现都使用 Bit-Packing Top-K 技术，确保结果一致
"""

import sys
import argparse
import torch
from src.parallel import htree_forward_v2
from src.naive_stable_topk import forward_kernel as naive_forward


def _cuda_sync_if_needed(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _warmup_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    compression_rate: int,
    max_top_nodes: int,
    top_k_per_layer: int,
    scale: float,
    warmup_iters: int,
    device: str,
) -> None:
    if warmup_iters <= 0:
        return
    if device != "cuda":
        # Triton only runs on CUDA; keep behavior explicit.
        return

    print(f"\n[Warmup] 预热 Triton kernel: {warmup_iters} 次 ...")
    with torch.no_grad():
        for i in range(warmup_iters):
            _ = htree_forward_v2(
                q, k, v,
                compression_rate=compression_rate,
                max_top_nodes=max_top_nodes,
                top_k_per_layer=top_k_per_layer,
                scale=scale,
            )
            _cuda_sync_if_needed(device)
            print(f"[Warmup] 完成 {i + 1}/{warmup_iters}")
    print("[Warmup] Triton 预热完成。\n")


def test_all_positions(*, warmup_triton_iters: int = 2, compare_naive: bool = True):
    """测试所有位置的输出结果"""
    # 设置随机种子保证可重现
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试配置
    # GQA Configuration: H (Query Heads), H_kv (KV Heads)
    B, T, H, H_kv, K, V = 1, 20000, 16, 4, 4, 4
    
    # 使用 float32，确保 naive 与 triton 输入精度一致
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32).contiguous()
    k = torch.randn(B, T, H_kv, K, device=device, dtype=torch.float32).contiguous()
    v = torch.randn(B, T, H_kv, V, device=device, dtype=torch.float32).contiguous()

    compression_rate = 16
    max_top_nodes = 8192
    top_k_per_layer = 512
    scale = K ** -0.5
    
    print("\n" + "="*80)
    print(f"全位置对比测试配置 (GQA): B={B}, T={T}, H={H}, H_kv={H_kv}, K={K}, V={V}")
    print(f"compression_rate={compression_rate}, top_k_per_layer={top_k_per_layer}")
    print(f"scale={scale}")
    print(f"compare_naive={compare_naive}")
    print("="*80 + "\n")
    
    # 测试所有位置
    test_positions = list(range(T))
    query_positions = torch.tensor([test_positions], device=device).expand(B, -1)
    
    # Naive Stable TopK 实现 (Bit-Packing 优化版)
    output_naive = None
    if compare_naive:
        print("运行 naive_stable_topk 实现 (Bit-Packing 优化版)...")
        try:
            output_naive = naive_forward(
                q, k, v,
                query_positions=query_positions,
                compression_rate=compression_rate,
                max_top_nodes=max_top_nodes,
                top_k_per_layer=top_k_per_layer,
                scale=scale
            )
            print(f"✓ Naive Stable TopK forward 完成: output shape {output_naive.shape}\n")
        except Exception as e:
            print(f"✗ Naive Stable TopK forward 失败: {e}")
            raise
    else:
        print("跳过 naive_stable_topk 实现的运行 (--compare-naive=False)\n")

    # Triton warmup
    _warmup_triton(
        q, k, v,
        compression_rate=compression_rate,
        max_top_nodes=max_top_nodes,
        top_k_per_layer=top_k_per_layer,
        scale=scale,
        warmup_iters=warmup_triton_iters,
        device=device,
    )
    
    # Parallel2 Stable TopK 实现 (Bit-Packing 优化版)
    print("运行 Parallel2 Stable TopK 实现 (Bit-Packing 优化版)...")
    try:
        _cuda_sync_if_needed(device)
        output_triton = htree_forward_v2(
            q, k, v,
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale,
        )
        _cuda_sync_if_needed(device)
        print(f"✓ Parallel2 Stable TopK forward 完成: output shape {output_triton.shape}\n")
    except Exception as e:
        print(f"✗ Parallel2 Stable TopK forward 失败: {e}")
        raise
    
    # 误差分析
    if compare_naive and output_naive is not None:
        print("="*80)
        print("开始逐位置误差分析...")
        print("="*80 + "\n")
        
        # 统计信息
        all_max_abs_diffs = []
        all_mean_abs_diffs = []
        all_max_rel_diffs = []
        all_mean_rel_diffs = []
        failed_positions = []
        
        # 用于显示进度
        report_interval = 1000
        
        for i, pos in enumerate(test_positions):
            # 比较结果
            naive_out = output_naive[:, pos, :, :].float()
            triton_out = output_triton[:, pos, :, :].float()
            
            # 计算误差
            abs_diff = torch.abs(naive_out - triton_out)
            max_abs_diff = abs_diff.max().item()
            mean_abs_diff = abs_diff.mean().item()
            
            rel_diff = abs_diff / (torch.abs(naive_out) + 1e-6)
            max_rel_diff = rel_diff.max().item()
            mean_rel_diff = rel_diff.mean().item()
            
            # 记录统计
            all_max_abs_diffs.append(max_abs_diff)
            all_mean_abs_diffs.append(mean_abs_diff)
            all_max_rel_diffs.append(max_rel_diff)
            all_mean_rel_diffs.append(mean_rel_diff)
            
            # 检查是否通过（宽松阈值）
            pass_abs = max_abs_diff < 0.01
            pass_mean = mean_abs_diff < 0.01
            
            if not (pass_abs and pass_mean):
                failed_positions.append({
                    'pos': pos,
                    'max_abs': max_abs_diff,
                    'mean_abs': mean_abs_diff,
                    'max_rel': max_rel_diff,
                    'mean_rel': mean_rel_diff,
                    'naive': naive_out[0, 0, :].cpu(),
                    'triton': triton_out[0, 0, :].cpu(),
                })
            
            # 定期报告进度
            if (i + 1) % report_interval == 0 or i == 0 or i == len(test_positions) - 1:
                status = "✓" if (pass_abs and pass_mean) else "✗"
                print(f"{status} 位置 {pos:5d}: "
                      f"abs=[max={max_abs_diff:.6f}, mean={mean_abs_diff:.6f}], "
                      f"rel=[max={max_rel_diff:.6f}, mean={mean_rel_diff:.6f}]")
        
        # 汇总统计
        print("\n" + "="*80)
        print("统计汇总:")
        print("="*80)
        
        max_abs_array = torch.tensor(all_max_abs_diffs)
        mean_abs_array = torch.tensor(all_mean_abs_diffs)
        max_rel_array = torch.tensor(all_max_rel_diffs)
        mean_rel_array = torch.tensor(all_mean_rel_diffs)
        
        print(f"\n绝对误差 (max):")
        print(f"  最小值: {max_abs_array.min().item():.8f}")
        print(f"  最大值: {max_abs_array.max().item():.8f}")
        print(f"  平均值: {max_abs_array.mean().item():.8f}")
        print(f"  中位数: {max_abs_array.median().item():.8f}")
        print(f"  标准差: {max_abs_array.std().item():.8f}")
        
        print(f"\n绝对误差 (mean):")
        print(f"  最小值: {mean_abs_array.min().item():.8f}")
        print(f"  最大值: {mean_abs_array.max().item():.8f}")
        print(f"  平均值: {mean_abs_array.mean().item():.8f}")
        print(f"  中位数: {mean_abs_array.median().item():.8f}")
        print(f"  标准差: {mean_abs_array.std().item():.8f}")
        
        print(f"\n相对误差 (max):")
        print(f"  最小值: {max_rel_array.min().item():.8f}")
        print(f"  最大值: {max_rel_array.max().item():.8f}")
        print(f"  平均值: {max_rel_array.mean().item():.8f}")
        print(f"  中位数: {max_rel_array.median().item():.8f}")
        print(f"  标准差: {max_rel_array.std().item():.8f}")
        
        print(f"\n相对误差 (mean):")
        print(f"  最小值: {mean_rel_array.min().item():.8f}")
        print(f"  最大值: {mean_rel_array.max().item():.8f}")
        print(f"  平均值: {mean_rel_array.mean().item():.8f}")
        print(f"  中位数: {mean_rel_array.median().item():.8f}")
        print(f"  标准差: {mean_rel_array.std().item():.8f}")
        
        # 报告失败位置
        if failed_positions:
            print(f"\n" + "="*80)
            print(f"失败位置数量: {len(failed_positions)}/{T}")
            print("="*80)
            
            # 显示前100个失败位置的详细信息
            print(f"\n显示前{min(100, len(failed_positions))}个失败位置的详细信息:\n")
            for i, fail_info in enumerate(failed_positions[:100]):
                print(f"位置 {fail_info['pos']}:")
                print(f"  max_abs={fail_info['max_abs']:.6f}, mean_abs={fail_info['mean_abs']:.6f}")
                print(f"  max_rel={fail_info['max_rel']:.6f}, mean_rel={fail_info['mean_rel']:.6f}")
                print(f"  Naive_StableTopK:     {fail_info['naive']}")
                print(f"  Parallel2_StableTopK: {fail_info['triton']}")
                print()
        else:
            print(f"\n✓ 所有{T}个位置测试通过!")
        
        print("="*80)
        
        return len(failed_positions) == 0
    else:
        # 不进行对比，仅运行 triton 版本
        print(f"\n✓ Parallel2 Stable TopK forward 已完成")
        print("="*80)
        return True


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="All-positions correctness test for stable top-k (naive vs triton).")
        parser.add_argument(
            "--warmup-triton-iters",
            type=int,
            default=2,
            help="Warm up triton kernels by running htree_forward_v2 N times before the real run (default: 2).",
        )
        parser.add_argument(
            "--no-compare",
            action="store_true",
            help="Disable running/comparing the naive implementation.",
        )
        args = parser.parse_args()

        success = test_all_positions(warmup_triton_iters=args.warmup_triton_iters, compare_naive=(not args.no_compare))
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
