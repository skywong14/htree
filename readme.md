# htree 项目文档

## 文档结构

本项目包含以下文档：

1. **`doc/htree.md`** - htree 算法的核心设计文档
   - htree 的树状结构和存储方式
   - 稀疏注意力机制设计
   - Forward Kernel 的概念流程（PyTorch 朴素实现）

2. **`doc/parallel_design.md`** - Triton kernel 设计方案
   - 分层归约架构设计
   - 核心问题与技术挑战分析
   - Kernel 功能设计和调用流程
   - 性能预期与优化方向

## 核心程序与参考实现

本项目目前的核心程序与参考实现如下：

- **核心程序 (Triton)**: `src/parallel.py`
  - 实现了高效的 Triton kernel 版本。
  - 采用 Stable Top-K 机制（Bit-Packing + Bitonic Sort）以保证与 PyTorch 实现的数值一致性。
  - 包含树构建、逐层前向传播（三阶段处理）、最终归一化等完整流程。

- **参考实现 (PyTorch)**: `src/naive_stable_topk.py`
  - 朴素实现的 PyTorch kernel，作为正确性验证的基准 ("Golden Reference")。
  - 同样实现了 Stable Top-K 逻辑，用于模拟 Triton kernel 的行为。

- **辅助工具**:
  - `compare_special_pos.py`: 深度调试工具。针对特定 query 位置（如边界情况），逐层对比 Triton 与 Naive 实现的中间状态（Score、Top-K 选择、Softmax 累积值），帮助精确定位数值差异。
  - `test_all_positions.py`: 全量测试工具。在大量数据上对比 Triton 与 Naive 实现的最终输出，计算绝对误差和相对误差，验证算法正确性。

## Forward Kernel 实现（PyTorch 参考版本）

`src/naive_stable_topk.py` 是一个朴素实现的 PyTorch kernel，旨在模拟 Triton kernel 的精确行为，特别是 Top-K 选择的稳定性。

它由以下组件构成：

1. **build_tree**: 根据传入的 KV State，通过 Mean Pooling 逐层向上构建 htree 的树状结构。

2. **online_softmax_merge**: 在线 Softmax 合并函数，用于将新的注意力分数和 Value 合并到已有的累积状态中。
   - 动态维护 `max_score` 以防止指数溢出。
   - 支持流式累积。

3. **compute_select_and_merge**: 核心逻辑单元，整合了以下步骤：
   - **候选节点确定**: 根据树结构和 Query 位置确定候选节点。
   - **RoPE 位置编码**: 对 Query 和 Key 应用旋转位置编码。
   - **Stable Top-K 选择**: 使用 Bit-Packing 技术（将索引编码到分数的低位）并进行排序，确保在分数相同时选择索引较小的节点，保证确定性。
   - **Online Softmax**: 将选中节点（或非选中节点，视层级而定）的贡献累积到当前状态。

4. **forward_kernel**: 主入口，串联整个流程。

## Triton Kernel 实现

`src/parallel.py` 实现了 htree 的高性能 Triton kernel 版本。

**主要 Kernel**：
1. **`htree_build_kernel_v2`** - 树构建：通过 Mean Pooling 逐层生成父节点。
2. **每层三个子 Kernel**（从顶层到底层逐层调用）：
   - `htree_compute_scores_and_select_kernel` - 候选遍历 + RoPE + Stable Top-K 选择（Bitonic Sort）。
   - `htree_accumulate_non_topk_kernel` - 流式加载 Value 并累积非 Top-K 节点的贡献。
   - `htree_merge_to_global_kernel_v2` - 将当前层的局部结果合并到全局累积状态。
3. **`htree_final_normalize_kernel_v2`** - 最终归一化。

**关键特性**：
- **Stable Top-K**: 使用 Bit-Packing 和 Bitonic Sort 保证 Top-K 选择的稳定性，消除不确定性。
- **在线 Softmax**: 边计算边累积，显著降低内存占用。
- **分块加载与计算**: 优化内存访问模式，提高带宽利用率。

详细设计请参考 `doc/parallel_design.md`。
