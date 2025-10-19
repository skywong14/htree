## Forward Kernel 实现

naive.py 是一个朴素实现的 PyTorch kernel。

它由以下组件构成：

1. **build_tree**: 根据传入的 final state（最底层所有 token 的 Key Value Query State），构建出 htree 数据结构的所有上层节点。关于构建"代表"token 的压缩算法，我们目前只支持并选择 mean pooling，后续将支持 window_attention。在这个组件中，根据需要创建 L-1 个 tensor（每个非叶子层用一个 tensor 存储）。这可以看作是初始化步骤。

2. **online_softmax_merge**: 在线 Softmax 合并函数，用于将新的注意力分数和 Value 合并到已有的累积状态中。实现流程：
   - **维护累积状态**：max_score（当前最大分数）、sum_exp（指数和）、weighted_output（加权输出）
   - **动态更新最大值**：当新分数中出现更大值时，重新缩放已有的 sum_exp 和 weighted_output
   - **累加新贡献**：计算新分数的指数值并累加到 sum_exp，计算新的加权 Value 并累加到 weighted_output
   - **数值稳定性**：通过维护最大值并动态缩放，避免指数计算溢出

3. **compute_select_and_merge**: 该函数整合了候选节点计算、Top-K 选择和在线 Softmax 合并三个步骤。实现流程：
   - **确定候选节点范围**：
     - 顶层：考虑整层所有左边界 ≤ m 的节点。包含位置 m 的节点因其左边界 ≤ m，会被保留在候选集中
     - 非顶层：考虑上一层选中的节点的所有子节点，对于上一层选中的最右端父节点在该层中的儿子，丢弃左边界 > m 的节点。包含位置 m 的节点必然被保留
   - **计算注意力分数**：
     - 对所有候选节点按索引排序（等价于按左边界排序），依次赋予 RoPE 位置编码 0, 1, 2, ..., (num_candidates-1)
     - 对当前 Query Vector q 应用 RoPE 位置编码 num_candidates
     - 计算 q 关于所有候选节点的注意力分数
   - **选择 Top-K**（仅非最底层执行）：
     - 给包含位置 m 的最右侧节点加极大分数加成（如 +1e9），确保它一定被 Top-K 选中
     - 选择注意力分数最高的 min(512, num_candidates) 个节点用于向下拓展
   - **确定参与在线 Softmax 的节点**：
     - 底层：所有候选节点参与计算
     - 非底层：候选节点中未被选中的节点参与计算（候选节点 - Top-K 节点）
   - **在线 Softmax 合并**：
     - 调用 online_softmax_merge，将当前层参与计算的节点的分数和 Value 合并到累积状态中
   - **返回值**：
     - 更新后的累积状态（max_score, sum_exp, weighted_output）
     - 选中的 Top-K 节点的索引（仅非最底层，底层返回 None），用于确定下一层候选集
   
   **边界情况处理**：
   - 当 query 位置 m 较小时，候选节点数量可能少于 512 或 8192。实现时需要动态处理：
     - 顶层候选节点数量可能 < 512（甚至可能只有几个节点）
     - 非顶层候选节点数量 = 上一层选中节点数 × 16（减去左边界 > m 的节点），可能远小于 8192
     - Top-K 选择时，K 应为 min(512, 实际候选节点数)
     - 所有数组和循环应使用实际的节点数量，而非硬编码的 512 或 8192

4. **forward_kernel**: 主 kernel，完整流程：
   - 调用 build_tree 构建树结构
   - 对每个 query 位置，初始化在线 Softmax 累积状态（max_score, sum_exp, weighted_output）
   - 从顶层到底层，逐层调用 compute_select_and_merge，边计算边合并注意力结果
   - 最终对累积的 weighted_output 进行归一化（除以 sum_exp），得到最终输出

**优化要点**：
- 通过在线 Softmax 算法，避免了缓存所有层的候选节点信息，显著降低内存占用
- 将逐层计算与最终 Softmax 合并，减少了两阶段处理的开销，提高了运行效率
- 保持了数值稳定性，通过动态维护最大值和缩放机制避免指数溢出

它完成了上述完整的 Forward 过程。