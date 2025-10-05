## Forward Kernel 实现

naive.py 是一个朴素实现的 PyTorch kernel。

它由以下组件构成：

1. **build_tree**: 根据传入的 final state（最底层所有 token 的 Key Value Query State），构建出 htree 数据结构的所有上层节点。关于构建"代表"token 的压缩算法，我们目前只支持并选择 mean pooling，后续将支持 window_attention。在这个组件中，根据需要创建 L-1 个 tensor（每个非叶子层用一个 tensor 存储）。这可以看作是初始化步骤。

2. **compute_and_select**: 从参数传入整个分层结构与当前需要处理的层级索引，返回该层选中的 512 个节点的索引和信息。具体流程：
   - 确定候选节点范围（顶层为整层，非顶层为上一层选中节点的子节点）
   - 识别包含位置 m-1 的最右侧父节点，作为必选节点
   - 从竞争节点中 mask 掉最右侧父节点和包含未来信息的节点
   - 按位置顺序（0, 1, 2, ...）给竞争节点和 Query State 加上 RoPE 编码
   - 计算注意力分数，选出 Top 511
   - 将 Top 511 与最右侧父节点合并，得到 512 个节点
   - 缓存选中节点的信息

3. **final_compute**: 从参数传入每一层选中的节点集合，执行最终注意力计算。实现分为三个阶段：
   - **预处理阶段**：利用树的确定性结构快速过滤掉子节点被选中的父节点，按原始序列位置排序，分配 RoPE 位置索引（0, 1, 2, ...）
   - **并行计算阶段**：各层独立计算在线注意力统计量（max_score, weighted_sum, exp_sum），避免跨层张量拼接
   - **整合阶段**：使用 logsumexp 技巧合并各层的统计量，保证数值稳定性，返回最终输出
   
   这种实现在数学上等价于简单地拼接序列 a 后计算注意力，但更高效

4. **forward_kernel**: 主 kernel，完整流程：
   - 调用 build_tree 构建树结构
   - 从顶层到底层，逐层调用 compute_and_select
   - 调用 final_compute 得到最终输出

它完成了上述完整的 Forward 过程。