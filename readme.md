# htree（核心思想）

htree 是一种“分层 KV 压缩 + 稀疏候选遍历”的因果注意力近似。

目标：对每个 query 位置 $t$，只访问有限数量的 KV 候选，同时用数值稳定的方式完成 softmax 归一化。

## 核心流程

1. **分层 KV 树（mean pooling）**
   - 以 `compression_rate` 为分支因子，把底层 KV 按固定窗口做均值池化，上采样出更高层的 KV 表示。
   - 得到从底层到顶层的一组 `(layers_k, layers_v)`，顶层节点数受 `max_top_nodes` 限制。

2. **自顶向下逐层展开（候选生成）**
   - 从顶层开始，对每层选中的 parent 节点展开出 `compression_rate` 个 child 作为下一层候选。
   - 候选以“流式”的方式处理：不需要把所有候选的 score 完整落盘。

3. **shared Top-K（跨 GQA groups 的重要度）**
   - 对同一 KV head 下的多个 query group（GQA），将候选的重要度汇总为一个 shared 指标，用它做 streaming Top-K。
   - Top-K 过程中被淘汰的候选会立刻进入累积并丢弃，避免保存全量 score / 额外重读 V。

4. **online-softmax（稳定累积）**
   - 以 running-max 的方式维护 `(max, sum, output)`，在遍历候选时稳定地累积 $p\cdot V$。
   - 各层的累积结果再合并到全局状态，最后做一次 `output/sum` 得到最终输出。

实现与 benchmark 入口：
- [src/parallel.py](src/parallel.py)
- [speed_test.py](speed_test.py)
