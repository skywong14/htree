# htree Triton Kernel 设计文档

## 整体架构

htree 的 Triton 实现（核心程序 `src/parallel.py`）采用三阶段流水线架构：

```
Phase 1: 树构建              Phase 2: 逐层前向传播              Phase 3: 最终归一化
(htree_build_kernel_v2)      (3个子kernel每层调用)           (htree_final_normalize_kernel_v2)
   
Token Layer                 ┌────────────────────────┐
      ↓                     │ htree_compute_scores_  │
  Mean Pooling              │   and_select_kernel    │         normalized_output
      ↓                     │ htree_accumulate_      │              ↑
Layer 1, 2, ..., L          │   non_topk_kernel      │         output / sum
  (递归构建)                 │ htree_merge_to_global_ │
                            │   kernel_v2            │
                            │  (顶层 → 底层逐层)      │
                            └────────────────────────┘
```

**核心设计理念**：
- **在线 Softmax 算法**：边计算边合并，避免缓存所有候选节点
- **父节点紧凑表示**：512 个父节点索引 × 16 子节点 = 8192 候选节点（固定 buffer）
- **32×16 分块加载**：每批 512 节点通过 32 次加载完成，最大化内存带宽
- **稳定 Top-K (Stable Top-K)**：采用 **Bit-Packing** 编码（将 buffer 索引编码进 score 低位） + **Bitonic Sort**，保证在分数相同时行为确定，便于与 PyTorch 参考实现对齐。
- **三步式处理**：每层调用 3 个 kernel（选择&Top-K、非 Top-K 累积、合并）

---

## Kernel 1: 树构建 (htree_build_kernel_v2)

### 功能
通过 Mean Pooling 从子节点生成父节点的 K 和 V 表示。

### 并行策略
```
Grid: (B × H)
每个 block 处理一条 (batch, head) 的父节点序列，内部循环写入
```

### 计算流程
1. 确定父节点对应的子节点范围（通常 16 个）
2. 累加所有子节点的 K 和 V
3. 除以子节点数量得到平均值
4. 存储父节点的 K 和 V

### 关键优化
- **向量化加载**：直接加载完整的 K/V 维度
- **边界处理**：最后一个父节点可能子节点数 < 16
- **数据局部性**：每个 block 独立处理，无需同步

---

## Kernel 2: 层处理（三个子 Kernel）

每层需要依次调用三个 kernel 完成前向传播：
1. **htree_compute_scores_and_select_kernel**：批次遍历候选 + RoPE + 写入 8192 buffer + 稳定 Top-K（非底层）
2. **htree_accumulate_non_topk_kernel**：按批次流式累积非 Top-K（底层累积全部）
3. **htree_merge_to_global_kernel_v2**：合并到全局状态

### Kernel 2.1: htree_compute_scores_and_select_kernel

#### 功能
批次遍历候选节点，计算分数写入固定 8192 buffer；非底层执行稳定 Top-K，输出 buffer 位置与升序节点索引。

#### 并行策略
```
Grid: (T, B × H)
每个 block 处理一个 (query_pos, batch, head) 组合
```

#### 算法流程

**阶段 1：确定候选节点范围**
- 加载父节点列表（512 个，padding 为 -1）
- 计算有效父节点数量
- 确定批次数：`num_batches = (num_valid_parents + 31) // 32`

**阶段 2：应用 RoPE 到 Query**
- 计算候选节点总数：`num_candidates = (num_valid_parents - 1) × 16 + (rightmost_idx % 16) + 1`
- Query 使用 RoPE 位置 `num_candidates - 1`
- 分别加载 cos/sin 并应用到 Q 的前后半部分

**阶段 3：批次遍历（BC=512 固定）**
- 外层：遍历 `num_batches`（32 父 × 16 子）批次
- 内层：`load_k_with_rope_v2` 加载 16 子节点并应用 RoPE；计算分数填充 `[32,16]`
- Flatten 写入固定 8192 buffer：`all_scores`, `all_node_indices`

**阶段 4：稳定 Top-K（非底层）**
- **Bit-Packing**: 将 buffer 位置编码进 score 低位。
  - 正数：`score_int | (~buffer_idx & mask)` （取反索引，使大分数相同时选小索引）
  - 负数：`score_int | (buffer_idx & mask)`
- **排序**: 对编码后的分数做单数组 **Bitonic Sort**。
- **解码**: 还原 buffer 位置。
- 提取前 512 的 buffer 位置 `topk_positions`，并获得升序的全局节点索引 `selected_indices`。
- 最右节点在排序前被赋值 `1e3`，确保必选。

**底层特殊**：跳过排序，`topk_positions/selected_indices` 置为 -1。

---

### Kernel 2.2: htree_accumulate_non_topk_kernel

#### 功能
按原批次顺序流式加载 V，并基于 `topk_positions` 构建 mask，仅累积非 Top-K；底层累积全部。

#### 并行策略
```
Grid: (T, B × H)
```

#### 算法流程

**步骤 1：加载元数据**
- 读取 `num_candidates`，`topk_positions`（buffer 位置，可能包含 -1）

**步骤 2：按批次加载 V（32 父 × 16 子）**
- 使用 `load_v_v2` 加载每批 512 个候选的 V
- 构建 `topk_mask`（基于 buffer 位置）并与有效性联合得到最终 mask
- 底层不使用 topk_mask

**步骤 3：在线 Softmax 累积**
- 对 masked scores 计算 block_max / block_sum / block_out
- 与累计状态 `(cur_max, cur_sum, cur_output)` 在线合并

**步骤 4：写回**
- 存储 `layer_max`, `layer_sum`, `layer_output`

---

### Kernel 2.3: htree_merge_to_global_kernel_v2

#### 功能
将当前层状态合并到全局状态。

#### 并行策略
```
Grid: (T, B × H)
```

#### 算法流程

**在线 Softmax 合并**：
```
new_max = max(global_max, layer_max)
scale_g = exp(global_max - new_max)
scale_c = exp(layer_max - new_max)

global_sum = global_sum × scale_g + layer_sum × scale_c
global_output = global_output × scale_g + layer_output × scale_c
global_max = new_max
```

**写回全局状态**。

---

## Kernel 3: 最终归一化 (htree_final_normalize_kernel_v2)

### 功能
对累积的输出进行归一化。

### 并行策略
```
Grid: (T, B × H)
每个 block 处理一个 (query_pos, batch, head) 组合
```

### 计算流程
```
output = global_output / global_sum
```

---

## 主调用函数 (htree_forward_v2)

### 四阶段流程

#### **Phase 1: 树构建**
1. 计算树的层数：`while temp_len > max_top_nodes: temp_len = ceil(temp_len / 16)`
2. 逐层调用 `htree_build_kernel_v2` 生成父节点 K/V
3. 存储每层：`layers_k = [k_layer0, k_layer1, ...]`, `layers_v`

#### **Phase 1.5: 预计算 RoPE Cache**
- 计算 `inv_freq`
- 生成 `cos_cache`, `sin_cache`

#### **Phase 2: 初始化全局状态**
- 初始化 `global_max`, `global_sum`, `global_output`

#### **Phase 3: 逐层前向传播**
1. 为顶层生成虚拟父节点列表
2. 从顶层到底层（`for layer_idx in range(num_layers-1, -1, -1)`）：
   - 调用 **Kernel 2.1**：`htree_compute_scores_and_select_kernel`
   - 调用 **Kernel 2.2**：`htree_accumulate_non_topk_kernel`
   - 调用 **Kernel 2.3**：`htree_merge_to_global_kernel_v2`
3. 每层的 `selected_indices` 传递给下一层作为 `prev_selected_parents`

#### **Phase 4: 最终归一化**
- 调用 `htree_final_normalize_kernel_v2`
- 输出：`output`

---

## 性能特性与优化

### 内存优化
- **父节点紧凑表示**：512 个索引 → 8192 个候选节点
- **层间状态复用**：`layer_max/sum/output` 每层复用
- **Top-K 信息临时存储**：`topk_indices/scores` 仅用于层间传递

### 计算优化
- **32×16 分块加载**：`tl.make_block_ptr` 批量加载连续子节点
- **向量化操作**：矩阵匹配同步 score
- **Bitonic Sort**: 在 SRAM 中进行高效排序

### 数值稳定性
- **在线 Softmax**：动态维护 max 值；合并阶段做安全缩放
- **Stable Top-K**: 使用 Bit-Packing 保证排序稳定性，便于调试和对比
