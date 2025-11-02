# htree Triton Kernel 设计文档

## 整体架构

htree 的 Triton 实现采用三阶段流水线架构：

```
Phase 1: 树构建              Phase 2: 逐层前向传播              Phase 3: 最终归一化
(htree_build_kernel)         (3个子kernel每层调用)           (htree_final_normalize_kernel)
   
Token Layer                 ┌────────────────────────┐
      ↓                     │ htree_compute_and_     │
  Mean Pooling              │   select_kernel        │         normalized_output
      ↓                     │ htree_subtract_topk_   │              ↑
Layer 1, 2, ..., L          │   kernel               │         output / sum
  (递归构建)                 │ htree_merge_to_global_ │
                            │   kernel               │
                            │  (顶层 → 底层逐层)      │
                            └────────────────────────┘
```

**核心设计理念**：
- **在线 Softmax 算法**：边计算边合并，避免缓存所有候选节点
- **父节点紧凑表示**：512 个父节点索引 × 16 子节点 = 8192 候选节点
- **32×16 分块加载**：每批 512 节点通过 32 次加载完成，最大化内存带宽
- **Bitonic Sort Top-K**：高效选择最重要的 512 个节点
- **三步式处理**：每层调用 3 个 kernel（选择、减法、合并）

---

## Kernel 1: 树构建 (htree_build_kernel)

### 功能
通过 Mean Pooling 从子节点生成父节点的 K 和 V 表示。

### 并行策略
```
Grid: (N_parent, B × H)
每个 block 处理一个父节点
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
1. **htree_compute_and_select_kernel**：候选遍历 + 在线 Softmax + Top-K 选择
2. **htree_subtract_topk_kernel**：反向减法移除 Top-K 贡献（非底层）
3. **htree_merge_to_global_kernel**：合并到全局状态

### Kernel 2.1: htree_compute_and_select_kernel

#### 功能
遍历候选节点，计算注意力分数，在线 Softmax 累积，选择 Top-K 节点。

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
- **初始化**：`cur_max = -inf`, `cur_sum = 0`, `cur_output = 0`，Top-K 缓冲区 `b_i`, `o_i`, `b_score`
- **外层循环**：遍历 `num_batches` 批次（每批 32 个父节点）
- **内层循环**：`for i_p in tl.static_range(32)` 逐个处理父节点
  - 使用辅助函数 `load_k_with_rope` 加载 16 个子节点的 K 并应用 RoPE
  - 使用辅助函数 `load_v` 加载 16 个子节点的 V
  - 计算注意力分数：`scores_16 = sum(q_rope × k_rope)`
  - 将分数和 Value 放入 2D 数组 `[32, 16]`
- **Flatten 到批次缓冲区**：`batch_scores[512]`, `batch_node_indices[512]`, `batch_values[512, V]`
- **在线 Softmax 累积**：
  ```
  new_max = max(cur_max, max(batch_scores))
  cur_sum = cur_sum × exp(cur_max - new_max) + sum(exp(batch_scores - new_max))
  cur_output = cur_output × exp(cur_max - new_max) + sum(exp(batch_scores - new_max) × batch_values)
  ```
- **Top-K 选择（非底层）**：
  - 计算重要性分数：最右侧节点 `importance = 1.0`，其他节点 `exp(score - cur_max)`
  - 使用 Bitonic Sort 对当前批次排序
  - 与历史 Top-K 合并（交错合并）
  - 向量化匹配矩阵同步更新 `b_score`

**阶段 4：存储结果**
- 存储 `layer_max`, `layer_sum`, `layer_output`
- 存储 `topk_indices`, `topk_scores`（未排序）

---

### Kernel 2.2: htree_subtract_topk_kernel

#### 功能
移除 Top-K 节点的贡献（反向减法），仅非底层调用。

#### 并行策略
```
Grid: (T, B × H)
```

#### 算法流程

**步骤 1：对 Top-K 索引排序**
- 加载 `topk_indices`（未排序）
- 将 -1 替换为 `MAX_IDX = 2147483647`
- 使用 Bitonic Sort 按升序排序
- 恢复 -1 并存储到 `selected_indices`（传递给下一层）

**步骤 2：汇总 Top-K 贡献**
- 计算 `topk_max = max(topk_scores)`（跳过无效节点）
- 逐个加载 Top-K 节点的 Value（循环 512 次）
- 累积到 `topk_values[512, V]`
- 计算加权和：`topk_output = sum(exp(topk_scores - topk_max) × topk_values)`

**步骤 3：反向减法**
```
topk_scale = exp(topk_max - cur_max)
cur_sum = cur_sum - topk_sum × topk_scale
cur_output = cur_output - topk_output × topk_scale
```

**步骤 4：写回**
- 更新 `layer_max`, `layer_sum`, `layer_output`

---

### Kernel 2.3: htree_merge_to_global_kernel

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

### 关键优化

**内存优化**：
- 父节点紧凑表示：512 个索引 → 8192 个候选节点
- 层间状态复用：`layer_max/sum/output` 每层复用
- Top-K 信息临时存储：`topk_indices/scores` 仅用于层间传递

**计算优化**：
- 32×16 分块加载：`tl.make_block_ptr` 批量加载连续子节点
- 向量化操作：矩阵匹配同步 score
- 辅助函数封装：`load_k_with_rope`, `load_v` 简化代码

**数值稳定性**：
- 在线 Softmax：动态维护 max 值
- 反向减法：缩放后再减去

---

## Kernel 3: 最终归一化 (htree_final_normalize_kernel)

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

### 边界处理
- 使用 `mask=o_v < V` 确保 V 维度边界安全

---

## 主调用函数 (htree_forward)

### 四阶段流程

#### **Phase 1: 树构建**
1. 计算树的层数：`while temp_len > max_top_nodes: temp_len = ceil(temp_len / 16)`
2. 逐层调用 `htree_build_kernel` 生成父节点 K/V
3. 存储每层：`layers_k = [k_layer0, k_layer1, ...]`, `layers_v`

#### **Phase 1.5: 预计算 RoPE Cache**
- 计算 `inv_freq = 1.0 / (rope_base ** (arange(0, K, 2) / K))`
- 生成 `positions = arange(cache_size)`, `freqs = outer(positions, inv_freq)`
- 存储：`cos_cache = freqs.cos()`, `sin_cache = freqs.sin()`
- `cache_size = 8192 + 1024`

#### **Phase 2: 初始化全局状态**
- `global_max[B, T, H]`: 初始化为 `-inf`
- `global_sum[B, T, H]`: 初始化为 `0.0`
- `global_output[B, T, H, V]`: 初始化为 `0.0`

#### **Phase 3: 逐层前向传播**
1. 为顶层生成虚拟父节点列表（每个 query 位置 t 独立计算）：
   ```
   rightmost_idx[t] = t // (16 ** (num_layers - 1))
   num_virtual_parents[t] = rightmost_idx[t] // 16 + 1
   prev_selected_parents[t] = [0, 1, ..., num_virtual_parents[t]-1, -1, ...]
   ```
2. 从顶层到底层（`for layer_idx in range(num_layers-1, -1, -1)`）：
   - 调用 **Kernel 2.1**：`htree_compute_and_select_kernel`
   - 调用 **Kernel 2.2**（非底层）：`htree_subtract_topk_kernel`
   - 调用 **Kernel 2.3**：`htree_merge_to_global_kernel`
3. 每层的 `selected_indices` 传递给下一层作为 `prev_selected_parents`

#### **Phase 4: 最终归一化**
- 调用 `htree_final_normalize_kernel`
- 输出：`output[B, T, H, V] = global_output / global_sum`

---

## 内存布局

### 输入/输出张量
```
Query:         [B, T, H, K]
Key:           [B, T, H, K]  (层 0)
Value:         [B, T, H, V]  (层 0)

层 K/V:        
  layers_k:    [k_layer0[B,T,H,K], k_layer1[B,N1,H,K], ...]
  layers_v:    [v_layer0[B,T,H,V], v_layer1[B,N1,H,V], ...]

RoPE Cache:    
  cos_cache:   [cache_size, K//2]
  sin_cache:   [cache_size, K//2]

全局状态（FP32）:
  global_max:    [B, T, H]
  global_sum:    [B, T, H]
  global_output: [B, T, H, V]

层临时状态（每层复用）:
  layer_max:     [B, T, H]
  layer_sum:     [B, T, H]
  layer_output:  [B, T, H, V]

Top-K 信息（层间传递）:
  topk_indices:     [B, T, H, 512]  (int32，未排序)
  topk_scores:      [B, T, H, 512]  (FP32，未排序)
  selected_indices: [B, T, H, 512]  (int32，已排序，传给下一层)

最终输出:      [B, T, H, V]  (dtype)
```

### 数据类型
- **输入/输出**：`dtype` (通常 `float16` 或 `bfloat16`)
- **全局状态/层状态**：`float32`（保证数值精度）
- **索引**：`int32`

---

## 关键约束与假设

### 硬编码常量
- `COMPRESSION_RATE = 16`: 每个父节点 16 个子节点
- `TOP_K = 512`: 每层选择 512 个节点
- `BC = 512`: 批次大小（必须等于 TOP_K）
- `PARENTS_PER_BATCH = 32`: 每批处理 32 个父节点
- `max_top_nodes = 8192`: 顶层最大节点数
- `n_dims = 9`: Bitonic Sort 维度（2^9 = 512）

### 关键假设
1. **父节点升序排列**：`selected_indices` 经过排序后传递给下一层
2. **最右侧节点必选**：使用 `importance = 1.0` 确保被 Top-K 选中
3. **BC = TOP_K**：批次大小必须等于 Top-K 大小（简化合并逻辑）
4. **TOP_K 是 2 的幂**：Bitonic Sort 要求（当前 512 = 2^9）
5. **数值稳定性**：反向减法可能引入误差，但算法设计可接受

### Triton 编程规范
- ✅ 使用 `tl.static_range` 替代 `range` 以利用编译时展开
- ✅ 使用 `tl.where` 替代条件分支
- ✅ 使用 `tl.make_block_ptr` 批量加载
- ✅ `boundary_check=(0, 1)` 确保边界安全
- ✅ 类型转换：`.to(dtype.element_ty)` 或 `.to(tl.float32)`

---

## 性能特性

### 计算复杂度（单层）
- **候选节点数**：最多 ~8192（非顶层），顶层取决于 query 位置
- **加载次数**：
  - Kernel 2.1：每个候选节点的 K/V 加载 1 次（32×16 分块加载）
  - Kernel 2.2：Top-K 节点的 V 加载 1 次（512 次独立加载）
- **Top-K 选择复杂度**：O(num_batches × BC × log²(BC))
  - 每批 Bitonic Sort：O(512 × 81)
  - 批间合并：O(512 × 9)
  - 总批次数：`num_batches = (num_valid_parents + 31) // 32`
- **在线 Softmax**：O(BC) = O(512) 每批

### 内存占用（单 block）
**Kernel 2.1 寄存器/共享内存**：
- 批次缓冲区：`batch_scores[512]`, `batch_node_indices[512]`, `batch_values[512, V]`
- Top-K 缓冲区：`b_i[512]`, `o_i[512]`, `b_score[512]`（非底层）
- 估算：~4KB（索引/分数）+ 512×V×4 bytes（Value）

**Kernel 2.2 寄存器/共享内存**：
- Top-K Value 缓冲区：`topk_values[512, V]`
- 估算：512×V×4 bytes

### 并行度
- **Grid 大小**：T × B × H（通常数千到数万个 block）
- **每个 block 独立**：无 block 间同步
- **内存访问模式**：
  - Kernel 2.1：合并访问（`tl.make_block_ptr` 加载 16×K/V）
  - Kernel 2.2：随机访问（逐个加载 Top-K 节点的 V）

---

## 扩展性与限制

### 可配置参数
- `compression_rate`: 压缩率（默认 16，硬编码）
- `top_k_per_layer`: 每层 Top-K 大小（默认 512）
- `max_top_nodes`: 顶层节点数阈值（默认 8192）
- `rope_base`: RoPE 基础频率（默认 10000.0）
- `scale`: 注意力缩放因子（默认 K^-0.5）

### 限制
- **K 维度**：必须是偶数（RoPE 前后半分）
- **TOP_K**：必须是 2 的幂（Bitonic Sort 要求），当前 512 = 2^9
- **COMPRESSION_RATE**：固定为 16（硬编码在 kernel 中）
- **BC = TOP_K**：批次大小必须等于 Top-K 大小
- **序列长度 T**：理论无上限（逐层压缩到 ≤8192）

### 潜在优化方向
1. **Kernel 2.2 优化**：512 次独立 V 加载可改为批量加载（需改变数据布局）
2. **Autotune**：根据 K/V 维度自动选择 num_warps
3. **V 维度并行**：对于超大 V（≥256）可考虑分块处理
4. **动态 TOP_K**：支持非 2 的幂（需修改 Bitonic Sort）
