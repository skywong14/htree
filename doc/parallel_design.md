# htree 算法的 Triton Kernel 设计方案

本文档描述了 htree 算法的 Triton kernel 实现的**架构设计和技术方案**。该方案采用**逐层累积架构**，在保持高并行度的同时，通过在线 Softmax 算法直接更新全局状态，有效解决了跨层状态累积的技术挑战。设计灵感来自 Flash Attention 的在线 Softmax 技术，并充分借鉴了 NSA (Native Sparse Attention) 的成熟实现经验。

> **注意**：本文档仅包含设计方案和原理说明，不包含具体代码实现

## 方案概述

htree 采用 **3-Kernel 逐层累积架构**：
1. **树构建阶段**：`htree_build_kernel` - 逐层构建树结构（mean pooling）
2. **每层 1 个 kernel**：
   - `htree_layer_kernel` - **批次遍历一体化处理**（BC=512 固定）：
     - 同时完成：分数计算、在线 Softmax 累积、Top-K 选择、缓存
     - 反向减法移除 Top-K 贡献
     - 合并到全局状态
3. **最终归一化**：`htree_final_normalize_kernel` - 输出归一化

**核心创新**：批量向量化 + 在线累积 + 反向减法，消除重复计算，显著降低内存访问和计算开销。
---

## 一、设计演进：新旧方案对比

### 1.1 旧方案的局限性

**旧方案的三阶段处理流程**：
1. **阶段 1**：遍历候选节点计算 LSE（排除最右侧节点）
2. **阶段 2**：再次遍历候选节点，计算注意力分数 + Top-K 选择
3. **阶段 3**：第三次遍历非选中节点，在线 Softmax 累积

**主要问题**：
- ❌ **重复计算严重**：每个候选节点的 Key 被加载 2-3 次，分数被计算 2 次
- ❌ **LSE 阶段必要性存疑**：LSE 仅用于归一化重要性分数，但可以用动态 max 替代
- ❌ **内存访问低效**：三次独立遍历导致三次完整的内存扫描
- ❌ **标量循环并行度低**：内层循环使用标量遍历（`for i_batch in range(BC)`），无法向量化

### 1.2 新方案的核心改进

**新方案的两阶段处理流程**：
1. **阶段 1**：分批遍历候选节点（BC=512）
   - 同时完成：分数计算 + 在线 Softmax 累积（所有节点）+ Top-K 选择 + 缓存
2. **阶段 2**：反向减法移除 Top-K 贡献（仅遍历 512 个节点）

**关键优势**：
- ✅ **消除重复计算**：每个候选节点的 Key 只加载 1 次，分数只计算 1 次
- ✅ **取消 LSE 阶段**：使用动态 `cur_max` 直接归一化重要性分数
- ✅ **紧凑父节点表示**：非顶层不展开子节点，显存占用从 `8192*4bytes` 降至 `512*4bytes`
- ✅ **32×16 分块加载**：利用 `tl.make_block_ptr` 批量加载连续子节点，最大化内存带宽
- ✅ **Padding 对齐简化逻辑**：所有数据 pad 到 512 的倍数，消除边界判断和分支
- ✅ **Payload 同步排序**：score/value 跟随 Bitonic Sort 移动，消除查找循环（O(512×512) → O(0)）
- ✅ **批量反向减法**：一次性处理 512 个节点，向量化操作替代标量循环

**性能提升估算**：
- **内存访问减少**：约 ~50-60%（从 3 次遍历降至 1.06 次）
- **计算量减少**：约 ~40%（消除 LSE 计算和重复分数计算）
- **显存占用减少**：约 ~85%（候选节点表示：8192 → 512）
- **查找开销消除**：O(512×512) → O(0)（Payload 同步排序）
- **并行度提升**：批量 block_ptr 加载，更好地利用 GPU 带宽

---

## 二、核心问题与设计挑战

### 1.1 htree 的独特约束

htree 与传统稀疏注意力（如 NSA）的根本差异在于**层间强依赖**：

```
Layer L (顶层)
    ↓ selected_indices_L (Top-512 节点)
Layer L-1
    ↓ selected_indices_{L-1}
Layer L-2
    ↓ ...
Layer 0 (底层 token)
```

**关键约束**：
- ❌ 无法预先确定所有层的候选节点（依赖上层结果）
- ❌ 必须逐层顺序执行（层间串行）
- ✅ 但层内可以高度并行

### 1.2 跨层状态累积设计

在 htree 中，每层需要将计算结果累积到全局在线 Softmax 状态：

```python
# 全局状态（跨所有层累积）
max_score[B, T, H]           # 当前最大分数
sum_exp[B, T, H]             # 指数和
weighted_output[B, T, H, V]  # 加权输出
```

**设计方案**：
- ✅ **采用方案：不在 V 维度并行**
  - 每个 thread block 一次性加载并处理完整的 V 维度
  - 直接更新全局状态，无需额外的归约 kernel
  - 避免跨 block 协同和状态合并的复杂性
  - 代码简洁，易于实现和调试
  
- ❌ **未采用：V 维度并行 + 分层归约**
  - 虽然在 V 很大（≥256）时可能有性能优势
  - 但需要额外的 Reduce kernel 来合并 NV 个局部状态
  - 增加内存占用和实现复杂度
  - 对于常见的 V=64 或 V=128，收益有限

---

## 二、Kernel 架构设计

### 2.1 完整系统流程

**执行模型**：树构建 → 逐层前向 → 最终归一化

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Tree Construction (逐层构建)                        │
│  htree_build_kernel: [B, N_child, H, K/V] → [B, N_parent, H, K/V] │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Layer-by-Layer Forward Pass (for layer in L...0)   │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ htree_layer_kernel: Grid (T, B*H)                     │  │
│  │                                                       │  │
│  │  阶段 1: 确定候选节点范围                              │  │
│  │    - 顶层：[0, rightmost_idx]                         │  │
│  │    - 非顶层：上层 Top-512 的子节点（过滤 >rightmost） │  │
│  │                                                       │  │
│  │  阶段 2: 应用 RoPE 到 Query                           │  │
│  │    - Query 使用位置 num_candidates-1 的 RoPE         │  │
│  │                                                       │  │
│  │  阶段 3: 批次遍历候选节点（BC=512 固定）               │  │
│  │    for each batch in candidates (step=BC):           │  │
│  │      3.1 批量加载 Key + 应用 RoPE [BC, K]             │  │
│  │      3.2 计算注意力分数 [BC]                          │  │
│  │      3.3 在线 Softmax 累积（所有 BC 个节点）          │  │
│  │          更新 cur_max, cur_sum, cur_output           │  │
│  │      3.4 Top-K 选择（非底层）：                        │  │
│  │          - 计算重要性分数（rightmost=1.0）            │  │
│  │          - Bitonic Sort 当前批次                      │  │
│  │          - 与历史 Top-K 合并 → 新 Top-K               │  │
│  │      3.5 同步缓存 Score（向量化匹配）                 │  │
│  │          通过矩阵匹配将当前批次的 score 同步到 Top-K  │  │
│  │                                                       │  │
│  │  阶段 4: 移除 Top-K 贡献（反向在线 Softmax）           │  │
│  │    4.1 对 Top-K 索引排序（升序，传递给下一层）        │  │
│  │    4.2 遍历 Top-K 节点（512个）：                     │  │
│  │        - 重新加载节点的 Value                         │  │
│  │        - 在线 Softmax 累积 Top-K 贡献                │  │
│  │    4.3 反向减法移除贡献：                             │  │
│  │        cur_sum -= topk_sum_scaled                    │  │
│  │        cur_output -= topk_output_scaled              │  │
│  │                                                       │  │
│  │  阶段 5: 合并到全局状态                                │  │
│  │    - 合并 cur_max/sum/output 到 global_max/sum/output│  │
│  │    - 存储 Top-K 节点索引（非底层）                     │  │
│  └──────────────────────────────────────────────────────┘  │
│      
**设计思路**：
1. **输入处理**：从上层传入的选中父节点（512个int定长tensor），值为-1表示无效节点
2. **核心任务**：
   - 任务A：选出注意力分数 Top-K 节点传给下一层
   - 任务B：对 Top-K 以外的节点做注意力计算
3. **批次遍历策略**（参考 NSA 设计）：
   - **批次大小固定为 BC=512**（必须等于 TOP_K）
   - 分批遍历最多 8192 个候选节点
   - **每批同时完成**：分数计算、在线Softmax累积、Top-K选择、缓存
4. **关键优化**：
   - ✅ 每个节点的 Key 只加载一次，分数只计算一次
   - ✅ 在线 Softmax 在第一次遍历中完成（包含所有节点）
   - ✅ Top-K 节点的 Score 通过向量化矩阵匹配同步缓存
   - ✅ 第二阶段重新加载 Value 并通过反向减法移除 Top-K 贡献
5. **最终输出**：Top-K 节点索引（升序排列）+ 更新后的全局状态（仅包含非 Top-K 节点的注意力输出）


                                                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Final Normalization                                 │
│  htree_final_normalize_kernel: Grid (T, B*H)                │
│  output = weighted_output / sum_exp                          │
└─────────────────────────────────────────────────────────────┘
```

**特性**：
- **层间串行**：必须等待上层 Top-K 结果才能处理下层
- **一体化处理**：LSE 计算、Top-K 选择、在线 Softmax 累积在单次遍历中完成
- **两阶段设计**：先遍历并选择 Top-K，再对非选中节点累积输出
- **状态管理**：边遍历边累积，直接更新全局状态（online softmax）

### 2.2 内存布局与优化

**状态张量设计**：

```python
# 全局状态（持久化，跨层复用）
max_score       : [B, T, H]          # 4 bytes/element (FP32)
sum_exp         : [B, T, H]          # 4 bytes (FP32)
weighted_output : [B, T, H, V]       # 4 * V bytes (FP32)

# 层间传递（每层临时，层间复用）
selected_indices : [B, T, H, 512]    # 4 * 512 bytes per (b,t,h) (int32, 被选中传递给下一层的节点)

# Kernel 内部临时存储
batch_scores         : [BC=512]      # 当前批次节点分数
batch_node_indices   : [BC=512]      # 当前批次节点索引
batch_values         : [BC=512, V]   # 当前批次节点 Value
b_i                  : [BC=512]      # Top-K 重要性分数缓冲区
o_i                  : [BC=512]      # Top-K 节点索引缓冲区
b_score              : [BC=512]      # Top-K 注意力分数缓冲区
```

**内存说明**：
- **全局状态持久化**：`max_score`, `sum_exp`, `weighted_output` 跨层复用，避免重复分配
- **选中节点索引**：只存储 Top-512 用于传递给下一层（底层不需要）
- **临时数组优化**：
  - 候选节点信息（索引、分数）在 kernel 内部使用局部数组存储
  - 不需要分配全局内存缓冲区，大幅减少内存占用
  - 利用寄存器和共享内存的高带宽
- **两阶段复用**：
  - Top-K Select 遍历时计算并缓存所有候选节点的分数
  - Online Softmax 直接复用这些分数，无需重新计算

---

## 三、Kernel 功能设计

### 3.1 树构建 Kernel (`htree_build_kernel`)

**功能**：自底向上逐层构建树结构（mean pooling）

**并行策略**：逐层构建，每层单独启动一次 kernel
- 从第 1 层开始（第 0 层是底层 token，无需构建）
- 每层的 Grid: `(N_parent, B * H)`
- 每个 thread block 处理当前层的一个父节点

**输入输出**：
- 输入：
  - 当前层的子节点 K, V `[B, N_child, H, K/V]`（第1层的输入是底层 token `[B, T, H, K/V]`）
  - `layer_idx`：当前正在构建的层索引
- 输出：当前层的父节点 `[B, N_parent, H, K/V]`，其中 `N_parent = ceil(N_child / compression_rate)`

**算法流程**（每个 thread block 处理一个父节点）：
1. 从 block 索引解析出 `(b, h, parent_idx)`：batch、head、父节点索引
2. 确定当前父节点对应的子节点范围：`[parent_idx * 16, (parent_idx + 1) * 16)`
3. 加载所有子节点的 K 和 V（如果子节点数不足16个，需要处理边界）
4. 计算平均值（mean pooling）
5. 存储到当前层的输出位置 `[b, parent_idx, h, :]`

### 3.2 层处理 Kernel (`htree_layer_kernel`)

**功能**：完成候选节点遍历、Top-K 选择、非选中节点的在线 Softmax 累积

**并行策略**：Grid: `(T, B*H)` - 每个 (query_pos, batch, head) 组合对应一个 thread block
- 所有 T 个 query 位置**并行且独立**处理，各自基于不同的候选集选择 Top-K 并累积输出
- 不同位置之间无数据依赖，各 block 独立计算并写入各自的输出切片

**RoPE 处理策略**：
- **预计算与缓存**：在主调用函数中预先计算并缓存 RoPE 的 cos/sin 值
  - 缓存大小：`cache_size = max_top_nodes + 1024`（8192 + 1024，额外空间处理边界情况）
  - 预计算：`positions = torch.arange(cache_size)`，`freqs = outer(positions, inv_freq)`
  - 存储：`cos_cache[cache_size, D//2]`, `sin_cache[cache_size, D//2]`
- **Kernel 内使用**：
  - 对候选节点：按索引顺序（0, 1, 2, ..., num_candidates-1）直接查找 RoPE 值
  - 对 Query：使用位置 `num_candidates - 1` 的 RoPE 值（与包含 query 位置的最右侧节点相同）
  - RoPE 应用：使用标准的旋转变换（与 naive.py 中 `apply_rotary_emb` 相同）
- **每层编码范围不同**：不同层的候选节点数量不同，因此每层使用的 RoPE 位置编码范围是动态的 `[0, num_candidates-1]`

**输入输出**：
- 输入：
  - Query `[B, T, H, K]`
  - 当前层 Key/Value `[B, N_layer, H, K/V]`
  - 上层选中节点索引 `selected_parents[B, T, H, top_k]`（顶层时为 None）
  - 当前层索引 `layer_idx`（用于判断是否为底层）
  - RoPE cache：`cos_cache[cache_size, D//2]`, `sin_cache[cache_size, D//2]`
  - 全局状态：`global_max[B, T, H]`, `global_sum[B, T, H]`, `global_output[B, T, H, V]`
- 输出：
  - `selected_indices[B, T, H, 512]`（int32）：Top-K 节点的全局索引（底层不需要）
  - 更新后的全局状态（直接写回）

**算法流程**（每个 thread block 独立执行以下步骤）：

**阶段 1：确定候选节点范围与 Padding**
- **顶层**：候选节点为 `[0, rightmost_idx]`，pad 到 512 的倍数
- **非顶层**：基于上层 Top-512 父节点，保持父节点列表形式（不展开子节点）
- **Padding 策略**：父节点值 -1 表示无效，对应的子节点为 padding（Key/Value=0, Score=-inf, Importance=0.0）

**阶段 2：应用 RoPE 到 Query**
- Query 使用 RoPE 位置 = `num_candidates - 1`（与最右侧候选节点相同）
- 从预计算的 cache 查表获取 cos/sin 值

**阶段 3：批次遍历候选节点（BC=512 固定）**
- 初始化在线 Softmax 状态和 Top-K 缓冲区
- 分批遍历，每批处理 BC=512 个节点：
  - 通过 32×16 分块加载：32 次循环，每次用 `tl.make_block_ptr` 加载 16 个连续子节点
  - 对加载的 Key 应用 RoPE，计算注意力分数
  - 在线 Softmax 累积（所有节点）
  - Top-K 选择（非底层）：计算重要性分数（rightmost=1.0），Bitonic Sort + 与历史合并
  - Payload 同步排序：score/value 跟随 Bitonic Sort 自动移动

**阶段 4：移除 Top-K 贡献（反向在线 Softmax）**
- 提取最终 Top-K 节点索引
- 直接使用 Bitonic Sort 后的 score 和 value（已对齐，无需查找）
- 批量计算并减去 Top-K 节点的贡献

**阶段 5：合并到全局状态**
- 使用在线 Softmax 将当前层状态合并到全局状态
- 存储 Top-K 节点索引（传递给下一层）

### 4.3 最终归一化 Kernel (`htree_final_normalize_kernel`)

**功能**：归一化输出

**并行策略**：Grid: `(T, B*H)`

**算法**：`output = weighted_output / sum_exp`
