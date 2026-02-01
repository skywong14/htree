# IO-Only Version Design Document

## 目标

创建 `src/parallel_io_only.py`，用于测量 htree kernel 的 **IO 带宽上界**。

通过简化所有计算逻辑但保留完整的内存访问模式，我们可以：
1. 测量纯 IO 操作的时间开销
2. 计算出计算逻辑的额外开销
3. 判断 kernel 是 IO-bound 还是 compute-bound

## 设计原则

### 保留的部分（完整 IO）
- ✅ 所有 `tl.load()` 操作
- ✅ 所有 `tl.store()` 操作
- ✅ 所有 buffer 分配和数据传输
- ✅ 完整的数据访问模式（非连续访问）

### 简化的部分（计算逻辑）
- ❌ RoPE 计算：移除复杂的 sin/cos 旋转
- ❌ Attention Score：用伪随机数 + 简单特征替代 dot-product
- ❌ TopK 选择：用伪随机哈希替代 bitonic sort
- ❌ 复杂的 softmax 计算：保留框架但简化数值

## 关键修改

### 1. Score 计算简化 (`htree_compute_scores_kernel_io_only`)

**原版本：**
```python
# 1. 应用 RoPE 到 Query 和 Key
k_rope_1, k_rope_2 = load_k_with_rope_v2(...)
q_rope_1 = (q1 * cos_q - q2 * sin_q) * scale
q_rope_2 = (q1 * sin_q + q2 * cos_q) * scale

# 2. 计算 dot-product attention scores
scores = tl.sum(q_rope_1[:, None, :] * k_rope_1[None, :, :], axis=2) + \
         tl.sum(q_rope_2[:, None, :] * k_rope_2[None, :, :], axis=2)
```

**IO-Only 版本：**
```python
# 1. 简化加载（移除 RoPE）
k1, k2 = load_k_simple(...)  # 仍然加载两部分
q_scaled_1 = q1 * scale
q_scaled_2 = q2 * scale

# 2. 伪随机 scores（保持访问模式不规则）
hash_vals = (child_positions * 1103515245 + 12345 + i_h_kv * 7 + i_t * 13) & 0x7FFFFFFF
pseudo_rand = hash_vals.to(tl.float32) / 2147483647.0

# 3. 叠加简单的 K 特征（保持 K 的 IO）
k_feature = tl.sum(k1, axis=1) + tl.sum(k2, axis=1)
k_feature_scaled = k_feature * 0.01

# 4. 组合
base_scores = pseudo_rand * 10.0 - 5.0 + k_feature_scaled
```

**关键点：**
- ✅ 仍然加载 Q 和 K 的所有数据
- ✅ 访问模式非连续（通过 parent_indices 间接访问）
- ❌ 避免昂贵的 dot-product 计算
- ⚠️ 使用哈希保证分数不完全规则（避免缓存友好）

### 2. TopK 选择简化 (`htree_select_topk_simple_kernel`)

**原版本：**
```python
# 1. NSA-style importance 聚合
lse = torch.logsumexp(scores, dim=1)
p = torch.exp(scores - lse[:, None])
imp = p.sum(dim=0)

# 2. Bit-packing 编码
encoded = bitpack_encode(imp, buffer_indices, log_n=13)

# 3. Bitonic sort
sorted_encoded = sort_single(encoded, N_DIMS, ...)

# 4. 解码
topk_positions = bitpack_decode(sorted_encoded, ...)
```

**IO-Only 版本：**
```python
# 1. 伪随机选择（基于哈希）
seed_base = i_t * 7 + i_h_kv * 13 + i_b * 31
hash_vals = ((o_k + seed_base) * 1103515245 + 12345) & 0x7FFFFFFF
pseudo_positions = (hash_vals % n_cand).to(tl.int32)

# 2. 强制选中 rightmost（保持因果性）
is_last_k = (o_k == (actual_k - 1))
selected_pos = tl.where(is_last_k, rightmost_pos, pseudo_positions)

# 3. 直接输出（无需排序）
tl.store(topk_positions + base_out + o_k, selected_pos)
```

**关键点：**
- ✅ 仍然读取 scores buffer（保持 IO）
- ✅ 选择结果不连续（伪随机分布）
- ❌ 避免 bitonic sort 的 O(n log²n) 比较
- ✅ 保持 rightmost 强制选中（算法正确性）

### 3. Next Parents 计算简化 (`htree_compute_next_parents_simple_kernel`)

**原版本：**
```python
# 1. 从 buffer positions 映射回 node indices
batch_id = pos_i64 // BC
within_batch = pos_i64 - batch_id * BC
parent_slot = within_batch // COMPRESSION_RATE
child_slot = within_batch - parent_slot * COMPRESSION_RATE
node_idx = parent_idx * COMPRESSION_RATE + child_slot

# 2. Bitonic sort 升序排序
sorted_indices = sort_single(node_indices, ...)
```

**IO-Only 版本：**
```python
# 直接读写，不做映射和排序
pos = tl.load(topk_positions + base + o_topk)
tl.store(next_selected_parents + base + o_topk, pos)
```

**关键点：**
- ✅ 保持读写操作
- ❌ 移除复杂的索引映射
- ❌ 移除排序

### 4. 其他 Kernels 保持不变

以下 kernels 保持完整实现：
- ✅ `htree_build_kernel_v2`（树构建，主要是 IO）
- ✅ `htree_accumulate_non_topk_kernel_io_only`（V 的加载和累积）
- ✅ `htree_merge_to_global_kernel_v2`（全局状态合并）
- ✅ `htree_final_normalize_kernel_v2`（最终归一化）

这些 kernels 本身就以 IO 为主，计算逻辑相对简单。

## 访问模式分析

### 为什么不能用连续访问？

如果 TopK 选择是连续的（如 `[0, 1, 2, ..., k-1]`），会导致：
1. **缓存友好**：连续访问会充分利用 L1/L2 cache
2. **预取友好**：硬件预取器会自动加载后续数据
3. **不能反映真实性能**：真实的 TopK 是分散的

### 伪随机策略

使用 **线性同余生成器 (LCG)** 生成伪随机数：
```python
hash = (x * 1103515245 + 12345) & 0x7FFFFFFF
```

这是经典的 ANSI C `rand()` 实现，特点：
- ✅ 计算极其简单（一次乘法 + 加法）
- ✅ 分布相对均匀（不会太规则）
- ✅ 确定性（相同输入产生相同输出，便于调试）
- ✅ 避免连续访问模式

加入多个种子（`i_t`, `i_h_kv`, `i_b`）确保不同位置的访问模式不同。

## 使用方法

### 运行速度测试

```bash
python speed_test_io_only.py \
    --batch 1 \
    --seq-len 120000 \
    --hq 16 \
    --h-kv 1 \
    --k-dim 16 \
    --v-dim 16 \
    --compression-rate 16 \
    --max-top-nodes 8192 \
    --top-k 512 \
    --warmup 3 \
    --iters 10 \
    --dtype float32
```

### 输出解读

```
Results (avg ms / iter):
  Full Computation: 1234.56 ms
  IO-Only        : 987.65 ms

Analysis:
  Speedup (Full/IO): 1.25x
  Computation overhead: 20.0% of total time
  IO time (approx): 987.65 ms (80.0% of total)
```

**含义：**
- **IO-Only 时间**：内存带宽的上界（假设计算开销为 0）
- **Computation overhead**：计算逻辑占总时间的百分比
- **Speedup**：移除计算后的加速比

**判断标准：**
- `overhead < 20%`：**IO-bound**，优化计算意义不大
- `20% ≤ overhead ≤ 50%`：**平衡型**，IO 和计算都有优化空间
- `overhead > 50%`：**Compute-bound**，应优化计算逻辑

## 局限性

1. **不是完美的 IO 测试**
   - 伪随机哈希仍有少量计算
   - 简化的 K 特征计算（sum）仍需访问数据

2. **访问模式不完全相同**
   - 伪随机选择的分布可能与真实 TopK 不同
   - 真实 TopK 可能有聚集性（clustered）

3. **不能用于正确性验证**
   - 输出结果无意义（scores 是随机的）
   - 仅用于性能分析

## 对比其他方法

### 方案 A：完全连续访问
```python
# 直接选前 k 个
selected = torch.arange(k)
```
❌ **问题**：访问模式太友好，无法反映真实 cache miss

### 方案 B：纯随机数
```python
# 每次调用 random()
selected = torch.randint(0, n, (k,))
```
❌ **问题**：随机数生成本身很慢，引入额外开销

### 方案 C：预先生成随机索引（我们的方法）
```python
# 用哈希函数生成确定性伪随机数
hash = (x * A + B) & MASK
```
✅ **优点**：
- 计算极快（单次乘加）
- 确定性（可重复）
- 分布均匀（避免聚集）

## 扩展：IO 带宽分析

假设：
- `T = 120000`, `H = 16`, `K = V = 16`
- `dtype = float32` (4 bytes)

**每层的 IO 量估算：**

1. **Load Q**: `B * T * H * K * 4` = `1 * 120000 * 16 * 16 * 4` ≈ 122.88 MB
2. **Load K**: `B * N_layer * H_kv * K * 4` ≈ 122.88 MB (取决于层)
3. **Load V**: `B * N_layer * H_kv * V * 4` ≈ 122.88 MB
4. **Store scores**: `B * T * H_kv * MAX_CAND * NUM_GROUPS * 4` ≈ 500 MB
5. **Store output**: `B * T * H * V * 4` ≈ 122.88 MB

**总 IO 量（粗估）：** ~1 GB per layer

如果有 4 层，总 IO ≈ 4 GB。

**理论带宽（A100）：** ~2000 GB/s (HBM2e)

**理论最快时间：** 4 GB / 2000 GB/s = **2 ms**

如果实际测到 987 ms，说明：
- 有效带宽 ≈ 4 GB / 0.987 s ≈ 4.05 GB/s
- 带宽利用率 ≈ 4.05 / 2000 ≈ **0.2%**

这说明还有很大优化空间（访问模式、block size、cache 等）。

## 总结

`parallel_io_only.py` 通过以下策略实现 IO 上界测试：

1. ✅ 保留完整的 `tl.load/store` 操作
2. ✅ 使用伪随机哈希保持非连续访问
3. ❌ 简化所有昂贵计算（RoPE, dot-product, bitonic sort）
4. ⚠️ 输出无意义，仅用于性能分析

与 `speed_test_io_only.py` 配合使用，可以：
- 测量 IO 带宽上界
- 量化计算开销
- 判断优化方向（IO vs Compute）
