
### 目标

本文档整理 `htree` **forward** 的数学定义与计算流程，抽象出与实现无关的算法语义，并标注 Triton 实现（`src/parallel.py`）中的关键张量/索引如何对应这些语义，方便后续实现 backward。

实现参考：
- **Naive PyTorch**: `src/naive_gqa.py`（更接近“定义”）
- **高效 Triton**: `src/parallel.py`（性能导向，张量布局/分块更复杂）

---

### 记号与张量形状

- **批大小** \(B\)，序列长度 \(T\)
- **Query heads** \(H\)，**KV heads** \(H_{kv}\)，要求 \(H \bmod H_{kv} = 0\)
- **GQA group size** \(G = H / H_{kv}\)
- **Key 维度** \(K\)，**Value 维度** \(V\)
- **压缩率** \(r\)（代码中 `COMPRESSION_RATE`，默认 16），要求为 2 的幂

输入输出：
- \(Q \in \mathbb{R}^{B \times T \times H \times K}\)
- \(K^{(0)} \in \mathbb{R}^{B \times T \times H_{kv} \times K}\)
- \(V^{(0)} \in \mathbb{R}^{B \times T \times H_{kv} \times V}\)
- 输出 \(O \in \mathbb{R}^{B \times T \times H \times V}\)

head 映射（GQA）：
- 对于 KV head \(h_{kv}\in[0,H_{kv})\) 与组内 index \(g\in[0,G)\)，对应 query head
  \[
  h = h_{kv}\cdot G + g
  \]

缩放：
- \(\alpha = K^{-1/2}\)（代码 `scale`）

---

### 层级树（Tree）构建：mean pooling

定义第 \(\ell\) 层的节点数 \(N_\ell\)：
- \(N_0 = T\)
- \(N_{\ell+1} = \lceil N_\ell / r\rceil\)
- 直到最顶层 \(N_{L-1} \le \texttt{max\_top\_nodes}\)

每层节点代表一段连续 token 区间。对理论描述，令第 \(\ell\) 层 node \(i\) 覆盖范围：
\[
\text{range}(\ell,i) = \left[i\cdot r^\ell,\;\min\big((i+1)\cdot r^\ell,\;T\big)\right)
\]

节点的 K/V 由其子节点做 **mean pooling**：
- 若 \(\ell>0\)，第 \(\ell\) 层节点 \(i\) 的子节点为 \(\ell-1\) 层的 \(j \in [ir,(i+1)r)\)（越界的子节点视为不存在）
- 对每个 \((b,h_{kv})\)：
  \[
  K^{(\ell)}_{b,i,h_{kv}} = \frac{1}{c}\sum_{j\in\text{children}(\ell,i)} K^{(\ell-1)}_{b,j,h_{kv}},\quad
  V^{(\ell)}_{b,i,h_{kv}} = \frac{1}{c}\sum_{j\in\text{children}(\ell,i)} V^{(\ell-1)}_{b,j,h_{kv}}
  \]
  其中 \(c\) 为有效子节点个数（末尾不足 \(r\) 个时 \(c<r\)）。

实现对应：
- Triton kernel: `htree_build_kernel_v2`
- Naive 参考: `build_tree(...)`

---

### 每个 query 位置的整体计算目标

对每个 \((b,t)\) 和每个 query head \(h\)，`htree` 不是对全部 \(T\) 个 token 做注意力，而是通过“自顶向下逐层细化”构造一组 **叶子节点集合**（来自不同层的聚合节点），然后对这些叶子节点做一次标准 softmax 加权：

令最终被“保留并参与 softmax”的节点集合为 \(\mathcal{L}(b,t,h_{kv})\)，其中每个元素是某一层的节点 \((\ell,i)\)。

则对组内 head \(g\)（对应 query head \(h=h_{kv}\cdot G+g\)）输出：
\[
O_{b,t,h}=\frac{\sum_{(\ell,i)\in\mathcal{L}} \exp(s^{(\ell)}_{g,i})\;V^{(\ell)}_{b,i,h_{kv}}}{\sum_{(\ell,i)\in\mathcal{L}} \exp(s^{(\ell)}_{g,i})}
\]
其中 \(s^{(\ell)}_{g,i}\) 为在第 \(\ell\) 层计算得到的打分（见下节）。

实现上用 **online-softmax** 累积 \((m,\;S,\;U)\) 来保证数值稳定（后述）。

---

### RoPE 位置约定（非常关键）

两份实现都使用同一套“**候选列表局部位置**”的 RoPE 约定，而不是 token 的绝对位置：

- 在某层对某个 \((b,t,h_{kv})\) 构造候选列表 \(C = [c_0,\dots,c_{n-1}]\)（长度 \(n=n_{\text{cand}}\)）
- **候选 key** \(K^{(\ell)}_{c_p}\) 使用 RoPE position \(p\)
- **query** 使用 RoPE position \(n-1\)（候选列表最后一个位置，即“rightmost”）

因此打分为（对组内 head \(g\)）：
\[
s_{g,p} = \alpha \cdot \left\langle \text{RoPE}\big(Q_{b,t,h_{kv}\cdot G+g},\;n-1\big),\;\text{RoPE}\big(K^{(\ell)}_{b,c_p,h_{kv}},\;p\big)\right\rangle
\]

实现对应：
- Naive: `positions = arange(num_candidates)`，`query` 用 `num_candidates-1`
- Triton: `rope_pos_q = n_cand-1`；每个候选的 `rope_pos_start + child_offset` 与其 buffer position 一致

---

### 候选集合（Candidates）与 “rightmost” 约束

处理固定的 \((b,t,h_{kv})\)。从顶层 \(\ell=L-1\) 向下到 \(\ell=0\) 迭代。

定义第 \(\ell\) 层中包含位置 \(t\) 的节点（因果边界对应的“最右节点”）：
\[
i^\star_\ell(t) = \left\lfloor t / r^\ell \right\rfloor
\]

#### 顶层候选（语义）
顶层候选包含所有满足 \(\text{range}(L-1,i).\text{start} \le t\) 的节点，等价于：
\[
C_{L-1} = [0,1,\dots,i^\star_{L-1}(t)]
\]

#### 非顶层候选（语义）
非顶层时，候选集合由上一层选出的节点集合 \(\text{Parents}_\ell\) 展开得到：
- 每个 parent 节点 \(p\) 展开成它的 \(r\) 个子节点 \([pr,\dots,pr+r-1]\)
- 并且对 **rightmost parent**（包含 \(t\) 的那一个 parent），只保留到其内部的 rightmost child（保证因果）

实现细节里，“只保留到 rightmost child”通过 `num_valid_children` 实现（见 `htree_compute_scores_kernel` / `htree_accumulate_non_topk_kernel`）。

#### 候选 buffer 的顺序（Triton 语义）
Triton 将候选写入一个线性 buffer（长度上限 `MAX_CANDIDATES`），顺序为：
- 依次遍历 `prev_selected_parents`（升序）
- 对每个 parent，依次遍历 child offset \(0..r-1\)
- 最后一个候选位置必然对应 \(i^\star_\ell(t)\)（因此称为 **rightmost candidate**，其 buffer pos 是 \(n_{\text{cand}}-1\)）

候选数量（Triton 使用的精确公式）：
设本层用于展开的 parent 数量为 \(P\)（`num_valid_parents`），rightmost child offset 为
\[
c^\star_\ell(t) = \left\lfloor t / r^\ell \right\rfloor \bmod r
\]
则
\[
n_{\text{cand}} = (P-1)\cdot r + c^\star_\ell(t) + 1
\]
对应 `src/parallel.py` 中 `n_cand_pos = (num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1`。

---

### shared Top-K（NSA-style）选择：只在非底层执行

对非底层 \(\ell>0\)，候选 \(C\) 会被分成两类：
- **Top-K** 节点：进入下一层继续细化（expand）
- **Non-TopK** 节点：在当前层直接作为“叶子”参与 softmax 累积（merge）

由于 GQA，一个 KV head 的同组 \(G\) 个 query heads **共享同一组 Top-K**（减少分支与带宽）。

对每个组内 head \(g\)，先计算
\[
\text{lse}_g = \log\sum_{p=0}^{n-1}\exp(s_{g,p})
\]
再定义每个候选的 group-importance：
\[
\text{imp}_p = \sum_{g=0}^{G-1}\exp(s_{g,p}-\text{lse}_g)
\]

然后对 \(\text{imp}\) 做 stable Top-K，得到 buffer position 集合 \(\mathcal{S}_\ell\subseteq[0,n)\)（大小 \(K\)，不足补 -1）。

**rightmost 强制选择**：
- 令 rightmost buffer position \(p^\star = n-1\)
- 强制使 \(\text{imp}_{p^\star}\) 为一个极大值（代码用 `1e3`），保证它必进入 Top-K

实现对应：
- Naive: `compute_select_and_merge_shared_gqa(...)` 中 `imp = sum_g exp(...)`，并强制 rightmost
- Triton: `htree_select_topk_shared_gqa_kernel`

---

### stable Top-K 的 bit-packing 约定（实现细节，但 backward 可能要复用）

Triton/Naive 都用相同技巧保证 Top-K 在分数相同/接近时稳定（偏好更小的 buffer index）：

- 将 float32 的低 `LOG_N` 位写入编码的 index（`LOG_N=log2(MAX_CANDIDATES)`，常见为 13）
- 对于非负分数，用 `~idx` 编码（使小 idx 在降序排序时排在前面）
- 排序在编码后的 float32 上进行（bitonic sort）
- 最后再解码回 buffer position

Naive 里对应：`bitpack_encode / bitpack_decode / stable_topk_positions_with_bitpacking`。

---

### non-TopK 的在线 softmax 累积（per head）

对固定 \((b,t,h)\)，维护 online-softmax 状态：
- \(m\)：当前累计的最大 score
- \(S\)：累计的 \(\sum \exp(\cdot)\)
- \(U\)：累计的 \(\sum \exp(\cdot)\,V\)（向量）

将一批新元素 \(\{(s_j, v_j)\}\) 合并进去：
1. \(m'=\max(m,\max_j s_j)\)
2. \(S' = S\cdot \exp(m-m') + \sum_j \exp(s_j-m')\)
3. \(U' = U\cdot \exp(m-m') + \sum_j \exp(s_j-m')\,v_j\)

最终输出：
\[
O = U / \max(S,\varepsilon)
\]

实现对应：
- Naive: `online_softmax_merge(...)`
- Triton:
  - 每层先得到 `(layer_max, layer_sum, layer_output)`（`htree_accumulate_non_topk_kernel`）
  - 再与全局累计合并（`htree_merge_to_global_kernel_v2`）
  - 最终归一化（`htree_final_normalize_kernel_v2`）

---

### “Top-K 不参与当前层累积”的实现方式（mask trick）

Triton forward 的一个关键技巧：为了让累积 kernel 不需要显式比较 `topk_hit`，它在非底层做：

1. 先算出 `topk_positions`
2. 将这些 buffer position 上的 `all_scores[..., pos, g]` 覆写成统一的 `NEG_INF = -1e10`
3. 在累积阶段，仅用阈值 `score > VALID_THRESHOLD` 判断“是否参与”

因此在非底层：
- **Non-TopK**: 正常 score（>阈值）→参与累积
- **Top-K**: 被改写为 -1e10（≤阈值）→不参与累积

底层 \(\ell=0\) 不做 Top-K，也不做阈值过滤：候选全部参与累积（对应 Naive 的“底层全 merge”）。

实现对应：
- 覆写: `htree_mask_topk_scores_kernel`
- 过滤: `htree_accumulate_non_topk_kernel` 中 `score_valid = scores > SCORE_VALID_THRESHOLD`

---

### 下一层 parents 的计算（从 topk buffer pos 映射回 node index）

对非底层，需要把当前层的 Top-K buffer positions \(\mathcal{S}_\ell\) 映射为下一层要展开的 node indices（作为 `prev_selected_parents` 输入给下一层）。

Triton 的候选 buffer 是按（batch_id, parent_slot, child_slot）线性展开的：
- `BC = 32 * r` 是固定 tile（一个 batch 含 32 个 parents）
- `pos -> batch_id = pos // BC`
- `within_batch = pos % BC`
- `parent_slot = within_batch // r`
- `child_slot = within_batch % r`
- `idx_in_topk = batch_id * 32 + parent_slot`（对应到 `prev_selected_parents` 中的 parent）
- `parent_idx = prev_selected_parents[idx_in_topk]`
- `child_node_idx = parent_idx * r + child_slot`

然后对得到的 `child_node_idx` 做升序排序并用 -1 padding。

实现对应：
- `htree_compute_next_parents_kernel`

---

### 与代码实现的关键张量对应（Triton）

- `layers_k[l]`, `layers_v[l]`:
  - 分别对应 \(K^{(\ell)}, V^{(\ell)}\)
- `prev_selected_parents[b,t,h_kv,:]`:
  - 语义上是本层将要展开的 parent 列表（升序，-1 padding）
  - 顶层初始化采用“virtual parents”以避免写出超大候选表（见 `htree_forward_v2` 顶层初始化）
- `all_scores[b,t,h_kv,pos,g]`:
  - 存储候选 buffer 上每个位置的 score \(s_{g,pos}\)
- `num_candidates[b,t,h_kv]`:
  - 存储 \(n_{\text{cand}}\)
- `topk_positions[b,t,h_kv,:]`:
  - 存储 shared Top-K 的 **buffer position**
- `global_max/global_sum/global_output`:
  - 对应跨层 online-softmax 的全局累计状态

---

### forward 伪代码（语义版）

对每个 \(b,t,h_{kv}\)：
- 初始化全局 online-softmax 状态（对组内每个 \(g\)）
- 初始化顶层 `parents`（覆盖到 rightmost）
- for \(\ell = L-1 \dots 0\):
  - 构造候选列表 \(C_\ell\)（由 `parents` 展开并截断到 rightmost）
  - 计算 scores \(s_{g,p}\)（RoPE 使用候选局部位置）
  - if \(\ell>0\):
    - 计算 \(\text{imp}\)，stable Top-K 得到 \(\mathcal{S}_\ell\)，强制包含 rightmost
    - 将 \(\mathcal{S}_\ell\) 从本层 merge 集合中移除（实现上通过 mask 分数到 -inf）
    - 计算下一层 `parents = sort( node_index(\mathcal{S}_\ell) )`
  - 将本层 non-TopK（或底层全部）用 online-softmax 合并进全局状态
- 输出 \(O = U/S\)

---

### 实现参数约束（来自 Triton）

当前 `src/parallel.py` 的实现有明确 regime 限制：
- `max_top_nodes == top_k_per_layer * compression_rate`（候选 buffer 长度固定为顶层节点数）
- `top_k_per_layer` 必须是 2 的幂（bitonic sort）
- `compression_rate` 必须是 2 的幂（`BC=32*r` 也需为 2 的幂）
- `max_top_nodes` 也需为 2 的幂（用于 bit-packing 的 `LOG_N`）

这些限制会直接影响 backward 的实现方式（尤其是 topk/排序与 buffer 布局）。


