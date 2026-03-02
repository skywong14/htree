# HTree Sparse Attention: Backward Pass Mathematical Derivation

## 1. Notation

### 1.1 Dimensions & Constants

| Symbol | Definition |
|--------|-----------|
| B, T, H, H_kv | Batch size, sequence length, query heads, KV heads |
| G = H / H_kv | GQA group size |
| K, V | Key/query head dim, value head dim |
| σ = K^{−1/2} | Scale factor |
| C | Compression rate |
| L | Number of tree layers (l = 0, …, L−1; layer 0 = bottom/original) |
| T_l | Number of nodes at tree layer l (T_0 = T) |
| TOP_K | Number of selected parents per layer |

### 1.2 Per-Layer Notation

For a given (b, t, h_kv) at tree layer l:

| Symbol | Definition |
|--------|-----------|
| P^(l) | Selected parent indices (sorted ascending, from `prev_selected_parents`) |
| N_p^(l) | Number of valid parents (entries ≥ 0 in P^(l)) |
| N_c^(l) | Number of candidates = (N_p − 1)·C + r + 1, where r = rightmost_child_idx |
| n(j) | Node index of candidate at flat position j: n(j) = P^(l)\[⌊j/C⌋\] · C + (j mod C) |
| p_c = j | RoPE position for K at flat candidate position j |
| p_q^(l) = N_c^(l) − 1 | RoPE position for Q at layer l |
| D^(l) | Set of candidate flat positions that **contribute to output** (dropped candidates for l > 0; ALL candidates for l = 0) |
| K_n^(l), V_n^(l) | Key / Value vector at node n of tree layer l |

### 1.3 RoPE

Half-rotary RoPE splitting vector x = [x_lo, x_hi]:

```
R(x, p)_lo = x_lo · cos(θ_p) − x_hi · sin(θ_p)
R(x, p)_hi = x_lo · sin(θ_p) + x_hi · cos(θ_p)
```

where θ_p is the frequency vector at position p. This is a linear (orthogonal) transformation: R(x, p) = M_p · x.

---

## 2. Forward Pass (Mathematical Summary)

### 2.1 Tree Building (bottom-up mean pooling)

For l = 1, …, L−1 and each parent node p:

```
K_p^(l) = (1/|Ch(p)|) · Σ_{i ∈ Ch(p)} K_i^(l−1)
V_p^(l) = (1/|Ch(p)|) · Σ_{i ∈ Ch(p)} V_i^(l−1)
```

where Ch(p) = {p·C, p·C+1, …, min(p·C+C−1, T_{l−1}−1)}.

### 2.2 Per-Layer Attention Score

For query position t, head h (in GQA group h_kv), at layer l, for candidate c at flat position j ∈ D^(l):

```
q̃_h^(l) = σ · R(q_{t,h}, p_q^(l))          ... scaled, RoPE'd query   [K]
k̃_c     = R(K_{n(j)}^(l), j)                ... RoPE'd key             [K]
S_{h,c}^(l) = q̃_h^(l) · k̃_c               ... attention score (scalar)
```

**Note**: Q's RoPE position p_q^(l) = N_c^(l) − 1 **differs across layers** (because each layer has a different number of candidates). K's RoPE position equals its flat position j in the candidate list (NOT the node's original sequence position).

### 2.3 Global Output

The online-softmax merge across all layers makes the final output mathematically equivalent to a **single softmax over the union of all layers' contributing candidates**:

```
O_{t,h} = Σ_l  Σ_{c ∈ D^(l)}  P_{h,c}^(l) · V_{n(c)}^(l)
```

where the attention probability is:

```
                        exp(S_{h,c}^(l))
P_{h,c}^(l) = ─────────────────────────────────────
               Σ_{l'} Σ_{c' ∈ D^(l')} exp(S_{h,c'}^(l'))
```

Define the logsumexp:

```
lse_h = log( Σ_l Σ_{c ∈ D^(l)} exp(S_{h,c}^(l)) )
```

Then:

```
P_{h,c}^(l) = exp(S_{h,c}^(l) − lse_h)
```

**Forward saves** (equivalently):
- `global_max_{t,h}` and `global_sum_{t,h}`, from which lse = log(global_sum) + global_max.
- Or directly use: P = exp(S − global_max) / global_sum.

---

## 3. Backward Pass Derivation

**Given**: dO = ∂L/∂O, shape [B, T, H, V].

**Goal**: Compute dq = ∂L/∂q, dk = ∂L/∂k, dv = ∂L/∂v w.r.t. the original inputs.

### 3.0 Key Observation: Top-K Is Non-Differentiable

The Top-K selection (importance-based candidate routing at non-bottom layers) is a **discrete operation**. We treat the selection pattern — which candidates are dropped (accumulated into output) vs. selected (passed to the next layer) — as **fixed constants** during backward. Gradients flow only through the attention weights and values of candidates that actually contributed to the output.

### 3.1 What Forward Must Save for Backward

| Quantity | Shape | Purpose |
|----------|-------|---------|
| global_max, global_sum | [B, T, H] each | Reconstruct attention probs P |
| output O | [B, T, H, V] | Compute delta D |
| selected_parents^(l) for each layer l | [B, T, H_kv, TOP_K] × L | Reconstruct sparsity pattern + RoPE positions |
| tree layers K^(l), V^(l) | [B, T_l, H_kv, K/V] × L | Recompute scores (can be rebuilt from k, v) |
| cos_cache, sin_cache | [cache_size, K/2] | RoPE (can be rebuilt from rope_base) |

### 3.2 Step 1: Delta Computation

```
┌─────────────────────────────────────────────────┐
│  D_{t,h} = Σ_v  dO_{t,h,v} · O_{t,h,v}        │
│                                                   │
│  Shape: [B, T, H].  Computed once, shared by     │
│  all layers.                                      │
└─────────────────────────────────────────────────┘
```

This is the standard "delta correction" term in softmax backward (same as FlashAttention).

**Proof sketch**: From O = P · V and ∂L/∂P_{h,c} = dO_h · V_c^T, the softmax backward formula involves the row-sum Σ_c P_{h,c} · (dO_h · V_c) = dO_h · O_h = D_h.

### 3.3 Step 2: Per-Layer Backward (Recompute + Accumulate)

Process layers in the **same order as forward** (top → bottom, i.e. l = L−1, …, 0). For each layer l and each (b, t, h_kv):

#### (a) Reconstruct Candidate Set

Use saved P^(l) (selected_parents) to reconstruct the flat candidate list, exactly as in the forward. This gives: flat positions j, node indices n(j), RoPE positions, validity masks, and the set D^(l) (dropped or all).

#### (b) Recompute Scores (FlashAttention-style recomputation)

```
q̃_h^(l) = σ · R(q_{t,h}, p_q^(l))                              [G, K]
k̃_c     = R(K_{n(c)}^(l), p_c)           for c ∈ D^(l)         [|D|, K]
S_{h,c}^(l) = q̃_h^(l) · k̃_c                                   [G, |D|]
```

#### (c) Reconstruct Attention Probabilities

```
┌──────────────────────────────────────────────────────────────┐
│  P_{h,c}^(l) = exp(S_{h,c}^(l) − global_max_h) / global_sum_h  │
└──────────────────────────────────────────────────────────────┘
```

This is numerically stable because S and global_max are in similar ranges.

#### (d) Score Gradient

```
┌──────────────────────────────────────────────────────────────────────┐
│  dS_{h,c}^(l) = P_{h,c}^(l) · ( Σ_v dO_{t,h,v} · V_{n(c),v}^(l)  −  D_{t,h} ) │
└──────────────────────────────────────────────────────────────────────┘
```

In matrix form, define dP_{h,c} = Σ_v dO_{t,h,v} · V_{n(c),v}^(l), i.e. dP = dO · V^T (shape [G, |D|]):

```
dS = P ⊙ (dP − D[:, None])           [G, |D|]
```

where D[:, None] broadcasts the per-head delta.

#### (e) dQ: Accumulate Across Layers

For each layer l, compute the gradient of Q through the scaled RoPE'd score:

```
dq̃_h^(l) = Σ_{c ∈ D^(l)} dS_{h,c}^(l) · k̃_c
```

In matrix form:

```
dQ̃^(l) = dS^(l) · K̃^(l)                                 [G, K] = [G, |D|] × [|D|, K]
```

Then un-RoPE and un-scale (since q̃ = σ · R(q, p_q), the chain rule gives):

```
┌───────────────────────────────────────────────────────────┐
│  dq_{t,h}  +=  σ · R⁻¹( dQ̃_h^(l),  p_q^(l) )          │
│                                                             │
│  (accumulated across all layers l = 0, …, L−1)             │
└───────────────────────────────────────────────────────────┘
```

**Derivation**: Since q̃ = σ · M_p · q where M_p is the RoPE rotation matrix:
- ∂L/∂q = σ · M_p^T · ∂L/∂q̃ = σ · R⁻¹(dq̃, p)

#### (f) dK: Per Tree Layer (requires atomic across t)

For each candidate c ∈ D^(l) with node index n(c) and RoPE position p_c:

```
dk̃_c = Σ_{h ∈ group(h_kv)} dS_{h,c}^(l) · q̃_h^(l)
```

In matrix form:

```
dK̃^(l) = (dS^(l))^T · Q̃^(l)                             [|D|, K] = [|D|, G] × [G, K]
```

Then un-RoPE (since k̃ = R(K, p_c), the chain rule gives):

```
┌───────────────────────────────────────────────────────────┐
│  dK_{n(c)}^(l)  +=  R⁻¹( dk̃_c,  p_c )                  │
│                                                             │
│  (accumulated across ALL query positions t that             │
│   include candidate c in their D^(l) set)                   │
└───────────────────────────────────────────────────────────┘
```

**Important**: Multiple query positions t may attend to the same node n in layer l (they share the same tree KV). These contributions must be summed — this requires **atomicAdd** in the Triton kernel, or an alternative accumulation strategy.

#### (g) dV: Per Tree Layer (requires atomic across t)

```
dV_c^(l) = Σ_{h ∈ group(h_kv)} P_{h,c}^(l) · dO_{t,h}
```

In matrix form:

```
dV^(l) = (P^(l))^T · dO                                   [|D|, V] = [|D|, G] × [G, V]
```

```
┌───────────────────────────────────────────────────────────┐
│  dV_{n(c)}^(l)  +=  dV_c^(l)                             │
│                                                             │
│  (accumulated across ALL query positions t;                 │
│   NO RoPE inverse needed — V has no RoPE)                   │
└───────────────────────────────────────────────────────────┘
```

### 3.4 Step 3: Inverse RoPE

The inverse RoPE R⁻¹(x, p) = R(x, −p) = M_p^T · x:

```
R⁻¹(x, p)_lo =  x_lo · cos(θ_p)  +  x_hi · sin(θ_p)
R⁻¹(x, p)_hi = −x_lo · sin(θ_p)  +  x_hi · cos(θ_p)
```

**Proof**: Since M_p is orthogonal (rotation matrix), M_p^T = M_p^{−1}. The transpose of:
```
| cos  −sin |       | cos   sin |
| sin   cos |   is  |−sin   cos |
```
which corresponds to negating the angle (cos(−θ) = cos(θ), sin(−θ) = −sin(θ)).

Cost: Same as forward RoPE — no extra overhead.

### 3.5 Step 4: Tree Gradient Back-Propagation

After computing dK^(l) and dV^(l) from attention backward for all layers, propagate gradients **top-down** through the mean-pooling:

For l = L−1, L−2, …, 1 (in decreasing order):

```
┌──────────────────────────────────────────────────────────────────┐
│  For each parent p at layer l:                                    │
│    For each child i ∈ Ch(p):                                      │
│      dK_i^(l−1)  +=  dK_p^(l) / |Ch(p)|                          │
│      dV_i^(l−1)  +=  dV_p^(l) / |Ch(p)|                          │
└──────────────────────────────────────────────────────────────────┘
```

**Derivation**: Since K_p^(l) = (1/|Ch(p)|) · Σ_{i} K_i^(l−1):
- ∂K_p / ∂K_i = 1/|Ch(p)| (identity scaled)
- By chain rule: dK_i^(l−1) += dK_p^(l) · (1/|Ch(p)|)

After propagation, **dK^(0) = dk** and **dV^(0) = dv** are the final gradients w.r.t. original inputs.

---

## 4. Matrix-Form Summary (Kernel Implementation Reference)

For each layer l, query position t, KV head h_kv, with |D| contributing candidates:

| Step | Formula | Shape |
|------|---------|-------|
| Load Q̃ | Q̃ = σ · R(Q, p_q^(l)) | [G, K] |
| Load K̃ | K̃_c = R(K_{n(c)}, p_c) for c ∈ D | [|D|, K] |
| Score | S = Q̃ · K̃^T | [G, |D|] |
| Prob | P = exp(S − max) / sum | [G, |D|] |
| dP | dP = dO · V^T | [G, |D|] |
| dS | dS = P ⊙ (dP − D) | [G, |D|] |
| dQ̃ | dQ̃ = dS · K̃ | **[G, K] = [G, |D|] × [|D|, K]** |
| dK̃ | dK̃ = dS^T · Q̃ | **[|D|, K] = [|D|, G] × [G, K]** |
| dV | dV = P^T · dO | **[|D|, V] = [|D|, G] × [G, V]** |
| Un-RoPE dQ | dq += σ · R⁻¹(dQ̃, p_q) | [G, K] |
| Un-RoPE dK | dK_{n(c)} += R⁻¹(dK̃_c, p_c) | [K] per candidate |

All matrix multiplies map directly to `tl.dot` operations, symmetric to the forward kernel's QK^T and PV.

---

## 5. Correctness Verification

The derivation can be verified by checking that for standard (non-sparse, single-layer, no-tree, no-RoPE) softmax attention, the formulas reduce to the well-known:

```
dV = P^T · dO
dS = P ⊙ (dO · V^T − D)           where D = rowsum(dO ⊙ O)
dQ = dS · K · scale
dK = dS^T · Q · scale
```

For HTree, the only additions are:
1. RoPE forward/inverse on Q and K
2. Per-layer decomposition of the candidate set (with a single shared softmax denominator)
3. Tree gradient propagation for non-bottom layers

---

## 6. Key Implementation Considerations

### 6.1 dK/dV Atomic Contention

Multiple query positions t write to the same dK_n^(l) and dV_n^(l). This requires either:
- **(A) atomicAdd**: Simple, but potential contention on popular nodes (especially at bottom layer where many queries access the same original tokens).
- **(B) Inverse index**: Precompute a mapping (node → list of queries). Iterate over KV nodes in outer loop. Avoids atomics but requires extra preprocessing.
- **(C) Per-query buffers + reduction**: Allocate per-(b,t) dK/dV, then reduce. High memory cost.

**Recommendation**: Start with (A) for simplicity. Optimize later if profiling shows contention bottleneck.

### 6.2 Per-Layer RoPE Position for Q

Q's RoPE position p_q^(l) = N_c^(l) − 1 **differs across layers** because each layer has a different number of candidates. This means:
- dq must be accumulated across L layers, each requiring its own inverse RoPE.
- In the kernel, this is handled naturally by the layer-sequential processing.

### 6.3 Non-Bottom Layer: Identifying Dropped Candidates

In the forward, the non-bottom kernel selects top-K and immediately accumulates dropped candidates. In the backward, we need to know **which candidates were dropped**. Two options:
- **(A) Re-run the importance + top-K logic** in the backward kernel (recompute scores, compute importance, select top-K, then the complement is dropped). This avoids extra storage but adds compute.
- **(B) Save an explicit dropped mask** from forward. Extra storage but simpler backward logic.

**Recommendation**: Option (A) is preferred — it's consistent with the FlashAttention philosophy of recomputation over storage, and the top-K computation is relatively cheap compared to the attention backward. *Note: The current code (parallel.py / backward.py) uses Option (B) — saves per_layer_parents from forward and reuses them in backward, without recomputing Top-K.*

### 6.4 Memory Budget for Saved States

For typical parameters (B=1, T=32768, H_kv=8, TOP_K=512, L=3, int32):
- selected_parents: L × B × T × H_kv × TOP_K × 4B ≈ 3 × 512 MB ≈ 1.5 GB
- global_max + global_sum: 2 × B × T × H × 4B (negligible)
- output O: B × T × H × V × 2B (already needed)

The selected_parents dominates. To reduce memory:
- Recompute instead of save (run forward's selection logic partially in backward)
- Only requires saving the topmost layer's initial parents (which is deterministic from T and compression_rate)

### 6.5 Numerical Stability

- P reconstruction: P = exp(S − max) / sum is stable when S and max are in similar ranges (guaranteed by the online-softmax forward).
- When global_sum is very small (≈ 0, meaning few valid candidates), P may have precision issues. Guard with the same threshold (1e-10) used in the forward merge.
