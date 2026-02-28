from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


HTREE_SCORE_NEG_INF: float = -1.0e10
HTREE_SCORE_VALID_THRESHOLD: float = HTREE_SCORE_NEG_INF * 0.9


@dataclass
class LayerTrace:
    layer_idx: int
    selected_parents: Optional[torch.Tensor]  # [B, T, H_kv, TOP_K], None for bottom layer
    layer_max: torch.Tensor  # [B, T, H]
    layer_sum: torch.Tensor  # [B, T, H]
    layer_output: torch.Tensor  # [B, T, H, V]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    y_lo = x_lo * cos - x_hi * sin
    y_hi = x_lo * sin + x_hi * cos
    return torch.cat([y_lo, y_hi], dim=-1)


def _inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse RoPE: R^{-1}(x, p) = M_p^T x, equivalent to RoPE with negated angle."""
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    y_lo = x_lo * cos + x_hi * sin
    y_hi = -x_lo * sin + x_hi * cos
    return torch.cat([y_lo, y_hi], dim=-1)


def _build_tree(
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int,
    max_top_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build mean-pooled tree bottom-up.
    Returns:
      layers_k/layers_v: index 0 is bottom layer (original k/v),
      index -1 is the top layer (<= max_top_nodes).
    """
    bsz, seq_len, h_kv, k_dim = k.shape
    v_dim = v.shape[-1]

    layers_k: List[torch.Tensor] = [k]
    layers_v: List[torch.Tensor] = [v]
    current_k = k
    current_v = v
    current_len = seq_len

    while current_len > max_top_nodes:
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(bsz, next_len, h_kv, k_dim, dtype=k.dtype, device=k.device)
        next_v = torch.empty(bsz, next_len, h_kv, v_dim, dtype=v.dtype, device=v.device)

        for parent_idx in range(next_len):
            start = parent_idx * compression_rate
            end = min(start + compression_rate, current_len)
            next_k[:, parent_idx] = current_k[:, start:end].mean(dim=1)
            next_v[:, parent_idx] = current_v[:, start:end].mean(dim=1)

        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k = next_k
        current_v = next_v
        current_len = next_len

    return layers_k, layers_v


def _build_rope_cache(
    k_dim: int,
    cache_size: int,
    device: torch.device,
    rope_base: float,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_base ** (torch.arange(0, k_dim, 2, dtype=dtype, device=device) / k_dim)
    )
    positions = torch.arange(cache_size, dtype=dtype, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _online_softmax_merge_group(
    cur_max: torch.Tensor,  # [G]
    cur_sum: torch.Tensor,  # [G]
    cur_output: torch.Tensor,  # [G, V]
    scores: torch.Tensor,  # [G, N]
    values: torch.Tensor,  # [N, V]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        return cur_max, cur_sum, cur_output

    tile_m = scores.max(dim=1).values
    new_max = torch.maximum(cur_max, tile_m)
    alpha = torch.exp(cur_max - new_max)

    probs = torch.exp(scores - new_max[:, None])
    new_sum = cur_sum * alpha + probs.sum(dim=1)
    new_output = cur_output * alpha[:, None] + probs @ values.to(probs.dtype)
    return new_max, new_sum, new_output


def _select_shared_topk(
    importance: torch.Tensor,  # [N]
    node_indices: torch.Tensor,  # [N]
    top_k: int,
    rightmost_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select shared Top-K by importance.
    Returns:
      selected_mask: [N] bool
      selected_nodes_sorted_padded: [TOP_K] int32
    """
    num_candidates = int(importance.numel())
    importance_for_sort = importance.clone()
    importance_for_sort[rightmost_pos] = 1e6

    _, sorted_pos = torch.sort(importance_for_sort, descending=True, stable=True)
    take = min(top_k, num_candidates)
    top_pos = sorted_pos[:take]

    selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=importance.device)
    if take > 0:
        selected_mask[top_pos] = True

    selected_nodes = node_indices[top_pos].to(torch.int32)
    selected_nodes_sorted, _ = torch.sort(selected_nodes)
    selected_nodes_padded = torch.full([top_k], -1, dtype=torch.int32, device=importance.device)
    if take > 0:
        selected_nodes_padded[:take] = selected_nodes_sorted

    return selected_mask, selected_nodes_padded


def _init_prev_selected_parents(
    seq_len: int,
    bsz: int,
    h_kv: int,
    top_k_per_layer: int,
    compression_rate: int,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(seq_len, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_candidates = (
        torch.arange(top_k_per_layer, dtype=torch.int32, device=device)[None, :]
        .expand(seq_len, -1)
    )
    valid_mask = parent_candidates < num_virtual_parents[:, None]
    prev = torch.where(valid_mask, parent_candidates, torch.full_like(parent_candidates, -1))
    return prev[None, :, None, :].expand(bsz, seq_len, h_kv, top_k_per_layer).contiguous()


# ---------------------------------------------------------------------------
#  Original forward (unchanged logic, kept for backward compatibility)
# ---------------------------------------------------------------------------

def htree_forward_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None,
    rope_base: float = 10000.0,
    selection_mode: str = "parallel_importance",
    return_trace: bool = False,
):
    """
    Naive Python reference for HTree forward (GQA only).

    Notes:
    - Uses the same candidate expansion order as Triton kernels:
      RoPE position is candidate flat position (0..num_candidates-1), not node index.
    - Group-shared Top-K follows Triton kernel strategy:
      importance = sum(exp(score - m)) across query heads, where m is a
      shared stabilizing max (equivalent to any globally shared shift).
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be rank-4 tensors")

    bsz, seq_len, h, k_dim = q.shape
    h_kv = k.shape[2]
    v_dim = v.shape[-1]
    if h % h_kv != 0:
        raise ValueError(f"H ({h}) must be divisible by H_kv ({h_kv})")
    if k_dim % 2 != 0:
        raise ValueError(f"K dim must be even for RoPE, got {k_dim}")
    if k.shape[:2] != (bsz, seq_len) or v.shape[:2] != (bsz, seq_len):
        raise ValueError("k/v first two dims must match q")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k/v must have same H_kv")
    if max_top_nodes != top_k_per_layer * compression_rate:
        raise ValueError(
            f"max_top_nodes ({max_top_nodes}) must equal "
            f"top_k_per_layer*compression_rate ({top_k_per_layer * compression_rate})"
        )

    # Keep "exp_sum" as backward-compatible alias.
    if selection_mode == "exp_sum":
        selection_mode = "parallel_importance"
    if selection_mode != "parallel_importance":
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    num_groups = h // h_kv
    if scale is None:
        scale = k_dim ** -0.5

    device = q.device
    out_dtype = q.dtype

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    layers_k, layers_v = _build_tree(k, v, compression_rate, max_top_nodes)
    num_layers = len(layers_k)

    rope_cache_size = max_top_nodes + 1024
    cos_cache, sin_cache = _build_rope_cache(k_dim, rope_cache_size, device, rope_base)

    global_max = torch.full([bsz, seq_len, h], HTREE_SCORE_NEG_INF, dtype=torch.float32, device=device)
    global_sum = torch.zeros([bsz, seq_len, h], dtype=torch.float32, device=device)
    global_output = torch.zeros([bsz, seq_len, h, v_dim], dtype=torch.float32, device=device)

    prev_selected_parents = _init_prev_selected_parents(
        seq_len, bsz, h_kv, top_k_per_layer, compression_rate, num_layers, device,
    )

    traces: Dict[int, LayerTrace] = {}

    for layer_idx in range(num_layers - 1, -1, -1):
        layer_k = layers_k[layer_idx]
        layer_v = layers_v[layer_idx]
        n_layer = layer_k.shape[1]
        layer_power = compression_rate ** layer_idx
        is_bottom = layer_idx == 0

        layer_max = torch.full([bsz, seq_len, h], HTREE_SCORE_NEG_INF, dtype=torch.float32, device=device)
        layer_sum = torch.zeros([bsz, seq_len, h], dtype=torch.float32, device=device)
        layer_output = torch.zeros([bsz, seq_len, h, v_dim], dtype=torch.float32, device=device)

        next_selected_parents = None
        if not is_bottom:
            next_selected_parents = torch.full_like(prev_selected_parents, -1)

        for b in range(bsz):
            for t in range(seq_len):
                rightmost_idx = t // layer_power
                rightmost_parent_idx = rightmost_idx // compression_rate
                rightmost_child_idx = rightmost_idx % compression_rate

                for kv_h in range(h_kv):
                    prev_list = prev_selected_parents[b, t, kv_h]
                    valid_parents = prev_list[prev_list >= 0]
                    num_valid_parents = int(valid_parents.numel())
                    if num_valid_parents <= 0:
                        raise RuntimeError(
                            f"No valid parents at layer={layer_idx}, b={b}, t={t}, kv_h={kv_h}"
                        )

                    num_candidates = (num_valid_parents - 1) * compression_rate + rightmost_child_idx + 1
                    rightmost_pos = num_candidates - 1

                    cand_pos = torch.arange(num_candidates, device=device, dtype=torch.long)
                    parent_off = torch.div(cand_pos, compression_rate, rounding_mode="floor")
                    child_slot = cand_pos - parent_off * compression_rate
                    parent_idx = valid_parents[parent_off].to(torch.long)
                    node_indices = parent_idx * compression_rate + child_slot

                    if bool((node_indices < 0).any()) or bool((node_indices >= n_layer).any()):
                        raise RuntimeError(
                            f"Invalid node index at layer={layer_idx}, b={b}, t={t}, kv_h={kv_h}"
                        )

                    k_nodes = layer_k[b, node_indices, kv_h].to(torch.float32)  # [N, K]
                    v_nodes = layer_v[b, node_indices, kv_h].to(torch.float32)  # [N, V]

                    cos_k = cos_cache[cand_pos]
                    sin_k = sin_cache[cand_pos]
                    k_rope = _apply_rope(k_nodes, cos_k, sin_k)

                    h_start = kv_h * num_groups
                    h_end = h_start + num_groups
                    q_group = q[b, t, h_start:h_end].to(torch.float32)  # [G, K]
                    cos_q = cos_cache[rightmost_pos]
                    sin_q = sin_cache[rightmost_pos]
                    q_rope = _apply_rope(q_group, cos_q, sin_q) * float(scale)
                    scores = q_rope @ k_rope.transpose(0, 1)  # [G, N]

                    if is_bottom:
                        merge_mask = torch.ones(num_candidates, dtype=torch.bool, device=device)
                    else:
                        local_m = scores.max()
                        importance = torch.exp(scores - local_m).sum(dim=0)

                        selected_mask, selected_nodes = _select_shared_topk(
                            importance=importance,
                            node_indices=node_indices,
                            top_k=top_k_per_layer,
                            rightmost_pos=rightmost_pos,
                        )
                        next_selected_parents[b, t, kv_h] = selected_nodes
                        merge_mask = ~selected_mask

                    if bool(merge_mask.any()):
                        merge_scores = scores[:, merge_mask]
                        merge_values = v_nodes[merge_mask]
                        cur_max = layer_max[b, t, h_start:h_end]
                        cur_sum = layer_sum[b, t, h_start:h_end]
                        cur_output = layer_output[b, t, h_start:h_end]
                        cur_max, cur_sum, cur_output = _online_softmax_merge_group(
                            cur_max,
                            cur_sum,
                            cur_output,
                            merge_scores,
                            merge_values,
                        )
                        layer_max[b, t, h_start:h_end] = cur_max
                        layer_sum[b, t, h_start:h_end] = cur_sum
                        layer_output[b, t, h_start:h_end] = cur_output

        cur_has = layer_sum > 1e-10
        g_has = global_sum > 1e-10
        new_max = torch.maximum(global_max, layer_max)
        scale_g = torch.where(g_has, torch.exp(global_max - new_max), torch.zeros_like(global_sum))
        scale_c = torch.where(cur_has, torch.exp(layer_max - new_max), torch.zeros_like(layer_sum))

        merged_sum = global_sum * scale_g + layer_sum * scale_c
        merged_output = global_output * scale_g[..., None] + layer_output * scale_c[..., None]

        global_max = torch.where(cur_has, torch.where(g_has, new_max, layer_max), global_max)
        global_sum = torch.where(cur_has, torch.where(g_has, merged_sum, layer_sum), global_sum)
        global_output = torch.where(
            cur_has[..., None],
            torch.where(g_has[..., None], merged_output, layer_output),
            global_output,
        )

        if return_trace:
            traces[layer_idx] = LayerTrace(
                layer_idx=layer_idx,
                selected_parents=None if is_bottom else next_selected_parents.clone(),
                layer_max=layer_max.clone(),
                layer_sum=layer_sum.clone(),
                layer_output=layer_output.clone(),
            )

        if not is_bottom:
            prev_selected_parents = next_selected_parents

    output = global_output / global_sum[..., None]
    output = output.to(out_dtype)

    if not return_trace:
        return output

    trace_obj = {
        "num_layers": num_layers,
        "num_groups": num_groups,
        "selection_mode": selection_mode,
        "layers": traces,
    }
    return output, trace_obj


# ---------------------------------------------------------------------------
#  Autograd wrapper with analytical backward
# ---------------------------------------------------------------------------

class HTreeFunction(torch.autograd.Function):
    """torch.autograd.Function wrapping HTree forward + analytical backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        compression_rate: int,
        max_top_nodes: int,
        top_k_per_layer: int,
        scale: float,
        rope_base: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        H_kv = k.shape[2]
        V = v.shape[-1]
        G = H // H_kv
        device = q.device
        cdtype = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32

        layers_k, layers_v = _build_tree(k, v, compression_rate, max_top_nodes)
        num_layers = len(layers_k)
        cache_size = max_top_nodes + 1024
        cos_cache, sin_cache = _build_rope_cache(K, cache_size, device, rope_base, dtype=cdtype)

        global_max = torch.full([B, T, H], HTREE_SCORE_NEG_INF, dtype=cdtype, device=device)
        global_sum = torch.zeros([B, T, H], dtype=cdtype, device=device)
        global_output = torch.zeros([B, T, H, V], dtype=cdtype, device=device)

        prev_selected_parents = _init_prev_selected_parents(
            T, B, H_kv, top_k_per_layer, compression_rate, num_layers, device,
        )

        per_layer_parents: Dict[int, torch.Tensor] = {}

        for layer_idx in range(num_layers - 1, -1, -1):
            per_layer_parents[layer_idx] = prev_selected_parents

            lk = layers_k[layer_idx]
            lv = layers_v[layer_idx]
            n_layer = lk.shape[1]
            layer_power = compression_rate ** layer_idx
            is_bottom = layer_idx == 0

            l_max = torch.full([B, T, H], HTREE_SCORE_NEG_INF, dtype=cdtype, device=device)
            l_sum = torch.zeros([B, T, H], dtype=cdtype, device=device)
            l_out = torch.zeros([B, T, H, V], dtype=cdtype, device=device)

            nsp = None
            if not is_bottom:
                nsp = torch.full_like(prev_selected_parents, -1)

            for b in range(B):
                for t in range(T):
                    ri = t // layer_power
                    rci = ri % compression_rate

                    for kv_h in range(H_kv):
                        pl = prev_selected_parents[b, t, kv_h]
                        vp = pl[pl >= 0]
                        nvp = int(vp.numel())
                        if nvp <= 0:
                            continue

                        nc = (nvp - 1) * compression_rate + rci + 1
                        rp = nc - 1

                        cp = torch.arange(nc, device=device, dtype=torch.long)
                        po = cp // compression_rate
                        cs = cp - po * compression_rate
                        pidx = vp[po].long()
                        ni = pidx * compression_rate + cs

                        kn = lk[b, ni, kv_h].to(cdtype)
                        vn = lv[b, ni, kv_h].to(cdtype)
                        ck = cos_cache[cp]; sk = sin_cache[cp]
                        kr = _apply_rope(kn, ck, sk)

                        h0 = kv_h * G; h1 = h0 + G
                        qg = q[b, t, h0:h1].to(cdtype)
                        cq = cos_cache[rp]; sq = sin_cache[rp]
                        qr = _apply_rope(qg, cq, sq) * float(scale)
                        scores = qr @ kr.T

                        if is_bottom:
                            mm = torch.ones(nc, dtype=torch.bool, device=device)
                        else:
                            lm = scores.max()
                            imp = torch.exp(scores - lm).sum(dim=0)
                            sel_mask, sel_nodes = _select_shared_topk(imp, ni, top_k_per_layer, rp)
                            nsp[b, t, kv_h] = sel_nodes
                            mm = ~sel_mask

                        if mm.any():
                            ms = scores[:, mm]; mv = vn[mm]
                            cm = l_max[b, t, h0:h1]
                            csm = l_sum[b, t, h0:h1]
                            co = l_out[b, t, h0:h1]
                            cm, csm, co = _online_softmax_merge_group(cm, csm, co, ms, mv)
                            l_max[b, t, h0:h1] = cm
                            l_sum[b, t, h0:h1] = csm
                            l_out[b, t, h0:h1] = co

            ch = l_sum > 1e-10
            gh = global_sum > 1e-10
            nm = torch.maximum(global_max, l_max)
            sg = torch.where(gh, torch.exp(global_max - nm), torch.zeros_like(global_sum))
            sc = torch.where(ch, torch.exp(l_max - nm), torch.zeros_like(l_sum))
            ms_ = global_sum * sg + l_sum * sc
            mo_ = global_output * sg[..., None] + l_out * sc[..., None]
            global_max = torch.where(ch, torch.where(gh, nm, l_max), global_max)
            global_sum = torch.where(ch, torch.where(gh, ms_, l_sum), global_sum)
            global_output = torch.where(ch[..., None], torch.where(gh[..., None], mo_, l_out), global_output)

            if not is_bottom:
                prev_selected_parents = nsp

        output = (global_output / global_sum[..., None]).to(q.dtype)

        # -- save for backward --
        ctx.save_for_backward(q, k, v, output)
        ctx.global_max = global_max.detach()
        ctx.global_sum = global_sum.detach()
        ctx.cos_cache = cos_cache
        ctx.sin_cache = sin_cache
        ctx.layers_k = [x.detach() for x in layers_k]
        ctx.layers_v = [x.detach() for x in layers_v]
        ctx.per_layer_parents = {l: p.detach() for l, p in per_layer_parents.items()}
        ctx.num_layers = num_layers
        ctx.compression_rate = compression_rate
        ctx.scale = scale
        ctx.top_k_per_layer = top_k_per_layer
        return output

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        q, k, v, output = ctx.saved_tensors
        gmax = ctx.global_max
        gsum = ctx.global_sum
        cos_c = ctx.cos_cache
        sin_c = ctx.sin_cache
        layers_k = ctx.layers_k
        layers_v = ctx.layers_v
        plp = ctx.per_layer_parents
        NL = ctx.num_layers
        CR = ctx.compression_rate
        scale = ctx.scale
        TK = ctx.top_k_per_layer

        B, T, H, K = q.shape
        H_kv = k.shape[2]
        V = v.shape[-1]
        G = H // H_kv
        device = q.device
        cd = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32

        dO = dO.to(cd)
        out_f = output.to(cd)

        # Step 1: delta D = rowsum(dO * O)
        D = (dO * out_f).sum(dim=-1)  # [B, T, H]

        dq = torch.zeros(B, T, H, K, dtype=cd, device=device)
        dk_layers = [torch.zeros_like(x, dtype=cd) for x in layers_k]
        dv_layers = [torch.zeros_like(x, dtype=cd) for x in layers_v]

        # Step 2: per-layer backward (same order as forward: top → bottom)
        for li in range(NL - 1, -1, -1):
            lk = layers_k[li]
            lv = layers_v[li]
            n_layer = lk.shape[1]
            lp = CR ** li
            is_bottom = li == 0
            psel = plp[li]

            # For non-bottom: derive dropped mask from per_layer_parents[li - 1]
            sel_at_layer: Optional[Dict[int, torch.Tensor]] = None
            if not is_bottom:
                sel_at_layer = plp[li - 1]

            for b in range(B):
                for t in range(T):
                    ri = t // lp
                    rci = ri % CR

                    for kv_h in range(H_kv):
                        pl = psel[b, t, kv_h]
                        vp = pl[pl >= 0]
                        nvp = int(vp.numel())
                        if nvp <= 0:
                            continue

                        nc = (nvp - 1) * CR + rci + 1
                        rp = nc - 1
                        cp = torch.arange(nc, device=device, dtype=torch.long)
                        po = cp // CR
                        cs = cp - po * CR
                        pidx = vp[po].long()
                        ni = pidx * CR + cs  # node indices at layer li

                        # Determine merge mask
                        if is_bottom:
                            mm = torch.ones(nc, dtype=torch.bool, device=device)
                        else:
                            st = sel_at_layer[b, t, kv_h]
                            sel_set = set(int(x) for x in st.tolist() if x >= 0)
                            mm = torch.tensor(
                                [int(n) not in sel_set for n in ni.tolist()],
                                dtype=torch.bool, device=device,
                            )

                        if not mm.any():
                            continue

                        m_ni = ni[mm]
                        m_cp = cp[mm]

                        kn = lk[b, m_ni, kv_h].to(cd)   # [|D|, K]
                        vn = lv[b, m_ni, kv_h].to(cd)   # [|D|, V]
                        ck = cos_c[m_cp]; sk = sin_c[m_cp]
                        kr = _apply_rope(kn, ck, sk)     # [|D|, K]

                        h0 = kv_h * G; h1 = h0 + G
                        qg = q[b, t, h0:h1].to(cd)       # [G, K]
                        cq = cos_c[rp]; sq = sin_c[rp]
                        qr = _apply_rope(qg, cq, sq) * float(scale)  # [G, K]

                        # Recompute scores & reconstruct P
                        S = qr @ kr.T                     # [G, |D|]
                        gm = gmax[b, t, h0:h1].to(cd)    # [G]
                        gs = gsum[b, t, h0:h1].to(cd)    # [G]
                        P = torch.exp(S - gm[:, None]) / gs[:, None]  # [G, |D|]

                        dO_g = dO[b, t, h0:h1]           # [G, V]
                        D_g = D[b, t, h0:h1]             # [G]

                        # dS = P * (dO @ V^T - D)
                        dP = dO_g @ vn.T                  # [G, |D|]
                        dS = P * (dP - D_g[:, None])      # [G, |D|]

                        # dQ̃ = dS @ K̃ → dq += σ R⁻¹(dQ̃, p_q)
                        dQt = dS @ kr                     # [G, K]
                        dq[b, t, h0:h1] += scale * _inverse_rope(dQt, cq, sq)

                        # dK̃ = dS^T @ Q̃ → dK += R⁻¹(dK̃, p_c)
                        dKt = dS.T @ qr                   # [|D|, K]
                        dK_pre = _inverse_rope(dKt, ck, sk)
                        dk_layers[li][b, :, kv_h].index_add_(0, m_ni, dK_pre)

                        # dV = P^T @ dO
                        dVc = P.T @ dO_g                  # [|D|, V]
                        dv_layers[li][b, :, kv_h].index_add_(0, m_ni, dVc)

        # Step 3: tree backward (top-down through mean-pooling)
        for li in range(NL - 1, 0, -1):
            parent_len = dk_layers[li].shape[1]
            child_len = dk_layers[li - 1].shape[1]
            for p in range(parent_len):
                s = p * CR
                e = min(s + CR, child_len)
                cnt = e - s
                if cnt > 0:
                    dk_layers[li - 1][:, s:e] += dk_layers[li][:, p:p+1] / cnt
                    dv_layers[li - 1][:, s:e] += dv_layers[li][:, p:p+1] / cnt

        return (
            dq.to(q.dtype),
            dk_layers[0].to(k.dtype),
            dv_layers[0].to(v.dtype),
            None, None, None, None, None,
        )


def htree_forward_naive_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Differentiable HTree forward (naive reference) with analytical backward."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return HTreeFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base,
    )


__all__ = [
    "HTREE_SCORE_NEG_INF",
    "HTREE_SCORE_VALID_THRESHOLD",
    "LayerTrace",
    "htree_forward_naive",
    "htree_forward_naive_autograd",
    "HTreeFunction",
]
