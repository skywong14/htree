from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


HTREE_SCORE_NEG_INF: float = -1.0e10
HTREE_SCORE_VALID_THRESHOLD: float = HTREE_SCORE_NEG_INF * 0.9


@dataclass
class LayerTrace:
    layer_idx: int
    selected_parents: Optional[torch.Tensor]
    layer_max: torch.Tensor
    layer_sum: torch.Tensor
    layer_output: torch.Tensor


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    y_lo = x_lo * cos - x_hi * sin
    y_hi = x_lo * sin + x_hi * cos
    return torch.cat([y_lo, y_hi], dim=-1)


def _inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """R^{-1}(x,p) = M_p^T x: cos/sin signs flipped vs forward."""
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    y_lo = x_lo * cos + x_hi * sin
    y_hi = -x_lo * sin + x_hi * cos
    return torch.cat([y_lo, y_hi], dim=-1)


def _build_tree(
    k: torch.Tensor, v: torch.Tensor, compression_rate: int, max_top_nodes: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    bsz, seq_len, h_kv, k_dim = k.shape
    v_dim = v.shape[-1]
    layers_k: List[torch.Tensor] = [k]
    layers_v: List[torch.Tensor] = [v]
    cur_k, cur_v, cur_len = k, v, seq_len
    while cur_len > max_top_nodes:
        nxt = (cur_len + compression_rate - 1) // compression_rate
        nk = torch.empty(bsz, nxt, h_kv, k_dim, dtype=k.dtype, device=k.device)
        nv = torch.empty(bsz, nxt, h_kv, v_dim, dtype=v.dtype, device=v.device)
        for p in range(nxt):
            s = p * compression_rate
            e = min(s + compression_rate, cur_len)
            nk[:, p] = cur_k[:, s:e].mean(dim=1)
            nv[:, p] = cur_v[:, s:e].mean(dim=1)
        layers_k.append(nk); layers_v.append(nv)
        cur_k, cur_v, cur_len = nk, nv, nxt
    return layers_k, layers_v


def _build_rope_cache(
    k_dim: int, cache_size: int, device: torch.device, rope_base: float,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, k_dim, 2, dtype=dtype, device=device) / k_dim))
    positions = torch.arange(cache_size, dtype=dtype, device=device)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _online_softmax_merge_group(
    cur_max: torch.Tensor, cur_sum: torch.Tensor, cur_output: torch.Tensor,
    scores: torch.Tensor, values: torch.Tensor,
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
    importance: torch.Tensor, node_indices: torch.Tensor, top_k: int, rightmost_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_candidates = int(importance.numel())
    imp = importance.clone()
    imp[rightmost_pos] = 1e6
    _, sorted_pos = torch.sort(imp, descending=True, stable=True)
    take = min(top_k, num_candidates)
    top_pos = sorted_pos[:take]
    selected_mask = torch.zeros(num_candidates, dtype=torch.bool, device=importance.device)
    if take > 0:
        selected_mask[top_pos] = True
    sel_nodes = node_indices[top_pos].to(torch.int32)
    sel_sorted, _ = torch.sort(sel_nodes)
    padded = torch.full([top_k], -1, dtype=torch.int32, device=importance.device)
    if take > 0:
        padded[:take] = sel_sorted
    return selected_mask, padded


def _init_prev_selected_parents(
    seq_len: int, bsz: int, h_kv: int, top_k: int,
    cr: int, num_layers: int, device: torch.device,
) -> torch.Tensor:
    tlp = cr ** (num_layers - 1)
    ti = torch.arange(seq_len, dtype=torch.int32, device=device)
    nvp = ti // tlp // cr + 1
    pc = torch.arange(top_k, dtype=torch.int32, device=device)[None, :].expand(seq_len, -1)
    vm = pc < nvp[:, None]
    prev = torch.where(vm, pc, torch.full_like(pc, -1))
    return prev[None, :, None, :].expand(bsz, seq_len, h_kv, top_k).contiguous()


def _expand_candidates(prev_sel, b, t, kv_h, cr, device):
    """Return (valid_parents, num_candidates, rightmost_child_idx, cand_pos, node_indices) or None."""
    pl = prev_sel[b, t, kv_h]
    vp = pl[pl >= 0]
    nvp = int(vp.numel())
    if nvp <= 0:
        return None
    return vp, nvp


# ===== Original forward (unchanged logic) =====

def htree_forward_naive(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    compression_rate: int = 16, max_top_nodes: int = 8192,
    top_k_per_layer: int = 512, scale: Optional[float] = None,
    rope_base: float = 10000.0, selection_mode: str = "parallel_importance",
    return_trace: bool = False,
):
    """Naive Python reference for HTree forward (GQA only)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v must be rank-4 tensors")
    bsz, seq_len, h, k_dim = q.shape
    h_kv = k.shape[2]; v_dim = v.shape[-1]
    assert h % h_kv == 0 and k_dim % 2 == 0
    assert k.shape[:2] == (bsz, seq_len) and v.shape[:2] == (bsz, seq_len)
    assert k.shape[2] == v.shape[2]
    assert max_top_nodes == top_k_per_layer * compression_rate
    if selection_mode == "exp_sum":
        selection_mode = "parallel_importance"
    assert selection_mode == "parallel_importance"

    G = h // h_kv
    if scale is None:
        scale = k_dim ** -0.5
    device = q.device; out_dtype = q.dtype
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()

    layers_k, layers_v = _build_tree(k, v, compression_rate, max_top_nodes)
    NL = len(layers_k)
    cs = max_top_nodes + 1024
    cos_cache, sin_cache = _build_rope_cache(k_dim, cs, device, rope_base)

    g_max = torch.full([bsz, seq_len, h], HTREE_SCORE_NEG_INF, dtype=torch.float32, device=device)
    g_sum = torch.zeros([bsz, seq_len, h], dtype=torch.float32, device=device)
    g_out = torch.zeros([bsz, seq_len, h, v_dim], dtype=torch.float32, device=device)

    psp = _init_prev_selected_parents(seq_len, bsz, h_kv, top_k_per_layer, compression_rate, NL, device)
    traces: Dict[int, LayerTrace] = {}

    for li in range(NL - 1, -1, -1):
        lk = layers_k[li]; lv = layers_v[li]; nl = lk.shape[1]
        lp = compression_rate ** li; ib = li == 0

        l_max = torch.full([bsz, seq_len, h], HTREE_SCORE_NEG_INF, dtype=torch.float32, device=device)
        l_sum = torch.zeros([bsz, seq_len, h], dtype=torch.float32, device=device)
        l_out = torch.zeros([bsz, seq_len, h, v_dim], dtype=torch.float32, device=device)
        nsp = None if ib else torch.full_like(psp, -1)

        for b in range(bsz):
            for t in range(seq_len):
                ri = t // lp; rci = ri % compression_rate
                for kv_h in range(h_kv):
                    pl = psp[b, t, kv_h]; vp = pl[pl >= 0]; nvp = int(vp.numel())
                    if nvp <= 0:
                        raise RuntimeError(f"No valid parents at layer={li}, b={b}, t={t}, kv_h={kv_h}")
                    nc = (nvp - 1) * compression_rate + rci + 1; rp = nc - 1
                    cp = torch.arange(nc, device=device, dtype=torch.long)
                    po = cp // compression_rate; cs_ = cp - po * compression_rate
                    pidx = vp[po].long(); ni = pidx * compression_rate + cs_
                    assert not ((ni < 0).any() or (ni >= nl).any())

                    kn = lk[b, ni, kv_h].to(torch.float32)
                    vn = lv[b, ni, kv_h].to(torch.float32)
                    kr = _apply_rope(kn, cos_cache[cp], sin_cache[cp])
                    h0 = kv_h * G; h1 = h0 + G
                    qg = q[b, t, h0:h1].to(torch.float32)
                    qr = _apply_rope(qg, cos_cache[rp], sin_cache[rp]) * float(scale)
                    scores = qr @ kr.T

                    if ib:
                        mm = torch.ones(nc, dtype=torch.bool, device=device)
                    else:
                        lm = scores.max()
                        imp = torch.exp(scores - lm).sum(dim=0)
                        sm, sn = _select_shared_topk(imp, ni, top_k_per_layer, rp)
                        nsp[b, t, kv_h] = sn; mm = ~sm

                    if mm.any():
                        cm, csm, co = _online_softmax_merge_group(
                            l_max[b, t, h0:h1], l_sum[b, t, h0:h1], l_out[b, t, h0:h1],
                            scores[:, mm], vn[mm])
                        l_max[b, t, h0:h1] = cm; l_sum[b, t, h0:h1] = csm; l_out[b, t, h0:h1] = co

        ch = l_sum > 1e-10; gh = g_sum > 1e-10
        nm = torch.maximum(g_max, l_max)
        sg = torch.where(gh, torch.exp(g_max - nm), torch.zeros_like(g_sum))
        sc = torch.where(ch, torch.exp(l_max - nm), torch.zeros_like(l_sum))
        ms = g_sum * sg + l_sum * sc
        mo = g_out * sg[..., None] + l_out * sc[..., None]
        g_max = torch.where(ch, torch.where(gh, nm, l_max), g_max)
        g_sum = torch.where(ch, torch.where(gh, ms, l_sum), g_sum)
        g_out = torch.where(ch[..., None], torch.where(gh[..., None], mo, l_out), g_out)

        if return_trace:
            traces[li] = LayerTrace(li, None if ib else nsp.clone(),
                                    l_max.clone(), l_sum.clone(), l_out.clone())
        if not ib:
            psp = nsp

    output = (g_out / g_sum[..., None]).to(out_dtype)
    if not return_trace:
        return output
    return output, {"num_layers": NL, "num_groups": G, "selection_mode": selection_mode, "layers": traces}


# ===== Autograd Function with analytical backward =====

class HTreeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base):
        B, T, H, K = q.shape; H_kv = k.shape[2]; V = v.shape[-1]; G = H // H_kv
        device = q.device
        cd = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
        CR = compression_rate; TK = top_k_per_layer

        layers_k, layers_v = _build_tree(k, v, CR, max_top_nodes)
        NL = len(layers_k)
        cos_c, sin_c = _build_rope_cache(K, max_top_nodes + 1024, device, rope_base, dtype=cd)
        g_max = torch.full([B, T, H], HTREE_SCORE_NEG_INF, dtype=cd, device=device)
        g_sum = torch.zeros([B, T, H], dtype=cd, device=device)
        g_out = torch.zeros([B, T, H, V], dtype=cd, device=device)
        psp = _init_prev_selected_parents(T, B, H_kv, TK, CR, NL, device)
        plp: Dict[int, torch.Tensor] = {}

        for li in range(NL - 1, -1, -1):
            plp[li] = psp
            lk = layers_k[li]; lv = layers_v[li]; nl = lk.shape[1]
            lp = CR ** li; ib = li == 0
            l_max = torch.full([B, T, H], HTREE_SCORE_NEG_INF, dtype=cd, device=device)
            l_sum = torch.zeros([B, T, H], dtype=cd, device=device)
            l_out = torch.zeros([B, T, H, V], dtype=cd, device=device)
            nsp = None if ib else torch.full_like(psp, -1)

            for b in range(B):
                for t in range(T):
                    ri = t // lp; rci = ri % CR
                    for kv_h in range(H_kv):
                        pl = psp[b, t, kv_h]; vp = pl[pl >= 0]; nvp = int(vp.numel())
                        if nvp <= 0:
                            continue
                        nc = (nvp - 1) * CR + rci + 1; rp = nc - 1
                        cp = torch.arange(nc, device=device, dtype=torch.long)
                        po = cp // CR; cs = cp - po * CR
                        pidx = vp[po].long(); ni = pidx * CR + cs
                        kn = lk[b, ni, kv_h].to(cd); vn = lv[b, ni, kv_h].to(cd)
                        kr = _apply_rope(kn, cos_c[cp], sin_c[cp])
                        h0 = kv_h * G; h1 = h0 + G
                        qg = q[b, t, h0:h1].to(cd)
                        qr = _apply_rope(qg, cos_c[rp], sin_c[rp]) * float(scale)
                        scores = qr @ kr.T

                        if ib:
                            mm = torch.ones(nc, dtype=torch.bool, device=device)
                        else:
                            lm = scores.max()
                            imp = torch.exp(scores - lm).sum(dim=0)
                            sm, sn = _select_shared_topk(imp, ni, TK, rp)
                            nsp[b, t, kv_h] = sn; mm = ~sm

                        if mm.any():
                            cm, csm, co = _online_softmax_merge_group(
                                l_max[b, t, h0:h1], l_sum[b, t, h0:h1], l_out[b, t, h0:h1],
                                scores[:, mm], vn[mm])
                            l_max[b, t, h0:h1] = cm; l_sum[b, t, h0:h1] = csm; l_out[b, t, h0:h1] = co

            ch = l_sum > 1e-10; gh = g_sum > 1e-10
            nm = torch.maximum(g_max, l_max)
            sg = torch.where(gh, torch.exp(g_max - nm), torch.zeros_like(g_sum))
            sc = torch.where(ch, torch.exp(l_max - nm), torch.zeros_like(l_sum))
            ms = g_sum * sg + l_sum * sc; mo = g_out * sg[..., None] + l_out * sc[..., None]
            g_max = torch.where(ch, torch.where(gh, nm, l_max), g_max)
            g_sum = torch.where(ch, torch.where(gh, ms, l_sum), g_sum)
            g_out = torch.where(ch[..., None], torch.where(gh[..., None], mo, l_out), g_out)
            if not ib:
                psp = nsp

        output = (g_out / g_sum[..., None]).to(q.dtype)

        ctx.save_for_backward(q, k, v, output)
        ctx.global_max = g_max.detach(); ctx.global_sum = g_sum.detach()
        ctx.cos_cache = cos_c; ctx.sin_cache = sin_c
        ctx.layers_k = [x.detach() for x in layers_k]
        ctx.layers_v = [x.detach() for x in layers_v]
        ctx.plp = {l: p.detach() for l, p in plp.items()}
        ctx.NL = NL; ctx.CR = CR; ctx.scale = scale; ctx.TK = TK
        return output

    @staticmethod
    def backward(ctx, dO):
        q, k, v, output = ctx.saved_tensors
        gmax = ctx.global_max; gsum = ctx.global_sum
        cos_c = ctx.cos_cache; sin_c = ctx.sin_cache
        layers_k = ctx.layers_k; layers_v = ctx.layers_v
        plp = ctx.plp; NL = ctx.NL; CR = ctx.CR; scale = ctx.scale; TK = ctx.TK

        B, T, H, K = q.shape; H_kv = k.shape[2]; V = v.shape[-1]; G = H // H_kv
        device = q.device
        cd = q.dtype if q.dtype in (torch.float32, torch.float64) else torch.float32
        dO = dO.to(cd); out_f = output.to(cd)

        D = (dO * out_f).sum(dim=-1)  # [B, T, H]
        dq = torch.zeros(B, T, H, K, dtype=cd, device=device)
        dk_l = [torch.zeros_like(x, dtype=cd) for x in layers_k]
        dv_l = [torch.zeros_like(x, dtype=cd) for x in layers_v]

        for li in range(NL - 1, -1, -1):
            lk = layers_k[li]; lv = layers_v[li]
            lp = CR ** li; ib = li == 0; psel = plp[li]
            sel_next = None if ib else plp[li - 1]

            for b in range(B):
                for t in range(T):
                    ri = t // lp; rci = ri % CR
                    for kv_h in range(H_kv):
                        pl = psel[b, t, kv_h]; vp = pl[pl >= 0]; nvp = int(vp.numel())
                        if nvp <= 0:
                            continue
                        nc = (nvp - 1) * CR + rci + 1; rp = nc - 1
                        cp = torch.arange(nc, device=device, dtype=torch.long)
                        po = cp // CR; cs = cp - po * CR
                        pidx = vp[po].long(); ni = pidx * CR + cs

                        if ib:
                            mm = torch.ones(nc, dtype=torch.bool, device=device)
                        else:
                            st = sel_next[b, t, kv_h]
                            ss = set(int(x) for x in st.tolist() if x >= 0)
                            mm = torch.tensor([int(n) not in ss for n in ni.tolist()],
                                              dtype=torch.bool, device=device)
                        if not mm.any():
                            continue

                        m_ni = ni[mm]; m_cp = cp[mm]
                        kn = lk[b, m_ni, kv_h].to(cd)
                        vn = lv[b, m_ni, kv_h].to(cd)
                        ck = cos_c[m_cp]; sk = sin_c[m_cp]
                        kr = _apply_rope(kn, ck, sk)

                        h0 = kv_h * G; h1 = h0 + G
                        qg = q[b, t, h0:h1].to(cd)
                        cq = cos_c[rp]; sq = sin_c[rp]
                        qr = _apply_rope(qg, cq, sq) * float(scale)

                        S = qr @ kr.T  # [G, |D|]
                        gm = gmax[b, t, h0:h1].to(cd)
                        gs = gsum[b, t, h0:h1].to(cd)
                        P = torch.exp(S - gm[:, None]) / gs[:, None]

                        dO_g = dO[b, t, h0:h1]; D_g = D[b, t, h0:h1]
                        dS = P * (dO_g @ vn.T - D_g[:, None])

                        dq[b, t, h0:h1] += scale * _inverse_rope(dS @ kr, cq, sq)
                        dk_l[li][b, :, kv_h].index_add_(0, m_ni, _inverse_rope(dS.T @ qr, ck, sk))
                        dv_l[li][b, :, kv_h].index_add_(0, m_ni, P.T @ dO_g)

        for li in range(NL - 1, 0, -1):
            plen = dk_l[li].shape[1]; clen = dk_l[li - 1].shape[1]
            for p in range(plen):
                s = p * CR; e = min(s + CR, clen); cnt = e - s
                if cnt > 0:
                    dk_l[li - 1][:, s:e] += dk_l[li][:, p:p+1] / cnt
                    dv_l[li - 1][:, s:e] += dv_l[li][:, p:p+1] / cnt

        return dq.to(q.dtype), dk_l[0].to(k.dtype), dv_l[0].to(v.dtype), None, None, None, None, None


def htree_forward_naive_autograd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    compression_rate: int = 16, max_top_nodes: int = 8192,
    top_k_per_layer: int = 512, scale: Optional[float] = None,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Differentiable HTree forward with analytical backward."""
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
