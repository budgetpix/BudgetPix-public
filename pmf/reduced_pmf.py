"""Training-free token reduction inside the released pMF trunk: ToMe and FeatSim.

  policy="tome"     Token Merging (Bolya et al., arXiv 2210.09461): in every block of the span, between attention and the
                    MLP, bipartite soft matching over the attention keys merges r tokens into their most similar partner;
                    proportional attention (logits + log size); constant r per block, sized so the target is reached
                    by the end of the span, and held there.
  policy="featsim"  RTI's similarity-merging arm (arXiv 2608.29281): one cut to R tokens after a short dense prelude,
                    the core runs on R tokens, then every merged token is written back to all patches it owns.

  * the 20 in-context conditioning tokens (class / omega / CFG interval / h) are never merged;
  * the u-path is shared_blocks followed by u_heads, treated as one trunk;
  * RoPE is on image tokens only; a surviving token keeps the (row, col) of its own patch;
  * matching metric = keys after qk-norm and before RoPE, averaged over heads;
  * spans: ToMe blocks 0 .. depth-3, FeatSim blocks 3 .. depth-3; both keep a 2-block dense coda, and u_final_layer
    decodes the full grid.
R = image tokens the core runs on. At the dense budget both are the identity."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_pmf_apt import AdaptivePMF, pmf_geometry, dense_L, _rope_complex


def bipartite_soft_matching(metric, r):
    """ToMe's bipartite soft matching."""
    with torch.no_grad():
        m = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = m[:, ::2, :], m[:, 1::2, :]
        scores = a @ b.transpose(-1, -2)
        node_max, node_idx = scores.max(dim=-1)
        edge = node_max.argsort(dim=-1, descending=True)[..., None]
        src_idx, unm_idx = edge[:, :r, :], edge[:, r:, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    return src_idx, dst_idx, unm_idx


def _apply_merge(x, src_idx, dst_idx, unm_idx, r, reduce="sum"):
    """Apply one bipartite merge (ToMe)."""
    n, _, c = x.shape
    src, dst = x[:, ::2, :], x[:, 1::2, :]
    unm = src.gather(dim=-2, index=unm_idx.expand(n, unm_idx.shape[1], c))
    mrg = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), mrg, reduce=reduce)
    return torch.cat([unm, dst], dim=1)


def _rot(x, rope_c):
    """RoPE on image tokens: x (B,N,H,d), rope_c (B,N,d/2) complex."""
    xf = x if x.dtype == torch.float64 else x.to(torch.float32)
    xc = torch.view_as_complex(xf.reshape(*xf.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(xc * rope_c[:, :, None, :]).flatten(-2).to(x.dtype)


class ReducedPMF(nn.Module):
    def __init__(self, net, policy="tome", budget=None, prop_attn=True, prelude=3, coda=2):
        super().__init__()
        assert policy in ("tome", "featsim"), policy
        self.policy, self.prop_attn = policy, bool(prop_attn)
        self.wrap = AdaptivePMF(net, pmf_geometry(net))
        self.wrap.has_v = False
        self.net, self.geo = net, self.wrap.geo
        self.blocks = list(net.shared_blocks) + list(net.u_heads)
        self.depth = len(self.blocks)
        self.n_img = self.geo.n_dense
        lo = 0 if policy == "tome" else prelude
        self.span = (lo, max(lo, self.depth - 1 - coda))
        self.set_budget(self.n_img if budget is None else budget)

    def set_budget(self, budget):
        """ToMe's constant r = ceil(drop / span length)."""
        self.budget = int(min(max(int(budget), 1), self.n_img))
        n_blocks = self.span[1] - self.span[0] + 1
        drop = self.n_img - self.budget
        self.r_per_block = 0 if drop <= 0 else max(1, -(-drop // n_blocks))
        return self.budget

    def _attn(self, blk, x, P, coords, size):
        """TransformerBlock attention, returning the pre-RoPE key metric."""
        attn = blk.attn
        B, S, D = x.shape
        H, d = attn.num_heads, attn.head_dim
        y = blk.norm1(x)
        q = attn.q_norm(attn.q_proj(y).reshape(B, S, H, d))
        k = attn.k_norm(attn.k_proj(y).reshape(B, S, H, d))
        v = attn.v_proj(y).reshape(B, S, H, d)
        metric = k[:, P:].mean(dim=2)                              # image keys BEFORE RoPE, mean over heads: (B, N, d)
        rope_c = _rope_complex(coords, d)
        q = torch.cat([q[:, :P], _rot(q[:, P:], rope_c)], dim=1)
        k = torch.cat([k[:, :P], _rot(k[:, P:], rope_c)], dim=1)
        w = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(d), k).float()
        if size is not None and self.prop_attn:                     # proportional attention on image keys
            bias = torch.cat([torch.zeros(B, P, device=x.device), size.clamp_min(1.0).log()], dim=1)
            w = w + bias[:, None, None, :]
        w = F.softmax(w, dim=-1)
        a = torch.einsum("bhqk,bkhd->bqhd", w.to(v.dtype), v).reshape(B, S, D)
        return x + attn.out_proj(a) * blk.attn_scale, metric

    @staticmethod
    def _mlp(blk, x):
        return x + blk.mlp(blk.norm2(x)) * blk.mlp_scale

    def forward(self, z, t, h, w, t_min, t_max, y, L=None):
        """Same signature as AdaptivePMF.forward; the layout argument is ignored. Returns (u, zeros)."""
        wrap = self.wrap
        B, dev = z.shape[0], z.device
        Ld = dense_L(B, self.geo, dev)
        prefix = wrap._prefix(h, w, t_min, t_max, y)
        P = prefix.shape[1]
        x = torch.cat([prefix, wrap.embed_tokens(z, Ld)], dim=1)
        coords_all = Ld["coords"]                                   # (B, 256, 2) patch centres, base units
        pos = torch.arange(self.n_img, device=dev).unsqueeze(0).expand(B, -1).clone()
        owner = pos.clone()
        size = torch.ones(B, self.n_img, device=dev)
        lo, hi = self.span
        target = self.budget
        for i, blk in enumerate(self.blocks):
            coords = coords_all.gather(1, pos.unsqueeze(-1).expand(-1, -1, 2))
            x, key = self._attn(blk, x, P, coords, size if self.prop_attn else None)
            n_now = x.shape[1] - P
            r = 0
            if lo <= i <= hi and n_now > target:
                if self.policy == "tome":
                    r = min(self.r_per_block, n_now - target, n_now // 2)
                else:
                    r = (n_now - target) if i == lo else 0
            if r > 0:
                img_x, img_key = x[:, P:], key
                while r > 0:
                    step_r = min(r, img_x.shape[1] // 2)
                    src_idx, dst_idx, unm_idx = bipartite_soft_matching(img_key, step_r)
                    img_x, size, pos, owner = _merge_step(img_x, size, pos, owner, src_idx, dst_idx, unm_idx, step_r)
                    img_key = _apply_merge(img_key, src_idx, dst_idx, unm_idx, step_r, reduce="mean")
                    r -= step_r
                x = torch.cat([x[:, :P], img_x], dim=1)
            x = self._mlp(blk, x)
            if i == hi and x.shape[1] - P < self.n_img:            # write back: the coda runs on the full grid
                img = x[:, P:].gather(1, owner.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                x = torch.cat([x[:, :P], img], dim=1)
                pos = torch.arange(self.n_img, device=dev).unsqueeze(0).expand(B, -1).clone()
                owner = pos.clone()
                size = torch.ones(B, self.n_img, device=dev)
        tok = x[:, P:]
        tok = tok.gather(1, owner.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        x_u = wrap.assemble(self.net.u_final_layer(tok), tok, Ld, wrap.refines_u)
        tt = torch.clamp(t.reshape(B, 1, 1, 1), min=0.05)
        u = (z - x_u) / tt
        return u, torch.zeros_like(u)


def _merge_step(x, size, pos, owner, src_idx, dst_idx, unm_idx, r):
    """One merge of tokens, sizes, positions and owners."""
    n, N, c = x.shape
    xs = _apply_merge(x * size.unsqueeze(-1), src_idx, dst_idx, unm_idx, r, reduce="sum")
    sz = _apply_merge(size.unsqueeze(-1), src_idx, dst_idx, unm_idx, r, reduce="sum").squeeze(-1)
    x = xs / sz.unsqueeze(-1)
    n_a = (N + 1) // 2
    n_unm = unm_idx.shape[1]
    new_of_a = x.new_full((n, n_a), -1, dtype=torch.long)
    ar = torch.arange(n_unm, device=x.device).unsqueeze(0).expand(n, -1)
    new_of_a.scatter_(1, unm_idx.squeeze(-1), ar)
    new_of_a.scatter_(1, src_idx.squeeze(-1), n_unm + dst_idx.squeeze(-1))
    new_of_b = n_unm + torch.arange(N // 2, device=x.device).unsqueeze(0).expand(n, -1)
    new_of_old = torch.empty(n, N, dtype=torch.long, device=x.device)
    new_of_old[:, ::2] = new_of_a
    new_of_old[:, 1::2] = new_of_b
    owner = new_of_old.gather(1, owner)
    pos_a, pos_b = pos[:, ::2], pos[:, 1::2]
    pos = torch.cat([pos_a.gather(1, unm_idx.squeeze(-1)), pos_b], dim=1)
    return x, sz, pos, owner
