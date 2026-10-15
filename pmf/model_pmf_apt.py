"""Adaptive-token pixel MeanFlow (pMF): the released pmfDiT made token-adaptive with the BudgetPix method.

Wraps `models.pmfDiT.pmfDiT` (B/16, L/16, H/16 at 256^2):

  in   a token of side s = k * base covers a k x k block of base cells. A merged token is the MEAN of its k*k cell
       embeddings plus a zero-init mixer. Its learned position is the box-average of the cells it covers; its RoPE
       angle is taken at the box centre.
  mid  the blocks run on [prefix ; image tokens], padded to a common length, with padded keys masked out.
  out  each branch's final layer emits a base x base x 3 patch per token (channel-last); for a merged token it is
       bilinearly upsampled to s x s and a zero-init per-scale refine head adds a residual conditioned on the token's
       final hidden state. u = (z - x_hat_u) / max(t, 0.05), v = (z - x_hat_v) / max(t, 0.05).

On the dense layout (all base tokens) this is the released model at init: every added output is zero-initialised.
The layout L is constant for every call on an image (it carries no tangent in the JVP).
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from heads import PixelDiTRefineHead            # noqa: E402
from layout import Geometry, all_base_tokens, pack   # noqa: E402


class ScaleMixer(nn.Module):
    """mean(k*k cell embeddings) + ZERO-INIT(depthwise k*k -> pointwise)."""

    def __init__(self, dim, k, bott=256):
        super().__init__()
        self.k = k
        self.pos = nn.Parameter(torch.zeros(1, dim, k, k))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.down = nn.Conv2d(dim, bott, 1)
        self.dw = nn.Conv2d(bott, bott, k, groups=bott)
        self.up = nn.Linear(bott, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, sub):                                          # (M, dim, k, k)
        h = self.dw(self.down(sub + self.pos.to(sub.dtype))).flatten(1)
        return sub.mean(dim=(2, 3)) + self.up(h)


def _rope_complex(coords, head_dim, theta=10000.0):
    """2D RoPE evaluated at arbitrary (row, col) positions. coords: (B,N,2) -> (B,N,head_dim/2) complex."""
    dim = head_dim // 2
    dt = torch.float64 if coords.dtype == torch.float64 else torch.float32
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dt, device=coords.device) / dim))   # (dim/2,)
    ang = torch.cat([coords[..., 0:1].to(dt) * freqs, coords[..., 1:2].to(dt) * freqs], dim=-1)   # (B,N,dim)
    return torch.polar(torch.ones_like(ang), ang)


def _apply_rope(x, rope_c, n_img):
    """x: (B,S,H,d); rotate the last n_img tokens (image), leave the prefix unrotated."""
    xf = x if x.dtype == torch.float64 else x.to(torch.float32)
    xc = torch.view_as_complex(xf.reshape(*xf.shape[:-1], -1, 2).contiguous())            # (B,S,H,d/2)
    rot = xc[:, -n_img:] * rope_c[:, :, None, :]                                           # (B,N,H,d/2)
    out = torch.cat([xc[:, :-n_img], rot], dim=1)
    return torch.view_as_real(out).flatten(-2).to(x.dtype)


def _block_masked(block, x, rope_c, bias, n_img):
    """TransformerBlock.forward, with a key-padding bias and per-token RoPE."""
    attn = block.attn
    B, S, D = x.shape
    H, d = attn.num_heads, attn.head_dim
    y = block.norm1(x)
    q = attn.q_norm(attn.q_proj(y).reshape(B, S, H, d))
    k = attn.k_norm(attn.k_proj(y).reshape(B, S, H, d))
    v = attn.v_proj(y).reshape(B, S, H, d)
    q = _apply_rope(q, rope_c, n_img)
    k = _apply_rope(k, rope_c, n_img)
    w = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(d), k)
    w = w if w.dtype == torch.float64 else w.float()
    w = F.softmax(w + bias.to(w.dtype), dim=-1)
    a = torch.einsum("bhqk,bkhd->bqhd", w.to(v.dtype), v).reshape(B, S, D)
    x = x + attn.out_proj(a) * block.attn_scale
    x = x + block.mlp(block.norm2(x)) * block.mlp_scale
    return x


class AdaptivePMF(nn.Module):
    def __init__(self, net, geo: Geometry, mix_bott: int = 256, head_width: int = 64, head_depth: int = 2,
                 head_patch: int = 4, head_heads: int = 4):
        super().__init__()
        self.net = net
        self.geo = geo
        self.dim = net.hidden_size
        self.base = geo.base_patch
        if net.patch_size != geo.base_patch or net.input_size != geo.img:
            raise ValueError(f"geometry {geo} does not match the network (patch {net.patch_size}, input {net.input_size})")
        self.P = net.prefix_tokens
        self.g = net.x_embedder.grid_size[0]
        coarse = [s for s in geo.scales if s != geo.base_patch]
        head = lambda s: PixelDiTRefineHead(self.dim, self.base, s, 3, head_width, head_depth, head_patch, head_heads)
        self.mixers = nn.ModuleDict({str(s): ScaleMixer(self.dim, s // self.base, mix_bott) for s in coarse})
        self.refines_u = nn.ModuleDict({str(s): head(s) for s in coarse})
        self.has_v = len(net.v_heads) > 0
        if self.has_v:
            self.refines_v = nn.ModuleDict({str(s): head(s) for s in coarse})

    # ------------------------------------------------------------------ tokens in
    def _gather_cells(self, grid, L, s):
        """grid (B,D,g,g) -> for every token of size s: its (M,D,k,k) block of base cells, plus the (bi, ti) index."""
        m = L["sizes"] == s
        bi, ti = m.nonzero(as_tuple=True)
        k = s // self.base
        gy = L["top_left"][bi, ti, 0] // self.base
        gx = L["top_left"][bi, ti, 1] // self.base
        o = torch.arange(k, device=grid.device)
        yy = (gy[:, None] + o[None, :])[:, :, None]
        xx = (gx[:, None] + o[None, :])[:, None, :]
        sub = grid[bi[:, None, None], :, yy, xx]                        # (M,k,k,D)
        return sub.permute(0, 3, 1, 2), bi, ti

    def embed_tokens(self, z, L):
        B, N = L["sizes"].shape
        emb = self.net.x_embedder
        cells = emb.proj2(emb.proj1(z))                                 # (B,D,g,g): embedding of every base cell
        pos = self.net.pos_embed[:, self.P:, :].reshape(1, self.g, self.g, self.dim).permute(0, 3, 1, 2)   # (1,D,g,g)
        out = z.new_zeros(B, N, self.dim, dtype=cells.dtype)
        for s in self.geo.scales:
            if not bool((L["sizes"] == s).any()):
                continue
            sub, bi, ti = self._gather_cells(cells, L, s)
            psub, _, _ = self._gather_cells(pos.expand(B, -1, -1, -1), L, s)
            if s == self.base:
                tok = sub[:, :, 0, 0]
            else:
                tok = self.mixers[str(s)](sub.contiguous())
            out[bi, ti] = tok + psub.mean(dim=(2, 3)).to(tok.dtype)     # box-averaged learned position
        return out

    def _prefix(self, h, w, t_min, t_max, y):
        """The conditioning tokens + their positions."""
        net = self.net
        u = lambda a: a.unsqueeze(1)
        seq = torch.cat([net.class_tokens + u(net.y_embedder(y)),
                         net.omega_tokens + u(net.omega_embedder(1 - 1 / w)),
                         net.t_min_tokens + u(net.cfg_t_start_embedder(t_min)),
                         net.t_max_tokens + u(net.cfg_t_end_embedder(t_max)),
                         net.time_tokens + u(net.h_embedder(h))], dim=1)
        return seq + net.pos_embed[:, :self.P, :]

    # ------------------------------------------------------------------ pixels out
    def assemble(self, base_out, hid, L, refines):
        """(B,N,b*b*3) per-token predictions -> image (B,3,H,W). Merged tokens: bilinear up + zero-init refine residual."""
        B, N = L["sizes"].shape
        b = self.base
        img = base_out.new_zeros(B, 3, self.geo.img, self.geo.img)
        for s in self.geo.scales:
            m = L["sizes"] == s
            if not bool(m.any()):
                continue
            bi, ti = m.nonzero(as_tuple=True)
            coarse = base_out[bi, ti].view(-1, b, b, 3).permute(0, 3, 1, 2)          # channel-last (p,q,c)
            if s == b:
                px = coarse
            else:
                up = F.interpolate(coarse, size=(s, s), mode="bilinear", align_corners=False)
                px = up + refines[str(s)](coarse, hid[bi, ti])
            ys, xs = L["top_left"][bi, ti, 0], L["top_left"][bi, ti, 1]
            o = torch.arange(s, device=img.device)
            iy = (ys[:, None] + o[None, :])[:, :, None].expand(-1, -1, s)
            ix = (xs[:, None] + o[None, :])[:, None, :].expand(-1, s, -1)
            ib = bi[:, None, None].expand(-1, s, s)
            for c in range(3):
                img.index_put_((ib, torch.full_like(ib, c), iy, ix), px[:, c].to(img.dtype), accumulate=True)
        return img

    # ------------------------------------------------------------------ forward
    def forward(self, z, t, h, w, t_min, t_max, y, L):
        """Same signature and outputs as pmfDiT.forward, plus the layout L (from layout.pack). Returns (u, v)."""
        net = self.net
        B = z.shape[0]
        n_img = L["sizes"].shape[1]
        seq = torch.cat([self._prefix(h, w, t_min, t_max, y), self.embed_tokens(z, L)], dim=1)
        rope_c = _rope_complex(L["coords"].to(torch.float64) if z.dtype == torch.float64 else L["coords"], net.head_dim)
        key_ok = torch.cat([torch.ones(B, self.P, dtype=torch.bool, device=z.device), L["valid"]], dim=1)
        bias = torch.zeros(B, 1, 1, key_ok.shape[1], device=z.device).masked_fill(~key_ok[:, None, None, :], float("-inf"))
        for blk in net.shared_blocks:
            seq = _block_masked(blk, seq, rope_c, bias, n_img)
        u_seq = seq
        for blk in net.u_heads:
            u_seq = _block_masked(blk, u_seq, rope_c, bias, n_img)
        u_tok = u_seq[:, self.P:]
        x_u = self.assemble(net.u_final_layer(u_tok), u_tok, L, self.refines_u)
        if self.has_v:
            v_seq = seq
            for blk in net.v_heads:
                v_seq = _block_masked(blk, v_seq, rope_c, bias, n_img)
            v_tok = v_seq[:, self.P:]
            x_v = self.assemble(net.v_final_layer(v_tok), v_tok, L, self.refines_v)
        else:
            x_v = torch.zeros_like(x_u)
        tt = torch.clamp(t.reshape(B, 1, 1, 1), min=0.05)
        return (z - x_u) / tt, (z - x_v) / tt


# ---------------------------------------------------------------------------------------------- helpers
def pmf_geometry(net):
    """Base patch = the model's patch, scales (p, 2p, 4p), thresholds 6.0 / 4.0 bits."""
    p = net.patch_size
    return Geometry(img=net.input_size, base_patch=p, scales=(p, 2 * p, 4 * p), thresholds=(6.0, 4.0))


def dense_L(B, geo, device):
    return pack([all_base_tokens(geo)] * B, geo, device, multiple=8)


def cat_L(*Ls):
    """Concatenate packed layouts along the batch (same padded length required)."""
    out = {}
    for k, v in Ls[0].items():
        out[k] = torch.cat([L[k] for L in Ls], dim=0) if torch.is_tensor(v) else v
    return out


def new_param_names(model):
    return [n for n, _ in model.named_parameters() if not n.startswith("net.")]
