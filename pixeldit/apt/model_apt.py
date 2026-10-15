"""AdaptivePixDiT: PixelDiT-T2I with a token-adaptive patch stream.

The patch stage runs over adaptive tokens (merged low-entropy regions); the pixel stage always runs
over every base cell. Two zero-initialised modules are added: ScaleMixer (merged-token embedding)
and CellRefine (per-cell correction to a merged token's conditioning vector). At the dense layout
the forward equals the base model's.
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .layout import Geometry


def analytic_freqs_cis_2d(dim, centres, Hs, Ws, theta=10000.0, scale=16.0):
    """2-D RoPE at fractional token centres, linearly interpolated on the grid linspace.

    centres: (..., 2) float in base-cell units.  Returns (..., dim//2) complex.
    """
    dev = centres.device

    def axis(pos, n):
        tab = torch.linspace(0, scale, n, device=dev)
        lo = pos.floor().clamp(0, n - 1).long()
        hi = (lo + 1).clamp(max=n - 1)
        w = (pos - lo.to(pos.dtype))
        return tab[lo] * (1 - w) + tab[hi] * w

    y_pos = axis(centres[..., 0].float(), Hs)
    x_pos = axis(centres[..., 1].float(), Ws)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4, device=dev)[: dim // 4].float() / dim))
    x_f = x_pos[..., None] * freqs                      # (..., dim//4)
    y_f = y_pos[..., None] * freqs
    x_cis = torch.polar(torch.ones_like(x_f), x_f)
    y_cis = torch.polar(torch.ones_like(y_f), y_f)
    out = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)   # (..., dim//4, 2)
    return out.reshape(*centres.shape[:-1], -1)                            # (..., dim//2)


def apply_rope_batched(q, k, freqs_cis):
    """Rotary embedding with per-image positions. q/k: (B,N,H,D); freqs_cis: (B,N,D/2)."""
    f = freqs_cis[:, :, None, :]
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    return (torch.view_as_real(q_ * f).flatten(3).type_as(q),
            torch.view_as_real(k_ * f).flatten(3).type_as(k))


class ScaleMixer(nn.Module):
    """Merged-token embedding: mean of k*k sub-patch embeddings + zero-init depthwise mix."""

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

    def forward(self, sub):                                   # sub: (M, dim, k, k)
        m = sub.mean(dim=(2, 3))
        h = self.dw(self.down(sub + self.pos.to(sub.dtype))).flatten(1)
        return m + self.up(h)


class RefineBlock(nn.Module):
    """adaLN-modulated (zero-init) transformer block over a merged token's k*k cell slots."""

    def __init__(self, width, heads, cond_dim):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.ada = nn.Linear(cond_dim, 6 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, z, c):                                  # z: (M,P,w)  c: (M,cond)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        M, P, w = z.shape
        h = self.n1(z) * (1 + sc1[:, None]) + sh1[:, None]
        qkv = self.qkv(h).reshape(M, P, 3, self.h, w // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        z = z + g1[:, None] * self.proj(a.transpose(1, 2).reshape(M, P, w))
        h = self.n2(z) * (1 + sc2[:, None]) + sh2[:, None]
        return z + g2[:, None] * self.mlp(h)


class CellRefine(nn.Module):
    """Zero-init per-cell correction to the conditioning vector a merged token broadcasts to its
    k*k base cells."""

    def __init__(self, dim, k, width=64, depth=2, heads=4):
        super().__init__()
        self.k = k
        self.slots = nn.Parameter(torch.zeros(1, k * k, width))
        nn.init.trunc_normal_(self.slots, std=0.02)
        self.blocks = nn.ModuleList([RefineBlock(width, heads, dim) for _ in range(depth)])
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):                                     # h: (M, dim) -> (M, k*k, dim)
        z = self.slots.expand(h.shape[0], -1, -1).to(h.dtype)
        for b in self.blocks:
            z = b(z, h)
        return self.out(z)


class AdaptivePixDiT(nn.Module):
    """Wrap a `PixDiT_T2I` and make its patch stream token-adaptive."""

    def __init__(self, base_model, geo: Geometry, mix_bott: int = 256,
                 refine_width: int = 64, refine_depth: int = 2, refine_heads: int = 4,
                 use_refine: bool = True, grad_ckpt: bool = False):
        super().__init__()
        self.net = base_model
        self.geo = geo
        self.grad_ckpt = bool(grad_ckpt)
        self.dim = base_model.hidden_size
        self.head_dim = self.dim // base_model.num_groups
        assert base_model.patch_size == geo.base_patch, \
            f"the released patch size {base_model.patch_size} must be the finest scale {geo.base_patch}"
        self.use_refine = use_refine
        ks = {s: s // geo.base_patch for s in geo.scales[1:]}
        self.mixers = nn.ModuleDict({str(s): ScaleMixer(self.dim, k, mix_bott) for s, k in ks.items()})
        self.refines = nn.ModuleDict(
            {str(s): CellRefine(self.dim, k, refine_width, refine_depth, refine_heads) for s, k in ks.items()}
        ) if use_refine else nn.ModuleDict()

    def gated_tensor_names(self):
        out = []
        for s, m in self.mixers.items():
            out += [f"mixers.{s}.up.weight", f"mixers.{s}.up.bias"]
        for s, m in self.refines.items():
            out += [f"refines.{s}.out.weight", f"refines.{s}.out.bias"]
            for i in range(len(m.blocks)):
                out += [f"refines.{s}.blocks.{i}.ada.weight", f"refines.{s}.blocks.{i}.ada.bias"]
        return out

    def new_parameter_names(self):
        return [n for n, _ in self.named_parameters() if n.startswith(("mixers.", "refines."))]

    def _cell_embeddings(self, x):
        """Every base cell's patch embedding: (B, L, dim)."""
        b = self.geo.base_patch
        xp = F.unfold(x, kernel_size=b, stride=b).transpose(1, 2)      # (B, L, 3*b*b)
        return self.net.s_embedder(xp)

    def _tokens_from_cells(self, cells, L):
        """Adaptive token embeddings (B, N, dim) from per-cell embeddings and packed layout `L`."""
        B, N = L["sizes"].shape
        Ws = self.geo.img // self.geo.base_patch
        out = cells.new_zeros(B, N, self.dim)
        base = self.geo.base_patch
        r0 = (L["top_left"][..., 0] // base)
        c0 = (L["top_left"][..., 1] // base)
        for s in self.geo.scales:
            m = L["sizes"] == s
            if not bool(m.any()):
                continue
            bi, ti = m.nonzero(as_tuple=True)
            k = s // base
            if k == 1:
                out[bi, ti] = cells[bi, r0[bi, ti] * Ws + c0[bi, ti]]
                continue
            rows = r0[bi, ti][:, None] + torch.arange(k, device=cells.device)[None, :]
            cols = c0[bi, ti][:, None] + torch.arange(k, device=cells.device)[None, :]
            idx = (rows[:, :, None] * Ws + cols[:, None, :]).reshape(len(bi), k * k)
            sub = cells[bi[:, None].expand(-1, k * k), idx]             # (M, k*k, dim)
            sub = sub.transpose(1, 2).reshape(-1, self.dim, k, k)
            out[bi, ti] = self.mixers[str(s)](sub).to(out.dtype)
        return out

    def _owner_map(self, L):
        """(B, L_dense) -> the index of the token that covers each base cell."""
        B, N = L["sizes"].shape
        Ws = self.geo.img // self.geo.base_patch
        base = self.geo.base_patch
        owner = L["sizes"].new_zeros(B, Ws * Ws)
        r0 = (L["top_left"][..., 0] // base)
        c0 = (L["top_left"][..., 1] // base)
        for s in self.geo.scales:
            m = L["sizes"] == s
            if not bool(m.any()):
                continue
            bi, ti = m.nonzero(as_tuple=True)
            k = s // base
            for a in range(k):
                for d in range(k):
                    owner[bi, (r0[bi, ti] + a) * Ws + (c0[bi, ti] + d)] = ti
        return owner

    def _joint_block(self, blk, x, y, c, pos_img, pos_txt, attn_mask):
        """`MMDiTBlockT2I.forward` with per-image positions pos_img (B,N,hd/2)."""
        a = blk.attn
        sh_x, sc_x, g_x, shm_x, scm_x, gm_x = blk.adaLN_modulation_img(c).chunk(6, dim=-1)
        sh_y, sc_y, g_y, shm_y, scm_y, gm_y = blk.adaLN_modulation_txt(c).chunk(6, dim=-1)
        xn = blk.norm_x1(x) * (1 + sc_x) + sh_x
        yn = blk.norm_y1(y) * (1 + sc_y) + sh_y

        B, Nx, C = xn.shape
        Ny = yn.shape[1]
        H, hd = a.num_heads, a.head_dim
        qx, kx, vx = a.qkv_x(xn).reshape(B, Nx, 3, H, hd).permute(2, 0, 1, 3, 4)
        qy, ky, vy = a.qkv_y(yn).reshape(B, Ny, 3, H, hd).permute(2, 0, 1, 3, 4)
        qx, kx = a.q_norm_x(qx), a.k_norm_x(kx)
        qy, ky = a.q_norm_y(qy), a.k_norm_y(ky)
        qx, kx = apply_rope_batched(qx, kx, pos_img)
        if pos_txt is not None:
            f = pos_txt[None, :, None, :]
            qy_ = torch.view_as_complex(qy.float().reshape(*qy.shape[:-1], -1, 2))
            ky_ = torch.view_as_complex(ky.float().reshape(*ky.shape[:-1], -1, 2))
            qy = torch.view_as_real(qy_ * f).flatten(3).type_as(qy)
            ky = torch.view_as_real(ky_ * f).flatten(3).type_as(ky)

        qx, kx, vx = qx.transpose(1, 2), kx.transpose(1, 2), vx.transpose(1, 2)
        qy, ky, vy = qy.transpose(1, 2), ky.transpose(1, 2), vy.transpose(1, 2)
        q = torch.cat([qy, qx], dim=2)
        k = torch.cat([ky, kx], dim=2)
        v = torch.cat([vy, vx], dim=2)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=attn_mask)
        oy = o[:, :, :Ny].transpose(1, 2).reshape(B, Ny, C)
        ox = o[:, :, Ny:].transpose(1, 2).reshape(B, Nx, C)
        x = x + g_x * a.proj_x(ox)
        y = y + g_y * a.proj_y(oy)
        x = x + gm_x * blk.mlp_x(blk.norm_x2(x) * (1 + scm_x) + shm_x)
        y = y + gm_y * blk.mlp_y(blk.norm_y2(y) * (1 + scm_y) + shm_y)
        return x, y

    def forward(self, x, t, y, L, text_mask=None):
        """x (B,3,H,W) · t (B,) · y (B,Ltxt,txt_dim) text embeddings · L = apt.layout.pack(...)
        · text_mask (B,Ltxt) bool/0-1, True = a real token."""
        net = self.net
        B, _, H, W = x.shape
        assert H == W == self.geo.img, f"image {H}x{W} does not match the geometry {self.geo.img}"
        b = self.geo.base_patch
        Hs = Ws = H // b
        Ldense = Hs * Ws
        N = L["sizes"].shape[1]

        cells = self._cell_embeddings(x)                         # (B, Ldense, dim)
        s = self._tokens_from_cells(cells, L)                    # (B, N, dim)
        pos_img = analytic_freqs_cis_2d(self.head_dim, L["coords"], Hs, Ws)   # (B, N, hd/2)

        t_emb = net.t_embedder(t.view(-1)).view(B, -1, self.dim)
        Ltxt = min(y.shape[1], net.txt_max_length)
        y_emb = net.y_embedder(y[:, :Ltxt]) + net.y_pos_embedding[:, :Ltxt].to(y.dtype)
        pos_txt = net.fetch_pos_text(Ltxt, x.device) if net.use_text_rope else None
        cond = F.silu(t_emb)

        # joint attention mask: True = attend
        keep_img = L["valid"]
        keep_txt = (text_mask[:, :Ltxt] > 0) if text_mask is not None else \
            torch.ones(B, Ltxt, dtype=torch.bool, device=x.device)
        attn_mask = None
        if not bool(keep_img.all()) or not bool(keep_txt.all()):
            attn_mask = torch.cat([keep_txt, keep_img], dim=1).view(B, 1, 1, Ltxt + N)

        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for i in range(net.patch_depth):
            if ckpt:
                s, y_emb = torch.utils.checkpoint.checkpoint(
                    self._joint_block, net.patch_blocks[i], s, y_emb, cond, pos_img, pos_txt, attn_mask,
                    use_reentrant=False)
            else:
                s, y_emb = self._joint_block(net.patch_blocks[i], s, y_emb, cond, pos_img, pos_txt, attn_mask)

        owner = self._owner_map(L)                                # (B, Ldense)
        s_cells = s.gather(1, owner[..., None].expand(-1, -1, self.dim))
        if self.use_refine:
            s_cells = self._apply_refine(s, s_cells, L, Ws)
        s_cells = F.silu(t_emb + s_cells)

        s_cond = s_cells.reshape(B * Ldense, self.dim)
        xp = net.pixel_embedder(x, img_height=H, img_width=W, patch_size=b)
        for blk in net.pixel_blocks:
            if ckpt:
                xp = torch.utils.checkpoint.checkpoint(blk, xp, s_cond, H, W, b, None, use_reentrant=False)
            else:
                xp = blk(xp, s_cond, H, W, b, None)
        xp = net.final_layer(xp)
        P2, C = b * b, net.out_channels
        xp = xp.view(B, Ldense, P2, C).permute(0, 3, 2, 1).contiguous().view(B, C * P2, Ldense)
        return F.fold(xp, (H, W), kernel_size=b, stride=b)

    def _apply_refine(self, s, s_cells, L, Ws):
        """Add the per-cell correction for every merged token."""
        base = self.geo.base_patch
        r0 = (L["top_left"][..., 0] // base)
        c0 = (L["top_left"][..., 1] // base)
        for sc in self.geo.scales[1:]:
            m = L["sizes"] == sc
            if not bool(m.any()):
                continue
            bi, ti = m.nonzero(as_tuple=True)
            k = sc // base
            d = self.refines[str(sc)](s[bi, ti])                  # (M, k*k, dim)
            rows = r0[bi, ti][:, None] + torch.arange(k, device=s.device)[None, :]
            cols = c0[bi, ti][:, None] + torch.arange(k, device=s.device)[None, :]
            idx = (rows[:, :, None] * Ws + cols[:, None, :]).reshape(len(bi), k * k)
            s_cells = s_cells.index_put((bi[:, None].expand(-1, k * k), idx),
                                        s_cells[bi[:, None].expand(-1, k * k), idx] + d.to(s_cells.dtype))
        return s_cells
