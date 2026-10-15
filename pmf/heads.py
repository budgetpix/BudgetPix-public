"""The refine head for coarse tokens: what turns one token's base prediction into an s x s pixel block.

A small DiT over the token's 4x4 pixel sub-patches, adaLN-modulated by the token's hidden state h. It returns the
RESIDUAL on the bilinearly upsampled base prediction and is zero-initialised at the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


class PixDiTBlock(nn.Module):
    """One adaLN-modulated transformer block over a token's pixel sub-patches. (M,P,w) -> (M,P,w).

    The adaLN projection is ZERO-INIT: shift = scale = gate = 0, so the block is the identity at init.
    """

    def __init__(self, width, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.qn, self.kn = RMSNorm(width // heads), RMSNorm(width // heads)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * width))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, h):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(h).unsqueeze(1).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + sc1) + s1
        M, P, W = y.shape
        q, k, v = self.qkv(y).reshape(M, P, 3, self.heads, W // self.heads).unbind(2)
        q, k = self.qn(q), self.kn(k)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)                      # (M,heads,P,d)
        sc = qh @ kh.transpose(-1, -2)
        sc = sc.float() if sc.dtype in (torch.float16, torch.bfloat16) else sc
        att = torch.softmax(sc / (qh.shape[-1] ** 0.5), dim=-1).to(vh.dtype)
        a = att @ vh
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(M, P, W))
        y = self.norm2(x) * (1 + sc2) + s2
        return x + g2 * self.mlp(y)


class PixelDiTRefineHead(nn.Module):
    """Residual on the resized base prediction, produced by a small DiT over the token's pixels.

    Runs at the BASE resolution b x b (patch x patch sub-patches -> (b/patch)^2 positions of `width`),
    emits r*r*3 channels per sub-patch and PixelShuffle(r) folds them into the s x s residual.
    """

    def __init__(self, dim, base, s, out=3, width=64, depth=2, patch=4, heads=4):
        super().__init__()
        self.base, self.s, self.out_ch = base, s, out
        self.r = s // base
        if base % patch:
            raise ValueError(f"pixel patch {patch} must divide the base patch {base}")
        self.patch, self.grid = patch, base // patch
        P = self.grid * self.grid
        self.tokenize = nn.Linear(patch * patch * out, width)
        self.pos = nn.Parameter(torch.zeros(1, P, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([PixDiTBlock(width, dim, heads) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.out = nn.Linear(width, patch * patch * self.r * self.r * out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.shuffle = nn.PixelShuffle(self.r)

    def forward(self, base_pred, h):                                # (M,3,b,b), (M,dim)
        M, C, H, W = base_pred.shape
        p, g = self.patch, self.grid
        t = base_pred.reshape(M, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(M, g * g, p * p * C)
        t = self.tokenize(t) + self.pos
        for blk in self.blocks:
            t = blk(t, h)
        t = self.out(self.norm(t))                                  # (M, g*g, p*p*r*r*C)
        rr = self.r * self.r
        t = t.reshape(M, g, g, p, p, rr * C).permute(0, 5, 1, 3, 2, 4).reshape(M, rr * C, H, W)
        return self.shuffle(t)
