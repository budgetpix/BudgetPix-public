"""BudgetPix on JiT: the JiT transformer with token-adaptive patch sizes.

A token is a base patch (16 px at 256, 32 px at 512) or a merged 2x2 / 4x4 block of base patches. The
added modules are zero-initialised, so on the dense layout the model is exactly the released JiT.

The JiT modules are vendored from github.com/LTH14/JiT (MIT).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from layout import Geometry, IN_CONTEXT_LEN

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

# JiT sizes; the token grid is 16 x 16 at every resolution
ARCH = {"B": dict(hidden=768, depth=12, heads=12, head_dim=64, bottleneck=128, in_context_start=4),
        "L": dict(hidden=1024, depth=24, heads=16, head_dim=64, bottleneck=128, in_context_start=8),
        "H": dict(hidden=1280, depth=32, heads=16, head_dim=80, bottleneck=256, in_context_start=10)}
NUM_CLASSES = 1000
NEW_MODULES = ("x_embedder.scale_emb", "x_embedder.enc", "final_layer.refine")


# ---------------------------------------------------------------------------------------------------------
# JiT building blocks
# ---------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * x).to(dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        hidden = int(hidden * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, patch, pca_dim, dim):
        super().__init__()
        self.proj1 = nn.Conv2d(3, pca_dim, patch, patch, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, dim, 1)

    def forward(self, p):                                        # (N,3,patch,patch) -> (N,dim)
        return self.proj2(self.proj1(p)).flatten(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, dim)   # the last row is the unconditional label

    def forward(self, labels):
        return self.embedding_table(labels)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def rotate_half(x):
    x1, x2 = rearrange(x, "... (d r) -> ... d r", r=2).unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


# ---------------------------------------------------------------------------------------------------------
# positions at token centres
# ---------------------------------------------------------------------------------------------------------
def _sincos_numpy(coords, dim):
    """JiT's 2D sin-cos embedding (float64 numpy) at (row, col) coordinates."""
    rc = coords.detach().cpu().double().numpy()
    omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim / 4.0))
    emb = []
    for pos in (rc[:, 1], rc[:, 0]):
        out = np.einsum("m,d->md", pos, omega)
        emb.append(np.concatenate([np.sin(out), np.cos(out)], axis=1))
    return torch.from_numpy(np.concatenate(emb, axis=1)).float().to(coords.device)


_SINCOS = {}


def sincos_pos(coords, dim):
    """The sin-cos embedding at token centres."""
    c2 = coords.detach() * 2
    if coords.numel() == 0 or not torch.equal(c2, c2.round()) or c2.min() < 0 or c2.max() >= 64:
        return _sincos_numpy(coords, dim)
    key = (dim, coords.device)
    if key not in _SINCOS:
        r = torch.arange(64, dtype=torch.float32) / 2
        lattice = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2)
        _SINCOS[key] = _sincos_numpy(lattice, dim).reshape(64, 64, dim).to(coords.device)
    i = c2.long()
    return _SINCOS[key][i[:, 0], i[:, 1]]


def rope(coords, head_dim, theta=10000.0):
    """2D RoPE cos/sin at token centres: row frequencies then column frequencies."""
    dim = head_dim // 2
    base = 1.0 / (theta ** (torch.arange(0, dim, 2, device=coords.device, dtype=torch.float32)[: dim // 2] / dim))
    freqs = torch.cat([torch.outer(coords[:, i].float(), base).repeat_interleave(2, dim=-1) for i in (0, 1)], dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):                                     # x (T,heads,hd), cos/sin (T,hd)
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


# ---------------------------------------------------------------------------------------------------------
# transformer on the packed sequence
# ---------------------------------------------------------------------------------------------------------
def varlen_attention(q, k, v, cu, max_s):
    """Attention inside each image of a packed batch. q, k, v (T,heads,hd); cu (B+1,) segment offsets."""
    if flash_attn_varlen_func is not None and q.is_cuda:
        dt = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        return flash_attn_varlen_func(q.to(dt).contiguous(), k.to(dt).contiguous(), v.to(dt).contiguous(),
                                      cu, cu, max_s, max_s, causal=False)
    out = torch.empty_like(q)
    bounds = cu.tolist()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            o = F.scaled_dot_product_attention(q[a:b].transpose(0, 1)[None], k[a:b].transpose(0, 1)[None],
                                               v[a:b].transpose(0, 1)[None])
            out[a:b] = o[0].transpose(0, 1)
    return out


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin, cu, max_s):
        T, C = x.shape
        q, k, v = self.qkv(x).reshape(T, 3, self.heads, self.head_dim).permute(1, 0, 2, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rope(k, cos.to(k.dtype), sin.to(k.dtype))
        return self.proj(varlen_attention(q, k, v, cu, max_s).to(x.dtype).reshape(T, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, cos, sin, cu, max_s, tok_batch=None):
        # c is one row per token, or one row per image with tok_batch to gather it
        m = self.adaLN_modulation(c)[tok_batch] if tok_batch is not None else self.adaLN_modulation(c)
        s_msa, sc_msa, g_msa, s_mlp, sc_mlp, g_mlp = m.chunk(6, dim=-1)
        x = x + g_msa * self.attn(modulate(self.norm1(x), s_msa, sc_msa), cos, sin, cu, max_s)
        return x + g_mlp * self.mlp(modulate(self.norm2(x), s_mlp, sc_mlp))


# ---------------------------------------------------------------------------------------------------------
# BudgetPix modules (every output projection is zero-initialised)
# ---------------------------------------------------------------------------------------------------------
class ScaleEncoder(nn.Module):
    """Pools the k x k sub-patch embeddings of a coarse token into a residual for its embedding."""

    def __init__(self, dim, k, bott=256):
        super().__init__()
        self.mini_pos = nn.Parameter(torch.randn(1, dim, k, k) * 0.02)
        self.down = nn.Conv2d(dim, bott, 1)
        self.mix_dw = nn.Conv2d(bott, bott, k, k, groups=bott)
        self.mix_pw = nn.Conv2d(bott, bott, 1)
        self.up = nn.Linear(bott, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, sub):                                      # (N,dim,k,k) -> (N,dim)
        x = F.gelu(self.down(sub + self.mini_pos))
        return self.up(F.gelu(self.mix_pw(self.mix_dw(x))).flatten(1))


class PatchEmbed(nn.Module):
    def __init__(self, geo, bottleneck, dim):
        super().__init__()
        self.base_patch, self.scales = geo.base_patch, geo.scales
        self.base = BottleneckPatchEmbed(geo.base_patch, bottleneck, dim)
        self.scale_emb = nn.Embedding(len(geo.scales), dim)
        nn.init.zeros_(self.scale_emb.weight)
        for s in geo.scales[1:]:
            setattr(self, f"enc{s}", ScaleEncoder(dim, s // geo.base_patch))

    def forward(self, p, s):                                     # (N,3,s,s) -> (N,dim)
        b = self.base_patch
        if s == b:
            z = self.base(p)
        else:
            k = s // b
            z = self.base(F.interpolate(p, size=(b, b), mode="bilinear", align_corners=False))
            subs = rearrange(p, "n c (nh ph) (nw pw) -> (n nh nw) c ph pw", ph=b, pw=b)
            sub = rearrange(self.base(subs), "(n nh nw) d -> n d nh nw", nh=k, nw=k)
            z = z + getattr(self, f"enc{s}")(sub).to(z.dtype)
        return z + self.scale_emb(torch.full((z.shape[0],), self.scales.index(s), dtype=torch.long, device=z.device))


class HeadRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class PixDiTBlock(nn.Module):
    """A DiT block over one token's pixel sub-patches, adaLN-modulated by the token's hidden state."""

    def __init__(self, width, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = HeadRMSNorm(width), HeadRMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.q_norm, self.k_norm = HeadRMSNorm(width // heads), HeadRMSNorm(width // heads)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.SiLU(), nn.Linear(int(width * mlp_ratio), width))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * width))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, h):                                     # (N,P,w), (N,dim)
        N, P, w = x.shape
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(h).unsqueeze(1).chunk(6, dim=-1)
        qkv = self.qkv(self.norm1(x) * (1 + sc1) + s1).reshape(N, P, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2])
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(N, P, w))
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)


class PixelDiTRefineHead(nn.Module):
    """Residual that turns the base-size prediction of a token of size s into s x s pixels: the
    prediction is cut into patch x patch sub-patches, refined by `depth` PixDiT blocks, projected to
    r*r times the pixels and folded up by PixelShuffle(r), r = s / base."""

    CHUNK = 16384                                                # tokens per call

    def __init__(self, dim, base, s, width=64, depth=2, patch=4, heads=4):
        super().__init__()
        self.r, self.patch, self.grid = s // base, patch, base // patch
        self.tokenize = nn.Linear(patch * patch * 3, width)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([PixDiTBlock(width, dim, heads) for _ in range(depth)])
        self.norm = HeadRMSNorm(width)
        self.out = nn.Linear(width, patch * patch * self.r * self.r * 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.shuffle = nn.PixelShuffle(self.r)

    def forward(self, x, h):                                     # (N,3,b,b), (N,dim) -> (N,3,s,s)
        if x.shape[0] > self.CHUNK:
            return torch.cat([self(x[i:i + self.CHUNK], h[i:i + self.CHUNK]) for i in range(0, x.shape[0], self.CHUNK)])
        N, C, H, W = x.shape
        p, g, rr = self.patch, self.grid, self.r * self.r
        t = self.tokenize(x.reshape(N, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(N, g * g, p * p * C)) + self.pos
        for blk in self.blocks:
            t = blk(t, h)
        t = self.out(self.norm(t)).reshape(N, g, g, p, p, rr * C).permute(0, 5, 1, 3, 2, 4).reshape(N, rr * C, H, W)
        return self.shuffle(t)


class FinalLayer(nn.Module):
    def __init__(self, dim, geo):
        super().__init__()
        self.norm_final = RMSNorm(dim)
        self.linear = nn.Linear(dim, geo.base_patch ** 2 * 3)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        for s in geo.scales[1:]:
            setattr(self, f"refine{s}", PixelDiTRefineHead(dim, geo.base_patch, s))

    def forward(self, x, c, tok_batch):
        shift, scale = self.adaLN_modulation(c)[tok_batch].chunk(2, dim=-1)
        h = modulate(self.norm_final(x), shift, scale)
        return self.linear(h), h


# ---------------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------------
class BudgetPixJiT(nn.Module):
    def __init__(self, variant="B", geo=Geometry()):
        super().__init__()
        a = ARCH[variant]
        self.geo, self.variant = geo, variant
        self.head_dim, self.in_context_start = a["head_dim"], a["in_context_start"]
        d = a["hidden"]
        self.t_embedder = TimestepEmbedder(d)
        self.y_embedder = LabelEmbedder(NUM_CLASSES, d)
        self.x_embedder = PatchEmbed(geo, a["bottleneck"], d)
        self.pos_embed = nn.Parameter(torch.zeros(1, geo.n_dense, d), requires_grad=False)
        self.in_context_posemb = nn.Parameter(torch.zeros(1, IN_CONTEXT_LEN, d))
        self.blocks = nn.ModuleList([Block(d, a["heads"]) for _ in range(a["depth"])])
        self.final_layer = FinalLayer(d, geo)

    def forward(self, z, t, y, L):
        """z (B,3,H,W) noisy image, t (B,) in [0,1] (1 = clean), y (B,) labels, L a packed layout -> x0 prediction."""
        geo, b = self.geo, self.geo.base_patch
        c = self.t_embedder(t) + self.y_embedder(y)                              # (B,d), one row per image
        tok = z.new_zeros(L["sizes"].shape[0], c.shape[1])
        for s in geo.scales:
            m = L["sizes"] == s
            if m.any():
                patches = rearrange(z, "b c (nh ph) (nw pw) -> b (nh nw) c ph pw", ph=s, pw=s)[L["tok_batch"][m], L[f"idx{s}"][m]]
                tok[m] = self.x_embedder(patches, s).to(tok.dtype)
        if "pos" not in L:
            pos = sincos_pos(L["coords"], tok.shape[1])
            m = L["sizes"] == b
            pos[m] = self.pos_embed[0][L[f"idx{b}"][m]].to(pos.dtype)
            L["pos"], (L["cos"], L["sin"]) = pos, rope(L["coords"], self.head_dim)
        tok = tok + L["pos"].to(tok.dtype)
        cos, sin = L["cos"].to(tok.dtype), L["sin"].to(tok.dtype)
        for blk in self.blocks[:self.in_context_start]:
            tok = blk(tok, c, cos, sin, L["cu_img"], L["max_n"], tok_batch=L["tok_batch"])
        # JiT's in-context class tokens join each image's sequence for the remaining blocks
        B, icl, hd = c.shape[0], IN_CONTEXT_LEN, self.head_dim
        ic = (self.y_embedder(y).unsqueeze(1).repeat(1, icl, 1) + self.in_context_posemb).reshape(B * icl, -1)
        buf = tok.new_zeros(L["Tic"], tok.shape[1])
        buf[L["img2ic"]], buf[L["ic_slots"]] = tok, ic.to(tok.dtype)
        cos_ic, sin_ic = tok.new_ones(L["Tic"], hd), tok.new_zeros(L["Tic"], hd)
        cos_ic[L["img2ic"]], sin_ic[L["img2ic"]] = cos, sin
        c_ic = c[L["tok_batch_ic"]]
        for blk in self.blocks[self.in_context_start:]:
            buf = blk(buf, c_ic, cos_ic, sin_ic, L["cu_ic"], L["max_nic"])
        base_out, h = self.final_layer(buf[L["img2ic"]], c, L["tok_batch"])
        return self.assemble(base_out.float(), h, L, B)

    def assemble(self, base_out, h, L, B):
        """Per-token predictions -> image: base patches as predicted, coarse tokens resized and refined."""
        geo, b = self.geo, self.geo.base_patch
        img = torch.zeros(B, 3, geo.img, geo.img, device=base_out.device)
        for s in geo.scales:
            m = L["sizes"] == s
            if not m.any():
                continue
            px = base_out[m].reshape(-1, b, b, 3).permute(0, 3, 1, 2)
            if s != b:
                px = F.interpolate(px, size=(s, s), mode="bilinear", align_corners=False) \
                    + getattr(self.final_layer, f"refine{s}")(px, h[m].float()).float()
            per = geo.img // s
            buf = torch.zeros(B, per * per, 3, s, s, device=img.device)
            buf[L["tok_batch"][m], L[f"idx{s}"][m]] = px
            img = img + rearrange(buf, "b (nh nw) c ph pw -> b c (nh ph) (nw pw)", nh=per, nw=per)
        return img


# ---------------------------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------------------------
def _load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _architecture(sd):
    """(variant, Geometry) of a state dict."""
    hidden = sd["blocks.0.attn.qkv.weight"].shape[1]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    variant = next((v for v, a in ARCH.items() if (a["hidden"], a["depth"]) == (hidden, depth)), None)
    if variant is None:
        raise ValueError(f"hidden {hidden} / depth {depth} is not a JiT-B, L or H")
    patch = sd["x_embedder.base.proj1.weight"].shape[-1]
    merged = sorted(int(k.split(".")[1][3:]) for k in sd if k.startswith("x_embedder.enc") and k.endswith(".up.weight"))
    scales = (patch, *merged) if merged else (patch, 2 * patch, 4 * patch)
    return variant, Geometry(img=16 * patch, base_patch=patch, scales=scales)


def noise_scale(geo):
    """JiT's noise scale: 1.0 at 256, 2.0 at 512."""
    return geo.img / 256


def load_budgetpix(path, device="cpu"):
    """A BudgetPix checkpoint (a release file or a train.py checkpoint) -> model with its EMA weights."""
    ck = _load_cpu(path)
    sd = dict(ck["model"])
    if not ck.get("ema_applied"):                                # train.py checkpoint: use the EMA weights
        sd.update(ck["ema"])
    model = BudgetPixJiT(*_architecture(sd))
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_jit(path, device="cpu"):
    """A released JiT checkpoint (its `model_ema1` weights) -> BudgetPix model whose added modules are at
    their zero init, i.e. the released model."""
    ck = _load_cpu(path)
    sd = {(k[4:] if k.startswith("net.") else k).replace("x_embedder.proj", "x_embedder.base.proj"): v
          for k, v in ck.get("model_ema1", ck.get("model")).items()}
    model = BudgetPixJiT(*_architecture(sd))
    missing, _ = model.load_state_dict(sd, strict=False)
    left = [k for k in missing if not k.startswith(NEW_MODULES)]
    if left:
        raise ValueError(f"{path} does not match the JiT architecture: {left[:4]} not found")
    return model.to(device)
