"""BudgetPix on MiniT2I: the MiniT2I MM-DiT (imported from MINIT2I_ROOT) with token-adaptive patch sizes."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layout import Geometry

NEW_MODULES = ("mixers.", "refines.")


# ---------------------------------------------------------------------------------------------------------
# the released code and weights
# ---------------------------------------------------------------------------------------------------------
def load_mmdit():
    """Load `$MINIT2I_ROOT/diffusers/mmdit.py` by file path."""
    name = "_minit2i_mmdit"
    if name not in sys.modules:
        root = os.environ.get("MINIT2I_ROOT")
        path = Path(root or ".") / "diffusers" / "mmdit.py"
        if not root or not path.exists():
            raise SystemExit("set MINIT2I_ROOT to a checkout of https://github.com/Hope7Happiness/minit2i-torch")
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def build_mmdit(net_cfg):
    up = load_mmdit()
    return up.MMJiT(up.MMJiTConfig(**{k: v for k, v in net_cfg.items() if k in up.MMJiTConfig.__dataclass_fields__}))


def _strip(k):
    """Drop wrapper prefixes such as `model.` and `net.`."""
    changed = True
    while changed:
        changed = False
        for p in ("model.", "net.", "module.", "_orig_mod."):
            if k.startswith(p):
                k, changed = k[len(p):], True
    return k


def load_released(path, image_size=512, patch_size=16):
    """MiniT2I weights -> MMJiT. `path` is an HF snapshot directory
    (transformer/{config.json,*.safetensors}) or a torch payload {"config": ..., "state_dict": ...}."""
    path = Path(path)
    if path.is_dir():
        d = path / "transformer" if (path / "transformer").exists() else path
        cfg = {k: v for k, v in json.load(open(d / "config.json")).items() if not k.startswith("_")}
        if isinstance(cfg.get("cfg_interval"), list):
            cfg["cfg_interval"] = tuple(cfg["cfg_interval"])
        from safetensors.torch import load_file
        sd = {}
        for shard in sorted(d.glob("*.safetensors")):
            sd.update(load_file(str(shard)))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg, sd = dict(payload.get("config", {})), payload["state_dict"]
    net = build_mmdit(dict(cfg, image_size=image_size, patch_size=patch_size))
    missing, _ = net.load_state_dict({_strip(k): v for k, v in sd.items()}, strict=False)
    if missing:
        raise SystemExit(f"{path}: {len(missing)} weights of the released model not found, e.g. {missing[:3]}")
    return net


# ---------------------------------------------------------------------------------------------------------
# positions at token centres
# ---------------------------------------------------------------------------------------------------------
def sincos_2d(coords, dim):
    """2D sin-cos embedding at (y, x) coordinates in patch units; x half first."""
    with torch.autocast(device_type=coords.device.type, enabled=False):
        omega = 1.0 / (10000 ** (torch.arange(dim // 4, device=coords.device, dtype=torch.float32) / (dim // 2 / 2.0)))
        ay, ax = coords.float()[..., 0][..., None] * omega, coords.float()[..., 1][..., None] * omega
        return torch.cat([ax.sin(), ax.cos(), ay.sin(), ay.cos()], dim=-1)


def rope_2d(coords, head_dim, theta=10000.0):
    """2D rotary angles at (y, x) coordinates: (..., head_dim)."""
    dim = head_dim // 2
    with torch.autocast(device_type=coords.device.type, enabled=False):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=coords.device, dtype=torch.float32)[: dim // 2] / dim))
        a = torch.cat([coords.float()[..., 0][..., None] * freqs, coords.float()[..., 1][..., None] * freqs], dim=-1)
        return torch.cat([a, a], dim=-1)


def rope_1d(length, head_dim, device, theta=10000.0):
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        a = torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv
        return torch.cat([a, a], dim=-1)[..., :head_dim]


def apply_rope(x, angles):                                       # x (B,L,H,D), angles (B,L,D)
    cos, sin = angles.cos().unsqueeze(2).to(x.dtype), angles.sin().unsqueeze(2).to(x.dtype)
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-b, a], dim=-1) * sin


# ---------------------------------------------------------------------------------------------------------
# BudgetPix modules (every output projection is zero-initialised)
# ---------------------------------------------------------------------------------------------------------
class ScaleMixer(nn.Module):
    """A coarse token = the mean of the k x k pretrained cell embeddings it covers + a zero-init mixer."""

    def __init__(self, dim, k, bott=256):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, dim, k, k))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.down = nn.Conv2d(dim, bott, 1)
        self.dw = nn.Conv2d(bott, bott, k, groups=bott)
        self.up = nn.Linear(bott, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, sub):                                      # (M,dim,k,k) -> (M,dim)
        return sub.mean(dim=(2, 3)) + self.up(self.dw(self.down(sub + self.pos.to(sub.dtype))).flatten(1))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


class PixDiTBlock(nn.Module):
    """A DiT block over one token's pixel sub-patches, adaLN-modulated by the token's hidden state."""

    def __init__(self, width, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.qn, self.kn = RMSNorm(width // heads), RMSNorm(width // heads)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp_ratio)), nn.SiLU(), nn.Linear(int(width * mlp_ratio), width))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * width))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, h):                                     # (M,P,w), (M,dim)
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(h).unsqueeze(1).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + sc1) + s1
        M, P, W = y.shape
        q, k, v = self.qkv(y).reshape(M, P, 3, self.heads, W // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(self.qn(q).transpose(1, 2), self.kn(k).transpose(1, 2), v.transpose(1, 2))
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(M, P, W))
        return x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + s2)


class PixelDiTRefineHead(nn.Module):
    """Residual that turns the 16-px prediction of an s-px token into s x s pixels: 4x4 sub-patches of the
    prediction, refined by PixDiT blocks, projected to r*r times the pixels, folded up by PixelShuffle(r)."""

    def __init__(self, dim, base, s, width=64, depth=2, patch=4, heads=4):
        super().__init__()
        self.r, self.patch, self.grid = s // base, patch, base // patch
        self.tokenize = nn.Linear(patch * patch * 3, width)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([PixDiTBlock(width, dim, heads) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.out = nn.Linear(width, patch * patch * self.r * self.r * 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.shuffle = nn.PixelShuffle(self.r)

    def forward(self, x, h):                                     # (M,3,b,b), (M,dim) -> (M,3,s,s)
        M, C, H, W = x.shape
        p, g, rr = self.patch, self.grid, self.r * self.r
        t = self.tokenize(x.reshape(M, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(M, g * g, p * p * C)) + self.pos
        for blk in self.blocks:
            t = blk(t, h)
        t = self.out(self.norm(t)).reshape(M, g, g, p, p, rr * C).permute(0, 5, 1, 3, 2, 4).reshape(M, rr * C, H, W)
        return self.shuffle(t)


# ---------------------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------------------
class AdaptiveMMJiT(nn.Module):
    def __init__(self, net, geo):
        super().__init__()
        cfg = net.cfg
        if (cfg.patch_size, cfg.image_size) != (geo.base_patch, geo.img):
            raise ValueError(f"released model is {cfg.image_size}/{cfg.patch_size}, geometry is {geo.img}/{geo.base_patch}")
        if any(hasattr(blk, "modulation") for blk in net.double_blocks):
            raise ValueError("this upstream version modulates its blocks; the masked forward below does not")
        self.net, self.geo = net, geo
        self.dim, self.head_dim = cfg.hidden_size, cfg.head_dim
        self.mixers = nn.ModuleDict({str(s): ScaleMixer(self.dim, s // geo.base_patch) for s in geo.scales[1:]})
        self.refines = nn.ModuleDict({str(s): PixelDiTRefineHead(self.dim, geo.base_patch, s) for s in geo.scales[1:]})

    def embed(self, img, L):
        """(B,3,H,W) -> (B,N,dim). Coarse tokens pool the embeddings of the 16-px cells they cover."""
        B, N = L["sizes"].shape
        b = self.geo.base_patch
        cells = self.net.img_embedder.proj2(self.net.img_embedder.proj1(img))          # (B,dim,g,g)
        out = img.new_zeros(B, N, self.dim, dtype=cells.dtype)
        gy, gx = L["top_left"][..., 0] // b, L["top_left"][..., 1] // b
        for s in self.geo.scales:
            bi, ti = (L["sizes"] == s).nonzero(as_tuple=True)
            if len(bi) == 0:
                continue
            if s == b:
                out[bi, ti] = cells[bi, :, gy[bi, ti], gx[bi, ti]]
            else:
                o = torch.arange(s // b, device=img.device)
                yy, xx = gy[bi, ti][:, None] + o, gx[bi, ti][:, None] + o
                sub = cells[bi[:, None, None], :, yy[:, :, None], xx[:, None, :]]           # (M,k,k,dim)
                out[bi, ti] = self.mixers[str(s)](sub.permute(0, 3, 1, 2).contiguous())
        return out

    def assemble(self, base_out, h, L):
        """Per-token predictions (B,N,b*b*3) -> image: base tokens as predicted, coarse ones resized + refined."""
        B = base_out.shape[0]
        b = self.geo.base_patch
        img = base_out.new_zeros(B, 3, self.geo.img, self.geo.img)
        for s in self.geo.scales:
            bi, ti = (L["sizes"] == s).nonzero(as_tuple=True)
            if len(bi) == 0:
                continue
            px = base_out[bi, ti].view(-1, b, b, 3).permute(0, 3, 1, 2)
            if s != b:
                px = F.interpolate(px.float(), size=(s, s), mode="bilinear", align_corners=False) \
                    + self.refines[str(s)](px.float(), h[bi, ti].float())
            o = torch.arange(s, device=img.device)
            iy = (L["top_left"][bi, ti, 0][:, None] + o)[:, :, None].expand(-1, -1, s)
            ix = (L["top_left"][bi, ti, 1][:, None] + o)[:, None, :].expand(-1, s, -1)
            ib = bi[:, None, None].expand(-1, s, s)
            for ch in range(3):
                img.index_put_((ib, torch.full_like(ib, ch), iy, ix), px[:, ch].to(img.dtype), accumulate=True)
        return img

    def forward(self, img, t, context, attn_mask, L):
        """img (B,3,H,W) noisy, t (B,) in [0,1] (1 = clean), context (B,P,1024) T5 states, attn_mask (B,P)."""
        net = self.net
        context = torch.where(attn_mask.to(context.device)[:, :, None] > 0.5, context, net.mask_token.to(dtype=context.dtype))
        x = self.embed(img, L)
        x = x + sincos_2d(L["coords"], self.dim).to(x.dtype)
        txt = net.txt_embedder(context.to(dtype=net.txt_embedder.weight.dtype))
        for block in net.txt_preamble_blocks:
            txt = block(txt)
        lt = txt.shape[1]
        angles = torch.cat([rope_1d(lt, self.head_dim, x.device).unsqueeze(0).expand(x.shape[0], -1, -1),
                            rope_2d(L["coords"], self.head_dim)], dim=1)
        key_ok = torch.cat([torch.ones(x.shape[0], lt, dtype=torch.bool, device=x.device), L["valid"]], dim=1)
        bias = torch.zeros(x.shape[0], 1, 1, key_ok.shape[1], device=x.device).masked_fill_(~key_ok[:, None, None, :], float("-inf"))
        for block in net.double_blocks:
            x, txt = double_block(block, x, txt, angles, bias, self.head_dim)
        base_out = net.final_layer(torch.cat([txt, x], dim=1))[:, lt:, :]
        return self.assemble(base_out, x, L).float()


def double_block(block, x, txt, angles, bias, head_dim):
    """DoubleStreamBlock with coordinate RoPE and padded image slots masked out."""
    b, li, lt = x.shape[0], x.shape[1], txt.shape[1]
    qkv_i = block.img_qkv(block.img_norm1(x)).reshape(b, li, 3, block.num_heads, block.head_dim)
    qkv_t = block.txt_qkv(block.txt_norm1(txt)).reshape(b, lt, 3, block.num_heads, block.head_dim)
    q = apply_rope(torch.cat([block.q_norm(qkv_t[:, :, 0]), block.q_norm(qkv_i[:, :, 0])], dim=1), angles)
    k = apply_rope(torch.cat([block.k_norm(qkv_t[:, :, 1]), block.k_norm(qkv_i[:, :, 1])], dim=1), angles)
    v = torch.cat([qkv_t[:, :, 2], qkv_i[:, :, 2]], dim=1)
    attn = torch.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim ** -0.5) + bias.to(q.dtype)
    out = torch.einsum("bhqk,bkhd->bqhd", attn.softmax(dim=-1), v)
    x = x + block.img_attn_proj(out[:, lt:].reshape(b, li, -1))
    txt = txt + block.txt_attn_proj(out[:, :lt].reshape(b, lt, -1))
    return x + block.img_mlp(block.img_norm2(x)), txt + block.txt_mlp(block.txt_norm2(txt))


# ---------------------------------------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------------------------------------
def geometry(cfg):
    return Geometry(img=cfg["image_size"], base_patch=cfg["patch_size"], scales=tuple(cfg["scales"]),
                    thresholds=tuple(cfg["thresholds"]))


def load_budgetpix(path, device="cpu"):
    """A BudgetPix checkpoint (a release file or a train.py checkpoint) -> (model, cfg), EMA weights."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = AdaptiveMMJiT(build_mmdit(dict(cfg["net_cfg"], image_size=cfg["image_size"], patch_size=cfg["patch_size"])),
                          geometry(cfg))
    sd = dict(ck["model"])
    if not ck.get("ema_applied"):                                # train.py checkpoint: use the EMA weights
        sd.update(ck["ema"])
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg


def load_model(ckpt=None, released=None, config=None, device="cpu"):
    """(model, cfg) from a BudgetPix checkpoint, or from the released weights with the geometry of `config`
    and the added modules at zero."""
    if ckpt:
        return load_budgetpix(ckpt, device)
    import yaml
    cfg = yaml.safe_load(open(config))
    return AdaptiveMMJiT(load_released(released, cfg["image_size"], cfg["patch_size"]), geometry(cfg)).to(device).eval(), cfg
