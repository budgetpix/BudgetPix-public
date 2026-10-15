"""Token layouts: which parts of the image get fine tokens and which get coarse ones.

A layout is a list of tokens (size_px, y, x) that tiles the image. The finest token is the 16-px
patch (1024 tokens at 512x512); flat regions are merged into 32- and 64-px tokens. A block merges when the
Shannon entropy of its luma histogram is below a per-scale threshold (6.0 bits at 32 px, 4.0 bits at
64 px), and one per-image offset shifts both thresholds to hit a token budget.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

LUMA = (0.2989, 0.5870, 0.1140)


@dataclass(frozen=True)
class Geometry:
    img: int = 512
    base_patch: int = 16              # finest token
    scales: tuple = (16, 32, 64)      # token sizes, base patch first, each double the previous
    thresholds: tuple = (6.0, 4.0)    # merge thresholds (bits) for scales[1:], finest first
    entropy_bins: int = 512

    def __post_init__(self):
        s = tuple(sorted(int(v) for v in self.scales))
        object.__setattr__(self, "scales", s)
        object.__setattr__(self, "thresholds", tuple(float(v) for v in self.thresholds))
        if s[0] != self.base_patch or any(b != 2 * a for a, b in zip(s, s[1:])):
            raise ValueError(f"scales {s} must start at the base patch {self.base_patch} and double each step")
        if self.img % s[-1] or len(self.thresholds) != len(s) - 1:
            raise ValueError(f"bad geometry: img {self.img}, scales {s}, thresholds {self.thresholds}")

    @property
    def n_dense(self):
        return (self.img // self.base_patch) ** 2

    @property
    def n_min(self):
        return (self.img // self.scales[-1]) ** 2

    @property
    def thresholds_by_scale(self):
        return dict(zip(self.scales[1:], self.thresholds))


# ---------------------------------------------------------------------------------------------------------
# entropy maps
# ---------------------------------------------------------------------------------------------------------
def _entropy(gray, cell, bins):
    """gray (B,H,W) in [0,255] -> (B, H/cell * W/cell): entropy in bits of each cell's luma histogram."""
    B, H, W = gray.shape
    g = (gray.clamp(0, 255) * (bins / 256.0)).long().clamp_(0, bins - 1)
    g = g.view(B, H // cell, cell, W // cell, cell).permute(0, 1, 3, 2, 4).reshape(B, -1, cell * cell)
    hist = torch.zeros(B, g.shape[1], bins, device=gray.device, dtype=torch.float32)
    hist.scatter_add_(2, g, torch.ones_like(g, dtype=torch.float32))
    p = hist / (cell * cell)
    return -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)


def detail_pyramid(images_u8, geo):
    """images (B,3,H,W) uint8 -> {s: (B, H/s, W/s) entropy} for every merge scale s."""
    x = images_u8.float()
    gray = torch.nan_to_num(LUMA[0] * x[:, 0] + LUMA[1] * x[:, 1] + LUMA[2] * x[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
    return {s: _entropy(gray, s, geo.entropy_bins).view(-1, geo.img // s, geo.img // s) for s in geo.scales[1:]}


# ---------------------------------------------------------------------------------------------------------
# quadtree and budget calibration
# ---------------------------------------------------------------------------------------------------------
def _leaf_masks(maps, offsets, geo):
    """maps {s: (B,g,g) float64}, offsets (B,) -> {s: (B,g,g) bool}, True where a token of size s sits.
    Walking from the coarsest scale down, a block that exists merges when its entropy is below the
    threshold plus the offset; otherwise its four children exist at the next scale."""
    g = geo.img // geo.scales[-1]
    alive = np.ones((len(offsets), g, g), dtype=bool)
    out = {}
    for s in reversed(geo.scales):
        if s == geo.base_patch:
            out[s] = alive
            break
        merged = alive & (maps[s] < (geo.thresholds_by_scale[s] + offsets)[:, None, None])
        out[s] = merged
        alive = np.repeat(np.repeat(alive & ~merged, 2, 1), 2, 2)
    return out


def _count(maps, offsets, geo):
    return sum(m.reshape(m.shape[0], -1).sum(1) for m in _leaf_masks(maps, offsets, geo).values())


def _morton(r, c, depth):
    """Sort key of the blocks of one level: coarsest ancestor row-major, then the child index at every level."""
    key = np.zeros_like(r)
    for d in range(depth, -1, -1):
        key = key * 4 + (((r >> d) & 1) << 1 | ((c >> d) & 1))
    return ((r >> (depth + 1)) * (1 << 30) + (c >> (depth + 1))) * (1 << (2 * (depth + 1))) + key


def _tokens(masks, i, geo):
    """Image i's tokens, coarsest first, in quadtree order."""
    toks = []
    for depth, s in enumerate(reversed(geo.scales)):
        r, c = np.nonzero(masks[s][i])
        if len(r):
            order = np.argsort(_morton(r, c, depth - 1), kind="stable") if depth > 0 else np.lexsort((c, r))
            toks += [(s, int(rr) * s, int(cc) * s) for rr, cc in zip(r[order], c[order])]
    return toks


def _as_numpy(pyr):
    return {s: v.detach().to("cpu", torch.float64).numpy() for s, v in pyr.items()}


def all_base_tokens(geo):
    """The dense layout: every token is a base patch."""
    p, g = geo.base_patch, geo.img // geo.base_patch
    return [(p, p * r, p * c) for r in range(g) for c in range(g)]


def layouts_for_budget(pyr, geo, budget, iters=24):
    """Per-image layouts with the largest token count that does not exceed `budget`."""
    if budget < geo.n_min:
        raise ValueError(f"budget {budget} is below the coarsest layout ({geo.n_min} tokens)")
    maps = _as_numpy(pyr)
    B = next(iter(maps.values())).shape[0]
    span = math.log2(geo.entropy_bins) + max(geo.thresholds)
    lo, hi = np.full(B, -span), np.full(B, span)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_many = _count(maps, mid, geo) > budget
        lo, hi = np.where(too_many, mid, lo), np.where(too_many, hi, mid)
    masks = _leaf_masks(maps, hi, geo)
    return [_tokens(masks, i, geo) for i in range(B)]


def training_layouts(images_u8, geo, offset_lo=0.0, offset_hi=2.0, anchor_p=0.15, scramble_p=0.10):
    """The training layout mixture, per image: the dense layout with probability `anchor_p`,
    otherwise the entropy quadtree at an offset drawn from
    U(offset_lo, offset_hi), with random entropy maps for a `scramble_p` share."""
    pyr = detail_pyramid(images_u8, geo)
    maps = _as_numpy(pyr)
    B = images_u8.shape[0]
    anchor, offsets = np.zeros(B, dtype=bool), np.zeros(B)
    for i in range(B):
        if torch.rand(1).item() < anchor_p:
            anchor[i] = True
            continue
        if torch.rand(1).item() < scramble_p:
            for s, v in pyr.items():
                maps[s][i] = (torch.rand_like(v[i].cpu()) * math.log2(geo.entropy_bins)).numpy()
        offsets[i] = offset_lo + (offset_hi - offset_lo) * torch.rand(1).item()
    masks = _leaf_masks(maps, offsets, geo)
    return [all_base_tokens(geo) if anchor[i] else _tokens(masks, i, geo) for i in range(B)]


# ---------------------------------------------------------------------------------------------------------
# packing: a padded batch the model reads
# ---------------------------------------------------------------------------------------------------------
def pack(token_lists, geo, device, multiple=64):
    """Pad the layouts of a batch to a common length (a multiple of `multiple`).

    sizes (B,N): token side in px, 0 where padded; coords (B,N,2): token centre in base-patch units (what
    the positions use); top_left (B,N,2): pixel corner; valid (B,N); n_per (B,): real tokens per image.
    """
    B, b = len(token_lists), geo.base_patch
    N = int(math.ceil(max(len(t) for t in token_lists) / multiple) * multiple)
    sizes = torch.zeros(B, N, dtype=torch.long)
    coords = torch.zeros(B, N, 2, dtype=torch.float32)
    tl = torch.zeros(B, N, 2, dtype=torch.long)
    valid = torch.zeros(B, N, dtype=torch.bool)
    for i, toks in enumerate(token_lists):
        a = np.asarray(toks, dtype=np.int64).reshape(-1, 3)
        s_, y_, x_ = a[:, 0], a[:, 1], a[:, 2]
        n, k = len(a), s_ / b
        sizes[i, :n] = torch.from_numpy(s_)
        tl[i, :n, 0], tl[i, :n, 1] = torch.from_numpy(y_), torch.from_numpy(x_)
        coords[i, :n, 0] = torch.from_numpy(((y_ / b) + (k - 1) / 2.0).astype(np.float32))
        coords[i, :n, 1] = torch.from_numpy(((x_ / b) + (k - 1) / 2.0).astype(np.float32))
        valid[i, :n] = True
    return {"sizes": sizes.to(device), "coords": coords.to(device), "top_left": tl.to(device),
            "valid": valid.to(device), "n_per": valid.sum(1).to(device)}
