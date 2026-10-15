"""Token layouts: which parts of the image get fine tokens and which get coarse ones.

A layout is a list of tokens (size_px, y, x) that tiles the image. The finest token is the base patch
(16 px at 256, 32 px at 512); flat regions are merged into 2x2 or 4x4 blocks of base patches. The rule
follows APT (github.com/rccchoudhury/apt): a block merges when the Shannon entropy of its luma histogram
is below a per-scale threshold, and one per-image offset shifts both thresholds to hit a token budget.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

LUMA = (0.2989, 0.5870, 0.1140)
IN_CONTEXT_LEN = 32          # JiT's in-context class tokens, placed in front of every image's tokens


@dataclass(frozen=True)
class Geometry:
    img: int = 256
    base_patch: int = 16              # the finest token
    scales: tuple = (16, 32, 64)      # token sizes, base patch first, each double the previous
    thresholds: tuple = (6.0, 4.0)    # merge thresholds (bits) for scales[1:], finest first
    entropy_bins: int = 512

    def __post_init__(self):
        s = tuple(sorted(int(v) for v in self.scales))
        object.__setattr__(self, "scales", s)
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
    """gray (B,H,W) in [0,255] -> (B, H/cell, W/cell): entropy in bits of each cell's luma histogram."""
    B, H, W = gray.shape
    n, m = H // cell, W // cell
    cells = gray.reshape(B, n, cell, m, cell).permute(0, 1, 3, 2, 4).reshape(B, n, m, cell * cell)
    idx = (cells.clamp(0, 255) * (bins / 256.0)).long().clamp_(0, bins - 1)
    hist = torch.zeros(B, n, m, bins, device=gray.device)
    hist.scatter_add_(3, idx, torch.ones_like(idx, dtype=torch.float32))
    p = hist / (cell * cell)
    return -(p * torch.log2(p + 1e-10)).sum(-1)


def detail_pyramid(img, geo):
    """img (B,3,H,W) in [0,255] -> {s: (B, H/s, W/s) entropy} for every merge scale s."""
    x = img.float()
    gray = torch.nan_to_num(LUMA[0] * x[:, 0] + LUMA[1] * x[:, 1] + LUMA[2] * x[:, 2],
                            nan=0.0, posinf=0.0, neginf=0.0)
    return {s: _entropy(gray, s, geo.entropy_bins) for s in geo.scales[1:]}


# ---------------------------------------------------------------------------------------------------------
# quadtree and budget calibration
# ---------------------------------------------------------------------------------------------------------
def _leaf_masks(maps, offsets, geo):
    """maps {s: (B,g,g) float64}, offsets (B,) -> {s: (B,g,g) bool}, True where a token of size s sits.
    Walking from the coarsest scale down, a block that exists merges when its entropy is below the
    threshold plus the offset; otherwise its four children exist at the next scale."""
    B = len(offsets)
    g = geo.img // geo.scales[-1]
    alive = np.ones((B, g, g), dtype=bool)
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
    return sum(m.reshape(len(offsets), -1).sum(1) for m in _leaf_masks(maps, offsets, geo).values())


def _tokens(masks, i):
    return [(s, int(r) * s, int(c) * s) for s, m in masks.items() for r, c in zip(*np.nonzero(m[i]))]


def _as_numpy(pyr):
    return {s: v.detach().to("cpu", torch.float64).numpy() for s, v in pyr.items()}


def all_base_tokens(geo):
    """The dense layout: every token is a base patch."""
    p, g = geo.base_patch, geo.img // geo.base_patch
    return [(p, p * r, p * c) for r in range(g) for c in range(g)]


def layouts_for_budget(pyr, geo, budget, iters=24):
    """Per-image layouts with the largest token count that does not exceed `budget`. The offset is
    bisected per image; raising it merges more."""
    maps = _as_numpy(pyr)
    B = next(iter(maps.values())).shape[0]
    span = math.log2(geo.entropy_bins) + max(geo.thresholds)
    lo, hi = np.full(B, -span), np.full(B, span)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_many = _count(maps, mid, geo) > budget
        lo, hi = np.where(too_many, mid, lo), np.where(too_many, hi, mid)
    masks = _leaf_masks(maps, hi, geo)
    return [_tokens(masks, i) for i in range(B)]


def training_layouts(images_u8, geo, offset_lo=0.0, offset_hi=2.0, anchor_p=0.15, scramble_p=0.10):
    """The training layout mixture, per image: the dense layout with probability `anchor_p`, otherwise
    the entropy quadtree at an offset drawn from U(offset_lo, offset_hi), with random entropy maps for a
    `scramble_p` share."""
    pyr = detail_pyramid(images_u8, geo)
    maps = _as_numpy(pyr)
    B = images_u8.shape[0]
    anchor = np.zeros(B, dtype=bool)
    offsets = np.zeros(B)
    for i in range(B):
        if torch.rand(1).item() < anchor_p:
            anchor[i] = True
            continue
        if torch.rand(1).item() < scramble_p:
            for s, v in pyr.items():
                maps[s][i] = (torch.rand_like(v[i]) * math.log2(geo.entropy_bins)).numpy()
        offsets[i] = offset_lo + (offset_hi - offset_lo) * torch.rand(1).item()
    masks = _leaf_masks(maps, offsets, geo)
    return [all_base_tokens(geo) if anchor[i] else _tokens(masks, i) for i in range(B)]


# ---------------------------------------------------------------------------------------------------------
# packing: the variable-length batch the model reads
# ---------------------------------------------------------------------------------------------------------
def pack(token_lists, geo, device="cpu"):
    """Token lists -> one packed sequence for the whole batch (flash-attn varlen style).

    sizes, coords (T, 2): token size and centre in base-patch units (a base patch at cell (r, c) sits at
    the integer (r, c)); idx{s}: index of the token in the s-pixel grid;
    tok_batch: image of each token; cu_img / max_n: segment offsets for the image-only layers;
    the *_ic entries: the same with the in-context class tokens in front of each image.
    """
    b, icl = geo.base_patch, IN_CONTEXT_LEN
    sizes, coords, tok_batch, n_per = [], [], [], []
    idx = {s: [] for s in geo.scales}
    for i, toks in enumerate(token_lists):
        toks = sorted(toks, key=lambda t: (t[1], t[2]))
        s_ = torch.tensor([t[0] for t in toks], dtype=torch.long)
        y_ = torch.tensor([t[1] for t in toks], dtype=torch.long)
        x_ = torch.tensor([t[2] for t in toks], dtype=torch.long)
        sizes.append(s_)
        coords.append(torch.stack([(y_.float() + s_.float() / 2) / b - 0.5,
                                   (x_.float() + s_.float() / 2) / b - 0.5], 1))
        for s in geo.scales:
            j = torch.full((len(toks),), -1, dtype=torch.long)
            m = s_ == s
            j[m] = (y_[m] // s) * (geo.img // s) + x_[m] // s
            idx[s].append(j)
        tok_batch.append(torch.full((len(toks),), i, dtype=torch.long))
        n_per.append(len(toks))
    B, n_per = len(token_lists), torch.tensor(n_per, dtype=torch.long)
    cu_img = torch.zeros(B + 1, dtype=torch.int32)
    cu_img[1:] = torch.cumsum(n_per, 0).to(torch.int32)
    cu_ic = torch.zeros(B + 1, dtype=torch.int32)
    cu_ic[1:] = torch.cumsum(n_per + icl, 0).to(torch.int32)
    Tic = int((n_per + icl).sum())
    img2ic = torch.empty(int(n_per.sum()), dtype=torch.long)
    ic_slots = torch.empty(B * icl, dtype=torch.long)
    tok_batch_ic = torch.empty(Tic, dtype=torch.long)
    for i in range(B):
        off, n, s = int(cu_ic[i]), int(n_per[i]), int(cu_img[i])
        ic_slots[i * icl:(i + 1) * icl] = torch.arange(off, off + icl)
        img2ic[s:s + n] = torch.arange(off + icl, off + icl + n)
        tok_batch_ic[off:off + icl + n] = i
    L = {"sizes": torch.cat(sizes), "coords": torch.cat(coords), "tok_batch": torch.cat(tok_batch),
         "n_per": n_per, "cu_img": cu_img, "max_n": int(n_per.max()), "cu_ic": cu_ic,
         "max_nic": int((n_per + icl).max()), "Tic": Tic, "img2ic": img2ic, "ic_slots": ic_slots,
         "tok_batch_ic": tok_batch_ic, **{f"idx{s}": torch.cat(v) for s, v in idx.items()}}
    return to_device(L, device)


def to_device(L, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in L.items()}


# ---------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------
PALETTE = {16: (231, 76, 60), 32: (241, 196, 15), 64: (52, 152, 219), 128: (46, 204, 113)}


def draw_layout(img, tokens, alpha=0.35):
    """img (3,H,W) in [0,1] -> the image with each token tinted by its size and its border drawn."""
    out = img.float().clone() * 255
    for s, y, x in tokens:
        c = torch.tensor(PALETTE.get(s, (155, 89, 182)), dtype=torch.float32)[:, None, None]
        out[:, y:y + s, x:x + s] = (1 - alpha) * out[:, y:y + s, x:x + s] + alpha * c
    for s, y, x in tokens:
        out[:, y:y + 1, x:x + s] = 255
        out[:, y:y + s, x:x + 1] = 255
    out[:, -1:, :] = 255
    out[:, :, -1:] = 255
    return out.clamp(0, 255) / 255
