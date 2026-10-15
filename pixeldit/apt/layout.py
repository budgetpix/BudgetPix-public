"""Adaptive token layouts: entropy quadtree, budget calibration, and packing.

A layout is a list of (size_px, y, x) tokens per image. Following APT, a block merges into scale s
when the Shannon entropy (bits) of its luma is below the threshold for s; the base patch is the
finest scale. Thresholds are finest-first (same order as scales[1:]) and a per-image offset shifts
them all; raising it yields fewer tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

LUMA = (0.2989, 0.5870, 0.1140)


@dataclass(frozen=True)
class Geometry:
    """The token geometry. `scales` is finest-first and each entry doubles the previous."""
    img: int = 512
    base_patch: int = 16
    scales: tuple = (16, 32, 64)
    thresholds: tuple = None                   # one per merge step, finest-first; None -> 6, 4, 2, 1 bits
    entropy_bins: int = 512

    def __post_init__(self):
        s = tuple(sorted(int(x) for x in self.scales))
        object.__setattr__(self, "scales", s)
        for a, b in zip(s, s[1:]):
            if b != 2 * a:
                raise ValueError(f"each scale must be double the previous, got {s}")
        if self.base_patch != s[0]:
            raise ValueError(f"base_patch {self.base_patch} must be the FINEST scale of {s} "
                             f"(layouts only merge; the base patch is the finest scale)")
        if self.img % s[-1]:
            raise ValueError(f"image size {self.img} must be divisible by the coarsest scale {s[-1]}")
        thr = self.thresholds if self.thresholds is not None else (6.0, 4.0, 2.0, 1.0)[:len(s) - 1]
        thr = tuple(float(t) for t in thr)
        if len(thr) != len(s) - 1:
            raise ValueError(f"scales {s} need {len(s)-1} thresholds, got {thr}")
        object.__setattr__(self, "thresholds", thr)

    @property
    def thresholds_by_scale(self):                 # {scale: threshold}, coarsest scales only
        return dict(zip(self.scales[1:], self.thresholds))

    @property
    def n_dense(self):
        return (self.img // self.base_patch) ** 2

    @property
    def n_min(self):
        return (self.img // self.scales[-1]) ** 2

    def grid(self, s):
        return self.img // s


# ---------------------------------------------------------------------------------------------
# entropy pyramid
# ---------------------------------------------------------------------------------------------
def _entropy_cells(gray, cell, bins):
    """Shannon entropy in bits of each `cell`x`cell` block's own intensity histogram. gray:(B,H,W) 0..255."""
    B, H, W = gray.shape
    g = (gray.clamp(0, 255) * (bins / 256.0)).long().clamp_(0, bins - 1)
    g = g.view(B, H // cell, cell, W // cell, cell).permute(0, 1, 3, 2, 4).reshape(B, -1, cell * cell)
    hist = torch.zeros(B, g.shape[1], bins, device=gray.device, dtype=torch.float32)
    hist.scatter_add_(2, g, torch.ones_like(g, dtype=torch.float32))
    p = hist / (cell * cell)
    return -(p * torch.log2(p.clamp_min(1e-12))).sum(-1)          # (B, ncell)


def detail_pyramid(images_u8, geo: Geometry):
    """{scale: (B, g, g) entropy in bits} for every scale ABOVE the base patch."""
    x = images_u8.float()
    gray = LUMA[0] * x[:, 0] + LUMA[1] * x[:, 1] + LUMA[2] * x[:, 2]
    gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
    out = {}
    for s in geo.scales[1:]:
        g = geo.grid(s)
        out[s] = _entropy_cells(gray, s, geo.entropy_bins).view(-1, g, g)
    return out


def pyramid_slice(pyr, i):
    """One image's maps, on the CPU."""
    return {s: v[i].detach().cpu() for s, v in pyr.items()}


def thresholds_at(geo: Geometry, delta=0.0):
    return {s: t + delta for s, t in geo.thresholds_by_scale.items()}


# ---------------------------------------------------------------------------------------------
# quadtree
# ---------------------------------------------------------------------------------------------
def quadtree(pyr1, thr, geo: Geometry):
    """One image's entropy maps -> [(size_px, y, x)]. Coarsest-first: a block merges when its
    entropy is below that scale's threshold; unmerged area descends to the next finer scale."""
    scales = list(geo.scales)
    coarsest = scales[-1]
    toks = []
    pending = [(coarsest, r, c) for r in range(geo.grid(coarsest)) for c in range(geo.grid(coarsest))]
    while pending:
        nxt = []
        for (s, r, c) in pending:
            if s == geo.base_patch:
                toks.append((s, r * s, c * s))
                continue
            e = float(pyr1[s][r, c])
            if e < thr[s]:
                toks.append((s, r * s, c * s))
            else:
                h = s // 2
                nxt += [(h, 2 * r, 2 * c), (h, 2 * r, 2 * c + 1), (h, 2 * r + 1, 2 * c), (h, 2 * r + 1, 2 * c + 1)]
        pending = nxt
    return toks


def count_tokens(pyr1, thr, geo: Geometry):
    return len(quadtree(pyr1, thr, geo))


def calibrate_offset(pyr1, geo: Geometry, target, iters=24):
    """Bisect a common threshold offset so the quadtree yields ~`target` tokens (never overshooting)."""
    if target < geo.n_min:
        raise ValueError(f"target {target} is below the coarsest grid's {geo.n_min} tokens; a merge-only "
                         f"quadtree cannot go lower (it would silently return {geo.n_min})")
    base = geo.thresholds_by_scale
    span = math.log2(geo.entropy_bins) + max(base.values())
    lo, hi = -span, span
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if count_tokens(pyr1, {s: t + mid for s, t in base.items()}, geo) > target:
            lo = mid
        else:
            hi = mid
    return hi


def all_base_tokens(geo: Geometry):
    g = geo.grid(geo.base_patch)
    p = geo.base_patch
    return [(p, p * r, p * c) for r in range(g) for c in range(g)]


def layout_for(images_u8, geo: Geometry, budget=None, offset=0.0):
    """Per-image layouts for a uint8 batch (B,3,H,W). `budget` bisects per image; else a fixed offset."""
    return layouts_batched(detail_pyramid(images_u8, geo), geo, budget=budget, offsets=offset)


# ---------------------------------------------------------------------------------------------
# batched quadtree + budget calibration
# ---------------------------------------------------------------------------------------------

def _pyr_np(pyr):
    """{s: (B,g,g) float64 numpy}."""
    return {s: v.detach().to("cpu", torch.float64).numpy() for s, v in pyr.items()}


def _leaf_masks_np(pyr_np, deltas, geo: Geometry):
    """deltas: (B,) offsets. -> {scale: (B,g,g) bool leaf masks} for every scale, coarse -> fine."""
    scales = list(geo.scales)                       # ascending (finest first)
    base = geo.thresholds_by_scale
    B = next(iter(pyr_np.values())).shape[0]
    coarsest = scales[-1]; g = geo.grid(coarsest)
    alive = np.ones((B, g, g), dtype=bool)
    out = {}
    for s in reversed(scales):                      # coarse -> fine
        if s == geo.base_patch:
            out[s] = alive; break
        thr = (base[s] + deltas)[:, None, None]     # (B,1,1)
        merged = alive & (pyr_np[s] < thr)
        out[s] = merged
        alive = np.repeat(np.repeat(alive & ~merged, 2, 1), 2, 2)   # children of split blocks
    return out


def _count_np(pyr_np, deltas, geo: Geometry):
    return sum(m.reshape(m.shape[0], -1).sum(1) for m in _leaf_masks_np(pyr_np, deltas, geo).values())


def calibrate_offsets_batched(pyr_np, geo: Geometry, target, iters=24):
    """(B,) offsets: calibrate_offset for every image at once."""
    if target < geo.n_min:
        raise ValueError(f"target {target} is below the coarsest grid's {geo.n_min} tokens")
    base = geo.thresholds_by_scale
    span = math.log2(geo.entropy_bins) + max(base.values())
    B = next(iter(pyr_np.values())).shape[0]
    lo = np.full(B, -span, dtype=np.float64); hi = np.full(B, span, dtype=np.float64)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_many = _count_np(pyr_np, mid, geo) > target
        lo = np.where(too_many, mid, lo); hi = np.where(too_many, hi, mid)
    return hi


def _morton_order(r, c, depth):
    """Key that sorts blocks (r,c) of a level `depth` below the coarsest by (parent order, child index),
    recursively -- the breadth-first emission order of the per-image quadtree."""
    key = np.zeros_like(r)
    for d in range(depth, -1, -1):                  # from the coarsest ancestor down to the block itself
        key = key * 4 + (((r >> d) & 1) << 1 | ((c >> d) & 1))
    top = (r >> (depth + 1)) * (1 << 30) + (c >> (depth + 1))   # row-major over the coarsest grid, first
    return top * (1 << (2 * (depth + 1))) + key


def tokens_from_masks(masks, i, geo: Geometry):
    """Image i's [(size, y, x)] in the per-image quadtree's order."""
    scales = list(geo.scales); toks = []
    for depth, s in enumerate(reversed(scales)):     # coarse -> fine; depth = levels below the coarsest
        m = masks[s][i]
        if not m.any(): continue
        r, c = np.nonzero(m)
        order = np.argsort(_morton_order(r, c, depth - 1), kind="stable") if depth > 0 else np.lexsort((c, r))
        toks += [(s, int(rr) * s, int(cc) * s) for rr, cc in zip(r[order], c[order])]
    return toks


def layouts_batched(pyr, geo: Geometry, budget=None, offsets=None):
    """Per-image token lists for a whole batch. `budget` bisects per image; else `offsets` (scalar or (B,))."""
    pn = _pyr_np(pyr)
    B = next(iter(pn.values())).shape[0]
    if budget is not None:
        d = calibrate_offsets_batched(pn, geo, budget)
    else:
        d = np.broadcast_to(np.asarray(offsets if offsets is not None else 0.0, dtype=np.float64), (B,)).astype(np.float64)
    masks = _leaf_masks_np(pn, d, geo)
    return [tokens_from_masks(masks, i, geo) for i in range(B)]


# ---------------------------------------------------------------------------------------------
# packing: variable-length layouts -> padded tensors + mask
# ---------------------------------------------------------------------------------------------
def pack(token_lists, geo: Geometry, device, multiple=64):
    """Pad a batch of layouts to a common length and return everything the model needs.

    Returns dict with
      sizes   (B,N) int64  token side length in px, 0 where padded
      coords  (B,N,2) float  token centre in base-patch units (used by RoPE)
      top_left(B,N,2) int64  pixel coordinates of the token's top-left corner
      valid   (B,N) bool
      n_per   (B,) int64     real token count per image
    """
    B = len(token_lists)
    N = max(len(t) for t in token_lists)
    N = int(math.ceil(N / multiple) * multiple)
    b = geo.base_patch
    sizes = torch.zeros(B, N, dtype=torch.long)
    coords = torch.zeros(B, N, 2, dtype=torch.float32)
    tl = torch.zeros(B, N, 2, dtype=torch.long)
    valid = torch.zeros(B, N, dtype=torch.bool)
    ok_scales = set(geo.scales)
    for i, toks in enumerate(token_lists):
        if not toks:
            continue
        a = np.asarray(toks, dtype=np.int64).reshape(-1, 3)             # (n,3) = (s, y, x)
        s_, y_, x_ = a[:, 0], a[:, 1], a[:, 2]
        bad = ~np.isin(s_, list(ok_scales)) | (y_ < 0) | (x_ < 0) | (y_ + s_ > geo.img) | (x_ + s_ > geo.img) | (y_ % s_ != 0) | (x_ % s_ != 0)
        assert not bad.any(), f"invalid token {tuple(int(v) for v in a[bad][0])} for geometry {geo}"
        n = a.shape[0]
        sizes[i, :n] = torch.from_numpy(s_)
        tl[i, :n, 0] = torch.from_numpy(y_); tl[i, :n, 1] = torch.from_numpy(x_)
        k = s_ / b
        coords[i, :n, 0] = torch.from_numpy(((y_ / b) + (k - 1) / 2.0).astype(np.float32))   # centre, base-patch units
        coords[i, :n, 1] = torch.from_numpy(((x_ / b) + (k - 1) / 2.0).astype(np.float32))
        valid[i, :n] = True
    return {
        "sizes": sizes.to(device), "coords": coords.to(device), "top_left": tl.to(device),
        "valid": valid.to(device), "n_per": valid.sum(1).to(device),
        "img": geo.img, "base_patch": b, "scales": geo.scales,
    }
