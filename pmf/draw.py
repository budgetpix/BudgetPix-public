"""Training layouts. Per image: with prob anchor_p the dense grid; else the entropy quadtree of the clean image at a
threshold offset ~ U(offset_lo, offset_hi), its entropy maps replaced by random ones with prob scramble_p."""
import numpy as np
import torch
from layout import all_base_tokens, detail_pyramid, pyramid_slice, _pyr_np, _leaf_masks_np, tokens_from_masks, pack


def draw_layouts(images_u8, geo, *, offset_lo=0.0, offset_hi=2.0, anchor_p=0.15, scramble_p=0.10):
    B = images_u8.shape[0]
    pyr = detail_pyramid(images_u8, geo)
    hi_bits = float(np.log2(geo.entropy_bins))
    maps = _pyr_np(pyr)
    anchor = np.zeros(B, dtype=bool); deltas = np.zeros(B, dtype=np.float64)
    for i in range(B):
        if torch.rand(1).item() < anchor_p:
            anchor[i] = True
            continue
        p = pyramid_slice(pyr, i)
        if torch.rand(1).item() < scramble_p:
            for s, v in p.items():
                maps[s][i] = (torch.rand_like(v) * hi_bits).numpy().astype(np.float64)
        deltas[i] = offset_lo + (offset_hi - offset_lo) * torch.rand(1).item()
    masks = _leaf_masks_np(maps, deltas, geo)
    return [all_base_tokens(geo) if anchor[i] else tokens_from_masks(masks, i, geo) for i in range(B)]


def draw_packed(images_u8, geo, device, **kw):
    return pack(draw_layouts(images_u8, geo, **kw), geo, device, multiple=8)
