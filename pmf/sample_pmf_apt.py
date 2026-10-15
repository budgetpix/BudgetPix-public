#!/usr/bin/env python
"""One-step (1 NFE) class-conditional samples from a BudgetPix pMF checkpoint at a token budget.

    PMF_ROOT=/path/to/pMF python sample_pmf_apt.py --ckpt ../checkpoints/budgetpix_pmf_l16_256.pt \
        --classes 207,88,417,933 --budget 128 --out samples.png

The layout is chosen up front:
  --layout self    (default) the entropy layout of the model's own dense one-step sample for the same class and noise,
                   bisected to the budget (costs one extra dense forward)
  --layout image   the entropy layout of the image(s) given with --layout_from (one per sample, cycled)
  --layout random  a content-free random entropy map
--budget >= 256 is the dense grid. --nfe 2 adds a second step whose layout comes from the first step's own x-hat.
Guidance defaults are the released pMF-L/16 settings (omega 7.0 on the CFG interval [0.2, 0.7]); for B/16 use 8.5 and
[0.1, 0.7]. The released model driven at a reduced budget without training: --ckpt released --released pMF-L-16.pt.
"""
from __future__ import annotations

import argparse, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upstream import add_pmf_root                                   # noqa: E402
add_pmf_root()
from eval_pmf_apt import build, make_L, NOISE_SCALE, cell_scale     # noqa: E402
from model_pmf_apt import dense_L                                   # noqa: E402
from layout import pack, detail_pyramid, _pyr_np, calibrate_offsets_batched, _leaf_masks_np, tokens_from_masks  # noqa: E402


def layouts_from_images(paths, idx, geo, budget, dev):
    """Entropy pyramids of user images (centre-cropped to 256), bisected to the budget: one layout per sample."""
    from PIL import Image
    from data_imagenet import center_crop_arr
    ims = []
    for j in range(len(idx)):
        im = Image.open(paths[j % len(paths)]).convert("RGB")
        ims.append(torch.from_numpy(np.array(center_crop_arr(im, geo.img), copy=True)).permute(2, 0, 1))
    maps = _pyr_np(detail_pyramid(torch.stack(ims), geo))
    off = calibrate_offsets_batched(maps, geo, budget)
    masks = _leaf_masks_np(maps, off, geo)
    return pack([tokens_from_masks(masks, j, geo) for j in range(len(idx))], geo, dev, multiple=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="BudgetPix pMF checkpoint, or 'released' (with --released)")
    ap.add_argument("--released", default="", help="released pMF weights, for --ckpt released (default: $PMF_WEIGHTS)")
    ap.add_argument("--arch", default="L16", choices=["B16", "L16", "H16"])
    ap.add_argument("--classes", default="207,88,417,933", help="ImageNet class ids, one sample each")
    ap.add_argument("--n", type=int, default=1, help="samples per class")
    ap.add_argument("--budget", type=int, default=256, help="tokens per image (256 = dense, 128 = 50 %%)")
    ap.add_argument("--layout", default="self", choices=["self", "image", "random"])
    ap.add_argument("--layout_from", nargs="*", default=[], help="image file(s) for --layout image")
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--omega", type=float, default=7.0); ap.add_argument("--tmin", type=float, default=0.2); ap.add_argument("--tmax", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="noise of sample k is pMF's BatchGenerator(seed + k)")
    ap.add_argument("--overlay", action="store_true", help="also write <out>_layout.png with the token layout drawn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="samples.png")
    a = ap.parse_args()
    dev = a.device
    model = build(a.arch, a.ckpt, released=a.released).to(dev).eval()
    geo = model.geo
    classes = [int(c) for c in a.classes.split(",")]
    labels = torch.tensor([c for c in classes for _ in range(a.n)], device=dev)
    B = labels.shape[0]
    idx = [a.seed + k for k in range(B)]
    from utils.torch_util import BatchGenerator
    z = BatchGenerator(dev, idx).randn((B, 3, geo.img, geo.img)).float() * NOISE_SCALE[a.arch]
    ts = torch.linspace(1.0, 0.0, a.nfe + 1, device=dev)
    full = lambda v: torch.full((B,), v, device=dev)
    xhat, L0, toks = None, None, []
    reduced = a.budget < geo.n_dense
    with torch.no_grad():
        if a.layout == "self" and reduced:                                  # the dense pre-pass that gives the layout
            u = model(z, full(1.0), full(1.0), full(a.omega), full(a.tmin), full(a.tmax), labels, dense_L(B, geo, dev))[0]
            xhat = z - u
        for i in range(a.nfe):
            t, r = ts[i].expand(B), ts[i + 1].expand(B)
            if i == 0:
                if not reduced:
                    L = dense_L(B, geo, dev)
                elif a.layout == "image":
                    if not a.layout_from:
                        raise SystemExit("--layout image needs --layout_from <image> ...")
                    L = layouts_from_images(a.layout_from, idx, geo, a.budget, dev)
                else:
                    L = make_L(a.layout, idx, geo, a.budget, dev, labels, xhat)
                L0 = L
            else:
                L = make_L("xhat", idx, geo, a.budget, dev, labels, xhat)
            toks.append(float(L["n_per"].float().mean()))
            u = model(z, t, t - r, full(a.omega), full(a.tmin), full(a.tmax), labels, L)[0]
            xhat = z - t.view(B, 1, 1, 1) * u
            z = z - (t - r).view(B, 1, 1, 1) * u
    imgs = ((z.float().clamp(-1, 1) + 1) / 2).cpu()
    from torchvision.utils import save_image
    save_image(imgs, a.out, nrow=len(classes), padding=2)
    print(f"tokens/img per step: {[round(v, 1) for v in toks]} of {geo.n_dense} dense; wrote {a.out}")
    if a.overlay:
        side = cell_scale(L0, geo).cpu()                                    # (B, g, g) token side per base cell
        g, b = geo.img // geo.base_patch, geo.base_patch
        ov = imgs.clone()
        tint = {geo.scales[1]: torch.tensor([1.0, 0.8, 0.0]), geo.scales[2]: torch.tensor([0.9, 0.2, 0.2])}
        for k in range(B):
            for gy in range(g):
                for gx in range(g):
                    s = int(side[k, gy, gx]); y0, x0 = gy * b, gx * b
                    if s in tint:
                        ov[k, :, y0:y0 + b, x0:x0 + b] = 0.65 * ov[k, :, y0:y0 + b, x0:x0 + b] + 0.35 * tint[s].view(3, 1, 1)
                    ov[k, :, y0, x0:x0 + b] = 1.0; ov[k, :, y0:y0 + b, x0] = 1.0
        path = os.path.splitext(a.out)[0] + "_layout.png"
        save_image(ov, path, nrow=len(classes), padding=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
