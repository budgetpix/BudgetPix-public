#!/usr/bin/env python
"""Export the FID reference set: N real images as PNG, at the generation resolution.

Use held-out data and the same N as the eval.

    python make_fid_ref.py --tars /data/shards_heldout --n 2000 --out eval/real_2k --size 1024
    python make_fid_ref.py --images /data/val --n 2000 --out eval/real_2k --size 1024
    python make_fid_ref.py --chunks /data/chunks_heldout --n 2000 --out eval/real_2k
"""
from __future__ import annotations

import argparse, glob, io, os, tarfile

import torch
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def square(im, size):
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((size, size), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=""); ap.add_argument("--images", default="")
    ap.add_argument("--tars", default="")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    k = 0
    if a.chunks:
        for f in sorted(glob.glob(os.path.join(a.chunks, "chunk_*.pt"))):
            c = torch.load(f, map_location="cpu", weights_only=False)
            pv = c["pixel_values"]
            if pv.dtype != torch.uint8:
                pv = ((pv.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            for i in range(pv.shape[0]):
                Image.fromarray(pv[i].permute(1, 2, 0).numpy()).save(os.path.join(a.out, f"{k:06d}.png")); k += 1
                if k >= a.n:
                    break
            if k >= a.n:
                break
    elif a.tars:
        for p in sorted(glob.glob(os.path.join(a.tars, "**", "*.tar"), recursive=True)):
            with tarfile.open(p) as tf:
                for m in tf:
                    if not m.isfile() or os.path.splitext(m.name)[1].lower() not in IMG_EXT:
                        continue
                    try:
                        im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
                    except Exception:
                        continue
                    square(im, a.size).save(os.path.join(a.out, f"{k:06d}.png")); k += 1
                    if k >= a.n:
                        break
            if k >= a.n:
                break
    else:
        files = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                       if p.lower().endswith(IMG_EXT))
        for p in files[: a.n]:
            square(Image.open(p).convert("RGB"), a.size).save(os.path.join(a.out, f"{k:06d}.png")); k += 1
    print(f"wrote {k} images to {a.out} at {a.size}x{a.size}")


if __name__ == "__main__":
    main()
