"""Training data streams.

Three input forms, all yielding images in [-1,1] (B,3,1024,1024), Gemma embeddings (B,300,2304)
and mask (B,300):

  1. `chunks`   folders of `chunk_*.pt` written by `precompute_data.py` (uint8 pixels and encoded
                captions). Recommended.
  2. `tars`     webdataset shards (`*.tar` holding an image and a `.json` per key). Captions come from
                the external caption jsons beside a shard (see CAPTION_SUFFIXES), else from the
                sample's own json, and are encoded on the fly.
  3. `images`   a folder of images with sidecar captions (`x.jpg` + `x.txt` or `x.json`), or a single
                `captions.json` mapping file name -> caption.

Images are resized short-side and centre-cropped to a square `img`.
"""
from __future__ import annotations

import io
import json
import os
import random
import tarfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
CAP_KEYS = ("prompt", "caption", "text", "blip_caption", "sharegpt4v")
# A shard `x.tar` may have `x_InternVL2-26B.json` etc. beside it, keyed by sample name; the first
# external caption found in suffix order is used, else the sample's own json.
CAPTION_SUFFIXES = ("", "_InternVL2-26B", "_VILA1-5-13B")


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0


def _world():
    return torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1


def _shard(files, name, rank, world):
    """Split a file list over ranks x dataloader workers."""
    w = torch.utils.data.get_worker_info()
    shard, n = rank, world
    if w is not None:
        shard = rank * w.num_workers + w.id
        n = world * w.num_workers
    out = files[shard::n]
    if not out:
        raise RuntimeError(f"{len(files)} {name} is fewer than ranks x workers = {n}; lower data.num_workers")
    return out


def _square(im, size):
    from PIL import Image
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return im.resize((size, size), Image.BICUBIC)


def _to_u8(im, size):
    import numpy as np
    a = np.array(_square(im.convert("RGB"), size), dtype="uint8")
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def _caption_from_json(obj):
    if isinstance(obj, str):
        return obj
    for k in CAP_KEYS:
        v = obj.get(k) if isinstance(obj, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0].strip()
    return ""


# ---------------------------------------------------------------------------------- 1. chunks
class ChunkStream(IterableDataset):
    """`chunk_*.pt`: pixel_values (n,3,H,W) uint8, y (n,300,2304) fp16, y_mask (n,300)."""

    def __init__(self, roots, seed=0):
        self.roots = [Path(r) for r in (roots if isinstance(roots, (list, tuple)) else [roots])]
        self.seed = int(seed)
        self.rank, self.world = _rank(), _world()
        self.files = sorted(f for r in self.roots for f in r.glob("chunk_*.pt"))
        if not self.files:
            raise RuntimeError(f"no chunk_*.pt under {[str(r) for r in self.roots]} — run precompute_data.py")

    def __iter__(self):
        files = _shard(self.files, "chunks", self.rank, self.world)
        w = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + self.rank * 1000 + (w.id if w is not None else 0))
        while True:
            order = list(files); rng.shuffle(order)
            for path in order:
                c = torch.load(path, map_location="cpu", weights_only=False)
                n = int(c["pixel_values"].shape[0])
                idx = list(range(n)); rng.shuffle(idx)
                for i in idx:
                    yield {"pixel_values": c["pixel_values"][i],
                           "y": c["y"][i].float(),
                           "y_mask": c["y_mask"][i].long()}


# ---------------------------------------------------------------------------------- 2. tars
class TarStream(IterableDataset):
    """Webdataset shards. Yields uint8 pixels + the raw caption."""

    def __init__(self, roots, img=1024, seed=0, caption_suffixes=CAPTION_SUFFIXES, once=False):
        roots = roots if isinstance(roots, (list, tuple)) else [roots]
        self.files = sorted(str(p) for r in roots for p in Path(r).rglob("*.tar"))
        if not self.files:
            raise RuntimeError(f"no *.tar under {roots}")
        self.img, self.seed = int(img), int(seed)
        self.rank, self.world = _rank(), _world()
        self.caption_suffixes = tuple(caption_suffixes)
        # once=True: every shard exactly once, in order, then stop.
        self.once = bool(once)

    def _external(self, tar_path):
        """External-caption jsons next to a shard (`shard_InternVL2-26B.json`)."""
        out = {}
        for suf in self.caption_suffixes:
            if not suf:
                continue
            p = tar_path.replace(".tar", f"{suf}.json")
            if os.path.exists(p):
                try:
                    out.update(json.load(open(p)))
                except Exception:
                    pass
        return out

    def __iter__(self):
        from PIL import Image
        files = _shard(self.files, "tars", self.rank, self.world)
        w = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + self.rank * 1000 + (w.id if w is not None else 0))
        while True:
            order = list(files)
            if not self.once:
                rng.shuffle(order)
            for path in order:
                ext = self._external(path)
                try:
                    tf = tarfile.open(path)
                except Exception:
                    continue
                with tf:
                    pending = {}
                    for m in tf:
                        if not m.isfile():
                            continue
                        key, e = os.path.splitext(m.name)
                        d = pending.setdefault(key, {})
                        try:
                            d[e.lower()] = tf.extractfile(m).read()
                        except Exception:
                            continue
                        img_e = next((k for k in d if k in IMG_EXT), None)
                        if img_e and ".json" in d:
                            cap = ""
                            base = os.path.basename(key)
                            if base in ext:
                                cap = _caption_from_json(ext[base])
                            if not cap:
                                try:
                                    cap = _caption_from_json(json.loads(d[".json"]))
                                except Exception:
                                    cap = ""
                            pending.pop(key, None)
                            if not cap:
                                continue
                            try:
                                im = Image.open(io.BytesIO(d[img_e]))
                                yield {"pixel_values": _to_u8(im, self.img), "caption": cap}
                            except Exception:
                                continue
            if self.once:
                return


# ---------------------------------------------------------------------------------- 3. images
class ImageFolderStream(IterableDataset):
    def __init__(self, roots, img=1024, seed=0, once=False):
        roots = roots if isinstance(roots, (list, tuple)) else [roots]
        self.files = sorted(str(p) for r in roots for p in Path(r).rglob("*")
                            if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise RuntimeError(f"no images under {roots}")
        self.img, self.seed = int(img), int(seed)
        self.rank, self.world = _rank(), _world()
        self.once = bool(once)
        self.maps = {}
        for r in roots:
            p = Path(r) / "captions.json"
            if p.exists():
                self.maps.update(json.load(open(p)))

    def _caption(self, path):
        base = os.path.basename(path)
        if base in self.maps:
            return _caption_from_json(self.maps[base])
        stem = os.path.splitext(path)[0]
        for e in (".txt", ".json"):
            if os.path.exists(stem + e):
                raw = open(stem + e).read()
                return raw.strip() if e == ".txt" else _caption_from_json(json.loads(raw))
        return ""

    def __iter__(self):
        from PIL import Image
        files = _shard(self.files, "images", self.rank, self.world)
        w = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + self.rank * 1000 + (w.id if w is not None else 0))
        while True:
            order = list(files)
            if not self.once:
                rng.shuffle(order)
            for p in order:
                cap = self._caption(p)
                if not cap:
                    continue
                try:
                    yield {"pixel_values": _to_u8(Image.open(p), self.img), "caption": cap}
                except Exception:
                    continue
            if self.once:
                return


# ---------------------------------------------------------------------------------- assembly
def collate(batch):
    out = {"pixel_values": torch.stack([b["pixel_values"] for b in batch])}
    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])
        out["y_mask"] = torch.stack([b["y_mask"] for b in batch])
    else:
        out["caption"] = [b["caption"] for b in batch]
    return out


def build_dataset(d, img):
    if d.get("chunks"):
        return ChunkStream(d["chunks"], seed=d.get("seed", 0)), True
    if d.get("tars"):
        return TarStream(d["tars"], img=img, seed=d.get("seed", 0),
                         caption_suffixes=tuple(d.get("caption_suffixes", CAPTION_SUFFIXES))), False
    if d.get("images"):
        return ImageFolderStream(d["images"], img=img, seed=d.get("seed", 0)), False
    raise SystemExit("set data.chunks (recommended), data.tars or data.images in the config")


def make_loader(d, img, micro_batch_size, num_workers=8, prefetch=2):
    ds, precomputed = build_dataset(d, img)
    return DataLoader(ds, batch_size=micro_batch_size, num_workers=num_workers, collate_fn=collate,
                      pin_memory=False, persistent_workers=num_workers > 0,
                      prefetch_factor=prefetch if num_workers > 0 else None, drop_last=True), precomputed


def prepare_images(pixel_values, device):
    """uint8 -> [-1,1] float on `device`; float inputs are assumed already in [-1,1]."""
    x = pixel_values.to(device, non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.to(torch.float32).mul_(1.0 / 127.5).add_(-1.0)
    return x
