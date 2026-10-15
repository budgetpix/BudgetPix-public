#!/usr/bin/env python3
"""Build MiniT2I's training data locally: CC12M + LLaVA-NeXT recaptions, high-quality subset.

Reads captions + URLs from CaptionEmporium/conceptual-captions-cc12m-llavanext and downloads the images.
Keeps rows with status == success, a non-empty `caption_llava`, and an image short side >= MIN_SIDE.
Writes chunk_*.pt: pixel_values uint8 (n,3,512,512) [Resize(512, bicubic) + CenterCrop(512)],
input_ids/attention_mask int64 (n,256) from flan-t5-large, caption list[str], key list[str].
Resumable: existing chunks are counted and skipped.

  CC12M_OUT=/data/cc12m512 python cc12m_build.py --n-train 1000000 --procs 32
"""
from __future__ import annotations
import argparse, gzip, io, itertools, json, os, pathlib, time
import multiprocessing as mp
import urllib.request

OUT = pathlib.Path(os.environ.get("CC12M_OUT", "data/cc12m512"))
INDEX_REPO = "CaptionEmporium/conceptual-captions-cc12m-llavanext"
UA = {"User-Agent": "Mozilla/5.0 (research dataset preparation)"}
_MIN_SIDE, _SIZE = 512, 512


def _init(min_side, size):
    global _MIN_SIDE, _SIZE
    _MIN_SIDE, _SIZE = min_side, size


def fetch(row):
    """-> (key, HxWx3 uint8 bytes, caption) or None."""
    try:
        import numpy as np
        from PIL import Image
        req = urllib.request.Request(row["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read(12_000_000)
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGB")
        if min(im.size) < _MIN_SIDE:
            return None
        w, h = im.size
        s = _SIZE / min(w, h)
        im = im.resize((max(_SIZE, round(w * s)), max(_SIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - _SIZE) // 2, (h - _SIZE) // 2
        im = im.crop((l, t, l + _SIZE, t + _SIZE))
        return row["key"], np.asarray(im, dtype="uint8").tobytes(), row["caption_llava"]
    except Exception:
        return None


def qualifying(idx, min_side):
    with gzip.open(idx, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "success" or not r.get("caption_llava"):
                continue
            try:
                if min(int(r["width"]), int(r["height"])) < min_side:
                    continue
            except Exception:
                continue
            yield {"url": r["url"], "key": r["key"], "caption_llava": r["caption_llava"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1000000)
    ap.add_argument("--n-heldout", type=int, default=5000)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--per-chunk", type=int, default=1024)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 8, help="download/decode processes")
    ap.add_argument("--prompt-len", type=int, default=256)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    (OUT / "chunks_512").mkdir(parents=True, exist_ok=True)
    (OUT / "chunks_heldout").mkdir(parents=True, exist_ok=True)

    n_ho = len(list((OUT / "chunks_heldout").glob("chunk_*.pt")))
    n_tr = len(list((OUT / "chunks_512").glob("chunk_*.pt")))
    have_ho, have_tr = n_ho * a.per_chunk, n_tr * a.per_chunk
    need_ho, need_tr = max(0, a.n_heldout - have_ho), max(0, a.n_train - have_tr)
    print(f"[cc12m] have {have_tr} train + {have_ho} held-out; need {need_tr} + {need_ho}", flush=True)
    if not need_tr and not need_ho:
        (OUT / ".build.done").touch()
        print("[cc12m] complete", flush=True)
        return 0

    import huggingface_hub as H
    idx = H.hf_hub_download(INDEX_REPO, "train.jsonl.gz", repo_type="dataset")
    rows = qualifying(idx, a.min_side)
    skip = have_tr + have_ho
    if skip:
        print(f"[cc12m] skipping {skip} qualifying rows already consumed", flush=True)
        for _ in itertools.islice(rows, skip):
            pass

    def save(buf, dest, n):
        px = torch.from_numpy(
            np.frombuffer(b"".join(b[1] for b in buf), dtype=np.uint8)
              .reshape(len(buf), a.size, a.size, 3).copy()
        ).permute(0, 3, 1, 2).contiguous()
        caps = [b[2] for b in buf]
        enc = tok(caps, padding="max_length", truncation=True, max_length=a.prompt_len, return_tensors="pt")
        tmp = dest / f".chunk_{n:06d}.tmp"
        torch.save({"pixel_values": px, "input_ids": enc["input_ids"].long(),
                    "attention_mask": enc["attention_mask"].long(), "caption": caps,
                    "key": [b[0] for b in buf]}, tmp)
        tmp.rename(dest / f"chunk_{n:06d}.pt")

    t0 = time.time()
    seen = 0
    with mp.Pool(a.procs, initializer=_init, initargs=(a.min_side, a.size)) as pool:
        results = pool.imap_unordered(fetch, rows, chunksize=4)
        for split, dest, need, start in (("heldout", OUT / "chunks_heldout", need_ho, n_ho),
                                         ("train", OUT / "chunks_512", need_tr, n_tr)):
            got, nch, buf = 0, start, []
            while got < need:
                try:
                    res = next(results)
                except StopIteration:
                    print(f"[cc12m] index exhausted during {split}", flush=True)
                    break
                seen += 1
                if res is None:
                    continue
                buf.append(res)
                got += 1
                if len(buf) == a.per_chunk:
                    save(buf, dest, nch)
                    nch += 1
                    buf = []
                    el = time.time() - t0
                    print(f"[cc12m] {split} {got}/{need}  {got/el:.0f} img/s  "
                          f"{100*got/max(1,seen):.0f}% of urls ok  {el/60:.0f} min  "
                          f"eta {(need-got)/max(1e-9, got/el)/3600:.1f} h", flush=True)
            if buf:
                save(buf, dest, nch)
    (OUT / ".build.done").touch()
    print(f"[cc12m] COMPLETE in {(time.time()-t0)/3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    main()
