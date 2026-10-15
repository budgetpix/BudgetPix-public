#!/usr/bin/env python3
"""Build MiniT2I's fine-tuning corpus locally: the "120K mix" of three Hugging Face Hub sets.
  BLIP3o/BLIP3o-60k                      WebDataset tars (.jpg + .txt); caption = the .txt as decoded.
  OpenDatasets/dalle-3-dataset           parquet files: PNG bytes + caption [+ synthetic_caption];
                                         caption = (caption or synthetic_caption).strip() + "\n".
  FreedomIntelligence/ShareGPT-4o-Image  text_to_image_part_0..9.tar (PNG) + text_to_image.json; prompts over 256 T5
                                         tokens are dropped; caption = input_prompt + "\n"; images get a JPEG q95 round trip.
Images: Resize(512, BICUBIC) + CenterCrop(512), uint8. Tokens: flan-t5-large, max_length 256, padded and truncated.
All samples are shuffled (seed 0) and written as chunk_*.pt (pixel_values uint8 (n,3,512,512), input_ids / attention_mask
int64 (n,256), caption, key = "<source>/<member>"); the last 1024 samples form the held-out chunk.
Pass 1 writes per-file uint8 shards under work<SUFFIX>/; pass 2 writes chunks_512<SUFFIX> / chunks_heldout<SUFFIX>.
Resumable; .build.done<SUFFIX> marks completion.

  MIX_OUT=/data/mix120k MIX_T5=google/flan-t5-large python mix120k_build.py --procs 24
"""
from __future__ import annotations
import argparse, io, json, os, pathlib, random, sys, tarfile, time
import multiprocessing as mp

OUT = pathlib.Path(os.environ.get("MIX_OUT", "data/mix120k"))
CACHE = OUT / "hf_cache"
SIZE, MIN_SIDE, PROMPT_LEN, PER_CHUNK, MAX_PROMPT_TOKENS = 512, 512, 256, 1024, 256
T5 = os.environ.get("MIX_T5", "google/flan-t5-large")   # HF id or a local snapshot dir
REPOS = {"blip3o": "BLIP3o/BLIP3o-60k", "dalle3": "OpenDatasets/dalle-3-dataset", "sharegpt4o": "FreedomIntelligence/ShareGPT-4o-Image"}
PATTERNS = {"blip3o": ["*.tar"], "dalle3": ["data/*.parquet"], "sharegpt4o": ["text_to_image_part_*.tar", "text_to_image.json"]}
SUFFIX = ""


def log(*a):
    print("[mix120k]", time.strftime("%H:%M:%S"), *a, flush=True)


def download():
    from huggingface_hub import snapshot_download
    roots = {}
    for name, repo in REPOS.items():
        t0 = time.time()
        roots[name] = pathlib.Path(snapshot_download(repo_id=repo, repo_type="dataset", cache_dir=str(CACHE),
                                                     allow_patterns=PATTERNS[name], max_workers=8))
        log(f"downloaded {name} -> {roots[name]} ({time.time() - t0:.0f} s)")
    return roots


_TF = None


def transform():
    global _TF
    if _TF is None:
        from torchvision import transforms
        _TF = transforms.Compose([transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                                  transforms.CenterCrop(SIZE), transforms.PILToTensor()])
    return _TF


def jpeg_roundtrip(im):
    """JPEG quality-95 round trip."""
    from PIL import Image
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=95); buf.seek(0)
    out = Image.open(buf).convert("RGB"); out.load()
    return out


def prep(im, roundtrip=False):
    """PIL image -> HxWx3 uint8 bytes at SIZE^2, or None if a side is below MIN_SIDE."""
    if min(im.size) < MIN_SIDE:
        return None
    if roundtrip:
        im = jpeg_roundtrip(im)
    t = transform()(im.convert("RGB"))                 # uint8 CHW
    return t.permute(1, 2, 0).contiguous().numpy().tobytes()


def iter_tar_pairs(path):
    """WebDataset tar -> (key, image bytes, caption or None) grouped by member stem, in file order."""
    cur, img, txt = None, None, None
    with tarfile.open(path, "r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            stem, _, ext = m.name.rpartition(".")
            ext = ext.lower()
            if stem != cur:
                if cur is not None and img is not None:
                    yield cur, img, txt
                cur, img, txt = stem, None, None
            data = tar.extractfile(m).read()
            if ext in ("jpg", "jpeg", "png", "webp"):
                img = data
            elif ext == "txt":
                txt = data.decode("utf-8", errors="replace")
        if cur is not None and img is not None:
            yield cur, img, txt


def _set_suffix(suffix):
    global SUFFIX
    SUFFIX = suffix


def work_file(source, path):
    return OUT / f"work{SUFFIX}" / f"{source}__{pathlib.Path(path).name}"


def pass1(task):
    """One source file -> work/<source>__<file>.npy (N,512,512,3 uint8) + .json (keys, captions). Returns stats."""
    source, path, captions_json = task
    import numpy as np
    from PIL import Image
    wf = work_file(source, path)
    if wf.with_suffix(".json").exists():
        return source, path, "skip", 0, 0
    kept, dropped, keys, caps, bufs = 0, 0, [], [], []
    t0 = time.time()
    if source == "dalle3":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "caption", "synthetic_caption", "image_hash"]):
            for i, row in enumerate(batch.to_pylist()):
                raw = row.get("image")
                data = raw.get("bytes") if isinstance(raw, dict) else raw
                cap = str(row.get("caption") or row.get("synthetic_caption") or "").strip()
                if not data or not cap:
                    dropped += 1; continue
                try:
                    px = prep(Image.open(io.BytesIO(data)))
                except Exception:
                    px = None
                if px is None:
                    dropped += 1; continue
                bufs.append(px); caps.append(cap + "\n"); keys.append(f"dalle3/{row.get('image_hash') or i}"); kept += 1
    else:
        cap_by_key = None
        if source == "sharegpt4o":
            cap_by_key = json.load(open(captions_json))
        for stem, img, txt in iter_tar_pairs(path):
            if source == "sharegpt4o":
                base = stem.split("/")[-1]
                cap = cap_by_key.get(stem) or cap_by_key.get(base)
            else:
                cap = txt
            if not cap or not cap.strip():
                dropped += 1; continue
            try:
                px = prep(Image.open(io.BytesIO(img)), roundtrip=(source == "sharegpt4o"))
            except Exception:
                px = None
            if px is None:
                dropped += 1; continue
            bufs.append(px); caps.append(cap); keys.append(f"{source}/{stem}"); kept += 1
    wf.parent.mkdir(parents=True, exist_ok=True)
    if kept:
        arr = np.frombuffer(b"".join(bufs), dtype=np.uint8).reshape(kept, SIZE, SIZE, 3)
        tmp = str(wf.with_suffix(".npy")) + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, str(wf.with_suffix(".npy")))
    json.dump({"keys": keys, "captions": caps, "n": kept, "dropped": dropped, "source": source,
               "file": str(path)}, open(wf.with_suffix(".json.tmp"), "w"))
    os.replace(wf.with_suffix(".json.tmp"), wf.with_suffix(".json"))
    return source, path, f"{time.time() - t0:.0f}s", kept, dropped


def sharegpt_captions(json_path):
    """text_to_image.json -> {key: input_prompt + "\\n"} for prompts <= 256 T5 tokens, keyed by image path without
    extension and by its basename."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T5)
    rows = json.load(open(json_path, encoding="utf-8"))
    out, dropped = {}, 0
    for r in rows:
        img = str(r.get("output_image") or ""); cap = str(r.get("input_prompt") or "")
        if not img or not cap:
            continue
        if len(tok(cap, truncation=False)["input_ids"]) > MAX_PROMPT_TOKENS:
            dropped += 1; continue
        stem = img.rsplit(".", 1)[0]
        out[stem] = cap + "\n"; out[stem.split("/")[-1]] = cap + "\n"
    log(f"sharegpt4o captions: {len(rows)} rows, {dropped} over {MAX_PROMPT_TOKENS} tokens dropped")
    return out


def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--n-heldout", type=int, default=PER_CHUNK)
    ap.add_argument("--suffix", default="", help="e.g. .v2: writes work.v2/, chunks_512.v2/, chunks_heldout.v2/, .build.done.v2")
    a = ap.parse_args()
    SUFFIX = a.suffix
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / f".build.done{SUFFIX}"
    if done.exists():
        log(f"complete ({done} exists)"); return 0
    roots = download()
    sg_json = OUT / f"sharegpt4o_captions{SUFFIX}.json"
    if not sg_json.exists():
        json.dump(sharegpt_captions(roots["sharegpt4o"] / "text_to_image.json"), open(sg_json, "w"))
    tasks = [("blip3o", str(p), None) for p in sorted(roots["blip3o"].glob("*.tar"))]
    tasks += [("dalle3", str(p), None) for p in sorted(roots["dalle3"].rglob("*.parquet"))]
    tasks += [("sharegpt4o", str(p), str(sg_json)) for p in sorted(roots["sharegpt4o"].glob("text_to_image_part_*.tar"))]
    log(f"pass 1: {len(tasks)} source files, {a.procs} processes, suffix '{SUFFIX}'")
    tot = {}
    with mp.Pool(a.procs, initializer=_set_suffix, initargs=(SUFFIX,)) as pool:
        for source, path, dt, kept, dropped in pool.imap_unordered(pass1, tasks):
            log(f"  {source} {pathlib.Path(path).name}: {dt} kept {kept} dropped {dropped}")
            tot.setdefault(source, [0, 0]); tot[source][0] += kept; tot[source][1] += dropped
    shards = sorted((OUT / f"work{SUFFIX}").glob("*.json"))
    meta = [json.load(open(s)) for s in shards]
    n_by = {}
    for m in meta:
        n_by[m["source"]] = n_by.get(m["source"], 0) + m["n"]
    log(f"pass 1 done: {n_by} (this run kept/dropped {tot})")
    index = [(i, j) for i, m in enumerate(meta) for j in range(m["n"])]
    random.Random(0).shuffle(index)
    ho, tr = index[:a.n_heldout], index[a.n_heldout:]
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T5)
    arrays = [np.load(str(s.with_suffix(".npy")), mmap_mode="r") if m["n"] else None for s, m in zip(shards, meta)]

    def write(split, items, dest):
        dest.mkdir(parents=True, exist_ok=True)
        for c in range(0, len(items), PER_CHUNK):
            f = dest / f"chunk_{c // PER_CHUNK:06d}.pt"
            if f.exists():
                continue
            part = items[c:c + PER_CHUNK]
            px = torch.from_numpy(np.stack([arrays[i][j] for i, j in part])).permute(0, 3, 1, 2).contiguous()
            caps = [meta[i]["captions"][j] for i, j in part]
            keys = [meta[i]["keys"][j] for i, j in part]
            enc = tok(caps, padding="max_length", truncation=True, max_length=PROMPT_LEN, return_tensors="pt")
            tmp = dest / f".chunk_{c // PER_CHUNK:06d}.tmp"
            torch.save({"pixel_values": px, "input_ids": enc["input_ids"].long(), "attention_mask": enc["attention_mask"].long(),
                        "caption": caps, "key": keys}, tmp)
            tmp.rename(f)
            if (c // PER_CHUNK) % 10 == 0:
                log(f"  {split}: chunk {c // PER_CHUNK} / {(len(items) + PER_CHUNK - 1) // PER_CHUNK}")

    write("heldout", ho, OUT / f"chunks_heldout{SUFFIX}")
    write("train", tr, OUT / f"chunks_512{SUFFIX}")
    json.dump({"train": len(tr), "heldout": len(ho), "per_source": n_by, "seed": 0, "size": SIZE, "min_side": MIN_SIDE,
               "prompt_len": PROMPT_LEN, "tokenizer": "google/flan-t5-large", "sources": REPOS,
               "preprocessing": "torchvision Resize(512, BICUBIC) + CenterCrop(512) + PILToTensor; ShareGPT-4o-Image through a JPEG q95 round trip; "
                                "captions: BLIP3o .txt as decoded, DALL-E 3 (caption or synthetic_caption).strip()+'\\n', ShareGPT4o input_prompt+'\\n' (<= 256 T5 tokens)"},
              open(OUT / f"manifest{SUFFIX}.json", "w"), indent=1)
    done.touch()
    log(f"complete: {len(tr)} train + {len(ho)} held-out samples in {OUT} (suffix '{SUFFIX}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
