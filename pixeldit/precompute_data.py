#!/usr/bin/env python
"""Turn a captioned image set into training chunks: uint8 pixels + encoded Gemma embeddings.

    python precompute_data.py --tars /data/shards --out /data/chunks_1024 --size 1024 --per_chunk 64
    python precompute_data.py --images /data/imgs --out /data/chunks_1024 --size 1024

Hold out the last shards for evaluation and never point `data.chunks` at them.
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", default=""); ap.add_argument("--images", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--per_chunk", type=int, default=64)
    ap.add_argument("--n", type=int, default=0, help="stop after N samples (0 = the whole source)")
    ap.add_argument("--bs", type=int, default=8, help="caption-encoding batch")
    ap.add_argument("--llm", default="Efficient-Large-Model/gemma-2-2b-it")
    ap.add_argument("--start_index", type=int, default=0, help="first chunk number (to extend a set)")
    ap.add_argument("--caption_suffixes", default=",_InternVL2-26B,_VILA1-5-13B",
                    help="external caption jsons beside each shard, tried in order (upstream stage 3's list); "
                         "'' alone = the sample's own json only")
    a = ap.parse_args()
    if not (a.tars or a.images):
        raise SystemExit("--tars or --images")
    from apt.data import ImageFolderStream, TarStream
    from apt.text import GemmaEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = GemmaEncoder(repo=a.llm, device=device)
    # once=True: visit the source exactly once and stop
    suf = tuple(a.caption_suffixes.split(","))
    ds = (TarStream(a.tars.split(","), img=a.size, once=True, caption_suffixes=suf) if a.tars
          else ImageFolderStream(a.images.split(","), img=a.size, once=True))
    os.makedirs(a.out, exist_ok=True)

    pix, caps, chunk_i, total = [], [], a.start_index, 0
    it = iter(ds)
    while True:
        try:
            s = next(it)
        except StopIteration:
            break
        pix.append(s["pixel_values"]); caps.append(s["caption"]); total += 1
        if len(pix) >= a.per_chunk:
            chunk_i = _flush(pix, caps, enc, a, chunk_i)
            pix, caps = [], []
        if a.n and total >= a.n:
            break
        if total and total % (a.per_chunk * 10) == 0:
            print(f"  {total} samples -> {chunk_i} chunks", flush=True)
    if pix:
        chunk_i = _flush(pix, caps, enc, a, chunk_i)
    print(f"wrote {chunk_i - a.start_index} chunks ({total} samples) to {a.out}")
    print("also write the null embedding once:  python encode_prompts.py --null_only --out eval/null_emb.pt")


def _flush(pix, caps, enc, a, chunk_i):
    ys, ms = [], []
    for i in range(0, len(caps), a.bs):
        e, m = enc.encode(caps[i:i + a.bs])
        ys.append(e.to(torch.float16)); ms.append(m)
    path = os.path.join(a.out, f"chunk_{chunk_i:06d}.pt")
    tmp = path + ".tmp"
    torch.save({"pixel_values": torch.stack(pix), "y": torch.cat(ys), "y_mask": torch.cat(ms),
                "caption": caps, "llm": a.llm, "size": a.size}, tmp)
    os.replace(tmp, path)
    return chunk_i + 1


if __name__ == "__main__":
    main()
