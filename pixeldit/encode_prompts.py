#!/usr/bin/env python
"""Build the fixed evaluation prompt set: Gemma embeddings + masks + the null embedding.

    python encode_prompts.py --captions eval/captions.txt --n 2000 --out eval/prompts_2k.pt
    python encode_prompts.py --chunks /data/chunks_heldout --n 2000 --out eval/prompts_2k.pt
    python encode_prompts.py --null_only --out eval/null_emb.pt
"""
from __future__ import annotations

import argparse, glob, os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="", help="text file, one caption per line")
    ap.add_argument("--chunks", default="", help="folder of precomputed chunk_*.pt (uses their captions)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--negative", default=None, help="the CFG negative prompt (default: the released one)")
    ap.add_argument("--llm", default="Efficient-Large-Model/gemma-2-2b-it")
    ap.add_argument("--null_only", action="store_true", help="write only the TRAINING null (empty string, upstream-style) for caption dropout")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from apt.text import NEGATIVE_PROMPT, GemmaEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = GemmaEncoder(repo=a.llm, device=device)
    # the training null (caption dropout) is the empty string; the eval null (CFG) is the negative prompt
    neg = ("" if a.null_only else NEGATIVE_PROMPT) if a.negative is None else a.negative
    null = enc.null_embedding(neg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    if a.null_only:
        torch.save({"emb": null.to(torch.float16), "negative": neg, "llm": a.llm}, a.out)
        print(f"wrote {a.out}: null embedding {tuple(null.shape)} for {neg!r}")
        return

    caps = []
    if a.captions:
        caps = [l.strip() for l in open(a.captions) if l.strip()][: a.n]
    elif a.chunks:
        for f in sorted(glob.glob(os.path.join(a.chunks, "chunk_*.pt"))):
            c = torch.load(f, map_location="cpu", weights_only=False)
            caps += list(c.get("caption") or [])
            if len(caps) >= a.n:
                break
        caps = caps[: a.n]
        if not caps:
            raise SystemExit(f"{a.chunks} holds no captions — pass --captions instead")
    else:
        raise SystemExit("--captions, --chunks or --null_only")

    embs, masks = [], []
    for i in range(0, len(caps), a.bs):
        e, m = enc.encode(caps[i:i + a.bs])
        embs.append(e.to(torch.float16)); masks.append(m)
        if (i // a.bs) % 20 == 0:
            print(f"  {i + len(e)}/{len(caps)}", flush=True)
    emb = torch.cat(embs); mask = torch.cat(masks)
    torch.save({"emb": emb, "mask": mask, "captions": caps, "null": null.to(torch.float16),
                "negative": neg, "llm": a.llm, "chi": not a.no_chi}, a.out)
    print(f"wrote {a.out}: {emb.shape[0]} prompts, emb {tuple(emb.shape)} fp16 "
          f"({emb.numel() * 2 / 2**30:.2f} GiB) + null {tuple(null.shape)}")


if __name__ == "__main__":
    main()
