#!/usr/bin/env python
"""Download the BudgetPix checkpoints from the Hugging Face model repo into checkpoints/.

    python scripts/download_checkpoints.py --all
    python scripts/download_checkpoints.py jit_b16_256 minit2i_l_512 pmf_l16_256

Every file is checked against checkpoints/SHA256SUMS; re-running skips verified files and resumes partial downloads.
"""
from __future__ import annotations

import argparse, hashlib, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HF_URL = "https://huggingface.co/public-release-bp/BudgetPix/resolve/main"
MODELS = {  # name -> file
    "jit_b16_256": "budgetpix_jit_b16_256.pth",
    "jit_b32_512": "budgetpix_jit_b32_512.pth",
    "jit_l16_256": "budgetpix_jit_l16_256.pth",
    "jit_l32_512": "budgetpix_jit_l32_512.pth",
    "minit2i_b_512": "budgetpix_minit2i_b_512.pt",
    "minit2i_l_512": "budgetpix_minit2i_l_512.pt",
    "pixeldit_1024": "budgetpix_pixeldit_1024.pt",
    "pmf_l16_256": "budgetpix_pmf_l16_256.pt",
    # optional, for the pMF line only: the released pMF weights converted to torch WITH the v-branch. Training needs them.
    "pmf_released_l16": "pmf_released_L16_full_uv.pt",
    "pmf_released_b16": "pmf_released_B16_full_uv.pt",
}
CORE = ["jit_b16_256", "jit_b32_512", "jit_l16_256", "jit_l32_512", "minit2i_b_512", "minit2i_l_512", "pixeldit_1024", "pmf_l16_256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(url, dst):
    tmp = dst + ".tmp"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        r = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:               # already complete
            os.replace(tmp, dst); return
        raise
    mode = "ab" if done and r.status == 206 else "wb"
    total = int(r.headers.get("Content-Length", 0)) + (done if mode == "ab" else 0)
    got = done if mode == "ab" else 0
    with open(tmp, mode) as f:
        while True:
            blk = r.read(1 << 22)
            if not blk:
                break
            f.write(blk); got += len(blk)
            if total:
                print(f"\r  {os.path.basename(dst)}  {got / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=f"any of {list(MODELS)}")
    ap.add_argument("--all", action="store_true", help="the eight BudgetPix checkpoints (not the optional pMF extras)")
    ap.add_argument("--base-url", default=HF_URL, help="another host serving the same file names")
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints"))
    a = ap.parse_args()
    names = CORE if a.all else a.models
    if not names:
        ap.error("name the models to download, or pass --all")
    sums = {}
    with open(os.path.join(ROOT, "checkpoints", "SHA256SUMS")) as f:
        for line in f:
            if line.strip():
                h, n = line.split()
                sums[n.lstrip("*")] = h
    os.makedirs(a.out, exist_ok=True)
    for name in names:
        fname = MODELS[name]
        dst = os.path.join(a.out, fname)
        if os.path.exists(dst) and sha256(dst) == sums.get(fname):
            print(f"[ok] {fname} already present and verified"); continue
        fetch(f"{a.base_url}/{fname}", dst)
        got = sha256(dst)
        if sums.get(fname) and got != sums[fname]:
            raise SystemExit(f"{fname}: sha256 mismatch ({got} != {sums[fname]}); delete it and retry")
        print(f"[ok] {fname}  sha256 {got[:16]}...")


if __name__ == "__main__":
    main()
