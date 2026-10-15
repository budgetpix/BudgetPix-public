#!/usr/bin/env python
"""Sample from a BudgetPix PixelDiT checkpoint at a chosen token budget.

    PIXELDIT_ROOT=/path/to/PixelDiT python sample_apt.py \
        --ckpt ../checkpoints/budgetpix_pixeldit_1024.pt --prompts "a photo of a cat" --budget 2048 --cfg 2.75

The dense grid at 1024 px is 4096 tokens; --budget 2048 is 50 %, 1024 is 25 %. Omit --budget for the
dense grid.
"""
from __future__ import annotations

import argparse, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apt import FlowSchedule, sample                                   # noqa: E402
from apt.text import GemmaEncoder, NEGATIVE_PROMPT                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="BudgetPix PixelDiT checkpoint (train_apt.py output or the release file)")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--negative", default=NEGATIVE_PROMPT, help="negative prompt for classifier-free guidance")
    ap.add_argument("--budget", type=int, default=None, help="tokens/img (dense = 4096 at 1024 px); omit for the dense grid")
    ap.add_argument("--cfg", type=float, default=2.75, help="guidance scale (the released default; raise it at low budgets)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--order", type=int, default=2, help="2 = DPM-Solver++(2M) (released default), 1 = Euler")
    ap.add_argument("--relayout_every", type=int, default=5)
    ap.add_argument("--warmup_steps", type=int, default=5, help="dense steps before the first layout")
    ap.add_argument("--llm", default="", help="text encoder (default: the checkpoint's, else Efficient-Large-Model/gemma-2-2b-it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="samples.png")
    a = ap.parse_args()
    device = a.device

    # 1. captions -> Gemma embeddings
    ck_cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)["cfg"]
    llm = a.llm or ck_cfg.get("llm") or "Efficient-Large-Model/gemma-2-2b-it"
    if not a.llm and (llm.startswith("/") or llm.startswith("<")) and not os.path.exists(llm):
        llm = "Efficient-Large-Model/gemma-2-2b-it"
    enc = GemmaEncoder(repo=llm, device=device)
    emb, mask = enc.encode(a.prompts)                                   # (B,300,2304) fp32, (B,300)
    null = enc.null_embedding(a.negative)                               # (1,300,2304)
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 2. the adaptive model (EMA weights) and its geometry
    from apt.convert import load_adaptive_checkpoint
    model, geo, cfg, _ = load_adaptive_checkpoint(a.ckpt, device=device)
    sched = FlowSchedule(int(cfg.get("diffusion_steps", 1000)), float(cfg.get("flow_shift", 4.0)))

    # 3. sample
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    x0 = torch.randn(len(a.prompts), 3, geo.img, geo.img, generator=g)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        x, L, tok_mean = sample(model, emb, geo, null_y=null, steps=a.steps, cfg_scale=a.cfg, sched=sched,
                                order=a.order, budget=(None if a.budget is None or a.budget >= geo.n_dense else a.budget),
                                relayout_every=a.relayout_every, warmup_steps=a.warmup_steps, text_mask=None,
                                x0=x0, device=device)
    print(f"trajectory-mean tokens/img: {tok_mean:.1f} of {geo.n_dense} dense ({100 * tok_mean / geo.n_dense:.0f}%)")
    from torchvision.utils import save_image
    save_image((x.float().clamp(-1, 1) + 1) / 2, a.out, nrow=len(a.prompts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
