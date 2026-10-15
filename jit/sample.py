"""Class-conditional samples from a BudgetPix JiT checkpoint at a token budget.

    python sample.py --ckpt ../checkpoints/budgetpix_jit_l32_512.pth --budget 128 --classes 207,88,417,933 --overlay
    python sample.py --ckpt ../checkpoints/budgetpix_jit_b16_256.pth --sweep --classes 207,88

--budget is the number of tokens per image (256 = dense, 128 = 50 %, 64 = 25 %); omit it for the dense grid.
--sweep draws one column per budget (100 / 75 / 50 / 25 %) from the same noise. --released runs a released
JiT checkpoint at the budget instead.
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from diffusion import sample
from layout import draw_layout
from model import load_budgetpix, load_jit, noise_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="BudgetPix checkpoint (release file or train.py checkpoint)")
    ap.add_argument("--released", help="a released JiT checkpoint instead of --ckpt (the untrained control)")
    ap.add_argument("--budget", type=int, default=None, help="tokens per image; default: the dense grid")
    ap.add_argument("--classes", default="207,88,417,933", help="ImageNet class ids")
    ap.add_argument("--n", type=int, default=1, help="samples per class")
    ap.add_argument("--sweep", action="store_true", help="one column per budget: 100 / 75 / 50 / 25 %% of the tokens")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=None, help="dense steps first (default: 20 %% at 256, 10 %% at 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlay", action="store_true", help="also write <out>_layout.png with the final token layout")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="samples.png")
    a = ap.parse_args()
    if bool(a.ckpt) == bool(a.released):
        raise SystemExit("pass --ckpt or --released")

    model = (load_budgetpix(a.ckpt, a.device) if a.ckpt else load_jit(a.released, a.device)).eval()
    geo = model.geo
    labels = torch.tensor([int(c) for c in a.classes.split(",") for _ in range(a.n)], device=a.device)
    torch.manual_seed(a.seed)
    noise = torch.randn(len(labels), 3, geo.img, geo.img, device=a.device) * noise_scale(geo)
    budgets = [geo.n_dense * f // 4 for f in (4, 3, 2, 1)] if a.sweep else [a.budget]

    images, layouts = [], []
    for b in budgets:
        with torch.autocast(a.device.split(":")[0], dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            x, tokens, mean_tok = sample(model, labels, noise, budget=b, steps=a.steps, cfg=a.cfg, warmup=a.warmup_steps)
        print(f"budget {b or geo.n_dense}: {mean_tok:.0f} tokens/img on average over the trajectory "
              f"({geo.n_dense} dense)")
        images.append(((x.float() + 1) / 2).clamp(0, 1).cpu())
        layouts.append(torch.stack([draw_layout(img, tok) for img, tok in zip(images[-1], tokens)]))
    nrow = len(budgets) if a.sweep else len(labels)
    order = lambda xs: torch.stack(xs, 1).flatten(0, 1)          # one row per sample, one column per budget
    save_image(order(images), a.out, nrow=nrow, padding=2)
    print("wrote", a.out)
    if a.overlay:
        path = os.path.splitext(a.out)[0] + "_layout.png"
        save_image(order(layouts), path, nrow=nrow, padding=2)
        print("wrote", path)


if __name__ == "__main__":
    main()
