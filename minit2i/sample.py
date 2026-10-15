"""Text-to-image samples from a BudgetPix MiniT2I checkpoint at a token budget.

    MINIT2I_ROOT=/path/to/minit2i-torch python sample.py --ckpt ../checkpoints/budgetpix_minit2i_l_512.pt \
        --prompts "a corgi in a spacesuit" "a watercolor of a lighthouse at dusk" --budget 512 --cfg 6.0

--budget is the number of tokens per image (1024 = dense, 512 = 50 %, 256 = 25 %); omit it for the dense grid.
--released <dir> --config <yml> runs the released MiniT2I at the budget instead.
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from data import TextEncoder
from diffusion import sample
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="BudgetPix checkpoint (release file or train.py checkpoint)")
    ap.add_argument("--released", help="released MiniT2I weights instead of --ckpt (the untrained control)")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "minit2i_l_512.yml"), help="geometry for --released")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=None, help="tokens per image; default: the dense grid")
    ap.add_argument("--cfg", type=float, default=5.0, help="guidance scale")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="samples.png")
    a = ap.parse_args()
    if bool(a.ckpt) == bool(a.released):
        raise SystemExit("pass --ckpt or --released")

    model, cfg = load_model(a.ckpt, a.released, a.config, a.device)
    text, mask = TextEncoder(cfg["llm"], a.device, cfg["prompt_length"])(a.prompts)
    torch.manual_seed(a.seed)
    noise = torch.randn(len(a.prompts), 3, model.geo.img, model.geo.img, device=a.device) * cfg["noise_scale"]
    x, mean_tok = sample(model, text, mask, noise, budget=a.budget, steps=a.steps, cfg=a.cfg)
    print(f"{mean_tok:.0f} tokens/img on average over the trajectory ({model.geo.n_dense} dense)")
    save_image((x.float().clamp(-1, 1) + 1) / 2, a.out, nrow=len(a.prompts))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
