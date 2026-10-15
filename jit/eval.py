"""FID / IS of a BudgetPix JiT checkpoint at a token budget (class-conditional ImageNet).

    torchrun --nproc_per_node=4 eval.py --ckpt ../checkpoints/budgetpix_jit_l32_512.pth --budget 128

50 images per class (50k), 50 Heun steps, CFG 3.0, PNGs scored by torch-fidelity against the reference
statistics in fid_stats/. Image k always gets the same noise.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from diffusion import sample
from model import load_budgetpix, load_jit, noise_scale

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="BudgetPix checkpoint (release file or train.py checkpoint)")
    ap.add_argument("--released", help="a released JiT checkpoint instead of --ckpt (the untrained control)")
    ap.add_argument("--budget", type=int, default=None, help="tokens per image; default: the dense grid")
    ap.add_argument("--num_images", type=int, default=50000, help="image k has class k // (num_images / 1000)")
    ap.add_argument("--batch_size", type=int, default=256, help="per GPU")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup_steps", type=int, default=None, help="dense steps first (default: 20 %% at 256, 10 %% at 512)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="folder for the generated PNGs (default: evals/<checkpoint folder>_<checkpoint>"
                                                 "_b<budget>_cfg<cfg>; its *.png files are replaced)")
    ap.add_argument("--keep_images", action="store_true")
    a = ap.parse_args()
    if bool(a.ckpt) == bool(a.released):
        raise SystemExit("pass --ckpt or --released")
    if a.out is None:
        p = os.path.abspath(a.ckpt or a.released)
        src = os.path.basename(os.path.dirname(p)) + "_" + os.path.splitext(os.path.basename(p))[0]
        a.out = os.path.join("evals", f"{src}_b{a.budget or 'dense'}_cfg{a.cfg:g}")

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    model = (load_budgetpix(a.ckpt, device) if a.ckpt else load_jit(a.released, device)).eval()
    geo = model.geo
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        for f in glob.glob(os.path.join(a.out, "*.png")):
            os.remove(f)
    if world > 1:
        dist.barrier()

    per_class = max(1, a.num_images // 1000)
    todo = np.arange(a.num_images)[rank::world]
    tok_sum = torch.zeros(2, device=device)
    for i in range(0, len(todo), a.batch_size):
        ks = todo[i:i + a.batch_size]
        labels = torch.tensor(ks // per_class % 1000, device=device)
        noise = torch.stack([torch.randn(3, geo.img, geo.img, generator=torch.Generator().manual_seed(a.seed * 1_000_003 + int(k)))
                             for k in ks]).to(device) * noise_scale(geo)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            x, _, mean_tok = sample(model, labels, noise, budget=a.budget, steps=a.steps, cfg=a.cfg, warmup=a.warmup_steps)
        tok_sum += torch.tensor([mean_tok * len(ks), len(ks)], device=device)
        u8 = np.round(np.clip(((x.float() + 1) / 2).cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255)).astype(np.uint8)
        for k, im in zip(ks, u8):
            Image.fromarray(im).save(os.path.join(a.out, f"{int(k):06d}.png"))
        if rank == 0:
            print(f"generated {min(i + a.batch_size, len(todo)) * world} / {a.num_images}", flush=True)
    if world > 1:
        dist.all_reduce(tok_sum)
        dist.barrier()
    if rank == 0:
        import torch_fidelity                                    # the LTH14 fork, see requirements.txt
        stats = os.path.join(HERE, "fid_stats", f"jit_in{geo.img}_stats.npz")
        m = torch_fidelity.calculate_metrics(input1=a.out, input2=None, fid_statistics_file=stats, cuda=device.startswith("cuda"),
                                             isc=True, fid=True, kid=False, prc=False, verbose=False)
        print(f"budget {a.budget or geo.n_dense}: {float(tok_sum[0] / tok_sum[1]):.1f} tokens/img (trajectory mean, "
              f"{geo.n_dense} dense)  FID {m['frechet_inception_distance']:.2f}  IS {m['inception_score_mean']:.1f}")
        if not a.keep_images:
            for f in glob.glob(os.path.join(a.out, "*.png")):
                os.remove(f)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
