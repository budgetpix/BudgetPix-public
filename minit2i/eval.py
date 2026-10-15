"""FID / IS of a BudgetPix MiniT2I checkpoint at a token budget, on held-out captions.

    MINIT2I_ROOT=/path/to/minit2i-torch torchrun --nproc_per_node=4 eval.py \
        --ckpt ../checkpoints/budgetpix_minit2i_l_512.pt --data /data/mix120k/chunks_heldout --budget 512 --cfg 5.0

The held-out chunks (data.py format) give the captions and reference images (all by default, or the first --n).
"""
import argparse
import glob
import os
import shutil
import uuid

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from data import TextEncoder
from diffusion import sample
from model import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


def prepare(data, n, work, tag):
    """Rank 0, once: the first n held-out samples (all if n is 0) -> <work>/real_<tag>/*.png and their caption tokens."""
    path, real = os.path.join(work, f"prompts_{tag}.pt"), os.path.join(work, f"real_{tag}")
    if os.path.exists(path):
        return
    tmp = f"{real}.{uuid.uuid4().hex}.tmp"
    os.makedirs(tmp)
    try:
        ids, masks = [], []
        for f in sorted(glob.glob(os.path.join(data, "chunk_*.pt"))):
            c = torch.load(f, map_location="cpu", weights_only=False)
            pv = c["pixel_values"]
            if pv.dtype != torch.uint8:
                pv = ((pv.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            for i in range(pv.shape[0] if not n else min(pv.shape[0], n - len(ids))):
                Image.fromarray(pv[i].permute(1, 2, 0).numpy()).save(os.path.join(tmp, f"{len(ids):06d}.png"))
                ids.append(c["input_ids"][i].long())
                masks.append(c["attention_mask"][i].long())
            if n and len(ids) == n:
                break
        if not ids or len(ids) < n:
            raise SystemExit(f"{data} holds {len(ids)} samples" + (f", fewer than --n {n}" if n else ""))
        try:
            os.rename(tmp, real)
        except OSError:
            if not os.path.isdir(real):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ptmp = f"{path}.{uuid.uuid4().hex}.tmp"
    torch.save({"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks)}, ptmp)
    os.replace(ptmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="BudgetPix checkpoint (release file or train.py checkpoint)")
    ap.add_argument("--released", help="released MiniT2I weights instead of --ckpt (the untrained control)")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "minit2i_l_512.yml"), help="geometry for --released")
    ap.add_argument("--data", required=True, help="held-out chunk_*.pt folder (captions and reference images)")
    ap.add_argument("--n", type=int, default=0, help="number of held-out captions / images (0 = all)")
    ap.add_argument("--budget", type=int, default=None, help="tokens per image; default: the dense grid")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", default="evals", help="reference set, prompts and generated images")
    ap.add_argument("--keep_images", action="store_true")
    a = ap.parse_args()
    if bool(a.ckpt) == bool(a.released):
        raise SystemExit("pass --ckpt or --released")

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    data = os.path.normpath(os.path.abspath(a.data))
    tag = os.path.basename(os.path.dirname(data)) + "_" + os.path.basename(data) + "_" + (str(a.n) if a.n else "all")
    p = os.path.normpath(os.path.abspath(a.ckpt or a.released))
    src = os.path.basename(os.path.dirname(p)) + "_" + os.path.splitext(os.path.basename(p))[0]
    gen = os.path.join(a.work, f"gen_{src}_b{a.budget or 'dense'}_cfg{a.cfg:g}")
    if rank == 0:
        os.makedirs(a.work, exist_ok=True)
        prepare(a.data, a.n, a.work, tag)
        shutil.rmtree(gen, ignore_errors=True)
        os.makedirs(gen)
    if world > 1:
        dist.barrier()
    model, cfg = load_model(a.ckpt, a.released, a.config, device)
    encoder = TextEncoder(cfg["llm"], device, cfg["prompt_length"])
    prompts = torch.load(os.path.join(a.work, f"prompts_{tag}.pt"))
    n = prompts["input_ids"].shape[0]

    todo = np.arange(n)[rank::world]
    tok_sum = torch.zeros(2, device=device)
    for i in range(0, len(todo), a.batch_size):
        ks = todo[i:i + a.batch_size]
        text, mask = encoder.encode_ids(prompts["input_ids"][ks], prompts["attention_mask"][ks])
        noise = torch.stack([torch.randn(3, model.geo.img, model.geo.img, generator=torch.Generator().manual_seed(a.seed * 1_000_003 + int(k)))
                             for k in ks]).to(device) * cfg["noise_scale"]
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            x, mean_tok = sample(model, text, mask, noise, budget=a.budget, steps=a.steps, cfg=a.cfg)
        tok_sum += torch.tensor([mean_tok * len(ks), len(ks)], device=device)
        u8 = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for k, im in zip(ks, u8):
            Image.fromarray(im).save(os.path.join(gen, f"{int(k):06d}.png"))
        if rank == 0:
            print(f"generated {min(i + a.batch_size, len(todo)) * world} / {n}", flush=True)
    if world > 1:
        dist.all_reduce(tok_sum)
        dist.barrier()
    if rank == 0:
        import torch_fidelity
        m = torch_fidelity.calculate_metrics(input1=gen, input2=os.path.join(a.work, f"real_{tag}"), cuda=device.startswith("cuda"),
                                             fid=True, isc=True, verbose=False)
        print(f"budget {a.budget or model.geo.n_dense}, cfg {a.cfg:g}: {float(tok_sum[0] / tok_sum[1]):.1f} tokens/img "
              f"(trajectory mean, {model.geo.n_dense} dense)  FID {m['frechet_inception_distance']:.2f}  IS {m['inception_score_mean']:.2f}")
        if not a.keep_images:
            shutil.rmtree(gen, ignore_errors=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
