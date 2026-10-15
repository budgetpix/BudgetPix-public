"""Fine-tune a released JiT checkpoint into BudgetPix on ImageNet-1k.

    torchrun --nproc_per_node=4 train.py --released /weights/jit-b-16/checkpoint-last.pth \
        --data /data/imagenet-1k/data --epochs 100 --batch_size 112 --grad_accum 2 --out runs/jit_b16_256 --resume

The architecture and resolution come from the released checkpoint. Every weight is trained: the released
ones at lr = blr * global_batch / 256, the added (zero-initialised) modules 8x faster.
"""
import argparse
import functools
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data import ImageNetParquet, collate
from diffusion import loss as diffusion_loss
from layout import to_device
from model import NEW_MODULES, load_jit, noise_scale

IMAGENET_TRAIN = 1281167


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released", required=True, help="released JiT checkpoint (github.com/LTH14/JiT)")
    ap.add_argument("--data", required=True, help="ImageNet-1k parquet dir (train-*.parquet)")
    ap.add_argument("--out", default="runs/budgetpix_jit")
    ap.add_argument("--epochs", type=float, default=100)
    ap.add_argument("--batch_size", type=int, default=112, help="per GPU")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--blr", type=float, default=1e-5)
    ap.add_argument("--new_lr_mult", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--grad_clip", type=float, default=3.0)
    ap.add_argument("--anchor_p", type=float, default=0.15, help="share of images on the dense layout")
    ap.add_argument("--scramble_p", type=float, default=0.10, help="share of layouts that ignore the content")
    ap.add_argument("--offset_lo", type=float, default=0.0, help="per-image threshold offset ~ U(lo, hi)")
    ap.add_argument("--offset_hi", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--snapshots", type=int, default=10, help="kept snapshots over the run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint-last.pth if present")
    a = ap.parse_args()

    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    torch.manual_seed(a.seed + rank)
    os.makedirs(a.out, exist_ok=True)

    model = load_jit(a.released, device)
    geo, ns = model.geo, noise_scale(model.geo)
    for n, p in model.named_parameters():                     # step 0 must be exactly the released model
        if n.startswith(NEW_MODULES) and any(k in n for k in ("scale_emb", ".up.", ".out.", ".adaLN.")):
            assert float(p.detach().abs().max()) == 0, f"{n} is not zero-initialised"
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    new = [p for n, p in params if n.startswith(NEW_MODULES)]
    released = [p for n, p in params if not n.startswith(NEW_MODULES)]
    gbs = a.batch_size * world * a.grad_accum
    lr = a.blr * gbs / 256
    opt = torch.optim.AdamW([{"params": released, "lr": lr}, {"params": new, "lr": lr * a.new_lr_mult}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    ema = {n: p.detach().clone().float() for n, p in params}
    steps = int(round(a.epochs * IMAGENET_TRAIN / gbs))
    snapshot_every = max(1, int(IMAGENET_TRAIN // gbs * a.epochs / a.snapshots))
    if main_proc:
        print(f"[train] JiT-{model.variant}/{geo.base_patch} at {geo.img}: global batch {gbs}, lr {lr:.2e} "
              f"(x{a.new_lr_mult:g} on {sum(p.numel() for p in new) / 1e6:.1f}M added parameters), {steps} steps", flush=True)

    step = 0
    last = os.path.join(a.out, "checkpoint-last.pth")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema = {n: v.to(device).float() for n, v in ck["ema"].items()}
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"]) + 1
        if main_proc:
            print(f"[train] resumed from {last} at step {step}", flush=True)
        del ck
    torch.manual_seed(a.seed + rank + 1000 * step)
    ds = ImageNetParquet(a.data, geo.img, seed=a.seed + 1000 * step, rank=rank, world_size=world)
    coll = functools.partial(collate, geo=geo, offset_lo=a.offset_lo, offset_hi=a.offset_hi,
                             anchor_p=a.anchor_p, scramble_p=a.scramble_p)
    loader = iter(DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=coll, pin_memory=device.type == "cuda",
                             drop_last=True, persistent_workers=a.num_workers > 0,
                             prefetch_factor=4 if a.num_workers > 0 else None))
    net = DDP(model, device_ids=[local]) if world > 1 else model
    model.train()

    def save(path, with_optimizer=True):
        state = {"model": model.state_dict(), "ema": ema, "step": step, "args": vars(a)}
        if with_optimizer:
            state["optimizer"] = opt.state_dict()
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    t0, step0, tokens = time.time(), step, []
    while step < steps:
        warm = min(1.0, (step + 1) / max(1, a.warmup))
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr * warm, lr * a.new_lr_mult * warm
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(a.grad_accum):
            imgs, labels, L = next(loader)
            L = to_device(L, device)
            sync = net.no_sync() if world > 1 and micro < a.grad_accum - 1 else nullcontext()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = diffusion_loss(net, imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), L, ns)
                # a batch without e.g. 64-px tokens leaves that scale's modules unused; keep them in the graph for DDP
                loss = loss / a.grad_accum + 0.0 * sum(p.float().sum() for p in new)
            loss.backward()
            step_loss += loss.item()
            tokens.append(float(L["n_per"].float().mean()))
        torch.nn.utils.clip_grad_norm_([p for _, p in params], a.grad_clip)
        opt.step()
        with torch.no_grad():
            for n, p in params:
                ema[n].mul_(a.ema).add_(p.detach().float(), alpha=1 - a.ema)
        if main_proc and (step % a.log_every == 0 or step == steps - 1):
            print(f"step {step:6d}  loss {step_loss:.4f}  tokens/img {sum(tokens) / len(tokens):.1f}  "
                  f"{(step + 1 - step0) * gbs / (time.time() - t0):.0f} img/s", flush=True)
            tokens = []
        if main_proc and (step + 1) % a.save_every == 0:
            save(last)
        if main_proc and (step + 1) % snapshot_every == 0:
            save(os.path.join(a.out, f"checkpoint-s{step + 1}.pth"), with_optimizer=False)
        step += 1
    if main_proc:
        save(os.path.join(a.out, "checkpoint-final.pth"), with_optimizer=False)
        print("[train] done:", os.path.join(a.out, "checkpoint-final.pth"), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
