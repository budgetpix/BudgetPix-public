"""Fine-tune a released MiniT2I checkpoint into BudgetPix.

    MINIT2I_ROOT=/path/to/minit2i-torch torchrun --nproc_per_node=4 train.py --config configs/minit2i_l_512.yml --resume

The config gives the released weights (`checkpoint`), the data (`data.local_dataset_dir`, see data.py) and the
recipe. The added modules are zero-initialised and train `new_lr_mult` times faster; sample.py and eval.py load the EMA weights.
"""
import argparse
import dataclasses
import datetime
import os
import time

import torch
import torch.distributed as dist
import yaml

from data import TextEncoder, make_loader, to_unit
from diffusion import loss as diffusion_loss
from layout import all_base_tokens, pack
from model import NEW_MODULES, AdaptiveMMJiT, geometry, load_released


def check_zero_init(model, net, geo, noise_scale, device):
    """The adapted model must equal the released one on the dense layout before any training."""
    model.eval()
    img = torch.randn(2, 3, geo.img, geo.img, device=device) * noise_scale
    t = torch.rand(2, device=device)
    ctx = torch.randn(2, 16, net.cfg.txt_input_size, device=device)
    mask = torch.ones(2, 16, device=device)
    with torch.no_grad():
        d = (net(img, t, ctx, mask) - model(img, t, ctx, mask, pack([all_base_tokens(geo)] * 2, geo, device))).abs().max().item()
    if d > 1e-4:
        raise SystemExit(f"step 0 differs from the released model by {d:.2e}; not training")
    model.train()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: output_dir of the config")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint-last.pt if present")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = a.out or cfg["output_dir"]
    os.makedirs(out, exist_ok=True)

    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))
        torch.cuda.set_device(local)
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    main_proc, accum = rank == 0, int(cfg["grad_accum"])
    torch.manual_seed(int(cfg["seed"]) + rank)

    geo = geometry(cfg)
    net = load_released(cfg["checkpoint"], cfg["image_size"], cfg["patch_size"])
    model = AdaptiveMMJiT(net, geo).to(device)
    d = check_zero_init(model, net, geo, cfg["noise_scale"], device)
    for n, p in model.named_parameters():
        p.requires_grad_(not n.startswith(("net.t_embedder.", "net.pooled_embedder.")))
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    new = [p for n, p in params if n.startswith(NEW_MODULES)]
    released = [p for n, p in params if not n.startswith(NEW_MODULES)]
    lr = cfg["blr"] * cfg["batch_size"] / 256.0
    opt = torch.optim.AdamW([{"params": released, "lr": lr}, {"params": new, "lr": lr * cfg["new_lr_mult"]}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    ema = {n: p.detach().clone().float() for n, p in params}
    net_ddp = model
    if world > 1:
        net_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    if main_proc:
        print(f"[train] dense step 0 = released model (max diff {d:.1e}); {sum(p.numel() for p in new) / 1e6:.1f}M added, "
              f"{sum(p.numel() for p in released) / 1e6:.0f}M released parameters; lr {lr:.2e}", flush=True)

    step, last = 0, os.path.join(out, "checkpoint-last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        ema = {n: v.to(device).float() for n, v in ck["ema"].items()}
        step = int(ck["step"])
        torch.manual_seed(int(cfg["seed"]) + rank + 1000 * step)
        if main_proc:
            print(f"[train] resumed from {last} at step {step}", flush=True)
        del ck
    encoder = TextEncoder(cfg["llm"], device, cfg["prompt_length"])
    d_cfg = cfg["data"]
    loader = iter(make_loader(d_cfg["local_dataset_dir"], int(d_cfg["micro_batch_size"]), int(d_cfg["num_workers"]),
                              seed=int(cfg["seed"]) + 7919 * step, rank=rank, world_size=world))
    saved_cfg = dict(cfg, net_cfg=dataclasses.asdict(net.cfg))
    bf16 = device.startswith("cuda")

    def save(path, with_optimizer=True):
        state = {"model": model.state_dict(), "ema": ema, "step": step, "cfg": saved_cfg}
        if with_optimizer:
            state["optimizer"] = opt.state_dict()
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    t0, step0, micro = time.time(), step, 0
    opt.zero_grad(set_to_none=True)
    while step < cfg["max_steps"]:
        batch = next(loader)
        if micro == 0 and step == step0:
            n = batch["pixel_values"].shape[0] * world * accum
            assert n == cfg["batch_size"], f"micro batch x GPUs x grad_accum = {n}, config batch_size = {cfg['batch_size']}"
        images = to_unit(batch["pixel_values"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            text, mask = encoder.encode_ids(batch["input_ids"], batch["attention_mask"])
            loss, tokens = diffusion_loss(net_ddp, images, text.float(), mask, geo, cfg["noise_scale"], cfg["layout"])
        (loss / accum).backward()
        micro += 1
        if micro % accum:
            continue
        warm = min(1.0, (step + 1) / max(1, cfg["warmup"]))
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr * warm, lr * cfg["new_lr_mult"] * warm
        torch.nn.utils.clip_grad_norm_([p for _, p in params], cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, p in params:
                ema[n].mul_(cfg["ema"]).add_(p.detach().float(), alpha=1 - cfg["ema"])
        step += 1
        if main_proc and (step % cfg["log_every"] == 0 or step == step0 + 1):
            print(f"step {step:7d}  loss {loss.item():.4f}  tokens/img {tokens.item():6.1f}  "
                  f"{(step - step0) * cfg['batch_size'] / (time.time() - t0):7.1f} img/s", flush=True)
        if main_proc and step % cfg["save_every"] == 0:
            save(last)
        if main_proc and cfg.get("snapshot_every") and step % cfg["snapshot_every"] == 0:
            save(os.path.join(out, f"checkpoint-s{step}.pt"), with_optimizer=False)
    if main_proc:
        save(os.path.join(out, "checkpoint-final.pt"), with_optimizer=False)
        print("[train] done:", os.path.join(out, "checkpoint-final.pt"), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
