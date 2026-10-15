#!/usr/bin/env python
"""Adaptive-token finetuning of the released PixelDiT-T2I 1024 px checkpoint.

    PIXELDIT_ROOT=/path/to/PixelDiT torchrun --nproc_per_node=4 train_apt.py \
        --config configs/pixeldit_1024.yml

Warm-starts from the released weights; every added module is zero-gated, so step 0 is the released model.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apt import AdaptivePixDiT, FlowSchedule, Geometry, all_base_tokens, pack, training_loss   # noqa: E402

NEW_PREFIXES = ("mixers.", "refines.")
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path, config_path=None):
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    for base in ([os.path.dirname(os.path.abspath(config_path))] if config_path else []) + [HERE]:
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(HERE, path)


# ------------------------------------------------------------------------------------- model
def geometry_of(cfg):
    return Geometry(img=int(cfg["image_size"]), base_patch=int(cfg["patch_size"]),
                    scales=tuple(cfg["scales"]), thresholds=tuple(cfg["thresholds"]))


def build_model(cfg, device, verbose=True):
    """Released PixDiT_T2I (+ weights) wrapped in AdaptivePixDiT. Returns (model, net, arch)."""
    from apt.convert import describe, load_released_weights, read_config
    from apt.upstream import build_released
    ck = cfg.get("checkpoint", "")
    arch = read_config(resolve(cfg["config_json"], cfg.get("_config_path"))) if cfg.get("config_json") else read_config(ck)
    arch = dict(arch)
    arch.update(image_size=int(cfg["image_size"]), patch_size=int(cfg["patch_size"]))
    net = build_released(arch)
    if ck:
        rep = load_released_weights(net, ck)
        if verbose:
            print("[init] released weights:", describe(rep))
    elif verbose:
        print("[init] WARNING: no checkpoint -- random weights. The contract still holds, but there "
              "is nothing to preserve.")
    geo = geometry_of(cfg)
    model = AdaptivePixDiT(net, geo,
                           mix_bott=int(cfg.get("mix_bott", 256)),
                           refine_width=int(cfg.get("refine_width", 64)),
                           refine_depth=int(cfg.get("refine_depth", 2)),
                           refine_heads=int(cfg.get("refine_heads", 4)),
                           use_refine=bool(cfg.get("use_refine", True)),
                           grad_ckpt=bool(cfg.get("grad_ckpt", True))).to(device)
    return model, net, arch


def assert_contract(model, net, cfg, geo, device, rel_tol=1e-3):
    """Refuse to train unless step 0 reproduces the released model on the dense grid."""
    was = model.training
    model.eval(); net.eval()
    sd = model.state_dict()
    nonzero = [n for n in model.gated_tensor_names() if float(sd[n].abs().max()) != 0.0]
    if nonzero:
        raise SystemExit(f"zero-init contract violated -- these gated tensors are not zero: {nonzero[:6]}")
    B, P = 1, 8
    x = torch.randn(B, 3, geo.img, geo.img, device=device)
    t = torch.full((B,), 500.0, device=device)
    y = torch.randn(B, P, net.txt_embed_dim, device=device)
    L = pack([all_base_tokens(geo)] * B, geo, device)
    tf32_mm, tf32_dnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            a = net(x, t, y, s=None, mask=None)
            b = model(x, t, y, L)
            d = (a - b).abs().max().item()
            rng = float(a.abs().max())
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_mm, tf32_dnn
    rel = d / max(rng, 1e-12)
    print(f"[contract] gated tensors all zero; dense max|adaptive - released| = {d:.3e} "
          f"= {rel:.2e} of the output range (+-{rng:.3f}), tf32 off")
    if rel > rel_tol:
        raise SystemExit(f"zero-init contract violated -- {rel:.2e} of range exceeds {rel_tol:.0e} "
                         f"with the heads provably zero: the adaptive path itself differs, not the init")
    if rng == 0.0 and cfg.get("checkpoint"):
        raise SystemExit("the released model output is identically zero -- the checkpoint did not load")
    model.train(was)


# ------------------------------------------------------------------------------------- data
def build_loader(cfg, device):
    """Yields (images[-1,1] (B,3,H,W), y (B,300,2304), y_mask (B,300))."""
    from apt.data import make_loader, prepare_images
    d = dict(cfg.get("data", {}) or {})
    d.setdefault("seed", cfg.get("seed", 0))
    loader, precomputed = make_loader(d, int(cfg["image_size"]), int(d.get("micro_batch_size", 2)),
                                      int(d.get("num_workers", 8)), int(d.get("dataloader_prefetch", 2)))
    enc = None
    if not precomputed:
        from apt.text import GemmaEncoder
        print("[data] captions are being encoded on the fly -- a frozen Gemma (~5 GB) now sits on "
              "every GPU. precompute_data.py removes it.")
        enc = GemmaEncoder(repo=cfg.get("llm", "Efficient-Large-Model/gemma-2-2b-it"), device=device)
    while True:
        for batch in loader:
            images = prepare_images(batch["pixel_values"], device)
            if enc is None:
                y, m = batch["y"].to(device, non_blocking=True), batch["y_mask"].to(device, non_blocking=True)
            else:
                e, m = enc.encode(batch["caption"])
                y, m = e.to(device), m.to(device)
            yield images, y.float(), m.float()


def load_null(cfg, device, net):
    """The caption-dropout embedding, made by `encode_prompts.py --null_only`."""
    p = resolve(cfg.get("null_emb", ""), cfg.get("_config_path"))
    if p and os.path.exists(p):
        e = torch.load(p, map_location="cpu", weights_only=False)
        e = e["emb"] if isinstance(e, dict) else e
        return e[:1].to(device).float()
    if cfg.get("caption_drop", 0.1) > 0:
        raise SystemExit(f"null_emb '{p}' not found; make it with encode_prompts.py --null_only, or set "
                         f"caption_drop: 0 (which changes the recipe -- upstream drops 10%)")
    return None


# ------------------------------------------------------------------------------------- loop
def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default="", help="'auto' = continue from <out>/checkpoint-last.pt")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["_config_path"] = os.path.abspath(a.config)          # so relative entries resolve against it
    run_training(cfg, out_dir=a.out, resume=a.resume)


def run_training(cfg, out_dir=None, resume=""):
    out = Path(out_dir or cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world > 1:
        import datetime
        torch.distributed.init_process_group(backend, timeout=datetime.timedelta(hours=6))
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if backend == "nccl" else "cpu"
    main_proc = rank == 0
    accum = int(cfg.get("grad_accum", 1))
    torch.manual_seed(int(cfg.get("seed", 0)) + rank)

    model, net, arch = build_model(cfg, device, verbose=main_proc)
    geo = geometry_of(cfg)
    assert_contract(model, net, cfg, geo, device)
    sched = FlowSchedule(int(cfg.get("diffusion_steps", 1000)), float(cfg.get("flow_shift", 4.0)))
    null_y = load_null(cfg, device, net)

    raw = model
    for p in raw.parameters():
        p.requires_grad_(True)
    if world > 1:
        # find_unused_parameters: a micro-batch with no 64-px token leaves mixers['64']/refines['64']
        # out of the graph on that rank.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=([local_rank] if backend == "nccl" else None), find_unused_parameters=True)

    new_params = [p for n, p in raw.named_parameters() if n.startswith(NEW_PREFIXES) and p.requires_grad]
    old_params = [p for n, p in raw.named_parameters() if not n.startswith(NEW_PREFIXES) and p.requires_grad]
    gbs = int(cfg["batch_size"])
    lr = float(cfg.get("lr", 2.0e-5))
    if main_proc:
        print(f"[opt] AdamW lr {lr:g} (new modules x{cfg.get('new_lr_mult', 8)}), "
              f"{sum(p.numel() for p in new_params)/1e6:.2f}M new + "
              f"{sum(p.numel() for p in old_params)/1e6:.2f}M pretrained trainable; world {world}, grad_accum {accum}, "
              f"global batch {gbs}")
    groups = [{"params": new_params, "lr": lr * float(cfg.get("new_lr_mult", 8.0)), "is_new": True}]
    if old_params:
        groups.insert(0, {"params": old_params, "lr": lr, "is_new": False})
    opt = torch.optim.AdamW(groups, betas=tuple(cfg.get("betas", (0.9, 0.999))),
                            eps=float(cfg.get("adam_eps", 1e-15)), weight_decay=float(cfg.get("weight_decay", 0.0)))
    ema_dev = cfg.get("ema_device", "cpu")
    ema = {n: p.detach().to(ema_dev).clone().float() for n, p in raw.named_parameters() if p.requires_grad}

    steps = int(cfg["max_steps"])
    step = 0
    last = out / "checkpoint-last.pt"
    if resume == "auto" and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        for n in ema:
            ema[n] = ck["ema"][n].to(ema[n].device).float()
        step = int(ck["step"])
        torch.manual_seed(int(cfg.get("seed", 0)) + rank + 1000 * step)
        if main_proc:
            print(f"[resume] {last} -> step {step}")
    elif resume and resume != "auto":
        raise SystemExit("--resume takes 'auto' only")

    loader = iter(build_loader(cfg, device))
    payload_cfg = dict(cfg, net_cfg=arch)
    loss_kw = dict(sched=sched, null_y=null_y, caption_drop=float(cfg.get("caption_drop", 0.1)),
                   layout_kw=dict(cfg.get("layout", {}) or {}))
    warm_steps = int(cfg.get("warmup", 2000))
    clip = float(cfg.get("grad_clip", 0.1))
    t0 = time.time(); step0 = step; micro = 0; checked = False
    opt.zero_grad(set_to_none=True)
    while step < steps:
        images, y, y_mask = next(loader)
        if not checked:
            got = images.shape[0] * world * accum
            if got != gbs:
                raise SystemExit(f"config batch_size {gbs} but per-gpu {images.shape[0]} x world {world} "
                                 f"x grad_accum {accum} = {got}; fix data.micro_batch_size or grad_accum")
            checked = True
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loss, stats = training_loss(model, images, y, geo, **loss_kw)
        ok = torch.tensor([1 if loss.requires_grad else 0], device=device)
        if world > 1:
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
        if int(ok.item()) == 0:
            continue
        (loss / accum).backward()
        micro += 1
        if micro % accum:
            continue
        w = min(1.0, (step + 1) / max(1, warm_steps))
        for g in opt.param_groups:
            g["lr"] = (lr * float(cfg.get("new_lr_mult", 8.0)) if g["is_new"] else lr) * w
        gn = torch.nn.utils.clip_grad_norm_([p for p in raw.parameters() if p.requires_grad], clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            d = float(cfg.get("ema", 0.9999))
            for n, p in raw.named_parameters():
                if n in ema:
                    ema[n].mul_(d).add_(p.detach().float().to(ema[n].device), alpha=1 - d)
        step += 1
        if main_proc and (step % int(cfg.get("log_every", 20)) == 0 or step == step0 + 1):
            print(f"step {step:7d} loss {loss.item():.4f} tok/img {stats['tok_img'].item():7.1f} "
                  f"sigma {stats['sigma'].item():.3f} gnorm {float(gn):.3e} "
                  f"{(step - step0) * gbs / (time.time() - t0):6.2f} img/s", flush=True)
        if main_proc and step % int(cfg.get("save_every", 500)) == 0:
            _atomic_save({"model": raw.state_dict(), "ema": ema, "optimizer": opt.state_dict(),
                          "step": step, "cfg": payload_cfg}, last)
        if main_proc and cfg.get("snapshot_every") and step % int(cfg["snapshot_every"]) == 0:
            _atomic_save({"model": raw.state_dict(), "ema": ema, "step": step, "cfg": payload_cfg},
                         out / f"checkpoint-s{step}.pt")
    if main_proc:
        _atomic_save({"model": raw.state_dict(), "ema": ema, "step": steps, "cfg": payload_cfg},
                     out / "checkpoint-final.pt")
        print("done")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
