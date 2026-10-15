#!/usr/bin/env python
"""Fine-tune a released pMF (pixel MeanFlow) with the BudgetPix adaptive tokens.

    PMF_ROOT=/path/to/pMF torchrun --nproc_per_node=7 train_pmf_apt.py --arch L16 --data /data/imagenet-1k/data \
        --ckpt ../checkpoints/pmf_released_L16_full_uv.pt --out runs/pmf_l16 --steps 64000 --batch 13 --anchor_p 0.30

The released weights must be the FULL training state (u- and v-branches): the inference files on the Hub carry the u-path
only; the converted B/16 and L/16 files are in the Hugging Face model repo next to the checkpoints."""
import argparse, copy, os, sys, time
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upstream import add_pmf_root                                   # noqa: E402
add_pmf_root()
from models.pmfDiT import pmfDiT_B_16, pmfDiT_L_16, pmfDiT_H_16   # noqa: E402
from model_pmf_apt import AdaptivePMF, pmf_geometry                 # noqa: E402
from loss_pmf_apt import PMFLoss, PerceptualAux                     # noqa: E402
from draw import draw_packed                                         # noqa: E402
from data_imagenet import ImageNetParquet, collate                   # noqa: E402

ARCH = {"B16": pmfDiT_B_16, "L16": pmfDiT_L_16, "H16": pmfDiT_H_16}


def load_pmf(arch, ckpt):
    net = ARCH[arch](eval_mode=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if k != "rope_freqs"]
    if missing or res.unexpected_keys:
        raise SystemExit(f"checkpoint mismatch: missing {missing[:5]} unexpected {res.unexpected_keys[:5]}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="B16"); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=32, help="per GPU")
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--new_lr_mult", type=float, default=8.0)
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--ema", type=float, default=0.9995); ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--anchor_p", type=float, default=0.15); ap.add_argument("--scramble_p", type=float, default=0.10)
    ap.add_argument("--offset_lo", type=float, default=0.0); ap.add_argument("--offset_hi", type=float, default=2.0)
    ap.add_argument("--perceptual", type=int, default=1); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data", required=True, help="ImageNet-1k parquet dir (train-*.parquet)")
    ap.add_argument("--mem_frac", type=float, default=0.0, help="cap this process's GPU memory (for a shared GPU)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--cfg_beta", type=float, default=None, help="pMF's omega-sampling exponent; default = the released config (B16 1.0, L16 2.0)")
    ap.add_argument("--lr_decay_start", type=int, default=-1,
                    help="cooldown: from this step the lr decays linearly to 0 over --lr_decay_steps (-1 = constant lr after warm-up)")
    ap.add_argument("--lr_decay_steps", type=int, default=8000)
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0)); lrank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl"); torch.cuda.set_device(lrank)
    dev = f"cuda:{lrank}"
    if a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, lrank)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1234 + rank)
    main_proc = rank == 0
    os.makedirs(a.out, exist_ok=True)
    say = (lambda *m: print(f"[train {time.strftime('%H:%M:%S')}]", *m, flush=True)) if main_proc else (lambda *m: None)

    net = load_pmf(a.arch, a.ckpt)
    geo = pmf_geometry(net)
    model = AdaptivePMF(net, geo).to(dev).train()
    new = [p for n, p in model.named_parameters() if not n.startswith("net.")]
    old = [p for n, p in model.named_parameters() if n.startswith("net.")]
    groups = [{"params": old, "lr": a.lr, "base": a.lr},
              {"params": new, "lr": a.lr * a.new_lr_mult, "base": a.lr * a.new_lr_mult}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    step = 0
    last = os.path.join(a.out, "checkpoint-last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        say(f"resumed {last} at step {step}")
    if a.cfg_beta is None:
        a.cfg_beta = {"B16": 1.0, "L16": 2.0}[a.arch]
    loss_fn = PMFLoss(model, cfg_beta=a.cfg_beta)
    aux = PerceptualAux(dev) if a.perceptual else None
    ds = ImageNetParquet(data_dir=a.data, seed=step * 7919, rank=rank, world_size=world)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=collate, drop_last=True,
                                     persistent_workers=True, prefetch_factor=4)
    it = iter(dl)
    say(f"arch {a.arch}, world {world} x batch {a.batch} = {world * a.batch}, new params {sum(p.numel() for p in new)/1e6:.2f}M, "
        f"lr {a.lr} (x{a.new_lr_mult} new), anchor {a.anchor_p}, scramble {a.scramble_p}, window U({a.offset_lo},{a.offset_hi}), perceptual {a.perceptual}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    t0 = time.time(); step0 = step; hist = []; bad_steps = 0
    while step < a.steps:
        imgs, labels = next(it)
        L = draw_packed(imgs, geo, dev, offset_lo=a.offset_lo, offset_hi=a.offset_hi, anchor_p=a.anchor_p, scramble_p=a.scramble_p)
        x = imgs.to(dev, non_blocking=True).float() / 127.5 - 1.0
        labels = labels.to(dev)
        loss, st = loss_fn(x, labels, L, aux_fn=aux)
        ok = torch.isfinite(loss.detach()).float()
        if world > 1:
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if ok.item() < 1:
            bad_steps += 1
            say(f'non-finite loss at step {step} on some rank ({bad_steps} skipped so far)')
            if bad_steps > 20:
                raise SystemExit('stopping: >20 non-finite losses')
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        if world > 1:
            flat = torch.cat([p.grad.reshape(-1) for p in trainable]); dist.all_reduce(flat); flat /= world
            o = 0
            for p in trainable:
                n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p)); o += n
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.clip)
        warm = min(1.0, (step + 1) / max(1, a.warmup))
        if a.lr_decay_start >= 0 and step >= a.lr_decay_start:
            warm *= max(0.0, 1.0 - (step - a.lr_decay_start) / max(1, a.lr_decay_steps))
        for g in opt.param_groups:
            g["lr"] = g["base"] * warm
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
        step += 1
        if main_proc:
            hist.append({k: float(v) for k, v in st.items()} | {"gn": float(gn)})
            if step % a.log_every == 0 or step == step0 + 1:
                m = {k: sum(h[k] for h in hist) / len(hist) for k in hist[0]}; hist = []
                ips = (step - step0) * a.batch * world / (time.time() - t0)
                say(f"step {step:6d} loss {m['loss']:.4f} mse_u {m['mse_u']:.4f} mse_v {m['mse_v']:.4f} lpips {m['lpips']:.4f} "
                    f"cnx {m['convnext']:.3f} gn {m['gn']:.3f} tok {m['tok']:.0f} {ips:.1f} img/s mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}G")
            if step % a.save_every == 0 or step == a.steps:
                ckd = {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a)}
                torch.save(ckd, last + ".tmp"); os.replace(last + ".tmp", last)
                torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step, "args": vars(a)}, os.path.join(a.out, f"checkpoint-s{step}.pt"))
                say(f"saved step {step}")
    if main_proc:
        say("done")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
