#!/usr/bin/env python
"""The FID-vs-budget ladder for one checkpoint.

    PIXELDIT_ROOT=/path/to/PixelDiT torchrun --nproc_per_node=4 eval_apt.py \
        --ckpt runs/pixeldit_1024/checkpoint-final.pt --line budgetpix \
        --prompts eval/prompts_2k.pt --real eval/real_2k \
        --budgets 4096,3645,3277,2867,2458,2048,1638,1229,1024 --tsv evals/ladder_1024.tsv

Every cell (line x budget x cfg) is tag-keyed in the TSV: re-running skips finished cells.

The table reports three FLOP columns:

    gflop_img = gflop_patch + gflop_floor

The pixel stage runs over every pixel of the image whatever the token budget, so `gflop_floor` is
constant down the whole ladder.
"""
from __future__ import annotations

import argparse, csv, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apt import FlowSchedule, sample                              # noqa: E402
from apt.cost import FlopMeter                                    # noqa: E402

COLS = ["tag", "line", "budget_tok", "cfg", "steps", "order", "n_prompts", "tok_final",
        "tok_traj", "n_fwd", "gflop_img", "gflop_patch", "gflop_floor", "FID", "IS", "minutes",
        "gen_bsz", "ckpt"]


def parse_cfgs(spec, budgets):
    """'2.75' -> every budget; '4096:2.75,1024:4' -> per budget (a missing budget uses the last)."""
    if ":" not in spec:
        return {b: [float(x) for x in spec.split("/")] for b in budgets}
    d = {}
    for part in spec.split(","):
        k, v = part.split(":")
        d[int(k)] = [float(x) for x in v.split("/")]
    last = list(d.values())[-1]
    return {b: d.get(b, last) for b in budgets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="adaptive checkpoint (train_apt.py output)")
    ap.add_argument("--released", default="", help="released .pth: the no-training control")
    ap.add_argument("--released_cfg", default="configs/pixeldit_1024.yml")
    ap.add_argument("--line", required=True)
    ap.add_argument("--prompts", required=True, help="encode_prompts.py output (emb/mask/null)")
    ap.add_argument("--real", required=True, help="folder of real images for FID")
    ap.add_argument("--real_name", default="", help="torch-fidelity feature cache name (default: folder name)")
    ap.add_argument("--n", type=int, default=0, help="prompts to use (default: all)")
    ap.add_argument("--budgets", default="4096,3645,3277,2867,2458,2048,1638,1229,1024",
                    help="the recipe ladder: 100/89/80/70/60/50/40/30%% of the dense grid + the 25%% point")
    ap.add_argument("--cfgs", default="2.75")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--order", type=int, default=2, help="2 = DPM-Solver++(2M), upstream's default")
    ap.add_argument("--gen_bsz", type=int, default=4, help="per GPU; keep IDENTICAL across every line")
    ap.add_argument("--relayout_every", type=int, default=5)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ema", action="store_true")
    ap.add_argument("--tsv", default="evals/ladder_1024.tsv")
    ap.add_argument("--work", default="evals/gen")
    ap.add_argument("--keep_images", action="store_true")
    a = ap.parse_args()

    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world > 1:
        import datetime
        torch.distributed.init_process_group(backend, timeout=datetime.timedelta(hours=6))
        if backend == "nccl":
            torch.cuda.set_device(local)
    device = f"cuda:{local}" if backend == "nccl" else "cpu"
    main_proc = rank == 0

    # ---- model
    import yaml
    if a.ckpt:
        from apt.convert import load_adaptive_checkpoint
        model, geo, cfg, _ = load_adaptive_checkpoint(a.ckpt, device=device, use_ema=not a.no_ema,
                                                      verbose=main_proc)
        src = a.ckpt
    elif a.released:
        from train_apt import build_model, geometry_of, resolve
        rc = resolve(a.released_cfg)
        cfg = yaml.safe_load(open(rc))
        cfg = dict(cfg, checkpoint=a.released, _config_path=os.path.abspath(rc))
        model, _, _ = build_model(cfg, device, verbose=main_proc)
        model.eval()
        geo = geometry_of(cfg)
        src = a.released
    else:
        raise SystemExit("--ckpt or --released")
    sched = FlowSchedule(int(cfg.get("diffusion_steps", 1000)), float(cfg.get("flow_shift", 4.0)))

    # ---- prompts + the null embedding for CFG
    P = torch.load(a.prompts, map_location="cpu", weights_only=False)
    N = a.n or P["emb"].shape[0]
    emb_all, mask_all = P["emb"][:N], P["mask"][:N]
    null_y = P.get("null")
    if null_y is None:
        raise SystemExit(f"{a.prompts} has no 'null' embedding — regenerate it with encode_prompts.py "
                         f"(classifier-free guidance needs the negative-prompt embedding, not a zero mask)")
    real_name = a.real_name or Path(a.real).name

    budgets = [int(x) for x in a.budgets.split(",")]
    cfgs = parse_cfgs(a.cfgs, budgets)
    cells = [(b, c) for b in budgets for c in cfgs[b]]

    meter = FlopMeter(model, geo, txt_len=emb_all.shape[1], device=device) if main_proc else None
    if main_proc:
        print("[cost]", meter.report(), flush=True)
    tsv = Path(a.tsv); tsv.parent.mkdir(parents=True, exist_ok=True)
    if main_proc and not tsv.exists():
        tsv.write_text("\t".join(COLS) + "\n")

    for (b, c) in cells:
        tag = f"{a.line}_b{b}_cfg{c:g}_s{a.steps}o{a.order}_n{N}"
        done = False
        if tsv.exists():
            with open(tsv) as f:
                done = any(r["tag"] == tag for r in csv.DictReader(f, delimiter="\t"))
        if done:
            if main_proc:
                print(f"[eval] {tag} done", flush=True)
            continue
        if main_proc:
            print(f"[eval] {tag}  {time.strftime('%m-%d %H:%M')}", flush=True)
        t0 = time.time()
        work = Path(a.work) / tag
        if main_proc:
            shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
        if world > 1:
            torch.distributed.barrier()
        idx = np.arange(N)[rank::world]
        tok_final, tok_traj, n_fwd, gf = [], [], [], []
        from PIL import Image
        for i0 in range(0, len(idx), a.gen_bsz):
            ii = idx[i0:i0 + a.gen_bsz]
            e = emb_all[ii].to(device=device, dtype=torch.float32)
            m = mask_all[ii].to(device=device, dtype=torch.float32)
            x0 = torch.stack([torch.randn(3, geo.img, geo.img,
                                          generator=torch.Generator().manual_seed(a.seed * 1_000_003 + int(k)))
                              for k in ii]).to(device)              # image k always gets noise k
            st = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                x, L, mean = sample(model, e, geo, null_y=null_y, steps=a.steps, cfg_scale=c, sched=sched,
                                    order=a.order, budget=(None if b >= geo.n_dense else b),
                                    relayout_every=a.relayout_every, warmup_steps=a.warmup_steps,
                                    text_mask=None, stats=st, x0=x0, device=device)
            tok_final.append(float(L["n_per"].float().mean())); tok_traj.append(mean); n_fwd.append(st["n_fwd"])
            if main_proc:
                per_img = [(n, {s: cnt / len(ii) for s, cnt in d.items()}) for n, d in st["fwd_log"]]
                gf.append(meter.gflop_of(per_img))
            u8 = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(lambda ka: Image.fromarray(ka[1]).save(work / f"{int(ka[0]):06d}.png"), zip(ii, u8)))
        if world > 1:
            torch.distributed.barrier()
        if main_proc:
            import torch_fidelity
            r = torch_fidelity.calculate_metrics(input1=str(work), input2=a.real, cuda=device.startswith("cuda"),
                                                 fid=True, isc=True, verbose=False,
                                                 input2_cache_name=real_name, cache_root=str(Path(a.work) / "cache"))
            tot = float(np.mean([g[0] for g in gf])); pat = float(np.mean([g[1] for g in gf]))
            flo = float(np.mean([g[2] for g in gf]))
            row = dict(tag=tag, line=a.line, budget_tok=b, cfg=c, steps=a.steps,
                       order=a.order, n_prompts=N, tok_final=f"{np.mean(tok_final):.1f}",
                       tok_traj=f"{np.mean(tok_traj):.1f}", n_fwd=int(np.mean(n_fwd)),
                       gflop_img=f"{tot:.3f}", gflop_patch=f"{pat:.3f}", gflop_floor=f"{flo:.3f}",
                       FID=f"{r['frechet_inception_distance']:.4f}", IS=f"{r['inception_score_mean']:.2f}",
                       minutes=int((time.time() - t0) / 60), gen_bsz=a.gen_bsz, ckpt=src)
            with open(tsv, "a") as f:
                f.write("\t".join(str(row[k]) for k in COLS) + "\n")
            print(f"[eval]   -> tokens {row['tok_final']} (traj {row['tok_traj']})  "
                  f"GFLOP/img {row['gflop_img']} = patch {row['gflop_patch']} + floor {row['gflop_floor']}  "
                  f"FID {row['FID']}  IS {row['IS']}  ({row['minutes']}m)", flush=True)
            if not a.keep_images:
                shutil.rmtree(work, ignore_errors=True)
        if world > 1:
            torch.distributed.barrier()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
