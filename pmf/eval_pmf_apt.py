#!/usr/bin/env python
"""FID/IS of (adaptive) pMF under N-NFE sampling at a token budget, with a chosen layout source per step.

    torchrun --nproc_per_node=1 eval_pmf_apt.py --model released --released pMF-L-16.pt --budget 128 --nfe 1 --first self
    torchrun ... --model runs/pmf_l16/checkpoint-s64000.pt --budget 128 --nfe 2 --first dense --later xhat

Sampling: z_1 = noise * noise_scale (per-image noise seeded by the image index), t on linspace(1, 0, nfe+1),
z <- z - (t - r) u(z, t, h = t - r, omega, t_min, t_max, y), with the layout L of each step from:
  dense   the full grid
  self    the entropy quadtree of the model's own dense one-step sample for the same class and noise, bisected to the
          budget (one extra dense forward)
  random  a content-free random entropy map bisected to the budget (control)
  xhat    the entropy quadtree of the previous step's x-prediction x_u = z_t - t u, bisected to the budget
Budget >= the dense token count -> the dense grid. FID/IS vs jit_in256_stats.npz. --num 10000 = 10 per class.
Rows -> evals/eval_pmf_apt.tsv

    PMF_ROOT=/path/to/pMF torchrun --nproc_per_node=4 eval_pmf_apt.py --arch L16 --model ../checkpoints/budgetpix_pmf_l16_256.pt \
        --budget 128 --nfe 1 --first self --num 50000
    PMF_ROOT=/path/to/pMF PMF_WEIGHTS=/weights/pMF-L-16.pt torchrun ... eval_pmf_apt.py --arch L16 --model released --budget 128   # no-training control"""
import argparse, os, sys, shutil, time
import numpy as np, torch, torch.distributed as dist
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upstream import add_pmf_root                                   # noqa: E402
add_pmf_root()
from models.pmfDiT import pmfDiT_B_16, pmfDiT_L_16, pmfDiT_H_16              # noqa: E402
from utils.torch_util import BatchGenerator                                    # noqa: E402
from model_pmf_apt import AdaptivePMF, pmf_geometry, dense_L                   # noqa: E402
from layout import pack, detail_pyramid, _pyr_np, calibrate_offsets_batched, _leaf_masks_np, tokens_from_masks  # noqa: E402

ARCH = {"B16": pmfDiT_B_16, "L16": pmfDiT_L_16, "H16": pmfDiT_H_16}
NOISE_SCALE = {"B16": 1.0, "L16": 1.0, "H16": 2.0}
WEIGHTS = os.environ.get("PMF_WEIGHTS", "")      # the released pMF weights (u-path .pt from the Hub, or the full u+v file)


def load_released(net, path):
    """Load a released pMF file into `net` (the Hub's inference .pt or the converted full training state)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    sd = {(k[4:] if k.startswith("net.") else k): v for k, v in sd.items()}
    res = net.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if k != "rope_freqs" and not k.startswith("v_")]   # inference files carry the u-path only
    assert not miss, miss
    return net


def build(arch, model_arg, use_ema=True, policy="adaptive", reduce_budget=None, released=None):
    """model_arg: 'released' (the pMF weights at `released` / $PMF_WEIGHTS) or a BudgetPix checkpoint (training format
    {model, ema, args} or the release format {model, ema_applied, args}); the latter is self-contained."""
    net = ARCH[arch](eval_mode=False)
    if model_arg == "released" or policy in ("tome", "featsim"):
        released = released or WEIGHTS
        if not released:
            raise SystemExit("the released model needs its weights: pass --released <pMF-L-16.pt> or set PMF_WEIGHTS")
        load_released(net, released)
    if policy in ("tome", "featsim"):
        from reduced_pmf import ReducedPMF
        assert model_arg == "released", "ToMe / FeatSim run on the released model"
        return ReducedPMF(net, policy=policy, budget=reduce_budget)
    ck = None
    if model_arg != "released":
        try:
            ck = torch.load(model_arg, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, TypeError, ValueError):
            ck = torch.load(model_arg, map_location="cpu", weights_only=False)
    model = AdaptivePMF(net, pmf_geometry(net))
    if ck is not None:
        sd = ck["model"] if (ck.get("ema_applied") or not use_ema) else ck["ema"]
        model.load_state_dict(sd, strict=True)
    model.has_v = False                                                      # sampling needs u only
    return model


def make_L(src, idx, geo, budget, dev, labels=None, xhat=None):
    B = len(idx)
    if src == "dense" or budget >= geo.n_dense:
        return dense_L(B, geo, dev)
    coarse = [s for s in geo.scales if s > geo.base_patch]                  # detail_pyramid's keys
    if src == "random":
        hi = float(np.log2(geo.entropy_bins))
        maps = {s: np.stack([np.random.default_rng([i, s]).uniform(0.0, hi, (geo.img // s, geo.img // s)) for i in idx]) for s in coarse}
    elif src in ("xhat", "self"):                                           # self: xhat = the dense pre-pass sample
        u8 = ((xhat.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        maps = _pyr_np(detail_pyramid(u8, geo))
    else:
        raise ValueError(src)
    off = calibrate_offsets_batched(maps, geo, budget)
    masks = _leaf_masks_np(maps, off, geo)
    return pack([tokens_from_masks(masks, j, geo) for j in range(B)], geo, dev, multiple=8)


def cell_scale(L, geo):
    """(B, g, g) token side length covering each base cell."""
    B = L["sizes"].shape[0]; b = geo.base_patch; g = geo.img // b
    out = torch.zeros(B, g, g, dtype=torch.long, device=L["sizes"].device)
    for s in geo.scales:
        bi, ti = (L["sizes"] == s).nonzero(as_tuple=True)
        k = s // b; gy = L["top_left"][bi, ti, 0] // b; gx = L["top_left"][bi, ti, 1] // b
        for dy in range(k):
            for dx in range(k):
                out[bi, gy + dy, gx + dx] = s
    return out


def adherence(L_given, x, idx, geo, budget, dev):
    """Given layout vs the layout of the GENERATED image at the same budget: IoU of the merged (coarse) area and the fraction
    of base cells whose token size agrees; chance = the same against another sample's generated layout (batch rolled by 1)."""
    a = cell_scale(L_given, geo)
    o = cell_scale(make_L("xhat", idx, geo, budget, dev, xhat=x), geo)
    def iou(p, q):
        pm, qm = p > geo.base_patch, q > geo.base_patch
        return ((pm & qm).sum((1, 2)).float() / (pm | qm).sum((1, 2)).clamp(min=1).float())
    oc = o.roll(1, 0)
    return [iou(a, o).sum().item(), iou(a, oc).sum().item(), (a == o).float().mean((1, 2)).sum().item(), (a == oc).float().mean((1, 2)).sum().item(), len(idx)]


def eval_noise(idx, dev, shape):
    """Noise of images idx: BatchGenerator seeded with the image index."""
    return BatchGenerator(dev, idx).randn(shape).float()


@torch.no_grad()
def generate(model, arch, idx, labels, nfe, omega, tmin, tmax, budget, first, later):
    geo = model.geo; B = labels.shape[0]; dev = labels.device
    z = eval_noise(idx, dev, (B, 3, geo.img, geo.img)) * NOISE_SCALE[arch]
    ts = torch.linspace(1.0, 0.0, nfe + 1, device=dev)
    xhat = None; toks = []; L0 = None
    full = lambda v: torch.full((B,), v, device=dev)
    if first == "self" and budget < geo.n_dense:                           # the dense pre-pass that gives the layout
        u = model(z, full(1.0), full(1.0), full(omega), full(tmin), full(tmax), labels, dense_L(B, geo, dev))[0]
        xhat = z - u
    for i in range(nfe):
        t, r = ts[i].expand(B), ts[i + 1].expand(B)
        L = make_L(first if i == 0 else later, idx, geo, budget, dev, labels, xhat)
        L0 = L if i == 0 else L0
        toks.append(L["n_per"].float().mean().item())
        u = model(z, t, t - r, full(omega), full(tmin), full(tmax), labels, L)[0]
        xhat = z - t.view(B, 1, 1, 1) * u
        z = z - (t - r).view(B, 1, 1, 1) * u
    return z, toks, L0


def make_tag(a):
    if a.tag:
        return a.tag
    if a.policy != "adaptive":
        return f"released{a.arch}-{a.policy}_r{a.reduce_budget}_n{a.nfe}_w{a.omega:g}_i{a.tmin:g}-{a.tmax:g}" + (f"_N{a.num}" if a.num != 10000 else "")
    if a.model == "released":
        name = f"released{a.arch}"
    else:
        name = f"{os.path.basename(os.path.dirname(os.path.abspath(a.model)))}-{os.path.basename(a.model)[:-3].replace('checkpoint-', '')}" + ("-raw" if a.raw else "")
    return f"{name}_b{a.budget}_n{a.nfe}_{a.first}" + (f"-{a.later}" if a.nfe > 1 else "") + f"_w{a.omega:g}_i{a.tmin:g}-{a.tmax:g}"


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="L16"); ap.add_argument("--model", default="released", help="'released' or a BudgetPix pMF checkpoint")
    ap.add_argument("--released", default="", help="released pMF weights for --model released (default: $PMF_WEIGHTS)")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--budget", type=int, default=256); ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--first", default="self", choices=["self", "random", "dense"], help="layout of the first step")
    ap.add_argument("--later", default="xhat", choices=["xhat", "dense"], help="layout of the later steps (nfe > 1)")
    ap.add_argument("--omega", type=float, default=7.0); ap.add_argument("--tmin", type=float, default=0.2); ap.add_argument("--tmax", type=float, default=0.7)
    ap.add_argument("--num", type=int, default=10000); ap.add_argument("--bs", type=int, default=125)
    ap.add_argument("--fid_ref", default=os.path.join(HERE, "..", "jit", "fid_stats", "jit_in256_stats.npz"))
    ap.add_argument("--work", default=os.path.join(HERE, "evals", "gen")); ap.add_argument("--tag", default="")
    ap.add_argument("--tsv", default=os.path.join(HERE, "evals", "eval_pmf_apt.tsv")); ap.add_argument("--keep", action="store_true")
    ap.add_argument("--policy", default="adaptive", choices=["adaptive", "tome", "featsim"], help="tome / featsim: training-free baselines inside the released trunk")
    ap.add_argument("--reduce_budget", type=int, default=256, help="tome / featsim: image tokens the reduced core runs on")
    return ap


def main():
    a = parser().parse_args()
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0)); lr_ = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
    dev = f"cuda:{lr_}" if torch.cuda.is_available() else "cpu"; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = build(a.arch, a.model, use_ema=not a.raw, policy=a.policy, reduce_budget=a.reduce_budget, released=a.released).to(dev).eval()
    if a.policy != "adaptive":
        a.first, a.later, a.budget = "dense", "dense", 256                # the layout is ignored; the trunk reduces
    tag = make_tag(a)
    out = os.path.join(a.work, tag)
    if rank == 0:
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    if world > 1: dist.barrier()
    assert a.num % 1000 == 0
    labels_all = np.arange(1000).repeat(a.num // 1000)
    t0 = time.time(); toks = []
    geo = model.geo; adh_on = a.first in ("self", "random") and a.budget < geo.n_dense; adh = torch.zeros(5, dtype=torch.float64, device=dev)
    for start in range(rank * a.bs, a.num, world * a.bs):
        idx = list(range(start, min(start + a.bs, a.num)))
        labels = torch.from_numpy(labels_all[idx]).long().to(dev)
        x, tk, L0 = generate(model, a.arch, idx, labels, a.nfe, a.omega, a.tmin, a.tmax, a.budget, a.first, a.later)
        toks.append(tk if a.policy == "adaptive" else [float(a.reduce_budget)] * len(tk))
        if adh_on:
            adh += torch.tensor(adherence(L0, x, idx, geo, a.budget, dev), dtype=torch.float64, device=dev)
        imgs = np.round(np.clip((x.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2 * 255, 0, 255)).astype(np.uint8)
        for j, i in enumerate(idx):
            Image.fromarray(imgs[j]).save(os.path.join(out, f"{i:05d}.png"))
        if rank == 0 and (start // (world * a.bs)) % 10 == 0:
            print(f"[eval {time.strftime('%H:%M:%S')}] {tag}: {start + world * a.bs}/{a.num} tokens/step {tk}", flush=True)
    tok_mean = torch.tensor(np.mean(np.array(toks), axis=0), device=dev, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(tok_mean); tok_mean /= world; dist.all_reduce(adh); dist.barrier()
    gen_min = (time.time() - t0) / 60
    if rank == 0:
        from utils.fidelity_wrapper import calculate_metrics
        m = calculate_metrics(input1=out, input2=a.fid_ref, cuda=dev.startswith("cuda"), isc=True, fid=True, kid=False, prc=False, verbose=False)
        fid, isc = m["frechet_inception_distance"], m["inception_score_mean"]
        os.makedirs(os.path.dirname(a.tsv), exist_ok=True)
        if not os.path.exists(a.tsv):
            open(a.tsv, "w").write("tag\tmodel\tema\tbudget\tnfe\tfirst\tlater\tomega\ttmin\ttmax\tnum\ttokens_per_step\tFID\tIS\tgen_min\n")
        with open(a.tsv, "a") as f:
            f.write(f"{tag}\t{a.model}\t{int(not a.raw)}\t{a.budget}\t{a.nfe}\t{a.first}\t{a.later if a.nfe > 1 else '-'}\t{a.omega:g}\t{a.tmin:g}\t{a.tmax:g}\t"
                    f"{a.num}\t{','.join(f'{v:.1f}' for v in tok_mean.tolist())}\t{fid:.4f}\t{isc:.2f}\t{gen_min:.1f}\n")
        print(f"[eval] {tag}: FID {fid:.4f} IS {isc:.2f} tokens/step {[round(v, 1) for v in tok_mean.tolist()]} (generation {gen_min:.1f} min)", flush=True)
        if adh_on:
            n = adh[4].item(); v = (adh[:4] / n).tolist()
            ap_ = os.path.join(os.path.dirname(a.tsv), "adherence_pmf_apt.tsv")
            if not os.path.exists(ap_):
                open(ap_, "w").write("tag\tn\tiou_merged\tiou_merged_chance\tcell_agree\tcell_agree_chance\n")
            with open(ap_, "a") as f:
                f.write(f"{tag}\t{int(n)}\t{v[0]:.4f}\t{v[1]:.4f}\t{v[2]:.4f}\t{v[3]:.4f}\n")
            print(f"[eval] {tag}: layout adherence IoU(merged) {v[0]:.3f} (chance {v[1]:.3f}), cell agreement {v[2]:.3f} (chance {v[3]:.3f})", flush=True)
        keep = os.path.join(a.work, "_samples", tag); os.makedirs(keep, exist_ok=True)
        for fn in sorted(os.listdir(out))[:64:4]:
            shutil.copy(os.path.join(out, fn), os.path.join(keep, fn))
        if not a.keep:
            shutil.rmtree(out)
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
