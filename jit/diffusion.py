"""JiT's flow-matching objective and Heun sampler, with the BudgetPix layout.

Forward process: z = t*x + (1 - t)*eps, eps ~ N(0, noise_scale^2), t ~ sigmoid(N(-0.8, 0.8)); the
model predicts x and the loss is the MSE of the implied velocity (x - z) / (1 - t).
"""
import numpy as np
import torch

from layout import all_base_tokens, detail_pyramid, layouts_for_budget, pack
from model import NUM_CLASSES

P_MEAN, P_STD, T_EPS, LABEL_DROP = -0.8, 0.8, 5e-2, 0.1


def loss(model, images_u8, labels, L, noise_scale):
    """images_u8 (B,3,H,W) uint8, labels (B,), L the packed training layout."""
    x = images_u8.float().div(255).mul_(2).sub_(1)
    drop = torch.rand(labels.shape[0], device=x.device) < LABEL_DROP                  # classifier-free guidance
    y = torch.where(drop, torch.full_like(labels, NUM_CLASSES), labels)
    t = torch.sigmoid(torch.randn(x.shape[0], device=x.device) * P_STD + P_MEAN)
    tb = t.view(-1, 1, 1, 1)
    z = tb * x + (1 - tb) * torch.randn_like(x) * noise_scale
    denom = (1 - tb).clamp_min(T_EPS)
    v_pred = (model(z, t, y, L).float() - z) / denom
    return (((x - z) / denom - v_pred) ** 2).mean()


def default_warmup(geo, steps):
    """Dense steps before the first layout: 10 of 50 at 256, 5 of 50 at 512."""
    return max(1, round(steps * (0.2 if geo.img <= 256 else 0.1)))


@torch.no_grad()
def sample(model, labels, noise, *, budget=None, steps=50, cfg=3.0, interval=(0.1, 1.0), warmup=None,
           relayout_every=5):
    """Heun sampler (CFG on t in `interval`). With a budget below the dense grid, the first `warmup`
    steps run dense; then the layout is re-derived from the model's x0 estimate every `relayout_every`
    steps. noise (B,3,H,W) is the scaled initial noise. Returns (images in [-1,1], final token lists,
    mean tokens per image over the trajectory)."""
    geo = model.geo
    B = labels.shape[0]
    warmup = default_warmup(geo, steps) if warmup is None else warmup
    ts = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=torch.float64).float().view(-1, 1, 1, 1, 1).expand(-1, B, -1, -1, -1)
    uncond = torch.full_like(labels, NUM_CLASSES)

    def velocity(z, t, L):
        denom = (1 - t).clamp_min(T_EPS)
        v_cond = (model(z, t.flatten(), labels, L) - z) / denom
        if cfg == 1.0:
            return v_cond
        v_unc = (model(z, t.flatten(), uncond, L) - z) / denom
        on = (t < interval[1]) & ((t > interval[0]) | (interval[0] == 0))
        scale = torch.where(on, torch.tensor(cfg, device=z.device), torch.tensor(1.0, device=z.device))
        return v_unc + scale * (v_cond - v_unc)

    tokens = [all_base_tokens(geo)] * B
    L = pack(tokens, geo, noise.device)
    adaptive = budget is not None and budget < geo.n_dense
    z, trace = noise, []
    for i in range(steps - 1):
        if adaptive and i >= warmup and (i - warmup) % relayout_every == 0:
            x0 = model(z, ts[i].flatten(), labels, L)
            tokens = layouts_for_budget(detail_pyramid((x0.clamp(-1, 1) + 1) / 2 * 255, geo), geo, budget)
            L = pack(tokens, geo, noise.device)
        v1 = velocity(z, ts[i], L)                               # Heun step
        v2 = velocity(z + (ts[i + 1] - ts[i]) * v1, ts[i + 1], L)
        z = z + (ts[i + 1] - ts[i]) * 0.5 * (v1 + v2)
        trace.append(float(L["n_per"].float().mean()))
    z = z + (ts[-1] - ts[-2]) * velocity(z, ts[-2], L)           # final Euler step
    trace.append(trace[-1] if trace else float(L["n_per"].float().mean()))
    return z, tokens, float(np.mean(trace))
