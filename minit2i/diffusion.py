"""Flow-matching objective and Euler sampler with BudgetPix token layouts.

Forward process: x_t = t*x + (1 - t)*eps, eps ~ N(0, noise_scale^2) (2.0 at 512),
t ~ sigmoid(N(-0.8, 0.8)); the model predicts x and the loss is the MSE of the implied velocity.
Captions are dropped for 10 % of the images by zeroing their attention mask.
"""
import numpy as np
import torch

from layout import all_base_tokens, detail_pyramid, layouts_for_budget, pack, training_layouts

P_MEAN, P_STD, T_EPS, CAPTION_DROP = -0.8, 0.8, 0.05, 0.1


def loss(model, images, text, mask, geo, noise_scale, layout_kw):
    """images (B,3,H,W) in [-1,1], text (B,P,1024) T5 states, mask (B,P). Returns (loss, mean tokens/img)."""
    b, device = images.shape[0], images.device
    drop = torch.rand(b, device=device) < CAPTION_DROP
    mask = torch.where(drop[:, None], torch.zeros_like(mask), mask)
    t = torch.sigmoid(torch.randn(b, device=device) * P_STD + P_MEAN).clamp(1e-5, 1 - 1e-5)
    u8 = ((images.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    L = pack(training_layouts(u8, geo, **layout_kw), geo, device)
    tb = t[:, None, None, None]
    x_t = images * tb + torch.randn_like(images) * noise_scale * (1.0 - tb)
    denom = (1.0 - tb).clamp_min(T_EPS)
    err = (((model(x_t, t, text, mask, L) - x_t) / denom) - ((images - x_t) / denom)).pow(2)
    return err.mean(), L["n_per"].float().mean()


@torch.no_grad()
def sample(model, text, mask, noise, *, budget=None, steps=50, cfg=5.0, warmup=5, relayout_every=5):
    """Euler sampler with classifier-free guidance. With a budget below the dense grid, the
    first `warmup` steps run dense; then the layout is re-derived from the model's x0 estimate every
    `relayout_every` steps. noise (B,3,H,W) is the scaled initial noise. Returns (images in [-1,1],
    mean tokens per image over the trajectory)."""
    geo = model.geo
    b, device = text.shape[0], noise.device
    dtype = next(model.parameters()).dtype
    text, mask, x = text.to(device, dtype), mask.to(device, dtype), noise.to(device, dtype)
    null_mask = torch.zeros_like(mask)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    L = pack([all_base_tokens(geo)] * b, geo, device)
    adaptive = budget is not None and budget < geo.n_dense
    trace = []
    for i in range(steps):
        t = ts[i].expand(b)
        if adaptive and i >= warmup and (i - warmup) % relayout_every == 0:
            x0 = model(x, t, text, mask, L)
            L = pack(layouts_for_budget(detail_pyramid(((x0.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8), geo),
                                        geo, budget), geo, device)
        trace.append(float(L["n_per"].float().mean()))
        pred = model(x, t, text, mask, L)
        if cfg != 1.0:
            uncond = model(x, t, text, null_mask, L)
            pred = uncond + (pred - uncond) * cfg
        v = (pred - x) / (1.0 - ts[i]).clamp_min(T_EPS)
        x = x + (ts[i + 1] - ts[i]) * v
    return x, float(np.mean(trace))
