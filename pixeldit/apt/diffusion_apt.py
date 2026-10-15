"""Training loss and sampler for AdaptivePixDiT.

  * sigmas: sigma_raw = 1 - linspace(1.0, 0.001, 1000), shifted by
    sigma = shift * sigma_raw / (1 + (shift - 1) * sigma_raw), shift = 4.0.
    sigma = 1 is pure noise, sigma = 0 is the clean image.
  * noising: x_sigma = (1 - sigma) * x + sigma * eps, eps ~ N(0, I).
  * target: v = eps - x, plain MSE.
  * the model's timestep input is sigma * 1000; x0 = x - sigma * v.
"""
from __future__ import annotations

import numpy as np
import torch

from .layout import (Geometry, all_base_tokens, detail_pyramid, layouts_batched, pack, pyramid_slice, _leaf_masks_np, _pyr_np,
                     tokens_from_masks)

DIFFUSION_STEPS = 1000
FLOW_SHIFT = 4.0
LOGIT_MEAN, LOGIT_STD = 0.0, 1.0


def flow_sigmas(steps=DIFFUSION_STEPS, shift=FLOW_SHIFT):
    """The released sigma ladder, float64, index 0 = clean .. index 999 = noise."""
    scale = 1000.0 / steps
    betas = np.linspace(scale * 1.0, scale * 0.001, steps, dtype=np.float64)
    s = 1.0 - betas
    return shift * s / (1.0 + (shift - 1.0) * s)


class FlowSchedule:
    """sigma bookkeeping shared by the loss and the sampler."""

    def __init__(self, steps=DIFFUSION_STEPS, shift=FLOW_SHIFT):
        self.steps, self.shift = int(steps), float(shift)
        self.sigmas = flow_sigmas(self.steps, self.shift)

    def sample_sigma(self, batch, device, generator=None):
        """u = sigmoid(N(logit_mean, logit_std)); index = floor(u * steps); sigma = sigmas[index]."""
        u = torch.sigmoid(torch.randn(batch, generator=generator) * LOGIT_STD + LOGIT_MEAN)
        idx = (u * self.steps).long().clamp(0, self.steps - 1)
        sig = torch.from_numpy(self.sigmas)[idx].float()
        return sig.to(device), idx.to(device)

    def t_input(self, sigma):
        """What the network's t_embedder consumes: sigma * diffusion_steps."""
        return sigma * float(self.steps)

    def timesteps(self, n, device):
        """betas = linspace(1, 1/steps, n+1), shift-mapped and flipped: sigma_max -> 0."""
        b = np.linspace(1.0, 1.0 / self.steps, n + 1)
        s = 1.0 - b
        s = self.shift * s / (1.0 + (self.shift - 1.0) * s)
        return torch.tensor(s[::-1].copy(), dtype=torch.float32, device=device)


# ------------------------------------------------------------------------------------ layouts
def draw_layouts(images_u8, geo: Geometry, *, offset_lo=0.0, offset_hi=2.0, anchor_p=0.15, scramble_p=0.10):
    """Training layouts, one per image: with probability `anchor_p` the dense grid; otherwise the entropy quadtree
    at a threshold offset ~ U(offset_lo, offset_hi), its entropy maps replaced by random ones with probability
    `scramble_p`."""
    B = images_u8.shape[0]
    pyr = detail_pyramid(images_u8, geo)
    hi_bits = float(np.log2(geo.entropy_bins))
    maps = _pyr_np(pyr)                                          # {s: (B,g,g) float64}
    anchor = np.zeros(B, dtype=bool); deltas = np.zeros(B, dtype=np.float64)
    for i in range(B):
        if torch.rand(1).item() < anchor_p:
            anchor[i] = True
            continue
        p = pyramid_slice(pyr, i)
        if torch.rand(1).item() < scramble_p:
            for s, v in p.items():
                maps[s][i] = (torch.rand_like(v) * hi_bits).numpy().astype(np.float64)
        deltas[i] = offset_lo + (offset_hi - offset_lo) * torch.rand(1).item()
    masks = _leaf_masks_np(maps, deltas, geo)
    return [all_base_tokens(geo) if anchor[i] else tokens_from_masks(masks, i, geo) for i in range(B)]


# ------------------------------------------------------------------------------------ loss
def training_loss(model, images, y, geo: Geometry, *, sched: FlowSchedule = None, null_y=None,
                  caption_drop=0.1, layout_kw=None, text_mask=None):
    """images (B,3,H,W) in [-1,1]; y (B,Ltxt,2304) Gemma embeddings. Returns (loss, stats)."""
    sched = sched or FlowSchedule()
    b, device = images.shape[0], images.device
    if caption_drop > 0 and null_y is not None:
        drop = torch.rand(b, device=device) < caption_drop
        y = torch.where(drop[:, None, None], null_y.to(y.dtype).to(device).expand_as(y), y)
    sigma, _ = sched.sample_sigma(b, device)
    imgs_u8 = ((images.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    L = pack(draw_layouts(imgs_u8, geo, **(layout_kw or {})), geo, device)
    noise = torch.randn_like(images)
    sb = sigma[:, None, None, None]
    x_t = images * (1.0 - sb) + noise * sb
    v_pred = model(x_t, sched.t_input(sigma), y, L, text_mask)
    loss = (v_pred.float() - (noise - images).float()).pow(2).mean()
    stats = {"tok_img": L["n_per"].float().mean().detach(), "sigma": sigma.mean().detach(), "loss": loss.detach()}
    return loss, stats


# ------------------------------------------------------------------------------------ sampler
@torch.no_grad()
def sample(model, y, geo: Geometry, *, null_y=None, steps=50, cfg_scale=2.75, sched: FlowSchedule = None,
           order=2, budget=None, relayout_every=5, warmup_steps=5, text_mask=None, generator=None, device=None,
           stats=None, x0=None):
    """DPM-Solver++ sampler with cfg on the null embedding and the layout re-estimated from the running x0 estimate.

    order          2 = DPM-Solver++(2M); 1 = Euler.
    budget         target tokens/img; None -> the dense grid.
    stats          filled with n_fwd / fwd_log / tok_trace.
    """
    sched = sched or FlowSchedule()
    device = device or y.device
    b = y.shape[0]
    dtype = next(model.parameters()).dtype
    y = y.to(device=device, dtype=dtype)
    if null_y is not None:
        null_y = null_y.to(device=device, dtype=dtype)
        if null_y.shape[0] == 1:
            null_y = null_y.expand(b, -1, -1)
    x = (x0.to(device=device, dtype=dtype) if x0 is not None else
         torch.randn(b, 3, geo.img, geo.img, device=device, dtype=dtype, generator=generator))
    fwd_log = []

    def fwd(xx, sig, ctx, LL):
        fwd_log.append((int(LL["sizes"].shape[1]),
                        {int(s): int((LL["sizes"] == s).sum()) for s in geo.scales if s != geo.base_patch}))
        return model(xx, sched.t_input(sig), ctx, LL, text_mask).float()

    if budget is not None and budget < geo.n_min:
        raise ValueError(f"budget {budget} below the coarsest grid ({geo.n_min})")
    sig = sched.timesteps(steps, device)                    # (steps+1,), sigma_max -> 0
    L = pack([all_base_tokens(geo) for _ in range(b)], geo, device)
    tok_trace, prev_x0, prev_lam = [], None, None

    for i in range(steps):
        s_i, s_n = sig[i], sig[i + 1]
        si = s_i.expand(b)
        if i >= warmup_steps and (i - warmup_steps) % relayout_every == 0:
            if budget is None or budget >= geo.n_dense:
                if not bool((L["n_per"] == geo.n_dense).all()):
                    L = pack([all_base_tokens(geo) for _ in range(b)], geo, device)
            else:
                x0_est = x.float() - float(s_i) * fwd(x, si, y, L)
                u8 = ((x0_est.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
                L = pack(layouts_batched(detail_pyramid(u8, geo), geo, budget=budget), geo, device)
        tok_trace.append(float(L["n_per"].float().mean()))
        v_c = fwd(x, si, y, L)
        if cfg_scale != 1.0 and null_y is not None:
            v_u = fwd(x, si, null_y, L)
            v = v_u + (v_c - v_u) * cfg_scale
        else:
            v = v_c
        xf = x.float()
        x0_hat = xf - float(s_i) * v
        if order >= 2 and prev_x0 is not None and float(s_n) > 0 and float(s_i) > 0:
            a_i, a_n = 1.0 - float(s_i), 1.0 - float(s_n)
            lam_i = np.log(max(a_i, 1e-12) / max(float(s_i), 1e-12))
            lam_n = np.log(max(a_n, 1e-12) / max(float(s_n), 1e-12))
            h = lam_n - lam_i
            r = (lam_i - prev_lam) / h if h != 0 else 0.0
            D = x0_hat if r == 0 else (1 + 1 / (2 * r)) * x0_hat - (1 / (2 * r)) * prev_x0
            xf = (float(s_n) / float(s_i)) * xf - a_n * (np.exp(-h) - 1.0) * D
            prev_lam = lam_i
        else:
            xf = xf + (float(s_n) - float(s_i)) * v
            if float(s_i) > 0 and float(s_n) > 0:
                prev_lam = np.log(max(1.0 - float(s_i), 1e-12) / max(float(s_i), 1e-12))
        prev_x0 = x0_hat
        x = xf.to(dtype)
    if stats is not None:
        stats["n_fwd"] = len(fwd_log); stats["fwd_log"] = fwd_log; stats["tok_trace"] = tok_trace
    return x, L, float(np.mean(tok_trace))
