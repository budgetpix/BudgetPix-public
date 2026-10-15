"""pMF training objective for the adaptive model: every network call on an image receives that image's layout L, which
is held constant. Perceptual terms: a paired random-resized crop to 224 of (x_hat, x), LPIPS-VGG and the squared
distance of ConvNeXt-V2-B pooled features. Images are in [-1, 1].
Conventions: z_t = (1 - t) x + t e, t = 1 is noise; u, v are velocities; x_hat = z_t - t u."""
import math
import torch
import torch.nn.functional as F
from model_pmf_apt import cat_L


class PerceptualAux:
    """Paired RandomResizedCrop(224, scale (0.08, 1), ratio (3/4, 4/3)) -> (lpips[B], convnext[B])."""

    def __init__(self, device, hf_home=None):
        import os, lpips
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        from transformers import ConvNextV2Model
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        self.cnx = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224",
                                                   cache_dir=(f"{hf_home}/hub" if hf_home else None)).to(device).eval()
        for p in list(self.lpips.parameters()) + list(self.cnx.parameters()):
            p.requires_grad_(False)

    @staticmethod
    def paired_crop(a, b, out=224, g=None):
        B, C, H, W = a.shape
        oa, ob = [], []
        for i in range(B):
            for _ in range(10):
                area = H * W * (0.08 + 0.92 * torch.rand(1, generator=g).item())
                ar = math.exp(math.log(3 / 4) + (math.log(4 / 3) - math.log(3 / 4)) * torch.rand(1, generator=g).item())
                w, h = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
                if 0 < w <= W and 0 < h <= H:
                    break
            else:
                w, h = W, H
            top = int(torch.randint(0, H - h + 1, (1,), generator=g)); left = int(torch.randint(0, W - w + 1, (1,), generator=g))
            oa.append(F.interpolate(a[i:i + 1, :, top:top + h, left:left + w], size=(out, out), mode="bilinear", align_corners=False))
            ob.append(F.interpolate(b[i:i + 1, :, top:top + h, left:left + w], size=(out, out), mode="bilinear", align_corners=False))
        return torch.cat(oa), torch.cat(ob)

    def __call__(self, pred_x, x):
        pa, pb = self.paired_crop(pred_x.float(), x.float())
        lp = self.lpips(pa, pb).flatten()
        cn = ((self.cnx(pixel_values=pa).pooler_output - self.cnx(pixel_values=pb).pooler_output) ** 2).sum(-1)
        return lp, cn


class PMFLoss:
    def __init__(self, model, num_classes=1000, noise_scale=1.0, P_mean=0.8, P_std=0.8, data_proportion=0.5,
                 cfg_beta=1.0, cfg_max=7.0, class_dropout_prob=0.1, norm_p=1.0, norm_eps=0.01,
                 lpips_lambda=0.4, convnext_lambda=0.1, perceptual_max_t=0.8):
        self.model = model                     # AdaptivePMF (raw module, not DDP)
        self.num_classes, self.noise_scale = num_classes, noise_scale
        self.P_mean, self.P_std, self.data_proportion = P_mean, P_std, data_proportion
        self.cfg_beta, self.cfg_max, self.class_dropout_prob = cfg_beta, cfg_max, class_dropout_prob
        self.norm_p, self.norm_eps = norm_p, norm_eps
        self.lpips_lambda, self.convnext_lambda, self.perceptual_max_t = lpips_lambda, convnext_lambda, perceptual_max_t

    def _logit_normal(self, bz, dev):
        return torch.sigmoid(torch.randn(bz, 1, 1, 1, device=dev) * self.P_std + self.P_mean)

    def sample_tr(self, bz, dev):
        t, r = self._logit_normal(bz, dev), self._logit_normal(bz, dev)
        fm = (torch.arange(bz, device=dev) < int(bz * self.data_proportion)).view(bz, 1, 1, 1)
        r = torch.where(fm, t, r)
        return torch.maximum(t, r), torch.minimum(t, r), fm

    def sample_cfg_scale(self, bz, dev):
        u = torch.rand(bz, 1, 1, 1, device=dev); smax = torch.tensor(float(self.cfg_max), device=dev)
        if self.cfg_beta == 1.0:
            return torch.exp(u * torch.log1p(smax))
        b = float(self.cfg_beta); lb = (1.0 - b) * torch.log1p(smax)
        return torch.exp(torch.log1p(u * torch.expm1(lb)) / (1.0 - b))

    def sample_cfg_interval(self, bz, fm, dev):
        tmin = torch.rand(bz, 1, 1, 1, device=dev) * 0.5
        tmax = 0.5 + torch.rand(bz, 1, 1, 1, device=dev) * 0.5
        return torch.where(fm, 0.0, tmin), torch.where(fm, 1.0, tmax)

    def net(self, x, t, h, omega, tmin, tmax, y, L):
        bz = x.shape[0]
        return self.model(x, t.reshape(bz), h.reshape(bz), omega.reshape(bz), tmin.reshape(bz), tmax.reshape(bz), y, L)

    def v_cond(self, x, t, omega, y, L):
        return self.net(x, t, torch.zeros_like(t), omega, torch.zeros_like(t), torch.ones_like(t), y, L)[1]

    @torch.no_grad()
    def guidance(self, v_t, z_t, t, y, fm, w, tmin, tmax, L):
        bz = z_t.shape[0]
        y_null = torch.full_like(y, self.num_classes)
        both = self.v_cond(torch.cat([z_t, z_t]), torch.cat([t, t]), torch.cat([w, torch.ones_like(w)]), torch.cat([y, y_null]), cat_L(L, L))
        v_c, v_u = both[:bz], both[bz:]
        v_g_fm = v_t + (1 - 1 / w) * (v_c - v_u)
        w = torch.where((t >= tmin) & (t <= tmax), w, torch.ones_like(w))
        v_c = self.v_cond(z_t, t, w, y, L)
        v_g = v_t + (1 - 1 / w) * (v_c - v_u)
        return torch.where(fm, v_g_fm, v_g), v_c

    def cond_drop(self, v_t, v_g, labels):
        bz = v_t.shape[0]
        nd = (torch.rand(bz, device=v_t.device) < self.class_dropout_prob).sum()
        drop = torch.arange(bz, device=v_t.device) < nd
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels), torch.where(drop.view(bz, 1, 1, 1), v_t, v_g)

    def __call__(self, x, labels, L, aux_fn=None):
        bz, dev = x.shape[0], x.device
        t, r, fm = self.sample_tr(bz, dev)
        e = torch.randn_like(x) * self.noise_scale
        z_t = (1 - t) * x + t * e
        v_t = (z_t - x) / t.clamp(0.05, 1.0)
        tmin, tmax = self.sample_cfg_interval(bz, fm, dev)
        omega = self.sample_cfg_scale(bz, dev)
        v_g, v_c = self.guidance(v_t, z_t, t, labels, fm, omega, tmin, tmax, L)
        labels, v_g = self.cond_drop(v_t, v_g, labels)

        def u_fn(z, tt, rr):
            return self.net(z, tt, tt - rr, omega, tmin, tmax, labels, L)       # (u, v): v rides along as aux

        u, du_dt, v = torch.func.jvp(u_fn, (z_t, t, r), (v_c, torch.ones_like(t), torch.zeros_like(r)), has_aux=True)
        V = u + (t - r) * du_dt.detach()
        v_g = v_g.detach()
        adp = lambda l: l / (l.detach() + self.norm_eps) ** self.norm_p
        lu = ((V - v_g) ** 2).sum(dim=(1, 2, 3)); lv = ((v - v_g) ** 2).sum(dim=(1, 2, 3))
        loss_u, loss_v = adp(lu), adp(lv)
        lp = cn = torch.zeros_like(loss_u); aux = torch.zeros_like(loss_u)
        if aux_fn is not None:
            lp, cn = aux_fn(z_t - t * u, x)
            m = t.flatten() < self.perceptual_max_t
            lp, cn = torch.where(m, lp, 0.0), torch.where(m, cn, 0.0)
            aux = adp(lp) * self.lpips_lambda + adp(cn) * self.convnext_lambda
        loss = (loss_u + loss_v + aux).mean()
        return loss, {"loss": loss.detach(), "mse_u": ((V - v_g) ** 2).mean().detach(), "mse_v": ((v - v_g) ** 2).mean().detach(),
                      "lpips": lp.mean().detach(), "convnext": cn.mean().detach(), "tok": L["n_per"].float().mean()}
