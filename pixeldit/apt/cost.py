"""Compute accounting for AdaptivePixDiT, with the pixel stage reported separately:

    total = patch(n_padded) + heads(coarse tokens) + pixel_floor

measured with `torch.utils.flop_counter` (2 FLOP per multiply-add).
"""
from __future__ import annotations

import contextlib
import torch

from .layout import Geometry, all_base_tokens, pack


class FlopMeter:
    def __init__(self, model, geo: Geometry, txt_len=300, device=None, batch=1):
        self.model, self.geo, self.txt_len, self.batch = model, geo, int(txt_len), int(batch)
        self.device = device or next(model.parameters()).device
        self.net = model.net
        self.dim_txt = self.net.txt_embed_dim
        self._total, self._head, self._floor = {}, {}, None

    def _inputs(self, B):
        x = torch.randn(B, 3, self.geo.img, self.geo.img, device=self.device)
        t = torch.full((B,), 500.0, device=self.device)
        y = torch.randn(B, self.txt_len, self.dim_txt, device=self.device)
        return x, t, y

    @contextlib.contextmanager
    def _no_graph(self):
        """Grad mode on, no parameter requiring grad; flags are restored afterwards."""
        flags = [(q, q.requires_grad) for q in self.model.parameters()]
        for q, _ in flags:
            q.requires_grad_(False)
        try:
            with torch.enable_grad():
                yield
        finally:
            for q, fl in flags:
                q.requires_grad_(fl)

    def _count(self, L):
        import gc
        from torch.utils.flop_counter import FlopCounterMode
        B = L["sizes"].shape[0]
        x, t, y = self._inputs(B)
        fc = FlopCounterMode(display=False)
        with fc, self._no_graph():
            out = self.model(x, t, y, L)
        del out
        flops = fc.get_total_flops()
        del fc; gc.collect(); torch.cuda.empty_cache()
        return flops / 1e9 / B

    def pixel_floor(self):
        """GFLOP/img of the pixel stage alone: embedder + PiT blocks + final layer."""
        if self._floor is None:
            from torch.utils.flop_counter import FlopCounterMode
            B, H = 1, self.geo.img
            b = self.geo.base_patch
            Ld = (H // b) ** 2
            x = torch.randn(B, 3, H, H, device=self.device, requires_grad=True)
            s_cond = torch.randn(B * Ld, self.net.hidden_size, device=self.device)
            fc = FlopCounterMode(display=False)
            with fc, self._no_graph():
                xp = self.net.pixel_embedder(x, img_height=H, img_width=H, patch_size=b)
                for blk in self.net.pixel_blocks:
                    xp = blk(xp, s_cond, H, H, b, None)
                out = self.net.final_layer(xp)
            del out, xp
            self._floor = fc.get_total_flops() / 1e9 / B
        return self._floor

    def total(self, n_pad):
        """GFLOP of one forward whose padded image length is n_pad, all tokens at the base scale."""
        if n_pad not in self._total:
            toks = all_base_tokens(self.geo)[:n_pad]
            L = pack([toks], self.geo, self.device, multiple=1)
            self._total[n_pad] = self._count(L)
        return self._total[n_pad]

    def head(self, s):
        """Marginal GFLOP of one refine head at scale s (one coarse token replacing k*k base tokens)."""
        if s not in self._head:
            base = all_base_tokens(self.geo)
            keep = [tk for tk in base if not (tk[1] < s and tk[2] < s)]
            LA = pack([[(s, 0, 0)] + keep], self.geo, self.device, multiple=1)
            LB = pack([keep], self.geo, self.device, multiple=1)
            self._head[s] = max(0.0, self._count(LA) - self._count(LB)
                                - (self.total(len(keep) + 1) - self.total(len(keep))))
        return self._head[s]

    def gflop_of(self, fwd_log):
        """fwd_log: [(N_pad, {scale: count}), ...] for ONE image. Returns (total, patch, floor)."""
        tot = 0.0
        for n_pad, counts in fwd_log:
            tot += self.total(n_pad) + sum(c * self.head(s) for s, c in counts.items())
        floor = self.pixel_floor() * len(fwd_log)
        return tot, tot - floor, floor

    def report(self):
        d = self.total(self.geo.n_dense)
        f = self.pixel_floor()
        return (f"dense forward {d:.1f} GFLOP/img = patch {d - f:.1f} + pixel floor {f:.1f} "
                f"({100 * f / d:.1f}% fixed); a zero-token patch stage would still cost {f:.1f}")
