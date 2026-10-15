"""Load the released PixelDiT-T2I weights into an upstream `PixDiT_T2I`.

The released file is `{'state_dict': {...}}` with every model tensor under a `core.` prefix, plus an
auxiliary `_repa_projector` (a REPA alignment head, unused at inference).
"""
from __future__ import annotations

import json
import os

import torch


def read_config(path) -> dict:
    """The architecture config that ships next to the weights."""
    p = path if os.path.isfile(path) else os.path.join(os.path.dirname(path), "config.json")
    return json.load(open(p))


def released_state_dict(path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    core = {k[len("core."):]: v for k, v in sd.items() if k.startswith("core.")}
    if not core:
        core = {k: v for k, v in sd.items() if not k.startswith("_repa")}
    return core


def load_released_weights(net, path) -> dict:
    """Load into an upstream `PixDiT_T2I`. Raises on any missing tensor."""
    core = released_state_dict(path)
    missing, unexpected = net.load_state_dict(core, strict=False)
    missing = [k for k in missing if "precompute" not in k]
    if missing:
        raise RuntimeError(f"released checkpoint did not fill {len(missing)} tensors, e.g. {missing[:5]}")
    return {"loaded": len(core), "missing": missing, "unexpected": list(unexpected)}


def describe(rep) -> str:
    return (f"loaded {rep['loaded']} tensors"
            + (f", {len(rep['unexpected'])} unexpected" if rep["unexpected"] else ""))


def _load_cpu(path):
    """torch.load on CPU, memory-mapped when possible."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location="cpu", weights_only=False)


def load_adaptive_checkpoint(path, device="cpu", use_ema=True, verbose=True):
    """Rebuild the adaptive model from a `train_apt.py` checkpoint. Returns (model, geo, cfg, arch).

    The geometry and head shape come from the checkpoint's own `cfg`. `use_ema` loads the EMA weights.
    """
    ck = _load_cpu(path)
    cfg = ck["cfg"]
    from .layout import Geometry
    from .model_apt import AdaptivePixDiT
    from .upstream import build_released
    arch = dict(cfg["net_cfg"])
    arch.update(image_size=int(cfg["image_size"]), patch_size=int(cfg["patch_size"]))
    net = build_released(arch)
    geo = Geometry(img=int(cfg["image_size"]), base_patch=int(cfg["patch_size"]),
                   scales=tuple(cfg["scales"]), thresholds=tuple(cfg["thresholds"]))
    model = AdaptivePixDiT(net, geo, mix_bott=int(cfg.get("mix_bott", 256)),
                           refine_width=int(cfg.get("refine_width", 64)),
                           refine_depth=int(cfg.get("refine_depth", 2)),
                           refine_heads=int(cfg.get("refine_heads", 4)),
                           use_refine=bool(cfg.get("use_refine", True)))
    sd = ck["model"]
    if use_ema and ck.get("ema"):
        sd = dict(sd)
        for k, v in ck["ema"].items():
            sd[k] = v
    model.load_state_dict(sd)
    model = model.to(device).eval()
    if verbose:
        ema = (use_ema and bool(ck.get("ema"))) or bool(ck.get("ema_applied"))
        print(f"[eval] {path}: step {ck.get('step', '?')}, {'EMA' if ema else 'raw'} weights")
    return model, geo, cfg, arch
