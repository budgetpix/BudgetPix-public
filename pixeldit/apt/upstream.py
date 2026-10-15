"""Load the released NVlabs PixelDiT model classes from a checkout, by path.

Set `PIXELDIT_ROOT` to the checkout (the directory that contains `pixdit_core/`), or pass `root=`.
"""
from __future__ import annotations

import importlib
import os
import sys

_CACHE = {}


def upstream_root(root: str | None = None) -> str:
    root = root or os.environ.get("PIXELDIT_ROOT", "")
    if not root:
        raise SystemExit("set PIXELDIT_ROOT to a checkout of github.com/NVlabs/PixelDiT "
                         "(the directory containing pixdit_core/)")
    if not os.path.isdir(os.path.join(root, "pixdit_core")):
        raise SystemExit(f"{root} does not contain pixdit_core/ — is it the PixelDiT checkout?")
    return os.path.abspath(root)


def load_pixdit(root: str | None = None):
    """Return the upstream `pixdit_core.pixeldit_t2i` module (with `PixDiT_T2I`)."""
    r = upstream_root(root)
    if r in _CACHE:
        return _CACHE[r]
    if r not in sys.path:
        sys.path.insert(0, r)
    mod = importlib.import_module("pixdit_core.pixeldit_t2i")
    _CACHE[r] = mod
    return mod


def build_released(cfg: dict, root: str | None = None):
    """Instantiate the released `PixDiT_T2I` from a config dict (the released `config.json`)."""
    m = load_pixdit(root)
    keys = ("in_channels", "num_groups", "hidden_size", "pixel_hidden_size", "pixel_attn_hidden_size",
            "pixel_num_groups", "patch_depth", "pixel_depth", "num_text_blocks", "patch_size",
            "txt_embed_dim", "txt_max_length", "use_text_rope", "text_rope_theta",
            "repa_encoder_index", "use_pixel_abs_pos")
    return m.PixDiT_T2I(**{k: cfg[k] for k in keys if k in cfg})
