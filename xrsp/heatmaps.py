"""Landmark heatmap targets and decoding.

The model predicts one Gaussian heatmap per landmark rather than a mask. A mask
cannot express an endplate on a projection -- the ala superimposes on the S1 body,
so a silhouette fit is 26-32 deg wrong at every slab fraction -- and a 1-2 px
endplate ribbon is unstable to segment and would still need a line fit afterwards.
Two well-localised corners are unambiguous, and the angles we care about are far
more sensitive to line ORIENTATION than to a couple of pixels of position.

Channel order is fixed by `channel_names()` and must not be reordered: checkpoints,
metrics and the BUU comparison all index by it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# cranial -> caudal, matching xrsp.build_dataset.CORNER_LEVELS
LEVELS: Tuple[str, ...] = tuple(
    [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 14)]
    + [f"L{i}" for i in range(1, 7)] + ["S1"]
)
CORNER_KEYS: Tuple[str, ...] = ("sup_ant", "sup_post", "inf_ant", "inf_post")

# Default working set. The full cervical->sacral list gives 109 channels, of which a
# lumbar-FOV film populates ~33; the masked loss handles that correctly but the unused
# channels still cost memory. T8..S1 covers what a lumbosacral acquisition actually
# shows while keeping every level needed for LL, PI/SS/PT and the segmental angles.
DEFAULT_LEVELS: Tuple[str, ...] = tuple(
    [f"T{i}" for i in range(8, 14)] + [f"L{i}" for i in range(1, 7)] + ["S1"]
)
FEMORAL_KEY = "bicoxofemoral"


def channel_names(levels: Sequence[str] = LEVELS) -> List[str]:
    """Fixed channel order: 4 corners per level, cranial->caudal, then the femoral head."""
    out = [f"{lv}.{k}" for lv in levels for k in CORNER_KEYS]
    out.append(FEMORAL_KEY)
    return out


def gaussian_heatmaps(points: Dict[str, Optional[Sequence[float]]], shape: Tuple[int, int],
                      *, sigma: float = 3.0, names: Sequence[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """(heatmaps [C,H,W] float32, valid [C] bool).

    `valid` is the loss mask: a level absent from this scan -- or a corner the 3-D
    fit could not resolve -- contributes NOTHING to the loss. Supervising an absent
    landmark with an all-zero target teaches the model that the landmark does not
    exist in images where it merely was not annotated, which is the standard way
    partial annotation quietly poisons a landmark model.
    """
    names = list(names) if names is not None else channel_names()
    H, W = shape
    hm = np.zeros((len(names), H, W), np.float32)
    valid = np.zeros(len(names), bool)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    r = int(np.ceil(3 * sigma))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    blob = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    for c, nm in enumerate(names):
        p = points.get(nm)
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H):
            continue                      # off-detector: absent, not zero-supervised
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)
        x0, x1 = max(0, xi - r), min(W, xi + r + 1)
        hm[c, y0:y1, x0:x1] = np.maximum(
            hm[c, y0:y1, x0:x1],
            blob[y0 - (yi - r):y1 - (yi - r), x0 - (xi - r):x1 - (xi - r)])
        valid[c] = True
    return hm, valid


def points_from_json(corners: Dict, bicox_px: Optional[Sequence[float]],
                     names: Sequence[str] = None) -> Dict[str, Optional[List[float]]]:
    """Flatten a `<view>_corners.json` into {channel_name: [x, y] or None}."""
    names = list(names) if names is not None else channel_names()
    pts: Dict[str, Optional[List[float]]] = {n: None for n in names}
    for lv, cs in (corners or {}).items():
        for k in CORNER_KEYS:
            if k in cs:
                key = f"{lv}.{k}"
                if key in pts:
                    pts[key] = [float(cs[k][0]), float(cs[k][1])]
    if bicox_px is not None:
        pts[FEMORAL_KEY] = [float(bicox_px[0]), float(bicox_px[1])]
    return pts


def soft_argmax(hm, *, beta: float = 100.0, thresh: float = 0.05):
    """Sub-pixel landmark from a heatmap, per channel.

    Soft-argmax rather than argmax because the metric is an ANGLE: quantising a
    corner to the nearest pixel puts a floor on the achievable angular accuracy
    (on a 35 mm endplate at 1 mm/px, +-0.5 px on each corner is already ~1.6 deg).

    Returns (points [C,2] in (x, y), confidence [C]). Confidence is the channel max
    BEFORE normalisation -- a channel whose peak is under `thresh` is reported as
    NaN, i.e. "this landmark is not in this image", not a spurious coordinate.
    """
    import torch
    single = hm.dim() == 3
    x = hm.unsqueeze(0) if single else hm
    B, C, H, W = x.shape
    conf = x.flatten(2).max(-1).values                       # [B, C]
    flat = (x.flatten(2) * beta).softmax(-1).view(B, C, H, W)
    ys = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
    py = (flat * ys).sum((-2, -1))
    px = (flat * xs).sum((-2, -1))
    pts = torch.stack([px, py], -1)                          # [B, C, 2]
    pts = torch.where((conf < thresh).unsqueeze(-1),
                      torch.full_like(pts, float("nan")), pts)
    if single:
        return pts[0], conf[0]
    return pts, conf


def masked_mse(pred, target, valid):
    """Per-channel MSE over VALID channels only. `valid` is [B, C] bool."""
    import torch
    m = valid.to(pred.dtype).view(*valid.shape, 1, 1)
    denom = m.sum() * pred.shape[-1] * pred.shape[-2]
    if float(denom) == 0:
        return pred.sum() * 0.0
    return ((pred - target) ** 2 * m).sum() / denom
