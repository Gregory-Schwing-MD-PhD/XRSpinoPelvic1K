"""Real-time spinal-level localization on radiographs / fluoroscopy.

Goal: given a (live) lateral radiograph or fluoro frame, name the vertebral level under a
query point (e.g. the C-arm centre / a tapped needle) — the intra-op "wrong-level surgery"
guard. Trained on the DRR dataset (build_dataset.py), where every level has a 2-D landmark
projected from 3-D ground truth.

This module provides the RUNNABLE pieces (heatmap targets from the projected landmarks, a
nearest-level readout) plus a thin model interface. The CNN trainer is a guarded-import
scaffold (PyTorch optional) — the data contract is fixed so the model is swappable."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---- targets (runnable, no ML dep) -----------------------------------------
def gaussian_heatmaps(levels: List[Dict], shape, sigma: float = 6.0,
                      names: Optional[List[str]] = None):
    """Per-level Gaussian heatmap stack from projected landmarks (the training target for a
    keypoint/heatmap localizer). `levels` = output of project_level_points. Returns
    (heatmaps (C,H,W) float32, channel_names)."""
    H, W = shape
    names = names or [lv["name"] for lv in levels]
    idx = {n: i for i, n in enumerate(names)}
    hm = np.zeros((len(names), H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for lv in levels:
        if lv["name"] not in idx:
            continue
        x, y = lv["point"]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
        hm[idx[lv["name"]]] = np.maximum(hm[idx[lv["name"]]], g)
    return hm, names


def level_at_point(levels: List[Dict], xy) -> Optional[Dict]:
    """Nearest projected level to a query point — the localization READOUT (used as the
    non-ML baseline on DRRs, and as the post-processing on a model's predicted points)."""
    if not levels:
        return None
    xy = np.asarray(xy, float)
    d = [np.hypot(*(np.asarray(lv["point"]) - xy)) for lv in levels]
    k = int(np.argmin(d))
    return {**levels[k], "distance_px": float(d[k])}


def points_from_heatmaps(hm, names) -> List[Dict]:
    """Argmax decode a heatmap stack back to per-level points (model inference → readout)."""
    out = []
    for i, n in enumerate(names):
        if hm[i].max() <= 0:
            continue
        y, x = np.unravel_index(int(hm[i].argmax()), hm[i].shape)
        out.append({"name": n, "point": [float(x), float(y)], "score": float(hm[i].max())})
    return out


# ---- model scaffold (PyTorch optional) -------------------------------------
def build_model(n_levels: int):
    """A small U-Net heatmap regressor. Requires torch (`pip install xrsp[train]`)."""
    try:
        import torch.nn as nn
    except Exception as e:                       # pragma: no cover
        raise ImportError("install torch to build/train the localizer: pip install torch") from e

    def cbr(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class UNetHeat(nn.Module):
        def __init__(self, c=n_levels):
            super().__init__()
            self.e1, self.e2, self.e3 = cbr(1, 32), cbr(32, 64), cbr(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d2, self.d1 = cbr(128 + 64, 64), cbr(64 + 32, 32)
            self.head = nn.Conv2d(32, c, 1)

        def forward(self, x):
            import torch
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.up(e3), e2], 1))
            d1 = self.d1(torch.cat([self.up(d2), e1], 1))
            return self.head(d1)
    return UNetHeat()


# Training/inference loops live in scripts/train_localizer.py (kept out of the importable
# package so the core DRR engine has no heavy deps). See docs/ROADMAP.md.
