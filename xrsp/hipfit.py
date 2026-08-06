"""Classical femoral-head circle fitting on real radiographs. No training, no labels.

This exists to VALIDATE the DRR-trained hip model, and it must share none of its
machinery to be worth anything. A learned validator trained on the same synthetic data
would carry the same synthetic-to-real bias and would agree with the model for the wrong
reason. Canny edges plus a Hough circle transform have never seen a DRR, so where the two
disagree, the disagreement means something.

The search is constrained by anatomy rather than run over the whole film: on a lateral the
femoral head centre sits roughly 1.2-1.8 S1-endplate-lengths below the S1 plate and
anterior to it, with a radius near 0.4-0.6 of that length. BUU ships the S1 annotation, so
that window comes for free and removes most of the false-positive surface (rib ends, bowel
gas, the acetabular rim).

Reported per case: the fitted centre, its radius, the edge support, and whether the fit
converged at all. A case with no confident fit is reported as such -- it is not filled in
with a guess, since the entire purpose is to be an independent reference.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def clahe(img, clip: float = 0.02, nbins: int = 256):
    """Local contrast equalisation. Lumbar-technique films wash the femoral head out
    almost completely, and the Hough edge map is useless without this."""
    from skimage.exposure import equalize_adapthist
    return equalize_adapthist(np.clip(img, 0, 1), clip_limit=clip, nbins=nbins)


def search_window(s1_ant, s1_post, image_shape, *, lo=0.6, hi=2.6, ant_pad=1.6):
    """Pixel bbox where the femoral head must lie, from the S1 endplate annotation.

    Returns (y0, y1, x0, x1). Anterior is assumed to be at DECREASING x (BUU's own
    convention, image-left); pass mirrored points if working in the DRR convention.
    """
    a = np.asarray(s1_ant, float)
    b = np.asarray(s1_post, float)
    L = float(np.linalg.norm(b - a))
    cy = 0.5 * (a[1] + b[1])
    cx = 0.5 * (a[0] + b[0])
    H, W = image_shape
    y0 = int(max(0, cy + lo * L))
    y1 = int(min(H, cy + hi * L))
    x0 = int(max(0, cx - ant_pad * L))
    x1 = int(min(W, cx + 0.8 * L))
    return y0, y1, x0, x1


def fit_head_circles(img, s1_ant, s1_post, *, n_peaks: int = 2,
                     sigma: float = 2.0, rel_lo: float = 0.28, rel_hi: float = 0.75
                     ) -> Dict[str, object]:
    """Hough circle fit for the femoral head(s) inside the anatomic window.

    Returns {'ok', 'center', 'radius', 'support', 'n_found', 'window'}. `center` is the
    MEAN of the accepted circle centres: on a lateral the two heads overlap and either
    both or one may be resolved, and their midpoint is the bicoxofemoral point either way
    (see femhead: for equal radii the arrangement is symmetric about that midpoint).
    """
    from skimage.feature import canny
    from skimage.transform import hough_circle, hough_circle_peaks

    im = np.clip(np.asarray(img, float), 0, 1)
    H, W = im.shape
    L = float(np.linalg.norm(np.asarray(s1_post, float) - np.asarray(s1_ant, float)))
    y0, y1, x0, x1 = search_window(s1_ant, s1_post, (H, W))
    if y1 - y0 < 20 or x1 - x0 < 20:
        return {"ok": False, "reason": "window degenerate", "window": (y0, y1, x0, x1)}

    patch = clahe(im[y0:y1, x0:x1])
    edges = canny(patch, sigma=sigma)
    radii = np.arange(max(4, int(rel_lo * L)), max(6, int(rel_hi * L)), 2)
    if len(radii) < 2 or edges.sum() < 50:
        return {"ok": False, "reason": "no edges", "window": (y0, y1, x0, x1)}
    acc = hough_circle(edges, radii)
    accum, cx, cy, rad = hough_circle_peaks(acc, radii, total_num_peaks=n_peaks,
                                            min_xdistance=int(0.4 * L),
                                            min_ydistance=int(0.4 * L))
    if not len(accum):
        return {"ok": False, "reason": "no peaks", "window": (y0, y1, x0, x1)}
    keep = accum >= 0.25 * float(accum.max())
    cx, cy, rad, accum = cx[keep], cy[keep], rad[keep], accum[keep]
    return {
        "ok": True,
        "center": [float(cx.mean() + x0), float(cy.mean() + y0)],
        "radius": float(np.mean(rad)),
        "support": float(np.mean(accum)),
        "n_found": int(len(cx)),
        "window": (y0, y1, x0, x1),
        "centers": [[float(a + x0), float(b + y0)] for a, b in zip(cx, cy)],
    }
