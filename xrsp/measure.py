"""Spinopelvic parameters from 2-D landmarks — the readout the model feeds.

Everything here takes ONLY predicted landmarks, no CT, so it is exactly what runs at
inference on a real radiograph.

Angle conventions follow the radiographic definitions:
  SS  S1 superior endplate vs the horizontal
  PT  (femoral head axis -> S1 endplate anchor) vs the vertical
  PI  the endplate normal vs (anchor -> femoral head axis); PI = SS + PT
  LL  Cobb between the L1 superior and S1 superior endplates
  segmental  Cobb between one level's inferior and the next level's superior endplate

The PI/PT anchor defaults to the CORNER MIDPOINT -- bisect the S1 superior endplate
between its corners. That is the operational method PI was defined with on lateral
radiographs (Legaye/Duval-Beaupere), so it is the convention the published norms
carry, and it is the only anchor derivable from landmarks a model can see. ostk's
over-mask anchor is available where it has been emitted; see docs/PIPELINE.md.

Validated against the 3-D pipeline on case 0003 (landmarks only, no CT):
    SS 32.89 vs 32.31   LL 45.06 vs 45.70   (sub-degree)
Fitting the same endplate from the 2-D silhouette instead is 26-32 deg wrong.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np


def _ang(v) -> float:
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def _acute(d: float) -> float:
    d = abs(d) % 180.0
    return 180.0 - d if d > 90.0 else d


def _pt(points: Dict[str, Optional[Sequence[float]]], key: str):
    p = points.get(key)
    if p is None:
        return None
    a = np.asarray(p, float)
    return None if not np.all(np.isfinite(a)) else a


def endplate(points, level: str, which: str = "sup"):
    """(anterior, posterior) corners of one endplate, or None."""
    a = _pt(points, f"{level}.{which}_ant")
    p = _pt(points, f"{level}.{which}_post")
    return None if (a is None or p is None) else (a, p)


def cobb(points, lv_a: str, which_a: str, lv_b: str, which_b: str) -> Optional[float]:
    """Angle between two endplate lines. Sign-free (Cobb convention)."""
    ea, eb = endplate(points, lv_a, which_a), endplate(points, lv_b, which_b)
    if ea is None or eb is None:
        return None
    return _acute(_ang(ea[0] - ea[1]) - _ang(eb[0] - eb[1]))


def spinopelvic(points: Dict[str, Optional[Sequence[float]]], *,
                anchor: str = "corner", upper: str = "L1") -> Dict[str, Optional[float]]:
    """PI / SS / PT / LL from landmarks alone.

    `anchor`: 'corner' (default) bisects the S1 superior endplate; 'overmask' uses the
    emitted `S1.sup_overmask` channel if present, falling back to 'corner'.
    """
    out: Dict[str, Optional[float]] = {"PI": None, "SS": None, "PT": None, "LL": None}
    s1 = endplate(points, "S1", "sup")
    if s1 is None:
        return out
    S1a, S1p = s1
    ep_dir = S1a - S1p
    out["SS"] = round(_acute(_ang(ep_dir)), 3)

    up = endplate(points, upper, "sup")
    if up is not None:
        out["LL"] = round(_acute(_ang(up[0] - up[1]) - _ang(ep_dir)), 3)

    mid = None
    if anchor == "overmask":
        mid = _pt(points, "S1.sup_overmask")
    if mid is None:
        mid = 0.5 * (S1a + S1p)                     # radiographic convention
    fem = _pt(points, "bicoxofemoral")
    if fem is None:
        return out                                   # PI/PT need the femoral head axis
    hip = mid - fem
    pt = _acute(_ang(hip) + 90.0)
    if hip[0] > 0:
        pt = -pt                                     # S1 behind the hip axis -> negative
    nrm = np.array([-ep_dir[1], ep_dir[0]], float)
    out["PT"] = round(pt, 3)
    out["PI"] = round(_acute(_ang(hip) - _ang(nrm)), 3)
    return out


def segmental(points, levels: Sequence[str]) -> Dict[str, Optional[float]]:
    """Disc angles: level i inferior endplate vs level i+1 superior endplate."""
    out = {}
    for a, b in zip(levels, levels[1:]):
        out[f"{a}/{b}"] = cobb(points, a, "inf", b, "sup")
    return out


def pi_identity_error(res: Dict[str, Optional[float]]) -> Optional[float]:
    """|SS + PT - PI|. A geometric necessity, so a non-zero value means the landmarks
    are internally inconsistent -- the cheapest possible QC on a prediction, and the
    check that localised a bad femoral-head point during development."""
    if any(res.get(k) is None for k in ("PI", "SS", "PT")):
        return None
    return round(abs(res["SS"] + res["PT"] - res["PI"]), 4)


def points_from_prediction(pts_xy, names, *, scale_xy=(1.0, 1.0)) -> Dict:
    """Decoded soft-argmax output -> the dict the functions above take.

    `scale_xy` maps network-resolution pixels back to the ORIGINAL detector grid, so
    errors can be reported in mm via the view's pixel spacing.
    """
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    out = {}
    arr = np.asarray(pts_xy, float)
    for i, nm in enumerate(names):
        p = arr[i]
        out[nm] = None if not np.all(np.isfinite(p)) else [p[0] / sx, p[1] / sy]
    return out
