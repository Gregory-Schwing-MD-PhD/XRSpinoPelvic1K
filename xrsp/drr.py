"""Digitally Reconstructed Radiograph (DRR) generation from a CT volume.

A DRR is a synthetic X-ray: integrate attenuation along the projection (ray) axis of a
CT volume. Because we render a CT that comes WITH a 3-D segmentation, we can project the
LABELS along the IDENTICAL geometry (see project_labels.py) and get a 2-D radiograph that
is paired, for free, with a 2-D mask + per-level landmarks — i.e. labelled training data
for radiograph models, generated from 3-D ground truth.

Parallel-beam projection (dependency-light, deterministic). Perspective/cone-beam (a la
DeepDRR) is a drop-in upgrade behind the same `projection_plan` interface."""
from __future__ import annotations

from typing import Dict

import numpy as np


def _u(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n else v


def projection_plan(affine, view: str = "lateral",
                    sup_world=(0, 0, 1), ant_world=(0, 1, 0)) -> Dict:
    """Map a viewing direction to array axes + display orientation, from the NIfTI affine.

    `view`: 'lateral' projects along the patient L–R axis; 'ap' along anterior–posterior.
    Returns a plan dict used by BOTH the DRR and the label projection so they stay aligned.
    Display convention (matches the PACS demo): superior up, anterior to the LEFT."""
    A = np.asarray(affine, float)[:3, :3]
    cols = [A[:, i] for i in range(3)]

    def axis_for(d):
        d = _u(d)
        return int(np.argmax([abs(_u(c) @ d) for c in cols]))

    proj = axis_for((1, 0, 0)) if view == "lateral" else axis_for((0, 1, 0))
    rem = [i for i in range(3) if i != proj]
    v_axis = max(rem, key=lambda i: abs(_u(cols[i]) @ _u(sup_world)))
    h_axis = [i for i in rem if i != v_axis][0]
    h_world = ant_world if view == "lateral" else (1, 0, 0)
    v_sign = 1 if (cols[v_axis] @ _u(sup_world)) >= 0 else -1
    h_sign = 1 if (cols[h_axis] @ _u(h_world)) >= 0 else -1
    return {"view": view, "proj": proj, "v_axis": v_axis, "h_axis": h_axis,
            "v_sign": int(v_sign), "h_sign": int(h_sign)}


def _orient2d(a2d, plan: Dict):
    """Reorient a (sum-over-proj-axis) 2-D array to display rows=superior↓, cols=anterior←."""
    rem = sorted([plan["v_axis"], plan["h_axis"]])
    img = a2d if rem[0] == plan["v_axis"] else a2d.T          # rows -> v_axis
    if plan["v_sign"] < 0:
        img = img[::-1, :]                                    # superior at top
    if plan["h_sign"] > 0:
        img = img[:, ::-1]                                    # anterior at left
    return np.ascontiguousarray(img)


def drr_project(volume, affine, *, view: str = "lateral", emphasis: str = "bone",
                bone_hu: float = 150.0, hu_floor: float = -1000.0, hu_ceil: float = 2000.0,
                gamma: float = 1.0, plan: Dict = None):
    """Render a DRR. Attenuation μ is path-integrated along the projection axis; bone (high μ)
    comes out bright like a film positive. `emphasis`:
      'bone' (default) — μ = ReLU(HU − bone_hu): the skeleton (spine/ribs/pelvis) dominates,
             soft tissue & bowel gas drop out — a clean SPINE film, ideal for level labels.
      'full'           — μ ∝ (HU − air): a conventional radiograph look (all tissue).
    Returns (drr float32 in [0,1], plan)."""
    vol = np.asarray(volume, float)
    plan = plan or projection_plan(affine, view)
    if emphasis == "bone":
        mu = np.clip(vol - bone_hu, 0.0, hu_ceil - bone_hu)  # only bone+ contributes
    else:
        mu = np.clip(vol, hu_floor, hu_ceil) - hu_floor      # >= 0, air ~ 0
    path = mu.sum(axis=plan["proj"])
    img = _orient2d(path, plan).astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    if gamma != 1.0:
        img = img ** float(gamma)
    return img, plan


def to_uint8(drr) -> np.ndarray:
    a = np.asarray(drr, float)
    a = (a - a.min()) / (a.ptp() or 1.0)
    return (255.0 * a).clip(0, 255).astype(np.uint8)
