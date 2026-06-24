"""Project a 3-D segmentation through the SAME geometry as the DRR → 2-D footprints +
per-level landmarks. This is what makes the DRRs *labelled*: the 2-D ground truth comes
straight from the 3-D masks, so no manual radiograph annotation is needed."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .drr import _orient2d, projection_plan
from .labels import LABELS, NAMES, SPINE_LEVELS


def project_footprints(label, affine, *, view: str = "lateral", plan: Dict = None,
                       ids=None) -> Dict[int, np.ndarray]:
    """2-D silhouette (footprint) of each label id under the projection — the union along
    the ray, oriented like the DRR. Returns {id: bool mask (H,W)}."""
    lab = np.asarray(label)
    plan = plan or projection_plan(affine, view)
    ids = ids if ids is not None else [int(v) for v in np.unique(lab) if v != 0]
    out = {}
    for i in ids:
        out[int(i)] = _orient2d((lab == i).any(axis=plan["proj"]), plan)
    return out


def _project_point_vox(ijk, plan: Dict, shape) -> List[float]:
    """A voxel index (i,j,k) → image (x=col, y=row), matching the DRR's flips exactly."""
    r = ijk[plan["v_axis"]]
    c = ijk[plan["h_axis"]]
    if plan["v_sign"] > 0:                          # match _orient2d's vertical flip
        r = shape[plan["v_axis"]] - 1 - r
    if plan["h_sign"] > 0:
        c = shape[plan["h_axis"]] - 1 - c
    return [float(c), float(r)]


def project_level_points(label, affine, *, view: str = "lateral", plan: Dict = None,
                         levels=SPINE_LEVELS) -> List[Dict]:
    """Per-vertebra LANDMARK for level localization: the 2-D projection of each present
    level's centroid + its 2-D bounding box. Returns a list of
    {name, id, point:[x,y], bbox:[x0,y0,x1,y1]} cranio-caudal."""
    lab = np.asarray(label)
    plan = plan or projection_plan(affine, view)
    out = []
    for name in levels:
        lid = LABELS.get(name)
        if lid is None:
            continue
        vox = np.argwhere(lab == lid)
        if not len(vox):
            continue
        ctr = vox.mean(0)
        x, y = _project_point_vox(ctr, plan, lab.shape)
        corners = np.array([_project_point_vox(v, plan, lab.shape)
                            for v in (vox.min(0), vox.max(0))])
        x0, y0 = corners.min(0)
        x1, y1 = corners.max(0)
        out.append({"name": name, "id": int(lid), "point": [x, y],
                    "bbox": [float(x0), float(y0), float(x1), float(y1)]})
    return out


def footprints_to_mask(footprints: Dict[int, np.ndarray]) -> np.ndarray:
    """Collapse per-id footprints into a single 2-D label image (cranial wins on overlap)."""
    if not footprints:
        return None
    shape = next(iter(footprints.values())).shape
    out = np.zeros(shape, np.int16)
    for i in sorted(footprints, reverse=True):     # lower id (cranial) painted last = wins
        out[footprints[i]] = i
    return out
