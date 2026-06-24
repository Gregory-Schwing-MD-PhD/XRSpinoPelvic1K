"""DRR engine + label projection + level localization — on a synthetic CT so it runs with
no data. Checks the DRR/label projections are aligned and the localizer reads the right
level back."""
import numpy as np

from xrsp import (drr_project, project_footprints, project_level_points,
                  projection_plan, gaussian_heatmaps, points_from_heatmaps,
                  level_at_point)
from xrsp.labels import LABELS


def _phantom(D=48):
    """A CT-like volume (HU) with three stacked 'vertebrae' along world-Z + femoral balls,
    on an identity affine (world == index, so +Z superior, +Y anterior, +X left–right)."""
    vol = np.full((D, D, D), -1000.0, np.float32)      # air
    lab = np.zeros((D, D, D), np.int16)
    zc = {"L3": 14, "L4": 22, "L5": 30}                # cranio-caudal: higher z = superior
    for name, z in zc.items():
        sl = (slice(18, 30), slice(18, 30), slice(z, z + 6))
        vol[sl] = 600.0                                # bone HU
        lab[sl][...] = LABELS[name]
    return vol, lab, np.eye(4), zc


def test_drr_shape_and_bone_bright():
    vol, _, aff, _ = _phantom()
    drr, plan = drr_project(vol, aff, view="lateral")
    assert drr.ndim == 2 and np.all(np.isfinite(drr))
    assert 0.0 <= drr.min() and drr.max() <= 1.0
    assert drr.max() > drr.mean()                       # bone is brighter than background


def test_drr_and_labels_share_geometry():
    vol, lab, aff, _ = _phantom()
    plan = projection_plan(aff, "lateral")
    drr, _ = drr_project(vol, aff, view="lateral", plan=plan)
    fps = project_footprints(lab, aff, view="lateral", plan=plan)
    # every footprint must lie inside the DRR canvas and overlap the bright (bone) pixels
    for i, m in fps.items():
        assert m.shape == drr.shape
        assert m.any()
        assert drr[m].mean() > drr.mean()               # footprint sits on attenuating bone


def test_level_points_craniocaudal_and_localize():
    vol, lab, aff, zc = _phantom()
    pts = project_level_points(lab, aff, view="lateral")
    names = [p["name"] for p in pts]
    assert names == ["L3", "L4", "L5"]                  # cranio-caudal order preserved
    # superior level (L3, higher z) must be HIGHER on the image (smaller row/y)
    ys = {p["name"]: p["point"][1] for p in pts}
    assert ys["L3"] < ys["L4"] < ys["L5"]
    # localize: a query at L4's point returns L4
    hit = level_at_point(pts, pts[1]["point"])
    assert hit["name"] == "L4" and hit["distance_px"] < 1e-6


def test_heatmap_roundtrip():
    vol, lab, aff, _ = _phantom()
    drr, plan = drr_project(vol, aff, view="lateral")
    pts = project_level_points(lab, aff, view="lateral", plan=plan)
    hm, names = gaussian_heatmaps(pts, drr.shape, sigma=2.0)
    assert hm.shape == (len(names), *drr.shape)
    decoded = points_from_heatmaps(hm, names)
    # argmax-decoded points match the source landmarks within a pixel
    src = {p["name"]: np.array(p["point"]) for p in pts}
    for d in decoded:
        assert np.allclose(d["point"], src[d["name"]], atol=1.0)
