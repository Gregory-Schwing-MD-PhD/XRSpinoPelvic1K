"""Tests for the parts where a silent error would invalidate results.

Deliberately not a coverage exercise. Each test guards a specific failure that would
be invisible in the metrics:

  * view leakage across folds  -> inflates every number, undetectable after the fact
  * absent landmarks supervised as zero -> teaches the model a level does not exist
  * channel-order drift -> silently permutes predictions vs checkpoints
  * the PI identity -> geometric necessity; a violation means bad landmarks
"""
from __future__ import annotations

import numpy as np
import pytest

from xrsp import measure as M
from xrsp import splits as S
from xrsp.heatmaps import (CORNER_KEYS, channel_names, gaussian_heatmaps,
                           points_from_json)


# ── splits ───────────────────────────────────────────────────────────────────

def _cohort(n_patients=60, seed=3):
    rng = np.random.default_rng(seed)
    recs = []
    for p in range(n_patients):
        for k in range(1 + int(rng.random() < 0.2)):
            lstv = ["normal", "SACRALIZATION", "LUMBARIZATION"][
                int(rng.choice(3, p=[0.9, 0.05, 0.05]))]
            recs.append(dict(case_id=f"c{p:03d}_{k}", patient_id=f"p{p:03d}",
                             lstv_label=lstv, has_l6=(lstv == "LUMBARIZATION"),
                             femoral_heads_visible=bool(rng.random() < 0.6)))
    return recs


def test_no_patient_leaks_across_folds():
    recs = _cohort()
    folds = S.build_folds(recs, seed=0)          # build_folds self-asserts too
    S.assert_no_leakage(folds, recs)
    seen = {}
    for f, cases in folds.items():
        for c in cases:
            assert c not in seen, f"{c} in {seen.get(c)} and {f}"
            seen[c] = f


def test_leakage_guard_actually_catches_leakage():
    recs = _cohort()
    folds = {k: list(v) for k, v in S.build_folds(recs, seed=0).items()}
    a, b = sorted(folds)[:2]
    folds[b].append(folds[a][0])                 # inject the failure
    with pytest.raises(AssertionError):
        S.assert_no_leakage(folds, recs)


def test_views_never_straddle_a_fold():
    recs = _cohort()
    folds = S.build_folds(recs, seed=0)
    views = [dict(case_id=r["case_id"], view=i) for r in recs for i in range(8)]
    tr = S.view_rows_for_fold(views, folds, 0, split="train")
    va = S.view_rows_for_fold(views, folds, 0, split="val")
    assert len(tr) + len(va) == len(views)
    assert not ({r["case_id"] for r in tr} & {r["case_id"] for r in va})


def test_splits_are_deterministic():
    recs = _cohort()
    assert S.build_folds(recs, seed=0) == S.build_folds(recs, seed=0)
    assert S.build_folds(recs, seed=0) != S.build_folds(recs, seed=1)


def test_patient_id_is_required():
    with pytest.raises(ValueError):
        S.build_folds([{"case_id": "c1"}])


# ── heatmaps ─────────────────────────────────────────────────────────────────

def test_absent_landmarks_are_masked_not_zero_supervised():
    names = channel_names(["L5", "S1"])
    pts = {n: None for n in names}
    pts["S1.sup_ant"] = [10.0, 20.0]
    hm, valid = gaussian_heatmaps(pts, (64, 64), sigma=2.0, names=names)
    assert valid.sum() == 1, "only the provided landmark may be supervised"
    assert valid[names.index("S1.sup_ant")]
    assert hm[names.index("L5.sup_ant")].max() == 0


def test_off_detector_landmark_is_not_supervised():
    names = channel_names(["S1"])
    hm, valid = gaussian_heatmaps({"S1.sup_ant": [999.0, 999.0]}, (32, 32),
                                  sigma=2.0, names=names)
    assert valid.sum() == 0
    assert hm.max() == 0


def test_channel_order_is_stable_and_complete():
    levels = ["T12", "L1", "S1"]
    names = channel_names(levels)
    assert names[-1] == "bicoxofemoral"
    assert len(names) == 4 * len(levels) + 1
    assert names[:4] == [f"T12.{k}" for k in CORNER_KEYS]
    assert names == channel_names(levels), "channel order must be deterministic"


def test_every_visible_level_gets_four_corners():
    """The level set follows the DATA -- a scan showing more vertebrae yields more."""
    small = points_from_json({"L5": {k: [1.0, 2.0] for k in CORNER_KEYS}}, None,
                             channel_names(["L5", "S1"]))
    assert sum(v is not None for v in small.values()) == 4
    big_levels = ["T10", "T11", "T12", "L1", "S1"]
    big = points_from_json({lv: {k: [1.0, 2.0] for k in CORNER_KEYS} for lv in big_levels},
                           [5.0, 6.0], channel_names(big_levels))
    assert sum(v is not None for v in big.values()) == 4 * len(big_levels) + 1


# ── measurement ──────────────────────────────────────────────────────────────

def _synthetic_points(ss_deg=30.0):
    """S1 endplate inclined ss_deg, L1 horizontal, femoral head directly below."""
    t = np.deg2rad(ss_deg)
    s1a = np.array([100.0, 300.0])
    s1p = s1a + 40.0 * np.array([np.cos(t), np.sin(t)])
    pts = {"S1.sup_ant": s1a.tolist(), "S1.sup_post": s1p.tolist(),
           "L1.sup_ant": [100.0, 100.0], "L1.sup_post": [140.0, 100.0],
           "bicoxofemoral": [120.0, 420.0]}
    return pts


def test_ss_matches_the_constructed_angle():
    for ss in (10.0, 25.0, 40.0):
        res = M.spinopelvic(_synthetic_points(ss))
        assert abs(res["SS"] - ss) < 1e-6


def test_pi_identity_holds():
    res = M.spinopelvic(_synthetic_points(30.0))
    assert M.pi_identity_error(res) < 1e-2, "PI = SS + PT is a geometric necessity"


def test_missing_femoral_head_yields_none_not_a_guess():
    pts = _synthetic_points()
    pts["bicoxofemoral"] = None
    res = M.spinopelvic(pts)
    assert res["SS"] is not None and res["LL"] is not None
    assert res["PI"] is None and res["PT"] is None


def test_anchor_choice_moves_pi_but_not_ss_or_ll():
    pts = _synthetic_points(30.0)
    a = M.spinopelvic(pts, anchor="corner")
    pts["S1.sup_overmask"] = [pts["S1.sup_ant"][0] + 8.0, pts["S1.sup_ant"][1] + 4.0]
    b = M.spinopelvic(pts, anchor="overmask")
    assert abs(a["SS"] - b["SS"]) < 1e-9, "the anchor must not change orientation"
    assert abs(a["LL"] - b["LL"]) < 1e-9
    assert abs(a["PI"] - b["PI"]) > 1e-6, "moving the anchor must move PI"


def test_segmental_angles_span_consecutive_levels():
    pts = {}
    for i, lv in enumerate(["L1", "L2", "L3"]):
        y = 100.0 + 50 * i
        pts[f"{lv}.sup_ant"] = [100.0, y]
        pts[f"{lv}.sup_post"] = [140.0, y]
        pts[f"{lv}.inf_ant"] = [100.0, y + 30]
        pts[f"{lv}.inf_post"] = [140.0, y + 30]
    seg = M.segmental(pts, ["L1", "L2", "L3"])
    assert set(seg) == {"L1/L2", "L2/L3"}
    assert all(v is not None and v < 1e-6 for v in seg.values())


# ---------------------------------------------------------------------------
# BUU union loader
# ---------------------------------------------------------------------------

def _buu_root():
    import os
    r = r"C:/Users/grego/OneDrive/Desktop/CTSpinoPelvic1K-1/BUU-LSPINE_400"
    return r if os.path.isdir(os.path.join(r, "LA")) else None


def test_buu_corner_order_and_mirror():
    """Anterior must survive the mirror, and the mirror must actually move x."""
    import numpy as np
    from xrsp.buu import load_corners, BUU_ROWS
    import tempfile, os
    rows = []
    for i in range(len(BUU_ROWS)):
        rows.append([100.0, 10.0 * i, 300.0, 10.0 * i + 5.0, 0])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.csv")
        np.savetxt(p, np.array(rows), delimiter=",")
        raw = load_corners(p)
        mir = load_corners(p, image_width=1000)
    # anterior is BUU's FIRST point, and stays the anterior corner after mirroring
    assert raw["L1.sup_ant"][0] == 100.0 and raw["L1.sup_post"][0] == 300.0
    assert mir["L1.sup_ant"][0] == 899.0 and mir["L1.sup_post"][0] == 699.0
    # mirroring must not move y
    assert mir["L1.sup_ant"][1] == raw["L1.sup_ant"][1]
    # after mirroring, anterior is at LARGER x -- the DRR convention
    assert mir["L1.sup_ant"][0] > mir["L1.sup_post"][0]


def test_buu_has_no_inferior_S1():
    """S1 is fused to S2: BUU annotates S1a with no S1b, and so do we."""
    from xrsp.buu import BUU_ROWS
    assert ("S1", "sup") in BUU_ROWS
    assert ("S1", "inf") not in BUU_ROWS


def test_buu_rejects_truncated_annotation():
    import numpy as np, tempfile, os
    from xrsp.buu import load_corners
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "short.csv")
        np.savetxt(p, np.zeros((4, 5)), delimiter=",")
        assert load_corners(p) is None      # partial file must not look like missing points


def test_buu_femoral_channel_is_masked_not_zeroed():
    """The whole point of the union: unlabelled != absent."""
    import numpy as np
    from xrsp.heatmaps import FEMORAL_KEY, channel_names, gaussian_heatmaps
    from xrsp.buu import BUU_LEVELS, load_corners
    import tempfile, os
    names = channel_names(BUU_LEVELS)
    # points must lie INSIDE the heatmap; an out-of-frame landmark is masked too
    rows = [[15.0, 5.0 + 10.0 * i, 45.0, 8.0 + 10.0 * i, 0] for i in range(11)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.csv")
        np.savetxt(p, np.array(rows), delimiter=",")
        pts = load_corners(p)
    hm, valid = gaussian_heatmaps(pts, (128, 64), names=names)
    fi = names.index(FEMORAL_KEY)
    assert not bool(valid[fi]), "bicoxofemoral must be MASKED for a BUU sample"
    assert bool(valid[names.index("L1.sup_ant")]), "annotated corners must be supervised"
    assert bool(valid[names.index("S1.sup_ant")]), "S1 superior IS annotated by BUU"
    assert not bool(valid[names.index("S1.inf_ant")]), "S1 has no inferior plate"


def test_landmark_outside_the_frame_is_masked():
    """A point off the image is unlabelled-here, not present-at-the-edge."""
    import numpy as np
    from xrsp.heatmaps import channel_names, gaussian_heatmaps
    names = channel_names(["L1"])
    hm, valid = gaussian_heatmaps({"L1.sup_ant": [500.0, 5.0]}, (64, 32), names=names)
    assert not bool(valid[names.index("L1.sup_ant")])
    assert float(np.abs(hm).max()) == 0.0


def test_buu_index_finds_the_real_dataset():
    root = _buu_root()
    if root is None:
        return
    from xrsp.buu import index_buu
    rows = index_buu(root)
    assert len(rows) == 400
    assert all(r["img"].endswith(".jpg") and r["csv"].endswith(".csv") for r in rows)


# ---------------------------------------------------------------------------
# Femoral head -> bicoxofemoral point
# ---------------------------------------------------------------------------

def _disc(shape, cx, cy, r):
    import numpy as np
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def test_union_centroid_is_the_midpoint_at_any_overlap():
    """The load-bearing claim: for EQUAL radii the union blob's centroid is exactly the
    midpoint of the two head centres, whatever the overlap -- so the two projected heads
    never have to be separated."""
    import numpy as np
    from xrsp.femhead import bicoxofemoral_from_mask
    H, W, r = 400, 400, 40
    for sep in (0, 5, 20, 40, 79, 120):          # total overlap -> fully separate
        cx1, cx2, cy = 200 - sep / 2, 200 + sep / 2, 200
        m = _disc((H, W), cx1, cy, r) | _disc((H, W), cx2, cy, r)
        p = bicoxofemoral_from_mask(m)
        assert abs(p[0] - 200.0) < 0.75, f"sep={sep}: x={p[0]}"
        assert abs(p[1] - 200.0) < 0.75, f"sep={sep}: y={p[1]}"


def test_unequal_radii_bias_is_small_and_toward_the_larger_head():
    """The known bias: near/far magnification makes one disc bigger. Direction and
    magnitude are both checked, so a regression that inflates it is caught."""
    import numpy as np
    from xrsp.femhead import bicoxofemoral_from_mask
    H, W = 400, 400
    m = _disc((H, W), 180, 200, 40) | _disc((H, W), 220, 200, 42)   # 5% larger, right
    p = bicoxofemoral_from_mask(m)
    assert p[0] > 200.0                      # biased toward the LARGER disc
    assert p[0] - 200.0 < 3.0                # but only slightly


def test_qc_flags_reject_the_failures_that_matter():
    import numpy as np
    from xrsp.femhead import qc_flags
    H, W = 300, 300
    good = _disc((H, W), 150, 150, 40)
    assert qc_flags(good)["ok"]
    assert not qc_flags(np.zeros((H, W), bool))["ok"]                    # empty
    assert not qc_flags(_disc((H, W), 2, 150, 40))["ok"]                 # touches border
    two = _disc((H, W), 80, 80, 25) | _disc((H, W), 230, 230, 25)
    assert qc_flags(two)["fragmented"] and not qc_flags(two)["ok"]       # two blobs
    assert not qc_flags(np.ones((H, W), bool))["ok"]                     # absurd area


def test_tta_spread_measures_scatter():
    from xrsp.femhead import tta_spread
    assert tta_spread([[10, 10], [10, 10], [10, 10]]) == 0.0
    assert tta_spread([[0, 0], [6, 8]]) > 4.0
    assert tta_spread([[1, 1]]) is None


def test_xray_appearance_preserves_shape_and_range():
    import numpy as np
    from xrsp.femhead import xray_appearance
    rng = np.random.default_rng(0)
    x = np.clip(rng.random((64, 48)).astype(np.float32), 0, 1)
    for _ in range(25):
        y = xray_appearance(x, rng)
        assert y.shape == x.shape and y.dtype == np.float32
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
