"""Build the XRSpinoPelvic1K dataset: turn CT + 3-D segmentation pairs into paired
DRRs + 2-D label masks + per-level landmarks + a manifest.

Input  : a directory of NIfTI pairs  <case>_ct.nii.gz  +  <case>_label.nii.gz
Output : data/xrsp1k/
           <case>/<view>_drr.png          # the synthetic radiograph (uint8)
           <case>/<view>_drr.npy          # float DRR [0,1]
           <case>/<view>_mask.png         # projected 2-D label image
           <case>/<view>_levels.json      # per-level landmarks (point + bbox)
         manifest.csv                      # one row per (case, view)

Usage:
  python -m xrsp.build_dataset --in /path/to/ct_label_pairs --out data/xrsp1k \
      --views lateral ap
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from .drr import drr_project, projection_plan, to_uint8
from .project_labels import (footprints_to_mask, project_footprints,
                             project_level_points)


def _save_png(path, arr_uint8):
    from PIL import Image
    Image.fromarray(arr_uint8).save(path)


# EVERY vertebra ostk knows, cranial->caudal. Emitting corners for all present levels
# is nearly free at generation time (the 3-D fit runs per level anyway) and cheap at
# training time (a few more heatmap channels, masked where a level is absent). It buys
# segmental lordosis and Cobb angles, and the extra landmarks act as auxiliary
# supervision. Levels missing from a scan are simply omitted from that view's json.
CORNER_LEVELS = tuple([f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 14)]
                      + [f"L{i}" for i in range(1, 7)] + ["S1"])
# Femoral heads share ONE channel on purpose. On a lateral you cannot tell which head is
# which -- on a true lateral they superimpose entirely -- so a left/right split is
# ill-posed. One channel holding both, with the BICOXOFEMORAL point taken as the centroid
# of the detected blobs, handles superimposed (one blob) and oblique (two) identically,
# and the blob separation is a free obliquity estimate. PI/PT need exactly this point.
FEMORAL_LABELS = ("femur_left", "femur_right")

# A level must be substantially IN the field of view before its endplate corners mean
# anything. A vertebra clipped by the scan edge still yields four geometrically valid
# corners -- they sit on the fragment -- but the "endplate" of a sliver is not an
# endplate, and supervising on it teaches the model to put corners on partial bodies.
# Seen on case 0003: T12 survived as 447 voxels (3% of a real body, 120 px projected)
# against L1-S1 at 15k-24k, and its corners rendered as a floating pair above the spine.
#
# Both a floor and a relative test, because absolute voxel counts scale with resolution:
# a level is kept if it clears MIN_LEVEL_VOXELS *and* MIN_LEVEL_FRACTION of the median
# labelled vertebra in that same scan.
MIN_LEVEL_VOXELS = 3000
MIN_LEVEL_FRACTION = 0.35


def _overmask_anchor_px(lab, aff, plan, level_name):
    """ostk's over-mask endplate anchor, projected to detector pixels. Kept as an
    OPTION (see oblique.pi_anchor_2d) -- not the default, which follows the
    radiographic convention of bisecting the endplate between its corners."""
    from ostk.spine import endplate_overmask_midpoint_from_label
    p3 = endplate_overmask_midpoint_from_label(lab, aff, level_name, which="superior")
    if p3 is None:
        return None
    p3 = np.asarray(p3, float)
    right = np.asarray(plan["right"], float); up = np.asarray(plan["up"], float)
    sp = plan["pixel_spacing_mm"]; H = plan["shape"][0]
    return [float((p3 @ right - plan["u0"]) / sp - 0.5),
            float((H - 1) - ((p3 @ up - plan["v0"]) / sp - 0.5))]


def build_case_oblique(ct_path, label_path, out_dir, *, n_views=8, seed=0, gamma=0.55,
                       pixel_spacing_mm=1.0, yaw_deg=12.0, pitch_deg=8.0, roll_deg=6.0,
                       drop_ids=(), ostk_path=None, pi_anchor="corner",
                       also_overmask=True):
    """N randomly-oblique lateral views of one CT, each with labels + endplate corners
    projected through the SAME geometry.

    Perfect laterals do not occur clinically, and the obliquity is what changes how the
    sacral ala superimposes on the S1 body -- the structure the model has to see through.
    Measured on case 0003, the PROJECTED S1 endplate inclination moves 28.8 -> 33.2 deg
    across yaw -12..+12, so positioning alone is a ~4 deg SS error source; the model has
    to learn that, which means it has to be trained on it.

    Corners come from the 3-D fit, projected (see oblique.endplate_corners_2d) -- never
    from the 2-D silhouette, which is 26-32 deg wrong because the ala is superimposed.
    """
    import nibabel as nib
    from .oblique import CORNER_KEYS
    from .oblique import (endplate_corners_2d, oblique_plan, pi_anchor_2d,
                          project_footprints, render, sample_view,
                          vertebra_corners_2d)
    try:
        from ostk.labels import LABELS
    except Exception:                                     # noqa: BLE001
        LABELS = {}
    rng = np.random.default_rng(seed)
    ct = nib.load(ct_path)
    vol, aff = np.asanyarray(ct.dataobj).astype(np.float32), ct.affine
    lab = np.asanyarray(nib.load(label_path).dataobj).astype(np.int16)
    if drop_ids:
        lab[np.isin(lab, list(drop_ids))] = 0
    os.makedirs(out_dir, exist_ok=True)
    idx = np.array(np.nonzero(lab)).T
    if not len(idx):
        return []
    bounds = (np.c_[idx, np.ones(len(idx))] @ np.asarray(aff, float).T)[:, :3]
    rows = []
    for k in range(n_views):
        v = sample_view(rng, yaw_deg=(0.0 if k == 0 else yaw_deg),   # view 0 = true lateral
                        pitch_deg=(0.0 if k == 0 else pitch_deg),
                        roll_deg=(0.0 if k == 0 else roll_deg))
        plan = oblique_plan(aff, v["direction"], v["sup"], roll_deg=v["roll_deg"],
                            bounds_world=bounds, pixel_spacing_mm=pixel_spacing_mm)
        drr = render(vol, aff, plan, gamma=gamma)
        fps = project_footprints(lab, aff, plan)
        # Bicoxofemoral point: femoral-head CENTRES fitted in 3-D, then projected.
        # NOT the centroid of the femur footprint -- `femur_left/right` covers the whole
        # proximal femur including shaft and trochanter, so its centroid sits well below
        # and lateral to the head. Using it put PT 6.8 deg and PI 6.2 deg out while SS and
        # LL stayed at 0.6 deg, and SS+PT=PI still held exactly -- i.e. the geometry was
        # fine and only this point was wrong. ostk fits a sphere to the superior slab of
        # each femur; do that, then project, exactly as for the endplate corners.
        fem_px = None
        heads_world = []
        try:
            from ostk.metrics import femoral_head_center
            for fem, hip in (("femur_left", "left_hip"), ("femur_right", "right_hip")):
                res = femoral_head_center(lab, aff, fem, hip)
                # returns (centre_xyz, radius_mm, rms) -- take the centre only
                if res is not None and len(res):
                    heads_world.append(np.asarray(res[0], float))
        except Exception:                                     # noqa: BLE001
            heads_world = []
        if heads_world:
            right3 = np.asarray(plan["right"], float)
            up3 = np.asarray(plan["up"], float)
            sp3 = plan["pixel_spacing_mm"]
            Hh = plan["shape"][0]
            px = [[(c @ right3 - plan["u0"]) / sp3 - 0.5,
                   (Hh - 1) - ((c @ up3 - plan["v0"]) / sp3 - 0.5)] for c in heads_world]
            fem_px = [float(np.mean([q[0] for q in px])),
                      float(np.mean([q[1] for q in px]))]
        # FOUR corners per level (both endplates) -- the standard vertebral annotation
        # on spine radiographs, so this is directly comparable to corner-annotated real
        # datasets, and it gives every segmental disc angle for free.
        # median labelled-vertebra size in THIS scan, for the relative FOV test
        _sizes = [int((lab == LABELS[n]).sum()) for n in CORNER_LEVELS
                  if LABELS.get(n) is not None and (lab == LABELS[n]).any()]
        _med = float(np.median(_sizes)) if _sizes else 0.0
        corners = {}
        skipped = []
        for name in CORNER_LEVELS:
            lid = LABELS.get(name)
            if lid is None or not (lab == lid).any():
                continue
            n_vox = int((lab == lid).sum())
            if n_vox < MIN_LEVEL_VOXELS or (_med and n_vox < MIN_LEVEL_FRACTION * _med):
                skipped.append((name, n_vox))       # FOV-truncated: not an endplate
                continue
            try:
                c4 = vertebra_corners_2d(lab, aff, plan, lid, level_name=name,
                                         ostk_path=ostk_path)
            except Exception:                             # noqa: BLE001
                c4 = None
            if not c4:
                continue
            # the alternative PI anchor, kept alongside so the convention stays a
            # choice at analysis time rather than baked into the generated data
            if also_overmask and name == "S1":
                try:
                    om = _overmask_anchor_px(lab, aff, plan, name)
                    if om is not None:
                        c4["sup_overmask"] = om
                except Exception:                         # noqa: BLE001
                    pass
            corners[name] = c4
        tag = f"lat{k:02d}"
        _save_png(os.path.join(out_dir, f"{tag}_drr.png"), to_uint8(drr))
        np.save(os.path.join(out_dir, f"{tag}_drr.npy"), drr.astype(np.float32))
        mask = footprints_to_mask(fps)
        if mask is not None:
            _save_png(os.path.join(out_dir, f"{tag}_mask.png"), mask.astype(np.uint8))
        json.dump({"view": tag, "geometry": plan, "shape": list(drr.shape),
                   "yaw_deg": v["yaw_deg"], "pitch_deg": v["pitch_deg"],
                   "roll_deg": v["roll_deg"], "endplate_corners": corners,
                   "bicoxofemoral_px": fem_px,
                   "skipped_truncated": {n: v for n, v in skipped},
                   "pi_anchor_mode": pi_anchor,
                   "pi_anchor_px": (pi_anchor_2d(corners["S1"], mode=pi_anchor)
                                    if "S1" in corners else None),
                   "n_channels": sum(len([k for k in c if k in CORNER_KEYS])
                                     for c in corners.values()) + (1 if fem_px else 0)},
                  open(os.path.join(out_dir, f"{tag}_corners.json"), "w"), indent=2)
        rows.append({"view": tag, "n_corner_levels": len(corners),
                     "skipped_truncated": ";".join(f"{n}:{v}" for n, v in skipped) or "",
                     "has_bicox": int(fem_px is not None),
                     "yaw_deg": round(v["yaw_deg"], 2), "pitch_deg": round(v["pitch_deg"], 2),
                     "roll_deg": round(v["roll_deg"], 2),
                     "drr": os.path.join(out_dir, f"{tag}_drr.png")})
    return rows


def build_case(ct_path, label_path, out_dir, views=("lateral", "ap"), gamma=0.5,
               drop_ids=()):
    import nibabel as nib
    ct = nib.load(ct_path)
    vol, aff = np.asanyarray(ct.dataobj).astype(np.float32), ct.affine
    lab = np.asanyarray(nib.load(label_path).dataobj).astype(np.int16)
    if drop_ids:                                          # e.g. exclude ribs (34-57) for now
        lab[np.isin(lab, list(drop_ids))] = 0
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for view in views:
        plan = projection_plan(aff, view)
        drr, _ = drr_project(vol, aff, view=view, gamma=gamma, plan=plan)
        fps = project_footprints(lab, aff, view=view, plan=plan)
        mask = footprints_to_mask(fps)
        levels = project_level_points(lab, aff, view=view, plan=plan)
        np.save(os.path.join(out_dir, f"{view}_drr.npy"), drr)
        _save_png(os.path.join(out_dir, f"{view}_drr.png"), to_uint8(drr))
        if mask is not None:
            _save_png(os.path.join(out_dir, f"{view}_mask.png"), mask.astype(np.uint8))
        json.dump({"view": view, "plan": plan, "shape": list(drr.shape), "levels": levels},
                  open(os.path.join(out_dir, f"{view}_levels.json"), "w"), indent=2)
        rows.append({"view": view, "n_levels": len(levels),
                     "drr": os.path.join(out_dir, f"{view}_drr.png")})
    return rows


def _base(name):
    """'<base>_ct.nii.gz' or '<base>.nii.gz' -> '<base>'."""
    for suf in ("_ct.nii.gz", ".nii.gz"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _pairs(in_dir):
    """Legacy: <case>_ct.nii.gz + <case>_label.nii.gz side-by-side in one dir."""
    for ct in sorted(glob.glob(os.path.join(in_dir, "*_ct.nii*"))):
        case = os.path.basename(ct).split("_ct")[0]
        lab = ct.replace("_ct", "_label")
        if os.path.exists(lab):
            yield case, ct, lab


def _pairs_split(ct_dir, label_dir):
    """CTSpinoPelvic1K layout: ct/<base>[_ct].nii.gz + labels/<base>_label.nii.gz."""
    for ct in sorted(glob.glob(os.path.join(ct_dir, "*.nii.gz"))):
        case = _base(os.path.basename(ct))
        lab = os.path.join(label_dir, f"{case}_label.nii.gz")
        if os.path.exists(lab):
            yield case, ct, lab


def _case_done(out_dir, case, views):
    """Resume check: all views' landmark json already written for this case."""
    return all(os.path.exists(os.path.join(out_dir, case, f"{v}_levels.json")) for v in views)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build XRSpinoPelvic1K DRR dataset from CT+seg")
    p.add_argument("--in", dest="in_dir", help="dir of side-by-side <case>_ct/_label.nii.gz")
    p.add_argument("--ct_dir", help="CTSpinoPelvic1K ct/ dir (use with --label_dir)")
    p.add_argument("--label_dir", help="CTSpinoPelvic1K labels/ dir (use with --ct_dir)")
    p.add_argument("--out", dest="out_dir", default="data/xrsp1k")
    p.add_argument("--views", nargs="+", default=["lateral", "ap"])
    p.add_argument("--gamma", type=float, default=0.5, help="DRR display gamma")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--oblique", type=int, default=0, metavar="N",
                   help="generate N randomly-oblique lateral views per CT (view 0 is a "
                        "true lateral). Replaces --views. Labels + endplate corners are "
                        "projected through the same geometry, so they stay exact.")
    p.add_argument("--yaw_deg", type=float, default=12.0)
    p.add_argument("--pitch_deg", type=float, default=8.0)
    p.add_argument("--roll_deg", type=float, default=6.0)
    p.add_argument("--pixel_spacing_mm", type=float, default=1.0)
    p.add_argument("--pi_anchor", choices=("corner", "overmask"), default="corner",
                   help="S1 point PI/PT are measured from. 'corner' (default) bisects "
                        "the superior endplate between its corners -- the radiographic "
                        "convention the published PI norms carry, and the only one a "
                        "model can derive from visible landmarks. 'overmask' is ostk's "
                        "bony-support centre (kept as an option).")
    p.add_argument("--ostk_path", default=None,
                   help="path to an OpenSpineToolkit checkout (needed for endplate corners)")
    p.add_argument("--no_ribs", action="store_true",
                   help="exclude ribs (ids 34-57) from the masks -- ship spine+sacrum+femurs now,"
                        " add ribs in a later version once rib numbering is finalised")
    a = p.parse_args(argv)
    if not a.in_dir and not (a.ct_dir and a.label_dir):
        p.error("give --in OR (--ct_dir and --label_dir)")
    os.makedirs(a.out_dir, exist_ok=True)
    views = tuple(a.views)
    drop_ids = tuple(range(34, 58)) if a.no_ribs else ()   # 34-57 = rib_left/right_1..12

    cases = list(_pairs_split(a.ct_dir, a.label_dir) if a.ct_dir else _pairs(a.in_dir))
    if a.n_shards > 1:
        cases = [c for i, c in enumerate(cases) if i % a.n_shards == a.shard_id]
    if a.limit:
        cases = cases[: a.limit]

    man, n_done, n_skip = [], 0, 0
    for case, ct, lab in cases:
        if not a.no_resume and _case_done(a.out_dir, case, views):
            n_skip += 1
            continue
        try:
            if a.oblique:
                rows = build_case_oblique(
                    ct, lab, os.path.join(a.out_dir, case), n_views=a.oblique,
                    seed=abs(hash(case)) % (2 ** 31),   # per-case seed: reproducible views
                    gamma=a.gamma, pixel_spacing_mm=a.pixel_spacing_mm,
                    yaw_deg=a.yaw_deg, pitch_deg=a.pitch_deg, roll_deg=a.roll_deg,
                    drop_ids=drop_ids, ostk_path=a.ostk_path,
                    pi_anchor=a.pi_anchor)
            else:
                rows = build_case(ct, lab, os.path.join(a.out_dir, case), views, a.gamma,
                                  drop_ids=drop_ids)
        except Exception as exc:                              # one bad case must not kill the shard
            print(f"[{case}] FAILED: {str(exc)[:160]}")
            continue
        for r in rows:
            r["case"] = case
            man.append(r)
        n_done += 1
        print(f"[{case}] {len(rows)} view(s)  ({n_done} done, {n_skip} skipped)")

    if man:
        # per-shard manifest (merge after the array finishes) to avoid clobbering
        suf = "" if a.n_shards == 1 else f"_shard{a.shard_id}"
        with open(os.path.join(a.out_dir, f"manifest{suf}.csv"), "w", newline="") as f:
            cols = ["case", "view", "n_levels", "n_corner_levels", "has_bicox",
                    "skipped_truncated",
                    "yaw_deg", "pitch_deg", "roll_deg", "drr"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(man)
    print(f"shard {a.shard_id}/{a.n_shards}: wrote {n_done} case(s), skipped {n_skip} -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
