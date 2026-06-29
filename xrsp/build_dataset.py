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


def build_case(ct_path, label_path, out_dir, views=("lateral", "ap"), gamma=0.5):
    import nibabel as nib
    ct = nib.load(ct_path)
    vol, aff = np.asanyarray(ct.dataobj).astype(np.float32), ct.affine
    lab = np.asanyarray(nib.load(label_path).dataobj).astype(np.int16)
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
    a = p.parse_args(argv)
    if not a.in_dir and not (a.ct_dir and a.label_dir):
        p.error("give --in OR (--ct_dir and --label_dir)")
    os.makedirs(a.out_dir, exist_ok=True)
    views = tuple(a.views)

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
            rows = build_case(ct, lab, os.path.join(a.out_dir, case), views, a.gamma)
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
            w = csv.DictWriter(f, fieldnames=["case", "view", "n_levels", "drr"])
            w.writeheader()
            w.writerows(man)
    print(f"shard {a.shard_id}/{a.n_shards}: wrote {n_done} case(s), skipped {n_skip} -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
