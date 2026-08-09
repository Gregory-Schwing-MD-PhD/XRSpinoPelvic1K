#!/usr/bin/env python3
"""Convert DRRs to YOLO keypoint format, INCLUDING the bicoxofemoral point.

    python scripts/drr_to_yolo.py --drr data/xrsp1k --out data/drr_yolo

WHY THIS EXISTS
---------------
PI needs the S1 endplate AND the hip axis on the same image. BUU annotates neither hip --
its 11 rows are L1sup..L5inf plus S1sup and nothing pelvic -- so no model trained only on
BUU can produce PI, which is why no pelvis point appears in any BUU render. The DRRs are
the only source that carries both, because the hip point there is a 3-D sphere fit to the
femoral head projected through the same geometry as the corners.

TWO CLASSES, because Ultralytics needs a fixed keypoint count per model and a hip point
is not four corners:

    class 0  vertebra   4 corners, all visible
    class 1  pelvis     the bicoxofemoral point in slot 0; slots 1-3 flagged invisible

The invisible slots are parked ON the hip point rather than at the origin: a keypoint
outside its own box is something several tools clamp or warn about, even at v=0.

The pelvis box is a square centred on the hip point, sized as a fraction of image height.
A landmark has no extent of its own, and YOLO cannot regress a keypoint without a box to
carry it. The size only has to be stable and large enough to survive the detector's
stride -- it is a carrier, not a measurement, and nothing downstream reads it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

CORNERS = ("sup_ant", "sup_post", "inf_ant", "inf_post")


def subject_of(case):
    m = re.match(r"^(\d+)", case)
    return m.group(1) if m else case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drr", required=True, help="data/xrsp1k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad_frac", type=float, default=0.06)
    ap.add_argument("--hip_box_frac", type=float, default=0.05,
                    help="pelvis box side as a fraction of image height")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symlink", action="store_true")
    a = ap.parse_args(argv)

    import shutil

    from PIL import Image

    views = sorted(glob.glob(os.path.join(a.drr, "*", "*_corners.json")))
    print(f"{len(views)} rendered views found")

    # Split by SUBJECT, as everywhere else in this project: a case contributes many views
    # of one anatomy, and splitting on views puts the same spine in train and test.
    subs = sorted({subject_of(os.path.basename(os.path.dirname(v))) for v in views})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(subs)
    n_te = max(1, int(round(a.test_frac * len(subs))))
    n_va = max(1, int(round(a.val_frac * len(subs))))
    te, va = set(subs[:n_te]), set(subs[n_te:n_te + n_va])
    which = lambda s: "test" if s in te else ("val" if s in va else "train")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    n_hip = n_vert = skipped = 0
    for vj in views:
        meta = json.load(open(vj))
        case = os.path.basename(os.path.dirname(vj))
        tag = meta.get("view") or os.path.basename(vj).replace("_corners.json", "")
        png = os.path.join(os.path.dirname(vj), f"{tag}_drr.png")
        if not os.path.exists(png):
            skipped += 1
            continue
        W, H = Image.open(png).size
        lines = []

        for lv, cs in (meta.get("endplate_corners") or {}).items():
            pts = [cs.get(k) for k in CORNERS]
            if any(p is None for p in pts):
                continue
            P = np.asarray(pts, float)
            if not np.all(np.isfinite(P)):
                continue
            x0, y0 = P.min(0)
            x1, y1 = P.max(0)
            pad = a.pad_frac * float(np.hypot(x1 - x0, y1 - y0))
            x0, y0 = max(x0 - pad, 0.0), max(y0 - pad, 0.0)
            x1, y1 = min(x1 + pad, W - 1.0), min(y1 + pad, H - 1.0)
            if x1 <= x0 or y1 <= y0:
                continue
            v = [f"{(x0+x1)/2/W:.6f}", f"{(y0+y1)/2/H:.6f}",
                 f"{(x1-x0)/W:.6f}", f"{(y1-y0)/H:.6f}"]
            for (kx, ky) in P:
                v += [f"{kx/W:.6f}", f"{ky/H:.6f}", "2"]
            lines.append("0 " + " ".join(v))
            n_vert += 1

        fem = meta.get("bicoxofemoral_px")
        if fem is not None and np.all(np.isfinite(np.asarray(fem, float))):
            fx, fy = float(fem[0]), float(fem[1])
            side = a.hip_box_frac * H
            v = [f"{fx/W:.6f}", f"{fy/H:.6f}", f"{side/W:.6f}", f"{side/H:.6f}"]
            v += [f"{fx/W:.6f}", f"{fy/H:.6f}", "2"]
            for _ in range(3):                      # parked on the point, flagged unseen
                v += [f"{fx/W:.6f}", f"{fy/H:.6f}", "0"]
            lines.append("1 " + " ".join(v))
            n_hip += 1

        if not lines:
            skipped += 1
            continue
        split = which(subject_of(case))
        stem = f"{case}__{tag}"
        dst = os.path.join(a.out, "images", split, stem + ".png")
        if not os.path.exists(dst):
            if a.symlink:
                os.symlink(os.path.abspath(png), dst)
            else:
                shutil.copy2(png, dst)
        with open(os.path.join(a.out, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        counts[split] += 1

    yaml = os.path.join(a.out, "drr.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\n"
                "train: images/train\nval: images/val\ntest: images/test\n\n"
                "kpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n"
                "names:\n  0: vertebra\n  1: pelvis\n")
    print(f"views written {counts}  (skipped {skipped})")
    print(f"instances: {n_vert} vertebrae, {n_hip} pelvis")
    print(f"subjects: {len(subs)}  train/val/test = "
          f"{len(subs)-len(te)-len(va)}/{len(va)}/{len(te)}")
    print(f"dataset yaml -> {yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
