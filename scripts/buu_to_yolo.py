#!/usr/bin/env python3
"""Convert BUU to Ultralytics keypoint format, replicating Bansal et al. 2026.

    python scripts/buu_to_yolo.py --buu data/BUU-LSPINE \
        --splits data/buu_splits.json --out data/buu_yolo

REPLICATED EXACTLY: single class ("vertebra"), four anatomical corner keypoints per
instance, COCO-style visibility flags. Their dataset is 208 Honduran + 508 BUU sagittal
films annotated the same way, so the task the model is asked to solve is identical.

NOT REPLICATED, DELIBERATELY: their images are pre-resized to 640x640 NON-UNIFORMLY by
the dataset authors, which destroys aspect ratio. Here the ORIGINAL films are used and
Ultralytics letterboxes them -- uniform scale plus padding -- so shape is preserved.
That matters because the downstream quantity is an ANGLE. On this dataset the source
aspect ratio has a median of 0.799; forcing it to square distorts a true 45 degree
endplate to 58 degrees, and by a different amount on every film (measured anisotropy
0.499 to 1.219). A squashed dataset can still score well on pixel metrics, which is why
the distortion is invisible in their results and fatal in ours.

The same flaw exists in our own heatmap pipeline, which resizes to 512x256 with
independent x and y scales. Fixing that is separate work; this script simply does not
introduce it.

GEOMETRY
--------
Boxes are the tight hull of a vertebra's four corners, then dilated by `--pad_frac` of
its own diagonal. A hull-tight box has zero margin, and IoU>=0.5 matching against a
zero-margin box is needlessly brittle on a structure whose corners are the very things
being predicted.

S1 carries only a superior endplate in BUU (11 annotated rows: L1sup..L5inf, S1sup), so
its two inferior keypoints are emitted with visibility 0. That keeps every instance a
uniform 4-keypoint object -- Ultralytics requires a fixed keypoint count per class -- and
the loss ignores flagged-invisible points rather than regressing them to a fabricated
location.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (level, [(row_index, corner_role)]) -- BUU_ROWS order is L1sup, L1inf, ... L5inf, S1sup
VERTEBRAE = [("L1", 0, 1), ("L2", 2, 3), ("L3", 4, 5), ("L4", 6, 7), ("L5", 8, 9)]
S1_ROW = 10


def instances(ann):
    """[(4x2 corners, 4 visibility flags, level)] per vertebra, in ORIGINAL pixels.

    Corner order is fixed at sup_ant, sup_post, inf_ant, inf_post for every instance --
    Ultralytics keypoints are positional, so a permuted order on one level would train
    the model to swap those landmarks there and nowhere else.
    """
    out = []
    for lv, r_sup, r_inf in VERTEBRAE:
        sa, sp = ann[r_sup, 0:2], ann[r_sup, 2:4]
        ia, ip = ann[r_inf, 0:2], ann[r_inf, 2:4]
        out.append((np.array([sa, sp, ia, ip], float), [2, 2, 2, 2], lv))
    sa, sp = ann[S1_ROW, 0:2], ann[S1_ROW, 2:4]
    # S1 has no inferior endplate here. The invisible pair is parked on the superior
    # points rather than at (0,0): a coordinate outside the box would still be written
    # into the label file, and some tooling clamps or warns on that even at v=0.
    out.append((np.array([sa, sp, sa, sp], float), [2, 2, 0, 0], "S1"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--splits", required=True,
                    help="buu_splits.json -- the SAME patient-grouped, sex/age-stratified "
                         "assignment the heatmap model used, so the comparison is on one "
                         "test set and not two draws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad_frac", type=float, default=0.06)
    ap.add_argument("--symlink", action="store_true",
                    help="symlink images instead of copying (BUU is ~4 GB)")
    a = ap.parse_args(argv)

    from PIL import Image

    from xrsp.buu import BUU_ROWS, index_buu

    assign = json.load(open(a.splits))["assignments"]
    rows = index_buu(a.buu)
    print(f"{len(rows)} films indexed; split file covers {len(assign)}")

    counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0
    for split in counts:
        os.makedirs(os.path.join(a.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", split), exist_ok=True)

    for r in rows:
        split = assign.get(r["case"])
        if split not in counts:
            skipped += 1
            continue
        ann = np.loadtxt(r["csv"], delimiter=",")
        if ann.ndim != 2 or ann.shape[0] < len(BUU_ROWS):
            skipped += 1
            continue
        W, H = Image.open(r["img"]).size

        lines = []
        for corners, vis, _lv in instances(ann):
            x0, y0 = corners.min(0)
            x1, y1 = corners.max(0)
            diag = float(np.hypot(x1 - x0, y1 - y0))
            pad = a.pad_frac * diag
            x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
            x0, y0 = max(x0, 0.0), max(y0, 0.0)
            x1, y1 = min(x1, W - 1.0), min(y1, H - 1.0)
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            bw, bh = (x1 - x0) / W, (y1 - y0) / H
            if bw <= 0 or bh <= 0:
                continue
            vals = [f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
            for (kx, ky), v in zip(corners, vis):
                vals += [f"{kx / W:.6f}", f"{ky / H:.6f}", str(v)]
            lines.append("0 " + " ".join(vals))

        stem = r["case"]
        dst_img = os.path.join(a.out, "images", split, stem + ".jpg")
        if not os.path.exists(dst_img):
            if a.symlink:
                os.symlink(os.path.abspath(r["img"]), dst_img)
            else:
                shutil.copy2(r["img"], dst_img)
        with open(os.path.join(a.out, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        counts[split] += 1

    # flip_idx is REQUIRED by Ultralytics whenever fliplr is enabled, and it is the
    # anterior/posterior swap made explicit: index 0<->1 and 2<->3. It is written here so
    # that a future run that turns flipping on cannot silently train the model to put the
    # anterior corner posteriorly -- the failure that augmentation ablation measured as
    # the most damaging on this exact task. This run keeps fliplr at 0.
    yaml = os.path.join(a.out, "buu.yaml")
    with open(yaml, "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\n"
                "train: images/train\nval: images/val\ntest: images/test\n\n"
                "kpt_shape: [4, 3]\n"
                "flip_idx: [1, 0, 3, 2]\n"
                "names:\n  0: vertebra\n")
    print(f"wrote {counts}  (skipped {skipped})")
    print(f"dataset yaml -> {yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
