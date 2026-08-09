"""Does a detector trained on L1-S1 extrapolate to the vertebrae ABOVE L1?

The training labels cover exactly six objects per film: L1-L5 and S1. Most BUU laterals
include T11/T12 and often more, entirely unannotated. So every film is already a held-out
test of the question -- if the model learned the generic concept "vertebral body" it
should fire on the thoracic ones for free, and if it learned "the six things I am scored
on" it will not.

The distinction matters because it decides whether whole-spine landmarks can be had
without paying for whole-spine annotation.

Note what a NEGATIVE result here would mean: not that the model is broken, but that
absent-annotation is trained as background. Every thoracic vertebra in every training
film was an object the loss actively pushed the model to ignore.

    python scripts/extrapolate_above_l1.py --model <onnx> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from standing_scale_sweep import (infer, tiled_infer, load_gt, gt_in_px,  # noqa: F401
                                  _iou, PAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="/data/buu_yolo/images/test")
    ap.add_argument("--labels", default="/data/buu_yolo/labels/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--render", type=int, default=6)
    ap.add_argument("--out", default="/data/results_extrapolate")
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(a.images).glob("*"))[:a.n]

    rows = []
    for n, ip in enumerate(imgs):
        lp = Path(a.labels) / (ip.stem + ".txt")
        if not lp.exists():
            continue
        im = Image.open(ip).convert("L")
        W, H = im.size
        gt = gt_in_px(load_gt(lp), W, H)
        if "L1" not in gt:
            continue
        # top of the annotated block: the highest GT corner on L1
        l1_top = float(gt["L1"]["pts"][:, 1].min())
        gt_boxes = [[g["pts"][:, 0].min(), g["pts"][:, 1].min(),
                     g["pts"][:, 0].max(), g["pts"][:, 1].max()] for g in gt.values()]
        l1_cy = float(gt["L1"]["pts"][:, 1].mean())

        for mode in ("single", "tiled"):
            dets = (tiled_infer(sess, im, a.imgsz, a.conf) if mode == "tiled"
                    else infer(sess, im, a.imgsz, a.conf))
            # "Above L1" = does not correspond to any annotated vertebra AND sits
            # cranial to L1. Requiring the whole box to clear L1's topmost corner was
            # too strict by a hair: a T12 sitting directly on L1 has its lower edge at
            # exactly that line, so a genuine extrapolation scored as zero.
            def extra(d):
                if max((_iou(d["box"], g) for g in gt_boxes), default=0) >= 0.2:
                    return False
                return (d["box"][1] + d["box"][3]) / 2 < l1_cy
            above = [d for d in dets if extra(d)]
            rows.append({"image": ip.name, "mode": mode, "n_det": len(dets),
                         "n_above_L1": len(above),
                         "conf_above": [round(d["conf"], 3) for d in above]})

            if n < a.render and mode == "tiled":
                rgb = im.convert("RGB")
                dr = ImageDraw.Draw(rgb)
                dr.line([(0, l1_top), (W, l1_top)], fill=(80, 170, 255), width=4)
                for lv, g in gt.items():
                    for p in g["pts"]:
                        dr.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7],
                                   outline=(60, 255, 120), width=4)
                for d in dets:
                    col = (255, 60, 220) if extra(d) else (255, 190, 40)
                    dr.rectangle(d["box"], outline=col, width=4)
                rgb.save(out / f"extrap_{ip.stem}.png")

    (out / "raw.json").write_text(json.dumps(rows, indent=1))

    print(f"\n{'mode':>8} {'films':>6} {'dets/film':>10} {'above L1/film':>14} "
          f"{'films with any':>15}")
    print("-" * 60)
    summary = []
    for mode in ("single", "tiled"):
        g = [r for r in rows if r["mode"] == mode]
        if not g:
            continue
        nd = np.mean([r["n_det"] for r in g])
        na = np.mean([r["n_above_L1"] for r in g])
        any_pct = np.mean([r["n_above_L1"] > 0 for r in g]) * 100
        summary.append(dict(mode=mode, dets=float(nd), above=float(na),
                            films_with_any_pct=float(any_pct)))
        print(f"{mode:>8} {len(g):6d} {nd:10.2f} {na:14.2f} {any_pct:14.0f}%")

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nGround truth is 6 objects per film (L1-S1). Anything above the blue line "
          f"in the renders is a vertebra the model was never asked to find.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
