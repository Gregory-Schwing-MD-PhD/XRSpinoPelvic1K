#!/usr/bin/env python3
"""Do the learned hip point and a classical circle fit agree on REAL radiographs?

    python scripts/hip_agreement.py --weights runs/yolo_exp/combined/weights/best.pt \
        --buu data/BUU-LSPINE --splits data/buu_splits.json --out results/hip_agreement

THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT
--------------------------------------------------
The hip point is supervised ONLY by synthetic DRRs, because BUU annotates L1-L5 and S1
and nothing pelvic. There is therefore no ground truth on a real film, and no way to ask
directly whether the synthetic supervision transferred. The reverse direction of the same
gap is already measured and it is not reassuring: corner channels trained on BUU produce
85 px median error on DRRs.

So this asks a weaker but answerable question. `xrsp.hipfit` finds the femoral head with
Canny edges and a Hough circle transform -- no training, no labels, and it has never seen
a DRR. Where the two agree, two methods sharing no machinery landed on the same anatomy,
which is real evidence. Where they disagree, at least one is wrong and the film is worth
a human.

It is NOT a substitute for annotation. Agreement between two automatic methods can still
be agreement on the same systematic error, and the Hough fit has its own failure modes
(the acetabular rim, bowel gas, a prosthesis). What it can do is TRIAGE: if agreement is
tight on most films, the annotation effort collapses to the disagreements plus a random
audit, instead of all 301.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="a model with a pelvis class")
    ap.add_argument("--buu", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--render", type=int, default=6)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO, settings

    settings.update({k: False for k in ("mlflow", "clearml", "comet", "dvc", "hub",
                                        "neptune", "raytune", "wandb", "tensorboard")})
    from xrsp.buu import BUU_ROWS, index_buu
    from xrsp.hipfit import fit_head_circles

    os.makedirs(a.out, exist_ok=True)
    assign = json.load(open(a.splits))["assignments"]
    rows = [r for r in index_buu(a.buu) if assign.get(r["case"]) == "test"]
    model = YOLO(a.weights)
    print(f"test films: {len(rows)}")

    recs = []
    for r in rows:
        im = Image.open(r["img"]).convert("L")
        W, H = im.size
        arr = np.asarray(im, dtype=np.float32) / 255.0
        ann = np.loadtxt(r["csv"], delimiter=",")
        if ann.ndim != 2 or ann.shape[0] < len(BUU_ROWS):
            continue
        s1_ant, s1_post = ann[10, 0:2], ann[10, 2:4]

        # 1) classical, independent of everything learned
        try:
            hf = fit_head_circles(arr, s1_ant, s1_post)
        except Exception as exc:                             # noqa: BLE001
            hf = {"ok": False, "reason": f"{type(exc).__name__}"}
        hough = np.asarray(hf["center"], float) if hf.get("ok") else None

        # 2) learned, supervised only by DRRs
        res = model.predict(r["img"], conf=a.conf, verbose=False)[0]
        pred = None
        if res.boxes is not None and len(res.boxes):
            cls = res.boxes.cls.cpu().numpy().astype(int)
            pel = np.where(cls == 1)[0]                       # class 1 = pelvis
            if len(pel):
                j = pel[int(np.argmax(res.boxes.conf.cpu().numpy()[pel]))]
                pred = res.keypoints.xy.cpu().numpy()[j][0]   # hip point is slot 0

        d = (float(np.linalg.norm(pred - hough))
             if (pred is not None and hough is not None) else float("nan"))
        L = float(np.linalg.norm(np.asarray(s1_post) - np.asarray(s1_ant)))
        recs.append(dict(case=r["case"], W=W, H=H, s1_len=L,
                         hough_ok=bool(hf.get("ok")),
                         hough_reason=str(hf.get("reason", "")),
                         hough_x=(hough[0] if hough is not None else None),
                         hough_y=(hough[1] if hough is not None else None),
                         pred_x=(float(pred[0]) if pred is not None else None),
                         pred_y=(float(pred[1]) if pred is not None else None),
                         dist_px=d,
                         dist_norm=d / float(np.hypot(W, H)),
                         # In S1-endplate lengths: scale-free AND anatomically meaningful,
                         # since PI error scales with hip offset over the S1-to-hip radius.
                         dist_s1L=d / L if L > 0 else float("nan")))

    with open(os.path.join(a.out, "hip_agreement.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader(); w.writerows(recs)

    dd = np.array([x["dist_px"] for x in recs], float)
    ok = np.isfinite(dd)
    n_h = sum(1 for x in recs if x["hough_ok"])
    n_p = sum(1 for x in recs if x["pred_x"] is not None)
    print(f"\nHough converged   : {n_h}/{len(recs)}")
    print(f"model found pelvis: {n_p}/{len(recs)}")
    print(f"comparable films  : {int(ok.sum())}")
    if ok.sum():
        s1L = np.array([x["dist_s1L"] for x in recs], float)[ok]
        print(f"\nagreement (px)        median {np.median(dd[ok]):7.1f}  "
              f"p25 {np.percentile(dd[ok],25):6.1f}  p75 {np.percentile(dd[ok],75):6.1f}  "
              f"p95 {np.percentile(dd[ok],95):7.1f}")
        print(f"agreement (S1 lengths) median {np.median(s1L):7.3f}  "
              f"p95 {np.percentile(s1L,95):7.3f}")
        # A hip offset d over an S1-to-hip radius R changes PI by atan(d/R); R is roughly
        # 3 S1-endplate lengths on a lateral film.
        deg = np.degrees(np.arctan(s1L / 3.0))
        print(f"implied PI disagreement  median {np.median(deg):5.2f} deg  "
              f"p95 {np.percentile(deg,95):5.2f} deg")
        for t in (0.1, 0.2, 0.4):
            print(f"  within {t:.1f} S1 lengths: {100*(s1L<=t).mean():5.1f}% of films")

    # render the extremes -- agreement is only believable if you can see it
    import matplotlib.patches as mp
    rdir = os.path.join(a.out, "renders")
    os.makedirs(rdir, exist_ok=True)
    order = [i for i in np.argsort(np.where(ok, dd, np.inf)) if ok[i]]
    picks = [("agree", order[: a.render]), ("disagree", order[-a.render:][::-1])]
    for tag, idxs in picks:
        for k, i in enumerate(idxs):
            rec = recs[i]
            row = next(r for r in rows if r["case"] == rec["case"])
            fig, ax = plt.subplots(figsize=(4.6, 7.2), dpi=115)
            ax.imshow(np.asarray(Image.open(row["img"]).convert("L")), cmap="gray")
            if rec["hough_x"] is not None:
                ax.plot([rec["hough_x"]], [rec["hough_y"]], "o", ms=9, mfc="none",
                        mec="#00E5A0", mew=2, label="Hough (no training)")
            if rec["pred_x"] is not None:
                ax.plot([rec["pred_x"]], [rec["pred_y"]], "x", ms=10, color="#FF3B30",
                        mew=2, label="model (DRR-supervised)")
            ax.set_title(f"{tag.upper()}  {rec['case']}\n"
                         f"{rec['dist_px']:.0f} px = {rec['dist_s1L']:.2f} S1 lengths",
                         fontsize=8)
            ax.legend(fontsize=6, loc="lower right", framealpha=0.4)
            ax.set_axis_off(); fig.tight_layout()
            fig.savefig(os.path.join(rdir, f"{tag}_{k:02d}_{rec['case']}.png"),
                        facecolor="white")
            plt.close(fig)
    print(f"\nrenders -> {rdir}")
    print(f"csv     -> {os.path.join(a.out, 'hip_agreement.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
