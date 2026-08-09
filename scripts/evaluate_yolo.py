#!/usr/bin/env python3
"""Evaluate the YOLO-Pose baseline with the SAME metrics as the heatmap model.

    python scripts/evaluate_yolo.py --weights runs/yolo/yolo11n-pose/weights/best.pt \
        --buu data/BUU-LSPINE --splits data/buu_splits.json \
        --out results/yolo --render 6

Deliberately reuses xrsp.evalmetrics rather than Ultralytics' own reporting. Ultralytics
gives COCO mAP/OKS; the heatmap model was scored on radial pixel error, ED thresholds,
detection F1 and a corner-identity matrix. Two models scored by two different pieces of
code are not comparable, however similar the metric names look.

LEVEL ASSIGNMENT
----------------
YOLO is single-class here (as in Bansal et al.), so detections are anonymous vertebrae.
They are matched to ground-truth boxes greedily by IoU, which recovers level identity and
mirrors their "sequential evaluation procedure": match boxes at IoU>=0.5 first, score
keypoints only inside matched boxes.

BOTH protocols are reported, because the choice moves the numbers a lot:
  matched_only  their protocol -- a vertebra the detector missed never reaches the
                keypoint score. Flattering, and what their published figures use.
  all_gt        every annotated landmark counts; a missed vertebra is a miss, not an
                absence. This is what the heatmap model was held to, so it is the only
                one of the two that may be compared against it.

GEOMETRY
--------
Everything is computed in ORIGINAL FILM PIXELS. The heatmap pipeline resizes to 512x256
with independent x and y scales, which distorts angles (a true 45 deg endplate reads as
58 deg at this dataset's median anisotropy). YOLO letterboxes, so its coordinates come
back undistorted -- meaning the SS and LL reported here are the real ones, and are NOT
directly comparable to the heatmap model's angles until that pipeline is fixed. Pixel
errors are additionally reported normalised by image diagonal, which is scale-free and
therefore IS comparable across both.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVELS = ["L1", "L2", "L3", "L4", "L5", "S1"]
CORNERS = ("sup_ant", "sup_post", "inf_ant", "inf_post")


def gt_from_csv(csv_path):
    """[(level, 4x2 corners, 4 visibility)] in original pixels, cranio-caudal."""
    from scripts.buu_to_yolo import instances          # single source of truth
    ann = np.loadtxt(csv_path, delimiter=",")
    return [(lv, c, v) for c, v, lv in instances(ann)]


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="Bansal et al. used 0.5 for YOLO box confidence")
    ap.add_argument("--iou_match", type=float, default=0.5)
    ap.add_argument("--render", type=int, default=6)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO, settings

    settings.update({k: False for k in ("mlflow", "clearml", "comet", "dvc", "hub",
                                        "neptune", "raytune", "wandb", "tensorboard")})

    from xrsp import evalmetrics as EM
    from xrsp import evalplots as EP
    from xrsp.buu import index_buu

    os.makedirs(a.out, exist_ok=True)
    assign = json.load(open(a.splits))["assignments"]
    rows = [r for r in index_buu(a.buu) if assign.get(r["case"]) == "test"]
    print(f"test films: {len(rows)}")
    model = YOLO(a.weights)

    err_matched, err_all, by_level = [], [], {lv: [] for lv in LEVELS}
    conf4 = np.zeros((4, 4), int)
    det = {"tp": 0, "fp": 0, "fn": 0}
    items, lm_rows = [], []

    for r in rows:
        W, H = Image.open(r["img"]).size
        diag = float(np.hypot(W, H))
        gt = gt_from_csv(r["csv"])
        res = model.predict(r["img"], conf=a.conf, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        pk = (res.keypoints.xy.cpu().numpy() if res.keypoints is not None
              else np.zeros((0, 4, 2)))

        used, P, T = set(), {}, {}
        for lv, corners, vis in gt:
            gb = (corners[:, 0].min(), corners[:, 1].min(),
                  corners[:, 0].max(), corners[:, 1].max())
            best, bi = 0.0, -1
            for j in range(len(pb)):
                if j in used:
                    continue
                v = iou(gb, tuple(pb[j]))
                if v > best:
                    best, bi = v, j
            matched = bi >= 0 and best >= a.iou_match
            if matched:
                used.add(bi)
            for k, cn in enumerate(CORNERS):
                if vis[k] == 0:
                    continue
                key = f"{lv}.{cn}"
                T[key] = [float(corners[k, 0]), float(corners[k, 1])]
                if matched:
                    p = pk[bi][k]
                    P[key] = [float(p[0]), float(p[1])]
                    d = float(np.linalg.norm(np.asarray(P[key]) - np.asarray(T[key])))
                    err_matched.append(d)
                    err_all.append(d)
                    by_level[lv].append(d)
                    det["tp" if d <= 5.0 else "fp"] += 1
                    if d > 5.0:
                        det["fn"] += 1
                else:
                    # Unmatched vertebra: a MISS in all_gt, absent in matched_only.
                    err_all.append(float("nan"))
                    det["fn"] += 1
                lm_rows.append({"case": r["case"], "channel": key, "level": lv,
                                "err_px": (err_all[-1] if not matched
                                           else err_matched[-1]),
                                "err_norm_diag": (err_matched[-1] / diag if matched
                                                  else float("nan")),
                                "matched": int(matched)})
        det["fp"] += max(0, len(pb) - len(used))
        conf4 += EM.corner_identity_confusion(P, T, LEVELS)
        items.append((r["case"], P, T, W, H, res))

    def params(pts):
        from ostk.metrics2d import spinopelvic_summary_2d
        eps = {lv: (np.asarray(pts[f"{lv}.sup_ant"], float),
                    np.asarray(pts[f"{lv}.sup_post"], float))
               for lv in LEVELS
               if f"{lv}.sup_ant" in pts and f"{lv}.sup_post" in pts}
        if not eps:
            return {}
        try:
            return spinopelvic_summary_2d(eps, None)
        except Exception:                                    # noqa: BLE001
            return {}

    prows = []
    for case, P, T, W, H, _ in items:
        g, q = params(T), params(P)
        prows.append({"case": case,
                      "SS_true": g.get("SS"), "SS_pred": q.get("SS"),
                      "LL_true": g.get("LL"), "LL_pred": q.get("LL")})

    diag_all = [np.hypot(*Image.open(r["img"]).size) for r in rows]
    med_diag = float(np.median(diag_all))
    summary = {
        "model": os.path.abspath(a.weights), "n_films": len(rows),
        "protocol_matched_only": {
            "corner_error_px": EM.error_summary(err_matched),
            "corner_ed_accuracy": EM.ed_accuracy(err_matched),
            "note": "Bansal protocol: keypoints scored only inside IoU>=0.5 matched boxes",
        },
        "protocol_all_gt": {
            "corner_error_px": EM.error_summary(err_all),
            "corner_ed_accuracy": EM.ed_accuracy(err_all),
            "n_missed": int(np.isnan(err_all).sum()),
            "note": "every annotated landmark counts; comparable to the heatmap model",
        },
        "corner_identity": {"matrix": conf4.tolist(), "classes": list(CORNERS),
                            **EM.prf_from_confusion(conf4)},
        "median_image_diagonal_px": med_diag,
        "detection": det,
    }
    for p in ("SS", "LL"):
        t = [r[f"{p}_true"] for r in prows
             if r[f"{p}_true"] is not None and r[f"{p}_pred"] is not None]
        q = [r[f"{p}_pred"] for r in prows
             if r[f"{p}_true"] is not None and r[f"{p}_pred"] is not None]
        if len(t) >= 3:
            d = np.abs(np.asarray(q) - np.asarray(t))
            summary.setdefault("parameters", {})[p] = {
                "n": len(t), "icc": EM.icc21(t, q),
                "mae": float(np.nanmean(d)),
                "within_5deg": float(np.nanmean(d <= 5)),
                "within_10deg": float(np.nanmean(d <= 10)),
                **{k: v for k, v in EM.bland_altman(t, q).items()
                   if k not in ("mean", "diff")}}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"),
              indent=2, default=str)
    with open(os.path.join(a.out, "per_landmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lm_rows[0]))
        w.writeheader(); w.writerows(lm_rows)
    with open(os.path.join(a.out, "per_item.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0]))
        w.writeheader(); w.writerows(prows)

    EP.plot_ced({"corners (matched)": EM.ced_curve(err_matched)}, a.out, "fig_ced_yolo")
    EP.plot_error_by_level(by_level, a.out, "fig_error_by_level_yolo")
    EP.plot_confusion(conf4, list(CORNERS), a.out, "fig_corner_identity_yolo",
                      "corner identity - YOLO", EM.prf_from_confusion(conf4))

    # ── render best and worst, chosen by SS error, never by hand ────────────
    def sserr(r):
        try:
            return abs(float(r["SS_pred"]) - float(r["SS_true"]))
        except (TypeError, ValueError):
            return -1.0
    ranked = sorted([r for r in prows if sserr(r) >= 0], key=sserr)
    picks = [("best", ranked[: a.render]), ("worst", ranked[-a.render:][::-1])]
    rdir = os.path.join(a.out, "renders")
    os.makedirs(rdir, exist_ok=True)
    byc = {c: (P, T, W, H) for c, P, T, W, H, _ in items}
    for tag, sel in picks:
        for i, pr in enumerate(sel):
            case = pr["case"]
            P, T, W, H = byc[case]
            row = next(r for r in rows if r["case"] == case)
            fig, ax = plt.subplots(figsize=(4.6, 7.4), dpi=120)
            ax.imshow(np.asarray(Image.open(row["img"]).convert("L")), cmap="gray")
            for key, t in T.items():
                ax.plot([t[0]], [t[1]], "o", ms=4, mfc="none", mec="#00E5A0", mew=1.3)
                if key in P:
                    p = P[key]
                    ax.plot([p[0]], [p[1]], "x", ms=5, color="#FF3B30", mew=1.4)
                    ax.plot([t[0], p[0]], [t[1], p[1]], "-", lw=0.8, color="#FFD60A")
            for src, col, lab in ((T, "#00E5A0", "truth"), (P, "#FF3B30", "pred")):
                if "S1.sup_ant" in src and "S1.sup_post" in src:
                    A, B = src["S1.sup_ant"], src["S1.sup_post"]
                    ax.plot([A[0], B[0]], [A[1], B[1]], "-", lw=2.2, color=col,
                            label=f"S1 endplate ({lab})")
            ax.set_title(f"{tag.upper()}  {case}\nSS {pr['SS_true']:.1f} -> "
                         f"{pr['SS_pred']:.1f}   LL {pr['LL_true']:.1f} -> "
                         f"{pr['LL_pred']:.1f}", fontsize=8)
            ax.legend(fontsize=6, loc="lower right", framealpha=0.4)
            ax.set_axis_off(); fig.tight_layout()
            fig.savefig(os.path.join(rdir, f"{tag}_{i:02d}_{case}.png"),
                        facecolor="white")
            plt.close(fig)

    m = summary["protocol_all_gt"]
    mm = summary["protocol_matched_only"]
    print(f"\n=== YOLO-Pose on {len(rows)} held-out films (original pixels) ===")
    print(f"  matched-only (Bansal protocol): median "
          f"{mm['corner_error_px']['median']:.2f}px  "
          f"<=5px {100*mm['corner_ed_accuracy']['within_5px']:.1f}%  "
          f"<=10px {100*mm['corner_ed_accuracy']['within_10px']:.1f}%  "
          f"<=15px {100*mm['corner_ed_accuracy']['within_15px']:.1f}%")
    print(f"  all-GT (comparable to heatmap): median "
          f"{m['corner_error_px']['median']:.2f}px  "
          f"<=5px {100*m['corner_ed_accuracy']['within_5px']:.1f}%   "
          f"missed {m['n_missed']}")
    print(f"  corner identity macro-F1 {summary['corner_identity']['macro_f1']:.3f}")
    for p, v in (summary.get("parameters") or {}).items():
        print(f"  {p}  ICC {v['icc']:.3f}  MAE {v['mae']:.2f}deg  "
              f"<=5deg {100*v['within_5deg']:.0f}%  bias {v['bias']:+.2f} "
              f"LoA {v['loa_low']:+.1f},{v['loa_high']:+.1f}")
    print(f"  renders -> {rdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
