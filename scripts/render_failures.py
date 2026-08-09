#!/usr/bin/env python3
"""Overlay predicted vs ground-truth landmarks on the BUU films that failed.

    python scripts/render_failures.py --model runs/unified/best.pt \
        --buu data/BUU-LSPINE --per_item results/unified/per_item_buu.csv \
        --out results/unified/failures --n 8

Aggregate metrics say a run failed on 3% of films; they cannot say WHY. Reading the
actual images is the only way to separate causes that need completely different fixes:
a corner on the wrong vertebra, a corner on the right vertebra but the wrong endplate,
an S1 endplate placed on the ala rather than the promontory, or ground truth that is
itself wrong. Each looks identical in a Bland-Altman plot.

Cases are chosen by SS error from the per-item CSV, not hand-picked, so the panel cannot
be flattering by selection. The BEST cases are rendered alongside on request, because a
failure gallery with no reference makes ordinary behaviour impossible to judge.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORNERS = ("sup_ant", "sup_post", "inf_ant", "inf_post")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--per_item", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--best", action="store_true",
                    help="render the BEST cases instead of the worst, as a reference")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from xrsp.buu import BUULandmarkDataset, index_buu
    from xrsp.dataset import collate
    from xrsp.heatmaps import soft_argmax
    from xrsp.model import build_unet

    rows = list(csv.DictReader(open(a.per_item)))
    def err(r):
        try:
            return abs(float(r["SS_pred"]) - float(r["SS_true"]))
        except (TypeError, ValueError):
            return -1.0
    ranked = sorted([r for r in rows if err(r) >= 0], key=err, reverse=not a.best)
    want = {r["case"]: err(r) for r in ranked[: a.n]}
    print(f"selected {len(want)} cases "
          f"({'best' if a.best else 'worst'} by SS error): "
          + ", ".join(f"{c}({e:.1f})" for c, e in want.items()))

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    names, size = list(ck["names"]), tuple(ck["size"])
    levels = sorted({n.split(".")[0] for n in names if "." in n})
    net = build_unet(len(names), features=tuple(ck.get("features",
                                                       (16, 32, 64, 128, 256))))
    net.load_state_dict(ck["model"])
    net.eval()

    brows = [r for r in index_buu(a.buu) if r["case"] in want]
    ds = BUULandmarkDataset(brows, levels=levels, out_size=size, sigma=2.0,
                            augment=False, p_flip=0.0, max_rot_deg=0.0)
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    made = []
    with torch.no_grad():
        for i, (img, hm, valid, _) in enumerate(dl):
            case = brows[i]["case"]
            pp, conf = soft_argmax(net(img))
            tp, _ = soft_argmax(hm)
            im = img[0, 0].numpy()

            fig, ax = plt.subplots(figsize=(4.6, 8.2), dpi=120)
            ax.imshow(im, cmap="gray", origin="upper")
            for c, n in enumerate(names):
                if not bool(valid[0, c]):
                    continue
                t = tp[0, c].numpy()
                p = pp[0, c].numpy()
                ax.plot([t[0]], [t[1]], "o", ms=3.4, mfc="none", mec="#00E5A0", mew=1.2)
                ax.plot([p[0]], [p[1]], "x", ms=4.2, color="#FF3B30", mew=1.3)
                # A line between truth and prediction: the length IS the error, and it
                # shows the DIRECTION, which a scatter of two point clouds does not.
                ax.plot([t[0], p[0]], [t[1], p[1]], "-", lw=0.7, color="#FFD60A",
                        alpha=0.85)
            # S1 drawn heavy: it is the endplate SS/PI/PT are measured from, so a failure
            # there is the one that propagates into every pelvic parameter.
            for pts, col, lab in ((tp, "#00E5A0", "truth"), (pp, "#FF3B30", "pred")):
                try:
                    ia = names.index("S1.sup_ant"); ip = names.index("S1.sup_post")
                except ValueError:
                    continue
                if bool(valid[0, ia]) and bool(valid[0, ip]):
                    A, P = pts[0, ia].numpy(), pts[0, ip].numpy()
                    ax.plot([A[0], P[0]], [A[1], P[1]], "-", lw=2.0, color=col,
                            label=f"S1 endplate ({lab})")
            r = next(x for x in rows if x["case"] == case)
            ax.set_title(f"{case}\nSS {float(r['SS_true']):.1f} -> "
                         f"{float(r['SS_pred']):.1f}   "
                         f"LL {float(r['LL_true']):.1f} -> {float(r['LL_pred']):.1f}",
                         fontsize=8)
            ax.legend(fontsize=6, loc="lower right", framealpha=0.4)
            ax.set_axis_off()
            fig.tight_layout()
            tagd = "best" if a.best else "worst"
            path = os.path.join(a.out, f"{tagd}_{i:02d}_{case}.png")
            fig.savefig(path, facecolor="white")
            plt.close(fig)
            made.append(path)
            print(f"  wrote {os.path.basename(path)}  "
                  f"(min channel confidence {float(conf[0][valid[0]].min()):.3f})")
    print(f"{len(made)} panels -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
