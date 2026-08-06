#!/usr/bin/env python3
"""Validate the hip point on REAL films against a classical circle fit. No labels.

    python scripts/validate_hip_circlefit.py --buu <BUU-LSPINE_400> \
        --model runs/unified/best.pt --out results/hip_circlefit.csv

The hip channel is supervised only by DRRs, so on real radiographs it has no ground truth
and no real-domain gradient -- it is the one part of the model that could be confidently
wrong. This measures that, using a reference that owes NOTHING to the DRRs: Canny edges
plus a Hough circle transform, constrained to the anatomic window implied by BUU's own S1
annotation. A learned validator would carry the same synthetic-to-real bias and agree for
the wrong reason; a classical fit cannot.

Bias and spread are reported SEPARATELY. A systematic offset is a correctable calibration
error; scatter is noise. Conflating them hides the one that matters.

Images are downsampled before fitting -- the Hough accumulator over a wide radius range is
minutes per case at full resolution, and a femoral head is hundreds of pixels across, so
nothing is lost.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--model", default=None, help="omit to report the circle fit alone")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--downsample", type=int, default=4)
    a = ap.parse_args(argv)

    from PIL import Image
    from scipy.ndimage import zoom

    from xrsp.buu import index_buu
    from xrsp.hipfit import fit_head_circles

    net = names = size = None
    if a.model:
        import torch
        from xrsp.model import build_unet
        ck = torch.load(a.model, map_location="cpu", weights_only=False)
        names = list(ck["names"])
        net = build_unet(len(names), features=tuple(ck.get("features", (16, 32, 64, 128, 256))))
        net.load_state_dict(ck["model"])
        net.eval()
        size = tuple(ck.get("size", (512, 256)))

    rows = index_buu(a.buu)
    if a.limit:
        rows = rows[: a.limit]
    ds = max(1, int(a.downsample))
    out, dxs, dys = [], [], []
    for r in rows:
        im = Image.open(r["img"]).convert("L")
        W0, H0 = im.size
        full = np.asarray(im, np.float32) / 255.0
        small = zoom(full, (1.0 / ds, 1.0 / ds), order=1)
        c = np.loadtxt(r["csv"], delimiter=",", ndmin=2)
        if len(c) < 11:
            continue
        s1a, s1p = c[10, 0:2] / ds, c[10, 2:4] / ds
        fit = fit_head_circles(small, s1a, s1p)
        row = {"case": r["case"], "fit_ok": bool(fit.get("ok")),
               "fit_x": None, "fit_y": None, "fit_r": None,
               "pred_x": None, "pred_y": None, "dx": None, "dy": None, "dist": None}
        if fit.get("ok"):
            row["fit_x"] = fit["center"][0] * ds
            row["fit_y"] = fit["center"][1] * ds
            row["fit_r"] = fit["radius"] * ds
        if net is not None:
            import torch
            from xrsp.heatmaps import FEMORAL_KEY, soft_argmax
            mir = full[:, ::-1].copy()                       # DRR convention
            x = zoom(mir, (size[0] / H0, size[1] / W0), order=1)
            with torch.no_grad():
                hm = net(torch.from_numpy(x[None, None].astype(np.float32)))[0]
            pk, _ = soft_argmax(hm)
            j = names.index(FEMORAL_KEY)
            px = float(pk[j, 0]) * W0 / size[1]
            py = float(pk[j, 1]) * H0 / size[0]
            px = W0 - 1 - px                                  # back to BUU orientation
            row["pred_x"], row["pred_y"] = px, py
            if fit.get("ok"):
                dx, dy = px - row["fit_x"], py - row["fit_y"]
                row["dx"], row["dy"] = dx, dy
                row["dist"] = float((dx * dx + dy * dy) ** 0.5)
                dxs.append(dx)
                dys.append(dy)
        out.append(row)
        print(f"  {r['case']:22s} fit={'Y' if fit.get('ok') else 'n'}"
              + ("" if row["dist"] is None else f"  d={row['dist']:6.1f}px"), flush=True)

    n_fit = sum(1 for r in out if r["fit_ok"])
    print(f"\ncircle fit converged: {n_fit}/{len(out)} ({100*n_fit/max(1,len(out)):.0f}%)")
    if dxs:
        dx, dy = np.array(dxs), np.array(dys)
        d = np.hypot(dx, dy)
        print(f"model vs circle fit, n={len(d)}")
        print(f"  BIAS  (systematic): dx {dx.mean():+.1f}  dy {dy.mean():+.1f} px")
        print(f"  SPREAD (noise)    : dx {dx.std():.1f}   dy {dy.std():.1f} px")
        print(f"  radial distance   : median {np.median(d):.1f}  p95 {np.percentile(d,95):.1f} px")
        print("  a bias larger than the fit's own residual is the only thing that would")
        print("  justify hand-annotating real films.")
    if a.out and out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
