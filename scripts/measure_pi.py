#!/usr/bin/env python3
"""Run BOTH models on real laterals and report PI/SS/PT, with QC.

    python scripts/measure_pi.py --buu <BUU-LSPINE_400> \
        --femhead runs/femhead/best.pt --corners runs/corners/best.pt \
        --out results/buu_pi.csv

The two halves of PI are measured independently and combined here:

  * the HIP AXIS, from the femoral-head segmenter (DRR-trained), as the centroid of the
    predicted head blob;
  * the S1 ENDPLATE, from the corner regressor (BUU-trained, real domain).

Nothing in the geometry requires one network to produce both, and the two tasks share no
useful supervision -- "find a sphere centre" and "find an endplate" are unrelated -- so
they are separate models on separate datasets, combined only at inference in the same
pixel space. Which also means the S1 corners come from radiologist labels in the
deployment domain, and only the hip point carries a synthetic-to-real gap.

QC is not optional here. The hip model is trained entirely on DRRs and run on real films,
and its output cannot be validated against anything BUU ships, so every prediction carries
its flags (blob area, component count, border contact, TTA scatter) and a rejected case is
reported as REJECTED rather than measured badly. A silently wrong PI is worse than a
missing one.

Free sanity check, printed at the end: population PI in asymptomatic adults is about
50 deg with an SD near 10. A predicted distribution far off that centre, or with a spread
of ~25 deg, means the transfer failed -- and it costs nothing to notice before anyone
annotates anything.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--femhead", required=True)
    ap.add_argument("--corners", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from PIL import Image
    from scipy.ndimage import zoom

    from xrsp import measure as M
    from xrsp.buu import index_buu, load_corners
    from xrsp.femhead import (bicoxofemoral_from_mask, qc_flags, tta_spread,
                              xray_appearance)
    from xrsp.heatmaps import FEMORAL_KEY, soft_argmax
    from xrsp.model import build_unet

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fh = torch.load(a.femhead, map_location=dev, weights_only=False)
    net_f = build_unet(1, features=tuple(fh.get("features", (16, 32, 64, 128, 256))))
    net_f.load_state_dict(fh["model"])
    net_f.to(dev).eval()
    fh_size = tuple(fh.get("size", (256, 128)))

    ck = torch.load(a.corners, map_location=dev, weights_only=False)
    names = list(ck["names"])
    net_c = build_unet(len(names), features=tuple(ck.get("features", (16, 32, 64, 128, 256))))
    net_c.load_state_dict(ck["model"])
    net_c.to(dev).eval()
    c_size = tuple(ck.get("size", (512, 256)))

    rows = index_buu(a.buu)
    if a.limit:
        rows = rows[: a.limit]
    rng = np.random.default_rng(0)
    out_rows, pis = [], []
    for r in rows:
        im = Image.open(r["img"]).convert("L")
        W0, H0 = im.size
        img0 = (np.asarray(im, np.float32) / 255.0)[:, ::-1].copy()   # DRR convention

        # ---- hip axis, with TTA-based uncertainty --------------------------------
        x = zoom(img0, (fh_size[0] / H0, fh_size[1] / W0), order=1)
        pts, masks = [], None
        with torch.no_grad():
            for t in range(max(1, a.tta)):
                xt = x if t == 0 else xray_appearance(x, rng)
                p = torch.sigmoid(net_f(torch.from_numpy(xt[None, None]).to(dev)))
                m = (p[0, 0].cpu().numpy() > 0.5)
                if t == 0:
                    masks = m
                pts.append(bicoxofemoral_from_mask(m))
        flags = qc_flags(masks)
        spread = tta_spread(pts)
        fem = bicoxofemoral_from_mask(masks)
        # scale the hip point back to ORIGINAL pixels, then into corner-model pixels
        fem_full = None if fem is None else [fem[0] * W0 / fh_size[1],
                                             fem[1] * H0 / fh_size[0]]

        # ---- corners ------------------------------------------------------------
        xc = zoom(img0, (c_size[0] / H0, c_size[1] / W0), order=1)
        with torch.no_grad():
            hm = net_c(torch.from_numpy(xc[None, None]).to(dev))[0].cpu()
        pk, _ = soft_argmax(hm)
        P = M.points_from_prediction(pk.numpy(), names,
                                     scale_xy=(c_size[1] / W0, c_size[0] / H0))
        if fem_full is not None and flags["ok"] and (spread is None or spread < 8.0):
            P[FEMORAL_KEY] = fem_full
        res = M.spinopelvic(P)
        ident = M.pi_identity_error(res)
        status = "ok" if flags["ok"] else "REJECTED"
        if spread is not None and spread >= 8.0:
            status = "REJECTED"
        out_rows.append({
            "case": r["case"], "status": status,
            "PI": res.get("PI"), "SS": res.get("SS"), "PT": res.get("PT"),
            "LL": res.get("LL"), "identity": ident,
            "tta_spread_px": spread, "area_frac": flags["area_frac"],
            "n_components": flags["n_components"], "touches_border": flags["touches_border"],
        })
        if status == "ok" and res.get("PI") is not None:
            pis.append(res["PI"])
        pi_s = "  --  " if res.get("PI") is None else f"{res['PI']:6.1f}"
        sp_s = "  --  " if spread is None else f"{spread:5.1f}px"
        print(f"  {r['case']:22s} {status:9s} PI={pi_s}  spread={sp_s}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as fh_:
            w = csv.DictWriter(fh_, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {a.out}")

    n_ok = sum(1 for r in out_rows if r["status"] == "ok")
    print(f"\n{n_ok}/{len(out_rows)} measured, {len(out_rows)-n_ok} rejected by QC")
    if pis:
        p = np.array(pis)
        print(f"PI over the accepted set: mean {p.mean():.1f}  SD {p.std():.1f}  "
              f"median {np.median(p):.1f}  n={len(p)}")
        print("reference: asymptomatic adult PI is ~50 deg, SD ~10")
        if abs(p.mean() - 50) > 15 or p.std() > 20:
            print("  *** distribution is implausible -- transfer probably FAILED ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
