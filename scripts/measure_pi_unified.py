#!/usr/bin/env python3
"""Run the unified model on real laterals and report PI/SS/PT/LL, with QC.

    python scripts/measure_pi_unified.py --buu <BUU-LSPINE_400> \
        --model runs/unified/best.pt --out results/buu_pi.csv

One model, all channels. The hip channel is the only one trained purely on DRRs, so it is
the only one that can be confidently wrong on real film -- hence the outlier flags, which
are reported per case rather than aggregated away:

  * predicted point outside the image
  * ORDERING violations: a posterior corner anterior of its anterior corner, or an
    inferior endplate above its superior one. These need no ground truth -- they are
    self-inconsistencies, and they catch failures a confidence score would not.
  * TTA scatter over 8 augmented passes, as an uncertainty estimate

Free transfer check at the end: population PI in asymptomatic adults is ~50 deg, SD ~10.
A distribution far off that centre, or with a ~25 deg spread, means the transfer failed --
and noticing that costs nothing and needs no annotation.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ordering_violations(P, names):
    """Self-consistency checks on the predicted landmark set."""
    bad = []
    lv = sorted({n.split(".")[0] for n in names if "." in n})
    for l in lv:
        sa, sp = P.get(f"{l}.sup_ant"), P.get(f"{l}.sup_post")
        ia, ip = P.get(f"{l}.inf_ant"), P.get(f"{l}.inf_post")
        if sa and sp and sa[0] <= sp[0]:
            bad.append(f"{l}:sup_ant_not_anterior")
        if ia and ip and ia[0] <= ip[0]:
            bad.append(f"{l}:inf_ant_not_anterior")
        if sa and ia and sa[1] >= ia[1]:
            bad.append(f"{l}:sup_below_inf")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from PIL import Image
    from scipy.ndimage import zoom

    from xrsp import measure as M
    from xrsp.buu import index_buu
    from xrsp.femhead import tta_spread, xray_appearance
    from xrsp.heatmaps import FEMORAL_KEY, soft_argmax
    from xrsp.model import build_unet

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    names = list(ck["names"])
    net = build_unet(len(names), features=tuple(ck.get("features", (16, 32, 64, 128, 256))))
    net.load_state_dict(ck["model"])
    net.to(dev).eval()
    size = tuple(ck.get("size", (512, 256)))
    fem_i = names.index(FEMORAL_KEY)

    rows = index_buu(a.buu)
    if a.limit:
        rows = rows[: a.limit]
    rng = np.random.default_rng(0)
    out, pis = [], []
    for r in rows:
        im = Image.open(r["img"]).convert("L")
        W0, H0 = im.size
        img0 = (np.asarray(im, np.float32) / 255.0)[:, ::-1].copy()
        x = zoom(img0, (size[0] / H0, size[1] / W0), order=1)
        hips = []
        with torch.no_grad():
            for t in range(max(1, a.tta)):
                xt = x if t == 0 else xray_appearance(x, rng)
                hm = net(torch.from_numpy(xt[None, None].astype(np.float32)).to(dev))[0].cpu()
                pk, _ = soft_argmax(hm)
                if t == 0:
                    P = M.points_from_prediction(pk.numpy(), names,
                                                 scale_xy=(size[1] / W0, size[0] / H0))
                hips.append([float(pk[fem_i, 0]), float(pk[fem_i, 1])])
        spread = tta_spread(hips)
        viol = ordering_violations(P, names)
        oob = [k for k, v in P.items() if v is not None
               and (v[0] < 0 or v[1] < 0 or v[0] >= W0 or v[1] >= H0)]
        res = M.spinopelvic(P)
        ident = M.pi_identity_error(res)
        status = "ok"
        if oob:
            status = "REJECTED_oob"
        elif spread is not None and spread >= 8.0:
            status = "REJECTED_tta"
        elif len(viol) > 2:
            status = "REJECTED_order"
        out.append({"case": r["case"], "status": status,
                    "PI": res.get("PI"), "SS": res.get("SS"), "PT": res.get("PT"),
                    "LL": res.get("LL"), "identity": ident,
                    "tta_spread_px": spread, "n_violations": len(viol),
                    "violations": ";".join(viol[:4]), "n_oob": len(oob)})
        if status == "ok" and res.get("PI") is not None:
            pis.append(res["PI"])
        pi_s = "  --  " if res.get("PI") is None else f"{res['PI']:6.1f}"
        print(f"  {r['case']:22s} {status:14s} PI={pi_s}  viol={len(viol)}"
              f"  spread={'--' if spread is None else f'{spread:.1f}'}", flush=True)

    n_ok = sum(1 for r in out if r["status"] == "ok")
    print(f"\n{n_ok}/{len(out)} measured, {len(out)-n_ok} rejected by QC")
    if pis:
        p = np.array(pis)
        print(f"PI: mean {p.mean():.1f}  SD {p.std():.1f}  median {np.median(p):.1f}  n={len(p)}")
        print("reference: asymptomatic adult PI ~50 deg, SD ~10")
        if abs(p.mean() - 50) > 15 or p.std() > 20:
            print("  *** implausible -- transfer probably FAILED ***")
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
