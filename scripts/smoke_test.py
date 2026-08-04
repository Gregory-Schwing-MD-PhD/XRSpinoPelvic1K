#!/usr/bin/env python3
"""Fast end-to-end check of the training path. Run this before a long GPU job.

    apptainer exec --nv containers/xrspinopelvic.sif python scripts/smoke_test.py \
        --data /data/xrsp1k

Exercises generation output -> dataset -> heatmaps -> MONAI U-Net -> soft-argmax ->
spinopelvic measurement, on a handful of views, in well under a minute.

Why it exists: the numpy half of this repo is covered by `make test`, but the torch
half (dataset collation, network shapes, decode) can only run where torch and MONAI
are installed -- i.e. inside the container. Discovering a shape error four hours into
a 24-hour job is an avoidable way to lose a day.

Exit code 0 means the pipeline is wired correctly end to end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="generated dataset root")
    ap.add_argument("--n", type=int, default=4, help="views to exercise")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from xrsp import measure as M
    from xrsp.dataset import (LandmarkDRRDataset, collate, index_views,
                              levels_present)
    from xrsp.heatmaps import channel_names, masked_mse, soft_argmax
    from xrsp.model import build_unet, landmark_error_px

    fail = []

    rows = index_views(a.data)[: a.n]
    if not rows:
        sys.exit(f"FAIL: no generated views under {a.data} — run generation first")
    print(f"[1/6] indexed {len(rows)} view(s) from {a.data}")

    levels = levels_present(rows)
    names = channel_names(levels)
    print(f"[2/6] levels derived from the data ({len(levels)}): {' '.join(levels)}")
    print(f"      -> {len(names)} channels (4 corners x {len(levels)} + bicoxofemoral)")
    if not levels:
        fail.append("no annotated levels found")

    ds = LandmarkDRRDataset(rows, out_size=(a.height, a.width), levels=levels,
                            augment=True, seed=0)
    img, hm, valid, meta = ds[0]
    print(f"[3/6] dataset item: img {tuple(img.shape)} hm {tuple(hm.shape)} "
          f"valid {int(valid.sum())}/{len(names)}")
    if img.shape != (1, a.height, a.width):
        fail.append(f"image shape {tuple(img.shape)}")
    if hm.shape != (len(names), a.height, a.width):
        fail.append(f"heatmap shape {tuple(hm.shape)}")
    if int(valid.sum()) == 0:
        fail.append("no valid channels in the first view")

    dl = DataLoader(ds, batch_size=min(2, len(ds)), collate_fn=collate)
    bimg, bhm, bvalid, bmeta = next(iter(dl))
    print(f"[4/6] collated batch: {tuple(bimg.shape)} / {tuple(bhm.shape)} / "
          f"{tuple(bvalid.shape)}")

    net = build_unet(len(names), features=(8, 16, 32, 64, 128))
    with torch.no_grad():
        out = net(bimg)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[5/6] MONAI BasicUNet -> {tuple(out.shape)}  ({n_par:.1f}M params)")
    if out.shape != bhm.shape:
        fail.append(f"net output {tuple(out.shape)} != target {tuple(bhm.shape)}")
    loss = masked_mse(out, bhm, bvalid)
    err = landmark_error_px(out, bhm, bvalid)
    print(f"      masked MSE {float(loss):.5f}   (untrained) landmark err {err:.1f} px")
    if not np.isfinite(float(loss)):
        fail.append("masked_mse is not finite")

    # decode the TARGETS: heatmap -> soft-argmax -> angles must round-trip
    pts, _ = soft_argmax(hm)
    sx, sy = meta["scale_xy"]
    res = M.spinopelvic(M.points_from_prediction(pts.numpy(), names, scale_xy=(sx, sy)))
    ident = M.pi_identity_error(res)
    print(f"[6/6] target round-trip: " +
          "  ".join(f"{k}={res[k]}" for k in ("PI", "SS", "PT", "LL")))
    print(f"      PI identity |SS+PT-PI| = {ident}")
    if res["SS"] is None:
        fail.append("SS could not be measured from the decoded targets")
    if ident is not None and ident > 0.5:
        fail.append(f"PI identity violated by {ident} deg — decode is inconsistent")

    if fail:
        print("\nSMOKE TEST FAILED:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nSMOKE TEST PASSED — the training path is wired correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
