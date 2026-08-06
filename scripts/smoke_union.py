#!/usr/bin/env python3
"""End-to-end check of the DRR + BUU union path. Exit 0 means it is wired correctly.

    python scripts/smoke_union.py --drr <generated_root> --buu <BUU-LSPINE_400>

Exercises the whole chain on a handful of samples from BOTH sources: index -> dataset ->
heatmap targets -> collate -> MONAI U-Net -> masked loss -> soft-argmax -> spinopelvic
angles. The point is to catch a shape or masking error here, in seconds, rather than
hours into a GPU job.

The masking assertions are the load-bearing ones. A BUU sample MUST supervise its
corners and MUST NOT supervise the bicoxofemoral point, because BUU does not annotate it;
if that mask ever silently flips to "supervised with a zero target", the model is being
taught that the femoral heads do not exist in real radiographs, and PI/PT quietly die.
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
    ap.add_argument("--drr", default=None, help="generated DRR root (optional)")
    ap.add_argument("--buu", default=None, help="BUU-LSPINE_400 root (optional)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    a = ap.parse_args(argv)
    if not a.drr and not a.buu:
        sys.exit("need --drr and/or --buu")

    import torch
    from torch.utils.data import DataLoader

    from xrsp import measure as M
    from xrsp.buu import BUULandmarkDataset, UnionDataset, index_buu
    from xrsp.dataset import LandmarkDRRDataset, collate, index_views, levels_present
    from xrsp.heatmaps import FEMORAL_KEY, channel_names, masked_mse, soft_argmax
    from xrsp.model import build_unet, landmark_error_px

    fail = []
    size = (a.height, a.width)

    drr_rows = index_views(a.drr)[: a.n] if a.drr else []
    buu_rows = index_buu(a.buu)[: a.n] if a.buu else []
    print(f"[1/7] indexed  DRR {len(drr_rows)}   BUU {len(buu_rows)}")
    if not drr_rows and not buu_rows:
        sys.exit("FAIL: nothing indexed")

    # channel set: the union of levels either source can carry
    levels = levels_present(drr_rows) if drr_rows else []
    for lv in [f"L{i}" for i in range(1, 6)] + ["S1"]:
        if lv not in levels:
            levels.append(lv)
    order = {lv: i for i, lv in enumerate(
        [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 14)]
        + [f"L{i}" for i in range(1, 7)] + ["S1"])}
    levels = sorted(levels, key=lambda lv: order.get(lv, 999))
    names = channel_names(levels)
    fem_i = names.index(FEMORAL_KEY)
    print(f"[2/7] channels {len(names)} over {len(levels)} levels: {' '.join(levels)}")

    drr_ds = LandmarkDRRDataset(drr_rows, out_size=size, levels=levels,
                                augment=True, seed=0) if drr_rows else None
    buu_ds = BUULandmarkDataset(buu_rows, levels=levels, out_size=size,
                                augment=True, seed=0) if buu_rows else None

    # --- the masking contract -------------------------------------------------------
    if buu_ds is not None:
        _, _, v, _ = buu_ds[0]
        n_sup = int(v.sum())
        if bool(v[fem_i]):
            fail.append("BUU sample supervises bicoxofemoral (must be MASKED)")
        if not bool(v[names.index("S1.sup_ant")]):
            fail.append("BUU sample does not supervise S1 superior")
        if bool(v[names.index("S1.inf_ant")]):
            fail.append("BUU sample supervises an S1 INFERIOR plate (S1 is fused to S2)")
        print(f"[3/7] BUU mask: {n_sup}/{len(names)} supervised, bicoxofemoral masked OK")
    if drr_ds is not None:
        _, _, v, _ = drr_ds[0]
        if not bool(v[fem_i]):
            fail.append("DRR sample does NOT supervise bicoxofemoral (it should)")
        print(f"[3/7] DRR mask: {int(v.sum())}/{len(names)} supervised, "
              f"bicoxofemoral supervised OK")

    union = UnionDataset(drr_ds, buu_ds, drr_weight=1, buu_weight=1)
    print(f"[4/7] union: {len(union)} samples")

    dl = DataLoader(union, batch_size=min(2, len(union)), collate_fn=collate)
    img, hm, valid, meta = next(iter(dl))
    print(f"[5/7] batch: img {tuple(img.shape)}  hm {tuple(hm.shape)}  "
          f"valid {tuple(valid.shape)}")
    if img.shape[-2:] != size or hm.shape[1] != len(names):
        fail.append(f"batch shapes wrong: {tuple(img.shape)} / {tuple(hm.shape)}")

    net = build_unet(len(names), features=(8, 16, 32, 64, 128))
    with torch.no_grad():
        out = net(img)
    if out.shape != hm.shape:
        fail.append(f"net output {tuple(out.shape)} != target {tuple(hm.shape)}")
    loss = masked_mse(out, hm, valid)
    err = landmark_error_px(out, hm, valid)
    print(f"[6/7] U-Net {tuple(out.shape)}  masked MSE {float(loss):.5f}  "
          f"(untrained) err {err:.1f}px")
    if not np.isfinite(float(loss)):
        fail.append("masked_mse not finite")

    # decode the TARGETS: heatmaps must round-trip to sane angles
    n_ok = 0
    for k in range(len(union)):
        _, h1, v1, m1 = union[k]
        pts, _ = soft_argmax(h1)
        sx, sy = m1["scale_xy"]
        res = M.spinopelvic(M.points_from_prediction(pts.numpy(), names, scale_xy=(sx, sy)))
        ident = M.pi_identity_error(res)
        src = m1.get("source", "drr")
        got = {kk: res[kk] for kk in ("PI", "SS", "PT", "LL") if res.get(kk) is not None}
        if res.get("LL") is not None:
            n_ok += 1
        if ident is not None and ident > 0.5:
            fail.append(f"{src} sample {k}: PI identity off by {ident}")
        print(f"      [{src}] " + "  ".join(f"{kk}={vv:.1f}" for kk, vv in got.items())
              + (f"   |SS+PT-PI|={ident}" if ident is not None else "   (no PI: no femoral pt)"))
    print(f"[7/7] {n_ok}/{len(union)} samples decoded an LL")
    if n_ok == 0:
        fail.append("no sample decoded a usable angle")

    if fail:
        print("\nSMOKE TEST FAILED:")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nSMOKE TEST PASSED — DRR + BUU union is wired correctly end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
