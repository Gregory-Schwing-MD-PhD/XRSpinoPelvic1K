#!/usr/bin/env python3
"""Train the unified spinopelvic landmark model. One model, two sources, disjoint channels.

    python scripts/train_unified.py --drr data/xrsp1k --buu /data/BUU-LSPINE_400 \
        --out runs/unified --epochs 150

  HIP POINT from DRRs   (3-D sphere fit to the femoral head, projected)
  CORNERS   from BUU    (radiologist annotations, deployment domain)

The two streams supervise disjoint channels, so the DRR corner convention never has to
agree with BUU's -- the DRR corners are simply not used. Sigma is annealed from coarse to
fine so early training gets a wide basin and late training gets precision.

Splits are by PATIENT on both sides: BUU filenames carry a subject id and a subject can
recur, and a DRR case contributes many views of one anatomy. Splitting on files would put
the same spine in train and test and report a number that does not survive new patients.
The leak guard is asserted, not assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def subject_of(case: str) -> str:
    m = re.match(r"^(\d+)", case)
    return m.group(1) if m else case


def split_by_subject(rows, val_frac, test_frac, seed):
    subs = sorted({subject_of(r["case"]) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_te = max(1, int(round(test_frac * len(subs)))) if subs else 0
    n_va = max(1, int(round(val_frac * len(subs)))) if subs else 0
    te, va = set(subs[:n_te]), set(subs[n_te:n_te + n_va])
    tr_r = [r for r in rows if subject_of(r["case"]) not in te | va]
    va_r = [r for r in rows if subject_of(r["case"]) in va]
    te_r = [r for r in rows if subject_of(r["case"]) in te]
    assert not ({subject_of(r["case"]) for r in tr_r}
                & {subject_of(r["case"]) for r in va_r + te_r}), "subject leak"
    return tr_r, va_r, te_r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drr", default=None)
    ap.add_argument("--buu", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--sigma_start", type=float, default=8.0)
    ap.add_argument("--sigma_end", type=float, default=2.0)
    ap.add_argument("--drr_weight", type=int, default=1)
    ap.add_argument("--buu_weight", type=int, default=1)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--p_flip", type=float, default=0.5,
                    help="P(left-right flip). Makes the model indifferent to which way "
                         "the patient faced; 0 disables.")
    ap.add_argument("--max_rot_deg", type=float, default=8.0,
                    help="random in-plane rotation. Makes the DETECTOR robust to a tilted "
                         "film -- it does NOT make SS/PT valid on one, since both are "
                         "measured against true vertical.")
    ap.add_argument("--resume", default=None, help="checkpoint to resume (preemption)")
    a = ap.parse_args(argv)
    if not a.drr and not a.buu:
        sys.exit("need --drr and/or --buu")

    import torch
    from torch.utils.data import DataLoader

    from xrsp.buu import BUU_LEVELS, index_buu
    from xrsp.dataset import collate, index_views, levels_present
    from xrsp.heatmaps import FEMORAL_KEY, masked_mse
    from xrsp.model import build_unet, landmark_error_px
    from xrsp.unified import build_union

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    drr_rows = index_views(a.drr) if a.drr else []
    buu_rows = index_buu(a.buu) if a.buu else []
    if a.limit:
        drr_rows, buu_rows = drr_rows[: a.limit], buu_rows[: a.limit]

    levels = levels_present(drr_rows) if drr_rows else []
    for lv in BUU_LEVELS:
        if lv not in levels:
            levels.append(lv)
    order = {lv: i for i, lv in enumerate(
        [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 14)]
        + [f"L{i}" for i in range(1, 7)] + ["S1"])}
    levels = sorted(levels, key=lambda lv: order.get(lv, 999))

    d_tr, d_va, d_te = split_by_subject(drr_rows, a.val_frac, a.test_frac, a.seed) \
        if drr_rows else ([], [], [])
    b_tr, b_va, b_te = split_by_subject(buu_rows, a.val_frac, a.test_frac, a.seed) \
        if buu_rows else ([], [], [])
    print(f"DRR {len(drr_rows)} views -> {len(d_tr)}/{len(d_va)}/{len(d_te)}")
    print(f"BUU {len(buu_rows)} films -> {len(b_tr)}/{len(b_va)}/{len(b_te)}")
    print(f"levels ({len(levels)}): {' '.join(levels)}")

    size = (a.height, a.width)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds_va, names = build_union(d_va, b_va, levels=levels, out_size=size,
                               sigma=a.sigma_end, augment=False, seed=a.seed,
                               drr_weight=1, buu_weight=1,
                               p_flip=0.0, max_rot_deg=0.0)   # val is never augmented
    net = build_unet(len(names), features=(16, 32, 64, 128, 256)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    start_ep, best = 0, float("inf")
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep, best = int(ck.get("epoch", 0)) + 1, float(ck.get("best", float("inf")))
        print(f"resumed from {a.resume} at epoch {start_ep}")

    json.dump({"levels": levels, "names": names, "size": list(size),
               "drr_test": [r["case"] for r in d_te],
               "buu_test": [r["case"] for r in b_te],
               "args": vars(a)},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=2)
    dl_va = DataLoader(ds_va, batch_size=a.batch, collate_fn=collate)
    fem_i = names.index(FEMORAL_KEY)

    for ep in range(start_ep, a.epochs):
        # sigma annealing: a wide basin early, precision late
        t = ep / max(1, a.epochs - 1)
        sig = a.sigma_start + t * (a.sigma_end - a.sigma_start)
        ds_tr, _ = build_union(d_tr, b_tr, levels=levels, out_size=size, sigma=sig,
                               augment=True, seed=a.seed + ep,
                               drr_weight=a.drr_weight, buu_weight=a.buu_weight,
                               p_flip=a.p_flip, max_rot_deg=a.max_rot_deg)
        dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, collate_fn=collate)
        net.train()
        tl, nb = 0.0, 0
        for img, hm, valid, _ in dl_tr:
            img, hm, valid = img.to(dev), hm.to(dev), valid.to(dev)
            opt.zero_grad()
            loss = masked_mse(net(img), hm, valid)
            loss.backward()
            opt.step()
            tl += float(loss)
            nb += 1
        net.eval()
        vl, errs, hip_errs = 0.0, [], []
        with torch.no_grad():
            for img, hm, valid, _ in dl_va:
                img, hm, valid = img.to(dev), hm.to(dev), valid.to(dev)
                out = net(img)
                vl += float(masked_mse(out, hm, valid))
                e = landmark_error_px(out, hm, valid)
                if np.isfinite(e):
                    errs.append(e)
                # the hip channel on its own -- it is the one with no real-domain labels
                hv = torch.zeros_like(valid)
                hv[:, fem_i] = valid[:, fem_i]
                if bool(hv.any()):
                    e2 = landmark_error_px(out, hm, hv)
                    if np.isfinite(e2):
                        hip_errs.append(e2)
        vl /= max(1, len(dl_va))
        print(f"  ep {ep:3d} sig {sig:4.1f}  train {tl/max(1,nb):.5f}  val {vl:.5f}  "
              f"all {np.mean(errs) if errs else float('nan'):5.1f}px  "
              f"hip {np.mean(hip_errs) if hip_errs else float('nan'):5.1f}px", flush=True)
        ck = {"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
              "best": best, "names": names, "size": list(size),
              "features": [16, 32, 64, 128, 256]}
        torch.save(ck, os.path.join(a.out, "last.pt"))      # resume point, every epoch
        if vl < best:
            best = vl
            ck["best"] = best
            torch.save(ck, os.path.join(a.out, "best.pt"))
    print(f"best val {best:.5f}  ->  {os.path.join(a.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
