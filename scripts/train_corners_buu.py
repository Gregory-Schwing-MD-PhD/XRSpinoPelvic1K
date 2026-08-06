#!/usr/bin/env python3
"""Train the endplate-corner regressor on BUU-LSpine. The spine half of PI.

    python scripts/train_corners_buu.py --buu <BUU-LSPINE_400> --out runs/corners \
        --epochs 120

Trained on BUU rather than on DRRs because BUU is radiologist ground truth in the
DEPLOYMENT domain -- real standing laterals -- which makes the whole 3-D corner-definition
question moot for L1..S1. The DRR corners remain useful for thoracic levels, which BUU
does not annotate.

Split is patient-grouped: BUU filenames carry a subject id (`0003-F-013Y1`), and the same
subject can appear more than once. Splitting on files would put one patient's spine in
both train and test and report a number that does not survive contact with new patients.

Held-out TEST is written out and never used for model selection -- selection uses VAL.
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
    """`0003-F-013Y1` -> `0003`. Falls back to the whole name if the pattern changes,
    which over-groups (safe) rather than under-groups (leaks)."""
    m = re.match(r"^(\d+)", case)
    return m.group(1) if m else case


def grouped_split(rows, *, val_frac=0.15, test_frac=0.15, seed=0):
    subs = sorted({subject_of(r["case"]) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_te = max(1, int(round(test_frac * len(subs))))
    n_va = max(1, int(round(val_frac * len(subs))))
    te, va = set(subs[:n_te]), set(subs[n_te:n_te + n_va])
    tr_r = [r for r in rows if subject_of(r["case"]) not in te | va]
    va_r = [r for r in rows if subject_of(r["case"]) in va]
    te_r = [r for r in rows if subject_of(r["case"]) in te]
    # the guard has to actually fire, not be assumed
    assert not ({subject_of(r["case"]) for r in tr_r}
                & {subject_of(r["case"]) for r in va_r + te_r}), "subject leak"
    return tr_r, va_r, te_r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from xrsp.buu import BUU_LEVELS, BUULandmarkDataset, index_buu
    from xrsp.dataset import collate
    from xrsp.heatmaps import channel_names, masked_mse, soft_argmax
    from xrsp.model import build_unet, landmark_error_px

    rows = index_buu(a.buu)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        sys.exit(f"no BUU laterals under {a.buu}")
    tr, va, te = grouped_split(rows, seed=a.seed)
    print(f"BUU {len(rows)} films -> train {len(tr)}  val {len(va)}  test {len(te)} "
          f"(patient-grouped)")

    levels = list(BUU_LEVELS)
    names = channel_names(levels)
    size = (a.height, a.width)
    mk = lambda rs, aug: BUULandmarkDataset(rs, levels=levels, out_size=size,
                                            sigma=a.sigma, augment=aug, seed=a.seed)
    dl_tr = DataLoader(mk(tr, True), batch_size=a.batch, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(mk(va, False), batch_size=a.batch, collate_fn=collate)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_unet(len(names), features=(16, 32, 64, 128, 256)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"levels": levels, "names": names, "size": list(size),
               "test_cases": [r["case"] for r in te],
               "val_cases": [r["case"] for r in va]},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=2)

    best = float("inf")
    for ep in range(a.epochs):
        net.train()
        tl = 0.0
        for img, hm, valid, _ in dl_tr:
            img, hm, valid = img.to(dev), hm.to(dev), valid.to(dev)
            opt.zero_grad()
            loss = masked_mse(net(img), hm, valid)
            loss.backward()
            opt.step()
            tl += float(loss)
        net.eval()
        vl, errs = 0.0, []
        with torch.no_grad():
            for img, hm, valid, _ in dl_va:
                img, hm, valid = img.to(dev), hm.to(dev), valid.to(dev)
                out = net(img)
                vl += float(masked_mse(out, hm, valid))
                e = landmark_error_px(out, hm, valid)
                if np.isfinite(e):
                    errs.append(e)
        vl /= max(1, len(dl_va))
        print(f"  ep {ep:3d}  train {tl/max(1,len(dl_tr)):.5f}  val {vl:.5f}  "
              f"landmark {np.mean(errs) if errs else float('nan'):.1f}px", flush=True)
        if vl < best:
            best = vl
            torch.save({"model": net.state_dict(), "names": names,
                        "size": list(size), "features": [16, 32, 64, 128, 256]},
                       os.path.join(a.out, "best.pt"))
    print(f"best val {best:.5f}  ->  {os.path.join(a.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
