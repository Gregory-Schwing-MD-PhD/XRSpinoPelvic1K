#!/usr/bin/env python3
"""Train the femoral-head SEGMENTER on DRRs. The hip half of PI.

    python scripts/train_femhead.py --data data/xrsp1k --out runs/femhead --epochs 60

Segmentation rather than a landmark heatmap because the bicoxofemoral point is the centre
of a sphere, and a sphere's centre has no local image evidence -- there is nothing at that
pixel to see. A network must infer it from the rim, which is what a segmenter does
explicitly, with a dense gradient over thousands of pixels instead of a near-delta target.
The point then falls out of the mask arithmetically (xrsp.femhead.bicoxofemoral_from_mask),
and a mask can be eyeballed for QC while a heatmap peak cannot.

Loss is Dice + BCE: Dice for the overlap the centroid actually depends on, BCE to keep
gradients alive when the prediction is empty early in training.

The model is validated on what it is FOR -- the distance between the centroid of the
predicted mask and of the true mask, in pixels -- not only on Dice. A mask can lose Dice
symmetrically and leave the centroid perfect, or hold Dice while the centroid shifts; the
second is the one that damages PI, at roughly 1.3 deg per 3 mm of hip-axis error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def index_head_views(root: str):
    """(DRR, head-mask) pairs. A view without a head mask is skipped: the femora were
    outside the CT FOV, which is missing data, not an empty-mask training target."""
    rows = []
    for npy in sorted(glob.glob(os.path.join(root, "*", "*_drr.npy"))):
        head = npy.replace("_drr.npy", "_head.npy")
        if os.path.exists(head):
            rows.append({"case": os.path.basename(os.path.dirname(npy)),
                         "view": os.path.basename(npy).replace("_drr.npy", ""),
                         "npy": npy, "head": head})
    return rows


def dice_bce(logits, target, eps: float = 1.0):
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1.0 - (num / den).mean())


def centroid_err_px(logits, target, thr: float = 0.5):
    """Mean |centroid(pred) - centroid(true)| in pixels -- the quantity PI depends on."""
    import torch
    p = (torch.sigmoid(logits) > thr).float()
    errs = []
    for i in range(p.shape[0]):
        a, b = p[i, 0], target[i, 0]
        if a.sum() < 1 or b.sum() < 1:
            continue
        ys, xs = torch.nonzero(a, as_tuple=True)
        yt, xt = torch.nonzero(b, as_tuple=True)
        errs.append(float(((xs.float().mean() - xt.float().mean()) ** 2
                           + (ys.float().mean() - yt.float().mean()) ** 2) ** 0.5))
    return float(np.mean(errs)) if errs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke runs)")
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from xrsp.femhead import FemHeadDataset
    from xrsp.model import build_unet

    rows = index_head_views(a.data)
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        sys.exit(f"no (DRR, head-mask) pairs under {a.data} — run generation first")

    # split by CASE, never by view: several views share one patient's anatomy, so a
    # view-level split leaks the same hips into train and val and reports a fantasy.
    cases = sorted({r["case"] for r in rows})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(a.val_frac * len(cases))))
    val_cases = set(cases[:n_val])
    tr = [r for r in rows if r["case"] not in val_cases]
    va = [r for r in rows if r["case"] in val_cases]
    if not tr or not va:
        tr, va = rows, rows                      # degenerate tiny set (smoke runs)
    print(f"cases {len(cases)}  train {len(tr)} views / val {len(va)} views")

    size = (a.height, a.width)
    dl_tr = DataLoader(FemHeadDataset(tr, out_size=size, augment=True, seed=a.seed),
                       batch_size=a.batch, shuffle=True,
                       collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                             torch.stack([x[1] for x in b]),
                                             [x[2] for x in b]))
    dl_va = DataLoader(FemHeadDataset(va, out_size=size, augment=False, seed=a.seed),
                       batch_size=a.batch,
                       collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                             torch.stack([x[1] for x in b]),
                                             [x[2] for x in b]))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_unet(1, features=(16, 32, 64, 128, 256)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    best = float("inf")
    hist = []
    for ep in range(a.epochs):
        net.train()
        tl = 0.0
        for img, msk, _ in dl_tr:
            img, msk = img.to(dev), msk.to(dev)
            opt.zero_grad()
            loss = dice_bce(net(img), msk)
            loss.backward()
            opt.step()
            tl += float(loss)
        net.eval()
        vl, ce = 0.0, []
        with torch.no_grad():
            for img, msk, _ in dl_va:
                img, msk = img.to(dev), msk.to(dev)
                out = net(img)
                vl += float(dice_bce(out, msk))
                e = centroid_err_px(out, msk)
                if np.isfinite(e):
                    ce.append(e)
        vl /= max(1, len(dl_va))
        cerr = float(np.mean(ce)) if ce else float("nan")
        hist.append({"epoch": ep, "train": tl / max(1, len(dl_tr)),
                     "val": vl, "centroid_px": cerr})
        print(f"  ep {ep:3d}  train {tl/max(1,len(dl_tr)):.4f}  val {vl:.4f}  "
              f"centroid {cerr:.2f}px", flush=True)
        if vl < best:
            best = vl
            torch.save({"model": net.state_dict(),
                        "size": list(size), "features": [16, 32, 64, 128, 256]},
                       os.path.join(a.out, "best.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=2)
    print(f"best val {best:.4f}  ->  {os.path.join(a.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
