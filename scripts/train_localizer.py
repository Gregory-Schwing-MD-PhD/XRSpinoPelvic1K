"""Train the vertebral-level heatmap localizer on a built XRSpinoPelvic1K dataset.

SCAFFOLD — fills in the loop around the fixed data contract (see docs/ROADMAP.md). Requires
torch: `pip install xrsp[train]`. Kept as a script (not in the package) so the core DRR
engine has no heavy deps.

Usage:
  python scripts/train_localizer.py --data data/xrsp1k --view lateral --epochs 50
"""
import argparse
import glob
import json
import os

import numpy as np

from xrsp.labels import SPINE_LEVELS
from xrsp.localize import build_model, gaussian_heatmaps, points_from_heatmaps


def _load_case(case_dir, view):
    drr = np.load(os.path.join(case_dir, f"{view}_drr.npy"))
    levels = json.load(open(os.path.join(case_dir, f"{view}_levels.json")))["levels"]
    return drr.astype(np.float32), levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--view", default="lateral")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--out", default="runs/localizer.pt")
    a = ap.parse_args()

    import torch
    from torch.utils.data import Dataset, DataLoader

    names = SPINE_LEVELS

    class DRRSet(Dataset):
        def __init__(self, root, view):
            self.items = sorted(glob.glob(os.path.join(root, "*", f"{view}_drr.npy")))
            self.view = view

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            case_dir = os.path.dirname(self.items[i])
            drr, levels = _load_case(case_dir, self.view)
            hm, _ = gaussian_heatmaps(levels, drr.shape, a.sigma, names=names)
            return (torch.from_numpy(drr)[None], torch.from_numpy(hm))

    ds = DRRSet(a.data, a.view)
    if not len(ds):
        raise SystemExit(f"no DRRs under {a.data} (run xrsp.build_dataset first)")
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(len(names)).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = torch.nn.MSELoss()

    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep + 1}/{a.epochs}  loss {tot / len(dl):.4f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "levels": names}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
