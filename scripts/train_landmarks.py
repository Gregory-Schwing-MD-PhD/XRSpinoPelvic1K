#!/usr/bin/env python3
"""Train the vertebral-corner + bicoxofemoral landmark model on generated DRRs.

One fold at a time; 5-fold CV is five jobs (see slurm/xrsp_train_array.sh).

    python scripts/train_landmarks.py --data data/xrsp1k --splits data/splits.json \
        --fold 0 --out runs/f0 --epochs 200

Idempotent: if `last.ckpt` exists in --out it resumes from it, so a job that hits the
SLURM wall clock can simply be resubmitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrsp import splits as S                                   # noqa: E402
from xrsp.dataset import (LandmarkDRRDataset, collate, index_views,   # noqa: E402
                          levels_present)
from xrsp.heatmaps import channel_names                         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="generated dataset root (xrsp1k)")
    ap.add_argument("--splits", required=True, help="splits.json from make_splits.py")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", required=True, help="run dir (checkpoints + logs)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=3.0, help="heatmap sigma, px")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--levels", nargs="+", default=None,
                   help="vertebral levels to predict. DEFAULT: every level annotated "
                        "anywhere in --data, derived from the data itself. Scans differ "
                        "in how many vertebrae they show, so this is not hardcoded.")
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader

    from xrsp.model import LandmarkNet

    pl.seed_everything(a.seed, workers=True)
    folds = S.load(a.splits)
    rows = index_views(a.data)
    if not rows:
        sys.exit(f"no generated views found under {a.data}")
    tr_rows = S.view_rows_for_fold(rows, folds, a.fold, split="train")
    va_rows = S.view_rows_for_fold(rows, folds, a.fold, split="val")
    if not tr_rows or not va_rows:
        sys.exit(f"fold {a.fold}: train={len(tr_rows)} val={len(va_rows)} — check --splits")

    if a.levels is None:
        a.levels = levels_present(rows)
        if not a.levels:
            sys.exit("no annotated levels found in --data")
        print(f"levels derived from the data ({len(a.levels)}): {' '.join(a.levels)}")
    names = channel_names(a.levels)
    size = (a.height, a.width)
    tr = LandmarkDRRDataset(tr_rows, out_size=size, sigma=a.sigma,
                            augment=not a.no_augment, levels=a.levels, seed=a.seed)
    va = LandmarkDRRDataset(va_rows, out_size=size, sigma=a.sigma,
                            augment=False, levels=a.levels, seed=a.seed)
    print(f"fold {a.fold}: {len(tr)} train views / {len(va)} val views, "
          f"{len(names)} channels, image {size}")

    dl_kw = dict(batch_size=a.batch_size, num_workers=a.workers, collate_fn=collate,
                 pin_memory=torch.cuda.is_available(),
                 persistent_workers=a.workers > 0)
    tl = DataLoader(tr, shuffle=True, drop_last=True, **dl_kw)
    vl = DataLoader(va, shuffle=False, **dl_kw)

    os.makedirs(a.out, exist_ok=True)
    json.dump({"fold": a.fold, "levels": list(a.levels), "channels": names,
               "image_size": list(size), "sigma": a.sigma, "epochs": a.epochs,
               "batch_size": a.batch_size, "lr": a.lr, "seed": a.seed,
               "n_train_views": len(tr), "n_val_views": len(va)},
              open(os.path.join(a.out, "run_config.json"), "w"), indent=2)

    model = LandmarkNet(len(names), lr=a.lr, names=names, max_epochs=a.epochs)
    ckpt = ModelCheckpoint(dirpath=a.out, filename="best",
                           monitor="val/landmark_px", mode="min",
                           save_last=True, save_top_k=1)
    trainer = pl.Trainer(
        default_root_dir=a.out, max_epochs=a.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        precision=a.precision if torch.cuda.is_available() else 32,
        logger=CSVLogger(a.out, name="", version=""),
        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=10, deterministic=False)
    resume = os.path.join(a.out, "last.ckpt")
    trainer.fit(model, tl, vl, ckpt_path=resume if os.path.exists(resume) else None)
    print(f"done. best checkpoint: {ckpt.best_model_path} "
          f"(val/landmark_px = {ckpt.best_model_score})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
