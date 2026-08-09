#!/usr/bin/env python3
"""YOLO-Pose baseline, configured to match Bansal et al. 2026 (PLoS One e0347290).

    python scripts/train_yolo_pose.py --data data/buu_yolo/buu.yaml \
        --model yolo11n-pose.pt --out runs/yolo

EVERY SETTING BELOW IS FROM THEIR TABLE 4, so a difference in results is a difference in
data or architecture rather than in tuning:

    input 640          batch 8            SGD               lr0 0.01
    cosine LR          warmup 3 epochs    AMP on            epochs 100
    patience 20        workers 8          seed 42           COCO-pretrained

AUGMENTATION IS FULLY DISABLED, which is the setting that makes the comparison mean
anything. Their Experimental Standardisation section says "all default data augmentation
operations within the YOLO training pipeline were disabled" to match Detectron2 -- and
Ultralytics silently applies mosaic, mixup, HSV jitter, translation, scaling and
left-right flip unless every one is explicitly zeroed. Leaving any of them on would
compare their configuration against a different one and call the difference architecture.

fliplr is 0 for a second, independent reason: their own ablation (Table 11) measured
horizontal flip as the single most damaging augmentation on this task, and on a lateral
film a mirror image is an anatomy that does not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Bansal et al. Table 4. Anything not listed here is an Ultralytics default they kept.
PAPER = dict(
    imgsz=640, batch=8, optimizer="SGD", lr0=0.01, cos_lr=True,
    warmup_epochs=3, amp=True, epochs=100, patience=20, workers=8, seed=42,
    deterministic=True, pretrained=True, val=True, plots=True,
)
# Every augmentation Ultralytics would otherwise apply, set to its identity value.
NO_AUG = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, erasing=0.0, auto_augment=None,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="buu.yaml from buu_to_yolo.py")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="yolo11n-pose.pt (their best balance), yolov8n-pose.pt, "
                         "or yolov8l-pose.pt (their highest accuracy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override Table 4's 100")
    ap.add_argument("--augment", action="store_true",
                    help="re-enable Ultralytics defaults -- NOT the paper configuration")
    a = ap.parse_args(argv)

    from ultralytics import YOLO, settings

    # Ultralytics AUTO-DETECTS every experiment tracker importable in the environment and
    # enables it. monai[all] drags in mlflow, whose file-store backend now raises on
    # import-and-log, so a YOLO run dies inside a logger it was never asked to use. Every
    # integration is turned off explicitly: this is a controlled baseline and none of them
    # should be in the loop.
    settings.update({k: False for k in
                     ("mlflow", "clearml", "comet", "dvc", "hub", "neptune",
                      "raytune", "wandb", "tensorboard")})

    cfg = dict(PAPER)
    if not a.augment:
        cfg.update(NO_AUG)
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    cfg.update(data=os.path.abspath(a.data), project=os.path.abspath(a.out),
               name=os.path.splitext(os.path.basename(a.model))[0], exist_ok=True)

    os.makedirs(a.out, exist_ok=True)
    # Written BEFORE training: if the run dies, the configuration it died under is still
    # on disk. A config recoverable only from a completed run is no record at all.
    with open(os.path.join(a.out, "paper_config.json"), "w") as f:
        json.dump({"source": "Bansal et al. 2026 PLoS One e0347290, Table 4",
                   "model": a.model, "augmentation_disabled": not a.augment,
                   "config": {k: v for k, v in cfg.items()}}, f, indent=2, default=str)
    print("=== configuration (Bansal et al. Table 4) ===")
    for k in sorted(cfg):
        print(f"  {k:16s} {cfg[k]}")

    model = YOLO(a.model)
    res = model.train(**cfg)
    print(f"\ntraining done -> {res.save_dir}")

    # Ultralytics reports val metrics during training; the TEST split is only touched
    # here, once, at the end.
    m = model.val(data=cfg["data"], split="test", imgsz=cfg["imgsz"], plots=True)
    out = {
        "box_map50": float(m.box.map50), "box_map": float(m.box.map),
        "pose_map50": float(m.pose.map50), "pose_map": float(m.pose.map),
        "save_dir": str(res.save_dir),
    }
    with open(os.path.join(a.out, "test_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== TEST split (Ultralytics/COCO metrics) ===")
    print(f"  bbox  mAP@0.5 {out['box_map50']:.4f}   mAP@0.5:0.95 {out['box_map']:.4f}")
    print(f"  pose  mAP@0.5 {out['pose_map50']:.4f}   mAP@0.5:0.95 {out['pose_map']:.4f}")
    print("  [Bansal Table 6 test, v11n: bbox 91.90 / 60.80, pose 67.90 / 63.50]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
