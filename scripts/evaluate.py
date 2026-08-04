#!/usr/bin/env python3
"""Evaluate a trained landmark model on held-out DRRs — INTERNAL validity.

    python scripts/evaluate.py --data data/xrsp1k --splits data/splits.json \
        --fold 0 --ckpt runs/f0/best.ckpt --out results/f0

Ground truth here is AMODAL and exact, so this measures whether the network learned
the target. It does NOT measure the domain: same renderer, same physics, same
distribution. Report it as in-silico and pair it with the real-radiograph evaluation
(scripts/evaluate_buu.py) and the DRR reader study — see docs/PIPELINE.md §4.

Reports, per fold:
  * landmark error in px and mm, overall and per channel
  * SS / PT / PI / LL error in degrees, predicted-vs-3-D-truth
  * |SS + PT - PI| on the PREDICTIONS: a geometric identity, so a non-zero value is
    internal inconsistency in the predicted landmarks and needs no ground truth at all
  * everything stratified by obliquity, since that is the domain-randomisation claim
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrsp import measure as M                                   # noqa: E402
from xrsp import splits as S                                    # noqa: E402
from xrsp.dataset import (LandmarkDRRDataset, collate,          # noqa: E402
                          index_views, levels_present)
from xrsp.heatmaps import channel_names, points_from_json, soft_argmax   # noqa: E402


def _angle_errors(pred_pts, true_pts, anchor="corner"):
    p = M.spinopelvic(pred_pts, anchor=anchor)
    t = M.spinopelvic(true_pts, anchor=anchor)
    out = {}
    for k in ("PI", "SS", "PT", "LL"):
        out[k] = (None if (p[k] is None or t[k] is None) else abs(p[k] - t[k]))
        out[f"{k}_pred"], out[f"{k}_true"] = p[k], t[k]
    out["pi_identity_pred"] = M.pi_identity_error(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--anchor", choices=("corner", "overmask"), default="corner")
    a = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from xrsp.model import LandmarkNet                          # noqa: F401

    folds = S.load(a.splits)
    rows = S.view_rows_for_fold(index_views(a.data), folds, a.fold, split="val")
    if not rows:
        sys.exit(f"fold {a.fold} has no held-out views")
    cfg_path = os.path.join(os.path.dirname(a.ckpt), "run_config.json")
    levels = (json.load(open(cfg_path))["levels"] if os.path.exists(cfg_path)
              else levels_present(rows))
    names = channel_names(levels)
    ds = LandmarkDRRDataset(rows, out_size=(a.height, a.width), levels=levels)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    from xrsp.model import _make_module
    model = _make_module().load_from_checkpoint(a.ckpt, map_location="cpu")
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    per_view, per_channel_err = [], {n: [] for n in names}
    with torch.no_grad():
        for img, hm, valid, metas in dl:
            pred = model(img.to(dev)).cpu()
            pp, _ = soft_argmax(pred)
            tp, _ = soft_argmax(hm)
            d = torch.linalg.norm(pp - tp, dim=-1)              # [B, C] px
            for b, meta in enumerate(metas):
                sx, sy = meta["scale_xy"]
                mm_per_px = meta["pixel_spacing_mm"] / max(sx, 1e-9)
                gt = json.loads(open(
                    os.path.join(a.data, meta["case"],
                                 f"{meta['view']}_corners.json")).read())
                true_pts = points_from_json(gt.get("endplate_corners"),
                                            gt.get("bicoxofemoral_px"), names)
                pred_pts = M.points_from_prediction(pp[b].numpy(), names,
                                                    scale_xy=(sx, sy))
                ok = valid[b].numpy() & np.isfinite(d[b].numpy())
                errs = d[b].numpy()[ok]
                for ci, nm in enumerate(names):
                    if ok[ci]:
                        per_channel_err[nm].append(float(d[b, ci]) * mm_per_px)
                row = {"case": meta["case"], "view": meta["view"],
                       "yaw_deg": gt.get("yaw_deg"), "pitch_deg": gt.get("pitch_deg"),
                       "n_valid": int(ok.sum()),
                       "landmark_px": float(errs.mean()) if errs.size else None,
                       "landmark_mm": float(errs.mean() * mm_per_px) if errs.size else None}
                row.update(_angle_errors(pred_pts, true_pts, anchor=a.anchor))
                per_view.append(row)

    os.makedirs(a.out, exist_ok=True)
    json.dump(per_view, open(os.path.join(a.out, "per_view.json"), "w"), indent=2)

    def agg(key):
        v = [r[key] for r in per_view if r.get(key) is not None]
        return (None if not v else
                {"mean": float(np.mean(v)), "median": float(np.median(v)),
                 "p95": float(np.percentile(v, 95)), "n": len(v)})

    summary = {"fold": a.fold, "n_views": len(per_view), "anchor": a.anchor,
               "levels": levels,
               "landmark_px": agg("landmark_px"), "landmark_mm": agg("landmark_mm"),
               "angles_deg": {k: agg(k) for k in ("PI", "SS", "PT", "LL")},
               "pi_identity_pred_deg": agg("pi_identity_pred"),
               "per_channel_mm": {n: (float(np.mean(v)) if v else None)
                                  for n, v in per_channel_err.items()}}
    # obliquity strata: the point of generating oblique views is that accuracy holds
    obl = [r for r in per_view if r.get("yaw_deg") is not None]
    if obl:
        summary["by_obliquity"] = {}
        for lo, hi in ((0, 3), (3, 8), (8, 90)):
            sel = [r for r in obl if lo <= abs(r["yaw_deg"]) < hi
                   and r.get("landmark_mm") is not None]
            if sel:
                summary["by_obliquity"][f"|yaw| {lo}-{hi} deg"] = {
                    "n": len(sel),
                    "landmark_mm": float(np.mean([r["landmark_mm"] for r in sel]))}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
