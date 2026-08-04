#!/usr/bin/env python3
"""Evaluate on BUU Spine (real standing lateral radiographs) — EXTERNAL validity.

    python scripts/evaluate_buu.py --images /path/to/buu/images \
        --annotations /path/to/buu/corners.json --ckpt runs/f0/best.ckpt \
        --out results/buu

READ THIS BEFORE INTERPRETING THE OUTPUT
----------------------------------------
BUU's corners are what a HUMAN COULD SEE on the film. For S1 that ground truth
carries the very ala/body superimposition this model is trained to see through --
published manual inter-rater ICC for the sacral endplate is as low as 0.41, and the
cause named in the literature is that overlap.

So this script does NOT score amodal accuracy, and a disagreement here is not
automatically an error. What it measures is AGREEMENT WITH HUMAN READERS, in the
statistic radiologists use (corner distance; angle differences).

The disagreement only becomes interpretable alongside the DRR reader study
(scripts/make_reader_set.py), which measures the SIGNED human bias against amodal
truth. If this model departs from BUU's annotations by the same signed bias measured
there, that is evidence the readers are systematically wrong -- not the model. Run
both; report both. See docs/PIPELINE.md section 4.

Reported per level and overall:
  * corner distance, normalised by vertebral body height (scale-free: BUU has no
    pixel spacing, so an unnormalised px error is not comparable across films)
  * SS / LL agreement, and PI/PT where the femoral heads are in the field
  * SIGNED S1 offset along the endplate normal -- the quantity the reader study
    predicts, and the one that carries the argument
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xrsp import measure as M                                   # noqa: E402
from xrsp.heatmaps import channel_names, soft_argmax            # noqa: E402


def load_annotations(path):
    """BUU-style corner annotations -> {image_id: {level: {corner: [x, y]}}}.

    Accepts the common layouts; adapt `_norm` if BUU ships a different schema. The
    dataset is EVALUATION ONLY and is not redistributed with this repo (ROADMAP 3).
    """
    raw = json.load(open(path))
    if isinstance(raw, dict) and "images" in raw:
        raw = raw["images"]
    out = {}
    items = raw.items() if isinstance(raw, dict) else ((r["id"], r) for r in raw)
    for k, v in items:
        lv = v.get("levels") or v.get("corners") or v
        out[str(k)] = {str(a): {str(c): [float(p[0]), float(p[1])]
                                for c, p in b.items()} for a, b in lv.items()
                       if isinstance(b, dict)}
    return out


def body_height(corners):
    """Mean of the two lateral body heights — the normaliser for corner distance."""
    try:
        sa, sp = np.array(corners["sup_ant"]), np.array(corners["sup_post"])
        ia, ip = np.array(corners["inf_ant"]), np.array(corners["inf_post"])
    except KeyError:
        return None
    return float(0.5 * (np.linalg.norm(ia - sa) + np.linalg.norm(ip - sp)))


def signed_normal_offset(pred, true):
    """Signed distance from the TRUE S1 endplate to the PREDICTED one, along the true
    normal. Positive = the model places the endplate cranial to the reader.

    This is the number the DRR reader study predicts the sign of. Magnitude alone
    cannot distinguish 'model is wrong' from 'reader is biased'; the sign can.
    """
    try:
        ta, tp = np.array(true["sup_ant"]), np.array(true["sup_post"])
        pa, pp = np.array(pred["sup_ant"]), np.array(pred["sup_post"])
    except (KeyError, TypeError):
        return None
    d = tp - ta
    n = np.array([-d[1], d[0]], float)
    nn = np.linalg.norm(n)
    if nn == 0:
        return None
    n /= nn
    return float(((0.5 * (pa + pp)) - (0.5 * (ta + tp))) @ n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="dir of BUU lateral radiographs")
    ap.add_argument("--annotations", required=True, help="BUU corner annotations (json)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--glob", default="*.png")
    a = ap.parse_args(argv)

    import torch
    from PIL import Image
    from scipy.ndimage import zoom
    from xrsp.model import _make_module

    ann = load_annotations(a.annotations)
    cfg_path = os.path.join(os.path.dirname(a.ckpt), "run_config.json")
    if not os.path.exists(cfg_path):
        sys.exit("run_config.json not found beside the checkpoint — needed for channel order")
    cfg = json.load(open(cfg_path))
    names = cfg["channels"]

    model = _make_module().load_from_checkpoint(a.ckpt, map_location="cpu").eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    rows = []
    files = sorted(glob.glob(os.path.join(a.images, a.glob)))
    if not files:
        sys.exit(f"no images matching {a.glob} under {a.images}")
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        truth = ann.get(key)
        if not truth:
            continue
        img = np.asarray(Image.open(f).convert("L"), np.float32) / 255.0
        H0, W0 = img.shape
        sy, sx = a.height / H0, a.width / W0
        x = torch.from_numpy(zoom(img, (sy, sx), order=1)[None, None].astype(np.float32))
        with torch.no_grad():
            pp, _ = soft_argmax(model(x.to(dev)).cpu()[0])
        pred_pts = M.points_from_prediction(pp.numpy(), names, scale_xy=(sx, sy))
        pred_lv = {}
        for nm, p in pred_pts.items():
            if p is None or "." not in nm:
                continue
            lv, ck = nm.split(".", 1)
            pred_lv.setdefault(lv, {})[ck] = p
        for lv, tc in truth.items():
            pc = pred_lv.get(lv)
            if not pc:
                continue
            h = body_height(tc)
            for ck, tp in tc.items():
                if ck in pc:
                    d = float(np.linalg.norm(np.array(pc[ck]) - np.array(tp)))
                    rows.append({"image": key, "level": lv, "corner": ck,
                                 "dist_px": d,
                                 "dist_norm": (d / h) if h else None})
        # angle agreement + the signed S1 offset that carries the argument
        tp_flat = {f"{lv}.{ck}": v for lv, cs in truth.items() for ck, v in cs.items()}
        pa = M.spinopelvic(pred_pts)
        ta = M.spinopelvic(tp_flat)
        rec = {"image": key, "_angles": True}
        for k in ("PI", "SS", "PT", "LL"):
            rec[f"{k}_pred"], rec[f"{k}_reader"] = pa[k], ta[k]
            rec[k] = (None if (pa[k] is None or ta[k] is None) else pa[k] - ta[k])
        if "S1" in truth and "S1" in pred_lv:
            off = signed_normal_offset(pred_lv["S1"], truth["S1"])
            h = body_height(truth["S1"])
            rec["s1_signed_offset_px"] = off
            rec["s1_signed_offset_norm"] = (off / h) if (off is not None and h) else None
        rows.append(rec)

    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "per_item.json"), "w"), indent=2)

    corners = [r for r in rows if "corner" in r]
    angles = [r for r in rows if r.get("_angles")]

    def stat(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return (None if not v else {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                                    "median": float(np.median(v)), "n": len(v)})

    summary = {
        "n_images": len({r["image"] for r in rows}),
        "corner_dist_px": stat([r["dist_px"] for r in corners]),
        "corner_dist_norm_body_height": stat([r["dist_norm"] for r in corners]),
        "per_level_norm": {
            lv: stat([r["dist_norm"] for r in corners if r["level"] == lv])
            for lv in sorted({r["level"] for r in corners})},
        "angle_diff_model_minus_reader_deg": {
            k: stat([r[k] for r in angles]) for k in ("PI", "SS", "PT", "LL")},
        "s1_signed_offset_norm": stat([r.get("s1_signed_offset_norm") for r in angles]),
        "INTERPRETATION": (
            "Differences are model-vs-HUMAN, not model-vs-truth. BUU corners are "
            "modal (what a reader could see); for S1 they carry the ala superimposition "
            "this model is trained to see through. Compare s1_signed_offset_norm against "
            "the signed human bias from the DRR reader study before calling any of this "
            "an error. See docs/PIPELINE.md section 4."),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
