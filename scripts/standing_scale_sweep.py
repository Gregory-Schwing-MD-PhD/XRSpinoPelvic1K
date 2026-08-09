"""How far does the lumbar detector survive when the film is a WHOLE-SPINE standing view?

The model was trained on coned lateral lumbar films, where L1-S1 fills most of the
frame. On a standing C2-S1 radiograph the same six vertebrae occupy roughly a third of
it. Nothing else changes -- same anatomy, same projection, same detector -- so the
question is purely one of SCALE at the network input, and that can be measured on the
films we already have ground truth for instead of waiting for a standing dataset.

Each test film is pasted into a canvas 1/f times its size, so the lumbar spine occupies
a fraction f of the frame. f = 1.0 is the coned film as acquired; f ~ 0.36 is where
L1-S1 sits on a C2-S1 standing view (a ~20 cm lumbar segment in a ~55 cm spine, plus
pelvis). Detections are mapped back to original film pixels, so every error below is in
the SAME units at every f and the numbers are directly comparable.

Padding is mid-grey, not black: a standing film's margins contain thoracic spine and
pelvis, not vacuum, and black padding would additionally hand the detector a hard edge
that no real film has. This still under-states the difficulty -- real margins contain
DISTRACTORS, more vertebrae that look exactly like the ones being counted -- so treat
the curve as an upper bound on standing-film performance.

Also evaluates the obvious mitigation: run the detector once to find the spine, crop to
it, and run again at native scale.

    python scripts/standing_scale_sweep.py --model <onnx> --out results_standing
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import onnxruntime as ort

CORNERS = ("sup_ant", "sup_post", "inf_ant", "inf_post")
LEVELS = ("L1", "L2", "L3", "L4", "L5", "S1")
PAD = 114                      # the Ultralytics letterbox fill, so the padding the
                               # network sees at the border matches what it trained with
FRACTIONS = (1.0, 0.8, 0.65, 0.5, 0.42, 0.36, 0.3, 0.25, 0.2)


# ── the same decode the browser runs (pacs/infer.js) ─────────────────────────────

def letterbox(im: Image.Image, S: int):
    w, h = im.size
    s = min(S / w, S / h)
    nw, nh = round(w * s), round(h * s)
    left, top = round((S - nw) / 2 - 0.1), round((S - nh) / 2 - 0.1)
    cv = Image.new("RGB", (S, S), (PAD, PAD, PAD))
    cv.paste(im.convert("RGB").resize((nw, nh), Image.BILINEAR), (left, top))
    return cv, s, left, top


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = x1 - x0, y1 - y0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def infer(sess, im: Image.Image, S: int, conf=0.5, nms_iou=0.7):
    cv, s, left, top = letterbox(im, S)
    x = np.asarray(cv, np.float32).transpose(2, 0, 1)[None] / 255.0
    o = sess.run(None, {"images": x})[0][0]
    idx = np.where(o[4] >= conf)[0]
    if not len(idx):
        return []
    cx, cy, bw, bh = o[0][idx], o[1][idx], o[2][idx], o[3][idx]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    cf = o[4][idx]
    kp = o[5:].T[idx].reshape(-1, 4, 3)
    keep = []
    for i in np.argsort(-cf):
        if all(_iou(boxes[k], boxes[i]) <= nms_iou for k in keep):
            keep.append(i)
    out = []
    for i in keep:
        b = boxes[i]
        out.append({
            "conf": float(cf[i]),
            "box": [(b[0] - left) / s, (b[1] - top) / s,
                    (b[2] - left) / s, (b[3] - top) / s],
            "kpts": [[(kp[i][k][0] - left) / s, (kp[i][k][1] - top) / s,
                      float(kp[i][k][2])] for k in range(4)],
        })
    return out


def tiled_infer(sess, im: Image.Image, S: int, conf=0.5, nms_iou=0.7, overlap=0.5):
    """Sliding-window detection, no prior knowledge of where the spine is.

    A whole-spine film letterboxed into 640 leaves each vertebra a few pixels tall, so
    the single-shot detector finds nothing. Square tiles the width of the film, stepped
    down it with 50% overlap, put each vertebra back at roughly the scale the detector
    was trained on. The overlap is what makes it safe: a vertebra straddling a tile
    boundary is whole in the neighbouring tile, and global NMS keeps the better copy.

    Costs one forward pass per tile -- about nine on a standing film.
    """
    CW, CH = im.size
    step = max(1, int(CW * (1 - overlap)))
    tops = list(range(0, max(1, CH - CW + 1), step))
    if not tops or tops[-1] + CW < CH:
        tops.append(max(0, CH - CW))
    alld = []
    for t in tops:
        tile = im.crop((0, t, CW, min(CH, t + CW)))
        for d in infer(sess, tile, S, conf, nms_iou):
            d["box"] = [d["box"][0], d["box"][1] + t, d["box"][2], d["box"][3] + t]
            d["kpts"] = [[k[0], k[1] + t, k[2]] for k in d["kpts"]]
            alld.append(d)
    keep = []
    for d in sorted(alld, key=lambda x: -x["conf"]):
        if all(_iou(k["box"], d["box"]) <= nms_iou for k in keep):
            keep.append(d)
    return keep


def assign_levels(dets):
    """Caudal end up, exactly as the page does it."""
    chain = ["S1", "L5", "L4", "L3", "L2", "L1"] + [f"T{n}" for n in range(12, 0, -1)]
    order = sorted(dets, key=lambda d: -(d["box"][1] + d["box"][3]) / 2)
    return {chain[i]: d for i, d in enumerate(order) if i < len(chain)}


def angle_h(v):
    a = abs(np.degrees(np.arctan2(v[1], v[0])))
    return 180 - a if a > 90 else a


def cobb(v1, v2):
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    a = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return 180 - a if a > 90 else a


def angles_from(by_level):
    out = {}
    s1, l1 = by_level.get("S1"), by_level.get("L1")
    if s1:
        v = np.array(s1["kpts"][1][:2]) - np.array(s1["kpts"][0][:2])
        out["SS"] = angle_h(v)
    if s1 and l1:
        v1 = np.array(l1["kpts"][1][:2]) - np.array(l1["kpts"][0][:2])
        v2 = np.array(s1["kpts"][1][:2]) - np.array(s1["kpts"][0][:2])
        out["LL"] = cobb(v1, v2)
    return out


# ── ground truth ─────────────────────────────────────────────────────────────────

def load_gt(label_path: Path):
    """Ultralytics keypoint labels -> {level: 4x2} in ORIGINAL film pixels, ordered
    caudal-first exactly like assign_levels, so the two are compared like for like."""
    rows = []
    for line in label_path.read_text().strip().splitlines():
        p = [float(v) for v in line.split()]
        rows.append(p)
    return rows


def gt_in_px(rows, W, H):
    inst = []
    for p in rows:
        k = np.array(p[5:]).reshape(4, 3)
        pts = np.stack([k[:, 0] * W, k[:, 1] * H], 1)
        inst.append({"pts": pts, "vis": k[:, 2], "cy": float(p[2] * H)})
    inst.sort(key=lambda d: -d["cy"])
    return {(["S1", "L5", "L4", "L3", "L2", "L1"] + [f"T{n}" for n in range(12, 0, -1)])[i]: d
            for i, d in enumerate(inst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="/data/buu_yolo/images/test")
    ap.add_argument("--labels", default="/data/buu_yolo/labels/test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tiled", action="store_true",
                    help="sliding-window instead of one shot at the whole film")
    ap.add_argument("--out", default="/data/results_standing")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    imgs = sorted(Path(a.images).glob("*"))[:a.n]
    print(f"{len(imgs)} films, imgsz {a.imgsz}, model {Path(a.model).name}\n")

    rec = []
    for n, ip in enumerate(imgs):
        lp = Path(a.labels) / (ip.stem + ".txt")
        if not lp.exists():
            continue
        im = Image.open(ip).convert("L")
        W, H = im.size
        gt = gt_in_px(load_gt(lp), W, H)
        gt_ang = {}
        if "S1" in gt:
            gt_ang["SS"] = angle_h(gt["S1"]["pts"][1] - gt["S1"]["pts"][0])
        if "S1" in gt and "L1" in gt:
            gt_ang["LL"] = cobb(gt["L1"]["pts"][1] - gt["L1"]["pts"][0],
                                gt["S1"]["pts"][1] - gt["S1"]["pts"][0])

        for f in FRACTIONS:
            # A standing film is TALLER, not wider: height grows to 1/f, width keeps a
            # normal lateral margin. Scaling both (the obvious thing) would make the
            # canvas absurdly wide and understate the achievable tile scale, which is
            # the whole point of the mitigation being tested here.
            CW, CH = int(round(W * 1.15)), int(round(H / f))
            cv = Image.new("L", (CW, CH), PAD)
            ox, oy = (CW - W) // 2, (CH - H) // 2
            cv.paste(im, (ox, oy))
            dets = (tiled_infer(sess, cv, a.imgsz) if a.tiled
                    else infer(sess, cv, a.imgsz))
            for d in dets:                       # canvas px -> original film px
                d["box"] = [d["box"][0] - ox, d["box"][1] - oy,
                            d["box"][2] - ox, d["box"][3] - oy]
                d["kpts"] = [[k[0] - ox, k[1] - oy, k[2]] for k in d["kpts"]]
            by = assign_levels(dets)
            ang = angles_from(by)

            errs = []
            for lv in LEVELS:
                if lv in gt and lv in by:
                    p = np.array(by[lv]["kpts"])[:, :2]
                    for j in range(4):
                        if gt[lv]["vis"][j] > 0:
                            errs.append(float(np.linalg.norm(p[j] - gt[lv]["pts"][j])))
            rec.append({
                "image": ip.name, "f": f, "n_det": len(dets),
                "n_levels": sum(1 for lv in LEVELS if lv in by),
                "med_px": float(np.median(errs)) if errs else None,
                "diag": float(np.hypot(W, H)),
                "ss_err": abs(ang["SS"] - gt_ang["SS"])
                          if "SS" in ang and "SS" in gt_ang else None,
                "ll_err": abs(ang["LL"] - gt_ang["LL"])
                          if "LL" in ang and "LL" in gt_ang else None,
            })
        print(f"  [{n + 1}/{len(imgs)}] {ip.name}")

    (out / "raw.json").write_text(json.dumps(rec, indent=1))

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'f':>6} {'det/6':>7} {'films 6/6':>10} {'med px':>9} "
          f"{'%diag':>7} {'SS MAE':>8} {'LL MAE':>8}")
    print("-" * 60)
    rows = []
    for f in FRACTIONS:
        g = [r for r in rec if r["f"] == f]
        if not g:
            continue
        lv = np.mean([r["n_levels"] for r in g])
        full = np.mean([r["n_levels"] == 6 for r in g]) * 100
        med = [r["med_px"] for r in g if r["med_px"] is not None]
        pd = [r["med_px"] / r["diag"] * 100 for r in g if r["med_px"] is not None]
        ss = [r["ss_err"] for r in g if r["ss_err"] is not None]
        ll = [r["ll_err"] for r in g if r["ll_err"] is not None]
        row = dict(f=f, det=lv, full_pct=full,
                   med_px=float(np.median(med)) if med else None,
                   med_pct_diag=float(np.median(pd)) if pd else None,
                   ss_mae=float(np.mean(ss)) if ss else None,
                   ll_mae=float(np.mean(ll)) if ll else None)
        rows.append(row)
        c_med = "%9.2f" % row["med_px"] if med else "        -"
        c_pct = "%6.3f%%" % row["med_pct_diag"] if pd else "      -"
        c_ss = "%8.2f" % row["ss_mae"] if ss else "       -"
        c_ll = "%8.2f" % row["ll_mae"] if ll else "       -"
        print(f"{f:6.2f} {lv:7.2f} {full:9.0f}% {c_med} {c_pct} {c_ss} {c_ll}")

    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}/summary.json")


if __name__ == "__main__":
    main()
