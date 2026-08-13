"""Turn the femoral-head ledger into YOLO-Pose keypoint labels.

WHY THE RIM CLICKS ARE WORTH KEEPING
------------------------------------
A keypoint model asked to regress the femoral-head CENTRE is being asked to predict a
point that does not exist in the image. There is no edge at the centre of a femoral head,
no texture and no gradient -- it is inside bone. The network can only learn it the way a
reader does, by inferring it from the rim, and it has to learn that inference implicitly
from the pixels with no supervision on the intermediate step.

The rim is the opposite. The subchondral cortex is one of the sharpest edges on a pelvic
radiograph. It is exactly the kind of thing a detector is good at. So the arc tool records
the rim clicks, and this script turns them into supervision for the visible thing, with
the centre riding along as a derived keypoint.

At inference the same trick runs in reverse: predict the rim points, fit a circle, and
take the FITTED centre rather than the directly-regressed one. Three predicted rim points
that are not equidistant from the predicted centre are also a free consistency check --
a way for the model to say "this hip is unreliable" that a bare centre regression cannot
express.

THE CATCH, AND THE FIX
----------------------
YOLO-Pose keypoints are a POSITIONAL CONTRACT: keypoint k must mean the same anatomical
thing on every image, because the loss compares keypoint k to keypoint k. Raw rim clicks
break that immediately -- a reader's first click is 12 o'clock on one film and 4 o'clock
on the next, and averaged over a dataset that trains the model to predict the mean of the
rim, which is the centre, badly.

So the raw clicks are NOT exported. They are used to fit a circle, and the export
resamples that circle at FIXED CLOCK ANGLES. The reader clicks wherever the cortex is
clear; the training target is deterministic. Keypoint 3 is always the same place on the
head.

The angles are defined in IMAGE directions (right, down, left, up), not anatomical ones.
That is deliberate. Whether image-left is anterior or posterior depends on which way the
patient faced, which varies across this set -- but the femoral head is a SPHERE, so its
rim looks the same at every clock angle and the distinction that would matter for, say,
a vertebral corner does not exist here. Image directions are consistent, checkable, and
sufficient.

VISIBILITY IS NOT DECORATION
----------------------------
A clock angle outside the arc the reader actually marked is EXTRAPOLATED. Writing it as
a confident target would teach the model to hallucinate rim in exactly the films where
the rim cannot be seen. YOLO-Pose has a per-keypoint visibility flag for this; angles
outside the observed span get v=0 and are dropped from the loss.

    python annot/export_keypoints.py --out kp_dataset
    python annot/export_keypoints.py --ledger local:/path/to/ledger --out kp --images DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# 4 rim points + the derived centre. The rim points are the supervision; the centre is the
# quantity the study needs. More rim points give the circle-refit at inference more to
# work with, at the cost of more of them falling outside the observed span on tight films.
RIM = 8
CENTRE_KP = 0                     # keypoint 0 is the centre, 1..RIM are the rim clockwise


def rim_angles(n: int = RIM) -> list:
    """Fixed clock angles, image coordinates (y down). 0 = image-right, pi/2 = image-down."""
    return [2 * math.pi * i / n for i in range(n)]


# ---------------------------------------------------------------- reading the ledger
def load_cases(spec: str) -> list:
    """Every case record, from a local ledger dir or an HF dataset repo."""
    if spec.startswith("local:"):
        root = Path(spec[6:]) / "cases"
        return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(spec, repo_type="dataset", allow_patterns="cases/*.json",
                                  max_workers=16, token=os.environ.get("HF_TOKEN")))
    return [json.loads(p.read_text()) for p in sorted((root / "cases").glob("*.json"))]


def _heads(p) -> list:
    if not p:
        return []
    if isinstance(p.get("heads"), list):
        return [q for q in p["heads"] if q]
    return [q for q in (p.get("left"), p.get("right")) if q]


def _pair(A: list, B: list) -> list:
    """Index pairs between two readers' marks, order-free.

    A lateral cannot tell you which head is left and which is right, so reader A's first
    mark is not necessarily reader B's first. Same rule the agreement score uses: with two
    marks each, try both pairings and keep the tighter one.
    """
    d = lambda u, v: math.hypot(u[0] - v[0], u[1] - v[1])   # noqa: E731
    if len(A) == 2 and len(B) == 2:
        straight = max(d(A[0], B[0]), d(A[1], B[1]))
        crossed = max(d(A[0], B[1]), d(A[1], B[0]))
        return [(0, 0), (1, 1)] if straight <= crossed else [(0, 1), (1, 0)]
    if not A or not B:
        return []
    i, j = min(((i, j) for i in range(len(A)) for j in range(len(B))),
               key=lambda t: d(A[t[0]], B[t[1]]))
    return [(i, j)]


def _spans(p, k: int):
    """(lo, span) in radians for head k, or None when the read carries no arc."""
    lo, sp = p.get("arc_lo"), p.get("arc_span")
    if not isinstance(lo, list) or not isinstance(sp, list) or k >= len(lo):
        return None
    return float(lo[k]), float(sp[k])


def _observed(ang: float, lo: float, span: float) -> bool:
    return ((ang - lo) % (2 * math.pi)) <= span + 1e-9


# ---------------------------------------------------------------- building instances
def instances(case: dict, circle_rim: bool, include_single: bool) -> list:
    """One record per femoral head: consensus centre, radius, and per-angle visibility.

    Uses BOTH readers where the case is finalised -- the centre and radius are averaged
    over the pairing, and a clock angle counts as observed only when BOTH readers marked
    evidence there. Intersection rather than union: a target one reader never looked at is
    not a target two readers agreed on.
    """
    slots = case.get("slots") or {}
    reads = [s for s in slots.values()
             if s.get("done") and not s.get("not_visible") and s.get("points")]
    if not reads:
        return []
    if len(reads) < 2 and not include_single:
        return []
    if case.get("final") and case["final"].get("points") is None:
        return []                                  # settled as no visible head

    P = [r["points"] for r in reads[:2]]
    dims = next((p for p in P if p.get("w") and p.get("h")), None)
    if not dims:
        return []
    W, H = float(dims["w"]), float(dims["h"])
    Hd = [_heads(p) for p in P]
    if not all(Hd):
        return []

    pairs = _pair(Hd[0], Hd[1]) if len(P) == 2 else [(k, k) for k in range(len(Hd[0]))]
    out = []
    for a, b in pairs:
        idx = [(0, a)] + ([(1, b)] if len(P) == 2 else [])
        cx = sum(Hd[i][k][0] for i, k in idx) / len(idx)
        cy = sum(Hd[i][k][1] for i, k in idx) / len(idx)
        rr = [(P[i].get("radii") or [])[k] for i, k in idx
              if k < len(P[i].get("radii") or [])]
        rr = [r for r in rr if r]
        if not rr:
            continue
        R = sum(rr) / len(rr)                       # fraction of image WIDTH

        vis = []
        for ang in rim_angles():
            ok = []
            for i, k in idx:
                sp = _spans(P[i], k)
                if sp is None:                       # a circle-tool read: the rim at this
                    ok.append(circle_rim)            # angle was asserted, never observed
                else:
                    ok.append(_observed(ang, sp[0], sp[1]))
            vis.append(2 if all(ok) else 0)
        tools = sorted({p.get("tool", "circle") for p in P})
        out.append({"cx": cx, "cy": cy, "R": R, "vis": vis, "W": W, "H": H,
                    "n_reads": len(P), "tool": "+".join(tools)})
    return out


def yolo_line(inst: dict, pad: float = 1.25) -> str:
    """`cls xc yc w h  (x y v) * (1 + RIM)`, all normalised, YOLO-Pose layout.

    x is normalised by width and y by height, so the circle of radius R (a fraction of
    WIDTH) becomes an ellipse in normalised space -- the aspect factor has to be applied
    to every y or the rim points land off the cortex on a tall lateral.
    """
    W, H, R = inst["W"], inst["H"], inst["R"]
    asp = W / H                                     # R is in width-fractions; y needs this
    bw, bh = min(1.0, 2 * R * pad), min(1.0, 2 * R * asp * pad)
    xc = min(1.0, max(0.0, inst["cx"]))
    yc = min(1.0, max(0.0, inst["cy"]))
    kp = [f"{xc:.6f} {yc:.6f} 2"]
    for ang, v in zip(rim_angles(), inst["vis"]):
        x = xc + R * math.cos(ang)
        y = yc + R * asp * math.sin(ang)
        inside = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        kp.append(f"{min(1.0, max(0.0, x)):.6f} {min(1.0, max(0.0, y)):.6f} "
                  f"{v if inside else 0}")
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp)


DATA_YAML = """# Femoral head keypoints, derived from reader rim clicks.
#
# kpt 0 is the CENTRE and is derived, never clicked: it is not visible in the image.
# kpts 1..{n} are the rim resampled at fixed clock angles, image directions, clockwise
# from image-right. Visibility 0 means that angle fell OUTSIDE the arc the readers
# actually marked -- it is extrapolation, and it is excluded from the loss on purpose.
#
# At inference, refit a circle to the predicted rim keypoints and prefer the FITTED
# centre over the regressed one. The rim is observable; the centre is not.
path: {path}
train: images/train
val: images/val
kpt_shape: [{k}, 3]
flip_idx: {flip}
names:
  0: femoral_head
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.environ.get(
        "ANNOT_REPO", "gregoryschwingmdphd/xrsp-femhead-annot"))
    ap.add_argument("--out", default="kp_dataset")
    ap.add_argument("--images", help="directory of source films to copy alongside")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--include-single", action="store_true",
                    help="also export films with only one read (default: two or none)")
    ap.add_argument("--circle-rim", action="store_true",
                    help="treat circle-tool reads as rim evidence at every angle. OFF by "
                         "default: that reader sized a circle, they did not assert that "
                         "the cortex was traceable all the way round, and training on it "
                         "teaches the model to hallucinate rim where none was seen.")
    a = ap.parse_args()

    cases = load_cases(a.ledger)
    out = Path(a.out)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "val").mkdir(parents=True, exist_ok=True)

    tally, kept, vis_hist = Counter(), 0, Counter()
    meta = []
    for n, case in enumerate(cases):
        cid = case.get("case_id") or f"case{n:05d}"
        inst = instances(case, a.circle_rim, a.include_single)
        if not inst:
            tally["no usable read"] += 1
            continue
        split = "val" if (hash(cid) % 1000) / 1000.0 < a.val_frac else "train"
        (out / "labels" / split / f"{cid}.txt").write_text(
            "\n".join(yolo_line(i) for i in inst) + "\n")
        for i in inst:
            vis_hist[sum(1 for v in i["vis"] if v)] += 1
            tally[f"tool={i['tool']}"] += 1
        tally[f"{len(inst)} head(s)"] += 1
        kept += 1
        mid = None
        if len(inst) == 2:
            mid = [(inst[0]["cx"] + inst[1]["cx"]) / 2,
                   (inst[0]["cy"] + inst[1]["cy"]) / 2]
        elif len(inst) == 1:
            mid = [inst[0]["cx"], inst[0]["cy"]]
        meta.append({"case": cid, "split": split, "heads": len(inst),
                     "tool": inst[0]["tool"], "bicoxofemoral": mid})
        if a.images:
            src = next((p for p in Path(a.images).glob(f"{cid}.*")), None)
            if src:
                shutil.copy2(src, out / "images" / split / src.name)

    (out / "data.yaml").write_text(DATA_YAML.format(
        n=RIM, path=str(out.resolve()), k=RIM + 1,
        # a horizontal flip maps image-right rim to image-left: angle -> -angle.
        # The centre maps to itself.
        flip=[0] + [1 + ((RIM - i) % RIM) for i in range(RIM)]))
    (out / "manifest.json").write_text(json.dumps(
        {"ledger": a.ledger, "cases": len(cases), "exported": kept,
         "rim_points": RIM, "circle_rim_as_evidence": a.circle_rim,
         "bicoxofemoral_note": "derived, not a training target: it is the midpoint of the "
                               "fitted centres and is recorded here so the study quantity "
                               "survives alongside the detector labels.",
         "films": meta}, indent=1))

    print(f"  cases in ledger      {len(cases)}")
    print(f"  exported             {kept}")
    for k, v in tally.most_common():
        print(f"      {k:24s} {v}")
    print(f"\n  observed rim angles per head (of {RIM}):")
    for k in sorted(vis_hist):
        bar = "#" * min(40, vis_hist[k] * 40 // max(1, max(vis_hist.values())))
        print(f"      {k}: {vis_hist[k]:5d}  {bar}")
    if vis_hist and max(vis_hist) == 0:
        print("\n  NOTE: no read carried an arc, so every rim point is extrapolated and "
              "masked out.\n  Only the centre keypoint is trainable from this export.")
    print(f"\n  wrote {out}/ (labels, data.yaml, manifest.json)")


if __name__ == "__main__":
    sys.exit(main())
