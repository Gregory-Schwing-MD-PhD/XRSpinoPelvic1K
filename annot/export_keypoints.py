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

A LANDMARK HAS TO BE FINDABLE, NOT MERELY CONSISTENT
----------------------------------------------------
An earlier version of this file resampled the fitted circle at fixed clock angles and
exported those as keypoints. They were perfectly consistent between films and still the
wrong target, for a reason worth writing down:

  * "the rim point at 45 degrees" is only defined ONCE YOU HAVE THE CENTRE. The target
    was a function of the answer.
  * a femoral head is rotationally symmetric, so there is no local image evidence
    whatsoever separating the rim at 45 degrees from the rim at 50. A detector could only
    find it by localising the whole head first -- which is to say, by regressing an
    inferred quantity in disguise, exactly the problem the rim was supposed to solve.

So the exported keypoints are the three NAMED EXTREMES the reader actually marks:
anterior, superior, posterior. An extreme is a tangency condition -- the cortex is
vertical at A and P, horizontal at S -- and that IS local image evidence, findable
without knowing the centre. It is also the positional contract YOLO-Pose needs, because
keypoint 2 is the superior extreme on every film by construction rather than by
resampling convention.

FACING IS RECORDED, NOT ASSUMED
-------------------------------
On a lateral, anterior is a DIRECTION, and which way the patient faced varies across this
set. If it is not carried through, every A/P label in the export is a coin flip and the
positional contract is broken again in a way no test would notice. The annotator derives
it from the order the reader marked A and P and stores it per head; this exporter uses it
only as a consistency check, since the labels themselves are already anatomical.

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

# keypoint 0 is the DERIVED centre; 1..3 are the marked extremes. Four, fixed, named.
ROLES = ["A", "S", "P"]
KP_NAMES = ["centre", "anterior", "superior", "posterior"]


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


# ---------------------------------------------------------------- building instances
def _lm(p, k: int) -> dict:
    """Reader `p`'s named landmarks for head k, or {} for a read that has none."""
    L = p.get("landmarks")
    return L[k] if isinstance(L, list) and k < len(L) and isinstance(L[k], dict) else {}


def _dir(role: str, facing: str):
    """Unit direction from the centre to a named extreme, image coordinates (y down).

    Only needed for an extreme NOBODY marked, which is then written as unobserved -- but it
    still has to land on the right side of the head, or the label file would carry a
    posterior point sitting anteriorly and quietly poison any later re-derivation.
    """
    if role == "S":
        return (0.0, -1.0)
    if not facing:
        return None                                   # unknowable: no A/P was ever marked
    ant_left = (facing == "left")
    if role == "P":
        ant_left = not ant_left
    return (-1.0, 0.0) if ant_left else (1.0, 0.0)


def _at(seq, k, default=""):
    return seq[k] if isinstance(seq, list) and k < len(seq) else default


def instances(case: dict, circle_rim: bool, include_single: bool) -> list:
    """One record per femoral head: consensus centre, radius, and the three extremes.

    Both readers are used where the case is finalised. A landmark seen by BOTH is the mean
    of their two clicks and is exported as visible; one seen by only one reader keeps that
    reader's position at lower confidence; one neither reader could trace is placed from
    the fitted circle and marked NOT VISIBLE so it never enters the loss. The asymmetry is
    deliberate -- a landmark half the readers could not find is not the same evidence as
    one they both put in the same place.
    """
    slots = case.get("slots") or {}
    reads = [s for s in slots.values()
             if s.get("done") and not s.get("not_visible") and s.get("points")]
    if not reads:
        return []
    if len(reads) < 2 and not include_single:
        return []
    if case.get("final") and case["final"].get("points") is None:
        return []                                     # settled as no visible head

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
        rr = [x for x in ((P[i].get("radii") or [None] * (k + 1))[k] for i, k in idx) if x]
        if not rr:
            continue
        R = sum(rr) / len(rr)                          # fraction of image WIDTH

        # Facing has to agree between the readers. It is derived from the order each of
        # them marked A and P, so a mismatch means one has the two swapped -- and an
        # anterior label sitting posteriorly is worse than no label at all.
        face = {_at(P[i].get("facing"), k) for i, k in idx} - {""}
        facing = face.pop() if len(face) == 1 else ""
        conflict = len(face) > 0

        lms, seen = {}, []
        for role in ROLES:
            got = [d["xy"] for i, k in idx
                   for d in [(_lm(P[i], k).get(role) or {})]
                   if d.get("src") == "obs" and d.get("xy")]
            if got and len(got) == len(idx):           # every reader traced it
                lms[role] = (sum(g[0] for g in got) / len(got),
                             sum(g[1] for g in got) / len(got), 2)
            elif got:                                  # one reader of two
                lms[role] = (got[0][0], got[0][1], 1)
            else:                                      # nobody could: place, do not train
                v = _dir(role, facing)
                lms[role] = ((cx + R * v[0], cy + R * (W / H) * v[1], 0) if v
                             else (cx, cy, 0))
            seen.append(lms[role][2])
        tilt = [t for t in (_at(P[i].get("ap_tilt_deg"), k, None) for i, k in idx)
                if t is not None]
        out.append({"cx": cx, "cy": cy, "R": R, "lms": lms, "W": W, "H": H,
                    "facing": facing, "facing_conflict": conflict,
                    "n_obs": sum(1 for v in seen if v == 2),
                    "n_reads": len(P), "ap_tilt": max(tilt) if tilt else None,
                    "tool": "+".join(sorted({p.get("tool", "circle") for p in P}))})
    return out


def yolo_line(inst: dict, pad: float = 1.25) -> str:
    """`cls xc yc w h  (x y v) * 4`, all normalised, YOLO-Pose layout.

    Keypoints in a fixed order: centre, anterior, superior, posterior. The centre is always
    v=2 -- derived rather than seen, but always determined, and it is the quantity the
    study needs. The three extremes carry the visibility the readers actually earned.
    """
    W, H, R = inst["W"], inst["H"], inst["R"]
    asp = W / H                                        # R is in width-fractions
    bw, bh = min(1.0, 2 * R * pad), min(1.0, 2 * R * asp * pad)
    cl = lambda v: min(1.0, max(0.0, v))               # noqa: E731
    xc, yc = cl(inst["cx"]), cl(inst["cy"])
    kp = [f"{xc:.6f} {yc:.6f} 2"]
    for role in ROLES:
        x, y, v = inst["lms"][role]
        inside = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        kp.append(f"{cl(x):.6f} {cl(y):.6f} {v if inside else 0}")
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp)


DATA_YAML = """# Femoral head keypoints.
#
#   0 centre     DERIVED, never clicked. There is no edge at the centre of a femoral head
#                -- it is inside bone -- so it is solved from the circle the extremes fit.
#                It is what the study needs (the bicoxofemoral point is the midpoint of
#                the two centres) but it is not a landmark.
#   1 anterior   most anterior point of the articular surface  (cortex vertical there)
#   2 superior   top of the head                               (cortex horizontal there)
#   3 posterior  most posterior point                          (cortex vertical there)
#
# The extremes are marked by readers rather than resampled off the fitted circle, because
# an extreme is a TANGENCY CONDITION and so locally findable. Points partway round the rim
# are not: a femoral head is rotationally symmetric, nothing in the image separates the rim
# at 45 degrees from the rim at 50, and such a point is only definable once you already
# have the centre -- i.e. it is the answer wearing a landmark's clothes.
#
# visibility 0 = NEITHER reader could trace that extreme. The position was placed from the
# fitted circle and is excluded from the loss on purpose. 1 = one reader of two.
#
# At inference, fit a circle through the predicted extremes and prefer THAT centre over the
# regressed one. Extremes that are not equidistant from the predicted centre are a free
# reliability signal a bare centre regression cannot express.
path: {path}
train: images/train
val: images/val
kpt_shape: [4, 3]
# a horizontal flip exchanges anterior and posterior; centre and superior map to themselves
flip_idx: [0, 3, 2, 1]
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
                    help="unused placeholder kept so old invocations do not break: a "
                         "circle-tool read names no extremes, so it can only ever "
                         "contribute a centre.")
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
            vis_hist[i["n_obs"]] += 1
            tally[f"tool={i['tool']}"] += 1
            if i["facing_conflict"]:
                tally["FACING CONFLICT (readers disagree on A/P)"] += 1
            if i["ap_tilt"] is not None and i["ap_tilt"] > 12:
                tally["A/P tilt > 12 deg (non-spherical or swapped)"] += 1
        tally[f"{len(inst)} head(s)"] += 1
        kept += 1
        mid = None
        if len(inst) == 2:
            mid = [(inst[0]["cx"] + inst[1]["cx"]) / 2,
                   (inst[0]["cy"] + inst[1]["cy"]) / 2]
        elif len(inst) == 1:
            mid = [inst[0]["cx"], inst[0]["cy"]]
        meta.append({"case": cid, "split": split, "heads": len(inst),
                     "tool": inst[0]["tool"], "facing": inst[0]["facing"],
                     "observed_extremes": [i["n_obs"] for i in inst],
                     "bicoxofemoral": mid})
        if a.images:
            src = next((p for p in Path(a.images).glob(f"{cid}.*")), None)
            if src:
                shutil.copy2(src, out / "images" / split / src.name)

    (out / "data.yaml").write_text(DATA_YAML.format(path=str(out.resolve())))
    (out / "manifest.json").write_text(json.dumps(
        {"ledger": a.ledger, "cases": len(cases), "exported": kept,
         "keypoints": KP_NAMES,
         "bicoxofemoral_note": "derived, not a training target: it is the midpoint of the "
                               "fitted centres and is recorded here so the study quantity "
                               "survives alongside the detector labels.",
         "films": meta}, indent=1))

    print(f"  cases in ledger      {len(cases)}")
    print(f"  exported             {kept}")
    for k, v in tally.most_common():
        print(f"      {k:44s} {v}")
    print("\n  extremes BOTH readers traced, per head (of 3):")
    for k in sorted(vis_hist):
        bar = "#" * min(40, vis_hist[k] * 40 // max(1, max(vis_hist.values())))
        print(f"      {k}: {vis_hist[k]:5d}  {bar}")
    if vis_hist and max(vis_hist) == 0:
        print("\n  NOTE: no read named any extreme, so only the centre keypoint is "
              "trainable\n  from this export. Those are circle-tool reads.")
    print(f"\n  wrote {out}/ (labels, data.yaml, manifest.json)")


if __name__ == "__main__":
    sys.exit(main())
