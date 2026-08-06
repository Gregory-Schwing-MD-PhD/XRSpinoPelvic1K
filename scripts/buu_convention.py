#!/usr/bin/env python3
"""Measure the READER convention on BUU-LSpine, then check ours reproduces it.

BUU-LSpine 400 ships reader-placed vertebral corners on real lateral radiographs:
per vertebra a superior plate (`L1a`) and an inferior plate (`L1b`), each as two corner
points, L1..L5, plus `S1a` -- and, notably, no `S1b`. The readers mark no inferior S1
endplate, which is the same call this pipeline makes for the same reason: S1 is fused to
S2, so there is no disc space and no inferior plate to mark.

Why this is the right target: it removes the eyeballing. Our corners come from CT, BUU's
from radiographs of different patients, so they cannot be compared point-for-point --
but the CONVENTION can be compared, through quantities that do not depend on scale or on
which patient it is:

  * segmental and global lordosis angles (degrees)
  * the wedge angle between a vertebra's own two endplates (degrees)
  * plate span / body height (dimensionless aspect ratio)

If our corner rule matches the readers' rule, these distributions coincide. If we are
systematically cutting the plates short or chasing osteophytes, the aspect ratio and the
wedge angle move and this says so, with a number rather than an opinion.

The convention being reproduced is the standard one for spinopelvic measurement:
  Legaye J, Duval-Beaupere G, Hecquet J, Marty C. "Pelvic incidence: a fundamental
    pelvic parameter for three-dimensional regulation of spinal sagittal curves."
    Eur Spine J 1998;7(2):99-103.
  Berthonnaud E, Dimnet J, Roussouly P, Labelle H. "Analysis of the sagittal balance of
    the spine and pelvis using shape and orientation parameters."
    J Spinal Disord Tech 2005;18(1):40-47.
  Frobin W, Brinckmann P, Biggemann M, Tillotson M, Burton K. "Precision measurement of
    disc height, vertebral height and sagittal plane displacement from lateral
    radiographic views of the lumbar spine." Clin Biomech 1997;12(Suppl 1):S1-S63.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

LEVELS = ["L1", "L2", "L3", "L4", "L5"]
# CSV row order, cranial -> caudal: L1sup, L1inf, L2sup, L2inf, ... L5inf, S1sup
ROWS = [(lv, f) for lv in LEVELS for f in ("sup", "inf")] + [("S1", "sup")]


def load_case(path):
    """{(level, face): ((x1,y1),(x2,y2))} from one BUU lateral CSV."""
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if len(a) < len(ROWS):
        return None
    out = {}
    for i, key in enumerate(ROWS):
        out[key] = (a[i, 0:2].astype(float), a[i, 2:4].astype(float))
    return out


def plate_angle(p, q, anterior_sign=+1.0):
    """Signed angle of an endplate line, degrees, in image coordinates (y grows DOWN).

    Positive = the anterior end sits HIGHER, the sense lordosis is measured in.
    """
    d = np.asarray(q, float) - np.asarray(p, float)
    if anterior_sign < 0:
        d = -d
    return float(np.degrees(np.arctan2(-d[1], d[0])))


def wrap180(d):
    """Normalise an angle difference to (-180, 180].

    arctan2 returns (-180, 180], so a plain subtraction of two plate angles can come back
    as e.g. 312 deg for what is really -48. That produced an "LL" of 312.7 and a "wedge"
    of 353.5 before this was applied -- absurd on their face, but only because the values
    were far outside the anatomic range; a wrap error of the same kind inside the range
    would pass unnoticed.
    """
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def cobb(p1, p2, anterior_sign=+1.0):
    """Angle between two endplate lines (degrees), the Cobb construction."""
    return float(wrap180(plate_angle(*p2, anterior_sign)
                         - plate_angle(*p1, anterior_sign)))


def case_metrics(c):
    """Scale-free descriptors of ONE case."""
    m = {}
    # global lordosis: L1 superior to S1 superior -- the standard LL landmarks
    m["LL"] = abs(cobb(c[("L1", "sup")], c[("S1", "sup")]))
    for lv in LEVELS:
        s, i = c[(lv, "sup")], c[(lv, "inf")]
        # wedge: a vertebra's own two plates. Small and positive going down the lumbar
        # spine (bodies are slightly taller anteriorly); a large value means one plate
        # was fitted to something that is not an endplate.
        m[f"wedge_{lv}"] = abs(float(wrap180(plate_angle(*s) - plate_angle(*i))))
        span = 0.5 * (np.linalg.norm(s[1] - s[0]) + np.linalg.norm(i[1] - i[0]))
        mid_s, mid_i = 0.5 * (s[0] + s[1]), 0.5 * (i[0] + i[1])
        h = np.linalg.norm(mid_i - mid_s)
        if h > 1e-6:
            m[f"aspect_{lv}"] = span / h        # A-P depth / height, dimensionless
    for a, b in zip(LEVELS, LEVELS[1:] + ["S1"]):
        m[f"disc_{a}{b}"] = cobb(c[(a, "inf")], c[(b, "sup")])

    # --- the S1 plate specifically -------------------------------------------------
    # S1 is the only endplate whose CORNERS affect a spinopelvic parameter: PI and PT
    # depend on it through the plate MIDPOINT, and nothing else in the chain does (a
    # perturbation test puts SS and LL at exactly 0.000 deg and every non-S1 corner at
    # 0.000 for PI/PT too). So it is worth comparing on its own.
    #
    # PI is an ANGLE, so absolute position and image scale are irrelevant; what has to
    # match the readers is the plate's SHAPE and ORIENTATION. Both are captured
    # scale-free: its length relative to a lumbar endplate, and its angle relative to the
    # vertebra above it. A sacral plate that is systematically too long or mis-angled
    # shows up here even though ours is in mm and BUU's is in pixels.
    s1 = c[("S1", "sup")]
    s1_len = float(np.linalg.norm(s1[1] - s1[0]))
    l5s = c[("L5", "sup")]
    l5i = c[("L5", "inf")]
    for nm, ref in (("s1_over_L5sup", l5s), ("s1_over_L5inf", l5i)):
        rl = float(np.linalg.norm(ref[1] - ref[0]))
        if rl > 1e-6:
            m[nm] = s1_len / rl
    m["s1_vs_L5inf_deg"] = cobb(l5i, s1)
    m["s1_vs_L1sup_deg"] = cobb(c[("L1", "sup")], s1)
    return m


def summarize(rows, keys=None):
    ks = keys or sorted({k for r in rows for k in r})
    print(f"{'metric':14s}{'n':>5s}{'p5':>9s}{'median':>9s}{'p95':>9s}{'mean':>9s}{'sd':>8s}")
    out = {}
    for k in ks:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if not len(v):
            continue
        out[k] = v
        print(f"{k:14s}{len(v):5d}{np.percentile(v,5):9.1f}{np.median(v):9.1f}"
              f"{np.percentile(v,95):9.1f}{v.mean():9.1f}{v.std():8.1f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True, help="BUU-LSPINE_400 directory")
    ap.add_argument("--ours", nargs="*", default=[],
                    help="generated *_corners.json to score against the reader convention")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.buu, "LA", "*.csv")))
    if not files:
        sys.exit(f"no lateral CSVs under {a.buu}/LA")
    rows, bad = [], 0
    for f in files:
        c = load_case(f)
        if c is None:
            bad += 1
            continue
        rows.append(case_metrics(c))
    print(f"BUU-LSpine lateral: {len(rows)} cases ({bad} unreadable)\n")
    KEYS = (["LL"] + [f"wedge_{l}" for l in LEVELS] + [f"aspect_{l}" for l in LEVELS]
            + [f"disc_{a_}{b_}" for a_, b_ in zip(LEVELS, LEVELS[1:] + ["S1"])]
            + ["s1_over_L5sup", "s1_over_L5inf", "s1_vs_L5inf_deg", "s1_vs_L1sup_deg"])
    ref = summarize(rows, KEYS)
    if not a.ours:
        return 0

    ours = []
    for f in a.ours:
        c = load_ours(f)
        if c:
            ours.append(case_metrics(c))
    if not ours:
        print("\nno usable generated cases")
        return 1
    print(f"\n\nOURS (CT-derived, projected): {len(ours)} cases\n")
    mine = summarize(ours, KEYS)

    # Score: is each of ours inside the reader distribution? p5-p95 is the band 90% of
    # real reader annotations fall in, so landing outside it is a concrete, checkable
    # failure rather than an impression.
    print("\n\nAGREEMENT with the reader convention (BUU p5-p95 band)\n")
    print(f"{'metric':14s}{'BUU p5-p95':>18s}{'ours median':>13s}{'in band':>9s}{'ours outside':>14s}")
    n_bad = 0
    for k in KEYS:
        if k not in ref or k not in mine:
            continue
        lo, hi = np.percentile(ref[k], 5), np.percentile(ref[k], 95)
        v = mine[k]
        inb = int(((v >= lo) & (v <= hi)).sum())
        out = len(v) - inb
        n_bad += out
        flag = "" if out == 0 else "   <<<"
        print(f"{k:14s}{f'{lo:.1f} - {hi:.1f}':>18s}{np.median(v):13.1f}"
              f"{f'{inb}/{len(v)}':>9s}{out:14d}{flag}")
    print(f"\ntotal outside the reader band: {n_bad}")
    return 0


def load_ours(path):
    """Our generated *_corners.json -> the same {(level, face): (p, q)} structure.

    Emitted corners are named by anatomy (sup_ant / sup_post) and are ordered here to
    match BUU's convention -- ANTERIOR FIRST -- so every metric below is computed by
    identical code for both sources. Nothing of ours gets its own formula; that is the
    only way the comparison means anything.

    BUU's anterior side was determined from the data, not assumed. Two candidate tests
    disagree at first glance and the anatomy settles it:

      * the second point of the S1 plate is the higher one in 398/400 cases;
      * L1's plate midpoint sits ~100 px to the RIGHT of S1's.

    The sacral endplate slopes DOWN AND FORWARD -- the reason L5 tends to slip down it --
    so the promontory, the anterior corner, is the LOWER point. That makes BUU's FIRST
    point anterior and anterior = image-left; the second test then agrees, since lordosis
    carries L1 posterior to S1, i.e. to the right. Our renders put anterior on the RIGHT
    (see oblique._grid), so the two are mirrored, and getting this backwards silently
    flips the sign of every segmental disc angle while leaving LL, wedge and aspect --
    all unsigned -- looking perfectly fine.
    """
    import json
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    C = d.get("endplate_corners", {})
    out = {}
    for lv in LEVELS + ["S1"]:
        cs = C.get(lv)
        if not cs:
            return None                                   # need the full L1..S1 chain
        for face, (ka, kp) in (("sup", ("sup_ant", "sup_post")),
                               ("inf", ("inf_ant", "inf_post"))):
            if ka in cs and kp in cs:
                # MIRROR x. BUU images carry anterior on the LEFT, ours on the RIGHT
                # (oblique._grid), and a mirror NEGATES every signed angle. Re-ordering
                # the two points does not fix it -- swapping both plates rotates each by
                # 180 deg and the difference is unchanged, so the disc angles came out
                # sign-flipped either way until the mirror itself was applied. LL, wedge
                # and aspect are all unsigned and hid this completely.
                mir = np.array([-1.0, 1.0])
                out[(lv, face)] = (np.asarray(cs[ka], float) * mir,
                                   np.asarray(cs[kp], float) * mir)
    if ("S1", "sup") not in out:
        return None
    for lv in LEVELS:
        if (lv, "sup") not in out or (lv, "inf") not in out:
            return None
    return out


if __name__ == "__main__":
    raise SystemExit(main())
