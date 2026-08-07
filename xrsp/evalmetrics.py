"""Evaluation metrics for the unified landmark model: agreement, not just error.

Separated from the plotting so every number a figure shows can be tested without
rendering anything, and so the JSON summary and the figures can never disagree.

WHAT IS BEING MEASURED, AND ON WHICH SET
----------------------------------------
The two streams supervise disjoint channels, so a single held-out image never carries
ground truth for everything:

  DRR test set  -- has BOTH the corners and the bicoxofemoral point (build_dataset emits
                   both; training merely ignores the corners). It is therefore the ONLY
                   set on which PI/PT can be evaluated end to end, because both need the
                   hip axis AND the S1 endplate on the same image.
  BUU test set  -- real radiographs, corners only, no hip ground truth. It is where
                   corner accuracy in the DEPLOYMENT domain is measured, and it cannot
                   produce PI or PT at all.

Reporting a single pooled number over both would be meaningless. Everything here is
computed per source and labelled.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CORNERS = ("sup_ant", "sup_post", "inf_ant", "inf_post")
ROUSSOULY_TYPES = ("1", "2", "3", "4")


# ── agreement ────────────────────────────────────────────────────────────────
def icc21(x: Sequence[float], y: Sequence[float]) -> float:
    """ICC(2,1) -- two-way random effects, absolute agreement, single measure.

    Absolute agreement, NOT consistency: a method that reads every PI 8 deg too high is
    perfectly *consistent* and clinically wrong. ICC(3,1) would score it ~1.0. Returns
    nan for fewer than 3 complete pairs rather than a confident number from noise.
    """
    a = np.asarray(x, float)
    b = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 3:
        return float("nan")
    M = np.column_stack([a, b])
    k = 2
    gm = M.mean()
    ms_r = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)              # between subjects
    ms_c = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)              # between raters
    ms_e = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float("nan") if den <= 0 else float((ms_r - ms_e) / den)


def bland_altman(truth: Sequence[float], pred: Sequence[float]) -> Dict:
    """Bias and 95% limits of agreement -- the standard for a new measurement method.

    Reported alongside a correlation on purpose. Correlation answers "do these move
    together", which a biased method can ace; Bland-Altman answers "how far apart is any
    single reading likely to be", which is the question a clinician actually has.
    """
    t = np.asarray(truth, float)
    p = np.asarray(pred, float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    if t.size < 2:
        return {"n": int(t.size), "bias": float("nan"), "sd": float("nan"),
                "loa_low": float("nan"), "loa_high": float("nan"),
                "mean": np.array([]), "diff": np.array([])}
    diff = p - t
    mean = (p + t) / 2.0
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    return {"n": int(t.size), "bias": bias, "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mean": mean, "diff": diff}


def error_summary(err: Sequence[float]) -> Dict:
    """Median and IQR lead, not the mean. Landmark error is right-skewed -- a handful of
    gross misses drag a mean somewhere no individual case sits. The tail is reported
    explicitly (p95, max) rather than smoothed into a standard deviation."""
    e = np.asarray([v for v in err if np.isfinite(v)], float)
    if e.size == 0:
        return {"n": 0, "median": float("nan"), "iqr": float("nan"),
                "mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    q1, q3 = np.percentile(e, [25, 75])
    return {"n": int(e.size), "median": float(np.median(e)), "iqr": float(q3 - q1),
            "mean": float(e.mean()), "p95": float(np.percentile(e, 95)),
            "max": float(e.max())}


def ed_accuracy(err: Sequence[float], thresholds=(5.0, 10.0, 15.0)) -> Dict:
    """Fraction of landmarks within each pixel threshold -- the statistic the vertebral
    keypoint literature reports, so results can be placed beside it directly.

    Bansal et al. (PLoS One 2026, e0347290) give ED<=5 px 75-79%, <=10 px ~98%, <=15 px
    ~100% for YOLOv8n/v11n-Pose and Detectron2 on 698 lateral lumbar films.

    READ THE COMPARISON CAREFULLY, because two things make it not like-for-like:
      * their images are resized to 640x640 NON-UNIFORMLY, so a pixel there is not a
        fixed physical distance and differs between x and y. Ours are 512x256 at a known
        mm/px. Equal pixel counts are not equal millimetres.
      * their keypoint metrics are computed only inside bounding boxes already matched at
        IoU>=0.5, so vertebrae they failed to detect are excluded from the keypoint score.
        Nothing here filters that way; a missed landmark counts against detection F1.
    Reported anyway, with those caveats, because an imperfect stated comparison beats an
    unstated one -- and mm is also reported when the pixel spacing is known.
    """
    e = np.asarray([v for v in err if np.isfinite(v)], float)
    if e.size == 0:
        return {f"within_{int(t)}px": float("nan") for t in thresholds}
    return {f"within_{int(t)}px": float((e <= t).mean()) for t in thresholds}


def ced_curve(err: Sequence[float], max_px: float = 20.0, n: int = 200):
    """Cumulative error distribution: fraction of landmarks within a threshold.

    The standard landmark-localisation figure, and more honest than any single number --
    a method with a good median and a heavy tail is instantly distinguishable from one
    that is uniformly mediocre, which no summary statistic separates.
    """
    e = np.asarray([v for v in err if np.isfinite(v)], float)
    th = np.linspace(0, max_px, n)
    if e.size == 0:
        return th, np.full_like(th, np.nan)
    return th, np.array([(e <= t).mean() for t in th])


# ── the two 4x4s ─────────────────────────────────────────────────────────────
def corner_identity_confusion(pred: Dict[str, Sequence[float]],
                              true: Dict[str, Sequence[float]],
                              levels: Sequence[str]) -> np.ndarray:
    """4x4: for each PREDICTED corner, which of its vertebra's 4 TRUE corners is nearest.

    A landmark model has no classes, so this is the nearest thing to a confusion matrix
    that means something: it asks whether each predicted point landed on the corner it
    was supposed to, or on one of its three neighbours. The off-diagonal is the
    interesting part --

        sup_ant <-> sup_post (or inf_ant <-> inf_post) is an ANTERIOR/POSTERIOR swap,
            the failure mode a left-right flip augmentation induces on a lateral film;
        sup_* <-> inf_*      is a superior/inferior slip, i.e. the wrong endplate,
            which propagates straight into the disc angles and so into LL.

    Compared WITHIN a vertebra, not globally: a corner landing on the next vertebra down
    is a different (and rarer) failure, and pooling the two would hide both.
    """
    M = np.zeros((4, 4), dtype=int)
    for lv in levels:
        tp = [true.get(f"{lv}.{c}") for c in CORNERS]
        if any(t is None or not np.all(np.isfinite(t)) for t in tp):
            continue                      # level absent from this view -> not a miss
        T = np.asarray(tp, float)
        for i, c in enumerate(CORNERS):
            p = pred.get(f"{lv}.{c}")
            if p is None or not np.all(np.isfinite(p)):
                continue                  # undetected -> counted in detection F1, not here
            M[i, int(np.argmin(np.linalg.norm(T - np.asarray(p, float), axis=1)))] += 1
    return M


def prf_from_confusion(M: np.ndarray) -> Dict:
    """Per-class precision/recall/F1 plus macro-F1, accuracy and Cohen's kappa.

    Kappa as well as accuracy because these matrices are not balanced -- Roussouly 3 is
    the commonest morphotype, so a model that answered "3" every time would post a
    respectable accuracy and a kappa near zero.
    """
    M = np.asarray(M, float)
    k = M.shape[0]
    tp = np.diag(M)
    prec = np.divide(tp, M.sum(0), out=np.full(k, np.nan), where=M.sum(0) > 0)
    rec = np.divide(tp, M.sum(1), out=np.full(k, np.nan), where=M.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.full(k, np.nan), where=np.nan_to_num(prec + rec) > 0)
    n = M.sum()
    acc = float(tp.sum() / n) if n else float("nan")
    pe = float((M.sum(0) * M.sum(1)).sum() / (n * n)) if n else float("nan")
    kappa = float((acc - pe) / (1 - pe)) if n and pe < 1 else float("nan")
    return {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
            "macro_f1": float(np.nanmean(f1)) if k else float("nan"),
            "accuracy": acc, "kappa": kappa, "support": M.sum(1).astype(int).tolist()}


FLAT_PROMINENCE = 0.02          # apex offset / chord length below which there is no apex


def lordosis_apex_level(corners: Dict[str, Sequence[float]],
                        lumbar: Sequence[str] = ("L1", "L2", "L3", "L4", "L5")
                        ) -> Optional[Tuple[str, float]]:
    """The lordosis apex: the level whose body centre is most ANTERIOR relative to the
    L1-S1 chord. Needed because SS cannot separate Roussouly 1 from 2.

    Apex-as-maximum-deviation-from-the-chord rather than apex-as-sign-change of the
    segmental angle: the sign-change definition is equivalent on a smooth spine but
    picks an essentially arbitrary level on a flat one -- which is precisely the type-2
    back this has to classify. Returns None when too few levels are visible to define a
    chord at all, so the caller reports "unclassifiable" rather than inventing a type.
    """
    cen, present = {}, []
    for lv in lumbar:
        pts = [corners.get(f"{lv}.{c}") for c in CORNERS]
        good = [np.asarray(p, float) for p in pts
                if p is not None and np.all(np.isfinite(p))]
        if len(good) >= 3:
            cen[lv] = np.mean(good, axis=0)
            present.append(lv)
    if len(present) < 3:
        return None
    top, bot = cen[present[0]], cen[present[-1]]
    chord = bot - top
    L = float(np.linalg.norm(chord))
    if L < 1e-6:
        return None
    # Signed perpendicular offset. +x is ANTERIOR: buu.py mirrors BUU into the DRR
    # convention on load, so both sources reach here anterior-right.
    nrm = np.array([chord[1], -chord[0]]) / L
    off = {lv: float(np.dot(cen[lv] - top, nrm)) for lv in present}
    if np.mean(list(off.values())) < 0:      # normal points posterior -> flip it
        off = {k: -v for k, v in off.items()}
    apex = max(off, key=off.get)
    # PROMINENCE, relative to the chord. On a straight lumbar spine every centre lies ON
    # the chord, every offset is ~0, and argmax then returns whichever level rounding
    # error favoured -- a completely arbitrary answer delivered with total confidence.
    # That is not a synthetic edge case: a spine with no apex is a FLAT BACK, which is
    # precisely Roussouly type 2, so the degenerate input is the clinically common one.
    # Returning the prominence lets the caller distinguish "the apex is at L4" from
    # "there is no apex", which are different findings.
    return apex, float(off[apex] / L)


def roussouly_type(ss: float, corners: Optional[Dict] = None) -> Optional[str]:
    """Roussouly morphotype 1-4, or None when it cannot be determined.

    SS carries most of it -- <35 is type 1 or 2, 35-45 is type 3, >45 is type 4 -- but
    NOT the 1/2 split, which is a shape distinction: type 1 is a short sharp lordosis low
    in the lumbar spine with a thoracolumbar kyphosis above it, type 2 is a globally flat
    back. Both have a low SS, so ostk.metrics.roussouly_type_from_ss honestly refuses to
    choose and returns "1-2".

    To produce a genuine 4x4 the apex is used: a lumbar spine with no measurable apex is
    flat, which is type 2; otherwise apex at L4 or below -> short low lordosis -> type 1,
    apex above L4 -> long shallow curve -> type 2.

    That rule is a DEFENSIBLE APPROXIMATION AND NOT THE PUBLISHED DEFINITION, which also
    weighs the thoracolumbar junction and the number of vertebrae in the lordosis. It is
    applied identically to the prediction and to the ground truth, so the confusion
    matrix measures the model rather than the rule -- but a 1-vs-2 disagreement should be
    read as "these two are hard to separate", not as a clinical claim. Passing no corners
    collapses 1 and 2 to None instead of guessing.
    """
    if not np.isfinite(ss):
        return None
    if ss > 45.0:
        return "4"
    if ss >= 35.0:
        return "3"
    if corners is None:
        return None
    got = lordosis_apex_level(corners)
    if got is None:
        return None
    apex, prominence = got
    # No measurable apex IS the type-2 finding -- a flat back -- rather than a missing
    # measurement. Without this the argmax over ~zero offsets returns an arbitrary level
    # and half of those flat backs would be labelled type 1.
    if prominence < FLAT_PROMINENCE:
        return "2"
    return "1" if apex in ("L4", "L5") else "2"


def confusion(true_lab: Sequence[Optional[str]], pred_lab: Sequence[Optional[str]],
              classes: Sequence[str] = ROUSSOULY_TYPES) -> Tuple[np.ndarray, int]:
    """Confusion matrix over `classes`, plus the count DROPPED because either side was
    unclassifiable. The drop count is returned rather than logged so a caller cannot
    quietly report a matrix over a filtered subset as if it covered everything."""
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    dropped = 0
    for t, p in zip(true_lab, pred_lab):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
        else:
            dropped += 1
    return M, dropped


def detection_prf(pred: Dict, true: Dict, names: Sequence[str],
                  thresh_px: float = 5.0) -> Dict:
    """Detection F1 at a pixel threshold, which the identity matrix deliberately excludes.

    TP  a landmark that exists and is predicted within `thresh_px`
    FP  predicted where there is no ground truth, OR predicted too far from it
    FN  exists but not predicted (the channel peak fell under the confidence floor)

    A landmark model can fail two distinct ways -- put a point in the wrong place, or
    decline to put one anywhere -- and a mean error over detections only sees the first.
    """
    tp = fp = fn = 0
    for n in names:
        t, p = true.get(n), pred.get(n)
        t_ok = t is not None and np.all(np.isfinite(t))
        p_ok = p is not None and np.all(np.isfinite(p))
        if t_ok and p_ok:
            d = float(np.linalg.norm(np.asarray(p, float) - np.asarray(t, float)))
            if d <= thresh_px:
                tp += 1
            else:
                fp += 1
                fn += 1          # both: nothing was found here AND a point was invented
        elif p_ok:
            fp += 1
        elif t_ok:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) \
        and (prec + rec) > 0 else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "thresh_px": thresh_px}
