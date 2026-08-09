#!/usr/bin/env python3
"""Held-out evaluation of the unified landmark model, with publication figures.

    python scripts/evaluate_unified.py --model runs/unified/best.pt \
        --drr data/xrsp1k --buu data/BUU-LSPINE --out results/unified

THE TEST SET IS READ, NOT RE-DERIVED
------------------------------------
train_unified.py writes the exact held-out case lists into run_config.json. Those are
used verbatim. Re-deriving the split here from the same seed would LOOK equivalent and
would silently stop being so the moment a file was added, removed or re-extracted --
which is the failure that turns a test score into a training score without anything
looking wrong. If run_config.json is missing this refuses to run rather than guessing.

TWO SETS, DIFFERENT QUESTIONS, NEVER POOLED
-------------------------------------------
  DRR test   amodal ground truth from the 3-D fit, and the ONLY set carrying BOTH the
             corners and the hip point on the same image -- so the only one on which
             PI and PT can be evaluated end to end. It does not measure the domain: same
             renderer, same physics. Read it as in-silico.
  BUU test   real standing radiographs. Corners only -- there is no hip ground truth on
             a BUU film, so PI and PT are unobtainable here by construction, not by
             oversight. This is the deployment-domain number for corner accuracy.

A single pooled metric over both would answer no question anyone has.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _levels_of(names):
    return sorted({n.split(".")[0] for n in names if "." in n},
                  key=lambda lv: ([f"C{i}" for i in range(1, 8)]
                                  + [f"T{i}" for i in range(1, 14)]
                                  + [f"L{i}" for i in range(1, 7)] + ["S1"]).index(lv)
                  if lv in ([f"C{i}" for i in range(1, 8)]
                            + [f"T{i}" for i in range(1, 14)]
                            + [f"L{i}" for i in range(1, 7)] + ["S1"]) else 999)


def _params_from_points(pts, levels):
    """PI/SS/PT/LL from a {channel: [x, y]} dict, via ostk's 2-D summary.

    Each level's SUPERIOR endplate line is (sup_ant, sup_post). Levels missing either
    corner are omitted rather than interpolated: a fabricated endplate would produce a
    confident angle from nothing.
    """
    from ostk.metrics2d import spinopelvic_summary_2d
    eps = {}
    for lv in levels:
        a, p = pts.get(f"{lv}.sup_ant"), pts.get(f"{lv}.sup_post")
        if a is not None and p is not None and np.all(np.isfinite(a)) \
                and np.all(np.isfinite(p)):
            eps[lv] = (np.asarray(a, float), np.asarray(p, float))
    fem = pts.get("bicoxofemoral")
    fem = np.asarray(fem, float) if fem is not None and np.all(np.isfinite(fem)) else None
    if not eps:
        return {"PI": None, "SS": None, "PT": None, "LL": None}
    try:
        return spinopelvic_summary_2d(eps, fem)
    except Exception:                                        # noqa: BLE001
        return {"PI": None, "SS": None, "PT": None, "LL": None}


LUMBAR_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "S1"]


def _ll_robust(pts, levels):
    """Lumbar lordosis from a least-squares fit over EVERY available endplate.

    The textbook LL is the Cobb angle between the most cranial and most caudal lumbar
    endplates -- so it rests on exactly two measurements and inherits both their errors in
    full. On this model those two are the WORST two: per-level median radial error is
    1.15-1.20 px at L3/L4 but 1.96 px at S1 and 1.35 px at L1, and LL is measured between
    S1 and L1. That is why LL fails (>10 deg on 8.3% of films) where SS fails on 3.0%.

    Endplate inclination varies smoothly and near-linearly down the lumbar spine, so a
    line through (level index, endplate angle) estimates the two end angles using every
    endplate rather than two.

    The slope is THEIL-SEN -- the median of all pairwise slopes -- not least squares.
    Least squares is not robust here and testing showed it is WORSE than the two-point
    Cobb: the endpoints carry the highest leverage, so wrecking the S1 endplate by 40 deg
    moved an LS estimate from 41 to 75 deg, amplifying exactly the error it was meant to
    absorb. Theil-Sen has a 29% breakdown point and the intercept is taken as the median
    residual, so one bad endplate is outvoted instead of steering the fit.

    Returns None below 4 endplates -- with 3 the fit has almost no redundancy and would
    simply launder the same noise through a regression.
    """
    # Index by POSITION AMONG THE LEVELS PRESENT, not by position in LUMBAR_ORDER. L6 is
    # absent in most spines, and a fixed ordering leaves a gap there: L5->S1 then spans two
    # index units for one anatomical level, which flattens the fitted slope and stretches
    # the span. On a clean synthetic lordosis that turned a true 40 deg into 48.
    ang, idx = [], []
    present = [lv for lv in LUMBAR_ORDER if lv in levels]
    for i, lv in enumerate(present):
        a, q = pts.get(f"{lv}.sup_ant"), pts.get(f"{lv}.sup_post")
        if a is None or q is None or not (np.all(np.isfinite(a)) and np.all(np.isfinite(q))):
            continue
        d = np.asarray(a, float) - np.asarray(q, float)
        ang.append(float(np.degrees(np.arctan2(d[1], d[0]))))
        idx.append(i)
    if len(ang) < 4:
        return None
    ang = np.unwrap(np.radians(ang))          # no 180 deg wrap mid-column
    x = np.asarray(idx, float)
    slopes = [(ang[j] - ang[i]) / (x[j] - x[i])
              for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    if not slopes:
        return None
    m = float(np.median(slopes))
    # The span is what LL is: slope x (levels between the ends). The intercept cancels,
    # so only the robust slope matters.
    return float(abs(np.degrees(m * (x.max() - x.min()))))


def _predict(net, ds, dev, names, rows, batch=8):
    """Run the model over a dataset, returning per-item (pred_pts, true_pts).

    Ground truth is recovered by arg-maxing the TARGET heatmaps rather than re-reading
    the source JSON, so prediction and truth pass through identical resize, augmentation
    and coordinate handling. Comparing a decoded prediction against a differently-derived
    truth would fold every pipeline discrepancy into the reported error.
    """
    import torch
    from torch.utils.data import DataLoader

    from xrsp.dataset import collate
    from xrsp.heatmaps import soft_argmax

    # `case` is taken from ROWS, by position, not from meta. The DRR meta is the
    # <view>_corners.json, which carries "view" but no "case" -- so keying the per-item
    # CSV on meta alone would silently blank the one column you need to trace an outlier
    # back to a scan. shuffle=False, so position is exact.
    out, idx = [], 0
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for img, hm, valid, meta in dl:
            pred = net(img.to(dev)).cpu()
            pp, pc = soft_argmax(pred)
            tp, _ = soft_argmax(hm)
            for b in range(img.shape[0]):
                P, T, C = {}, {}, {}
                for c, n in enumerate(names):
                    if not bool(valid[b, c]):
                        continue                     # unannotated -> not a miss
                    T[n] = [float(tp[b, c, 0]), float(tp[b, c, 1])]
                    P[n] = [float(pp[b, c, 0]), float(pp[b, c, 1])]
                    # Peak height BEFORE normalisation. The only per-landmark signal the
                    # network offers about its own certainty, and it was being thrown
                    # away -- which left no way to tell a confident-and-wrong prediction
                    # (SS 23.8 -> 71.9) from a confident-and-right one.
                    C[n] = float(pc[b, c])
                have_meta = isinstance(meta, (list, tuple)) and b < len(meta)
                m = dict(meta[b]) if have_meta else {}
                if idx < len(rows):
                    m.setdefault("case", rows[idx].get("case", ""))
                    m.setdefault("view", rows[idx].get("view", m.get("view", "")))
                idx += 1
                out.append((P, T, C, m))
    return out


def _evaluate(items, names, levels, tag, out_dir, do_params, corners_trained=True):
    """Metrics + figures for one source.

    `corners_trained` says whether the corner channels were SUPERVISED in this source's
    domain. They are not on the DRRs: unified.py gives corners to BUU and the hip point to
    the DRRs, on disjoint channels. So DRR corner error measures channels that have never
    seen a DRR -- 85 px median, detection F1 0.22 -- and every angle derived from them is
    an artefact, not a result. The flag marks them in the summary so nobody reads those
    numbers as performance, and switches on the hybrid below.
    """
    from xrsp import evalmetrics as EM
    from xrsp import evalplots as EP

    err_all, err_by_level, err_hip = [], {lv: [] for lv in levels}, []
    conf4 = np.zeros((4, 4), int)
    det = {"tp": 0, "fp": 0, "fn": 0}
    rows, lm_rows = [], []
    for P, T, CONF, meta in items:
        for n in names:
            t, p = T.get(n), P.get(n)
            if t is None:
                continue
            d = (float(np.linalg.norm(np.asarray(p, float) - np.asarray(t, float)))
                 if p is not None and np.all(np.isfinite(p)) else float("nan"))
            if n == "bicoxofemoral":
                err_hip.append(d)
            else:
                err_all.append(d)
                lv = n.split(".")[0]
                if lv in err_by_level:
                    err_by_level[lv].append(d)
        conf4 += EM.corner_identity_confusion(P, T, levels)
        d1 = EM.detection_prf(P, T, names)
        for k in det:
            det[k] += d1[k]
        row = {"case": str(meta.get("case", "")), "view": str(meta.get("view", ""))}
        if do_params:
            gt = _params_from_points(T, levels)
            pr = _params_from_points(P, levels)
            for k in ("PI", "SS", "PT", "LL"):
                row[f"{k}_true"], row[f"{k}_pred"] = gt.get(k), pr.get(k)
            # Reported ALONGSIDE the textbook LL, never instead of it: the two-endplate
            # Cobb is what the literature reports and what a reader would measure, so
            # replacing it silently would make the number incomparable.
            row["LLrobust_true"] = _ll_robust(T, levels)
            row["LLrobust_pred"] = _ll_robust(P, levels)
            row["roussouly_true"] = EM.roussouly_type(
                gt.get("SS") if gt.get("SS") is not None else float("nan"), T)
            row["roussouly_pred"] = EM.roussouly_type(
                pr.get("SS") if pr.get("SS") is not None else float("nan"), P)
            if not corners_trained:
                # PI and PT need the S1 endplate AND the hip axis on the SAME image, and
                # no image has both a trained corner channel and hip ground truth. So they
                # cannot be evaluated end to end at all under the disjoint-channel design.
                # This substitutes GROUND-TRUTH corners and keeps the PREDICTED hip, which
                # isolates exactly one thing: how many degrees of PI/PT the hip channel's
                # error costs. It is not an end-to-end number and is named so it cannot be
                # mistaken for one.
                mix = {k: v for k, v in T.items() if k != "bicoxofemoral"}
                if P.get("bicoxofemoral") is not None:
                    mix["bicoxofemoral"] = P["bicoxofemoral"]
                hy = _params_from_points(mix, levels)
                row["PI_gtcorners_predhip"] = hy.get("PI")
                row["PT_gtcorners_predhip"] = hy.get("PT")
        rows.append(row)
        for n in names:
            t, p = T.get(n), P.get(n)
            if t is None:
                continue
            d = (float(np.linalg.norm(np.asarray(p, float) - np.asarray(t, float)))
                 if p is not None and np.all(np.isfinite(p)) else float("nan"))
            lm_rows.append({"case": row["case"], "channel": n,
                            "level": n.split(".")[0], "err_px": d,
                            "confidence": CONF.get(n, float("nan"))})

    prec = det["tp"] / (det["tp"] + det["fp"]) if det["tp"] + det["fp"] else float("nan")
    rec = det["tp"] / (det["tp"] + det["fn"]) if det["tp"] + det["fn"] else float("nan")
    summary = {
        "source": tag, "n_items": len(items),
        # Load-bearing flag, not a note: when false, every corner and angle metric below
        # describes untrained channels and must not be reported as performance.
        "corners_trained_in_this_domain": bool(corners_trained),
        "corner_error_px": EM.error_summary(err_all),
        # ED-threshold accuracy: the statistic the keypoint literature reports, so these
        # numbers can sit beside Bansal et al. 2026 without hand-waving. See the caveats
        # in evalmetrics.ed_accuracy -- their pixels are not ours.
        "corner_ed_accuracy": EM.ed_accuracy(err_all),
        "hip_error_px": EM.error_summary(err_hip),
        "hip_ed_accuracy": EM.ed_accuracy(err_hip),
        "detection": {**det, "precision": prec, "recall": rec,
                      "f1": (2 * prec * rec / (prec + rec)
                             if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0
                             else float("nan"))},
        "corner_identity": {"matrix": conf4.tolist(), "classes": list(EM.CORNERS),
                            **EM.prf_from_confusion(conf4)},
    }

    EP.plot_ced({f"corners ({tag})": EM.ced_curve(err_all),
                 **({f"hip ({tag})": EM.ced_curve(err_hip)}
                    if any(np.isfinite(err_hip)) else {})},
                out_dir, stem=f"fig_ced_{tag}")
    EP.plot_error_by_level(err_by_level, out_dir, stem=f"fig_error_by_level_{tag}")
    EP.plot_confusion(conf4, list(EM.CORNERS), out_dir, stem=f"fig_corner_identity_{tag}",
                      title=f"corner identity — {tag}",
                      prf=EM.prf_from_confusion(conf4))

    if do_params:
        pairs, stats, ba = {}, {}, {}
        keys = ["PI", "SS", "PT", "LL", "LLrobust"]
        if not corners_trained:
            keys += ["PI_gtcorners_predhip", "PT_gtcorners_predhip"]
        for k in keys:
            base = ("PI" if k.startswith("PI")
                    else "PT" if k.startswith("PT") else k)
            t = [r[f"{base}_true"] for r in rows if r.get(f"{base}_true") is not None
                 and r.get(f"{k}_pred" if k == base else k) is not None]
            p = [r[f"{k}_pred" if k == base else k] for r in rows
                 if r.get(f"{base}_true") is not None
                 and r.get(f"{k}_pred" if k == base else k) is not None]
            if len(t) < 2:
                continue
            pairs[k] = (t, p)
            d = np.abs(np.asarray(p, float) - np.asarray(t, float))
            stats[k] = {"n": len(t), "icc": EM.icc21(t, p),
                        "pearson_r": float(np.corrcoef(t, p)[0, 1]) if len(t) > 1
                        else float("nan"),
                        "mae": float(np.nanmean(d)),
                        "within_5deg": float(np.nanmean(d <= 5)),
                        "within_10deg": float(np.nanmean(d <= 10))}
            ba[k] = EM.bland_altman(t, p)
        summary["parameters"] = stats
        summary["bland_altman"] = {k: {kk: vv for kk, vv in v.items()
                                       if kk not in ("mean", "diff")}
                                   for k, v in ba.items()}
        if ba:
            EP.plot_bland_altman(ba, out_dir, stem=f"fig_bland_altman_{tag}")
        if pairs:
            EP.plot_scatter_agreement(pairs, stats, out_dir, stem=f"fig_agreement_{tag}")

        M, dropped = EM.confusion([r.get("roussouly_true") for r in rows],
                                  [r.get("roussouly_pred") for r in rows])
        summary["roussouly"] = {"matrix": M.tolist(), "classes": list(EM.ROUSSOULY_TYPES),
                                "unclassifiable_excluded": dropped,
                                **EM.prf_from_confusion(M)}
        EP.plot_confusion(M, list(EM.ROUSSOULY_TYPES), out_dir,
                          stem=f"fig_roussouly_{tag}", title=f"Roussouly type — {tag}",
                          prf=EM.prf_from_confusion(M), dropped=dropped)
        # PI = SS + PT needs both, and on a source with no hip ground truth PI/PT are
        # None for every row -- the figure would be an empty axis presented as a result.
        if any(r.get("PI_pred") is not None for r in rows):
            EP.plot_identity_residual([r.get("PI_pred") for r in rows],
                                      [r.get("SS_pred") for r in rows],
                                      [r.get("PT_pred") for r in rows],
                                      out_dir, stem=f"fig_pi_identity_{tag}")

    if lm_rows:
        with open(os.path.join(out_dir, f"per_landmark_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "channel", "level", "err_px",
                                              "confidence"])
            w.writeheader()
            w.writerows(lm_rows)
    if rows:
        with open(os.path.join(out_dir, f"per_item_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="runs/unified/best.pt")
    ap.add_argument("--drr", default=None)
    ap.add_argument("--buu", default=None)
    ap.add_argument("--buu_splits", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)

    import torch

    from xrsp.buu import BUULandmarkDataset, index_buu
    from xrsp.dataset import LandmarkDRRDataset, index_views
    from xrsp.model import build_unet

    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    names, size = list(ck["names"]), tuple(ck["size"])
    levels = _levels_of(names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_unet(len(names), features=tuple(ck.get("features",
                                                       (16, 32, 64, 128, 256)))).to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(a.model)), "run_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"{cfg_path} not found. It carries the exact held-out case lists; "
                 f"re-deriving the split here would look equivalent and quietly stop "
                 f"being so as soon as the file list changed. Evaluate a run that has "
                 f"one.")
    cfg = json.load(open(cfg_path))
    drr_test, buu_test = set(cfg.get("drr_test") or []), set(cfg.get("buu_test") or [])
    print(f"held-out: {len(drr_test)} DRR views, {len(buu_test)} BUU films")

    summaries = {}
    if a.drr and drr_test:
        rows = [r for r in index_views(a.drr) if r["case"] in drr_test]
        if rows:
            ds = LandmarkDRRDataset(rows, out_size=size, levels=levels, sigma=2.0,
                                    augment=False)
            # do_params=True: the DRRs are the only set with corners AND hip together.
            summaries["drr"] = _evaluate(_predict(net, ds, dev, names, rows, a.batch),
                                         names, levels, "drr", a.out, do_params=True,
                                         corners_trained=False)
    if a.buu:
        rows = [r for r in index_buu(a.buu) if r["case"] in buu_test] if buu_test else []
        if rows:
            ds = BUULandmarkDataset(rows, levels=levels, out_size=size, sigma=2.0,
                                    augment=False, p_flip=0.0, max_rot_deg=0.0)
            # do_params=False: no hip ground truth on a BUU film, so PI/PT cannot exist.
            summaries["buu"] = _evaluate(_predict(net, ds, dev, names, rows, a.batch),
                                         # SS and LL need ONLY corners, so both are
                                         # computable on real film. spinopelvic_summary_2d
                                         # returns None for PI/PT with no femoral point,
                                         # so enabling this cannot fabricate them.
                                         names, levels, "buu", a.out, do_params=True)

    # Training curves live beside the checkpoint, not with a source, so they are plotted
    # once rather than per source.
    hist = os.path.join(os.path.dirname(os.path.abspath(a.model)), "history.csv")
    if os.path.exists(hist):
        from xrsp import evalplots as EP
        EP.plot_training_curves(hist, a.out)
        print(f"training curves <- {hist}")
    else:
        print(f"no history.csv beside the checkpoint -- training curves skipped")

    if not summaries:
        sys.exit("no held-out items were evaluated -- check --drr/--buu point at the "
                 "same data the run_config.json test lists came from.")
    json.dump({"model": os.path.abspath(a.model), "levels": levels, **summaries},
              open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    for tag, s in summaries.items():
        trained = s.get("corners_trained_in_this_domain", True)
        print(f"\n[{tag}] n={s['n_items']}")
        if not trained:
            print("    !! CORNER CHANNELS WERE NOT TRAINED IN THIS DOMAIN.")
            print("       unified.py supervises corners from BUU and the hip point from")
            print("       the DRRs, on disjoint channels. Every corner and angle number")
            print("       below therefore measures untrained channels -- report NONE of")
            print("       them. The hip error and the *_gtcorners_predhip rows are valid.")
        print(f"    corner median {s['corner_error_px']['median']:.2f}px  "
              f"det-F1 {s['detection']['f1']:.3f}  "
              f"identity macro-F1 {s['corner_identity']['macro_f1']:.3f}"
              f"{'   <- not a result' if not trained else ''}")
        ed = s["corner_ed_accuracy"]
        print(f"    corners within  5px {100*ed['within_5px']:.1f}%  "
              f"10px {100*ed['within_10px']:.1f}%  15px {100*ed['within_15px']:.1f}%"
              + ("   [Bansal 2026: 75-79 / ~98 / ~100 on 640x640]" if trained else ""))
        he = s["hip_error_px"]
        if he["n"]:
            print(f"    hip point   median {he['median']:.2f}px  p95 {he['p95']:.2f}px"
                  f"   (trained on this domain -- VALID)")
        for k, v in (s.get("parameters") or {}).items():
            note = ""
            if k.endswith("_gtcorners_predhip"):
                note = "  <- GT corners + predicted hip: isolates the hip channel"
            elif not trained:
                note = "  <- artefact"
            print(f"    {k:22s} ICC {v['icc']:6.3f}  MAE {v['mae']:6.2f}deg  "
                  f"<=5deg {100*v['within_5deg']:3.0f}%{note}")
        if "roussouly" in s:
            print(f"    Roussouly macro-F1 {s['roussouly']['macro_f1']:.3f}  "
                  f"kappa {s['roussouly']['kappa']:.3f}"
                  f"{'  <- artefact' if not trained else ''}")
    print(f"\nfigures + summary.json -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
