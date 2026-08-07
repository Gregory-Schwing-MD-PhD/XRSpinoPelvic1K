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
            pp, _ = soft_argmax(pred)
            tp, _ = soft_argmax(hm)
            for b in range(img.shape[0]):
                P, T = {}, {}
                for c, n in enumerate(names):
                    if not bool(valid[b, c]):
                        continue                     # unannotated -> not a miss
                    T[n] = [float(tp[b, c, 0]), float(tp[b, c, 1])]
                    P[n] = [float(pp[b, c, 0]), float(pp[b, c, 1])]
                have_meta = isinstance(meta, (list, tuple)) and b < len(meta)
                m = dict(meta[b]) if have_meta else {}
                if idx < len(rows):
                    m.setdefault("case", rows[idx].get("case", ""))
                    m.setdefault("view", rows[idx].get("view", m.get("view", "")))
                idx += 1
                out.append((P, T, m))
    return out


def _evaluate(items, names, levels, tag, out_dir, do_params):
    """Metrics + figures for one source. Returns the summary dict."""
    from xrsp import evalmetrics as EM
    from xrsp import evalplots as EP

    err_all, err_by_level, err_hip = [], {lv: [] for lv in levels}, []
    conf4 = np.zeros((4, 4), int)
    det = {"tp": 0, "fp": 0, "fn": 0}
    rows = []
    for P, T, meta in items:
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
            row["roussouly_true"] = EM.roussouly_type(
                gt.get("SS") if gt.get("SS") is not None else float("nan"), T)
            row["roussouly_pred"] = EM.roussouly_type(
                pr.get("SS") if pr.get("SS") is not None else float("nan"), P)
        rows.append(row)

    prec = det["tp"] / (det["tp"] + det["fp"]) if det["tp"] + det["fp"] else float("nan")
    rec = det["tp"] / (det["tp"] + det["fn"]) if det["tp"] + det["fn"] else float("nan")
    summary = {
        "source": tag, "n_items": len(items),
        "corner_error_px": EM.error_summary(err_all),
        "hip_error_px": EM.error_summary(err_hip),
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
        for k in ("PI", "SS", "PT", "LL"):
            t = [r[f"{k}_true"] for r in rows if r.get(f"{k}_true") is not None
                 and r.get(f"{k}_pred") is not None]
            p = [r[f"{k}_pred"] for r in rows if r.get(f"{k}_true") is not None
                 and r.get(f"{k}_pred") is not None]
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
        EP.plot_identity_residual([r.get("PI_pred") for r in rows],
                                  [r.get("SS_pred") for r in rows],
                                  [r.get("PT_pred") for r in rows],
                                  out_dir, stem=f"fig_pi_identity_{tag}")

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
                                         names, levels, "drr", a.out, do_params=True)
    if a.buu:
        rows = [r for r in index_buu(a.buu) if r["case"] in buu_test] if buu_test else []
        if rows:
            ds = BUULandmarkDataset(rows, levels=levels, out_size=size, sigma=2.0,
                                    augment=False, p_flip=0.0, max_rot_deg=0.0)
            # do_params=False: no hip ground truth on a BUU film, so PI/PT cannot exist.
            summaries["buu"] = _evaluate(_predict(net, ds, dev, names, rows, a.batch),
                                         names, levels, "buu", a.out, do_params=False)

    if not summaries:
        sys.exit("no held-out items were evaluated -- check --drr/--buu point at the "
                 "same data the run_config.json test lists came from.")
    json.dump({"model": os.path.abspath(a.model), "levels": levels, **summaries},
              open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=str)
    for tag, s in summaries.items():
        print(f"\n[{tag}] n={s['n_items']}  corner median "
              f"{s['corner_error_px']['median']:.2f}px  det-F1 {s['detection']['f1']:.3f}"
              f"  corner-identity macro-F1 {s['corner_identity']['macro_f1']:.3f}")
        if "parameters" in s:
            for k, v in s["parameters"].items():
                print(f"    {k:3s} ICC {v['icc']:.3f}  MAE {v['mae']:.2f}deg  "
                      f"<=5deg {100*v['within_5deg']:.0f}%")
            print(f"    Roussouly macro-F1 {s['roussouly']['macro_f1']:.3f}  "
                  f"kappa {s['roussouly']['kappa']:.3f}")
    print(f"\nfigures + summary.json -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
