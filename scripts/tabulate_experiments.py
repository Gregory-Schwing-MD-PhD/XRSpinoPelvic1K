#!/usr/bin/env python3
"""Evaluate every finished YOLO run on the SAME metrics and print one comparison table.

    python scripts/tabulate_experiments.py --runs data/runs/yolo_exp \
        --buu data/BUU-LSPINE --splits data/buu_splits.json --out results/experiments

One table, one metric implementation, one test set. Reading six separate summary.json
files by hand is how a sweep turns into a claim nobody can reproduce -- and Ultralytics'
own val numbers are OKS mAP, which is not what the heatmap model was scored on and not
what the paper's headline ED thresholds are either.

Runs are evaluated only when a weights/best.pt exists, so this can be run repeatedly
while the sweep is still going and simply reports more rows each time.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bansal et al. 2026 Table 6, test split, for the columns that are directly comparable.
PAPER = {
    "YOLOv8n-Pose (paper)":  dict(ed5=75.63, ed10=98.51, ed15=100.00),
    "YOLOv11n-Pose (paper)": dict(ed5=76.75, ed10=97.37, ed15=100.00),
    "Detectron2-R50 (paper)": dict(ed5=75.77, ed10=98.34, ed15=99.78),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_eval", action="store_true",
                    help="re-read existing summaries instead of re-running inference")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    for w in sorted(glob.glob(os.path.join(a.runs, "*", "weights", "best.pt"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(w)))
        odir = os.path.join(a.out, name)
        summ = os.path.join(odir, "summary.json")
        if not (a.skip_eval and os.path.exists(summ)):
            print(f"--- evaluating {name}")
            r = subprocess.run(
                [sys.executable, os.path.join(here, "evaluate_yolo.py"),
                 "--weights", w, "--buu", a.buu, "--splits", a.splits,
                 "--out", odir, "--render", "3"],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    FAILED: {r.stderr.strip().splitlines()[-1:]}")
                continue
        if not os.path.exists(summ):
            continue
        s = json.load(open(summ))
        m, mm = s["protocol_all_gt"], s["protocol_matched_only"]
        p = s.get("parameters", {})
        rows.append(dict(
            name=name,
            med=mm["corner_error_px"]["median"],
            ed5=100 * mm["corner_ed_accuracy"]["within_5px"],
            ed10=100 * mm["corner_ed_accuracy"]["within_10px"],
            ed15=100 * mm["corner_ed_accuracy"]["within_15px"],
            missed=m["n_missed"],
            idf1=s["corner_identity"]["macro_f1"],
            ss_icc=p.get("SS", {}).get("icc", float("nan")),
            ss_mae=p.get("SS", {}).get("mae", float("nan")),
            ss_5=100 * p.get("SS", {}).get("within_5deg", float("nan")),
            ll_mae=p.get("LL", {}).get("mae", float("nan")),
        ))

    if not rows:
        print("no finished runs yet")
        return 0
    rows.sort(key=lambda r: -r["ss_5"] if r["ss_5"] == r["ss_5"] else 0)

    hdr = (f"{'run':<18}{'med px':>8}{'ED5%':>7}{'ED10%':>7}{'ED15%':>7}"
           f"{'miss':>6}{'idF1':>7}{'SS ICC':>8}{'SS MAE':>8}{'SS<5':>7}{'LL MAE':>8}")
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in rows:
        print(f"{r['name']:<18}{r['med']:>8.2f}{r['ed5']:>7.1f}{r['ed10']:>7.1f}"
              f"{r['ed15']:>7.1f}{r['missed']:>6d}{r['idf1']:>7.3f}"
              f"{r['ss_icc']:>8.3f}{r['ss_mae']:>8.2f}{r['ss_5']:>7.0f}{r['ll_mae']:>8.2f}")
    print("-" * len(hdr))
    # The paper's ED columns are on 640x640 SQUASHED images -- a different pixel. Printed
    # for orientation, never as a like-for-like row.
    for nm, v in PAPER.items():
        print(f"{nm:<18}{'--':>8}{v['ed5']:>7.2f}{v['ed10']:>7.2f}{v['ed15']:>7.2f}"
              f"{'--':>6}{'--':>7}{'--':>8}{'--':>8}{'--':>7}{'--':>8}")
    print("=" * len(hdr))
    print("paper rows are ED on 640x640 NON-UNIFORMLY resized images; ours are original")
    print("film pixels (median diagonal ~3167 px). Not the same unit -- see summary.json")
    print("for error normalised by image diagonal, which is comparable.")

    json.dump(rows, open(os.path.join(a.out, "table.json"), "w"), indent=2)
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write("| run | med px | ED5% | ED10% | ED15% | missed | id F1 | SS ICC | "
                "SS MAE | SS<5deg | LL MAE |\n")
        f.write("|" + "---|" * 11 + "\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['med']:.2f} | {r['ed5']:.1f} | {r['ed10']:.1f} "
                    f"| {r['ed15']:.1f} | {r['missed']} | {r['idf1']:.3f} | "
                    f"{r['ss_icc']:.3f} | {r['ss_mae']:.2f} | {r['ss_5']:.0f} | "
                    f"{r['ll_mae']:.2f} |\n")
    print(f"\ntable -> {os.path.join(a.out, 'table.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
