#!/usr/bin/env python3
"""Build the 5-fold splits, ONCE, from a case-level manifest.

    python scripts/make_splits.py --manifest data/cases.csv --out data/splits.json

The manifest needs `case_id` and `patient_id`; `lstv_label` / `has_l6` /
`femoral_heads_visible` are used for stratification when present.

Write once and commit the result. Regenerating a split silently invalidates every
number previously reported against it, so this refuses to overwrite unless --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xrsp import splits as S                                   # noqa: E402


def _read(path):
    if path.endswith(".json"):
        d = json.load(open(path))
        return d if isinstance(d, list) else d.get("records") or d.get("cases") or []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("has_l6", "femoral_heads_visible"):
            if k in r:
                r[k] = str(r[k]).strip().lower() in ("1", "true", "yes")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV or JSON of case records")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_folds", type=int, default=S.N_FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits file")
    a = ap.parse_args(argv)

    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists. A resplit invalidates every result reported against "
                 f"the old one — pass --force only if you mean it.")
    recs = _read(a.manifest)
    if not recs:
        sys.exit(f"no records in {a.manifest}")
    folds = S.build_folds(recs, n_folds=a.n_folds, seed=a.seed)   # self-checks leakage
    S.save(folds, a.out)
    print(S.summarise(folds, recs))
    print(f"\nwrote {a.out}  ({a.n_folds} folds, seed {a.seed})")
    print("patient-grouped and leakage-checked. Commit this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
