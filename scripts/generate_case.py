#!/usr/bin/env python3
"""Generate DRR views + projected landmarks for ONE case. The Nextflow unit of work.

    python scripts/generate_case.py --ct <ct.nii.gz> --label <label.nii.gz> --out <dir>

One case per process so generation parallelises across the cluster and, more usefully,
so failures are per-case: `nextflow -resume` re-runs only what failed instead of the whole
stage. Exits non-zero if the case produced nothing, which is what lets Nextflow see it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ostk_path", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    from xrsp.build_dataset import build_case_oblique

    case = os.path.basename(a.ct).replace("_ct.nii.gz", "").replace(".nii.gz", "")
    seed = a.seed if a.seed is not None else (abs(hash(case)) % 100000)
    os.makedirs(a.out, exist_ok=True)
    build_case_oblique(a.ct, a.label, a.out, n_views=a.n_views, seed=seed,
                       pixel_spacing_mm=a.spacing, ostk_path=a.ostk_path,
                       n_workers=a.workers)
    made = [f for f in os.listdir(a.out) if f.endswith("_corners.json")]
    if not made:
        sys.exit(f"{case}: produced no views")
    print(f"{case}: {len(made)} view(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
