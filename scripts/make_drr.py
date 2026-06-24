"""Render a DRR (+ mask + per-level landmarks) for a single CT/seg pair.

Usage:
  python scripts/make_drr.py CT.nii.gz LABEL.nii.gz OUT_DIR [--views lateral ap]
"""
import argparse

from xrsp.build_dataset import build_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ct")
    p.add_argument("label")
    p.add_argument("out_dir")
    p.add_argument("--views", nargs="+", default=["lateral", "ap"])
    p.add_argument("--gamma", type=float, default=0.5)
    a = p.parse_args()
    rows = build_case(a.ct, a.label, a.out_dir, tuple(a.views), a.gamma)
    for r in rows:
        print(f"{r['view']:8s} levels={r['n_levels']}  -> {r['drr']}")


if __name__ == "__main__":
    main()
