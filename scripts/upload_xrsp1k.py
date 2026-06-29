"""Merge per-shard manifests and upload XRSpinoPelvic1K to the Hugging Face Hub.

Run after the gen_xrsp1k.slurm array finishes. Merges manifest_shard*.csv -> manifest.csv,
then pushes the dataset folder to <repo_id>@<revision>.

  HF_TOKEN=hf_xxx python scripts/upload_xrsp1k.py \
      --out_dir data/xrsp1k --repo_id <org>/XRSpinoPelvic1K --revision main

The token is read from $HF_TOKEN (never hard-code it). Pixel masks/DRRs (.npy/.png) and
per-level landmark JSON are uploaded; per-shard manifests and caches are skipped.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path


def merge_manifests(out_dir: Path) -> int:
    """Combine manifest_shard*.csv into one manifest.csv. If a single-shard run already
    wrote manifest.csv (no shard files), leave it. Returns the row count."""
    shards = sorted(glob.glob(str(out_dir / "manifest_shard*.csv")))
    if not shards:
        man = out_dir / "manifest.csv"
        return sum(1 for _ in open(man)) - 1 if man.exists() else 0
    rows = []
    for s in shards:
        with open(s, newline="") as f:
            rows += list(csv.DictReader(f))
    rows.sort(key=lambda r: (r.get("case", ""), r.get("view", "")))
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "view", "n_levels", "drr"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True, help="dataset folder (gen output)")
    ap.add_argument("--repo_id", required=True, help="e.g. anonymous-mlhc/XRSpinoPelvic1K")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    n = merge_manifests(a.out_dir)
    n_cases = len({p.parent.name for p in a.out_dir.glob("*/lateral_levels.json")})
    print(f"[upload] merged manifest: {n} rows across {n_cases} cases")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the environment (do not hard-code it)")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="dataset", private=a.private, exist_ok=True)
    if a.revision != "main":
        try:
            api.create_branch(a.repo_id, repo_type="dataset", branch=a.revision, exist_ok=True)
        except Exception as exc:                              # noqa: BLE001
            print(f"[upload] branch note: {exc}")

    print(f"[upload] pushing {a.out_dir} -> {a.repo_id}@{a.revision}")
    api.upload_folder(
        repo_id=a.repo_id, repo_type="dataset", folder_path=str(a.out_dir),
        revision=a.revision, commit_message="XRSpinoPelvic1K (DRR + dense masks + landmarks) from CTSpinoPelvic1K v4",
        ignore_patterns=["manifest_shard*.csv", "**/__pycache__/**", ".cache/**", "logs/**"],
    )
    print(f"[upload] done -> https://huggingface.co/datasets/{a.repo_id}/tree/{a.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
