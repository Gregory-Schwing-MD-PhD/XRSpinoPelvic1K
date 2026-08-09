#!/usr/bin/env python3
"""Seed the femoral-head annotation ledger, and upload the films it refers to.

    python annot/seed_cases.py --buu data/BUU-LSPINE --splits data/buu_splits.json \
        --annot-repo <org>/xrsp-femhead-annot --image-repo <org>/xrsp-femhead-images \
        --n 120 --apply

WHICH FILMS
-----------
--split all (the default) seeds EVERY film. --split test seeds only the held-out ones.

Annotate everything. At two clicks per film with prefetch and keyboard shortcuts, 2000
films is roughly 4.4 h per reader (~8.9 person-hours double-annotated, ~2.2 h each across
four people). That is a small price for what it changes:

  test only (301)  -> a reference set. The hip point stays SYNTHETIC, supervised by DRRs,
                      and every claim about it rests on transfer from a renderer.
  everything (2000)-> 1398 train / 301 val / 301 test of REAL hip labels. The hip point
                      stops being synthetic at all, the domain-transfer question
                      disappears rather than being measured, and BUU+femoral-heads
                      becomes a public resource that does not currently exist.

The split assignment is carried into each case from buu_splits.json, so train films can
never be mistaken for test ones later however the annotations are used.

STILL DO A PILOT FIRST. --n 120 --split test is not an alternative to annotating
everything, it is the first hour of it: it calibrates inter-rater agreement and finds
tooling problems while they cost an hour instead of nine. Seeding is incremental -- run
it again without --n to add the rest, and existing cases are left untouched.

Random sampling under a recorded seed when --n is given, because any human selection
biases the reference: picking "clear" films measures the model on easy cases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buu", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--annot-repo", required=True)
    ap.add_argument("--image-repo", required=True)
    ap.add_argument("--n", type=int, default=0,
                    help="sample only N films (0 = all). Use for the pilot.")
    ap.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true",
                    help="actually create repos and upload (default is a dry run)")
    a = ap.parse_args(argv)

    from xrsp.buu import index_buu

    assign = json.load(open(a.splits))["assignments"]
    pool = [r for r in index_buu(a.buu)
            if a.split == "all" or assign.get(r["case"]) == a.split]
    if a.n:
        rng = random.Random(a.seed)
        pick = sorted(rng.sample(pool, min(a.n, len(pool))), key=lambda r: r["case"])
    else:
        pick = sorted(pool, key=lambda r: r["case"])
    import collections
    by = collections.Counter(assign.get(r["case"], "?") for r in pick)
    print(f"pool ({a.split}) {len(pool)} -> seeding {len(pick)}   splits: {dict(by)}")
    for r in pick[:5]:
        print("   ", r["case"])
    print("    ...")

    if not a.apply:
        print("\nDRY RUN — nothing created. Re-run with --apply.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN must be set (write access to both repos)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    for repo in (a.annot_repo, a.image_repo):
        api.create_repo(repo, repo_type="dataset", private=a.private, exist_ok=True)
        print(f"  repo ready: {repo}")

    # Incremental: never overwrite a case that already exists, or a pilot's completed
    # annotations would be wiped when the rest of the set is seeded on top of it.
    existing = set()
    try:
        for f in api.list_repo_files(a.annot_repo, repo_type="dataset"):
            if f.startswith("cases/") and f.endswith(".json"):
                existing.add(os.path.basename(f)[:-5])
    except Exception:                                        # noqa: BLE001
        pass
    have_img = set()
    try:
        for f in api.list_repo_files(a.image_repo, repo_type="dataset"):
            if f.startswith("images/"):
                have_img.add(os.path.splitext(os.path.basename(f))[0])
    except Exception:                                        # noqa: BLE001
        pass
    new = [r for r in pick if r["case"] not in existing]
    print(f"  {len(existing)} cases already seeded; adding {len(new)}")

    # Images go to their OWN repo, and the Space streams them through itself, so
    # annotators need no read access to any dataset -- the same containment the review
    # service uses for labels. Only films not already uploaded are sent: at 2000 films
    # a second pass would otherwise re-push several GB to no effect.
    todo = [r for r in pick if r["case"] not in have_img]
    for i, r in enumerate(todo, 1):
        api.upload_file(path_or_fileobj=r["img"], path_in_repo=f"images/{r['case']}.jpg",
                        repo_id=a.image_repo, repo_type="dataset")
        if i % 200 == 0:
            print(f"    uploaded {i}/{len(todo)}", flush=True)
    print(f"  uploaded {len(todo)} new films to {a.image_repo} "
          f"({len(have_img)} already present)")
    ops = []
    from huggingface_hub import CommitOperationAdd
    for r in new:
        # The split travels WITH the case. Without it, a later analysis cannot tell a
        # training film from a held-out one and every number becomes ambiguous.
        case = {"case_id": r["case"], "source": "BUU-LSPINE",
                "split": assign.get(r["case"], "unknown"),
                "slots": {}, "seeded_seed": a.seed}
        ops.append(CommitOperationAdd(
            path_in_repo=f"cases/{r['case']}.json",
            path_or_fileobj=json.dumps(case, indent=2).encode()))
    manifest = {"n": len(pick), "seed": a.seed, "split": a.split,
                "source_splits": os.path.abspath(a.splits),
                "cases": [r["case"] for r in pick]}
    ops.append(CommitOperationAdd(path_in_repo="manifest.json",
                                  path_or_fileobj=json.dumps(manifest, indent=2).encode()))
    api.create_commit(repo_id=a.annot_repo, repo_type="dataset", operations=ops,
                      commit_message=f"seed {len(pick)} femoral-head cases (seed {a.seed})")
    print(f"  seeded {len(pick)} cases + manifest.json in {a.annot_repo}")
    print("\nSpace secrets to set:")
    print(f"  HF_TOKEN=<write token>   ANNOT_REPO={a.annot_repo}")
    print(f"  IMAGE_REPO={a.image_repo}   ADJUDICATORS=<your hf username>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
