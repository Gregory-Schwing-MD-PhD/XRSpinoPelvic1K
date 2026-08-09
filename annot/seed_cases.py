#!/usr/bin/env python3
"""Seed the femoral-head annotation ledger, and upload the films it refers to.

    python annot/seed_cases.py --buu data/BUU-LSPINE --splits data/buu_splits.json \
        --annot-repo <org>/xrsp-femhead-annot --image-repo <org>/xrsp-femhead-images \
        --n 120 --apply

WHICH FILMS, AND WHY IT MATTERS
-------------------------------
The TEST split only, and a random sample of it under a fixed seed.

Test-only because these annotations exist to VALIDATE a hip point learned from synthetic
DRRs. Annotating training films would let the number be read as training accuracy, and
annotating a mixture makes it impossible to say which it was.

Random rather than chosen, because any human selection biases the reference. Picking
"clear" films measures the model on easy cases; picking the model's failures measures a
worst case. The seed is recorded in the manifest so the sample is reproducible.

120 films is the default because it is enough to put a useful confidence interval on a
mean error (roughly +-0.2 SD at n=120) while staying inside an afternoon of clicking at
two points per film. Raise it with --n if the first pass looks borderline.
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
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--apply", action="store_true",
                    help="actually create repos and upload (default is a dry run)")
    a = ap.parse_args(argv)

    from xrsp.buu import index_buu

    assign = json.load(open(a.splits))["assignments"]
    test = [r for r in index_buu(a.buu) if assign.get(r["case"]) == "test"]
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(test, min(a.n, len(test))), key=lambda r: r["case"])
    print(f"test films {len(test)} -> sampling {len(pick)} (seed {a.seed})")
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

    # Images go to their OWN repo, and the Space streams them through itself. Annotators
    # then need no read access to any dataset -- the same containment the review service
    # uses for labels.
    for r in pick:
        api.upload_file(path_or_fileobj=r["img"], path_in_repo=f"images/{r['case']}.jpg",
                        repo_id=a.image_repo, repo_type="dataset")
    print(f"  uploaded {len(pick)} films to {a.image_repo}")

    ops = []
    from huggingface_hub import CommitOperationAdd
    for r in pick:
        case = {"case_id": r["case"], "source": "BUU-LSPINE", "split": "test",
                "slots": {}, "seeded_seed": a.seed}
        ops.append(CommitOperationAdd(
            path_in_repo=f"cases/{r['case']}.json",
            path_or_fileobj=json.dumps(case, indent=2).encode()))
    manifest = {"n": len(pick), "seed": a.seed, "split": "test",
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
