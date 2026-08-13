"""Seed a fresh 100-film pilot ledger for the A/S/P landmark tool.

WHY A SEPARATE LEDGER, AND WHY THESE FILMS
------------------------------------------
The existing ledger is finished: 2000 films, both reads in, every slot taken -- which is
why the tool says "nothing left to annotate". Reopening slots in place would overwrite
3468 completed reads, and the two tools would then share one ledger with no clean way to
compare them. So the pilot gets its own dataset repo and the old one is left frozen.

The 100 films are drawn from those the circle tool FINISHED AND SETTLED. That is the
whole point of the design: the same films now get read twice more with the landmark tool,
so the pilot measures two things instead of one --

    inter-reader agreement WITHIN the new tool     (is it tighter than 0.0183?)
    the new consensus against the OLD consensus    (do the two methods agree on the
                                                    same anatomy, or has the landmark
                                                    definition moved the point?)

The second is not available from a fresh sample of unread films, and it is the one that
says whether the 3468 existing reads can be pooled with what comes next or have to be
treated as a different measurement.

Nothing about the old read is copied into the pilot ledger. The readers must not see it:
a reader shown the previous answer is measuring their own suggestibility. It stays where
it already is, in the frozen repo, keyed by case_id.

    python annot/seed_pilot.py                     # dry run: says what it would do
    python annot/seed_pilot.py --apply
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from datetime import date

SRC = "gregoryschwingmdphd/xrsp-femhead-annot"
DST = "gregoryschwingmdphd/xrsp-femhead-asp-pilot"


def token() -> str:
    t = os.environ.get("HF_TOKEN") or ""
    if not t:
        from pathlib import Path
        p = Path.home() / ".cache/huggingface/token"
        t = p.read_text().strip() if p.exists() else ""
    if not t:
        sys.exit("set HF_TOKEN (a write token)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    tok = token()
    api = HfApi(token=tok)

    idx = json.loads(open(hf_hub_download(a.src, "index.json", repo_type="dataset",
                                          token=tok, force_download=True), "rb").read())
    print(f"  source ledger        {a.src}: {len(idx)} cases")

    def reads(c):
        s = c.get("slots") or {}
        return [s[k] for k in ("1", "2")
                if (s.get(k) or {}).get("done") and (s.get(k) or {}).get("points")]

    # TWO COMPLETED READS, not "finalised". Only 185 of 2000 ever settled -- the circle
    # tool's readers disagreed by a median 0.0183 against a 0.005 tolerance, so the rest
    # are sitting in adjudication. Requiring `final` would leave 71 films, all of them the
    # ones that were easy enough to agree on first time, which is precisely the sample
    # that cannot answer whether a new tool helps.
    #
    # The two reads' MIDPOINT is the old-method estimate for the comparison. The 0.005
    # tolerance is a QC gate on whether a film needs a third look; it is not what defines
    # the answer, and a pair that missed it still localises the head to a few millimetres.
    have = [c for c in idx if len(reads(c)) == 2 and c.get("agree") is not None]
    tight = sorted([c for c in have if c["agree"] <= 0.005], key=lambda c: c["agree"])
    loose = sorted([c for c in have if c["agree"] > 0.005], key=lambda c: c["agree"])
    print(f"  two completed reads   {len(have)}")
    print(f"      circle tool AGREED (<=0.005)  {len(tight)}")
    print(f"      circle tool DISAGREED         {len(loose)}")

    # Stratified half and half, deliberately. A pilot drawn only from films the old tool
    # found easy would report a flattering agreement number and say nothing about the
    # films that generated 1288 adjudications. Half easy, half hard, and the two halves
    # are reported separately.
    rng = random.Random(a.seed)
    half = a.n // 2
    pick_t = rng.sample(tight, min(half, len(tight)))
    pick_l = rng.sample(loose, min(a.n - len(pick_t), len(loose)))
    if len(pick_t) + len(pick_l) < a.n:                         # top up from whichever has room
        rest = [c for c in have if c not in pick_t and c not in pick_l]
        pick_t += rng.sample(rest, min(a.n - len(pick_t) - len(pick_l), len(rest)))
    pick = sorted(pick_t + pick_l, key=lambda c: c["case_id"])
    hard = {c["case_id"] for c in pick_l}
    two = sum(1 for c in pick
              if max(len((r["points"].get("heads") or [])) for r in reads(c)) == 2)
    print(f"\n  sampled               {len(pick)} (seed {a.seed})")
    print(f"      from the AGREED half          {len(pick_t)}")
    print(f"      from the DISAGREED half       {len(pick_l)}")
    print(f"      where a reader saw TWO heads  {two}")

    cases = []
    for c in pick:
        # A clean slate, carrying only what is not an answer: the id and the split.
        # Copying `final`, `agree` or the old slots forward would put the previous
        # reading in front of the new readers.
        cases.append({"case_id": c["case_id"], "split": c.get("split", ""),
                      "slots": {}})

    print(f"\n  destination          {a.dst}")
    if not a.apply:
        print("  DRY RUN -- pass --apply to create it")
        print("  first five:", [c["case_id"] for c in cases[:5]])
        return 0

    try:
        api.repo_info(a.dst, repo_type="dataset")
        print("  exists -- refusing to overwrite a live ledger")
        return 1
    except Exception:                                          # noqa: BLE001
        api.create_repo(a.dst, repo_type="dataset", private=True, exist_ok=True)

    ops = []
    from huggingface_hub import CommitOperationAdd
    for c in cases:
        ops.append(CommitOperationAdd(f"cases/{c['case_id']}.json",
                                      json.dumps(c).encode()))
    man = {"cases": [c["case_id"] for c in cases],
           "created": str(date.today()),
           "tool": "arc — three named extremes (anterior, superior, posterior)",
           "sampled_from": a.src,
           "sampling": f"random, seed {a.seed}, stratified half from films the circle "
                       f"tool's two readers AGREED on (<=0.005 of image width) and half "
                       f"from films they did NOT — so the pilot spans the easy films and "
                       f"the ones that generated the adjudication backlog",
           "hard_half": sorted(hard),
           "purpose": "pilot of the landmark tool. Same films as an existing settled "
                      "circle-tool read, so this measures BOTH the new tool's "
                      "inter-reader agreement AND new-vs-old agreement on identical "
                      "anatomy. No previous answer is stored here -- showing a reader "
                      "the earlier reading would measure suggestibility.",
           "compare_against": f"{a.src} (frozen), same case_id"}
    ops.append(CommitOperationAdd("manifest.json",
                                  json.dumps(man, indent=1).encode()))
    # index.json is a boot cache and is only trusted when it covers exactly the manifest's
    # case set -- so it is written here rather than left to be rebuilt on first request.
    ops.append(CommitOperationAdd("index.json", json.dumps(cases).encode()))
    api.create_commit(a.dst, repo_type="dataset", operations=ops,
                      commit_message=f"seed {len(cases)}-film A/S/P pilot")
    print(f"  wrote {len(cases)} cases + manifest.json + index.json")
    print(f"\n  now point the Space at it:")
    print(f"      ANNOT_REPO = {a.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
