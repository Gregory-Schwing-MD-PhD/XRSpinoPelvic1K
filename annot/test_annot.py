"""End-to-end test of the annotation Space, on the filesystem.

Runs the real app over real HTTP with a real ledger and real films (ANNOT_REPO=local:...)
so everything the readers touch is exercised: claiming, the two-reader rule, submitting
points, NOT VISIBLE as an answer, consensus, disagreement, the status board and the
teaching images.

The properties worth guarding are the ones that fail silently:
  * one person must never hold both slots on a film -- that would make "double read" one
    person twice and inter-reader agreement meaningless;
  * "not visible" must FILL a slot, not release it, or the not-visible rate (the number
    this whole exercise exists to produce) undercounts;
  * two readers disagreeing must be held, never averaged into a final answer.

    python annot/test_annot.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
TMP = pathlib.Path(tempfile.mkdtemp(prefix="annot_test_"))
LEDGER, IMAGES = TMP / "ledger", TMP / "images"
(LEDGER / "cases").mkdir(parents=True)
IMAGES.mkdir(parents=True)

N_CASES = 5
os.environ["ANNOT_REPO"] = f"local:{LEDGER}"
os.environ["IMAGE_REPO"] = f"local:{IMAGES}"
os.environ["FLUSH_SECONDS"] = "0.2"
os.environ["ADJUDICATORS"] = "chief"
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

for i in range(N_CASES):
    cid = f"case{i:03d}"
    Image.new("L", (900, 1400), 40 + 30 * i).save(IMAGES / f"{cid}.jpg")
    (LEDGER / "cases" / f"{cid}.json").write_text(
        json.dumps({"case_id": cid, "split": "train", "slots": {}}))

import app as A  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


def hdr(name):
    return {"Authorization": f"Bearer {name}"}


with TestClient(A.app) as c:
    print("\n[1] sign-in gate")
    check(c.get("/").status_code == 200, "page serves")
    check(c.get("/next").status_code == 401, "anonymous request is refused")
    check(c.get("/whoami").json()["user"] is None, "whoami reports nobody signed in")

    print("\n[2] teaching images")
    for path, what in (("/reference", "where-to-look figure"),
                       ("/example", "worked example")):
        r = c.get(path)
        check(r.status_code == 200 and r.content[:4] == b"\x89PNG",
              f"{what} is served as a PNG ({len(r.content)//1000} kB)")

    print("\n[3] a reader takes films")
    got = []
    for _ in range(N_CASES):
        r = c.get("/next", headers=hdr("alice"))
        assert r.status_code == 200, r.text
        j = r.json()
        got.append((j["case_id"], j["slot"]))
        img = c.get(j["image_url"], headers=hdr("alice"))
        check(img.status_code == 200 and img.headers["content-type"] == "image/jpeg",
              f"{j['case_id']} film streams") if len(got) == 1 else None
        # answer every film, mixing in one not-visible
        if len(got) == 2:
            c.post("/submit", headers=hdr("alice"),
                   data={"case_id": j["case_id"], "slot": j["slot"], "points": "",
                         "not_visible": "1", "reason": "prosthesis"})
        else:
            c.post("/submit", headers=hdr("alice"), data={
                "case_id": j["case_id"], "slot": j["slot"],
                "points": json.dumps({"left": [0.5, 0.6], "right": None,
                                      "w": 900, "h": 1400})})
    check(len({g[0] for g in got}) == N_CASES, "each film handed out once")
    check(all(s == "1" for _, s in got), "first reader always gets slot 1")
    check(c.get("/next", headers=hdr("alice")).status_code == 404,
          "a reader is never offered a film they already read")

    print("\n[4] second reader, and how the three outcomes resolve")
    # agree on case000, disagree on case001, agree-not-visible on case001? -> use ids
    plan = {
        "case000": ("point", [0.5, 0.6]),        # within tolerance -> consensus
        "case001": ("nv", None),                 # both not visible -> settled, no point
        "case002": ("point", [0.9, 0.9]),        # far apart -> disagreement
        "case003": ("nv", None),                 # alice marked a point -> disagreement
        "case004": ("point", [0.5, 0.6]),
    }
    for _ in range(N_CASES):
        r = c.get("/next", headers=hdr("bob"))
        if r.status_code != 200:
            break
        j = r.json()
        kind, pt = plan[j["case_id"]]
        if kind == "nv":
            c.post("/submit", headers=hdr("bob"),
                   data={"case_id": j["case_id"], "slot": j["slot"], "points": "",
                         "not_visible": "1", "reason": "out of field"})
        else:
            c.post("/submit", headers=hdr("bob"), data={
                "case_id": j["case_id"], "slot": j["slot"],
                "points": json.dumps({"left": pt, "right": None, "w": 900, "h": 1400})})

    def case(cid):
        return A._INDEX[cid]

    check(case("case000").get("final", {}).get("by") == "consensus",
          "two close reads -> consensus, point = their mean")
    check(case("case001").get("final", {}).get("by") == "consensus-not-visible"
          and case("case001")["final"]["points"] is None,
          "both readers say not visible -> settled with NO point")
    check(not case("case002").get("final") and case("case002").get("disagree"),
          "two distant reads -> held for adjudication, never averaged")
    check(not case("case003").get("final") and case("case003").get("disagree"),
          "one marked / one not-visible -> held, not silently resolved either way")

    print("\n[5] not visible fills a slot, it does not release the film")
    s = case("case001")["slots"]
    check(all(s[k].get("done") for k in ("1", "2")), "both slots recorded as done")
    check(all(s[k].get("not_visible") for k in ("1", "2")), "both marked not_visible")

    print("\n[6] pass releases the slot")
    # Its OWN case. The first version of this reset case000's slots, which silently
    # deleted one finished read from each of alice and bob and made the read counts in
    # [7] wrong -- the test breaking the thing it was about to measure.
    cid = "passcase"
    A._INDEX[cid] = {"case_id": cid, "slots": {
        "1": {"annotator": "carol", "done": False, "claimed_at": 0, "expires_at": 9e18}}}
    A._ORDER.append(cid)
    c.post("/skip", headers=hdr("carol"),
           data={"case_id": cid, "slot": "1", "reason": "interrupted"})
    check("1" not in A._INDEX[cid]["slots"], "passed slot is freed for another reader")
    check(A._INDEX[cid].get("passed_by"), "the pass is recorded")

    print("\n[7] status board")
    r = c.get("/stats", headers=hdr("alice"))
    j = r.json()
    who = {x["annotator"]: x for x in j["readers"]}
    check(r.status_code == 200, "stats served")
    check(who["alice"]["reads"] == 5 and who["bob"]["reads"] >= 4,
          f"per-reader read counts ({ {k: v['reads'] for k, v in who.items()} })")
    check(who["alice"]["not_visible"] == 1, "per-reader not-visible count")
    check(j["counts"]["reads_needed"] == (N_CASES + 1) * 2,
          "two reads per film is the target")
    check(j["counts"]["needs_adjudication"] == 2, "both disagreements surfaced")
    check(j["agreement"] and j["agreement"]["n"] >= 2, "agreement stats computed")
    check(c.get("/board").status_code == 200, "board page serves")

    print("\n[8] durability: the background flush actually writes")
    import time
    time.sleep(1.0)
    A._flush_once()
    disk = json.loads((LEDGER / "cases" / "case001.json").read_text())
    check(disk.get("final", {}).get("by") == "consensus-not-visible",
          "resolved case reached the ledger on disk")

print()
print("FAILED: " + "; ".join(fails) if fails else "ALL CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
