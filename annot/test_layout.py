"""Screenshot the reading page in a real browser, against a local ledger.

The API tests say the right data moves; they say nothing about whether the film is
reachable with a mouse. These are the three things that were wrong in use:

  * criteria ran full width above the film, so the film started below the fold;
  * the film reached the bottom of the screen, where the Windows taskbar auto-shows
    and swallows the pointer before it reaches the caudal anatomy;
  * the magnifier followed the cursor and covered the arc being judged.

    python annot/test_layout.py [--headed]
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading

ROOT = pathlib.Path(__file__).resolve().parent
TMP = pathlib.Path(tempfile.mkdtemp(prefix="annot_layout_"))
LEDGER, IMAGES = TMP / "ledger", TMP / "images"
(LEDGER / "cases").mkdir(parents=True)
IMAGES.mkdir(parents=True)
SHOTS = pathlib.Path(tempfile.gettempdir()) / "annotshots"
SHOTS.mkdir(exist_ok=True)

os.environ["ANNOT_REPO"] = f"local:{LEDGER}"
os.environ["IMAGE_REPO"] = f"local:{IMAGES}"
os.environ["FLUSH_SECONDS"] = "0.5"
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

# a tall lateral-shaped film, like BUU
for i in range(3):
    cid = f"film{i:03d}"
    Image.new("L", (1392, 2428), 60).save(IMAGES / f"{cid}.jpg")
    (LEDGER / "cases" / f"{cid}.json").write_text(
        json.dumps({"case_id": cid, "slots": {}}))

import app as A  # noqa: E402
import uvicorn  # noqa: E402

PORT = 7861
cfg = uvicorn.Config(A.app, host="127.0.0.1", port=PORT, log_level="error")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()

fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


from playwright.sync_api import sync_playwright  # noqa: E402

VIEW = {"width": 1600, "height": 900}
with sync_playwright() as p:
    br = p.chromium.launch(headless="--headed" not in sys.argv)
    pg = br.new_page(viewport=VIEW)
    # ANY uncaught JS error is fatal here. Both #gate and #appui start hidden and boot()
    # reveals one, so a single SyntaxError anywhere in the script leaves the reader a
    # blank page with nothing in the UI to hint at why. That shipped once, from a bare
    # newline inside a JS string literal -- this file is a Python string, so the escape
    # has to survive two layers.
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
    pg.evaluate("localStorage.setItem('hf_tok','alice')")
    pg.reload(wait_until="networkidle")
    pg.wait_for_function("document.getElementById('c').width > 0", timeout=20000)
    pg.wait_for_timeout(400)

    check(not errs, f"no JS errors on the reading page ({errs[:2]})")
    check(pg.eval_on_selector("#appui", "e => !e.hidden"),
          "the app actually became visible (not a blank page)")

    # the adjudication page is a second script and fails the same way
    rerrs = []
    pg2 = br.new_page(viewport=VIEW)
    pg2.on("pageerror", lambda e: rerrs.append(str(e)))
    pg2.on("console", lambda m: rerrs.append(f"console.error: {m.text}")
           if m.type == "error" else None)
    pg2.goto(f"http://127.0.0.1:{PORT}/review", wait_until="networkidle")
    pg2.wait_for_timeout(600)
    syntax = [e for e in rerrs if "SyntaxError" in e or "Unexpected" in e]
    check(not syntax, f"no JS syntax errors on /review ({syntax[:1]})")
    pg2.close()

    guide = pg.locator("#guide").bounding_box()
    stage = pg.locator("#stage").bounding_box()
    canvas = pg.locator("#c").bounding_box()
    print(f"       guide  {guide}")
    print(f"       canvas {canvas}")

    check(guide["x"] < stage["x"], "guide is on the LEFT of the film")
    check(canvas["x"] > guide["x"] + guide["width"] - 2,
          "film is on the RIGHT, clear of the guide")
    check(guide["height"] <= VIEW["height"], "guide fits the window (scrolls internally)")
    check(pg.evaluate("(()=>{const g=document.getElementById('guide');"
                      "return g.scrollHeight > g.clientHeight})()"),
          "guide actually scrolls rather than clipping")

    gap = VIEW["height"] - (canvas["y"] + canvas["height"])
    check(gap >= 50, f"bottom gutter clear of the taskbar ({gap:.0f} px below the film)")
    check(canvas["y"] + canvas["height"] <= VIEW["height"], "film is fully on screen")

    # magnifier must be OFF until asked for
    pg.mouse.move(canvas["x"] + canvas["width"] / 2, canvas["y"] + canvas["height"] / 2)
    pg.wait_for_timeout(200)
    check(pg.eval_on_selector("#loupe", "e => getComputedStyle(e).display") == "none",
          "magnifier is off by default")
    pg.screenshot(path=str(SHOTS / "1_reading.png"))

    pg.keyboard.press("m")
    pg.mouse.move(canvas["x"] + canvas["width"] / 2, canvas["y"] + canvas["height"] / 3)
    pg.wait_for_timeout(250)
    check(pg.eval_on_selector("#loupe", "e => getComputedStyle(e).display") != "none",
          "pressing m turns the magnifier on")
    pg.keyboard.press("m")
    pg.wait_for_timeout(150)
    check(pg.eval_on_selector("#loupe", "e => getComputedStyle(e).display") == "none",
          "pressing m again turns it off")

    # Hiding the guide gives the film the whole window. It does NOT necessarily make the
    # film bigger: a lateral is much taller than it is wide, so on a normal monitor the
    # window HEIGHT is what binds and extra width changes nothing. Assert what is
    # actually being claimed -- the stage widens and the film never shrinks.
    c_before = pg.locator("#c").bounding_box()["width"]
    s_before = pg.locator("#stage").bounding_box()["width"]
    pg.keyboard.press("g")
    pg.wait_for_timeout(300)
    c_after = pg.locator("#c").bounding_box()["width"]
    s_after = pg.locator("#stage").bounding_box()["width"]
    check(s_after > s_before,
          f"hiding the guide widens the film column ({s_before:.0f} -> {s_after:.0f} px)")
    check(c_after >= c_before - 1,
          f"the film never shrinks when the guide hides ({c_before:.0f} -> {c_after:.0f})")
    pg.screenshot(path=str(SHOTS / "2_guide_hidden.png"))
    pg.keyboard.press("g")
    pg.wait_for_timeout(250)

    # a click must land where it was aimed, in normalised coords
    pg.mouse.click(canvas["x"] + canvas["width"] * 0.5,
                   canvas["y"] + canvas["height"] * 0.25)
    pg.wait_for_timeout(150)
    pt = pg.evaluate("pts[0]")
    check(pt and abs(pt[0] - 0.5) < 0.02 and abs(pt[1] - 0.25) < 0.02,
          f"click maps to the right spot on the film ({pt})")
    pg.screenshot(path=str(SHOTS / "3_marked.png"))
    br.close()

srv.should_exit = True
print(f"\n  shots in {SHOTS}")
print("FAILED: " + "; ".join(fails) if fails else "\nALL CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
