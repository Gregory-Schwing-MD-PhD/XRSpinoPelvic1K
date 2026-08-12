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
    print("       diag:", pg.evaluate("""(()=>{
      const st=document.getElementById('stage').getBoundingClientRect();
      const c=document.getElementById('c');
      const cs=getComputedStyle(document.getElementById('stage'));
      return {stageTop:+st.top.toFixed(1), stageH:+st.height.toFixed(1),
              padBottom:cs.paddingBottom, innerH:window.innerHeight,
              canvasCssH:c.style.height, imgH:img.height, zoom:zoom};
    })()"""))

    example = pg.locator("#example").bounding_box()
    print(f"       example {example}")
    check(guide["x"] < stage["x"], "guide is on the LEFT of the film")
    check(example["x"] > stage["x"] + stage["width"] - 2,
          "worked example is on the RIGHT of the film")
    check(pg.eval_on_selector_all("#example img", "e => e.length") == 3,
          "three example panels are shown")
    check(pg.evaluate("""(()=>{const e=document.getElementById('example');
                               return e.scrollHeight > e.clientHeight})()"""),
          "the example column scrolls independently")
    # panels must actually be served, not broken images
    check(pg.evaluate("""(()=>[...document.querySelectorAll('#example img')]
                               .every(i=>i.naturalWidth>0))()"""),
          "every example panel image loaded")
    check(canvas["x"] > guide["x"] + guide["width"] - 2,
          "film is on the RIGHT, clear of the guide")
    check(guide["height"] <= VIEW["height"], "guide fits the window (scrolls internally)")
    check(pg.evaluate("(()=>{const g=document.getElementById('guide');"
                      "return g.scrollHeight > g.clientHeight})()"),
          "guide actually scrolls rather than clipping")

    gap = VIEW["height"] - (canvas["y"] + canvas["height"])
    check(gap >= 50, f"bottom gutter clear of the taskbar ({gap:.0f} px below the film)")
    check(canvas["y"] + canvas["height"] <= VIEW["height"], "film is fully on screen")

    # PACS mapping: wheel zooms, left-drag windows, right-drag pans, click marks.
    # Zoom is the one that matters -- fit-to-window puts a BUU film at ~436 px wide on a
    # 1600x900 screen, so the 0.005 consensus tolerance is 2.2 SCREEN PIXELS and the
    # display, not the reader, sets the floor on agreement.
    cx = canvas["x"] + canvas["width"] / 2
    cy = canvas["y"] + canvas["height"] / 2
    pg.mouse.move(cx, cy)
    w0 = pg.locator("#c").bounding_box()["width"]
    for _ in range(10):                        # 1.15^10 ~ 4x, a normal working zoom
        pg.mouse.wheel(0, -120)
        pg.wait_for_timeout(60)
    w1 = pg.locator("#c").bounding_box()["width"]
    check(w1 > w0 * 1.3, f"scroll zooms, no modifier ({w0:.0f} -> {w1:.0f} px)")
    check(0.005 * w1 > 6,
          f"zoomed, the tolerance is {0.005 * w1:.1f} screen px -- clickable "
          f"(2.2 px at fit-to-window)")

    # left-drag windows, and must NOT drop a mark
    n_before = pg.evaluate("circles.length")
    c0 = pg.evaluate("gainC")
    pg.mouse.move(cx, cy); pg.mouse.down()
    pg.mouse.move(cx + 90, cy - 40, steps=8)
    pg.mouse.up()
    pg.wait_for_timeout(150)
    check(pg.evaluate("gainC") > c0,
          f"left-drag raises contrast ({c0} -> {pg.evaluate('gainC')})")
    check(pg.evaluate("gainB") > 1, "left-drag upward raises brightness")
    check(pg.evaluate("circles.length") == n_before,
          "a windowing drag on bare film does NOT drop a circle")

    # right-drag pans
    sl0 = pg.evaluate("document.getElementById('stage').scrollLeft")
    pg.mouse.move(cx, cy)
    pg.mouse.down(button="right")
    pg.mouse.move(cx - 70, cy, steps=6)
    pg.mouse.up(button="right")
    pg.wait_for_timeout(150)
    check(pg.evaluate("document.getElementById('stage').scrollLeft") > sl0,
          "right-drag pans the film")

    # zoom and pan survive the next film -- the anatomy is in the same place every time
    z0 = pg.evaluate("zoom")
    pg.keyboard.press("Enter") if False else None
    pg.evaluate("load()")
    pg.wait_for_timeout(1200)
    check(abs(pg.evaluate("zoom") - z0) < 1e-6,
          f"zoom persists onto the next film ({pg.evaluate('zoom'):.2f}x)")
    check(pg.locator("#c").bounding_box()["width"] > w0 * 1.3,
          "and the next film is actually drawn at that zoom")

    pg.keyboard.press("r")
    pg.wait_for_timeout(300)
    check(abs(pg.evaluate("zoom") - 1) < 1e-6 and abs(pg.evaluate("gainC") - 1) < 1e-6,
          "r resets zoom and windowing")
    pg.screenshot(path=str(SHOTS / "1_reading.png"))

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

    # a click drops a CIRCLE where it was aimed, in normalised coords
    box = pg.locator("#c").bounding_box()
    tx = box["x"] + box["width"] * 0.5
    ty = box["y"] + box["height"] * 0.25
    pg.mouse.click(tx, ty)
    pg.wait_for_timeout(150)
    c0 = pg.evaluate("circles[0]")
    check(c0 and abs(c0[0] - 0.5) < 0.02 and abs(c0[1] - 0.25) < 0.02,
          f"click drops a circle where aimed ({c0})")
    check(c0 and c0[2] > 0, f"the circle has a radius ({c0[2] if c0 else None})")

    # hovering it and scrolling resizes THAT circle, and must not zoom the film
    r0 = pg.evaluate("circles[0][2]")
    w_before = pg.locator("#c").bounding_box()["width"]
    pg.mouse.move(tx, ty)
    pg.mouse.wheel(0, -120)
    pg.wait_for_timeout(150)
    check(pg.evaluate("circles[0][2]") > r0,
          f"scrolling over a circle grows it ({r0:.4f} -> {pg.evaluate('circles[0][2]'):.4f})")
    check(abs(pg.locator("#c").bounding_box()["width"] - w_before) < 2,
          "and does NOT zoom the film")

    # dragging it nudges the circle rather than windowing
    cbefore = pg.evaluate("circles[0].slice()")
    gbefore = pg.evaluate("gainC")
    pg.mouse.move(tx, ty); pg.mouse.down()
    pg.mouse.move(tx + 30, ty + 10, steps=6); pg.mouse.up()
    pg.wait_for_timeout(150)
    cafter = pg.evaluate("circles[0]")
    check(cafter[0] > cbefore[0], "dragging a circle moves it")
    check(abs(pg.evaluate("gainC") - gbefore) < 1e-9,
          "dragging a circle does NOT window the film")

    # the radius carries to the second circle
    pg.mouse.click(box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.3)
    pg.wait_for_timeout(150)
    check(pg.evaluate("circles.length") == 2, "a second circle can be placed")
    check(abs(pg.evaluate("circles[1][2]") - pg.evaluate("circles[0][2]")) < 1e-6,
          "the second circle inherits the sized radius")
    # OVERLAPPING circles must both stay readable, so the interiors are stroke-only.
    # Two heads on a rotated film overlap by construction; a filled disc would hide the
    # arc of whichever circle was drawn first, which is the thing being matched.
    pg.evaluate("""(()=>{
        circles.length=0;
        circles.push([0.44,0.55,0.10],[0.56,0.57,0.10]);   // deliberately overlapping
        sel=1; draw();
    })()""")
    pg.wait_for_timeout(200)
    px = pg.evaluate("""(()=>{
        const g=C.getContext('2d');
        const rd=(fx,fy)=>{const d=g.getImageData(Math.round(fx*C.width),
                                                  Math.round(fy*C.height),1,1).data;
                           return [d[0],d[1],d[2]];};
        // sample EMPTY interior: the centres carry a crosshair and the amber hip-axis
        // line runs exactly through both, so sampling a centre reads the marker, not
        // the fill, which is what this check is actually about
        return {inside1: rd(0.415,0.585),    // inside circle 0, off its crosshair
                inside2: rd(0.585,0.545),    // inside circle 1, off its crosshair
                lens:    rd(0.500,0.530),    // the overlap, above the axis line
                film:    rd(0.200,0.200)};   // bare film for reference
    })()""")
    print(f"       pixels {px}")
    same = lambda a, b: all(abs(x - y) <= 6 for x, y in zip(a, b))
    check(same(px["inside1"], px["film"]),
          f"first circle is not filled -- film shows through ({px['inside1']})")
    check(same(px["inside2"], px["film"]),
          f"second circle is not filled ({px['inside2']})")
    check(same(px["lens"], px["film"]),
          f"the overlap of two circles stays transparent ({px['lens']})")
    pg.screenshot(path=str(SHOTS / "4_two_circles.png"))
    pg.screenshot(path=str(SHOTS / "3_marked.png"))
    br.close()

srv.should_exit = True
print(f"\n  shots in {SHOTS}")
print("FAILED: " + "; ".join(fails) if fails else "\nALL CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
