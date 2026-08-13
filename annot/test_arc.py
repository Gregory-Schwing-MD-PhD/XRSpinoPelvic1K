"""The arc tool in a real browser, on a film with a femoral head that actually exists.

test_layout.py checks the page geometry. This checks the thing readers touch: whether the
fit is right, whether the placement QC points at the cortex, whether the workflow bends
when a reader works out of order -- and whether any of it is slow enough to feel.

The synthetic film is not decoration. Placement QC scores edge strength, so testing it
needs an image with a known edge in a known place: a bright thin cortical rim around a
mid-grey head on a dark ground, so "on the cortex", "inside the head" and "out in the
joint space" are three distinguishable answers the way they are on a radiograph.

EVERYTHING HERE IS IN NORMALISED COORDINATES. The app serves films downscaled to
DISPLAY_W, and the QC maps are built on what the reader is actually shown -- so a probe
written in source pixels lands somewhere else entirely. An earlier version of this file
did exactly that and "passed" while measuring bare film.

    python annot/test_arc.py [--headed]
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shutil
import sys
import tempfile
import threading

ROOT = pathlib.Path(__file__).resolve().parent
TMP = pathlib.Path(tempfile.mkdtemp(prefix="annot_arc_"))
LEDGER, IMAGES = TMP / "ledger", TMP / "images"
(LEDGER / "cases").mkdir(parents=True)
IMAGES.mkdir(parents=True)
SHOTS = pathlib.Path(tempfile.gettempdir()) / "annotshots"
SHOTS.mkdir(exist_ok=True)

os.environ["ANNOT_REPO"] = f"local:{LEDGER}"
os.environ["IMAGE_REPO"] = f"local:{IMAGES}"
os.environ["FLUSH_SECONDS"] = "0.5"
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # the readout uses arrows

import numpy as np                                              # noqa: E402
from PIL import Image                                           # noqa: E402

W, H = 1392, 2428                                               # source film
HEAD_X, HEAD_Y, HEAD_R = 700.0, 900.0, 120.0                    # ground truth, source px
CX, CY, CR = HEAD_X / W, HEAD_Y / H, HEAD_R / W                 # ...normalised
yy, xx = np.mgrid[0:H, 0:W]
d = np.hypot(xx - HEAD_X, yy - HEAD_Y)
film = np.full((H, W), 42.0)
film += 55 * (d < HEAD_R)                                       # cancellous interior
film += 150 * np.exp(-((d - HEAD_R) ** 2) / (2 * 2.2 ** 2))     # the cortical rim
film += np.random.default_rng(0).normal(0, 3.0, film.shape)     # sensor noise
film = np.clip(film, 0, 255).astype(np.uint8)
for i in range(10):
    cid = f"film{i:03d}"
    Image.fromarray(film).save(IMAGES / f"{cid}.jpg", quality=95)
    (LEDGER / "cases" / f"{cid}.json").write_text(
        json.dumps({"case_id": cid, "slots": {}}))

import app as A                                                 # noqa: E402
import uvicorn                                                  # noqa: E402

PORT = 7863
cfg = uvicorn.Config(A.app, host="127.0.0.1", port=PORT, log_level="error")
srv = uvicorn.Server(cfg)
threading.Thread(target=srv.run, daemon=True).start()

fails = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        fails.append(what)


def rim(deg):
    """Normalised coordinates of the point at `deg` on the true cortical rim."""
    t = math.radians(deg)
    return [CX + CR * math.cos(t), CY + CR * (W / H) * math.sin(t)]


from playwright.sync_api import sync_playwright                 # noqa: E402

VIEW = {"width": 1600, "height": 900}
with sync_playwright() as p:
    br = p.chromium.launch(headless="--headed" not in sys.argv)
    pg = br.new_page(viewport=VIEW)
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(f"console.error: {m.text}")
          if m.type == "error" else None)
    pg.goto(f"http://127.0.0.1:{PORT}/?tool=arc", wait_until="networkidle")
    pg.evaluate("localStorage.setItem('hf_tok','alice')")
    pg.evaluate("localStorage.removeItem('annot_face')")
    pg.reload(wait_until="networkidle")
    pg.wait_for_function("document.getElementById('c').width > 0", timeout=20000)
    pg.wait_for_function("qcReady === true", timeout=20000)
    pg.wait_for_timeout(200)

    def put(spec, head=0, dx=0.0, dy=0.0):
        """Place (angle, role) pairs on head `head`, offset by (dx,dy) if given."""
        pts = [[rim(g)[0] + dx, rim(g)[1] + dy, r] for g, r in spec]
        pg.evaluate("""([pts,k]) => {
             if(k===0) HD=[newHd()];
             while(HD.length<=k) HD.push(newHd());
             HD[k].pts.length=0; HD[k].skip={};
             pts.forEach(q=>HD[k].pts.push({x:q[0], y:q[1], role:q[2]}));
             acur=k; sel2=null; refit(); draw(); readout(); }""", [pts, head])

    print("\n[1] the page holds together")
    check(not errs, f"no JS errors ({errs[:2]})")
    check(pg.eval_on_selector("#appui", "e => !e.hidden"), "the app is visible")
    check(pg.evaluate("TOOL") == "arc", "arc is the default tool")
    SW, SH = pg.evaluate("[img.width, img.height]")
    print(f"       source {W}x{H} -> served {SW}x{SH}")
    check(abs(SW / W - SH / H) < 1e-3, "the downscale preserves aspect ratio")
    # where the head really is in the pixels the QC maps were built from
    PX, PY, PR = CX * SW, CY * SH, CR * SW

    print("\n[2] the fit")

    def fit_at(degs):
        return pg.evaluate(
            """pts => { HD=[newHd()]; acur=0; sel2=null;
                 pts.forEach(q=>HD[0].pts.push({x:q[0], y:q[1],
                                                role:nextRole(HD[0])}));
                 refit(); draw(); readout();
                 const f=fits[0];
                 return f && {a:f.a/img.width, b:f.b/img.height, R:f.R/img.width,
                              rms:f.rms, two:2*f.e1/img.width,
                              swapped:f.swapped, tilt:f.aptilt};
               }""", [rim(g) for g in degs])

    wide = fit_at([180, 270, 0])                 # A left, S top, P right: facing left
    shallow = fit_at([255, 270, 285])
    print(f"       A/S/P   {wide}")
    print(f"       shallow {shallow}")
    check(wide and abs(wide["a"] - CX) < 2e-4 and abs(wide["b"] - CY) < 2e-4
          and abs(wide["R"] - CR) < 2e-4,
          f"three extremes recover the centre and radius exactly "
          f"(true {CX:.5f},{CY:.5f} r={CR:.5f})")
    check(wide["rms"] < 1e-6 and shallow["rms"] < 1e-6,
          f"both fits have zero residual ({wide['rms']:.1e}, {shallow['rms']:.1e}) -- "
          f"residuals cannot tell them apart")
    check(shallow["two"] > 4 * wide["two"],
          f"but the shallow one's error bar is far bigger "
          f"({wide['two']:.5f} -> {shallow['two']:.5f})")
    check(0.005 < wide["two"] < 0.010,
          f"A/S/P alone lands just ABOVE the 0.005 tolerance ({wide['two']:.5f}) -- they "
          f"are the findable three, not the best-conditioned three")
    even = fit_at([0, 120, 240])
    check(even["two"] < wide["two"] * 0.8,
          f"evenly spaced points are tighter ({even['two']:.5f} vs {wide['two']:.5f}) -- "
          f"which is why extra rim points are asked for, not merely tolerated")
    extras = fit_at([180, 270, 0, 225, 315, 90])
    check(extras["two"] <= 0.005,
          f"and A/S/P plus three extras clears the tolerance ({extras['two']:.5f})")

    print("\n[3] placement QC finds the cortex")
    # A profile straight through the cortex at 12 o'clock. Positive d is into the head,
    # negative is out into the joint space; both are wrong places to click and the crest
    # is the right one, so the score must plateau at 0 and fall away on BOTH sides.
    prof = {k: pg.evaluate("([x,y]) => qcAt(x,y)", [PX, PY - PR + k])
            for k in (0, 3, -3, 12, 20, -25)}
    for k in sorted(prof):
        print(f"       {k:+4d} px from the rim: {prof[k]}")
    check(prof[0]["pct"] >= 70,
          f"a point ON the cortex clears the on-cortex threshold ({prof[0]['pct']}%)")
    # THE regression this guards. |grad| dips at the crest of a thin bright line and peaks
    # on its flanks, so an AVERAGED score ranks the crest below points 3 px off it and
    # tells a reader who is exactly right to move. A 3x3 mean did this; so did a 5x5.
    check(prof[0]["pct"] >= prof[3]["pct"] and prof[0]["pct"] >= prof[-3]["pct"],
          f"and is never beaten by its own flanks "
          f"({prof[-3]['pct']}/{prof[0]['pct']}/{prof[3]['pct']}%)")
    check(prof[0]["d"] == 0,
          "and a correct click is given NO hint to move -- a hint that fires when the "
          "reader is already right is worse than none")
    check(prof[20]["pct"] < 40,
          f"20 px inside the head scores low ({prof[20]['pct']}%) -- it is bone, but it "
          f"is not cortex")
    check(prof[-25]["pct"] < 40,
          f"out in the joint space scores low ({prof[-25]['pct']}%)")
    check(prof[12]["dy"] < 0 and prof[12]["bpct"] > prof[12]["pct"] + 20,
          f"a click off the cortex gets an OUTWARD hint to a better spot "
          f"({prof[12]['pct']}% -> {prof[12]['bpct']}% at {prof[12]['d']} px)")
    # featureless film must not score high merely because everything around it ties
    empty = pg.evaluate("([x,y]) => qcAt(x,y)", [SW * 0.15, SH * 0.12])
    check(empty["flat"] and empty["pct"] == 0,
          f"empty film says 'no structure nearby' rather than ranking noise ({empty})")
    # the hover probe must reach the same verdict as the click that follows it
    box = pg.locator("#c").bounding_box()
    pg.mouse.move(box["x"] + box["width"] * rim(270)[0],
                  box["y"] + box["height"] * rim(270)[1])
    pg.wait_for_timeout(250)
    hov = pg.eval_on_selector("#qcread", "e => e.textContent")
    print(f"       hover on the cortex reads: {hov!r}")
    check("on cortex" in hov, "the live hover readout says so before the click is made")
    pg.mouse.move(box["x"] + box["width"] * CX, box["y"] + box["height"] * CY)
    pg.wait_for_timeout(250)
    hov2 = pg.eval_on_selector("#qcread", "e => e.textContent")
    print(f"       hover in the middle reads: {hov2!r}")
    check("on cortex" not in hov2,
          "and does NOT say so in the middle of the head, where the centre is")

    print("\n[4] the workflow bends")
    pg.evaluate("() => { HD=[newHd()]; acur=0; sel2=null; fits=[]; draw(); }")
    for g in (45, 180, 315, 270, 0):             # deliberately not in A,S,P order
        q = rim(g)
        pg.mouse.click(box["x"] + box["width"] * q[0], box["y"] + box["height"] * q[1])
        pg.wait_for_timeout(50)
    check(pg.evaluate("HD[0].pts.length") == 5, "five clicks land as five points")
    pg.keyboard.press("w")
    pg.wait_for_timeout(200)
    named = pg.evaluate("ROLES.filter(r=>roleAt(HD[0],r)).join('')")
    check(named == "ASP", f"Auto-name works out A, S and P from the shape ({named!r})")
    check(abs(pg.evaluate("roleAt(HD[0],'A').x") - rim(180)[0]) < 0.01,
          "and puts A on the anterior side, given the film faces left")
    check(pg.evaluate("HD[0].pts.filter(p=>!p.role).length") == 2,
          "the other two stay as extra rim points rather than being deleted")
    pg.evaluate("sel2={arc:0,i:0}")
    pg.keyboard.press("2")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts[0].role") == "S", "pressing 2 renames the selected point S")
    check(pg.evaluate("HD[0].pts.filter(p=>p.role==='S').length") == 1,
          "and whatever used to be S is demoted, not deleted")
    check(pg.evaluate("HD[0].pts.length") == 5, "no point is ever lost to a rename")
    pg.keyboard.press("0")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts[0].role") == "",
          "0 turns a landmark back into an ordinary rim point")

    print("\n[5] facing is asserted, and catches a swap")
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    check(pg.evaluate("FACE") == "left", "facing defaults to anterior-left")
    check(pg.evaluate("fits[0].swapped") is False, "A on the left is not flagged")
    check(abs(pg.evaluate("fits[0].aptilt")) < 0.5, "A and P are level, so no tilt warning")
    pg.keyboard.press("m")
    pg.wait_for_timeout(150)
    check(pg.evaluate("FACE") == "right", "m flips which way the patient faces")
    check(pg.evaluate("fits[0].swapped") is True,
          "and the very same marks are now flagged as A/P swapped")
    check("SWAPPED" in pg.eval_on_selector("#fitread", "e=>e.textContent"),
          "the header says so in words, not just in colour")
    pg.keyboard.press("m")
    pg.wait_for_timeout(150)
    check(pg.evaluate("localStorage.getItem('annot_face')") == "left",
          "and the choice sticks to the next film")

    print("\n[6] a skipped extreme is derived, never guessed")
    put([(270, 'S'), (0, 'P'), (300, ''), (330, '')])
    pg.evaluate("() => { HD[0].skip['A']=true; refit(); draw(); readout(); }")
    check(pg.evaluate("nextRole(HD[0])") == "",
          "a skipped extreme stops being asked for")
    check(pg.evaluate("fits[0] !== null"),
          "and the circle is still determined from what is left")
    pg.evaluate("() => { HD[0].pts.length=0; refit(); }")
    pg.keyboard.press("u")
    pg.wait_for_timeout(120)
    check(not pg.evaluate("HD[0].skip['A']"),
          "u undoes a skip -- it is a decision too, and it is invisible on the film")

    print("\n[7] latency")
    lat = pg.evaluate("""([cx,cy,cr,asp]) => {
        const t=(f,n)=>{const s=performance.now(); for(let i=0;i<n;i++) f(i);
                        return (performance.now()-s)/n;};
        HD=[newHd()]; acur=0;
        for(let i=0;i<8;i++){const a=i*Math.PI/4;
          HD[0].pts.push({x:cx+cr*Math.cos(a), y:cy+cr*asp*Math.sin(a), role:''});}
        return {probe: t(i=>qcAt(cx*img.width+((i*37)%60)-30,
                                 cy*img.height+((i*53)%60)-30), 200),
                refit: t(()=>refit(), 60),
                draw:  t(()=>draw(), 40),
                full:  t(()=>{refit();draw();readout();}, 30)};
      }""", [CX, CY, CR, W / H])
    print("       " + "  ".join(f"{k}={v:.2f}ms" for k, v in lat.items()))
    check(lat["probe"] < 4, f"hover probe is imperceptible ({lat['probe']:.2f} ms)")
    check(lat["refit"] < 8, f"a refit of 8 points is cheap ({lat['refit']:.2f} ms)")
    check(lat["draw"] < 25, f"a redraw of the film is well under a frame "
                            f"({lat['draw']:.2f} ms)")
    check(lat["full"] < 33,
          f"so dragging a point stays inside one 30 fps frame ({lat['full']:.2f} ms)")
    tqc = pg.evaluate("() => { const s=performance.now(); buildQC(); "
                      "return performance.now()-s; }")
    print(f"       buildQC={tqc:.0f}ms (once per film, after first paint)")
    check(tqc < 1500, f"the one-off Sobel and dilation pass is tolerable ({tqc:.0f} ms)")

    put([(180, 'A'), (270, 'S'), (0, 'P')])
    sx = box["x"] + box["width"] * rim(270)[0]
    sy = box["y"] + box["height"] * rim(270)[1]
    y0 = pg.evaluate("roleAt(HD[0],'S').y")
    pg.mouse.move(sx, sy)
    pg.mouse.down()
    pg.mouse.move(sx + 40, sy + 15, steps=30)
    pg.mouse.up()
    pg.wait_for_timeout(150)
    check(pg.evaluate("roleAt(HD[0],'S').y") > y0,
          "and a real 30-step drag moves the landmark it grabbed")

    print("\n[8] two heads, and what gets submitted")
    put([(180, 'A'), (270, 'S'), (0, 'P'), (225, ''), (315, '')])
    pg.keyboard.press("h")
    pg.wait_for_timeout(150)
    check(pg.evaluate("HD.length") == 2 and pg.evaluate("acur") == 1,
          "h starts a second head")
    put([(180, 'A'), (270, 'S'), (0, 'P')], head=1, dx=90.0 / W, dy=12.0 / H)
    sent = []
    pg.on("request", lambda r: sent.append(r) if r.method == "POST"
          and r.url.endswith("/submit") else None)
    pg.evaluate("send()")
    pg.wait_for_timeout(1000)
    m = re.search(r'name="points"\r?\n\r?\n(.*?)\r?\n--', sent[0].post_data if sent
                  else "", re.S)
    body = json.loads(m.group(1)) if m else {}
    print("       payload keys:", sorted(body))
    check(len(body.get("heads", [])) == 2, "two fitted centres are submitted")
    check(len(body.get("landmarks", [])) == 2, "with a landmark set for each head")
    lm = body.get("landmarks", [{}])[0]
    check(set(lm) == {"A", "S", "P"}, f"named A, S and P ({sorted(lm)})")
    check(all(lm[r]["src"] == "obs" for r in ("A", "S", "P")),
          "each marked as observed rather than derived")
    check(body.get("facing") == ["left", "left"],
          f"facing travels with the read ({body.get('facing')})")
    check(len(body.get("extra", [[]])[0]) == 2,
          "extra rim points ride along, separately from the landmarks")
    check(body.get("tool") == "arc", "and the tool used is stamped on it")
    # the shape everything downstream already reads must be untouched
    check(all(isinstance(q, list) and len(q) == 2 for q in body.get("heads", [])),
          "heads is still a plain list of centres, so agreement and the board still work")
    pg.screenshot(path=str(SHOTS / "arc_two_heads.png"))

    print("\n[9] guard rails")
    pg.evaluate("""() => { HD=[newHd()]; acur=0;
        HD[0].pts.push({x:0.4,y:0.30,role:'A'},{x:0.5,y:0.30,role:'S'},
                       {x:0.6,y:0.30,role:'P'}); refit(); }""")
    check(pg.evaluate("fits[0] === null"),
          "three collinear points define no circle, and none is invented")
    put([(180, ''), (270, ''), (0, '')])
    pg.evaluate("send()")
    pg.wait_for_timeout(300)
    check("still to mark" in pg.eval_on_selector("#msg", "e=>e.textContent"),
          "submitting with nothing named says exactly what is missing")
    put([(180, 'A'), (270, 'S'), (0, 'P')], head=1, dx=0.05)
    pg.evaluate("() => { HD[1].pts.length=1; refit(); }")
    pg.evaluate("send()")
    pg.wait_for_timeout(300)
    check("at least 3" in pg.eval_on_selector("#msg", "e=>e.textContent"),
          "and a half-marked second head is refused rather than submitted as a stub")
    pg.screenshot(path=str(SHOTS / "arc_qc.png"))
    br.close()

srv.should_exit = True
print(f"\n  shots in {SHOTS}")
print("FAILED: " + "; ".join(fails) if fails else "\nALL CHECKS PASSED")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if fails else 0)
