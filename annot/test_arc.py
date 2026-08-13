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

QC_WEAK = 40          # mirrors ui.py; the hint must clear it
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

    # Pure geometry -- these RATIOS hold whatever the per-click noise actually is, so they
    # are the defensible half of the "add more points" advice. The absolute numbers are
    # not: they scale with an assumed sigma that this pilot exists to measure.
    asp = fit_at([180, 270, 0])
    sup2 = fit_at([180, 270, 0, 225, 315])       # two more, bunched superiorly
    inf1 = fit_at([180, 270, 0, 90])             # ONE more, at the inferior margin
    print(f"       A/S/P {asp['two']:.5f}   +2 superior {sup2['two']:.5f}   "
          f"+1 inferior {inf1['two']:.5f}")
    check(abs(asp["two"] / even["two"] - 1.5) < 0.02,
          f"A/S/P is exactly 1.5x looser than three evenly spaced points "
          f"({asp['two'] / even['two']:.3f}x) -- geometry, not an assumption")
    check(sup2["two"] > 0.85 * asp["two"],
          f"TWO extra points bunched superiorly barely help "
          f"({sup2['two'] / asp['two']:.2f}x) -- 'add a few more anywhere' was wrong")
    check(inf1["two"] < 0.6 * asp["two"] and inf1["two"] < even["two"],
          f"but ONE at the inferior margin beats even the ideal triple "
          f"({inf1['two'] / asp['two']:.2f}x)")
    fit_at([180, 270, 0])
    pg.wait_for_timeout(150)
    tag = pg.eval_on_selector("#fitread", "e => e.textContent")
    print(f"       header advice: {tag!r}")
    check("add a point" in tag,
          "so the header names a part of the rim rather than asking for more of the same")
    # THE HINT MUST NEVER POINT AT BONE THAT IS NOT THERE. Conditioning alone always picks
    # the inferior pole, and on a femoral head that is inside the neck -- measured on the
    # segmented CT, femur continues outward past the sphere at 6 o'clock. So the suggested
    # angle has to have visible cortex on this film.
    hint = pg.evaluate("""() => {
        const t = bestNextAngle(HD[0].pts, fits[0]);
        if (t === null) return null;
        const q = qcAt(fits[0].a + fits[0].R*Math.cos(t),
                       fits[0].b + fits[0].R*Math.sin(t));
        return {deg: Math.round(((t*180/Math.PI)%360+360)%360),
                edge: q ? q.pct : null, name: rimName(t)};
      }""")
    print(f"       hint: {hint}")
    check(hint is not None and hint["edge"] is not None and hint["edge"] >= QC_WEAK,
          f"the suggested spot has visible cortex on this film ({hint})")
    check(hint["deg"] == 90,
          "on a film whose whole rim is visible it does name the inferior pole, which is "
          "the geometric optimum")
    # Now take the inferior rim away, which is what a real femoral neck does to a real
    # head. The hint must MOVE rather than keep naming bone that is no longer there.
    moved = pg.evaluate("""() => {
        const f=fits[0], saved=[];
        const px=Math.round(f.a), py=Math.round(f.b+f.R), rad=Math.round(f.R*0.45);
        for(let dy=-rad; dy<=rad; dy++) for(let dx=-rad; dx<=rad; dx++){
          if(dx*dx+dy*dy>rad*rad) continue;
          const i=(py+dy)*QCW+(px+dx);
          saved.push([i, GMAX[i]]); GMAX[i]=QNOISE;
        }
        const t=bestNextAngle(HD[0].pts, fits[0]);
        saved.forEach(([i,v])=>{GMAX[i]=v;});
        return t===null ? null : {deg: Math.round(((t*180/Math.PI)%360+360)%360),
                                  name: rimName(t)};
      }""")
    print(f"       with the inferior rim hidden: {moved}")
    check(moved is not None and moved["deg"] != 90,
          f"with the inferior rim hidden the hint moves off it ({moved})")
    check(moved is not None and 45 <= moved["deg"] <= 135,
          f"to the low rim on one side or the other, not back up to the top ({moved})")

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

    print("\n[6b] second heads are the reader's call")
    pg.evaluate("() => { HD=[newHd()]; acur=0; sel2=null; HIST=[]; offered=false; "
                "badFit=false; fits=[]; peek=false; draw(); }")
    box = pg.locator("#c").bounding_box()

    def click(deg, dx=0.0):
        q = rim(deg)
        pg.mouse.click(box["x"] + box["width"] * (q[0] + dx),
                       box["y"] + box["height"] * q[1])
        pg.wait_for_timeout(80)

    for g in (180, 270, 0):
        click(g)
    check(pg.evaluate("HD.length") == 1, "finishing A/S/P does not create a second head")
    click(225)
    click(315)
    check(pg.evaluate("HD.length") == 1 and pg.evaluate("HD[0].pts.length") == 5,
          "later clicks keep going to the head in hand")
    pg.keyboard.press("h")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD.length") == 2 and pg.evaluate("acur") == 1,
          "h is what starts the second head, when the reader says so")
    pg.keyboard.press("h")
    pg.wait_for_timeout(120)
    check(pg.evaluate("acur") == 0, "and h switches back")

    # a genuine second head fitted on top of the first is still called out
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    put([(180, 'A'), (270, 'S'), (0, 'P')], head=1, dx=0.004)
    pg.wait_for_timeout(150)
    tag = pg.eval_on_selector("#fitread", "e => e.textContent")
    check("SAME head" in tag,
          "two centres almost on top of each other are called out, not accepted")

    print("\n[6c] undo and delete")
    pg.evaluate("() => { HD=[newHd()]; acur=0; sel2=null; HIST=[]; "
                "autoSwitched=false; pendingSwitch=false; fits=[]; draw(); }")
    for g in (180, 270, 0):
        click(g)
    click(225)                                   # comes back to head 1
    check(pg.evaluate("HD[0].pts.length") == 4, "four points on one head")
    pg.keyboard.press("Control+z")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts.length") == 3, "Ctrl+Z undoes the last click")
    pg.keyboard.press("Control+z")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts.length") == 2,
          "and the next one goes back another click")

    # a RENAME must undo to the previous name, not delete the landmark -- the thing a
    # pop-the-last-point undo gets wrong
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    pg.evaluate("() => { HIST=[]; sel2={arc:0,i:0}; }")
    pg.keyboard.press("2")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts[0].role") == "S", "renaming A to S works")
    check(pg.evaluate("HD[0].pts.filter(p=>p.role==='S').length") == 1,
          "and demotes whatever used to be S")
    pg.keyboard.press("Control+z")
    pg.wait_for_timeout(120)
    check(pg.evaluate("ROLES.filter(r=>roleAt(HD[0],r)).join('')") == "ASP"
          and pg.evaluate("HD[0].pts.length") == 3,
          "and Ctrl+Z puts BOTH names back rather than deleting a point")

    # click a mark, press Delete
    click(270)
    check(pg.evaluate("sel2 !== null"), "clicking an existing point selects it")
    pg.keyboard.press("Delete")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts.length") == 2
          and pg.evaluate("roleAt(HD[0],'S')") is None,
          "Delete removes the selected landmark")
    pg.keyboard.press("Control+z")
    pg.wait_for_timeout(120)
    check(pg.evaluate("roleAt(HD[0],'S') !== null"), "and Ctrl+Z brings it back")
    pg.evaluate("sel2={arc:0,i:0}")
    pg.keyboard.press("Backspace")
    pg.wait_for_timeout(120)
    check(pg.evaluate("HD[0].pts.length") == 2, "Backspace does the same")
    check(pg.evaluate("location.pathname") == "/",
          "and does NOT navigate the browser back, which would lose the whole read")
    pg.keyboard.press("Control+z")
    pg.wait_for_timeout(120)

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
    put([(180, 'A'), (270, 'S'), (0, 'P'), (225, ''), (315, ''), (90, '')])
    pg.keyboard.press("h")
    pg.wait_for_timeout(150)
    check(pg.evaluate("HD.length") == 2 and pg.evaluate("acur") == 1,
          "h starts a second head")
    put([(180, 'A'), (270, 'S'), (0, 'P'), (90, '')], head=1,
        dx=90.0 / W, dy=12.0 / H)
    sent = []
    pg.on("request", lambda r: sent.append(r) if r.method == "POST"
          and r.url.endswith("/submit") else None)
    # Submit is gated when the tool's own error bar is outside tolerance, so press
    # until it goes: the second press is the deliberate "send it anyway".
    pg.evaluate("send()")
    pg.wait_for_timeout(400)
    if not sent:
        pg.evaluate("send()")
        pg.wait_for_timeout(900)
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
    check(len(body.get("extra", [[]])[0]) == 3,
          "extra rim points ride along, separately from the landmarks")
    check(body.get("tool") == "arc", "and the tool used is stamped on it")
    # the shape everything downstream already reads must be untouched
    check(all(isinstance(q, list) and len(q) == 2 for q in body.get("heads", [])),
          "heads is still a plain list of centres, so agreement and the board still work")
    pg.screenshot(path=str(SHOTS / "arc_two_heads.png"))

    print("\n[8a] what the first round of the pilot broke")
    pg.evaluate("() => { HD=[newHd()]; acur=0; sel2=null; HIST=[]; offered=false; "
                "badFit=false; fits=[]; peek=false; draw(); }")
    box = pg.locator("#c").bounding_box()

    def clk(deg, dx=0.0):
        # re-measure every time: a successful submit calls load(), which restores the pan
        # and scrolls the canvas, so a box captured earlier aims at the wrong pixels
        b = pg.locator("#c").bounding_box()
        q = rim(deg)
        pg.mouse.click(b["x"] + b["width"] * (q[0] + dx),
                       b["y"] + b["height"] * q[1])
        pg.wait_for_timeout(80)

    # NO AUTOMATIC HAND-OFF. Three of four readers marked two heads on ~90% of films
    # while the fourth marked one on 91%: being told "now marking HEAD 2" reads as an
    # instruction that a second head exists, and only the reader can see whether it does.
    for g in (180, 270, 0):
        clk(g)
    check(pg.evaluate("HD.length") == 1 and pg.evaluate("acur") == 0,
          "naming A, S and P does NOT start a second head")
    m = pg.eval_on_selector("#msg", "e => e.textContent")
    check("press h" in m and "TWO" in m,
          f"it offers one instead, and says what a second head looks like ({m!r})")
    clk(225)
    check(pg.evaluate("HD.length") == 1 and pg.evaluate("HD[0].pts.length") == 4,
          "and the next click is a rim point for the head in hand, not a new head")

    # THE ERROR BAR NOW BINDS. 80% of pilot heads were submitted with the tool's own
    # 2 sigma above tolerance, and advice everyone ignores is not advice.
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    two = pg.evaluate("2*fits[0].e1/img.width")
    check(two > 0.005, f"a bare A/S/P fit is above tolerance ({two:.4f})")
    sent = []
    pg.on("request", lambda r: sent.append(r) if r.method == "POST"
          and r.url.endswith("/submit") else None)
    pg.evaluate("send()")
    pg.wait_for_timeout(400)
    check(not sent, "Submit on a loose fit does NOT send it")
    m = pg.eval_on_selector("#msg", "e => e.textContent")
    print(f"       first Submit says: {m!r}")
    check("Submit again" in m and "margin" in m,
          "it says how loose, where to click, and that a second press sends it anyway")
    check("warnbtn" in pg.eval_on_selector("#btnsend", "e => e.className"),
          "and the Submit button itself carries the warning")
    pg.evaluate("send()")
    pg.wait_for_timeout(900)
    check(len(sent) >= 1,
          "pressing Submit again does send it -- a hard film is not blocked")

    # ...and adding points re-arms the gate rather than leaving it disarmed
    sent.clear()
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    pg.evaluate("send()")
    pg.wait_for_timeout(300)
    clk(90)
    # read the fit BEFORE submitting: a successful send calls load(), which clears it
    tight = pg.evaluate("2*fits[0].e1/img.width")
    pg.evaluate("send()")
    pg.wait_for_timeout(600)
    check(len(sent) >= 1 and tight <= 0.005,
          f"a rim point at the inferior margin takes it inside tolerance and it goes "
          f"({tight:.4f})")

    print("\n[8c] the film stops getting crowded")
    put([(180, 'A'), (270, 'S'), (0, 'P')])
    put([(180, 'A'), (270, 'S'), (0, 'P')], head=1, dx=0.13)
    pg.evaluate("() => { acur=1; draw(); }")
    pg.wait_for_timeout(200)
    marks = pg.evaluate("""() => {
        // count non-background pixels in a box around head 1, which is now the INACTIVE
        // head: fewer marks means fewer coloured pixels over its cortex
        const g=C.getContext('2d'), f=fits[0], R=Math.round(f.R*1.6);
        const d=g.getImageData(Math.round(f.a-R), Math.round(f.b-R), 2*R, 2*R).data;
        let n=0;
        for(let i=0;i<d.length;i+=4){
          // green channel well above red = the head-1 marker colour
          if(d[i+1]>d[i]+40 && d[i+1]>90) n++;
        }
        return n;
      }""")
    pg.evaluate("() => { acur=0; draw(); }")
    pg.wait_for_timeout(200)
    marks_active = pg.evaluate("""() => {
        const g=C.getContext('2d'), f=fits[0], R=Math.round(f.R*1.6);
        const d=g.getImageData(Math.round(f.a-R), Math.round(f.b-R), 2*R, 2*R).data;
        let n=0;
        for(let i=0;i<d.length;i+=4){
          if(d[i+1]>d[i]+40 && d[i+1]>90) n++;
        }
        return n;
      }""")
    print(f"       head-1 marker pixels: active {marks_active}, inactive {marks}")
    check(marks < marks_active * 0.6,
          f"the head you are not working on fades right back "
          f"({marks_active} -> {marks} marker pixels)")
    check(marks > 0, "but stays visible, so the same arc is not marked twice")

    # hold b to blink every mark off
    pg.evaluate("() => { acur=0; draw(); }")
    pg.keyboard.down("b")
    pg.wait_for_timeout(200)
    blank = pg.evaluate("""() => {
        const g=C.getContext('2d'), f=fits[0], R=Math.round(f.R*1.6);
        const d=g.getImageData(Math.round(f.a-R), Math.round(f.b-R), 2*R, 2*R).data;
        let n=0;
        for(let i=0;i<d.length;i+=4) if(d[i+1]>d[i]+40 && d[i+1]>90) n++;
        return n;
      }""")
    check(blank == 0, f"holding b blinks every mark off ({blank} marker pixels left)")
    pg.keyboard.up("b")
    pg.wait_for_timeout(200)
    check(pg.evaluate("peek === false"), "and letting go puts them back")


    print("\n[8b] calibration: the one film with a known answer")
    pg.evaluate("() => calibrate()")
    pg.wait_for_function("calMode === true", timeout=20000)
    pg.wait_for_timeout(700)
    check(pg.evaluate("cur.case_id") == "CALIBRATION", "the calibration film loads")
    # The truth must not be reachable before the attempt, or this is a reading test
    # rather than a measurement of the reader.
    check(pg.evaluate("calTruth === null"),
          "and no truth reaches the page before it is attempted")
    body = pg.evaluate("async () => (await fetch('/calib',{headers:H()})).text()")
    check("landmarks" not in body and '"centre"' not in body,
          f"nor does the /calib payload leak it ({body[:70]})")

    # mark it deliberately 6 px low and see whether it says so, in millimetres
    T = json.loads((ROOT / "calib_truth.json").read_text())
    off = 6.0 / T["h"]
    pg.evaluate("""([t,off]) => {
        HD=[newHd()]; acur=0; sel2=null;
        ['A','S','P'].forEach(r=>HD[0].pts.push(
            {x:t.landmarks[r][0], y:t.landmarks[r][1]+off, role:r}));
        refit(); draw(); }""", [T, off])
    pg.evaluate("send()")
    pg.wait_for_function("calTruth !== null", timeout=20000)
    pg.wait_for_timeout(400)
    txt = pg.eval_on_selector("#calbox", "e => e.textContent")
    print(f"       calibration says: {txt[:110]!r}")
    check(not pg.eval_on_selector("#calbox", "e => e.hidden"),
          "submitting shows the score")
    mm = 6.0 * T["mm_per_px"]
    check(f"{mm:.1f} mm" in txt,
          f"and the centre error is right: marked {mm:.1f} mm low, and it says so")
    check(pg.evaluate("calTruth !== null"),
          "the truth comes back only AFTER the attempt, and is drawn on the film")
    check("attempt 1" in txt.lower(), "attempts are numbered")

    # a wrong facing silently mirrors every A/P label, so it has to be called out
    pg.evaluate("() => { FACE = FACE === 'left' ? 'right' : 'left'; }")
    pg.evaluate("send()")
    pg.wait_for_timeout(900)
    check("wrong way round" in pg.eval_on_selector("#calbox", "e => e.textContent"),
          "marking with the wrong facing is called out")
    pg.evaluate("() => { FACE='left'; leaveCalib(); }")
    pg.wait_for_timeout(900)
    check(pg.evaluate("calMode === false"), "and you can get back to reading")


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
