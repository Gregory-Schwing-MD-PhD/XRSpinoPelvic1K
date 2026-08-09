"""
Femoral-head annotation Space â€” two clicks per lateral radiograph, distributed.

Mirrors the CTSpinoPelvic1K review service deliberately: HuggingFace token as identity,
a dataset repo as the ledger, claim slots with a TTL, and double annotation with
adjudication on disagreement. Same shape, so the same people already know how to use it.

  HF_TOKEN        write token for the ledger dataset (Space secret)
  ANNOT_REPO      dataset repo holding cases/ and annotations/
  IMAGE_REPO      dataset repo holding the films  (read-only)
  ADJUDICATORS    comma/space-separated HF usernames allowed to adjudicate

WHY THIS AND NOT ITK-SNAP
-------------------------
ITK-SNAP is a 3-D volume tool. Annotating two points on a 2-D radiograph through it means
converting every film to a single-slice volume, installing software, and training people
on a segmentation UI to place two dots. A browser canvas is two clicks and no install,
and the Space is already how this group works.

NO AUTOMATIC PROPOSAL IS EVER SHOWN. The whole purpose of these annotations is an
INDEPENDENT reference for a hip point predicted from synthetic data. Displaying the
model's guess -- or the classical circle fit -- as a starting position would anchor the
annotator to it, and agreement would then measure suggestibility rather than anatomy.
This is the single most important design decision in the tool, and it is why the file
contains no call to any predictor.

WHAT IS COLLECTED, AND WHY BOTH HEADS
-------------------------------------
Left and right femoral head centres separately, not the midpoint. On a true lateral the
two heads superimpose, but rotation separates them -- and the bicoxofemoral point is
DEFINED as the midpoint of the two centres (Legaye & Duval-Beaupere 1998). Collecting
both lets the midpoint be derived, lets their separation flag an oblique film, and lets a
reader mark only one when the other is genuinely invisible.

CRITERIA — shown in the app and repeated here so the definition lives with the code.

ANATOMICAL DEFINITION
  The bicoxofemoral (hip) axis is the line joining the CENTRES of the two femoral heads.
  The point used for PI, PT and SS is the MIDPOINT of that line (Legaye & Duval-Beaupere,
  Eur Spine J 1998). It is a geometric centre, not a surface point and not a palpable
  landmark: the femoral head is very nearly a sphere, so its projection on any radiograph
  is a circle and the point wanted is the centre of that circle.

HOW TO FIND IT ON A LATERAL FILM
  1. Identify the femoral head: the round dense structure below and anterior to the S1
     endplate, seated in the acetabulum.
  2. Trace the SUBCHONDRAL CORTICAL MARGIN -- the thin dense arc of the articular surface.
     That arc is the circle to centre on. This is the Mose concentric-circle method done
     by eye.
  3. Place the point at the centre of curvature of that arc, NOT at the densest part of
     the shadow. Overlap with the acetabulum and the opposite head makes the brightest
     region sit medial to the true centre.

WHAT NOT TO USE
  * the fovea capitis -- the medial notch where ligamentum teres attaches. It is a defect
    in the sphere; centring on it pulls the point medially.
  * the greater trochanter, the femoral neck, or the head-neck junction. None of them is
    part of the sphere.
  * the acetabular roof or teardrop. They belong to the pelvis, not the head.

ONE HEAD OR TWO
  On a well-positioned lateral the two heads superimpose almost exactly and you will see
  one circle -- mark it as LEFT and leave RIGHT empty. Rotation separates them into two
  overlapping circles; then mark both, and their midpoint is derived automatically. Their
  separation is itself recorded, because a wide separation means an oblique film and the
  spinopelvic parameters from it are less trustworthy.

WHEN TO SKIP
  Prosthesis, both heads outside the collimated field, or a film so underexposed that the
  cortical arc cannot be traced. Skipping is a valid answer. A guessed centre becomes a
  fabricated error in whatever this set is used to measure.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

from fastapi import Depends, FastAPI, Form, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

import store as store_mod  # noqa: E402  (sibling, as in review_service)

ANNOT_REPO = os.environ.get("ANNOT_REPO", "")
IMAGE_REPO = os.environ.get("IMAGE_REPO", "")
CLAIM_TTL = int(os.environ.get("CLAIM_TTL_SECONDS", 3 * 24 * 3600))
N_PRIMARY = 2
# Consensus tolerance, as a fraction of image width. Judged in NORMALISED units so it
# means the same thing on every film regardless of pixel size.
#
# 0.005, not the 0.02 this started at, and the difference matters. On a typical BUU film
# (2012 px across roughly 400 mm, so ~5 px/mm) with the S1 midpoint about 125 mm from the
# hip axis, a hip displacement of d mm moves PI and PT by atan(d/125):
#
#   0.02   -> 40 px -> 8.0 mm -> up to 3.7 deg of PI error
#   0.01   -> 20 px -> 4.0 mm -> up to 1.8 deg
#   0.005  -> 10 px -> 2.0 mm -> up to 0.9 deg
#
# The model being validated has an SS MAE of 2.3 deg. A reference whose own two readers
# may disagree by 3.7 deg cannot measure a 2.3 deg error -- the ruler would be coarser
# than the thing it is measuring. 0.005 keeps the reference's internal disagreement well
# under the effect size; anything looser silently caps the precision of every conclusion
# drawn from it.
CONSENSUS_TOL = float(os.environ.get("CONSENSUS_TOL", 0.005))
ADJUDICATORS = {u.strip().lower()
                for u in os.environ.get("ADJUDICATORS", "").replace(",", " ").split()
                if u.strip()}

app = FastAPI(title="Femoral head annotation")
_WHOAMI: dict = {}


def _store():
    if not ANNOT_REPO or not os.environ.get("HF_TOKEN"):
        raise HTTPException(500, "ANNOT_REPO and HF_TOKEN must be set")
    return store_mod.ReviewStore(
        store_mod.HFBackend(ANNOT_REPO, os.environ["HF_TOKEN"]))


def _hf_username(token: str) -> Optional[str]:
    """Verified HF username, cached by sha256(token). The token itself is never stored."""
    if not token:
        return None
    k = hashlib.sha256(token.encode()).hexdigest()
    hit = _WHOAMI.get(k)
    if hit and time.time() - hit[1] < 900:
        return hit[0]
    try:
        from huggingface_hub import HfApi
        name = HfApi().whoami(token=token).get("name")
    except Exception:                                        # noqa: BLE001
        return None
    if name:
        _WHOAMI[k] = (name, time.time())
    return name


def user(authorization: str = Header(default="")) -> dict:
    tok = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    name = _hf_username(tok)
    if not name:
        raise HTTPException(401, "send your HuggingFace token as: Authorization: Bearer hf_...")
    return {"id": name, "role": "adjudicator" if name.lower() in ADJUDICATORS else "primary"}


def _now() -> float:
    return time.time()


def _claimable_slot(case: dict, who: str) -> Optional[str]:
    """Which primary slot this annotator may take, or None.

    Same rule as the review service: a person cannot hold two slots on one case (that
    would make "double annotation" one person twice), and an expired unsubmitted claim is
    reclaimable so an abandoned case does not sit forever.
    """
    slots = case.setdefault("slots", {})
    if any(s.get("annotator") == who for s in slots.values()):
        return None
    for k in ("1", "2"):
        s = slots.get(k)
        if s is None:
            return k
        if not s.get("done") and s.get("expires_at", 0) < _now():
            return k
    return None


@app.get("/example")
def example():
    """A worked example, served from a bundled PNG.

    Built from a DRR, NOT from a real film, and that is the point: on a DRR the hip point
    is a 3-D sphere fitted to the femoral head and projected, so the example shows an
    objectively correct answer rather than one annotator's opinion. Using a marked-up real
    radiograph would put whoever marked it into every reader's head.

    It is a fixed teaching image and never one of the films being annotated, so it cannot
    anchor a specific case the way an on-image proposal would.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("example_femhead.png", "static/example_femhead.png"):
        fp = os.path.join(here, name)
        if os.path.exists(fp):
            return Response(open(fp, "rb").read(), media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404, "example image not bundled")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/next")
def next_case(u: dict = Depends(user)):
    st = _store()
    for case in st.list_cases():
        if case.get("final"):
            continue
        slot = _claimable_slot(case, u["id"])
        if slot is None:
            continue
        case["slots"][slot] = {"annotator": u["id"], "claimed_at": _now(),
                               "expires_at": _now() + CLAIM_TTL, "done": False}
        st.put_case(case)
        return {"case_id": case["case_id"], "slot": slot,
                "image_url": f"/image/{case['case_id']}"}
    return JSONResponse({"detail": "nothing left to annotate"}, status_code=404)


# Display width. A BUU film is ~2000x2400 and several MB; sending that raw makes every
# "next film" a multi-second wait, which is the difference between annotating 120 films in
# an afternoon and giving up. At 1100 px one displayed pixel is under 2 source pixels,
# and the consensus tolerance is 0.005 of width (~10 source px), so the downscale costs
# nothing that matters. Coordinates are normalised anyway, so precision is set by the
# MAGNIFIER, not by the transport size.
DISPLAY_W = int(os.environ.get("DISPLAY_W", 1100))
_IMG_CACHE: dict = {}


def _render_jpeg(case_id: str) -> bytes:
    if case_id in _IMG_CACHE:
        return _IMG_CACHE[case_id]
    from io import BytesIO

    from huggingface_hub import hf_hub_download
    from PIL import Image
    try:
        p = hf_hub_download(IMAGE_REPO, f"images/{case_id}.jpg", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
    except Exception as exc:                                 # noqa: BLE001
        raise HTTPException(404, f"image not found: {exc}")
    im = Image.open(p).convert("L")
    if im.width > DISPLAY_W:
        im = im.resize((DISPLAY_W, round(im.height * DISPLAY_W / im.width)),
                       Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    data = buf.getvalue()
    if len(_IMG_CACHE) < 400:              # bounded: a Space has finite memory
        _IMG_CACHE[case_id] = data
    return data


@app.get("/image/{case_id}")
def image(case_id: str, u: dict = Depends(user)):
    """Stream the film THROUGH the Space so annotators never need read access to the
    image repo, exactly as the review service streams labels."""
    return Response(_render_jpeg(case_id), media_type="image/jpeg",
                    headers={"Cache-Control": "private, max-age=3600"})


@app.get("/peek")
def peek(u: dict = Depends(user)):
    """The next case id WITHOUT claiming it, so the browser can prefetch its image.

    Claiming on prefetch would hand every annotator a second case the moment they opened
    one, and abandoned claims would pile up behind the TTL.
    """
    st = _store()
    for case in st.list_cases():
        if case.get("final"):
            continue
        if _claimable_slot(case, u["id"]) is not None:
            return {"case_id": case["case_id"]}
    return JSONResponse({"detail": "none"}, status_code=404)


@app.post("/submit")
def submit(case_id: str = Form(...), slot: str = Form(...), points: str = Form(...),
           u: dict = Depends(user)):
    """points: JSON {"left":[x,y]|null, "right":[x,y]|null, "note":str, "w":W, "h":H}

    Coordinates arrive NORMALISED to the displayed image and are stored that way. The
    browser scales the film to fit a screen of unknown size, so raw canvas pixels would
    silently encode each annotator's window size.
    """
    st = _store()
    case = st.get_case(case_id)
    if case is None:
        raise HTTPException(404, "unknown case")
    s = case.get("slots", {}).get(slot)
    if not s or s.get("annotator") != u["id"]:
        raise HTTPException(403, "that slot is not yours")
    p = json.loads(points)
    if p.get("left") is None and p.get("right") is None:
        raise HTTPException(400, "mark at least one femoral head, or use /skip")
    s.update(done=True, submitted_at=_now(), points=p)

    done = [case["slots"][k] for k in ("1", "2")
            if case.get("slots", {}).get(k, {}).get("done")]
    if len(done) >= N_PRIMARY:
        case["agree"] = _agreement(done[0]["points"], done[1]["points"])
        if case["agree"] is not None and case["agree"] <= CONSENSUS_TOL:
            case["final"] = {"points": _mean_points(done[0]["points"], done[1]["points"]),
                             "by": "consensus"}
    st.put_case(case)
    return {"ok": True, "status": "final" if case.get("final") else "awaiting second"}


@app.post("/skip")
def skip(case_id: str = Form(...), slot: str = Form(...), reason: str = Form(""),
         u: dict = Depends(user)):
    """Release a film whose femoral heads genuinely cannot be seen.

    A forced guess on an unreadable film is worse than no annotation: this set exists to
    be a reference, and a fabricated reference point silently becomes a fabricated error
    for the model being validated.
    """
    st = _store()
    case = st.get_case(case_id)
    if case is None:
        raise HTTPException(404, "unknown case")
    case.setdefault("skipped_by", []).append({"by": u["id"], "reason": reason})
    case.get("slots", {}).pop(slot, None)
    st.put_case(case)
    return {"ok": True}


def _agreement(a: dict, b: dict) -> Optional[float]:
    import math
    ds = []
    for k in ("left", "right"):
        if a.get(k) and b.get(k):
            ds.append(math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]))
    return max(ds) if ds else None


def _mean_points(a: dict, b: dict) -> dict:
    out = {}
    for k in ("left", "right"):
        if a.get(k) and b.get(k):
            out[k] = [(a[k][0] + b[k][0]) / 2, (a[k][1] + b[k][1]) / 2]
        else:
            out[k] = a.get(k) or b.get(k)
    return out


PAGE = """<!doctype html><meta charset=utf-8>
<title>Femoral head annotation</title>
<style>
 body{font:14px system-ui;margin:0;background:#111;color:#eee}
 header{padding:8px 12px;background:#1b1b1f;display:flex;gap:10px;align-items:center;
        flex-wrap:wrap}
 #wrap{position:relative;display:inline-block}
 canvas{cursor:none;max-height:86vh}
 #loupe{position:absolute;width:190px;height:190px;border:2px solid #0072B2;
        border-radius:50%;pointer-events:none;display:none;box-shadow:0 0 12px #000;
        background:#000;z-index:5}
 button{padding:6px 12px;border-radius:6px;border:0;cursor:pointer}
 .go{background:#0072B2;color:#fff}.sk{background:#666;color:#fff}
 #msg{margin-left:auto;color:#9ad}
 input{background:#222;color:#eee;border:1px solid #444;border-radius:5px;padding:5px}
 kbd{background:#333;padding:1px 5px;border-radius:3px;font-size:11px}
</style>
<header>
  <input id=tok type=password placeholder="hf_... token" size=24>
  <button class=go onclick=load()>Next</button>
  <span>click <b style="color:#00E5A0">LEFT</b> head then
        <b style="color:#FF3B30">RIGHT</b></span>
  <button onclick=undo()>Undo <kbd>u</kbd></button>
  <button class=go onclick=send()>Submit <kbd>enter</kbd></button>
  <button class=sk onclick=skip()>Can't see <kbd>s</kbd></button>
  <span id=msg></span>
</header>
<details id=help open>
 <summary style="cursor:pointer;padding:8px 12px;background:#22222a">
   Criteria — read once, then collapse</summary>
 <div style="display:flex;gap:18px;padding:10px 14px;background:#191920">
  <div style="max-width:640px;line-height:1.5">
   <b>What the point is.</b> The hip axis is the line joining the <i>centres</i> of the two
   femoral heads; the point used for PI/PT/SS is its midpoint
   (Legaye &amp; Duval-Beaupère 1998). The femoral head is very nearly a sphere, so its
   projection is a circle — you are marking <b>the centre of that circle</b>.
   <br><br>
   <b>How to find it.</b>
   <ol style="margin:4px 0 0 18px;padding:0">
    <li>Find the round dense head below and anterior to the S1 endplate, seated in the
        acetabulum.</li>
    <li>Trace the <b>subchondral cortical arc</b> — the thin dense line of the articular
        surface. That arc defines the circle.</li>
    <li>Mark its <b>centre of curvature</b>, not the brightest spot. Overlap with the
        acetabulum and the opposite head puts the densest shadow <i>medial</i> to the
        true centre.</li>
    <li>Use the magnifier — it follows the cursor at 4×.</li>
   </ol>
   <br>
   <b>Do not centre on:</b> the fovea capitis (medial notch — a defect in the sphere),
   the greater trochanter, the femoral neck or head–neck junction, the acetabular roof
   or teardrop.
   <br><br>
   <b>One circle or two.</b> On a well-positioned lateral the heads superimpose — mark it
   as <span style="color:#00E5A0">LEFT</span> and leave
   <span style="color:#FF3B30">RIGHT</span> empty. If rotation separates them into two
   overlapping circles, mark both; the midpoint is derived and their separation is
   recorded, since a wide separation means an oblique film.
   <br><br>
   <b>Skip</b> for a prosthesis, heads outside the collimated field, or an exposure where
   the cortical arc cannot be traced. Skipping is a valid answer — a guessed centre
   becomes a fabricated error in whatever this set measures.
  </div>
  <div><img src="/example" style="max-height:330px;border:1px solid #444;border-radius:6px"
       alt="worked example" onerror="this.style.display='none'">
   <div style="font-size:11px;color:#888;max-width:300px;margin-top:4px">
     Worked example on a synthetic radiograph, where the centre is a 3-D sphere fit rather
     than anyone's opinion.</div></div>
 </div>
</details>
<div id=wrap><canvas id=c></canvas><canvas id=loupe width=190 height=190></canvas></div>
<script>
let img=new Image(), pts=[], cur=null, nextId=null, nextImg=null;
const C=document.getElementById('c'), X=C.getContext('2d');
const LP=document.getElementById('loupe'), LX=LP.getContext('2d');
const H=()=>({Authorization:'Bearer '+document.getElementById('tok').value});
const msg=t=>document.getElementById('msg').textContent=t;

// PREFETCH: fetch the next film's bytes while this one is being annotated, so "Next"
// is instant instead of a round trip to HuggingFace. /peek does not claim the case --
// claiming on prefetch would hand everyone a second case they never opened.
async function prefetch(){
  try{
    const r=await fetch('/peek',{headers:H()}); if(!r.ok)return;
    const j=await r.json(); if(!j.case_id||j.case_id===nextId)return;
    nextId=j.case_id;
    const b=await fetch('/image/'+nextId,{headers:H()});
    nextImg=URL.createObjectURL(await b.blob());
  }catch(e){}
}
async function load(){
  pts=[]; const t0=performance.now();
  const r=await fetch('/next',{headers:H()});
  if(!r.ok){msg('nothing left to annotate');return}
  cur=await r.json();
  const src=(cur.case_id===nextId&&nextImg)?nextImg:null;
  img=new Image();
  img.onload=()=>{C.width=img.width;C.height=img.height;draw();
                  msg(cur.case_id+'  slot '+cur.slot+'  ('+Math.round(performance.now()-t0)+' ms)');
                  nextId=null;nextImg=null;prefetch();};
  if(src){img.src=src}
  else{const b=await fetch(cur.image_url,{headers:H()});
       img.src=URL.createObjectURL(await b.blob());}
}
function draw(){
  X.drawImage(img,0,0);
  pts.forEach((p,i)=>{
    X.strokeStyle=i===0?'#00E5A0':'#FF3B30';X.lineWidth=Math.max(1.5,img.width/700);
    const x=p[0]*img.width,y=p[1]*img.height,r=img.width/80;
    X.beginPath();X.arc(x,y,r,0,7);X.stroke();
    X.beginPath();X.moveTo(x-r*1.7,y);X.lineTo(x+r*1.7,y);
    X.moveTo(x,y-r*1.7);X.lineTo(x,y+r*1.7);X.stroke();
  });
}
// MAGNIFIER: a femoral head centre is judged by the curvature of a faint arc, and the
// film is displayed scaled down to fit the screen. Without this the annotator's precision
// is set by the display scale rather than by the anatomy.
C.addEventListener('mousemove',e=>{
  const r=C.getBoundingClientRect();
  const fx=(e.clientX-r.left)/r.width, fy=(e.clientY-r.top)/r.height;
  const sx=fx*img.width, sy=fy*img.height, Z=4, S=190/Z;
  LX.fillStyle='#000';LX.fillRect(0,0,190,190);
  LX.drawImage(img, sx-S/2, sy-S/2, S, S, 0,0,190,190);
  LX.strokeStyle='#0072B2';LX.lineWidth=1;
  LX.beginPath();LX.moveTo(95,80);LX.lineTo(95,110);LX.moveTo(80,95);LX.lineTo(110,95);
  LX.stroke();
  LP.style.display='block';
  LP.style.left=(e.clientX-r.left+18)+'px';
  LP.style.top=(e.clientY-r.top-210)+'px';
});
C.addEventListener('mouseleave',()=>LP.style.display='none');
C.addEventListener('click',e=>{
  if(pts.length>=2)return;
  const r=C.getBoundingClientRect();
  pts.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw();
});
function undo(){pts.pop();draw()}
async function send(){
  if(!cur||!pts.length){msg('mark at least one head');return}
  const b=new FormData();
  b.append('case_id',cur.case_id);b.append('slot',cur.slot);
  b.append('points',JSON.stringify({left:pts[0]||null,right:pts[1]||null,
                                    w:img.width,h:img.height}));
  const r=await fetch('/submit',{method:'POST',headers:H(),body:b});
  if(r.ok){load()}else{msg('error saving')}
}
async function skip(){
  const b=new FormData();
  b.append('case_id',cur.case_id);b.append('slot',cur.slot);
  b.append('reason',prompt('why is it unreadable?')||'');
  await fetch('/skip',{method:'POST',headers:H(),body:b});load();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='u')undo(); else if(e.key==='Enter')send(); else if(e.key==='s')skip();
});
</script>
"""
