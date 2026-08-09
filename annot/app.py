"""
Femoral-head annotation Space — two clicks per lateral radiograph, distributed.

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


@app.get("/image/{case_id}")
def image(case_id: str, u: dict = Depends(user)):
    """Stream the film THROUGH the Space so annotators never need read access to the
    image repo, exactly as the review service streams labels."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(IMAGE_REPO, f"images/{case_id}.jpg", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
    except Exception as exc:                                 # noqa: BLE001
        raise HTTPException(404, f"image not found: {exc}")
    return Response(open(p, "rb").read(), media_type="image/jpeg")


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
 header{padding:8px 12px;background:#1b1b1f;display:flex;gap:12px;align-items:center}
 #wrap{position:relative;display:inline-block}
 canvas{cursor:crosshair;max-height:88vh}
 button{padding:6px 12px;border-radius:6px;border:0;cursor:pointer}
 .go{background:#0072B2;color:#fff}.sk{background:#666;color:#fff}
 #msg{margin-left:auto;color:#9ad}
 input{background:#222;color:#eee;border:1px solid #444;border-radius:5px;padding:5px}
</style>
<header>
  <input id=tok type=password placeholder="hf_... token" size=26>
  <button class=go onclick=load()>Next film</button>
  <span>click <b style="color:#00E5A0">LEFT</b> head, then
        <b style="color:#FF3B30">RIGHT</b></span>
  <button onclick=undo()>Undo</button>
  <button class=go onclick=send()>Submit</button>
  <button class=sk onclick=skip()>Can't see them</button>
  <span id=msg></span>
</header>
<div id=wrap><canvas id=c></canvas></div>
<script>
let img=new Image(), pts=[], cur=null, C=document.getElementById('c'), X=C.getContext('2d');
const H=()=>({Authorization:'Bearer '+document.getElementById('tok').value});
function msg(t){document.getElementById('msg').textContent=t}
async function load(){
  pts=[];const r=await fetch('/next',{headers:H()});
  if(!r.ok){msg(await r.text());return}
  cur=await r.json();
  img=new Image();
  img.onload=()=>{C.width=img.width;C.height=img.height;draw()};
  const b=await fetch(cur.image_url,{headers:H()});
  img.src=URL.createObjectURL(await b.blob());
  msg(cur.case_id+'  slot '+cur.slot);
}
function draw(){
  X.drawImage(img,0,0);
  pts.forEach((p,i)=>{
    X.strokeStyle=i===0?'#00E5A0':'#FF3B30';X.lineWidth=Math.max(2,img.width/500);
    X.beginPath();X.arc(p[0]*img.width,p[1]*img.height,img.width/70,0,7);X.stroke();
    X.beginPath();X.moveTo(p[0]*img.width-img.width/40,p[1]*img.height);
    X.lineTo(p[0]*img.width+img.width/40,p[1]*img.height);
    X.moveTo(p[0]*img.width,p[1]*img.height-img.width/40);
    X.lineTo(p[0]*img.width,p[1]*img.height+img.width/40);X.stroke();
  });
}
C.addEventListener('click',e=>{
  if(pts.length>=2)return;
  const r=C.getBoundingClientRect();
  pts.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw();
});
function undo(){pts.pop();draw()}
async function send(){
  if(!cur||!pts.length){msg('mark at least one head');return}
  const body=new FormData();
  body.append('case_id',cur.case_id);body.append('slot',cur.slot);
  body.append('points',JSON.stringify({left:pts[0]||null,right:pts[1]||null,
                                       w:img.width,h:img.height}));
  const r=await fetch('/submit',{method:'POST',headers:H(),body});
  msg(r.ok?'saved — loading next':'error');if(r.ok)load();
}
async function skip(){
  const body=new FormData();
  body.append('case_id',cur.case_id);body.append('slot',cur.slot);
  body.append('reason',prompt('why?')||'');
  await fetch('/skip',{method:'POST',headers:H(),body});load();
}
</script>
"""
