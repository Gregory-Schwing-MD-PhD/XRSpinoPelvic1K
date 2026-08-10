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

CRITERIA � shown in the app and repeated here so the definition lives with the code.

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

from fastapi import Cookie, Depends, FastAPI, Form, Header, HTTPException
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
def _userset(name: str) -> set:
    return {u.strip().lower()
            for u in os.environ.get(name, "").replace(",", " ").split() if u.strip()}


ADJUDICATORS = _userset("ADJUDICATORS")
# Who may read films. Empty = anyone signed in, which is right for an open pilot and
# wrong here: the films are BUU-LSPINE, redistribution is not ours to grant, and the
# alternative (a private Space) 404s for every reader until each one is added as a
# collaborator -- including the owner, because the .hf.space subdomain does not share a
# huggingface.co login cookie. A public Space plus this list gives the readers a URL
# that simply works while keeping the films to named people.
READERS = _userset("READERS")

app = FastAPI(title="Femoral head annotation")
_WHOAMI: dict = {}


# "local:/path" runs the whole Space off the filesystem: real ledger, real images, real
# HTTP, no HuggingFace. That is what the test suite drives, and it is the only way to
# exercise sign-in, claiming and the not-visible path without minting tokens and writing
# to a live repo. On the Space ANNOT_REPO is an org/name repo id, so DEV_LOCAL is False.
DEV_LOCAL = ANNOT_REPO.startswith("local:")


def _store():
    if DEV_LOCAL:
        return store_mod.ReviewStore(store_mod.LocalBackend(ANNOT_REPO[6:]))
    if not ANNOT_REPO or not os.environ.get("HF_TOKEN"):
        raise HTTPException(500, "ANNOT_REPO and HF_TOKEN must be set")
    return store_mod.ReviewStore(
        store_mod.HFBackend(ANNOT_REPO, os.environ["HF_TOKEN"]))


# ── in-memory ledger ────────────────────────────────────────────────────────────
# store.list_cases() reads EVERY case json out of the dataset repo. At 2000 cases that
# is 2000 file reads to answer one "give me the next film", on both /next and /peek --
# tens of seconds per click, which is not a slow tool, it is an unusable one.
#
# So the ledger is loaded once and held in memory, and writes go through a batched
# background flush. Two consequences worth being explicit about:
#
#   * a submit returns as soon as memory is updated, so the annotator never waits on a
#     git commit to see the next film. Durability lags by at most FLUSH_SECONDS.
#   * dirty cases are batched into ONE commit. Per-case commits would mean ~4000 commits
#     for a double-read pass over 2000 films, which is slow and abuses the repo.
#
# A Space runs a single replica, so memory is authoritative while it is up; on restart
# the index is rebuilt from the repo. Nothing is stored only in RAM for longer than the
# flush interval.
import threading  # noqa: E402

FLUSH_SECONDS = float(os.environ.get("FLUSH_SECONDS", 3.0))
_LOCK = threading.RLock()
_INDEX: dict = {}
_ORDER: list = []
_DIRTY: set = set()
_READY = threading.Event()
_LOAD_ERR: list = []


def _load_index() -> None:
    """One pass over the repo at startup. Slow by nature; it happens once."""
    try:
        cases = _store().list_cases()
    except Exception as exc:                                   # noqa: BLE001
        _LOAD_ERR.append(str(exc))
        _READY.set()
        return
    with _LOCK:
        _INDEX.clear()
        for c in cases:
            _INDEX[c["case_id"]] = c
        _ORDER[:] = sorted(_INDEX)
    _READY.set()


def _flush_once() -> int:
    with _LOCK:
        ids = list(_DIRTY)
        _DIRTY.clear()
        batch = [_INDEX[i] for i in ids if i in _INDEX]
    if not batch:
        return 0
    try:
        _store().put_cases(batch)
    except Exception:                                          # noqa: BLE001
        with _LOCK:                       # put them back; try again next tick
            _DIRTY.update(ids)
        return 0
    return len(batch)


def _flusher() -> None:
    while True:
        time.sleep(FLUSH_SECONDS)
        try:
            _flush_once()
        except Exception:                                      # noqa: BLE001
            pass


@app.on_event("startup")
def _boot() -> None:
    threading.Thread(target=_load_index, daemon=True).start()
    threading.Thread(target=_flusher, daemon=True).start()


def _wait_ready() -> None:
    if not _READY.wait(timeout=120):
        raise HTTPException(503, "ledger still loading, try again in a moment")
    if _LOAD_ERR:
        raise HTTPException(500, f"ledger failed to load: {_LOAD_ERR[0]}")


def _touch(case: dict) -> None:
    with _LOCK:
        _INDEX[case["case_id"]] = case
        _DIRTY.add(case["case_id"])


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


# ── sign in with HuggingFace ────────────────────────────────────────────────────
# Set `hf_oauth: true` in the Space README and HF injects OAUTH_CLIENT_ID/SECRET. Then
# signing in is one button, which matters: the readers are radiologists and medical
# students, and "create a token, choose the right scopes, paste the string" is a step
# where a good fraction of them simply stop.
#
# The user's access token is used ONCE, to read their username, and is then discarded --
# nothing here needs to act on their behalf. The session cookie carries the verified
# username and an expiry, signed with the client secret. Pasting a token still works, so
# the tool runs locally and in tests with no OAuth app at all.
OAUTH_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
SPACE_HOST = os.environ.get("SPACE_HOST", "")
OAUTH_ENABLED = bool(OAUTH_ID and OAUTH_SECRET)
SESSION_DAYS = 30
_SECRET = (OAUTH_SECRET or os.environ.get("HF_TOKEN", "") or "dev").encode()


def _sign(payload: str) -> str:
    import hmac
    mac = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}.{mac}"


def _unsign(cookie: str) -> Optional[str]:
    import hmac
    if not cookie or "." not in cookie:
        return None
    payload, _, mac = cookie.rpartition(".")
    good = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(mac, good):
        return None
    name, _, exp = payload.partition("|")
    try:
        if float(exp) < time.time():
            return None
    except ValueError:
        return None
    return name or None


def _redirect_uri() -> str:
    host = SPACE_HOST or "localhost:7860"
    scheme = "https" if SPACE_HOST else "http"
    return f"{scheme}://{host}/auth/callback"


@app.get("/auth/login")
def auth_login():
    from fastapi.responses import RedirectResponse
    if not OAUTH_ENABLED:
        raise HTTPException(503, "OAuth is not configured on this Space; paste a token")
    state = _sign(f"s|{time.time() + 600}")
    url = ("https://huggingface.co/oauth/authorize"
           f"?client_id={OAUTH_ID}&redirect_uri={_redirect_uri()}"
           f"&response_type=code&scope=openid%20profile&state={state}&prompt=consent")
    r = RedirectResponse(url, status_code=302)
    r.set_cookie("oauth_state", state, max_age=600, httponly=True, samesite="lax",
                 secure=bool(SPACE_HOST))
    return r


@app.get("/auth/callback")
def auth_callback(code: str = "", state: str = "", error: str = ""):
    from fastapi.responses import RedirectResponse
    import requests
    if error:
        raise HTTPException(400, f"HuggingFace returned: {error}")
    if not code or not _unsign(state):
        raise HTTPException(400, "bad or expired sign-in state — start again from /")
    tok = requests.post(
        "https://huggingface.co/oauth/token",
        data={"client_id": OAUTH_ID, "client_secret": OAUTH_SECRET,
              "grant_type": "authorization_code", "code": code,
              "redirect_uri": _redirect_uri()}, timeout=20)
    if not tok.ok:
        raise HTTPException(400, f"token exchange failed: {tok.text[:200]}")
    access = tok.json().get("access_token", "")
    info = requests.get("https://huggingface.co/oauth/userinfo",
                        headers={"Authorization": f"Bearer {access}"}, timeout=20)
    name = (info.json() or {}).get("preferred_username") if info.ok else None
    if not name:
        raise HTTPException(400, "could not read your HuggingFace username")
    r = RedirectResponse("/", status_code=302)
    r.set_cookie("annot_session",
                 _sign(f"{name}|{time.time() + SESSION_DAYS * 86400}"),
                 max_age=SESSION_DAYS * 86400, httponly=True, samesite="lax",
                 secure=bool(SPACE_HOST))
    r.delete_cookie("oauth_state")
    return r


@app.get("/auth/logout")
def auth_logout():
    from fastapi.responses import RedirectResponse
    r = RedirectResponse("/", status_code=302)
    r.delete_cookie("annot_session")
    return r


def user(authorization: str = Header(default=""),
         x_annot_token: str = Header(default=""),
         annot_session: str = Cookie(default="")) -> dict:
    """Identity, in order of preference: the signed session cookie (OAuth), then a
    pasted token.

    The pasted token is read from X-Annot-Token, NOT just Authorization. On a PRIVATE
    Space the Hub's own proxy consumes the Authorization header to decide whether the
    caller may reach the Space at all, so it never arrives here -- the token fallback
    looked broken while being eaten one layer up. Authorization is still accepted for
    local runs and tests, where nothing is in front of the app.
    """
    name = _unsign(annot_session)
    if not name and DEV_LOCAL:
        # dev only: whoami would need a real token and a network round trip
        name = (authorization[7:] if authorization.lower().startswith("bearer ")
                else "") or None
    if not name:
        tok = x_annot_token.strip() or (
            authorization[7:] if authorization.lower().startswith("bearer ") else "")
        name = _hf_username(tok)
    if not name:
        raise HTTPException(401, "sign in with HuggingFace")
    low = name.lower()
    if READERS and low not in READERS and low not in ADJUDICATORS:
        raise HTTPException(
            403, f"{name} is not on the reader list for this study. Ask Greg to add "
                 f"your HuggingFace username.")
    return {"id": name, "role": "adjudicator" if low in ADJUDICATORS else "primary"}


@app.get("/whoami")
def whoami_route(annot_session: str = Cookie(default="")):
    return {"user": _unsign(annot_session), "oauth": OAUTH_ENABLED}


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


@app.get("/reference")
def reference():
    """The 'where is it / what am I marking' figure, built by make_reference.py.

    Same reasoning as /example: it is drawn on a DRR, so the crosshair is ostk's own
    sphere fit projected, not a person's opinion, and it is a fixed teaching image rather
    than any film in the queue.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(here, "reference_femhead.png")
    if os.path.exists(fp):
        return Response(open(fp, "rb").read(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(404, "reference image not bundled")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/next")
def next_case(u: dict = Depends(user)):
    _wait_ready()
    with _LOCK:
        for cid in _ORDER:
            case = _INDEX.get(cid)
            if not case or case.get("final"):
                continue
            slot = _claimable_slot(case, u["id"])
            if slot is None:
                continue
            case["slots"][slot] = {"annotator": u["id"], "claimed_at": _now(),
                                   "expires_at": _now() + CLAIM_TTL, "done": False}
            _touch(case)
            return {"case_id": case["case_id"], "slot": slot,
                    "image_url": f"/image/{case['case_id']}",
                    "progress": _progress()}
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
    from PIL import Image
    if DEV_LOCAL:
        p = os.path.join(IMAGE_REPO[6:], f"{case_id}.jpg")
        if not os.path.exists(p):
            raise HTTPException(404, f"image not found: {p}")
        return _encode(Image.open(p).convert("L"), case_id)

    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(IMAGE_REPO, f"images/{case_id}.jpg", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
    except Exception as exc:                                 # noqa: BLE001
        raise HTTPException(404, f"image not found: {exc}")
    return _encode(Image.open(p).convert("L"), case_id)


def _encode(im, case_id: str) -> bytes:
    """Downscale + JPEG. case_id is REQUIRED: it is the cache key, and defaulting it to
    "" cached every film under the same entry and served the first one forever."""
    from io import BytesIO

    from PIL import Image
    if im.width > DISPLAY_W:
        im = im.resize((DISPLAY_W, round(im.height * DISPLAY_W / im.width)),
                       Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=88, optimize=True)
    data = buf.getvalue()
    if case_id and len(_IMG_CACHE) < 400:  # bounded: a Space has finite memory
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
    _wait_ready()
    with _LOCK:
        for cid in _ORDER:
            case = _INDEX.get(cid)
            if not case or case.get("final"):
                continue
            # The case being annotated right now already holds a slot belonging to this
            # user, and _claimable_slot refuses a second slot to the same person -- so
            # the first hit here is genuinely the NEXT film, not the current one.
            if _claimable_slot(case, u["id"]) is not None:
                return {"case_id": case["case_id"]}
    return JSONResponse({"detail": "none"}, status_code=404)


@app.post("/submit")
def submit(case_id: str = Form(...), slot: str = Form(...), points: str = Form(""),
           not_visible: str = Form(""), reason: str = Form(""),
           u: dict = Depends(user)):
    """points: JSON {"left":[x,y]|null, "right":[x,y]|null, "note":str, "w":W, "h":H}

    Coordinates arrive NORMALISED to the displayed image and are stored that way. The
    browser scales the film to fit a screen of unknown size, so raw canvas pixels would
    silently encode each annotator's window size.

    NOT VISIBLE IS AN ANSWER, NOT AN ABANDONMENT. It fills the reader's slot and needs
    the same double read as a marked film. This matters because the open question these
    annotations exist to settle is precisely "on what fraction of films can a trained eye
    place this point at all" -- so an unreadable film is a DATUM. Releasing the slot
    instead (which is what /skip does) would drop that film back in the pool and quietly
    delete the very observation being collected.
    """
    _wait_ready()
    with _LOCK:
        case = _INDEX.get(case_id)
        if case is None:
            raise HTTPException(404, "unknown case")
        s = case.get("slots", {}).get(slot)
        if not s or s.get("annotator") != u["id"]:
            raise HTTPException(403, "that slot is not yours")

        nv = str(not_visible).lower() in ("1", "true", "yes", "on")
        p = json.loads(points) if points else {}
        if not nv and p.get("left") is None and p.get("right") is None:
            raise HTTPException(400, "mark at least one femoral head, or use Not visible")
        if nv:
            s.update(done=True, submitted_at=_now(), points=None,
                     not_visible=True, reason=reason)
        else:
            s.update(done=True, submitted_at=_now(), points=p, not_visible=False)

        done = [case["slots"][k] for k in ("1", "2")
                if case.get("slots", {}).get(k, {}).get("done")]
        if len(done) >= N_PRIMARY:
            _resolve(case, done)
        _touch(case)
        status = ("final" if case.get("final")
                  else "needs adjudication" if case.get("disagree")
                  else "awaiting second read")
        return {"ok": True, "status": status, "progress": _progress()}


def _resolve(case: dict, done: list) -> None:
    """Decide a case once both reads are in.

    Three outcomes, kept distinct because they mean different things downstream:
      both readable and close  -> final, point = mean of the two
      both not visible         -> final with NO point. A settled, usable observation:
                                  this film has no placeable hip landmark.
      anything else            -> disagreement, held for adjudication. That includes one
                                  reader marking a point the other could not see, which
                                  is the most interesting disagreement in the set and
                                  must not be silently resolved either way.
    """
    nv = [bool(s.get("not_visible")) for s in done]
    case.pop("disagree", None)
    if all(nv):
        case["final"] = {"points": None, "by": "consensus-not-visible"}
        case["agree"] = None
        return
    if any(nv):
        case["disagree"] = "one reader marked a point the other could not see"
        case["agree"] = None
        return
    case["agree"] = _agreement(done[0]["points"], done[1]["points"])
    if case["agree"] is not None and case["agree"] <= CONSENSUS_TOL:
        case["final"] = {"points": _mean_points(done[0]["points"], done[1]["points"]),
                         "by": "consensus"}
    else:
        case["disagree"] = f"readers differ by {case['agree']:.4f} of image width"


@app.post("/skip")
def skip(case_id: str = Form(...), slot: str = Form(...), reason: str = Form(""),
         u: dict = Depends(user)):
    """Put a film BACK for someone else -- "not now", not "not visible".

    Use this for an interruption or a film you would rather another reader took. It
    releases the slot and the case returns to the pool. If the femoral heads genuinely
    cannot be seen, that is an ANSWER and belongs in /submit with not_visible=1, where it
    counts toward the double read.
    """
    _wait_ready()
    with _LOCK:
        case = _INDEX.get(case_id)
        if case is None:
            raise HTTPException(404, "unknown case")
        case.setdefault("passed_by", []).append({"by": u["id"], "reason": reason,
                                                 "at": _now()})
        case.get("slots", {}).pop(slot, None)
        _touch(case)
    return {"ok": True}


def _progress() -> dict:
    """Cheap counters for the header. Runs over memory, so it is free to call per film."""
    with _LOCK:
        total = len(_INDEX)
        reads = final = 0
        for c in _INDEX.values():
            reads += sum(1 for k in ("1", "2")
                         if c.get("slots", {}).get(k, {}).get("done"))
            final += bool(c.get("final"))
    return {"total": total, "reads": reads, "reads_needed": total * N_PRIMARY,
            "final": final}


@app.get("/stats")
def stats(u: dict = Depends(user)):
    """Who has done what, and how much is left.

    Deliberately visible to every annotator, not just adjudicators: people finish a long
    repetitive job far more reliably when they can see the pile going down and see that
    others are carrying their share.
    """
    _wait_ready()
    import statistics
    with _LOCK:
        cases = list(_INDEX.values())

    by: dict = {}
    agrees, nv_cases = [], 0
    counts = {"total": len(cases), "untouched": 0, "one_read": 0, "two_reads": 0,
              "final": 0, "final_consensus": 0, "final_not_visible": 0,
              "needs_adjudication": 0, "in_progress": 0}
    for c in cases:
        slots = c.get("slots", {})
        done = [slots.get(k) for k in ("1", "2") if slots.get(k, {}).get("done")]
        claimed = [slots.get(k) for k in ("1", "2")
                   if slots.get(k) and not slots[k].get("done")]
        for s in done:
            r = by.setdefault(s["annotator"], {"reads": 0, "not_visible": 0})
            r["reads"] += 1
            r["not_visible"] += bool(s.get("not_visible"))
        n = len(done)
        counts["untouched"] += (n == 0 and not claimed)
        counts["in_progress"] += bool(claimed)
        counts["one_read"] += (n == 1)
        counts["two_reads"] += (n >= 2)
        if c.get("final"):
            counts["final"] += 1
            if c["final"].get("by") == "consensus-not-visible":
                counts["final_not_visible"] += 1
                nv_cases += 1
            else:
                counts["final_consensus"] += 1
        counts["needs_adjudication"] += bool(c.get("disagree"))
        if c.get("agree") is not None:
            agrees.append(c["agree"])

    counts["reads_done"] = sum(r["reads"] for r in by.values())
    counts["reads_needed"] = counts["total"] * N_PRIMARY
    counts["pct_complete"] = round(100 * counts["final"] / max(1, counts["total"]), 1)
    # The headline number this whole exercise exists to produce.
    settled = counts["final"] + counts["needs_adjudication"]
    counts["not_visible_rate_pct"] = (round(100 * nv_cases / settled, 1)
                                      if settled else None)
    agree_stats = None
    if agrees:
        agree_stats = {
            "n": len(agrees),
            "median": round(statistics.median(agrees), 5),
            "p90": round(sorted(agrees)[int(0.9 * (len(agrees) - 1))], 5),
            "within_tol_pct": round(100 * sum(a <= CONSENSUS_TOL for a in agrees)
                                    / len(agrees), 1),
            "tolerance": CONSENSUS_TOL,
        }
    readers = sorted(({"annotator": k, **v} for k, v in by.items()),
                     key=lambda r: -r["reads"])
    return {"counts": counts, "readers": readers, "agreement": agree_stats,
            "you": u["id"], "pending_writes": len(_DIRTY)}


@app.get("/board", response_class=HTMLResponse)
def board():
    return BOARD


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


from ui import BOARD, PAGE  # noqa: E402  (markup lives next door)
