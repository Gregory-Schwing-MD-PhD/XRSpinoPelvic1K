"""Browser UI for the femoral-head annotator: the reading page and the status board.

Split out of app.py because it is markup, not logic, and because it is the part that
gets iterated on while the API stays still.

Three things here are deliberate rather than decorative:

  * NOT VISIBLE is a primary button, the same size as Submit. It is an ANSWER -- the
    fraction of films with no placeable hip landmark is the number this whole exercise
    exists to produce -- and burying it behind a prompt would bias that fraction
    downward by making the honest response the inconvenient one.

  * The criteria panel cites Radiopaedia and the primary source, and states what the
    point is FOR. A reader who knows the centre feeds PI, and that PI is fixed for life
    and sets the lordosis target, places it differently from one told to "click the hip".

  * No predicted point is ever drawn. These annotations are the independent reference
    for a hip point learned from synthetic data; showing a proposal would measure
    suggestibility instead of anatomy.
"""

CRITERIA = """
<aside id=guide>
 <div class=helpbody>
  <div class=helptext>
   <p><b>What you are marking.</b> The <b>centre of the femoral head</b>. The head is
   very nearly a sphere, so on any radiograph its projection is a circle and the target
   is <b>the centre of that circle</b> &mdash; a geometric centre, not a surface point
   and not a palpable landmark.</p>

   <p class=cite><b>Definition.</b> The bicoxofemoral (hip) axis is the line joining the
   centres of the two femoral heads; the point used for pelvic incidence is its
   <b>midpoint</b>.<br>
   &mdash; Legaye J, Duval-Beaup&egrave;re G, et&nbsp;al. <i>Pelvic incidence: a
   fundamental pelvic parameter for three-dimensional regulation of spinal sagittal
   curves.</i> Eur Spine J 1998;7:99&ndash;103.<br>
   &mdash; <a href="https://radiopaedia.org/articles/pelvic-incidence"
   target="_blank" rel="noopener">Radiopaedia &mdash; pelvic incidence</a>
   (femoral head centres as the pelvic reference axis).</p>

   <p><b>How to find it.</b></p>
   <ol>
    <li>Find the round dense head below and anterior to the S1 endplate, seated in the
        acetabulum.</li>
    <li>Trace the <b>subchondral cortical arc</b> &mdash; the thin dense line of the
        articular surface. That arc defines the circle. (This is the Mose
        concentric-circle method, done by eye.)</li>
    <li>Mark its <b>centre of curvature</b>, <u>not</u> the brightest spot. Overlap with
        the acetabulum and the opposite head puts the densest shadow <i>medial</i> to the
        true centre &mdash; that error is systematic, not noise.</li>
    <li>If the arc is faint, press <kbd>m</kbd> for a 4&times; magnifier that
        follows the cursor. It is off by default because it covers the anatomy
        beside the point you are placing.</li>
   </ol>

   <p><b>Do not centre on</b> the fovea capitis (the medial notch &mdash; a defect in the
   sphere, centring there pulls you medially), the greater trochanter, the femoral neck
   or head&ndash;neck junction, or the acetabular roof and teardrop.</p>

   <p class=cite><b>You cannot tell left from right on a lateral, and you do not need
   to.</b> The two heads are superimposed along the beam and nothing in the image
   distinguishes them. The point we need is the <b>midpoint</b> of the two centres,
   which is the same whichever way round you mark them. <b>Order does not matter</b>
   and the two marks are never compared by side.</p>

   <p><b>One circle or two?</b> Most well-positioned laterals show <b>one</b> circle,
   because the heads superimpose almost exactly &mdash; mark it once and submit. Add a
   second mark <u>only</u> when you can genuinely resolve <b>two overlapping circles of
   the same diameter, each with its own concentric subchondral arc</b>. Rotation
   separates them mostly <i>front-to-back</i>, so two heads sit side by side along the
   AP direction and are the same size. If your two candidates differ in size, or are
   stacked well above and below each other, they are almost certainly not two heads.</p>

   <p class=cite><b>Unsure about a film?</b> Press <kbd>f</kbd> to <b>Flag</b> it with a
   note. It goes straight to the adjudication queue with your name on it and we will look
   at it together &mdash; you cannot send films out of this tool, so this is the way to
   raise one. Flagging does not use up your read: answer the film as best you can
   afterwards, or mark it not visible.</p>

   <p class=warn><b>If you are not sure it is a second head, mark only one.</b> One
   confident centre is worth far more to us than two uncertain ones: a single head still
   gives a usable hip point, whereas a wrong second mark drags the derived midpoint off
   by half its error.</p>

   <p><b>Things that are NOT the femoral head</b>, and how to tell:</p>
   <ul>
    <li><b>Acetabular roof / teardrop</b> &mdash; part of the pelvis. Its arc is
        <i>concave toward</i> the head (it is the socket) and it runs continuous with the
        pelvic ring, rather than closing into a circle.</li>
    <li><b>Greater trochanter</b> &mdash; lateral, not round, and continuous with the
        femoral shaft rather than seated in a socket.</li>
    <li><b>Femoral neck / head&ndash;neck junction</b> &mdash; a narrowing, not part of
        the sphere.</li>
    <li><b>The outer body-wall convexity</b> &mdash; a soft-tissue edge, low contrast,
        no cortical arc, and it does not close into a circle.</li>
   </ul>

   <p class=warn><b>&ldquo;Femoral head not visible&rdquo; is a real answer, and it is
   recorded as one.</b> Use it for a prosthesis, heads outside the collimated field, or
   an exposure where the cortical arc simply cannot be traced. It counts as your read and
   still needs a second reader to agree. A guessed centre becomes a fabricated error in
   everything measured from this set &mdash; please do not guess.<br>
   Use <b>Pass</b> only to hand a film to someone else (an interruption, or one you would
   rather another reader took). Pass records nothing.</p>
  </div>

  <div class=helpimgs>
   <figure>
    <img src="/reference" alt="where the femoral head is on a lateral film"
         onerror="this.parentNode.style.display='none'">
    <figcaption>Where to look, and the centre of curvature of the subchondral arc.</figcaption>
   </figure>
   <figure>
    <img src="/example" alt="worked example"
         onerror="this.parentNode.style.display='none'">
    <figcaption>Worked example on a synthetic radiograph, where the centre is a 3-D
    sphere fit rather than anyone's opinion.</figcaption>
   </figure>
  </div>
 </div>
</aside>
"""

STYLE = """
 :root{--go:#0072B2;--lft:#00E5A0;--rgt:#FF3B30;--warn:#f5a524}
 *{box-sizing:border-box}
 body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
      background:#111;color:#eee}
 a{color:#6cf}
 :root{--hdr:53px}
 header{padding:8px 12px;background:#1b1b1f;display:flex;gap:10px;align-items:center;
        flex-wrap:wrap;position:sticky;top:0;z-index:10;border-bottom:1px solid #2a2a33}
 button{padding:7px 13px;border-radius:6px;border:0;cursor:pointer;font:inherit;
        font-weight:600}
 button:disabled{opacity:.45;cursor:not-allowed}
 .go{background:var(--go);color:#fff}
 .nv{background:var(--warn);color:#231a00}
 .sk{background:#3a3a44;color:#ddd;font-weight:500}
 .ghost{background:transparent;color:#9aa;border:1px solid #3a3a44;font-weight:500}
 kbd{background:#333;padding:1px 5px;border-radius:3px;font-size:11px;font-weight:400}
 #msg{margin-left:auto;color:#9ad;font-variant-numeric:tabular-nums}
 #who{color:#8a8;font-size:12px}
 .bar{height:6px;background:#23232b;border-radius:3px;width:150px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--lft)}
 .lft{color:var(--lft)}.rgt{color:var(--rgt)}
 /* Two panels. The guide scrolls on its own so the film never moves while you read,
    and the film column is what the window resize actually gives space to. */
 #split{display:grid;grid-template-columns:var(--guide,430px) 1fr;
        height:calc(100vh - var(--hdr));min-height:0}
 #split.noguide{grid-template-columns:0 1fr}
 /* visibility, NOT display:none. Removing the guide from the grid drops the
    film into the first (zero-width) column and it collapses to nothing. */
 #split.noguide #guide{visibility:hidden;border-right:0}
 #guide{overflow-y:auto;overflow-x:hidden;border-right:1px solid #2a2a33;min-width:0;
        background:#191920}
 /* BOTTOM GUTTER. The Windows taskbar auto-shows when the pointer reaches the last few
    pixels of the screen, which made the bottom of the film unclickable. Reserving a
    strip below the image means the cursor never has to go there to reach anatomy. */
 #stage{display:flex;align-items:center;justify-content:center;overflow:auto;min-width:0;
        padding:10px 12px calc(64px + env(safe-area-inset-bottom,0px))}
 #wrap{position:relative;display:block}
 /* Display size is set in JS by fit(), not by CSS. A percentage max-height has no
    definite parent height to resolve against through this flex chain, so the canvas
    rendered at its natural 1100x1919 and ran off the bottom of the window. */
 canvas{cursor:crosshair;display:block}
 body.zoom canvas{cursor:none}
 #loupe{position:absolute;width:190px;height:190px;border:2px solid var(--go);
        border-radius:50%;pointer-events:none;display:none;box-shadow:0 0 12px #000;
        background:#000;z-index:5}
 details>summary{cursor:pointer;padding:8px 12px;background:#22222a;font-weight:600}
 .helpbody{padding:12px 16px;background:#191920}
 .helptext{max-width:none}
 .helptext p{margin:0 0 10px}
 .helptext ol{margin:4px 0 10px 18px;padding:0}
 .helptext li{margin:3px 0}
 .cite{background:#15151b;border-left:3px solid var(--go);padding:8px 10px;
       font-size:13px;color:#c8d2dc}
 .warn{background:#241d0c;border-left:3px solid var(--warn);padding:8px 10px}
 .helpimgs{display:flex;flex-direction:column;gap:14px;margin-top:12px}
 .helpimgs figure{margin:0;max-width:100%}
 .helpimgs img{max-height:none;max-width:100%;cursor:zoom-in;border:1px solid #444;border-radius:6px;
               display:block}
 .helpimgs figcaption{font-size:11px;color:#888;margin-top:4px}
 .signin{padding:40px;text-align:center}
 .signin button{font-size:16px;padding:12px 22px}
 table{border-collapse:collapse;width:100%;max-width:900px}
 th,td{padding:6px 10px;border-bottom:1px solid #2a2a33;text-align:left}
 th{color:#9aa;font-weight:600;font-size:12px;text-transform:uppercase;
    letter-spacing:.08em}
 td.n{text-align:right;font-variant-numeric:tabular-nums}
 .cards{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
 .card{background:#1b1b22;border:1px solid #2a2a33;border-radius:8px;padding:12px 16px;
       min-width:140px}
 .card b{display:block;font-size:26px;font-variant-numeric:tabular-nums}
 .card span{color:#9aa;font-size:12px}
 main{padding:16px}
"""

PAGE = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Femoral head annotation</title>
<style>""" + STYLE + """</style>

<div id=gate class=signin hidden>
  <h2>Femoral head annotation</h2>
  <p>Two clicks per lateral radiograph. Sign in with your HuggingFace account so your
     reads are attributed to you.</p>
  <p><button class=go onclick="location.href='/auth/login'">Sign in with HuggingFace</button></p>
  <p style="color:#888;font-size:12px">No HuggingFace account?
     <a href="https://huggingface.co/join" target="_blank" rel="noopener">Create one</a>
     &mdash; it takes about a minute and is free.</p>
  <details style="max-width:520px;margin:18px auto;text-align:left">
    <summary>Or paste an access token</summary>
    <p style="font-size:13px">For local use, or if the sign-in button is unavailable.
       Create a <b>read</b> token at
       <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">
       huggingface.co/settings/tokens</a>.</p>
    <input id=tok type=password placeholder="hf_..." size=30
           style="background:#222;color:#eee;border:1px solid #444;border-radius:5px;
                  padding:7px;width:100%">
    <p><button class=go onclick="useToken()">Use this token</button></p>
  </details>
</div>

<div id=appui hidden>
<header>
  <button class=go onclick=load()>Next <kbd>n</kbd></button>
  <span>click the head centre &mdash; a <b class=rgt>2nd</b> only if you see two distinct circles</span>
  <button class=ghost onclick=undo()>Undo <kbd>u</kbd></button>
  <button class=go onclick=send()>Submit <kbd>&crarr;</kbd></button>
  <button class=nv onclick=notVisible()>Not visible <kbd>v</kbd></button>
  <button class=sk onclick=pass()>Pass <kbd>p</kbd></button>
  <button class=ghost onclick=flagIt()>Flag <kbd>f</kbd></button>
  <a href="/board" target="_blank" class=ghost
     style="text-decoration:none;padding:7px 13px;border-radius:6px;
            border:1px solid #3a3a44">Board</a>
  <button class=ghost onclick=toggleGuide()>Guide <kbd>g</kbd></button>
  <button class=ghost id=zoomBtn onclick=toggleZoom()>Zoom <kbd>m</kbd></button>
  <span id=who></span>
  <div class=bar title="cases finalised"><i id=barfill style="width:0"></i></div>
  <span id=msg></span>
</header>
<main id=split>
""" + CRITERIA + """
  <section id=stage>
    <div id=wrap><canvas id=c></canvas><canvas id=loupe width=190 height=190></canvas></div>
  </section>
</main>
</div>

<script>
let img=new Image(), pts=[], cur=null, nextId=null, nextImg=null, token=null, busy=false;
// The loupe follows the cursor and covers the anatomy next to the point you are
// placing. It is a real precision aid on a faint arc, so it stays -- but OFF by
// default and remembered, rather than imposed on every reader.
let zoomOn=false, guideOn=true;
const $=i=>document.getElementById(i);
// X-Annot-Token, not Authorization: on a private Space the Hub proxy
// consumes Authorization before the app ever sees it.
const H=()=>token?{'X-Annot-Token':token}:{};
const msg=t=>$('msg').textContent=t;

function useToken(){
  const v=$('tok').value.trim(); if(!v)return;
  token=v; try{localStorage.setItem('hf_tok',v)}catch(e){}
  start();
}

async function boot(){
  try{token=localStorage.getItem('hf_tok')||null}catch(e){}
  const r=await fetch('/whoami');
  const j=await r.json();
  if(j.user){ $('who').textContent='signed in as '+j.user; start(); return; }
  if(token){ start(); return; }
  $('gate').hidden=false;
}
async function start(){
  $('gate').hidden=true; $('appui').hidden=false;
  try{
    toggleZoom(localStorage.getItem('annot_zoom')==='1');
    toggleGuide(localStorage.getItem('annot_guide')!=='0');
  }catch(e){ toggleZoom(false); }
  if(token && !$('who').textContent){
    $('who').textContent='using pasted token';
  }
  load();
}

// PREFETCH: pull the next film's bytes while this one is being annotated, so Next is
// instant instead of a round trip to HuggingFace. /peek does not claim the case --
// claiming on prefetch would hand everyone a second case they never opened.
async function prefetch(){
  try{
    const r=await fetch('/peek',{headers:H()}); if(!r.ok)return;
    const j=await r.json(); if(!j.case_id||j.case_id===nextId)return;
    nextId=j.case_id;
    const b=await fetch('/image/'+nextId,{headers:H()});
    if(b.ok) nextImg=URL.createObjectURL(await b.blob());
  }catch(e){}
}
function progress(p){
  if(!p)return;
  $('barfill').style.width=(100*p.final/Math.max(1,p.total)).toFixed(1)+'%';
  $('barfill').parentNode.title=p.final+' / '+p.total+' films finalised  ('
    +p.reads+' of '+p.reads_needed+' reads done)';
}
async function load(){
  if(busy)return; busy=true;
  pts=[]; const t0=performance.now();
  try{
    const r=await fetch('/next',{headers:H()});
    if(r.status===401){$('appui').hidden=true;$('gate').hidden=false;return}
    if(r.status===403){msg(await r.text());
      $('appui').hidden=true;$('gate').hidden=false;
      $('gate').insertAdjacentHTML('afterbegin',
        '<p style="color:#f5a524;max-width:60ch;margin:0 auto 14px">'
        +'You are signed in, but not on the reader list for this study. '
        +'Send Greg your HuggingFace username.</p>');
      return}
    if(!r.ok){msg('nothing left to annotate — thank you');cur=null;return}
    cur=await r.json(); progress(cur.progress);
    const src=(cur.case_id===nextId&&nextImg)?nextImg:null;
    img=new Image();
    img.onload=()=>{C.width=img.width;C.height=img.height;fit();draw();
      msg(cur.case_id+'  slot '+cur.slot+'  ('+Math.round(performance.now()-t0)+' ms)');
      nextId=null;nextImg=null;prefetch();};
    if(src){img.src=src}
    else{const b=await fetch(cur.image_url,{headers:H()});
         img.src=URL.createObjectURL(await b.blob());}
  } finally { busy=false; }
}
const C=$('c'), X=C.getContext('2d');
const LP=$('loupe'), LX=LP.getContext('2d');
// Fit the film to the space left over, preserving aspect ratio, and leave a strip at
// the bottom: the Windows taskbar auto-shows at the last few pixels of the screen and
// eats the pointer before it reaches the caudal anatomy.
const BOTTOM_GUTTER = 64;
function fit(){
  if(!img.width) return;
  const st=$('stage').getBoundingClientRect();
  const availW=Math.max(80, st.width-24);
  const availH=Math.max(80, window.innerHeight-st.top-BOTTOM_GUTTER-12);
  const k=Math.min(availW/img.width, availH/img.height);
  C.style.width =(img.width*k)+'px';
  C.style.height=(img.height*k)+'px';
}
function draw(){
  X.drawImage(img,0,0);
  pts.forEach((p,i)=>{
    X.strokeStyle=i===0?'#00E5A0':'#FF3B30';   // 1st / 2nd mark, NOT left/rightX.lineWidth=Math.max(1.5,img.width/700);
    const x=p[0]*img.width,y=p[1]*img.height,r=img.width/80;
    X.beginPath();X.arc(x,y,r,0,7);X.stroke();
    X.beginPath();X.moveTo(x-r*1.7,y);X.lineTo(x+r*1.7,y);
    X.moveTo(x,y-r*1.7);X.lineTo(x,y+r*1.7);X.stroke();
  });
}
// MAGNIFIER: a femoral head centre is judged by the curvature of a faint arc, and the
// film is displayed scaled down to fit the screen. Without this the annotator's
// precision is set by the display scale rather than by the anatomy.
C.addEventListener('mousemove',e=>{
  if(!img.width||!zoomOn)return;
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
  if(pts.length>=2||!img.width)return;
  const r=C.getBoundingClientRect();
  pts.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw();
});
function undo(){pts.pop();draw()}

function toggleZoom(on){
  zoomOn = (on===undefined) ? !zoomOn : on;
  document.body.classList.toggle('zoom', zoomOn);
  $('zoomBtn').classList.toggle('ctl--on', zoomOn);
  $('zoomBtn').style.borderColor = zoomOn ? '#0072B2' : '';
  if(!zoomOn) LP.style.display='none';
  try{localStorage.setItem('annot_zoom', zoomOn?'1':'0')}catch(e){}
}
function toggleGuide(on){
  guideOn = (on===undefined) ? !guideOn : on;
  $('split').classList.toggle('noguide', !guideOn);
  requestAnimationFrame(fit);
  try{localStorage.setItem('annot_guide', guideOn?'1':'0')}catch(e){}
}

async function post(url,fields){
  const b=new FormData();
  for(const k in fields) b.append(k,fields[k]);
  const r=await fetch(url,{method:'POST',headers:H(),body:b});
  if(!r.ok){msg('error: '+(await r.text()).slice(0,120));return null}
  return r.json();
}
async function send(){
  if(!cur){return}
  if(!pts.length){msg('mark the head centre, or use Not visible');return}
  // an unordered list: a lateral cannot tell you which head is which
  const j=await post('/submit',{case_id:cur.case_id,slot:cur.slot,
    points:JSON.stringify({heads:pts,w:img.width,h:img.height})});
  if(j){progress(j.progress);load()}
}
async function notVisible(){
  if(!cur)return;
  const why=prompt('Why is the femoral head not visible?\\n'+
    '(prosthesis / out of field / underexposed / other)','out of field');
  if(why===null)return;                 // cancelled -- do not record an answer
  const j=await post('/submit',{case_id:cur.case_id,slot:cur.slot,points:'',
                                not_visible:'1',reason:why});
  if(j){progress(j.progress);load()}
}
// A reader sees one film at a time and cannot browse, re-open or export -- so without
// this there is no way to raise a film they are unsure about. It does not consume the
// read: flag it, then answer it as best you can, or use Not visible.
async function flagIt(){
  if(!cur)return;
  const note=prompt('What is unclear about this film?
'
    +'(it goes to the adjudication queue with your name)','');
  if(note===null)return;
  await post('/flag',{case_id:cur.case_id,note:note});
  msg('flagged for discussion — now answer it as best you can');
}
async function pass(){
  if(!cur)return;
  await post('/skip',{case_id:cur.case_id,slot:cur.slot,reason:'passed'});
  load();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
  if(e.key==='u')undo();
  else if(e.key==='Enter')send();
  else if(e.key==='v')notVisible();
  else if(e.key==='p')pass();
  else if(e.key==='n')load();
  else if(e.key==='m')toggleZoom();
  else if(e.key==='g')toggleGuide();
  else if(e.key==='f')flagIt();
});
window.addEventListener('resize', fit);
boot();
</script>
"""

BOARD = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Annotation status board</title>
<style>""" + STYLE + """</style>
<header><b>Femoral head annotation &mdash; status</b>
  <a href="/">back to reading</a>
  <span id=msg></span>
</header>
<main>
 <div class=cards id=cards></div>
 <h3>Readers</h3>
 <table id=readers><thead><tr><th>Annotator</th><th class=n>Reads</th>
   <th class=n>Not visible</th><th class=n>Share</th></tr></thead><tbody></tbody></table>
 <h3>Inter-reader agreement</h3>
 <div id=agree></div>
 <p style="color:#888;font-size:12px" id=foot></p>
</main>
<script>
const $=i=>document.getElementById(i);
function card(v,l,t){return '<div class=card title="'+(t||'')+'"><b>'+v+'</b><span>'+l+
  '</span></div>'}
async function tick(){
  let r;
  try{ r=await fetch('/stats',{headers:(localStorage.getItem('hf_tok')?
        {Authorization:'Bearer '+localStorage.getItem('hf_tok')}:{})}); }
  catch(e){ $('msg').textContent='offline'; return; }
  if(!r.ok){ $('msg').textContent='sign in on the reading page first'; return; }
  const j=await r.json(), c=j.counts;
  $('cards').innerHTML=
     card(c.final+' / '+c.total,'films finalised','both reads in and agreeing')
   + card(c.pct_complete+'%','complete')
   + card(c.reads_done+' / '+c.reads_needed,'reads done','2 reads per film')
   + card(c.one_read,'awaiting 2nd read')
   + card(c.needs_adjudication,'need adjudication','the two readers disagree')
   + card(c.final_not_visible,'no visible head','both readers agreed it is unreadable')
   + card(c.not_visible_rate_pct===null?'—':c.not_visible_rate_pct+'%',
          'not-visible rate','of every settled film');
  const tot=c.reads_done||1;
  $('readers').querySelector('tbody').innerHTML=j.readers.map(x=>
    '<tr><td>'+x.annotator+(x.annotator===j.you?' <b style="color:#00E5A0">(you)</b>':'')
    +'</td><td class=n>'+x.reads+'</td><td class=n>'+x.not_visible
    +'</td><td class=n>'+(100*x.reads/tot).toFixed(0)+'%</td></tr>').join('')
    || '<tr><td colspan=4 style="color:#888">no reads yet</td></tr>';
  $('agree').innerHTML = j.agreement
    ? '<div class=cards>'
      + card(j.agreement.within_tol_pct+'%','within tolerance',
             'both readers within '+j.agreement.tolerance+' of image width')
      + card(j.agreement.median,'median disagreement','fraction of image width')
      + card(j.agreement.p90,'90th percentile')
      + card(j.agreement.n,'doubly-read films') + '</div>'
    : '<p style="color:#888">No film has two marked reads yet.</p>';
  $('foot').textContent='pending writes: '+j.pending_writes
    +'  ·  refreshed '+new Date().toLocaleTimeString();
}
tick(); setInterval(tick,10000);
</script>
"""


REVIEW = """<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Adjudication</title>
<style>""" + STYLE + """
 #rsplit{display:grid;grid-template-columns:330px 1fr;height:calc(100vh - var(--hdr))}
 #rlist{overflow-y:auto;border-right:1px solid #2a2a33;background:#15151b}
 .ritem{padding:9px 12px;border-bottom:1px solid #23232b;cursor:pointer;font-size:12.5px}
 .ritem:hover{background:#1d1d26}
 .ritem.is-on{background:#1f2a3a;box-shadow:inset 3px 0 0 var(--go)}
 .ritem b{font-family:var(--mono)}
 .ritem span{display:block;color:#8a94a0;margin-top:2px}
 .ritem.done b{color:#6b7684;text-decoration:line-through}
 #rstage{display:flex;align-items:center;justify-content:center;overflow:auto;
         padding:10px 12px 64px;min-width:0}
 .legend{display:flex;gap:14px;font-size:12px;margin-left:8px}
 .legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
</style>
<header>
  <b>Adjudication</b>
  <span id=legend class=legend></span>
  <button class=go onclick=useMine()>Use my mark <kbd>&crarr;</kbd></button>
  <button class=ghost onclick=useReader(0)>Accept 1st reader <kbd>1</kbd></button>
  <button class=ghost onclick=useReader(1)>Accept 2nd reader <kbd>2</kbd></button>
  <button class=nv onclick=settleNV()>Not visible <kbd>v</kbd></button>
  <button class=ghost onclick=clearMine()>Clear <kbd>u</kbd></button>
  <a href="/" class=ghost style="text-decoration:none;padding:7px 13px;
     border-radius:6px;border:1px solid #3a3a44">Reading</a>
  <span id=msg></span>
</header>
<div id=rsplit>
  <div id=rlist></div>
  <section id=rstage><div id=rwrap><canvas id=rc></canvas></div></section>
</div>
<script>
const $=i=>document.getElementById(i);
const H=()=>{const t=localStorage.getItem('hf_tok');return t?{'X-Annot-Token':t}:{}};
const msg=t=>$('msg').textContent=t;
const COL=['#00E5A0','#FF3B30'];
let cases=[], cur=null, img=new Image(), mine=[];
const C=$('rc'), X=C.getContext('2d');

async function loadQueue(){
  const r=await fetch('/queue',{headers:H()});
  if(!r.ok){msg(await r.text());return}
  cases=(await r.json()).cases;
  $('rlist').innerHTML=cases.map((c,i)=>
    '<div class="ritem'+(c.settled?' done':'')+'" data-i="'+i+'" onclick="pick('+i+')">'
    +'<b>'+c.case_id+'</b><span>'+c.why+'</span></div>').join('')
    || '<div class=ritem><span>nothing to adjudicate</span></div>';
  if(cases.length) pick(0);
}
// Fit like the reading page: leave a strip at the bottom so the taskbar cannot
// swallow the pointer over the caudal anatomy.
function fit(){
  if(!img.width)return;
  const st=$('rstage').getBoundingClientRect();
  const k=Math.min((st.width-24)/img.width,
                   (window.innerHeight-st.top-76)/img.height);
  C.style.width=(img.width*k)+'px'; C.style.height=(img.height*k)+'px';
}
function draw(){
  if(!img.width)return;
  X.drawImage(img,0,0);
  const r=img.width/85, lw=Math.max(2,img.width/450);
  (cur.reads||[]).forEach((rd,i)=>{
    X.strokeStyle=COL[i]; X.lineWidth=lw;
    (rd.heads||[]).forEach(p=>{
      const x=p[0]*img.width,y=p[1]*img.height;
      X.beginPath();X.arc(x,y,r,0,7);X.stroke();
      X.beginPath();X.moveTo(x-r*1.7,y);X.lineTo(x+r*1.7,y);
      X.moveTo(x,y-r*1.7);X.lineTo(x,y+r*1.7);X.stroke();
    });
  });
  X.strokeStyle='#ffffff'; X.lineWidth=lw*1.2;
  mine.forEach(p=>{
    const x=p[0]*img.width,y=p[1]*img.height;
    X.beginPath();X.arc(x,y,r*1.25,0,7);X.stroke();
    X.beginPath();X.moveTo(x-r*2,y);X.lineTo(x+r*2,y);
    X.moveTo(x,y-r*2);X.lineTo(x,y+r*2);X.stroke();
  });
}
async function pick(i){
  cur=cases[i]; mine=[];
  [...document.querySelectorAll('.ritem')].forEach(e=>
    e.classList.toggle('is-on', e.dataset.i==String(i)));
  $('legend').innerHTML=(cur.reads||[]).map((rd,k)=>
    '<span><i style="background:'+COL[k]+'"></i>'+rd.annotator
    +(rd.not_visible?' — not visible'+(rd.reason?' ('+rd.reason+')':''):
      ' — '+(rd.heads||[]).length+' mark(s)')+'</span>').join('')
    +'<span><i style="background:#fff"></i>your mark</span>'
    +(cur.agree!=null?'<span>readers differ by '+(100*cur.agree).toFixed(2)+'% of width</span>':'')
    +(cur.flagged||[]).map(f=>'<span>flagged by '+f.by+(f.note?': '+f.note:'')+'</span>').join('');
  const b=await fetch('/image/'+cur.case_id,{headers:H()});
  img=new Image();
  img.onload=()=>{C.width=img.width;C.height=img.height;fit();draw();};
  img.src=URL.createObjectURL(await b.blob());
}
C.addEventListener('click',e=>{
  if(mine.length>=2)return;
  const r=C.getBoundingClientRect();
  mine.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]); draw();
});
function clearMine(){mine=[];draw()}
async function settle(body){
  const f=new FormData(); f.append('case_id',cur.case_id);
  for(const k in body) f.append(k,body[k]);
  const r=await fetch('/adjudicate',{method:'POST',headers:H(),body:f});
  if(!r.ok){msg(await r.text());return}
  msg(cur.case_id+' settled');
  const i=cases.indexOf(cur); cases[i].settled=true;
  document.querySelector('.ritem[data-i="'+i+'"]').classList.add('done');
  if(cases[i+1]) pick(i+1);
}
function useMine(){
  if(!mine.length){msg('click the centre first');return}
  settle({points:JSON.stringify({heads:mine})});
}
function useReader(k){
  const rd=(cur.reads||[])[k];
  if(!rd){msg('no such read');return}
  if(rd.not_visible) return settle({not_visible:'1',note:'accepted '+rd.annotator});
  settle({points:JSON.stringify({heads:rd.heads}),note:'accepted '+rd.annotator});
}
function settleNV(){settle({not_visible:'1'})}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='Enter')useMine(); else if(e.key==='1')useReader(0);
  else if(e.key==='2')useReader(1); else if(e.key==='v')settleNV();
  else if(e.key==='u')clearMine();
});
window.addEventListener('resize',fit);
loadQueue();
</script>
"""
