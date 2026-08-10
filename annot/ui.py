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
<details id=help open>
 <summary>Criteria &mdash; read once, then collapse</summary>
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

   <p><b>Why this point matters.</b> <b>PI</b> is the angle between the line
   perpendicular to the S1 endplate at its midpoint and the line from that midpoint to
   this hip axis. PI is <b>morphological and posture-invariant</b> &mdash; fixed once the
   skeleton matures, the same supine, standing or sitting &mdash; and
   <b>PI = PT + SS</b>. It sets how much lumbar lordosis a given patient <i>needs</i>,
   drives the Roussouly type, and is the basis of the PI&minus;LL target used to plan
   deformity correction. <b>Without this point PI and PT cannot be computed at all</b>,
   which is exactly why these reads are being collected.</p>

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
    <li>Use the magnifier: it follows the cursor at 4&times;.</li>
   </ol>

   <p><b>Do not centre on</b> the fovea capitis (the medial notch &mdash; a defect in the
   sphere, centring there pulls you medially), the greater trochanter, the femoral neck
   or head&ndash;neck junction, or the acetabular roof and teardrop.</p>

   <p><b>One circle or two.</b> On a well-positioned lateral the two heads superimpose
   almost exactly &mdash; mark the single circle as
   <b class=lft>LEFT</b> and leave <b class=rgt>RIGHT</b> empty. If rotation separates
   them into two overlapping circles, mark both: the midpoint is derived and their
   separation is recorded, because a wide separation means an oblique film.</p>

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
</details>
"""

STYLE = """
 :root{--go:#0072B2;--lft:#00E5A0;--rgt:#FF3B30;--warn:#f5a524}
 *{box-sizing:border-box}
 body{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
      background:#111;color:#eee}
 a{color:#6cf}
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
 #wrap{position:relative;display:inline-block}
 canvas{cursor:none;max-height:82vh}
 #loupe{position:absolute;width:190px;height:190px;border:2px solid var(--go);
        border-radius:50%;pointer-events:none;display:none;box-shadow:0 0 12px #000;
        background:#000;z-index:5}
 details>summary{cursor:pointer;padding:8px 12px;background:#22222a;font-weight:600}
 .helpbody{display:flex;gap:20px;padding:12px 16px;background:#191920;flex-wrap:wrap}
 .helptext{max-width:660px}
 .helptext p{margin:0 0 10px}
 .helptext ol{margin:4px 0 10px 18px;padding:0}
 .helptext li{margin:3px 0}
 .cite{background:#15151b;border-left:3px solid var(--go);padding:8px 10px;
       font-size:13px;color:#c8d2dc}
 .warn{background:#241d0c;border-left:3px solid var(--warn);padding:8px 10px}
 .helpimgs{display:flex;gap:14px;flex-wrap:wrap}
 .helpimgs figure{margin:0;max-width:320px}
 .helpimgs img{max-height:330px;max-width:100%;border:1px solid #444;border-radius:6px;
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
  <span>click <b class=lft>LEFT</b> head, then <b class=rgt>RIGHT</b></span>
  <button class=ghost onclick=undo()>Undo <kbd>u</kbd></button>
  <button class=go onclick=send()>Submit <kbd>&crarr;</kbd></button>
  <button class=nv onclick=notVisible()>Not visible <kbd>v</kbd></button>
  <button class=sk onclick=pass()>Pass <kbd>p</kbd></button>
  <a href="/board" target="_blank" class=ghost
     style="text-decoration:none;padding:7px 13px;border-radius:6px;
            border:1px solid #3a3a44">Board</a>
  <span id=who></span>
  <div class=bar title="cases finalised"><i id=barfill style="width:0"></i></div>
  <span id=msg></span>
</header>
""" + CRITERIA + """
<div id=wrap><canvas id=c></canvas><canvas id=loupe width=190 height=190></canvas></div>
</div>

<script>
let img=new Image(), pts=[], cur=null, nextId=null, nextImg=null, token=null, busy=false;
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
    img.onload=()=>{C.width=img.width;C.height=img.height;draw();
      msg(cur.case_id+'  slot '+cur.slot+'  ('+Math.round(performance.now()-t0)+' ms)');
      nextId=null;nextImg=null;prefetch();};
    if(src){img.src=src}
    else{const b=await fetch(cur.image_url,{headers:H()});
         img.src=URL.createObjectURL(await b.blob());}
  } finally { busy=false; }
}
const C=$('c'), X=C.getContext('2d');
const LP=$('loupe'), LX=LP.getContext('2d');
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
// film is displayed scaled down to fit the screen. Without this the annotator's
// precision is set by the display scale rather than by the anatomy.
C.addEventListener('mousemove',e=>{
  if(!img.width)return;
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

async function post(url,fields){
  const b=new FormData();
  for(const k in fields) b.append(k,fields[k]);
  const r=await fetch(url,{method:'POST',headers:H(),body:b});
  if(!r.ok){msg('error: '+(await r.text()).slice(0,120));return null}
  return r.json();
}
async function send(){
  if(!cur){return}
  if(!pts.length){msg('mark at least one head, or use Not visible');return}
  const j=await post('/submit',{case_id:cur.case_id,slot:cur.slot,
    points:JSON.stringify({left:pts[0]||null,right:pts[1]||null,
                           w:img.width,h:img.height})});
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
});
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
