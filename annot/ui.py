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
   <p class=ifarc><b>What you are doing.</b> Mark <b>three named points</b> on the femoral
   head articular surface:</p>
   <ol class="lmlist ifarc">
    <li><b class=lmA>A &mdash; anterior extreme</b> <kbd>1</kbd>. The most anterior point of
        the articular surface: where the cortex stops running forward and starts curving
        back.</li>
    <li><b class=lmS>S &mdash; superior extreme</b> <kbd>2</kbd>. The top of the head, where
        the cortex is horizontal.</li>
    <li><b class=lmP>P &mdash; posterior extreme</b> <kbd>3</kbd>. The mirror of A at the
        back.</li>
   </ol>
   <p class=ifarc>A circle is fitted to them live and its <b>centre</b> is what the study
   needs. You never judge where the middle is.</p>

   <p class="cite ifarc"><b>There is no required order and nothing is locked in.</b> Click
   points wherever the cortex is clear; the tool names them A, S, P as you go, purely as a
   guess. <b>Click any point to select it and press <kbd>1</kbd>, <kbd>2</kbd> or
   <kbd>3</kbd> to rename it</b>, or <kbd>0</kbd> to make it an ordinary rim point. Renaming
   never deletes anything &mdash; whatever held that name becomes a rim point. If you would
   rather mark first and name later, click five or six points and press <kbd>w</kbd>
   (<b>Auto-name</b>): it works out which of <i>your</i> clicks are the three extremes from
   the shape they fit. It never invents a point, and every name it picks is still
   editable.</p>

   <p class="cite ifarc"><b>Why these three and not just &ldquo;points on the rim&rdquo;.</b>
   Because an <i>extreme</i> is something you can find, and a point partway round an arc is
   not. At the anterior extreme the cortex is vertical; at the superior extreme it is
   horizontal. Those are properties of the image at that spot, so two readers looking at
   the same film are looking for the same thing. Anywhere else on a round head, one bit of
   cortex is indistinguishable from the next, and &ldquo;the point a third of the way
   round&rdquo; only means something once you already know where the centre is &mdash;
   which is the thing we are trying to find.</p>

   <p class="warn ifarc"><b>Add points away from A, S and P &mdash; and where matters more
   than how many.</b> A, S and P are the points you can <i>name</i>; they are not the three
   that pin a circle best. With A and P at the sides and S on top, the <i>side-to-side</i>
   position of the centre is fixed by two points at once, but its <i>up-and-down</i>
   position rests on S almost alone &mdash; which is why the uncertainty ellipse comes out
   as a vertical cigar. Two more clicks near the top barely help (11% tighter). <b>One click
   on the inferior margin cuts it by 42%</b>, and does better than three perfectly spaced
   points. The header names the part of the rim that would help most; click there. If the
   inferior margin is buried under the acetabular shadow, spread the extras as far from
   your existing three as the cortex allows.</p>

   <p class="cite ifarc"><b>How much is enough is still being calibrated.</b> The
   <b>&plusmn;</b> figure assumes a click is accurate to about 0.3% of the film width. That
   is an estimate, not a measurement &mdash; it is the number this pilot exists to pin down.
   If you are more precise than that, the tool is being pessimistic and A, S and P alone
   would already be inside tolerance. Until then, treat <b>widen</b> as a nudge rather than
   a verdict, and do not hunt for extra points on cortex you cannot actually see.</p>

   <p class="cite ifarc"><b>Why not click the centre directly.</b> The hip point is not a
   landmark. It is <i>derived</i> from landmarks, exactly like the centre of a vertebral
   body &mdash; there is no edge at it, no line and no texture, because it is inside the
   bone. Every reader who clicks it is inferring it, and no two readers infer identically.
   The head is very nearly a sphere, so its projection is a circle and solving that circle
   is exact rather than an approximation: this is the Mose concentric-circle method with
   the circle solved for instead of slid into place. Legaye defines the hip axis through
   the head <i>centres</i>, treating the head as a sphere, so the fitted centre is the
   published landmark itself.</p>

   <p class="warn ifarc"><b>Get A and P at the widest part.</b> Points crowded along the
   top of the head fit a circle perfectly and still put the centre in the wrong place,
   because a short arc barely constrains how <i>far away</i> the centre is. The dashed
   dashed ellipse drawn around the centre <b>is</b> that uncertainty, at true scale. When
   the marks are well spread it sits inside the crosshair; when they are bunched at the top
   of the head it stretches into a vertical cigar <b>longer than the head itself</b>, in
   red. Treat it as an alarm rather than a gauge &mdash; for the fine reading, the
   <b>&plusmn;</b> figure in the header is the same quantity as a number, against the 0.005
   tolerance. <b>If it will not settle, add rim points anywhere the cortex is clear.</b> If
   it still will not, the arc you can see really is too short: press <kbd>f</kbd> to flag
   the film rather than forcing it.</p>

   <p class="warn ifarc"><b>The tool tells you whether you are on the cortex.</b> As you
   move the cursor the header shows <b>edge&nbsp;NN%</b> &mdash; how strong a structural
   edge sits under the pointer compared with the surrounding film. The subchondral cortex
   is an edge; the dark joint space and the flat cancellous bone inside the head are not.
   <b>Hunt with the cursor until the number is high, then click.</b> Above about 70% you
   are on a structure; below 40% you are on flat film.<br>
   If a mark you have already placed is not on an edge, an <b>arrow appears on it pointing
   at the nearest better spot</b>, with the distance in pixels. Drag the point along the
   arrow. Nothing moves on its own and nothing is blocked &mdash; it is a hint from the
   pixels, and if it is pointing at the acetabular roof instead of the head, ignore it. Your
   anatomy wins.</p>

   <p class="cite ifarc"><b>Zoom the whole film, do not squint.</b> Scroll to zoom about the
   cursor and <b>right-drag to pan</b>; both carry over to the next film, so you set your
   working magnification once. There is no magnifier &mdash; a loupe covers the anatomy
   beside the point you are placing, which is the part you need to see. At fit-to-window one
   screen pixel is several pixels of film, so the display rather than your eye would set the
   floor on how well two readers can agree. <b>Zoom in before you place anything.</b></p>

   <p class="cite ifarc"><b>If one extreme is genuinely not traceable</b> &mdash; overlapped
   by the other head, off the collimated field, lost in an overexposed corner &mdash; press
   <kbd>k</kbd> for <b>Can&rsquo;t see it</b> instead of guessing. It is then computed from
   the circle and drawn as a dashed square, and recorded as <i>not observed</i>. A guessed
   landmark is worse than a missing one: it trains everything downstream to look for
   something that was never there. You will need a few extra rim points to pin the circle
   when you skip one.</p>

   <p class=ifcircle><b>What you are doing.</b> <b>Fit a circle to the femoral head.</b>
   Click to drop
   one, hover it and <b>scroll</b> to size it to the subchondral arc, drag to nudge it.
   We take its <b>centre</b> &mdash; you never have to judge where the middle is.</p>

   <p class="cite ifcircle"><b>Why a circle rather than a dot.</b> The head is very nearly a sphere,
   so its projection is a circle, and matching that circle constrains the centre with the
   <i>whole arc</i> instead of one judgement. It is also the published landmark rather
   than an approximation of it: Legaye defines the hip axis through the head
   <i>centres</i>, treating the head as a sphere. This is the Mose concentric-circle
   method with the circle drawn for you.</p>

   <p class=cite><b>Definition.</b> The bicoxofemoral (hip) axis is the line joining the
   centres of the two femoral heads; the point used for pelvic incidence is its
   <b>midpoint</b>.<br>
   &mdash; Legaye J, Duval-Beaup&egrave;re G, et&nbsp;al. <i>Pelvic incidence: a
   fundamental pelvic parameter for three-dimensional regulation of spinal sagittal
   curves.</i> Eur Spine J 1998;7:99&ndash;103.<br>
   &mdash; <a href="https://radiopaedia.org/articles/pelvic-incidence"
   target="_blank" rel="noopener">Radiopaedia &mdash; pelvic incidence</a>
   (femoral head centres as the pelvic reference axis).</p>

   <p><b>How to do it.</b></p>
   <ol class=ifarc>
    <li>Find the round dense head below and anterior to the S1 endplate, seated in the
        acetabulum. <b>Anterior is the side the pubis and abdomen are on</b>; the sacrum
        and the buttock are posterior. You do not need to know left from right &mdash;
        that is unknowable on a lateral and nothing here asks for it.</li>
    <li><b>Zoom in first</b> (scroll). At fit-to-window one screen pixel is several pixels
        of film &mdash; close enough that the display, rather than your eye, would set the
        limit on how well two readers can agree.</li>
    <li><b>Set which way the patient faces once</b>, with the <b>Faces</b> button or
        <kbd>m</kbd>. It sticks to every film afterwards, so on a series shot the same way
        you set it on the first film and never touch it again. It is what makes
        &ldquo;anterior&rdquo; mean the same direction twice, and it lets the tool tell you
        when A and P have gone the wrong way round.</li>
    <li>Click points on the <b>subchondral cortical arc</b>, the thin dense line of the
        articular surface. Follow the <i>arc</i>, not the bright shadow: overlap with the
        acetabulum puts the densest region away from the true surface, and that error is
        systematic, not noise.</li>
    <li>Name three of them <b class=lmA>A</b>, <b class=lmS>S</b>, <b class=lmP>P</b>
        &mdash; as you go, or afterwards, or with <kbd>w</kbd>. The header shows what is
        still unnamed.</li>
    <li>Then add <b>as many extra rim points as the cortex is clear for</b>. They are drawn
        as small rings rather than labelled squares, they are not landmarks, and they exist
        only to tighten the circle. More is always better.</li>
    <li><b>Drag any point</b> to correct it; the circle, the centre and the ellipse
        re-fit live. <kbd>u</kbd> removes the last point, and also undoes a
        <b>Can&rsquo;t see it</b>.</li>
    <li>The fitted circle is drawn <b>solid where you gave it evidence and dashed where it
        is extrapolating</b>. A mostly-dashed circle is a warning, not a finished mark.</li>
    <li>For a <b>second head</b>, press <kbd>h</kbd> (or New head) and mark A, S and P on
        it too. <kbd>h</kbd> also switches back if you want to add points to the first.</li>
    <li><b>Elsewhere the controls are a PACS.</b> Scroll zooms about the cursor,
        <b>left-drag</b> on bare film windows it (left-right contrast, up-down
        brightness), <b>right-drag</b> pans, <kbd>r</kbd> resets. A faint arc usually
        appears with more contrast. Zoom and pan carry over between films.</li>
   </ol>
   <ol class=ifcircle>
    <li>Find the round dense head below and anterior to the S1 endplate, seated in the
        acetabulum.</li>
    <li><b>Click</b> on it to drop a circle.</li>
    <li><b>Hover the circle and scroll</b> until its edge sits on the <b>subchondral
        cortical arc</b> &mdash; the thin dense line of the articular surface. Match the
        <i>arc</i>, not the bright shadow: overlap with the acetabulum puts the densest
        region <i>medial</i> to the true centre, and that error is systematic, not
        noise.</li>
    <li><b>Drag the circle</b> to seat it. Work the size and position together, the way
        you would slide a Mose template &mdash; when the whole visible arc lies on the
        circle, the centre is right.</li>
    <li>The radius carries to the next circle and the next film, so after the first one
        it usually needs only a nudge.</li>
    <li><b>Elsewhere the controls are a PACS.</b> Scroll <i>off</i> a circle zooms about
        the cursor, <b>left-drag</b> on bare film windows it (left-right contrast,
        up-down brightness), <b>right-drag</b> pans, <kbd>r</kbd> resets. A faint arc
        usually appears with more contrast. Zoom and pan carry over between films.</li>
    <li><b>Zoom in before you fit.</b> At fit-to-window one screen pixel is several
        pixels of film &mdash; close enough that the display, rather than your eye,
        would set the limit on how well two readers can agree.</li>
   </ol>

   <p><b>Do not centre on</b> the fovea capitis (the medial notch &mdash; a defect in the
   sphere, centring there pulls you medially), the greater trochanter, the femoral neck
   or head&ndash;neck junction, or the acetabular roof and teardrop.</p>

   <p class=cite><b>You cannot tell left from right on a lateral, and you do not need
   to.</b> The two heads are superimposed along the beam and nothing in the image
   distinguishes them. The point we need is the <b>midpoint</b> of the two centres,
   which is the same whichever way round you mark them. <b>Order does not matter</b>
   and the two marks are never compared by side.</p>

   <p><b>One circle or two?</b> Three cases, and only the third is a judgement call:</p>
   <ul>
    <li><b>You see one circle</b> &mdash; the usual well-positioned lateral, where the
        heads superimpose. <b>One circle.</b> Its centre already sits at the midpoint.</li>
    <li><b>You can clearly resolve two</b>, each with its own concentric subchondral arc.
        <b>Fit BOTH.</b> Do not pick one. The point we derive is the midpoint, so
        marking a single head on a film where the heads are separated by
        <i>S</i> puts that point <b><i>S</i>/2 away from the truth</b> &mdash; on a
        rotated film that is far larger than the agreement tolerance. Two circles is the
        <i>correct</i> answer here, not the ambitious one.</li>
    <li><b>You can see one clearly and suspect a second</b> without being able to trace
        its arc. Mark the one you are sure of and press <kbd>f</kbd> to flag it. That is
        the only case where one mark is the cautious choice.</li>
   </ul>
   <p>Rotation separates the heads mostly <i>front-to-back</i>, so two real heads sit
   side by side along the AP direction and are <b>the same diameter</b>. Candidates that
   differ in size, or are stacked well above and below one another, are almost certainly
   not two heads.</p>

   <p class=cite><b>Unsure about a film?</b> Press <kbd>f</kbd> to <b>Flag</b> it with a
   note. It goes straight to the adjudication queue with your name on it and we will look
   at it together &mdash; you cannot send films out of this tool, so this is the way to
   raise one. Flagging does not use up your read: answer the film as best you can
   afterwards, or mark it not visible.</p>

   <p class=warn><b>&ldquo;Mark only one when unsure&rdquo; means unsure that a second
   head is there</b> &mdash; not unsure which of two visible circles to choose. If you
   can see two, mark two. Both errors halve into the midpoint: a wrong second mark that
   is off by <i>E</i> costs <i>E</i>/2, and a missing second head costs <i>S</i>/2, where
   <i>S</i> is the separation. On a rotated film the missing head is usually the bigger
   error of the two.</p>

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

 </div>
</aside>
"""

EXAMPLES = """
<aside id=example>
  <h3>Worked example</h3>
  <figure>
   <img src="/reference/c" alt="two femoral heads, each with a fitted circle">
   <figcaption><b>Two heads.</b> Same diameter, offset front-to-back, a circle on each.
   The midpoint between the centres is derived for you.</figcaption>
  </figure>
  <figure>
   <img src="/reference/b" alt="the concentric-circle construction on one head">
   <figcaption><b>One head.</b> Size the circle to the subchondral cortical arc, not to
   the brightest shadow.</figcaption>
  </figure>
  <figure>
   <img src="/reference/a" alt="where the femoral head sits on a lateral film">
   <figcaption><b>Where to look.</b> Below and anterior to the S1 endplate, seated in
   the acetabulum.</figcaption>
  </figure>
  <p class=exnote>Drawn on a DRR of a segmented CT, so the marked centre is a 3-D sphere
  fit projected &mdash; not anyone's opinion. The second head in the top panel is a
  schematic: a DRR integrates along the hip axis, so its two heads superimpose exactly.</p>
</aside>
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
 /* Only one tool's instructions and controls are ever shown. The class goes on <body>
    so it reaches the guide, the header hint and the buttons from one place. */
 body.tool-arc .ifcircle{display:none}
 body.tool-circle .ifarc{display:none}
 /* The fit readout. Fixed width for the same reason as #wl: it changes on every click,
    and a header that re-wraps moves the film out from under the cursor. */
 #fitread{font-family:var(--mono);font-size:11.5px;min-width:30ch;display:inline-block;
          color:#8a94a0}
 #qcread{font-family:var(--mono);font-size:11.5px;min-width:24ch;display:inline-block;
         color:#8a94a0}
 #fitread b{font-variant-numeric:tabular-nums}
 .lmlist{margin:4px 0 10px 18px;padding:0}
 .lmlist li{margin:4px 0}
 .lmA,.lmS,.lmP{font-variant-numeric:tabular-nums}
 .lmA{color:#7cc4ff}.lmS{color:#00E5A0}.lmP{color:#f5a524}
 .ok{color:var(--lft)}.mid{color:var(--warn)}.bad{color:var(--rgt)}
 /* fixed width: this text changes on every scroll, and a header that
    re-wraps when it grows moves the film out from under the cursor. */
 #wl{font-family:var(--mono);font-size:11.5px;color:#8a94a0;
     min-width:15ch;display:inline-block}
 .bar{height:6px;background:#23232b;border-radius:3px;width:150px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--lft)}
 .lft{color:var(--lft)}.rgt{color:var(--rgt)}
 /* Two panels. The guide scrolls on its own so the film never moves while you read,
    and the film column is what the window resize actually gives space to. */
 /* Flex column, not a hardcoded header height: the header wraps to a second line as
    buttons are added, and any fixed --hdr is wrong the moment it does. */
 #appui{display:flex;flex-direction:column;height:100vh}
 #appui[hidden]{display:none}
 /* Three columns: instructions, the film, the worked example. The film keeps the
    middle and grows with the window; the two reference columns are fixed so the
    example is always beside the thing being matched rather than scrolled away. */
 #split{flex:1 1 auto;display:grid;
        grid-template-columns:var(--guide,410px) 1fr var(--example,330px);
        min-height:0}
 #split.noguide{grid-template-columns:0 1fr var(--example,330px)}
 #split.noex{grid-template-columns:var(--guide,410px) 1fr 0}
 #split.noguide.noex{grid-template-columns:0 1fr 0}
 #split.noex #example{visibility:hidden;border-left:0}
 #example{overflow-y:auto;overflow-x:hidden;min-width:0;background:#15151b;
          border-left:1px solid #2a2a33;padding:12px 14px}
 #example h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;
             letter-spacing:.1em;color:#8a94a0}
 #example figure{margin:0 0 14px}
 #example img{width:100%;border:1px solid #333;border-radius:6px;display:block;
              background:#000;cursor:zoom-in}
 #example img:hover{border-color:var(--go)}
 #example figcaption{font-size:11.5px;color:#9aa6b2;line-height:1.45;margin-top:5px}
 .exnote{font-size:11px;color:#6d7681;line-height:1.45;border-top:1px solid #2a2a33;
         padding-top:10px}
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
  <span class=ifarc>click round the rim &middot; name them
    <b class=lmA>A</b><kbd>1</kbd> <b class=lmS>S</b><kbd>2</kbd> <b class=lmP>P</b><kbd>3</kbd>
    in any order</span>
  <span class=ifcircle>click to drop a circle &middot; hover it and <b>scroll</b> to size &middot; drag to nudge</span>
  <button class="ghost ifarc" id=btnface onclick=toggleFace()
    title="which way the patient faces on this film. Sticky — set it once and it carries to every film after. m">Faces &#9664; anterior left</button>
  <button class="ghost ifarc" onclick=autoName()
    title="name your existing clicks A/S/P from the shape of the circle they fit. Never adds a point; every name stays editable.">Auto-name <kbd>w</kbd></button>
  <button class="ghost ifarc" id=btnskip onclick=skipRole()
    title="this extreme cannot be traced — derive it from the circle and record it as unobserved">Can&rsquo;t see it <kbd>k</kbd></button>
  <button class="ghost ifarc" id=btnhead onclick=newHead()>New head <kbd>h</kbd></button>
  <button class=ghost onclick=undo()>Undo <kbd>u</kbd></button>
  <button class=go onclick=send()>Submit <kbd>&crarr;</kbd></button>
  <button class=nv onclick=notVisible()>Not visible <kbd>v</kbd></button>
  <button class=sk onclick=pass()>Pass <kbd>p</kbd></button>
  <button class=ghost onclick=flagIt()>Flag <kbd>f</kbd></button>
  <a href="/board" target="_blank" class=ghost
     style="text-decoration:none;padding:7px 13px;border-radius:6px;
            border:1px solid #3a3a44">Board</a>
  <button class=ghost onclick=toggleGuide()>Guide <kbd>g</kbd></button>
  <button class=ghost onclick=toggleEx()>Example <kbd>x</kbd></button>
  <button class=ghost onclick=toggleTool()
    title="which primitive gives better agreement is an empirical question — the tool used is recorded on every read">Tool: <b id=toolname>arc</b></button>
  <span id=qcread class=ifarc title="live edge strength under the cursor, ranked against the surrounding film. The subchondral cortex is a structural edge; flat joint space and flat cancellous bone are not. An arrow points at the nearest stronger edge within a nudge."></span>
  <span id=fitread class=ifarc title="how tightly your points constrain the centre. 2σ of the fitted centre, as a fraction of image width; the consensus tolerance is 0.005."></span>
  <span id=wl title="scroll = zoom · left-drag = window/level · right-drag = pan · click = mark · r = reset. Zoom and pan carry over to the next film."></span>
  <span id=who></span>
  <div class=bar title="cases finalised"><i id=barfill style="width:0"></i></div>
  <span id=msg></span>
</header>
<main id=split>
""" + CRITERIA + """
  <section id=stage>
    <div id=wrap><canvas id=c></canvas></div>
  </section>
""" + EXAMPLES + """
</main>
</div>

<script>
let img=new Image(), cur=null, nextId=null, nextImg=null, token=null, busy=false;
// A circle per head: [cx, cy, r], all as fractions of image WIDTH (r too), so the mark
// is scale-free like the centres were. sel is the one the wheel resizes.
//
// Circles rather than bare points because the centre of a circle matched to the
// subchondral arc is over-determined by the whole arc, where a directly-clicked centre
// rests on a single judgement. It is also closer to the published landmark: Legaye
// defines the hip axis through the head CENTRES, treating the head as a sphere, so
// fitting the circle is the definition rather than an approximation of it.
let circles=[], sel=-1;
// Radius carries across films. Adult heads are a tight distribution and the films are
// the same study, so after the first one the default is already about right.
let lastR = 0.07;
let guideOn=true, exOn=true;

/* ---------------------------------------------------------------------------
   TWO ANNOTATION PRIMITIVES, and the ledger records which one produced each read.

     circle : drop a circle, scroll to size it            (the original)
     arc    : mark three NAMED extremes of the head, fit the circle

   The bicoxofemoral point is not a landmark. It is DERIVED from landmarks, the way the
   centre of a vertebral body is -- there is no edge at it, no texture, no gradient,
   because it is inside bone. So the reader marks landmarks and the centre is solved for.

   The three are the anterior, superior and posterior extremes of the femoral head
   articular surface. They are named rather than free clicks anywhere on the rim, and
   that distinction is the whole point:

     * an EXTREME is locally detectable. It is where the tangent to the cortex turns
       vertical or horizontal -- a property of the image at that spot, findable without
       already knowing the answer.
     * a free rim click is not. On a rotationally symmetric arc nothing whatsoever
       distinguishes the point at 45 degrees from the point at 50, so "the rim point at
       45 degrees" can only be defined once you have the centre. Asking a detector to
       find one is asking it to regress an inferred quantity in disguise.

   They are well spread -- anterior and posterior sit at the widest part of the head,
   level with the centre -- but they are NOT the best-conditioned three points on a
   circle. Three points 120 degrees apart are, and A/S/P at 9, 12 and 3 o'clock come out
   about half again worse. That is the price of using landmarks a reader can actually
   find, and it is why extra rim points are asked for rather than merely tolerated: the
   three names carry the identity, the extras carry the precision.

   Extra rim points are welcome and tighten the fit. They are not landmarks and are not
   exported as keypoints -- they are evidence, not targets.

   Which primitive actually agrees better between readers is an empirical question, so
   both stay available and the tool is stamped into every submission.
--------------------------------------------------------------------------- */
let TOOL='arc';
try{ TOOL = new URLSearchParams(location.search).get('tool')
          || localStorage.getItem('annot_tool') || 'arc'; }catch(e){}
if(TOOL!=='circle') TOOL='arc';

const ROLES=['A','S','P'];
// WHICH WAY THE PATIENT FACES, asserted once and then sticky.
//
// "Anterior" is a direction on a lateral, not a side of the image, and if it is not
// carried through then every A/P label in the export is a coin flip -- the exact failure
// naming the landmarks was meant to prevent. An earlier version inferred it from the
// order the reader clicked A and P, which worked but forced them to mark in a fixed
// order and gave the tool no way to check them. Asking for it outright is one control,
// it is obvious from the film, it is almost always the same for a whole series, and it
// lets the tool CATCH a swapped pair instead of faithfully recording it.
let FACE='left';
try{ FACE = localStorage.getItem('annot_face')==='right' ? 'right' : 'left'; }catch(e){}
function toggleFace(){
  FACE = FACE==='left' ? 'right' : 'left';
  try{localStorage.setItem('annot_face',FACE)}catch(e){}
  showFace(); refit(); draw(); readout();
}
function showFace(){
  const b=$('btnface'); if(!b) return;
  b.innerHTML = FACE==='left' ? 'Faces &#9664; anterior left'
                              : 'anterior right &#9654; Faces';
}
// The image direction anterior points in, and the unit vector to each extreme.
const antSign = ()=> FACE==='left' ? -1 : 1;
function roleDir(r){
  if(r==='S') return [0,-1];
  return [(r==='A' ? antSign() : -antSign()), 0];
}
const ROLE_NAME={A:'ANTERIOR extreme', S:'SUPERIOR extreme', P:'POSTERIOR extreme', '':'extra rim point'};
// HD[k] = one femoral head:
//   pts  [{x,y,role}]  in placement order, role '' for an extra rim point
//   skip {A,S,P}       an extreme the reader could not see; derived from the fit instead
let HD=[{pts:[], skip:{}}], acur=0, fits=[], sel2=null;   // sel2 = {arc,i} selected point
const newHd=()=>({pts:[], skip:{}});
// The landmark the next click will become: the first of A,S,P neither placed nor skipped,
// then '' for extras. This is a SUGGESTION, never a constraint -- see relabel().
function nextRole(h){
  for(const r of ROLES) if(!h.skip[r] && !h.pts.some(p=>p.role===r)) return r;
  return '';
}
const roleAt=(h,r)=>h.pts.find(p=>p.role===r) || null;
// NOTHING IS LOCKED IN. Any point can be renamed to any role at any time, in any order,
// and the previous holder of that role becomes an ordinary rim point rather than being
// deleted. A reader who marks four points and only then decides which is the anterior
// extreme is working normally; a tool that forces A-then-S-then-P makes them undo their
// way backwards to fix a label, which is how you train people to submit a wrong one.
function relabel(r, target){
  if(TOOL!=='arc') return;
  const t = target || sel2 || (HD[acur].pts.length
              ? {arc:acur, i:HD[acur].pts.length-1} : null);
  if(!t){ msg('click a point first, then press '+(r||'e')+' to name it'); return; }
  const h=HD[t.arc], p=h.pts[t.i];
  if(r && p.role!==r){
    const prev=h.pts.find(q=>q.role===r);
    if(prev) prev.role='';                    // demoted, not discarded
    delete h.skip[r];                         // naming it un-skips it
  }
  p.role = r;
  acur=t.arc; sel2={arc:t.arc, i:t.i};
  refit(); draw(); readout();
  msg(r ? ('that point is now the '+ROLE_NAME[r]) : 'that point is now an extra rim point');
}
// Assumed click noise, as a fraction of image width. With exactly three points the
// residuals are ZERO by construction, so residual-based error bars would report a perfect
// centre on a 20-degree arc -- precisely the case this is meant to catch. Propagating a
// floor of click noise through the fit is what makes the ellipse tell the truth there.
//
// 0.003 of image width is ~4 px on a BUU film: a prior, not a measurement. Once there are
// arc reads in the ledger it can be replaced by the residual spread actually observed,
// and the readout recalibrated rather than argued about.
const SIG0 = 0.003;
// The consensus tolerance the pair of reads is scored against, so the readout is in the
// units that decide whether this film needs adjudication.
const TOL = 0.005;

/* ---------------------------------------------------------------------------
   PLACEMENT QC: is this click actually ON the cortex?

   The failure mode is specific and common: the reader lands in the dark joint space or in
   flat cancellous bone a few pixels off the cortical line, and nothing on screen says so.
   At working zoom that error is invisible and it is systematic, which is the worst kind.

   The test is EDGE STRENGTH, not brightness. The subchondral cortex is a structural
   boundary, so it is a gradient ridge whichever way the film is displayed -- and reads
   the same on an inverted film, where a brightness test would be exactly backwards. Flat
   dark joint space and flat cancellous bone both score low, which is the distinction that
   matters. It is a HINT computed from pixels, not a truth: the reader's anatomy always
   wins, and nothing here blocks a submission.

   Sobel over the whole film once, then every probe is a disc of array reads.
--------------------------------------------------------------------------- */
let LUM=null, GRAD=null, GMAX=null, QCW=0, QCH=0, qcReady=false;
function buildQC(){
  qcReady=false; LUM=null; GRAD=null; GMAX=null;
  if(!img.width) return;
  const w=img.width, h=img.height;
  let dat;
  try{
    const oc=document.createElement('canvas'); oc.width=w; oc.height=h;
    const og=oc.getContext('2d',{willReadFrequently:true});
    og.drawImage(img,0,0);
    dat=og.getImageData(0,0,w,h).data;
  }catch(err){ return; }        // tainted canvas: QC stays off rather than lying
  const lum=new Uint8Array(w*h);
  for(let i=0,p=0;i<lum.length;i++,p+=4)
    lum[i]=(dat[p]*299+dat[p+1]*587+dat[p+2]*114)/1000;
  const g=new Uint8Array(w*h);
  for(let y=1;y<h-1;y++){
    const r0=(y-1)*w, r1=y*w, r2=(y+1)*w;
    for(let x=1;x<w-1;x++){
      const gx=(lum[r0+x+1]+2*lum[r1+x+1]+lum[r2+x+1])
              -(lum[r0+x-1]+2*lum[r1+x-1]+lum[r2+x-1]);
      const gy=(lum[r2+x-1]+2*lum[r2+x]+lum[r2+x+1])
              -(lum[r0+x-1]+2*lum[r0+x]+lum[r0+x+1]);
      const m=Math.sqrt(gx*gx+gy*gy)/4;
      g[r1+x]= m>255 ? 255 : m;
    }
  }
  // separable max filter: the plateau that makes the whole cortical band score alike
  const rd=Math.max(2, Math.round(w*0.0025)), tmp=new Uint8Array(w*h), mx=new Uint8Array(w*h);
  for(let y=0;y<h;y++){
    const row=y*w;
    for(let x=0;x<w;x++){
      let m=0, x0=x-rd<0?0:x-rd, x1=x+rd>=w?w-1:x+rd;
      for(let i=x0;i<=x1;i++) if(g[row+i]>m) m=g[row+i];
      tmp[row+x]=m;
    }
  }
  for(let x=0;x<w;x++){
    for(let y=0;y<h;y++){
      let m=0, y0=y-rd<0?0:y-rd, y1=y+rd>=h?h-1:y+rd;
      for(let j=y0;j<=y1;j++) if(tmp[j*w+x]>m) m=tmp[j*w+x];
      mx[y*w+x]=m;
    }
  }
  // the noise floor: the median of the dilated gradient, which in any radiograph is
  // dominated by the large smooth areas rather than by the few edges
  const hist=new Uint32Array(256);
  for(let i=0;i<mx.length;i++) hist[mx[i]]++;
  let acc=0; QNOISE=0;
  for(let v=0;v<256;v++){ acc+=hist[v]; if(acc>=mx.length/2){ QNOISE=v; break; } }
  LUM=lum; GRAD=g; GMAX=mx; QCW=w; QCH=h; qcReady=true;
}
const QC_R1=()=>Math.max(4, Math.round(QCW*0.007));   // how far a nudge may be suggested
const QC_R2=()=>Math.max(9, Math.round(QCW*0.020));   // what counts as "nearby structure"
const _gm=(x,y)=>GMAX[y*QCW+x];

/* The score is a RATIO anchored on the film's own noise floor, not a rank.

   Ranking the spot against its neighbourhood was tried first and is quietly broken: in a
   flat region every pixel ties, so the rank is decided by sensor noise and a click in the
   middle of featureless cancellous bone came back at 60% -- indistinguishable from the
   cortex. Worse, the rank depended on how much cortex happened to fall inside the
   sampling disc, so the same anatomy scored differently at different points along it.

   What the reader is asking is absolute: "is there real structure here, or am I on flat
   film?". So the answer is measured between two references from the film itself -- its
   noise floor, and the strongest structure within reach:

       0%   = indistinguishable from noise
       100% = as strong as the best edge anywhere nearby

   If there is no structure nearby at all, there is nothing to be near and the answer is
   0 rather than a meaningless 100. */
let QNOISE=0;
function qcAt(pxf, pyf){
  if(!qcReady) return null;
  const x=Math.round(pxf), y=Math.round(pyf);
  const r1=QC_R1(), r2=QC_R2();
  if(x<r2+3||y<r2+3||x>=QCW-r2-3||y>=QCH-r2-3) return null;
  const here=_gm(x,y);
  let hi=0, bx=x, by=y, bv=here;
  for(let dy=-r2;dy<=r2;dy++){
    for(let dx=-r2;dx<=r2;dx++){
      const d2=dx*dx+dy*dy;
      if(d2>r2*r2) continue;
      const v=_gm(x+dx,y+dy);
      if(v>hi) hi=v;
      if(d2<=r1*r1 && v>bv){ bv=v; bx=x+dx; by=y+dy; }
    }
  }
  const span=hi-QNOISE;
  // Nothing within reach stands out from the noise: say so instead of ranking noise
  // against noise and returning a confident number.
  if(span < Math.max(6, QNOISE*0.8))
    return {pct:0, dx:0, dy:0, bpct:0, d:0, flat:true};
  const sc=v => Math.max(0, Math.min(100, Math.round(100*(v-QNOISE)/span)));
  const pct=sc(here), bp=sc(bv);
  // DEADBAND. The score is normalised against the strongest structure within reach, so
  // there is almost always SOME pixel a little stronger -- rim cortex varies by a few
  // percent along its length from noise and resampling alone. Without this, a click dead
  // on the cortex still gets an arrow telling it to move, and a hint that fires when the
  // reader is already right is worse than no hint: they stop believing it.
  if(pct>=QC_GOOD || bp-pct < 15)
    return {pct:pct, dx:0, dy:0, bpct:pct, d:0, flat:false};
  return {pct:pct, dx:bx-x, dy:by-y, bpct:bp,
          d:Math.round(Math.hypot(bx-x,by-y)), flat:false};
}
const QC_GOOD=70, QC_WEAK=40;
const qcClass=q => !q ? '' : (q.pct>=QC_GOOD ? 'ok' : (q.pct>=QC_WEAK ? 'mid' : 'bad'));

function solve3(M, v){                       // Gaussian elimination, partial pivoting
  const A=[[M[0][0],M[0][1],M[0][2],v[0]],
           [M[1][0],M[1][1],M[1][2],v[1]],
           [M[2][0],M[2][1],M[2][2],v[2]]];
  for(let c=0;c<3;c++){
    let p=c;
    for(let r=c+1;r<3;r++) if(Math.abs(A[r][c])>Math.abs(A[p][c])) p=r;
    if(Math.abs(A[p][c])<1e-12) return null;
    const t=A[c]; A[c]=A[p]; A[p]=t;
    for(let r=0;r<3;r++){
      if(r===c) continue;
      const f=A[r][c]/A[c][c];
      for(let k=c;k<4;k++) A[r][k]-=f*A[c][k];
    }
  }
  return [A[0][3]/A[0][0], A[1][3]/A[1][1], A[2][3]/A[2][2]];
}
function inv3(M){
  const [a,b,c]=M[0], [d,e,f]=M[1], [g,h,i]=M[2];
  const det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  if(Math.abs(det)<1e-18) return null;
  return [[ (e*i-f*h)/det, (c*h-b*i)/det, (b*f-c*e)/det],
          [ (f*g-d*i)/det, (a*i-c*g)/det, (c*d-a*f)/det],
          [ (d*h-e*g)/det, (b*g-a*h)/det, (a*e-b*d)/det]];
}

/* Least-squares circle through N>=3 points, in PIXELS.

   Kasa's algebraic fit for the starting guess, then Gauss-Newton on the GEOMETRIC cost
   sum(|p-c| - R)^2. The refinement is not cosmetic: the algebraic fit minimises an
   algebraic residual that is weighted by distance from the centre, which biases the
   radius on SHORT ARCS -- exactly the films we are worried about.

   Returns the centre, the radius, the observed angular interval, and the 2x2 covariance
   of the centre. That covariance is the point of the whole exercise: for a shallow arc it
   is enormous along the radial direction, which is how the tool can say "your points do
   not determine this centre" instead of drawing a crisp crosshair in the wrong place. */
function fitCircle(P){
  const n=P.length;
  if(n<3) return null;
  let Sx=0,Sy=0,Sxx=0,Syy=0,Sxy=0,Sz=0,Sxz=0,Syz=0;
  for(const p of P){ const x=p[0], y=p[1], z=x*x+y*y;
    Sx+=x; Sy+=y; Sxx+=x*x; Syy+=y*y; Sxy+=x*y; Sz+=z; Sxz+=x*z; Syz+=y*z; }
  const s=solve3([[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,n]], [-Sxz,-Syz,-Sz]);
  if(!s) return null;                        // collinear clicks: no circle exists
  let a=-s[0]/2, b=-s[1]/2;
  let R=Math.sqrt(Math.max(1e-6, a*a+b*b-s[2]));
  if(!isFinite(a)||!isFinite(b)||!isFinite(R)) return null;

  let JTJ=null, sse=0;
  for(let it=0; it<20; it++){
    JTJ=[[0,0,0],[0,0,0],[0,0,0]];
    let JTr=[0,0,0]; sse=0; let bad=false;
    for(const p of P){
      const dx=p[0]-a, dy=p[1]-b, d=Math.hypot(dx,dy);
      if(d<1e-9){ bad=true; break; }
      const J=[-dx/d, -dy/d, -1], r=d-R;
      sse+=r*r;
      for(let i=0;i<3;i++){ JTr[i]+=J[i]*r;
        for(let k=0;k<3;k++) JTJ[i][k]+=J[i]*J[k]; }
    }
    if(bad) break;
    const dl=solve3(JTJ, [-JTr[0],-JTr[1],-JTr[2]]);
    if(!dl) break;
    a+=dl[0]; b+=dl[1]; R+=dl[2];
    if(Math.hypot(dl[0],dl[1],dl[2])<1e-8) break;
  }
  if(!(R>0) || !isFinite(a) || !isFinite(b)) return null;

  // observed angular interval = everything except the largest gap between clicks
  const th=P.map(p=>Math.atan2(p[1]-b, p[0]-a)).sort((u,v)=>u-v);
  let gi=-1, gm=th[0]+2*Math.PI-th[n-1];
  for(let i=0;i<n-1;i++) if(th[i+1]-th[i]>gm){ gm=th[i+1]-th[i]; gi=i; }
  const lo = gi<0 ? th[0] : th[gi+1];
  const hi = gi<0 ? th[n-1] : th[gi]+2*Math.PI;
  const span = hi-lo;

  // sigma^2 (J'J)^-1, with a floor of click noise. n===3 gives zero residual and no
  // degrees of freedom, so without the floor the error bars would be identically zero.
  const dof = n-3;
  const s2 = Math.max(dof>0 ? sse/dof : 0, Math.pow(SIG0*img.width, 2));
  const C = JTJ ? inv3(JTJ) : null;
  let cxx=Infinity, cxy=0, cyy=Infinity;
  if(C){ cxx=s2*C[0][0]; cxy=s2*C[0][1]; cyy=s2*C[1][1]; }

  // 2x2 eigen-decomposition -> the uncertainty ellipse. For a short arc the major axis
  // points along the radius, away from the marked cortex, which is the honest picture.
  const tr=cxx+cyy, det=cxx*cyy-cxy*cxy;
  const disc=Math.sqrt(Math.max(0, tr*tr/4-det));
  const l1=tr/2+disc, l2=Math.max(0, tr/2-disc);
  const ang = Math.abs(cxy)<1e-12 ? (cxx>=cyy?0:Math.PI/2)
                                  : Math.atan2(l1-cxx, cxy);
  return {a:a, b:b, R:R, n:n, lo:lo, hi:hi, span:span,
          rms: Math.sqrt(sse/n), sa: Math.sqrt(Math.max(0,cxx)),
          sb: Math.sqrt(Math.max(0,cyy)),
          e1: Math.sqrt(l1), e2: Math.sqrt(l2), eang: ang};
}
/* WHERE an extra point would help, scored rather than guessed.

   Extra clicks are NOT equally useful, and saying "add a few more anywhere" was wrong.
   Starting from A/S/P, two more points on the superior rim buy 11%; ONE point at the
   inferior margin buys 42%, and takes the fit past what three evenly-spaced points can
   do. The reason is visible in the covariance: with A and P at the sides and S on top,
   the horizontal position of the centre is pinned by two points averaging, while the
   VERTICAL position rests on S almost alone -- so S's noise passes through undiluted and
   the uncertainty ellipse is a vertical cigar. Only a point away from that axis fixes it.

   Each candidate angle is scored by the centre covariance it would produce, so the tool
   names the part of the rim to click instead of asking for more of the same. */
function bestNextAngle(P, f){
  if(!f || P.length<3) return null;
  const rows=P.map(p=>{
    const t=Math.atan2(p.y*img.height-f.b, p.x*img.width-f.a);
    return [-Math.cos(t), -Math.sin(t), -1];
  });
  const score=extra=>{
    const R = extra===null ? rows
            : rows.concat([[-Math.cos(extra), -Math.sin(extra), -1]]);
    const M=[[0,0,0],[0,0,0],[0,0,0]];
    R.forEach(r=>{ for(let i=0;i<3;i++) for(let k=0;k<3;k++) M[i][k]+=r[i]*r[k]; });
    const C=inv3(M);
    if(!C) return Infinity;
    const cxx=C[0][0], cxy=C[0][1], cyy=C[1][1], tr=cxx+cyy;
    return tr/2 + Math.sqrt(Math.max(0, tr*tr/4 - (cxx*cyy-cxy*cxy)));
  };
  const now=score(null);
  let best=null, bv=Infinity;
  for(let i=0;i<24;i++){ const t=i*Math.PI/12, v=score(t); if(v<bv){ bv=v; best=t; } }
  // only worth saying if it is a real improvement, not a rounding one
  return (best!==null && bv < now*0.8) ? best : null;
}
// The rim named the way a reader thinks about it, which depends on which way the patient
// faces -- image-left is anterior on one film and posterior on the next.
function rimName(t){
  const L = FACE==='left' ? 'anterior' : 'posterior';
  const Rt = FACE==='left' ? 'posterior' : 'anterior';
  const names=[Rt, 'infero-'+Rt, 'inferior', 'infero-'+L, L, 'supero-'+L,
               'superior', 'supero-'+Rt];
  const d=((t*180/Math.PI)%360+360)%360;
  return names[Math.round(d/45)%8];
}

function refit(){
  if(!img.width){ fits = HD.map(()=>null); return; }
  // every mark carries its own placement score, recomputed as it is dragged
  HD.forEach(h=>h.pts.forEach(p=>{ p.qc = qcAt(p.x*img.width, p.y*img.height); }));
  fits = HD.map(h => h.pts.length>=3
    ? fitCircle(h.pts.map(p=>[p.x*img.width, p.y*img.height])) : null);
  // Which way the patient faces, read off the two extremes rather than asked for. On a
  // lateral this is the only thing that makes "anterior" mean the same direction twice.
  fits.forEach((f,k)=>{
    if(!f) return;
    const a=roleAt(HD[k],'A'), p=roleAt(HD[k],'P');
    f.facing = FACE;
    // A congruent spherical head puts the anterior and posterior extremes at the SAME
    // height, level with the centre. A large tilt between them is either a mislabelled
    // pair or a head that is not spherical -- dysplasia, migration, arthritic collapse --
    // and both are worth surfacing rather than averaging away.
    f.aptilt = (a&&p)
      ? Math.abs(Math.atan2((p.y-a.y)*img.height, (p.x-a.x)*img.width))*180/Math.PI : null;
    // Now that facing is asserted, a swapped pair is CATCHABLE: the anterior extreme has
    // to sit on the anterior side of the fitted centre. Silently recording a swap would
    // put an anterior label on posterior cortex in the training set.
    f.swapped = !!(a && p && (a.x - p.x) * antSign() < 0);
  });
}
// The header readout, in the units the consensus test actually uses: 2 sigma of the
// fitted centre as a fraction of image width, against the 0.005 tolerance. A reader who
// can see this number stops guessing whether the arc was long enough.
function readout(){
  const el=$('fitread'); if(!el) return;
  if(TOOL!=='arc'){ el.textContent=''; return; }
  const bits=[];
  let worst=0, any=false, tilt=0;
  HD.forEach((h,k)=>{
    const f=fits[k];
    if(!h.pts.length) return;
    if(!f){ bits.push((k+1)+': '+h.pts.length+' pt'+(h.pts.length===1?'':'s')+' (need 3)');
            worst=Infinity; return; }
    any=true;
    const two=2*f.e1/img.width;
    worst=Math.max(worst, two);
    if(f.aptilt!==null) tilt=Math.max(tilt, f.aptilt);
    const named=ROLES.filter(r=>roleAt(h,r)).join('');
    bits.push((k+1)+': '+(named||'-')+(h.pts.length>named.length
                 ? '+'+(h.pts.length-named.length) : '')
              +'  \\u00b1'+two.toFixed(4));
  });
  if(!bits.length){
    el.className='ifarc';
    el.textContent='mark the '+ROLE_NAME[nextRole(HD[acur])];
    return;
  }
  // how many marks are actually sitting on a structural edge
  const all=HD.flatMap(h=>h.pts).filter(p=>p.qc);
  const onc=all.filter(p=>p.qc.pct>=QC_GOOD).length;
  const qcbit = all.length ? '   cortex '+onc+'/'+all.length : '';
  let cls = !any||worst>2*TOL ? 'bad' : (worst>TOL ? 'mid' : 'ok');
  let tag = !any ? '' : (worst>2*TOL ? '  shallow' : (worst>TOL ? '  widen' : '  good'));
  // 12 degrees is about a head-radius/10 offset between the two extremes: past that it is
  // not click noise.
  if(tilt>12){ tag += '  A/P tilt '+Math.round(tilt)+'\\u00b0';
               if(cls==='ok') cls='mid'; }
  if(fits.some(f=>f&&f.swapped)){ tag += '  A/P LOOK SWAPPED'; cls='bad'; }
  if(all.length && onc<all.length && cls==='ok') cls='mid';
  // not just "add more points" -- which part of the rim, measured
  if(cls!=='ok' && fits[acur]){
    const w=bestNextAngle(HD[acur].pts, fits[acur]);
    if(w!==null) tag += '  → add a point: '+rimName(w)+' margin';
  }
  el.className='ifarc '+cls;
  el.textContent=bits.join('   ')+qcbit+tag;
}
// Name the reader's existing clicks from the shape of what they drew. It never invents a
// point -- it only decides which of the ones already on screen are the extremes, which is
// the tedious part -- and every assignment stays editable afterwards.
function autoName(){
  if(TOOL!=='arc')return;
  const h=HD[acur], f=fits[acur];
  if(!f){ msg('mark at least 3 points first'); return; }
  const used=new Set();
  ROLES.forEach(r=>{
    if(h.skip[r]) return;
    const d=roleDir(r);
    let best=-1, bestc=-2;
    h.pts.forEach((p,i)=>{
      if(used.has(i)) return;
      const vx=(p.x*img.width-f.a)/f.R, vy=(p.y*img.height-f.b)/f.R;
      const n=Math.hypot(vx,vy)||1, c=(vx*d[0]+vy*d[1])/n;
      if(c>bestc){ bestc=c; best=i; }
    });
    // 0.5 is 60 degrees off the extreme. Naming a point further away than that would be
    // asserting an extreme the reader never went near.
    if(best>=0 && bestc>0.5){ used.add(best); h.pts.forEach(p=>{ if(p.role===r) p.role=''; });
                              h.pts[best].role=r; }
  });
  refit(); draw(); readout();
  const got=ROLES.filter(r=>roleAt(h,r));
  msg(got.length===3 ? 'named A, S and P from the shape — correct any that are wrong'
      : 'named '+(got.join('')||'nothing')+' — no click was close enough for the rest');
}
// Windowing and zoom, PACS-style. The loupe this replaces was covering the anatomy
// beside the point being placed, which is the one thing you need to see.
//
// ZOOM IS NOT A LUXURY HERE. Fit-to-window puts a BUU film at ~436 px wide on a
// 1600x900 screen, and the consensus tolerance of 0.005 of image width is then 2.2
// SCREEN PIXELS. No one clicks that accurately, so at fit-to-window the tool itself
// was setting the floor on inter-reader agreement.
let gainB=1, gainC=1, zoom=1;
// Zoom AND pan persist across films. Every film in this set is the same view of the same
// anatomy, so a reader who has zoomed to the hips wants to still be there on the next
// one -- resetting per film means re-navigating 2000 times. Pan is kept as a FRACTION of
// the scrollable range, not pixels, so it lands in the same anatomical region on films
// of different size.
let panFx=0.5, panFy=0.5;
function rememberPan(){
  const st=$('stage');
  const mx=st.scrollWidth-st.clientWidth, my=st.scrollHeight-st.clientHeight;
  if(mx>1) panFx=clamp(st.scrollLeft/mx,0,1);
  if(my>1) panFy=clamp(st.scrollTop/my,0,1);
}
function restorePan(){
  const st=$('stage');
  st.scrollLeft=panFx*Math.max(0, st.scrollWidth-st.clientWidth);
  st.scrollTop =panFy*Math.max(0, st.scrollHeight-st.clientHeight);
}
const clamp=(v,lo,hi)=>Math.min(hi,Math.max(lo,v));
function showWL(){
  $('wl').textContent = (zoom>1.01?('zoom '+zoom.toFixed(1)+'x  '):'')
    + 'B '+Math.round(gainB*100)+'%  C '+Math.round(gainC*100)+'%';
}
function resetView(){ gainB=1; gainC=1; zoom=1; panFx=0.5; panFy=0.5;
                      showWL(); fit(); draw(); }
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
  applyTool(); showFace();
  try{ toggleGuide(localStorage.getItem('annot_guide')!=='0');
       toggleEx(localStorage.getItem('annot_ex')!=='0'); }catch(e){}
  // click any example to open it full size in a tab
  document.querySelectorAll('#example img').forEach(im=>
    im.onclick=()=>window.open(im.src,'_blank','noopener'));
  showWL();
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
  circles=[]; sel=-1; HD=[newHd()]; acur=0; fits=[]; sel2=null; sel2=null; readout();
  const t0=performance.now();
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
    img.onload=()=>{C.width=img.width;C.height=img.height;showWL();fit();
                  refit();draw();readout();restorePan();
      // after first paint: the reader is orienting, and a Sobel pass over a 3 MP film
      // costs tens of milliseconds that must not sit in front of the image appearing
      setTimeout(()=>{buildQC(); refit(); draw(); readout();}, 0);
      msg(cur.case_id+'  slot '+cur.slot+'  ('+Math.round(performance.now()-t0)+' ms)');
      nextId=null;nextImg=null;prefetch();};
    if(src){img.src=src}
    else{const b=await fetch(cur.image_url,{headers:H()});
         img.src=URL.createObjectURL(await b.blob());}
  } finally { busy=false; }
}
const C=$('c'), X=C.getContext('2d');
// Fit the film to the space left over, preserving aspect ratio, and leave a strip at
// the bottom: the Windows taskbar auto-shows at the last few pixels of the screen and
// eats the pointer before it reaches the caudal anatomy.
function fit(){
  if(!img.width) return;
  // Size from the CONTAINER'S OWN content box, never from window.innerHeight minus the
  // stage's top. The header wraps to a second line as buttons are added, and a canvas
  // sized against the one-line header hangs 30 px past the gutter and back into the
  // taskbar. #stage is a grid row, so its box already accounts for whatever the header
  // is doing -- and the bottom padding IS the gutter.
  const st=$('stage'), cs=getComputedStyle(st);
  const availW=Math.max(80, st.clientWidth
    - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight));
  const availH=Math.max(80, st.clientHeight
    - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom));
  const k=Math.min(availW/img.width, availH/img.height)*zoom;
  C.style.width =(img.width*k)+'px';
  C.style.height=(img.height*k)+'px';
}
function draw(){
  // filter the film only -- the markers must stay full contrast, or a bright window
  // washes out the very thing you just placed
  X.filter='brightness('+gainB+') contrast('+gainC+')';
  X.drawImage(img,0,0);
  X.filter='none';
  const lw=Math.max(1.5, img.width/700);
  if(TOOL==='arc'){ drawArcs(); return; }
  circles.forEach((c,i)=>{
    // 1st / 2nd circle, NOT left/right -- a lateral cannot tell you which is which
    X.strokeStyle = i===0 ? '#00E5A0' : '#FF3B30';
    X.lineWidth = lw;
    const x=c[0]*img.width, y=c[1]*img.height, R=c[2]*img.width;
    X.beginPath(); X.arc(x,y,R,0,7); X.stroke();          // the fitted head
    const t=Math.max(6, R*0.28);                          // centre crosshair
    X.beginPath(); X.moveTo(x-t,y); X.lineTo(x+t,y);
    X.moveTo(x,y-t); X.lineTo(x,y+t); X.stroke();
    if(i===sel){                                          // the one the wheel resizes
      X.setLineDash([lw*3, lw*3]); X.lineWidth=lw*0.8;
      X.beginPath(); X.arc(x,y,R*1.18,0,7); X.stroke();
      X.setLineDash([]);
    }
  });
  if(circles.length===2){                                 // the derived hip point
    const a=circles[0], b=circles[1];
    const mx=(a[0]+b[0])/2*img.width, my=(a[1]+b[1])/2*img.height;
    X.strokeStyle='#f5a524'; X.lineWidth=lw;
    X.beginPath();
    X.moveTo(a[0]*img.width,a[1]*img.height);
    X.lineTo(b[0]*img.width,b[1]*img.height);
    X.stroke();
    X.fillStyle='#f5a524';
    X.beginPath(); X.arc(mx,my,Math.max(3,lw*2),0,7); X.fill();
  }
}
const COL=['#00E5A0','#FF3B30'];      // 1st / 2nd head. NOT left/right -- see heads().
// Marks are sized in SCREEN pixels, not image pixels.
//
// The canvas is drawn at the film's native size and then CSS-scaled, so anything sized in
// image pixels grows with the zoom: at 4x a marker becomes a blob covering the cortex it
// is marking, and at fit-to-window the same marker is too small to grab. Dividing by the
// display scale keeps every mark the same size under the reader's eye at every zoom,
// which is what makes zooming in to work actually usable.
// unit vector from the fitted centre out through a mark, for placing its label clear of
// the anatomy. Falls back to straight up before there is a fit to point away from.
function outward(f,x,y){
  if(!f) return [0,-1];
  const dx=x-f.a, dy=y-f.b, L=Math.hypot(dx,dy);
  return L<1e-6 ? [0,-1] : [dx/L, dy/L];
}
function drawArcs(){
  const bb=C.getBoundingClientRect();
  const k = bb.width>1 ? img.width/bb.width : 1;      // image px per screen px
  const lw=1.7*k, rr=5*k, fs=13*k, xh=7*k;
  const ctr=[];
  HD.forEach((h,kk)=>{
    const col=COL[kk%2], f=fits[kk], active=(kk===acur);
    // The marks, always on top of the fit. Named landmarks are drawn as labelled squares
    // and extras as small rings, so a reader can see at a glance which of their clicks
    // are the three that get exported and which are only feeding the circle.
    h.pts.forEach((p,i)=>{
      const x=p.x*img.width, y=p.y*img.height;
      // the selected point, which 1/2/3 will rename
      if(sel2 && sel2.arc===kk && sel2.i===i){
        X.strokeStyle='#fff'; X.lineWidth=lw*0.8; X.globalAlpha=0.9;
        X.beginPath(); X.arc(x,y,rr*2.4,0,7); X.stroke(); X.globalAlpha=1;
      }
      X.strokeStyle=col; X.lineWidth=lw;
      if(p.role){
        X.strokeRect(x-rr, y-rr, rr*2, rr*2);
        // The letter sits RADIALLY OUTWARD from the centre, so it never covers the
        // cortex it is labelling -- always above S, out to the side for A and P.
        const o=outward(f,x,y);
        // dark halo behind the letter: it has to stay readable over bright cortex as
        // well as over dark joint space, and it sits on both
        X.font='bold '+fs+'px system-ui';
        X.textAlign='center'; X.textBaseline='middle';
        X.lineWidth=fs*0.28; X.strokeStyle='rgba(0,0,0,.85)';
        X.strokeText(p.role, x+o[0]*rr*2.8, y+o[1]*rr*2.8);
        X.fillStyle=col; X.fillText(p.role, x+o[0]*rr*2.8, y+o[1]*rr*2.8);
        X.lineWidth=lw; X.strokeStyle=col;
      } else {
        X.beginPath(); X.arc(x,y,rr*0.75,0,7); X.stroke();
      }
      X.fillStyle=col; X.globalAlpha=0.55;
      X.beginPath(); X.arc(x,y,rr*0.45,0,7); X.fill();
      X.globalAlpha=1;
      // PLACEMENT HINT. A mark that is not on an edge gets an arrow to the nearest spot
      // that is, so the correction is a glance and a drag rather than a hunt. Nothing is
      // moved automatically -- the reader decides whether the arrow is pointing at the
      // cortex or at some other structure that happens to have an edge.
      const q=p.qc;
      if(q && q.pct<QC_GOOD && (q.dx||q.dy)){
        X.strokeStyle = q.pct<QC_WEAK ? '#FF3B30' : '#f5a524';
        X.lineWidth=lw*1.1;
        const L=Math.hypot(q.dx,q.dy)||1, ux=q.dx/L, uy=q.dy/L;
        // the arrow is drawn at the TRUE offset, so its length is the distance the reader
        // has to move -- a hint that exaggerated the gap would be its own error
        const x0=x+ux*rr*1.6, y0=y+uy*rr*1.6;
        const x1=x+q.dx, y1=y+q.dy;
        X.beginPath(); X.moveTo(x0,y0); X.lineTo(x1,y1); X.stroke();
        const ah=5*k;
        X.beginPath();
        X.moveTo(x1,y1);
        X.lineTo(x1-ah*(ux*0.87-uy*0.5), y1-ah*(uy*0.87+ux*0.5));
        X.moveTo(x1,y1);
        X.lineTo(x1-ah*(ux*0.87+uy*0.5), y1-ah*(uy*0.87-ux*0.5));
        X.stroke();
        X.beginPath(); X.arc(x,y,rr*1.7,0,7); X.stroke();
      }
    });
    if(!f) return;
    // SOLID where the reader gave evidence, DASHED where the circle is extrapolating.
    // A mostly-dashed circle is the visual form of the same warning the readout gives.
    X.strokeStyle=col; X.lineWidth=lw;
    X.beginPath(); X.arc(f.a,f.b,f.R,f.lo,f.hi); X.stroke();
    X.setLineDash([lw*3,lw*4]); X.globalAlpha=0.6;
    X.beginPath(); X.arc(f.a,f.b,f.R,f.hi,f.lo+2*Math.PI); X.stroke();
    X.setLineDash([]); X.globalAlpha=1;
    // the derived centre -- the thing we are actually collecting. Deliberately SMALL:
    // it used to be sized off the head radius, which drew a 26 px cross over a 10 px
    // uncertainty ellipse and hid the one number the reader is supposed to react to.
    const t=xh;
    X.lineWidth=lw*1.3;
    X.beginPath(); X.moveTo(f.a-t,f.b); X.lineTo(f.a+t,f.b);
    X.moveTo(f.a,f.b-t); X.lineTo(f.a,f.b+t); X.stroke();
    // ...and how well it is pinned. 2 sigma, so the ellipse is the region the centre
    // could plausibly be in. On a wide arc it collapses inside the crosshair; on a
    // superior-only arc it stretches into a long radial cigar, which is the truth.
    X.setLineDash([lw*2,lw*2]); X.lineWidth=lw*1.1;
    X.strokeStyle = 2*f.e1/img.width > 2*TOL ? '#FF3B30'
                  : (2*f.e1/img.width > TOL ? '#f5a524' : col);
    X.beginPath();
    X.ellipse(f.a, f.b, Math.max(1.5*k,2*f.e1), Math.max(1.5*k,2*f.e2), f.eang, 0, 7);
    X.stroke(); X.setLineDash([]);
    // An extreme the reader could not see is derived from the fit and drawn hollow, so it
    // is obvious on the film which of the three are evidence and which are inference.
    ROLES.forEach(r=>{
      if(!h.skip[r] || !f) return;
      const d=roleDir(r);
      const x=f.a+f.R*d[0], y=f.b+f.R*d[1];
      X.strokeStyle=col; X.globalAlpha=0.5; X.setLineDash([lw*1.5,lw*2]); X.lineWidth=lw;
      X.strokeRect(x-rr, y-rr, rr*2, rr*2);
      X.font='italic '+fs+'px system-ui';
      X.textAlign='center'; X.textBaseline='middle';
      X.lineWidth=fs*0.28; X.strokeStyle='rgba(0,0,0,.85)'; X.setLineDash([]);
      X.strokeText(r, x+d[0]*rr*2.8, y+d[1]*rr*2.8);
      X.fillStyle=col; X.fillText(r, x+d[0]*rr*2.8, y+d[1]*rr*2.8);
      X.globalAlpha=1;
    });
    if(active && HD.length>1){                   // which head the next click joins
      X.strokeStyle=col; X.globalAlpha=0.35; X.lineWidth=lw*0.8;
      X.beginPath(); X.arc(f.a,f.b,f.R+6*k,0,7); X.stroke(); X.globalAlpha=1;
    }
    ctr.push([f.a,f.b]);
  });
  if(ctr.length===2){                            // the bicoxofemoral point
    X.strokeStyle='#f5a524'; X.lineWidth=lw;
    X.beginPath(); X.moveTo(ctr[0][0],ctr[0][1]); X.lineTo(ctr[1][0],ctr[1][1]); X.stroke();
    X.fillStyle='#f5a524';
    X.beginPath(); X.arc((ctr[0][0]+ctr[1][0])/2,(ctr[0][1]+ctr[1][1])/2,
                         Math.max(3,lw*2),0,7); X.fill();
  }
}
/* PACS mapping:
     wheel        zoom, about the cursor
     left drag    window / level  (left-right contrast, up-down brightness)
     left click   place the mark
     right drag   pan
   The left button does double duty, so a press only becomes a window drag once it has
   moved past DRAG_MIN; below that it is still a click. Without the threshold a slightly
   shaky hand would re-window the film instead of marking it. */
function zoomAt(e, inwards){
  const st=$('stage'), r=C.getBoundingClientRect();
  const fx=(e.clientX-r.left)/r.width, fy=(e.clientY-r.top)/r.height;
  const before=zoom;
  zoom=clamp(zoom*(inwards?1.15:1/1.15), 1, 12);
  if(zoom===before) return;
  fit();
  const r2=C.getBoundingClientRect();      // keep what was under the cursor there
  st.scrollLeft += (fx*r2.width  - (e.clientX - r2.left));
  st.scrollTop  += (fy*r2.height - (e.clientY - r2.top));
  rememberPan(); draw(); showWL();
}
// Image coordinates (fraction of width/height) under the pointer.
function at(e){
  const r=C.getBoundingClientRect();
  return [(e.clientX-r.left)/r.width, (e.clientY-r.top)/r.height];
}
// Which circle the pointer is inside, or -1. Aspect matters: x is a fraction of width
// and y a fraction of height, so the y term has to be rescaled before comparing to r.
function hit(p){
  const asp=img.height/img.width;
  for(let i=circles.length-1;i>=0;i--){
    const c=circles[i];
    const dx=p[0]-c[0], dy=(p[1]-c[1])*asp;
    if(Math.hypot(dx,dy) <= c[2]*1.15) return i;
  }
  return -1;
}

// The wheel resizes the circle under the cursor and zooms everywhere else. Hovering the
// thing you want to change is unambiguous, and it keeps the wheel doing one obvious job
// in each place rather than needing a modifier.
// Which rim click the pointer is on, or null. The grab radius is in SCREEN pixels, so it
// stays grabbable at every zoom level instead of shrinking to nothing when zoomed out.
function hitPt(e){
  const r=C.getBoundingClientRect(), sx=r.width/img.width, sy=r.height/img.height;
  const p=at(e);
  for(let k=HD.length-1;k>=0;k--){
    for(let i=HD[k].pts.length-1;i>=0;i--){
      const dx=(p[0]-HD[k].pts[i].x)*img.width*sx, dy=(p[1]-HD[k].pts[i].y)*img.height*sy;
      if(Math.hypot(dx,dy) <= 11) return {arc:k, i:i};
    }
  }
  return null;
}

C.addEventListener('wheel',e=>{
  if(!img.width)return;
  e.preventDefault();
  // In arc mode there is nothing to resize -- the radius is solved, not set -- so the
  // wheel does one job everywhere.
  if(TOOL==='arc'){ zoomAt(e, e.deltaY<0); return; }
  const i=hit(at(e));
  if(i>=0){
    sel=i;
    circles[i][2]=clamp(circles[i][2]*(e.deltaY<0?1.06:1/1.06), 0.01, 0.35);
    lastR=circles[i][2];                 // carries to the next circle and the next film
    draw();
    return;
  }
  zoomAt(e, e.deltaY < 0);
},{passive:false});

const DRAG_MIN = 4;
let drag = null;
C.addEventListener('pointerdown',e=>{
  if(!img.width || (e.button !== 0 && e.button !== 2)) return;
  const st=$('stage'), p=at(e);
  const pt=(TOOL==='arc' && e.button===0) ? hitPt(e) : null;
  const i=(TOOL!=='arc' && e.button===0) ? hit(p) : -1;
  if(i>=0) sel=i;
  if(pt){ acur=pt.arc; sel2={arc:pt.arc, i:pt.i}; }   // grabbing a point selects it
  drag={x:e.clientX, y:e.clientY, b:gainB, c:gainC, moved:0, btn:e.button,
        circle:i, cx:i>=0?circles[i][0]:0, cy:i>=0?circles[i][1]:0,
        pt:pt, px:pt?HD[pt.arc].pts[pt.i].x:0, py:pt?HD[pt.arc].pts[pt.i].y:0,
        sl:st.scrollLeft, stp:st.scrollTop};
  try{ C.setPointerCapture(e.pointerId); }catch(err){}
  if(i>=0||pt) draw();
});
C.addEventListener('pointermove',e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.moved=Math.max(drag.moved, Math.hypot(dx,dy));
  if(drag.btn === 2){                       // right: pan, image follows the cursor
    const st=$('stage');
    st.scrollLeft = drag.sl - dx;
    st.scrollTop  = drag.stp - dy;
    rememberPan();
    return;
  }
  if(drag.pt){                              // drag a rim click; everything re-fits live
    if(drag.moved < DRAG_MIN) return;
    const r=C.getBoundingClientRect();
    HD[drag.pt.arc].pts[drag.pt.i].x=clamp(drag.px + dx/r.width, 0, 1);
    HD[drag.pt.arc].pts[drag.pt.i].y=clamp(drag.py + dy/r.height, 0, 1);
    refit(); draw(); readout();
    return;
  }
  if(drag.circle >= 0){                     // nudge the circle you grabbed
    if(drag.moved < DRAG_MIN) return;
    const r=C.getBoundingClientRect();
    circles[drag.circle][0]=clamp(drag.cx + dx/r.width, 0, 1);
    circles[drag.circle][1]=clamp(drag.cy + dy/r.height, 0, 1);
    draw();
    return;
  }
  if(drag.moved < DRAG_MIN) return;
  gainC=clamp(drag.c + dx*0.005, 0.3, 5);
  gainB=clamp(drag.b - dy*0.004, 0.2, 3);
  showWL(); draw();
});
C.addEventListener('pointerup',e=>{
  const d=drag; drag=null;
  if(!d) return;
  try{ C.releasePointerCapture(e.pointerId); }catch(err){}
  // a click on empty film drops a mark; a click on an existing one just selects it
  if(d.moved < DRAG_MIN && d.btn === 0 && d.circle < 0 && !d.pt) place(e);
});
/* LIVE HOVER PROBE. The reader can find the cortex BEFORE committing a click, which is
   the difference between a QC that scolds and one that helps.

   Deliberately text-only: it never redraws the film. A full redraw on every mousemove
   would repaint a 3 MP image through a brightness/contrast filter, and the pointer lag
   that produces is exactly the thing that makes an annotation tool horrible to use. The
   probe itself is a couple of thousand array reads. */
const ARROWS=['\\u2192','\\u2198','\\u2193','\\u2199','\\u2190','\\u2196','\\u2191','\\u2197'];
let hoverRAF=0;
function hoverProbe(e){
  const el=$('qcread'); if(!el) return;
  if(TOOL!=='arc' || !img.width){ el.textContent=''; return; }
  if(!qcReady){ el.className='ifarc'; el.textContent='reading film\\u2026'; return; }
  const p=at(e);
  const q=qcAt(p[0]*img.width, p[1]*img.height);
  if(!q){ el.className='ifarc'; el.textContent=''; return; }
  let t='edge '+q.pct+'%';
  if(q.pct<QC_GOOD && (q.dx||q.dy)){
    const a=ARROWS[(Math.round(Math.atan2(q.dy,q.dx)/(Math.PI/4))+8)%8];
    t += '   '+a+' '+q.d+'px \\u2192 '+q.bpct+'%';
  } else if(q.pct>=QC_GOOD){ t += '  on cortex'; }
  el.className='ifarc '+qcClass(q);
  el.textContent=t;
}
C.addEventListener('pointermove',e=>{
  if(drag) return;                         // dragging has its own, heavier, path
  if(hoverRAF) return;
  hoverRAF=requestAnimationFrame(()=>{ hoverRAF=0; hoverProbe(e); });
});
C.addEventListener('pointerleave',()=>{ const el=$('qcread'); if(el) el.textContent=''; });
C.addEventListener('pointercancel',()=>{drag=null});
C.addEventListener('contextmenu',e=>e.preventDefault());   // right-drag is panning

function place(e){
  if(!img.width)return;
  const p=at(e);
  if(TOOL==='arc'){
    const h=HD[acur], r=nextRole(h);
    h.pts.push({x:p[0], y:p[1], role:r});
    sel2={arc:acur, i:h.pts.length-1};
    refit(); draw(); readout();
    const nx=nextRole(h);
    msg(!r ? (h.pts.length+' points on head '+(acur+1))
        : nx ? (ROLE_NAME[r]+' marked \\u2014 now the '+ROLE_NAME[nx])
             : 'all three marked \\u2014 add rim points to tighten it, or Submit');
    return;
  }
  if(circles.length>=2)return;
  circles.push([p[0], p[1], lastR]);
  sel=circles.length-1;
  draw();
}
function undo(){
  if(TOOL==='arc'){
    const h=HD[acur];
    // A skip is a decision too, and it is invisible on the film, so it has to be
    // undoable -- otherwise a mis-keyed skip permanently downgrades a landmark that the
    // reader could actually see.
    const lastSkip=ROLES.filter(r=>h.skip[r]).pop();
    if(h.pts.length) h.pts.pop();
    else if(lastSkip) delete h.skip[lastSkip];
    // an emptied second head is removed rather than left as a stub that submits as a head
    else if(acur>0){ HD.splice(acur,1); acur=HD.length-1; }
    sel2=null; refit(); draw(); readout();
    return;
  }
  circles.pop(); sel=circles.length-1; draw();
}
// An extreme that genuinely cannot be traced. It is derived from the fitted circle and
// exported as NOT observed, so the film is still usable for the centre without teaching a
// detector to find a landmark that was never visible.
function skipRole(){
  if(TOOL!=='arc')return;
  const h=HD[acur], r=nextRole(h);
  if(!r){ msg('all three extremes are already marked'); return; }
  h.skip[r]=true;
  refit(); draw(); readout();
  const nx=nextRole(h);
  msg(ROLE_NAME[r]+' skipped \\u2014 it will be derived'
      + (nx ? '. Now the '+ROLE_NAME[nx] : '. Add rim points so the circle is determined'));
}
// h starts a second head, and thereafter switches between them -- a reader who spots a
// missed piece of the first rim after starting the second should not have to undo to it.
function newHead(){
  if(TOOL!=='arc')return;
  if(HD.length<2){
    if(HD[acur].pts.length<3){ msg('finish this head first — 3 points minimum'); return; }
    HD.push(newHd()); acur=1;
  } else {
    acur=(acur+1)%HD.length;
  }
  msg('head '+(acur+1)+' \\u2014 mark the '+ROLE_NAME[nextRole(HD[acur])]);
  draw(); readout();
}
function applyTool(){
  document.body.classList.toggle('tool-arc', TOOL==='arc');
  document.body.classList.toggle('tool-circle', TOOL==='circle');
  const t=$('toolname'); if(t) t.textContent=TOOL;
}
function toggleTool(){
  TOOL = (TOOL==='arc') ? 'circle' : 'arc';
  try{localStorage.setItem('annot_tool',TOOL)}catch(e){}
  circles=[]; sel=-1; HD=[newHd()]; acur=0; fits=[];
  applyTool(); draw(); readout();
  msg('tool: '+TOOL+' — marks cleared');
}

function toggleGuide(on){
  guideOn = (on===undefined) ? !guideOn : on;
  $('split').classList.toggle('noguide', !guideOn);
  requestAnimationFrame(fit);
  try{localStorage.setItem('annot_guide', guideOn?'1':'0')}catch(e){}
}
function toggleEx(on){
  exOn = (on===undefined) ? !exOn : on;
  $('split').classList.toggle('noex', !exOn);
  requestAnimationFrame(fit);
  try{localStorage.setItem('annot_ex', exOn?'1':'0')}catch(e){}
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
  let payload;
  if(TOOL==='arc'){
    const use=HD.map((h,k)=>({h:h,f:fits[k]})).filter(u=>u.h.pts.length>0);
    if(!use.length){msg('mark the anterior, superior and posterior extremes, or Not visible');return}
    if(use.some(u=>u.h.pts.length<3)){
      msg('a head needs at least 3 points — add them or press u to remove it');return}
    if(use.some(u=>!u.f)){
      msg('those points do not define a circle — spread them around the rim');return}
    // Every extreme has to be accounted for: marked, or explicitly skipped. Submitting
    // with one silently unfilled would export a landmark that was never decided on.
    const open=use.map(u=>ROLES.filter(r=>!roleAt(u.h,r) && !u.h.skip[r])).flat();
    if(open.length){
      msg('still to mark: '+open.map(r=>ROLE_NAME[r]).join(', ')
          +' — or press k if it cannot be seen');return}
    const F=use.map(u=>u.f);
    // heads = the FITTED centres, unordered, in exactly the shape the circle tool sent --
    // so agreement, adjudication and the board keep working untouched.
    //
    // landmarks = the three NAMED extremes, the only part of this a detector can be
    // trained on directly. The centre is not an image feature; an extreme is. Each is
    // either a click ("obs") or derived from the fit because the reader could not see it
    // ("derived") -- and that distinction is what stops a training set from teaching a
    // model to find rim that was never visible.
    //
    // facing is recorded because "anterior" is a direction on a lateral, not a side of
    // the image, and it varies between films. Without it every A/P label in the export
    // would be a coin flip.
    payload={tool:'arc',
             heads:F.map(f=>[f.a/img.width, f.b/img.height]),
             radii:F.map(f=>f.R/img.width),
             landmarks:use.map(u=>{
               const o={};
               ROLES.forEach(r=>{
                 const p=roleAt(u.h,r);
                 o[r]= p ? {xy:[+p.x.toFixed(5), +p.y.toFixed(5)], src:'obs'}
                         : {src:'derived'};
               });
               return o;
             }),
             facing:F.map(()=>FACE),
             ap_swapped:F.map(f=>!!f.swapped),
             ap_tilt_deg:F.map(f=>f.aptilt===null?null:+f.aptilt.toFixed(1)),
             // extra rim points: evidence for the fit, never keypoint targets
             extra:use.map(u=>u.h.pts.filter(p=>!p.role)
                                 .map(p=>[+p.x.toFixed(5), +p.y.toFixed(5)])),
             // observed angular interval per head, in radians, image coordinates: which
             // part of each circle is evidence and which part is extrapolation.
             arc_lo:F.map(f=>+f.lo.toFixed(4)),
             arc_span:F.map(f=>+f.span.toFixed(4)),
             // 2 sigma of the fitted centre along the major/minor axes, as a fraction of
             // image width, with the major axis direction. A read whose own error bar
             // exceeds the consensus tolerance is not a disagreement to adjudicate.
             sigma2:F.map(f=>[+(2*f.e1/img.width).toFixed(5),
                              +(2*f.e2/img.width).toFixed(5),
                              +f.eang.toFixed(4)]),
             rms:F.map(f=>+(f.rms/img.width).toFixed(5)),
             w:img.width,h:img.height};
  } else {
    if(!circles.length){msg('circle a femoral head, or use Not visible');return}
    // radii ride along: they give a per-film scale reference and flag an ambiguous arc
    // when two readers fit very different circles.
    payload={tool:'circle',
             heads:circles.map(c=>[c[0],c[1]]),
             radii:circles.map(c=>c[2]),
             w:img.width,h:img.height};
  }
  const j=await post('/submit',{case_id:cur.case_id,slot:cur.slot,
                                points:JSON.stringify(payload)});
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
  // \\n, not a real line break: this file is a Python string, so a bare \\n here becomes
  // an actual newline inside a JS single-quoted literal -- a SyntaxError that takes the
  // whole script down and leaves the reader a blank page.
  const note=prompt('What is unclear about this film?\\n'
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
  else if(e.key==='r')resetView();
  else if(e.key==='g')toggleGuide();
  else if(e.key==='x')toggleEx();
  else if(e.key==='f')flagIt();
  else if(e.key==='h')newHead();
  else if(e.key==='k')skipRole();
  // 1/2/3 rename the selected point A/S/P, 0 demotes it to an extra rim point. This is
  // what makes the order free: mark whatever you can see, decide the names afterwards.
  //
  // Digits rather than the obvious a/s/p because p is Pass and always has been. Rebinding
  // a destructive key that readers already have in their fingers, so that a new frequent
  // key could have a nicer mnemonic, is how you get films passed by accident. a and s are
  // free and work as aliases; there is deliberately no p alias.
  else if(e.key==='1')relabel('A');
  else if(e.key==='2')relabel('S');
  else if(e.key==='3')relabel('P');
  else if(e.key==='0')relabel('');
  else if(e.key==='a')relabel('A');
  else if(e.key==='s')relabel('S');
  else if(e.key==='e')relabel('');
  else if(e.key==='w')autoName();
  else if(e.key==='m')toggleFace();
});
window.addEventListener('resize', fit);
// The header wraps as its contents change, which moves #stage without firing a window
// resize. Sizing the canvas from a stale #stage box left the film hanging past the
// bottom gutter -- straight back into the taskbar.
if(window.ResizeObserver) new ResizeObserver(()=>fit()).observe($('stage'));
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
 body{display:flex;flex-direction:column;height:100vh}
 #rsplit{flex:1 1 auto;display:grid;grid-template-columns:330px 1fr;min-height:0}
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
  const st=$('rstage'), cs=getComputedStyle(st);
  const aw=Math.max(80, st.clientWidth
    - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight));
  const ah=Math.max(80, st.clientHeight
    - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom));
  const k=Math.min(aw/img.width, ah/img.height);
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
  img.onload=()=>{C.width=img.width;C.height=img.height;showWL();fit();draw();
                  restorePan();};
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
if(window.ResizeObserver) new ResizeObserver(()=>fit()).observe($('rstage'));
loadQueue();
</script>
"""
