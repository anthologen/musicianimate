# Drum animation research

Notes behind `fingering.py` (the striking planner) and `animate_drums.py`
(the keyframer). The goal: drive a two-handed, two-footed drummer from a
General MIDI drum track, with motion that reads as real playing and a kit
whose strike targets can be swapped out (`kit_layout.py`).

## 1. Prior work on animating drummers from MIDI

- **IK-driven virtual drummers.** Sporka et al.'s *Drum Set Tutorial System*
  and *An Animated Virtual Drummer* convert a note description (essentially a
  MIDI drum track) into a seated human animation by (a) assigning each stroke
  to a hand and (b) solving inverse kinematics so the stick tip reaches the
  struck surface, inferring a drumstick trajectory between hits. The hard part
  they identify is the **hand-to-stroke assignment** and keeping the arms out
  of each other's way on a kit of arbitrary layout.
- **Learned trajectories (DRUMS, SIGGRAPH MIG 2025).** A more recent system
  predicts fine-grained 3-D hand trajectories from raw MIDI with a
  bi-directional LSTM, **parents the sticks to the hands**, matches upper-body
  and facial motion to the MIDI, and adds a **procedural foot module** for the
  pedals. The takeaways that generalise to a lightweight procedural rig: sticks
  are rigid children of the hands (so you animate the hand and the tip
  follows), and feet are handled separately from hands.

Our approach is the lightweight, deterministic version of the IK papers: no
learned model and no runtime IK. The sticks are rigid (tip at a fixed point in
the hand's local frame), so placing the hand *is* placing the tip — we solve
for the hand location in closed form and bake plain keyframes, exactly the
pattern the guitar pick-hand already uses in this repo.

## 2. The General MIDI percussion map (channel 10)

GM fixes a note-number -> drum-sound table (notes 35-81) on MIDI channel 10,
so we route entirely off **note number**, never channel (the repo's
`parse_midi` masks the channel away anyway). The demo writes on channel 10 for
correct DAW playback. The subset folded onto this 5-piece kit:

| Note | GM sound | Voice | Note | GM sound | Voice |
|-----:|----------|-------|-----:|----------|-------|
| 35/36 | Bass Drum | `kick` | 46 | Open Hi-Hat | `hihat_open` |
| 37 | Side Stick | `side_stick` | 49 | Crash 1 | `crash` |
| 38/40 | Snare | `snare` | 51/59 | Ride | `ride` |
| 41/43 | Floor Tom | `tom_floor` | 52 | Chinese | `china` |
| 45/47 | Low/Low-mid Tom | `tom_mid` | 53 | Ride Bell | `ride_bell` |
| 48/50 | Hi-mid/High Tom | `tom_hi` | 55 | Splash | `splash` |
| 42 | Closed Hi-Hat | `hihat_closed` | 57 | Crash 2 | `crash2` |
| 44 | Pedal Hi-Hat | `hihat_pedal` | | | |

Voices with no dedicated object fold onto an existing one (china/splash/crash2
-> `Crash`, ride bell -> `Ride`, side stick -> `Snare`). Unmapped notes are
dropped with a warning. All of this — the note map, each voice's physical
target, strike point, mechanism and default limb — lives in `kit_layout.py`,
the one file to edit when swapping kits.

## 3. Sticking heuristics (which limb plays each hit)

Drumming sticking is a well-worn craft; the rules we encode:

- **Standard right-handed convention.** Right hand keeps time on the hi-hat /
  ride, left hand plays the snare, right foot the kick, left foot the hi-hat
  pedal. On most kits the hi-hat sits to the player's left, so the right hand
  **crosses over** the left to reach it (cross-arm playing) — the default we
  use. (Open-handed playing avoids the cross; it's a valid alternative the
  `kit_layout` conventions could be flipped to.)
- **Feet are deterministic.** Kick -> right foot, hi-hat pedal -> left foot,
  with no interaction with the hands. A pedal hi-hat note ("chick") also
  drives the hi-hat open/closed state.
- **Alternating single strokes for speed.** Fast runs and fills are played
  hand-to-hand (R L R L) so no single hand has to move faster than it can. We
  trigger alternation whenever the inter-onset gap drops below `FAST_GAP`
  (0.14 s ≈ sixteenths at 120 BPM); slower passages keep the convention hand.
- **Minimise arm travel on flex voices.** Toms and crashes have no fixed hand;
  each goes to the hand whose *last position* is nearest the target, with a
  penalty (`CONV_PENALTY`) for abandoning a voice's convention hand — so a
  groove keeps its natural grip while a fill grabs the closest stick.
- **Split simultaneous hits.** A chord of hits (e.g. snare + hi-hat on the
  backbeat) is divided across the two hands to minimise total travel, so one
  hand is never asked to be in two places at once.

These map directly to the planner: convention comes from
`kit_layout.VOICES[...]['limb']`, alternation is the `FAST_GAP` branch, and
travel minimisation is a nearest-hand cost with the convention penalty.

## 4. Motion model — the Moeller method (whip + full/down/tap/up strokes)

A real stroke is *anticipation -> contact -> rebound*: the stick lifts to a
backswing apex, drops to the head at the note onset, and bounces back. Louder
notes are struck from a **taller backswing dropped faster** — the velocity model
we reuse from the guitar pick-hand:

    v = velocity / 127
    lift   = LIFT_MIN + v·(LIFT_MAX − LIFT_MIN)     # accent apex above the head
    strike = STRIKE_SLOW − v·(STRIKE_SLOW − STRIKE_FAST)  # apex→contact seconds

But scaling *every* hit's backswing by its own velocity in isolation is not how a
drummer actually moves. Sanford **Moeller's method** (Chapin's *Speed, Power,
Control, Endurance*; Moeller's *The Art of Snare Drumming*) — the canonical
efficiency technique — says a player **doesn't re-lift the stick between every
note**. Each hit is played as one of two gestures:

- an **accent**: a whole-arm **whip** — the forearm/elbow leads down and the bead
  *trails then cracks through* (like snapping a whip), the stick reared up high
  beforehand; or
- a **tap**: a small **wrist flick** from just above the head.

Crucially, *which gesture, plus what comes next,* sets the stroke's **start and
end height**, giving Moeller's four canonical strokes (each named for where the
bead starts→ends):

| this hit | next hit | stroke | backswing → rebound |
|----------|----------|--------|---------------------|
| accent | accent | **Full** | high → high |
| accent | tap    | **Down** | high → low  |
| tap    | tap    | **Tap**  | low → low   |
| tap    | accent | **Up**   | low → high  |

So the stick **rears up high only to load an accent** and **rides low through
taps**, and — the economy the method is prized for — a single up-motion doubles
as a tap's rebound *and* the next accent's backswing ("two notes for one arm
movement"). The classic **down→tap→up triple** and **up↔down double** fall
straight out of the table; no special-casing.

`_moeller_strokes(notes)` implements this per hand. A hit is an **accent** if it
is loud in absolute terms (`v ≥ ACCENT_ABS ≈ 0.62`) *or* stands out from that
hand's own **median dynamic** (`ACCENT_REL` louder) — so a flat ghost-note roll
reads as all taps, a backbeat or the loud half of a crescendo pops as accents,
and the classification tracks the music rather than a fixed threshold. The
backswing height is the velocity-scaled accent apex (up to ~20 cm) for an accent
and a small `TAP_LIFT` (~3 cm) for a tap; the **rebound is set to the *next*
hit's backswing height** so the two coincide (staying low before a tap, rearing
up before an accent), easing to a neutral `READY_LIFT` across a rest longer than
`MOELLER_RESET_GAP`.

The **whip itself** lives in how the wrist empty (the forearm-IK target) is
keyed relative to the tip empty. On an **accent** the wrist bobs a large fraction
(`FOREARM_FLEX_ACCENT`) of the apex and, decisively, **drops onto the play anchor
early** (`WHIP_LEAD` of the way down) so the *arm arrives before the bead* — the
stick lags and then snaps through at contact, the visual "crack". A **tap** barely
moves the arm (`FOREARM_FLEX_TAP`, a wrist-only flick) and gets no early lead. In
both cases the wrist is **exactly on the anchor at contact**, so `|wrist − tip| ==
STICK_LEN` and the bead still lands on the head (verified: median 7.5 mm across
both demos — a constant bead-radius offset, i.e. exact — with the sole outlier a
pre-existing >2-voice simultaneous cluster the *planner* stacks on one hand, not a
motion error). `_replane_strokes` then re-aims each stroke's now-per-height apex
and rebound perpendicular to the stick (gravity-aligned, §5) without touching the
contact key. Between strikes the hand still glides from one target toward the
next, honouring the "minimise movement" heuristic visually.

Feet follow the same idea: the kick beater cocks **further back when louder**
before swinging into the batter head (rest −10° -> +6° at contact), and the
footboards press in step (carrying the parented shoes). The hi-hat foot holds
the top cymbal open or closed along the planner's hi-hat timeline, with a
quick pedal chick on each left-foot note. Struck cymbals get a short,
velocity-scaled wobble that decays back to rest.

## 5. A seated humanoid stand-in, driven by target + velocity

The hands are not free-floating: they belong to a blocky, faceless mock
humanoid (`build_drummer.py`) seated at the kit with **average adult proportions**
(`ANTHRO`). Segment lengths follow the Drillis & Contini / Winter anthropometric
tables (length as a fraction of stature H, taken at H = 1.75 m): upper arm 0.186 H,
forearm 0.146 H, thigh 0.245 H, shank 0.246 H, biacromial shoulder breadth 0.259 H,
head height ~0.130 H. The leg bones are solved from these lengths (a two-bone knee
placement between the fixed hip and the pedal-planted ankle), so the whole figure
stays in proportion if `ANTHRO` is swapped for a different body. A small **seat
offset** (`SEAT_DX/SEAT_DY`) slides the seated upper body toward the kit (the feet
stay on their pedals, the knees re-solve) so the arms play with lower, more folded
elbows; it is kept modest because the drums are spread out — shifting the seat far
toward the hi-hat pulls the shoulder off the floor tom on the other side.

**Facing the snare (ergonomic angle).** A right-handed drummer sits angled slightly
toward the snare / hi-hat rather than square to the kick, so the **whole body** —
pelvis, hips and legs as well as the torso/shoulders/head — yaws ~18 deg toward +X
(`SEAT_YAW` about the pelvis, applied by `_seat`), which puts the snare centred
between the thighs like the ergonomic reference (Fig 2). The **feet** are placed at
the raw pedal positions (not seated), so yawing the hips only splays the thighs to
the pedals; each knee is then solved into its **pedal's vertical plane**
(`_solve_knee`) so the shin runs straight down the pedal and the feet stay planted
(only the round throne stays square — a symmetric seat needs no facing). (Earlier
only the torso yawed over square hips; rotating the pelvis to match reads as the
player squaring their seat to the snare, not just twisting over it.)
Turning toward that side leaves the far-side drums (ride, floor tom) a little behind
the turn, so the one piece that now needs a cross-body reach is the floor tom — it is
pulled in toward the seat in `kit_layout` (roughly matching a real setup) so the
angled drummer reaches it in front instead of across. With that plus elbow poles
chosen to keep the elbow out of the (now-turned) torso, arm-through-body clipping is
essentially eliminated (2 of 188 sampled frames) while every strike still lands
(median 0 mm) — the drummer reads as ergonomically angled *relative to* the kit. This matters
because a seated drummer is the **centre of lateral rotation** — a stick reaching
the floor tom must swing out from the shoulder, not stay pointed at the audience.
Rooting the arms at the shoulders makes that happen for free, and the cross-arm
hi-hat grip emerges naturally (the right arm reaches across to the player's left).

Each arm has a **wrist joint** (three bones): a two-bone IK chain (upper arm +
forearm) reaches a *wrist* empty, and a separate hand bone carrying the
drumstick Damped-Tracks the *stick-tip* empty at the drum. This decouples the
elbow from the stick: without it, a rigid forearm-plus-stick segment over-folds
reaching a close, low drum like the snare and the elbow rides up at shoulder
height. With the wrist, the forearm only reaches the wrist, the elbow drops
~0.25 m below the shoulder, and the wrist angles the stick onto the head — while
the tracked tip still lands on the strike point.

The stick is a **standard 16" (5A) drumstick**, held **near its butt end** — only
a small (~3 cm) overhang trails behind the fist, so almost the whole stick is
working length swinging at the drum. Its **attitude is set by the target**, because
`wrist_target()` places the grip one stick-length back down a shaft held at a
**chosen pitch** (`stick_pitch(voice)`), horizontal heading running from the shoulder
out to the drum:

- **Flat drums played from above** (snare, floor tom, side-stick — `FLAT_VOICES`):
  the stick sits **nearly level, angled just slightly down** (`STICK_PITCH_FLAT`
  ≈ −8°), hovering over the head. Straight-down stabbing is what a jab looks like and
  it read as unnatural.
- **Everything else** (rack toms, hats, ride, crashes): the **bead points up**
  (`STICK_PITCH_UP` ≈ +28°). Reaching up/across to those pieces that way is more
  ergonomic, and a real strike's rebound flicks the tip back up — so an up-pitched
  ready pose is what the wrist rotates from.

Because the arm is short (~0.58 m reach) and the shoulder sits high, a bead-up grip
on a *far* piece drops the hand too low to reach. Rather than diving the grip to the
shoulder (which jabs the bead steeply **down** — the old failure), `wrist_target()`
**lowers the pitch only as far as anatomy forces**, taking the *highest reachable*
stick: so the hi-hat lands ~+18°, the crashes ~+25°, the ride +28°, while snare and
floor tom stay ~−8°. Only genuine cross-body reaches beyond arm+stick (e.g. the left
hand crossing to the floor tom) fall through to the tip→shoulder pull and tip down.
`|wrist − tip|` stays STICK_LEN throughout, so the tip still lands exactly (verified
≤~2 cm both hands). A pole target keeps each elbow low and tucked.

The hand's `LIMIT_ROTATION` wrist envelope (`WRIST_ROT_LIMIT`) is applied on **both
hands with the same values** (`_WRIST_ENVELOPE`). For that shared limit to mean the
same thing on each side, both hand bones are first **rolled to the same reference —
local Z up** (`hand_b.align_roll` in `_build_skeleton`). Without that, the left hand
inherited a *sideways*-rolled frame from its forearm, so its local-Euler play range
sat ~60° off the right's; a right-sized limit then clamped the *lifted apex* and flung
the left stick sideways (the bead missing its vertical target empty by up to ~45 cm).
With the frames aligned, one envelope — sized to contain the full observed play range
(wider on Z, where the left's cross-body snare reach twists the wrist most) — bounds
both hands' gross extremes **without clamping the strokes**: the bead tracks its target
empty (≤~1.5 cm, i.e. no clamp) and the left wind-up stays vertical (lateral component
~0). Earlier this limit was disabled on the left as a stopgap; re-conditioning the
frame let it be restored, consistent with the right.

Even with the left stroke vertical, the left elbow rode into the (slim) chest box —
first forward on the raised wind-up, then, more subtly, **inboard while idle**: with
no left note playing, the ending-bar torso twist toward the ride/cymbals rotated the
arm base past the two-bone IK's **elbow-solution bifurcation**, flipping the elbow
between an outboard (~x 0.33) and an inboard, chest-clipping (~x 0.15) branch. Two
changes hold it on the outboard branch: the IK **pole angle** is raised
(`POLE_ANGLE["L"]` 45→100°) so the elbow hangs back-and-down behind the chest, and the
left **pole target is parented to the chest bone** so it *twists with the torso* —
keeping the pole on the same side of the shoulder→wrist axis regardless of the spine
twist, so the solution can't flip. (Cranking only the pole angle high enough to clear
the twisted idle instead winged the *active* elbow out; parenting the pole fixes it at
a tucked angle.) The right elbow's pole is pinned by its far cross-body hi-hat reach
(moving it to clear a ~3 mm graze of the pelvis box wrecks that landing), so that
negligible touch is left as-is.

**Elbows hang and tuck, the wrist does the stroke.** Studying seated-drummer
posture references (Drummerworld, Melodics, Drum Helper) the consistent advice is:
let the shoulders hang and the elbows sit *low and tucked near the sides*, forearms
doing the reaching — "you only really need the action of your forearms," and "at no
point should you have to fully extend your arm to reach any part of the kit." A
raised or winged-out elbow is both tiring and unnatural.

Two mechanisms give this. First, the **stroke is mostly wrist, with a little
forearm.** The animator keyframes the wrist empty (which the forearm IK reaches,
placing the elbow) with a small vertical **bob** through each stroke — up a fraction
(`FOREARM_FLEX ≈ 0.3`) of the tip lift at the wind-up apex, then back onto the
contact-height anchor at impact — while the tip empty alone winds up and drops the
rest of the way. So the forearm/elbow now *joins* the stroke (the elbow flexes ~4–9°
per hit) but the hand/wrist still leads it, matching how a real player hits with a
combination of the two. Because the wrist returns exactly to the anchor at contact,
`|wrist − tip| == STICK_LEN` there and the tip still lands on the head. For repeated
hits on one drum the anchor is identical frame to frame, so the arm holds station and
only the small per-hit flex plus the wrist flick play — the arm only *travels* when
moving to a *different* drum. (Before, the wrist empty bobbed the *whole* apex height
on every hit, so the arm rose and fell ~7 cm per hi-hat note; before that fix it held
perfectly still with zero forearm motion. The current small bob sits between the two:
an economical groove where the elbow visibly helps without the arm heaving.)

The wind-up and rebound are then swung so the bead **cocks straight up in the stick's
vertical plane** — perpendicular to the stick (a clean wrist hinge, no change in
`|wrist − tip|`) and along gravity. Two earlier attempts were worse: a pure world-Z
lift twisted the wrist ("comes in from the side"), and lifting in the forearm–stick
plane flung the *cross-body* left bead sideways (the wind-up went "all the way left")
because the left forearm reaches down-and-across, tilting that plane out to the side.
Taking instead the vertical component of the up-vector perpendicular to the stick
keeps the swing gravity-aligned: for a forward-pointing stick (the snare) it cocks
essentially straight up (the left now lifts only ~2% sideways, ~8° off vertical); for
a stick angled up and across (the hats/ride) it tilts to stay perpendicular but stays
vertical (~19°, unchanged from before). A refinement pass (`_replane_strokes`) re-aims
each apex/rebound this way after the arms are keyed; the contact key is left untouched,
so the landing is unchanged. Because at contact `wrist == anchor` and `tip == p`, the
stick direction is just `p − anchor`, so no rig read-back is needed.

Second, elbow position is set by the IK **pole targets**. Far-out poles (well off to
each side) winged the elbows outboard — the left splayed ~0.26 m past its shoulder on
the backbeat, and the right rode *up to shoulder height* on its constant cross-body
reaches (hi-hat/ride live on the player's left), which also made the IK flip the
elbow between a high and low solution between adjacent frames, so the arm appeared to
snap mid-fill. Pulling both poles IN close to the body and — critically — *behind*
the shoulder (pole y > shoulder y) rolls the elbows down, tucked, and hanging back
rather than forward: measured worst case is still below the shoulder, mean outboard
splay is ~0, and the mid-fill snap is gone. (A pole placed *forward* of the shoulder
had pulled the resting elbow forward across the chest — the over-flexed frame-1 pose
where the left elbow sat dead-centre and 19 cm in front of the shoulder.) The torso
stays comparatively still and the forearm/wrist do the reaching.

**Joint range-of-motion limits (AAOS human norms).** On top of the pole placement,
each joint is capped so the IK can never resolve into an anatomically impossible
pose. The norms used: shoulder flexion 0-180 / extension 0-60 / abduction 0-180 /
internal rotation 0-70 / external rotation 0-90; elbow flexion 0-150 (no
hyperextension); wrist flexion 0-80 / extension 0-70 / radial 0-20 / ulnar 0-30 deg.
How each maps onto the rig:

- **Elbow** is a true 1-DOF hinge, so it takes real anatomical limits: the forearm
  roll is aligned to the hinge axis, its IK twist and sideways bend are locked
  (`lock_ik_x/y`), and its flexion is clamped to fold no tighter than 150 deg
  (interior >= ~30 deg) and to stop just shy of straight (no hyperextension) via
  `ik_min/max_z` measured from the rest bend. This tightened an earlier 160 deg
  over-fold back into the human range.
- **Shoulder** (upper_arm) is positioned by the IK target + pole, so its orientation
  is a consequence of where the drum is, not a free DOF. Rather than tight symmetric
  caps — which would forbid the large but genuinely human cross-body reach to the
  hi-hat — it is caged per side (`use_ik_limit_x/y/z` + min/max) in a human-plausible
  envelope that contains the natural seated + cross-arm motion with margin, blocking
  impossible IK branches (like the old frame-1 over-rotation) without binding play.
- **Wrist** (hand bone) is aimed at the stick tip by a Damped Track; a `LIMIT_ROTATION`
  in the bone's local space then bounds it to a human wrist envelope. **Both hands use
  the same envelope**, made meaningful across sides by first rolling both hand bones to
  a common local-Z-up reference; it is sized to the full play range so it catches only
  real extremes and never clamps the (vertical) strokes (see the stroke section).

All limits were verified non-destructive: stick-tip landing accuracy is unchanged
(median ~1 mm, mean ~4 mm across the strikes) with the limits enabled.

**Feet, legs and pedal technique.** The legs hang off a *static pelvis* bone, not
the twisting spine, so the torso can turn toward the snare without dragging the feet
off the pedals (the feet were sliding around before). Each leg is a two-bone IK
(thigh + shin) reaching an **Ankle target**, with a per-leg knee pole, so the knee
can move while the foot stays planted. The footboards rest **sloped up toward the
toe** (`PEDAL_SLOPE`), and the feet are built with the toe higher than the ankle to
sit flush on them. Pedal strokes follow real technique, scaled by velocity: a loud
or fast note uses **heel-up** -- the Ankle target bobs up to cock (heel lifts, toe
stays down) then drives down into the strike, so the *knee moves* and the leg pushes
the pedal (~7 cm knee travel); a soft note barely moves the leg and is mostly
**heel-down** ankle rotation. The kick beater and footboard swing in step.

For the **hi-hat foot** the two techniques split by musical role (GM notes: 42 closed,
44 pedal/foot-chick, 46 open): opening and closing the hats (46 vs 42 -> pedal up vs
down) is **heel-down** -- the knee stays put and only the ankle rocks the pedal, so
the top cymbal lifts to open and drops to close; a foot **chick** (44) is **heel-up**
-- the knee lifts then drives down to stomp the hats shut. The pedal is yawed about
its **toe** (`build_pedal`): the toe end stays at the stand centre (the pull-rod
point) while the footboard angles **back toward the drummer** (heel toward the body)
-- the ergonomic setup -- and the foot follows (the ankle rotates about the toe). The
tripod is rotated (`leg_yaw`) so a leg *gap* faces that angled pedal, so the pedal
runs between two legs instead of into one. The demo MIDI shows the foot work -- an
open-hat groove (bar 2) rocking the pedal open/closed, and foot chicks on 2 & 4 in
the ride bar.

This makes the animation **body-agnostic**: it is authored purely as *target
areas to hit at target velocities* — the planner's strike points and MIDI
velocities. Swap `ANTHRO` (or drop in a real character mesh weighted to the
same skeleton) and the identical targets re-solve against the new limb lengths.
A longer arm covers the same reach with a smaller sweep; a shorter arm sweeps
more or stretches toward the target — the "required speed to hit the area in
the available time" is what changes per body, which is exactly the tailoring
knob a future model swap needs. The velocity model (§4) sets the apex height
and the apex→contact time; the arm's angular speed to satisfy it falls out of
the geometry the IK solves.

## 6. What this kit does / doesn't cover

- One seated humanoid: torso, faceless head, two IK arms with sticks, two legs
  with shod feet on the pedals; four independent striking limbs, cross-arm
  hi-hat by default.
- The rest pose is **upright** (the upper spine, neck and head sit essentially
  over the hips). The spine **twists to FACE the centre of activity**: it aims the
  chest a fraction (`TWIST_FOLLOW`) of the way to the *bearing* of the active
  drum(s), measured off the seated baseline facing (`SEAT_YAW`, ~18° toward the
  hi-hat/snare on +X) — so turning toward the RIGHT-side kit (floor tom, ride)
  must cross that whole left bias, which is why the drummer used to stay facing
  away from the floor tom when both sticks struck it. The activity centre weights
  the **two hands equally** (not each note), so a split ride-right + snare-left
  groove faces near centre while a same-side tom fill turns the torso fully toward
  it. A per-frame **reach cap** (`REACH_MARGIN`) then keeps the twist from turning
  so far right that the left hand's live target (or its idle snare home) falls out
  of reach — which both flipped the snare stick and drove the left elbow into the
  chest; the cap binds only when the left hand is on the far (left/centre) side, so
  the floor-tom fill is unaffected. It also **leans forward only in proportion to
  how far forward the hands reach** — ~0 for the close-in snare / hi-hat groove,
  ramping to a modest maximum (`LEAN_REACH`) out at the toms and crashes. So the
  drummer sits tall and dips in only when a reach demands it. The lean stays gentle
  and the chest is slim because the cross-arm reach to the hi-hat passes just in
  front of the torso — a deep chest or big lean would tilt the body into that path.
- Simplifications in the stand-in: the legs are static (only the arms, wrists,
  ankles and spine animate) and there is no finger articulation — deliberate,
  since this is a placeholder for a real character model. The wrist empty and
  tip empty interpolate independently, so a handful of the fastest hi-hat hits
  land up to ~2 cm off at the exact strike frame (median error is 0); the vast
  majority are exact.
- Not modelled: buzz/press rolls, flams as grace-note offsets (a >2-hit cluster
  is stacked on alternating hands and warned about), and remote/auxiliary
  hi-hats.

## Sources

- DRUMS: Drummer Reconstruction Using MIDI Sequences — ACM SIGGRAPH MIG 2025.
  https://dl.acm.org/doi/10.1145/3769047.3769066
- Sporka et al., *The Drum Set Tutorial System by Means of Inverse Kinematics*
  (CEMVRC 2005); *An Animated Virtual Drummer*.
  https://www.researchgate.net/publication/228976881_An_animated_virtual_drummer
- General MIDI percussion key map (notes 35-81).
  https://soundprogramming.net/file-formats/general-midi-drum-note-numbers/
- The Moeller method (whip motion; Full/Down/Tap/Up strokes; doubles & triples).
  Sanford A. Moeller, *The Art of Snare Drumming*; Jim Chapin, *Speed, Power,
  Control, Endurance*.
  https://en.wikipedia.org/wiki/Moeller_method
  https://melodics.com/blog/the-moeller-method
  https://drumhelper.com/learning-drums/moeller-technique-for-drumming/
  https://www.masterclass.com/articles/the-moeller-method-for-playing-the-drums
- Cross-arm vs open-handed drumming / hi-hat ergonomics — Modern Drummer,
  Drummer Cafe, Wikipedia "Open-handed drumming".
- Seated drumming posture (elbows low/tucked, forearms reach, shoulders relaxed):
  Drummerworld "Drumming Posture"; Melodics "How to have good drumming posture";
  Drum Helper "Correct Drumming Posture".
  https://www.drummerworld.com/articles/news/drumming-posture-tips/
  https://melodics.com/blog/how-to-have-good-drumming-posture
  https://drumhelper.com/learning-drums/correct-drumming-posture/
- Normal joint range-of-motion values (AAOS norms): shoulder / elbow / wrist.
  https://goniometer.io/range-of-motion
  https://www.healthline.com/health/shoulder-range-of-motion
- Body segment lengths as a fraction of stature (Drillis & Contini / Winter).
  https://www1.udel.edu/biology/rosewc/kaap686/notes/anthropometry.html
  https://en.wikipedia.org/wiki/Anthropometry_of_the_upper_arm
