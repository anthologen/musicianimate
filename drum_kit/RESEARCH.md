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

## 4. Motion model — wind-up proportional to volume

A real stroke is *anticipation -> contact -> rebound*: the stick lifts to a
backswing apex, drops to the head at the note onset, and bounces back. Louder
notes are struck from a **taller backswing dropped faster**. We reuse the
guitar pick-hand's velocity model:

    v = velocity / 127
    lift   = LIFT_MIN + v·(LIFT_MAX − LIFT_MIN)     # apex height above the head
    strike = STRIKE_SLOW − v·(STRIKE_SLOW − STRIKE_FAST)  # apex→contact seconds

so a hit at velocity 127 winds up ~14 cm and drops in ~45 ms, while a soft hit
winds up ~3 cm and drifts down over ~120 ms. Between strikes the hand glides
from one target toward the next, which reads as natural arm travel and honours
the "minimise movement" heuristic visually.

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
toward the snare / hi-hat rather than square to the kick, so the **torso** yaws
~18 deg toward +X (`SEAT_YAW` about the pelvis). Only the upper body twists: the
hips, legs and throne stay **square to the pedals** (`_seat_sq`, no yaw), and each
knee is solved into its **pedal's vertical plane** (`_solve_knee`) so the shin runs
straight down the pedal and the thigh splays from the hip out to it -- the legs read
square-on rather than angled off to the side, like a player who turns their upper
body but keeps their feet planted.
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
working length swinging at the drum. The wrist is placed by `wrist_target()` as a
**blend** (`WRIST_BACK`) of two directions from the tip: a shallow **back-and-up**
component (so the shaft lies fairly flat across the head, a glancing stroke rather
than a straight-down stab) and an **in-line "toward the shoulder"** component. A
pure back-and-up grip sat so far behind the tip that on a close drum the forearm
doubled back and the **wrist hyper-bent (~140°)** — the "strange" wrist seen in
reference-matching; mixing in the shoulder direction keeps the grip roughly in line
with the forearm, so the wrist rests at a **natural cock (~25–45°)** like a real
player's while the stick stays reasonably shallow. For a far cross-body target (the
hi-hat) even that is out of reach, so the wrist is pulled fully **in along the
tip→shoulder line** — the closest the grip can get to the shoulder — keeping the
elbow bent rather than locked straight. `|wrist − tip|` stays STICK_LEN either way,
so the tip still lands exactly. A pole target keeps each elbow low and tucked.

**Elbows hang and tuck, the wrist does the stroke.** Studying seated-drummer
posture references (Drummerworld, Melodics, Drum Helper) the consistent advice is:
let the shoulders hang and the elbows sit *low and tucked near the sides*, forearms
doing the reaching — "you only really need the action of your forearms," and "at no
point should you have to fully extend your arm to reach any part of the kit." A
raised or winged-out elbow is both tiring and unnatural.

Two mechanisms give this. First, the **stroke comes from the wrist, not the arm.**
The animator keyframes the wrist empty (which the forearm IK reaches, placing the
elbow) *only at each hit's contact-height anchor* and lets it hold; the tip empty
alone winds up and drops. For repeated hits on one drum the anchor is identical
frame to frame, so the forearm and elbow are motionless and the hand bone pivots at
the wrist to flick the stick — the arm only travels when moving to a *different*
drum. (Before, the wrist empty bobbed up to the wind-up apex on every hit, so the
whole arm rose and fell ~7 cm on each hi-hat note; now the arm holds and the wrist
rotates, which reads as an economical groove and matches how a real player keeps
time.)

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
  in the bone's local space then clamps it to a human wrist envelope
  (flexion/extension/deviation). The natural stroke already stays within it (measured
  wrist bend peaked ~86 deg), so the cap only catches extremes.

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
  over the hips). The spine **twists** toward the centre of what the hands are
  playing (a smoothed activity centre, SINE-eased), which swings the shoulders so
  the IK arms adjust; and it **leans forward only in proportion to how far
  forward the hands reach** — ~0 for the close-in snare / hi-hat groove, ramping
  to a modest maximum (`LEAN_REACH`) out at the toms and crashes. So the drummer
  sits tall and dips in only when a reach demands it, rather than hunching over
  the kit the whole time. The lean stays gentle and the chest is slim because the
  cross-arm reach to the hi-hat passes just in front of the torso — a deep chest
  or a big lean would tilt the body into that path and the arm would clip.
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
