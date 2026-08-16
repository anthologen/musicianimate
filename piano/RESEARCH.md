# Piano Fingering: Research Notes

How to turn a MIDI file into a realistic piano fingering — which hand plays
each note and which finger (1 = thumb .. 5 = pinky) presses each key — so
the result can drive an animated hand rig. This is the background for
`piano/fingering.py`.

## The problem

Unlike guitar, a piano pitch is unambiguous: one key per note. The
difficulty is elsewhere. First, a raw MIDI file usually doesn't say which
hand plays what, so notes must be split between two hands that share the
keyboard and sometimes cross. Second, each hand's finger assignment is
combinatorial: a chord maps several simultaneous keys onto five fingers,
and melodic lines chain those choices over time — a fingering that is
comfortable now may force an impossible stretch three notes later, which is
why the choice must be optimized globally, not greedily. There is no single
"correct" answer: professional pianists agree with each other on only about
70% of notes, so ~60–65% agreement with any one human annotation is
effectively the ceiling for an algorithm — and for animation, *plausible
and physically consistent* matters more than matching one pianist's habits.

## The consensus method: ergonomic cost + dynamic programming

The literature converges on a minimum-cost path search over per-event
finger assignments:

- **[Parncutt, Sloboda, Clarke, Raekallio & Desain 1997, "An Ergonomic
  Model of Keyboard Fingering for Parsimonious Fingering of Melodic
  Fragments"](https://www.semanticscholar.org/paper/An-Ergonomic-Model-of-Keyboard-Fingering-for-Parncutt-Sloboda/1df73cbfa091a54f5760d4dab9bf508b03f1b98d)
  — the canonical model. Defines per-finger-pair **span tables** (minimum /
  comfortable / relaxed / practical interval each finger pair can cover, in
  semitones, with negative minima for crossings) plus ~12 weighted
  difficulty rules: stretch beyond comfort, weak fingers 4/5, thumb on a
  black key, position changes, thumb passing, same-finger repetition.
  Minimum-total-difficulty fingerings are found by dynamic programming.
  Later refined by Jacobs (1) toward modern pedagogical practice.
- **[Al Kasimi, Nichols & Raphael 2007, "A Simple Algorithm for Automatic
  Generation of Polyphonic Piano
  Fingerings"](https://www.semanticscholar.org/paper/A-Simple-Algorithm-for-Automatic-Generation-of-Kasimi-Nichols/2d1e519c087fc3ec9d4104427140697720052b0c)
  — extends the DP to chords by making each DP state a *complete
  finger-to-key assignment of the sounding chord*, with a vertical
  (within-chord) cost plus a horizontal (transition) cost between
  consecutive states. This is the shape our engine takes.
- **Balliauw, Herremans, Palhazi Cuervo & Sörensen** — variable
  neighborhood search over the same kind of ergonomic cost, including both
  hands; confirms the cost-model design but a Viterbi/beam DP is simpler
  and adequate at this scale.
- **[Zhao et al. 2022, pitch-difference fingering match model
  (EURASIP)](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-022-00237-8)**
  — emphasizes *playability* constraints (what a hand physically can do)
  over stylistic preference, a useful framing for animation work.

## Statistical and neural alternatives (not used)

- **[Nakamura, Saito & Yoshii 2020, "Statistical Learning and Estimation of
  Piano Fingering"](https://arxiv.org/abs/1904.10237)** with the **[PIG
  dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/)**
  (150 classical pieces, fingerings annotated by pianists; Bach/Mozart/
  Chopin subsets have ≥4 annotators per piece). Second/third-order HMMs
  decode fingering at ~65% match to human annotations — the published
  state of the art, beating the DNN/LSTM baselines they compare against.
- **[Ramoneda et al. 2022, "Automatic Piano Fingering from Partially
  Annotated Scores using Autoregressive Neural
  Networks"](https://dl.acm.org/doi/10.1145/3503161.3548372)** — modern
  neural take; can complete partial human annotations.

Rejected here for the same reasons as on the guitar side: the PIG data is
licensed for academic research, learned models need training corpora and ML
dependencies in a deliberately stdlib-only repo, and the accuracy gap over
a tuned ergonomic DP (~60% vs ~65%, against a ~70% human ceiling) buys
little for animation, where a hand-tuned interpretable cost model is easier
to steer ("less thumb on black keys" is one weight).

## Hand separation

Most fingering literature assumes the score is already split into hands;
arbitrary MIDI isn't. Reference methods:

- **[Nakamura, Ono & Sagayama 2014, "Merged-Output HMM for Piano Fingering
  of Both Hands"
  (ISMIR)](https://eita-nakamura.github.io/articles/Nakamura_etal_MergedOutputHMMForPianoFingering_ISMIR2014.pdf)**
  — models the note stream as the merged output of two per-hand Markov
  processes and decodes hand attribution and fingering jointly; error rates
  1.4–18.7% depending on the piece.
- **[Hadjakos, Waloschek & Leemhuis 2019, "Detecting Hands in Piano MIDI
  Data"](http://www.cemfi.de/wp-content/papercite-data/pdf/hadjakos-2019-detectinghands.pdf)**
  — surveys rule-based, HMM, and neural hand-separation approaches.

Our engine borrows the *idea* (two hand processes competing for notes,
decoded globally) in simplified DP form — see below.

## Existing libraries (not used)

- **[pianoplayer (marcomusy)](https://github.com/marcomusy/pianoplayer)** —
  MIT, music21-based; minimizes finger-travel "effort" over a 5–9 note
  lookahead with hand-size presets. Heavy dependencies, raw-MIDI input is
  its weakest path, and we'd still need an adapter from score annotations
  back to a timed animation timeline.
- **[piano_fingering (Kanma)](https://github.com/Kanma/piano_fingering)** —
  lighter, same trade-offs.

## What the animation literature says about output format

Piano hand-motion synthesis systems — from **Handrix** (ElKoura & Singh,
SCA 2003) through **[PianoMotion10M 2024](https://arxiv.org/html/2406.09326v1)**,
**[FürElise (SIGGRAPH Asia 2024)](https://arxiv.org/abs/2410.05791)** and
**[Tipiano](https://arxiv.org/html/2604.09692v1)** — all share the pipeline
*fingering / fingertip targets → wrist trajectory → hand pose (IK or
learned)*. So a fingering engine for animation must emit more than
note→finger labels: it needs fingertip key targets in instrument-local
coordinates and a smooth per-hand wrist curve. That drove the design of
`fingering.json` (per-note `x/y/is_black` from `key_layout.py` plus 30 Hz
`wrist_x` curves), which `animate_hands.py` consumes.

## What `piano/fingering.py` implements

A dependency-free three-stage pipeline (the guitar engine in
`guitar/fingering.py` later mirrored this structure):

1. **Onset clustering** — notes starting within 30 ms form one chord event.
2. **Hand assignment** — Viterbi over per-event split points (bottom *k*
   notes → left hand), carrying each hand's last centroid in the state.
   Costs: span beyond a hand's reach, centroid movement scaled by urgency
   (`_time_factor`), crossed hands, a weak pull of LH below / RH above
   middle C, and an **activation cost** for engaging an idle hand — the
   term that stops the DP from splitting one-hand material (an octave, a
   scale) across two conveniently-placed hands. Placing both hands before
   the first event is free. A `hands_from_tracks` mode trusts a two-track
   MIDI instead.
3. **Per-hand fingering** — Al Kasimi-style Viterbi where a state is a
   strictly-monotonic finger assignment of the sounding chord (left hand
   handled by mirroring pitches, `q = −midi`, so one orientation of the
   span tables serves both hands). Chord feasibility and stretch costs come
   from the Parncutt span tables scaled by a hand-size preset (XXS–XXL);
   within-chord costs add weak-finger and thumb/pinky-on-black penalties;
   transition costs cover hand repositioning (thumb-anchor position via
   `FINGER_OFFSET`), thumb passing (extra onto a black key), same-finger
   restrikes, and finger substitution on held notes (allowed, heavily
   penalized). Fingers holding sustained notes are pinned; impossible
   sustain pileups (>5 sounding notes) release the oldest holds. Exact
   Viterbi with a beam of 200 as a safety valve.

Weights live in `WEIGHTS` and were hand-tuned against golden selftest
cases (`python -m piano.fingering --selftest`): C-major scale must produce
1-2-3-1-2-3-4-5 in both hands, C-E-G → 1-3-5, octave → 1-5 in one hand,
two-octave runs must cross thumb-under, and a bass line + melody must split
cleanly. Debug visualization: `./inspect_midi.py <file> --fingering`.

## Hand-pose realism: dimensions and joint limits

The fingering solver decides *which* finger presses *which* key;
`build_hands.py` then has to be a hand shaped like a hand, and
`animate_hands.py` has to pose it without the joints doing things a hand
cannot do. Both come from the guitar/bass fretting hands (see
`guitar/RESEARCH.md`, `bass_guitar/RESEARCH.md`) so that all three players
read as the same size of human.

### Dimensions

Phalanx lengths, knuckle spread and per-phalanx cross-sections are adult-hand
anthropometry, shared verbatim with the fret hands (`FINGERS`,
`FINGER_CROSS`): proximal phalanges ~44–48 mm on the index/middle down to
~37 mm on the little finger, each phalanx roughly 0.6 of the one before it;
digit breadths ~18 mm at the index/middle proximal tapering to ~11–14 mm at
the fingertip, with the little finger slimmest and the thumb thickest. The
knuckle line keeps its natural arc and spans a realistic 74 mm index-to-little
(~25 mm pitch, up from the 19 mm the rig started with), on an 86 × 72 × 24 mm
palm.

Finger 1 is a driven **thumb column**, not a fifth finger: metacarpal
(~46 mm) + proximal (~31 mm) + distal (~22 mm) rooted at the CMC joint near
the wrist on the radial edge and a little below the palm plane. Extended, its
tip reaches about to the index finger's PIP joint, as a real thumb does.

### Joint range of motion

Every phalanx is caged with a local `LIMIT_ROTATION` constraint at build time
(`FINGER_ROT_LIMIT`, `THUMB_ROT_LIMIT`), the same way `build_guitarist` /
`build_bassist` cage the elbow and knee. Per-finger means for index → little
(Thieme 2024; AAOS goniometry):

| Joint | Flexion | Extension (backward) | Abduction/adduction |
|---|---|---|---|
| **MCP** (knuckle) | ~85–90° | ~25–30° (little finger the most) | **~±25°** |
| **PIP** | ~95–110° | ~0° | none |
| **DIP** | ~80–85° | ~0–10° | none |

and for the thumb: CMC ~15–20° flexion/extension but **45–60° of palmar/radial
abduction**, MCP ~55° flexion, IP ~80° flexion with 15–20° hyperextension.

The thumb's sideways bound is the one **lopsided** joint in the hand, and it is
the only cage that flips with the side of the body. All of that 45–60° is
*abduction*, away from the palm: zero in this rig is the thumb lying alongside
the index, which is already the end of its adduction, and a real thumb crosses
the palm by rotating **under** it (palmar flexion and opposition — carried here
by the metacarpal's pitch) rather than swinging further into the index it is
touching. Capped symmetrically at 45°, the solve took the second option, and on
the reach take the right thumb lay flat in the palm plane and swung 32° *past*
the index — threading between the index and middle fingers at 98% of its own
extension — to hold a key 42 mm treble-ward of a wrist the smoothing had left
behind. `THUMB_CMC_ADDUCT` (12°) is what it may do toward the palm; the wrist
takes back the rest of the reach.

Two joints are deliberately looser than their own norm, for the same reason as
on the guitar: the closed-form IK lumps the distal phalanx into the middle
link, so `f<n>_mid` carries the **combined** PIP+DIP (~190°) — or, on the
thumb, MCP+IP (~135°) — fold, while `f<n>_dist` holds a fixed natural flexion.
The constraints are guards, and nothing is allowed to lean on them: a keyframe
past a `LIMIT_ROTATION` is not a pose that plays, it is a pose Blender rewrites
at render time, so the hand that shows up is one nothing solved, checked for
clearance or measured for range of motion. The clearance search already refuses
out-of-cage candidates; `_cage_pose` catches the digit whose *whole* search the
cage rejects and pulls its wish inside before it is keyed, so the fingertip
sits off its key by exactly what the joint could not do — a hand short of a
key, not a broken one. The reach take used to key the right thumb 48° back at
the CMC and show it at the cage's 30°; both takes now key *inside* the cage on
every bone and every frame (verified in-scene against the baked actions).

### Reaching is the wrist's job

The animator caps the IK's knuckle yaw at the anatomical `FINGER_MCP_SPLAY`
(26°; the thumb's CMC gets the lopsided `THUMB_CMC_ABDUCT` / `THUMB_CMC_ADDUCT`
pair, 45° and 12°). Uncapped it splayed pressing fingers up to 48° on the demo —
a finger swinging sideways under its neighbours rather than a hand moving. So
the *wrist* does the reaching, in three places:

- **`_splay_clamp`** slides the wrist along the keyboard until every pressing
  finger's key is within its splay window (`tan(bound) ×` the knuckle-to-key
  distance, measured from the knuckle in the same signed bounds the IK clamps
  its yaw to, so the thumb's window is as lopsided as its joint). The
  Gaussian-smoothed glide — what produces the crossunder/crossover look —
  survives wherever the fingers can absorb it.
- **`BLACK_KEY_LIFT`** rides the hand ~18 mm higher over any event touching a
  black key. Black keys sit 12 mm up and 55 mm further in, leaving so little
  drop from knuckle to fingertip that the fingers folded *backward* (~39° of
  MCP hyperextension on the demo's black-key scale). Pianists meet the black
  keys with a higher, flatter hand.
- **`_wrist_fit`** searches a grid around the smoothed target — ±16 mm in and
  out from the keys, and down from the hover height to a 45 mm floor — for the
  placement its fingers can actually hold, scoring the two failure modes
  against each other: fingertip off its key (too far away, or pinned at the
  splay cap) versus knuckle bowed backward past `FINGER_MCP_HYPEREXT` (too
  close). A regularizer keeps the hand on its glide, so this only bites on
  stretched voicings: on the demo it moves 10 of 40 events, and all but the two
  stretched chords by a few millimetres.

What is left over is honest: at full stretch (a 1-2-3-5 octave chord, 168 mm
between the outer keys) the hand runs out of span and the residual is shared
between the thumb and the outer fingers, ~4.5 mm of sideways error on a 23 mm
key rather than a dislocated knuckle. Every other press in the demo lands on
its key exactly.

Sources: [Normal Active ROM of the Index–Little Fingers (Thieme, 2024)](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/s-0044-1788593.pdf);
[AAOS normal-ROM goniometry chart](https://goniometer.io/range-of-motion);
[Physiopedia: MCP joint abduction goniometry](https://www.physio-pedia.com/Goniometry:_Finger_Metacarpophalangeal_Joint_Abduction);
[StatPearls: Metacarpophalangeal joints](https://www.ncbi.nlm.nih.gov/books/NBK538428/);
Garrett (1971) hand anthropometry / ANSUR digit breadths.

## The seated player: where the bench puts the arms

`build_pianist.py` sits the same blocky stand-in the other four players use
(Drillis & Contini / Winter segment fractions, H = 1.75 m) on a bench at the
piano, and `animate_pianist.py` points its arm IK at the two hand rigs. Almost
everything about that figure is decided by one number.

**The bench height sets the wrist.** The piano hand rigs are *axis-locked*:
`animate_hands` keys only their location, so the fingers always run along +y and
the palm always faces down. A hand that cannot rotate cannot meet the forearm
halfway — so the wrist angle is entirely the arm's problem, and the only way to
keep it straight is to bring the forearm in level with the keys. Standard
ergonomics get this for free: a 0.50 m bench under a 0.71 m key surface puts the
seated shoulders 0.59 m above the seat, from which the upper arm hangs nearly
vertically and the forearm runs out almost horizontally. Measured on the demo,
the wrists sit 5° (L) and 27° (R, at full cross-body reach) off the forearm.
Raise or lower the bench and the forearm starts diving onto the keys instead;
`animate_pianist._check_wrist_pose` fails the build past 60°, because no IK limit
can catch it (the hand is not a joint the rig solves).

**The elbow is a trade, not a free parameter.** `ELBOW_BEND` pulls the elbow off
the shoulder→wrist line down and back, which is what hangs the upper arm. Its
small *outboard* term is contested: this demo's right hand plays as far in as the
player's own centre line, so that arm reaches across the body, and too little
outboard push cuts the forearm ~12 mm into the torso while too much opens the
wrist and flares the elbow 0.34 m off centre. The settled value has the forearm
resting *against* the ribs (~5 mm) at the worst frame — which is what it does in
life.

**A torso is not one block.** No elbow-pole angle fixes a forearm that crosses in
front of the abdomen: the elbow is not what clips. The fix is anatomical — a
ribcage above the waist and a narrower, set-back abdomen below it, which is
exactly the clearance a cross-body reach needs.

**Solve the IK poles, don't tune them.** Blender measures an IK pole angle from a
reference frame that depends on the chain root's roll, so `_solve_pole_angle`
sweeps the angle and keeps whatever reproduces the built rest pose (to ~1 mm),
rather than shipping four hand-tuned constants that would silently twist the
limbs if the proportions or the seat ever changed. Sweep **−180…180**: Blender
hard-clamps `pole_angle` to ±π, so a 0…360 search quietly pins half the circle to
the same pose and finds a false minimum.

**The ends of the board are out of reach sitting up — so lean.** The keyboard is
1.25 m wide and the arm is 0.58 m from shoulder to wrist, so an upright spine
cannot get to either end of it: on `make_reach_piano_midi.py`'s stress piece
(A0 and C8, alternated and together) the right arm came up **40 mm short** for
63% of the take and the hand rig simply floated off the end of it — a failure no
joint limit catches, because a stretched IK chain is not an illegal pose, it is
an unreachable target. Pianists answer this by leaning in from the hips, and so
does `_reach_lean`: per body keyframe it steps the forward lean up until *both*
wrists are inside `REACH_MARGIN` of the arm's length (leaning by θ swings the
shoulder forward `lever·sinθ` and drops it `lever·(1−cosθ)`; the drop matters,
because the keys are already well below the shoulder, so it is stepped rather
than solved in closed form). The reach take needs 0…17°, the demo needs 0° and
is unchanged.

**What a lean cannot fix is the wrist — the HAND has to turn.** Leaning got the
arm onto the key but left 68° (R) / 48° (L) of wrist, over the 60° cap, and no
body pose takes that out: at full stretch the forearm *is* the shoulder→hand
line, so it arrives diagonally while a square hand insists on pointing +y. A
pianist turns the hand out to follow the arm, and `animate_hands` now does the
same. The wrist target carries a **yaw** as well as a position (`_event_yaw`):
`HAND_YAW_FRAC` of the angle the arm comes in on — `atan2` of the hand's offset
from *its own shoulder* over `ARM_DEPTH`, the shoulder's setback behind the keys
— capped at `HAND_YAW_MAX` = 25°. The wrist supplies the remainder, which is the
real division of labour: a hand that turned the whole way would be pointing its
fingers along the keys instead of across them.

Everything below the wrist is then solved in **the hand's own frame**
(`_hand_xy`): the wrist at the origin with the fingers along +y, which is the
frame `build_hands.FINGERS` and the bone poses were always written in. Only the
keys have to be rotated into it. Heights are untouched (the yaw is about Z), so
the reach, hover and press geometry are unchanged — and at yaw 0 every formula
reduces to the square-handed one it replaced, which is worth keeping true as a
check. Two places needed real generalizing: the splay clamp slides the wrist
along the hand's own x (sliding across the *keyboard* would change how far the
fingers reach as well as how far they splay), and `_event_root_target` sets the
wrist a *rotated* knuckle-plus-reach offset back from each key.

The one thing a yaw costs is **depth**: meeting a wide grip at an angle racks
its outer fingers to different depths along the keys, sin(yaw) times the width
of the grip, which they must find out of their own length on top of whatever the
voicing already asks. So each event's yaw is capped at what its own width can
afford (`YAW_SPREAD` = 8 mm) — a pianist squares up for a stretch and angles the
hand for single notes, which is exactly what the ends of the keyboard are played
with. Measured: wrists 68°/48° → **48°/27°**, inside the cap, so the reach take
now passes `strict=True`; the demo's cross-body right hand improved 28° → 18°;
contact at A0/C8 is 0.01 mm mean, and chord contact is within half a millimetre
of the square-handed solve (demo 0.36 → 0.54 mm mean). `animate_pianist`'s
`strict=False` remains as an escape hatch for pieces that still ask too much.

Sources: Drillis & Contini (1966) body-segment parameters; Winter,
*Biomechanics and Motor Control of Human Movement*, anthropometric tables
(seated acromial height ≈ 0.59 m and seated stature ≈ 0.92 m above the seat, at
H = 1.75 m). The 0.71 m key height is `build_piano`'s own geometry; the 0.50 m
bench is the standard piano-bench height that pairs with it.

## Getting from one position to the next: the hand has to leave the keys

Everything above is about where the hand *plays*. What it did in between was
slide: the wrist glided flat, at the height it plays at, straight across the
keyboard. Three things were wrong with that, and on the A0↔C8 reach take
(`animated_piano_reach.blend`) all three are plainly visible — the worst
fingertip during a move sat **28 mm below the white key tops**, i.e. inside the
instrument.

- **A hovering fingertip is not clear of the keys.** It hovers 7.5 mm over a
  white key, but a black key stands 12 mm higher — so the same fingertip is
  4.5 mm *under* the top of every black key it passes.
- **A finger still holding its key was dragged.** The wrist departs as late as
  the hold allows but never later than `MIN_TRAVEL` before the next event needs
  it, so in a legato piece with big leaps it is already moving while the key is
  down. Pinned to that key, the finger flattens out, the IK runs out of length
  and the fingertip is towed sideways *at press depth* — through everything
  between the two positions. The reach take is nothing but this: 0.45 s notes
  0.5 s apart, leaping up to half a metre.
- **The clearance search could not see the keyboard.** `_solve_clear` knows only
  about the other four digits, and "give way" includes giving way *downward*.
  A digit with a neighbour in its way therefore took the full 36 mm retreat and
  ended up buried in the keybed — the thumb, whose idle pose sits lowest, did
  this for most of the take. Worse, the search walked its grid in index order
  and returned the first pose the joint cage allowed, which is that same
  downward corner, so a digit with *nothing* in its way did it too.

The fix is what a pianist does: lift off, travel above the keys, come down onto
the next position. Each gap between events gets a **`_travel_lift`** — a smooth
hump added to the wrist height and to every fingertip target that is not holding
a key down, zero at both ends so the pose at either event is exactly the one
`_wrist_fit` chose, peaking halfway across. Its height scales with the distance
travelled (12%, capped at 32 mm, ignored under 45 mm of travel), because a shift
to the neighbouring chord should barely lift and a leap across the board should
clear the black keys with room. Because the hand translates **rigidly** — wrist
and fingertips by the same amount — the pose at the top of the arc is one
already checked against the joint cage and the finger-to-finger clearances;
only its height is new. A key the hand walks out on is **released** when the
wrist goes rather than dragged (the key itself stays down: it is the MIDI, and
a leap like that is pedalled anyway). And the clearance search now charges for
sinking a digit into the keybed (`KEYBED_COST`) and walks its grid **cheapest
first**, so the digit's own wish is always tried before any nudge and the
nearest legal pose wins when the cage refuses it. The halfway point of every
travel is posed as well: the fingers are keyed at the wrist's own key times, so
they agree with it exactly there and drift in between, and a digit still holding
a key while the wrist walks off is precisely where that drift shows.

One more thing had to change for the arc to read: the wrist's f-curves are now
**BEZIER/`AUTO_CLAMPED`**, not SINE. SINE is one-sided (ease-in): the hand left
a position at zero speed, was still accelerating when it reached the next one,
and stopped dead — measured on the reach take's 530 mm leap, 91 mm/frame into
the arrival. That is fine for a *press*, which should land at speed and stays on
SINE (and in step with the key dips `piano_midi_animator` bakes), and wrong for
travel. It is the same fix the bass fret hand needed. `_ease_wrist` models the
new curve as smoothstep, which is exactly what auto-clamped handles give between
two keys of equal value — every dwell, and the top of every arc.

Measured over the whole reach take, per hand per frame, how far the deepest
fingertip is below the key surface under it: worst **−31.4 mm → −7.5 mm** (which
is press depth: nothing is below the keys any more but the fingers pressing
them), and summed over every frame **12.3 m → 4.4 m** (−64%; the demo, which
barely travels, still improves −62%). Mid-leap every fingertip now sits 25–40 mm
clear of the keys. Onset contact is unchanged to the hundredth of a millimetre
(1.55 mm mean on the reach take, 1.87 on the demo — the same wide-chord
compromises as before), the strict wrist-ROM check still passes (26.6° / 47.9°
of 60), and total finger-to-finger overlap is a third better on the reach take
and unchanged on the demo. The cost is deliberate: the reach take's fingers hold
their keys 86% of the notated length rather than 91%, which is what releasing a
leap early means.

What is left is not travel at all: an idle fingertip hovers at a fixed height
above the *white* keys, so wherever it happens to sit over a black one it is
4.5 mm inside it — which is most of the 4.4 m that remains, and a static
property of the hover pose rather than anything the hand does on its way.

### How fast the hand may get there

Clearing the keys is half of a leap; the other half is *how* the hand covers
the distance, and a smooth-looking curve is not the same as a humanly
accelerated one. Two problems remained.

**The profile was a cubic, so every move began and ended with an acceleration
step.** A Bezier between two keys is smoothstep: velocity eases from zero at
both ends (good), but acceleration jumps from nothing to its peak at the instant
of departure and falls off a cliff at the arrival — infinite jerk twice per
move. Human point-to-point reaches instead follow the **minimum-jerk profile**
([Flash & Hogan 1985](https://www.jneurosci.org/content/5/7/1688)), a symmetric
bell of speed with zero velocity *and* zero acceleration at both ends. Each
travel is now sampled along that profile (`_min_jerk`), and the arc rides on the
same samples with a shape (`_arc_shape`, sin³) that is likewise flat in
acceleration where it meets the keys. The samples are placed **on the rendered
frames**, because what anyone sees of a move is the sequence of whole-frame
positions; a travel under 8 frames — few frames to be described by, and the
steepest accelerations — is sampled at half frames as well, so the Bezier drawn
through them cannot bulge far past the profile.

**A leap given too little time is not a smooth leap, it is a fast one.** For a
move of D metres in T seconds the minimum-jerk peaks are `v = 1.875 D/T` and
`a = 5.7735 D/T²`, so ceilings on speed and acceleration are really a floor on
**time**. The reach take's 160–180 mm leaps were being crammed into `MIN_TRAVEL`
(0.15 s), which asks for 37–43 m/s² — around 4 g at the wrist. Where the window
is too short, the hand now takes the time off the note it is leaving
(`_travel_time` → an earlier departure, floored by `LEAP_MIN_HOLD`), which is
exactly what a pianist does: a leap is played short. `LEAP_SPEED_MAX` = 2.4 m/s
and `LEAP_ACCEL_MAX` = 24 m/s² are the fast end of a real reach rather than a
comfortable one — they put a half-metre leap at 0.39 s, about what an A0→C8 jump
takes a player who can make it at all — and `LEAP_TIME_MARGIN` keeps them true
of the *baked* curve rather than the ideal one, since the interpolation between
samples and the vertical share of the arc both add a little.

Measured on the reach take's wrist curves, peak over the whole take, sampled
both at whole frames (what is seen) and at quarter frames (the underlying
curve):

| | before | after |
|---|---|---|
| peak speed | 2.11 m/s | 2.18 m/s |
| peak acceleration (whole frames) | 22.0 m/s² | 18.5 m/s² |
| peak acceleration (quarter frames) | 37.3 m/s² | 21.2 m/s² |
| peak jerk (quarter frames) | 2972 m/s³ | 1474 m/s³ |
| quarter-frame samples over the ceilings | 34 | **0** |

The 530 mm leap's frame-to-frame speed now reads 2, 16, 39, 63, 81, 90, 87, 74,
53, 29, 8, 0 mm/frame — a symmetric bell, against the old 6, 21, 36, 50, 62, 73,
81, 87, 91, 37, 0, which was still accelerating when it arrived. Fingertip
clearance improves with it (the worst is now press depth), and the extra time
costs one more point of note length: 86% held rather than 87%.

## Fingers that cannot pass through one another

Everything above places a digit against the *keyboard*. What it does not settle
is what the five do to **each other**: each finger's IK aims at its own key and
knows nothing about where its neighbours are, and `_solve_clear` — which does —
could only nudge a digit a few millimetres around the pose it already wanted.
Measured on the baked curves of the reach take, phalanx capsule against phalanx
capsule, that left 24 rendered frames with two digits inside each other, the
worst by 15 mm; the demo take, 85 frames and 13 mm. Three distinct causes:

- **A finger reaching for its key through the finger still holding one.** The
  left middle finger's approach to B0 began four frames before the note, while
  the ring finger still held the A♯ behind it — and the *approach* pose is on
  the far side of that finger, so no nudge around it helps. What a hand does
  there is wait: the finger stays back over its own knuckle and goes to the key
  once the neighbour lifts. A digit whose grid cannot clear its neighbours now
  **withdraws** toward its idle pose in steps (`WITHDRAW_STEPS`) and takes the
  first one that gets it out. It costs nothing to look at, because the finger
  has no note yet; it simply arrives from further out (0.65 → 0.83 m/s of tip
  speed on the drop, still inside a real strike).
- **A splayed finger pressing into a neighbour that is already home.** MCP
  abduction caps at ~25°, and a finger *at* that cap is 19 mm sideways of its
  own knuckle — but adjacent knuckles are only 26 mm apart and two proximal
  phalanges need ~17 mm between their axes, so 25° of adduction toward a
  neighbour is a pose no amount of joint range makes room for. The neighbour
  has to move, which is exactly what a real hand does (adduct a finger into the
  one beside it and it carries that finger along). A digit **being crossed
  into** is therefore released from its usual `HOVER_GIVE` ration and gets the
  whole nudge grid — a finger with another one inside it is more conspicuous
  than any nudge that gets it out.
- **The order the five are placed in.** Least-free-first is right while someone
  is holding a key down and wrong once nobody is: an index on its way to a key,
  splayed to its cap, was placed *before* the idle middle finger merely because
  it was the less free of the two, and the middle finger then had nowhere to go
  under a digit that was not even sounding a note. The digit left with no room
  now goes again, ahead of everything not holding a key down, and the retry is
  kept only if it actually helped.

The last of these only became visible once the search stopped **lying about the
clearance it had achieved**. Where the joint cage refused every candidate in the
grid, `_solve_clear` fell back on the wish — and reported the clearance as
infinite, because it had scored nothing. That is the one pose most likely to be
in trouble, and it is keyed like any other; it hid a thumb 3 mm inside the index
finger. The fallback is now measured, caged, like everything else.

Finally, clearance at the moments the fingers are keyed at is not clearance at
the frames that get **rendered**: two digits apart at both ends of a span can
still brush in the middle, because the ease between them is not a straight line.
Every rendered frame is now measured on the baked f-curves (`_baked_chains`
reads the pose straight back off them), and any frame where two digits have got
into each other is solved outright and becomes a keyed moment of its own, which
leaves the ease only the shorter spans either side. Two or three passes settle
it. Both takes now run with **no interpenetrating frame at all**, the tightest
moment of each holding 0.03–0.6 mm of surface clearance, and that measurement —
not the solver's own opinion of itself — is what `animate_hands` returns as
`finger_clear_mm`. A negative number there means two *pressing* fingers were
asked for keys the hand cannot hold at once, which is a fingering to look at
rather than something the animation can fix.

## The other half of it: not needing to dodge in the first place

Getting two fingers out of each other is a rescue, and a rescue is visible. The
clearance search sees one frame at a time, so when it does have to move a digit
it moves it *in that frame* — the reach take's left ring finger left 23 mm
sideways between two frames, which reads as a teleport even though nothing
intersects. Three things were behind it, and all three are upstream of the
search.

**The wrist was working to the wrong splay limit.** `_splay_clamp` slides the
hand until every pressing finger is inside its knuckle's range, and that range —
26° — is what a finger has when there is *nothing beside it*. Adducting the same
26° toward a neighbour is not a pose a hand has: knuckles are 24–26 mm apart and
two proximal phalanges need ~17 mm between their axes, so swept against a
neighbour held at its own idle pose, contact comes at 8° (index into middle),
11–12° (middle/ring) and 11–13° (ring/little). Clamping to the joint's figure,
the wrist barely moved through the take's bass run: four notes 48 mm apart played
by a hand that stayed put while each finger in turn splayed to its cap. The clamp
now takes its window twice — first as what the fingers have beside each other
(`SPLAY_BESIDE`), then, only if a chord is too wide for that, as the joint's own,
which is the case the fingers really are fanned apart in and nobody is adducting
into anybody. The wrist path grows ~3%, peak speed and acceleration are unchanged
(2.18 m/s, 21.4 m/s²), and the fingertip-to-key miss is unchanged or better (the
demo take's worst falls from 46 mm to 22 mm).

**The release tail outstayed its welcome.** A finger's tail runs 5 frames past
the note; adjacent fingers share the space between their knuckles; so a finger
whose note ends 1.8 frames before its neighbour lands on the key in front of it
is still lying there when the neighbour arrives, and the search has one frame to
get it out. It now has a deadline — it must be home before a finger beside it
lands on a nearby key (`BESIDE_CLEAR_LEAD`, gated on `BESIDE_NEAR` so a neighbour
playing half a keyboard away costs nothing) — and gives the room up along the
ease it was already leaving on.

**And the tail eased the wrong way.** Blender's SINE is ease-in, which is right
for a strike (land at speed, with the key) and backwards for a release: the
finger stayed on its key for most of the tail and then covered the whole way home
in the last frame or two. Spans in which a digit is giving a key back are now
keyed EASE_OUT, and sampled with the matching shape (`_ease_off`), so the curve
and the poses along it describe one motion: away at once, settling as it arrives.

The same reasoning caught one more, and the worst in either take: a finger whose
next note is soon in *time* keeps hovering over the key it just played, which is
only sensible if the hand is still there. The right thumb played B3 and hovered
over it while the hand leapt two octaves, ending up stretched to a point 111 mm
behind its own wrist, then snapping forward at 2.62 m/s. Hovering is now also
conditional on the hand not lifting off after that event.

Peak fingertip speed *relative to the hand*, over the whole reach take, falls
from 2.62 m/s to 1.40 (and that one is now slower than the wrist carrying it, so
it is the hand moving, not the finger whipping); on the demo take 1.12 → 0.92.
No frame of either take has two digits in each other.

## Out of scope (future work)

- **Learned cost weights** from the PIG data (would need its academic
  license and a fitting step; the DP structure would not change).
- **Interdependence of the hands** beyond splitting (voice sharing,
  hand-over-hand passages) and phrase-boundary awareness — the known gaps
  Nakamura et al. list for all current models.
- Deliberate **finger substitution on held notes** as a technique (organ
  style) — currently only an emergency with a heavy penalty.
- Pedaling (would relax sustain pinning), ornaments, glissandi, and
  fingered octaves/double notes idioms (1-4/1-5 alternation in fast
  octave passages).
- Style profiles (editions differ: Czerny vs Chopin fingering habits could
  be alternate `WEIGHTS` presets).
