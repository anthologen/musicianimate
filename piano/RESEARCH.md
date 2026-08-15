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

Two joints are deliberately looser than their own norm, for the same reason as
on the guitar: the closed-form IK lumps the distal phalanx into the middle
link, so `f<n>_mid` carries the **combined** PIP+DIP (~190°) — or, on the
thumb, MCP+IP (~135°) — fold, while `f<n>_dist` holds a fixed natural flexion.
The constraints are pure guards: the shipped performance keyframes *inside*
the cage on every bone and every frame, with ~3° to spare at the tightest
(verified in-scene against the baked action).

### Reaching is the wrist's job

The animator caps the IK's knuckle yaw at the anatomical `FINGER_MCP_SPLAY`
(26°; the thumb's CMC gets `THUMB_CMC_SPLAY`, 45°). Uncapped it splayed
pressing fingers up to 48° on the demo — a finger swinging sideways under its
neighbours rather than a hand moving. So the *wrist* does the reaching, in
three places:

- **`_splay_clamp_x`** slides the wrist along the keyboard until every
  pressing finger's key is within its splay window (`tan(cap) ×` the
  knuckle-to-key distance). The Gaussian-smoothed glide — what produces the
  crossunder/crossover look — survives wherever the fingers can absorb it.
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
