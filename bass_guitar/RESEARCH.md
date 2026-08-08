# Bass Fingering: Research Notes

How to turn a MIDI file into a realistic **bass** tab — which string and
fret sounds each note, which left-hand finger frets it, and which
right-hand finger plucks it — so the result can drive an animated hand
rig. This is the background for `bass_guitar/fingering.py`. It builds on
`guitar/RESEARCH.md`; only the bass-specific differences are covered here.

## Same problem, smaller and lower

A bass pitch is ambiguous across (string, fret) positions exactly like a
guitar pitch, so the same cost-based **dynamic-programming / optimum-path**
method applies (Sayegh 1989; Radisavljevic & Driessen 2004; Hori 2021 —
surveyed in the guitar notes). `bass_guitar/fingering.py` reuses the
guitar engine's beam-searched Viterbi wholesale, keyed on
`(grip, hand_position)`, with three bass-specific changes.

### 1. Four low strings, and (almost) monophonic

Standard four-string bass is tuned **EADG = E1 A1 D2 G2 = MIDI
(28, 33, 38, 43)**, an octave below the guitar's low four strings.
`fret_layout.TUNING` is the single knob; range is MIDI 28–63 over 20 frets
on an 864 mm (34") scale.

Bass lines are **overwhelmingly monophonic** — chords and double-stops are
occasional, and full barre chords essentially never occur. So the guitar
engine's index-barre machinery is dropped entirely: `_event_states`
generates only the one-finger-per-note model, and the search collapses to
a near-linear per-note (string, fret, finger) choice dominated by
hand-position continuity. Dense clusters are still trimmed to the bass
note plus the top three (rare), and a fifth/octave double-stop is handled
naturally by the multi-note grip model.

### 2. Left hand — position-dependent finger model (Simandl low, one-finger-per-fret high)

On a 34" scale the low frets are physically enormous (frets 1→3 span
~91 mm, and 1→4 would be ~132 mm — an unplayable stretch). Bassists
therefore split the neck into two regimes:

- **Low (at/below ~5th fret): the Simandl 1-2-4 system.** One hand
  "position" spans a minor third (three consecutive frets P, P+1, P+2)
  covered by fingers **1, 2, 4**. The **3rd finger is not used alone** —
  it only *reinforces* the 4th. So fret offsets from the index are
  `{1:0, 2:1, 3:2, 4:2}` and a lone 3rd finger is penalized.
- **High (above ~5th fret): one-finger-per-fret.** As the frets shrink,
  the hand reverts to the guitar's model — offsets `{1:0, 2:1, 3:2, 4:3}`
  over four frets.

Both regimes fall out of a single `_finger_offset(finger, fret)` table,
switched at `SIMANDL_MAX_FRET`. Two cost terms encode the ergonomics:
`pos_spread` (all fretted fingers should agree on one implied hand
position `fret - offset(finger)`) and `weak_low_ring` (finger 3 alone in a
low position). The **metric** reach limit `REACH_MAX_M` (in metres, so it
tightens up the neck where frets shrink) is calibrated to permit a low
1-2-4 span and a high 1-2-3-4 span while forbidding a low four-fret
stretch, doing most of the physical pruning for free.

Sources: Franz Simandl, *New Method for the Double Bass* (the 1-2-4
system, adapted to electric bass); TalkBass and BassBros "Simandl
technique" discussions; "One Finger One Fret" (bassguitarblog) for the
high-position crossover.

### 3. Right hand — fingerstyle i-m alternation (with raking)

The characteristic bass right hand is **fingerstyle alternating
index/middle plucking** ("i-m-i-m"): melodic playing uses strict
alternation, with no finger played twice in a row on a repeated string.
`assign_plucking` writes `note["pluck_finger"] ∈ {"i","m"}` with three
rules:

- **Strict alternation** on a repeated string, and when ascending to a
  thinner (higher-index) string.
- **Raking** when a line *descends* across strings toward a thicker
  (lower-index) string: the finger that just plucked drags across and
  repeats, instead of alternating — the natural follow-through of a
  plucking finger coming to rest against the next string down. This is the
  standard economy-of-motion move for descending string crossings.
- **Re-anchoring** to the index finger after a rest (`≥ PLUCK_GAP_RESET`),
  matching how players reset. In a double-stop the index takes the lower
  string, the middle the next up.

A `--style pick` mode instead reuses the guitar's metric ("pendulum")
down/up pick-stroke model verbatim (down on the beat, up on the off-beat;
see the guitar notes), so the same fingering engine can drive either a
plucking-finger rig or a pick rig.

Sources: TalkBass right-hand-technique threads; BassBuzz "Plucking: Index
vs Middle"; classical-guitar strict i-m alternation (Wikipedia). Slap/pop
and thumb technique are out of scope.

## Hand-pose realism: joint limits and anti-crossing

The fingering solver decides *which* finger frets *where*; `animate_hands.py`
then has to pose a real-shaped hand into that grip without the fingers doing
things a hand cannot do. Two anatomical rules drive that posing.

### Finger joint range of motion

Every finger joint is caged to measured active-ROM norms (`build_hands.py`
`FINGER_ROT_LIMIT`, applied as Blender `LIMIT_ROTATION` constraints, mirroring
how `build_bassist` cages the elbow/knee). Per-finger means for index →
little (Thieme 2024 normative study; AAOS goniometry charts):

| Joint | Flexion | Extension (backward) | Abduction/adduction |
|---|---|---|---|
| **MCP** (knuckle) | ~85–90° | ~25–30° (little finger the most, ~26° mean, ~38° at +2 SD) | **~±25°** (only when extended; the collateral ligaments lock it out in flexion) |
| **PIP** | ~95–110° | ~0° | none |
| **DIP** | ~80–85° | ~0–10° | none |

The rig's closed-form IK lumps the distal phalanx into the middle link, so the
`mid` (PIP) bone actually carries the **combined PIP+DIP fold** (~190° total in
a real hand); that is why its cap is generous (~140°) while `dist` barely moves.
Everything else sits at the anatomical envelope, and the shipped performance is
verified to keyframe *inside* the cage on every frame.

### Anti-crossing = the MCP splay limit, not a knuckle-order rule

A finger "crossing under" a neighbour (the index sweeping under the middle/ring
to reach a bass string in the octave double-stop — `bass_demo.mid` frame 117) is
at root **one joint exceeding its range**: to reach that far sideways the MCP
would have to abduct far past its ~25° limit. So the anti-crossing rule is simply
that limit, enforced in two places: the fret-finger IK hard-caps knuckle yaw at
`FINGER_MCP_SPLAY` (~26°), and the per-event wrist grid search penalizes any pose
whose pressing fingers would *demand* more splay (or more MCP hyperextension) than
that. Reach therefore becomes the **wrist's** job — a small yaw/roll and slide of
the whole hand — exactly as a real player angles the hand for an octave grip
rather than swinging one finger under the others. Idle (non-pressing) fingers
likewise lift *within* the MCP's extension range instead of bowing the knuckle
backward to clear a presser.

An earlier attempt penalized a presser whose target lay across another finger's
*knuckle x*; because it ignored that the fingers are spread along the neck (in y)
and lifted (in z), it fired on the natural bunched-knuckle pose and drove the hand
into a ~57° over-splayed diagonal — the very artifact it was meant to prevent.
The joint-limit formulation replaces it.

Sources: [Normal Active ROM of the Index–Little Fingers (Thieme, 2024)](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/s-0044-1788593.pdf);
[AAOS normal-ROM goniometry chart](https://goniometer.io/range-of-motion);
[Physiopedia: MCP joint abduction goniometry](https://www.physio-pedia.com/Goniometry:_Finger_Metacarpophalangeal_Joint_Abduction);
[StatPearls: Metacarpophalangeal joints](https://www.ncbi.nlm.nih.gov/books/NBK538428/).

## Out of scope (future work)

- Slap/pop, ghost notes, and palm/thumb muting.
- Techniques: hammer-ons/pull-offs, slides, bends, harmonics, vibrato.
- 5-/6-string basses and alternate tunings (`fret_layout.TUNING` is the
  single knob).
- Learned cost weights — weights here are hand-tuned against the golden
  `--selftest` cases, like the piano and guitar engines.
- Ring-finger (a) use in fast three-string figures; only i-m is modelled.
