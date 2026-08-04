# bass_guitar

Procedurally builds a minimalist electric bass and a stand-in bassist in
Blender, then animates a fretting hand and a plucking (or picking) hand from any
MIDI bass line: MIDI → tab/fingering → rigged hands → optional full-body player.

Like a guitar pitch, a bass pitch is ambiguous — most notes are playable at
several `(string, fret)` positions — so the heart of this folder is an ergonomic
tab/fingering solver, not just a mesh builder. It reuses the guitar engine with
three bass-specific changes (four low strings and near-monophonic lines; a
position-dependent left hand — Simandl 1-2-4 low, one-finger-per-fret high; and
a fingerstyle i-m right hand). See `RESEARCH.md` for the survey behind it.

## Pipeline

```
your_song.mid
    │
    ▼
python -m bass_guitar.fingering song.mid -o fingering.json   # tab + fingering solver (outside Blender)
    │
    ▼
build_bass_guitar.build_bass_guitar()   # bass meshes                       ┐
build_hands.build_hands()               # FretHand + PluckHand/PickHand rigs (inside Blender)
animate_hands.animate_hands("fingering.json")  # keyframes                   ┘
    │
    ▼  (optional: put the bass in a player's hands)
build_bassist.build_bassist()           # standing humanoid rig
animate_bassist.animate_bassist("fingering.json")  # full-body playing shot
    │
    ▼
bpy.ops.wm.save_as_mainfile(filepath="animated_bassist.blend")
```

`fingering.py` and `fret_layout.py` run as plain, dependency-free Python
outside Blender. `build_bass_guitar.py`, `build_hands.py`, `build_bassist.py`,
`animate_hands.py` and `animate_bassist.py` need `bpy` and must run inside
Blender (Scripting tab, Text Editor "Run Script", `blender --background
--python …`, or the Blender MCP `execute_blender_code` tool).

## Files

| File | Role |
|---|---|
| `fret_layout.py` | bpy-free source of truth: neck geometry, tuning (EADG, an octave below the guitar's low four strings), fret spacing, and string/press target coordinates. Imported by both the builder (inside Blender) and the solver (outside), so computed fingering targets land exactly on the built meshes. The single knob for tuning/scale length. |
| `build_bass_guitar.py` | Builds a realistically sized 34" solid-body electric bass (soft-cornered slab body, tapered neck, 20 frets, four strings, two pickups, bridge, four inline tuners) into a `BassGuitar` collection. Lies face-up (+Z), neck pointing +Y. Self-contained (bmesh/mathutils only). |
| `build_hands.py` | Builds the `FretHand` (wrist + four three-bone finger chains wrapped around the neck with a static thumb behind it) and, by default, the fingerstyle `PluckHand` (a floating-thumb hand with two articulated plucking fingers, `pi`/`pm`, near the bridge pickup) into a `BassHands` collection. With `style="pick"` it builds the guitar's `PickHand` instead. Run after `build_bass_guitar.py`. |
| `fingering.py` | The tab/fingering solver — beam-searched Viterbi DP over feasible `(string, fret, finger)` grips, picking the cheapest ergonomic path and emitting `fingering.json`. Reuses the guitar engine; barres are dropped, the left hand is position-dependent (Simandl/one-finger-per-fret), and it also assigns the right-hand pluck finger (i/m alternation with raking) or, with `--style pick`, the guitar's down/up pick direction. |
| `animate_hands.py` | Keyframes the two hand rigs from `fingering.json`: the FretHand glides the neck and arches fingers over their frets (closed-form two-link IK); the PluckHand curls the assigned finger (pi/pm) onto each struck string and follows through, or the PickHand sweeps a pick across the strings with metric down/up strokes. |
| `build_bassist.py` | Builds a blocky, faceless, walk-capable humanoid `Bassist` armature (IK arms + legs, human ROM limits on every joint, a `hand.*` stub at each wrist as the attach point for the hand rigs). Same axis convention as the drummer/guitarist stand-ins. |
| `animate_bassist.py` | Stitches it all together: rebuilds bass + hands + bassist, runs `animate_hands`, then rigidly mounts the "bass + playing hands" assembly across the standing player's torso, straps it on, makes the arms' IK follow the hands, and adds a gentle body sway. The end-to-end shot. |
| `make_demo_bass_midi.py` | Generates `bass_demo.mid` — a left-hand test bench: an ascending walking line low on the neck (Simandl 1-2-4 + open strings), a descending cross-string run (raking), a high position shift and back (one-finger-per-fret), a few double-stops (a fifth, octaves), and an out-of-range low note (octave folding). |
| `make_fingerstyle_demo_midi.py` | Generates `fingerstyle_demo.mid` — a right-hand showcase that keeps the left hand simple (mostly open/first position): an eighth-note pedal (pure i-m), an octave riff (crossing + rake), a descending open-string run (textbook rake), and a sixteenth burst (fast i-m). |
| `scan_finger_collisions.py` | Regression tool: animates `fingering.json` and reports inter-finger mesh collisions per frame. Run with `blender --background --python`. |
| `RESEARCH.md` | Design notes and literature survey behind the fingering solver and the plucking/picking model. Builds on `guitar/RESEARCH.md`; covers only the bass-specific differences. |

## Quick start

```bash
# 1. Generate a demo MIDI (or point at your own file)
python bass_guitar/make_demo_bass_midi.py bass_guitar/bass_demo.mid              # left-hand test bench
python bass_guitar/make_fingerstyle_demo_midi.py bass_guitar/fingerstyle_demo.mid  # right-hand showcase

# 2. Solve the tab + fingering (outside Blender)
python -m bass_guitar.fingering bass_guitar/bass_demo.mid -o bass_guitar/fingering.json
```

Then, inside Blender (Text Editor or `execute_blender_code`):

```python
from bass_guitar import build_bass_guitar, build_hands, animate_hands
build_bass_guitar.build_bass_guitar()
build_hands.build_hands()                       # fingerstyle (default); pass style="pick" for a pick
animate_hands.animate_hands("bass_guitar/fingering.json")

# Optional — hand the bass to a full standing player:
from bass_guitar import build_bassist, animate_bassist
animate_bassist.animate_bassist("bass_guitar/fingering.json")
```

`fingering.py` also has a self-test: `python -m bass_guitar.fingering --selftest`
(and `--hand-size S|M|L`, `--style pick`). The `--style` used to solve the
fingering must match the `style=` used to build the hands.

## Coordinate & rig conventions

- Metres, +Z up. The bass lies face-up (+Z) with the neck pointing +Y:
  bridge near the world origin, headstock out at +Y, strings on top — ready
  for a top-down or 3/4 render. String 0 is the low E on the bass (−X) side,
  matching the `Bass_String_<i>` object names.
- The `FretHand` armature wraps the neck (`build_hands.WRAP_TILT`): palm beside
  the treble edge below string level, knuckle line just above it, fingers
  arching up and over the strings, and a static thumb box pressing the back of
  the neck. It is a realistic hand size matching the plucking hand; the wide
  bass frets are covered by finger splay (z-rotation), not a stretched palm.
  Finger chains use the piano rig's convention (pose-space x-rotation = curl,
  z-rotation = sideways reach), so it shares the piano's closed-form IK.
- The `PluckHand` hovers above the strings near the bridge pickup with a static
  floating thumb; each note curls the assigned finger (`pi`/`pm`) onto the
  struck string and follows through toward the palm, while the object tracks x
  to keep that finger over the string. The `PickHand` (`--style pick`) is driven
  purely by object location, sweeping the pick tip across the strings.
- The `Bassist` stands at the origin facing −Y (left = +X, right = −X). Its
  arms and legs are two-bone IK; the hand rigs bone-parent to the `hand.*`
  stubs so the arm IK carries them. In `animate_bassist`, the bass + hands ride
  one holder bone-parented to a torso bone, so the whole upper assembly (bass,
  strap, and hands) sways as one and the hands stay glued to the strings.

## Fingering model (`fingering.py`)

Ergonomic dynamic programming, mirroring `piano/`- and `guitar/fingering.py`'s
beam-searched Viterbi (Sayegh 1989; Radisavljevic & Driessen 2004; Hori 2021),
with three bass-specific changes:

- **Four low strings, near-monophonic.** Tuning is EADG = E1 A1 D2 G2 (MIDI
  28, 33, 38, 43), range MIDI 28–63 over 20 frets on an 864 mm (34") scale.
  Bass lines are overwhelmingly monophonic, so the guitar's index-barre grips
  are dropped and the search collapses to a near-linear per-note
  `(string, fret, finger)` choice dominated by hand-position continuity.
- **Position-dependent left hand.** On a 34" scale the low frets are physically
  enormous, so below `SIMANDL_MAX_FRET` the hand uses the **Simandl 1-2-4**
  system (fingers 1, 2, 4 over three frets; the 3rd finger only reinforces the
  4th and is penalized when used alone), and above it reverts to
  **one-finger-per-fret** (1-2-3-4). Both regimes fall out of one
  `_finger_offset` table. A **metric** reach limit `REACH_MAX_M` (in metres, so
  it tightens up the neck where frets shrink) does most of the physical pruning
  for free.
- **States** are feasible grips per onset; **static costs** favour open strings,
  low positions, and compact grips agreeing on one implied hand position
  (`pos_spread`); **transition costs** (urgency-scaled) penalize hand-position
  shifts (the dominant term), same-finger fret hops, and string changes. Held
  notes never change string or fret. Out-of-range pitches are octave-folded with
  a warning; over-full chords are thinned to bass + top three.

## Plucking model (`fingering.py` / `animate_hands.py`)

The default fingerstyle right hand plucks with alternating index/middle
(`note["pluck_finger"] ∈ {"i","m"}`):

- **Strict i-m alternation** on a repeated string and when ascending to a
  thinner (higher-index) string.
- **Raking** when a line descends across strings toward a thicker string: the
  finger that just plucked drags across and repeats instead of alternating — the
  natural follow-through as a plucking finger comes to rest on the next string
  down.
- **Re-anchoring** to the index after a rest (`PLUCK_GAP_RESET`). In a
  double-stop the index takes the lower string, the middle the next up.

A `--style pick` mode instead reuses the guitar's metric ("pendulum") down/up
pick-stroke model verbatim (down on the beat, up off it), so the same fingering
engine can drive either the plucking-finger rig or a pick rig.

## Output

Demo `.blend` files saved by prior runs:

- `bass_guitar.blend` — bass only, no hands/animation.
- `animated_bass_guitar.blend` — the bass with the two hand rigs playing it
  (no body), straight from `animate_hands`.
- `animated_bassist.blend` — `bass_demo.mid` (`fingering.json`) played by the
  full standing bassist (`animate_bassist`).
- `fingerstyle_demo.blend` — `fingerstyle_demo.mid` (`fingerstyle_demo.json`)
  right-hand showcase.

See `RESEARCH.md` for the full design rationale and out-of-scope items (slap/pop,
ghost notes, muting, hammer-ons/slides/bends/harmonics, 5-/6-string basses and
alternate tunings, ring-finger use, learned cost weights).
