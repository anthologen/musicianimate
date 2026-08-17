# piano

Procedurally builds a minimalist 88-key piano and a stand-in pianist in
Blender, then animates the keys, two hand rigs and the seated player from any
MIDI file: MIDI → fingering → key dips + rigged hands → optional full body.

Unlike a guitar, a piano pitch is unambiguous — every note has exactly one key —
so the hard problem here is not *where* to play a note but *with what*: which
hand takes it and which of the five fingers presses it, and then how a real hand
gets from one position to the next without sliding through the keys or through
its own fingers. Most of this folder is that: an ergonomic fingering solver and
a hand-motion model built on top of it. See `RESEARCH.md` for the survey and the
design rationale behind every constant.

## Pipeline

```
your_song.mid
    │
    ▼
python -m piano.fingering song.mid -o fingering.json   # hand split + fingering solver (outside Blender)
    │
    ▼
build_piano.build_piano()               # 88 keys + case + camera/lights      ┐
build_hands.build_hands()               # Hand_L / Hand_R armatures           │ (inside Blender)
piano_midi_animator.animate_piano("song.mid")   # key dips                    │
animate_hands.animate_hands("fingering.json")   # wrist + finger keyframes    ┘
    │
    ▼  (optional: put a player on the bench)
build_pianist.build_pianist()           # seated humanoid rig on a stool
animate_pianist.animate_pianist("fingering.json", "song.mid")   # full-body shot
    │
    ▼
bpy.ops.wm.save_as_mainfile(filepath="animated_piano.blend")
```

`fingering.py` and `key_layout.py` run as plain, dependency-free Python outside
Blender (the MIDI parser is hand-rolled). `build_piano.py`, `build_hands.py`,
`build_pianist.py`, `piano_midi_animator.py`, `animate_hands.py` and
`animate_pianist.py` need `bpy` and must run inside Blender (Scripting tab, Text
Editor "Run Script", `blender --background --python …`, or the Blender MCP
`execute_blender_code` tool).

`animate_pianist.animate_pianist()` runs steps 1–4 itself, so it is usually the
only call you need inside Blender.

## Files

| File | Role |
|---|---|
| `key_layout.py` | bpy-free source of truth: key dimensions, the per-semitone black-key offsets, the 88-key span centred on x = 0, and the fingertip target (`key_x`/`fingertip_y`/`rest_z`) for every MIDI note. Imported by both the builder (inside Blender) and the solver (outside), so computed fingertip targets land exactly on the built key meshes. The single knob for keyboard geometry. |
| `build_piano.py` | Builds the `Piano` collection: 88 keys named `WhiteKey_<midi>` / `BlackKey_<midi>` (21 = A0 … 108 = C8 — those names are the animator's contract), a matte-black case (keybed slab, back rail, two trestle legs, stretcher), materials, a camera and two-point studio lighting, and Standard color management so blacks stay black. Also exports `FLOOR_Z`, the floor the pianist's stool and feet stand on. |
| `piano_midi_animator.py` | Parses a Standard MIDI File (no dependencies) and keyframes the key objects. Note-on velocity controls press *speed*, not depth: a loud note reaches the bottom of key travel in fewer frames. |
| `fingering.py` | The fingering solver — hand-splitting Viterbi pass, then a per-hand beam-searched Viterbi over complete finger-to-key assignments of each chord, with Parncutt-style ergonomic costs. Emits `fingering.json` (per note: hand, finger, fingertip x/y, plus a smoothed per-hand `wrist_x` curve). Runs outside Blender. |
| `build_hands.py` | Builds the `Hands` collection: `Hand_L` / `Hand_R` armatures, each a `wrist` root plus five three-bone chains `f<n>_prox → f<n>_mid → f<n>_dist` (f1 = thumb … f5 = pinky), every phalanx caged to its human range of motion, with rigid boxes bone-parented on. Adult-hand anthropometry, shared with the guitarist's and bassist's fretting hands so all the players read as the same size of human. The left hand mirrors the right across x; bone names and axes are identical. |
| `animate_hands.py` | The big one. Keyframes both hand rigs from `fingering.json`: the object transform carries the wrist (position, yaw toward the reaching arm, and a pitch stroke into each note), finger bones are solved by closed-form two-link IK in the hand's own frame, travel between positions lifts over an arc on a minimum-jerk profile, and the whole hand is solved together so no two digits intersect on any rendered frame. Baked FK — no IK constraints in the result. |
| `build_pianist.py` | Builds a blocky, faceless humanoid `Pianist` armature *seated* on a minimalist stool at `build_piano.FLOOR_Z` — same stand-in family and anthropometry as the guitarist/bassist/drummer, but posed sitting, because the bench height is what puts the forearms level with the keys. Two-bone IK arms (`Wrist_*` target, `Elbow_*` pole, `hand.*` stub) and two-bone IK legs (`Ankle_*`, `Knee_*`). |
| `animate_pianist.py` | Stitches it all together: rebuilds piano + pianist + hands, dips the keys from the MIDI, runs `animate_hands`, points each arm's IK wrist target at the matching hand rig's wrist bone so the arms follow the hands along the keyboard, adds a torso sway / head bob / gaze that tracks the hands, and validates both wrists against human ROM across the take. |
| `make_demo_piano_midi.py` | Generates `piano_demo.mid` — a one-octave C major scale from soft to loud, a soft-then-loud C major chord (velocity → press speed), and a two-hand section (walking bass under a two-octave scale run) that exercises hand splitting and thumb-under crossings. |
| `make_reach_piano_midi.py` | Generates `piano_reach.mid` — a reach test, not a demo: the lowest and highest keys one at a time, A0 ↔ C8 alternation, a contrary-motion sweep across the full 1.2 m span, and the widest chord the board allows. This is what exercises the arm IK, the wrist ROM guard and the gaze. |
| `RESEARCH.md` | Design notes and literature survey behind the fingering solver, the hand anthropometry and ROM cage, the seated-player geometry, the travel/lift model, the finger-clearance solve, the thumb-under rule and the wrist stroke — plus what is deliberately out of scope. |

## Quick start

```bash
# 1. Generate a demo MIDI (or point at your own file)
python piano/make_demo_piano_midi.py piano/piano_demo.mid       # musical demo
python -m piano.make_reach_piano_midi piano/piano_reach.mid     # full-span reach test

# 2. Solve the hand split + fingering (outside Blender)
python -m piano.fingering piano/piano_demo.mid -o piano/fingering.json
```

Then, inside Blender (Text Editor or `execute_blender_code`):

```python
from piano import animate_pianist
animate_pianist.animate_pianist("piano/fingering.json", "piano/piano_demo.mid")
```

Or drive the pieces yourself, without a player:

```python
from piano import build_piano, build_hands, piano_midi_animator, animate_hands
build_piano.build_piano()
build_hands.build_hands()
piano_midi_animator.animate_piano("piano/piano_demo.mid")
animate_hands.animate_hands("piano/fingering.json")
```

`fingering.py` also has a self-test: `python -m piano.fingering --selftest`
(plus `--hand-size`, one of `XXS XS S M L XL XXL`, which scales the span tables,
and `--hands-from-tracks`, which takes the hand split from the MIDI's own two
tracks instead of solving it).

The MIDI is a separate argument from the `fingering.json` because the JSON
records only the basename it came from — pass the matching pair, or the keys
will dip under nothing.

## Coordinate & rig conventions

- Metres, +Z up. x runs along the keyboard (0 at its centre, low notes at −x),
  y runs from the player (0 at the key fronts) toward the fallboard. White key
  tops sit at z = 0.02; the floor is at `build_piano.FLOOR_Z`, one keybed slab
  plus one trestle leg below the keys.
- **Note the pianist faces the opposite way from the standing players.** They sit
  at negative y facing +Y, so their LEFT hand is at −x (bass) and their RIGHT at
  +x (treble) — which is how `build_hands.REST_LOCATION` places `Hand_L` /
  `Hand_R`.
- Finger bones point +y in rest pose with roll 0, so pose-space x-rotation is
  curl and z-rotation is sideways reach — the two axes `animate_hands.py` drives
  with closed-form IK. The same convention is reused by the guitar and bass
  fretting hands, so they share this folder's IK.
- The hand rigs are wholly independent of the pianist: `animate_hands` authors
  them in world space over the real keys, and the arm IK only ever *reads* from
  them. That is why joining the body on is far simpler here than for the
  guitarist — a piano is furniture, not something worn, so there is no holder,
  no re-parenting and no strap, and no dependency cycle to reason about.
- Press timing is shared between `piano_midi_animator.py` and `animate_hands.py`
  on purpose (velocity → attack speed, same release tail), so fingertips and
  keys dip together when both are run on the same MIDI.

## Fingering model (`fingering.py`)

Dependency-free implementation of the standard ergonomic dynamic-programming
method (Parncutt et al. 1997; Al Kasimi et al. 2007; Nakamura et al.):

1. **Onset clustering** — notes within `ONSET_EPS` become one chord event.
2. **Hand split** — a Viterbi pass over per-event split points (everything below
   a boundary goes to the left hand), scoring hand span, movement speed and hand
   crossing; a simplified take on Nakamura's merged-output HMM. `--hands-from-tracks`
   skips it when the MIDI already separates the hands.
3. **Per-hand fingering** — a second beam-searched Viterbi whose *states* are
   complete finger-to-key assignments of the sounding chord, with Parncutt's
   ergonomic rules as costs: finger-pair span tables (scaled by `--hand-size`),
   weak-finger use, thumb on black keys, thumb passing, position change and
   same-finger repeats. Held notes pin their finger.

Output is `fingering.json`: `{source, hand_size, note_count, notes[], hands{}}`,
where each note carries `start`, `end`, `midi`, `velocity`, `hand`, `finger`
(1 = thumb … 5 = pinky), the fingertip target `x`/`y` from `key_layout`, and
`is_black`; and `hands` holds a smoothed `wrist_x` curve per hand.

## Hand motion model (`animate_hands.py`)

The realism lives here rather than in the mesh. Briefly, in the order it matters:

- **The hand goes to the notes, not the fingers after them.** How much reaching
  the wrist takes is set by what the fingers have room for *beside each other*
  (`SPLAY_BESIDE`), not by each knuckle's isolated abduction limit — a finger
  adducted to its own limit is already through its neighbour.
- **Travel lifts.** Moving between positions the wrist and every non-holding
  fingertip rise together over a smooth arc, so the hand clears the keyboard
  instead of sliding across it; a key the hand walks out on is released rather
  than dragged. The move is sampled along a minimum-jerk profile and given
  enough time to stay inside human speed and acceleration limits.
- **The wrist yaws and strokes.** It turns out toward the arm reaching it, so a
  hand playing far from its own shoulder does not leave the whole diagonal in
  the wrist; and it flexes down into each note and rebounds off it, as deep as
  the note is loud and the chord is wide, where the passage leaves room.
  Everything below the wrist is solved in the hand's own frame, so the
  fingertips hold their keys through the stroke.
- **The thumb is not a finger.** Its column is rolled about its own length
  (the pronation half of opposition) so folding it carries the tip across the
  palm; it only goes *under* the palm on a real crossing, and otherwise stays
  turned out on its own side of the hand while the wrist covers the note.
- **No two digits intersect.** The whole hand is solved at the union of every
  finger's keyframe times; idle fingers settle back over their own knuckles,
  non-holding digits are slid aside and lifted until their phalanges clear by
  real surface distance, and anything with nowhere to go withdraws toward its
  knuckle and waits. The result is then re-measured on the *baked* curves at
  every rendered frame — an ease between two clear poses is not itself clear —
  and any frame still crossed is solved outright. `animate_hands` returns that
  measurement as `finger_clear_mm` per hand.
- **Every keyframe is legal.** Poses are clamped into the joint cage
  `build_hands.py` puts on the bones rather than leaving the bone constraints to
  rewrite them, so the baked FK is what you see.

## Output

Demo `.blend` files saved by prior runs (untracked — `*.blend*` is gitignored):

- `animated_piano.blend` — `piano_demo.mid` played by the full seated pianist.
- `animated_piano_reach.blend` — `piano_reach.mid`, the A0 ↔ C8 full-span reach
  test.

See `RESEARCH.md` for the full rationale and out-of-scope items (learned cost
weights from the PIG dataset, hand interdependence beyond splitting, deliberate
finger substitution, pedalling, ornaments and glissandi, per-editor style
profiles).
