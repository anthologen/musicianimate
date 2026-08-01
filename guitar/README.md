# guitar

Procedurally builds a minimalist electric guitar and a stand-in guitarist in
Blender, then animates a fretting hand and a picking hand from any MIDI melody:
MIDI → tablature/fingering → rigged hands → optional full-body player.

Unlike a piano key, a guitar pitch is ambiguous — most notes can be played at
several `(string, fret)` positions, each takeable by any of four fingers — so
the core of this folder is an ergonomic tab/fingering solver, not just a
mesh builder. See `RESEARCH.md` for the survey behind it.

## Pipeline

```
your_song.mid
    │
    ▼
python -m guitar.fingering song.mid -o fingering.json   # tab + fingering solver (outside Blender)
    │
    ▼
build_guitar.build_guitar()          # guitar meshes                ┐
build_hands.build_hands()            # FretHand + PickHand rigs      (inside Blender)
animate_hands.animate_hands("fingering.json")  # keyframes           ┘
    │
    ▼  (optional: put the guitar in a player's hands)
build_guitarist.build_guitarist()    # standing humanoid rig
animate_guitarist.animate_guitarist("fingering.json")  # full-body playing shot
    │
    ▼
bpy.ops.wm.save_as_mainfile(filepath="animated_guitar.blend")
```

`fingering.py` and `fret_layout.py` run as plain, dependency-free Python
outside Blender. `build_guitar.py`, `build_hands.py`, `build_guitarist.py`,
`animate_hands.py` and `animate_guitarist.py` need `bpy` and must run inside
Blender (Scripting tab, Text Editor "Run Script", `blender --background
--python …`, or the Blender MCP `execute_blender_code` tool).

## Files

| File | Role |
|---|---|
| `fret_layout.py` | bpy-free source of truth: neck geometry, tuning (EADGBE), fret spacing, and string/press target coordinates. Imported by both the builder (inside Blender) and the solver (outside), so computed fingering targets land exactly on the built meshes. The single knob for tuning/scale length. |
| `build_guitar.py` | Builds a realistically sized 25.5" solid-body electric (slab body, tapered neck, 22 frets, six strings, two humbuckers, bridge, inline tuners) into a `Guitar` collection. Lies face-up (+Z), neck pointing +Y. Self-contained (bmesh/mathutils only). |
| `build_hands.py` | Builds the `FretHand` (wrist + four three-bone finger chains, wrapped around the neck with a static thumb behind it) and `PickHand` (a loose fist pinching a flat pick) armatures into a `GuitarHands` collection. Run after `build_guitar.py`. |
| `fingering.py` | The tab/fingering solver — beam-searched Viterbi DP over feasible `(string, fret, finger)` grips (incl. index barres), picking the cheapest ergonomic path and emitting `fingering.json`. Mirrors `piano/fingering.py`. Also decides right-hand pick direction (down/up) from beat position. |
| `animate_hands.py` | Keyframes the two hand rigs from `fingering.json`: the FretHand glides the neck and arches fingers over their frets (closed-form two-link IK); the PickHand sweeps the pick tip across the strings — a tight flick for 1–3 strings (PICK), a big velocity-scaled arc for 4+ (STRUM). |
| `build_guitarist.py` | Builds a blocky, faceless, walk-capable humanoid `Guitarist` armature (IK arms + legs, human ROM limits on every joint, a `hand.*` stub at each wrist as the attach point for the hand rigs). Same axis convention as the drummer stand-in. |
| `animate_guitarist.py` | Stitches it all together: rebuilds guitar + hands + guitarist, runs `animate_hands`, then rigidly mounts the "guitar + playing hands" assembly onto the standing player's torso, straps it on, makes the arms' IK follow the hands, and adds a gentle body sway. The end-to-end shot. |
| `make_demo_guitar_midi.py` | Generates `guitar_demo.mid` — a fingering test bench: C-major scale, chromatic run, open chords (C/Am/E), barre chords (F/Bm), a high position shift, plus a five-note cluster and an out-of-range note to exercise the solver's fallbacks. |
| `make_strum_guitar_midi.py` | Generates `strum_demo.mid` — one G–Em–C–D progression strummed "D D U U D U", swelling pp→ff→pp so the velocity-scaled strum motion is on display. |
| `scan_finger_collisions.py` | Regression tool: animates `fingering.json` and reports inter-finger mesh collisions per frame. Run with `blender --background --python`. |
| `RESEARCH.md` | Design notes and literature survey behind the fingering solver and the picking-direction model. |

## Quick start

```bash
# 1. Generate a demo MIDI (or point at your own file)
python guitar/make_demo_guitar_midi.py guitar/guitar_demo.mid    # fingering test bench
python guitar/make_strum_guitar_midi.py guitar/strum_demo.mid    # strumming pattern

# 2. Solve the tab + fingering (outside Blender)
python -m guitar.fingering guitar/guitar_demo.mid -o guitar/fingering.json
```

Then, inside Blender (Text Editor or `execute_blender_code`):

```python
from guitar import build_guitar, build_hands, animate_hands
build_guitar.build_guitar()
build_hands.build_hands()
animate_hands.animate_hands("guitar/fingering.json")

# Optional — hand the guitar to a full standing player:
from guitar import build_guitarist, animate_guitarist
animate_guitarist.animate_guitarist("guitar/fingering.json")
```

`fingering.py` also has a self-test: `python -m guitar.fingering --selftest`
(and `--benchmark`, `--hand-size S|M|L`).

## Coordinate & rig conventions

- Metres, +Z up. The guitar lies face-up (+Z) with the neck pointing +Y:
  bridge near the world origin, headstock out at +Y, strings on top — ready
  for a top-down or 3/4 render. String 0 is the low E on the bass (−X) side,
  matching the `Guitar_String_<i>` object names.
- The `FretHand` armature wraps the neck (`build_hands.WRAP_TILT`): palm beside
  the treble edge below string level, knuckle line just above it, fingers
  arching up and over the strings, and a static thumb box pressing the back of
  the neck — opposite the fingers, like a real grip. Finger chains use the
  piano rig's convention (pose-space x-rotation = curl, z-rotation = sideways
  reach), so it shares the piano's closed-form IK.
- The `PickHand` is driven purely by object location: the animator sweeps the
  pick tip (at `PICK_TIP_LOCAL`) across the strings; the fingers/pick are rigid.
- The `Guitarist` stands at the origin facing −Y (left = +X, right = −X). Its
  arms and legs are two-bone IK; the hand rigs bone-parent to the `hand.*`
  stubs so the arm IK carries them. In `animate_guitarist`, the guitar +
  hands ride one holder bone-parented to a torso bone, so the whole upper
  assembly sways as one and the hands stay glued to the strings.

## Fingering model (`fingering.py`)

Ergonomic dynamic programming, mirroring `piano/fingering.py`'s beam-searched
Viterbi (Sayegh 1989; Radisavljevic & Driessen 2004; Hori 2021):

- **States** are feasible `(string, fret, finger)` grips per onset event:
  distinct strings, ≤ 4 fretted notes, fretted extent within a *metric* reach
  limit (so reach loosens up the neck where frets shrink), fingers ordered
  index-to-pinky up the frets — plus **index-barre** grips where finger 1
  flattens across the lowest fret and fingers 2–4 take higher notes.
- **Hand position** follows Hori's form model: `mean(fret − (finger − 1))`
  over fretted notes; all-open grips inherit the previous position.
- **Static costs** favour open strings, low positions, and compact grips;
  **transition costs** (urgency-scaled) penalize hand-position shifts (the
  dominant term), same-finger fret hops, and string changes. A held note may
  never change string or fret.
- Out-of-range pitches are octave-folded with a warning; over-full chords are
  thinned to bass + top three.

## Picking model (`animate_hands.py`)

Right-hand stroke direction follows **metric ("pendulum") picking** — the hand
oscillates with the beat, so a note is struck whichever way the hand is moving:
**down on the beat, up off it** (`slot = round(beat / S)`; even = down, odd =
up), with the subdivision `S` auto-detected from the median inter-onset gap. A
loud note after a rest is forced to a downstroke for attack. Velocity scales
backswing distance and strike speed; onsets of 4+ strings become full STRUM
arcs whose size grows with velocity, driving the arm from the elbow. Beat
positions come from the MIDI tempo map (stored as `note["beat"]` in the JSON).

## Output

Demo `.blend` files saved by prior runs:

- `guitar.blend` — guitar only, no hands/animation.
- `animated_guitar.blend` — `guitar_demo.mid` (`fingering.json`) played by the
  full standing guitarist (`animate_guitarist`).
- `animated_strum.blend` — `strum_demo.mid` (`strum_fingering.json`) strumming
  showcase.

See `RESEARCH.md` for the full design rationale and out-of-scope items (bends,
slides, hammer-ons, fingerstyle p-i-m-a, capo/alternate tunings, learned cost
weights).
