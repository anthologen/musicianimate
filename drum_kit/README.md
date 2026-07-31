# drum_kit

Procedurally builds a minimalist 5-piece drum kit and a stand-in drummer in
Blender, then animates both hands (sticks) and both feet (kick + hi-hat
pedals) from a MIDI drum track: MIDI → limb plan → rigged hands/feet.

## Pipeline

```
your_song.mid
    │
    ▼
python -m drum_kit.fingering song.mid -o fingering.json   # limb scheduler (outside Blender)
    │
    ▼
build_drum_kit.build_drum_kit()      # kit meshes            ┐
build_drummer.build_drummer()        # drummer armature + sticks   (inside Blender)
animate_drums.animate_drums("fingering.json")  # keyframes   ┘
    │
    ▼
bpy.ops.wm.save_as_mainfile(filepath="animated_drum_kit.blend")
```

`fingering.py` runs as plain, dependency-free Python outside Blender.
`build_drum_kit.py`, `build_drummer.py` and `animate_drums.py` need `bpy` and
must run inside Blender (Scripting tab, Text Editor "Run Script", or the
Blender MCP `execute_blender_code` tool).

## Files

| File | Role |
|---|---|
| `kit_layout.py` | bpy-free source of truth: kit geometry, the General MIDI drum map, and per-voice strike targets. The only file to touch to swap in a different kit model. |
| `build_drum_kit.py` | Builds the kit meshes (kick, snare, 2 rack toms, floor tom, hi-hat pair, crash, ride) into a `DrumKit` collection. Self-contained (bmesh/mathutils only). |
| `build_drummer.py` | Builds a blocky, faceless humanoid `Drummer` armature (two-bone IK arms + wrist joint, sticks in hand, shoes, throne) into a `Drummer` collection. Run after `build_drum_kit.py`. |
| `fingering.py` | Drum striking planner — assigns each MIDI hit to a limb (right/left hand or foot) and emits `fingering.json`. A limb *scheduler*, not a fingering search (no pitch ambiguity: each GM percussion note maps to exactly one voice). |
| `animate_drums.py` | Keyframes the built rig from `fingering.json`: wind-up → contact → rebound strokes for the hands (Moeller method — accents whip from the forearm, taps flick from the wrist) and press/release for the feet. |
| `make_demo_drum_midi.py` | Generates `drum_demo.mid` — a musical groove touching every piece of the kit (basic beat, open-hat groove, dynamics, tom fill, sixteenth burst, ride section, cymbal tags, ending). |
| `make_demo_drum_midi2.py` | Generates `drum_demo2.mid` — a dynamics showcase: solo ppp→fff crescendos on every GM voice in turn, so the velocity→wind-up/strike-speed scaling reads clearly per voice before anything overlaps. |
| `RESEARCH.md` | Design notes and survey of the sticking model and stroke mechanics. |

## Quick start

```bash
# 1. Generate a demo MIDI (or point at your own file)
python -m drum_kit.make_demo_drum_midi        # -> drum_demo.mid

# 2. Plan the sticking (outside Blender)
python -m drum_kit.fingering drum_kit/drum_demo.mid -o drum_kit/fingering.json
```

Then, inside Blender (Text Editor or `execute_blender_code`):

```python
from drum_kit import build_drum_kit, build_drummer, animate_drums
build_drum_kit.build_drum_kit()
build_drummer.build_drummer()
animate_drums.animate_drums("drum_kit/fingering.json")
```

`fingering.py` also has a self-test: `python -m drum_kit.fingering --selftest`.

## Coordinate & rig conventions

- Metres, +Z up, floor at z = 0. Drummer sits at +Y looking toward -Y (the
  audience), so the drummer's LEFT hand points +X and RIGHT points -X;
  right-handed layout (hi-hat far left, floor tom & ride right).
- Each drum is shell + 2 heads + 2 hoops parented to a group empty at the
  drum centre (`Kick`/`Snare`/`Tom1`/`Tom2`/`FloorTom`) — animate that empty
  to shake the whole drum. Cymbals and the kick beater are single meshes
  with origin at their pivot (cymbal centre / pedal axle) so rotation alone
  produces tilt, wobble, and hi-hat open/close.
- The drummer's sticks are driven purely as *target areas + velocities*:
  each hand has a two-bone IK chain (upper arm + forearm) reaching a
  `Wrist_*` empty, with a separate hand bone Damped-Tracking the
  `IK_Hand_*` stick-tip empty — so the elbow hangs naturally and the whole
  arm reorients toward each strike point, pivoting from the shoulder rather
  than staying pointed at the audience.

## Sticking model (`fingering.py`)

Right-handed convention with automatic fallbacks:

- Feet are deterministic — kick = right foot, hi-hat pedal = left foot.
- Hands default to convention — right hand on hi-hat/ride, left hand on
  snare; toms and crashes are "flex" voices with no fixed hand.
- Fast successions (inter-onset gap below a threshold) force strict
  left/right alternation (a single-stroke roll) so runs never ask one hand
  to outpace a real arm.
- Otherwise each hit goes to whichever hand minimizes travel from its last
  position, penalized for abandoning a voice's convention hand.
- Simultaneous hits split across the two hands to minimize total travel.
- Reach feasibility overrides everything else: a hand cannot cross the kit
  faster than `REACH_V_MAX` allows — if the convention pick can't physically
  reach in time, the planner reassigns rather than teleporting a hand
  across the kit.

## Stroke model (`animate_drums.py`)

Hand strokes follow the **Moeller method** rather than scaling every hit's
wind-up by its own velocity in isolation: each hit is classified as an
*accent* or a *tap* (relative to that hand's recent velocities), and the
accent/tap of a hit plus the next one select backswing and rebound heights
(full/down/tap/up strokes). Accents are a whole-arm whip — the forearm leads
down early and the stick bead trails and cracks through; taps are a
wrist-only flick. Velocity scales apex height and strike speed. Feet follow
a simpler press/release model for the kick and hi-hat pedals.

## Output

Demo `.blend` files saved by prior runs:

- `drum_kit.blend` — kit only, no drummer/animation.
- `animated_drum_kit.blend` — `drum_demo.mid` (`fingering.json`) fully animated.
- `animated_drum_kit2.blend` / `animated_drum_kit2_moeller.blend` —
  `drum_demo2.mid` (`fingering2.json`) dynamics showcase, before/after the
  Moeller stroke model.

See `RESEARCH.md` for the full design rationale and open limitations (e.g.
simultaneous >2-voice clusters on one hand).
