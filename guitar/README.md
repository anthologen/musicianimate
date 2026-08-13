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
| `build_hands.py` | Builds the `FretHand` (wrist + four three-bone finger chains, wrapped around the neck with a static jointed thumb behind it) and `PickHand` (a loose fist of full curled fingers pinching a flat pick) armatures into a `GuitarHands` collection. Both are a realistic, matched hand size; every phalanx takes its cross-section from hand anthropometry and is caged to its human range of motion. Run after `build_guitar.py`. |
| `fingering.py` | The tab/fingering solver — beam-searched Viterbi DP over feasible `(string, fret, finger)` grips (incl. index barres), picking the cheapest ergonomic path and emitting `fingering.json`. Mirrors `piano/fingering.py`. Also decides right-hand pick direction (down/up) from beat position. |
| `animate_hands.py` | Keyframes the two hand rigs from `fingering.json`: the FretHand glides the neck and arches fingers over their frets (closed-form two-link IK, within anatomical joint limits, with idle fingers held in a relaxed arch over the strings and every transition eased and anticipated); the PickHand sweeps the pick tip across the strings — a tight flick for 1–3 strings (PICK), a big velocity-scaled arc for 4+ (STRUM), each carried partly by a radial/ulnar rock of the wrist. |
| `build_guitarist.py` | Builds a blocky, faceless, walk-capable humanoid `Guitarist` armature (IK arms + legs, human ROM limits on every joint, a `hand.*` stub at each wrist as the attach point for the hand rigs). Same axis convention as the drummer stand-in. |
| `animate_guitarist.py` | Stitches it all together: rebuilds guitar + hands + guitarist, runs `animate_hands`, then rigidly mounts the "guitar + playing hands" assembly onto the standing player's torso, straps it on, makes the arms' IK follow the hands, adds a gentle body sway, and validates both wrists against human ROM (`_check_wrist_pose`). The end-to-end shot. |
| `make_demo_guitar_midi.py` | Generates `guitar_demo.mid` — a fingering test bench: C-major scale, chromatic run, open chords (C/Am/E), barre chords (F/Bm), a high position shift, plus a five-note cluster and an out-of-range note to exercise the solver's fallbacks. |
| `make_strum_guitar_midi.py` | Generates `strum_demo.mid` — one G–Em–C–D progression strummed "D D U U D U", swelling pp→ff→pp so the velocity-scaled strum motion is on display. |
| `scan_finger_collisions.py` | Regression tool: animates `fingering.json` and reports inter-finger surface clearances (axis distance minus both phalanges' half-widths) per frame. Run with `blender --background --python`. |
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
  arching up and over the strings, and a static two-segment thumb on the
  index/radial edge of the palm pressing the back of the neck — opposite the
  fingers, like a real grip. Finger chains use the piano rig's convention
  (pose-space x-rotation = curl, z-rotation = sideways reach), so it shares the
  piano's closed-form IK.
- The `PickHand` is driven by object location plus a small wrist rock: the
  animator sweeps the pick tip (at `PICK_TIP_LOCAL`) across the strings; the
  fingers/pick are rigid. Its base orientation (`PICK_ROT`) is **mount-solved**
  against the standing guitarist's picking forearm so the wrist reads straight
  — see below — and each stroke rotates the rig off it about the palm normal
  (`PICK_ULNAR` / `STRUM_ULNAR`), radial on the backswing through ulnar on the
  follow-through, so part of the crossing comes from the wrist deviating rather
  than the whole arm sliding. The tip is pinned to the same contacts either
  way: the rig's location is solved around whatever the rock angle is.
  It is a **right** hand: fingers +y, palm −z, and therefore thumb, index and
  pick all on the **−x** side (`PICK_THUMB_AXIS`), the same convention as
  `bass_guitar/build_hands.py`. Mounted, local +x points nearly straight down,
  so mirroring that sign builds a *left* hand on the right arm — thumb at the
  floor, pinky on top — while every other check still passes. `PICK_THUMB_UP_MIN`
  in `animate_guitarist` is the guard against exactly that.
- The `Guitarist` stands at the origin facing −Y (left = +X, right = −X). Its
  arms and legs are two-bone IK; the hand rigs bone-parent to the `hand.*`
  stubs so the arm IK carries them. In `animate_guitarist`, the guitar +
  hands ride one holder bone-parented to a torso bone, so the whole upper
  assembly sways as one and the hands stay glued to the strings. How the guitar
  is *worn* is three constants on that holder: `NECK_ELEV` (neck up), `BODY_YAW`
  (neck angled out toward the audience rather than lying flat across the chest,
  as on the bassist) and `ANCHOR_WORLD` (wear height). All three feed the
  mount-solved `PICK_ROT` — re-run its fixed point after touching any of them.

## Hand realism model

Both hands are built and animated to human proportions and limits, so the
player reads as a player rather than as sticks pointed at the right places.

- **Anthropometric fingers.** Every phalanx box takes its own cross-section
  from adult hand norms (`build_hands.FINGER_CROSS`): ~18 mm across an
  index/middle proximal tapering to ~11 mm at a little fingertip, depth ~1 mm
  under breadth. The two hands share one realistic size (knuckle span ~74 mm,
  86 × 72 mm palm) and one set of phalanx lengths, so they read as a pair.
- **Joint range of motion.** Each finger bone carries a `LIMIT_ROTATION` cage
  from measured active-ROM norms (`FINGER_ROT_LIMIT`; Thieme 2024, AAOS
  goniometry): MCP flexion ~100°/extension 35°/splay ±30°, and one-way flexion
  hinges at PIP/DIP. The animator's IK caps knuckle splay at `FINGER_MCP_SPLAY`
  (26°) and the wrist search *pays* for any grip that would demand more, so
  reach is the wrist's job — a finger never swings 45° sideways under its
  neighbour to grab a string.
- **Idle fingers.** A finger with no note holds a relaxed arch in its own fret
  lane, hovering just over the strings (`IDLE_*`) and curling toward whichever
  string is being played, instead of the piano rig's flat rest — which laid the
  idle fingertips *through* the strings for most of a take. When a neighbour
  presses close, an idle finger leans aside at the knuckle and/or curls back,
  picking the cheapest pose that clears; idle fingers are solved in order so
  they also stay clear of each other.
- **Linger and neck order.** A released finger does not snap straight: it lifts
  just off the string and rests in its own curl, unless staying there would put
  it on the wrong side of the next presser along the neck
  (`NECK_ORDER_MARGIN`), in which case it vacates to its ordered lane.
- **Easing and anticipation.** Fret-hand F-curves are Bezier/auto-clamped, so
  every move accelerates and decelerates. The runway before a press is *split*
  between the reach across to the fret and the descent onto it, scaled by how
  far the fingertip actually travels; a big neck shift is spread over the whole
  preceding rest; and a grip lifts `GRIP_LIFT_LEAD_FR` frames before the next
  one lands (chords notated as ringing to the last instant otherwise leave the
  incoming fingers no lane). On the two demos this took peak fingertip speed
  from 34–52 mm/frame to 12–15 and peak acceleration from 24–48 to ≤ 13.
- **Wrists.** The hand rigs are separate armatures whose orientation is
  *authored*, so no IK limit constrains them; `animate_guitarist._check_wrist_pose`
  validates both wrists against human ROM on the finished mounted rig and fails
  the build loudly otherwise. The picking hand's `PICK_ROT` is solved against
  the mounted picking forearm (finger axis along the forearm, palm onto the
  strings), which took the picking wrist from 51–93° bent — past the guard — to
  under 20°.
- **Radial/ulnar deviation.** `PICK_ROT` is the *base* pose only. A rigid hand
  slid across the strings reads as dragged, so each stroke also rocks the rig
  about its palm normal — the true deviation axis, as opposed to the knuckle
  axis (flexion) or the finger axis (forearm roll) — radial at the backswing,
  through neutral at the contact, to ulnar on the follow-through, sized by
  velocity (`PICK_ULNAR` / `STRUM_ULNAR`, ~7–9° half-swing). The pick tip is
  still pinned to exactly the same contacts: the rig's location is solved
  around the rock, so it *replaces* part of the slide rather than adding to it
  (on `guitar_demo`, 12% of the crossing comes from the wrist, and 95% of the
  hand's own rotation over the take is deviation). Because the wrist carries
  its share, the arm swings less: peak picking-wrist bend fell 19.9° → 15.2°.
- **Handedness.** Bend and palm-facing are both blind to *chirality*: a mirrored
  picking fist still runs its fingers along the forearm with its palm on the
  strings, so it passes both and simply plays upside down. `_check_wrist_pose`
  therefore also requires the picking thumb to ride on top (`PICK_THUMB_UP_MIN`).
  Note the mirror is not free: it moves the pinch — and so the wrist, which is
  the picking arm's IK target — to the other side of the hand, which drops the
  wrist ~4 cm for the same pick contact. That is why `ANCHOR_WORLD` is worn
  35 mm higher and `PICK_ROT` was re-solved on top of it; without both, the
  picking arm reaches 96% of its span at the loud-strum apexes and the elbow
  locks out at 149°.

Known residual: a guitar's strings sit only ~7–8 mm apart at the low frets, so
two realistically-sized fingers pressing adjacent strings at the same fret must
touch, exactly as a real player's do. `scan_finger_collisions.py` reports those
grip contacts as small negative surface clearances; what it is really watching
for is deep overlap or a finger crossing to the wrong side of a neighbour.

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
