"""Animates the drummer's sticks and feet from a drum fingering.json.

Consumes the output of ``python -m drum_kit.fingering`` (per-hit limb, target
object, strike type and world strike point) and keyframes the rigs built by
drum_kit/build_sticks.py plus the animation-ready origins baked into
drum_kit/build_drum_kit.py:

  - Sticks (Stick_R / Stick_L): each assigned hit is a wind-up -> contact ->
    rebound stroke. The hand root's location carries the rigid stick tip to
    the strike point at the note's onset; between hits it glides toward the
    next target so arm travel reads naturally. Velocity sets both the wind-up
    height and the strike speed (loud = a taller backswing dropped faster),
    the same model as the guitar pick hand.
  - Kick (Kick_Beater + Kick_Footboard, with Shoe_R): the beater cocks back
    (further when loud), swings into the batter head at the onset, and
    rebounds; the footboard presses in step, carrying the parented shoe.
  - Hi-hat (HiHat_Top + HiHat_Footboard, with Shoe_L): the foot holds the top
    cymbal open or closed following the planner's hi-hat timeline, with a
    quick pedal "chick" on each left-foot note.
  - Cymbals (Crash / Ride / HiHat_Top): a short velocity-scaled wobble decays
    after each stick hit.

Usage (inside Blender, after build_drum_kit.py + drum_kit/build_sticks.py)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_drums", "/path/to/drum_kit/animate_drums.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_drums("/path/to/drum_kit/fingering.json")
"""

import json
import math
import os

import bpy
import mathutils

try:
    from .build_sticks import (STICK_ROT0, STICK_TIP_LOCAL, _tip_offset,
                               _rest_location)
    from piano.piano_midi_animator import _iter_action_fcurves
except ImportError:  # loaded as a loose script via importlib
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    from build_sticks import (STICK_ROT0, STICK_TIP_LOCAL, _tip_offset,
                              _rest_location)
    from piano.piano_midi_animator import _iter_action_fcurves


# --- velocity -> motion (louder = taller wind-up, faster strike) -----------
LIFT_MIN, LIFT_MAX = 0.03, 0.14        # stick apex height above the surface
STRIKE_SLOW, STRIKE_FAST = 0.12, 0.045  # apex->contact seconds (soft -> loud)
WINDUP_PITCH_MIN, WINDUP_PITCH_MAX = 0.18, 0.65  # extra tip-up at the apex, rad
HOVER = 0.03                            # rebound height above the surface

# --- kick ------------------------------------------------------------------
BEATER_STRIKE = math.radians(6.0)      # beater angle at contact (into head)
BEATER_COCK_MIN, BEATER_COCK_MAX = math.radians(5), math.radians(18)  # backswing
BOARD_PRESS_MIN, BOARD_PRESS_MAX = 0.11, 0.30  # footboard toe-down, rad

# --- hi-hat ----------------------------------------------------------------
HAT_CLOSE_DROP = 0.039                 # how far HiHat_Top slides down to close
HAT_FOOT_PRESS = math.radians(15)      # footboard press while closed

# --- cymbal wobble ---------------------------------------------------------
WOBBLE_MIN, WOBBLE_MAX = 0.02, 0.09    # rad of tip wobble
WOBBLE_SETTLE = 0.22                   # seconds to settle back to rest


def _cached_rot(obj):
    """Rest rotation_euler, cached on the object the first time (so re-runs
    mid-animation don't mistake a struck pose for rest)."""
    if "drum_rest_rot" in obj:
        return list(obj["drum_rest_rot"])
    r = list(obj.rotation_euler)
    obj["drum_rest_rot"] = r
    return r


def _cached_z(obj):
    if "drum_rest_z" in obj:
        return obj["drum_rest_z"]
    z = obj.location.z
    obj["drum_rest_z"] = z
    return z


def _monotonic(fps, frame_start):
    """A keyframe-time gate keeping successive frames strictly increasing."""
    state = {"f": None}
    min_df = 0.75

    def frame(t):
        f = frame_start + t * fps
        if state["f"] is not None and f <= state["f"] + min_df:
            f = state["f"] + min_df
        state["f"] = f
        return f

    return frame


def _animate_stick(root, side, notes, fps, frame_start):
    tip_off = _tip_offset(side)
    rx, ry, rz = STICK_ROT0[side]
    frame = _monotonic(fps, frame_start)
    last = frame_start

    def key(t, loc, pitch_extra):
        f = frame(t)
        root.location = loc
        root.rotation_euler = (rx + pitch_extra, ry, rz)
        root.keyframe_insert(data_path="location", frame=f)
        root.keyframe_insert(data_path="rotation_euler", frame=f)
        return f

    # Start hovering over this hand's home so it doesn't snap in from the rig
    # rest at frame 1.
    root.location = _rest_location(side)
    root.rotation_euler = (rx, ry, rz)
    root.keyframe_insert(data_path="location", frame=frame_start)
    root.keyframe_insert(data_path="rotation_euler", frame=frame_start)
    frame(0.0)

    prev_t = None
    for n in sorted(notes, key=lambda m: m["start"]):
        t = n["start"]
        p = mathutils.Vector((n["x"], n["y"], n["z"]))
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        lift = LIFT_MIN + v * (LIFT_MAX - LIFT_MIN)
        strike = STRIKE_SLOW - v * (STRIKE_SLOW - STRIKE_FAST)
        wpitch = WINDUP_PITCH_MIN + v * (WINDUP_PITCH_MAX - WINDUP_PITCH_MIN)
        gap = (t - prev_t) if prev_t is not None else None
        lead = min(0.08, 0.5 * gap) if gap is not None else 0.08

        apex = p + mathutils.Vector((0.0, 0.0, lift))
        key(t - strike - lead, apex - tip_off, wpitch)      # wind-up apex
        key(t, p - tip_off, 0.0)                            # contact on surface
        last = key(t + strike * 0.6 + 0.03,
                   p + mathutils.Vector((0.0, 0.0, HOVER)) - tip_off, wpitch * 0.4)
        prev_t = t
    return last


def _animate_kick(beater, board, notes, fps, frame_start):
    rest_b = _cached_rot(beater)[0]
    rest_board = _cached_rot(board)[0]
    frame = _monotonic(fps, frame_start)
    last = frame_start

    def key_beater(t, rot_x):
        beater.rotation_euler.x = rot_x
        beater.keyframe_insert(data_path="rotation_euler", index=0, frame=frame(t))

    fb = _monotonic(fps, frame_start)

    def key_board(t, rot_x):
        board.rotation_euler.x = rest_board + rot_x
        board.keyframe_insert(data_path="rotation_euler", index=0, frame=fb(t))

    beater.rotation_euler.x = rest_b
    beater.keyframe_insert(data_path="rotation_euler", index=0, frame=frame_start)
    board.rotation_euler.x = rest_board
    board.keyframe_insert(data_path="rotation_euler", index=0, frame=frame_start)
    frame(0.0)
    fb(0.0)

    for n in sorted(notes, key=lambda m: m["start"]):
        t = n["start"]
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        cock = BEATER_COCK_MIN + v * (BEATER_COCK_MAX - BEATER_COCK_MIN)
        press = BOARD_PRESS_MIN + v * (BOARD_PRESS_MAX - BOARD_PRESS_MIN)
        key_beater(t - 0.10, rest_b - cock)     # cock back
        key_beater(t, BEATER_STRIKE)            # swing into the head
        key_beater(t + 0.12, rest_b)            # rebound
        key_board(t - 0.06, 0.0)
        key_board(t, press)                     # stomp
        last = max(last, frame_start + (t + 0.14) * fps)
        key_board(t + 0.14, 0.0)                # release
    return last


def _animate_hihat(top, board, hihat_events, pedal_notes, fps, frame_start):
    rest_z = _cached_z(top)
    rest_board = _cached_rot(board)[0]
    open_z, closed_z = rest_z, rest_z - HAT_CLOSE_DROP

    top_frame = _monotonic(fps, frame_start)
    board_frame = _monotonic(fps, frame_start)

    def key_top(t, z):
        top.location.z = z
        top.keyframe_insert(data_path="location", index=2, frame=top_frame(t))

    def key_board(t, rot_x):
        board.rotation_euler.x = rest_board + rot_x
        board.keyframe_insert(data_path="rotation_euler", index=0, frame=board_frame(t))

    # Start closed (the foot holding the hats down under the groove).
    key_top(0.0, closed_z)
    key_board(0.0, HAT_FOOT_PRESS)

    prev_state = "closed"
    for e in hihat_events:
        t, state = e["t"], e["state"]
        z = open_z if state == "open" else closed_z
        b = 0.0 if state == "open" else HAT_FOOT_PRESS
        pz = open_z if prev_state == "open" else closed_z
        pb = 0.0 if prev_state == "open" else HAT_FOOT_PRESS
        key_top(max(0.0, t - 0.03), pz)   # hold the old state right up to the change
        key_board(max(0.0, t - 0.03), pb)
        key_top(t, z)
        key_board(t, b)
        prev_state = state

    # A quick pedal "chick" on each left-foot note: lift the toe then stomp.
    for n in sorted(pedal_notes, key=lambda m: m["start"]):
        t = n["start"]
        key_board(t - 0.05, HAT_FOOT_PRESS * 0.3)
        key_board(t, HAT_FOOT_PRESS * 1.1)
        key_board(t + 0.07, HAT_FOOT_PRESS)


def _animate_cymbal(obj, notes, fps, frame_start):
    rest = _cached_rot(obj)
    frame = _monotonic(fps, frame_start)

    def key(t, rx):
        obj.rotation_euler = (rx, rest[1], rest[2])
        obj.keyframe_insert(data_path="rotation_euler", frame=frame(t))

    key(0.0, rest[0])
    for n in sorted(notes, key=lambda m: m["start"]):
        t = n["start"]
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        amp = WOBBLE_MIN + v * (WOBBLE_MAX - WOBBLE_MIN)
        key(t, rest[0] + amp)                       # struck
        key(t + WOBBLE_SETTLE * 0.5, rest[0] - amp * 0.35)  # swing back
        key(t + WOBBLE_SETTLE, rest[0])             # settle


def animate_drums(fingering_json, fps=24, frame_start=1):
    """Keyframe the drummer rigs from a drum fingering.json file."""
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    with open(fingering_json) as fh:
        data = json.load(fh)
    notes = data["notes"]
    hihat_events = data.get("hihat", [])

    def find(name):
        return bpy.data.objects.get(name)

    stick = {"R": find("Stick_R"), "L": find("Stick_L")}
    beater, kboard = find("Kick_Beater"), find("Kick_Footboard")
    htop, hboard = find("HiHat_Top"), find("HiHat_Footboard")
    crash, ride = find("Crash"), find("Ride")

    touched = [stick["R"], stick["L"], beater, kboard, htop, hboard, crash, ride]

    # Cache rest transforms (first run only) BEFORE clearing, then wipe old
    # keyframes so re-running is idempotent.
    for obj in (beater, kboard, hboard, crash, ride):
        if obj is not None:
            _cached_rot(obj)
    if htop is not None:
        _cached_z(htop)
        _cached_rot(htop)
    for obj in touched:
        if obj is not None and obj.animation_data is not None:
            obj.animation_data_clear()

    by_hand = {"R": [], "L": []}
    kick, pedal = [], []
    wobble = {"Crash": [], "Ride": [], "HiHat_Top": []}
    for n in notes:
        limb = n["limb"]
        if limb in ("R", "L"):
            by_hand[limb].append(n)
        elif limb == "footR":
            kick.append(n)
        elif limb == "footL":
            pedal.append(n)
        if n["strike"] == "stick" and n["target"] in wobble:
            wobble[n["target"]].append(n)

    last = frame_start
    for side in ("R", "L"):
        if stick[side] is not None:
            last = max(last, _animate_stick(stick[side], side, by_hand[side],
                                            fps, frame_start))
    if beater is not None and kboard is not None:
        last = max(last, _animate_kick(beater, kboard, kick, fps, frame_start))
    if htop is not None and hboard is not None:
        _animate_hihat(htop, hboard, hihat_events, pedal, fps, frame_start)
    for name, obj in (("Crash", crash), ("Ride", ride), ("HiHat_Top", htop)):
        if obj is not None and wobble[name]:
            _animate_cymbal(obj, wobble[name], fps, frame_start)

    for obj in touched:
        if obj is None or obj.animation_data is None:
            continue
        for fcurve in _iter_action_fcurves(obj.animation_data.action):
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'SINE'

    scene.frame_start = frame_start
    scene.frame_end = int(round(last)) + fps
    scene.frame_set(frame_start)
    return {"hits": len(notes), "frame_end": scene.frame_end, "fps": fps}


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fingering.json")
    animate_drums(path)
