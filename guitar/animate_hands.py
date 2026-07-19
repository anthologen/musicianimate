"""Animates the FretHand / PickHand rigs from a guitar fingering.json.

Consumes the output of ``python -m guitar.fingering`` (per-note string,
fret, finger, and fingertip/pluck targets) and keyframes the armatures
built by guitar/build_hands.py:

  - The FretHand object location carries the wrist along the neck: for
    every fretted event it is placed so the pressing fingers' knuckles
    hover over their frets (index toward the nut, pinky toward the
    bridge), gliding between events with the piano animator's smoothing
    so position shifts read as one sweep. Finger bones are driven by the
    piano's closed-form two-link IK; open strings need no press, so the
    fret hand simply keeps gliding through them.
  - The PickHand object location sweeps the pick tip across the strings:
    each onset gets a windup / dip / cross / lift stroke through the
    struck string's pluck point. Chords strum bass-to-treble across the
    onset; fast single-note runs alternate down- and up-strokes.
  - Press timing mirrors the piano animator: velocity sets attack speed
    (loud = fast), with the same release tail.

Usage (inside Blender, after build_guitar.py + guitar/build_hands.py)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_guitar_hands", "/path/to/guitar/animate_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_hands("/path/to/guitar/fingering.json")
"""

import json
import os

import bpy

try:
    from . import fret_layout
    from .build_hands import (FRET_FINGERS, PICK_TIP_LOCAL, hand_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger,
                                     _relax_finger)
except ImportError:  # loaded as a loose script via importlib
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    import fret_layout
    from build_hands import FRET_FINGERS, PICK_TIP_LOCAL, hand_world_offset
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger,
                                     _relax_finger)


# --- fret hand -------------------------------------------------------------
FRET_HOVER_Z = 0.085   # wrist hover height above the guitar's face
HOVER_LIFT = 0.012     # fingertip height above the string while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
REACH_FRAC = 0.50      # fraction of finger length the knuckle sits back
DIST_FLEX_PRESS = 0.45
DIST_FLEX_HOVER = 0.30

# --- pick hand -------------------------------------------------------------
PICK_HOVER = 0.012     # pick tip above the string plane between strokes
PICK_DEPTH = 0.004     # pick tip below the string plane while crossing
PICK_LEAD_X = 0.012    # windup distance before the first struck string
PICK_FOLLOW_X = 0.010  # follow-through past the last struck string
STRUM_TIME = 0.05      # seconds a chord strum spans around the onset
ALT_PICK_GAP = 0.25    # alternate stroke direction under this note gap


def _fret_event_root_target(event):
    """Wrist location placing the event's pressing knuckles over their
    frets: knuckles sit REACH_FRAC of the finger length toward the treble
    side (+x) of the fingertip target, and directly over its fret in y.
    Uses the midrange like the piano animator so stretched grips split
    the residual between their outer fingers."""
    xs, ys = [], []
    for n in event["notes"]:
        if n["fret"] == 0:
            continue
        spec = FRET_FINGERS[n["finger"]]
        kx, ky, _kz = spec["knuckle"]
        reach = REACH_FRAC * sum(spec["lengths"])
        # knuckle_world = wrist + (-ky, kx, kz) under HAND_ROT_Z
        xs.append(n["x"] + reach + ky)
        ys.append(n["y"] - kx)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
            FRET_HOVER_Z)


def animate_fret_hand(arm_obj, notes, fps, frame_start,
                      min_attack_frames, max_attack_frames, release_frames):
    """Keyframe the fretting hand from the note list. Returns last frame."""
    pbones = arm_obj.pose.bones
    arm_obj.animation_data_clear()

    def to_frame(t):
        return frame_start + t * fps

    for f in FRET_FINGERS:
        _relax_finger(pbones, f, frame_start)

    events = [ev for ev in _group_events(notes)
              if any(n["fret"] > 0 for n in ev["notes"])]
    if not events:
        return frame_start

    targets = _smooth_targets(events, [_fret_event_root_target(ev)
                                       for ev in events])

    def key_root(t, target):
        arm_obj.location = target
        arm_obj.keyframe_insert(data_path="location", frame=to_frame(t))

    prev_t = None
    for i, (ev, target) in enumerate(zip(events, targets)):
        arrive = ev["t"] - ARRIVE_LEAD
        if prev_t is not None:
            arrive = max(arrive, prev_t + 0.6 * (ev["t"] - prev_t))
        arrive = max(arrive, 0.0)
        key_root(arrive, target)
        end = max(n["end"] for n in ev["notes"] if n["fret"] > 0)
        depart = end
        if i + 1 < len(events):
            depart = min(depart, events[i + 1]["t"] - ARRIVE_LEAD - MIN_TRAVEL)
        depart = max(depart, ev["t"])
        if depart > arrive + 0.02:
            key_root(depart, target)
            prev_t = depart
        else:
            prev_t = arrive

    # Fingers: per finger so release tails clamp against the next press.
    per_finger = {}
    for ev, target in zip(events, targets):
        for n in ev["notes"]:
            if n["fret"] > 0:
                per_finger.setdefault(n["finger"], []).append((n, target))

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    last_frame = frame_start
    for f, items in per_finger.items():
        spec = FRET_FINGERS[f]
        kx, ky, kz = spec["knuckle"]
        prev_off = None
        for i, (n, target) in enumerate(items):
            # knuckle_world = wrist + (-ky, kx, kz); IK frame: local +y is
            # world -X (across the strings), local +x is world +Y.
            knuckle = (target[0] - ky, target[1] + kx, target[2] + kz)
            dy = knuckle[0] - n["x"]
            dx = n["y"] - knuckle[1]

            pressed = _finger_ik(dx, dy, knuckle[2] - n["z"],
                                 spec["lengths"], DIST_FLEX_PRESS)
            hover = _finger_ik(dx, dy, knuckle[2] - (n["z"] + HOVER_LIFT),
                               spec["lengths"], DIST_FLEX_HOVER)

            on_frame = to_frame(n["start"])
            off_frame = max(on_frame, to_frame(n["end"]))
            hover_frame = max(frame_start, on_frame - attack_frames(n))
            if prev_off is not None:
                hover_frame = max(hover_frame, prev_off + 0.5)
            hover_frame = min(hover_frame, on_frame)

            _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                         DIST_FLEX_HOVER, hover_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         DIST_FLEX_PRESS, on_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         DIST_FLEX_PRESS, off_frame)

            release_frame = off_frame + release_frames
            if i + 1 < len(items):
                nxt = items[i + 1][0]
                nxt_hover = max(frame_start,
                                to_frame(nxt["start"]) - attack_frames(nxt))
                release_frame = min(release_frame, nxt_hover - 0.5)
            if release_frame > off_frame + 0.25:
                _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                             DIST_FLEX_HOVER, release_frame)
            prev_off = off_frame
            last_frame = max(last_frame, off_frame + release_frames)

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


def animate_pick_hand(arm_obj, notes, fps, frame_start):
    """Keyframe the picking hand's strokes. Returns the last frame."""
    arm_obj.animation_data_clear()
    tip_off = hand_world_offset(PICK_TIP_LOCAL)
    obj_y = fret_layout.PLUCK_Y - tip_off[1]
    z_hover = fret_layout.STRING_Z + PICK_HOVER - tip_off[2]
    z_pluck = fret_layout.STRING_Z - PICK_DEPTH - tip_off[2]

    def key(t, tip_x, z):
        arm_obj.location = (tip_x - tip_off[0], obj_y, z)
        arm_obj.keyframe_insert(data_path="location",
                                frame=frame_start + t * fps)

    events = _group_events(notes)
    if not events:
        return frame_start

    min_dt = 0.75 / fps  # keep successive keyframes distinct
    last_t = None

    def key_after(t, tip_x, z):
        nonlocal last_t
        if last_t is not None:
            t = max(t, last_t + min_dt)
        key(t, tip_x, z)
        last_t = t
        return t

    prev_dir = -1
    prev_t = None
    last_frame = frame_start
    for ev in events:
        t = ev["t"]
        gap = t - prev_t if prev_t is not None else None
        direction = 1
        if gap is not None and gap < ALT_PICK_GAP and len(ev["notes"]) == 1:
            direction = -prev_dir
        xs = sorted(n["pluck_x"] for n in ev["notes"])
        first, last = (xs[0], xs[-1]) if direction > 0 else (xs[-1], xs[0])
        chord = len(ev["notes"]) > 1
        dip_lead = STRUM_TIME if chord else 0.02
        cross_lag = STRUM_TIME if chord else 0.015

        windup = t - (min(0.10, 0.6 * gap) if gap is not None else 0.10)
        key_after(windup, first - direction * PICK_LEAD_X, z_hover)
        key_after(t - dip_lead, first - direction * PICK_LEAD_X * 0.5, z_pluck)
        key_after(t + cross_lag, last + direction * PICK_FOLLOW_X * 0.5,
                  z_pluck)
        rise_t = key_after(t + cross_lag + 0.06,
                           last + direction * PICK_FOLLOW_X, z_hover)
        last_frame = max(last_frame, frame_start + rise_t * fps)
        prev_dir = direction
        prev_t = t

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


def animate_hands(fingering_json, fps=24, frame_start=1,
                  min_attack_frames=1.0, max_attack_frames=9.0,
                  release_frames=5.0):
    """Keyframe both guitar hand rigs from a fingering.json file."""
    with open(fingering_json) as fh:
        data = json.load(fh)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    for name in ("FretHand", "PickHand"):
        if bpy.data.objects.get(name) is None:
            raise RuntimeError(
                f"{name} not found - run guitar/build_hands.py first")

    notes = data["notes"]
    last_frame = animate_fret_hand(
        bpy.data.objects["FretHand"], notes, fps, frame_start,
        min_attack_frames, max_attack_frames, release_frames)
    last_frame = max(last_frame, animate_pick_hand(
        bpy.data.objects["PickHand"], notes, fps, frame_start))

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, int(round(last_frame)) + fps)
    scene.frame_set(frame_start)

    fretted = sum(1 for n in notes if n["fret"] > 0)
    return {"notes_animated": len(notes), "fretted": fretted,
            "open": len(notes) - fretted, "frame_end": scene.frame_end,
            "fps": fps}


if __name__ == "__main__":
    animate_hands(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "fingering.json"))
