"""Animates the Hand_L / Hand_R rigs from a fingering.json timeline.

Consumes the output of ``python -m piano.fingering`` (per-note hand, finger
and fingertip target) and keyframes the armatures built by build_hands.py:

  - The armature object's location carries the wrist: for every chord event
    it is placed so the pressing fingers' knuckles sit over their keys
    (arriving slightly early, gliding between events with SINE easing).
  - Finger bones are driven by closed-form two-link IK in the vertical
    plane through the knuckle: the proximal bone pitches, the middle joint
    flexes, the proximal z-rotation supplies sideways reach, and the
    distal phalanx keeps a fixed natural flexion. No IK constraints are
    used, so the result is plain baked FK keyframes.
  - Press timing mirrors piano_midi_animator.py: velocity sets attack
    speed (loud = fast), with the same release tail, so fingertips and
    keys dip together when both animators are run on the same MIDI.

Usage (inside Blender, after build_piano.py + build_hands.py)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_hands", "/path/to/animate_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_hands("/path/to/fingering.json")
"""

import json
import math
import os

import bpy

try:
    from .piano_midi_animator import _iter_action_fcurves
    from .build_hands import FINGERS
    from . import key_layout
except ImportError:  # loaded as a loose script via importlib
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from piano_midi_animator import _iter_action_fcurves
    from build_hands import FINGERS
    import key_layout


HOVER_Z = 0.065        # wrist hover height above the keybed
HOVER_LIFT = 0.015     # fingertip height above a key while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
SMOOTH_SIGMA = 1.2     # events; Gaussian smoothing width of wrist targets
SEGMENT_GAP = 0.6      # seconds; a longer silence starts a new phrase
ONSET_EPS = 0.03       # notes closer than this form one chord event
MAX_YAW = 1.0          # radians of sideways finger reach
DIST_FLEX_PRESS = 0.35
DIST_FLEX_HOVER = 0.26
RELAXED = (0.45, 0.70, 0.40)  # prox/mid/dist downward flexion at rest

# The thumb strikes white keys near their front edge (it is much shorter
# than the fingers), so its target sits closer to the player.
THUMB_FRONT_PULL = 0.012

# In a chord that mixes black and white keys the hand moves in and the
# white keys are pressed deep, between the black keys - otherwise the
# depth spread between front-of-white and black targets exceeds any
# fixed-wrist reach.
DEEP_WHITE_Y = 0.050
DEEP_WHITE_THUMB_Y = 0.045


def _target_y(note, has_black=False):
    if note["is_black"]:
        return note["y"]
    if has_black:
        return DEEP_WHITE_THUMB_Y if note["finger"] == 1 else DEEP_WHITE_Y
    if note["finger"] == 1:
        return note["y"] - THUMB_FRONT_PULL
    return note["y"]


def _finger_ik(dx, dy, dv, lengths, dist_flex):
    """Closed-form 2-link IK for one finger with a rigidly flexed distal.

    dx/dy: fingertip target offset from the knuckle in the keyboard plane;
    dv: drop from knuckle to target (positive down); dist_flex: the fixed
    distal flexion the pose will use. The mid+distal pair is treated as one
    link along the elbow-to-tip chord (length b, hanging gamma below the
    mid bone), which makes the fingertip land exactly on the target.
    Returns (yaw, prox_pitch, mid_flex), pitch/flex positive = down.
    """
    a = lengths[0]
    l2, l3 = lengths[1], lengths[2]
    b = math.hypot(l2 + l3 * math.cos(dist_flex), l3 * math.sin(dist_flex))
    gamma = math.atan2(l3 * math.sin(dist_flex), l2 + l3 * math.cos(dist_flex))
    yaw = max(-MAX_YAW, min(MAX_YAW, math.atan2(dx, max(dy, 0.012))))
    dh = math.hypot(dx, dy)
    d = math.hypot(dh, dv)
    d = max(abs(a - b) + 0.002, min(a + b - 0.002, d))
    delta = math.atan2(dv, dh)
    psi = math.acos((a * a + d * d - b * b) / (2 * a * d))
    phi = math.acos((a * a + b * b - d * d) / (2 * a * b))
    return yaw, delta - psi, math.pi - phi - gamma


def _press_z(is_black, depth_white, depth_black):
    top = key_layout.WHITE_H + (key_layout.BLACK_H if is_black else 0.0)
    return top - (depth_black if is_black else depth_white)


def _group_events(notes):
    events = []
    for n in sorted(notes, key=lambda n: (n["start"], n["midi"])):
        if events and n["start"] - events[-1]["t"] <= ONSET_EPS:
            events[-1]["notes"].append(n)
        else:
            events.append({"t": n["start"], "notes": [n]})
    return events


def _event_root_target(event, mirror):
    """Wrist position placing the event's pressing knuckles over their keys.

    Uses the midrange (not the mean) of the per-finger requirements: in a
    stretched chord the outer fingers have the least reach to spare, so the
    residual is split between them instead of letting the comfortable
    middle fingers drag the wrist off-center.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    xs, ys = [], []
    for n in event["notes"]:
        spec = FINGERS[n["finger"]]
        kx, ky, _kz = spec["knuckle"]
        reach = 0.55 * sum(spec["lengths"])
        xs.append(n["x"] - kx * mirror)
        ys.append(_target_y(n, has_black) - (ky + reach))
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, HOVER_Z)


def _smooth_targets(events, targets, sigma=SMOOTH_SIGMA):
    """Gaussian-smooth wrist targets over time.

    Per-event targets zigzag during scale runs (each note wants the wrist
    centered on a different finger), which reads as a lurching hand. Real
    pianists glide the wrist and let the fingers - especially a tucked
    thumb - cover the difference, so smoothing here is what produces the
    crossunder/crossover look: the finger IK reaches sideways from the
    gliding wrist to its key. Chords are weighted by their note count so
    stretched voicings keep (almost) exact placement; isolated events have
    no near neighbors and are unchanged.
    """
    # Segment the events at rhythmic gaps: smoothing must never blend
    # across a phrase boundary (e.g. a closing chord pulling the end of a
    # fast run toward it mid-press).
    segment = [0] * len(events)
    for i in range(1, len(events)):
        segment[i] = segment[i - 1]
        if events[i]["t"] - events[i - 1]["t"] > SEGMENT_GAP:
            segment[i] += 1

    smoothed = []
    for i in range(len(events)):
        acc = [0.0, 0.0, 0.0]
        wsum = 0.0
        for j, tgt in enumerate(targets):
            if abs(j - i) > 3 * sigma or segment[j] != segment[i]:
                continue
            # The width is in *events*, so a scale glides the same whether
            # it is played slowly or fast. Chords weigh more than single
            # notes: stretched voicings need near-exact placement.
            w = math.exp(-0.5 * ((j - i) / sigma) ** 2) * len(events[j]["notes"])
            for k in range(3):
                acc[k] += w * tgt[k]
            wsum += w
        smoothed.append(tuple(a / wsum for a in acc))
    return smoothed


def _pose_finger(pbones, finger, yaw, prox, mid, dist_flex, frame):
    for seg, rx, rz in (("prox", -prox, -yaw), ("mid", -mid, 0.0),
                        ("dist", -dist_flex, 0.0)):
        pb = pbones[f"f{finger}_{seg}"]
        pb.rotation_euler = (rx, 0.0, rz)
        pb.keyframe_insert(data_path="rotation_euler", frame=frame)


def _relax_finger(pbones, finger, frame):
    prox, mid, dist = RELAXED
    _pose_finger(pbones, finger, 0.0, prox, mid, dist, frame)


def animate_hand(arm_obj, notes, fps, frame_start,
                 press_depth_white, press_depth_black,
                 min_attack_frames, max_attack_frames, release_frames):
    """Keyframe one hand rig from its note list. Returns the last frame."""
    mirror = -1.0 if arm_obj.name.endswith("_L") else 1.0
    pbones = arm_obj.pose.bones
    arm_obj.animation_data_clear()

    def to_frame(t):
        return frame_start + t * fps

    for f in FINGERS:
        _relax_finger(pbones, f, frame_start)

    events = _group_events(notes)
    if not events:
        return frame_start

    # --- wrist (armature object) location ---------------------------------
    targets = _smooth_targets(events, [_event_root_target(ev, mirror)
                                       for ev in events])

    def key_root(t, target):
        arm_obj.location = target
        arm_obj.keyframe_insert(data_path="location", frame=to_frame(t))

    prev_t = None
    for i, (ev, target) in enumerate(zip(events, targets)):
        arrive = ev["t"] - ARRIVE_LEAD
        if prev_t is not None:
            # In fast passages give at least 60% of the window to travel,
            # so the wrist flows instead of hop-and-waiting.
            arrive = max(arrive, prev_t + 0.6 * (ev["t"] - prev_t))
        arrive = max(arrive, 0.0)
        key_root(arrive, target)
        # Depart as late as the hold allows, but always leave the wrist at
        # least MIN_TRAVEL of glide to the next event - a dwell followed by
        # a near-instant hop reads as a lurch.
        end = max(n["end"] for n in ev["notes"])
        depart = end
        if i + 1 < len(events):
            depart = min(depart, events[i + 1]["t"] - ARRIVE_LEAD - MIN_TRAVEL)
        depart = max(depart, ev["t"])
        if depart > arrive + 0.02:
            key_root(depart, target)
            prev_t = depart
        else:
            prev_t = arrive

    # --- fingers -----------------------------------------------------------
    # Process notes per finger so consecutive notes on the same finger can
    # be clamped against each other: a release tail must never land on top
    # of the next note's approach or press.
    per_finger = {}
    for ev, target in zip(events, targets):
        has_black = any(n["is_black"] for n in ev["notes"])
        for n in ev["notes"]:
            per_finger.setdefault(n["finger"], []).append(
                (n, target, has_black))

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    last_frame = frame_start
    for f, items in per_finger.items():
        spec = FINGERS[f]
        kx, ky, kz = spec["knuckle"]
        prev_off = None
        for i, (n, target, has_black) in enumerate(items):
            knuckle = (target[0] + kx * mirror, target[1] + ky,
                       target[2] + kz)
            press_z = _press_z(n["is_black"], press_depth_white,
                               press_depth_black)
            dx = n["x"] - knuckle[0]
            dy = _target_y(n, has_black) - knuckle[1]

            pressed = _finger_ik(dx, dy, knuckle[2] - press_z,
                                 spec["lengths"], DIST_FLEX_PRESS)
            hover = _finger_ik(dx, dy, knuckle[2] - (press_z + HOVER_LIFT),
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


def animate_hands(
    fingering_json,
    fps=24,
    frame_start=1,
    press_depth_white=0.0075,
    press_depth_black=0.0045,
    min_attack_frames=1.0,
    max_attack_frames=9.0,
    release_frames=5.0,
):
    """Keyframe both hand rigs from a fingering.json file."""
    with open(fingering_json) as fh:
        data = json.load(fh)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    last_frame = frame_start
    counts = {}
    for hand in ("L", "R"):
        arm_obj = bpy.data.objects.get(f"Hand_{hand}")
        if arm_obj is None:
            raise RuntimeError(
                f"Hand_{hand} not found - run build_hands.py first")
        notes = [n for n in data["notes"] if n["hand"] == hand]
        counts[hand] = len(notes)
        last_frame = max(last_frame, animate_hand(
            arm_obj, notes, fps, frame_start,
            press_depth_white, press_depth_black,
            min_attack_frames, max_attack_frames, release_frames))

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, int(round(last_frame)) + fps)
    scene.frame_set(frame_start)

    return {"notes_animated": counts, "frame_end": scene.frame_end,
            "fps": fps}


if __name__ == "__main__":
    animate_hands(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "fingering.json"))
