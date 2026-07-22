"""Animates the FretHand / PluckHand (or PickHand) rigs from a bass
fingering.json.

Consumes the output of ``python -m bass_guitar.fingering`` and keyframes
the armatures built by bass_guitar/build_hands.py:

  - The FretHand wraps the neck (build_hands.WRAP_TILT): its palm hangs
    beside the treble edge with the static thumb pressing the neck back,
    the object location carries the wrist along the neck so the pressing
    fingers' knuckles ride the treble edge over their frets, gliding
    between events with the piano animator's smoothing. Finger bones arch
    over the strings, driven by the piano's closed-form two-link IK in the
    tilted frame; the per-event wrist yaw/roll come from a
    collision-aware grid search. Open strings need no press, so the fret
    hand simply keeps gliding through them. (Reused wholesale from the
    guitar animator; the bass never barres, so barre handling is dropped.)
  - The PluckHand (fingerstyle, default) hovers above the strings near the
    bridge pickup; per note the assigned finger (pi = index, pm = middle)
    curls down onto the struck string's pluck point and follows through,
    while the object tracks x to keep that finger over the string.
  - The PickHand (``--style pick``) sweeps a pick across the strings with
    the guitar's metric (down/up) pendulum strokes.

Usage (inside Blender, after build_bass_guitar.py + bass_guitar/build_hands.py)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_bass_hands", "/path/to/bass_guitar/animate_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_hands("/path/to/bass_guitar/fingering.json")
"""

import json
import math
import os

import bpy
import mathutils

try:
    from . import fret_layout
    from .build_hands import (FRET_FINGERS, PLUCK_FINGERS, HAND_ROT_Z,
                              WRAP_TILT, PLUCK_ROT, PICK_TIP_LOCAL,
                              pick_world_offset)
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
    from build_hands import (FRET_FINGERS, PLUCK_FINGERS, HAND_ROT_Z,
                             WRAP_TILT, PLUCK_ROT, PICK_TIP_LOCAL,
                             pick_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger,
                                     _relax_finger)


# --- fret hand -------------------------------------------------------------
KNUCKLE_Z = 0.058      # knuckle-line height while fretting (arch clearance)
HOVER_LIFT = 0.012     # fingertip height above the string while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
REACH_FRAC = 0.50      # fraction of finger length the knuckle sits back
DIST_FLEX_PRESS = 0.45
DIST_FLEX_HOVER = 0.30
PRESS_STAGGER = 0.014  # fret-slot y spread between fingers sharing a fret

# Per-event wrist rotation freedom, chosen by a collision-penalizing grid
# search (forward kinematics of the pressing fingers). Yaw turns the hand
# in the fretboard plane; roll spins about the reach axis, arching a
# far-reaching finger over its neighbour instead of through it.
WRIST_YAW_MAX = 0.56
WRIST_YAW_STEP = 0.08
WRIST_YAW_REG = 0.5
WRIST_ROLL_MAX = 0.42
WRIST_ROLL_STEP = 0.07
WRIST_ROLL_REG = 0.3
WRIST_COHERE = 2.0     # per rad^2 of pose change from the previous event
TOUCH_CLEAR = 0.013    # axis distance where finger boxes sit flush
COLLIDE_W = 25000.0    # per m^2 of clearance deficit between finger axes
RETARGET_DIST = 0.006  # lateral move beyond which a finger curls up while travelling

# --- pluck hand ------------------------------------------------------------
# The hand is over the strings on the thick (-x) side with fingers reaching
# across (+x) and down; plucking flexes the finger so its tip is drawn back
# toward the palm - across the string (in the x-z plane) and up - which is
# the motion perpendicular to the string. See build_hands for the posture.
PLUCK_FLEX = 0.75      # distal curl of a plucking finger (a fuller hook for
                       # the full-length fingers, keeps the proximal down)
PLUCK_ACROSS = 0.045   # the wrist sits this far -x of the plucked string
PLUCK_HOVER = 0.014    # fingertip above the string between plucks
PLUCK_PULL_X = 0.013   # tip drawn this far toward the palm (-x) on the pluck
PLUCK_PULL_Z = 0.008   # ...and lifted this far up as it releases the string
PLUCK_LEAD = 0.05      # seconds the finger arrives above the string before onset
PLUCK_SETTLE = 0.07    # seconds after onset the follow-through completes
HAND_ARRIVE = 0.12     # seconds the hand slides to a new string before onset

# --- pick hand (--style pick) ---------------------------------------------
PICK_HOVER = 0.012
PICK_DEPTH = 0.004
PICK_LEAD_X = 0.012
PICK_FOLLOW_X = 0.010
STRUM_TIME = 0.05
PENDULUM_SUBDIV = None
ACCENT_VEL = 100
GAP_RESET = 0.35
STRIKE_SLOW = 0.12
STRIKE_FAST = 0.045


# ===========================================================================
# Fret hand (ported from the guitar animator, barre logic removed)
# ===========================================================================

def _press_notes(event):
    """The event's fretted notes as press items, with same-fret fingers
    staggered along the fret slot so converging fingers stack diagonally
    instead of colliding (the lower finger presses farther behind the
    wire). Staggered items are copies; originals are never mutated."""
    normal = [n for n in event["notes"] if n["fret"] > 0]
    by_fret = {}
    for n in normal:
        by_fret.setdefault(n["fret"], []).append(n)
    out = []
    for group in by_fret.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        group.sort(key=lambda n: n["finger"])
        for i, n in enumerate(group):
            rep = dict(n)
            wire = fret_layout.fret_y(n["fret"])
            width = fret_layout.fret_width(n["fret"])
            rep["y"] = max(wire + 0.10 * width,
                           min(wire + 0.85 * width,
                               n["y"] + PRESS_STAGGER
                               * ((len(group) - 1) / 2.0 - i)))
            out.append(rep)
    return out


def _fret_rotation(yaw, roll):
    """The FretHand's world rotation: base wrap pose, then forearm roll
    about the reach axis, then yaw in the fretboard plane."""
    return (mathutils.Matrix.Rotation(yaw, 3, 'Z')
            @ mathutils.Matrix.Rotation(roll, 3, 'X')
            @ mathutils.Matrix.Rotation(WRAP_TILT, 3, 'Y')
            @ mathutils.Matrix.Rotation(HAND_ROT_Z, 3, 'Z'))


def _solve_wrist(press, rot, knuckle_z):
    """Wrist location placing each pressing knuckle REACH_FRAC of its
    finger length back along the rotated rest direction from its fingertip
    target, at the knuckle_z line."""
    rest_dir = rot @ mathutils.Vector((0.0, 1.0, 0.0))
    xs, ys, zs = [], [], []
    for n in press:
        spec = FRET_FINGERS[n["finger"]]
        reach = REACH_FRAC * sum(spec["lengths"])
        ko = rot @ mathutils.Vector(spec["knuckle"])
        xs.append(n["x"] - reach * rest_dir.x - ko.x)
        ys.append(n["y"] - reach * rest_dir.y - ko.y)
        zs.append(knuckle_z - ko.z)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0,
            (min(zs) + max(zs)) / 2.0)


def _seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments p1-q1 and p2-q2 (Ericson)."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = d1.length_squared, d2.length_squared, d2.dot(r)
    if a < 1e-12 and e < 1e-12:
        return r.length
    if a < 1e-12:
        s, t = 0.0, max(0.0, min(1.0, f / e))
    else:
        c = d1.dot(r)
        if e < 1e-12:
            t, s = 0.0, max(0.0, min(1.0, -c / a))
        else:
            b = d1.dot(d2)
            denom = a * e - b * b
            s = max(0.0, min(1.0, (b * f - c * e) / denom)) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, max(0.0, min(1.0, (b - c) / a))
    return ((p1 + d1 * s) - (p2 + d2 * t)).length


def _finger_fk(wrist, rot, spec, tip, flex):
    """Predicted world joint points [knuckle, prox, mid, tip] of a finger
    posed by the IK at this wrist pose."""
    ko = rot @ mathutils.Vector(spec["knuckle"])
    knuckle = mathutils.Vector(wrist) + ko
    local = rot.transposed() @ (mathutils.Vector(tip) - knuckle)
    yaw, prox, mid = _finger_ik(local.x, local.y, -local.z,
                                spec["lengths"], flex)
    sy, cy = math.sin(yaw), math.cos(yaw)
    pts = [knuckle]
    p = mathutils.Vector(spec["knuckle"])
    pitch = prox
    for length, dflex in zip(spec["lengths"], (0.0, mid, flex)):
        pitch += dflex
        p = p + length * mathutils.Vector(
            (sy * math.cos(pitch), cy * math.cos(pitch), -math.sin(pitch)))
        pts.append(mathutils.Vector(wrist) + rot @ p)
    return pts


def _pose_cost(press, wrist, rot):
    """IK strain plus predicted finger-collision penalty of one pose."""
    cost = 0.0
    inv = rot.transposed()
    chains = []
    for n in press:
        spec = FRET_FINGERS[n["finger"]]
        ko = rot @ mathutils.Vector(spec["knuckle"])
        delta = mathutils.Vector((n["x"] - (wrist[0] + ko.x),
                                  n["y"] - (wrist[1] + ko.y),
                                  n["z"] - (wrist[2] + ko.z)))
        local = inv @ delta
        cost += math.atan2(local.x, max(local.y, 0.012)) ** 2
        chains.append(_finger_fk(wrist, rot, spec,
                                 (n["x"], n["y"], n["z"]), DIST_FLEX_PRESS))
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            dmin = min(_seg_dist(chains[i][k], chains[i][k + 1],
                                 chains[j][m], chains[j][m + 1])
                       for k in range(3) for m in range(3))
            if dmin < TOUCH_CLEAR:
                cost += COLLIDE_W * (TOUCH_CLEAR - dmin) ** 2
    return cost


def _fret_event_pose(event, prev_pose=None, dt=None):
    """(wrist location, yaw, roll) for one fretted event, from a grid
    search over the wrist's rotation freedom that trades IK strain, a
    neutral wrist, and coherence with the previous pose against predicted
    finger-finger collisions."""
    press = _press_notes(event)
    if not press:
        return _solve_wrist(press, _fret_rotation(0, 0), KNUCKLE_Z), 0.0, 0.0

    cohere = 0.0
    if prev_pose is not None and dt is not None:
        cohere = WRIST_COHERE * max(0.0, 1.0 - dt / 1.5)
    ysteps = int(round(WRIST_YAW_MAX / WRIST_YAW_STEP))
    rsteps = (int(round(WRIST_ROLL_MAX / WRIST_ROLL_STEP))
              if len(press) >= 2 else 0)
    best = None
    for ri in range(-rsteps, rsteps + 1):
        roll = ri * WRIST_ROLL_STEP
        for yi in range(-ysteps, ysteps + 1):
            yaw = yi * WRIST_YAW_STEP
            rot = _fret_rotation(yaw, roll)
            wrist = _solve_wrist(press, rot, KNUCKLE_Z)
            cost = (WRIST_YAW_REG * yaw * yaw
                    + WRIST_ROLL_REG * roll * roll
                    + _pose_cost(press, wrist, rot))
            if cohere:
                cost += cohere * ((yaw - prev_pose[0]) ** 2
                                  + (roll - prev_pose[1]) ** 2)
            if best is None or cost < best[0]:
                best = (cost, wrist, yaw, roll)
    return best[1], best[2], best[3]


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

    poses = []
    prev_pose, prev_t = None, None
    for ev in events:
        pose = _fret_event_pose(
            ev, prev_pose, None if prev_t is None else ev["t"] - prev_t)
        poses.append(pose)
        prev_pose, prev_t = (pose[1], pose[2]), ev["t"]
    targets = _smooth_targets(events, [p[0] for p in poses])
    rots = [(r[0], r[1]) for r in _smooth_targets(
        events, [(yaw, roll, 0.0) for _, yaw, roll in poses])]

    def key_root(t, target, rot):
        arm_obj.location = target
        arm_obj.rotation_euler = _fret_rotation(*rot).to_euler()
        frame = to_frame(t)
        arm_obj.keyframe_insert(data_path="location", frame=frame)
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    prev_t = None
    for i, (ev, target, rot) in enumerate(zip(events, targets, rots)):
        arrive = ev["t"] - ARRIVE_LEAD
        if prev_t is not None:
            arrive = max(arrive, prev_t + 0.6 * (ev["t"] - prev_t))
        arrive = max(arrive, 0.0)
        key_root(arrive, target, rot)
        end = max(n["end"] for n in ev["notes"] if n["fret"] > 0)
        depart = end
        if i + 1 < len(events):
            depart = min(depart, events[i + 1]["t"] - ARRIVE_LEAD - MIN_TRAVEL)
        depart = max(depart, ev["t"])
        if depart > arrive + 0.02:
            key_root(depart, target, rot)
            prev_t = depart
        else:
            prev_t = arrive

    per_finger = {}
    for ev, target, rot in zip(events, targets, rots):
        for n in _press_notes(ev):
            per_finger.setdefault(n["finger"], []).append((n, target, rot))

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    last_frame = frame_start
    for f, items in per_finger.items():
        spec = FRET_FINGERS[f]
        prev_off = None
        for i, (n, target, rot) in enumerate(items):
            rmat = _fret_rotation(*rot)
            rinv = rmat.transposed()
            ko = rmat @ mathutils.Vector(spec["knuckle"])
            knuckle = (target[0] + ko.x, target[1] + ko.y, target[2] + ko.z)

            def ik_inputs(tip_z):
                local = rinv @ mathutils.Vector(
                    (n["x"] - knuckle[0], n["y"] - knuckle[1],
                     tip_z - knuckle[2]))
                return local.x, local.y, -local.z

            dx, dy, dv = ik_inputs(n["z"])
            pressed = _finger_ik(dx, dy, dv, spec["lengths"], DIST_FLEX_PRESS)
            dx, dy, dv = ik_inputs(n["z"] + HOVER_LIFT)
            hover = _finger_ik(dx, dy, dv, spec["lengths"], DIST_FLEX_HOVER)

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
            nxt_hover = None
            retarget = False
            if i + 1 < len(items):
                nxt = items[i + 1][0]
                nxt_hover = max(frame_start,
                                to_frame(nxt["start"]) - attack_frames(nxt))
                release_frame = min(release_frame, nxt_hover - 0.5)
                retarget = (abs(nxt["x"] - n["x"]) + abs(nxt["y"] - n["y"])
                            > RETARGET_DIST)
            if (retarget and release_frame <= off_frame + 0.25
                    and nxt_hover - off_frame > 1.0):
                release_frame = (off_frame + nxt_hover) / 2.0
            if release_frame > off_frame + 0.25:
                if retarget:
                    _relax_finger(pbones, f, release_frame)
                else:
                    _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                                 DIST_FLEX_HOVER, release_frame)
            prev_off = off_frame
            last_frame = max(last_frame, off_frame + release_frames)

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


# ===========================================================================
# Pluck hand (fingerstyle) - default
# ===========================================================================

def _pluck_knuckle(hand_loc, finger):
    """World knuckle position of a plucking finger at a hand location."""
    ko = PLUCK_ROT @ mathutils.Vector(PLUCK_FINGERS[finger]["knuckle"])
    return (hand_loc[0] + ko.x, hand_loc[1] + ko.y, hand_loc[2] + ko.z), ko


def _pluck_ik(hand_loc, finger, target, flex):
    """(yaw, prox, mid) posing `finger` so its tip reaches `target`."""
    knuckle, _ = _pluck_knuckle(hand_loc, finger)
    inv = PLUCK_ROT.transposed()
    local = inv @ mathutils.Vector((target[0] - knuckle[0],
                                    target[1] - knuckle[1],
                                    target[2] - knuckle[2]))
    return _finger_ik(local.x, local.y, -local.z,
                      PLUCK_FINGERS[finger]["lengths"], flex)


def animate_pluck_hand(arm_obj, notes, fps, frame_start,
                       min_attack_frames, max_attack_frames):
    """Keyframe the fingerstyle plucking hand. Returns the last frame."""
    pbones = arm_obj.pose.bones
    arm_obj.animation_data_clear()
    rest_y, rest_z = arm_obj.location[1], arm_obj.location[2]

    for name in PLUCK_FINGERS:
        _relax_finger(pbones, name, frame_start)

    starts = sorted((n for n in notes), key=lambda n: n["start"])
    if not starts:
        return frame_start

    def to_frame(t):
        return frame_start + t * fps

    def _finger_of(n):
        return "pi" if n.get("pluck_finger", "i") == "i" else "pm"

    # The hand tracks x so the plucking finger sits over its string. One
    # shared x per onset event (centred on the struck strings) lets both
    # fingers reach their own strings by IK in a double-stop, instead of
    # two notes fighting over the object's x. Each pluck is then hover ->
    # contact -> follow-through -> hover on the assigned finger.
    # The wrist sits PLUCK_ACROSS to the -x (thick) side of the struck
    # string so the finger reaches across (+x) onto it; keyed once per
    # onset event (both fingers of a double-stop then reach their own
    # strings by IK) rather than per note.
    per_finger = {}
    hand_keys = []  # (t, x)
    for ev in _group_events(starts):
        mean_x = sum(fret_layout.pluck_point(n["string"])[0]
                     for n in ev["notes"]) / len(ev["notes"])
        hand_x = mean_x - PLUCK_ACROSS
        hand_keys.append((ev["t"], hand_x))
        for n in ev["notes"]:
            px, py, _ = fret_layout.pluck_point(n["string"])
            per_finger.setdefault(_finger_of(n), []).append(
                (n, (hand_x, rest_y, rest_z), (px, py)))

    for items in per_finger.values():
        items.sort(key=lambda it: it[0]["start"])

    # Keyframe the object x: arrive at each string a hair before the onset.
    prev_t = None
    for t, x in hand_keys:
        arrive = max(0.0, t - HAND_ARRIVE)
        if prev_t is not None and arrive < prev_t:
            arrive = prev_t
        arm_obj.location = (x, rest_y, rest_z)
        arm_obj.keyframe_insert(data_path="location", frame=to_frame(arrive))
        arm_obj.keyframe_insert(data_path="location", frame=to_frame(t))
        prev_t = t

    def attack_s(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        frames = max_attack_frames - vel_t * (max_attack_frames -
                                              min_attack_frames)
        return frames / fps

    last_frame = frame_start
    for finger, items in per_finger.items():
        # Each finger plucks the string directly below its own knuckle: the
        # index/middle knuckles straddle the pluck point along the string
        # (world Y), so aiming both at the same Y makes them angle inward
        # and CROSS. Offsetting the target Y to the finger's knuckle keeps
        # the finger pointing straight down its reach axis (no sideways
        # yaw) - still the same string, just contacted a few mm apart.
        ky = (PLUCK_ROT @ mathutils.Vector(PLUCK_FINGERS[finger]["knuckle"])).y
        prev_off = None
        for i, (n, hand_loc, (px, py)) in enumerate(items):
            fy = py + ky
            z0 = fret_layout.STRING_Z
            hover = _pluck_ik(hand_loc, finger,
                              (px, fy, z0 + PLUCK_HOVER), PLUCK_FLEX)
            contact = _pluck_ik(hand_loc, finger, (px, fy, z0), PLUCK_FLEX)
            # Follow-through: the tip is drawn back toward the palm (-x)
            # and lifted (+z) as the string releases - perpendicular to the
            # string, no motion along it.
            follow = _pluck_ik(hand_loc, finger,
                               (px - PLUCK_PULL_X, fy, z0 + PLUCK_PULL_Z),
                               PLUCK_FLEX + 0.25)

            on = n["start"]
            hover_t = max(0.0, on - max(attack_s(n), PLUCK_LEAD))
            if prev_off is not None:
                hover_t = max(hover_t, prev_off + 0.5 / fps)
            hover_t = min(hover_t, on)

            _pose_finger(pbones, finger, *hover, PLUCK_FLEX, to_frame(hover_t))
            _pose_finger(pbones, finger, *contact, PLUCK_FLEX, to_frame(on))
            follow_t = on + PLUCK_SETTLE * 0.5
            _pose_finger(pbones, finger, *follow, PLUCK_FLEX + 0.25,
                         to_frame(follow_t))
            back_t = on + PLUCK_SETTLE
            if i + 1 < len(items):
                back_t = min(back_t, items[i + 1][0]["start"] - 0.5 / fps)
            back_t = max(back_t, follow_t + 0.5 / fps)
            _pose_finger(pbones, finger, *hover, PLUCK_FLEX, to_frame(back_t))
            prev_off = back_t
            last_frame = max(last_frame, to_frame(back_t))

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


# ===========================================================================
# Pick hand (--style pick) - ported from the guitar animator
# ===========================================================================

def _event_beat(ev):
    beats = [n["beat"] for n in ev["notes"] if "beat" in n]
    return min(beats) if beats else None


def _pendulum_subdiv(events):
    if PENDULUM_SUBDIV is not None:
        return PENDULUM_SUBDIV
    beats = [b for b in (_event_beat(ev) for ev in events) if b is not None]
    gaps = sorted(b1 - b0 for b0, b1 in zip(beats, beats[1:]) if b1 - b0 > 1e-4)
    if not gaps:
        return 0.5
    med = gaps[len(gaps) // 2]
    grid = (1.0, 0.5, 1.0 / 3.0, 0.25)
    snapped = min(grid, key=lambda g: abs(g - med))
    return max(0.25, min(1.0, snapped))


def _pick_directions(events):
    subdiv = _pendulum_subdiv(events)
    dirs = []
    prev_t = None
    for ev in events:
        beat = _event_beat(ev)
        if beat is None:
            direction = 1 if not dirs else -dirs[-1]
        else:
            slot = round(beat / subdiv)
            direction = 1 if slot % 2 == 0 else -1
        gap = None if prev_t is None else ev["t"] - prev_t
        vel = max(n["velocity"] for n in ev["notes"])
        if gap is not None and gap >= GAP_RESET and vel >= ACCENT_VEL:
            direction = 1
        dirs.append(direction)
        prev_t = ev["t"]
    return dirs


def animate_pick_hand(arm_obj, notes, fps, frame_start):
    """Keyframe the picking hand's strokes. Returns the last frame."""
    arm_obj.animation_data_clear()
    tip_off = pick_world_offset(PICK_TIP_LOCAL)
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

    directions = _pick_directions(events)
    min_dt = 0.75 / fps
    last_t = None

    def key_after(t, tip_x, z):
        nonlocal last_t
        if last_t is not None:
            t = max(t, last_t + min_dt)
        key(t, tip_x, z)
        last_t = t
        return t

    prev_t = None
    last_frame = frame_start
    for ev, direction in zip(events, directions):
        t = ev["t"]
        gap = t - prev_t if prev_t is not None else None
        xs = sorted(n["pluck_x"] for n in ev["notes"])
        first, last = (xs[0], xs[-1]) if direction > 0 else (xs[-1], xs[0])
        chord = len(ev["notes"]) > 1
        vel_norm = max(0.0, min(1.0, max(n["velocity"]
                                         for n in ev["notes"]) / 127.0))
        lead = PICK_LEAD_X * (0.6 + 1.3 * vel_norm)
        follow = PICK_FOLLOW_X * (0.6 + 0.8 * vel_norm)
        strike = STRIKE_SLOW - vel_norm * (STRIKE_SLOW - STRIKE_FAST)
        cross_lag = STRUM_TIME if chord else 0.015
        windup = t - strike - (min(0.06, 0.4 * gap) if gap is not None else 0.06)
        key_after(windup, first - direction * lead, z_hover)
        key_after(t - 0.5 * strike, first - direction * lead * 0.4, z_pluck)
        key_after(t + cross_lag, last + direction * follow * 0.5, z_pluck)
        rise_t = key_after(t + cross_lag + 0.06,
                           last + direction * follow, z_hover)
        last_frame = max(last_frame, frame_start + rise_t * fps)
        prev_t = t

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


# ===========================================================================
# Driver
# ===========================================================================

def animate_hands(fingering_json, fps=24, frame_start=1,
                  min_attack_frames=1.0, max_attack_frames=9.0,
                  release_frames=5.0):
    """Keyframe both bass hand rigs from a fingering.json file."""
    with open(fingering_json) as fh:
        data = json.load(fh)

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    style = data.get("style", "finger")
    right = "PickHand" if style == "pick" else "PluckHand"
    for name in ("FretHand", right):
        if bpy.data.objects.get(name) is None:
            raise RuntimeError(
                f"{name} not found - run bass_guitar/build_hands.py "
                f"(style={style!r}) first")

    notes = data["notes"]
    last_frame = animate_fret_hand(
        bpy.data.objects["FretHand"], notes, fps, frame_start,
        min_attack_frames, max_attack_frames, release_frames)
    if style == "pick":
        last_frame = max(last_frame, animate_pick_hand(
            bpy.data.objects["PickHand"], notes, fps, frame_start))
    else:
        last_frame = max(last_frame, animate_pluck_hand(
            bpy.data.objects["PluckHand"], notes, fps, frame_start,
            min_attack_frames, max_attack_frames))

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, int(round(last_frame)) + fps)
    scene.frame_set(frame_start)

    fretted = sum(1 for n in notes if n["fret"] > 0)
    return {"notes_animated": len(notes), "fretted": fretted,
            "open": len(notes) - fretted, "frame_end": scene.frame_end,
            "style": style, "fps": fps}


if __name__ == "__main__":
    animate_hands(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "fingering.json"))
