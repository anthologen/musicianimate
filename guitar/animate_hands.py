"""Animates the FretHand / PickHand rigs from a guitar fingering.json.

Consumes the output of ``python -m guitar.fingering`` (per-note string,
fret, finger, and fingertip/pluck targets) and keyframes the armatures
built by guitar/build_hands.py:

  - The FretHand wraps the neck (see build_hands.WRAP_TILT): its palm
    hangs beside the treble edge with the static thumb pressing the back
    of the neck, and the object location carries the wrist along the
    neck so the pressing fingers' knuckles ride the treble edge over
    their frets (index toward the nut, pinky toward the bridge), gliding
    between events with the piano animator's smoothing so position
    shifts read as one sweep. Finger bones arch over the strings, driven
    by the piano's closed-form two-link IK expressed in the tilted
    frame; open strings need no press, so the fret hand simply keeps
    gliding through them. Barre grips collapse their barred notes into
    one index press aimed at the bass-most string with a nearly straight
    finger, and the knuckle line drops to KNUCKLE_Z_BARRE so the
    flattened index lies across the strings.
  - The PickHand object location sweeps the pick tip across the strings:
    each onset gets a windup / dip / cross / lift stroke through the
    struck string's pluck point. Onsets of 1-3 strings PICK (a tight,
    low wrist flick); onsets of 4+ strings STRUM (the hand arcs clear of
    the strings and sweeps a windup/follow-through, so the arm IK drives
    the stroke from the elbow) -- and the strum's SIZE scales with velocity,
    soft strums moving little and loud strums arcing high and wide. Chords
    cross bass-to-treble across the onset; single-note runs alternate down-
    and up-strokes.
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
import math
import os

import bpy
import mathutils

try:
    from . import fret_layout
    from .build_hands import (FRET_FINGERS, HAND_ROT_Z, PICK_TIP_LOCAL,
                              WRAP_TILT, pick_world_offset)
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
    from build_hands import (FRET_FINGERS, HAND_ROT_Z, PICK_TIP_LOCAL,
                             WRAP_TILT, hand_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger,
                                     _relax_finger)


# --- fret hand -------------------------------------------------------------
KNUCKLE_Z = 0.056      # knuckle-line height while fretting (arch clearance)
KNUCKLE_Z_BARRE = 0.048  # lower knuckles while barring, so the index lies flat
HOVER_LIFT = 0.012     # fingertip height above the string while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
REACH_FRAC = 0.50      # fraction of finger length the knuckle sits back
DIST_FLEX_PRESS = 0.45
DIST_FLEX_HOVER = 0.30
DIST_FLEX_BARRE = 0.06         # flattened index while barring
DIST_FLEX_BARRE_HOVER = 0.12
PRESS_STAGGER = 0.013  # fret-slot y spread between fingers sharing a fret

# Per-event wrist rotation freedom, chosen by a grid search that
# penalizes predicted finger-finger collisions (forward kinematics of
# the pressing fingers). Yaw turns the hand in the fretboard plane and
# is locked to 0 while barring (the flattened index must stay parallel
# to the fret wire); roll spins about the reach axis - safe during
# barres - lifting one side of the knuckle line so a far-reaching
# finger arches OVER its neighbour instead of slicing through it.
WRIST_YAW_MAX = 0.56
WRIST_YAW_STEP = 0.08
WRIST_YAW_REG = 0.5    # cost per rad^2 of yaw - prefer a neutral wrist
WRIST_ROLL_MAX = 0.42
WRIST_ROLL_STEP = 0.07
WRIST_ROLL_REG = 0.3
WRIST_COHERE = 2.0     # per rad^2 of pose change from the previous event,
                       # fading out over 1.5 s - stops the wrist flipping
                       # between opposite extremes on back-to-back chords
                       # (the interpolating fingers would cross mid-swing)
TOUCH_CLEAR = 0.012    # axis distance where finger boxes sit flush
COLLIDE_W = 25000.0    # per m^2 of clearance deficit between finger axes
RETARGET_DIST = 0.006  # lateral move beyond which a finger curls up
                       # (relax pose) while travelling to its next spot

# --- pick hand -------------------------------------------------------------
# Two right-hand gestures, chosen per onset by how many strings it strikes:
#   * PICK  (1..STRUM_STRINGS-1 strings): a tight wrist/finger flick that barely
#     clears the strings -- crisp for single notes and small partial chords.
#   * STRUM (>= STRUM_STRINGS strings): the hand arcs clear of the strings and
#     sweeps a windup/follow-through, so the guitarist's arm IK (which copies the
#     PickHand wrist) drives the stroke from the elbow/shoulder across the whole
#     string set, the way a real player strums a full chord. The strum's SIZE
#     tracks velocity (see the SOFT..LOUD bounds below): a quiet strum barely
#     clears the strings and keeps a tight windup, a loud one arcs high and sweeps
#     a wide backswing/follow-through -- so the motion reads as its volume.
# The struck-string count is len(event notes) -- one note per string per onset.
STRUM_STRINGS = 4      # >3 strings in one motion -> strum, else pick
PICK_HOVER = 0.012     # pick tip above the string plane between picked strokes
PICK_DEPTH = 0.004     # pick tip below the string plane while crossing
PICK_LEAD_X = 0.012    # picked windup distance before the first struck string
PICK_FOLLOW_X = 0.010  # picked follow-through past the last struck string
# Strum size scales linearly with velocity between these soft/loud bounds.
STRUM_HOVER_SOFT = 0.014   # quiet strum: hand just skims the strings
STRUM_HOVER_LOUD = 0.075   # loud strum: hand arcs well clear
STRUM_LEAD_SOFT = 0.012    # quiet strum: tight windup into the first string
STRUM_LEAD_LOUD = 0.075    # loud strum: wide arm sweep in
STRUM_FOLLOW_SOFT = 0.010  # quiet strum: little follow-through
STRUM_FOLLOW_LOUD = 0.060  # loud strum: long follow-through past the last string
STRUM_TIME = 0.05      # seconds a chord strum spans around the onset
REVERSAL_GAP = 0.4     # up to this gap (s) between opposite strokes, the hand
                       # reverses at the shared apex instead of re-winding up:
                       # the prior stroke's follow-through already carried it to
                       # this stroke's backswing side, so a fresh (opposite,
                       # farther) windup crammed into the tiny interval would
                       # fling the hand out and snap it back within a frame.

# Stroke direction follows metric ("pendulum") picking: down on the beat,
# up on the off-beat. slot = round(beat / PENDULUM_SUBDIV); even = down,
# odd = up. This gives alternate down-up for even runs, repeated
# downstrokes for on-beat notes, and a clean re-anchor after rests. A
# loud note after a rest is forced to a power downstroke. See RESEARCH.md.
PENDULUM_SUBDIV = None  # beats per half-swing; None = auto (median gap)
ACCENT_VEL = 100       # velocity at/above which a post-rest note down-picks
GAP_RESET = 0.35       # seconds of silence that re-anchor the pendulum

# Velocity dynamics: louder = bigger backswing and a faster strike.
STRIKE_SLOW = 0.12     # apex->contact time for a soft note (slow strike)
STRIKE_FAST = 0.045    # apex->contact time for a loud note (fast strike)


def _barre_press_notes(event):
    """The event's fretted notes as (note, is_barre) press items.

    Barred groups collapse to their bass-most note (a flattened index
    presses every barred string on the way there, so only that one needs
    an IK target). Fingers sharing a fret get their targets staggered
    along the fret slot - the lower finger (whose knuckle sits nut-side)
    presses farther behind the wire, the higher finger right up against
    it - so converging fingers stack diagonally instead of colliding,
    the way real players place e.g. Am's middle/ring or F's ring/pinky.
    Staggered/barre items are copies; originals are never mutated."""
    barred = [n for n in event["notes"] if n["fret"] > 0 and n.get("barre")]
    normal = [n for n in event["notes"]
              if n["fret"] > 0 and not n.get("barre")]
    out = []
    if len(barred) >= 2:
        rep = dict(min(barred, key=lambda n: n["x"]))
        rep["end"] = max(n["end"] for n in barred)
        rep["velocity"] = max(n["velocity"] for n in barred)
        out.append((rep, True))
    else:  # a lone flagged note is just a normal press
        normal += barred
    by_fret = {}
    for n in normal:
        by_fret.setdefault(n["fret"], []).append(n)
    for group in by_fret.values():
        if len(group) == 1:
            out.append((group[0], False))
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
            out.append((rep, False))
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
    finger length back along the rotated rest direction from its
    fingertip target, at the knuckle_z line. Midrange like the piano
    animator, so stretched grips split the residual between their outer
    fingers."""
    rest_dir = rot @ mathutils.Vector((0.0, 1.0, 0.0))
    xs, ys, zs = [], [], []
    for n, _is_barre in press:
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
    """Predicted world joint points [knuckle, prox, mid, tip] of a
    finger posed by the IK at this wrist pose - the same geometry
    _pose_finger will bake, reconstructed for collision checks."""
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
    for n, is_barre in press:
        spec = FRET_FINGERS[n["finger"]]
        ko = rot @ mathutils.Vector(spec["knuckle"])
        delta = mathutils.Vector((n["x"] - (wrist[0] + ko.x),
                                  n["y"] - (wrist[1] + ko.y),
                                  n["z"] - (wrist[2] + ko.z)))
        local = inv @ delta
        cost += math.atan2(local.x, max(local.y, 0.012)) ** 2
        chains.append(_finger_fk(
            wrist, rot, spec, (n["x"], n["y"], n["z"]),
            DIST_FLEX_BARRE if is_barre else DIST_FLEX_PRESS))
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
    neutral wrist, and coherence with the previous event's pose against
    predicted finger-finger collisions. Barre events lock yaw (the bar
    must stay parallel to the fret wire) but keep roll, which spins
    about the bar's own axis."""
    press = _barre_press_notes(event)
    has_barre = any(b for _, b in press)
    knuckle_z = KNUCKLE_Z_BARRE if has_barre else KNUCKLE_Z
    if not press:
        return _solve_wrist(press, _fret_rotation(0, 0), knuckle_z), 0.0, 0.0

    cohere = 0.0
    if prev_pose is not None and dt is not None:
        cohere = WRIST_COHERE * max(0.0, 1.0 - dt / 1.5)
    ysteps = 0 if has_barre else int(round(WRIST_YAW_MAX / WRIST_YAW_STEP))
    rsteps = (int(round(WRIST_ROLL_MAX / WRIST_ROLL_STEP))
              if len(press) >= 2 else 0)
    best = None
    for ri in range(-rsteps, rsteps + 1):
        roll = ri * WRIST_ROLL_STEP
        for yi in range(-ysteps, ysteps + 1):
            yaw = yi * WRIST_YAW_STEP
            rot = _fret_rotation(yaw, roll)
            wrist = _solve_wrist(press, rot, knuckle_z)
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
    # Smooth the wrist rotation with the same weights/segments as the
    # location so they always travel together.
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

    # Fingers: per finger so release tails clamp against the next press.
    per_finger = {}
    for ev, target, rot in zip(events, targets, rots):
        for n, is_barre in _barre_press_notes(ev):
            per_finger.setdefault(n["finger"], []).append(
                (n, target, rot, is_barre))

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    last_frame = frame_start
    for f, items in per_finger.items():
        spec = FRET_FINGERS[f]
        prev_off = None
        for i, (n, target, rot, is_barre) in enumerate(items):
            rmat = _fret_rotation(*rot)
            rinv = rmat.transposed()
            ko = rmat @ mathutils.Vector(spec["knuckle"])
            knuckle = (target[0] + ko.x, target[1] + ko.y, target[2] + ko.z)

            def ik_inputs(tip_z):
                local = rinv @ mathutils.Vector(
                    (n["x"] - knuckle[0], n["y"] - knuckle[1],
                     tip_z - knuckle[2]))
                return local.x, local.y, -local.z

            flex_press = DIST_FLEX_BARRE if is_barre else DIST_FLEX_PRESS
            flex_hover = (DIST_FLEX_BARRE_HOVER if is_barre
                          else DIST_FLEX_HOVER)
            dx, dy, dv = ik_inputs(n["z"])
            pressed = _finger_ik(dx, dy, dv, spec["lengths"], flex_press)
            dx, dy, dv = ik_inputs(n["z"] + HOVER_LIFT)
            hover = _finger_ik(dx, dy, dv, spec["lengths"], flex_hover)

            on_frame = to_frame(n["start"])
            off_frame = max(on_frame, to_frame(n["end"]))
            hover_frame = max(frame_start, on_frame - attack_frames(n))
            if prev_off is not None:
                hover_frame = max(hover_frame, prev_off + 0.5)
            hover_frame = min(hover_frame, on_frame)

            _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                         flex_hover, hover_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         flex_press, on_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         flex_press, off_frame)

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
                    # Curl up while repositioning so the travelling
                    # finger clears pressed neighbours instead of
                    # sweeping through them at hover height.
                    _relax_finger(pbones, f, release_frame)
                else:
                    _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                                 flex_hover, release_frame)
            prev_off = off_frame
            last_frame = max(last_frame, off_frame + release_frames)

    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'SINE'
    return last_frame


def _event_beat(ev):
    """Fractional beat of an event (min note beat), or None if the JSON
    predates the beat field."""
    beats = [n["beat"] for n in ev["notes"] if "beat" in n]
    return min(beats) if beats else None


def _pendulum_subdiv(events):
    """Beats per pendulum half-swing: the median inter-onset gap in beats,
    snapped to a musical grid. This makes an eighth-note passage swing at
    eighths (alternating) and a sixteenth passage at sixteenths, while
    quarter/chordal material lands on even slots (downstrokes)."""
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
    """+1 (down = bass->treble) / -1 (up) per event, by metric picking:
    down on the beat, up on the off-beat, with a loud post-rest note
    forced to a power downstroke.

    Successive strokes closer than ~1.5 subdivisions strictly ALTERNATE
    (pendulum picking): there is no time for the hand to reset between two
    same-direction strokes that close together, so a metric slot that would
    repeat the previous direction is flipped. This keeps fast runs swinging
    down-up-down-up and, crucially, avoids two same-direction strokes a beat
    apart -- which would fling the picking hand back across the strings and
    snap it in a single frame (a same-direction pair recovers on opposite
    apex sides, unlike a reversal, which shares one apex)."""
    subdiv = _pendulum_subdiv(events)
    dirs = []
    prev_t, prev_beat = None, None
    for ev in events:
        beat = _event_beat(ev)
        if beat is None:  # defensive: no metric info -> simple alternate
            direction = 1 if not dirs else -dirs[-1]
        else:
            slot = round(beat / subdiv)
            direction = 1 if slot % 2 == 0 else -1
            if (dirs and prev_beat is not None
                    and beat - prev_beat <= 1.5 * subdiv
                    and direction == dirs[-1]):
                direction = -dirs[-1]      # enforce the pendulum on fast pairs
        gap = None if prev_t is None else ev["t"] - prev_t
        vel = max(n["velocity"] for n in ev["notes"])
        if gap is not None and gap >= GAP_RESET and vel >= ACCENT_VEL:
            direction = 1  # accented attack after a rest -> downstroke
        dirs.append(direction)
        prev_t, prev_beat = ev["t"], beat
    return dirs


def animate_pick_hand(arm_obj, notes, fps, frame_start):
    """Keyframe the picking hand's strokes. Returns the last frame."""
    arm_obj.animation_data_clear()
    tip_off = pick_world_offset(PICK_TIP_LOCAL)
    obj_y = fret_layout.PLUCK_Y - tip_off[1]
    z_pluck = fret_layout.STRING_Z - PICK_DEPTH - tip_off[2]

    def hover_z(lift):  # pick-tip clearance -> object z for a hover apex
        return fret_layout.STRING_Z + lift - tip_off[2]

    def key(t, tip_x, z):
        arm_obj.location = (tip_x - tip_off[0], obj_y, z)
        arm_obj.keyframe_insert(data_path="location",
                                frame=frame_start + t * fps)

    events = _group_events(notes)
    if not events:
        return frame_start

    directions = _pick_directions(events)
    min_dt = 0.75 / fps  # keep successive keyframes distinct
    last_t = None

    def key_after(t, tip_x, z):
        nonlocal last_t
        if last_t is not None:
            t = max(t, last_t + min_dt)
        key(t, tip_x, z)
        last_t = t
        return t

    prev_t, prev_dir = None, None
    last_frame = frame_start
    for ev, direction in zip(events, directions):
        t = ev["t"]
        gap = t - prev_t if prev_t is not None else None
        xs = sorted(n["pluck_x"] for n in ev["notes"])
        first, last = (xs[0], xs[-1]) if direction > 0 else (xs[-1], xs[0])
        n_strings = len(ev["notes"])
        chord = n_strings > 1
        strum = n_strings >= STRUM_STRINGS

        # Velocity 0..1 -- louder = bigger, faster stroke.
        vel_norm = max(0.0, min(1.0, max(n["velocity"]
                                         for n in ev["notes"]) / 127.0))

        # A strum's size scales with velocity: a soft strum barely clears the
        # strings with a tight windup, a loud one arcs high and sweeps a wide
        # backswing/follow-through (the arm IK follows into the elbow), so the
        # gesture reads as its volume. A pick stays a tight, low flick. z_hover
        # is the apex the hand rises to on either side of the stroke.
        if strum:
            z_hover = hover_z(STRUM_HOVER_SOFT + vel_norm
                              * (STRUM_HOVER_LOUD - STRUM_HOVER_SOFT))
            lead = STRUM_LEAD_SOFT + vel_norm * (STRUM_LEAD_LOUD - STRUM_LEAD_SOFT)
            follow = (STRUM_FOLLOW_SOFT + vel_norm
                      * (STRUM_FOLLOW_LOUD - STRUM_FOLLOW_SOFT))
        else:
            z_hover = hover_z(PICK_HOVER)
            lead = PICK_LEAD_X * (0.6 + 1.3 * vel_norm)
            follow = PICK_FOLLOW_X * (0.6 + 0.8 * vel_norm)
        strike = STRIKE_SLOW - vel_norm * (STRIKE_SLOW - STRIKE_FAST)
        # A wider strum drags across more strings, so let it span more time
        # (scaled by string count) instead of snapping through the set.
        cross_lag = (STRUM_TIME * n_strings / 3.0 if strum
                     else STRUM_TIME if chord else 0.015)

        # Backswing apex, held above the strings, then the strike drops
        # to the string and crosses; apex->contact time = `strike`. On a quick
        # direction reversal the previous stroke's follow-through already left
        # the hand at this stroke's backswing apex (the pendulum reverses at a
        # shared apex), so skip the fresh windup -- emitting it would fling the
        # hand out and snap it back within a frame (see REVERSAL_GAP).
        reversal = prev_dir is not None and direction != prev_dir
        if not (reversal and gap is not None and gap <= REVERSAL_GAP):
            windup = t - strike - (min(0.06, 0.4 * gap)
                                   if gap is not None else 0.06)
            key_after(windup, first - direction * lead, z_hover)
        key_after(t - 0.5 * strike, first - direction * lead * 0.4, z_pluck)
        key_after(t + cross_lag, last + direction * follow * 0.5, z_pluck)
        rise_t = key_after(t + cross_lag + 0.06,
                           last + direction * follow, z_hover)
        last_frame = max(last_frame, frame_start + rise_t * fps)
        prev_t, prev_dir = t, direction

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
