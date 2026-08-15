"""Animates the Hand_L / Hand_R rigs from a fingering.json timeline.

Consumes the output of ``python -m piano.fingering`` (per-note hand, finger
and fingertip target) and keyframes the armatures built by build_hands.py:

  - The armature object's location carries the wrist: for every chord event
    it is placed so the pressing fingers' knuckles sit over their keys
    (arriving slightly early, gliding between events with SINE easing),
    then nudged by _wrist_fit to a placement its fingers can hold within
    their range of motion - the hand moves so the fingers need not contort.
  - Finger bones are driven by closed-form two-link IK in the vertical
    plane through the knuckle: the proximal bone pitches, the middle joint
    flexes, the proximal z-rotation supplies sideways reach (capped at the
    knuckle's anatomical abduction), and the distal phalanx keeps a fixed
    natural flexion. No IK constraints are used, so the result is plain
    baked FK keyframes - every one of which lands inside the joint cage
    build_hands.py puts on the bones.
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
MAX_YAW = 1.0          # radians of sideways finger reach (generic fallback;
                       # the piano's own call sites pass the anatomical caps
                       # below, and the guitar/bass rigs that import
                       # _finger_ik clamp its yaw themselves)
DIST_FLEX_PRESS = 0.35
DIST_FLEX_HOVER = 0.26
RELAXED = (0.45, 0.70, 0.40)  # prox/mid/dist downward flexion at rest

# --- anatomical sideways reach (MCP abduction / CMC abduction) --------------
# A finger reaches a key that is not straight ahead of its knuckle by deviating
# sideways at the knuckle, and real MCP abduction/adduction tops out around
# 25 deg (Thieme 2024; AAOS goniometry) - the same limit build_hands cages the
# bones to, and the same one the guitar/bass fret hands enforce. Left to the old
# 57 deg MAX_YAW the IK splayed fingers up to 48 deg on the demo, which now that
# the cage exists would simply be clamped by the constraint (fingertip sliding
# off its key) and in any case reads as a finger swinging under its neighbours.
# So the cap lives in the IK, and _splay_clamp_x below moves the WRIST so the
# finger rarely has to reach that far - which is what a pianist actually does,
# gliding the hand along the keyboard rather than fanning the fingers.
# The thumb is not a finger: its CMC saddle joint carries 45-60 deg of palmar/
# radial abduction, and that wide swing is exactly what a thumb-under in a scale
# run is made of, so it gets its own (much larger) cap.
FINGER_MCP_SPLAY = math.radians(26.0)
THUMB_CMC_SPLAY = math.radians(45.0)

# Black keys sit 12 mm above the white keybed and 55 mm further from the player,
# so a hand held at the white-key hover height has barely any drop left from
# knuckle to fingertip and the fingers have to fold BACKWARD (MCP
# hyperextension, ~39 deg on the demo's black-key scale - past the ~30 deg a
# knuckle really has) to stay on the keys. Pianists meet the black keys with a
# slightly higher, flatter hand instead; lifting the wrist over any event that
# touches one restores a natural downward arch. The lift is smoothed with the
# rest of the wrist target, so the hand rises into a black-key passage and
# settles again after it.
BLACK_KEY_LIFT = 0.018

# The knuckle-to-target distance the splay geometry treats as the minimum, so a
# target level with (or behind) the knuckle cannot demand an infinite reach;
# matches the clamp inside _finger_ik.
MIN_REACH_Y = 0.012

# How far out along its own length each finger is expected to reach for its key,
# which is what sets where the wrist sits behind the keys. At 1.0 the finger
# would have to be straight; a pianist's is arched, its fingertip a good way
# short of full extension. This is the piano's version of the guitar hand's
# REACH_FRAC, and the same trade-off applies: too small and the fingers fold up
# under themselves (bowing the knuckles backward and demanding big sideways
# deviations for small offsets), too large and they flatten out.
REACH_FRAC = 0.55

# --- wrist fit ---------------------------------------------------------------
# Bounds and weights for the little per-event search that settles where the
# wrist sits (see _wrist_fit). The steps are ~2 mm, finer than the hand's
# placement error matters, and the ranges are small: this trims the smoothed
# glide, it does not re-plan it.
MIN_WRIST_Z = 0.045    # floor for the wrist drop: below this the palm would be
                       # down among the keys
WRIST_Z_STEP = 0.002
WRIST_Y_RANGE = 0.016  # how far in/out from the smoothed target it may look
WRIST_Y_STEP = 0.002
REACH_W = 4.0e6        # per m^2 a pressing fingertip lands off its key
HYPEREXT_W = 3.0e3     # per rad^2 a knuckle is driven past its extension range
                       # (~1 mm of miss weighs about the same as 4 deg of bend)
WRIST_REG = 2.0e4      # per m^2 of departure from the smoothed target: keeps
                       # the hand on its glide wherever the fingers are happy

# A knuckle bends BACKWARD only so far: ~25-30 deg for the fingers (the little
# finger the most), and the thumb's CMC a similar amount in this rig's frame,
# where the metacarpal also carries the thumb column's palmar rotation. Beyond
# it the finger is bowing back over the key instead of arching onto it. Used by
# the wrist-height search; the build-time cage (build_hands.FINGER_ROT_LIMIT)
# sits a few degrees looser so the two never fight.
FINGER_MCP_HYPEREXT = math.radians(28.0)
THUMB_CMC_HYPEREXT = math.radians(25.0)

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


def _splay_cap(finger):
    """How far sideways this digit's knuckle may deviate (radians)."""
    return THUMB_CMC_SPLAY if finger == 1 else FINGER_MCP_SPLAY


def _hyperext_cap(finger):
    """How far this digit's knuckle may bend backward (radians)."""
    return THUMB_CMC_HYPEREXT if finger == 1 else FINGER_MCP_HYPEREXT


def _finger_ik(dx, dy, dv, lengths, dist_flex, splay_cap=MAX_YAW):
    """Closed-form 2-link IK for one finger with a rigidly flexed distal.

    dx/dy: fingertip target offset from the knuckle in the keyboard plane;
    dv: drop from knuckle to target (positive down); dist_flex: the fixed
    distal flexion the pose will use; splay_cap: the knuckle's anatomical
    sideways limit (the caller's business - the guitar/bass fret hands clamp
    the returned yaw themselves, so the default is the old loose value).
    The mid+distal pair is treated as one link along the elbow-to-tip chord
    (length b, hanging gamma below the mid bone), which makes the fingertip
    land exactly on the target.
    Returns (yaw, prox_pitch, mid_flex), pitch/flex positive = down.
    """
    a = lengths[0]
    l2, l3 = lengths[1], lengths[2]
    b = math.hypot(l2 + l3 * math.cos(dist_flex), l3 * math.sin(dist_flex))
    gamma = math.atan2(l3 * math.sin(dist_flex), l2 + l3 * math.cos(dist_flex))
    yaw = max(-splay_cap,
              min(splay_cap, math.atan2(dx, max(dy, MIN_REACH_Y))))
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
        reach = REACH_FRAC * sum(spec["lengths"])
        xs.append(n["x"] - kx * mirror)
        ys.append(_target_y(n, has_black) - (ky + reach))
    z = HOVER_Z + (BLACK_KEY_LIFT if has_black else 0.0)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, z)


def _splay_clamp_x(event, tgt, mirror):
    """Slide one wrist target sideways until every pressing finger can reach
    its key within the knuckle's anatomical splay (see FINGER_MCP_SPLAY).

    Smoothing the wrist path is what gives the hand its glide - the fingers
    reach out of a gliding wrist rather than the wrist hopping key to key - but
    left unbounded it asks a finger to deviate 40-50 deg sideways, which no
    knuckle does (and which the build-time LIMIT_ROTATION cage would clamp,
    sliding the fingertip off its key). Each pressing finger admits a window of
    wrist x whose width is tan(cap) * its knuckle-to-key distance; the target is
    clamped into the intersection, so the glide survives wherever the fingers
    can absorb it and the WRIST does the rest of the reaching. A chord too wide
    for any one window (empty intersection) takes the window midpoint, splitting
    the residual between its outer fingers - the same compromise
    _event_root_target makes with its midrange.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    lo, hi = float("-inf"), float("inf")
    for n in event["notes"]:
        kx, ky, _kz = FINGERS[n["finger"]]["knuckle"]
        dy = max(_target_y(n, has_black) - (tgt[1] + ky), MIN_REACH_Y)
        span = math.tan(_splay_cap(n["finger"])) * dy
        centre = n["x"] - kx * mirror
        lo = max(lo, centre - span)
        hi = min(hi, centre + span)
    x = (lo + hi) / 2.0 if lo > hi else max(lo, min(hi, tgt[0]))
    return (x, tgt[1], tgt[2])


def _tip_miss(dx, dy, dv, lengths, dist_flex, splay_cap):
    """How far the posed fingertip ends up from the key it was asked for (m).

    Zero for anything the finger can actually do; positive once the IK's own
    limits bite - the chain too short for the target, or the knuckle pinned at
    its splay cap with the key still further sideways.
    """
    yaw, prox, mid = _finger_ik(dx, dy, dv, lengths, dist_flex, splay_cap)
    l1, l2, l3 = lengths
    p1, p2, p3 = prox, prox + mid, prox + mid + dist_flex
    h = l1 * math.cos(p1) + l2 * math.cos(p2) + l3 * math.cos(p3)
    v = l1 * math.sin(p1) + l2 * math.sin(p2) + l3 * math.sin(p3)
    return math.dist((h * math.sin(yaw), h * math.cos(yaw), v), (dx, dy, dv))


def _event_pose_cost(event, tgt, mirror, press_white, press_black):
    """How badly a wrist placement serves this event's pressing fingers.

    Two things go wrong, and they pull in opposite directions:

      * hold the hand too far from the keys and a finger cannot reach - the
        chain runs out of length, or the knuckle hits its sideways limit, and
        the fingertip is left hanging off its key (_tip_miss);
      * hold it too close and the finger has nowhere to drop to, so it folds
        BACKWARD at the knuckle (MCP hyperextension) to stay on the key - the
        bowed-back knuckle the joint cage forbids.

    Both are scored against the anatomy and squared, so a stretched chord
    settles by splitting the difference between its fingers rather than pinning
    one at its limit. The miss is weighted the heavier: a fingertip visibly off
    its key reads worse than a few degrees of knuckle. Both the press and the
    hover pose are scored, since the hand holds one wrist position for both.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    cost = 0.0
    for n in event["notes"]:
        spec = FINGERS[n["finger"]]
        kx, ky, kz = spec["knuckle"]
        press_z = _press_z(n["is_black"], press_white, press_black)
        dx = n["x"] - (tgt[0] + kx * mirror)
        dy = _target_y(n, has_black) - (tgt[1] + ky)
        cap = _splay_cap(n["finger"])
        for dv, flex in ((tgt[2] + kz - press_z, DIST_FLEX_PRESS),
                         (tgt[2] + kz - (press_z + HOVER_LIFT),
                          DIST_FLEX_HOVER)):
            cost += REACH_W * _tip_miss(dx, dy, dv, spec["lengths"],
                                        flex, cap) ** 2
            _yaw, prox, _mid = _finger_ik(dx, dy, dv, spec["lengths"], flex,
                                          cap)
            back = -prox - _hyperext_cap(n["finger"])
            cost += HYPEREXT_W * max(back, 0.0) ** 2
    return cost


def _wrist_fit(events, targets, mirror, press_white, press_black):
    """Nudge each smoothed wrist target to a placement the fingers can hold.

    The smoothed path says where the hand would LIKE to be; this asks what its
    fingers can actually do there, and if the answer is "not this" it searches a
    small grid around it - in and out from the keys (y) and up and down (z),
    with the sideways clamp re-applied at each candidate since moving in or out
    changes how far the fingers can splay - for the placement that best serves
    the whole event (_event_pose_cost). Moving the hand rather than contorting
    the fingers is the whole point, and it is what a pianist does: dropping and
    reaching in for a stretched voicing, riding higher and flatter over the
    black keys. A regularizer keeps the hand on its smoothed glide wherever the
    fingers are comfortable, so this only bites where the geometry is tight -
    ordinary single notes and close chords score zero and do not move at all.
    """
    out = []
    for ev, tgt in zip(events, targets):
        base = _splay_clamp_x(ev, tgt, mirror)
        best = (_event_pose_cost(ev, base, mirror, press_white, press_black),
                base)
        if best[0] > 0.0:
            steps = int(WRIST_Y_RANGE / WRIST_Y_STEP)
            for i in range(-steps, steps + 1):
                dy = i * WRIST_Y_STEP
                z = tgt[2]
                while z >= MIN_WRIST_Z - 1e-9:
                    cand = _splay_clamp_x(ev, (tgt[0], tgt[1] + dy, z), mirror)
                    cost = (_event_pose_cost(ev, cand, mirror, press_white,
                                             press_black)
                            + WRIST_REG * (dy * dy + (tgt[2] - z) ** 2))
                    if cost < best[0]:
                        best = (cost, cand)
                    z -= WRIST_Z_STEP
        out.append(best[1])
    return out


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
    # Wrist path: per-event ideal placement, smoothed into a glide, then pulled
    # back inside what the fingers can anatomically do (sideways splay, then
    # reach) - the hand moves so the fingers do not have to contort.
    targets = _wrist_fit(events, _smooth_targets(
        events, [_event_root_target(ev, mirror) for ev in events]),
        mirror, press_depth_white, press_depth_black)

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

            cap = _splay_cap(f)
            pressed = _finger_ik(dx, dy, knuckle[2] - press_z,
                                 spec["lengths"], DIST_FLEX_PRESS, cap)
            hover = _finger_ik(dx, dy, knuckle[2] - (press_z + HOVER_LIFT),
                               spec["lengths"], DIST_FLEX_HOVER, cap)

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
