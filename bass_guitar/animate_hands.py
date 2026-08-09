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
                              PICK_HAND_ROT, pick_world_offset)
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
                             PICK_HAND_ROT, pick_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger,
                                     _relax_finger)


# --- fret hand -------------------------------------------------------------
KNUCKLE_Z = 0.058      # knuckle-line height while fretting (arch clearance)
HOVER_LIFT = 0.012     # fingertip height above the string while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
# A big neck position shift is spread over the WHOLE rest before it, not lunged
# in the last few frames with the hand then sitting parked. The larger the shift,
# the later the wrist is allowed to arrive (down to MIN_ARRIVE_LEAD) and the
# smaller its "arrive at least 60% through the gap" floor, so the long glide up
# the neck uses every frame the previous note's release leaves it - the way a
# player eases into a shift in anticipation instead of snapping.
MIN_ARRIVE_LEAD = 0.06   # s: least lead before a press, even for a big shift
SHIFT_SPREAD_MIN = 0.04  # m: shifts below this aren't spread (a normal reach)
SHIFT_SPREAD_FULL = 0.12  # m: shifts this large get the full spread
SHIFT_FLOOR_FRAC = 0.12  # the gap-fraction floor on arrival for a full shift
REACH_FRAC = 0.50      # fraction of finger length the knuckle sits back
DIST_FLEX_PRESS = 0.45
DIST_FLEX_HOVER = 0.30
PRESS_STAGGER = 0.014  # fret-slot y spread between fingers sharing a fret
RELAX_LIFT = 2.5       # frames to lift a finger off a note back to its idle hover
# Anticipation: a real fretting hand does not hold a finger still until the last
# instant and then snap it onto the next note - it starts repositioning the
# finger (and wrist) in advance and eases it into place. These give every finger
# transition a distance-independent minimum runway so the pose change spreads
# over several eased frames instead of one, killing the acceleration spikes.
SETTLE_FR = 1.0        # frames a finger holds the hover just before it presses
REACH_LEAD_FR = 4.0    # frames a finger glides from its idle lane into the hover
CROSS_GLIDE_FR = 4.0   # frames a finger takes crossing straight from one press
                       # to the next (it lifts off the old fret this early)

# An idle (non-pressing) fret finger holds a RELAXED ARCH in its own fret lane,
# NOT a reach to a hover point. The knuckles ride the treble edge spread ALONG
# the neck (one per fret, index nut-side, pinky bridge-side), so the fingers are
# already in their own lanes; an idle finger only crosses a neighbour if it
# reaches ACROSS or FORWARD out of that lane.
#
# The pose is authored DIRECTLY as gentle joint angles rather than solved by IK
# to a target near the strings. Solving to a target is what produced the tense
# "claw": a ~90 mm finger told to touch a point only ~35 mm from its own knuckle
# (just over the strings at its own fret) can only get there by clenching the PIP
# to ~130 deg. Extending the finger to un-clench it - reaching toward the nut -
# fixed the curl but made the fingertip drape over pressing neighbours AND forced
# the wrist search to yaw the whole hand ~30 deg to keep it clear, over-bending
# the fret wrist past its ROM. Holding a fixed arch instead keeps the fingertip
# in its own lane (past the knuckle, not curled under the palm), so it neither
# reaches across a neighbour nor needs any wrist yaw to make room.
#
# The MCP is tilted DOWN at the knuckle (rather than lifted back) so the whole
# arch points toward the strings: with a small positive tilt the idle fingertip
# rests ~6 mm above the string plane (instead of ~45 mm lifted back, or ~14 mm
# with the arch merely level), so the hand sits READY over the strings and a
# press only glides down a hair instead of dropping from high like a hammer-on.
# The prior 0.0 (arch level) left the fingertips floating ~14 mm up at the start
# frame near the nut, where no wrist roll tilts them lower; nudging the MCP down
# parks the average reach among the strings. It only tilts the arch as a rigid
# unit - the PIP/distal curl (the arch SHAPE) is unchanged - so the tip stays in
# its own lane, still safely clear of the strings, free to move without fretting.
IDLE_PROX = 0.10           # relaxed idle MCP (proximal) angle: aimed at the board
IDLE_MID = 1.26            # relaxed idle PIP (middle) flex - a natural ~72 deg arch
IDLE_DIST_FLEX = 0.26      # distal curl of an idle finger
IDLE_CLEAR_TARGET = 0.020  # clearance (m) from a pressing finger that counts as clear
# When a neighbour presses close by, the idle finger gets out of the way two ways,
# and the search below picks the cheapest mix that clears the presser:
#  - it LEANS ASIDE at the knuckle (a small sympathetic MCP yaw), the way a real
#    hand's fingers drift together when one abducts rather than one finger swinging
#    across a still neighbour, and
#  - it CURLS BACK - tightening the PIP to draw the fingertip toward the palm
#    within its own lane.
# Among poses that clear, the one closest to the relaxed neutral (no yaw, loosest
# arch) wins; leaning is cheaper than a hard curl, so a crowded finger mostly
# slides aside and only clenches if leaning alone can't open the gap.
_IDLE_MID_GRID = (IDLE_MID, 1.6, 1.9, 2.2, 2.5)
IDLE_SYMPATHY_MAX = math.radians(16.0)  # farthest an idle finger leans out of the way
_IDLE_YAW_GRID = tuple(i * IDLE_SYMPATHY_MAX / 3.0 for i in range(-3, 4))
IDLE_YAW_COST = 0.030     # score penalty per rad of sympathetic lean
IDLE_CURL_COST = 0.012    # score penalty per rad of extra PIP curl


def _idle_finger_pose(f, target, rot, press_chains=()):
    """(yaw, prox, mid) for a relaxed idle finger arched over its own fret lane.

    Holds IDLE_PROX/IDLE_MID (a loose ~72 deg arch, no sideways yaw) by default.
    When ``press_chains`` (the world joint chains of the fingers pressing this
    event) come within IDLE_CLEAR_TARGET, the finger searches a small grid of
    sympathetic knuckle LEANS (_IDLE_YAW_GRID) and PIP CURLS (_IDLE_MID_GRID) and
    takes the pose that clears the presser closest to the relaxed neutral - so it
    drifts aside (and curls a little) instead of a neighbour swinging across it.
    Among poses that clear, the least-deviating one wins."""
    rmat = _fret_rotation(*rot)
    spec = FRET_FINGERS[f]
    if not press_chains:
        return 0.0, IDLE_PROX, IDLE_MID
    best = None
    for yaw in _IDLE_YAW_GRID:
        for mid in _IDLE_MID_GRID:
            chain = _finger_fk_from_angles(target, rmat, spec, yaw, IDLE_PROX,
                                           mid, IDLE_DIST_FLEX)
            clr = min(_seg_dist(chain[k], chain[k + 1], pc[m], pc[m + 1])
                      for pc in press_chains
                      for k in range(3) for m in range(3))
            # Clear the presser up to the target; once clear, sit as close to the
            # relaxed neutral (no lean, loosest arch) as possible.
            score = (min(clr, IDLE_CLEAR_TARGET)
                     - IDLE_YAW_COST * abs(yaw)
                     - IDLE_CURL_COST * (mid - IDLE_MID))
            if best is None or score > best[0]:
                best = (score, yaw, mid)
    return best[1], IDLE_PROX, best[2]

# Per-event wrist rotation freedom, chosen by a collision-penalizing grid
# search (forward kinematics of the pressing fingers). Yaw turns the hand
# in the fretboard plane; roll spins about the reach axis, arching a
# far-reaching finger over its neighbour instead of through it.
WRIST_YAW_MAX = 0.56
WRIST_YAW_STEP = 0.08
WRIST_YAW_REG = 0.5
# The fretting forearm reaches the neck from a roughly fixed elbow by the body, so
# a hand held STRICTLY perpendicular to the neck has to bend the wrist ever more
# sharply toward either end of the fretboard: measured against the actual mounted
# arm, the wrist sits ~58 deg off the forearm at a comfortable middle position
# (~fret 8) but climbs past 80-90 deg reaching the nut, and back up toward the top
# frets. A real player relieves that by letting the whole hand ROTATE to follow the
# forearm, trading wrist deviation for a little yaw. So the wrist search relaxes
# toward a neck-position-dependent NEUTRAL yaw instead of toward a fixed square-on
# zero: ~0 near NEUTRAL_YAW_Y (where the wrist is naturally straight), yawing
# NEGATIVE reaching toward the nut (higher flat-frame y) and POSITIVE reaching
# toward the high frets. It follows only a FRACTION of the full forearm angle (the
# bend-minimizing yaw wants >=0.5 rad at the extremes) and is capped, so the hand
# still reads as roughly square to the neck -- just no longer robotically so, and it
# stops bending the wrist hardest exactly where the reach is already longest.
NEUTRAL_YAW_Y = 0.45       # flat-frame neck y where the fret wrist sits naturally straight
NEUTRAL_YAW_SLOPE = 0.65   # rad of neutral yaw per metre the hand reaches off NEUTRAL_YAW_Y
NEUTRAL_YAW_MAX = 0.28     # cap (rad, ~16 deg) that keeps the rotation "a little"
# Reaching HIGH on the neck (toward the nut / headstock), a relaxed player lets the
# wrist hang lower and TRAIL behind the reaching fingers instead of holding it up
# level with the neck. The trail is applied along the finger REACH AXIS (see
# _fret_event_pose), so it re-extends the fingers to keep the fingertips on their
# frets while the wrist drops and backs off -- rather than a flat-Y slide, which
# would deviate the knuckle sideways of its target and force a spurious MCP splay.
# Applied only above WRIST_RELAX_Y (the same comfortable middle the yaw neutral
# uses) and only a little.
WRIST_RELAX_Y = NEUTRAL_YAW_Y   # start relaxing the wrist down above this neck position
WRIST_RELAX_DROP = 0.09         # metres the wrist trails (-Y, i.e. down) per metre above it
WRIST_RELAX_MAX = 0.034         # cap on the trail/drop (m), keeping it slight
WRIST_ROLL_MAX = 0.42
WRIST_ROLL_STEP = 0.07
WRIST_ROLL_REG = 0.3
WRIST_COHERE = 2.0     # per rad^2 of pose change from the previous event
TOUCH_CLEAR = 0.013    # axis distance where finger boxes sit flush
COLLIDE_W = 25000.0    # per m^2 of clearance deficit between finger axes
# The wrist search also gives the IDLE fingers room: it penalizes rotations that
# park a pressing finger's box close to where an idle finger's default hover
# sits (their proximals near the shared treble edge - the residual index/ring
# contact at the octave). Softer than the press-press term so it only nudges the
# wrist when a real gap is available, and the per-finger idle search still does
# the fine clearing on top.
WRIST_IDLE_CLEAR = 0.015   # idle-vs-press clearance the wrist search aims to open
IDLE_COLLIDE_W = 9000.0    # weight of the idle-clearance term (< COLLIDE_W)
# Biomechanical anti-crossing. A finger crossing UNDER a neighbour (the index
# reaching under the idle middle to the bass string at the octave) is, at root, a
# finger deviating too far SIDEWAYS at the knuckle: real MCP abduction/adduction
# tops out around 25 deg (Thieme 2024; AAOS goniometry). So the finger IK's yaw is
# hard-capped at FINGER_MCP_SPLAY, and the wrist search PENALIZES any pose that
# would demand more splay than that from a pressing finger - which pushes the hand
# (wrist yaw/roll and slide) to do the reaching instead, exactly as a real player
# angles the whole hand for an octave grip rather than swinging one finger under
# the others. (This replaces an earlier knuckle-x "overshoot" term that, because it
# ignored how the fingers spread ALONG the neck, fired on the natural bunched-
# knuckle pose and drove the hand into a ~57 deg over-splayed diagonal instead.)
# A second cap forbids MCP hyperextension past its ~25-30 deg norm, so a finger
# stretching for a far string bends at the knuckle rather than bowing backward.
FINGER_MCP_SPLAY = math.radians(26.0)    # max abduction/adduction of a fret finger
FINGER_MCP_HYPEREXT = math.radians(28.0)  # max backward (extension) tilt at the MCP
SPLAY_W = 8000.0            # per rad^2 a presser's target demands beyond the splay cap
HYPEREXT_W = 3000.0         # per rad^2 a presser's MCP is driven past its extension cap

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
# The pick stroke is a small wrist pendulum (see animate_pick_hand). Bass is
# picked, not strummed, so the swing is DAMPENED and stays close to the
# strings, and loudness drives the STRIKE SPEED (how fast the wrist dips onto
# the string) rather than how far it reaches.
PICK_SWING = 0.38      # half-swing of the wrist pendulum (rad): the pick clears
                       # the strings by ~2 mm at the top of the swing and its
                       # travel stays INSIDE the 18 mm string spacing, so it ticks
                       # one string without raking its neighbours.
PICK_DEPTH = 0.0018    # tip dip below the string centre at the bottom of the stroke
PICK_FOLLOW_FRAC = 0.6  # how far the wrist swings out after the final note
# The bare neck-axis swing rolls the palm (pronation/supination) as much as it rocks the
# wrist. PICK_DEVIATE tilts the swing axis OFF the forearm/finger axis (which is the part
# that rolls the palm) toward pure ulnar/radial DEVIATION, so the wrist rocks side-to-side
# and the palm holds a steadier down-facing angle -- classic wrist picking. 0 = the old
# pure neck-axis swing (palm rolls, no along-string drift); 1 = no palm roll but the pick
# arcs far along its own string. 0.5 halves the roll for a ~20 mm along-string arc.
PICK_DEVIATE = 0.5
ACCENT_VEL = 100        # a note this loud after a rest restarts on a downstroke
GAP_RESET = 0.35        # ...if the rest before it is at least this long (s)
STRIKE_SLOW = 0.12
STRIKE_FAST = 0.045
# The strike is not pure wrist: the ARM joins in. The hand winds up a little
# OPPOSITE the stroke just before contact and follows THROUGH in the stroke
# direction just after (object translation across the strings -> the wrist head
# moves -> the forearm/elbow follow). Amplitude scales with velocity, so soft
# notes are almost pure wrist and loud/fast attacks throw more arm in. The strike
# key itself sits exactly on the struck string, so contact stays precise.
ARM_WIND = 0.007        # max wind-up across the strings at full velocity (m)
ARM_FOLLOW = 0.012      # max follow-through in the stroke direction (m)
ARM_FOLLOW_T = 0.05     # s after the strike the follow-through peaks


# ===========================================================================
# Fret hand (ported from the guitar animator, barre logic removed)
# ===========================================================================

def _fret_ik(dx, dy, dv, lengths, dist_flex):
    """Fret-finger IK: the piano's closed-form two-link solve, but with the
    knuckle yaw (MCP abduction) clamped to the anatomical FINGER_MCP_SPLAY
    instead of the piano's looser MAX_YAW. Keeping the cap here (rather than
    mutating the shared piano constant) means every predicted and keyed fret
    pose reaches its string within a realistic sideways deviation; the wrist
    search below is what actually moves the hand so the finger does not HAVE to
    exceed it. Returns (yaw, prox, mid) like _finger_ik."""
    yaw, prox, mid = _finger_ik(dx, dy, dv, lengths, dist_flex)
    return (max(-FINGER_MCP_SPLAY, min(FINGER_MCP_SPLAY, yaw)), prox, mid)


def _splay_demand(dx, dy):
    """The knuckle yaw the raw IK WANTS for this in-plane target, before the
    FINGER_MCP_SPLAY clamp - i.e. how far the finger would have to deviate
    sideways. The wrist search penalizes the part of this beyond the cap."""
    return abs(math.atan2(dx, max(dy, 0.012)))


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


def _neutral_yaw(wrist_y):
    """The wrist yaw the search relaxes toward at this neck position: zero at
    NEUTRAL_YAW_Y and tilting to follow the forearm as the hand reaches toward
    either end of the neck, so the wrist deviates less sharply there (see the
    NEUTRAL_YAW_* notes). Capped so the hand only rotates a little."""
    ny = -NEUTRAL_YAW_SLOPE * (wrist_y - NEUTRAL_YAW_Y)
    return max(-NEUTRAL_YAW_MAX, min(NEUTRAL_YAW_MAX, ny))


def _wrist_relax_drop(wrist_y):
    """How far to trail the wrist back along the finger reach axis so it hangs
    lower and more relaxed reaching high on the neck (the direction is applied in
    _fret_event_pose). Zero at/below WRIST_RELAX_Y, growing (capped) toward the
    nut. See the WRIST_RELAX_* notes."""
    return min(WRIST_RELAX_MAX,
               WRIST_RELAX_DROP * max(0.0, wrist_y - WRIST_RELAX_Y))


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


def _finger_fk_from_angles(wrist, rot, spec, yaw, prox, mid, flex):
    """World joint points [knuckle, prox, mid, tip] of a finger posed at the
    given joint angles (curl chain prox->mid->distal, sideways yaw)."""
    sy, cy = math.sin(yaw), math.cos(yaw)
    pts = [mathutils.Vector(wrist) + rot @ mathutils.Vector(spec["knuckle"])]
    p = mathutils.Vector(spec["knuckle"])
    pitch = prox
    for length, dflex in zip(spec["lengths"], (0.0, mid, flex)):
        pitch += dflex
        p = p + length * mathutils.Vector(
            (sy * math.cos(pitch), cy * math.cos(pitch), -math.sin(pitch)))
        pts.append(mathutils.Vector(wrist) + rot @ p)
    return pts


def _finger_fk(wrist, rot, spec, tip, flex):
    """Predicted world joint points [knuckle, prox, mid, tip] of a finger
    posed by the IK to reach ``tip`` at this wrist pose."""
    ko = rot @ mathutils.Vector(spec["knuckle"])
    knuckle = mathutils.Vector(wrist) + ko
    local = rot.transposed() @ (mathutils.Vector(tip) - knuckle)
    yaw, prox, mid = _fret_ik(local.x, local.y, -local.z,
                              spec["lengths"], flex)
    return _finger_fk_from_angles(wrist, rot, spec, yaw, prox, mid, flex)


def _pose_cost(press, wrist, rot, idle=()):
    """IK strain, biomechanical-limit penalties, and predicted finger-collision
    penalty of one pose. ``idle`` are the non-pressing fingers, whose default
    hover is kept clear of the pressing fingers too (a softer penalty than
    press-press)."""
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
        # Gentle strain (prefer little sideways reach) plus the HARD anatomical
        # caps: a target the finger could only reach by splaying past
        # FINGER_MCP_SPLAY would cross under/over its neighbours, and one it
        # could only reach by bending back past FINGER_MCP_HYPEREXT would stretch
        # unnaturally. Penalizing both here steers the wrist search to a pose
        # (yaw/roll/slide) where the finger reaches its string within limits.
        demand = _splay_demand(local.x, local.y)
        cost += demand * demand
        over = demand - FINGER_MCP_SPLAY
        if over > 0.0:
            cost += SPLAY_W * over * over
        _, prox, _ = _fret_ik(local.x, local.y, -local.z,
                              spec["lengths"], DIST_FLEX_PRESS)
        hyper = -prox - FINGER_MCP_HYPEREXT   # prox < 0 == MCP bending backward
        if hyper > 0.0:
            cost += HYPEREXT_W * hyper * hyper
        chains.append(_finger_fk(wrist, rot, spec,
                                 (n["x"], n["y"], n["z"]), DIST_FLEX_PRESS))
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            dmin = min(_seg_dist(chains[i][k], chains[i][k + 1],
                                 chains[j][m], chains[j][m + 1])
                       for k in range(3) for m in range(3))
            if dmin < TOUCH_CLEAR:
                cost += COLLIDE_W * (TOUCH_CLEAR - dmin) ** 2
    for f in idle:
        spec = FRET_FINGERS[f]
        # The idle finger's actual relaxed arch (see _idle_finger_pose), so the
        # wrist search reads where it truly sits - in its own lane - and is not
        # driven to yaw the whole hand to clear a reach the finger never makes.
        ich = _finger_fk_from_angles(wrist, rot, spec, 0.0, IDLE_PROX,
                                     IDLE_MID, IDLE_DIST_FLEX)
        for pc in chains:
            dmin = min(_seg_dist(ich[k], ich[k + 1], pc[m], pc[m + 1])
                       for k in range(3) for m in range(3))
            if dmin < WRIST_IDLE_CLEAR:
                cost += IDLE_COLLIDE_W * (WRIST_IDLE_CLEAR - dmin) ** 2
    return cost


def _fret_event_pose(event, prev_pose=None, dt=None):
    """(wrist location, yaw, roll) for one fretted event, from a grid
    search over the wrist's rotation freedom that trades IK strain, a
    neutral wrist, and coherence with the previous pose against predicted
    finger-finger collisions."""
    press = _press_notes(event)
    if not press:
        return _solve_wrist(press, _fret_rotation(0, 0), KNUCKLE_Z), 0.0, 0.0
    pressed_fingers = {n["finger"] for n in press}
    idle = [f for f in FRET_FINGERS if f not in pressed_fingers]

    # Relax the wrist yaw toward the forearm-following neutral for this neck
    # position (not a fixed square-on zero), read from the base-pose wrist y so it
    # doesn't feed back on the yaw being searched.
    ny = _neutral_yaw(_solve_wrist(press, _fret_rotation(0, 0), KNUCKLE_Z)[1])

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
            cost = (WRIST_YAW_REG * (yaw - ny) ** 2
                    + WRIST_ROLL_REG * roll * roll
                    + _pose_cost(press, wrist, rot, idle))
            if cohere:
                cost += cohere * ((yaw - prev_pose[0]) ** 2
                                  + (roll - prev_pose[1]) ** 2)
            if best is None or cost < best[0]:
                best = (cost, wrist, yaw, roll)
    wrist = best[1]
    drop = _wrist_relax_drop(wrist[1])
    if drop and len(press) == 1:
        # SINGLE-finger grip: trail the wrist back along the finger REACH AXIS
        # (-rest_dir), not flat -Y. A raw flat-Y slide moves the knuckle sideways
        # of its target in the wrap-tilted finger frame, which the IK reads as a
        # large MCP splay (the lone pressing finger then hits the anatomical cap
        # and jams into its idle neighbour - the "strange abduction" near the nut).
        # Backing off along the reach axis instead is pure finger EXTENSION -
        # exactly the "re-extend to keep the fingertips on their frets" the relax
        # is meant to be - so the finger stays perpendicular while the wrist still
        # hangs lower and trails the reach.
        rest = _fret_rotation(best[2], best[3]) @ mathutils.Vector((0.0, 1.0, 0.0))
        wrist = (wrist[0] - drop * rest.x,
                 wrist[1] - drop * rest.y,
                 wrist[2] - drop * rest.z)
    elif drop:
        # Multi-finger grip (a double-stop): keep the flat-Y trail. Its two
        # fingers sit at different frets and reach apart; the flat-Y slide happens
        # to separate them (it extends them toward the nut), which the reach-axis
        # trail would not, so the tight double-stop stays clear. Bass is
        # near-monophonic, so this branch is rare.
        wrist = (wrist[0], wrist[1] - drop, wrist[2])
    return wrist, best[2], best[3]


def animate_fret_hand(arm_obj, notes, fps, frame_start,
                      min_attack_frames, max_attack_frames, release_frames):
    """Keyframe the fretting hand from the note list. Returns last frame."""
    pbones = arm_obj.pose.bones
    arm_obj.animation_data_clear()

    def to_frame(t):
        return frame_start + t * fps

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
    prev_target = None
    for i, (ev, target, rot) in enumerate(zip(events, targets, rots)):
        # Spread a big neck shift across the whole preceding rest: the farther the
        # hand travels, the later it may arrive (nearer the press) and the weaker
        # its 60%-through-the-gap floor, so the glide isn't crammed into the last
        # few frames. Small reaches keep the original snappy-but-safe lead.
        lead, floor_frac = ARRIVE_LEAD, 0.6
        if prev_target is not None:
            move = math.dist(prev_target, target)
            s = max(0.0, min(1.0, (move - SHIFT_SPREAD_MIN) /
                             (SHIFT_SPREAD_FULL - SHIFT_SPREAD_MIN)))
            lead = ARRIVE_LEAD + s * (MIN_ARRIVE_LEAD - ARRIVE_LEAD)
            floor_frac = 0.6 + s * (SHIFT_FLOOR_FRAC - 0.6)
        arrive = ev["t"] - lead
        if prev_t is not None:
            arrive = max(arrive, prev_t + floor_frac * (ev["t"] - prev_t))
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
        prev_target = target

    # Which finger presses which note on each event (idle otherwise), plus the
    # predicted world joint chain of each pressing finger, so idle fingers can be
    # posed to keep clear of them.
    event_press = [{n["finger"]: n for n in _press_notes(ev)} for ev in events]
    event_press_chains = []
    for ev, target, rot in zip(events, targets, rots):
        rmat = _fret_rotation(*rot)
        ch = {}
        for n in _press_notes(ev):
            ch[n["finger"]] = _finger_fk(target, rmat, FRET_FINGERS[n["finger"]],
                                         (n["x"], n["y"], n["z"]), DIST_FLEX_PRESS)
        event_press_chains.append(ch)

    def other_press_chains(f, i):
        """Chains of the fingers pressing event i other than f (the ones an
        idle/holding f must stay clear of)."""
        return [c for g, c in event_press_chains[i].items() if g != f]

    # When the PREVIOUS event's grip releases (its pressing fingers lift): an
    # approaching finger must not reach across the neck until then, or it drifts
    # through a finger still holding the previous grip (index crossing the still-
    # pressing middle at ~106). The finger f itself is EXCLUDED: a finger never
    # collides with its own just-vacated fret, so a finger re-fretting soon is
    # free to lift off and glide straight across (see CROSS_GLIDE_FR) rather than
    # being pinned by its own note-off. frame_start for the first event.
    def prev_release_for(f, i):
        if i == 0:
            return frame_start
        ends = [to_frame(nn["end"]) for g, nn in event_press[i - 1].items()
                if g != f]
        return max(ends) if ends else frame_start

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    event_onsets = sorted(to_frame(ev["t"]) for ev in events)

    def next_onset_after(frame):
        for o in event_onsets:
            if o > frame + 0.5:
                return o
        return None

    def key_idle(f, target, rot, frame, press_chains):
        y, p, m = _idle_finger_pose(f, target, rot, press_chains)
        _pose_finger(pbones, f, y, p, m, IDLE_DIST_FLEX, frame)

    last_frame = frame_start
    for f in FRET_FINGERS:
        spec = FRET_FINGERS[f]
        # Start idle over the finger's own string, at the first event's hand pose.
        key_idle(f, targets[0], rots[0], frame_start, other_press_chains(f, 0))
        prev_end = frame_start
        prev_tr = None   # (target, rot) of the previous event, for holding idle
        for i, (ev, target, rot) in enumerate(zip(events, targets, rots)):
            onset = to_frame(ev["t"])
            if f not in event_press[i]:
                # Hold the previous idle pose until the previous grip releases,
                # so a finger idle across two grips doesn't drift bass-ward over
                # a still-pressing neighbour during the first grip's sustain (the
                # idle ring over the pressing index through the octave). Only when
                # the previous grip sustains well past this event's arrival, so a
                # tight reach-in transition is left to glide.
                hold_prev = prev_release_for(f, i) - 0.5
                if (prev_tr is not None
                        and prev_end + 0.75 < hold_prev < onset - 3.0):
                    key_idle(f, prev_tr[0], prev_tr[1], hold_prev,
                             other_press_chains(f, i - 1))
                    prev_end = hold_prev
                # Idle: hover over the finger's own string, gliding into place
                # with the hand. One key at the event onset keeps it in its lane.
                frame = max(onset, prev_end + 0.5)
                key_idle(f, target, rot, frame, other_press_chains(f, i))
                prev_end = frame
                prev_tr = (target, rot)
                continue

            n = event_press[i][f]
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
            pressed = _fret_ik(dx, dy, dv, spec["lengths"], DIST_FLEX_PRESS)
            dx, dy, dv = ik_inputs(n["z"] + HOVER_LIFT)
            hover = _fret_ik(dx, dy, dv, spec["lengths"], DIST_FLEX_HOVER)

            on_frame = to_frame(n["start"])
            off_frame = max(on_frame, to_frame(n["end"]))
            pr = prev_release_for(f, i)
            was_pressing = i > 0 and f in event_press[i - 1]

            if was_pressing:
                # CROSS: this finger just pressed the previous event and now
                # re-frets. It already lifted off the old fret early (that press's
                # pressed_end, pulled back by CROSS_GLIDE_FR below), so it is
                # already gliding across - place the hover LATE (a small settle
                # before the press) and let that long eased glide, not a
                # last-instant snap, carry the fingertip to the new fret. No
                # idle-lane detour: going press -> lane -> press would yank it
                # back and forth (the fret-2 string-1 -> string-3 jump at ~96).
                hover_frame = on_frame - SETTLE_FR
                hover_frame = max(hover_frame, prev_end + 0.5, pr)
                hover_frame = min(hover_frame, on_frame - 0.5)
                hover_frame = max(hover_frame, prev_end + 0.5)
            else:
                # REACH: coming from the idle lane. Hover early, then hold the
                # finger in its own lane until REACH_LEAD_FR frames before the
                # hover, so it drifts to the fret over several eased frames
                # instead of the old ~0.75-frame snap.
                hover_frame = max(frame_start, on_frame - attack_frames(n))
                hover_frame = max(hover_frame, prev_end + 0.5, pr)
                hover_frame = min(hover_frame, on_frame - SETTLE_FR, on_frame)
                hover_frame = max(hover_frame, prev_end + 0.5)
                # Hold the finger's PREVIOUS idle pose (in the previous event's
                # hand frame) until that grip releases, so an idle finger doesn't
                # drift toward this press during the previous event's sustain.
                hold_prev = pr - 0.5
                if (prev_tr is not None
                        and prev_end + 0.75 < hold_prev < hover_frame - 0.5):
                    key_idle(f, prev_tr[0], prev_tr[1], hold_prev,
                             other_press_chains(f, i - 1))
                    prev_end = hold_prev
                # Start the eased reach REACH_LEAD_FR frames before the hover,
                # but never before the previous grip releases (so the finger
                # doesn't cross a still-pressing neighbour).
                hold_frame = max(hover_frame - REACH_LEAD_FR, pr)
                if hold_frame > prev_end + 0.75:
                    key_idle(f, target, rot, hold_frame, other_press_chains(f, i))
                    prev_end = hold_frame

            # If this same finger presses again very soon, it must lift off THIS
            # fret early to glide across in time (anticipation), and it should
            # NOT relax to the idle lane in between - the next event's late hover
            # is its next key, so it glides straight from one press to the next.
            presses_next_soon = (
                i + 1 < len(events) and f in event_press[i + 1]
                and to_frame(events[i + 1]["t"]) - off_frame < CROSS_GLIDE_FR)

            _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                         DIST_FLEX_HOVER, hover_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         DIST_FLEX_PRESS, on_frame)

            if presses_next_soon:
                next_on = to_frame(events[i + 1]["t"])
                pressed_end = min(off_frame, next_on - CROSS_GLIDE_FR)
                pressed_end = max(pressed_end, on_frame + 0.5)
                _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                             DIST_FLEX_PRESS, pressed_end)
                prev_end = pressed_end
            else:
                # Lift back to the idle hover promptly after the note (never
                # lingering in the pressed pose while the next grip forms), and be
                # clear a frame before the next grip's onset.
                relax_frame = off_frame + RELAX_LIFT
                nxt_grip = next_onset_after(off_frame)
                if nxt_grip is not None:
                    relax_frame = min(relax_frame, nxt_grip - 1.0)
                # Begin the lift early enough to finish in time - even a hair
                # before the notated note end when the next grip lands at once.
                pressed_end = min(off_frame, relax_frame - RELAX_LIFT)
                pressed_end = max(pressed_end, on_frame + 0.5)
                relax_frame = max(relax_frame, pressed_end + 1.0)
                _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                             DIST_FLEX_PRESS, pressed_end)
                key_idle(f, target, rot, relax_frame, other_press_chains(f, i))
                prev_end = relax_frame
            prev_tr = (target, rot)
            last_frame = max(last_frame, off_frame + release_frames)

    # Bezier with auto-clamped handles so every finger and the wrist EASE like a
    # real hand: speed goes to zero at each rest key and peaks mid-move, instead
    # of the one-sided, constant-rate ramp SINE gives (which let a finger arrive
    # at full speed and then dead-stop - the acceleration spike). Same easing the
    # pick hand already uses; clamped handles keep a press from overshooting.
    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'AUTO_CLAMPED'
            kp.handle_right_type = 'AUTO_CLAMPED'
        fcurve.update()
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

def _pick_directions(events):
    """Down/up stroke per event. Bass is ALTERNATE-picked: the wrist swings
    like a pendulum, so each note simply flips the previous stroke's direction.
    That makes every pluck a pass-THROUGH of the swing (the pick is moving
    fastest as it crosses the string), which reads far more naturally than
    re-striking from the same side. The one exception is a downstroke reset: an
    accented note landing after a real rest starts a fresh downstroke."""
    dirs = []
    prev_t = None
    for ev in events:
        gap = None if prev_t is None else ev["t"] - prev_t
        vel = max(n["velocity"] for n in ev["notes"])
        if not dirs:
            direction = 1
        elif gap is not None and gap >= GAP_RESET and vel >= ACCENT_VEL:
            direction = 1
        else:
            direction = -dirs[-1]
        dirs.append(direction)
        prev_t = ev["t"]
    return dirs


def _pick_swing_rig():
    """Geometry of the pick's wrist swing. The pick tip hangs a fixed offset
    ``uw`` (flat world) from the wrist bone's HEAD; rotating the wrist about the
    swing ``axis`` sweeps the tip so it dips through the strings. The neck axis
    (Y) keeps the tip's along-string y FIXED (no rake) but rolls the palm; tilting
    the axis by PICK_DEVIATE off the forearm axis trades that palm roll for wrist
    DEVIATION (and a little along-string arc). Because the swing pivots about the
    bone head, which the bassist's right arm IK target COPY_LOCATIONs, the stroke
    costs the arm NO motion -- it only sees the smooth per-string object slide.
    Returns:

      head_off : flat-world offset from the object origin to the wrist-bone head
      tip_off  : flat-world tip offset from the head AT THE STRIKE (phi_bot), so
                 object loc = pluck point - tip_off - head_off
      phi_bot  : wrist angle placing the tip deepest (through the strings)
      axis     : the swing axis in armature-local space
    """
    head_local = mathutils.Vector((0.0, -0.025, 0.0))
    head_off = PICK_HAND_ROT @ head_local
    uw = PICK_HAND_ROT @ (mathutils.Vector(PICK_TIP_LOCAL) - head_local)
    # Swing axis = neck axis (Y) with PICK_DEVIATE of its FINGER-axis component
    # removed (that component is what rolls the palm), tilting toward deviation.
    finger = (PICK_HAND_ROT @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
    a = (mathutils.Vector((0.0, 1.0, 0.0)) - PICK_DEVIATE * finger.y * finger)
    a.normalize()
    # phi_bot = wrist angle putting the tip DEEPEST (min flat z = through the strings).
    phi_bot = min((i * math.tau / 720.0 for i in range(720)),
                  key=lambda phi: (mathutils.Matrix.Rotation(phi, 3, a) @ uw).z)
    tip_off = mathutils.Matrix.Rotation(phi_bot, 3, a) @ uw
    axis = PICK_HAND_ROT.transposed() @ a
    return head_off, tip_off, phi_bot, axis


def animate_pick_hand(arm_obj, notes, fps, frame_start):
    """Keyframe the picking hand's strokes as WRIST ROTATION over smooth
    object tracking. The object only slides along the strings (X) to sit over
    the struck string, at a constant height and depth; each pluck is a pendulum
    of the wrist bone that swings the pick tip down through the string and out
    the other side (a down/up stroke), so the fast per-note stroke stays in the
    wrist and never jerks the arm that follows the wrist. Returns the last
    frame."""
    arm_obj.animation_data_clear()
    pbone = arm_obj.pose.bones["wrist"]
    pbone.rotation_mode = 'XYZ'

    head_off, tip_off, phi_bot, axis = _pick_swing_rig()
    # Object placement so the arc BOTTOM (phi_bot) lands the tip at the pluck point:
    # tip_world = obj_loc + head_off + tip_off, wanted at (x_head, PLUCK_Y, STRING_Z-DEPTH).
    # The across-string x is per-string; the depth/along-string parts are the same every
    # strike, so the arm target still only slides across in smooth per-string steps.
    def objloc(x_head):
        return (x_head - head_off.x - tip_off.x,
                fret_layout.PLUCK_Y - head_off.y - tip_off.y,
                fret_layout.STRING_Z - PICK_DEPTH - head_off.z - tip_off.z)

    def swing_euler(phi):
        return tuple(mathutils.Matrix.Rotation(phi, 3, axis).to_euler())

    events = _group_events(notes)
    if not events:
        return frame_start
    directions = _pick_directions(events)

    def to_frame(t):
        return frame_start + t * fps

    min_dt = 0.6 / fps

    # 1. Object tracking. The hand eases from string to string (this is what the
    #    arm target follows) AND adds a per-strike ARM sweep: it winds up a little
    #    opposite the stroke just before contact and follows THROUGH in the stroke
    #    direction just after, so the forearm/elbow join the wrist in the hit.
    #    Amplitude scales with velocity; the mid key sits exactly on the struck
    #    string on the beat, so the pick still contacts precisely.
    loc_t = None
    prev_t = None
    n_ev = len(events)
    for i, (ev, d) in enumerate(zip(events, directions)):
        x_head = sum(n["pluck_x"] for n in ev["notes"]) / len(ev["notes"])
        vel_norm = max(0.0, min(1.0,
                                max(n["velocity"] for n in ev["notes"]) / 127.0))
        t = ev["t"]
        strike = STRIKE_SLOW - vel_norm * (STRIKE_SLOW - STRIKE_FAST)
        strike = min(strike, 0.7 * (t - prev_t)) if prev_t is not None \
            else min(strike, 0.10)
        gap_next = (events[i + 1]["t"] - t) if i + 1 < n_ev else 1.0
        wind = ARM_WIND * vel_norm
        follow = ARM_FOLLOW * vel_norm

        def key_loc(dx, tt):
            arm_obj.location = objloc(x_head + dx)
            arm_obj.keyframe_insert(data_path="location",
                                    frame=to_frame(max(0.0, tt)))

        t_wind = t - strike
        if loc_t is not None:
            t_wind = max(t_wind, loc_t + min_dt)
        t_strike = max(t_wind + min_dt, t)
        t_follow = max(t + min(ARM_FOLLOW_T, 0.4 * gap_next), t_strike + min_dt)
        key_loc(-d * wind, t_wind)      # wound up, off the stroke's back side
        key_loc(0.0, t_strike)          # exact contact on the struck string
        key_loc(d * follow, t_follow)   # follow-through in the stroke direction
        loc_t = t_follow
        prev_t = t

    # 2. Wrist pendulum -- the pick's whole stroke, kept small and close to the
    #    strings (bass is picked, not strummed). Two keys per note: the wrist is
    #    POISED to one side (pick a few mm off the strings), then dips THROUGH the
    #    string exactly on the onset. Rising out the far side to the next note's
    #    poise IS the follow-through, so alternating strokes flow as one gentle
    #    pendulum of fixed, dampened amplitude (PICK_SWING).
    #
    #    Loudness sets the DESCENT TIME, not the reach: a loud note stays poised
    #    and then snaps down over a short `strike`; a soft note eases down slowly
    #    over a long one. Either way the bottom is keyed on the onset, so every
    #    note lands in time -- only the approach speed changes with dynamics.
    def key_rot(phi, t):
        pbone.rotation_euler = swing_euler(phi)
        pbone.keyframe_insert(data_path="rotation_euler", frame=to_frame(t))

    last_frame = frame_start
    prev_t = None
    for ev, d in zip(events, directions):
        t = ev["t"]
        vel_norm = max(0.0, min(1.0,
                                max(n["velocity"] for n in ev["notes"]) / 127.0))
        strike = STRIKE_SLOW - vel_norm * (STRIKE_SLOW - STRIKE_FAST)
        # Keep the poise after the previous pluck (and off the very first frame).
        strike = min(strike, 0.7 * (t - prev_t)) if prev_t is not None \
            else min(strike, 0.10)
        key_rot(phi_bot - d * PICK_SWING, max(0.0, t - strike))  # poised, off string
        key_rot(phi_bot, t)                                      # dip through, on beat
        last_frame = max(last_frame, to_frame(t))
        prev_t = t
    # Lift off the strings after the final note.
    key_rot(phi_bot + directions[-1] * PICK_SWING * PICK_FOLLOW_FRAC,
            events[-1]["t"] + 0.12)
    last_frame = max(last_frame, to_frame(events[-1]["t"] + 0.12))

    # Bezier with auto-clamped handles so the wrist EASES like a real hand:
    # angular speed goes to zero at each swing turnaround (the poise) and peaks
    # as the pick crosses the string, instead of the constant-rate ramp that
    # single-sided SINE easing gives. Clamped handles keep it from overshooting
    # past the poise or digging past the string bottom.
    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'AUTO_CLAMPED'
            kp.handle_right_type = 'AUTO_CLAMPED'
        fcurve.update()
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
