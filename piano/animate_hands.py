"""Animates the Hand_L / Hand_R rigs from a fingering.json timeline.

Consumes the output of ``python -m piano.fingering`` (per-note hand, finger
and fingertip target) and keyframes the armatures built by build_hands.py:

  - The armature object's transform carries the wrist: for every chord event
    it is placed so the pressing fingers' knuckles sit over their keys
    (arriving slightly early, easing in and out of each position), then
    nudged by _wrist_fit to a placement its fingers can hold within
    their range of motion - the hand moves so the fingers need not contort.
    Moving between two positions it LIFTS: the wrist and every fingertip
    not holding a key down rise together over a smooth arc (_travel_lift),
    so the hand clears the keyboard on its way instead of sliding over it,
    and a key the hand walks out on is released rather than dragged. The
    move itself is sampled frame by frame along a minimum-jerk profile
    (_min_jerk) and given enough time to stay inside human speed and
    acceleration (LEAP_SPEED_MAX) - taken off the note it is leaving if
    the music does not otherwise allow it.
    It also carries a YAW (_event_yaw), turning the hand out toward the arm
    that reaches it, so a hand playing far from its own shoulder does not
    leave the whole diagonal in the wrist. Everything below the wrist is
    then solved in the hand's own frame (_hand_xy).
  - Finger bones are driven by closed-form two-link IK in the vertical
    plane through the knuckle: the proximal bone pitches, the middle joint
    flexes, the proximal z-rotation supplies sideways reach (capped at the
    knuckle's anatomical abduction), and the distal phalanx keeps a fixed
    natural flexion. No IK constraints are used, so the result is plain
    baked FK keyframes - every one of which lands inside the joint cage
    build_hands.py puts on the bones.
  - The whole hand is solved together, at the union of every finger's
    keyframe times, so no two fingers pass through each other: a finger
    with no note to play settles back over its own knuckle instead of
    holding the sideways reach of the key it last played, and whatever is
    not holding a key down is then slid aside and lifted until its
    phalanges clear their neighbours' by real surface distance.
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
    from .build_hands import (FINGERS, PALM_CENTRE, PALM_SIZE, rot_limit,
                              _finger_cross)
    from . import key_layout
except ImportError:  # loaded as a loose script via importlib
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from piano_midi_animator import _iter_action_fcurves
    from build_hands import (FINGERS, PALM_CENTRE, PALM_SIZE, rot_limit,
                             _finger_cross)
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
# So the cap lives in the IK, and _splay_clamp below moves the WRIST so the
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

# --- hand yaw ----------------------------------------------------------------
# The hand TURNS OUT to meet its own arm. Held square to the keys - fingers
# always along +y, which is how this rig was originally solved - the hand is
# fine in front of its own shoulder and increasingly wrong away from it: the
# forearm arrives diagonally, and since the hand will not follow, the difference
# is left in the wrist. At the ends of an 88-key board that reached 68 deg,
# past any human wrist, and no arm or body pose could take it out (at full
# stretch the forearm IS the shoulder-to-hand line).
#
# So the wrist target carries a yaw as well as a position, and every finger is
# solved in the hand's own frame. The angle is the direction the arm comes from
# - atan2 of the hand's offset from its shoulder over how far the shoulder sits
# behind the keys - taken only partway (the wrist supplies the rest, as it does
# in life) and capped at a comfortable turn.
HAND_YAW_FRAC = 0.7
HAND_YAW_MAX = math.radians(25.0)
ARM_DEPTH = 0.26        # m the shoulder sits behind the wrists at the keyboard
SHOULDER_DX = 0.227     # m from the player's centre line to either shoulder
#                         (half biacromial breadth at H = 1.75; the same figure
#                         build_pianist seats at the keyboard). Kept here rather
#                         than imported: the hands must animate with no player
#                         in the scene, and any seated adult is within a cm or
#                         two of this.
HOME_X = key_layout.key_x(60)   # the bench is centred on middle C

# A yawed hand meets a wide grip at an angle, which racks its outer fingers to
# DIFFERENT depths along the keys - sin(yaw) times the width of the grip - and
# they have to find that difference out of their own length, on top of whatever
# the voicing already asks of them. Which is why a pianist squares the hand up
# for a big stretch and only angles it for what one finger, or a close grip, can
# cover: the yaw of an event is capped at the angle its own width can afford.
# Measured on the demo, 8 mm keeps chord contact within half a millimetre of the
# square-handed solve while leaving single notes - which is what the ends of the
# keyboard are played with - free to turn the full amount.
YAW_SPREAD = 0.008      # m of depth spread across a grip a yaw may impose

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

# --- idle fingers ------------------------------------------------------------
# The IK aims each finger at its own key and knows nothing about where the other
# four are, so a finger that has finished its note used to sit there holding the
# sideways reach it played with, frame after frame, while the wrist glided on to
# the next chord. That stale splay is what put fingers through one another: by
# the time the hand had moved a key or two along, a finger still pointing back at
# the note it had released was lying across its neighbour.
#
# A real hand does the opposite - the moment a finger is done it comes back over
# its own knuckle, curved and hovering. So any gap in a finger's note list is
# spent in an IDLE pose: straight ahead of its knuckle, no sideways reach at all,
# at the hover height above the keybed.
IDLE_GAP = 0.30        # s: a gap at least this long between a finger's notes is
                       # spent idle rather than holding the last hover pose
IDLE_SETTLE = 0.10     # s: how long the finger takes to fall into / leave idle

# An idle finger reaches further along its own length than a pressing one does
# (REACH_FRAC), because it is not arching down onto a key: at 0.55 the knuckle
# has to hyperextend ~18 deg to hold the tip at hover height, which puts every
# idle PIP joint ABOVE its own knuckle - a tent rather than a hand, and the pose
# from which two neighbours most easily sweep across each other. At 0.75 the MCP
# sits near neutral and the finger drapes.
IDLE_REACH = 0.75

# The thumb rests ON the keys beside the fingers rather than curled under the
# palm: its CMC sits only 14 mm below the knuckle line here, so a folded thumb
# would have to arch its metacarpal UP over the palm - past the CMC's extension
# range, and straight through the index - to bring its tip back. It needs its own
# reach because IDLE_REACH of a thumb is one of the poses that arches; 62 mm
# keeps the column extended and the CMC comfortably inside its cage.
THUMB_IDLE_Y = 0.062
# ...and it rests to the RADIAL side, not straight ahead of its own CMC, which in
# this rig sits directly behind the index knuckle (1 mm apart across the
# keyboard) - so a thumb pointing straight ahead lies along the index for its
# whole length. A real one is turned out across a key or two, which is both where
# it rests and what gives its column room to extend.
THUMB_IDLE_X = 0.030

# --- finger-to-finger clearance ----------------------------------------------
# Measured between finger SURFACES, not bone axes, exactly as the guitarist's
# fret hand measures it (guitar/animate_hands.FINGER_RADIUS): each phalanx is a
# capsule whose radius comes from its anthropometric box, so the 18 mm index
# proximal and the 11 mm little fingertip are judged by what they actually
# occupy. Two chains are clear when every pair of their phalanges is.
#
# The digit holding a key down is the one that never moves; everyone else gives
# way around it, by however much they can afford to. An idle finger has a free
# hand and takes the whole search; a finger only on its way to (or off) a key is
# not sounding a note yet, so it may be nudged - HOVER_GIVE of the same freedom -
# which is what stops a finger stretched out early toward a key the wrist has
# not reached yet from landing on the finger already playing there. The search is
# scored like the guitar's _idle_finger_pose: clear the obstacle up to
# IDLE_CLEAR_TARGET, then sit as close to where the digit wanted to be as
# possible.
#
# Phalanges that meet INSIDE the palm box are not judged at all. The thumb's
# column is rooted a millimetre across from the index knuckle and 14 mm below it,
# so the two are in contact there in almost any pose - as the thenar mass and the
# index MCP are in a real hand. Both are buried in the palm; only the part of a
# digit that sticks out of it can be seen crossing anything, and that is still
# tested.
SEGS = ("prox", "mid", "dist")
FINGER_RADIUS = {(f, seg): max(_finger_cross(f, seg)) / 2.0
                 for f in FINGERS for seg in SEGS}
IDLE_CLEAR_TARGET = 0.0015   # m of surface clearance worth searching for
HOVER_GIVE = 0.4             # an approaching/lifting finger's share of an idle
                             # one's freedom to be nudged aside. A finger holding
                             # a key DOWN gets none at all: it is sounding a
                             # note, and moving it even a couple of mm off its
                             # key reads worse than the crossing it would fix.
SLIDE_STEP, SLIDE_MAX = 0.0035, 0.0175      # sideways, either way
RETREAT_STEP, RETREAT_MAX = 0.006, 0.036    # up off the keys, or down onto them
IDLE_SLIDE_COST = 0.06       # per m of sideways slide from the neutral idle
IDLE_RETREAT_COST = 0.03     # per m of retreat; cheaper, because a finger
                             # lifting away is the less conspicuous of the two
IDLE_RETREAT_TUCK = 0.35     # fraction of a retreat also taken out of the reach,
                             # so the digit curls up rather than pointing out

# ...but the KEYS are an obstacle too, and the clearance search only knew about
# the other fingers. A digit told to give way downward would take the whole
# retreat and end up 30 mm INSIDE the keybed (the thumb, whose idle pose sits
# lowest, did this for most of the reach take), which is not clearance, it is
# just a collision with something the search could not see. So a nudge below the
# key tops is charged for - steeply enough that any pose out in the air wins,
# and gradually enough that a digit with nowhere else to go still dips a little
# rather than crossing a neighbour.
KEYBED_Z = key_layout.WHITE_H     # top of a white key
KEYBED_CLEAR = 0.002              # m a nudged digit tries to keep above it
KEYBED_COST = 0.2                 # per m below that: ~7x the cost of the same
                                  # retreat taken upward, into free air. Higher
                                  # (0.5) and a digit boxed in between the thumb
                                  # and its neighbour on the demo's thumb-under
                                  # would rather lift 20 mm clear and sweep back
                                  # down through the finger beside it.

# --- lifting the hand between positions --------------------------------------
# Between two events the wrist used to glide flat, at the same height it plays
# at, so the whole hand SLID across the keyboard: idle fingertips hover 7.5 mm
# over a white key but 4.5 mm UNDER the top of a black one, and a finger still
# holding its key while the wrist departs (see the note on early release below)
# was dragged sideways at press depth, straight through the keys it passed.
#
# A pianist moving from one position to another lifts off, travels above the
# keys and comes down onto the next one - the wrist rises and falls in one arc
# and the fingers ride up with it. So every gap between events gets a LIFT: a
# smooth hump added to the wrist height and to every fingertip target that is
# not holding a key down, peaking mid-travel and back to zero on arrival, so the
# pose at either end is exactly the one the wrist fit chose. Since the hand
# translates rigidly, the finger poses at the apex are the ones already checked
# for range of motion and mutual clearance; only the height changes.
#
# How high scales with how far the hand goes: a neighbouring chord barely lifts,
# a leap across the board clears the black keys comfortably.
TRAVEL_LIFT_MIN = 0.045   # m of travel below which the hand just glides
TRAVEL_LIFT_FRAC = 0.12   # of the travel distance
TRAVEL_LIFT_MAX = 0.032   # m; a hand does not fly, and the arm has to follow
TRAVEL_LIFT_RATE = 0.5    # m/s ceiling on the rise, so a one-frame hop between
                          # two close positions cannot pop the hand up

# A key still sounding when the wrist leaves for such a move is LET GO of. The
# wrist departs as late as the hold allows but never later than MIN_TRAVEL
# before the next event needs it, so in a legato piece with big leaps it is
# already moving while the key is down; pinning the fingertip to that key then
# stretches the finger flat and drags its tip through the keyboard for the whole
# travel. A real hand releases and goes - the key is what stays behind (and with
# the sustain of a pedalled leap, so does the note). Only leaps that get a lift
# release early: a hand shuffling between neighbouring positions holds its notes
# out in full, as before.

# --- how a hand accelerates ---------------------------------------------------
# A leap from one position to another is a ballistic reach, and human reaches
# follow the MINIMUM-JERK profile (Flash & Hogan 1985): a symmetric bell of
# speed that starts and ends with zero velocity AND zero acceleration, so the
# arm never jerks into or out of a move. The travel is sampled along that
# profile. What it replaces - a single eased span between two keys - is a cubic,
# whose acceleration STEPS from nothing to its peak at the instant of departure
# and off a cliff at the arrival: infinite jerk at both ends of every move.
#
# For a move of D metres in T seconds the profile's peaks are
#     v = 1.875 D / T        a = 5.7735 D / T^2
# so a ceiling on speed and acceleration is really a floor on TIME: a big leap
# needs time, and where the music does not leave enough, the hand takes it from
# the note it is leaving (which is what a pianist does - a leap is played short)
# rather than moving inhumanly fast. Measured on the reach take before this, the
# 160-180 mm leaps crammed into MIN_TRAVEL were pulling 37-43 m/s^2.
#
# The ceilings are the fast end of a real reach rather than a comfortable one:
# 2.4 m/s and ~2.4 g put a half-metre leap at 0.39 s, which is about what an
# A0 -> C8 jump takes a player who can make it at all.
LEAP_SPEED_MAX = 2.4      # m/s at the peak of the bell
LEAP_ACCEL_MAX = 24.0     # m/s^2 at its shoulders
LEAP_MIN_HOLD = 0.08      # s of the key a leap may not take back: a note still
                          # has to be heard being played before the hand goes
LEAP_TIME_MARGIN = 1.10   # the profile is what the RENDERED frames sit on; the
                          # Bezier drawn through them bulges a little past it,
                          # and the lift adds a vertical share of its own, so
                          # the time solve buys a margin and the ceilings hold
                          # against the baked curve rather than the ideal one
TRAVEL_SAMPLE_MIN = 0.003    # m of travel worth sampling a profile through
TRAVEL_SAMPLE_MAX = 48       # keys, so a long slow drift stays cheap
TRAVEL_SAMPLE_DENSE = 8      # frames: a travel shorter than this is sampled at
                             # half frames (see below)


def _target_y(note, has_black=False):
    if note["is_black"]:
        return note["y"]
    if has_black:
        return DEEP_WHITE_THUMB_Y if note["finger"] == 1 else DEEP_WHITE_Y
    if note["finger"] == 1:
        return note["y"] - THUMB_FRONT_PULL
    return note["y"]


def _digit_offset_x(note, mirror):
    """Where this digit's fingertip sits ACROSS the hand, in the hand's own
    frame: at its knuckle for a finger, and for the thumb turned OUT to the
    radial side (THUMB_IDLE_X), which is both where a thumb rests and what keeps
    its column from lying along the index finger, the two being rooted a
    millimetre apart. The wrist is then placed this much the other way.
    """
    kx = FINGERS[note["finger"]]["knuckle"][0]
    out = mirror * THUMB_IDLE_X if note["finger"] == 1 else 0.0
    return kx * mirror - out


def _hand_xy(point, wrist, yaw):
    """A world point in the hand's own frame: the wrist at the origin and +y
    along the fingers, so a knuckle sits at exactly its build_hands offset and
    every finger solve reads the same as it did before the hand could turn.

    Only x and y rotate - the yaw is about Z - so the z that comes back is still
    the world height, which is what the reach/hover geometry is written in.
    """
    dx, dy = point[0] - wrist[0], point[1] - wrist[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, c * dy - s * dx, point[2])


def _event_yaw(event, mirror):
    """How far the hand turns out to meet its arm for this event (radians,
    positive = counter-clockwise seen from above).

    The arm arrives from the shoulder, so the hand turns toward it by the same
    reasoning a pianist's does: partway (HAND_YAW_FRAC) toward the line the
    forearm comes in on, never past HAND_YAW_MAX, and less than that for a grip
    wide enough that the angle would rack its fingers to different depths
    (YAW_SPREAD).
    """
    xs = [n["x"] for n in event["notes"]]
    centre, span = (min(xs) + max(xs)) / 2.0, max(xs) - min(xs)
    shoulder = HOME_X + mirror * SHOULDER_DX
    cap = HAND_YAW_MAX
    if span > 1e-6:
        cap = min(cap, math.asin(min(1.0, YAW_SPREAD / span)))
    yaw = -HAND_YAW_FRAC * math.atan2(centre - shoulder, ARM_DEPTH)
    return max(-cap, min(cap, yaw))


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


def _finger_chain(knuckle, lengths, yaw, prox, mid, dist_flex):
    """[knuckle, PIP, DIP, tip] for one posed finger, in whatever frame the
    knuckle was given in (the piano solves in the hand's own - see _hand_xy).

    Only the proximal bone carries the sideways rotation (see _pose_finger), so
    the whole chain lies in the vertical plane the knuckle's yaw picks out.
    """
    pts = [tuple(knuckle)]
    pitch = 0.0
    for length, bend in zip(lengths, (prox, mid, dist_flex)):
        pitch += bend
        h = length * math.cos(pitch)
        pts.append((pts[-1][0] + h * math.sin(yaw),
                    pts[-1][1] + h * math.cos(yaw),
                    pts[-1][2] - length * math.sin(pitch)))
    return pts


def _seg_dist(p1, q1, p2, q2):
    """(shortest distance, midpoint of the closest approach) for two segments."""
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    d1, d2, r = sub(q1, p1), sub(q2, p2), sub(p1, p2)
    a, e = dot(d1, d1), dot(d2, d2)
    f, c, b = dot(d2, r), dot(d1, r), dot(d1, d2)
    denom = a * e - b * b
    s = 0.0 if denom < 1e-12 else max(0.0, min(1.0, (b * f - c * e) / denom))
    t = max(0.0, min(1.0, (b * s + f) / e)) if e > 1e-12 else 0.0
    s = max(0.0, min(1.0, (b * t - c) / a)) if a > 1e-12 else 0.0
    ca = tuple(p1[k] + d1[k] * s for k in range(3))
    cb = tuple(p2[k] + d2[k] * t for k in range(3))
    return math.dist(ca, cb), tuple((ca[k] + cb[k]) / 2.0 for k in range(3))


def _in_palm(point, wrist):
    """Whether a point falls inside the palm box, both in the wrist's frame."""
    return all(abs(p - (w + c)) <= h / 2.0
               for p, w, c, h in zip(point, wrist, PALM_CENTRE, PALM_SIZE))


def _chain_clearance(fa, ca, fb, cb, wrist):
    """Surface clearance (m, negative = interpenetrating) between two posed
    finger chains, each [knuckle, PIP, DIP, tip] in the wrist's own frame.

    Phalanges that meet inside the palm are skipped (see the notes above): the
    thumb's column is rooted a millimetre across from the index knuckle and
    14 mm below it, so the two touch there in almost any pose, and both are
    inside the palm box where nothing can be seen crossing anything."""
    out = []
    for k in range(3):
        for m in range(3):
            d, mid = _seg_dist(ca[k], ca[k + 1], cb[m], cb[m + 1])
            if _in_palm(mid, wrist):
                continue
            out.append(d - FINGER_RADIUS[(fa, SEGS[k])]
                       - FINGER_RADIUS[(fb, SEGS[m])])
    return min(out) if out else float("inf")


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


def _event_root_target(event, mirror, yaw):
    """Wrist position placing the event's pressing knuckles over their keys,
    with the hand turned out by `yaw`.

    Each digit wants the wrist one rotated knuckle-plus-reach offset back from
    its key. Uses the midrange (not the mean) of those requirements: in a
    stretched chord the outer fingers have the least reach to spare, so the
    residual is split between them instead of letting the comfortable
    middle fingers drag the wrist off-center.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    c, s = math.cos(yaw), math.sin(yaw)
    xs, ys = [], []
    for n in event["notes"]:
        spec = FINGERS[n["finger"]]
        _kx, ky, _kz = spec["knuckle"]
        ox = _digit_offset_x(n, mirror)
        oy = ky + REACH_FRAC * sum(spec["lengths"])
        xs.append(n["x"] - (c * ox - s * oy))
        ys.append(_target_y(n, has_black) - (s * ox + c * oy))
    z = HOVER_Z + (BLACK_KEY_LIFT if has_black else 0.0)
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, z)


def _splay_clamp(event, tgt, mirror, yaw):
    """Slide one wrist target sideways until every pressing finger can reach
    its key within the knuckle's anatomical splay (see FINGER_MCP_SPLAY).

    Smoothing the wrist path is what gives the hand its glide - the fingers
    reach out of a gliding wrist rather than the wrist hopping key to key - but
    left unbounded it asks a finger to deviate 40-50 deg sideways, which no
    knuckle does (and which the build-time LIMIT_ROTATION cage would clamp,
    sliding the fingertip off its key). Each pressing finger admits a window of
    wrist travel whose width is tan(cap) * its knuckle-to-key distance; the
    target is clamped into the intersection, so the glide survives wherever the
    fingers can absorb it and the WRIST does the rest of the reaching. A chord
    too wide for any one window (empty intersection) takes the window midpoint,
    splitting the residual between its outer fingers - the same compromise
    _event_root_target makes with its midrange.

    "Sideways" is ACROSS THE HAND, not across the keyboard: sliding along the
    hand's own x is what changes a finger's splay without also changing how far
    it has to reach, so with the hand turned out the slide follows it.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    lo, hi = float("-inf"), float("inf")
    for n in event["notes"]:
        _kx, ky, _kz = FINGERS[n["finger"]]["knuckle"]
        tx, ty, _tz = _hand_xy((n["x"], _target_y(n, has_black), 0.0), tgt, yaw)
        dx = tx - _digit_offset_x(n, mirror)
        span = math.tan(_splay_cap(n["finger"])) * max(ty - ky, MIN_REACH_Y)
        lo = max(lo, dx - span)
        hi = min(hi, dx + span)
    slide = (lo + hi) / 2.0 if lo > hi else max(lo, min(hi, 0.0))
    return (tgt[0] + slide * math.cos(yaw),
            tgt[1] + slide * math.sin(yaw), tgt[2])


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


def _event_pose_cost(event, tgt, mirror, yaw, press_white, press_black):
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
        tx, ty, _tz = _hand_xy((n["x"], _target_y(n, has_black), 0.0), tgt, yaw)
        dx, dy = tx - kx * mirror, ty - ky
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


def _wrist_fit(events, targets, yaws, mirror, press_white, press_black):
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
    for ev, tgt, yaw in zip(events, targets, yaws):
        base = _splay_clamp(ev, tgt, mirror, yaw)
        best = (_event_pose_cost(ev, base, mirror, yaw, press_white,
                                 press_black), base)
        if best[0] > 0.0:
            steps = int(WRIST_Y_RANGE / WRIST_Y_STEP)
            for i in range(-steps, steps + 1):
                dy = i * WRIST_Y_STEP
                z = tgt[2]
                while z >= MIN_WRIST_Z - 1e-9:
                    cand = _splay_clamp(ev, (tgt[0], tgt[1] + dy, z), mirror,
                                        yaw)
                    cost = (_event_pose_cost(ev, cand, mirror, yaw, press_white,
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
        acc = [0.0] * len(targets[i])
        wsum = 0.0
        for j, tgt in enumerate(targets):
            if abs(j - i) > 3 * sigma or segment[j] != segment[i]:
                continue
            # The width is in *events*, so a scale glides the same whether
            # it is played slowly or fast. Chords weigh more than single
            # notes: stretched voicings need near-exact placement.
            w = math.exp(-0.5 * ((j - i) / sigma) ** 2) * len(events[j]["notes"])
            for k in range(len(acc)):
                acc[k] += w * tgt[k]
            wsum += w
        smoothed.append(tuple(a / wsum for a in acc))
    return smoothed


def _ease(u):
    """Blender's SINE keyframe interpolation (ease-in), so sampling a pose
    between two keys here reproduces what the baked f-curve will do."""
    return 1.0 - math.cos(max(0.0, min(1.0, u)) * math.pi / 2.0)


def _ease_wrist(u):
    """The same for the WRIST curves, which are baked BEZIER/AUTO_CLAMPED.

    SINE is one-sided: the hand would leave a position at zero speed and still
    be accelerating when it got to the next one, then stop dead - the arrival
    snap the bass hand had (bass_guitar/animate_hands). A travel curve is not an
    attack, so it eases at BOTH ends; auto-clamped handles go flat wherever a
    key repeats its neighbour's value (every dwell, and the top of a travel
    arc), which makes such a span a cubic between two horizontal tangents -
    exactly smoothstep. The finger bones keep SINE: a press has to arrive at
    speed, and it must stay in step with the key dips.

    The SHAPE of a travel no longer comes from this, though - it is sampled
    frame by frame along the minimum-jerk profile (see LEAP_SPEED_MAX), and this
    only fills the gaps between those samples, which are a frame apart or less.
    """
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _sample(keys, frame, ease=_ease):
    """Interpolate a list of (frame, value-tuple) keys the way Blender will."""
    if frame <= keys[0][0]:
        return keys[0][1]
    if frame >= keys[-1][0]:
        return keys[-1][1]
    for (fa, va), (fb, vb) in zip(keys, keys[1:]):
        if fa <= frame <= fb:
            u = ease(0.0 if fb <= fa else (frame - fa) / (fb - fa))
            return tuple(a + (b - a) * u for a, b in zip(va, vb))
    return keys[-1][1]


def _travel_dist(a, b):
    """How far the hand goes between two wrist targets, across the keyboard."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _travel_lift(dist, dt):
    """How high the hand arcs crossing `dist` metres in `dt` seconds."""
    if dist < TRAVEL_LIFT_MIN or dt <= 0.0:
        return 0.0
    return min(TRAVEL_LIFT_FRAC * dist, TRAVEL_LIFT_MAX,
               TRAVEL_LIFT_RATE * dt / 2.0)


def _travel_time(dist):
    """The least time a move of `dist` metres may take (see LEAP_SPEED_MAX).

    Inverts the minimum-jerk peaks: T >= 1.875 D / v_max from the speed ceiling
    and T >= sqrt(5.7735 D / a_max) from the acceleration one, whichever is
    longer. Short moves are governed by the acceleration, long ones by the speed.
    """
    return LEAP_TIME_MARGIN * max(1.875 * dist / LEAP_SPEED_MAX,
                                  math.sqrt(5.7735 * dist / LEAP_ACCEL_MAX))


def _min_jerk(u):
    """Flash & Hogan's minimum-jerk profile: 0 at u=0, 1 at u=1, with zero
    velocity and zero acceleration at both ends."""
    u = max(0.0, min(1.0, u))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def _arc_shape(u):
    """The travel arc's height, 0 -> 1 -> 0. Cubed rather than squared so the
    lift-off and the landing are flat in acceleration as well as in speed - the
    hand leaves the keys the way the rest of the move starts."""
    return math.sin(math.pi * max(0.0, min(1.0, u))) ** 3


def _sample_plan(plan, frame):
    """(key target or None, distal flexion, idle mix, give) for one plan.

    A plan entry's target is None when the finger is idle there, and an idle
    pose is only a point once the knuckle is known - so rather than a target
    this returns how far the finger has eased TOWARD idle (0 = on its key
    target, 1 = fully idle), leaving the caller to resolve the idle end. `give`
    is the entry's own freedom to be nudged: 0 while a key is held down, 1 the
    rest of the time.
    """
    if frame <= plan[0][0]:
        _fr, tgt, flex, give = plan[0]
    elif frame >= plan[-1][0]:
        _fr, tgt, flex, give = plan[-1]
    else:
        for (fa, ta, xa, ga), (fb, tb, xb, gb) in zip(plan, plan[1:]):
            if fa <= frame <= fb:
                u = _ease(0.0 if fb <= fa else (frame - fa) / (fb - fa))
                flex, give = xa + (xb - xa) * u, ga + (gb - ga) * u
                if ta is None and tb is None:
                    return None, flex, 1.0, give
                if ta is None:
                    return tb, flex, 1.0 - u, give
                if tb is None:
                    return ta, flex, u, give
                return (tuple(a + (b - a) * u for a, b in zip(ta, tb)),
                        flex, 0.0, give)
        _fr, tgt, flex, give = plan[-1]
    return tgt, flex, 1.0 if tgt is None else 0.0, give


def _knuckle(finger, wrist, mirror):
    kx, ky, kz = FINGERS[finger]["knuckle"]
    return (wrist[0] + kx * mirror, wrist[1] + ky, wrist[2] + kz)


def _idle_target(finger, knuckle, mirror, hover_z, slide=0.0, retreat=0.0):
    """Where a digit that is not playing puts its fingertip.

    Straight ahead of its own knuckle - no sideways reach at all - arched to the
    same fraction of its length a press uses, hovering over the keys; the thumb
    turned out to the radial side (THUMB_IDLE_X). ``slide`` and ``retreat`` are
    the nudges _solve_clear uses to get a digit out of a neighbour's way: aside,
    and up off the keys - or, negative, out and down onto them, which is what a
    digit with little drop left beneath it (a low wrist) has to do instead. A
    retreat also trades reach for height (IDLE_RETREAT_TUCK), so the digit curls
    in rather than pointing up.
    """
    if finger == 1:
        reach, out = THUMB_IDLE_Y, -mirror * THUMB_IDLE_X
    else:
        reach, out = IDLE_REACH * sum(FINGERS[finger]["lengths"]), 0.0
    reach = max(MIN_REACH_Y, reach - IDLE_RETREAT_TUCK * retreat)
    return (knuckle[0] + out + slide, knuckle[1] + reach, hover_z + retreat)


def _in_cage(finger, yaw, prox, mid, dist_flex):
    """Whether this pose lands inside build_hands' joint cage.

    The LIMIT_ROTATION constraints on the bones are guards, never clampers: a
    keyframe outside them would be silently pulled back in the scene, and the
    pose that then shows up is not the one the clearance search checked. So the
    search only ever considers poses that survive the cage.
    """
    for seg, rx, rz in (("prox", -prox, -yaw), ("mid", -mid, 0.0),
                        ("dist", -dist_flex, 0.0)):
        limit = rot_limit(finger, seg)
        for axis, val in (("x", rx), ("z", rz)):
            lo, hi = limit[axis]
            if not math.radians(lo) <= val <= math.radians(hi):
                return False
    return True


def _pose_from_target(finger, knuckle, target, dist_flex):
    """(yaw, prox, mid) putting this finger's tip on a world-space target."""
    return _finger_ik(target[0] - knuckle[0], target[1] - knuckle[1],
                      knuckle[2] - target[2], FINGERS[finger]["lengths"],
                      dist_flex, _splay_cap(finger))


def _grid(step, span, signed):
    n = int(span / step + 1e-9)
    return tuple(i * step for i in range(-n if signed else 0, n + 1))


def _solve_clear(finger, knuckle, mirror, wrist, hover_z, dist_flex,
                 obstacles, key_target, mix, budget):
    """Where `finger`'s tip goes so it keeps clear of the chains already placed.

    `key_target`/`mix` say what the digit wants: a point on a key, the idle pose
    over its own knuckle, or - through the ease either side of a rest - a blend
    of the two. That wish is tried first and kept if it is already clear.
    Otherwise a grid of sideways slides and retreats around it is searched, in
    order of how far each strays (stray_cost), for the pose that clears its
    neighbours while straying least, which is how a hand keeps its fingers
    apart: the free ones move aside and away, the one holding a key does not
    move at all, and none of them gives way into the keyboard.

    `budget` (0..1) is how far that search may stray - near zero for a finger on
    its key, whose grid then collapses to the wish itself. `obstacles` are
    (finger, chain) pairs: every digit placed before this one, so no two fingers
    settle into each other either. Candidates that would take a joint outside its
    range of motion are not considered at all (_in_cage).
    """
    def wish(slide=0.0, retreat=0.0):
        idle = _idle_target(finger, knuckle, mirror, hover_z, slide,
                            retreat)
        if key_target is None:
            return idle
        if mix <= 0.0:
            return (key_target[0] + slide, key_target[1],
                    key_target[2] + retreat)
        return tuple(k + (i - k) * mix for k, i in zip(key_target, idle))

    def stray_cost(retreat, slide):
        """What this nudge costs before any clearance it buys is counted.

        Sideways and (upward) away are the ordinary currencies. The third term
        is the KEYS: how far the nudge would push the digit below the key tops,
        and below where it wanted to be anyway - a digit ON a key is not diving,
        it is playing. Dodging a neighbour by burying a finger 30 mm inside the
        keyboard, which is what the thumb (lowest of the five) used to do for
        most of the reach take, is no clearance at all, so it is charged for:
        steeply enough that any pose out in the air wins, gradually enough that
        a digit truly boxed in dips a millimetre rather than crossing.
        """
        sunk = max(0.0, min(KEYBED_Z + KEYBED_CLEAR, wish(slide)[2])
                   - wish(slide, retreat)[2])
        return (IDLE_SLIDE_COST * abs(slide) + IDLE_RETREAT_COST * abs(retreat)
                + KEYBED_COST * sunk)

    # Searched CHEAPEST FIRST, so the wish itself (cost 0) is always tried
    # before any nudge and the first pose the joint cage allows is the nearest
    # legal one. Walked in plain grid order instead, a digit whose own wish the
    # cage refused took whichever corner of the search came first - the full
    # downward nudge - and a digit with nothing in its way at all did the same,
    # which is how the thumb ended up inside the keybed with no neighbour
    # anywhere near it.
    nudges = sorted((stray_cost(retreat, slide), retreat, slide)
                    for retreat in _grid(RETREAT_STEP, budget * RETREAT_MAX,
                                         True)
                    for slide in _grid(SLIDE_STEP, budget * SLIDE_MAX, True))

    best, fallback = None, None
    for cost, retreat, slide in nudges:
        target = wish(slide, retreat)
        pose = _pose_from_target(finger, knuckle, target, dist_flex)
        if not _in_cage(finger, *pose, dist_flex):
            continue
        if fallback is None:
            fallback = target
        if not obstacles:
            return target
        chain = _finger_chain(knuckle, FINGERS[finger]["lengths"], *pose,
                              dist_flex)
        clear = min(_chain_clearance(finger, chain, g, oc, wrist)
                    for g, oc in obstacles)
        if clear >= IDLE_CLEAR_TARGET and cost <= 0.0:
            return target
        score = min(clear, IDLE_CLEAR_TARGET) - cost
        if best is None or score > best[0]:
            best = (score, target)
    if best is not None:
        return best[1]
    # Nothing in the grid is inside the cage: a key the hand cannot really hold.
    # Keep what was asked for, exactly as the wrist search's own compromises are
    # kept, and let the bone constraints show it for what it is.
    return fallback if fallback is not None else wish()


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

    events = _group_events(notes)
    if not events:
        arm_obj.rotation_euler = (0.0, 0.0, 0.0)
        for f in FINGERS:
            _relax_finger(pbones, f, frame_start)
        return frame_start

    # --- wrist (armature object) placement ---------------------------------
    # Wrist path: per-event ideal placement, smoothed into a glide, then pulled
    # back inside what the fingers can anatomically do (sideways splay, then
    # reach) - the hand moves so the fingers do not have to contort.
    #
    # A placement is (x, y, z, yaw): where the wrist sits AND how far the hand
    # is turned out to meet its arm (_event_yaw). The yaw goes through the same
    # smoothing as the position - they must travel together, or the hand would
    # snap round between two events it glides between - and the fit then re-scores
    # each event's fingers in the frame that yaw puts them in.
    yaws = [_event_yaw(ev, mirror) for ev in events]
    smoothed = _smooth_targets(events, [_event_root_target(ev, mirror, y) + (y,)
                                        for ev, y in zip(events, yaws)])
    yaws = [s[3] for s in smoothed]
    targets = [t + (y,) for t, y in zip(
        _wrist_fit(events, [s[:3] for s in smoothed], yaws, mirror,
                   press_depth_white, press_depth_black), yaws)]

    # Kept as well as keyed: the finger solve below needs to know where the
    # wrist actually is at each frame it poses a finger, not just where the
    # event it belongs to wanted it.
    wrist_keys = []

    def key_root(t, target):
        arm_obj.location = target[:3]
        arm_obj.rotation_euler = (0.0, 0.0, target[3])
        arm_obj.keyframe_insert(data_path="location", frame=to_frame(t))
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=to_frame(t))
        wrist_keys.append((to_frame(t), target))

    # When the wrist leaves each event and when it has to be at the next one.
    # Worked out for the whole take up front because the finger plans below need
    # it too: a note whose key the wrist walks out on is released rather than
    # dragged (see TRAVEL_LIFT_MIN).
    sched = []
    prev_t = None
    for i, ev in enumerate(events):
        arrive = ev["t"] - ARRIVE_LEAD
        if prev_t is not None:
            # In fast passages give at least 60% of the window to travel,
            # so the wrist flows instead of hop-and-waiting.
            arrive = max(arrive, prev_t + 0.6 * (ev["t"] - prev_t))
        arrive = max(arrive, 0.0)
        # Depart as late as the hold allows, but always leave the wrist at
        # least MIN_TRAVEL of glide to the next event - a dwell followed by
        # a near-instant hop reads as a lurch.
        end = max(n["end"] for n in ev["notes"])
        depart = end
        if i + 1 < len(events):
            depart = min(depart, events[i + 1]["t"] - ARRIVE_LEAD - MIN_TRAVEL)
        depart = max(depart, ev["t"])
        # A leap the remaining window cannot carry at human speed LEAVES EARLIER
        # - the time has to come from somewhere, and a pianist takes it off the
        # end of the note rather than out of the move (see LEAP_SPEED_MAX). The
        # window is max(next_t - ARRIVE_LEAD - d, 0.6 * (next_t - d)) by the
        # arrival rule above; both shrink as the departure `d` gets later, so
        # the latest departure that still leaves `need` of travel is the larger
        # of the two solutions.
        if i + 1 < len(events):
            dist = _travel_dist(targets[i], targets[i + 1])
            if dist >= TRAVEL_LIFT_MIN:
                need = _travel_time(dist)
                nt = events[i + 1]["t"]
                latest = max(nt - ARRIVE_LEAD - need, nt - need / 0.6)
                depart = min(depart, max(latest, ev["t"] + LEAP_MIN_HOLD,
                                         arrive))
        # `leave` is when the wrist's last key at this event sits - the moment
        # it starts moving on, which is the departure if it dwelled here at all
        # and its arrival if the passage is too fast for a dwell.
        dwell = depart > arrive + 0.02
        leave = depart if dwell else arrive
        sched.append((arrive, depart, leave))
        prev_t = leave

    # Each travel is then SAMPLED along the minimum-jerk profile (_min_jerk),
    # about one key per frame, rather than left to a single eased span - that is
    # what gives the move a human acceleration at both ends instead of a step
    # (see LEAP_SPEED_MAX). The arc rides on top of the same samples, so the
    # lift and the travel are one motion. `arc_keys` keeps the height alone,
    # because the FINGERS have to ride up by it too.
    arc_keys = [(frame_start, (0.0,))]
    lifts = [0.0] * len(events)

    for i, (ev, target) in enumerate(zip(events, targets)):
        arrive, depart, leave = sched[i]
        key_root(arrive, target)
        if leave > arrive:
            key_root(depart, target)
        if i + 1 >= len(events):
            continue
        nxt, span = sched[i + 1][0], sched[i + 1][0] - leave
        dist = _travel_dist(target, targets[i + 1])
        lifts[i] = _travel_lift(dist, span)
        if span <= 0.0 or (dist < TRAVEL_SAMPLE_MIN and lifts[i] <= 0.0):
            continue
        arc_keys += [(to_frame(leave), (0.0,)), (to_frame(nxt), (0.0,))]
        # Sampled ON THE RENDERED FRAMES, not at some spacing of its own: what
        # anyone sees of this move is the sequence of positions at whole frames,
        # so those are the ones that have to lie on the profile exactly. Between
        # them the baked curve is a Bezier through the samples, which bulges a
        # little past the profile it is drawn through - so a SHORT travel, which
        # has few frames to be described by and the steepest accelerations, is
        # sampled at half frames as well. A travel too short to contain a whole
        # frame still gets its midpoint: never a bare two-key jump.
        f0, f1 = to_frame(leave), to_frame(nxt)
        per = 1.0 if f1 - f0 >= TRAVEL_SAMPLE_DENSE else 0.5
        first = math.ceil((f0 + 1e-6) / per) * per
        inner = [first + k * per
                 for k in range(int((f1 - 1e-6 - first) / per) + 1)]
        if not inner:
            inner = [0.5 * (f0 + f1)]
        if len(inner) > TRAVEL_SAMPLE_MAX:      # a long drift: thin them out
            inner = inner[::math.ceil(len(inner) / TRAVEL_SAMPLE_MAX)]
        for f in inner:
            u = (f - f0) / (f1 - f0)
            lift = lifts[i] * _arc_shape(u)
            s = _min_jerk(u)
            p = [a + (b - a) * s for a, b in zip(target, targets[i + 1])]
            p[2] += lift
            key_root(leave + u * span, tuple(p))
            arc_keys.append((f, (lift,)))
    arc_keys.sort(key=lambda k: k[0])

    # --- fingers -----------------------------------------------------------
    # Each finger first gets a PLAN: where its fingertip should be at a handful
    # of frames - approach, press, hold, release - plus the idle poses it
    # settles into between notes. Nothing is keyed yet, because where a finger
    # can go is not its own business alone: the plans are resolved together
    # below, at the union of all their frames, so an idle finger can be moved
    # out of a pressing neighbour's way.
    #
    # Notes are planned per finger so consecutive notes on the same finger can
    # be clamped against each other: a release tail must never land on top of
    # the next note's approach or press.
    per_finger = {}
    for i, ev in enumerate(events):
        has_black = any(n["is_black"] for n in ev["notes"])
        for n in ev["notes"]:
            per_finger.setdefault(n["finger"], []).append((n, has_black, i))

    def attack_frames(note):
        vel_t = max(0, min(127, note["velocity"])) / 127.0
        return max_attack_frames - vel_t * (max_attack_frames -
                                            min_attack_frames)

    hover_z = _press_z(False, press_depth_white, press_depth_black) + HOVER_LIFT
    idle_gap, idle_settle = IDLE_GAP * fps, IDLE_SETTLE * fps

    # A plan entry is (frame, target, dist_flex, give); target None means
    # "idle", which only becomes a point once the wrist - and so the knuckle -
    # is known, and give is 0 only while the key is actually held down.
    IDLE = (None, DIST_FLEX_HOVER, 1.0)
    plans = {f: [(frame_start,) + IDLE] for f in FINGERS}

    def plan_idle(plan, until):
        """Spend a real gap before `until` back in the idle pose."""
        since = plan[-1][0]
        if until - since >= idle_gap:
            plan.append((since + idle_settle,) + IDLE)
            plan.append((until - idle_settle,) + IDLE)

    last_frame = frame_start
    for f, items in per_finger.items():
        plan = plans[f]
        prev_off = None
        for i, (n, has_black, ev_i) in enumerate(items):
            press_z = _press_z(n["is_black"], press_depth_white,
                               press_depth_black)
            key = (n["x"], _target_y(n, has_black), press_z)
            up = (key[0], key[1], key[2] + HOVER_LIFT)

            on_frame = to_frame(n["start"])
            # A note the hand leaves for another position is let go of when the
            # wrist goes, not when the MIDI says: held any longer, the fingertip
            # is pinned to a key the hand is no longer over and ploughs through
            # everything between the two (see TRAVEL_LIFT_MIN).
            off_t = n["end"]
            if lifts[ev_i] > 0.0:
                off_t = max(n["start"], min(off_t, sched[ev_i][1]))  # depart
            off_frame = max(on_frame, to_frame(off_t))
            hover_frame = max(frame_start, on_frame - attack_frames(n))
            if prev_off is not None:
                hover_frame = max(hover_frame, prev_off + 0.5)
            hover_frame = min(hover_frame, on_frame)

            plan_idle(plan, hover_frame)
            plan.append((hover_frame, up, DIST_FLEX_HOVER, 1.0))
            plan.append((on_frame, key, DIST_FLEX_PRESS, 0.0))
            plan.append((off_frame, key, DIST_FLEX_PRESS, 0.0))

            release_frame = off_frame + release_frames
            if i + 1 < len(items):
                nxt = items[i + 1][0]
                nxt_hover = max(frame_start,
                                to_frame(nxt["start"]) - attack_frames(nxt))
                release_frame = min(release_frame, nxt_hover - 0.5)
            if release_frame > off_frame + 0.25:
                # Lift off the key just played only while the finger is going
                # to stay in that neighbourhood - a repeat, or a quick
                # alternation. With a real gap ahead of it the finger comes
                # straight back over its own knuckle instead: by the time the
                # tail is over, the wrist has moved on, and a fingertip still
                # held above the key it played is a finger reaching backwards
                # across whatever the hand has arrived at.
                soon = (i + 1 < len(items)
                        and to_frame(items[i + 1][0]["start"])
                        - release_frame < idle_gap)
                plan.append((release_frame, up, DIST_FLEX_HOVER, 1.0)
                            if soon else (release_frame,) + IDLE)
            prev_off = off_frame
            last_frame = max(last_frame, to_frame(n["end"]) + release_frames)
        plan.append((plan[-1][0] + idle_gap,) + IDLE)

    # --- resolve the whole hand, frame by frame ----------------------------
    # At every frame any digit is keyed at, all five are posed together and one
    # at a time, least free first: a finger holding a key down goes exactly where
    # its key is and becomes an obstacle, then the ones approaching or lifting
    # off, then the idle - each nudged aside and away until it clears everything
    # already placed. Keying all five at every moment also means the f-curves
    # share their key times, so fingers that are apart at two keys stay apart
    # through the SINE ease between them.
    wrist_keys.sort(key=lambda k: k[0])
    moments = sorted({frame_start}
                     | {fr for fr, _loc in wrist_keys}
                     | {fr for fr, _lift in arc_keys}
                     | {e[0] for plan in plans.values() for e in plan})
    for frame in moments:
        # The wrist samples already carry the arc; what the FINGERS need is its
        # height on its own, because every fingertip free to go with the hand
        # (`give`, 0 only while a key is actually held down) rides up by the
        # same amount. The hand then simply translates - the poses at the top of
        # the arc are the ones already solved for range of motion and mutual
        # clearance, just higher up.
        wx, wy, wz, yaw = _sample(wrist_keys, frame, _ease_wrist)
        lift = _sample(arc_keys, frame, _ease_wrist)[0]
        # The whole hand is solved in ITS OWN frame (_hand_xy): the wrist at the
        # origin with the fingers along +y, which is the frame the bones are
        # posed in and the one every offset in build_hands.FINGERS is written
        # in. Only the keys have to be brought into it. Heights are untouched by
        # a yaw, so z stays the world height throughout.
        wrist = (0.0, 0.0, wz)
        knuckles, digits = {}, []
        for f in FINGERS:
            knuckles[f] = _knuckle(f, wrist, mirror)
            target, flex, mix, give = _sample_plan(plans[f], frame)
            if target is not None:
                target = _hand_xy(target, (wx, wy, wz), yaw)
                target = (target[0], target[1], target[2] + lift * give)
            digits.append((max(mix, HOVER_GIVE * give), f, target, flex, mix,
                           give))
        # Least free first: the finger holding a key down is placed exactly
        # where its key is and becomes an obstacle, then the ones with a little
        # give, then the fully idle - so what has to move is what can.
        placed = []
        for budget, f, target, flex, mix, give in sorted(digits,
                                                         key=lambda d: d[0]):
            target = _solve_clear(f, knuckles[f], mirror, wrist,
                                  hover_z + lift * give,
                                  flex, placed, target, mix, budget)
            pose = _pose_from_target(f, knuckles[f], target, flex)
            _pose_finger(pbones, f, *pose, flex, frame)
            placed.append((f, _finger_chain(knuckles[f], FINGERS[f]["lengths"],
                                            *pose, flex)))

    # Two easings, for two kinds of motion (see _ease_wrist): the wrist travels,
    # so it eases at both ends and arcs smoothly over the top of a leap; the
    # fingers strike, so they keep the one-sided SINE ramp that lands a
    # fingertip on its key at speed, in step with the key dips.
    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        travel = fcurve.data_path in ("location", "rotation_euler")
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'BEZIER' if travel else 'SINE'
            if travel:
                kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'

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
