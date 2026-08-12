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
    from .build_hands import (FRET_FINGERS, HAND_ROT_Z, PICK_ROT,
                              PICK_TIP_LOCAL, WRAP_TILT, _finger_cross,
                              pick_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger)
except ImportError:  # loaded as a loose script via importlib
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    import fret_layout
    from build_hands import (FRET_FINGERS, HAND_ROT_Z, PICK_ROT,
                             PICK_TIP_LOCAL, WRAP_TILT, _finger_cross,
                             pick_world_offset)
    from piano.piano_midi_animator import _iter_action_fcurves
    from piano.animate_hands import (_finger_ik, _smooth_targets,
                                     _group_events, _pose_finger)


# Every clearance below is measured between finger SURFACES, not bone axes: each
# phalanx is treated as a capsule whose radius comes from its anthropometric box
# (build_hands.FINGER_CROSS). A flat axis threshold was fine when every phalanx
# was the same thin 11 mm stick, but a real index proximal is 18 mm across and a
# little-finger tip 11 mm, so the same axis gap means very different things -- and
# the guitar puts four fingers down at once on a chord, where that difference is
# exactly what decides whether the boxes intersect.
SEGS = ("prox", "mid", "dist")
FINGER_RADIUS = {(f, seg): max(_finger_cross(f, seg)) / 2.0
                 for f in FRET_FINGERS for seg in SEGS}


def _chain_clearance(fa, ca, fb, cb):
    """Surface clearance (m, negative = interpenetrating) between two posed
    finger chains, each [knuckle, prox, mid, tip] in world space."""
    return min(_seg_dist(ca[k], ca[k + 1], cb[m], cb[m + 1])
               - FINGER_RADIUS[(fa, SEGS[k])] - FINGER_RADIUS[(fb, SEGS[m])]
               for k in range(3) for m in range(3))


# --- fret hand -------------------------------------------------------------
KNUCKLE_Z = 0.056      # knuckle-line height while fretting (arch clearance)
KNUCKLE_Z_BARRE = 0.048  # lower knuckles while barring, so the index lies flat
HOVER_LIFT = 0.012     # fingertip height above the string while not pressing
ARRIVE_LEAD = 0.15     # seconds the wrist arrives before a press
MIN_TRAVEL = 0.12      # seconds of glide the wrist gets between events
# A big neck position shift is spread over the WHOLE rest before it, not lunged
# in the last few frames with the hand then sitting parked. The larger the shift,
# the later the wrist is allowed to arrive (down to MIN_ARRIVE_LEAD) and the
# smaller its "arrive at least 60% through the gap" floor, so the long glide up
# the neck uses every frame the previous note's release leaves it - the way a
# player eases into a position shift in anticipation instead of snapping.
MIN_ARRIVE_LEAD = 0.06   # s: least lead before a press, even for a big shift
SHIFT_SPREAD_MIN = 0.04  # m: shifts below this aren't spread (a normal reach)
SHIFT_SPREAD_FULL = 0.12  # m: shifts this large get the full spread
SHIFT_FLOOR_FRAC = 0.12  # the gap-fraction floor on arrival for a full shift
# How far back along the finger's rest direction the knuckle line sits from the
# fingertip target -- i.e. how EXTENDED each pressing finger is. At 0.50 the
# fingers had to fold into a claw to reach the board (knuckle-to-tip only ~0.5 of
# the finger's length) and, worse, they then approached their string almost
# straight DOWN: with so little reach in the knuckle plane, the small along-neck
# offset to a fret demanded a huge MCP azimuth (40-55 deg, i.e. past any human
# abduction), so a capped finger landed up to 19 mm short of its string. Sitting
# the hand further back off the treble edge lets each finger reach ACROSS to its
# string in a natural ~0.70 arch, which is both what a real fretting hand does and
# what keeps the sideways deviation small (max demand 54 -> 40 deg, worst press
# error 19 -> 7 mm, worst inter-finger clearance -8.7 -> -3.1 mm).
REACH_FRAC = 0.75      # fraction of finger length the knuckle sits back
DIST_FLEX_PRESS = 0.45
DIST_FLEX_HOVER = 0.30
DIST_FLEX_BARRE = 0.06         # flattened index while barring
DIST_FLEX_BARRE_HOVER = 0.12
PRESS_STAGGER = 0.022  # fret-slot y spread between fingers sharing a fret -
                       # a diagonal placement (lower finger farther behind the
                       # wire) that keeps same-fret finger bodies from crossing
                       # even when the wrist roll that would tilt them apart is
                       # averaged out by the pose smoothing across a chord run
# After a note, a real hand does NOT snap the finger straight - that is tiring
# and unnatural. It lifts the tip just off the string and lets the finger LINGER
# in a relaxed curl, only returning to the neutral idle lane when the curl would
# foul a coming press. So a released finger eases up to its own arched hover
# (retaining the curl) over LINGER_LIFT_FR frames and holds that pose; the idle
# branch keeps holding it as long as it stays LINGER_CLEAR clear of the pressers.
# A chord that is notated as ringing right up to the instant the next chord lands
# leaves the incoming fingers no lane to move through: they end up crammed into
# the last frame (the fret-hand's remaining snap). A real player lifts the old
# grip a shade EARLY to change - the strings keep ringing - so a grip's fingers
# release this many frames before the next grip's onset when the notation would
# otherwise hold them to the last instant.
GRIP_LIFT_LEAD_FR = 2.5
LINGER_LIFT_FR = 3.0   # frames to ease a released finger up into its relaxed curl
LINGER_CLEAR = 0.001   # m: min SURFACE clearance a lingering curl keeps from a presser
# A lingering finger must also keep the natural neck ORDER: a lower-numbered
# finger stays nut-ward (higher flat y) of a higher-numbered one. A finger that
# would linger on the WRONG side of a pressing neighbour (e.g. the middle left
# behind at fret 2 while the index presses fret 3 - a finger crossing no player
# makes) is moved out of the way to its ordered idle lane instead. The margin
# tolerates near-same-fret overlap without tripping on a genuine cross.
NECK_ORDER_MARGIN = 0.008  # m of along-neck overlap allowed before it reads as a cross
# A press must not be crammed into the last frame: the descent onto the fret
# (hover -> press) needs a distance-independent runway so the fingertip eases
# down instead of slamming (the acceleration spike). When a reaching finger's
# hover and press both clear the still-pressing neighbour, the search may start
# the descent this many frames early rather than being pinned to that release.
PRESS_LEAD_FR = 3.0    # min eased frames for the hover -> press descent on a reach
# Anticipation: a real fretting hand does not hold a finger still until the last
# instant and then snap it onto the next note - it starts repositioning the
# finger (and wrist) in advance and eases it into place. These give every finger
# transition a distance-independent minimum runway so the pose change spreads
# over several eased frames instead of one, killing the acceleration spikes.
SETTLE_FR = 1.0        # frames a finger holds the hover just before it presses
# How long a finger gets to glide from its idle lane into the hover over its next
# fret. A FIXED lead cannot serve both cases: a neighbouring fret is a few mm away,
# but a chord change can throw a finger the width of the neck (the strum demo's
# D -> C shape moves the ring finger ~100 mm), and cramming that into four frames
# is the one snap left after the easing pass. So the lead grows with how far the
# fingertip actually has to travel, up to REACH_LEAD_MAX - still bounded by when
# the previous grip frees the lane.
REACH_LEAD_FR = 4.0    # frames for a short reach (a neighbouring fret)
REACH_LEAD_MAX = 11.0  # frames for a full cross-neck reposition
REACH_LEAD_DIST = 0.06  # m of fingertip travel that earns the full lead
CROSS_GLIDE_FR = 4.0   # frames a finger takes crossing straight from one press
                       # to the next (it lifts off the old fret this early)

# An idle (non-pressing) fret finger holds a RELAXED ARCH in its own fret lane,
# NOT a reach to a hover point, and never the piano rig's flat RELAXED pose
# (which lay right across the strings, the idle fingertips dipping BELOW the
# string plane between chords). The knuckles ride the treble edge spread ALONG
# the neck (one per fret, index nut-side, pinky bridge-side), so the fingers are
# already in their own lanes; an idle finger only crosses a neighbour if it
# reaches ACROSS or FORWARD out of that lane.
#
# The pose is authored DIRECTLY as gentle joint angles rather than solved by IK
# to a target near the strings. Solving to a target is what produces a tense
# "claw": a ~90 mm finger told to touch a point only ~35 mm from its own knuckle
# (just over the strings at its own fret) can only get there by clenching the PIP
# to ~130 deg. Holding a fixed arch instead keeps the fingertip in its own lane
# (past the knuckle, not curled under the palm), so it neither reaches across a
# neighbour nor needs any wrist yaw to make room.
#
# The MCP is tilted DOWN at the knuckle (rather than lifted back) so the whole
# arch points toward the strings and the hand sits READY over them - a press then
# only glides down a hair instead of dropping from high like a hammer-on. It only
# tilts the arch as a rigid unit; the PIP/distal curl (the arch SHAPE) is
# unchanged, so the tip stays in its own lane and clear of the strings.
IDLE_PROX = 0.10           # relaxed idle MCP (proximal) angle: aimed at the board
IDLE_MID = 1.26            # relaxed idle PIP (middle) flex - a natural ~72 deg arch
IDLE_DIST_FLEX = 0.26      # distal curl of an idle finger
IDLE_CLEAR_TARGET = 0.008  # surface clearance (m) from a presser that counts as clear
# When a neighbour presses close by, the idle finger gets out of the way two ways,
# and the search below picks the cheapest mix that clears the presser:
#  - it LEANS ASIDE at the knuckle (a small sympathetic MCP yaw), the way a real
#    hand's fingers drift together when one abducts rather than one finger
#    swinging across a still neighbour, and
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
# A finger with no note of its own does not hang out in its own string lane -
# that leaves it splayed straight across to a string nobody is playing (it reads
# as an outstretched finger frozen in place). A real player's unused fingers ride
# WITH the hand, curled just over the string(s) being played. So an idle finger is
# aimed at the group's AVERAGE across-string position (the mean fingertip x of
# whoever is pressing), kept in its own along-neck lane, hovering IDLE_HOVER above
# the strings. Reaching that nearer string means curling in more, which also stops
# it looking outstretched.
IDLE_HOVER = 0.008         # m the idle fingertip hovers above the string plane


def _idle_hover_pose(f, target, rmat, across_target):
    """(yaw, prox, mid) placing idle finger ``f``'s fingertip over the group's
    average string (``across_target``) at its own along-neck lane, hovering just
    above the strings. Solved with the same two-link IK a press uses, but aimed
    at a point lifted off the strings and slid across to the played string."""
    spec = FRET_FINGERS[f]
    # Keep the finger in its own lane: reach the along-neck (y) its neutral arch
    # already points to, only sliding the tip across (x) to the played string and
    # up (z) to a hover. Aiming at its own y avoids bunching every idle finger at
    # the pressed fret.
    lane_y = _finger_fk_from_angles(target, rmat, spec, 0.0, IDLE_PROX, IDLE_MID,
                                    IDLE_DIST_FLEX)[3].y
    knuckle = mathutils.Vector(target) + rmat @ mathutils.Vector(spec["knuckle"])
    tgt = mathutils.Vector((across_target, lane_y,
                            fret_layout.STRING_Z + IDLE_HOVER))
    local = rmat.transposed() @ (tgt - knuckle)
    return _fret_ik(local.x, local.y, -local.z, spec["lengths"], IDLE_DIST_FLEX)


def _idle_finger_pose(f, target, rot, press_chains=(), across_target=None,
                      n_press=None):
    """(yaw, prox, mid) for a relaxed idle finger.

    When notes are being fretted, ``across_target`` is the mean across-string x
    of the pressing fingertips: the idle finger curls in to hover over that same
    string (see IDLE_HOVER), riding with the hand instead of splaying out into
    its own lane. That hover pose is used as long as it stays clear of the
    pressers.

    Otherwise - a chord grip, or the hover pose would foul a presser - it falls
    back to the loose neutral arch (IDLE_PROX/IDLE_MID), searching a small grid
    of sympathetic knuckle LEANS (_IDLE_YAW_GRID) and PIP CURLS (_IDLE_MID_GRID)
    for the pose that clears the presser closest to neutral, so it drifts aside
    rather than a neighbour swinging across it.

    ``press_chains`` are (finger, world chain) obstacles: the pressing fingers
    plus any idle neighbour already solved for this event, so two idle fingers
    never settle into each other. ``n_press`` is how many of those are actual
    presses (the hover branch is for near-monophonic grips only)."""
    rmat = _fret_rotation(*rot)
    spec = FRET_FINGERS[f]
    if n_press is None:
        n_press = len(press_chains)
    # The average-string hover is used only in near-monophonic grips (at most one
    # other finger pressing). When two or more fingers press (a chord) the mean
    # string sits between them and curling an idle finger there crowds its
    # neighbours, so it keeps the neutral lane instead.
    if across_target is not None and n_press <= 1:
        hover = _idle_hover_pose(f, target, rmat, across_target)
        if _angles_clear(f, target, rot,
                         (hover[0], hover[1], hover[2], IDLE_DIST_FLEX),
                         press_chains):
            return hover
    if not press_chains:
        return 0.0, IDLE_PROX, IDLE_MID
    best = None
    for yaw in _IDLE_YAW_GRID:
        for mid in _IDLE_MID_GRID:
            chain = _finger_fk_from_angles(target, rmat, spec, yaw, IDLE_PROX,
                                           mid, IDLE_DIST_FLEX)
            clr = min(_chain_clearance(f, chain, g, pc)
                      for g, pc in press_chains)
            # Clear the presser up to the target; once clear, sit as close to the
            # relaxed neutral (no lean, loosest arch) as possible.
            score = (min(clr, IDLE_CLEAR_TARGET)
                     - IDLE_YAW_COST * abs(yaw)
                     - IDLE_CURL_COST * (mid - IDLE_MID))
            if best is None or score > best[0]:
                best = (score, yaw, mid)
    return best[1], IDLE_PROX, best[2]


# Per-event wrist rotation freedom, chosen by a grid search that penalizes
# predicted finger-finger collisions (forward kinematics of the pressing
# fingers). Yaw turns the hand in the fretboard plane and is locked to the
# neutral while barring (the flattened index must stay parallel to the fret
# wire); roll spins about the reach axis - safe during barres - lifting one side
# of the knuckle line so a far-reaching finger arches OVER its neighbour instead
# of slicing through it.
WRIST_YAW_MAX = 0.56
WRIST_YAW_STEP = 0.08
WRIST_YAW_REG = 0.5    # cost per rad^2 off the neutral yaw - prefer a neutral wrist
# The fretting forearm reaches the neck from a roughly fixed elbow by the body, so
# a hand held STRICTLY perpendicular to the neck has to bend the wrist ever more
# sharply toward either end of the fretboard. A real player relieves that by
# letting the whole hand ROTATE to follow the forearm, trading wrist deviation for
# a little yaw. So the wrist search relaxes toward a neck-position-dependent
# NEUTRAL yaw instead of toward a fixed square-on zero: ~0 near NEUTRAL_YAW_Y
# (where the wrist is naturally straight), yawing NEGATIVE reaching toward the nut
# (higher flat-frame y) and POSITIVE toward the high frets. It follows only a
# FRACTION of the full forearm angle and is capped, so the hand still reads as
# roughly square to the neck -- just no longer robotically so, and it stops
# bending the wrist hardest exactly where the reach is already longest.
NEUTRAL_YAW_Y = 0.46       # flat-frame neck y where the fret wrist sits naturally straight
#                            (~the middle of a 25.5" neck: the wrist spans ~0.30 at the
#                            12th fret to ~0.62 at the 1st)
NEUTRAL_YAW_SLOPE = 0.65   # rad of neutral yaw per metre the hand reaches off NEUTRAL_YAW_Y
NEUTRAL_YAW_MAX = 0.28     # cap (rad, ~16 deg) that keeps the rotation "a little"
# Reaching HIGH on the neck (toward the nut / headstock), a relaxed player lets the
# wrist hang lower and TRAIL behind the reaching fingers instead of holding it up
# level with the neck. The trail is applied along the finger REACH AXIS (see
# _fret_event_pose), so it re-extends the fingers to keep the fingertips on their
# frets while the wrist drops and backs off -- rather than a flat-Y slide, which
# would deviate the knuckle sideways of its target and force a spurious MCP splay.
# Applied only above WRIST_RELAX_Y and only a little.
WRIST_RELAX_Y = NEUTRAL_YAW_Y   # start relaxing the wrist down above this neck position
WRIST_RELAX_DROP = 0.09         # metres the wrist trails per metre above it
WRIST_RELAX_MAX = 0.034         # cap on the trail/drop (m), keeping it slight
WRIST_ROLL_MAX = 0.42
WRIST_ROLL_STEP = 0.07
WRIST_ROLL_REG = 0.3
WRIST_COHERE = 2.0     # per rad^2 of pose change from the previous event,
                       # fading out over 1.5 s - stops the wrist flipping
                       # between opposite extremes on back-to-back chords
                       # (the interpolating fingers would cross mid-swing)
TOUCH_CLEAR = 0.001    # m of air between two pressing fingers' surfaces
COLLIDE_W = 25000.0    # per m^2 of surface-clearance deficit between fingers
# The wrist search also gives the IDLE fingers room: it penalizes rotations that
# park a pressing finger's box close to where an idle finger's default arch sits
# (their proximals near the shared treble edge). Softer than the press-press term
# so it only nudges the wrist when a real gap is available, and the per-finger
# idle search still does the fine clearing on top.
WRIST_IDLE_CLEAR = 0.003   # idle-vs-press surface clearance the wrist search opens
IDLE_COLLIDE_W = 9000.0    # weight of the idle-clearance term (< COLLIDE_W)
# Biomechanical anti-crossing. A finger crossing UNDER a neighbour (reaching under
# an idle finger to a bass string) is, at root, a finger deviating too far SIDEWAYS
# at the knuckle: real MCP abduction/adduction tops out around 25 deg (Thieme 2024;
# AAOS goniometry), yet the raw IK was splaying pressing fingers up to 45 deg on
# the demo. So the finger IK's yaw is hard-capped at FINGER_MCP_SPLAY, and the
# wrist search PENALIZES any pose that would demand more splay than that from a
# pressing finger - which pushes the hand (wrist yaw/roll and slide) to do the
# reaching instead, exactly as a real player angles the whole hand for a stretched
# chord rather than swinging one finger under the others. A second cap forbids MCP
# hyperextension past its ~25-30 deg norm, so a finger stretching for a far string
# bends at the knuckle rather than bowing backward.
FINGER_MCP_SPLAY = math.radians(26.0)    # max abduction/adduction of a fret finger
FINGER_MCP_HYPEREXT = math.radians(28.0)  # max backward (extension) tilt at the MCP
SPLAY_W = 8000.0            # per rad^2 a presser's target demands beyond the splay cap
HYPEREXT_W = 3000.0         # per rad^2 a presser's MCP is driven past its extension cap

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

# --- picking-wrist radial/ulnar deviation ----------------------------------
# The stroke above is authored as pure TRANSLATION of the PickHand rig: the hand
# used to slide across the strings at a fixed, solved orientation (PICK_ROT), so
# the wrist never moved relative to the forearm. Real picking is not a shoulder
# slide -- a sizeable part of the stroke is the wrist rocking in the plane of the
# palm, toward the thumb (RADIAL) on the backswing and toward the little finger
# (ULNAR) through the follow-through. Adding a small rock of that kind is what
# makes the hand read as picking rather than being dragged.
#
# The deviation axis is the PALM NORMAL (the hand rig's local -z, which PICK_ROT
# turns onto the strings); rotating about it swings the fingers/pick sideways
# WITHIN the palm plane, which is exactly radial/ulnar deviation -- as opposed to
# the knuckle axis (flexion/extension) or the finger axis (forearm roll). NB the
# bass rig's like-named PICK_ULNAR is a dimensionless tilt of its pendulum axis;
# here the hand carries no swing bone, so this is the deviation angle itself.
#
# It is a COMPONENT of the stroke, not an addition to it: animate_pick_hand pins
# the pick TIP to the same contact points as before and solves the rig's location
# around whatever the deviation angle is, so the rock replaces part of the slide
# instead of throwing the tip off the string. Half-amplitude is kept small -- at
# 7 deg the tip sweeps ~3.4 mm to either side (the swing radius, wrist to pick
# tip, is ~64 mm), a third or so of a single-note stroke's crossing distance, and
# the wrist bend stays far inside the _check_wrist_pose envelope.
PICK_ULNAR = math.radians(7.0)   # half-swing of the picked stroke's wrist rock
STRUM_ULNAR = math.radians(9.0)  # ditto for a full strum (the arm still leads it)
PICK_ULNAR_VEL = 0.5   # fraction of the half-swing that scales with velocity
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


def _angles_clear(f, wrist, rot, angles, press_chains):
    """True if finger ``f`` posed at the given (yaw, prox, mid, dist) joint angles
    at this wrist/(yaw,roll) keeps at least LINGER_CLEAR from every pressing
    finger's world chain. Used to decide whether a released finger may LINGER in
    its relaxed curl (rather than straighten to the idle lane), and whether a
    reaching finger's descent clears a still-pressing neighbour. Both the posed
    finger and the press chains are in world space, so the comparison is valid
    even across a hand-position change."""
    if not press_chains:
        return True
    ch = _finger_fk_from_angles(wrist, _fret_rotation(*rot), FRET_FINGERS[f],
                                *angles)
    return min(_chain_clearance(f, ch, g, pc)
               for g, pc in press_chains) >= LINGER_CLEAR


def _finger_fk(wrist, rot, spec, tip, flex):
    """Predicted world joint points [knuckle, prox, mid, tip] of a
    finger posed by the IK to reach ``tip`` at this wrist pose - the same
    geometry _pose_finger will bake, reconstructed for collision checks."""
    ko = rot @ mathutils.Vector(spec["knuckle"])
    knuckle = mathutils.Vector(wrist) + ko
    local = rot.transposed() @ (mathutils.Vector(tip) - knuckle)
    yaw, prox, mid = _fret_ik(local.x, local.y, -local.z,
                              spec["lengths"], flex)
    return _finger_fk_from_angles(wrist, rot, spec, yaw, prox, mid, flex)


def _pose_cost(press, wrist, rot, idle=()):
    """IK strain, biomechanical-limit penalties, and predicted finger-collision
    penalty of one pose. ``idle`` are the non-pressing fingers, whose default
    arch is kept clear of the pressing fingers too (a softer penalty than
    press-press)."""
    cost = 0.0
    inv = rot.transposed()
    chains = []
    for n, is_barre in press:
        spec = FRET_FINGERS[n["finger"]]
        flex = DIST_FLEX_BARRE if is_barre else DIST_FLEX_PRESS
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
        _, prox, _ = _fret_ik(local.x, local.y, -local.z, spec["lengths"], flex)
        hyper = -prox - FINGER_MCP_HYPEREXT   # prox < 0 == MCP bending backward
        if hyper > 0.0:
            cost += HYPEREXT_W * hyper * hyper
        chains.append((n["finger"],
                       _finger_fk(wrist, rot, spec,
                                  (n["x"], n["y"], n["z"]), flex)))
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            dmin = _chain_clearance(chains[i][0], chains[i][1],
                                    chains[j][0], chains[j][1])
            if dmin < TOUCH_CLEAR:
                cost += COLLIDE_W * (TOUCH_CLEAR - dmin) ** 2
    for f in idle:
        spec = FRET_FINGERS[f]
        # The idle finger's actual relaxed arch (see _idle_finger_pose), so the
        # wrist search reads where it truly sits - in its own lane - and is not
        # driven to yaw the whole hand to clear a reach the finger never makes.
        ich = _finger_fk_from_angles(wrist, rot, spec, 0.0, IDLE_PROX,
                                     IDLE_MID, IDLE_DIST_FLEX)
        for g, pc in chains:
            dmin = _chain_clearance(f, ich, g, pc)
            if dmin < WRIST_IDLE_CLEAR:
                cost += IDLE_COLLIDE_W * (WRIST_IDLE_CLEAR - dmin) ** 2
    return cost


def _fret_event_pose(event, prev_pose=None, dt=None):
    """(wrist location, yaw, roll) for one fretted event, from a grid
    search over the wrist's rotation freedom that trades IK strain, the
    anatomical joint caps, a neutral wrist, and coherence with the previous
    event's pose against predicted finger-finger collisions. Barre events
    hold the neutral yaw (the bar must stay parallel to the fret wire) but
    keep roll, which spins about the bar's own axis."""
    press = _barre_press_notes(event)
    has_barre = any(b for _, b in press)
    knuckle_z = KNUCKLE_Z_BARRE if has_barre else KNUCKLE_Z
    if not press:
        return _solve_wrist(press, _fret_rotation(0, 0), knuckle_z), 0.0, 0.0
    pressed_fingers = {n["finger"] for n, _ in press}
    idle = [f for f in FRET_FINGERS if f not in pressed_fingers]

    # Relax the wrist yaw toward the forearm-following neutral for this neck
    # position (not a fixed square-on zero), read from the base-pose wrist y so it
    # doesn't feed back on the yaw being searched.
    base_y = _solve_wrist(press, _fret_rotation(0, 0), knuckle_z)[1]
    ny = _neutral_yaw(base_y)

    cohere = 0.0
    if prev_pose is not None and dt is not None:
        cohere = WRIST_COHERE * max(0.0, 1.0 - dt / 1.5)
    ysteps = int(round(WRIST_YAW_MAX / WRIST_YAW_STEP))
    rsteps = (int(round(WRIST_ROLL_MAX / WRIST_ROLL_STEP))
              if len(press) >= 2 else 0)
    best = None
    for ri in range(-rsteps, rsteps + 1):
        roll = ri * WRIST_ROLL_STEP
        # A barre keeps the bar parallel to the wire: no free yaw, but it still
        # follows the neck-position neutral so the wrist isn't cranked at the
        # ends of the neck (the whole hand, bar included, turns as one).
        yaws = [ny] if has_barre else [yi * WRIST_YAW_STEP
                                       for yi in range(-ysteps, ysteps + 1)]
        for yaw in yaws:
            rot = _fret_rotation(yaw, roll)
            wrist = _solve_wrist(press, rot, knuckle_z)
            cost = (WRIST_YAW_REG * (yaw - ny) ** 2
                    + WRIST_ROLL_REG * roll * roll
                    + _pose_cost(press, wrist, rot, idle))
            if cohere:
                cost += cohere * ((yaw - prev_pose[0]) ** 2
                                  + (roll - prev_pose[1]) ** 2)
            if best is None or cost < best[0]:
                best = (cost, wrist, yaw, roll)
    wrist = best[1]
    # Let the wrist hang lower / trail the reach high on the neck. A barre is
    # left alone: the bar has to stay pressed flat across every barred string,
    # and trailing it would lift one end off the wire.
    drop = 0.0 if has_barre else _wrist_relax_drop(wrist[1])
    if drop and len(press) == 1:
        # SINGLE-finger grip: trail the wrist back along the finger REACH AXIS
        # (-rest_dir), not flat -Y. A raw flat-Y slide moves the knuckle sideways
        # of its target in the wrap-tilted finger frame, which the IK reads as a
        # large MCP splay (the lone pressing finger then hits the anatomical cap
        # and jams into its idle neighbour). Backing off along the reach axis
        # instead is pure finger EXTENSION - exactly the "re-extend to keep the
        # fingertips on their frets" the relaxation is meant to be - so the finger
        # stays perpendicular while the wrist still hangs lower and trails.
        rest = _fret_rotation(best[2], best[3]) @ mathutils.Vector((0.0, 1.0, 0.0))
        wrist = (wrist[0] - drop * rest.x,
                 wrist[1] - drop * rest.y,
                 wrist[2] - drop * rest.z)
    elif drop:
        # Multi-finger grip (a chord): keep the flat-Y trail. Its fingers sit at
        # different frets and reach apart; the flat-Y slide happens to separate
        # them (it extends them toward the nut), which the reach-axis trail
        # would not, so a tight chord stays clear.
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
    event_press = [{n["finger"]: (n, is_barre)
                    for n, is_barre in _barre_press_notes(ev)} for ev in events]
    # Mean across-string x of the fingers pressing each event - where the idle
    # fingers hover so they ride over the played string(s) (see _idle_finger_pose).
    event_across = [(sum(n["x"] for n, _ in d.values()) / len(d)) if d else None
                    for d in event_press]
    event_press_chains = []
    for i, (target, rot) in enumerate(zip(targets, rots)):
        rmat = _fret_rotation(*rot)
        ch = {}
        for f, (n, is_barre) in event_press[i].items():
            ch[f] = _finger_fk(target, rmat, FRET_FINGERS[f],
                               (n["x"], n["y"], n["z"]),
                               DIST_FLEX_BARRE if is_barre else DIST_FLEX_PRESS)
        event_press_chains.append(ch)

    def other_press_chains(f, i):
        """(finger, chain) of the fingers pressing event i other than f (the
        ones an idle/holding f must stay clear of)."""
        return [(g, c) for g, c in event_press_chains[i].items() if g != f]

    def linger_ok(f, target, rot, angles, i):
        """Whether finger f may LINGER in its relaxed curl at event i: it must
        both clear the pressers (3-D) AND keep the natural neck ORDER - a
        lower-numbered finger stays nut-ward (higher flat y) of a higher one. A
        finger that would sit on the wrong side of a presser (a crossing) is
        refused so it relaxes to its ordered idle lane and moves out of the way
        instead."""
        if not _angles_clear(f, target, rot, angles, other_press_chains(f, i)):
            return False
        ftip = _finger_fk_from_angles(target, _fret_rotation(*rot),
                                      FRET_FINGERS[f], *angles)[3]
        for g, ch in event_press_chains[i].items():
            if g == f:
                continue
            d = ftip.y - ch[3].y   # >0: f is nut-ward of g
            if (f - g) * d > 0.0 and abs(d) > NECK_ORDER_MARGIN:
                return False       # f on the wrong side of g -> crossing
        return True

    # When the PREVIOUS event's grip releases (its pressing fingers lift): an
    # approaching finger must not reach across the neck until then, or it drifts
    # through a finger still holding the previous grip. The finger f itself is
    # EXCLUDED: a finger never collides with its own just-vacated fret, so a
    # finger re-fretting soon is free to lift off and glide straight across (see
    # CROSS_GLIDE_FR) rather than being pinned by its own note-off. frame_start
    # for the first event.
    def grip_release(i, note):
        """When event i's grip actually lifts its fingers: the notated note-off,
        pulled back to GRIP_LIFT_LEAD_FR before the next grip's onset when the
        notation sustains right up to it (a real chord change lifts early), never
        before the note has had a frame of sustain."""
        off = max(to_frame(note["start"]), to_frame(note["end"]))
        if i + 1 < len(events):
            off = min(off, to_frame(events[i + 1]["t"]) - GRIP_LIFT_LEAD_FR)
        return max(off, to_frame(note["start"]) + 0.5)

    def prev_release_for(f, i):
        if i == 0:
            return frame_start
        ends = [grip_release(i - 1, nn) for g, (nn, _) in event_press[i - 1].items()
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

    # Idle poses are solved ONE FINGER AT A TIME, index to pinky, each against the
    # pressing fingers AND the idle neighbours already solved for that event - so
    # two idle fingers can't settle into each other (the chord-free stretches,
    # where nobody presses and every finger just holds its arch, otherwise let
    # neighbouring arches overlap). Cached per (finger, event, hover-or-lane).
    idle_cache = {}

    def idle_angles(f, i, target, rot, across_target):
        key = (f, i, across_target is not None)
        if key in idle_cache:
            return idle_cache[key][0]
        n_press = len(event_press_chains[i]) - (1 if f in event_press[i] else 0)
        obstacles = other_press_chains(f, i)
        for g in FRET_FINGERS:
            if g >= f or g in event_press[i]:
                continue
            idle_angles(g, i, target, rot, across_target)
            obstacles.append((g, idle_cache[(g, i, across_target is not None)][1]))
        ang = _idle_finger_pose(f, target, rot, obstacles, across_target, n_press)
        chain = _finger_fk_from_angles(target, _fret_rotation(*rot),
                                       FRET_FINGERS[f], ang[0], ang[1], ang[2],
                                       IDLE_DIST_FLEX)
        idle_cache[key] = (ang, chain)
        return ang

    def key_idle(f, i, target, rot, frame, across_target):
        y, p, m = idle_angles(f, i, target, rot, across_target)
        _pose_finger(pbones, f, y, p, m, IDLE_DIST_FLEX, frame)

    last_frame = frame_start
    for f in FRET_FINGERS:
        spec = FRET_FINGERS[f]
        # Start idle hovering over the first event's played string.
        key_idle(f, 0, targets[0], rots[0], frame_start, event_across[0])
        prev_end = frame_start
        prev_tr = None   # (target, rot) of the previous event, for holding idle
        linger = None    # relaxed curl a just-released finger holds
        for i, (ev, target, rot) in enumerate(zip(events, targets, rots)):
            onset = to_frame(ev["t"])
            if f not in event_press[i]:
                # Hold the previous idle pose until the previous grip releases,
                # so a finger idle across two grips doesn't drift over a
                # still-pressing neighbour during the first grip's sustain. Only
                # when the previous grip sustains well past this event's arrival,
                # so a tight reach-in transition is left to glide. Skipped while
                # the finger is lingering in a curl - that curl IS its held pose.
                hold_prev = prev_release_for(f, i) - 0.5
                if (linger is None and prev_tr is not None
                        and prev_end + 0.75 < hold_prev < onset - 3.0):
                    # A transitional hold - relax to the neutral lane, not a hover.
                    key_idle(f, i - 1, prev_tr[0], prev_tr[1], hold_prev, None)
                    prev_end = hold_prev
                # Prefer to LINGER in the relaxed curl the finger lifted into
                # after its last note (a real hand rests its fingers curled, not
                # straight) as long as that curl stays clear of whoever is
                # pressing now; only then straighten out to the neutral idle lane.
                frame = max(onset, prev_end + 0.5)
                if linger is not None and linger_ok(f, target, rot, linger, i):
                    _pose_finger(pbones, f, *linger, frame)
                else:
                    key_idle(f, i, target, rot, frame, event_across[i])
                    linger = None
                prev_end = frame
                prev_tr = (target, rot)
                continue

            n, is_barre = event_press[i][f]
            flex_press = DIST_FLEX_BARRE if is_barre else DIST_FLEX_PRESS
            flex_hover = DIST_FLEX_BARRE_HOVER if is_barre else DIST_FLEX_HOVER
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
            pressed = _fret_ik(dx, dy, dv, spec["lengths"], flex_press)
            dx, dy, dv = ik_inputs(n["z"] + HOVER_LIFT)
            hover = _fret_ik(dx, dy, dv, spec["lengths"], flex_hover)

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
                # back and forth.
                hover_frame = on_frame - SETTLE_FR
                hover_frame = max(hover_frame, prev_end + 0.5, pr)
                hover_frame = min(hover_frame, on_frame - 0.5)
                hover_frame = max(hover_frame, prev_end + 0.5)
            else:
                # REACH: coming from the idle lane (or a lingering curl). Give the
                # descent a real runway: if this finger's hover AND press both
                # clear the still-pressing neighbour, the reach need not wait for
                # that neighbour to release (pr) - it may start the eased descent
                # PRESS_LEAD_FR frames early instead of being crammed into the
                # last frame (the "slam onto the fret" acceleration spike). Only
                # when the descent would actually cross a held neighbour is the pr
                # floor kept.
                prev_chains = ([(g, c) for g, c in event_press_chains[i - 1].items()
                                if g != f] if i > 0 else [])
                reach_ok = (
                    _angles_clear(f, target, rot,
                                  (hover[0], hover[1], hover[2], flex_hover),
                                  prev_chains)
                    and _angles_clear(f, target, rot,
                                      (pressed[0], pressed[1], pressed[2],
                                       flex_press), prev_chains))
                floor = prev_end + 0.5 if reach_ok else max(prev_end + 0.5, pr)
                if not reach_ok:
                    # Hold the finger's PREVIOUS idle pose (previous event's hand
                    # frame) until that grip releases, so it doesn't drift toward
                    # this press during the neighbour's sustain.
                    hold_prev = pr - 0.5
                    if (prev_tr is not None
                            and prev_end + 0.75 < hold_prev < on_frame - 1.0):
                        key_idle(f, i - 1, prev_tr[0], prev_tr[1], hold_prev, None)
                        prev_end = hold_prev
                # The finger PARKS in its own clear lane, then makes two moves: the
                # REACH across to over its fret, and the DESCENT onto the string.
                # Both need runway, so the time between the finger's last key and
                # the onset is SPLIT between them (proportionally when it is too
                # short for both). Placing the hover as early as the attack lead
                # allowed - the obvious reading - is what left a long cross-neck
                # reach crammed into the last 3/4 of a frame while the finger then
                # sat at the hover for five: the reach, not the descent, is the big
                # move. Keep the relaxed curl as the park pose if the finger was
                # lingering and it stays clear, so it does not straighten only to
                # re-curl for the press.
                if linger is not None and linger_ok(f, target, rot, linger, i):
                    park = linger
                else:
                    park = idle_angles(f, i, target, rot, None) + (IDLE_DIST_FLEX,)
                travel = (_finger_fk_from_angles(target, rmat, spec, *park)[3]
                          - _finger_fk_from_angles(
                              target, rmat, spec, hover[0], hover[1], hover[2],
                              flex_hover)[3]).length
                want_reach = (REACH_LEAD_FR + (REACH_LEAD_MAX - REACH_LEAD_FR)
                              * min(1.0, travel / REACH_LEAD_DIST))
                want_desc = max(attack_frames(n), PRESS_LEAD_FR)
                start = max(floor, prev_end + 0.5)
                runway = on_frame - start
                if runway >= want_reach + want_desc:
                    hover_frame = on_frame - want_desc
                    hold_frame = hover_frame - want_reach
                else:
                    hover_frame = on_frame - max(
                        runway * want_desc / (want_reach + want_desc), 0.75)
                    hold_frame = start
                hover_frame = min(max(hover_frame, start), on_frame - SETTLE_FR)
                hold_frame = min(hold_frame, hover_frame - 0.75)
                # The parking key stays DISTINCT from - and just before - the hover
                # so the long pre-reach span can't interpolate straight through a
                # neighbour.
                if hold_frame > prev_end + 0.75:
                    _pose_finger(pbones, f, *park, hold_frame)
                    prev_end = hold_frame

            # If this same finger presses again very soon, it must lift off THIS
            # fret early to glide across in time (anticipation), and it should
            # NOT relax to the idle lane in between - the next event's late hover
            # is its next key, so it glides straight from one press to the next.
            presses_next_soon = (
                i + 1 < len(events) and f in event_press[i + 1]
                and to_frame(events[i + 1]["t"]) - off_frame < CROSS_GLIDE_FR)

            _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                         flex_hover, hover_frame)
            _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                         flex_press, on_frame)

            if presses_next_soon:
                next_on = to_frame(events[i + 1]["t"])
                pressed_end = min(off_frame, next_on - CROSS_GLIDE_FR)
                pressed_end = max(pressed_end, on_frame + 0.5)
                _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                             flex_press, pressed_end)
                prev_end = pressed_end
                linger = None   # stays engaged; glides straight to the next press
            else:
                # RELEASE. A finger left curled at its played fret would CROSS
                # whoever presses next if that presser lands on the wrong side of
                # it along the neck. So decide up front:
                hov_ang = (hover[0], hover[1], hover[2], flex_hover)
                nxt = i + 1 if i + 1 < len(events) else None
                vacate = (nxt is not None
                          and not linger_ok(f, targets[nxt], rots[nxt],
                                            hov_ang, nxt))
                if not vacate:
                    # LINGER: hold the note its full length, then lift the
                    # fingertip just off the string into the SAME relaxed curl
                    # (its own arched hover, ~HOVER_LIFT up) - NOT straight to the
                    # idle lane. A real hand rests its fingers curled after a note;
                    # snapping them straight is tiring and reads wrong. Because the
                    # lift only rises in place it can overlap the next grip freely,
                    # so the note keeps its full sustain.
                    pressed_end = grip_release(i, n)
                    lift_frame = pressed_end + LINGER_LIFT_FR
                    _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                                 flex_press, pressed_end)
                    _pose_finger(pbones, f, hover[0], hover[1], hover[2],
                                 flex_hover, lift_frame)
                    linger = hov_ang
                    prev_end = lift_frame
                else:
                    # VACATE: get out of the way. Relax to the ordered idle lane,
                    # arriving before the next onset so the finger has cleared the
                    # spot the next presser needs (no hanging at the old fret).
                    relax_frame = off_frame + LINGER_LIFT_FR
                    nxt_grip = next_onset_after(off_frame)
                    if nxt_grip is not None:
                        relax_frame = min(relax_frame, nxt_grip - 1.0)
                    # Start the lift early enough to ease over the full runway
                    # (even a little before the notated note-off when the next
                    # grip lands at once), so vacating doesn't slam.
                    pressed_end = min(grip_release(i, n),
                                      relax_frame - LINGER_LIFT_FR)
                    pressed_end = max(pressed_end, on_frame + 0.5)
                    relax_frame = max(relax_frame, pressed_end + 1.0)
                    _pose_finger(pbones, f, pressed[0], pressed[1], pressed[2],
                                 flex_press, pressed_end)
                    # Vacating = get OUT of the way; relax to the neutral lane
                    # rather than hover over the string just left.
                    key_idle(f, i, target, rot, relax_frame, None)
                    linger = None
                    prev_end = relax_frame
            prev_tr = (target, rot)
            last_frame = max(last_frame, off_frame + release_frames)

    # Bezier with auto-clamped handles so every finger and the wrist EASE like a
    # real hand: speed goes to zero at each rest key and peaks mid-move, instead
    # of the one-sided, constant-rate ramp SINE gives (which let a finger arrive
    # at full speed and then dead-stop - the acceleration spike). Same easing the
    # pick hand uses; clamped handles keep a press from overshooting.
    for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                       and arm_obj.animation_data.action):
        for kp in fcurve.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'AUTO_CLAMPED'
            kp.handle_right_type = 'AUTO_CLAMPED'
        fcurve.update()
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
    arm_obj.rotation_euler = PICK_ROT.to_euler()
    tip_off = mathutils.Vector(pick_world_offset(PICK_TIP_LOCAL))
    # Deviation axis: the palm normal (hand-local -z), signed so that a POSITIVE
    # angle carries the pick tip toward the treble side -- i.e. along a
    # downstroke, so `dev` can be keyed straight off the stroke direction.
    dev_axis = -mathutils.Vector(pick_world_offset((0.0, 0.0, 1.0)))
    tip_y = fret_layout.PLUCK_Y
    z_pluck = fret_layout.STRING_Z - PICK_DEPTH

    def hover_z(lift):  # pick-tip clearance -> tip z for a hover apex
        return fret_layout.STRING_Z + lift

    def key(t, tip_x, z, dev):
        # Rock the wrist by `dev` about the palm normal, then solve the rig's
        # location so the pick TIP still lands on the point the stroke asked
        # for: the deviation takes over part of the crossing motion rather than
        # displacing the contact.
        rot = mathutils.Matrix.Rotation(dev, 3, dev_axis)
        arm_obj.rotation_euler = (rot @ PICK_ROT).to_euler('XYZ',
                                                          arm_obj.rotation_euler)
        arm_obj.location = mathutils.Vector((tip_x, tip_y, z)) - rot @ tip_off
        frame = frame_start + t * fps
        arm_obj.keyframe_insert(data_path="location", frame=frame)
        arm_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    events = _group_events(notes)
    if not events:
        return frame_start

    directions = _pick_directions(events)
    min_dt = 0.75 / fps  # keep successive keyframes distinct
    last_t = None

    def key_after(t, tip_x, z, dev):
        nonlocal last_t
        if last_t is not None:
            t = max(t, last_t + min_dt)
        key(t, tip_x, z, dev)
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
        # The wrist rock runs with the stroke: cocked RADIAL at the backswing
        # apex, sweeping through neutral around the contact to ULNAR on the
        # follow-through (mirrored for an upstroke). Its size tracks velocity
        # the same way the stroke's does, so a hard note snaps the wrist and a
        # quiet one barely rolls it.
        ulnar = ((STRUM_ULNAR if strum else PICK_ULNAR)
                 * (1.0 - PICK_ULNAR_VEL + PICK_ULNAR_VEL * vel_norm)
                 * direction)
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
            key_after(windup, first - direction * lead, z_hover, -ulnar)
        key_after(t - 0.5 * strike, first - direction * lead * 0.4, z_pluck,
                  -0.4 * ulnar)
        key_after(t + cross_lag, last + direction * follow * 0.5, z_pluck,
                  0.5 * ulnar)
        rise_t = key_after(t + cross_lag + 0.06,
                           last + direction * follow, z_hover, ulnar)
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
