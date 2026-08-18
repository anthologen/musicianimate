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
    leave the whole diagonal in the wrist, and a PITCH (_pitch_keys): the
    wrist stroke, flexing the hand down into each note, holding it there
    while the note sounds and extending slightly as the finger comes off,
    as deep as the note is loud and the chord is wide and only where the
    passage leaves room for the gesture. Everything below the wrist is
    then solved in the hand's own frame (_hand_frame), so the
    fingertips hold their keys through the whole stroke and what gives is
    the fingers.
    How much of the reaching the wrist takes is set by what the fingers
    have room for BESIDE EACH OTHER (SPLAY_BESIDE), not by what their
    knuckles could do in isolation - a finger adducted to its joint's own
    limit is already through the finger next to it, so the hand goes to the
    notes rather than the fingers fanning after them.
  - Finger bones are driven by closed-form two-link IK in the vertical
    plane through the knuckle: the proximal bone pitches, the middle joint
    flexes, the proximal z-rotation supplies sideways reach (capped at the
    knuckle's anatomical abduction - which for the THUMB is lopsided, wide
    away from the palm and barely open toward it), and the distal phalanx
    keeps a fixed natural flexion. The thumb's plane is not vertical: its
    column is ROLLED about its own length (THUMB_ROLL, the pronation half of
    opposition), so folding it carries the tip across the palm the way a real
    thumb's does, rather than hooking it into the keyboard like a finger.
    It only goes UNDER the palm to cross, though: a thumb-under is a
    transitory move and a weak place to strike from, so an event that is
    not one (_mark_thumb_crossings) keeps the thumb turned out on its own
    side of the hand (THUMB_STANCE_ABDUCT) and the wrist goes to the note.
    No IK constraints are used, so the result is plain baked FK keyframes -
    every one of which lands inside the joint
    cage build_hands.py puts on the bones, _cage_pose being what guarantees
    it rather than leaving the bone constraints to rewrite the pose.
  - The whole hand is solved together, at the union of every finger's
    keyframe times, so no two fingers pass through each other: a finger
    with no note to play settles back over its own knuckle instead of
    holding the sideways reach of the key it last played, and whatever is
    not holding a key down is then slid aside and lifted until its
    phalanges clear their neighbours' by real surface distance. A digit no
    nudge can get clear WITHDRAWS toward its own knuckle and waits there
    (WITHDRAW_STEPS) - a finger on its way to a key does not get there
    through the neighbour still holding one - and a digit left with no
    room at all is placed again ahead of whoever took it. The result is
    then measured on the baked curves at every RENDERED frame, since the
    ease between two clear poses is not itself clear, and any frame that
    is still crossed is solved outright (REFINE_PASSES). What comes back
    is that measurement: `finger_clear_mm` per hand.
  - Press timing mirrors piano_midi_animator.py: velocity sets attack
    speed (loud = fast), with the same release tail, so fingertips and
    keys dip together when both animators are run on the same MIDI. The
    tail runs the ease backwards (_ease_off): a finger leaves a key at
    once and settles as it arrives, where only a strike lands at speed.
    It is also cut short where the finger BESIDE it is about to land on a
    nearby key (BESIDE_CLEAR_LEAD), so the room is given up in that eased
    motion rather than snatched away by the clearance search a frame
    before the neighbour arrives.

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
#
# But only ONE WAY. A finger's knuckle deviates either side of straight ahead by
# about as much, so a single cap describes it; the thumb's range is wildly
# lopsided. Away from the palm (abduction) it opens the 45-60 deg web space that
# spans a tenth. TOWARD the palm it has almost nothing: yaw 0 in this rig is the
# thumb lying alongside the index, which is already the end of the joint's
# adduction - a real thumb gets across the palm by rotating UNDER it (palmar
# flexion and opposition, which this rig carries as the metacarpal's pitch), not
# by swinging further sideways into the index it is touching.
#
# Capped symmetrically at 45 deg, the solve took the second option: on the reach
# take the right thumb lay flat in the palm plane and swung 32 deg PAST the
# index, threading between the index and middle fingers, at 98% of its own
# extension, to hold a key 42 mm treble-ward of a wrist the smoothing had left
# behind. It reads exactly like what it is - a thumb bent a way a thumb does not
# bend. So adduction gets a small cap of its own, and the wrist takes back the
# reach it was borrowing from the joint (_splay_clamp, which now slides the hand
# along the keyboard until the thumb is inside BOTH bounds).
# The abduction figure is the TOP of that 45-60 range rather than the bottom,
# because in this rig the yaw is paying for two things. A real CMC abducts and
# pronates in one motion; here the pronation is a separate roll (THUMB_ROLL), and
# a rolled column folds toward the palm, so reaching a key out on the radial side
# costs the yaw the roll's whole bearing on top of the key's own (_roll_bearing).
# At 45 the octave stretches of both takes ran out of thumb and left the tip up
# to 5 mm off its key; at 60 every one of them lands inside a millimetre.
FINGER_MCP_SPLAY = math.radians(26.0)
THUMB_CMC_ABDUCT = math.radians(60.0)
THUMB_CMC_ADDUCT = math.radians(12.0)

# ...but a knuckle's own range is only what it has when there is NOTHING BESIDE
# IT. 26 deg is the passive limit of a finger spread away from its neighbours;
# adducting the same 26 deg TOWARD one puts the two through each other long
# before the joint runs out, because knuckles are 24-26 mm apart and two
# proximal phalanges need ~17 mm between their axes to stay off one another.
# Swept against a neighbour held at its own idle pose, contact comes at:
#
#     f2 into f3   8 deg      f3 into f2   8 deg
#     f3 into f4  11 deg      f4 into f3  12 deg
#     f4 into f5  11 deg      f5 into f4  13 deg
#
# (the index/middle pair the tightest, being the widest fingers with the
# narrowest gap). So this is the splay the WRIST has to work to, not 26: it is
# the point at which a finger stops being able to reach further sideways without
# taking its neighbour with it, and a hand whose fingers are all inside it needs
# no one to dodge anyone. _splay_clamp slides the wrist to keep every pressing
# finger in this window and only falls back on the joint's own when a chord is
# too wide for it - which is the case the fingers really are fanned apart in,
# where the caps above are the true limit and nobody is adducting into anybody.
#
# Left at 26, the wrist barely moved through the reach take's bass run: four
# notes 48 mm apart played by a hand that stayed put while each finger in turn
# splayed to its cap, the middle finger 19 mm sideways of its own knuckle and
# straight through the ring finger beside it.
SPLAY_BESIDE = {(2, 3): math.radians(8.0), (3, 2): math.radians(8.0),
                (3, 4): math.radians(11.0), (4, 3): math.radians(12.0),
                (4, 5): math.radians(11.0), (5, 4): math.radians(13.0)}

# --- the thumb's column is TURNED, not just bent ------------------------------
# A finger flexes in the vertical plane through its knuckle: the tip goes
# straight down onto the key, which is what a finger does. The thumb does not.
# Its column is rotated about its own long axis - the pronation half of
# opposition, which the CMC saddle carries along with the abduction above - so
# the thumb's flexion axis is nowhere near the fingers'. Flex a real thumb and
# the tip travels ACROSS the palm and down; the pad turns to face the index and
# the nail faces outward, which is why a pianist's thumb meets a key on the
# side of its tip rather than the flat of the pad.
#
# Solved flat like a finger (roll 0), the rig's thumb hooked straight into the
# keyboard instead: on the reach take, frame 383, the right thumb's proximal
# phalanx was folded 120 deg - the cage's own maximum - dead in the vertical
# plane, a claw where a thumb should have been.
#
# So the whole column gets a roll, applied at the CMC (the prox bone) and
# inherited by the two phalanges above it, which tilts the plane every joint in
# the chain bends in. 35 deg is well inside the 45-90 deg of axial rotation a
# real CMC has through opposition, and is what turns the fold visibly sideways
# without reading as a twisted thumb.
#
# It costs the solve nothing: rolling the plane does not move the knuckle or
# change how FAR the tip is from it, so the two-link fold is the same angle it
# always was (see _finger_ik) - only its direction, and the yaw and pitch that
# aim it, change. What the roll cannot do is press a key lying directly beneath
# the CMC: a turned column reaches down and sideways together, so a target with
# no sideways room to give un-rolls the thumb by exactly as much as it must
# (_roll_fit). That is the real joint's bargain too - you cannot hold the thumb
# opposed and drive it straight down at the same time.
THUMB_ROLL = math.radians(35.0)

# --- ...and it only goes UNDER the palm to cross ------------------------------
# The roll above is what a thumb-under is made of, and having it costs the solve
# something it did not use to be able to do wrong: with the column turned, the
# thumb reaches a key WELL across the palm without any adduction to speak of -
# the fold carries the tip there (_roll_bearing) - so the splay clamp, which
# judges a wrist placement by the joint angles it demands, reads the pose as
# comfortable and leaves the hand where the smoothing dropped it.
#
# It is not comfortable. Passing under the hand is a TRANSITORY move: a real
# thumb goes there to hand the run over to the fingers crossing above it and
# comes straight back out, and while it is there it is at its weakest - the
# column is folded across the palm and the tip meets the key on its outside
# edge, which is not a position anyone strikes a note from by choice. A thumb
# with no crossing to do plays from its own side of the hand, where it is
# strong: turned out across a key or two (THUMB_IDLE_X), column extended, the
# whole arch of the hand behind it.
#
# On the reach take - single notes leaping half the board, so nothing crossing
# anywhere - the smoothing left the wrist up to 95 mm short of where a thumb
# note wanted it and the thumb simply reached: 34 mm across its own knuckle at
# frame 314, 39 mm at 325, 38 at 349, and the left hand 32 mm at 361. A thumb
# tucked under the palm for a whole held note, four times, with no run in sight.
#
# So the across-the-palm window is opened only for an event that is actually
# crossing (_mark_thumb_crossings), and any other pressing thumb keeps a stance
# of its own: the tip stays at least this far out from its knuckle line, and the
# WRIST goes to the note instead. 12 deg is about a third of the way from
# straight ahead to the idle thumb's own turn-out (26 deg at THUMB_IDLE_X /
# THUMB_IDLE_Y), which leaves the hand ~55 mm of the sideways glide the clamp
# exists to preserve while keeping the column out from under the index.
THUMB_STANCE_ABDUCT = math.radians(12.0)

# How near the crossing finger's key has to be for a thumb note to BE a crossing.
# A thumb passes under to the next step of a scale or, in an arpeggio, up to
# about a fourth - five white keys, 110 mm. Past that the two notes are not one
# hand position being handed over, they are a leap, and a leap is played by
# taking the hand there. (The reach take's thumb notes sit 168 mm from their
# neighbours, which is the whole point of it.)
THUMB_CROSS_SPAN = 0.11

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

# How many passes _splay_clamp takes to settle (its window moves with the slide
# once a digit is rolled) and the movement below which it is called settled.
SPLAY_CLAMP_PASSES = 4
SPLAY_CLAMP_EPS = 0.0005

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

# --- wrist pitch: the drop into the key --------------------------------------
# The hand also PITCHES. Held level - where it sits and how far it is turned out
# were the only two things the wrist carried - every note is played by the
# fingers alone out of a plate of a hand that stays exactly parallel to the
# keybed, and a piano is not played that way: the wrist FLEXES into the key and
# EXTENDS back off it. That is the whole of the wrist stroke (the wrist staccato
# octaves and chords are played with, where the forearm holds still and the hand
# swings down at the wrist to strike and lifts to release), and it is there in
# miniature in ordinary playing - the "down-up" every teacher draws over a
# two-note slur.
#
# So the armature object's rotation carries a pitch as well as a yaw: NEGATIVE
# is flexion, the fingers swinging down toward the palm side, positive is
# extension. Everything below the wrist is then solved in the frame the pitch
# puts the hand in (_hand_frame), so the fingertips stay on their keys through
# the whole gesture and it is the HAND that moves around them.
#
# Which means the stroke needs a HINGE, and the choice is not free. A fingertip
# on a key cannot go anywhere, so tilting the hand has to move everything else
# instead, and where it hinges decides what: hinge at the wrist and the wrist
# stands still while the palm swings down and the fingers fold up to stay on
# their keys; hinge out at the fingertips and the fingers keep their pose exactly
# while the whole hand rides up and forward over them. The first spends the
# fingers' REACH - the tilt carries a key ~8 mm further out along a hand that is
# already stretched for it - which is the one thing they have least of, and on a
# take at full stretch the wrist fit answers by diving at the keyboard to get it
# back, dragging the resting fingers down among the keys with it.
#
# So the hinge sits under the knuckles at KEY LEVEL (PITCH_PIVOT_*): the reach a
# pressing finger needs is untouched by the tilt, and what the stroke does spend
# is the DROP beneath the knuckles - the palm settling toward the keys as the
# hand flexes, which is what it looks like anyway - plus a centimetre of rise
# and reach at the wrist itself, which the arm follows.
#
# How far it goes is a question of EFFORT, not of note. A mezzo-piano single
# note is a finger; a fortissimo chord is the hand with the arm behind it, and
# the wrist that lands it visibly gives. Loudness leads, and width ADDS to it
# rather than standing in for it: a chord is more of a gesture than one note at
# the same dynamic, because it is more of the hand, but a pianissimo chord is
# still pianissimo and is not landed like a fortissimo one.
PITCH_FLEX_MIN = math.radians(2.5)    # the settle even a quiet note has
PITCH_FLEX_MAX = math.radians(11.0)   # a fortissimo chord. Playing lives well
#                                       inside the wrist's ~70 deg of flexion;
#                                       what limits this is that the hand has to
#                                       stay over the keys, not the joint
PITCH_VEL_SOFT = 0.35   # MIDI velocity (0..1) under which nothing is loud...
PITCH_VEL_LOUD = 0.95   # ...and at which it is as loud as this counts
PITCH_CHORD_FULL = 4    # notes at once that read as the whole hand
PITCH_CHORD_W = 0.45    # what such a chord is worth next to the dynamic

# The shape of one gesture: the hand comes DOWN into the note - bottoming out
# just after the key lands, the wrist absorbing the arrival rather than leading
# it - stays there for as long as the note is held, and then extends slightly as
# the finger comes off it and settles back to level. The flexion is the stroke;
# the extension belongs to the RELEASE and is small, because it is only the hand
# leaving the key rather than a gesture of its own.
#
# STYLE NOTE - the bouncy wrist. Two constants and one key time away, and this
# is where it lives. The earlier take cocked the hand UP before every strike
# (PITCH_PREP_FRAC = 0.35) and threw it back PAST level by half the drop a fixed
# 0.13 s after the bottom (PITCH_RELEASE_FRAC = 0.5, timed off `bottom` rather
# than off the note ending) - the wrist visibly reacting to each hit, whatever
# the note was doing. It grooves, and that is exactly why it is off by default:
# it reads as a player nodding along to the music rather than one playing it.
# To bring it back, set PITCH_PREP_FRAC = 0.35 and PITCH_RELEASE_FRAC = 0.5 and
# time the extension off `bottom` in _pitch_keys (there is one marked line).
PITCH_PREP_FRAC = 0.0     # the cock-up ahead of the strike. At 0 the hand
#                           starts level and simply flexes into the note; the
#                           key at PITCH_PREP_LEAD is then just the level the
#                           drop falls from
PITCH_RELEASE_FRAC = 0.2  # how far past level the hand comes back up as it
#                           leaves the key. Slight: a fifth of the drop, so
#                           half a degree off a quiet note and ~2 deg off a
#                           fortissimo chord
PITCH_PREP_LEAD = 0.13    # s the descent begins ahead of the strike (inside
#                           ARRIVE_LEAD: the hand is already in position)
PITCH_SINK = 0.05         # s after the strike that the flexion bottoms out
PITCH_RELEASE_LAG = 0.10  # s from the note ENDING to the top of the extension:
#                           the hand rises with the key it is letting up
PITCH_SETTLE = 0.20       # s from there back to level

# ...and none of it happens in a fast passage. A gesture belongs to a note the
# hand has the time to make one on; asked for every note of a run the wrist
# would either flutter, or - since the space between the keys is what gets
# dropped first - simply sit flexed for the whole run, which is the one shape a
# level hand at least never had. So the amplitude is scaled by the room the
# event has around it, and a run comes out played the way it is in life: the
# fingers alone, out of a quiet hand.
PITCH_FULL_GAP = 0.50   # s of clear time either side buys the whole gesture
#                         (about what the shape above takes end to end)
PITCH_MIN_GAP = 0.04    # s; a key closer than this to the last one is dropped

# Nor over a THUMB CROSSING. A thumb passing under the palm is the tightest the
# hand ever is - the thumb is beneath its own index finger with a key at the end
# of it - and a wrist bouncing on top of that is what puts the two through each
# other (on the demo's ascending run, 1.5 mm of it). It is also just not how a
# scale is played: the hand glides and the thumb goes under a quiet wrist.
PITCH_CROSS_FRAC = 0.3

# The hinge, in the hand's own frame (see above). Forward of the wrist by the
# knuckle line and down by the height a hovering hand holds over the keys, so
# the point that stays put through the stroke is roughly the key a pressing
# finger is standing on. Further out and the wrist would swing more and the
# fingers less; further in and the fingers would take a fold they have not got.
PITCH_PIVOT_Y = 0.050
PITCH_PIVOT_Z = key_layout.WHITE_H - HOVER_Z

# And a stroke is still a LUXURY. Even hinged where it costs the least, tilting
# the hand costs something - the drop under the knuckles - and a hand already at
# full stretch has nothing to give: on the reach take the left hand's 4-1 stretch
# from B3 up to G4 lost 3 mm of its thumb to a stroke it could not afford, which
# is a fingertip sitting visibly off its key so the wrist could wave. So the fit
# is asked at a ladder of depths and the deepest one that costs the fingers
# nothing is kept. A pianist reaching that far has a straight wrist too.
PITCH_FIT_STEPS = (1.0, 0.6, 0.3)   # fractions of the intended stroke, tried
#                                     deepest first
PITCH_FIT_SLACK = 0.0005            # m of extra fingertip error a stroke may
#                                     cost before it is given up as unaffordable

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

# ...and only so far FORWARD. The mid bone carries the lumped PIP+DIP fold of a
# finger and the lumped MCP+IP fold of a thumb (see build_hands' cage), and the
# wrist-height search had no idea it existed: a hand parked directly over its
# key can always reach it by curling the digit up under itself, tip exactly on
# the note and knuckle perfectly straight, so both terms above score it zero.
# That is how the reach take's thumb ended up folded to the cage's own 120 deg
# limit for a whole held note. These are a few degrees inside the build cage, so
# the two never fight, and cost is charged only past them.
FINGER_PIP_FOLD = math.radians(135.0)
THUMB_MCP_FOLD = math.radians(112.0)
FOLD_W = 3.0e3         # per rad^2 of fold past it - the same weight as bending
                       # the joint the other way, the failures being as bad as
                       # each other and neither as bad as missing the key

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

# A finger also has a DEADLINE for coming home: the moment the finger beside it
# lands on a nearby key. Two adjacent fingers share the space between their
# knuckles, so a release tail still draped over the key just played is something
# the neighbour has to come down through - and the clearance search, which sees
# only one frame at a time, can then only pull the loitering finger out of the
# way in the single frame it first notices, which is a jump. On the reach take
# the left ring finger's note ended 1.8 frames before the middle finger landed
# on the white key in front of it, and its 5-frame tail meant it was still lying
# there: it left 23 mm sideways in one frame. Given the deadline instead, it
# comes off its key and home in one eased motion, and there is nothing left to
# dodge. The tail is a nicety; being out of the way is not.
BESIDE_CLEAR_LEAD = 0.5   # frames of margin, as the same clamp against the
                          # finger's OWN next note uses
BESIDE_NEAR = 0.050       # m between the two keys, about two white keys: past
                          # that a neighbour is landing somewhere else entirely
                          # and this finger can leave in its own time

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

# --- when the nudges are not enough: WITHDRAW --------------------------------
# A slide-and-retreat around where the digit wanted to be assumes it wanted to be
# somewhere reachable in the first place. Sometimes it does not. A finger on its
# way to a key can be asked, several frames before its note, to hover over a key
# on the far side of a neighbour that is still holding one down - and then no
# nudge in the grid helps, because the pose is not near a clear one, it is
# through the neighbour's finger. On the reach take that is exactly what the left
# middle finger did: reaching across for B0 while the ring finger still held the
# A# behind it, its middle phalanx passed 15 mm INSIDE the ring's proximal, and
# the search dutifully returned the shallowest crossing it could find.
#
# What a hand does there is wait. The finger stays back over its own knuckle
# until the neighbour lifts, and only then goes to its key - it never gets there
# by going through. So a digit that cannot clear its neighbours where it stands
# WITHDRAWS: its target is blended back toward the idle pose (which is where a
# finger with nothing to do is, and is clear by construction) in steps, taking
# the first that gets it out of everyone. The withdrawal costs nothing visually,
# because the finger has no note yet - it just arrives from further out.
#
# Only a digit with give may do this. A finger holding a key DOWN is sounding a
# note and does not withdraw from anything (see HOVER_GIVE); if two pressing
# fingers cross, that is the fingering asking for a pose the hand does not have,
# and it is reported (see _worst_clearance) rather than quietly hidden.
WITHDRAW_STEPS = (0.25, 0.5, 0.75, 1.0)

# The clearance solved at the moments the fingers are keyed at is not quite the
# clearance that gets RENDERED: between two keys the bone curves ease from one
# pose to the other, and two fingers that are apart at both ends can still brush
# on the way, the eased path bulging a little off the straight line between them.
# So after the whole hand is keyed, every rendered frame is measured on the baked
# curves, and any frame where two digits have got into each other is added to the
# moments and solved outright - which pins the pose there and leaves the ease
# only the shorter spans either side. A couple of passes settles it, each pass
# checking the curves the previous one changed.
REFINE_PASSES = 3

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
# A charge is not a wall, though, and against a clearance it cannot buy any
# other way it will be paid: on the reach take's octave stretch, with the wrist
# stroke tilting the hand over it, the releasing index finger bought its way out
# of the pinky by sinking 12 mm into the key it had just played - which reads as
# a fingertip inside the keyboard and then snapping out of it, and no clearance
# between two fingers is worth that. So the dip is also CAPPED, at about what
# the key gives under a finger. Past it the pose is not offered at all, and a
# digit with nothing else left crosses its neighbour by a hair instead, where
# the refinement pass can see it.
KEYBED_DIVE_MAX = 0.002           # m below its own wish a nudge may push a digit

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


def _hand_frame(point, wrist, yaw, pitch=0.0):
    """A world point in the hand's own frame: the wrist at the origin, +y along
    the fingers and +z out the back of the hand, so a knuckle sits at exactly
    its build_hands offset and every finger solve reads the same as it did
    before the hand could turn or tilt.

    The hand's transform is Rz(yaw) . Rx(pitch) - the XYZ euler the object is
    keyed with, the pitch about the hand's own sideways axis and the yaw about
    world Z - so this is its inverse, applied in the other order.

    Heights come back in the convention the rest of the solve is written in:
    measured from the wrist, but reported as `wrist[2] +` that, so a level hand
    (pitch 0) returns the world height it always did and nothing downstream can
    tell the difference.
    """
    dx, dy = point[0] - wrist[0], point[1] - wrist[1]
    dz = point[2] - wrist[2]
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = c * dx + s * dy, c * dy - s * dx
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (x, cp * y + sp * dz, wrist[2] + cp * dz - sp * y)


def _stroke_offset(pitch, yaw=0.0):
    """Where the wrist itself has to go for the hand to be pitched by `pitch`
    about the stroke's hinge (PITCH_PIVOT_*) instead of about the wrist: a world
    offset to add to the wrist placement.

    The hinge is a point in the hand, so it moves when the hand turns; this is
    what puts it back. Flexing, the wrist comes up and forward over it; on the
    prep and the rebound it settles back and down.
    """
    c, s = math.cos(pitch), math.sin(pitch)
    y0, z0 = PITCH_PIVOT_Y, PITCH_PIVOT_Z
    dy = y0 - (c * y0 - s * z0)
    dz = z0 - (s * y0 + c * z0)
    return (-math.sin(yaw) * dy, math.cos(yaw) * dy, dz)


def _stroke_wrist(tgt, yaw, pitch):
    """A wrist placement as the stroke actually holds it at `pitch` - the same
    point the fit is scoring, moved onto its hinge."""
    return tuple(t + o for t, o in zip(tgt, _stroke_offset(pitch, yaw)))


def _world_z(point, wrist, pitch):
    """How high above the keybed a point in the hand's own frame actually sits:
    the inverse of the height _hand_frame hands back."""
    return (wrist[2] + point[1] * math.sin(pitch)
            + (point[2] - wrist[2]) * math.cos(pitch))


def _local_z(height, y, wrist, pitch):
    """The hand-frame height a point at hand-frame `y` needs so that it sits at
    world `height`.

    What every HOVER is written in: HOVER_LIFT is a distance above a key, and a
    key does not tilt with the hand. Left in the hand's frame, a fingertip told
    to wait 15 mm over the keys would ride 15 mm over a plane that pitches with
    the wrist - and with the hand flexed into a chord, the digits out at the end
    of it would be waiting inside the keyboard.
    """
    return (wrist[2]
            + (height - wrist[2] - y * math.sin(pitch)) / math.cos(pitch))


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


def _event_flex(event, room):
    """How far the wrist flexes into this event (radians, >= 0; see
    PITCH_FLEX_MAX).

    `room` (0..1) is how much of a gesture the passage leaves time for. Note
    that it is the LOUDEST note of a chord that sets the effort, not the mean:
    a hand landing one accented voice out of four is still landing it.

    Needs the event to have been through _mark_thumb_crossings, since a hand
    with its thumb under it barely strokes at all (PITCH_CROSS_FRAC).
    """
    if event.get("thumb_cross"):
        room *= PITCH_CROSS_FRAC
    vel = max(n["velocity"] for n in event["notes"]) / 127.0
    loud = ((vel - PITCH_VEL_SOFT) / (PITCH_VEL_LOUD - PITCH_VEL_SOFT))
    loud = max(0.0, min(1.0, loud))
    wide = max(0.0, min(1.0, (len(event["notes"]) - 1.0)
                        / max(1.0, PITCH_CHORD_FULL - 1.0)))
    effort = min(1.0, loud + PITCH_CHORD_W * wide)
    return room * (PITCH_FLEX_MIN
                   + effort * (PITCH_FLEX_MAX - PITCH_FLEX_MIN))


def _event_room(events, i):
    """0..1: how much of a wrist gesture event `i` has the time to make, from
    the closer of its two neighbours (see PITCH_FULL_GAP). The first and last
    events of a take have all the room in the world on the outside."""
    gaps = []
    if i > 0:
        gaps.append(events[i]["t"] - events[i - 1]["t"])
    if i + 1 < len(events):
        gaps.append(events[i + 1]["t"] - events[i]["t"])
    return min(1.0, min(gaps) / PITCH_FULL_GAP) if gaps else 1.0


def _pitch_keys(events, flexes, to_frame, frame_start, fps):
    """The wrist's pitch over the whole take, as (frame, (pitch,)) keys - the
    same shape as the travel arc, and sampled the same way.

    Each event contributes the corners of one gesture: level, bottom, held there
    while the note sounds, a slight extension as it is let up, level again. Only
    the BOTTOM is guaranteed - everything after it is dropped where the music has
    not left room for it, since the next event's own descent owns any moment it
    gets to first. So a passage too fast for the full shape loses the release
    lift and simply rises back to level in time for the next note, which is a
    hand playing quietly rather than a wrist stuck down or fluttering.

    The extension is timed off the note ENDING rather than off the bottom, which
    is the difference between a hand leaving a key and a hand bouncing off one
    (see the style note by PITCH_RELEASE_FRAC).
    """
    keys = [(frame_start, (0.0,))]
    step = PITCH_MIN_GAP * fps

    def add(t, pitch):
        frame = to_frame(t)
        if frame <= keys[-1][0] + step:
            return False
        keys.append((frame, (pitch,)))
        return True

    for i, ev in enumerate(events):
        flex = flexes[i]
        # The next event's own descent is the deadline for this one's tail:
        # whichever gesture gets to a moment first owns it.
        nxt = (events[i + 1]["t"] - PITCH_PREP_LEAD if i + 1 < len(events)
               else float("inf"))
        add(ev["t"] - PITCH_PREP_LEAD, PITCH_PREP_FRAC * flex)
        bottom = ev["t"] + PITCH_SINK
        if not add(bottom, -flex):
            # Two strikes on top of each other. The deeper flexion stands rather
            # than a key that will not fit quietly losing the drop it was for.
            keys[-1] = (keys[-1][0], (min(keys[-1][1][0], -flex),))
        # The event is over when its LAST voice is - that is when the hand comes
        # off the keys, and a chord whose notes end raggedly leaves once.
        release = max(n["end"] for n in ev["notes"])
        lift = max(release, bottom) + PITCH_RELEASE_LAG   # off `bottom` for the
        #                                                   bouncy style
        tail = [(release, -flex)] if release > bottom else []
        tail += [(lift, PITCH_RELEASE_FRAC * flex), (lift + PITCH_SETTLE, 0.0)]
        for t, pitch in tail:
            if t >= nxt:
                break
            # A corner too close to the last one to key is simply skipped: the
            # ones after it are still wanted (a staccato note has no hold, but
            # it does have a hand coming off it).
            add(t, pitch)
    return keys


def _splay_range(finger, mirror):
    """How far sideways this digit's knuckle may deviate: (lo, hi) radians in
    the hand's own frame, positive toward the little finger.

    Symmetric for the fingers. For the thumb the two directions are different
    joints' worth of motion (see THUMB_CMC_ABDUCT), and which one is which
    depends on the hand: the thumb sits at -x on a right hand and +x on a left,
    so moving toward the palm - the tight direction - is +yaw on the right and
    -yaw on the left.
    """
    if finger != 1:
        return (-FINGER_MCP_SPLAY, FINGER_MCP_SPLAY)
    return ((-THUMB_CMC_ABDUCT, THUMB_CMC_ADDUCT) if mirror > 0 else
            (-THUMB_CMC_ADDUCT, THUMB_CMC_ABDUCT))


def _splay_beside(finger, mirror):
    """The window _splay_range leaves once the NEIGHBOURS are counted in:
    (lo, hi) radians, tightened on any side this digit has a finger on.

    See SPLAY_BESIDE. Unchanged on a side with nothing beside it - the index's
    radial side, the little finger's ulnar side, and both of the thumb's.
    """
    lo, hi = _splay_range(finger, mirror)
    for nbr, sign in ((finger + 1, mirror), (finger - 1, -mirror)):
        cap = SPLAY_BESIDE.get((finger, nbr))
        if cap is None:
            continue
        if sign > 0:
            hi = min(hi, cap)
        else:
            lo = max(lo, -cap)
    return lo, hi


def _hyperext_cap(finger):
    """How far this digit's knuckle may bend backward (radians)."""
    return THUMB_CMC_HYPEREXT if finger == 1 else FINGER_MCP_HYPEREXT


def _fold_cap(finger):
    """How far this digit's middle joint may fold forward (radians)."""
    return THUMB_MCP_FOLD if finger == 1 else FINGER_PIP_FOLD


def _roll_fit(roll, dh, dv):
    """How much of `roll` a target this steep leaves usable (radians).

    A rolled column bends down and sideways together: dropping the tip `dv`
    below the knuckle costs |dv * tan(roll)| of sideways travel, and only `dh`
    of sideways travel exists. Past that the tip cannot land on the target at
    all, so the column un-rolls to the steepest turn that still reaches - which
    is what a real thumb does when it has to press something underneath itself.
    """
    return math.copysign(min(abs(roll), math.atan2(dh, abs(dv))), roll)


def _digit_roll(finger, mirror, dx, dy, dv):
    """How far this digit's column is turned about its own long axis to reach a
    target dx/dy across and dv below its knuckle (radians).

    Zero for the fingers, which bend in the plane through their knuckle. The
    thumb's column is rolled THUMB_ROLL toward the palm - +x on a right hand,
    where the thumb sits at -x - so that flexing it carries the tip across the
    palm instead of straight down (see THUMB_ROLL), as far as the target allows.
    """
    if finger != 1:
        return 0.0
    return _roll_fit(-mirror * THUMB_ROLL, math.hypot(dx, dy), dv)


def _roll_plane(roll, dh, dv):
    """The chain's own bending plane for this target: (usable roll, how far the
    tip travels DOWN that plane, how far ALONG it).

    The plane is tilted `roll` about the chain's length, so reaching dv below
    the knuckle takes dv / cos(roll) of in-plane drop, and what is left of the
    knuckle-to-target distance is the forward reach. Everything the roll
    changes about the solve is one of these three numbers.
    """
    roll = _roll_fit(roll, dh, dv)
    drop = dv / math.cos(roll)
    return roll, drop, math.sqrt(max(0.0, dh * dh + dv * dv - drop * drop))


def _roll_bearing(roll, dh, dv):
    """How far a rolled column's fold carries its tip off straight ahead
    (radians), and so how much the yaw has to give back to land on the target.

    A column rolled toward the palm folds toward the palm, so a key out on the
    radial side is reached by swinging the whole thumb FURTHER out - which the
    yaw pays for, out of a cap (THUMB_CMC_ABDUCT) it is often already sitting
    on. Anything that decides where a digit can reach has to know this or it
    will place the hand where only an unrolled thumb could have played.
    """
    roll, drop, along = _roll_plane(roll, dh, dv)
    return math.atan2(-drop * math.sin(roll), along) if roll else 0.0


def _finger_ik(dx, dy, dv, lengths, dist_flex, splay_cap=MAX_YAW, roll=0.0):
    """Closed-form 2-link IK for one finger with a rigidly flexed distal.

    dx/dy: fingertip target offset from the knuckle in the keyboard plane;
    dv: drop from knuckle to target (positive down); dist_flex: the fixed
    distal flexion the pose will use; splay_cap: the knuckle's anatomical
    sideways limit, either a symmetric cap or a signed (lo, hi) pair for a
    joint whose two directions differ (the thumb's - see _splay_range). It is
    the caller's business either way: the guitar/bass fret hands clamp the
    returned yaw themselves, so the default is the old loose value.
    `roll` turns the whole chain about its own long axis (the thumb's - see
    THUMB_ROLL), so the plane it bends in is tilted rather than vertical; at 0
    every line below is the plain vertical solve the fingers have always used.
    The mid+distal pair is treated as one link along the elbow-to-tip chord
    (length b, hanging gamma below the mid bone), which makes the fingertip
    land exactly on the target.
    Returns (yaw, prox_pitch, mid_flex), pitch/flex positive = down.
    """
    a = lengths[0]
    l2, l3 = lengths[1], lengths[2]
    b = math.hypot(l2 + l3 * math.cos(dist_flex), l3 * math.sin(dist_flex))
    gamma = math.atan2(l3 * math.sin(dist_flex), l2 + l3 * math.cos(dist_flex))
    lo, hi = (-splay_cap, splay_cap) if isinstance(splay_cap,
                                                   (int, float)) else splay_cap
    dh = math.hypot(dx, dy)
    # Straight-line knuckle-to-target distance, and so the fold the two links
    # settle at, is the same whatever the roll: turning the plane about the
    # chain's own axis moves neither end of it.
    d = math.hypot(dh, dv)
    if roll:
        # In the tilted plane the tip has to drop FURTHER to end up dv below the
        # knuckle, and that overshoot leans across the plane, which the yaw
        # gives back off the target's own bearing (_roll_bearing).
        roll, drop, along = _roll_plane(roll, dh, dv)
        aim = math.atan2(dx, dy) - _roll_bearing(roll, dh, dv)
    else:
        drop, along = dv, dh
        aim = math.atan2(dx, max(dy, MIN_REACH_Y))
    yaw = max(lo, min(hi, aim))
    d = max(abs(a - b) + 0.002, min(a + b - 0.002, d))
    delta = math.atan2(drop, along)
    psi = math.acos((a * a + d * d - b * b) / (2 * a * d))
    phi = math.acos((a * a + b * b - d * d) / (2 * a * b))
    return yaw, delta - psi, math.pi - phi - gamma


def _finger_chain(knuckle, lengths, yaw, prox, mid, dist_flex, roll=0.0):
    """[knuckle, PIP, DIP, tip] for one posed finger, in whatever frame the
    knuckle was given in (the piano solves in the hand's own - see
    _hand_frame).

    Only the proximal bone carries the sideways rotation and the roll (see
    _pose_finger), so the whole chain lies in ONE plane through the knuckle -
    vertical for a finger, and for the thumb the same plane tilted `roll` about
    its own axis, which is what leans each phalanx toward the palm as it folds.
    """
    pts = [tuple(knuckle)]
    pitch = 0.0
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    for length, bend in zip(lengths, (prox, mid, dist_flex)):
        pitch += bend
        h = length * math.cos(pitch)
        side = -length * math.sin(pitch) * sr
        pts.append((pts[-1][0] + h * sy + side * cy,
                    pts[-1][1] + h * cy - side * sy,
                    pts[-1][2] - length * math.sin(pitch) * cr))
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


def _worst_clearance(chains, wrist=(0.0, 0.0, 0.0)):
    """(clearance, pair) of the two closest digits in one posed hand."""
    worst, digits = (float("inf"), None), sorted(chains)
    for i, fa in enumerate(digits):
        for fb in digits[i + 1:]:
            c = _chain_clearance(fa, chains[fa], fb, chains[fb], wrist)
            if c < worst[0]:
                worst = (c, (fa, fb))
    return worst


def _baked_chains(curves, mirror, frame):
    """Every digit's chain at `frame` read off the BAKED f-curves - the pose
    that will be rendered, ease and all, rather than the one solved at the
    nearest moment. Undoes exactly what _pose_finger keyed.

    The wrist is taken to be at the origin: the whole hand shares one frame, so
    where it sits changes every chain together and none of the distances between
    them (nor the palm test, which is relative to the wrist anyway)."""
    chains = {}
    for f in FINGERS:
        rot = {}
        for seg in SEGS:
            path = 'pose.bones["f%d_%s"].rotation_euler' % (f, seg)
            rot[seg] = tuple(curves[(path, i)].evaluate(frame)
                             if (path, i) in curves else 0.0 for i in range(3))
        chains[f] = _finger_chain(
            _knuckle(f, (0.0, 0.0, 0.0), mirror), FINGERS[f]["lengths"],
            -rot["prox"][2], -rot["prox"][0], -rot["mid"][0],
            -rot["dist"][0], rot["prox"][1])
    return chains


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


def _mark_thumb_crossings(events, mirror):
    """Flag every event whose thumb is CROSSING - passing under the hand on its
    way to a key the fingers are handing over to it, or being crossed over by
    the finger taking the run back. Only those events may put the thumb across
    the palm (see THUMB_STANCE_ABDUCT).

    What a crossing looks like in a note list: the event next to this one - the
    one before it going up, the one after it coming back down, and the hand does
    not move between the two - is played by a FINGER, on a key the thumb's own
    note lies past. Ascending on a right hand that is finger 3 on E and the thumb
    on the F above it; descending it is the thumb on F and finger 3 coming over
    onto the E below. Both read the same way round, because "past" is measured
    from the thumb toward the fingers' side of the hand, which is the direction
    the thumb has to travel under the palm to get there. On a left hand the whole
    test mirrors with the keyboard, as the technique does.

    Near in time (one phrase - a crossing is the join between two notes, not two
    passages) and near in space (THUMB_CROSS_SPAN): a thumb note half the
    keyboard away from its neighbour is a leap, and a leap the hand travels.
    """
    for i, ev in enumerate(events):
        cross = False
        for n in ev["notes"]:
            if n["finger"] != 1:
                continue
            for j in (i - 1, i + 1):
                if not 0 <= j < len(events):
                    continue
                nbr = events[j]
                if abs(nbr["t"] - ev["t"]) > SEGMENT_GAP:
                    continue
                cross = cross or any(
                    0.0 < mirror * (n["x"] - m["x"]) <= THUMB_CROSS_SPAN
                    for m in nbr["notes"] if m["finger"] != 1)
        ev["thumb_cross"] = cross
    return events


def _bearing_window(finger, mirror, splay, bend, cross):
    """Where this digit's TIP may bear from its knuckle: (lo, hi) radians in the
    hand's own frame, positive toward the little finger.

    That is the splay window shifted by whatever the digit's roll already
    carries the tip off straight ahead (_roll_bearing), because what the wrist
    has to be placed for is where the tip ENDS UP, not what the knuckle is doing
    on its own. For a finger (roll 0) the shift is nothing and this is
    _splay_range unchanged.

    For a thumb it is the whole story: the roll turns a joint that barely
    adducts into a column that reaches a good 45 deg across the palm, and
    THUMB_STANCE_ABDUCT is what keeps it from spending that anywhere but a
    crossing.
    """
    lo, hi = splay[0] + bend, splay[1] + bend
    if finger != 1 or cross:
        return lo, hi
    # Out is -yaw on a right hand (the thumb sits at -x) and +yaw on a left.
    stance = -mirror * THUMB_STANCE_ABDUCT
    if mirror > 0:
        return lo, max(lo, min(hi, stance))
    return min(hi, max(lo, stance)), hi


def _event_root_target(event, mirror, yaw):
    """Wrist position placing the event's pressing knuckles over their keys,
    with the hand turned out by `yaw`.

    Each digit wants the wrist one rotated knuckle-plus-reach offset back from
    its key. Uses the midrange (not the mean) of those requirements: in a
    stretched chord the outer fingers have the least reach to spare, so the
    residual is split between them instead of letting the comfortable
    middle fingers drag the wrist off-center.

    Nothing here knows about the wrist stroke: this is where the hand is HELD,
    and the stroke swings around it (_stroke_offset).
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


def _splay_clamp(event, tgt, mirror, yaw, press_white, press_black, pitch=0.0):
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

    The window is only symmetric about the digit's key when its joint is: it is
    measured from the KNUCKLE, in the same signed bounds the IK clamps its yaw
    to (_splay_range), so the thumb's - wide on the side it abducts toward and
    barely open on the other - makes the hand travel along the keyboard where it
    used to let the thumb reach across the palm. The whole window then SHIFTS by
    whatever the digit's roll costs it (_roll_bearing): a thumb that folds toward
    the palm has to be aimed further out to land in the same place, and this is
    the one place that can buy it the room, since the yaw it would otherwise take
    it from is already sitting on its cap.

    That shift is the one thing here that depends on where the slide ENDS UP -
    the closer a key comes to sitting under its own knuckle, the more of the
    reach the roll eats - so the clamp is walked to a fixed point instead of
    being read off the position it started from. Measured once, from a smoothed
    target 92 mm away, it under-read the reach-take thumb's shift by 18 deg and
    left the tip 15 mm off its key; two or three passes settle to a millimetre.

    The window is taken TWICE. First as what the fingers have room for beside
    each other (_splay_beside): no finger adducted into its neighbour is a hand
    with no dodging in it, and putting the hand where that is true is the wrist's
    job, not the clearance search's - the search can only rescue one frame at a
    time, and a rescue is something you can see. Where a chord is too wide for
    that window - the fingers really fanned apart, which is the case those caps
    were never about - the joint's own range is used instead, as before.
    """
    for window in (_splay_beside, _splay_range):
        settled, feasible = tgt, True
        for _ in range(SPLAY_CLAMP_PASSES):
            slid, feasible = _splay_pass(event, settled, mirror, yaw,
                                         press_white, press_black, window,
                                         pitch)
            done = math.dist(slid[:2], settled[:2]) < SPLAY_CLAMP_EPS
            settled = slid
            if done:
                break
        if feasible:
            return settled
    return settled


def _splay_pass(event, tgt, mirror, yaw, press_white, press_black,
                window=_splay_range, pitch=0.0):
    """One pass of _splay_clamp: the window as it looks from `tgt`, the smallest
    slide that lands inside it, and whether it was a window at all - two fingers
    can want the hand in two places, and then the best there is is to split the
    difference between them.

    Looked at from the hand as the wrist stroke holds it at the strike, not from
    a level one: the tilt leaves a key about where it was along the fingers (the
    hinge is what buys that) but a good deal less far below them, and a digit
    with less drop under it bears differently on its key."""
    has_black = any(n["is_black"] for n in event["notes"])
    at = _stroke_wrist(tgt, yaw, pitch)
    lo, hi = float("-inf"), float("inf")
    for n in event["notes"]:
        kx, ky, kz = FINGERS[n["finger"]]["knuckle"]
        key = (n["x"], _target_y(n, has_black),
               _press_z(n["is_black"], press_white, press_black))
        tx, ty, tz = _hand_frame(key, at, yaw, pitch)
        dx = tx - kx * mirror
        reach = max(ty - ky, MIN_REACH_Y)
        dv = at[2] + kz - tz
        bend = _roll_bearing(_digit_roll(n["finger"], mirror, dx, reach, dv),
                             math.hypot(dx, reach), dv)
        blo, bhi = _bearing_window(n["finger"], mirror,
                                   window(n["finger"], mirror), bend,
                                   event.get("thumb_cross", False))
        lo = max(lo, dx - math.tan(bhi) * reach)
        hi = min(hi, dx - math.tan(blo) * reach)
    slide = (lo + hi) / 2.0 if lo > hi else max(lo, min(hi, 0.0))
    return ((tgt[0] + slide * math.cos(yaw),
             tgt[1] + slide * math.sin(yaw), tgt[2]), lo <= hi)


def _tip_miss(finger, mirror, dx, dy, dv, dist_flex, roll=0.0):
    """How far the posed fingertip ends up from the key it was asked for (m).

    Zero for anything the digit can actually do; positive once its limits bite -
    the chain too short for the target, the knuckle pinned at its splay cap with
    the key still further sideways, or a joint driven past the cage and pulled
    back by _cage_pose. That last one is measured HERE, on the caged pose, not on
    what the IK wished for: what the wrist search is being asked is how well a
    placement will look once it is keyed, and a pose the cage rewrites is not the
    pose the IK solved. Scoring the wish instead let the hand settle where the
    thumb had to bow its CMC 35 deg back to reach - a pose the cage clamps at 30,
    leaving the tip 16 mm off the key, at a cost the search read as zero.

    A rolled column (_digit_roll) is measured in the plane it actually bends in,
    which is the same one _finger_chain walks.
    """
    lengths = FINGERS[finger]["lengths"]
    pose = _finger_ik(dx, dy, dv, lengths, dist_flex,
                      _splay_range(finger, mirror), roll)
    yaw, prox, mid = _cage_pose(finger, mirror, pose, dist_flex)
    roll = _roll_fit(roll, math.hypot(dx, dy), dv)   # what the IK actually used
    l1, l2, l3 = lengths
    p1, p2, p3 = prox, prox + mid, prox + mid + dist_flex
    h = l1 * math.cos(p1) + l2 * math.cos(p2) + l3 * math.cos(p3)
    v = l1 * math.sin(p1) + l2 * math.sin(p2) + l3 * math.sin(p3)
    side = -v * math.sin(roll)
    tip = (h * math.sin(yaw) + side * math.cos(yaw),
           h * math.cos(yaw) - side * math.sin(yaw), v * math.cos(roll))
    return math.dist(tip, (dx, dy, dv))


def _event_pose_cost(event, tgt, mirror, yaw, press_white, press_black,
                     pitch=(0.0, 0.0)):
    """How badly a wrist placement serves this event's pressing fingers.

    Three things go wrong, and they pull against each other:

      * hold the hand too far from the keys and a finger cannot reach - the
        chain runs out of length, or the knuckle hits its sideways limit, and
        the fingertip is left hanging off its key (_tip_miss);
      * hold it too close and the finger has nowhere to drop to, so it folds
        BACKWARD at the knuckle (MCP hyperextension) to stay on the key - the
        bowed-back knuckle the joint cage forbids;
      * park it right on TOP of the key and the digit has to curl up under
        itself to get down to it - the opposite failure, and the one the two
        above cannot see, because a fully folded chain still lands its tip
        exactly on the key and still has a perfectly straight knuckle. It was
        invisible here until the reach take's thumb was found holding a white
        key 16 mm in front of its own CMC with the middle joint at 120 deg, the
        cage's own maximum, hooked into the keyboard.

    All three are scored against the anatomy and squared, so a stretched chord
    settles by splitting the difference between its fingers rather than pinning
    one at its limit. The miss is weighted the heavier: a fingertip visibly off
    its key reads worse than a few degrees of knuckle. Both the press and the
    hover pose are scored, since the hand holds one wrist position for both -
    and each in the tilt the wrist stroke has the hand at when it happens
    (`pitch`, flexed at the bottom and extended over the hover the finger leaves
    the key through). One placement has to serve the whole of that swing, so
    scoring it at the two extremes is what keeps a chord the hand drops into
    from being fitted flat and then played tilted.
    """
    has_black = any(n["is_black"] for n in event["notes"])
    cost = 0.0
    for n in event["notes"]:
        spec = FINGERS[n["finger"]]
        kx, ky, kz = spec["knuckle"]
        press_z = _press_z(n["is_black"], press_white, press_black)
        cap = _splay_range(n["finger"], mirror)
        for tip_z, flex, tilt in ((press_z, DIST_FLEX_PRESS, pitch[0]),
                                  (press_z + HOVER_LIFT, DIST_FLEX_HOVER,
                                   pitch[1])):
            at = _stroke_wrist(tgt, yaw, tilt)
            tx, ty, tz = _hand_frame(
                (n["x"], _target_y(n, has_black), tip_z), at, yaw, tilt)
            dx, dy, dv = tx - kx * mirror, ty - ky, at[2] + kz - tz
            roll = _digit_roll(n["finger"], mirror, dx, dy, dv)
            cost += REACH_W * _tip_miss(n["finger"], mirror, dx, dy, dv,
                                        flex, roll) ** 2
            _yaw, prox, mid = _finger_ik(dx, dy, dv, spec["lengths"], flex,
                                         cap, roll)
            back = -prox - _hyperext_cap(n["finger"])
            cost += HYPEREXT_W * max(back, 0.0) ** 2
            cost += FOLD_W * max(mid - _fold_cap(n["finger"]), 0.0) ** 2
    return cost


def _fit_event(event, tgt, mirror, yaw, pitch, press_white, press_black):
    """One event's best wrist placement at one stroke depth, as (cost, target).

    The grid: in and out from the keys (y) and down from the hover height (z),
    with the sideways clamp re-applied at each candidate since moving in or out
    changes how far the fingers can splay.
    """
    base = _splay_clamp(event, tgt, mirror, yaw, press_white, press_black,
                        pitch[0])
    best = (_event_pose_cost(event, base, mirror, yaw, press_white,
                             press_black, pitch), base)
    if best[0] <= 0.0:
        return best
    steps = int(WRIST_Y_RANGE / WRIST_Y_STEP)
    for i in range(-steps, steps + 1):
        dy = i * WRIST_Y_STEP
        z = tgt[2]
        while z >= MIN_WRIST_Z - 1e-9:
            cand = _splay_clamp(event, (tgt[0], tgt[1] + dy, z), mirror,
                                yaw, press_white, press_black, pitch[0])
            cost = (_event_pose_cost(event, cand, mirror, yaw, press_white,
                                     press_black, pitch)
                    + WRIST_REG * (dy * dy + (tgt[2] - z) ** 2))
            if cost < best[0]:
                best = (cost, cand)
            z -= WRIST_Z_STEP
    return best


def _wrist_fit(events, targets, yaws, flexes, mirror, press_white,
               press_black):
    """Nudge each smoothed wrist target to a placement the fingers can hold.

    The smoothed path says where the hand would LIKE to be; this asks what its
    fingers can actually do there, and if the answer is "not this" it searches a
    small grid around it (_fit_event) for the placement that best serves the
    whole event. Moving the hand rather than contorting the fingers is the whole
    point, and it is what a pianist does: dropping and reaching in for a
    stretched voicing, riding higher and flatter over the black keys. A
    regularizer keeps the hand on its smoothed glide wherever the fingers are
    comfortable, so this only bites where the geometry is tight - ordinary
    single notes and close chords score zero and do not move at all.

    It is also where the WRIST STROKE is settled, because the two decide each
    other: the fit asks what the fingers can do out of the hand as the stroke
    will actually hold it, and where the answer is "less than they could out of
    a level one" the stroke is the thing that gives (PITCH_FIT_STEPS). So this
    returns the depth each event's stroke ended up with as well as where the
    hand is held for it.

    The events are marked for thumb crossings first, because the sideways clamp
    inside this search is the one place that knows the difference between a
    thumb passing under the hand and a thumb left behind by it.
    """
    _mark_thumb_crossings(events, mirror)
    out, kept = [], []
    for ev, tgt, yaw, flex in zip(events, targets, yaws, flexes):
        level = None
        for step in PITCH_FIT_STEPS:
            stroke = (-flex * step,
                      max(PITCH_PREP_FRAC, PITCH_RELEASE_FRAC) * flex * step)
            cost, cand = _fit_event(ev, tgt, mirror, yaw, stroke,
                                    press_white, press_black)
            if cost <= REACH_W * PITCH_FIT_SLACK ** 2:
                break            # free: nothing a level hand could do better
            if level is None:    # what the fingers manage with no stroke at all
                level = _fit_event(ev, tgt, mirror, yaw, (0.0, 0.0),
                                   press_white, press_black)
            if cost <= level[0] + REACH_W * PITCH_FIT_SLACK ** 2:
                break
        else:
            step, (cost, cand) = 0.0, level
        out.append(cand)
        kept.append(flex * step)
    return out, kept


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


def _ease_off(u):
    """The same SINE run backwards (Blender's EASE_OUT): away fast, settling
    in.

    Which is what LETTING GO of a key is. Eased in like an attack, a finger
    whose note has ended stays lying on its key for most of the release and
    then covers the whole way home in the last frame or two - the release read
    as a flick rather than a lift, and where a neighbour was landing beside it
    (BESIDE_CLEAR_LEAD) it had all of one frame to do it in. A finger leaves a
    key at once and slows as it arrives; only the strike lands at speed.
    """
    return math.sin(max(0.0, min(1.0, u)) * math.pi / 2.0)


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
    rest of the time - which is also what says which WAY a span is going, and
    so which way it eases: a span that gives a key back (give rising) is a
    release and eases out, everything else lands on something and eases in.
    """
    if frame <= plan[0][0]:
        _fr, tgt, flex, give = plan[0]
    elif frame >= plan[-1][0]:
        _fr, tgt, flex, give = plan[-1]
    else:
        for (fa, ta, xa, ga), (fb, tb, xb, gb) in zip(plan, plan[1:]):
            if fa <= frame <= fb:
                shape = _ease_off if gb > ga else _ease
                u = shape(0.0 if fb <= fa else (frame - fa) / (fb - fa))
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


def _curve_digit(fcurve):
    """Which digit a bone f-curve belongs to, from its data path:
    pose.bones["f4_prox"].rotation_euler -> 4. None for the object's own."""
    _, _, rest = fcurve.data_path.partition('bones["f')
    return int(rest.split("_")[0]) if rest else None


def _knuckle(finger, wrist, mirror):
    kx, ky, kz = FINGERS[finger]["knuckle"]
    return (wrist[0] + kx * mirror, wrist[1] + ky, wrist[2] + kz)


def _idle_target(finger, knuckle, mirror, hover_z, slide=0.0, retreat=0.0,
                 wrist=(0.0, 0.0, 0.0), pitch=0.0):
    """Where a digit that is not playing puts its fingertip.

    Straight ahead of its own knuckle - no sideways reach at all - arched to the
    same fraction of its length a press uses, hovering over the keys; the thumb
    turned out to the radial side (THUMB_IDLE_X). ``slide`` and ``retreat`` are
    the nudges _solve_clear uses to get a digit out of a neighbour's way: aside,
    and up off the keys - or, negative, out and down onto them, which is what a
    digit with little drop left beneath it (a low wrist) has to do instead. A
    retreat also trades reach for height (IDLE_RETREAT_TUCK), so the digit curls
    in rather than pointing up.

    The height is over the KEYS, so it is turned back into the tilted hand's own
    frame (_local_z): a resting finger waits the same distance above the keybed
    whatever the wrist stroke is doing above it.
    """
    if finger == 1:
        reach, out = THUMB_IDLE_Y, -mirror * THUMB_IDLE_X
    else:
        reach, out = IDLE_REACH * sum(FINGERS[finger]["lengths"]), 0.0
    reach = max(MIN_REACH_Y, reach - IDLE_RETREAT_TUCK * retreat)
    y = knuckle[1] + reach
    return (knuckle[0] + out + slide, y,
            _local_z(hover_z + retreat, y, wrist, pitch))


def _in_cage(finger, mirror, yaw, prox, mid, dist_flex, roll=0.0):
    """Whether this pose lands inside build_hands' joint cage.

    The LIMIT_ROTATION constraints on the bones are guards, never clampers: a
    keyframe outside them would be silently pulled back in the scene, and the
    pose that then shows up is not the one the clearance search checked. So the
    search only ever considers poses that survive the cage.
    """
    for seg, rx, ry, rz in (("prox", -prox, roll, -yaw),
                            ("mid", -mid, 0.0, 0.0),
                            ("dist", -dist_flex, 0.0, 0.0)):
        limit = rot_limit(finger, seg, mirror)
        for axis, val in (("x", rx), ("y", ry), ("z", rz)):
            lo, hi = limit[axis]
            if not math.radians(lo) <= val <= math.radians(hi):
                return False
    return True


def _cage_pose(finger, mirror, pose, dist_flex):
    """`pose`, pulled inside build_hands' joint cage.

    Nothing may be KEYED outside it either. A keyframe past a LIMIT_ROTATION is
    not a pose that plays: Blender rewrites it at render time, and the hand that
    then shows up is one nothing solved, checked for clearance or measured for
    range of motion - on the reach take that was a thumb keyed 48 deg back at
    the CMC and shown at the cage's 30. _solve_clear already refuses out-of-cage
    candidates, but a digit whose whole search the cage rejects falls back on
    what it wished for; this is where that wish is made honest. The tip then
    sits off its key by exactly what the joint could not do, which is the truth
    about the reach and reads as one - a hand short of a key, not a broken one.

    The roll is not clamped here because it cannot need it: THUMB_ROLL is inside
    the cage's own axial bound by construction, and _roll_fit only ever turns
    the column less.
    """
    def clamp(val, bounds):
        lo, hi = bounds
        return max(math.radians(lo), min(math.radians(hi), val))

    yaw, prox, mid = pose
    prox_lim = rot_limit(finger, "prox", mirror)
    return (-clamp(-yaw, prox_lim["z"]), -clamp(-prox, prox_lim["x"]),
            -clamp(-mid, rot_limit(finger, "mid", mirror)["x"]))


def _pose_from_target(finger, mirror, knuckle, target, dist_flex):
    """(yaw, prox, mid) putting this finger's tip on a world-space target."""
    return _finger_ik(target[0] - knuckle[0], target[1] - knuckle[1],
                      knuckle[2] - target[2], FINGERS[finger]["lengths"],
                      dist_flex, _splay_range(finger, mirror),
                      _target_roll(finger, mirror, knuckle, target))


def _target_roll(finger, mirror, knuckle, target):
    """_digit_roll for a digit reaching from `knuckle` to a world-space
    `target` - the roll every pose built from that pair has to be walked with
    (_finger_chain) and keyed with (_pose_finger)."""
    return _digit_roll(finger, mirror, target[0] - knuckle[0],
                       target[1] - knuckle[1], knuckle[2] - target[2])


def _grid(step, span, signed):
    n = int(span / step + 1e-9)
    return tuple(i * step for i in range(-n if signed else 0, n + 1))


def _solve_clear(finger, knuckle, mirror, wrist, hover_z, dist_flex,
                 obstacles, key_target, mix, budget, pitch=0.0):
    """Where `finger`'s tip goes so it keeps clear of the chains already placed.

    `key_target`/`mix` say what the digit wants: a point on a key, the idle pose
    over its own knuckle, or - through the ease either side of a rest - a blend
    of the two. That wish is tried first and kept if it is already clear.
    Otherwise a grid of sideways slides and retreats around it is searched, in
    order of how far each strays (stray_cost), for the pose that clears its
    neighbours while straying least, which is how a hand keeps its fingers
    apart: the free ones move aside and away, the one holding a key does not
    move at all, and none of them gives way into the keyboard.

    A digit that no nudge can get clear of its neighbours does not settle for the
    shallowest crossing: it WITHDRAWS toward its idle pose and searches again,
    one step further back each time, until it is out of them (WITHDRAW_STEPS).
    A finger with no note yet has nothing to lose by waiting off to the side.

    `budget` (0..1) is how far that search may stray - zero for a finger on
    its key, whose grid then collapses to the wish itself and which never
    withdraws. `obstacles` are (finger, chain) pairs: every digit placed before
    this one, so no two fingers settle into each other either. Candidates that
    would take a joint outside its range of motion are not considered at all
    (_in_cage). Returns (target, clearance).
    """
    def wish(slide=0.0, retreat=0.0, back=0.0):
        """The tip target `back` of the way further out toward the idle pose."""
        idle = _idle_target(finger, knuckle, mirror, hover_z, slide,
                            retreat, wrist, pitch)
        if key_target is None:
            return idle
        m = mix + (1.0 - mix) * back
        if m <= 0.0:
            return (key_target[0] + slide, key_target[1],
                    key_target[2] + retreat)
        return tuple(k + (i - k) * m for k, i in zip(key_target, idle))

    def height(*nudge):
        """The world height of a nudged wish - out of the hand's own frame,
        because the KEYS do not tilt with the wrist."""
        return _world_z(wish(*nudge), wrist, pitch)

    def sunk(retreat, slide):
        """How far this nudge would push the digit below the keys AND below
        where it was going anyway: a digit ON a key is not diving, it is
        playing."""
        return max(0.0, min(KEYBED_Z + KEYBED_CLEAR, height(slide))
                   - height(slide, retreat))

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

        Measured against the KEYBED, which is where the keys are and which does
        not tilt with the wrist (see `sunk`).
        """
        return (IDLE_SLIDE_COST * abs(slide) + IDLE_RETREAT_COST * abs(retreat)
                + KEYBED_COST * sunk(retreat, slide))

    # Searched CHEAPEST FIRST, so the wish itself (cost 0) is always tried
    # before any nudge and the first pose the joint cage allows is the nearest
    # legal one. Walked in plain grid order instead, a digit whose own wish the
    # cage refused took whichever corner of the search came first - the full
    # downward nudge - and a digit with nothing in its way at all did the same,
    # which is how the thumb ended up inside the keybed with no neighbour
    # anywhere near it.
    def nudge_grid(freedom):
        return sorted((stray_cost(retreat, slide), retreat, slide)
                      for retreat in _grid(RETREAT_STEP,
                                           freedom * RETREAT_MAX, True)
                      for slide in _grid(SLIDE_STEP, freedom * SLIDE_MAX,
                                         True)
                      if sunk(retreat, slide) <= KEYBED_DIVE_MAX)

    def search(nudges, back):
        """Best (clearance, target) over the whole grid, withdrawn by `back`.

        (None, None) where the joint cage refused every candidate in it.
        """
        best = None
        for cost, retreat, slide in nudges:
            target = wish(slide, retreat, back)
            pose = _pose_from_target(finger, mirror, knuckle, target,
                                     dist_flex)
            roll = _target_roll(finger, mirror, knuckle, target)
            if not _in_cage(finger, mirror, *pose, dist_flex, roll):
                continue
            if not obstacles:
                return float("inf"), target
            chain = _finger_chain(knuckle, FINGERS[finger]["lengths"], *pose,
                                  dist_flex, roll)
            clear = min(_chain_clearance(finger, chain, g, oc, wrist)
                        for g, oc in obstacles)
            if clear >= IDLE_CLEAR_TARGET and cost <= 0.0:
                return clear, target
            score = min(clear, IDLE_CLEAR_TARGET) - cost
            if best is None or score > best[0]:
                best = (score, clear, target)
        return (best[1], best[2]) if best is not None else (None, None)

    def measure(target):
        """What the pose this target really gives has left between it and the
        digits already placed - CAGED, because that is the pose that gets keyed.

        The search only ever scores candidates the cage allows, so it has no
        number for the one case where it allows none: the fallback below, which
        is keyed anyway and is as able to cross a neighbour as any other pose.
        Reported as clear, it hid a thumb 3 mm inside the index finger.
        """
        if not obstacles:
            return float("inf")
        roll = _target_roll(finger, mirror, knuckle, target)
        pose = _cage_pose(finger, mirror,
                          _pose_from_target(finger, mirror, knuckle, target,
                                            dist_flex), dist_flex)
        chain = _finger_chain(knuckle, FINGERS[finger]["lengths"], *pose,
                              dist_flex, roll)
        return min(_chain_clearance(finger, chain, g, oc, wrist)
                   for g, oc in obstacles)

    clear, target = search(nudge_grid(budget), 0.0)
    if clear is None:
        # Nothing in the grid is inside the cage: a key the hand cannot really
        # hold. Keep what was asked for, exactly as the wrist search's own
        # compromises are kept, and let _cage_pose show it for what it is -
        # but measure it, because it is still a pose among four others.
        target = wish()
        clear = measure(target)
    if clear < 0.0 and budget > 0.0:
        # In someone else's finger where it stands. A digit being CROSSED INTO
        # is not held to its usual allowance any more: the give it was rationed
        # (HOVER_GIVE) is about how conspicuous it is to shift a finger on its
        # way to or from a key, and a finger with another one inside it is more
        # conspicuous than any nudge. So the whole grid is opened to it - a real
        # hand does the same, a finger adducted into its neighbour carrying that
        # neighbour along rather than passing through it - and if being pushed
        # aside is still not enough it steps back out toward its idle pose until
        # it is out of the way. Whichever step comes closest is kept, so a hand
        # the wrist search has left in a real bind still shows its best pose.
        full = nudge_grid(1.0)
        for back in (0.0,) + WITHDRAW_STEPS:
            c, t = search(full, back)
            if c is None:
                continue   # nothing the cage allows here; a step further back
                           # is a different point, and may well be reachable
            if c > clear:
                clear, target = c, t
            if clear >= 0.0:
                break
    return target, clear


def _pose_finger(pbones, finger, yaw, prox, mid, dist_flex, frame, roll=0.0):
    """Key one digit's three bones.

    Only the knuckle bone is turned sideways or rolled; the two above it are
    pure flexion. With XYZ Euler order that reads, on the knuckle, as: swing to
    the yaw, roll the column about its own new length, then pitch about the axis
    the roll just moved - and the phalanges above inherit that axis, so the
    whole chain folds in the one tilted plane _finger_chain walks (see
    THUMB_ROLL). At roll 0 the plane is vertical, which is every finger.
    """
    for seg, rx, ry, rz in (("prox", -prox, roll, -yaw),
                            ("mid", -mid, 0.0, 0.0),
                            ("dist", -dist_flex, 0.0, 0.0)):
        pb = pbones[f"f{finger}_{seg}"]
        pb.rotation_euler = (rx, ry, rz)
        pb.keyframe_insert(data_path="rotation_euler", frame=frame)


def _relax_finger(pbones, finger, frame, roll=0.0):
    prox, mid, dist = RELAXED
    _pose_finger(pbones, finger, 0.0, prox, mid, dist, frame, roll)


def animate_hand(arm_obj, notes, fps, frame_start,
                 press_depth_white, press_depth_black,
                 min_attack_frames, max_attack_frames, release_frames):
    """Keyframe one hand rig from its note list.

    Returns (last frame, the worst finger-to-finger clearance anywhere in the
    take, in metres - negative if two digits could not be got out of each
    other)."""
    mirror = -1.0 if arm_obj.name.endswith("_L") else 1.0
    pbones = arm_obj.pose.bones
    arm_obj.animation_data_clear()

    def to_frame(t):
        return frame_start + t * fps

    events = _group_events(notes)
    if not events:
        arm_obj.rotation_euler = (0.0, 0.0, 0.0)
        for f in FINGERS:
            # Nothing to reach for, so the thumb takes its whole roll.
            _relax_finger(pbones, f, frame_start,
                          -mirror * THUMB_ROLL if f == 1 else 0.0)
        return frame_start, float("inf")

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
    #
    # How far the wrist FLEXES into each event is settled first, because the fit
    # has to ask what the fingers can do out of the hand as the stroke will
    # actually hold it - tilted, and swung around its hinge - rather than out of
    # a level one. It is a property of the event alone: how loud, how wide, and
    # how much room the passage leaves for a gesture. So nothing here is
    # circular, and a placement is still just (x, y, z, yaw) - where the hand is
    # HELD, with the stroke moving around it.
    _mark_thumb_crossings(events, mirror)   # the stroke has to know about them
    flexes = [_event_flex(ev, _event_room(events, i))
              for i, ev in enumerate(events)]
    yaws = [_event_yaw(ev, mirror) for ev in events]
    smoothed = _smooth_targets(
        events, [_event_root_target(ev, mirror, y) + (y,)
                 for ev, y in zip(events, yaws)])
    yaws = [s[3] for s in smoothed]
    fitted, flexes = _wrist_fit(events, [s[:3] for s in smoothed], yaws, flexes,
                                mirror, press_depth_white, press_depth_black)
    targets = [t + (y,) for t, y in zip(fitted, yaws)]

    # The pitch is a curve of its own, on its own key times: the wrist stroke
    # runs on the MUSIC's clock (strike, hold, release, settle), while the
    # position and the yaw run on the hand's travel schedule, and neither has
    # any business keying the other's channel. So the object's rotation is keyed
    # one axis at a time - Z where the hand arrives and travels, X where the
    # stroke turns over - and the two curves cross without disturbing each
    # other.
    pitch_keys = _pitch_keys(events, flexes, to_frame, frame_start, fps)

    arm_obj.rotation_euler = (0.0, 0.0, 0.0)   # nothing keys Y; it stays level
    for frame, (pitch,) in pitch_keys:
        arm_obj.rotation_euler[0] = pitch
        arm_obj.keyframe_insert(data_path="rotation_euler", index=0,
                                frame=frame)

    # The wrist path is PLANNED here and keyed further down, once the stroke has
    # been laid over it: the hinge moves the wrist, so every moment the pitch
    # turns over needs a placement of its own, and those moments belong to the
    # music rather than to the hand's travel schedule. Keeping the list is what
    # the finger solve wants anyway - it needs to know where the wrist actually
    # is at each frame it poses a finger, not just where the event it belongs to
    # wanted it.
    wrist_keys = []

    def key_root(t, target):
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

    # The stroke, laid over the path the hand travels. Where the pitch turns
    # over between two wrist keys the placement is read off the path there and
    # becomes a key of its own, so the hinge can move the wrist without the
    # curve cutting the corner; then every placement is swung onto the hinge for
    # whatever the pitch is doing at it. From here on `wrist_keys` is where the
    # wrist actually is - which is what the fingers are solved against and what
    # the pianist's arm will follow.
    wrist_keys.sort(key=lambda k: k[0])
    keyed = {fr for fr, _t in wrist_keys}
    wrist_keys += [(fr, _sample(wrist_keys, fr, _ease_wrist))
                   for fr, _p in pitch_keys
                   if fr not in keyed
                   and wrist_keys[0][0] < fr < wrist_keys[-1][0]]
    wrist_keys.sort(key=lambda k: k[0])
    wrist_keys = [(fr, _stroke_wrist(t, t[3],
                                     _sample(pitch_keys, fr, _ease_wrist)[0])
                   + (t[3],))
                  for fr, t in wrist_keys]

    for frame, target in wrist_keys:
        arm_obj.location = target[:3]
        arm_obj.rotation_euler[2] = target[3]
        arm_obj.keyframe_insert(data_path="location", frame=frame)
        arm_obj.keyframe_insert(data_path="rotation_euler", index=2,
                                frame=frame)

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

    def beside_lands(finger, key_x, after):
        """When a finger BESIDE this one next comes down on a key near this
        one - the moment this one has to be out of its way by.

        Adjacent fingers share the space between their knuckles, so a finger
        still draped over the key it has finished with is a finger the one next
        to it has to land through. Only a NEARBY key counts: a neighbour taking
        a note half the keyboard away is not coming near this finger, and this
        finger can leave in its own time.
        """
        return min((fr for nbr in (finger - 1, finger + 1)
                    for fr, x in landings.get(nbr, ())
                    if fr > after and abs(x - key_x) <= BESIDE_NEAR),
                   default=float("inf"))

    landings = {f: [(to_frame(n["start"]), n["x"]) for n, _b, _e in items]
                for f, items in per_finger.items()}

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

    # Which frames each digit spends giving a key BACK, so those spans can be
    # keyed with the ease that belongs to them (see set_interpolation).
    releases = {}

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
            # ...and never later than the finger BESIDE it needs the room. The
            # tail is a nicety; being out of the way is not, and a finger that
            # is still lying across the next key when its neighbour lands has to
            # be got out of the way by the clearance search instead - which can
            # only do it in the one frame it notices, so the finger jumps. Given
            # the deadline it simply comes home a little sooner, along the ease
            # it was already leaving on. (See BESIDE_CLEAR_LEAD.)
            release_frame = min(release_frame,
                                beside_lands(f, key[0], off_frame)
                                - BESIDE_CLEAR_LEAD)
            if release_frame > off_frame + 0.25:
                # Lift off the key just played only while the finger is going
                # to stay in that neighbourhood - a repeat, or a quick
                # alternation. With a real gap ahead of it the finger comes
                # straight back over its own knuckle instead: by the time the
                # tail is over, the wrist has moved on, and a fingertip still
                # held above the key it played is a finger reaching backwards
                # across whatever the hand has arrived at.
                #
                # Soon in TIME is not enough for that: what settles it is
                # whether the hand is still there. On the reach take the right
                # thumb played B3, and its next note - three frames after the
                # tail - was two octaves up, so it hovered over B3 while the
                # hand leapt 168 mm away, stretched out backwards to a point
                # 111 mm behind its own wrist, and snapped forward in a single
                # frame when it finally let go. If the hand LIFTS OFF after this
                # event (lifts[ev_i], the same thing that makes the note itself
                # let go early), the key being hovered over is one the hand is
                # leaving, and the finger comes home instead.
                soon = (i + 1 < len(items)
                        and to_frame(items[i + 1][0]["start"])
                        - release_frame < idle_gap
                        and lifts[ev_i] <= 0.0)
                plan.append((release_frame, up, DIST_FLEX_HOVER, 1.0)
                            if soon else (release_frame,) + IDLE)
                releases.setdefault(f, []).append((off_frame, release_frame))
            prev_off = off_frame
            last_frame = max(last_frame, to_frame(n["end"]) + release_frames)
        plan.append((plan[-1][0] + idle_gap,) + IDLE)

    # --- resolve the whole hand, frame by frame ----------------------------
    # At every frame any digit is keyed at, all five are posed together and one
    # at a time, least free first: a finger holding a key down goes exactly where
    # its key is and becomes an obstacle, then the ones approaching or lifting
    # off, then the idle - each nudged aside and away (and, if that is not
    # enough, withdrawn toward its own knuckle) until it clears everything
    # already placed. Keying all five at every moment also means the f-curves
    # share their key times, so fingers that are apart at two keys stay apart
    # through the SINE ease between them - and where the ease still brushes them
    # together, the frame it happens on is solved too (REFINE_PASSES).
    wrist_keys.sort(key=lambda k: k[0])
    moments = sorted({frame_start}
                     | {fr for fr, _loc in wrist_keys}
                     | {fr for fr, _lift in arc_keys}
                     | {fr for fr, _pitch in pitch_keys}
                     | {e[0] for plan in plans.values() for e in plan})

    def solve_at(frame):
        """Pose and key all five digits at `frame`. Returns their worst
        clearance there - negative if two of them are in each other."""
        # The wrist samples already carry the arc; what the FINGERS need is its
        # height on its own, because every fingertip free to go with the hand
        # (`give`, 0 only while a key is actually held down) rides up by the
        # same amount. The hand then simply translates - the poses at the top of
        # the arc are the ones already solved for range of motion and mutual
        # clearance, just higher up.
        wx, wy, wz, yaw = _sample(wrist_keys, frame, _ease_wrist)
        lift = _sample(arc_keys, frame, _ease_wrist)[0]
        pitch = _sample(pitch_keys, frame, _ease_wrist)[0]
        # The whole hand is solved in ITS OWN frame (_hand_frame): the wrist at
        # the origin with the fingers along +y, which is the frame the bones are
        # posed in and the one every offset in build_hands.FINGERS is written
        # in. Only the keys have to be brought into it - turned by the yaw and
        # tilted by the wrist stroke's pitch, both of which the object carries
        # above the bones, so the fingers see a key that moves under a hand that
        # is otherwise still.
        wrist = (0.0, 0.0, wz)
        knuckles, digits = {}, []
        for f in FINGERS:
            knuckles[f] = _knuckle(f, wrist, mirror)
            target, flex, mix, give = _sample_plan(plans[f], frame)
            if target is not None:
                # The arc rides on the key target in WORLD height, before the
                # tilt: it is the hand leaving the keyboard, not the hand
                # turning over.
                target = _hand_frame((target[0], target[1],
                                      target[2] + lift * give),
                                     (wx, wy, wz), yaw, pitch)
            digits.append((max(mix, HOVER_GIVE * give), f, target, flex, mix,
                           give))
        # Least free first: the finger holding a key down is placed exactly
        # where its key is and becomes an obstacle, then the ones with a little
        # give, then the fully idle - so what has to move is what can.
        def place(order):
            """Key every digit in this order. Returns (worst clearance, the
            digit that got it) - the one everything before it left no room
            for."""
            placed, worst, blocked = [], float("inf"), None
            for budget, f, target, flex, mix, give in order:
                target, clear = _solve_clear(f, knuckles[f], mirror, wrist,
                                             hover_z + lift * give,
                                             flex, placed, target, mix, budget,
                                             pitch)
                if clear < worst:
                    worst, blocked = clear, (f if budget > 0.0 else None)
                roll = _target_roll(f, mirror, knuckles[f], target)
                pose = _cage_pose(f, mirror,
                                  _pose_from_target(f, mirror, knuckles[f],
                                                    target, flex), flex)
                _pose_finger(pbones, f, *pose, flex, frame, roll)
                placed.append((f, _finger_chain(knuckles[f],
                                                FINGERS[f]["lengths"],
                                                *pose, flex, roll)))
            return worst, blocked

        order = sorted(digits, key=lambda d: d[0])
        worst, blocked = place(order)
        if worst < 0.0 and blocked is not None:
            # Freest last is the right order for a hand whose fingers are all
            # doing something, and the wrong one when the digit that has to give
            # way is the one that CAN. An index on its way to a key, splayed to
            # its cap, is placed before the idle middle finger only because it
            # is the less free of the two - and then the middle finger, which
            # has the whole grid to move in, cannot get out from under a finger
            # that is not even sounding a note yet and would happily have waited
            # a few millimetres further out. So the digit left with no room goes
            # again, this time ahead of everything that is not holding a key
            # down: whoever is in its way then has to do the withdrawing. Kept
            # only if it actually helped.
            promoted = ([d for d in order if d[0] <= 0.0]
                        + [d for d in order if d[0] > 0.0 and d[1] == blocked]
                        + [d for d in order if d[0] > 0.0 and d[1] != blocked])
            again, _ = place(promoted)
            if again > worst:
                worst = again
            else:
                place(order)
        return worst

    solved = {}
    for frame in moments:
        solved[frame] = solve_at(frame)

    # Two easings, for two kinds of motion (see _ease_wrist): the wrist travels,
    # so it eases at both ends and arcs smoothly over the top of a leap; the
    # fingers strike, so they keep the one-sided SINE ramp that lands a
    # fingertip on its key at speed, in step with the key dips.
    #
    # Run the other way (EASE_OUT) over the span in which a digit is LETTING GO
    # of a key, which is the same motion in reverse: away at once, settling as
    # it arrives (_ease_off, which is how the poses inside the span were sampled
    # too, so the curve and the samples describe one shape).
    def set_interpolation():
        for fcurve in _iter_action_fcurves(arm_obj.animation_data
                                           and arm_obj.animation_data.action):
            travel = fcurve.data_path in ("location", "rotation_euler")
            spans = () if travel else releases.get(_curve_digit(fcurve), ())
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'BEZIER' if travel else 'SINE'
                if travel:
                    kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'
                else:
                    kp.easing = ('EASE_OUT'
                                 if any(a <= kp.co[0] < b for a, b in spans)
                                 else 'AUTO')
            fcurve.update()

    set_interpolation()

    # ...and then check what those easings actually do. Clearance solved at the
    # moments is clearance at the moments; between two of them the curves are
    # what the audience sees, and they can bulge two digits into each other on
    # the way. So every RENDERED frame is measured on the baked curves, and any
    # frame where two digits have got into each other is solved outright and
    # becomes a moment of its own (see REFINE_PASSES). What comes back is that
    # same measurement: the clearance the take will actually be seen with.
    first, last = int(math.floor(frame_start)), int(math.ceil(last_frame))
    worst = float("inf")
    for _ in range(REFINE_PASSES + 1):
        curves = {(fc.data_path, fc.array_index): fc
                  for fc in _iter_action_fcurves(arm_obj.animation_data.action)}
        seen = [(fr, _worst_clearance(_baked_chains(curves, mirror, fr))[0])
                for fr in range(first, last + 1)]
        worst = min((c for _fr, c in seen), default=float("inf"))
        bad = [fr for fr, c in seen if c < 0.0 and fr not in solved]
        if not bad:
            break
        for fr in bad:
            solved[fr] = solve_at(fr)
        set_interpolation()

    return last_frame, worst


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
    counts, clear = {}, {}
    for hand in ("L", "R"):
        arm_obj = bpy.data.objects.get(f"Hand_{hand}")
        if arm_obj is None:
            raise RuntimeError(
                f"Hand_{hand} not found - run build_hands.py first")
        notes = [n for n in data["notes"] if n["hand"] == hand]
        counts[hand] = len(notes)
        end, worst = animate_hand(
            arm_obj, notes, fps, frame_start,
            press_depth_white, press_depth_black,
            min_attack_frames, max_attack_frames, release_frames)
        last_frame = max(last_frame, end)
        # What the tightest moment of the take had left between two digits. A
        # negative number is a crossing the solve could not get out of - two
        # fingers asked to hold keys that the hand cannot hold at once - and is
        # a fingering to look at, not something the animation can fix.
        clear[hand] = None if worst == float("inf") else round(worst * 1000, 2)

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, int(round(last_frame)) + fps)
    scene.frame_set(frame_start)

    return {"notes_animated": counts, "frame_end": scene.frame_end,
            "fps": fps, "finger_clear_mm": clear}


if __name__ == "__main__":
    animate_hands(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "fingering.json"))
