"""Animates the drummer's sticks and feet from a drum fingering.json.

Consumes the output of ``python -m drum_kit.fingering`` (per-hit limb, target
object, strike type and world strike point) and keyframes the humanoid rig
built by drum_kit/build_drummer.py plus the animation-ready origins baked into
drum_kit/build_drum_kit.py:

  - Hands: each stick hit drives that hand's IK-target empty (IK_Hand_L /
    IK_Hand_R) through a wind-up -> contact -> rebound stroke to the drum's
    strike point; the drummer's two-bone arm IK reorients the shoulder, elbow
    and stick to reach it, so the swing pivots about the seated drummer's
    shoulder instead of the stick staying pointed at the audience. The stroke
    follows the MOELLER METHOD (see _moeller_strokes): each hit is an accent
    (a whole-arm whip -- the forearm leads down and the bead trails then cracks
    through) or a tap (a wrist flick), which sets that hit's backswing height so
    the stick rears up only to load an accent and rides low through taps.
    Velocity scales the accent apex height and strike speed. Each windup PEAKS
    just before its own downswing rather than snapping up early and hovering:
    after contact the stick takes only a small bounce and then climbs
    continuously into the next hit's apex, so it is always winding up or
    striking, never held still at the top. The motion is authored purely as
    target areas + velocities, so a different ANTHRO body re-solves the same
    targets. The vertical axis accelerates into contact and decelerates out of
    the rebound; travel glides.
  - Kick (Kick_Beater + Kick_Footboard + foot.R ankle): the beater cocks back
    (further when loud), swings into the batter head at the onset, and
    rebounds; the footboard and the drummer's right ankle stomp in step.
  - Hi-hat (HiHat_Top + HiHat_Footboard + foot.L ankle): the left foot holds
    the top cymbal open or closed following the planner's hi-hat timeline, with
    a quick pedal "chick" on each left-foot note.
  - Cymbals (Crash / Ride / HiHat_Top): a short velocity-scaled wobble decays
    after each stick hit.

Usage (inside Blender, after build_drum_kit.py + drum_kit/build_drummer.py)::

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
    from .build_drummer import (_rest_tip, shoulder, wrist_target,
                                 stick_pitch, home_voice, SEAT_YAW)
    from piano.piano_midi_animator import _iter_action_fcurves
except ImportError:  # loaded as a loose script via importlib
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    from build_drummer import (_rest_tip, shoulder, wrist_target,
                               stick_pitch, home_voice, SEAT_YAW)
    from piano.piano_midi_animator import _iter_action_fcurves


# --- velocity -> motion (louder = taller wind-up, faster strike) -----------
LIFT_MIN, LIFT_MAX = 0.03, 0.30        # accent apex height above the surface
STRIKE_SLOW, STRIKE_FAST = 0.12, 0.045  # apex->contact seconds (soft -> loud)

# --- Moeller stroke system (accent/tap backswing heights) ------------------
# Sanford Moeller's method plays each hit as either an ACCENT -- a whole-arm
# WHIP where the forearm leads down and the bead trails then snaps through, the
# "crack" -- or a TAP, a small wrist flick. Which one THIS hit is (plus its
# velocity) sets its BACKSWING HEIGHT: an accent rears up to a full,
# velocity-scaled apex, a tap barely lifts. So the stick rides low through a
# ghost-note roll and rears up only to load an accent (Moeller's down/tap/up
# shaping falls out of neighbouring hits simply having different heights).
#
# Crucially the windup is TIMED, not held: each stroke rises to peak exactly at
# the top of its own backswing just before the downswing, so between two hits
# the stick is always either winding up or swinging down -- never suspended at
# the top. After contact it takes only a small physical BOUNCE off the head and
# then immediately begins the next windup, a single continuous rise that arrives
# at the top just in time ("two notes for one arm movement", but never a hover).
# Because the peak right before a strike is that strike's OWN apex, the windup
# height tracks the note's volume: loud = high, soft = low.
ACCENT_ABS = 0.62          # v>=this (norm) is always an accent (absolute floor)
ACCENT_REL = 0.18          # ...or this much louder than the hand's median hit
TAP_LIFT = 0.028           # backswing height of a low (tap) stroke, m
REBOUND_MIN, REBOUND_MAX = 0.018, 0.065  # small bounce off the head (soft->loud)
MOELLER_RESET_GAP = 0.6    # s; a longer gap is a real rest -> return the hand home
# A real rest is not a slow-motion drift across the kit. When the gap to the next
# hit is a genuine rest (rest_after), the idle hand pulls back to its NEUTRAL HOME
# (the _rest_tip over its convention voice, close to the body) shortly after the
# last stroke, HOLDS there, and only leaves to approach the next drum a short beat
# before that hit's windup -- rather than SINE-gliding straight from one drum to
# the next across the whole empty gap. Crucially the home pull only happens when
# the neighbouring hit is on a DIFFERENT spot: repeated hits on the SAME drum (a
# crash ridden once a bar) keep the hand poised over it and merely bob for each
# windup, instead of pumping all the way back to the body between identical strokes.
HOME_SETTLE = 0.30         # s; ease the idle hand back to home after a rest-ending hit
HOME_APPROACH = 0.34       # s; leave home this long before the next hit's windup
SAME_SPOT = 0.06           # m; hits closer than this count as the same drum (no home pull)
# The whip: an accent LEADS with the arm -- the wrist/elbow drops to play height
# before the bead reaches the head, so the stick trails and then cracks through;
# a tap is wrist-only. So the forearm bob is present on accents, small on taps,
# and only accents get the early wrist lead-in.
#
# The forearm bob raises the WRIST empty by this fraction of the tip's apex; the
# hand bone then ROTATES at the wrist to cover the remaining (1-flex) of the
# rise. So a smaller fraction means the arm heaves less and the WRIST FLICK
# supplies more of the height the bead reaches -- the raised stick reads as a
# wrist whip rather than a whole-arm lift.
FOREARM_FLEX_ACCENT = 0.3              # accent forearm bob, fraction of apex height
FOREARM_FLEX_TAP = 0.12                # tap wrist bob (mostly a wrist flick)
WHIP_LEAD = 0.38                       # accent: arm reaches the anchor this
                                       # fraction of the way early (bead trails)

# --- kick ------------------------------------------------------------------
BEATER_STRIKE = math.radians(6.0)      # beater angle at contact (into head)
BEATER_COCK_MIN, BEATER_COCK_MAX = math.radians(5), math.radians(18)  # backswing
BOARD_PRESS_MIN, BOARD_PRESS_MAX = 0.11, 0.30  # footboard toe-down, rad
# Foot bones share an up-aligned roll (build_drummer), so a NEGATIVE local-X
# rotation lowers the toe (a pedal press) on either foot.
ANKLE_KICK = -0.34                     # drummer's ankle press on a kick, rad
ANKLE_HAT = -0.20                      # ankle press for a hi-hat chick, rad
# Heel-up vs heel-down technique. Louder/faster notes drive the LEG (the knee/
# ankle bobs, heel lifts, the toe stays on the pedal); soft notes barely move the
# leg and are mostly ankle rotation (heel down, whole foot on the pedal). The
# knee drive moves the Ankle IK target: up to cock (heel up), down to strike.
HEEL_LIFT = 0.055                      # ankle/knee rise to cock (m), scaled by vel
KNEE_DRIVE = 0.03                      # ankle/knee drop into the strike (m), by vel

# --- hi-hat ----------------------------------------------------------------
HAT_CLOSE_DROP = 0.039                 # how far HiHat_Top slides down to close
HAT_FOOT_PRESS = math.radians(15)      # footboard press while closed

# --- torso (spine twist + lean toward the centre of activity) --------------
# The upper body turns to FACE the centre of activity. The spine twist aims the
# chest a fraction (TWIST_FOLLOW) of the way to the bearing of the active drum(s),
# and must overcome the seated body's baseline facing (SEAT_YAW, ~18 deg toward the
# hi-hat/snare on +X) to turn toward the RIGHT-side kit (floor tom, ride on -X) --
# so `twist = TWIST_FOLLOW*bearing - SEAT_YAW`. Facing right needs the larger swing,
# hence a generous cap.
TWIST_FOLLOW = 0.72                    # fraction of the activity bearing the chest faces
TWIST_MAX = math.radians(34)           # cap the spine twist (either direction)
# The torso must not turn so far RIGHT that the left hand's current target sits too
# far to the left of where the chest faces: past this margin the left stick reaches
# across (the snare backbeat flips) AND the left elbow rides into the chest. Capping
# the rightward twist by the LEFT hand's target holds the torso near centre while a
# snare is live, yet lets it turn fully right for a same-side fill (both hands on the
# toms, where this never binds). The right hand reaching its own side needs no cap.
REACH_MARGIN = math.radians(20)
# The drummer sits upright and leans forward only in proportion to how far
# forward the hands are reaching: none for the snare / hi-hat groove (close in),
# growing as they reach out to the toms and crashes. (A big lean also tilts the
# chest into the path of the cross-body arms, so the max is kept modest.)
LEAN_REACH = math.radians(9)           # max forward lean, at the farthest reach
REACH_NEAR = 0.34                      # forward reach (m) at/below which: upright
REACH_FAR = 0.72                       # forward reach (m) at which the lean is full
TORSO_SAMPLE = 0.18                    # seconds between torso keyframes
TORSO_WIN = 0.35                       # smoothing window for the activity centre
# The lean must COMMIT: a run of reaches to the same far target (e.g. the crash
# struck once a beat) leaves the ±TORSO_WIN activity window empty in the gaps
# between hits, which would sag the torso back to upright and then forward again
# on every stroke -- a random-looking bob unrelated to the beat. So the sampled
# lean is morphologically CLOSED over ±LEAN_HOLD seconds: dips shorter than
# 2*LEAN_HOLD (the gaps between reaches of one passage) are filled so the lean is
# held for the whole passage, while a gap wider than that (the reaching has truly
# stopped -- a lone crash bars later) still releases the drummer to upright.
LEAN_HOLD = 0.85
# The twist (the gaze/facing) needs the same steadying, but it is BIDIRECTIONAL --
# it swings left AND right -- so closing (which only fills dips toward a peak) does
# not fit. Instead the sampled twist is MOVING-AVERAGED over ±TWIST_SMOOTH seconds:
# a hand briefly idling between hits (which jumps the activity centre to the other
# hand and whips the head back and forth) is averaged out, so the gaze holds on the
# centre of a passage and turns only when the activity genuinely, sustainedly moves.
TWIST_SMOOTH = 0.6

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


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _apply_stick_easing(obj, roles):
    """Shape each hand's motion so it flows rather than teleports.

    The travel axes glide (SINE ease-in-out) between targets, but the vertical
    strike axis accelerates *into* the contact (fast at impact, where the
    swing's energy goes into the hit) and decelerates *out* of the rebound.
    Contact is a fast pass, not a full stop - what a real stroke looks like.
    """
    def role_of(frame):
        best, bd = None, 1e9
        for f, role in roles:
            d = abs(f - frame)
            if d < bd:
                best, bd = role, d
        return best if bd < 0.51 else None

    for fc in _iter_action_fcurves(obj.animation_data.action):
        strike_axis = fc.data_path == "location" and fc.array_index == 2
        for kp in fc.keyframe_points:
            role = role_of(kp.co[0])
            if strike_axis and role == "apex":
                kp.interpolation, kp.easing = 'CUBIC', 'EASE_IN'    # accelerate down
            elif strike_axis and role == "contact":
                kp.interpolation, kp.easing = 'CUBIC', 'EASE_OUT'   # rebound, then settle
            else:
                kp.interpolation, kp.easing = 'SINE', 'EASE_IN_OUT'  # smooth glide


def _moeller_strokes(notes):
    """Classify a hand's hits by the Moeller accent/tap system and derive each
    stroke's backswing HEIGHT plus its post-strike behaviour.

    A hit is an ACCENT if it is loud in absolute terms (>= ACCENT_ABS) or stands
    out from the hand's own median dynamic (>= ACCENT_REL louder) -- so a flat
    ghost-note roll is all taps, while a backbeat or the loud end of a crescendo
    pops as accents. The backswing height (``pre_h``) is a full velocity-scaled
    apex for an accent, a small TAP_LIFT for a tap: each stroke rears up to its
    OWN height, so the windup a viewer sees before a strike tracks that note's
    volume. The stick does not preload the next hit's height and then wait --
    instead each stroke reaches its peak just before its downswing (see
    _animate_arm), so the next hit's cock IS the continuous rise out of this
    hit's small rebound. ``rest_after`` marks a gap long enough to be a genuine
    rest, where the hand instead pulls back to its neutral home (see _animate_arm).
    Returns a per-hit list aligned to ``notes`` sorted by start: {accent, pre_h,
    rebound, rest_after}."""
    ns = sorted(notes, key=lambda m: m["start"])
    if not ns:
        return []
    vs = sorted(n["velocity"] for n in ns)
    baseline = vs[len(vs) // 2] / 127.0            # median hit -> normalized
    def apex(n):                                   # a full backswing by volume
        v = _clamp(n["velocity"] / 127.0, 0.0, 1.0)
        return LIFT_MIN + v * (LIFT_MAX - LIFT_MIN)
    def is_accent(n):
        v = n["velocity"] / 127.0
        return v >= ACCENT_ABS or (v - baseline) >= ACCENT_REL
    def bounce(n):                                 # small velocity-scaled bounce
        v = _clamp(n["velocity"] / 127.0, 0.0, 1.0)
        return REBOUND_MIN + v * (REBOUND_MAX - REBOUND_MIN)
    acc = [is_accent(n) for n in ns]

    out = []
    for i, n in enumerate(ns):
        pre_h = apex(n) if acc[i] else TAP_LIFT
        rest_after = (i + 1 >= len(ns) or
                      ns[i + 1]["start"] - n["start"] > MOELLER_RESET_GAP)
        out.append({"accent": acc[i], "pre_h": pre_h,
                    "rebound": bounce(n), "rest_after": rest_after})
    return out


def _animate_arm(target, wrist, side, notes, fps, frame_start):
    """Drive a hand's stick-tip empty (`target`) and its wrist empty so the
    stroke comes from the WRIST, not the whole arm.

    A real drummer keeping a groove holds the forearm/elbow fairly still and
    flicks the stick with the wrist; only when moving to a different drum does
    the arm travel. The rig supports this directly: the forearm IK reaches the
    wrist empty (placing the elbow/forearm) while the hand bone Damped-Tracks
    the tip empty independently. So we keyframe the wrist empty ONLY at each
    hit's contact-height anchor and let it *hold* across the stroke -- for
    repeated hits on one drum the anchor is identical frame to frame, so the arm
    is motionless and the hand bone pivots at the wrist to lift and drop the
    tip. Between different drums the anchor changes, so the wrist glides there
    (the arm relocates) over the gap. The tip empty still winds up -> contacts
    -> rebounds; the wind-up is now a wrist rotation, not an arm bob.

    Because the anchor is wrist_target(sh, contact_point) (from the contact, not
    the raised apex), |wrist - tip| == STICK_LEN at contact and the tip lands
    exactly on the head."""
    sh = shoulder(side)
    tframe = _monotonic(fps, frame_start)   # tip empty (3 keys/hit)
    wframe = _monotonic(fps, frame_start)   # wrist empty (1 key/hit -> holds)
    roles = []
    strokes = []                # (apex_f, contact_f, rebound_f, p, pre_h, post_h, anchor)
    last = frame_start
    cls = _moeller_strokes(notes)

    rest = _rest_tip(side)
    tip_home = mathutils.Vector(rest)
    wrist_home = mathutils.Vector(wrist_target(sh, rest, stick_pitch(home_voice(side))))
    target.location = rest
    target.keyframe_insert(data_path="location", frame=float(frame_start))
    roles.append((float(frame_start), "rest"))
    tframe(0.0)
    wrist.location = wrist_home
    wrist.keyframe_insert(data_path="location", frame=float(frame_start))
    wframe(0.0)

    def key_tip(t, loc, role):
        f = tframe(t)
        target.location = loc
        target.keyframe_insert(data_path="location", frame=f)
        roles.append((f, role))
        return f

    def key_wrist(t, loc):
        wrist.location = loc
        wrist.keyframe_insert(data_path="location", frame=wframe(t))

    ns = sorted(notes, key=lambda m: m["start"])
    prev_t = None
    for idx, n in enumerate(ns):
        t = n["start"]
        p = mathutils.Vector((n["x"], n["y"], n["z"]))
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        gap = (t - prev_t) if prev_t is not None else 1.0
        # Rapid hits get a shorter wind-up so the tip stays near the drum
        # instead of flinging up and darting between poses.
        busy = _clamp(gap / 0.30, 0.4, 1.0)
        # Moeller: this hit's backswing height (pre_h) comes from its accent/tap
        # type and velocity -- an accent rears up high, a tap barely lifts. The
        # apex is placed just before the downswing (t - down) so the stick is
        # still RISING right up to it; after contact it takes only a small bounce
        # (post_h) and then climbs continuously into the NEXT hit's apex, so it is
        # never suspended at the top. A genuine rest instead pulls the hand HOME.
        accent = cls[idx]["accent"]
        pre_h = cls[idx]["pre_h"] * busy
        post_h = cls[idx]["rebound"] * busy
        strike = STRIKE_SLOW - v * (STRIKE_SLOW - STRIKE_FAST)
        down = min(strike, 0.40 * gap)    # apex->contact; never precede the last hit
        up = min(0.07, 0.45 * gap)        # contact->rebound
        # A rest before this hit (or the opening rest) means the hand is parked at
        # its neutral home; it stays there and only travels out to the drum a beat
        # (HOME_APPROACH, clamped to the gap) before the windup rather than drifting
        # across the whole gap. A rest AFTER this hit brings the hand back home.
        # But only when the OTHER hit is on a different spot -- a rest spent
        # repeating the SAME drum keeps the hand out over it (see SAME_SPOT), so a
        # crash ridden once a bar bobs in place instead of retracting every stroke.
        same_prev = idx > 0 and (mathutils.Vector(
            (ns[idx - 1]["x"], ns[idx - 1]["y"], ns[idx - 1]["z"])) - p).length < SAME_SPOT
        same_next = idx + 1 < len(ns) and (mathutils.Vector(
            (ns[idx + 1]["x"], ns[idx + 1]["y"], ns[idx + 1]["z"])) - p).length < SAME_SPOT
        rested_before = idx == 0 or (cls[idx - 1]["rest_after"] and not same_prev)
        rest_after = cls[idx]["rest_after"] and not same_next
        next_gap = (ns[idx + 1]["start"] - t) if idx + 1 < len(ns) else 2.0 * HOME_SETTLE

        if rested_before:
            approach = min(HOME_APPROACH, 0.5 * gap)
            key_tip(t - down - approach, tip_home, "home")   # hold home, then approach
            key_wrist(t - down - approach, wrist_home)

        # Arm: on an accent the forearm/elbow WHIPS -- it cocks up with the
        # backswing, then drops to the anchor EARLY (WHIP_LEAD) so the arm leads
        # and the bead trails before snapping through, and it bobs a large
        # fraction of the stroke. On a tap the arm barely moves (a wrist flick).
        # It always sits back exactly on the anchor at contact, so
        # |wrist - tip| == STICK_LEN and the bead still lands on the head.
        anchor = mathutils.Vector(wrist_target(sh, p, stick_pitch(n["voice"])))
        flex = FOREARM_FLEX_ACCENT if accent else FOREARM_FLEX_TAP
        key_wrist(t - down, anchor + mathutils.Vector((0.0, 0.0, flex * pre_h)))  # cock up
        if accent:
            key_wrist(t - down * (1.0 - WHIP_LEAD), anchor)   # arm whips down first
        key_wrist(t, anchor)                                  # settled onto the head
        key_wrist(t + up, anchor + mathutils.Vector((0.0, 0.0, flex * post_h)))
        af = key_tip(t - down, p + mathutils.Vector((0.0, 0.0, pre_h)), "apex")  # cocks up
        cf = key_tip(t, p, "contact")                                           # onto the head
        rf = key_tip(t + up, p + mathutils.Vector((0.0, 0.0, post_h)), "rebound")
        last = rf
        strokes.append((af, cf, rf, p.copy(), pre_h, post_h, anchor.copy()))

        if rest_after:
            settle = min(HOME_SETTLE, 0.4 * next_gap)
            last = key_tip(t + up + settle, tip_home, "home")   # pull back to neutral
            key_wrist(t + up + settle, wrist_home)
        prev_t = t

    _apply_stick_easing(target, roles)
    # The wrist empty only ever travels between drums (it never strikes), so
    # glide it smoothly; holds between same-drum hits keep the arm still.
    for fc in _iter_action_fcurves(wrist.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation, kp.easing = 'SINE', 'EASE_IN_OUT'
    return last, strokes


def _replane_strokes(target, side, strokes):
    """Re-aim each stroke's wind-up/rebound so the bead cocks straight UP in the
    stick's VERTICAL plane -- perpendicular to the stick, along gravity -- rather
    than straight up in world Z (which twisted the wrist) or in the forearm-stick
    plane (which flung the cross-body left bead sideways -- "all the way left").

    Lifting perpendicular to the stick keeps it a clean wrist hinge (no change in
    |wrist - tip|), and taking the vertical component of that keeps the swing in
    the vertical plane so the bounce is gravity-aligned. For a forward-pointing
    stick (the snare) this cocks essentially straight up; for a stick angled up
    and across (the hats/ride) it tilts to stay perpendicular but stays vertical.
    The offset magnitude (lift/HOVER) and the CONTACT key are untouched, so the
    tip still lands exactly on the head. At contact wrist == anchor and tip == p,
    so the stick direction is simply p - anchor (no rig readback needed)."""
    if not strokes or target.animation_data is None:
        return
    fcs = {}
    for fc in _iter_action_fcurves(target.animation_data.action):
        if fc.data_path == "location":
            fcs[fc.array_index] = fc

    def set_key(frame, loc):
        for i in (0, 1, 2):
            fc = fcs.get(i)
            if fc is None:
                continue
            for kp in fc.keyframe_points:
                if abs(kp.co[0] - frame) < 0.01:
                    kp.co[1] = loc[i]
                    break
        for fc in fcs.values():
            fc.update()

    z = mathutils.Vector((0.0, 0.0, 1.0))
    for af, cf, rf, p, pre_h, post_h, anchor in strokes:
        stick = p - anchor                            # wrist -> bead (contact)
        if stick.length < 1e-6:
            continue
        stick.normalize()
        lift_dir = z - z.dot(stick) * stick           # up, perpendicular to stick
        lift_dir = lift_dir.normalized() if lift_dir.length > 1e-6 else z
        set_key(af, p + pre_h * lift_dir)             # Moeller backswing height
        set_key(rf, p + post_h * lift_dir)            # small bounce off the head


def _key_ankle(arm, bone, frame_fn):
    """Return a keyframer for a foot bone's local-X press (toe-down)."""
    pb = arm.pose.bones[bone]
    path = f'pose.bones["{bone}"].rotation_euler'

    def key(t, press):
        pb.rotation_euler.x = press
        arm.keyframe_insert(data_path=path, index=0, frame=frame_fn(t))

    return key


def _animate_kick(beater, board, arm, ankle, ankle_empty, notes, fps, frame_start):
    rest_b = _cached_rot(beater)[0]
    rest_board = _cached_rot(board)[0]
    frame = _monotonic(fps, frame_start)
    last = frame_start
    faz = _monotonic(fps, frame_start)
    az0 = _cached_z(ankle_empty)

    def key_az(t, z):
        ankle_empty.location.z = z
        ankle_empty.keyframe_insert(data_path="location", index=2, frame=faz(t))

    def key_beater(t, rot_x):
        beater.rotation_euler.x = rot_x
        beater.keyframe_insert(data_path="rotation_euler", index=0, frame=frame(t))

    fb = _monotonic(fps, frame_start)

    def key_board(t, rot_x):
        board.rotation_euler.x = rest_board + rot_x
        board.keyframe_insert(data_path="rotation_euler", index=0, frame=fb(t))

    fa = _monotonic(fps, frame_start)
    key_ankle = _key_ankle(arm, ankle, fa)

    beater.rotation_euler.x = rest_b
    beater.keyframe_insert(data_path="rotation_euler", index=0, frame=frame_start)
    board.rotation_euler.x = rest_board
    board.keyframe_insert(data_path="rotation_euler", index=0, frame=frame_start)
    key_ankle(0.0, 0.0)
    key_az(0.0, az0)
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
        key_board(t + 0.14, 0.0)                # release
        # Heel-up (loud) drives the leg: the knee/ankle bobs up to cock then down
        # into the strike, and the toe plantar-flexes so it stays on the pedal as
        # the heel lifts. Soft notes barely bob -> mostly ankle, heel down.
        key_az(t - 0.10, az0 + HEEL_LIFT * v)          # cock: knee up, heel lifts
        key_ankle(t - 0.10, ANKLE_KICK * 0.5 * v)      # toe stays down on the pedal
        key_az(t, az0 - KNEE_DRIVE * v)                # strike: knee drives down
        key_ankle(t, ANKLE_KICK * (0.6 + 0.4 * v))     # foot press
        key_az(t + 0.16, az0)                          # settle back
        key_ankle(t + 0.16, 0.0)
        last = max(last, frame_start + (t + 0.16) * fps)
    return last


def _animate_hihat(top, board, arm, ankle, ankle_empty, hihat_events, pedal_notes, fps, frame_start):
    rest_z = _cached_z(top)
    rest_board = _cached_rot(board)[0]
    open_z, closed_z = rest_z, rest_z - HAT_CLOSE_DROP

    top_frame = _monotonic(fps, frame_start)
    board_frame = _monotonic(fps, frame_start)
    ankle_frame = _monotonic(fps, frame_start)
    key_ankle = _key_ankle(arm, ankle, ankle_frame)
    faz = _monotonic(fps, frame_start)
    az0 = _cached_z(ankle_empty)

    def key_az(t, z):
        ankle_empty.location.z = z
        ankle_empty.keyframe_insert(data_path="location", index=2, frame=faz(t))
    key_az(0.0, az0)

    def key_top(t, z):
        top.location.z = z
        top.keyframe_insert(data_path="location", index=2, frame=top_frame(t))

    def key_board(t, rot_x):
        board.rotation_euler.x = rest_board + rot_x
        board.keyframe_insert(data_path="rotation_euler", index=0, frame=board_frame(t))

    # Start closed (the foot holding the hats down under the groove).
    key_top(0.0, closed_z)
    key_board(0.0, HAT_FOOT_PRESS)
    key_ankle(0.0, ANKLE_HAT)

    prev_state = "closed"
    for e in hihat_events:
        t, state = e["t"], e["state"]
        z = open_z if state == "open" else closed_z
        b = 0.0 if state == "open" else HAT_FOOT_PRESS
        a = 0.0 if state == "open" else ANKLE_HAT
        pz = open_z if prev_state == "open" else closed_z
        pb = 0.0 if prev_state == "open" else HAT_FOOT_PRESS
        pa = 0.0 if prev_state == "open" else ANKLE_HAT
        key_top(max(0.0, t - 0.03), pz)   # hold the old state right up to the change
        key_board(max(0.0, t - 0.03), pb)
        key_ankle(max(0.0, t - 0.03), pa)
        key_top(t, z)
        key_board(t, b)
        key_ankle(t, a)
        prev_state = state

    # A quick pedal "chick" on each left-foot note: lift the toe then stomp.
    for n in sorted(pedal_notes, key=lambda m: m["start"]):
        t = n["start"]
        key_board(t - 0.05, HAT_FOOT_PRESS * 0.3)
        key_board(t, HAT_FOOT_PRESS * 1.1)
        key_board(t + 0.07, HAT_FOOT_PRESS)
        key_ankle(t - 0.05, ANKLE_HAT * 0.3)
        key_ankle(t, ANKLE_HAT * 1.1)
        key_ankle(t + 0.07, ANKLE_HAT)
        # A pedal chick is a heel-UP move: the knee lifts (heel up) then drives
        # down to stomp the hats shut. (Open/closed above is heel-DOWN: the knee
        # stays and only the ankle rocks the pedal.)
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        key_az(t - 0.06, az0 + HEEL_LIFT * (0.7 + 0.3 * v))   # heel up, cock
        key_az(t, az0 - KNEE_DRIVE * (0.7 + 0.3 * v))         # knee drives, stomp
        key_az(t + 0.10, az0)


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


def _time_closing(times, vals, radius):
    """Morphological CLOSING of a time series: dilate (max) then erode (min), each
    over a ±`radius`-second window about every sample. Fills short dips BETWEEN
    peaks -- so a value held across a run of closely spaced peaks stops sagging in
    the gaps -- while leaving wider gaps, and the outer edges of a run, where they
    are. `times` must be ascending."""
    def sweep(src, op):
        out = []
        for ti in times:
            lo, hi = ti - radius, ti + radius
            out.append(op(v for v, tj in zip(src, times) if lo <= tj <= hi))
        return out
    return sweep(sweep(vals, max), min)


def _time_smooth(times, vals, radius):
    """Moving-average of a time series over a ±`radius`-second window about every
    sample. Damps rapid back-and-forth (a bidirectional signal) without shifting a
    sustained level, so a value that genuinely moves still follows. `times` must be
    ascending."""
    out = []
    for ti in times:
        lo, hi = ti - radius, ti + radius
        win = [v for v, tj in zip(vals, times) if lo <= tj <= hi]
        out.append(sum(win) / len(win))
    return out


def _animate_torso(arm, stick_notes, fps, frame_start):
    """Twist and lean the spine toward the centre of what the hands are
    playing. The spine's local Y twists the upper body (left/right), local X
    leans it forward; sampling a smoothed activity centre and SINE-easing the
    keyframes gives natural acceleration and deceleration. The lean is then
    COMMITTED across a reaching passage (see LEAN_HOLD) so it holds steady while
    the hands keep reaching out instead of bobbing back on every stroke."""
    if "spine" not in arm.pose.bones:
        return frame_start
    sp = arm.pose.bones["spine"]
    path = 'pose.bones["spine"].rotation_euler'
    frame = _monotonic(fps, frame_start)

    def key(t, lean, twist):
        sp.rotation_euler = (lean, twist, 0.0)
        arm.keyframe_insert(data_path=path, frame=frame(t))

    # Chest pivot (shoulder midpoint): the origin the activity bearing is measured
    # from and about which the twist turns.
    cxp = (shoulder("L")[0] + shoulder("R")[0]) / 2.0
    cyp = (shoulder("L")[1] + shoulder("R")[1]) / 2.0
    sh_y = shoulder("L")[1]               # seated shoulder Y (the reach origin)
    key(0.0, 0.0, 0.0)                    # rest: seated baseline facing (SEAT_YAW)
    notes = sorted(stick_notes, key=lambda n: n["start"])
    if not notes:
        return frame_start
    by = {"R": [n for n in notes if n["limb"] == "R"],
          "L": [n for n in notes if n["limb"] == "L"]}

    def hand_centre(hand_notes, t):
        """Proximity-weighted target of ONE hand near time t (or None if idle)."""
        ws = [(n, 1.0 / (0.06 + abs(n["start"] - t)))
              for n in hand_notes if abs(n["start"] - t) <= TORSO_WIN]
        if not ws:
            return None
        w = sum(x for _, x in ws)
        return (sum(n["x"] * x for n, x in ws) / w,
                sum(n["y"] * x for n, x in ws) / w)

    t, t_end = notes[0]["start"], notes[-1]["start"]
    # Pass 1: sample the raw per-frame twist and lean demand.
    times, twists, leans = [], [], []
    while t <= t_end + 1e-6:
        # Face the midpoint of where the TWO hands are, weighting each hand equally
        # (not each note): so a split ride-right + snare-left groove compromises
        # near centre and keeps both reachable, while a same-side fill (both on the
        # floor tom) turns the torso fully toward it. Over-weighting the busier hand
        # used to twist so far that the other hand's stick had to flip to reach.
        cR, cL = hand_centre(by["R"], t), hand_centre(by["L"], t)
        hands = [c for c in (cR, cL) if c]
        if hands:
            cx = sum(c[0] for c in hands) / len(hands)
            cy = sum(c[1] for c in hands) / len(hands)
            bearing = math.atan2(cx - cxp, cyp - cy)            # +left / -right
            # Twist RELATIVE to the seated baseline facing (SEAT_YAW): no twist when
            # the activity sits at the baseline, turning further only as it moves
            # off it. Turning right (toward the floor tom / ride) therefore has to
            # cross the whole ~18 deg left bias, which the follow factor scales.
            twist = TWIST_FOLLOW * (bearing - SEAT_YAW)
            # Don't turn further right than the LEFT hand allows (facing == SEAT_YAW
            # + twist): when the left hand is on the snare this holds the torso near
            # centre so the backbeat stays reachable, but when it is also on the
            # right (a tom fill) the floor never binds and the torso turns fully
            # toward the kit's right side.
            if cL is not None:
                bL = math.atan2(cL[0] - cxp, cyp - cL[1])
                twist = max(twist, (bL - REACH_MARGIN) - SEAT_YAW)
            else:
                # Left hand idle (e.g. a ride passage): hold at/left of the seated
                # baseline rather than turning right, so the resting left arm is not
                # dragged into the torso. It turns right only when the left hand is
                # actively on the right side (a tom fill).
                twist = max(twist, 0.0)
            twist = _clamp(twist, -TWIST_MAX, TWIST_MAX)
            reach = sh_y - cy
            lean = LEAN_REACH * _clamp((reach - REACH_NEAR) / (REACH_FAR - REACH_NEAR),
                                       0.0, 1.0)
        else:
            twist, lean = 0.0, 0.0                              # idle -> seated baseline
        times.append(t)
        twists.append(twist)
        leans.append(lean)
        t += TORSO_SAMPLE

    # Pass 2: commit the lean over each reaching passage, then keyframe. The raw
    # lean spikes once per reach and sags to upright in the (noteless) gaps
    # between strokes -- closing it holds the lean while the reaching continues.
    leans = _time_closing(times, leans, LEAN_HOLD)
    twists = _time_smooth(times, twists, TWIST_SMOOTH)
    last = frame_start
    for tt, tw, ln in zip(times, twists, leans):
        f = frame(tt)
        sp.rotation_euler = (ln, tw, 0.0)
        arm.keyframe_insert(data_path=path, frame=f)
        last = f
    return last


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

    # The hands are IK-target empties on the drummer; the feet are ankle bones
    # on the Drummer armature. Everything reaches its target via the rig.
    drummer = find("Drummer")
    ik = {"R": find("IK_Hand_R"), "L": find("IK_Hand_L")}
    wrist = {"R": find("Wrist_R"), "L": find("Wrist_L")}
    beater, kboard = find("Kick_Beater"), find("Kick_Footboard")
    htop, hboard = find("HiHat_Top"), find("HiHat_Footboard")
    crash, ride = find("Crash"), find("Ride")
    ankle_e = {"R": find("Ankle_R"), "L": find("Ankle_L")}   # leg-IK targets (knee drive)

    hand_empties = [ik["R"], ik["L"], wrist["R"], wrist["L"]]
    obj_touched = hand_empties + [beater, kboard, htop, hboard, crash, ride,
                                  ankle_e["R"], ankle_e["L"]]

    # Cache rest transforms (first run only) BEFORE clearing, then wipe old
    # keyframes so re-running is idempotent.
    for obj in (beater, kboard, hboard, crash, ride):
        if obj is not None:
            _cached_rot(obj)
    if htop is not None:
        _cached_z(htop)
        _cached_rot(htop)
    for e in (ankle_e["R"], ankle_e["L"]):
        if e is not None:
            _cached_z(e)          # cache the pedal-height rest before clearing
    for obj in obj_touched:
        if obj is not None and obj.animation_data is not None:
            obj.animation_data_clear()
    if drummer is not None and drummer.animation_data is not None:
        drummer.animation_data_clear()   # foot-bone (ankle) keyframes

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
    stroke_rec = {}
    for side in ("R", "L"):
        if ik[side] is not None and wrist[side] is not None:
            lst, stroke_rec[side] = _animate_arm(ik[side], wrist[side], side,
                                                 by_hand[side], fps, frame_start)
            last = max(last, lst)
    # Re-aim each hand's wind-up/rebound to cock straight up in the stick's
    # vertical plane (gravity-aligned, no sideways fling).
    for side in ("R", "L"):
        if ik[side] is not None and side in stroke_rec:
            _replane_strokes(ik[side], side, stroke_rec[side])
    if beater is not None and kboard is not None and drummer is not None:
        last = max(last, _animate_kick(beater, kboard, drummer, "foot.R",
                                       ankle_e["R"], kick, fps, frame_start))
    if htop is not None and hboard is not None and drummer is not None:
        _animate_hihat(htop, hboard, drummer, "foot.L", ankle_e["L"],
                       hihat_events, pedal, fps, frame_start)
    if drummer is not None:
        last = max(last, _animate_torso(drummer, by_hand["R"] + by_hand["L"],
                                        fps, frame_start))
    for name, obj in (("Crash", crash), ("Ride", ride), ("HiHat_Top", htop)):
        if obj is not None and wobble[name]:
            _animate_cymbal(obj, wobble[name], fps, frame_start)

    # Blanket SINE for the pedals/cymbals/feet; the hand tip/wrist empties keep
    # the per-axis accelerate-into-contact easing set by _animate_arm.
    for obj in obj_touched + [drummer]:
        if obj is None or obj.animation_data is None or obj in hand_empties:
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
