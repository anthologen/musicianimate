"""Animates the drummer's sticks and feet from a drum fingering.json.

Consumes the output of ``python -m drum_kit.fingering`` (per-hit limb, target
object, strike type and world strike point) and keyframes the humanoid rig
built by drum_kit/build_drummer.py plus the animation-ready origins baked into
drum_kit/build_drum_kit.py:

  - Hands: each stick hit drives that hand's IK-target empty (IK_Hand_L /
    IK_Hand_R) through a wind-up -> contact -> rebound stroke to the drum's
    strike point; the drummer's two-bone arm IK reorients the shoulder, elbow
    and stick to reach it, so the swing pivots about the seated drummer's
    shoulder instead of the stick staying pointed at the audience. Velocity
    sets the wind-up height and strike speed (loud = a taller, faster stroke),
    and the motion is authored purely as target areas + velocities, so a
    different ANTHRO body re-solves the same targets. The vertical axis
    accelerates into contact and decelerates out of the rebound; travel glides.
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
    from .build_drummer import _rest_tip, shoulder, wrist_target
    from piano.piano_midi_animator import _iter_action_fcurves
except ImportError:  # loaded as a loose script via importlib
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    from build_drummer import _rest_tip, shoulder, wrist_target
    from piano.piano_midi_animator import _iter_action_fcurves


# --- velocity -> motion (louder = taller wind-up, faster strike) -----------
LIFT_MIN, LIFT_MAX = 0.03, 0.14        # stick apex height above the surface
STRIKE_SLOW, STRIKE_FAST = 0.12, 0.045  # apex->contact seconds (soft -> loud)
HOVER = 0.03                            # rebound height above the surface

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
TWIST_GAIN = 0.32                      # rad of spine twist per metre of centre-x
TWIST_MAX = math.radians(15)           # cap the twist
# The drummer sits upright and leans forward only in proportion to how far
# forward the hands are reaching: none for the snare / hi-hat groove (close in),
# growing as they reach out to the toms and crashes. (A big lean also tilts the
# chest into the path of the cross-body arms, so the max is kept modest.)
LEAN_REACH = math.radians(9)           # max forward lean, at the farthest reach
REACH_NEAR = 0.34                      # forward reach (m) at/below which: upright
REACH_FAR = 0.72                       # forward reach (m) at which the lean is full
TORSO_SAMPLE = 0.18                    # seconds between torso keyframes
TORSO_WIN = 0.35                       # smoothing window for the activity centre

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
    last = frame_start

    rest = _rest_tip(side)
    target.location = rest
    target.keyframe_insert(data_path="location", frame=float(frame_start))
    roles.append((float(frame_start), "rest"))
    tframe(0.0)
    wrist.location = wrist_target(sh, rest)
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

    prev_t = None
    for n in sorted(notes, key=lambda m: m["start"]):
        t = n["start"]
        p = mathutils.Vector((n["x"], n["y"], n["z"]))
        v = max(0.0, min(1.0, n["velocity"] / 127.0))
        gap = (t - prev_t) if prev_t is not None else 1.0
        # Rapid hits get a shorter wind-up so the tip stays near the drum
        # instead of flinging up and darting between poses.
        busy = _clamp(gap / 0.30, 0.4, 1.0)
        lift = (LIFT_MIN + v * (LIFT_MAX - LIFT_MIN)) * busy
        strike = STRIKE_SLOW - v * (STRIKE_SLOW - STRIKE_FAST)
        down = min(strike, 0.40 * gap)    # apex->contact; never precede the last hit
        up = min(0.07, 0.45 * gap)        # contact->rebound

        # Arm anchor: the wrist sits at play height and holds; the hand pivots.
        key_wrist(t, wrist_target(sh, p))
        key_tip(t - down, p + mathutils.Vector((0.0, 0.0, lift)), "apex")  # wrist cocks up
        key_tip(t, p, "contact")                                          # onto the head
        last = key_tip(t + up, p + mathutils.Vector((0.0, 0.0, HOVER)), "rebound")
        prev_t = t

    _apply_stick_easing(target, roles)
    # The wrist empty only ever travels between drums (it never strikes), so
    # glide it smoothly; holds between same-drum hits keep the arm still.
    for fc in _iter_action_fcurves(wrist.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation, kp.easing = 'SINE', 'EASE_IN_OUT'
    return last


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


def _animate_torso(arm, stick_notes, fps, frame_start):
    """Twist and lean the spine toward the centre of what the hands are
    playing. The spine's local Y twists the upper body (left/right), local X
    leans it forward; sampling a smoothed activity centre and SINE-easing the
    keyframes gives natural acceleration and deceleration."""
    if "spine" not in arm.pose.bones:
        return frame_start
    sp = arm.pose.bones["spine"]
    path = 'pose.bones["spine"].rotation_euler'
    frame = _monotonic(fps, frame_start)

    def key(t, lean, twist):
        sp.rotation_euler = (lean, twist, 0.0)
        arm.keyframe_insert(data_path=path, frame=frame(t))

    sh_y = shoulder("L")[1]               # seated shoulder Y (the reach origin)
    key(0.0, 0.0, 0.0)                    # rest: upright
    notes = sorted(stick_notes, key=lambda n: n["start"])
    if not notes:
        return frame_start

    t, t_end = notes[0]["start"], notes[-1]["start"]
    last = frame_start
    while t <= t_end + 1e-6:
        near = [n for n in notes if abs(n["start"] - t) <= TORSO_WIN]
        if near:
            cx = sum(n["x"] for n in near) / len(near)          # centre of activity
            cy = sum(n["y"] for n in near) / len(near)          # how far forward
        else:
            cx, cy = 0.0, sh_y                                  # idle -> upright
        twist = max(-TWIST_MAX, min(TWIST_MAX, -TWIST_GAIN * cx))  # +x (left) -> face left
        # Lean only as far as the reach demands: 0 when the hands are close in
        # (snare/hi-hat), ramping to LEAN_REACH out at the toms/crashes.
        reach = sh_y - cy
        lean = LEAN_REACH * _clamp((reach - REACH_NEAR) / (REACH_FAR - REACH_NEAR), 0.0, 1.0)
        f = frame(t)
        sp.rotation_euler = (lean, twist, 0.0)
        arm.keyframe_insert(data_path=path, frame=f)
        last = f
        t += TORSO_SAMPLE
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
    for side in ("R", "L"):
        if ik[side] is not None and wrist[side] is not None:
            last = max(last, _animate_arm(ik[side], wrist[side], side,
                                          by_hand[side], fps, frame_start))
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
