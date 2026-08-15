"""Animates a full seated pianist PLAYING the piano -- the piece that joins the
piano pipeline together end to end.

The piano pipeline already produces, independently:

  * build_piano.py    -> a "Piano" collection (88 keys + case, key tops at z = 0.02)
  * piano/fingering.py -> a fingering.json (per-note hand, finger, fingertip target)
  * piano_midi_animator.py -> dips the 88 keys from the same MIDI
  * build_hands.py    -> "Hand_L" / "Hand_R" armatures that play those keys
  * animate_hands.py  -> keyframes those two hand rigs from the fingering.json
                         (the armature object's location carries the wrist along
                         the keyboard; the finger bones are baked FK)
  * build_pianist.py  -> a seated humanoid "Pianist" rig on a stool, with IK arms
                         + legs and a hand.* stub at each wrist

This module stitches them into one shot:

  1. (Re)build the piano, the pianist and the hands.
  2. Press the keys from the MIDI and run animate_hands() on the fingering.json
     derived from it. The two animators share a press-timing model on purpose
     (see animate_hands' docstring), so fingertips and keys dip together.
  3. Make the pianist's arms FOLLOW the hands: each arm's IK wrist target copies
     the matching hand rig's wrist bone, so the two-bone arm IK reaches from the
     shoulder out to wherever the hand has glided. The pianist's own blocky hand
     stubs are hidden so the detailed hand rigs read instead.
  4. Give the body a gentle performance -- a slow torso sway, a head bob, and a
     head that turns to follow the hands along the keyboard -- while the feet
     stay planted (leg IK).
  5. Validate both wrists against human range of motion across the take.

The MIDI is a separate argument from the fingering.json because the JSON records
only the basename it was generated from; pass the two together (they must be the
same piece, or the keys will dip under nothing).

Note how much SIMPLER this is than animate_guitarist.py, and why. A guitar is
worn: its hands are authored in the instrument's own flat frame and the whole
guitar+hands assembly has to be rigidly lifted onto the player's torso. A piano
is furniture -- it does not move, and animate_hands already authors the hands in
world space over the real keys. So there is no holder, no re-parenting and no
strap here; the body simply has to reach the hands where they already are. That
also means there is no dependency cycle to reason about: the hands are wholly
independent of the pianist, and the arm IK only ever reads from them.

Usage (inside Blender / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_pianist", "/path/to/piano/animate_pianist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_pianist("/path/to/piano/fingering.json")
"""

import importlib
import json
import math
import os
import sys

import bpy
import mathutils

V = mathutils.Vector
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # repo root, so `piano` imports as a package
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name):
    """Import a piano/*.py module as a proper ``piano`` package submodule, so its
    package-relative imports resolve (the loose-script fallback branches in those
    files are for standalone use and are not what this pipeline wants)."""
    return importlib.import_module("piano." + name)


# ---------------------------------------------------------------------------
# Body performance
# ---------------------------------------------------------------------------
BODY_SWAY_PERIOD = 4.0     # s per sway cycle -- slower than the guitarist's; a
#                            seated player rocks from the hips, not the knees
BODY_LEAN = math.radians(2.0)   # forward lean amplitude (into the keyboard)
BODY_TWIST = math.radians(2.5)  # torso twist amplitude
HEAD_NOD = math.radians(2.5)    # head bob amplitude
BODY_SAMPLE = 0.2          # s between body keyframes

# The head follows the hands. A pianist watches whichever end of the keyboard is
# busy, so the head yaws toward the midpoint of the two wrists -- but only part
# of the way (the eyes do the rest), and never past a plausible neck turn.
GAZE_FRAC = 0.55
GAZE_MAX = math.radians(20.0)
GAZE_DEPTH = 0.45          # m from the head to the keys; sets deg per m of hand
#                            travel, so the turn scales with the actual reach

# --- Reaching for the far keys ----------------------------------------------
# The seated arm is 0.58 m from shoulder to wrist and the 88-key board is 1.25 m
# wide, so the ends of it are simply OUT OF REACH from an upright spine: at C8
# the right arm came up 40 mm short and the hand rig floated off the end of it.
# A pianist answers that the way anyone does - by leaning in from the hips - so
# the torso here leans forward exactly as far as the needier arm requires, and
# no further. It is solved per keyframe against the hand curves rather than
# tuned as a constant, because how far out the hands go is a property of the
# piece, not of the rig.
REACH_LEAN_MAX = math.radians(22.0)  # spine's own cap is 40; leave room for the
#                                      sway that is layered on top
REACH_MARGIN = 0.98        # fraction of the arm's length treated as reachable,
#                            so it arrives with a hair of bend left, not locked
REACH_STEP = math.radians(1.0)       # resolution of the lean solve

# Axis conventions on this rig, measured on the built bones (build_pianist poses
# the figure facing +Y, so they differ from the standing players'):
#   * spine/head local Y is the bone's own (vertical) axis -> Y = twist/yaw, and
#     a POSITIVE yaw turns the figure toward its LEFT (world -x);
#   * local X leans forward/back, NEGATIVE being forward into the keyboard.

# --- Wrist range-of-motion guard --------------------------------------------
# The playing hands are SEPARATE armatures, and on the piano they are axis-locked
# (animate_hands keys their location only -- fingers always along +y, palm always
# down), so nothing about the HAND can drift out of range. What can, and what
# this guard is really watching, is the ARM: the wrist bend is set entirely by
# where the shoulder sits, i.e. by the bench height and the seat distance in
# build_pianist. Get those wrong and the forearm dives onto the keys with the
# wrist broken 50 deg, which no IK limit catches because the hand is not a joint
# the rig solves.
WRIST_BEND_MAX = math.radians(60.0)   # combined flexion/extension + deviation the
#                                       hand's long axis may sit off the forearm.
#                                       Tighter than the guitar's 90: a piano hand
#                                       is not wrapped around anything, and real
#                                       technique keeps the wrist near level.


# ---------------------------------------------------------------------------
# Build / reset
# ---------------------------------------------------------------------------
def _ensure_built():
    """(Re)build the piano, the pianist and the hands in an order that is safe
    against build_piano.clear_scene (which wipes every mesh object): piano first,
    then the pianist, then the hand rigs."""
    _load("build_piano").build_piano()
    _load("build_pianist").build_pianist()
    _load("build_hands").build_hands()


def _require(*names):
    missing = [n for n in names if bpy.data.objects.get(n) is None]
    if missing:
        raise RuntimeError("missing rig object(s): " + ", ".join(missing)
                           + " -- run with build=True or build them first")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def _wire_arms():
    """Point each pianist arm's IK wrist target at the matching hand rig so the
    arm reaches out to wherever that hand plays. The pianist faces +Y, so the
    LEFT arm (-x) follows Hand_L and the right follows Hand_R. Hide the blocky
    pianist hand stubs so the detailed hand rigs show instead."""
    for side in ("L", "R"):
        target = bpy.data.objects.get(f"Wrist_{side}")
        hand = bpy.data.objects.get(f"Hand_{side}")
        if target is None or hand is None:
            continue
        for c in list(target.constraints):
            target.constraints.remove(c)
        con = target.constraints.new('COPY_LOCATION')
        con.target = hand
        con.subtarget = "wrist"          # the hand rig's wrist bone (the joint)
        # Hide the pianist's placeholder fist -- the hand rig replaces it.
        stub = bpy.data.objects.get(f"Fist_{side}")
        if stub is not None:
            stub.hide_viewport = True
            stub.hide_render = True


# ---------------------------------------------------------------------------
# Body performance
# ---------------------------------------------------------------------------
def _wrist_sampler():
    """A function frame -> {"L": world wrist, "R": world wrist} for the two hand
    rigs, read straight off their baked location f-curves. Evaluating the curves
    (rather than stepping the scene frame by frame) keeps this free of scene
    state while the body is being keyed.

    animate_hands keys each hand armature's LOCATION and its YAW about Z (the
    hand turns out toward the arm reaching it), so the joint the arm has to
    reach is that location plus the wrist bone's rest head, turned by the
    yaw."""
    from piano.piano_midi_animator import _iter_action_fcurves
    rigs = {}
    for side in ("L", "R"):
        hand = bpy.data.objects.get(f"Hand_{side}")
        if hand is None:
            continue
        act = hand.animation_data and hand.animation_data.action
        curves = {}
        for fc in (_iter_action_fcurves(act) if act else []):
            if fc.data_path in ("location", "rotation_euler"):
                curves[(fc.data_path, fc.array_index)] = fc
        rigs[side] = (hand, curves, V(hand.data.bones["wrist"].head_local))

    def sample(frame):
        out = {}
        for side, (hand, curves, head) in rigs.items():
            def value(path, i, default):
                fc = curves.get((path, i))
                return fc.evaluate(frame) if fc is not None else default
            loc = V(tuple(value("location", i, hand.location[i])
                          for i in range(3)))
            yaw = value("rotation_euler", 2, hand.rotation_euler[2])
            out[side] = loc + (mathutils.Matrix.Rotation(yaw, 3, 'Z') @ head)
        return out
    return sample


def _arm_metrics(arm):
    """(rest shoulder positions, arm length, hip-to-shoulder lever) in world
    space, measured off the built rig rather than assumed."""
    bones = arm.data.bones
    shoulders, arm_len = {}, 0.0
    for side in ("L", "R"):
        ua, fore = bones[f"upper_arm.{side}"], bones[f"forearm.{side}"]
        shoulders[side] = arm.matrix_world @ V(ua.head_local)
        arm_len = max(arm_len, ua.length + fore.length)
    pivot = arm.matrix_world @ V(bones["spine"].head_local)
    lever = shoulders["L"].z - pivot.z
    return shoulders, arm_len, lever


def _reach_lean(wrists, shoulders, arm_len, lever):
    """The smallest forward lean (radians, >= 0) that puts BOTH wrists inside
    the arms' reach, capped at REACH_LEAN_MAX.

    Leaning by theta about the hips swings each shoulder forward by
    lever*sin(theta) and drops it by lever*(1 - cos(theta)); the drop matters
    here, because the keys are already well below the shoulder, so the lean is
    stepped rather than solved in closed form."""
    if not wrists or lever <= 0.0:
        return 0.0
    theta = 0.0
    while theta <= REACH_LEAN_MAX + 1e-9:
        s, c = math.sin(theta), math.cos(theta)
        if all((w - (shoulders[side] + V((0.0, lever * s, -lever * (1.0 - c)))))
               .length <= arm_len * REACH_MARGIN
               for side, w in wrists.items() if side in shoulders):
            return theta
        theta += REACH_STEP
    return REACH_LEAN_MAX


def _animate_body(arm, duration, fps, frame_start):
    """Slow torso sway, head bob, and a head that turns to follow the hands along
    the keyboard. The feet stay planted (leg IK to static Ankle empties) and the
    arms re-solve to keep the hands on the keys, so the body can groove without
    pulling the shoulders off the reach."""
    arm.animation_data_clear()          # re-runnable: never layer on a stale take
    if duration <= 0.0:
        return frame_start
    spine, head = arm.pose.bones.get("spine"), arm.pose.bones.get("head")
    sp_path = 'pose.bones["spine"].rotation_euler'
    hd_path = 'pose.bones["head"].rotation_euler'
    wrists = _wrist_sampler()
    shoulders, arm_len, lever = _arm_metrics(arm)
    centre_x = _load("build_pianist").PLAYER_X

    t, last = 0.0, frame_start
    while t <= duration + 1e-6:
        ph = 2.0 * math.pi * t / BODY_SWAY_PERIOD
        frame = frame_start + t * fps
        at = wrists(frame)
        if spine is not None:
            # X leans forward (negative = into the keyboard), Y twists about the
            # vertical bone axis. The sway rides on top of however far the torso
            # has had to lean in to put the hands within reach.
            lean = -BODY_LEAN * (0.5 - 0.5 * math.cos(2.0 * ph))
            lean -= _reach_lean(at, shoulders, arm_len, lever)
            spine.rotation_euler = (lean, BODY_TWIST * math.sin(ph), 0.0)
            arm.keyframe_insert(data_path=sp_path, frame=frame)
        if head is not None:
            x = (sum(w.x for w in at.values()) / len(at)) if at else None
            # Positive Y yaws toward the player's LEFT (-x), so the gaze term is
            # negated to turn the head the way the hands went.
            gaze = 0.0
            if x is not None:
                gaze = -max(-GAZE_MAX, min(GAZE_MAX, GAZE_FRAC * math.atan2(
                    x - centre_x, GAZE_DEPTH)))
            head.rotation_euler = (HEAD_NOD * math.sin(2.0 * ph),
                                   gaze - 0.4 * BODY_TWIST * math.sin(ph), 0.0)
            arm.keyframe_insert(data_path=hd_path, frame=frame)
        last = frame
        t += BODY_SAMPLE

    act = arm.animation_data.action if arm.animation_data else None
    if act is not None:
        from piano.piano_midi_animator import _iter_action_fcurves
        for fc in _iter_action_fcurves(act):
            if fc.data_path in (sp_path, hd_path):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'SINE'
    return last


# ---------------------------------------------------------------------------
# Range-of-motion check
# ---------------------------------------------------------------------------
def _check_wrist_pose(arm, frames, strict=True):
    """Validate each playing wrist against human range of motion, sampled over
    `frames` (so the far ends of the keyboard are covered too). Raises
    RuntimeError -- loud at build time -- if a wrist sits more than
    WRIST_BEND_MAX off the incoming forearm, unless `strict` is False, in which
    case the offence is only reported.

    Returns per side the worst bend (degrees) and the worst gap (mm) between the
    arm's own wrist and the hand rig it is supposed to be holding -- a non-zero
    gap means the arm ran out of length and the hand is floating off the end of
    it, which is a different failure from a bent wrist and is not caught by any
    joint limit either."""
    scene = bpy.context.scene
    saved = scene.frame_current
    bend_worst, gap_worst = {}, {}
    for f in frames:
        scene.frame_set(int(round(f)))
        bpy.context.view_layer.update()
        for side in ("L", "R"):
            fore = arm.pose.bones.get(f"forearm.{side}")
            hand = bpy.data.objects.get(f"Hand_{side}")
            if fore is None or hand is None:
                continue
            tail = arm.matrix_world @ fore.tail
            e = (tail - arm.matrix_world @ fore.head).normalized()
            # The hand's finger axis is its object +Y (fingers point up the keys).
            finger = (hand.matrix_world.to_3x3() @ V((0.0, 1.0, 0.0))).normalized()
            bend = e.angle(finger)
            if side not in bend_worst or bend > bend_worst[side]:
                bend_worst[side] = bend
            gap = (tail - hand.matrix_world @ hand.pose.bones["wrist"].head).length
            if side not in gap_worst or gap > gap_worst[side]:
                gap_worst[side] = gap
    scene.frame_set(saved)
    over = {side: b for side, b in bend_worst.items() if b > WRIST_BEND_MAX}
    if over and strict:
        side, bend = max(over.items(), key=lambda kv: kv[1])
        raise RuntimeError(
            f"Hand_{side} wrist is bent {math.degrees(bend):.0f} deg off the "
            f"forearm (max {math.degrees(WRIST_BEND_MAX):.0f}); the arm is "
            f"meeting the axis-locked hand at the wrong angle -- adjust the "
            f"seat in build_pianist (SEAT_Z sets the shoulder height, HIP_Y "
            f"the distance to the keys, ELBOW_BEND where the elbow hangs). Pass "
            f"strict=False to render the take anyway and read the numbers back.")
    for side, bend in over.items():
        print(f"WARNING: Hand_{side} wrist bends {math.degrees(bend):.0f} deg off "
              f"the forearm (max {math.degrees(WRIST_BEND_MAX):.0f})")
    return ({side: round(math.degrees(b), 1) for side, b in bend_worst.items()},
            {side: round(g * 1000.0, 1) for side, g in gap_worst.items()})


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def _setup_camera(scene):
    """Frame the seated pianist from a front-3/4 on the treble side -- close
    enough to read the hands on the keys, wide enough to hold the stool. Kept on
    the player's side of the piano (-y), which is where build_piano puts both
    lights."""
    cam = bpy.data.objects.get("Camera")
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(cam)
    cam.location = (1.70, -1.65, 1.55)
    look = V((-0.10, -0.10, -0.02))
    cam.rotation_euler = (look - V(cam.location)).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 38
    scene.camera = cam


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def animate_pianist(fingering_json=None, midi_path=None, fps=24, frame_start=1,
                    build=True, camera=True, strict=True):
    """Build (optionally), press the keys, animate the hands, make the seated
    pianist's arms follow them, and add a body groove.

    `strict` (default True) makes a wrist outside human range of motion a hard
    error. Set it False for a deliberate stress piece -- one that plays the top
    and bottom of the board at once, say -- where the take is still wanted, and
    read `wrist_bend_deg` in the result to see what it cost."""
    if fingering_json is None:
        fingering_json = os.path.join(_HERE, "fingering.json")
    if midi_path is None:
        midi_path = os.path.join(_HERE, "piano_demo.mid")

    scene = bpy.context.scene
    if build:
        _ensure_built()
    _require("Pianist", "Hand_L", "Hand_R")

    arm = bpy.data.objects["Pianist"]

    # 1. Dip the keys, then author the hand performance over them.
    keys_result = None
    if midi_path and os.path.exists(midi_path):
        keys_result = _load("piano_midi_animator").animate_piano(
            midi_path, fps=fps, frame_start=frame_start)
    hand_result = _load("animate_hands").animate_hands(
        fingering_json, fps=fps, frame_start=frame_start)

    # 2. Point the arm IK at the hands, 3. groove the body for the take.
    _wire_arms()
    with open(fingering_json) as fh:
        notes = json.load(fh)["notes"]
    duration = max((n["end"] for n in notes), default=0.0)
    _animate_body(arm, duration, fps, frame_start)

    # 4. Both wrists must be humanly posed on the finished rig -- the hands are
    #    axis-locked, so only the arm can get this wrong, and only here.
    bend, gap = _check_wrist_pose(
        arm, range(frame_start, hand_result["frame_end"] + 1,
                   max(1, int(round(fps / 6)))), strict=strict)

    if camera:
        _setup_camera(scene)

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, hand_result["frame_end"])
    scene.frame_set(frame_start)

    return {"pianist": arm.name, "frame_end": scene.frame_end, "fps": fps,
            "keys_animated": keys_result and keys_result["notes_animated"],
            "wrist_bend_deg": bend, "arm_reach_gap_mm": gap,
            **hand_result}


if __name__ == "__main__":
    animate_pianist()
