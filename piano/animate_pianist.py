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
def _hand_x_sampler():
    """A function frame -> mean world x of the two hands, read straight off their
    baked location f-curves. Evaluating the curves (rather than stepping the
    scene frame by frame) keeps this free of scene state while the body is being
    keyed."""
    from piano.piano_midi_animator import _iter_action_fcurves
    curves = []
    for side in ("L", "R"):
        hand = bpy.data.objects.get(f"Hand_{side}")
        act = hand and hand.animation_data and hand.animation_data.action
        fc = next((f for f in _iter_action_fcurves(act)
                   if f.data_path == "location" and f.array_index == 0), None)
        if fc is not None:
            curves.append(fc)
        elif hand is not None:
            curves.append(hand.location.x)     # unanimated: a constant

    def sample(frame):
        if not curves:
            return None
        xs = [c.evaluate(frame) if hasattr(c, "evaluate") else c for c in curves]
        return sum(xs) / len(xs)
    return sample


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
    hand_x = _hand_x_sampler()
    centre_x = _load("build_pianist").PLAYER_X

    t, last = 0.0, frame_start
    while t <= duration + 1e-6:
        ph = 2.0 * math.pi * t / BODY_SWAY_PERIOD
        frame = frame_start + t * fps
        if spine is not None:
            # X leans forward (negative = into the keyboard), Y twists about the
            # vertical bone axis.
            lean = -BODY_LEAN * (0.5 - 0.5 * math.cos(2.0 * ph))
            spine.rotation_euler = (lean, BODY_TWIST * math.sin(ph), 0.0)
            arm.keyframe_insert(data_path=sp_path, frame=frame)
        if head is not None:
            x = hand_x(frame)
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
def _check_wrist_pose(arm, frames):
    """Validate each playing wrist against human range of motion, sampled over
    `frames` (so the far ends of the keyboard are covered too). Raises
    RuntimeError -- loud at build time -- if a wrist sits more than
    WRIST_BEND_MAX off the incoming forearm. Returns the worst bend per side, in
    degrees, so a merely-tight posture is still visible in the return value."""
    scene = bpy.context.scene
    saved = scene.frame_current
    worst = {}
    for f in frames:
        scene.frame_set(int(round(f)))
        bpy.context.view_layer.update()
        for side in ("L", "R"):
            fore = arm.pose.bones.get(f"forearm.{side}")
            hand = bpy.data.objects.get(f"Hand_{side}")
            if fore is None or hand is None:
                continue
            e = (arm.matrix_world @ fore.tail
                 - arm.matrix_world @ fore.head).normalized()
            # The hand's finger axis is its object +Y (fingers point up the keys).
            finger = (hand.matrix_world.to_3x3() @ V((0.0, 1.0, 0.0))).normalized()
            bend = e.angle(finger)
            if side not in worst or bend > worst[side]:
                worst[side] = bend
    scene.frame_set(saved)
    for side, bend in worst.items():
        if bend > WRIST_BEND_MAX:
            raise RuntimeError(
                f"Hand_{side} wrist is bent {math.degrees(bend):.0f} deg off the "
                f"forearm (max {math.degrees(WRIST_BEND_MAX):.0f}); the arm is "
                f"meeting the axis-locked hand at the wrong angle -- adjust the "
                f"seat in build_pianist (SEAT_Z sets the shoulder height, HIP_Y "
                f"the distance to the keys, ELBOW_BEND where the elbow hangs).")
    return {side: math.degrees(b) for side, b in worst.items()}


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
                    build=True, camera=True):
    """Build (optionally), press the keys, animate the hands, make the seated
    pianist's arms follow them, and add a body groove."""
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
    bend = _check_wrist_pose(
        arm, range(frame_start, hand_result["frame_end"] + 1,
                   max(1, int(round(fps / 6)))))

    if camera:
        _setup_camera(scene)

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, hand_result["frame_end"])
    scene.frame_set(frame_start)

    return {"pianist": arm.name, "frame_end": scene.frame_end, "fps": fps,
            "keys_animated": keys_result and keys_result["notes_animated"],
            "wrist_bend_deg": {k: round(v, 1) for k, v in bend.items()},
            **hand_result}


if __name__ == "__main__":
    animate_pianist()
