"""Assembles the full singer shot: a blocky minimalist standing body, the
mouth mounted on its head and lip-synced from a timeline.json, a fist
wrapped around a handheld wireless mic held up to the mouth by the right
hand, and an empty mic stand in front of the singer.

The singer pipeline already produces, independently:

  * build_singer.py   -> a standing "Singer" humanoid (adapted from
                          guitar/build_guitarist.py), IK arms + legs,
                          hand.* stubs at each wrist
  * build_mouth.py     -> the flat "Mouth" object with a shape key per
                          viseme per loudness, authored facing -Y at
                          MOUTH_SCALE world size
  * animate_mouth.py   -> keyframes those shape keys from timeline.json
  * build_mic.py        -> a "Mic" prop (handle + windscreen) and a
                          "MicStand" prop (tripod + telescoping pole + an
                          intentionally EMPTY clip bracket)
  * build_hand.py       -> a "MicHand" armature: a fist (adapted from
                          guitar/build_hands.py's PickHand) with fingers
                          curled around the mic's handle

This module stitches them into one shot:

  1. (Re)build the mouth first (its own build_mouth() is the only builder
     that clears the whole scene), then the singer body, then the
     mic-gripping hand and the empty stand.
  2. Mount the mouth onto the singer's head bone: singer/build_singer.py
     stands the figure facing -Y with left = +X, right = -X -- exactly the
     axis convention singer/mouth_shapes.py already draws the mouth in, so
     mounting it is a plain translate + scale, no reorientation.
  3. Stamp the MicHand assembly's world pose so its wrist bone lands below
     the mouth at the mic's own reach and its shaft (local +x) points at
     the mouth, then -- the SAME pattern guitar/animate_guitarist.py uses
     to wire the guitarist's arms to FretHand/PickHand -- give the singer's
     Wrist_R IK target a COPY_LOCATION constraint onto MicHand's own wrist
     bone, so the arm's two-bone IK reaches out to wherever the hand rig
     is, and hide the singer's blocky Hand_R stub (the detailed hand rig
     replaces it).
  4. Build the mic stand as a static prop standing in front of the singer,
     its clip left empty (the only mic in the shot is the one gripped in
     the singer's hand).
  5. Animate the mouth's shape keys from timeline.json.
  6. Camera + lighting, then save.

Usage (inside Blender / MCP execute_blender_code)::

    import singer.animate_singer as anim
    anim.animate_singer()
"""

import importlib
import math
import os
import sys

import bpy
import mathutils

V = mathutils.Vector
M = mathutils.Matrix
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name):
    return importlib.import_module("singer." + name)


# ---------------------------------------------------------------------------
# Mouth mount: a plain translate + scale onto the front of the head box (see
# build_singer.ANTHRO["head"], centred at (0, -0.01, 1.635)). Placed in the
# lower half of the face, a hair proud of the head's front (-Y) surface.
# ---------------------------------------------------------------------------
MOUTH_SCALE = 0.055
MOUTH_MOUNT = (0.0, -0.113, 1.56)

# ---------------------------------------------------------------------------
# Right wrist / mic shaft: rather than aiming the mic from wherever the
# wrist happens to be, the mic's own shaft direction is fixed FIRST -- along
# the line from the empty stand's clip height up to the mouth, so the
# windscreen reads as pointed at the mouth and the BUTT of the mic reads as
# pointed back down at the stand it presumably came from -- and the wrist
# position is then derived from that direction (mouth, minus one mic's
# reach along it). STAND_CLIP_Z mirrors build_mic.STAND_CLIP_HEIGHT (the
# height an actually-clipped mic would sit at); kept as a separate literal
# so this module doesn't need to import build_mic just for one constant.
# ---------------------------------------------------------------------------
STAND_LOCATION = (0.0, -0.65, 0.0)
STAND_CLIP_Z = 1.40
MIC_TIP_GAP = 0.015    # clearance between the windscreen and the lips

# build_singer's default elbow pole (tuned for the resting, hands-at-sides
# pose) points behind the body; with the wrist raised this far across toward
# the mouth it lets the two-bone IK swing the elbow across the midline, deep
# into the chest box. Re-aimed out to the singer's own right side (further
# -x, away from the centreline) and a little forward/down so the elbow reads
# as bent out to the side -- the natural way to hold a mic up to the mouth --
# instead of tucked in front of the torso.
ELBOW_R_POLE = (-0.65, -0.15, 0.85)


# ---------------------------------------------------------------------------
# Build / reset
# ---------------------------------------------------------------------------
def _ensure_built():
    """Build order matters: build_mouth's own clear_scene() is the only step
    that wipes the whole scene, so it must run first."""
    bm = _load("build_mouth")
    bm.MOUTH_SCALE = MOUTH_SCALE
    bm.build_mouth(face=False, camera=False, clear=True)
    _load("build_singer").build_singer()
    _load("build_hand").build_mic_hand()
    _load("build_mic").build_mic_stand(location=STAND_LOCATION)
    # The singer's idle left hand: reuses the pianist's own hand rig
    # unchanged (piano/build_hands.py) rather than a bespoke build, since
    # its anatomy -- a wrist bone with the palm centred ON its axis, only
    # offset along its length -- is exactly what _mount_relaxed_hand wants
    # to mount naturally onto an arm with nothing to hold. build_singer()
    # already claimed the name "Hand_L" for its own blocky stub, so this
    # one is renamed to avoid Blender silently suffixing it "Hand_L.001".
    import piano.build_hands as piano_hands
    left_hand = piano_hands.build_hand("L", bpy.data.collections["Singer"],
                                       piano_hands._skin_material())
    left_hand.name = "Hand_L_Detail"


def _require(*names):
    missing = [n for n in names if bpy.data.objects.get(n) is None]
    if missing:
        raise RuntimeError("missing rig object(s): " + ", ".join(missing)
                           + " -- run with build=True or build them first")


# ---------------------------------------------------------------------------
# Mounting
# ---------------------------------------------------------------------------
def _mount_mouth(arm):
    """Bone-parent the flat Mouth card onto the head and stamp its world
    position. Shape keys are unaffected by the object's transform, so this
    can happen before or after animate_mouth keys them."""
    mouth = bpy.data.objects["Mouth"]
    mouth.parent = arm
    mouth.parent_type = 'BONE'
    mouth.parent_bone = "head"
    bpy.context.view_layer.update()
    mouth.matrix_world = M.Translation(V(MOUTH_MOUNT))
    # Freshly built shape keys default to value=1.0 (harmless once
    # animate_mouth keys them, since it zeroes everything first anyway, but
    # left alone the un-animated mesh would show every viseme summed at
    # once) -- zero them so a static preview reads as a closed mouth too.
    for block in mouth.data.shape_keys.key_blocks:
        block.value = 0.0
    return mouth


def _mount_mic_hand(arm):
    """Stamp MicHand's world pose -- shaft (local +x) along the fixed
    stand-clip-to-mouth line, wrist bone (== the mic's own grip point, see
    build_hand._BONE_TAIL) one mic's reach back down that line from the
    mouth -- then wire the singer's right arm to it exactly the way
    animate_guitarist.py wires the guitarist's arms to FretHand/PickHand:
    the Wrist_R IK target COPY_LOCATIONs onto MicHand's own wrist bone, and
    the blocky Hand_R stub (which the hand rig replaces) is hidden.

    The mic's shaft direction alone only pins ONE axis of the hand's
    rotation (local +x); the other two were left for to_track_quat to pick
    (nearest to world +z), with no regard for which way the singer's actual
    forearm arrives -- so the hand's own "wrist end" (local -y, opposite the
    fingers, see build_hand._build_wrist) could end up pointing off at an
    angle unrelated to the forearm, reading as attached to it at the right
    SPOT but the wrong ANGLE. Fixed with a two-pass solve: mount once with a
    placeholder rotation so the position-only arm IK settles (elbow depends
    on the wrist's target POSITION, never its rotation, so this doesn't need
    to repeat), read the real elbow->wrist direction off the now-posed arm,
    and re-mount with local -y aimed at the elbow (orthogonalised against
    the fixed shaft axis) -- same position, corrected rotation."""
    mic_hand = bpy.data.objects["MicHand"]
    mouth_world = V(MOUTH_MOUNT)
    stand_aim = V((STAND_LOCATION[0], STAND_LOCATION[1], STAND_CLIP_Z))
    shaft_dir = (mouth_world - stand_aim).normalized()

    tip_local_z = bpy.data.objects["Mic"]["tip_local_z"]
    wrist_pos = mouth_world - shaft_dir * (tip_local_z + MIC_TIP_GAP)
    # MicHand's wrist bone TAIL is the grip point (build_hand._BONE_TAIL is
    # the zero offset the fingers/palm/mic are all built around) -- but a
    # bone's own local origin is its HEAD, 0.055 away at (0, -0.025, 0), so
    # stamping the object's raw origin at wrist_pos would leave the actual
    # grip that far off. Pre-translate by -tail so the TAIL lands at
    # wrist_pos instead. Changing the ROTATION below never moves this: the
    # pre-translation cancels the tail offset before the rotation is
    # applied, so the tail always lands exactly at wrist_pos regardless.
    tail_local = V((0.0, 0.030, 0.0))

    def _stamp(rot):
        mic_hand.matrix_world = M.Translation(wrist_pos) @ rot @ M.Translation(-tail_local)

    _stamp(shaft_dir.to_track_quat('X', 'Z').to_matrix().to_4x4())
    bpy.context.view_layer.update()

    wrist_target = bpy.data.objects["Wrist_R"]
    for con in list(wrist_target.constraints):
        wrist_target.constraints.remove(con)
    copy_loc = wrist_target.constraints.new('COPY_LOCATION')
    copy_loc.target = mic_hand
    copy_loc.subtarget = "wrist"
    # Sample the TAIL first (head_tail=1.0): unlike the HEAD, the tail's
    # world position is rotation-invariant (see _stamp -- the pre-
    # translation by -tail_local cancels out whatever `rot` is), always
    # exactly wrist_pos. That makes it a stable target to settle the arm's
    # IK against while the hand still has the first pass's placeholder
    # rotation. Sampling the HEAD here instead would be circular: the
    # HEAD's world position depends on the hand's rotation, which itself
    # isn't final until after this settle, based on the resulting elbow.
    copy_loc.head_tail = 1.0

    # Re-aim the elbow pole so the arm bends out to the side rather than
    # swinging across the midline into the chest (see ELBOW_R_POLE).
    bpy.data.objects["Elbow_R"].location = ELBOW_R_POLE
    bpy.context.view_layer.update()

    # Second pass: the arm has now solved against wrist_pos, so read the
    # real elbow -> wrist direction and re-aim the hand's -y (its wrist end)
    # to continue it, keeping +x pinned to the mic's shaft direction.
    elbow_world = (arm.matrix_world @ arm.pose.bones["forearm.R"].matrix).translation
    forearm_dir = (wrist_pos - elbow_world).normalized()
    y_axis = (forearm_dir - forearm_dir.dot(shaft_dir) * shaft_dir).normalized()
    z_axis = shaft_dir.cross(y_axis)
    _stamp(M((shaft_dir, y_axis, z_axis)).transposed().to_4x4())
    bpy.context.view_layer.update()

    # Now that the hand's rotation is final, the wrist bone's HEAD -- the
    # "wrist end" opposite the fingers, see build_hand._build_wrist -- has
    # a fixed world position 5.5cm back up the shaft from the grip. Point
    # the constraint there instead of the grip-point TAIL so the forearm's
    # own IK reaches the actual wrist joint, not the middle of the fist.
    copy_loc.head_tail = 0.0
    bpy.context.view_layer.update()

    stub = bpy.data.objects.get("Hand_R")
    if stub is not None:
        stub.hide_viewport = True
        stub.hide_render = True
    return mic_hand


# On piano/build_hands.py's own rig (rest pose, no roll on any bone -- see
# its FINGERS docstring), a finger's local Y is its length and local Z is
# the axis flexion curls it away from: a NEGATIVE local-x pose rotation
# moves the tip toward -z (confirmed empirically -- 30 deg gave a tail
# delta of (0, -0.006, -0.022), i.e. almost entirely -z), so -z is the
# PALMAR direction (what the palm faces) and +z is dorsal (back of hand).
PALM_CURL_DEG = {"prox": -8.0, "mid": -12.0, "dist": -8.0}


def _mount_relaxed_hand(arm):
    """Mount the idle left hand (piano/build_hands.py's own rig, see
    _ensure_built) on the singer's left wrist exactly the way
    animate_pianist._wire_arms mounts a hand: COPY_LOCATION straight onto
    the hand's wrist bone at its default HEAD sample -- no head_tail
    override, because this hand's palm is centred on that bone's own axis
    (the fix in build_hand.py's own palm_center), so the head IS the joint
    with nothing else to reach past.

    With nothing to hold, there is no mic-shaft axis to pin the hand's
    rotation to the way the mic hand's shaft does -- but the ROLL about the
    forearm (which way the palm faces) still needs pinning to something, or
    to_track_quat is free to pick whatever roll keeps some arbitrary axis
    closest to world +z, unrelated to a natural hanging-arm pose. Pin it to
    the palm (-z, see above) facing medially -- world -x for this LEFT arm,
    which sits on the +x side of the body (build_singer.shoulder) -- so the
    hand rests the way an arm actually hangs at your side, palm toward the
    leg, not turned out to whatever angle the solver happened to land on.
    Finally, curl every finger (see PALM_CURL_DEG) a few degrees so the
    idle hand reads as relaxed rather than a stiff, flat plane."""
    hand = bpy.data.objects["Hand_L_Detail"]
    wrist_target = bpy.data.objects["Wrist_L"]
    target_pos = V(wrist_target.location)

    elbow_world = (arm.matrix_world @ arm.pose.bones["forearm.L"].matrix).translation
    forearm_dir = (target_pos - elbow_world).normalized()

    medial = V((-1.0, 0.0, 0.0))
    dorsal_dir = (medial - medial.dot(forearm_dir) * forearm_dir).normalized() * -1.0
    x_axis = forearm_dir.cross(dorsal_dir)
    rot = M((x_axis, forearm_dir, dorsal_dir)).transposed().to_4x4()

    head_local = V(hand.data.bones["wrist"].head_local)
    hand.matrix_world = M.Translation(target_pos) @ rot @ M.Translation(-head_local)
    bpy.context.view_layer.update()

    for f in range(1, 6):
        for seg, deg in PALM_CURL_DEG.items():
            pb = hand.pose.bones[f"f{f}_{seg}"]
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = (math.radians(deg), 0.0, 0.0)
    bpy.context.view_layer.update()

    for con in list(wrist_target.constraints):
        wrist_target.constraints.remove(con)
    copy_loc = wrist_target.constraints.new('COPY_LOCATION')
    copy_loc.target = hand
    copy_loc.subtarget = "wrist"
    bpy.context.view_layer.update()

    stub = bpy.data.objects.get("Hand_L")
    if stub is not None:
        stub.hide_viewport = True
        stub.hide_render = True
    return hand


# ---------------------------------------------------------------------------
# Camera + lighting
# ---------------------------------------------------------------------------
def _setup_camera_and_lighting(scene):
    cam = bpy.data.objects.get("Camera")
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(cam)
    cam.data.type = 'PERSP'
    cam.data.lens = 40
    cam.location = (0.75, -2.35, 1.20)
    look = V((0.0, -0.25, 1.15))
    cam.rotation_euler = (look - V(cam.location)).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam

    key = bpy.data.objects.get("Light")
    if key is None or key.type != 'LIGHT':
        key = bpy.data.objects.new("Light", bpy.data.lights.new("Light", type='AREA'))
        scene.collection.objects.link(key)
    key.data.type = 'AREA'
    key.data.energy = 350
    key.data.size = 2.2
    key.location = (1.3, -1.9, 2.3)
    key.rotation_euler = (math.radians(55), 0.0, math.radians(25))

    fill_name = "FillLight"
    fill = bpy.data.objects.get(fill_name)
    if fill is None:
        fill = bpy.data.objects.new(fill_name, bpy.data.lights.new(fill_name, type='AREA'))
        scene.collection.objects.link(fill)
    fill.data.type = 'AREA'
    fill.data.energy = 120
    fill.data.size = 2.5
    fill.location = (-1.4, -1.6, 1.2)
    fill.rotation_euler = (math.radians(75), 0.0, math.radians(-30))

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs[0].default_value = (0.16, 0.17, 0.20, 1.0)
        background.inputs[1].default_value = 1.0

    scene.view_settings.view_transform = 'Standard'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 960


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def animate_singer(timeline_json=None, fps=24, frame_start=1, build=True,
                   camera=True, save_path=None):
    if timeline_json is None:
        timeline_json = os.path.join(_HERE, "timeline.json")

    scene = bpy.context.scene
    if build:
        _ensure_built()
    _require("Singer", "Mouth", "MicHand", "MicStand", "Hand_L_Detail")

    arm = bpy.data.objects["Singer"]
    _mount_mouth(arm)
    _mount_mic_hand(arm)
    _mount_relaxed_hand(arm)

    animate_mouth = _load("animate_mouth").animate_mouth
    frame_start, frame_end = animate_mouth(timeline_json, fps=fps,
                                           frame_start=frame_start)

    if camera:
        _setup_camera_and_lighting(scene)

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, frame_end)
    scene.frame_set(frame_start)

    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)

    return {"singer": arm.name, "frame_start": frame_start,
           "frame_end": scene.frame_end, "fps": fps}


if __name__ == "__main__":
    animate_singer()
