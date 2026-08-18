"""Assembles the full singer shot: a blocky minimalist standing body, the
mouth mounted on its head and lip-synced from a timeline.json, a handheld
wireless mic held up to the mouth by the right hand, and an empty mic stand
in front of the singer.

The singer pipeline already produces, independently:

  * build_singer.py   -> a standing "Singer" humanoid, IK arms + legs,
                          hand.* stubs at each wrist (adapted from
                          guitar/build_guitarist.py)
  * build_mouth.py     -> the flat "Mouth" object with a shape key per
                          viseme per loudness, authored facing -Y at
                          MOUTH_SCALE world size
  * animate_mouth.py   -> keyframes those shape keys from timeline.json
  * build_mic.py        -> a "Mic" prop (handle + windscreen) and a
                          "MicStand" prop (tripod + telescoping pole + an
                          intentionally EMPTY clip bracket)

This module stitches them into one shot:

  1. (Re)build the mouth first (its own build_mouth() is the only builder
     that clears the whole scene), then the singer body.
  2. Mount the mouth onto the singer's head bone: singer/build_singer.py
     stands the figure facing -Y with left = +X, right = -X -- exactly the
     axis convention singer/mouth_shapes.py already draws the mouth in, so
     mounting it is a plain translate + scale, no reorientation.
  3. Raise the right wrist IK target to a point below-and-forward of the
     mouth at the mic's own reach (MIC_TIP_LOCAL_Z + a small gap), so the
     mic -- rigidly bone-parented to hand.R -- lands with its windscreen
     right at the singer's lips, and the blocky hand stub (which follows
     the same bone) lands right where the mic is gripped.
  4. Build the mic stand as a static prop standing in front of the singer,
     its clip left empty (the only mic in the shot is the one in the
     singer's hand).
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
# Right wrist: lifted to hold the mic up to the mouth. Chosen so the
# wrist-to-mouth distance matches the mic's own grip-to-tip reach (see
# _mount_mic), which is what keeps the blocky hand stub (bone-parented,
# follows automatically) sitting right where the mic is gripped instead of
# floating off on its own.
# ---------------------------------------------------------------------------
WRIST_R_TARGET = (-0.02, -0.133, 1.45)
MIC_TIP_GAP = 0.015    # clearance between the windscreen and the lips

STAND_LOCATION = (0.0, -0.65, 0.0)


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
    _load("build_mic").build_mic()
    _load("build_mic").build_mic_stand(location=STAND_LOCATION)


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


def _mount_mic(arm):
    """Lift the right wrist to the mic's reach below the mouth, then
    bone-parent the mic to hand.R and stamp a world transform whose grip
    point sits at the wrist and whose shaft points at the mouth."""
    wrist_target = bpy.data.objects["Wrist_R"]
    wrist_target.location = WRIST_R_TARGET
    bpy.context.view_layer.update()

    hb = arm.pose.bones["hand.R"]
    wrist_world = (arm.matrix_world @ hb.matrix).translation
    mouth_world = V(MOUTH_MOUNT)
    direction = (mouth_world - wrist_world).normalized()

    mic = bpy.data.objects["Mic"]
    tip_local_z = mic["tip_local_z"]
    grip_pos = mouth_world - direction * (tip_local_z + MIC_TIP_GAP)

    mic.parent = arm
    mic.parent_type = 'BONE'
    mic.parent_bone = "hand.R"
    bpy.context.view_layer.update()
    rot = direction.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    mic.matrix_world = M.Translation(grip_pos) @ rot
    bpy.context.view_layer.update()
    return mic


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
    _require("Singer", "Mouth", "Mic", "MicStand")

    arm = bpy.data.objects["Singer"]
    _mount_mouth(arm)
    _mount_mic(arm)

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
