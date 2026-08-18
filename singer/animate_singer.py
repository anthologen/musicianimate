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
# Right wrist: lifted to hold the mic up to the mouth. Chosen so the
# wrist-to-mouth distance matches MicHand's own reach from its wrist bone to
# the mic's tip (see _mount_mic_hand), so the fist lands right at the
# singer's lips.
# ---------------------------------------------------------------------------
WRIST_R_TARGET = (-0.0475, -0.0941, 1.4728)

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
    _load("build_hand").build_mic_hand()
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


def _mount_mic_hand(arm):
    """Stamp MicHand's world pose -- wrist bone at WRIST_R_TARGET, shaft
    (local +x) pointing at the mouth -- then wire the singer's right arm to
    it exactly the way animate_guitarist.py wires the guitarist's arms to
    FretHand/PickHand: the Wrist_R IK target COPY_LOCATIONs onto MicHand's
    own wrist bone, and the blocky Hand_R stub (which the hand rig replaces)
    is hidden."""
    mic_hand = bpy.data.objects["MicHand"]
    wrist_pos = V(WRIST_R_TARGET)
    mouth_world = V(MOUTH_MOUNT)
    direction = (mouth_world - wrist_pos).normalized()
    rot = direction.to_track_quat('X', 'Z').to_matrix().to_4x4()
    # MicHand's wrist bone tail sits at armature-local (0, 0.030, 0) (see
    # build_hand._build_wrist); translate that to the origin before rotating
    # into place so the BONE (not the armature object's own origin) lands at
    # wrist_pos.
    mic_hand.matrix_world = (M.Translation(wrist_pos) @ rot
                             @ M.Translation(V((0.0, -0.030, 0.0))))
    bpy.context.view_layer.update()

    wrist_target = bpy.data.objects["Wrist_R"]
    for con in list(wrist_target.constraints):
        wrist_target.constraints.remove(con)
    copy_loc = wrist_target.constraints.new('COPY_LOCATION')
    copy_loc.target = mic_hand
    copy_loc.subtarget = "wrist"

    stub = bpy.data.objects.get("Hand_R")
    if stub is not None:
        stub.hide_viewport = True
        stub.hide_render = True
    return mic_hand


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
    _require("Singer", "Mouth", "MicHand", "MicStand")

    arm = bpy.data.objects["Singer"]
    _mount_mouth(arm)
    _mount_mic_hand(arm)

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
