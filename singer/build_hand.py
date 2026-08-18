"""Builds the singer's mic-gripping right hand: a fist wrapped around the
mic's cylindrical handle, styled directly on guitar/build_hands.py's
PickHand -- the guitarist's own fist holding a small handheld object. This
reuses that rig's finger lengths (FRET_FINGERS), cross-sections
(FINGER_CROSS) and curl angles (PICK_FIST) outright; the only thing that
changes is WHERE each finger's knuckle starts -- on the mic handle's ~11mm
circle instead of pinching down to a flat pick -- so the same closed-fist
curl (``_curled_finger``'s cumulative-rotation technique) wraps around a
chunkier cylinder instead.

Local frame, matching PICK_FIST's own: +x is the knuckle spread, which
doubles as the mic's own shaft axis -- the mic passes THROUGH the fist the
way a drumstick or a mic handle actually sits, across the curled fingers,
each of which curls in its own y-z plane (+y out of the knuckle, curling
toward -z, into the palm).

Produces a "MicHand" armature (one "wrist" bone) with rigid boxes for the
palm, four full three-phalanx fingers wrapped around the handle, a
two-segment opposing thumb, and the mic itself (build_mic.build_mic(),
re-mapped from its own Z-shaft frame onto this rig's X-shaft frame) mounted
through the grip. Meant to be positioned once and then COPY_LOCATION'd onto
the singer's Wrist_R, the same way animate_guitarist.py wires the
guitarist's arms to FretHand/PickHand (see animate_singer.py).

Usage (inside Blender / MCP execute_blender_code)::

    import singer.build_hand as bh
    bh.build_mic_hand()
"""

import math

import bpy
import bmesh
import mathutils

V = mathutils.Vector
M = mathutils.Matrix

try:
    from . import build_mic
    from guitar.build_hands import FRET_FINGERS, PICK_FIST, FINGER_CROSS, FINGER_PROFILE
except ImportError:  # loaded as a loose script via importlib
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import build_mic
    from guitar.build_hands import FRET_FINGERS, PICK_FIST, FINGER_CROSS, FINGER_PROFILE

ARMATURE_NAME = "MicHand"

# ---------------------------------------------------------------------------
# Wrap geometry: each finger's knuckle sits on a circle of this radius
# (a hair outside the mic handle's own radius) and curls, via the SAME
# cumulative-rotation technique as guitar/build_hands._curled_finger, using
# PICK_FIST's authored per-joint flex angles (that fist's own curl already
# closes to roughly this scale, tuned for a handheld object between the
# fingers and the palm -- it did not need retuning to wrap a ~22mm-diameter
# handle instead of pinching to a flat pick).
# ---------------------------------------------------------------------------
WRAP_RADIUS = build_mic.MIC_HANDLE_RADIUS + 0.010
KNUCKLE_ANGLE = 68.0     # deg from +y toward +z -- where the fingers start
THUMB_ANGLE = 250.0      # opposite side of the circle
THUMB_X = -0.044         # beyond the index, the thumb's own edge (PICK_THUMB_AXIS side)
THUMB_LENGTHS = (0.030, 0.024)
THUMB_FLEX = (60.0, 70.0)

PALM_SIZE = (0.070, 0.030, 0.052)
# Bone-parent space (see _bone_box) is relative to the wrist bone's TAIL;
# the wrist bone below runs head=(0,-0.025,0) -> tail=(0,0.030,0), so an
# armature-local point converts to bone-parent space by subtracting the
# tail's own armature-local position (0, 0.030, 0).
_BONE_TAIL = V((0.0, 0.030, 0.0))


def _mat():
    m = bpy.data.materials.get("SingerBody") or bpy.data.materials.new("SingerBody")
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.30, 0.33, 0.38, 1.0)
    b.inputs["Roughness"].default_value = 0.6
    return m


def _finger_cross(key, seg):
    return FINGER_CROSS[FINGER_PROFILE[key]][seg]


def _box_mesh(name, sx, sy, sz, bevel=0.0018):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, affect='EDGES')
    bm.to_mesh(mesh); bm.free()
    return mesh


def _bone_box(arm_obj, coll, mat, size, location, rotation=(0.0, 0.0, 0.0), mesh=None):
    if mesh is None:
        mesh = _box_mesh("MicHandSegMesh", *size)
    obj = bpy.data.objects.new("MicHand_part", mesh)
    coll.objects.link(obj)
    obj.parent = arm_obj
    obj.parent_type = 'BONE'
    obj.parent_bone = "wrist"
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(mat)
    return obj


def _seg_box(arm_obj, coll, mat, width, height, p0, p1):
    """A box of the given cross-section spanning armature-local points
    p0->p1, its long (local-y) axis along the segment -- matches
    guitar/build_hands._seg_box exactly."""
    p0, p1 = V(p0), V(p1)
    d = p1 - p0
    center = (p0 + p1) / 2.0
    rot = d.to_track_quat('Y', 'Z').to_euler()
    _bone_box(arm_obj, coll, mat, (width, d.length, height),
             tuple(center - _BONE_TAIL), rotation=rot)


def _curled_wrap(angle_deg, lengths, flex_deg, inward=True):
    """Points knuckle -> ... -> tip for a finger that starts tangent to the
    WRAP_RADIUS circle at `angle_deg` (measured from +y toward +z) and
    curls -- via the same cumulative-rotation-about-x technique as
    guitar/build_hands._curled_finger -- around the handle instead of down
    to a point.

    At a cumulative rotation of 90 deg the swept direction points exactly
    along the knuckle's own radial line; `inward` picks which of the two
    rotation senses makes that 90 deg point INTO the axis (wrapping the
    handle) rather than away from it (the fist opening up)."""
    a = math.radians(angle_deg)
    knuckle = V((0.0, WRAP_RADIUS * math.cos(a), WRAP_RADIUS * math.sin(a)))
    tangent = V((0.0, -math.sin(a), math.cos(a)))
    sign = 1.0 if inward else -1.0
    pts = [knuckle]
    cum = 0.0
    for length, flex in zip(lengths, flex_deg):
        cum += flex
        d = M.Rotation(math.radians(sign * cum), 3, 'X') @ tangent
        pts.append(pts[-1] + d * length)
    return pts


def _build_wrist(arm_obj):
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.pose.bones["wrist"].rotation_mode = 'XYZ'


def _mount_mic(arm_obj, coll):
    """Bring in build_mic.build_mic()'s handle+windscreen and re-map its own
    Z-shaft local frame onto this rig's X-shaft frame (rotate +90 deg about
    Y: local Z -> local X), grip origin at the fist's centre -- the same
    point the fingers wrap around."""
    grip = build_mic.build_mic()
    grip.parent = arm_obj
    grip.parent_type = 'BONE'
    grip.parent_bone = "wrist"
    rot = M.Rotation(math.radians(90.0), 4, 'Y')
    grip.rotation_euler = rot.to_euler()
    grip.location = tuple(V((0.0, 0.0, 0.0)) - _BONE_TAIL)
    # Fold the standalone Mic collection into this one so build_mic_hand()
    # produces a single self-contained assembly.
    for obj in list(bpy.data.collections["Mic"].objects):
        bpy.data.collections["Mic"].objects.unlink(obj)
        coll.objects.link(obj)
    bpy.data.collections.remove(bpy.data.collections["Mic"])
    return grip


def build_mic_hand():
    scene = bpy.context.scene
    old = bpy.data.collections.get(ARMATURE_NAME)
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(ARMATURE_NAME)
    scene.collection.children.link(coll)

    arm_data = bpy.data.armatures.new(ARMATURE_NAME + "Rig")
    arm_obj = bpy.data.objects.new(ARMATURE_NAME, arm_data)
    coll.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    _build_wrist(arm_obj)

    mat = _mat()

    # Back-of-hand mass, sitting behind the curl (away from the palm side).
    palm_center = V((0.0, WRAP_RADIUS + 0.014, 0.006))
    _bone_box(arm_obj, coll, mat, PALM_SIZE, tuple(palm_center - _BONE_TAIL))

    # Four full fingers, spread along x (PICK_FIST's own knuckle-x values,
    # already right-hand chirality), each curling in its own y-z plane
    # around the WRAP_RADIUS circle.
    for name, (knuckle_xyz, lengths, flex) in PICK_FIST.items():
        x = knuckle_xyz[0]
        pts = _curled_wrap(KNUCKLE_ANGLE, lengths, flex, inward=True)
        for seg, (p0, p1) in zip(("prox", "mid", "dist"), zip(pts[:-1], pts[1:])):
            w, h = _finger_cross(name, seg)
            a, b = V((x, p0.y, p0.z)), V((x, p1.y, p1.z))
            _seg_box(arm_obj, coll, mat, w, h, tuple(a), tuple(b))

    # Thumb: opposes the fingers from the far side of the circle, curling
    # the other way so its pad meets the curled fingertips.
    tpts = _curled_wrap(THUMB_ANGLE, THUMB_LENGTHS, THUMB_FLEX, inward=True)
    tp = [V((THUMB_X, p.y, p.z)) for p in tpts]
    _seg_box(arm_obj, coll, mat, 0.018, 0.017, tuple(tp[0]), tuple(tp[1]))
    _seg_box(arm_obj, coll, mat, 0.016, 0.015, tuple(tp[1]), tuple(tp[2]))

    _mount_mic(arm_obj, coll)

    bpy.context.view_layer.update()
    print(f"Built {ARMATURE_NAME}: fist wrapped around a "
          f"{build_mic.MIC_HANDLE_RADIUS * 2000:.0f}mm handle")
    return arm_obj


if __name__ == "__main__":
    build_mic_hand()
