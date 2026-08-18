"""Builds the singer's mic-gripping right hand: a fist wrapped around the
mic's cylindrical handle, styled on guitar/build_hands.py's PickHand -- the
guitarist's own fist holding a small handheld object. It reuses that rig's
finger cross-sections (FINGER_CROSS), the relative prox/mid/dist length
PROPORTIONS of FRET_FINGERS, PICK_FIST's knuckle x-spread (already
right-hand chirality) and the ``_curled_finger`` cumulative-rotation
TECHNIQUE outright.

What it does NOT reuse verbatim is PICK_FIST's absolute lengths and flex
angles: those were tuned to curl a real adult finger (~9cm) down to a
PINCH near the palm, not to wrap a ~19mm-radius circle. Used as-is on a
circle this small, the (long) proximal phalanx alone overshoots past the
far side before the finger has flexed a third of the way, so the fingers
read as resting ON TOP of the handle rather than wrapped around it. Solved
numerically instead (see the module docstring's tuning note below):
FINGER_LENGTH_SCALE shrinks FRET_FINGERS's lengths (keeping their relative
proportions) until a smooth, monotonically-tightening curl closes to the
handle's own radius over FINGER_TOTAL_FLEX degrees, split prox/mid/dist by
FINGER_FLEX_FRACS -- the low-poly equivalent of a hand making a properly
closed fist around a slim object instead of a loose one around a fat neck.

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
# Wrap geometry: each finger's knuckle sits on a circle of this radius (a
# hair outside the mic handle's own radius) and curls, via the SAME
# cumulative-rotation technique as guitar/build_hands._curled_finger, closing
# smoothly down to the handle instead of resting on top of it. Tuned
# numerically (scratch script, not checked in): for each finger, shrink
# FRET_FINGERS's (prox, mid, dist) lengths by FINGER_LENGTH_SCALE, split
# FINGER_TOTAL_FLEX degrees of cumulative curl across the three joints by
# FINGER_FLEX_FRACS, and the resulting joint-by-joint distance from the
# shaft axis DECREASES MONOTONICALLY from the WRAP_RADIUS knuckle down to
# 3-9mm -- inside the ~11mm handle radius, i.e. every finger closes snugly
# around it with a hair of interpenetration (reads as a snug grip, not a
# gap) rather than flaring out past the far side and swinging back (what
# PICK_FIST's own, much longer-reaching, flex angles did here).
# ---------------------------------------------------------------------------
WRAP_RADIUS = build_mic.MIC_HANDLE_RADIUS + 0.008
KNUCKLE_ANGLE = 90.0        # deg from +y toward +z -- back of the hand, top
FINGER_LENGTH_SCALE = 0.32
FINGER_FLEX_FRACS = (0.40, 0.35, 0.25)
FINGER_TOTAL_FLEX = 140.0

# Thumb: starts on the front/near side of the circle (the gap the fingers'
# own ~175 deg sweep leaves open) and curls the SAME rotational sense
# (inward) back toward the fingers' own starting knuckle, so its pad closes
# over the top of the curled fingers -- an over-the-top power grip rather
# than trying to span the whole remaining gap to their fingertips.
THUMB_ANGLE = 330.0
THUMB_X = -0.044         # beyond the index, the thumb's own edge (PICK_THUMB_AXIS side)
THUMB_LENGTHS = (0.015, 0.012)
THUMB_FLEX_FRACS = (0.55, 0.45)
THUMB_TOTAL_FLEX = 100.0

# Sits flush behind the knuckle line (same KNUCKLE_ANGLE direction, i.e.
# +z since that angle is 90 deg), spanning the fingers' x-spread.
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


def _curled_wrap(angle_deg, lengths, flex_fracs, total_flex, inward=True):
    """Points knuckle -> ... -> tip for a finger that starts tangent to the
    WRAP_RADIUS circle at `angle_deg` (measured from +y toward +z) and
    curls -- via the same cumulative-rotation-about-x technique as
    guitar/build_hands._curled_finger -- around the handle instead of down
    to a point. `total_flex` degrees of cumulative curl are split across
    the joints by `flex_fracs`.

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
    for length, frac in zip(lengths, flex_fracs):
        cum += frac * total_flex
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

    # Back-of-hand mass: flush behind the knuckle line, same direction as
    # KNUCKLE_ANGLE (== 90 deg == +z), spanning the fingers' x-spread.
    palm_center = V((0.0, 0.0, WRAP_RADIUS + PALM_SIZE[2] / 2.0))
    _bone_box(arm_obj, coll, mat, PALM_SIZE, tuple(palm_center - _BONE_TAIL))

    # Four full fingers, spread along x (PICK_FIST's own knuckle-x values,
    # already right-hand chirality), each curling in its own y-z plane
    # around the WRAP_RADIUS circle -- lengths shrunk (proportions kept) so
    # the curl actually closes to the handle instead of overshooting past
    # it (see FINGER_LENGTH_SCALE above).
    for name, (knuckle_xyz, lengths0, _pick_flex) in PICK_FIST.items():
        x = knuckle_xyz[0]
        lengths = tuple(l * FINGER_LENGTH_SCALE for l in lengths0)
        pts = _curled_wrap(KNUCKLE_ANGLE, lengths, FINGER_FLEX_FRACS,
                           FINGER_TOTAL_FLEX, inward=True)
        for seg, (p0, p1) in zip(("prox", "mid", "dist"), zip(pts[:-1], pts[1:])):
            w, h = _finger_cross(name, seg)
            a, b = V((x, p0.y, p0.z)), V((x, p1.y, p1.z))
            _seg_box(arm_obj, coll, mat, w, h, tuple(a), tuple(b))

    # Thumb: curls inward from the front/near side of the circle, closing
    # back over the top of the curled fingers' own knuckle line.
    tpts = _curled_wrap(THUMB_ANGLE, THUMB_LENGTHS, THUMB_FLEX_FRACS,
                        THUMB_TOTAL_FLEX, inward=True)
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
