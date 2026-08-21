"""Builds the singer's mic-gripping right hand: a fist wrapped around the
mic's cylindrical handle, styled on guitar/build_hands.py's PickHand -- the
guitarist's own fist holding a small handheld object. It reuses that rig's
finger cross-sections (FINGER_CROSS), the relative prox/mid/dist length
PROPORTIONS of FRET_FINGERS, PICK_FIST's knuckle x-spread (already
right-hand chirality) and the ``_curled_finger`` cumulative-rotation
TECHNIQUE outright.

What it does NOT reuse verbatim is PICK_FIST's own flex angles: those were
tuned for gripping a fat guitar neck, and this needs a much tighter closure
around a slim mic handle. Two earlier versions of this rig tried to force
that closure by sweeping each finger around a circle centred on the mic's
own axis (shrinking the length to stop it overshooting, then widening it
again with a different flex split) -- both numerically closed the gap, but
even the better of the two read as fingers laid ALONGSIDE the handle, not
curled around it: from the back of the hand, the proximal segments sat
right at the top of that circle with the handle exposed underneath, because
a circular sweep never actually reaches OVER the object the way a real
finger's hinge does.

Rebuilt on _curled_finger's own technique instead (verbatim, not adapted):
each finger extends forward from its knuckle, then folds through a
cumulative flexion about its own local x -- the one motion a finger's
joints actually make. At real FRET_FINGERS length with the flex split
tuned below, that fold swings the knuckle out into the bulge a real fist's
own knuckles make, then closes the fingertip back in to within a couple of
mm of the shaft -- so the handle disappears UNDER the curled fingers from
every angle, the way it does in a real gripped fist, rather than just
having its silhouette's radius match at one single point.

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
# Wrap geometry. Each finger's knuckle sits at WRAP_RADIUS -- a hair outside
# the mic handle's own radius -- but unlike a knuckle-line's usual meaning
# elsewhere in this rig, it is now only the STARTING point of a real fold,
# not a circle every joint is forced to stay on: _curled_finger extends the
# finger out from there and lets the cumulative flexion swing it through
# its own natural arc, closing to the shaft only once, at the tip, rather
# than trying to hold every joint near the object's radius throughout.
#
# Tuned numerically (scratch script, not checked in) at FRET_FINGERS' real,
# unscaled length: search FINGER_TOTAL_FLEX and the FINGER_FLEX_FRACS split
# for whichever keeps the fingertip closing to within a couple of mm of the
# shaft, REQUIRING every joint (including the MCP/knuckle itself) to carry
# a reasonable share of the flex. An earlier search minimised tip distance
# alone and found a split that put almost none of the curl at the knuckle
# (17%) -- the proximal phalanx stayed nearly straight, shooting the whole
# finger out to a 34mm bulge before the other two joints did all the work
# snapping it back in over a much shorter distance. Numerically that still
# closed near the shaft, but it reads as a flagpole with a hook on the end,
# not a curled finger -- there is no gentle arc, because there is barely
# any bend where the biggest joint in the chain is. Every joint here
# carries at least 22% of the total instead, which is what turns that hook
# into a rounded C the object actually sits inside of.
# ---------------------------------------------------------------------------
WRAP_RADIUS = build_mic.MIC_HANDLE_RADIUS + 0.008
KNUCKLE_ANGLE = 90.0        # deg from +y toward +z -- back of the hand, top
FINGER_FLEX_FRACS = (0.24, 0.40, 0.36)
FINGER_TOTAL_FLEX = 318.0

# Thumb: starts on the front/near side of the circle (the gap the fingers'
# own ~175 deg sweep leaves open) and folds the same way, closing back
# in toward the shaft to lie over the top of the curled fingers -- an
# over-the-top power grip. An earlier, shorter thumb (32mm total) closed
# just as cleanly numerically, but next to the fingers' own full FRET_
# FINGERS length (82-102mm) it read as a stub, not a shorter digit -- a
# real thumb is shorter than the fingers (~55-65mm against theirs), not a
# fraction that small. Lengths below are close to that real ratio (~0.6 of
# the fingers' own average); flex tuned the same balanced-joint way.
THUMB_ANGLE = 330.0
# +x is toward the windscreen, -x toward the butt (see _mount_mic's 90 deg
# remap of the mic's own +z-toward-tip convention onto this rig's +x-shaft
# one) -- PICK_FIST's own x-spread runs thumb/index negative, pinky
# positive, which put the thumb+index cluster at the BUTT end instead of
# the grille end a real raised-mic grip has them at (the reference photo).
# Negated below (both here and where PICK_FIST's own knuckle x is read) to
# flip which end each finger cluster sits at, without touching any of the
# y-z curl geometry that actually shapes the grip.
THUMB_X = 0.044          # beyond the index, the thumb's own edge (PICK_THUMB_AXIS side)
THUMB_LENGTHS = (0.035, 0.020)
THUMB_FLEX_FRACS = (0.30, 0.70)
THUMB_TOTAL_FLEX = 222.0

# The wrist bone itself (see _build_wrist): head is the "wrist end" that
# meets the forearm, tail is the grip point every finger/palm/mic is built
# around (== the origin of this whole bone-parent space, see _BONE_TAIL).
WRIST_HEAD_LOCAL = (0.0, -0.025, 0.0)
WRIST_TAIL_LOCAL = (0.0, 0.030, 0.0)
WRIST_SPAN = WRIST_TAIL_LOCAL[1] - WRIST_HEAD_LOCAL[1]   # 0.055

# Spans the fingers' x-spread, and in y reaches from the knuckle line (0,
# where PALM_SIZE used to be centered) all the way back to the wrist head
# at -WRIST_SPAN -- otherwise the palm is a thin slab hovering near the
# grip point with nothing bridging it to the wrist joint the forearm's IK
# now (correctly) reaches, which reads as the hand floating apart from the
# arm even though the joint itself is positioned right.
#
# Thickness (z) was 0.052 -- more than double a real palm's ~24mm (see
# piano/build_hands.PALM_SIZE, whose z is 0.024) -- which read as a slab
# next to that hand's slimmer one. Brought down to just enough to still
# reach the knuckle line at z=WRAP_RADIUS from a centre at z=0 (half the
# thickness must be >= WRAP_RADIUS, ~19mm, or the palm stops short of the
# fingers and re-opens the gap the on-axis centring fix closed), with a
# millimetre of margin rather than matched exactly, since this hand -- a
# fist closed around a mic -- is inherently bulkier through the knuckles
# than the pianist's open, flat one.
PALM_SIZE = (0.070, WRIST_SPAN, 2.0 * WRAP_RADIUS + 0.002)
# Bone-parent space (see _bone_box) is relative to the wrist bone's TAIL,
# which is also the point animate_singer.py's mount stamps onto the arm's
# wrist joint (WRIST_R_TARGET) -- so leaving this at the origin (no offset)
# is what makes the wrap circle's centre (the grip point every finger, the
# palm and the mic itself are built around) land EXACTLY on that joint,
# instead of a few cm off it (which read as the hand floating apart from
# the forearm). Kept as a named offset, not a bare V((0,0,0)) literal, so a
# future rebuilt wrist bone with a genuinely offset tail only needs this
# one constant changed.
_BONE_TAIL = V((0.0, 0.0, 0.0))


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


def _curled_finger(knuckle, lengths, flex_deg):
    """Points knuckle -> ... -> tip for a finger extending +y from `knuckle`
    and folding through a CUMULATIVE flexion about local x -- verbatim the
    same technique as guitar/build_hands._curled_finger (a finger's joints
    only ever hinge one way, so each phalanx adds its own flex on top of
    every joint before it, rather than each aiming independently)."""
    pts = [V(knuckle)]
    cum = 0.0
    for length, flex in zip(lengths, flex_deg):
        cum += flex
        d = M.Rotation(math.radians(-cum), 3, 'X') @ V((0.0, 1.0, 0.0))
        pts.append(pts[-1] + d * length)
    return pts


def _curled_thumb(angle_deg, lengths, flex_deg):
    """The same fold as _curled_finger, but for a knuckle that does not sit
    at the top of the WRAP_RADIUS circle (KNUCKLE_ANGLE=90, where +y is
    already the natural "extend forward" direction) -- the thumb starts at
    THUMB_ANGLE instead, so its own "extend" and "fold inward" directions
    are the tangent and inward radial of the circle AT THAT ANGLE, not the
    fixed world axes _curled_finger assumes are already lined up."""
    a = math.radians(angle_deg)
    knuckle = V((0.0, WRAP_RADIUS * math.cos(a), WRAP_RADIUS * math.sin(a)))
    tangent = V((0.0, -math.sin(a), math.cos(a)))
    radial = V((0.0, math.cos(a), math.sin(a)))
    pts = [knuckle]
    cum = 0.0
    for length, flex in zip(lengths, flex_deg):
        cum += flex
        theta = math.radians(cum)
        d = tangent * math.cos(theta) - radial * math.sin(theta)
        pts.append(pts[-1] + d * length)
    return pts


def _build_wrist(arm_obj):
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = WRIST_HEAD_LOCAL
    wrist.tail = WRIST_TAIL_LOCAL
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

    # Centred ON the wrist bone's own axis (x=0, z=0), the same anatomy as
    # the pianist's hand (piano/build_hands.py: PALM_CENTRE sits right on
    # the bone, only ever offset along its length) -- NOT pushed out to
    # z=WRAP_RADIUS to sit "behind the knuckle line" as this used to do.
    # That offset put the whole palm mass to one side of the wrist bone's
    # own centreline instead of straddling it the way the forearm's own
    # box does, which is what read as the hand hinging off the EDGE of the
    # arm rather than growing out of a natural, centred wrist joint. In y,
    # spans from the knuckle line back to the wrist head so it bridges to
    # the forearm (see PALM_SIZE); half its own thickness (0.026) already
    # reaches past the knuckle line at z=WRAP_RADIUS (0.019), so it still
    # meets the fingers with no gap.
    palm_center = V((0.0, -WRIST_SPAN / 2.0, 0.0))
    _bone_box(arm_obj, coll, mat, PALM_SIZE, tuple(palm_center - _BONE_TAIL))

    # Four full fingers, spread along x (PICK_FIST's own knuckle-x values,
    # already right-hand chirality but negated -- see THUMB_X), each
    # folding in its own y-z plane from a knuckle at KNUCKLE_ANGLE on the
    # WRAP_RADIUS circle (see the wrap-geometry tuning note above) at their
    # own real, full length.
    finger_flex = tuple(f * FINGER_TOTAL_FLEX for f in FINGER_FLEX_FRACS)
    a90 = math.radians(KNUCKLE_ANGLE)
    finger_knuckle = (0.0, WRAP_RADIUS * math.cos(a90), WRAP_RADIUS * math.sin(a90))
    for name, (knuckle_xyz, lengths, _pick_flex) in PICK_FIST.items():
        x = -knuckle_xyz[0]
        pts = _curled_finger(finger_knuckle, lengths, finger_flex)
        for seg, (p0, p1) in zip(("prox", "mid", "dist"), zip(pts[:-1], pts[1:])):
            w, h = _finger_cross(name, seg)
            a, b = V((x, p0.y, p0.z)), V((x, p1.y, p1.z))
            _seg_box(arm_obj, coll, mat, w, h, tuple(a), tuple(b))

    # Thumb: folds inward from the front/near side of the circle, closing
    # back over the top of the curled fingers to lie across the shaft.
    thumb_flex = tuple(f * THUMB_TOTAL_FLEX for f in THUMB_FLEX_FRACS)
    tpts = _curled_thumb(THUMB_ANGLE, THUMB_LENGTHS, thumb_flex)
    tp = [V((THUMB_X, p.y, p.z)) for p in tpts]
    w0, h0 = _finger_cross(1, "prox")
    w1, h1 = _finger_cross(1, "mid")
    _seg_box(arm_obj, coll, mat, w0, h0, tuple(tp[0]), tuple(tp[1]))
    _seg_box(arm_obj, coll, mat, w1, h1, tuple(tp[1]), tuple(tp[2]))

    _mount_mic(arm_obj, coll)

    bpy.context.view_layer.update()
    print(f"Built {ARMATURE_NAME}: fist wrapped around a "
          f"{build_mic.MIC_HANDLE_RADIUS * 2000:.0f}mm handle")
    return arm_obj


if __name__ == "__main__":
    build_mic_hand()
