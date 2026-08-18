"""Builds two minimalist props for the singer scene: a handheld wireless mic
and a free-standing mic stand with an EMPTY clip (no mic in it -- the singer
holds the only mic in the scene).

Run inside Blender. Produces two independent collections:

    Mic         a small parent empty ("Mic") at the grip point, local +Z the
                shaft axis running from the grip toward the windscreen tip,
                with two child meshes (MicHandle, MicWindscreen). Meant to be
                BONE-PARENTED onto the singer's ``hand.R`` (see
                animate_singer.py), which then carries it wherever the wrist
                IK target goes.
    MicStand    a static tripod stand (base legs + telescoping pole + an
                open clip bracket at the top) built directly in world space.
                The clip is left empty on purpose.

Both share the repo's flat, unlit-adjacent minimalist palette (matte dark
metal bodies, no logos, no cable -- it's a wireless handheld).
"""

import math

import bpy
import bmesh
import mathutils

V = mathutils.Vector

# ---------------------------------------------------------------------------
# Mic dimensions (metres)
# ---------------------------------------------------------------------------
MIC_HANDLE_RADIUS = 0.011
MIC_HANDLE_LEN = 0.125
MIC_HEAD_RADIUS = 0.019
MIC_GRIP_FRAC = 0.38     # how far up the handle the grip point sits (0=butt end)

# ---------------------------------------------------------------------------
# Stand dimensions (metres)
# ---------------------------------------------------------------------------
STAND_LEG_LEN = 0.30
STAND_LEG_RADIUS = 0.009
STAND_LEG_SPLAY = math.radians(28.0)
STAND_POLE_RADIUS = 0.011
STAND_POLE2_RADIUS = 0.008
STAND_BASE_HEIGHT = 0.045     # hub height above the floor where legs meet
STAND_CLIP_HEIGHT = 1.40      # where the empty clip sits (mic-in-use height)
STAND_TELESCOPE_JOINT = 0.85  # where the thinner upper pole section begins


def _mat(name, color, rough, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def _cyl_mesh(name, radius, depth, segments=20):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                          radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(mesh); bm.free()
    return mesh


def _sphere_mesh(name, radius, segments=20, rings=14):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=radius)
    bm.to_mesh(mesh); bm.free()
    return mesh


def _link(coll, obj):
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Handheld wireless mic
# ---------------------------------------------------------------------------
def build_mic(name="Mic"):
    """A minimalist handheld wireless mic: a cylindrical handle and a rounded
    windscreen head, no cable, no buttons. Parent empty's origin is the grip
    point; local +Z points from the grip toward the windscreen tip."""
    old = bpy.data.collections.get(name)
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)

    body_mat = _mat("MicBody", (0.05, 0.05, 0.06), 0.35, metallic=0.75)
    head_mat = _mat("MicHead", (0.16, 0.16, 0.17), 0.85, metallic=0.05)

    grip = bpy.data.objects.new(name, None)
    grip.empty_display_type = 'PLAIN_AXES'
    grip.empty_display_size = 0.03
    _link(coll, grip)

    # Handle: centred on its own length; butt end sits below the grip origin
    # by MIC_GRIP_FRAC, tip end above it.
    handle = bpy.data.objects.new("MicHandle",
                                  _cyl_mesh("MicHandleMesh", MIC_HANDLE_RADIUS,
                                           MIC_HANDLE_LEN))
    handle.rotation_euler = (math.pi / 2.0, 0.0, 0.0)   # cylinder axis -> local Z
    handle.location = (0.0, 0.0, MIC_HANDLE_LEN * (0.5 - MIC_GRIP_FRAC))
    handle.data.materials.append(body_mat)
    handle.parent = grip
    _link(coll, handle)

    tip_z = MIC_HANDLE_LEN * (1.0 - MIC_GRIP_FRAC)
    head = bpy.data.objects.new("MicWindscreen",
                                _sphere_mesh("MicWindscreenMesh", MIC_HEAD_RADIUS))
    head.location = (0.0, 0.0, tip_z + MIC_HEAD_RADIUS * 0.55)
    head.data.materials.append(head_mat)
    head.parent = grip
    _link(coll, head)

    grip["tip_local_z"] = tip_z + MIC_HEAD_RADIUS
    print(f"Built {name}: handle {MIC_HANDLE_LEN * 100:.1f}cm, "
          f"tip at local z={grip['tip_local_z']:.3f}")
    return grip


# ---------------------------------------------------------------------------
# Mic stand (no mic in the clip)
# ---------------------------------------------------------------------------
def _leg(coll, mat, splay_angle, facing):
    """One tripod leg: a thin cylinder from the hub, angled out and down."""
    length = STAND_LEG_LEN
    dx = math.sin(splay_angle) * math.cos(facing)
    dy = math.sin(splay_angle) * math.sin(facing)
    dz = -math.cos(splay_angle)
    direction = V((dx, dy, dz))
    top = V((0.0, 0.0, STAND_BASE_HEIGHT))
    bottom = top + direction * length
    mid = (top + bottom) * 0.5

    mesh = _cyl_mesh("StandLegMesh", STAND_LEG_RADIUS, length, segments=10)
    obj = bpy.data.objects.new("StandLeg", mesh)
    obj.data.materials.append(mat)
    obj.location = tuple(mid)
    # Aim the cylinder's local Z (its long axis) along `direction`.
    obj.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    _link(coll, obj)
    return obj


def build_mic_stand(name="MicStand", location=(0.0, -0.60, 0.0)):
    """A minimalist tripod mic stand: three splayed legs, a two-section
    telescoping pole, and an open clip bracket at the top left EMPTY -- no mic
    mesh is placed in it."""
    old = bpy.data.collections.get(name)
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)

    metal = _mat("StandMetal", (0.10, 0.10, 0.11), 0.4, metallic=0.8)
    clip_mat = _mat("StandClip", (0.05, 0.05, 0.06), 0.5, metallic=0.6)

    root = bpy.data.objects.new(name, None)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 0.03
    root.location = location
    _link(coll, root)

    for i in range(3):
        facing = math.radians(90.0) + i * (2.0 * math.pi / 3.0)
        leg = _leg(coll, metal, STAND_LEG_SPLAY, facing)
        leg.parent = root

    # Lower (fixed) pole section: hub up to the telescope joint.
    lower_len = STAND_TELESCOPE_JOINT - STAND_BASE_HEIGHT
    lower = bpy.data.objects.new("StandPoleLower",
                                 _cyl_mesh("StandPoleLowerMesh", STAND_POLE_RADIUS,
                                          lower_len, segments=14))
    lower.location = (0.0, 0.0, STAND_BASE_HEIGHT + lower_len / 2.0)
    lower.data.materials.append(metal)
    lower.parent = root
    _link(coll, lower)

    # Upper (thinner, telescoping) pole section: joint up to the clip.
    upper_len = STAND_CLIP_HEIGHT - STAND_TELESCOPE_JOINT
    upper = bpy.data.objects.new("StandPoleUpper",
                                 _cyl_mesh("StandPoleUpperMesh", STAND_POLE2_RADIUS,
                                          upper_len, segments=12))
    upper.location = (0.0, 0.0, STAND_TELESCOPE_JOINT + upper_len / 2.0)
    upper.data.materials.append(metal)
    upper.parent = root
    _link(coll, upper)

    # Collar at the telescope joint (reads as the height-lock knob).
    collar = bpy.data.objects.new("StandCollar",
                                  _cyl_mesh("StandCollarMesh", STAND_POLE_RADIUS * 1.5,
                                           0.02, segments=14))
    collar.location = (0.0, 0.0, STAND_TELESCOPE_JOINT)
    collar.data.materials.append(clip_mat)
    collar.parent = root
    _link(coll, collar)

    # Open clip bracket at the top: a shallow ring with a gap (like a real
    # mic-clip cradle) so it unambiguously reads as EMPTY -- no mic mesh sits
    # inside it. Built as a partial torus via a thin arc of small cylinders.
    ring_r = MIC_HANDLE_RADIUS + 0.006
    seg_count = 10
    gap = math.radians(50.0)
    span = 2.0 * math.pi - gap
    for i in range(seg_count):
        a0 = math.radians(90.0) + gap / 2.0 + span * (i / seg_count)
        a1 = math.radians(90.0) + gap / 2.0 + span * ((i + 1) / seg_count)
        p0 = V((ring_r * math.cos(a0), ring_r * math.sin(a0), 0.0))
        p1 = V((ring_r * math.cos(a1), ring_r * math.sin(a1), 0.0))
        mid = (p0 + p1) * 0.5
        seg_len = (p1 - p0).length * 1.15
        seg = bpy.data.objects.new("StandClipSeg",
                                   _cyl_mesh("StandClipSegMesh", 0.004, seg_len,
                                            segments=6))
        seg.location = (mid.x, mid.y, STAND_CLIP_HEIGHT)
        tangent = (p1 - p0).normalized()
        seg.rotation_euler = tangent.to_track_quat('Z', 'Y').to_euler()
        seg.data.materials.append(clip_mat)
        seg.parent = root
        _link(coll, seg)

    print(f"Built {name} at {location}: tripod + telescoping pole, "
          f"clip empty at z={STAND_CLIP_HEIGHT}")
    return root


if __name__ == "__main__":
    build_mic()
    build_mic_stand()
