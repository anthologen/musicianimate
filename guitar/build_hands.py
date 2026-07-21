"""Builds the guitar fretting- and picking-hand rigs used by animate_hands.py.

Run inside Blender (after build_guitar.py) to create a "GuitarHands"
collection containing two armatures:

  - ``FretHand``: wrist root bone plus four finger chains f1 (index) ..
    f4 (pinky), each ``f<n>_prox -> f<n>_mid -> f<n>_dist`` pointing +y in
    armature space with roll 0 - the same convention as the piano rig, so
    pose-space x-rotation is curl and z-rotation is sideways reach. The
    armature object is rotated 90 degrees about Z (HAND_ROT_Z) and then
    tilted WRAP_TILT about the neck axis, giving a hand *wrapped around
    the neck*: the palm hangs beside the treble edge below string level,
    the knuckle line rides just above the treble edge (index toward the
    nut, pinky toward the bridge, matching the engine's finger-per-fret
    hand positions), the fingers arch up and over the strings, and a
    static thumb box reaches under the neck to press its back - opposite
    the fingers, like a real fretting grip.
  - ``PickHand``: a stylized loose fist holding a pick, a single wrist
    bone carrying rigid palm/finger/thumb boxes and a flat pick whose tip
    sits at PICK_TIP_LOCAL in armature space. It is oriented (PICK_PITCH
    /PICK_YAW) so the fingers run *parallel to the strings* (along the
    neck axis) and curl into the palm, the thumb crosses over the index
    to pinch the pick, and the pick protrudes down toward the strings.
    The animator drives only the object location, sweeping the pick tip
    across the strings.

Rigid boxes bone-parented to each bone match the piano hands' blocky
aesthetic; no skinning.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_hands", "/path/to/guitar/build_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_hands()
"""

import math

import bpy
import bmesh
import mathutils

try:
    from . import fret_layout
except ImportError:  # loaded as a loose script via importlib
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import fret_layout


# Both hand armatures are rotated so armature-local +y points across the
# strings (world -X) and local +x points up the neck (world +Y).
HAND_ROT_Z = math.pi / 2.0

# The fret hand is additionally tilted about the neck axis so it wraps
# the neck from the treble side: at 0 the palm hovers flat above the
# fretboard (lap-steel style); at WRAP_TILT the palm hangs beside the
# neck and the finger rest direction points up-and-over the strings.
WRAP_TILT = 0.6


def hand_world_offset(v):
    """Armature-local offset -> world offset under the HAND_ROT_Z pose."""
    return (-v[1], v[0], v[2])


# The pick hand is not laid flat across the strings like a lap-steel; it
# is a loose fist whose fingers run *along* the neck (parallel to the
# strings). Built in a canonical local frame (+y = toward the nut along
# the strings, +z = up/back-of-hand, +x = toward the treble/thumb side),
# then pitched forward-down over the strings and yawed so the forearm
# comes in from the treble side.
PICK_PITCH = 0.55   # forward-down tilt of the whole hand
PICK_YAW = -0.25    # forearm angle across the strings

PICK_ROT = (mathutils.Matrix.Rotation(PICK_YAW, 3, 'Z')
            @ mathutils.Matrix.Rotation(-PICK_PITCH, 3, 'X'))


def pick_world_offset(v):
    """Armature-local offset -> world offset under the PickHand pose."""
    return tuple(PICK_ROT @ mathutils.Vector(v))


def fret_world_offset(v):
    """Armature-local offset -> world offset for the FretHand's pose
    (HAND_ROT_Z, then WRAP_TILT about the world neck axis)."""
    ct, st = math.cos(WRAP_TILT), math.sin(WRAP_TILT)
    return (v[2] * st - v[1] * ct, v[0], v[1] * st + v[2] * ct)


def _fret_local_offset(w):
    """Inverse of fret_world_offset: world offset -> armature-local."""
    ct, st = math.cos(WRAP_TILT), math.sin(WRAP_TILT)
    return (w[1], -(w[0] * ct - w[2] * st), w[0] * st + w[2] * ct)


# ---------------------------------------------------------------------------
# Fret-hand dimensions (metres), in armature-local space. Knuckle x spreads
# the fingers along the neck (one per fret, ~mid-neck fret spacing); y is
# the reach direction across the strings.
# ---------------------------------------------------------------------------

FRET_FINGERS = {
    #        knuckle (x, y, z)            (prox, mid, dist) segment lengths
    1: {"knuckle": (0.042, 0.050, 0.0), "lengths": (0.042, 0.026, 0.020)},
    2: {"knuckle": (0.014, 0.054, 0.0), "lengths": (0.046, 0.030, 0.022)},
    3: {"knuckle": (-0.014, 0.052, 0.0), "lengths": (0.043, 0.028, 0.021)},
    4: {"knuckle": (-0.042, 0.046, 0.0), "lengths": (0.035, 0.024, 0.019)},
}

FINGER_BOX_W = 0.011
FINGER_BOX_H = 0.012
FRET_PALM_SIZE = (0.095, 0.075, 0.020)
PICK_PALM_SIZE = (0.075, 0.085, 0.026)

# Pick tip in PickHand armature-local space: below the thumb/index pinch,
# reaching down toward the strings.
PICK_PINCH = (0.014, 0.044, -0.008)
PICK_TIP_LOCAL = (0.014, 0.044, -0.047)

# Where idle hands hover before the animator takes over.
REST_LOCATION = {
    "FretHand": (0.085, fret_layout.fret_y(2), 0.026),
    "PickHand": (0.045, fret_layout.PLUCK_Y + 0.045, 0.100),
}


def _make_box_mesh(name, sx, sy, sz, bevel=0.0018):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2,
                    affect='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _skin_material():
    mat = bpy.data.materials.get("HandSkin") or bpy.data.materials.new("HandSkin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.72, 0.55, 0.45, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec:
        spec.default_value = 0.15
    return mat


def _pick_material():
    mat = bpy.data.materials.get("PickPlastic") or bpy.data.materials.new("PickPlastic")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.85, 0.30, 0.10, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.35
    return mat


def _bone_box(arm_obj, coll, mat, bone_name, size, location,
              rotation=(0.0, 0.0, 0.0), mesh=None):
    """Create a box (or a supplied mesh) rigidly parented to a bone.

    Bone parent space has its origin at the bone *tail* with y running
    along the bone.
    """
    if mesh is None:
        mesh = _make_box_mesh(f"{arm_obj.name}_{bone_name}Mesh", *size)
    obj = bpy.data.objects.new(f"{arm_obj.name}_{bone_name}", mesh)
    coll.objects.link(obj)
    obj.parent = arm_obj
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(mat)
    return obj


def _new_armature(name, coll, tilt=0.0):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = REST_LOCATION[name]
    rot = (mathutils.Matrix.Rotation(tilt, 4, 'Y')
           @ mathutils.Matrix.Rotation(HAND_ROT_Z, 4, 'Z'))
    arm_obj.rotation_euler = rot.to_euler()
    coll.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    return arm_obj


def build_fret_hand(coll, mat):
    """The articulated fretting hand: four IK-driven finger chains
    wrapping the neck from the treble side, thumb under the neck."""
    arm_obj = _new_armature("FretHand", coll, tilt=WRAP_TILT)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones

    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)

    for f, spec in FRET_FINGERS.items():
        kx, ky, kz = spec["knuckle"]
        y = ky
        parent = wrist
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            bone = eb.new(f"f{f}_{seg}")
            bone.head = (kx, y, kz)
            bone.tail = (kx, y + length, kz)
            bone.roll = 0.0
            bone.parent = parent
            bone.use_connect = seg != "prox"
            parent = bone
            y += length

    bpy.ops.object.mode_set(mode='OBJECT')
    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = 'XYZ'

    for f, spec in FRET_FINGERS.items():
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            _bone_box(arm_obj, coll, mat, f"f{f}_{seg}",
                      (FINGER_BOX_W, length * 0.92, FINGER_BOX_H),
                      (0.0, -length / 2.0, 0.0))

    _bone_box(arm_obj, coll, mat, "wrist", FRET_PALM_SIZE, (0.0, -0.013, 0.0))

    # Static thumb reaching under the neck to press its back, opposite
    # the fingers. Specified as a *world* offset from the wrist (the
    # animator keeps the wrist near x 0.07-0.09, z ~0.02, so the thumb
    # tip lands under the neck's centreline around z just below 0) and
    # converted into the tilted armature's local frame.
    thumb_center_w = (-0.044, 0.008, -0.020)
    thumb_dir_w = mathutils.Vector((-0.94, 0.12, -0.32)).normalized()
    c_local = _fret_local_offset(thumb_center_w)
    d_local = mathutils.Vector(_fret_local_offset(thumb_dir_w))
    _bone_box(arm_obj, coll, mat, "wrist", (0.018, 0.070, 0.015),
              (c_local[0], c_local[1] - 0.030, c_local[2]),
              rotation=d_local.to_track_quat('Y', 'Z').to_euler())
    return arm_obj


def _make_pick_mesh(name, length):
    """A flat rounded-triangle pick, tip pointing down (-z) `length`
    below the object origin, blade in the y-z plane (thickness along x)
    so its face points across the strings after the hand's rotation."""
    verts = []
    outline = [(-0.010, 0.002), (0.010, 0.002), (0.0, -length)]  # (y, z)
    for dx in (-0.0008, 0.0008):
        for y, z in outline:
            verts.append((dx, y, z))
    faces = [(0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _seg_box(arm_obj, coll, mat, width, height, p0, p1):
    """A box of the given cross-section spanning armature points p0->p1,
    its long (local-y) axis along the segment."""
    p0, p1 = mathutils.Vector(p0), mathutils.Vector(p1)
    d = p1 - p0
    center = (p0 + p1) / 2.0
    rot = d.to_track_quat('Y', 'Z').to_euler()
    _bone_box(arm_obj, coll, mat, "wrist", (width, d.length, height),
              (center.x, center.y - 0.030, center.z), rotation=rot)


def build_pick_hand(coll, mat):
    """The stylized picking fist: fingers curl along the strings, the
    thumb crosses over to pinch the pick against the index finger."""
    arm_obj = _new_armature("PickHand", coll)
    arm_obj.rotation_euler = PICK_ROT.to_euler()
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.pose.bones["wrist"].rotation_mode = 'XYZ'

    # Back of the hand / fist mass, sitting above and behind the pinch.
    _bone_box(arm_obj, coll, mat, "wrist", (0.066, 0.062, 0.030),
              (0.0, 0.008 - 0.030, 0.012))

    # Four fingers curling into the palm: each runs forward-and-down from
    # its knuckle (prox), then folds back under (dist). They are spread
    # across x and point along +y, i.e. parallel to the strings. The
    # middle finger is a touch longer, the outer fingers shorter.
    knuckles = {  # x, forward-reach scale
        "index": (0.026, 1.00), "middle": (0.009, 1.08),
        "ring": (-0.009, 1.00), "pinky": (-0.027, 0.86),
    }
    for kx, scale in knuckles.values():
        knuckle = mathutils.Vector((kx, 0.040, 0.006))
        # Prox drops forward-down from the knuckle; the distal folds back
        # under the palm and up, so the fingertips tuck away from the
        # strings and only the pick reaches down.
        mid = knuckle + mathutils.Vector((0.0, 0.011 * scale, -0.024 * scale))
        tip = mid + mathutils.Vector((0.0, -0.023 * scale, 0.007 * scale))
        _seg_box(arm_obj, coll, mat, FINGER_BOX_W, FINGER_BOX_H, knuckle, mid)
        _seg_box(arm_obj, coll, mat, FINGER_BOX_W, FINGER_BOX_H, mid, tip)

    # Thumb: from the treble side of the palm, crossing forward over the
    # index to the pinch, where its pad holds the pick.
    thumb_base = (0.030, 0.004, 0.010)
    thumb_knuckle = (0.026, 0.028, -0.002)
    _seg_box(arm_obj, coll, mat, 0.016, 0.015, thumb_base, thumb_knuckle)
    _seg_box(arm_obj, coll, mat, 0.015, 0.014, thumb_knuckle, PICK_PINCH)

    # The pick, pinched between thumb and index, protruding toward the
    # strings. Built vertical in armature space (tip straight below the
    # pinch); the hand's forward pitch angles it down at the strings.
    length = PICK_PINCH[2] - PICK_TIP_LOCAL[2]
    _bone_box(arm_obj, coll, _pick_material(), "wrist", None,
              (PICK_PINCH[0], PICK_PINCH[1] - 0.030, PICK_PINCH[2]),
              mesh=_make_pick_mesh("PickMesh", length))
    return arm_obj


def build_hands():
    scene = bpy.context.scene

    old = bpy.data.collections.get("GuitarHands")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("GuitarHands")
    scene.collection.children.link(coll)
    mat = _skin_material()

    hands = [build_fret_hand(coll, mat), build_pick_hand(coll, mat)]
    print(f"Built {len(hands)} guitar hand rigs: " +
          ", ".join(h.name for h in hands))
    return hands


if __name__ == "__main__":
    build_hands()
