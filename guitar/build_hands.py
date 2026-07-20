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
  - ``PickHand``: a stylized fist - a single wrist bone carrying rigid
    palm/finger/thumb boxes and a flat pick whose tip sits at
    PICK_TIP_LOCAL in armature space. The animator drives only the object
    location, sweeping the pick tip across the strings.

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

FINGER_BOX_W = 0.012
FINGER_BOX_H = 0.012
FRET_PALM_SIZE = (0.095, 0.075, 0.020)
PICK_PALM_SIZE = (0.075, 0.085, 0.026)

# Pick tip in PickHand armature-local space: under the palm's front edge
# at the thumb pinch, reaching down toward the strings.
PICK_TIP_LOCAL = (0.0, 0.018, -0.046)

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


def _make_pick_mesh(name):
    """A flat rounded-triangle pick, tip pointing down (-z), thickness
    along y; the tip sits 0.026 below the object origin."""
    verts, faces = [], []
    outline = [(-0.0095, 0.0), (0.0095, 0.0), (0.0, -0.026)]
    for dy in (-0.00075, 0.00075):
        for x, z in outline:
            verts.append((x, dy, z))
    faces = [(0, 1, 2), (5, 4, 3), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def build_pick_hand(coll, mat):
    """The stylized picking fist: one bone, rigid boxes, and the pick."""
    arm_obj = _new_armature("PickHand", coll)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.pose.bones["wrist"].rotation_mode = 'XYZ'

    _bone_box(arm_obj, coll, mat, "wrist", PICK_PALM_SIZE, (0.0, -0.013, 0.0))
    # Four finger stubs folded under the palm's leading edge.
    for i, lx in enumerate((0.030, 0.010, -0.010, -0.030)):
        _bone_box(arm_obj, coll, mat, "wrist",
                  (FINGER_BOX_W, 0.040, FINGER_BOX_H),
                  (lx, 0.028, -0.017), rotation=(1.15, 0.0, 0.0))
    # Thumb along the bridge side, its tip covering the pick pinch.
    _bone_box(arm_obj, coll, mat, "wrist", (0.055, 0.018, 0.014),
              (-0.022, 0.000, -0.015))

    # The pick: object origin at the pinch point so the tip lands exactly
    # at PICK_TIP_LOCAL (armature space = bone parent space here, because
    # the wrist bone lies along +y with roll 0 and its tail is at +0.030;
    # compensate the tail offset in the y coordinate).
    px, py, pz = PICK_TIP_LOCAL
    _bone_box(arm_obj, coll, _pick_material(), "wrist", None,
              (px, py - 0.030, pz + 0.026),
              mesh=_make_pick_mesh("PickMesh"))
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
