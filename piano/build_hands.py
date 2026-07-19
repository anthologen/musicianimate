"""Builds the two minimalist hand/finger rigs used by animate_hands.py.

Run inside Blender (after build_piano.py) to create a "Hands" collection
containing one armature per hand, named ``Hand_L`` / ``Hand_R``:

  - bone ``wrist`` (root), plus per finger f1 (thumb) .. f5 (pinky) a chain
    ``f<n>_prox`` -> ``f<n>_mid`` -> ``f<n>_dist``. All finger bones point
    +y (toward the fallboard) in rest pose with roll 0, so their pose-space
    x-rotation is pitch (curl) and z-rotation is sideways reach - the two
    axes animate_hands.py drives with closed-form IK.
  - box meshes bone-parented to every phalanx and a palm box on the wrist,
    matching the piano's blocky aesthetic. No skinning/weights: rigid
    boxes are enough at this level of stylization.

The left hand mirrors the right across x (thumb knuckle toward the treble
side instead of the bass side); bone names and axes are identical.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_hands", "/path/to/build_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_hands()
"""

import bpy
import bmesh


# ---------------------------------------------------------------------------
# Hand dimensions (meters), right-hand orientation; x mirrors for the left.
# Knuckle offsets are relative to the armature object origin (the wrist).
# ---------------------------------------------------------------------------

FINGERS = {
    #        knuckle (x, y, z)          (prox, mid, dist) segment lengths
    1: {"knuckle": (-0.040, 0.020, -0.012), "lengths": (0.036, 0.030, 0.024)},
    2: {"knuckle": (-0.014, 0.055, 0.000),  "lengths": (0.042, 0.026, 0.020)},
    3: {"knuckle": (0.005, 0.058, 0.000),   "lengths": (0.046, 0.030, 0.022)},
    4: {"knuckle": (0.024, 0.055, 0.000),   "lengths": (0.043, 0.028, 0.021)},
    5: {"knuckle": (0.043, 0.050, 0.000),   "lengths": (0.035, 0.024, 0.019)},
}

FINGER_BOX_W = 0.014
FINGER_BOX_H = 0.012
THUMB_BOX_W = 0.017
PALM_SIZE = (0.095, 0.080, 0.020)

# Where idle hands rest before/after playing (animator overrides location).
REST_LOCATION = {"L": (-0.25, -0.11, 0.075), "R": (0.25, -0.11, 0.075)}


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


def _bone_box(arm_obj, coll, mat, bone_name, size, offset_y):
    """Create a box and rigidly parent it to a bone.

    Bone parent space has its origin at the bone *tail* with y running along
    the bone, so a box centered on the bone sits at y = -length/2.
    """
    mesh = _make_box_mesh(f"{arm_obj.name}_{bone_name}Mesh", *size)
    obj = bpy.data.objects.new(f"{arm_obj.name}_{bone_name}", mesh)
    coll.objects.link(obj)
    obj.parent = arm_obj
    obj.parent_type = 'BONE'
    obj.parent_bone = bone_name
    obj.location = (0.0, offset_y, 0.0)
    obj.data.materials.append(mat)
    return obj


def build_hand(side, coll, mat):
    """Create one hand armature (side "L" or "R") with its display boxes."""
    name = f"Hand_{side}"
    mirror = -1.0 if side == "L" else 1.0

    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = REST_LOCATION[side]
    coll.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)

    bone_lengths = {}
    for f, spec in FINGERS.items():
        kx, ky, kz = spec["knuckle"]
        kx *= mirror
        y = ky
        parent = wrist
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            bone = eb.new(f"f{f}_{seg}")
            bone.head = (kx, y, kz)
            bone.tail = (kx, y + length, kz)
            bone.roll = 0.0
            bone.parent = parent
            bone.use_connect = seg != "prox"
            bone_lengths[bone.name] = length
            parent = bone
            y += length

    bpy.ops.object.mode_set(mode='OBJECT')

    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = 'XYZ'

    for f, spec in FINGERS.items():
        w = THUMB_BOX_W if f == 1 else FINGER_BOX_W
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            _bone_box(arm_obj, coll, mat, f"f{f}_{seg}",
                      (w, length * 0.92, FINGER_BOX_H), -length / 2.0)

    # Palm spans from the wrist to the knuckle line (wrist bone tail at
    # y=0.030 is the parent-space origin).
    _bone_box(arm_obj, coll, mat, "wrist", PALM_SIZE, -0.013)

    return arm_obj


def build_hands():
    scene = bpy.context.scene

    old = bpy.data.collections.get("Hands")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Hands")
    scene.collection.children.link(coll)
    mat = _skin_material()

    hands = [build_hand("L", coll, mat), build_hand("R", coll, mat)]
    print(f"Built {len(hands)} hand rigs: " +
          ", ".join(h.name for h in hands))
    return hands


if __name__ == "__main__":
    build_hands()
