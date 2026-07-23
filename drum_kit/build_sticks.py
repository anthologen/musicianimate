"""Builds the drummer's limbs: two stick-holding hands and two shod feet.

Run inside Blender AFTER build_drum_kit.py to create a "Drummer" collection:

  - ``Stick_R`` / ``Stick_L``: an Empty root carrying a rigid stylized fist
    and a drumstick mesh whose TIP sits at STICK_TIP_LOCAL in the root's local
    space. The animator moves and rotates the root so the stick tip lands on a
    drum/cymbal strike point; because the hands are rigid there is no armature
    or IK to solve (mirrors the guitar PickHand's "animate the object, the
    rigid tip tracks the target" approach). STICK_ROT0 is the neutral playing
    orientation both this builder and animate_drums.py use, so the exported
    tip offset is consistent on both sides.
  - ``Shoe_R`` / ``Shoe_L``: minimalist shoe meshes parented to the kick and
    hi-hat footboards built by build_drum_kit.py (origin at the heel hinge),
    so rotating a footboard presses its shoe. Right foot = kick, left foot =
    hi-hat, per the standard convention.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_sticks", "/path/to/drum_kit/build_sticks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_drummer()
"""

import math

import bpy
import bmesh
import mathutils

try:
    from . import kit_layout
except ImportError:  # loaded as a loose script via importlib
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import kit_layout


# Drumstick tip in the hand root's local space (+Y forward, +Z up): the stick
# runs forward from the grip and the tip sits a little below the grip line.
STICK_LEN = 0.40                       # ~16" standard stick
STICK_GRIP = (0.0, -0.03, 0.0)         # butt end, in the fist
STICK_TIP_LOCAL = (0.0, STICK_LEN - 0.03, -0.03)

# Neutral playing orientation of each hand root (Euler XYZ). The drummer sits
# at +Y, so the hands must approach each surface from the +Y (drummer) side
# and angle down toward -Y onto the head: pitch about X tilts the stick down,
# and the ~pi Z yaw turns the hand around so it reaches from behind the kit
# rather than from the audience side. A small offset from pi angles each stick
# inward. Shared with animate_drums.py so its tip-offset math matches the rig.
STICK_ROT0 = {
    "R": (-0.60, 0.0, math.pi - 0.30),
    "L": (-0.60, 0.0, math.pi + 0.30),
}


def _tip_offset(side):
    """World offset from a hand root's origin to its stick tip at rest."""
    rot = mathutils.Euler(STICK_ROT0[side], "XYZ").to_matrix()
    return rot @ mathutils.Vector(STICK_TIP_LOCAL)


def _rest_location(side):
    """Hover the stick tip ~12 cm above this hand's convention home."""
    home = kit_layout.strike_point("hihat_closed" if side == "R" else "snare")
    off = _tip_offset(side)
    return (home[0] - off[0], home[1] - off[1], home[2] + 0.12 - off[2])


# ---------------------------------------------------------------------------
# Materials + mesh helpers
# ---------------------------------------------------------------------------
def _mat(name, color, roughness, metallic=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _box_mesh(name, sx, sy, sz, bevel=0.004):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2,
                        affect='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _stick_mesh(name, grip, tip):
    """A tapered drumstick from `grip` (butt) to `tip`, plus a small tip bead,
    built directly in the hand root's local frame."""
    grip, tip = mathutils.Vector(grip), mathutils.Vector(tip)
    vec = tip - grip
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    shaft = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12,
                                  radius1=0.008, radius2=0.0045, depth=vec.length)
    # create_cone builds along +Z centred at origin; orient it along the stick.
    rot = vec.to_track_quat('Z', 'Y').to_matrix()
    bmesh.ops.rotate(bm, verts=shaft["verts"], matrix=rot)
    bmesh.ops.translate(bm, verts=shaft["verts"], vec=(grip + vec / 2.0))
    bead = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=0.007)
    bmesh.ops.translate(bm, verts=bead["verts"], vec=tip)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _shoe_mesh(name):
    """A minimalist shoe: a low beveled slab, heel at the local origin (+Y),
    toe extending -Y so it sits along a footboard hinged at its heel."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.09, 0.22, 0.06), verts=bm.verts)
    # Cube spans [-0.11, 0.11] in y; shift so the heel (+Y) is at the origin
    # and the toe runs to y = -0.22, and lift it onto the footboard top.
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, -0.11, 0.035))
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.012, segments=2, affect='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _add_child(coll, name, mesh, mat, parent, location=(0, 0, 0)):
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.parent = parent
    obj.location = location
    if mat is not None:
        obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_stick_hand(coll, side, skin_mat, wood_mat):
    """An Empty root holding a rigid fist + drumstick, at its rest hover."""
    root = bpy.data.objects.new(f"Stick_{side}", None)
    root.empty_display_size = 0.04
    root.location = _rest_location(side)
    root.rotation_euler = STICK_ROT0[side]
    coll.objects.link(root)

    # Back of the hand / loose fist gripping the butt of the stick.
    _add_child(coll, f"Stick_{side}_Fist", _box_mesh(f"Stick_{side}_FistMesh",
               0.052, 0.062, 0.036), skin_mat, root, (0.0, -0.012, 0.006))
    # A couple of stylized knuckle boxes wrapping over the grip.
    for i, kx in enumerate((0.016, -0.004, -0.022)):
        _add_child(coll, f"Stick_{side}_Knuckle_{i}",
                   _box_mesh(f"Stick_{side}_Knuckle_{i}Mesh", 0.012, 0.030, 0.014),
                   skin_mat, root, (kx, 0.014, -0.004))
    # Thumb crossing over the grip.
    _add_child(coll, f"Stick_{side}_Thumb",
               _box_mesh(f"Stick_{side}_ThumbMesh", 0.014, 0.034, 0.014),
               skin_mat, root, (0.026, 0.004, 0.010))
    # The drumstick itself, tip at STICK_TIP_LOCAL.
    _add_child(coll, f"Stick_{side}_Stick",
               _stick_mesh(f"Stick_{side}_StickMesh", STICK_GRIP, STICK_TIP_LOCAL),
               wood_mat, root)
    return root


def build_shoe(coll, name, footboard_name, rubber_mat):
    """A shoe parented to a footboard so pressing the pedal presses the shoe."""
    shoe = bpy.data.objects.new(name, _shoe_mesh(f"{name}Mesh"))
    coll.objects.link(shoe)
    shoe.data.materials.append(rubber_mat)
    footboard = bpy.data.objects.get(footboard_name)
    if footboard is not None:
        shoe.parent = footboard
        shoe.location = (0.0, 0.0, 0.0)
    else:  # kit not built yet - park it near the floor at the origin
        shoe.location = (0.0, 0.0, 0.02)
        print(f"warning: {footboard_name} not found; {name} left unparented")
    return shoe


def build_drummer():
    scene = bpy.context.scene

    old = bpy.data.collections.get("Drummer")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Drummer")
    scene.collection.children.link(coll)

    skin = _mat("DrummerSkin", (0.72, 0.55, 0.45), 0.7)
    wood = _mat("StickWood", (0.80, 0.66, 0.44), 0.45)
    rubber = _mat("ShoeRubber", (0.04, 0.04, 0.05), 0.6)

    limbs = [
        build_stick_hand(coll, "R", skin, wood),
        build_stick_hand(coll, "L", skin, wood),
        build_shoe(coll, "Shoe_R", "Kick_Footboard", rubber),
        build_shoe(coll, "Shoe_L", "HiHat_Footboard", rubber),
    ]
    print(f"Built drummer: {', '.join(l.name for l in limbs)}")
    return coll


if __name__ == "__main__":
    build_drummer()
