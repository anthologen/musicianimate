"""Procedurally builds the minimalist 88-key piano used by this project.

Run this inside Blender (Scripting tab / Text Editor "Run Script", the
Python console, or ``blender --background --python build_piano.py``) to
fully reconstruct the scene from scratch if the .blend file is ever lost.

Produces a "Piano" collection containing:
  - 88 key objects named WhiteKey_<midi> / BlackKey_<midi>, where <midi> is
    the standard MIDI note number (21 = A0 ... 108 = C8). These names are
    required by piano_midi_animator.py.
  - A minimalist matte-black case: keybed slab, back rail, two trestle legs,
    and a stretcher bar.
  - Materials for white keys, black keys, and the case.

Also positions a camera and two-point studio lighting for a clean render,
and sets color management to Standard so blacks render as true black
instead of being lifted by the default filmic/AgX tone mapping.
"""

import bpy
import bmesh
import mathutils

# Key geometry lives in key_layout.py (bpy-free) so the fingering engine can
# compute fingertip targets that match the scene exactly. The fallback import
# covers running this file as a standalone script inside Blender, where it has
# no package context.
try:
    from .key_layout import (
        WHITE_W, WHITE_LEN, WHITE_H, BLACK_W, BLACK_LEN, BLACK_H,
        WHITE_NOTES, START_MIDI, END_MIDI, WHITE_REST_Z, BLACK_REST_Z,
        _raw_x,
    )
except ImportError:
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from key_layout import (
        WHITE_W, WHITE_LEN, WHITE_H, BLACK_W, BLACK_LEN, BLACK_H,
        WHITE_NOTES, START_MIDI, END_MIDI, WHITE_REST_Z, BLACK_REST_Z,
        _raw_x,
    )


def _make_cube_mesh(name, sx, sy, sz):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _bevel_mesh(mesh, offset, segments=2):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=offset, segments=segments, affect='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def clear_scene():
    """Remove all mesh objects (keeps Light/Camera) and any stale Piano collection."""
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    if "Piano" in bpy.data.collections:
        old = bpy.data.collections["Piano"]
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)


def build_keys(piano_coll):
    """Create the 88 key objects and return the computed total keyboard width."""
    white_mesh = _make_cube_mesh("WhiteKeyMesh", WHITE_W, WHITE_LEN, WHITE_H)
    black_mesh = _make_cube_mesh("BlackKeyMesh", BLACK_W, BLACK_LEN, BLACK_H)
    _bevel_mesh(white_mesh, 0.0015)
    _bevel_mesh(black_mesh, 0.0012)

    keys = []  # (midi, is_white, x)
    for midi in range(START_MIDI, END_MIDI + 1):
        keys.append((midi, midi % 12 in WHITE_NOTES, _raw_x(midi)))

    min_x = min(k[2] for k in keys)
    max_x = max(k[2] for k in keys) + WHITE_W
    total_width = max_x - min_x
    shift = -min_x - total_width / 2.0

    white_y = WHITE_LEN / 2.0
    black_y = WHITE_LEN - BLACK_LEN / 2.0

    for midi, is_white, x in keys:
        x_final = x + shift
        if is_white:
            obj = bpy.data.objects.new(f"WhiteKey_{midi}", white_mesh)
            obj.location = (x_final + WHITE_W / 2.0, white_y, WHITE_REST_Z)
        else:
            obj = bpy.data.objects.new(f"BlackKey_{midi}", black_mesh)
            obj.location = (x_final + WHITE_W / 2.0, black_y, BLACK_REST_Z)
        piano_coll.objects.link(obj)

    return total_width


def build_case(piano_coll, total_width):
    """Minimalist keybed slab, back rail, two trestle legs, and a stretcher bar."""
    def add_box(name, sx, sy, sz, loc):
        mesh = _make_cube_mesh(name + "Mesh", sx, sy, sz)
        obj = bpy.data.objects.new(name, mesh)
        obj.location = loc
        piano_coll.objects.link(obj)
        return obj

    body_w = total_width + 0.03
    body_depth = WHITE_LEN + 0.16
    body_h = 0.07
    body_y_center = (body_depth / 2.0) - 0.005
    add_box("Case_Body", body_w, body_depth, body_h, (0, body_y_center, -body_h / 2.0))

    rail_h, rail_depth = 0.02, 0.03
    rail_y = body_depth - rail_depth / 2.0
    add_box("Case_BackRail", body_w, rail_depth, rail_h, (0, rail_y, rail_h / 2.0))

    leg_h, leg_thick = 0.62, 0.02
    leg_depth = body_depth - 0.04
    leg_x = body_w / 2.0 - leg_thick / 2.0 - 0.01
    leg_z = -body_h - leg_h / 2.0
    leg_y = body_y_center
    add_box("Leg_Left", leg_thick, leg_depth, leg_h, (-leg_x, leg_y, leg_z))
    add_box("Leg_Right", leg_thick, leg_depth, leg_h, (leg_x, leg_y, leg_z))

    stretcher_h = 0.02
    stretcher_len = (leg_x * 2) - leg_thick
    add_box("Stretcher", stretcher_len, 0.04, stretcher_h,
            (0, leg_y, -body_h - leg_h + stretcher_h / 2.0 + 0.03))


def setup_materials(piano_coll):
    def make_mat(name, color, roughness, specular):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        spec_input = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
        if spec_input:
            spec_input.default_value = specular
        return mat

    # Specular is kept low: at the shallow, grazing camera angles typical of
    # a keyboard shot, Fresnel reflection of a bright studio background can
    # wash out even a near-black base color if specular is left at default.
    white_mat = make_mat("PianoWhiteKey", (0.93, 0.93, 0.92), 0.25, 0.25)
    black_mat = make_mat("PianoBlackKey", (0.015, 0.015, 0.016), 0.15, 0.15)
    case_mat = make_mat("PianoCase", (0.02, 0.02, 0.023), 0.4, 0.2)

    for obj in piano_coll.objects:
        obj.data.materials.clear()
        if obj.name.startswith("WhiteKey_"):
            obj.data.materials.append(white_mat)
        elif obj.name.startswith("BlackKey_"):
            obj.data.materials.append(black_mat)
        elif obj.name.startswith(("Case_", "Leg_", "Stretcher")):
            obj.data.materials.append(case_mat)


def setup_camera_and_lighting(scene):
    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
    cam.location = (0.9, -1.1, 0.95)
    look_at = mathutils.Vector((0, 0.05, 0.02))
    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 45
    scene.camera = cam

    key_light = bpy.data.objects.get("Light")
    if key_light is None:
        light_data = bpy.data.lights.new("Light", type='AREA')
        key_light = bpy.data.objects.new("Light", light_data)
        bpy.context.collection.objects.link(key_light)
    key_light.data.type = 'AREA'
    key_light.data.energy = 60
    key_light.data.size = 2.0
    key_light.location = (1.2, -1.0, 1.6)

    fill_light = bpy.data.objects.get("FillLight")
    if fill_light is None:
        fill_data = bpy.data.lights.new("FillLightData", type='AREA')
        fill_light = bpy.data.objects.new("FillLight", fill_data)
        bpy.context.collection.objects.link(fill_light)
    fill_light.data.energy = 20
    fill_light.data.size = 3.0
    fill_light.location = (-1.4, -0.8, 1.0)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.55, 0.55, 0.56, 1.0)
    bg.inputs[1].default_value = 1.0

    # Standard (not filmic/AgX) so near-black materials render truly dark.
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'

    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800


def build_piano():
    scene = bpy.context.scene
    clear_scene()

    piano_coll = bpy.data.collections.new("Piano")
    scene.collection.children.link(piano_coll)

    total_width = build_keys(piano_coll)
    build_case(piano_coll, total_width)
    setup_materials(piano_coll)
    setup_camera_and_lighting(scene)

    print(f"Piano built: 88 keys, total width {total_width:.3f}m")
    return piano_coll


if __name__ == "__main__":
    build_piano()
