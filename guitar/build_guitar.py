"""Procedurally builds a minimalist, realistically sized electric guitar in Blender.

Run this inside Blender (Scripting tab / Text Editor "Run Script", the Python
console, or ``blender --background --python build_guitar.py``) to reconstruct
the scene from scratch.

All dimensions are in metres and taken from a standard 25.5" scale-length
solid-body electric (Stratocaster class): 648 mm nut-to-bridge, ~450 mm body,
43 mm nut width, 22 frets with equal-tempered spacing. The look is deliberately
minimalist — a soft-cornered slab body, a plain tapered neck/fretboard, and only
the hardware that reads as "electric guitar": frets, six strings, two humbuckers,
a bridge, six inline tuners, and a pair of control knobs.

Produces a "Guitar" collection. The guitar lies face-up (+Z), neck pointing +Y:
the bridge is near the world origin and the headstock is out at +Y, so the whole
instrument sits flat on the z=0 plane ready for a top-down or 3/4 render.
"""

import math

import bpy
import bmesh
import mathutils

# ---------------------------------------------------------------------------
# Geometry constants (metres). Change BODY_COLOR / SCALE_LEN etc. to taste.
# ---------------------------------------------------------------------------
SCALE_LEN = 0.648          # nut-to-bridge speaking length (25.5")
NUM_FRETS = 22
NUM_STRINGS = 6

BRIDGE_Y = 0.0             # bridge saddles sit here, on the body
NUT_Y = BRIDGE_Y + SCALE_LEN

BODY_Y0, BODY_Y1 = -0.20, 0.25     # body extent along the neck axis
BODY_W = 0.33                       # widest point of the slab
BODY_THICK = 0.045

NECK_THICK = 0.021                  # wood behind the fretboard
FB_THICK = 0.006                    # fretboard
FB_Y0 = 0.16                        # fretboard overhangs the body to reach fret 22
HEEL_Y = BODY_Y1                    # neck wood joins the body here

NUT_W = 0.043                       # neck width at the nut
END_W = 0.058                       # neck width at the fretboard's body end

FRET_H = 0.0012
STRING_R = 0.0006
STRING_Z = NECK_THICK + FB_THICK + 0.004   # string plane above the fretboard

HEAD_LEN = 0.17
HEAD_W = 0.072
HEAD_DROP = 0.010                   # headstock sits below the neck (break angle)

# Minimalist palette — a single muted body colour plus natural wood + metal.
BODY_COLOR = (0.055, 0.075, 0.085)  # matte deep slate; swap for any accent


def _neck_width(y):
    """Interpolated fretboard width at a given y (widest at the body end)."""
    t = (NUT_Y - y) / (NUT_Y - FB_Y0)
    t = max(0.0, min(1.0, t))
    return NUT_W + (END_W - NUT_W) * t


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def _make_cube_mesh(name, sx, sy, sz):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _bevel_mesh(mesh, offset, segments=3):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=offset, segments=segments, affect='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _make_cyl_mesh(name, radius, depth, segments=24):
    """Cylinder centred at the origin, axis along +Z."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                          radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _make_tapered_box(name, y0, y1, w0, w1, z0, z1):
    """Box spanning y0..y1, tapering in width from w0 (at y0) to w1 (at y1),
    spanning z0..z1. Built in world space, so the object's location is origin."""
    hw0, hw1 = w0 / 2.0, w1 / 2.0
    verts = [
        (-hw0, y0, z0), (hw0, y0, z0), (hw1, y1, z0), (-hw1, y1, z0),  # bottom
        (-hw0, y0, z1), (hw0, y0, z1), (hw1, y1, z1), (-hw1, y1, z1),  # top
    ]
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def clear_scene():
    """Remove mesh objects (keeps Light/Camera) and any stale Guitar collection."""
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    if "Guitar" in bpy.data.collections:
        old = bpy.data.collections["Guitar"]
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)


def _add(coll, name, mesh, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    coll.objects.link(obj)
    return obj


def _add_cylinder_between(coll, name, p1, p2, radius, segments=16):
    """A capped cylinder spanning two arbitrary points — used for strings and
    tuner posts. Orients the +Z cylinder axis onto the p1->p2 vector."""
    p1, p2 = mathutils.Vector(p1), mathutils.Vector(p2)
    vec = p2 - p1
    length = vec.length
    mesh = _make_cyl_mesh(name + "Mesh", radius, length, segments)
    obj = _add(coll, name, mesh, location=(p1 + vec / 2.0))
    obj.rotation_euler = vec.to_track_quat('Z', 'Y').to_euler()
    return obj


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def build_body(coll):
    """Soft-cornered slab body, offset slightly toward the bass side for a
    subtly asymmetric minimalist silhouette."""
    mesh = _make_cube_mesh("BodyMesh", BODY_W, BODY_Y1 - BODY_Y0, BODY_THICK)
    _bevel_mesh(mesh, 0.035, segments=4)   # generously rounded corners
    _add(coll, "Guitar_Body", mesh,
         location=(0.0, (BODY_Y0 + BODY_Y1) / 2.0, -BODY_THICK / 2.0))


def build_neck_and_fretboard(coll):
    # Neck wood: heel to nut, its back flush with the body top (z=0 down).
    neck = _make_tapered_box("NeckMesh", HEEL_Y, NUT_Y,
                             _neck_width(HEEL_Y), NUT_W,
                             0.0, NECK_THICK)
    _add(coll, "Guitar_Neck", neck)

    # Fretboard: overhangs the body to FB_Y0 so the top frets are supported.
    fb = _make_tapered_box("FretboardMesh", FB_Y0, NUT_Y,
                           _neck_width(FB_Y0), NUT_W,
                           NECK_THICK, NECK_THICK + FB_THICK)
    _add(coll, "Guitar_Fretboard", fb)

    # Nut: a slim bone strip across the top of the neck.
    nut = _make_cube_mesh("NutMesh", NUT_W + 0.002, 0.005, FB_THICK + 0.002)
    _add(coll, "Guitar_Nut", nut,
         location=(0.0, NUT_Y + 0.0025, NECK_THICK + FB_THICK / 2.0))

    fb_top = NECK_THICK + FB_THICK
    for n in range(1, NUM_FRETS + 1):
        y = NUT_Y - SCALE_LEN * (1.0 - 2.0 ** (-n / 12.0))
        if y < FB_Y0:
            continue
        w = _neck_width(y) - 0.002
        fret = _make_cyl_mesh(f"Fret_{n}Mesh", 0.0009, w, segments=8)
        _add(coll, f"Guitar_Fret_{n}", fret,
             location=(0.0, y, fb_top + FRET_H / 2.0),
             rotation=(0.0, math.pi / 2.0, 0.0))   # lay the cylinder across the neck


def build_headstock(coll):
    y0, y1 = NUT_Y, NUT_Y + HEAD_LEN
    head = _make_tapered_box("HeadstockMesh", y0, y1, NUT_W + 0.006, HEAD_W,
                             -HEAD_DROP, -HEAD_DROP + NECK_THICK)
    _add(coll, "Guitar_Headstock", head)

    # Six inline tuner posts down the bass side, strings break to them.
    post_top_z = -HEAD_DROP + NECK_THICK + 0.012
    post_x = -0.024
    posts = []
    for i in range(NUM_STRINGS):
        py = y0 + 0.028 + i * (HEAD_LEN - 0.05) / (NUM_STRINGS - 1)
        top = (post_x, py, post_top_z)
        _add_cylinder_between(coll, f"Guitar_TunerPost_{i}",
                              (post_x, py, -HEAD_DROP + NECK_THICK - 0.003),
                              top, 0.004)
        # Button on the back of the headstock.
        _add(coll, f"Guitar_TunerButton_{i}",
             _make_cyl_mesh(f"TunerBtn_{i}Mesh", 0.006, 0.010),
             location=(post_x, py, -HEAD_DROP - 0.006))
        posts.append(top)
    return posts


def build_strings(coll, tuner_tops):
    """Six strings fanning from the bridge to the nut, then breaking down to the
    tuner posts. Low-E is string 0 on the bass (-x) side."""
    bridge_span, nut_span = 0.052, 0.035
    for i in range(NUM_STRINGS):
        f = (i - (NUM_STRINGS - 1) / 2.0) / (NUM_STRINGS - 1)
        r = STRING_R * (1.0 - 0.35 * i / (NUM_STRINGS - 1))  # bass strings thicker
        x_bridge = f * bridge_span
        x_nut = f * nut_span
        speaking_a = (x_bridge, BRIDGE_Y, STRING_Z)
        speaking_b = (x_nut, NUT_Y, STRING_Z)
        _add_cylinder_between(coll, f"Guitar_String_{i}", speaking_a, speaking_b, r, segments=8)
        # From the nut over the headstock to the post.
        _add_cylinder_between(coll, f"Guitar_StringHead_{i}",
                              speaking_b, tuner_tops[i], r, segments=8)


def build_hardware(coll):
    # Bridge plate at the body/bridge end, straddling the string anchors.
    bridge = _make_cube_mesh("BridgeMesh", 0.075, 0.05, 0.010)
    _bevel_mesh(bridge, 0.002)
    _add(coll, "Guitar_Bridge", bridge, location=(0.0, BRIDGE_Y + 0.01, 0.005))

    # Two humbucker pickups between the bridge and the neck, under the strings.
    for name, py in (("Bridge", 0.075), ("Neck", 0.205)):
        pu = _make_cube_mesh(f"Pickup{name}Mesh", 0.075, 0.038, 0.014)
        _bevel_mesh(pu, 0.003)
        _add(coll, f"Guitar_Pickup_{name}", pu, location=(0.0, py, 0.005))

    # Volume + tone knobs on the lower bass bout.
    for i, ky in enumerate((-0.02, -0.075)):
        knob = _make_cyl_mesh(f"Knob_{i}Mesh", 0.012, 0.016)
        _add(coll, f"Guitar_Knob_{i}", knob, location=(0.085, ky, 0.008))


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def setup_materials(coll):
    def make_mat(name, color, roughness, metallic=0.0, specular=0.5):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        spec_input = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
        if spec_input:
            spec_input.default_value = specular
        return mat

    body_mat = make_mat("GuitarBody", BODY_COLOR, 0.35, specular=0.4)
    neck_mat = make_mat("GuitarNeck", (0.60, 0.45, 0.27), 0.5, specular=0.3)
    fb_mat = make_mat("GuitarFretboard", (0.05, 0.035, 0.03), 0.55, specular=0.3)
    metal_mat = make_mat("GuitarMetal", (0.85, 0.85, 0.87), 0.32, metallic=1.0)
    plastic_mat = make_mat("GuitarPlastic", (0.015, 0.015, 0.017), 0.4, specular=0.3)
    bone_mat = make_mat("GuitarNut", (0.90, 0.88, 0.80), 0.4, specular=0.3)

    def pick(name):
        if name.startswith(("Guitar_Fret_", "Guitar_String", "Guitar_Bridge",
                            "Guitar_TunerPost", "Guitar_TunerButton")):
            return metal_mat
        if name.startswith(("Guitar_Pickup_", "Guitar_Knob_")):
            return plastic_mat
        if name == "Guitar_Nut":
            return bone_mat
        if name in ("Guitar_Neck", "Guitar_Headstock"):
            return neck_mat
        if name == "Guitar_Fretboard":
            return fb_mat
        return body_mat

    for obj in coll.objects:
        obj.data.materials.clear()
        obj.data.materials.append(pick(obj.name))


def setup_camera_and_lighting(scene):
    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
    cam.location = (0.85, -0.55, 1.05)
    look_at = mathutils.Vector((0.0, 0.30, 0.0))
    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 42
    scene.camera = cam

    key_light = bpy.data.objects.get("Light")
    if key_light is None:
        light_data = bpy.data.lights.new("Light", type='AREA')
        key_light = bpy.data.objects.new("Light", light_data)
        bpy.context.collection.objects.link(key_light)
    key_light.data.type = 'AREA'
    key_light.data.energy = 120
    key_light.data.size = 2.5
    key_light.location = (1.2, -0.6, 1.8)

    fill_light = bpy.data.objects.get("FillLight")
    if fill_light is None:
        fill_data = bpy.data.lights.new("FillLightData", type='AREA')
        fill_light = bpy.data.objects.new("FillLight", fill_data)
        bpy.context.collection.objects.link(fill_light)
    fill_light.data.energy = 40
    fill_light.data.size = 3.5
    fill_light.location = (-1.4, -0.4, 1.2)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.55, 0.55, 0.56, 1.0)
    bg.inputs[1].default_value = 1.0

    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800


def build_guitar():
    scene = bpy.context.scene
    clear_scene()

    coll = bpy.data.collections.new("Guitar")
    scene.collection.children.link(coll)

    build_body(coll)
    build_neck_and_fretboard(coll)
    tuner_tops = build_headstock(coll)
    build_strings(coll, tuner_tops)
    build_hardware(coll)
    setup_materials(coll)
    setup_camera_and_lighting(scene)

    total_len = (NUT_Y + HEAD_LEN) - BODY_Y0
    print(f"Guitar built: {NUM_FRETS} frets, {NUM_STRINGS} strings, "
          f"scale {SCALE_LEN*1000:.0f} mm, overall length {total_len*1000:.0f} mm")
    return coll


if __name__ == "__main__":
    build_guitar()
