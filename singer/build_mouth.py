"""Procedurally builds the 2D vector mouth in Blender.

Run inside Blender (Scripting tab, the Python console, or
``blender --background --python singer/build_mouth.py``) to rebuild the rig
from scratch.  Produces a "Mouth" collection holding:

    Mouth       the animated mouth itself - one flat mesh with three material
                groups (dark cavity, teeth, tongue) and ONE SHAPE KEY PER
                VISEME PER LOUDNESS, 15 x 3 = 45 in all, named
                V_<viseme>_<level> (V_AA_loud, V_MBP_soft, ...).  There are no
                lips - the mouth is a flat opening cut straight into the
                face, matching a flat graphic-novel style.
    FaceCard    an optional plain backdrop so the mouth reads as a face

This is deliberately a 2D prototype: the mouth is a flat card in the X-Z
plane facing -Y (the camera), authored in normalised units where 1.0 is the
neutral mouth width, and scaled by MOUTH_SCALE.  Everything that decides what
a mouth position looks like lives in the bpy-free mouth_shapes.py, so the
planner outside Blender and the mesh inside it can never disagree.

Why shape keys rather than a Bezier curve object: animating this mouth means
*interpolating between* mouth positions, and shape keys do exactly that -
Blender's result is the basis plus the weighted sum of the offsets, so two
keys crossfading with weights that sum to 1 give the exact linear blend of
the two shapes.  Curve control points cannot be blended nearly as cleanly.
"""

import bpy

# Shape parameters and outline geometry are the shared source of truth.
try:
    from . import mouth_shapes
except ImportError:  # loaded as a loose script via importlib, not a package
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import mouth_shapes

COLLECTION = "Mouth"
MOUTH_NAME = "Mouth"
FACE_NAME = "FaceCard"
MOUTH_SCALE = 1.0          # world size of a neutral mouth width

# Flat, graphic palette - this is a vector-art mouth, not a skin shader.  No
# lip colour: the mouth is just an opening in the face, outlined only by
# where the dark cavity meets the surrounding skin.
CAVITY_COLOR = (0.045, 0.012, 0.020)
TEETH_COLOR = (0.94, 0.94, 0.90)
TONGUE_COLOR = (0.80, 0.33, 0.38)
FACE_COLOR = (0.93, 0.78, 0.68)

EMISSION = 1.0             # the palette is emitted verbatim (see _material)
MATERIAL_ORDER = ("interior", "teeth", "tongue")
MATERIAL_COLOR = {"interior": CAVITY_COLOR,
                  "teeth": TEETH_COLOR, "tongue": TONGUE_COLOR}

FACE_RX, FACE_RZ = 1.30, 1.70      # backdrop ellipse radii, mouth widths
FACE_Y = 0.08                      # ... sat behind the mouth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _material(name, color, roughness=0.9):
    """Flat vector-art shading: the colour is purely EMITTED (base colour
    black), so what renders is exactly the palette above - no light rig to
    balance, no washing out.  This is flat 2D artwork, not a lit surface."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if "Specular IOR Level" in bsdf.inputs:      # Blender 4.x / 5.x
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
            bsdf.inputs["Emission Strength"].default_value = EMISSION
    mat.diffuse_color = (*color, 1.0)               # solid-shading fallback
    return mat


def clear_scene():
    """Remove mesh objects (keeps Light/Camera) so the 2D prototype scene is
    on its own, matching the other builders in this repo."""
    for obj in list(bpy.data.objects):
        if obj.type in ('MESH', 'ARMATURE', 'EMPTY', 'CURVE'):
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _collection():
    """A fresh, empty "Mouth" collection (replacing any previous build)."""
    old = bpy.data.collections.get(COLLECTION)
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(COLLECTION)
    bpy.context.scene.collection.children.link(coll)
    return coll


def _scaled(shape):
    s = MOUTH_SCALE
    return [(x * s, y * s, z * s) for x, y, z in mouth_shapes.mesh_vertices(shape)]


# ---------------------------------------------------------------------------
# The mouth
# ---------------------------------------------------------------------------
def build_mouth_object(coll):
    """The mouth mesh, with one shape key per viseme per loudness variant."""
    basis_shape = mouth_shapes.shape_for("SIL", "medium")
    verts = _scaled(basis_shape)

    faces, material_index = [], []
    for slot, group in enumerate(MATERIAL_ORDER):
        for face in mouth_shapes.FACES[group]:
            faces.append(list(face))
            material_index.append(slot)

    mesh = bpy.data.meshes.new(MOUTH_NAME)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for polygon, slot in zip(mesh.polygons, material_index):
        polygon.material_index = slot
    for group in MATERIAL_ORDER:
        mesh.materials.append(_material(f"Mouth_{group.capitalize()}",
                                        MATERIAL_COLOR[group]))

    obj = bpy.data.objects.new(MOUTH_NAME, mesh)
    coll.objects.link(obj)

    # Basis first, then the 45 variants.  Every key is absolute (a full
    # mouth position), and the animator only ever crossfades between two.
    obj.shape_key_add(name="Basis", from_mix=False)
    for name, _viseme, _level, shape in mouth_shapes.all_variants():
        key = obj.shape_key_add(name=name, from_mix=False)
        key.slider_min, key.slider_max = 0.0, 1.0
        for index, point in enumerate(_scaled(shape)):
            key.data[index].co = point
    return obj


def build_face_card(coll):
    """A plain elliptical backdrop, so the mouth reads as part of a face."""
    import math
    segments = 64
    s = MOUTH_SCALE
    verts = [(FACE_RX * s * math.cos(2 * math.pi * i / segments),
              FACE_Y * s,
              FACE_RZ * s * math.sin(2 * math.pi * i / segments) - 0.15 * s)
             for i in range(segments)]
    mesh = bpy.data.meshes.new(FACE_NAME)
    mesh.from_pydata(verts, [], [list(range(segments))])
    mesh.update()
    mesh.materials.append(_material("Mouth_Face", FACE_COLOR, roughness=0.9))
    obj = bpy.data.objects.new(FACE_NAME, mesh)
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Camera + lighting -- straight on, orthographic: this is a 2D prototype.
# ---------------------------------------------------------------------------
def setup_camera_and_lighting(scene):
    import math
    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.4 * MOUTH_SCALE       # close on the mouth
    cam.location = (0.0, -3.0 * MOUTH_SCALE, 0.0)
    cam.rotation_euler = (math.pi / 2.0, 0.0, 0.0)      # look along +Y
    scene.camera = cam

    key = bpy.data.objects.get("Light")
    if key is None:
        key = bpy.data.objects.new("Light", bpy.data.lights.new("Light",
                                                                type='AREA'))
        scene.collection.objects.link(key)
    key.data.type = 'AREA'
    key.data.energy = 20 * MOUTH_SCALE ** 2
    key.data.size = 4.0 * MOUTH_SCALE
    key.location = (0.6 * MOUTH_SCALE, -2.2 * MOUTH_SCALE, 1.2 * MOUTH_SCALE)
    key.rotation_euler = (math.radians(65), 0.0, math.radians(20))
    # Flat vector art: the mouth's parts are stacked a few millimetres apart
    # in Y purely for draw order, and a cast shadow turns that offset into a
    # visible ghost outline on the face card.
    key.data.use_shadow = False

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs[0].default_value = (0.16, 0.17, 0.20, 1.0)
        background.inputs[1].default_value = 1.0

    scene.view_settings.view_transform = 'Standard'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_mouth(face=True, camera=True, clear=True):
    """Build (or rebuild) the mouth rig.  Returns the Mouth object."""
    if clear:
        clear_scene()
    coll = _collection()
    mouth = build_mouth_object(coll)
    if face:
        build_face_card(coll)
    if camera:
        setup_camera_and_lighting(bpy.context.scene)
    return mouth


if __name__ == "__main__":
    obj = build_mouth()
    keys = obj.data.shape_keys.key_blocks
    print(f"Built {obj.name}: {len(obj.data.vertices)} verts, "
          f"{len(obj.data.polygons)} faces, {len(keys) - 1} viseme shape keys")
