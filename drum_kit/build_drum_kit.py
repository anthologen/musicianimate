"""Procedurally builds a minimalist, realistically sized 5-piece drum kit in Blender.

Run this inside Blender (Scripting tab / Text Editor "Run Script", the Python
console, or ``blender --background --python build_drum_kit.py``) to reconstruct
the scene from scratch.

All dimensions are in metres, taken from a standard "fusion/rock" 5-piece kit:

    Kick       22" x 18"   (0.559 x 0.457 m)   floor, horizontal axis
    Snare      14" x 5.5"  (0.356 x 0.140 m)   on a stand, between the knees
    Rack tom 1 12" x 8"    (0.305 x 0.203 m)   mounted on the kick, tilted
    Rack tom 2 13" x 9"    (0.330 x 0.229 m)   mounted on the kick, tilted
    Floor tom  16" x 16"   (0.406 x 0.406 m)   on its own three legs
    Hi-hats    14" pair                        on a pedal stand, drummer's left
    Crash      16"                             boom-less stand, high left
    Ride       20"                             boom-less stand, over floor tom

The look is deliberately minimalist: plain shells, two heads and a hoop per
drum, shallow-cone cymbals with a small bell, and thin chrome tube stands. No
lugs, tension rods or logos.

Coordinate convention (metres, +Z up, floor at z = 0):
    The drummer sits at +Y looking toward -Y (the audience). So the drummer's
    LEFT hand points +X and their RIGHT points -X. The kick lies on its side
    with its axis along Y; the batter head faces the drummer at +Y and the
    pedal sits on the floor just +Y of it.

Everything that will later be ANIMATED is authored with a deliberate origin so
a single transform channel drives the motion:
    * Cymbals (Crash, Ride, HiHat_Top, HiHat_Bottom) are single meshes whose
      origin is the cymbal centre, so rotating them tilts/wobbles about the
      stand, and HiHat_Top can slide down in Z to close on HiHat_Bottom.
    * Kick_Beater has its origin at the pedal axle, so rotating it about X
      swings the beater into the batter head.
    * Kick_Footboard / HiHat_Footboard have their origin at the heel hinge, so
      rotating about X presses the pedal.
Produces a single "DrumKit" collection.
"""

import math

import bpy
import bmesh
import mathutils

# Kit dimensions and positions live in the bpy-free kit_layout module, the
# single source of truth shared with the (non-Blender) striking planner so
# strike coordinates land exactly on the meshes built here.
try:
    from . import kit_layout
except ImportError:  # loaded as a loose script via importlib, not a package
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import kit_layout

INCH = kit_layout.INCH
KICK_DIA, KICK_DEPTH = kit_layout.KICK_DIA, kit_layout.KICK_DEPTH
KICK_R = kit_layout.KICK_R
KICK_CENTER = kit_layout.KICK_CENTER          # rests on the floor: centre = radius
KICK_BATTER_Y = kit_layout.KICK_BATTER_Y      # batter head plane (drummer side, +Y)
TOMS = kit_layout.TOMS                         # (name, dia_in, depth_in, center, tilt_deg)
SNARE = kit_layout.SNARE
CRASH = kit_layout.CRASH                        # (name, dia_in, center, tilt_deg, base_xy)
RIDE = kit_layout.RIDE
HIHAT_DIA = kit_layout.HIHAT_DIA
HIHAT_CENTER = kit_layout.HIHAT_CENTER
HIHAT_BASE_XY = kit_layout.HIHAT_BASE_XY

# ---------------------------------------------------------------------------
# Palette -- minimalist: one muted shell colour, off-white heads, chrome
# hardware, bronze cymbals, matte-black pedals.
# ---------------------------------------------------------------------------
SHELL_COLOR = (0.045, 0.06, 0.10)   # deep muted midnight blue, semi-gloss
HEAD_COLOR = (0.90, 0.90, 0.87)
CHROME_COLOR = (0.85, 0.86, 0.88)
BRONZE_COLOR = (0.72, 0.55, 0.24)
BLACK_COLOR = (0.02, 0.02, 0.022)

TUBE_R = 0.011          # stand upright radius
LEG_R = 0.008           # stand leg radius
HOOP_H = 0.022          # rim band height
HEAD_T = 0.004          # drum head thickness


# ---------------------------------------------------------------------------
# Mesh helpers (bmesh, so it runs headless without bpy.ops surprises)
# ---------------------------------------------------------------------------
def _cyl_mesh(name, radius, depth, segments=48, cap_ends=True):
    """Cylinder centred at the origin, axis along +Z."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap_ends, cap_tris=False, segments=segments,
                          radius1=radius, radius2=radius, depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _cube_mesh(name, sx, sy, sz):
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


def _cymbal_mesh(name, edge_radius, segments=64):
    """A shallow inverted cone (wide low edge -> narrow high centre) plus a
    small domed bell, merged into one mesh centred on the origin. The result
    reads as a cymbal and rotates naturally about its centre."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bow_h = edge_radius * 0.12               # gentle taper from edge to bell
    # Bow: radius1 (bottom, outer edge) wide, radius2 (top, near bell) narrow.
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                          radius1=edge_radius, radius2=edge_radius * 0.22,
                          depth=bow_h)
    # Bell: a little dome sitting on top of the bow's inner rim.
    bell = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                                 radius1=edge_radius * 0.22, radius2=edge_radius * 0.05,
                                 depth=edge_radius * 0.10)
    bmesh.ops.translate(bm, verts=bell["verts"],
                        vec=(0.0, 0.0, bow_h / 2.0 + edge_radius * 0.05))
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def _add(coll, name, mesh, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    coll.objects.link(obj)
    return obj


def _cyl_between(coll, name, p1, p2, radius, segments=16):
    """Capped cylinder spanning two arbitrary points -- stand legs, rods."""
    p1, p2 = mathutils.Vector(p1), mathutils.Vector(p2)
    vec = p2 - p1
    mesh = _cyl_mesh(name + "Mesh", radius, vec.length, segments)
    obj = _add(coll, name, mesh, location=(p1 + vec / 2.0))
    obj.rotation_euler = vec.to_track_quat('Z', 'Y').to_euler()
    return obj


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------
def clear_scene():
    """Remove mesh objects (keeps Light/Camera) and any stale DrumKit collection."""
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if "DrumKit" in bpy.data.collections:
        old = bpy.data.collections["DrumKit"]
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)


# ---------------------------------------------------------------------------
# Drum builders. A drum is a shell + two heads + two hoops. `axis` is the
# world axis the drum's cylinder points along ('Z' vertical, 'Y' horizontal).
# All parts are parented to a small Empty at the drum centre so future
# animation can nudge the whole drum with one handle.
# ---------------------------------------------------------------------------
def _drum_axis_rotation(axis, tilt_deg):
    """Euler that points a +Z cylinder along `axis`, then leans it toward the
    drummer (+Y) by tilt_deg."""
    if axis == 'Z':
        base = mathutils.Euler((math.radians(tilt_deg), 0.0, 0.0))
    else:  # 'Y' -- lay the drum on its side, axis along Y
        base = mathutils.Euler((math.radians(90.0), 0.0, 0.0))
    return base


def build_drum(coll, name, dia_in, depth_in, center, axis='Z', tilt_deg=0.0):
    r = dia_in * INCH / 2.0
    depth = depth_in * INCH
    parent = bpy.data.objects.new(name, None)   # Empty group handle
    parent.empty_display_size = 0.05
    parent.location = center
    parent.rotation_euler = _drum_axis_rotation(axis, tilt_deg)
    coll.objects.link(parent)

    parts = []
    # Shell: open cylinder (heads cap the ends).
    shell = _cyl_mesh(f"{name}_ShellMesh", r, depth, cap_ends=False)
    parts.append(_add(coll, f"{name}_Shell", shell))
    # Two heads: thin discs just inside each end.
    for tag, z in (("Batter", depth / 2.0 - HEAD_T / 2.0),
                   ("Reso", -depth / 2.0 + HEAD_T / 2.0)):
        head = _cyl_mesh(f"{name}_Head{tag}Mesh", r * 0.995, HEAD_T)
        parts.append(_add(coll, f"{name}_Head_{tag}", head, location=(0, 0, z)))
    # Two hoops: chrome bands wrapping each rim.
    for tag, z in (("Top", depth / 2.0), ("Bot", -depth / 2.0)):
        hoop = _cyl_mesh(f"{name}_Hoop{tag}Mesh", r * 1.03, HOOP_H, cap_ends=False)
        parts.append(_add(coll, f"{name}_Hoop_{tag}", hoop, location=(0, 0, z)))

    for p in parts:
        p.parent = parent
        p.matrix_parent_inverse = parent.matrix_world.inverted()
    return parent


def build_floor_tom_legs(coll):
    """Three chrome legs so the floor tom stands on the floor by itself."""
    name, dia, depth_in, center, _tilt = TOMS[2]   # FloorTom
    r = dia * INCH / 2.0
    depth = depth_in * INCH                       # inches -> metres
    cx, cy, cz = center
    top_z = cz - depth * 0.25                     # legs clamp low on the shell
    for i in range(3):
        a = math.radians(30 + i * 120)
        top = (cx + (r - 0.01) * math.cos(a), cy + (r - 0.01) * math.sin(a), top_z)
        foot = (cx + (r + 0.06) * math.cos(a), cy + (r + 0.06) * math.sin(a), 0.0)
        _cyl_between(coll, f"FloorTom_Leg_{i}", top, foot, LEG_R)


def build_kick(coll):
    parent = build_drum(coll, "Kick", 22, 18, KICK_CENTER, axis='Y')
    # Front spurs: two short chrome legs angling forward off the audience side.
    for sx in (-0.20, 0.20):
        _cyl_between(coll, f"Kick_Spur_{'L' if sx > 0 else 'R'}",
                     (sx, -KICK_R * 0.6, KICK_R * 0.25),
                     (sx * 1.15, -KICK_R - 0.06, 0.0), LEG_R)
    return parent


# ---------------------------------------------------------------------------
# Cymbal builder -- single mesh, origin at centre, sitting atop a tube stand.
# ---------------------------------------------------------------------------
def build_cymbal(coll, name, dia_in, center, tilt_deg, base_xy):
    r = dia_in * INCH / 2.0
    cym = _add(coll, name, _cymbal_mesh(f"{name}Mesh", r), location=center,
               rotation=(math.radians(tilt_deg), 0.0, 0.0))   # tilt toward drummer, kept
    build_boom_stand(coll, f"{name}_Stand", base_xy, center)
    return cym


def build_boom_stand(coll, name, base_xy, cymbal_center):
    """A boom stand: a tripod post that stands back/outboard, with an angled
    boom arm reaching over to the cymbal (brought in toward the drummer) and a
    short counterweight stub on the near side of the boom clamp."""
    bx, by = base_xy
    cx, cy, cz = cymbal_center
    post_top_z = cz - 0.22                       # clamp sits below the cymbal
    # Vertical post + tripod legs.
    build_tube_stand(coll, name, base_xy, post_top_z)
    clamp = mathutils.Vector((bx, by, post_top_z))
    tip = mathutils.Vector((cx, cy, cz - 0.008))  # just under the bell
    # Boom arm from the clamp out to the cymbal.
    _cyl_between(coll, f"{name}_Boom", tuple(clamp), tuple(tip), 0.010)
    # Counterweight: short stub opposite the cymbal + a small weight.
    back = clamp - (tip - clamp).normalized() * 0.12
    _cyl_between(coll, f"{name}_BoomCW", tuple(clamp), tuple(back), 0.010)
    _add(coll, f"{name}_BoomWeight", _cyl_mesh(f"{name}_BoomWeightMesh", 0.022, 0.05),
         location=tuple(back), rotation=(math.radians(90), 0, 0))


def build_tube_stand(coll, name, base_xy, top_z, spread=0.24, leg_yaw=0.0):
    """A minimalist chrome stand: one upright tube + three splayed legs.
    ``leg_yaw`` (deg) rotates the tripod about the post so, e.g., the hi-hat's
    pedal passes through the GAP between two legs instead of hitting one."""
    bx, by = base_xy
    _cyl_between(coll, f"{name}_Post", (bx, by, 0.02), (bx, by, top_z), TUBE_R)
    for i in range(3):
        a = math.radians(90 + leg_yaw + i * 120)
        _cyl_between(coll, f"{name}_Leg_{i}", (bx, by, 0.10),
                     (bx + spread * math.cos(a), by + spread * math.sin(a), 0.0), LEG_R)


# ---------------------------------------------------------------------------
# Hi-hats: two cymbals (bottom fixed, top animatable) on a pedal stand.
# ---------------------------------------------------------------------------
def build_hihat(coll):
    r = HIHAT_DIA * INCH / 2.0
    cx, cy, cz = HIHAT_CENTER
    # Bottom cymbal, upside down (bell pointing down), essentially flat.
    bottom = _add(coll, "HiHat_Bottom", _cymbal_mesh("HiHat_BottomMesh", r),
                  location=(cx, cy, cz), rotation=(math.radians(180), 0, 0))
    # Top cymbal a small gap above -- slides down in Z to "close".
    top = _add(coll, "HiHat_Top", _cymbal_mesh("HiHat_TopMesh", r),
               location=(cx, cy, cz + 0.045))
    # Rotate the tripod so a GAP between two legs faces the pedal (which runs
    # back toward the drummer at an angle), not a leg.
    build_tube_stand(coll, "HiHat_Stand", HIHAT_BASE_XY, cz - 0.02, leg_yaw=90.0)
    # Pull rod up the centre; the top cymbal clutches onto it.
    _cyl_between(coll, "HiHat_PullRod", (cx, cy, 0.05), (cx, cy, cz + 0.06), 0.004)
    # Pedal toe end at the stand centre (cx, cy) -- the pull rod / linkage point.
    build_pedal(coll, "HiHat", (cx, cy), yaw=kit_layout.HIHAT_PEDAL_YAW)
    return top, bottom


# ---------------------------------------------------------------------------
# Pedals -- footboard hinged at the heel + (for the kick) a swinging beater.
# ---------------------------------------------------------------------------
PEDAL_SLOPE = math.radians(13.0)   # footboards rest sloped up toward the toe


def build_pedal(coll, name, base_xy, yaw=0.0):
    """A footboard hinged at its heel. ``base_xy`` is the TOE end, which stays
    fixed; ``yaw`` swings the heel about that toe (so the pedal can point back
    toward the drummer while its toe stays aligned with, e.g., the hi-hat stand).
    A rotation about X presses the toe down; the board also rests sloped up
    toward the toe (PEDAL_SLOPE)."""
    bx, by = base_xy            # toe end (kept fixed under the yaw)
    s, c = math.sin(yaw), math.cos(yaw)
    heel_x = bx - 0.24 * s      # heel is 0.24 back from the toe, swung by the yaw
    heel_y = by + 0.24 * c
    board = _cube_mesh(f"{name}_FootboardMesh", 0.075, 0.24, 0.010)
    board_obj = _add(coll, f"{name}_Footboard", board, location=(heel_x, heel_y, 0.018))
    # Shift geometry so the origin is the heel hinge (its +Y edge).
    for v in board.vertices:
        v.co.y -= 0.12
    board.update()
    # Real pedals slope UP from the heel (on the floor) to the toe. A negative
    # X tilt raises the toe end; the animator's positive X press then presses it
    # down from that resting slope.
    board_obj.rotation_euler = (-PEDAL_SLOPE, 0.0, yaw)
    # Heel plate on the floor, angled to match.
    heel = _add(coll, f"{name}_HeelPlate", _cube_mesh(f"{name}_HeelMesh", 0.09, 0.05, 0.012),
                location=(heel_x, heel_y + 0.01, 0.006))
    heel.rotation_euler = (0.0, 0.0, yaw)
    return board_obj


def _beater_mesh(name, rod_len):
    """Beater as ONE mesh in local space: a rod up +Z from the origin (the
    axle) with a felt puck on top whose axis points +Y (the striking face)."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rod = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=16,
                                radius1=0.006, radius2=0.006, depth=rod_len)
    bmesh.ops.translate(bm, verts=rod["verts"], vec=(0, 0, rod_len / 2.0))
    puck = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=20,
                                 radius1=0.022, radius2=0.022, depth=0.03)
    bmesh.ops.rotate(bm, verts=puck["verts"],
                     matrix=mathutils.Matrix.Rotation(math.radians(90), 3, 'X'))
    bmesh.ops.translate(bm, verts=puck["verts"], vec=(0, 0, rod_len))
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def build_kick_pedal(coll):
    bx = 0.0
    base_y = KICK_BATTER_Y + 0.02
    build_pedal(coll, "Kick", (bx, base_y))
    # Axle: the beater's pivot, low and just in front of the batter head.
    axle_z = 0.055
    axle_y = KICK_BATTER_Y + 0.02
    # Rod reaches from the axle up to the batter-head centre (z = KICK_R),
    # minus the puck radius so the felt just kisses the head at rest lean.
    rod_len = (KICK_R - axle_z) - 0.02
    # Rest lean: cocked BACK toward the drummer (+Y) so the puck sits just
    # outside the batter head; a +X rotation then swings it forward (-Y) to
    # strike. (A +tilt would push the beater INTO the drum -- hence negative.)
    beater = _add(coll, "Kick_Beater", _beater_mesh("Kick_BeaterMesh", rod_len),
                  location=(bx, axle_y, axle_z),
                  rotation=(math.radians(-10.0), 0.0, 0.0))
    # Upright posts flanking the axle.
    for sx, tag in ((-0.05, "L"), (0.05, "R")):
        _cyl_between(coll, f"Kick_PedalPost_{tag}",
                     (bx + sx, axle_y, 0.0), (bx + sx, axle_y, axle_z + 0.01), 0.008)
    return beater


# ---------------------------------------------------------------------------
# Minimalist tom mount: a single post standing on the kick top, between the two
# rack toms, with a short arm reaching out to each tom's inner SHELL at mid
# height. The arms meet the shells from the side (never the drumheads), and the
# post sits between the toms. Positions derive from the TOMS list.
# ---------------------------------------------------------------------------
def build_tom_mount(coll):
    tom1 = TOMS[0][3]   # (x, y, z) centres
    tom2 = TOMS[1][3]
    r1, r2 = 12 * INCH / 2, 13 * INCH / 2
    post_x, post_y = 0.0, -0.05                 # between the toms
    post_top = max(tom1[2], tom2[2]) + 0.03
    # Post rises from just inside the kick top up between the toms.
    _cyl_between(coll, "TomMount_Post", (post_x, post_y, KICK_R - 0.02),
                 (post_x, post_y, post_top), 0.012)
    # A short arm from the post into each tom's inner shell (at the tom's
    # centre height, so it hits the shell wall, not the membrane).
    for name, tom, r in (("Tom1", tom1, r1), ("Tom2", tom2, r2)):
        tx, ty, tz = tom
        inner_x = tx - math.copysign(r - 0.01, tx)   # shell edge facing centre
        _cyl_between(coll, f"TomMount_Arm_{name}",
                     (post_x, post_y, tz), (inner_x, ty, tz), 0.009)


# ---------------------------------------------------------------------------
# Snare stand.
# ---------------------------------------------------------------------------
def build_snare_stand(coll):
    _name, _d, _dep, center, _t = SNARE
    build_tube_stand(coll, "Snare_Stand", (center[0], center[1]), center[2] - 0.09, spread=0.22)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def _make_mat(name, color, roughness, metallic=0.0, specular=0.5):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    spec = bsdf.inputs.get("Specular IOR Level") or bsdf.inputs.get("Specular")
    if spec:
        spec.default_value = specular
    return mat


def setup_materials(coll):
    shell_mat = _make_mat("Drum_Shell", SHELL_COLOR, 0.28, specular=0.5)
    head_mat = _make_mat("Drum_Head", HEAD_COLOR, 0.55, specular=0.3)
    chrome_mat = _make_mat("Drum_Chrome", CHROME_COLOR, 0.22, metallic=1.0)
    bronze_mat = _make_mat("Drum_Bronze", BRONZE_COLOR, 0.34, metallic=1.0)
    black_mat = _make_mat("Drum_Black", BLACK_COLOR, 0.42, specular=0.3)

    def pick(name):
        if "_Head_" in name:
            return head_mat
        if "_Shell" in name:
            return shell_mat
        if name in ("Crash", "Ride", "HiHat_Top", "HiHat_Bottom"):
            return bronze_mat
        if ("Footboard" in name or "Heel" in name or "Beater" in name
                or "Pedal" in name):
            return black_mat
        return chrome_mat   # hoops, stands, spurs, mounts, rods, posts

    for obj in coll.objects:
        if obj.type != 'MESH':
            continue
        obj.data.materials.clear()
        obj.data.materials.append(pick(obj.name))


# ---------------------------------------------------------------------------
# Camera + lighting -- a 3/4 view from the audience side, framing the whole kit.
# ---------------------------------------------------------------------------
def setup_camera_and_lighting(scene):
    cam_data = bpy.data.cameras.get("Camera") or bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam)
    cam.location = (1.75, -2.35, 1.35)
    look_at = mathutils.Vector((-0.02, -0.02, 0.62))
    cam.rotation_euler = (look_at - mathutils.Vector(cam.location)).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 40
    scene.camera = cam

    key = bpy.data.objects.get("Light")
    if key is None:
        key = bpy.data.objects.new("Light", bpy.data.lights.new("Light", type='AREA'))
        bpy.context.collection.objects.link(key)
    key.data.type = 'AREA'
    key.data.energy = 400
    key.data.size = 3.0
    key.location = (2.0, -1.6, 2.6)

    fill = bpy.data.objects.get("FillLight")
    if fill is None:
        fill = bpy.data.objects.new("FillLight", bpy.data.lights.new("FillLightData", type='AREA'))
        bpy.context.collection.objects.link(fill)
    fill.data.energy = 120
    fill.data.size = 4.0
    fill.location = (-2.2, -1.2, 1.8)

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


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------
def build_drum_kit():
    scene = bpy.context.scene
    clear_scene()

    coll = bpy.data.collections.new("DrumKit")
    scene.collection.children.link(coll)

    build_kick(coll)
    build_kick_pedal(coll)
    build_tom_mount(coll)
    for name, dia, depth, center, tilt in TOMS:
        build_drum(coll, name, dia, depth, center, axis='Z', tilt_deg=tilt)
    sname, sdia, sdepth, scenter, stilt = SNARE
    build_drum(coll, sname, sdia, sdepth, scenter, axis='Z', tilt_deg=stilt)
    build_snare_stand(coll)
    build_floor_tom_legs(coll)

    build_cymbal(coll, *CRASH)
    build_cymbal(coll, *RIDE)
    build_hihat(coll)

    setup_materials(coll)
    setup_camera_and_lighting(scene)

    print(f"Drum kit built: kick + snare + {len(TOMS)} toms, "
          f"hi-hat + crash + ride; {len([o for o in coll.objects])} objects")
    return coll


if __name__ == "__main__":
    build_drum_kit()
