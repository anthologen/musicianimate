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
    static two-segment thumb reaches under the neck to press its back -
    opposite the fingers, like a real fretting grip. Every phalanx is
    caged to its human range of motion (FINGER_ROT_LIMIT), so reach is
    the WRIST's job rather than an unanatomical knuckle splay.
  - ``PickHand``: a loose fist holding a pick, a single wrist bone
    carrying rigid palm/finger/thumb boxes and a flat pick whose tip sits
    at PICK_TIP_LOCAL in armature space. It is oriented (PICK_PITCH
    /PICK_YAW) so the fingers run *parallel to the strings* (along the
    neck axis) and curl into the palm, the thumb crosses over the index
    to pinch the pick against the index fingertip, and the pick protrudes
    down toward the strings. The animator drives only the object
    location, sweeping the pick tip across the strings.

Both hands are the same realistic size, and every finger phalanx takes
its cross-section from adult hand anthropometry (FINGER_CROSS), so the
two read as a matched pair rather than as thin sticks.

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


# The pick hand is a loose fist built in a canonical local frame (+y = the
# finger axis, +z = the back of the hand, -z = the PALM, where the fingers curl
# and the pick is pinched, +x = the knuckle spread / thumb side), then rotated
# into playing pose by PICK_ROT.
#
# MOUNT-COUPLED: PICK_ROT is SOLVED against the standing guitarist's picking
# forearm, not chosen by eye, because it is the only thing that sets how bent the
# picking WRIST reads -- the hand rig is a separate armature whose orientation is
# authored, so no IK limit constrains it. It was previously a hand-picked
# forward-down pitch with the fingers running ALONG the strings; measured on the
# mounted player, the forearm arrives across the body at ~60 deg to that finger
# axis, i.e. the wrist sat 51-83 deg bent through the whole take (past the
# _check_wrist_pose guard in animate_guitarist, and reading as a snapped wrist).
#
# The solve (mirroring bass_guitar's): sample the picking forearm direction over
# the take, expressed in the GUITAR's own frame; build the hand basis whose
# columns are [across, finger, back] with
#   * finger (+y) = that mean forearm direction, so the hand CONTINUES the
#     forearm line instead of folding off it, and
#   * back (+z) = the guitar's outward face normal, orthogonalized against the
#     finger axis, so the PALM (-z) turns onto the strings -- the pick then hangs
#     off the palm and still reaches down to them.
# Iterated to a fixed point (moving the hand moves the arm's IK target, which
# moves the forearm). Result: wrist bend 51-83 deg (mean 60) -> 0.8-17.4 deg
# (mean 5), palm-into-strings >= 0.95, and the blade still points into the guitar
# face. Re-run scratch solver `solve_pick.py` if ANCHOR_WORLD / NECK_ELEV /
# ELBOW_POLE_OVERRIDE["R"] change much.
#
# RE-SOLVED 2026-08-11 for the corrected RIGHT-hand fist (see PICK_THUMB_AXIS).
# Mirroring the fist to the correct chirality moves the PINCH -- and with it the
# pick tip the animator pins to the strings -- to the other side of the wrist, so
# the wrist bone (the arm's IK target) landed ~4 cm lower and the forearm arrived
# at a new angle: the wrist bend doubled (5.9 -> 12.8 deg mean, 33 deg peak) even
# though nothing about the stroke itself had changed. Re-running the fixed point
# (together with the +35 mm wear height it forced, see animate_guitarist's
# ANCHOR_WORLD) brought it back to 0.5-18.0 deg, mean 5.0 -- at or better than the
# mirrored hand's 1.3-16.3/5.9. Because the solve only sets the FINGER axis (yaw
# and pitch together fully determine a direction; there is no roll freedom left in
# this parametrization), the palm stays turned onto the strings at 0.95 and the
# thumb keeps pointing up (world +Z component 0.94).
PICK_PITCH = -0.3103   # tilt of the hand about its knuckle axis (solved)
PICK_YAW = -0.9700     # turn of the hand across the strings (solved)

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
#
# The knuckle span (index->pinky ~74 mm) and palm are those of a realistic
# adult hand -- the SAME hand size as the picking hand -- so the two hands
# read as a matched pair. Frets wider than the knuckle pitch are covered NOT
# by a stretched palm but by finger SPLAY (the IK's z-rotation / knuckle
# abduction) within its anatomical cap, and above all by moving the WRIST
# (see FINGER_ROT_LIMIT and animate_hands' FINGER_MCP_SPLAY / wrist search).
# ---------------------------------------------------------------------------

FRET_FINGERS = {
    #        knuckle (x, y, z)             (prox, mid, dist) segment lengths
    1: {"knuckle": (0.037, 0.050, 0.0), "lengths": (0.044, 0.027, 0.021)},
    2: {"knuckle": (0.013, 0.055, 0.0), "lengths": (0.048, 0.031, 0.023)},
    3: {"knuckle": (-0.013, 0.053, 0.0), "lengths": (0.045, 0.029, 0.022)},
    4: {"knuckle": (-0.037, 0.046, 0.0), "lengths": (0.037, 0.025, 0.020)},
}

# --- Finger cross-section (metres), from hand anthropometry ------------------
# Each phalanx box approximates an average adult finger: broadest at the
# proximal phalanx and TAPERING to the tip, with the little finger the
# slimmest and the index/middle the widest. Breadths (side-to-side) follow
# measured digit-breadth norms -- ~18 mm proximal for the index/middle down to
# ~15 mm for the little finger, tapering to ~11-14 mm at the fingertip (Garrett
# 1971 hand anthropometry; ANSUR). Depth (dorsopalmar) runs ~1 mm under the
# breadth, a finger being a touch wider than it is thick. This replaced a flat
# 11x12 mm box on every phalanx, which -- being far thinner than a real finger
# -- left oversized gaps between the fingers. The thumb keeps its own (larger)
# width where it is built, so it stays the thickest digit.
#                    (width, height) for prox / mid / dist phalanx
FINGER_CROSS = {
    "index":  {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "middle": {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "ring":   {"prox": (0.017, 0.016), "mid": (0.015, 0.014), "dist": (0.013, 0.012)},
    "pinky":  {"prox": (0.015, 0.014), "mid": (0.013, 0.012), "dist": (0.011, 0.011)},
}

# Which anthropometric profile each finger key -- across the fret (1..4) and
# pick-fist (index..pinky) rigs -- draws its section from.
FINGER_PROFILE = {
    1: "index", 2: "middle", 3: "ring", 4: "pinky",   # fret fingers
    "index": "index", "middle": "middle",             # pick fist
    "ring": "ring", "pinky": "pinky",
}


def _finger_cross(key, seg):
    """(width, height) of finger `key`'s `seg` phalanx box, from anthropometry."""
    return FINGER_CROSS[FINGER_PROFILE[key]][seg]


# Both palms are the same realistic size so the two hands match; the fret palm
# just covers the (now realistic) knuckle span rather than a stretched one.
FRET_PALM_SIZE = (0.086, 0.072, 0.024)
PICK_PALM_SIZE = (0.070, 0.066, 0.032)

# --- Finger joint range-of-motion limits (human norms; degrees) --------------
# The finger joints are caged the same way build_guitarist cages the ELBOW and
# KNEE: real hinges that flex freely but must never bend BACKWARD past their
# small natural extension, and (for the inter-phalangeal joints) neither splay
# sideways nor twist. They are LOCAL LIMIT_ROTATION constraints applied at BUILD
# time, so they only ever GUARD future hand poses -- the shipped fret
# performance already lives inside them (its curls are baked keyframes within
# these bounds).
#
# In this rig's finger frame (bone runs +y): X = curl (flexion is NEGATIVE x),
# Z = knuckle splay/abduction, Y = axial twist. Bounds follow measured
# active-ROM norms for the index-through-little fingers (Thieme 2024 normative
# study; AAOS / goniometry references):
#   * MCP flexion ~85-90, extension (hyperextension) ~25-30 (the little finger
#     the most, mean ~26 and up toward ~38 at +2SD), abduction/adduction ~+/-25.
#   * PIP flexion ~95-110, ~0 hyperextension, no splay/twist.
#   * DIP flexion ~80-85, ~0-10 hyperextension, no splay/twist.
# The splay bound (+/-30 = the ~25 deg norm plus a little headroom over the
# 26 deg IK cap in animate_hands) is what stops a pressing finger deviating far
# enough sideways to sweep UNDER its neighbours; reach is the wrist's job. The
# one joint kept generous is the MID (PIP): this rig's closed-form IK lumps the
# distal phalanx INTO the middle link (see animate_hands _finger_ik), so
# f<n>_mid.x carries the COMBINED PIP+DIP fold (~190 deg in a real hand) while
# f<n>_dist.x barely moves -- the stylized fret curl drives it to ~140, within
# that combined envelope even though it is past the PIP-alone norm. Clamping it
# would unseat the press.
FINGER_ROT_LIMIT = {
    "prox": {"x": (-100.0, 35.0), "y": (-8.0, 8.0), "z": (-30.0, 30.0)},  # MCP
    "mid":  {"x": (-142.0, 5.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # PIP(+lumped DIP)
    "dist": {"x": (-95.0, 10.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # DIP: flex only
}


def _limit_rot(pbone, limit):
    """Add a LOCAL-space LIMIT_ROTATION constraint (degrees) to a hand bone --
    the same soft cage build_guitarist puts on the body's FK joints."""
    lr = pbone.constraints.new('LIMIT_ROTATION')
    lr.owner_space = 'LOCAL'
    for axis in ("x", "y", "z"):
        setattr(lr, f"use_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(lr, f"min_{axis}", math.radians(lo))
        setattr(lr, f"max_{axis}", math.radians(hi))


# CHIRALITY: the picking fist is a RIGHT hand -- fingers +y, palm -z, and
# therefore the THUMB on the NEGATIVE local x side (thumb = fingers x palm_out =
# y x -z = -x). Get that sign backwards and the fist is a mirrored LEFT hand:
# nothing about the *strike* breaks (the tip is re-placed each stroke, the palm
# still faces the strings and the wrist still reads straight), so it slips past
# every guard -- it just plays with the thumb underneath. Mounted, local +x
# points almost straight DOWN (world ~(-0.29, 0, -0.96)), so a +x thumb hangs at
# the floor and the pinky rides on top: the "upside-down hand" bug. Matches
# bass_guitar/build_hands.py, whose picking fist is built the same way.
PICK_THUMB_AXIS = (-1.0, 0.0, 0.0)
# The pick is pinched between the THUMB PAD and the tip of the curled INDEX
# (see PICK_FIST): it sits low on the palm rather than out at arm's length past
# the fingertips, and protrudes DOWNWARD (-z) toward the strings, so the fist
# reads as a hand holding a pick instead of a mitt with a spike. Moving the
# pinch does NOT disturb the strike -- animate_hands re-places the whole hand
# each stroke so the pick TIP still lands on the string plane -- as long as the
# blade VECTOR (PICK_TIP_LOCAL - PICK_PINCH) keeps pointing along local -z.
# Blade protruding past the pinch. Because animate_hands lands the pick TIP on
# the strings, the length also sets how high the FIST rides: at 22 mm (about what
# shows past a real pick grip) the closest curled fingertip clears the string
# plane by ~11 mm with the tip on it -- close enough that the hand reads as
# playing the strings rather than hovering, with room for the strum dip.
PICK_LENGTH = 0.022
PICK_PINCH = (-0.028, 0.010, -0.035)
PICK_TIP_LOCAL = (PICK_PINCH[0], PICK_PINCH[1], PICK_PINCH[2] - PICK_LENGTH)

# Where idle hands hover before the animator takes over.
REST_LOCATION = {
    "FretHand": (0.085, fret_layout.fret_y(2), 0.026),
    "PickHand": (0.045, fret_layout.PLUCK_Y + 0.045, 0.115),
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


def _build_finger_chains(arm_obj, fingers):
    """Add wrist + finger chains (piano convention) to an armature in edit
    mode, one chain per entry of `fingers`, each phalanx caged to its human
    range of motion."""
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)
    for name, spec in fingers.items():
        kx, ky, kz = spec["knuckle"]
        y = ky
        parent = wrist
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            bone = eb.new(f"f{name}_{seg}")
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
    # Cage each phalanx to its human range of motion (one-way flexion hinges).
    for name in fingers:
        for seg in ("prox", "mid", "dist"):
            _limit_rot(arm_obj.pose.bones[f"f{name}_{seg}"], FINGER_ROT_LIMIT[seg])


def _add_finger_boxes(arm_obj, coll, mat, fingers):
    for name, spec in fingers.items():
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            w, h = _finger_cross(name, seg)
            _bone_box(arm_obj, coll, mat, f"f{name}_{seg}",
                      (w, length * 0.92, h),
                      (0.0, -length / 2.0, 0.0))


def build_fret_hand(coll, mat):
    """The articulated fretting hand: four IK-driven finger chains
    wrapping the neck from the treble side, thumb under the neck."""
    arm_obj = _new_armature("FretHand", coll, tilt=WRAP_TILT)
    _build_finger_chains(arm_obj, FRET_FINGERS)
    _add_finger_boxes(arm_obj, coll, mat, FRET_FINGERS)
    _bone_box(arm_obj, coll, mat, "wrist", FRET_PALM_SIZE, (0.0, -0.013, 0.0))

    # Static thumb reaching under the neck to press its back, opposite the
    # fingers. Specified as a *world* offset from the wrist (the animator keeps
    # the wrist near x 0.07-0.09, z ~0.02, so the thumb tip lands under the
    # neck's centreline around z just below 0) and converted into the tilted
    # armature's local frame. It sits on the INDEX/radial edge of the palm
    # (world +y here maps to armature-local +x, the index side -- cf.
    # FRET_FINGERS, index knuckle x=+0.037), where a real hand's thumb attaches,
    # rather than bisecting the back of the hand.
    # It is built as TWO segments meeting at a knuckle (the thumb's IP joint) so
    # it reads as a jointed thumb rather than one rigid stick: a longer proximal
    # segment leaving the palm, then a shorter distal segment that flexes at the
    # joint to lie flatter against the neck (as a thumb pad pressing the back
    # does), tapering toward the tip.
    thumb_center_w = (-0.058, 0.040, -0.020)
    thumb_dir_w = mathutils.Vector((-0.94, 0.12, -0.32)).normalized()
    c_local = mathutils.Vector(_fret_local_offset(thumb_center_w))
    d_local = mathutils.Vector(_fret_local_offset(thumb_dir_w)).normalized()
    thumb_len, prox_len, dist_len = 0.076, 0.046, 0.030
    base = c_local - d_local * (thumb_len / 2.0)   # meets the palm
    knuckle = base + d_local * prox_len            # IP joint
    # Distal phalanx flexes ~18 deg at the knuckle, curling the tip to lie
    # flatter along the neck's back.
    dist_dir = (mathutils.Matrix.Rotation(math.radians(18), 4, 'X')
                @ d_local).normalized()
    tip = knuckle + dist_dir * dist_len
    _seg_box(arm_obj, coll, mat, 0.019, 0.016, base, knuckle)
    _seg_box(arm_obj, coll, mat, 0.017, 0.015, knuckle, tip)
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


def _curled_finger(knuckle, lengths, flex_deg):
    """Points knuckle -> PIP -> DIP -> tip for a finger curling toward the
    palm (local -z). Each phalanx adds its flexion about the knuckle-parallel
    local x axis (curl is negative x in this rig's finger frame, cf.
    FINGER_ROT_LIMIT), and the angle is CUMULATIVE, so a big enough total folds
    the fingertip back into the fist."""
    pts = [mathutils.Vector(knuckle)]
    cum = 0.0
    for length, flex in zip(lengths, flex_deg):
        cum += flex
        d = (mathutils.Matrix.Rotation(math.radians(-cum), 3, 'X')
             @ mathutils.Vector((0.0, 1.0, 0.0)))
        pts.append(pts[-1] + d * length)
    return pts


# The picking fist's four fingers are FULL three-phalanx fingers -- the SAME
# phalanx lengths as the fretting hand (FRET_FINGERS), so the two hands read as
# a matched pair -- curled into a relaxed fist rather than left as two-segment
# stubs (which made the right hand read as fingerless). The thumb and the INDEX
# pinch the pick: the index curls until its tip meets PICK_PINCH, holding the
# pick at the index fingertip against the thumb pad; the middle/ring/pinky fold
# in behind it (sitting clear, ~-x of the pinch), each a touch tighter, for a
# natural cascade. Each entry is (knuckle xyz, phalanx lengths, per-joint
# flexion MCP/PIP/DIP in degrees). The MCP flexion is kept moderate (the long
# proximal is what dives deepest) and the PIP/DIP tuck the tip back UP into the
# palm, so the whole fist stays compact and its lowest point clears the string
# plane (the pick tip reaches ~26 mm further down) instead of punching through
# the strings. The index is curled MORE at the MCP than the others (70 vs ~52)
# precisely so its fingertip reaches DOWN to the pinch rather than staying
# extended past it.
# RIGHT hand (PICK_THUMB_AXIS): the knuckle line runs index at -x out to pinky at
# +x, so mounted the index/pick ride ON TOP and the pinky hangs below.
PICK_FIST = {
    "index":  ((-0.030, 0.036, 0.020), FRET_FINGERS[1]["lengths"], (70, 70, 50)),
    "middle": ((-0.010, 0.037, 0.021), FRET_FINGERS[2]["lengths"], (52, 98, 62)),
    "ring":   ((0.010, 0.036, 0.020), FRET_FINGERS[3]["lengths"], (54, 100, 62)),
    "pinky":  ((0.028, 0.034, 0.019), FRET_FINGERS[4]["lengths"], (56, 102, 60)),
}


def build_pick_hand(coll, mat):
    """The picking fist: full fingers curl along the strings into the palm,
    the thumb crosses over to pinch the pick against the index fingertip."""
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
    _bone_box(arm_obj, coll, mat, "wrist", PICK_PALM_SIZE,
              (0.0, 0.008 - 0.030, 0.013))

    # Finger chains as static boxes, spread across x (RIGHT hand: index on the
    # -x/thumb side, pinky on +x) and pointing along +y (parallel to the
    # strings), curling into the palm -- see PICK_FIST.
    for name, (knuckle, lengths, flex) in PICK_FIST.items():
        pts = _curled_finger(knuckle, lengths, flex)
        for seg, (a, b) in zip(("prox", "mid", "dist"),
                               zip(pts[:-1], pts[1:])):
            w, h = _finger_cross(name, seg)
            _seg_box(arm_obj, coll, mat, w, h, tuple(a), tuple(b))

    # Thumb: lies along the thumb (-x) edge of the palm -- the RIGHT hand's
    # radial edge, see PICK_THUMB_AXIS -- and curls down toward the PALM (-z),
    # its pad meeting the curled index fingertip so the pick is pinched between
    # the two.
    thumb_base = (-0.042, 0.002, -0.006)
    thumb_knuckle = (-0.037, 0.015, -0.022)
    _seg_box(arm_obj, coll, mat, 0.017, 0.016, thumb_base, thumb_knuckle)
    _seg_box(arm_obj, coll, mat, 0.016, 0.015, thumb_knuckle, PICK_PINCH)

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
