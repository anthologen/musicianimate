"""Builds the fretting- and plucking-hand rigs used by animate_hands.py.

Run inside Blender (after build_bass_guitar.py) to create a "BassHands"
collection containing two armatures:

  - ``FretHand``: wrist root bone plus four finger chains f1 (index) ..
    f4 (pinky), each ``f<n>_prox -> f<n>_mid -> f<n>_dist`` pointing +y in
    armature space with roll 0 - the same convention as the piano/guitar
    rigs, so pose-space x-rotation is curl and z-rotation is sideways
    reach. The armature is rotated 90 deg about Z (HAND_ROT_Z) then tilted
    WRAP_TILT about the neck axis: the hand wraps the neck from the treble
    side, palm beside the treble edge below string level, knuckles riding
    the edge (index toward the nut, pinky toward the bridge), fingers
    arching over the strings, and a static thumb box pressing the back of
    the neck. The hand is a realistic size matching the plucking hand; the
    wide bass frets are covered by finger splay, not a stretched palm.
  - ``PluckHand`` (fingerstyle, the default): a hand hovering above the
    strings near the bridge pickup, with a static thumb box anchored on
    the pickup (floating-thumb style) and two articulated plucking
    fingers - ``pi`` (index) and ``pm`` (middle) - pitched down at the
    strings so the animator can curl the assigned finger onto a string
    and follow through toward the palm.

For ``--style pick`` a ``PickHand`` (the guitar's pick rig) is built
instead of the PluckHand.

Rigid boxes bone-parented to each bone match the piano/guitar hands'
blocky aesthetic; no skinning.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_hands", "/path/to/bass_guitar/build_hands.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_hands()              # fingerstyle
    mod.build_hands(style="pick")  # pick
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

# The fret hand is additionally tilted about the neck axis so it wraps the
# neck from the treble side (see guitar/build_hands.WRAP_TILT).
WRAP_TILT = 0.6

# The pluck hand hovers above the strings with the wrist draped over the
# top of the instrument on the THICK-string side, palm facing down, and
# the fingers reaching ACROSS the strings (perpendicular to them) toward
# the thinner side while angling down at the string plane - the realistic
# fingerstyle posture. Built in a canonical local frame (+y = finger rest
# direction): first yawed -90 about Z so +y points across the strings
# (world +X, toward the thinner strings), then pitched PLUCK_DOWN down
# about world Y so the fingertips angle down at the strings. Curling a
# finger then draws its tip back toward the palm - across the string
# (in the X-Z plane), i.e. perpendicular to the string - which is the
# plucking motion.
PLUCK_DOWN = math.radians(22)
PLUCK_ROT = (mathutils.Matrix.Rotation(PLUCK_DOWN, 3, 'Y')
             @ mathutils.Matrix.Rotation(-math.pi / 2.0, 3, 'Z'))


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


def pluck_world_offset(v):
    """Armature-local offset -> world offset under the PluckHand pose."""
    return tuple(PLUCK_ROT @ mathutils.Vector(v))


def pluck_local_offset(w):
    """Inverse of pluck_world_offset: world offset -> armature-local, so
    thumb/finger boxes can be laid out in intuitive world axes (+x toward
    the thinner strings, +y toward the nut, +z up) then bone-parented."""
    return tuple(PLUCK_ROT.transposed() @ mathutils.Vector(w))


# ---------------------------------------------------------------------------
# Fret-hand dimensions (metres), in armature-local space. Knuckle x spreads
# the fingers along the neck; y is the reach direction across the strings.
# The knuckle span (index->pinky ~74 mm) and palm are sized to a realistic
# human hand -- the SAME hand size as the plucking hand -- so the two hands
# read as a matched pair. The wide bass frets are reached NOT by a stretched
# palm but by finger SPLAY (the IK's z-rotation / knuckle abduction), the
# reach mechanism the animator already drives (see FINGER_ROT_LIMIT and
# animate_hands' _fret_rotation).
# ---------------------------------------------------------------------------

FRET_FINGERS = {
    #        knuckle (x, y, z)             (prox, mid, dist) segment lengths
    1: {"knuckle": (0.037, 0.050, 0.0), "lengths": (0.044, 0.027, 0.021)},
    2: {"knuckle": (0.013, 0.055, 0.0), "lengths": (0.048, 0.031, 0.023)},
    3: {"knuckle": (-0.013, 0.053, 0.0), "lengths": (0.045, 0.029, 0.022)},
    4: {"knuckle": (-0.037, 0.046, 0.0), "lengths": (0.037, 0.025, 0.020)},
}

# Plucking fingers: index (pi) and middle (pm), close together. Under
# PLUCK_ROT local +x maps to world -Y, so a more-negative knuckle x sits
# further toward the neck (+Y): the index is placed on the neck side next
# to the thumb, the middle just bridge-ward of it (a right hand). Middle
# is a touch longer.
PLUCK_FINGERS = {
    # Full finger length, matching the fretting hand's index/middle for
    # realism. A long finger reaching a close string folds hard, so the
    # wrist is held high enough (~50 mm above the strings, see
    # REST_LOCATION) that the fingers drape DOWN and the proximal flexes
    # down at the knuckle instead of hyperextending up.
    "pi": {"knuckle": (-0.011, 0.040, 0.0), "lengths": (0.044, 0.027, 0.021)},
    "pm": {"knuckle": (0.011, 0.040, 0.0), "lengths": (0.048, 0.031, 0.023)},
}

# --- Finger cross-section (metres), from hand anthropometry ------------------
# Each phalanx box approximates an average adult finger: broadest at the
# proximal phalanx and TAPERING to the tip, with the little finger the
# slimmest and the index/middle the widest. Breadths (side-to-side) follow
# measured digit-breadth norms -- ~18 mm proximal for the index/middle down
# to ~15 mm for the little finger, tapering to ~11-14 mm at the fingertip
# (Garrett 1971 hand anthropometry; ANSUR). Depth (dorsopalmar) runs ~1 mm
# under the breadth, a finger being a touch wider than it is thick. This
# replaced a flat 12x13 mm box on every phalanx, which -- being far thinner
# than a real finger -- left oversized gaps between the fingers.
#
# The gaps are ALSO bounded by the fret-hand knuckle PITCH (FRET_FINGERS
# knuckle x: ~24-26 mm centre-to-centre, a large-hand span chosen so the
# fingers cover the wide bass frets). That pitch is baked into the animation
# and so is left as-is; realistic finger widths shrink the gaps to a natural
# few millimetres rather than closing them entirely. The thumb keeps its own
# (larger) width where it is built, so it stays the thickest digit.
#                    (width, height) for prox / mid / dist phalanx
FINGER_CROSS = {
    "index":  {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "middle": {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "ring":   {"prox": (0.017, 0.016), "mid": (0.015, 0.014), "dist": (0.013, 0.012)},
    "pinky":  {"prox": (0.015, 0.014), "mid": (0.013, 0.012), "dist": (0.011, 0.011)},
}

# Which anthropometric profile each finger key -- across the fret (1..4),
# pluck (pi/pm) and pick-fist (index..pinky) rigs -- draws its section from.
FINGER_PROFILE = {
    1: "index", 2: "middle", 3: "ring", 4: "pinky",   # fret fingers
    "pi": "index", "pm": "middle",                     # pluck fingers
    "index": "index", "middle": "middle",              # pick fist
    "ring": "ring", "pinky": "pinky",
}


def _finger_cross(key, seg):
    """(width, height) of finger `key`'s `seg` phalanx box, from anthropometry."""
    return FINGER_CROSS[FINGER_PROFILE[key]][seg]


# Both palms are the same realistic size so the two hands match; the fret
# palm just covers the (now realistic) knuckle span rather than a stretched one.
FRET_PALM_SIZE = (0.086, 0.072, 0.024)
PLUCK_PALM_SIZE = (0.085, 0.070, 0.026)

# --- Finger joint range-of-motion limits (human norms; degrees) --------------
# The finger joints are caged the same way build_bassist cages the ELBOW and KNEE:
# real hinges that flex freely but must never bend BACKWARD past their small natural
# extension, and (for the inter-phalangeal joints) neither splay sideways nor twist.
# They are LOCAL LIMIT_ROTATION constraints applied at BUILD time, so they only ever
# GUARD future hand poses -- the shipped fret/pluck performance already lives inside
# them (its curls are baked keyframes within these bounds).
#
# In this rig's finger frame (bone runs +y): X = curl (flexion is NEGATIVE x),
# Z = knuckle splay/abduction, Y = axial twist. Bounds follow measured active-ROM
# norms for the index-through-little fingers (Thieme 2024 normative study; AAOS /
# goniometry references):
#   * MCP flexion ~85-90, extension (hyperextension) ~25-30 (the little finger the
#     most, mean ~26 and up toward ~38 at +2SD), abduction/adduction ~+/-25.
#   * PIP flexion ~95-110, ~0 hyperextension, no splay/twist.
#   * DIP flexion ~80-85, ~0-10 hyperextension, no splay/twist.
# The MCP was previously widened to +/-65 splay / +40 extension so the fret hand could
# REACH across the wide bass frets by knuckle-splay alone -- but that let a pressing
# finger deviate ~57 deg and sweep UNDER its neighbours (the octave double-stop at
# frame 117). Reach is now the WRIST's job (animate_hands caps the finger IK splay at
# FINGER_MCP_SPLAY and the wrist search penalizes any pose that demands more), so the
# splay comes back to +/-30 (the ~25 deg norm plus a little guard headroom over the
# 26 deg IK cap) and extension to +35 (the little finger's real backward range, which
# the pinky uses reaching up to a high octave string). The one joint kept generous is
# the MID (PIP): this rig's closed-form IK lumps the distal phalanx INTO the middle
# link (see animate_hands _finger_ik), so f<n>_mid.x carries the COMBINED PIP+DIP fold
# (~190 deg total in a real hand) while f<n>_dist.x barely moves -- the stylized fret
# curl drives it to ~140, which is within that combined envelope even though it is past
# the PIP-alone norm. Clamping it would unseat the press. See animate_hands.
FINGER_ROT_LIMIT = {
    "prox": {"x": (-100.0, 35.0), "y": (-8.0, 8.0), "z": (-30.0, 30.0)},  # MCP
    "mid":  {"x": (-142.0, 5.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # PIP(+lumped DIP): flex only
    "dist": {"x": (-95.0, 10.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # DIP: flex only
}


def _limit_rot(pbone, limit):
    """Add a LOCAL-space LIMIT_ROTATION constraint (degrees) to a hand bone --
    the same soft cage build_bassist puts on the body's FK joints."""
    lr = pbone.constraints.new('LIMIT_ROTATION')
    lr.owner_space = 'LOCAL'
    for axis in ("x", "y", "z"):
        setattr(lr, f"use_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(lr, f"min_{axis}", math.radians(lo))
        setattr(lr, f"max_{axis}", math.radians(hi))

# Where idle hands hover before the animator takes over.
REST_LOCATION = {
    "FretHand": (0.095, fret_layout.fret_y(2), 0.028),
    # Wrist draped over the top on the thick-string (-x) side, held FLAT
    # (small PLUCK_DOWN) so the palm is ~parallel to the strings, and high
    # enough (~50 mm above the strings) that the full-length fingers drape
    # down and hook to pluck without the proximal hyperextending. Kept
    # close over the strings (small PLUCK_ACROSS) - moving it out toward
    # the body edge would flatten the reach and bow the fingers back up.
    "PluckHand": (-0.052, fret_layout.PLUCK_Y, fret_layout.STRING_Z + 0.068),
    "PickHand": (0.045, fret_layout.PLUCK_Y + 0.045, 0.110),
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
    along the bone."""
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


def _seg_box(arm_obj, coll, mat, width, height, p0, p1):
    """A box of the given cross-section spanning armature points p0->p1,
    its long (local-y) axis along the segment, parented to the wrist."""
    p0, p1 = mathutils.Vector(p0), mathutils.Vector(p1)
    d = p1 - p0
    center = (p0 + p1) / 2.0
    rot = d.to_track_quat('Y', 'Z').to_euler()
    _bone_box(arm_obj, coll, mat, "wrist", (width, d.length, height),
              (center.x, center.y - 0.030, center.z), rotation=rot)


def _new_armature(name, coll, rot_matrix):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = REST_LOCATION[name]
    arm_obj.rotation_euler = rot_matrix.to_euler()
    coll.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    return arm_obj


def _build_finger_chains(arm_obj, fingers):
    """Add wrist + finger chains (piano convention) to an armature in edit
    mode, one chain per entry of `fingers`."""
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
    rot = (mathutils.Matrix.Rotation(WRAP_TILT, 4, 'Y')
           @ mathutils.Matrix.Rotation(HAND_ROT_Z, 4, 'Z'))
    arm_obj = _new_armature("FretHand", coll, rot)
    _build_finger_chains(arm_obj, FRET_FINGERS)
    _add_finger_boxes(arm_obj, coll, mat, FRET_FINGERS)
    _bone_box(arm_obj, coll, mat, "wrist", FRET_PALM_SIZE, (0.0, -0.013, 0.0))

    # Static thumb reaching under the neck to press its back, opposite the
    # fingers (specified as a world offset from the wrist, converted into
    # the tilted armature's local frame). It sits on the INDEX/radial edge of
    # the palm (world +y here maps to armature-local +x, the index side --
    # cf. FRET_FINGERS, index knuckle x=+0.037), where a real hand's thumb
    # attaches, rather than bisecting the back of the hand.
    # It is built as TWO segments meeting at a knuckle (the thumb's IP joint)
    # so it reads as a jointed thumb rather than one rigid stick: a longer
    # proximal segment leaving the palm, then a shorter distal segment that
    # flexes at the joint to lie flatter against the neck (as a thumb pad
    # pressing the back does), tapering toward the tip. Same static parenting
    # to the wrist as the pluck/pick-hand thumbs.
    thumb_center_w = (-0.050, 0.040, -0.022)
    thumb_dir_w = mathutils.Vector((-0.94, 0.12, -0.32)).normalized()
    c_local = mathutils.Vector(_fret_local_offset(thumb_center_w))
    d_local = mathutils.Vector(_fret_local_offset(thumb_dir_w)).normalized()
    thumb_len, prox_len, dist_len = 0.075, 0.045, 0.030
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


def build_pluck_hand(coll, mat):
    """The fingerstyle plucking hand: two articulated fingers (pi, pm)
    pitched down at the strings that do the plucking, plus loosely-curled
    static ring and pinky fingers (present for a complete hand but not
    used to pluck), a static anchored thumb, and a palm."""
    arm_obj = _new_armature("PluckHand", coll, PLUCK_ROT.to_4x4())
    _build_finger_chains(arm_obj, PLUCK_FINGERS)
    _add_finger_boxes(arm_obj, coll, mat, PLUCK_FINGERS)

    # Back-of-hand / palm mass, above and behind the knuckles. Shifted to
    # the +x (bridge-ward, pinky) side in local space so it covers all four
    # fingers, since the index/middle sit on the -x (neck, thumb) side.
    _bone_box(arm_obj, coll, mat, "wrist", PLUCK_PALM_SIZE,
              (0.020, 0.006 - 0.030, 0.010))

    # Ring and pinky: FULL-length relaxed fingers (3 phalanges, ~86/80 mm,
    # matching the fretting hand for realism) trailing the plucking fingers
    # toward the bridge (world -Y). Because the knuckles sit only ~53 mm
    # above the strings, a straight long finger would punch through, so -
    # like a real non-plucking finger - they curl into a loose HOOK: the
    # proximal flexes DOWN from the knuckle, the middle curls under, and
    # the tip tucks gently back up toward the palm, staying above the
    # strings. Laid out as world offsets (knuckle K -> P -> M -> tip T).
    for profile, (K, P, M, T) in (
            ("ring", ((0.037, -0.032, -0.015), (0.062, -0.032, -0.049),
                      (0.040, -0.032, -0.060), (0.022, -0.032, -0.052))),
            ("pinky", ((0.037, -0.052, -0.016), (0.060, -0.052, -0.048),
                       (0.040, -0.052, -0.058), (0.024, -0.052, -0.050)))):
        for seg, (wa, wb) in zip(("prox", "mid", "dist"),
                                 ((K, P), (P, M), (M, T))):
            w, h = _finger_cross(profile, seg)
            _seg_box(arm_obj, coll, mat, w, h,
                     pluck_local_offset(wa), pluck_local_offset(wb))

    # Thumb: floating-thumb anchor resting on the pickup near the low-E
    # (thick, -x) string and pointing TOWARD THE NECK (+y) - the natural
    # right-hand thumb direction, opposite the fingers. World offsets from
    # the wrist.
    for wa, wb, bw, bh in (
            ((0.006, 0.010, -0.032), (0.015, 0.030, -0.050), 0.017, 0.016),
            ((0.015, 0.030, -0.050), (0.022, 0.050, -0.062), 0.015, 0.014)):
        _seg_box(arm_obj, coll, mat, bw, bh,
                 pluck_local_offset(wa), pluck_local_offset(wb))
    return arm_obj


def _make_pick_mesh(name, length):
    """A flat rounded-triangle pick, tip pointing down (-z) `length` below
    the object origin (see guitar/build_hands)."""
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


# Pick geometry (armature-local), for --style pick. The constants below place the
# picking fist and its pick in the FLAT authoring frame; animate_hands re-derives the
# whole pluck swing from PICK_HAND_ROT + PICK_TIP_LOCAL, and the assembly is then mounted
# rigidly onto the standing bassist (see animate_bassist). So all of them are solved for
# the MOUNTED playing pose and would need re-solving if the wear angle (NECK_ELEV etc.)
# or the plucking arm's pose changed a lot. Bass-only: the guitarist has its own
# guitar/build_hands.py (PICK_ROT), unaffected.
#
# PICK_HAND_ROT -- the fist orientation, solved against the mounted plucking forearm so
# the wrist READS AS NEUTRAL: the finger axis (+y) is aligned with the incoming forearm
# (only a slight ~5 deg flexion remains) and -- crucially -- the PALM faces ONTO the
# strings, fingers/pick curling down toward them, back of the hand up (the correct pick
# posture). NOTE the palm is the hand's local -Z side (fingers curl toward -z, the pick
# is pinched there); the +Z side is the BACK of the hand. Getting that backwards points
# the palm out at the audience -- the hand then reads as snapped 180 deg the wrong way.
# Recipe (mounted pose, frame ~20): e = plucking forearm dir (elbow->wrist, world); build
# a right-handed basis R_des with columns [across, finger, +Z=back] so that the -Z (palm)
# axis faces the STRINGS (the back +Z points OUT of the bass face, toward the audience and
# up); finger=e; then PICK_HAND_ROT = holderR^-1 @ R_des (holderR = BassRig world rot).
# The mounted result is validated against human wrist ROM in animate_bassist
# (_check_wrist_pose), which checks the -Z palm faces the strings, so a palm-away /
# over-bent re-solve is caught loudly at build time.
# Solved for the RELAXED-elbow plucking arm (animate_bassist ELBOW_POLE_OVERRIDE["R"] hangs
# the upper arm nearly vertical, elbow tucked at the side, forearm coming gently ACROSS to
# the bridge). The whole point of this round: the STRIKE pose -- not some rest pose -- must
# be the straight one.
# The pluck is a wrist swing whose bottom (deepest dip, the onset) is where the hand
# actually reads; the earlier solves aligned finger=forearm at wrist-rotation ZERO, but the
# swing then rotated the wrist ~74 deg to dip the pick to the strings, so the played wrist
# was ALWAYS ~70 deg bent. Here the pick is pinched to hang straight DOWN off the palm
# (PICK_PINCH + PICK_TIP_LOCAL below), so that dip-bottom IS the object pose: the swing
# adds ~0 at the onset. PICK_HAND_ROT is then the strike orientation directly -- finger
# axis (+y) along the (near-level) forearm with a slight 15 deg downward break (the natural
# slight bend), and the -Z palm turned onto the strings. So the finger axis reads straight
# and the pick still crosses the strings squarely, with only the gentle +/-swing breaking
# the wrist a little between notes. Because the pick hangs off the PALM (not out along the
# fingers), the hand can follow whatever forearm the relaxed elbow gives -- straightness no
# longer needs a steep, elbow-raising forearm. MOUNT-COUPLED: redo this trio (and see
# ANCHOR_WORLD / ELBOW_POLE_OVERRIDE) if the wear height or R elbow pole changes -- re-run
# the solver in the round-19 notes (finger=forearm+15deg break, palm onto strings, blade so
# the pick's head->tip is straight down => the swing argmin sits at the strike).
PICK_HAND_ROT = mathutils.Euler((0.294463, 0.0, -0.832313), 'XYZ').to_matrix()
# The pick is pinched between the THUMB PAD and the side of the INDEX. It sits close under
# the wrist on the PALM (-z) side and protrudes DOWNWARD toward the strings, rather than
# out along the fingertips -- that is what lets the pick be short while still hanging down
# far enough for a straight wrist to reach the strings. The hand is a RIGHT hand (palm -z,
# fingers +y), so the thumb/index/pick live at NEGATIVE local x.
PICK_PINCH = (-0.015, 0.010, -0.035)
# PICK_MESH_ROT / PICK_TIP_LOCAL aim the pick BLADE so that, in the mounted pose, it lies
# in the across-string / depth plane and CROSSES the strings at ~90 deg (instead of
# raking nearly ALONG them) while still hanging down far enough to strike. Because the
# pluck swing is a wrist rotation about the string axis, a blade with no along-string
# component stays ~perpendicular to the strings through the whole stroke. Recipe: choose
# the world-flat blade direction w ~ (0.6, 0, -0.8) (zero along-string y => 90 deg; the
# -z dips it to the strings), d_local = PICK_HAND_ROT^-1 @ w, then
# PICK_TIP_LOCAL = PICK_PINCH + d_local * pick_len  and  PICK_MESH_ROT = the track-quat
# rotation from the pick mesh's built -z axis to d_local.
# Here the blade is aimed straight DOWN off the palm (its head->tip vector, in the flat
# frame, points along -Z perpendicular to the swing axis) so the dip-bottom coincides with
# the straight object pose -- that is the whole trick that lets the played wrist be
# straight. ~35 mm, just the tip past the fist; still crosses the strings ~square (no along-
# string rake) because the swing pivots about the string axis and the blade has no along-
# string component.
PICK_MESH_ROT = (-1.180094, 0.293406, -0.197252)
PICK_TIP_LOCAL = (-0.026860, -0.026297, -0.049950)


def pick_world_offset(v):
    """Armature-local offset -> world offset under the PickHand pose."""
    return tuple(PICK_HAND_ROT @ mathutils.Vector(v))


def _curled_finger(knuckle, lengths, flex_deg):
    """Points knuckle -> PIP -> DIP -> tip for a finger curling toward the
    palm (local -z). Each phalanx adds its flexion about the knuckle-parallel
    local x axis (curl is negative x in this rig's finger frame, cf.
    FINGER_ROT_LIMIT), and the angle is CUMULATIVE, so a big enough total
    folds the fingertip back into the fist."""
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
# a matched pair -- curled into a relaxed fist rather than left as stubs. The
# thumb and a loosely-curled index pinch the pick (the index flexes least, so it
# lies alongside the pick edge); the middle/ring/pinky fold in behind it, each a
# touch tighter, for a natural cascade. Each entry is
# (knuckle xyz, phalanx lengths, per-joint flexion MCP/PIP/DIP in degrees).
# The MCP flexion is kept moderate (the long proximal is what dives deepest) and
# the PIP/DIP tuck the tip back UP into the palm, so the whole fist stays compact
# and its lowest point clears the string plane (~the pick tip's depth) instead of
# punching through the strings -- the same clearance concern the short stubs had.
PICK_FIST = {
    "index":  ((-0.030, 0.036, 0.020), FRET_FINGERS[1]["lengths"], (40, 66, 50)),
    "middle": ((-0.010, 0.037, 0.021), FRET_FINGERS[2]["lengths"], (52, 98, 62)),
    "ring":   (( 0.010, 0.036, 0.020), FRET_FINGERS[3]["lengths"], (54, 100, 62)),
    "pinky":  (( 0.028, 0.034, 0.019), FRET_FINGERS[4]["lengths"], (56, 102, 60)),
}


def build_pick_hand(coll, mat):
    """The stylized picking fist (guitar rig), for --style pick."""
    arm_obj = _new_armature("PickHand", coll, PICK_HAND_ROT.to_4x4())
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    wrist = eb.new("wrist")
    wrist.head = (0.0, -0.025, 0.0)
    wrist.tail = (0.0, 0.030, 0.0)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm_obj.pose.bones["wrist"].rotation_mode = 'XYZ'

    _bone_box(arm_obj, coll, mat, "wrist", (0.070, 0.066, 0.032),
              (0.0, 0.008 - 0.030, 0.013))
    # Finger chains as static boxes (RIGHT hand: thumb/index on the -x side,
    # pinky on +x). Full three-phalanx fingers (matching the fretting hand)
    # curled into a relaxed fist around the pick -- see PICK_FIST.
    for name, (knuckle, lengths, flex) in PICK_FIST.items():
        pts = _curled_finger(knuckle, lengths, flex)
        for seg, (a, b) in zip(("prox", "mid", "dist"),
                               zip(pts[:-1], pts[1:])):
            w, h = _finger_cross(name, seg)
            _seg_box(arm_obj, coll, mat, w, h, tuple(a), tuple(b))

    # Thumb lies along the thumb (-x) edge and curls down toward the PALM (-z), its pad
    # pressing on the pick so the pick is pinched between the thumb and the index's side.
    # The pinch now sits low on the palm (see PICK_PINCH), so the thumb rolls under to meet
    # it there instead of reaching out along the fingertips.
    thumb_base = (-0.032, 0.004, -0.004)
    thumb_knuckle = (-0.028, 0.016, -0.024)
    _seg_box(arm_obj, coll, mat, 0.017, 0.016, thumb_base, thumb_knuckle)
    _seg_box(arm_obj, coll, mat, 0.016, 0.015, thumb_knuckle, PICK_PINCH)

    length = (mathutils.Vector(PICK_PINCH) - mathutils.Vector(PICK_TIP_LOCAL)).length
    _bone_box(arm_obj, coll, _pick_material(), "wrist", None,
              (PICK_PINCH[0], PICK_PINCH[1] - 0.030, PICK_PINCH[2]),
              rotation=PICK_MESH_ROT,
              mesh=_make_pick_mesh("PickMesh", length))
    return arm_obj


def build_hands(style="finger"):
    scene = bpy.context.scene

    old = bpy.data.collections.get("BassHands")
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("BassHands")
    scene.collection.children.link(coll)
    mat = _skin_material()

    hands = [build_fret_hand(coll, mat)]
    if style == "pick":
        hands.append(build_pick_hand(coll, mat))
    else:
        hands.append(build_pluck_hand(coll, mat))
    print(f"Built {len(hands)} bass hand rigs ({style}): " +
          ", ".join(h.name for h in hands))
    return hands


if __name__ == "__main__":
    build_hands()
