"""Builds the two minimalist hand/finger rigs used by animate_hands.py.

Run inside Blender (after build_piano.py) to create a "Hands" collection
containing one armature per hand, named ``Hand_L`` / ``Hand_R``:

  - bone ``wrist`` (root), plus per finger f1 (thumb) .. f5 (pinky) a chain
    ``f<n>_prox`` -> ``f<n>_mid`` -> ``f<n>_dist``. All finger bones point
    +y (toward the fallboard) in rest pose with roll 0, so their pose-space
    x-rotation is pitch (curl) and z-rotation is sideways reach - the two
    axes animate_hands.py drives with closed-form IK. Every phalanx is
    caged to its human range of motion (FINGER_ROT_LIMIT / THUMB_ROT_LIMIT).
  - box meshes bone-parented to every phalanx and a palm box on the wrist,
    matching the piano's blocky aesthetic. No skinning/weights: rigid
    boxes are enough at this level of stylization. Segment lengths, knuckle
    spread and per-phalanx cross-sections are adult-hand anthropometry,
    shared with the guitarist's and bassist's fretting hands so all three
    players read as the same size of human.

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

import math

import bpy
import bmesh


# ---------------------------------------------------------------------------
# Hand dimensions (meters), right-hand orientation; x mirrors for the left.
# Knuckle offsets are relative to the armature object origin (the wrist).
#
# Fingers 2..5 are the SAME hand the guitarist and bassist play with
# (guitar/build_hands.FRET_FINGERS, bass_guitar/build_hands.FRET_FINGERS),
# mirrored into this rig's x convention -- for a right hand the thumb sits at
# -x and the little finger at +x, the opposite order from the fret hands, which
# spread index->pinky along the neck. All three players therefore read as the
# same size of human, and the phalanx lengths are the measured adult means
# rather than the slightly small ones this rig started with: proximal ~44-48 mm
# on the index/middle down to ~37 mm on the little finger, each phalanx roughly
# 0.6 of the one before it (Garrett 1971 hand anthropometry; ANSUR).
#
# The knuckle line keeps its natural arc (the middle knuckle furthest forward,
# the little finger ~9 mm behind it) and now spans a realistic 74 mm from index
# to little knuckle centre -- ~25 mm pitch, up from the 19 mm this rig used,
# which had the fingers bunched noticeably narrower than the boxes on them.
#
# Finger 1 is the THUMB, and unlike the fret hands (whose thumbs are static
# boxes pressing the back of the neck) it is a driven chain, because on a piano
# the thumb plays. Its three bones are the anatomical thumb column -- metacarpal
# (~46 mm), proximal (~31 mm), distal (~22 mm) -- rooted at the CMC joint, which
# sits close to the wrist on the radial edge of the palm and a little below the
# palm plane, not up on the knuckle line. Fully extended, the thumb tip then
# reaches about to the index finger's PIP joint, as a real thumb does.
# ---------------------------------------------------------------------------

FINGERS = {
    #        knuckle (x, y, z)             (prox, mid, dist) segment lengths
    1: {"knuckle": (-0.038, 0.004, -0.014), "lengths": (0.046, 0.031, 0.022)},
    2: {"knuckle": (-0.037, 0.050, 0.000),  "lengths": (0.044, 0.027, 0.021)},
    3: {"knuckle": (-0.013, 0.055, 0.000),  "lengths": (0.048, 0.031, 0.023)},
    4: {"knuckle": (0.013, 0.053, 0.000),   "lengths": (0.045, 0.029, 0.022)},
    5: {"knuckle": (0.037, 0.046, 0.000),   "lengths": (0.037, 0.025, 0.020)},
}

# --- Finger cross-section (metres), from hand anthropometry ------------------
# Shared verbatim with the guitar/bass hands (their FINGER_CROSS): each phalanx
# box approximates an average adult finger, broadest at the proximal phalanx and
# TAPERING to the tip, with the little finger the slimmest and the index/middle
# the widest. Breadths follow measured digit-breadth norms -- ~18 mm proximal on
# the index/middle down to ~15 mm on the little finger, tapering to ~11-14 mm at
# the fingertip (Garrett 1971; ANSUR) -- and depth (dorsopalmar) runs ~1 mm under
# the breadth, a finger being a touch wider than it is thick. This replaced a
# flat 14x12 mm box on every phalanx, which was thinner than a real finger and
# left oversized gaps between neighbours (now a natural few millimetres) while
# reading as sticks rather than fingers. The thumb is the thickest digit, ~22 mm
# across the metacarpal/thenar mass and ~17 mm at the tip.
#                    (width, height) for prox / mid / dist phalanx
FINGER_CROSS = {
    "thumb":  {"prox": (0.022, 0.020), "mid": (0.019, 0.017), "dist": (0.017, 0.016)},
    "index":  {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "middle": {"prox": (0.018, 0.017), "mid": (0.016, 0.015), "dist": (0.014, 0.013)},
    "ring":   {"prox": (0.017, 0.016), "mid": (0.015, 0.014), "dist": (0.013, 0.012)},
    "pinky":  {"prox": (0.015, 0.014), "mid": (0.013, 0.012), "dist": (0.011, 0.011)},
}

# Which anthropometric profile each finger number draws its section from.
FINGER_PROFILE = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}

# Palm sized to the (now realistic) knuckle span and to the same hand the
# guitarist/bassist play with (their FRET_PALM_SIZE), rather than the wider,
# flatter slab this rig started with: a hand is ~86 mm across the metacarpals
# and ~24 mm thick through the palm, not 95 x 20.
PALM_SIZE = (0.086, 0.072, 0.024)

# Where the wrist bone and its palm box sit in armature space. The box is
# bone-parented to the wrist bone, whose parent space has its origin at the bone
# TAIL, so its centre lands PALM_OFFSET_Y in front of that. Kept as constants
# because animate_hands needs the same box: a phalanx inside the palm is buried
# in the hand's own mass, and nothing there can be seen crossing anything.
WRIST_BONE_Y = (-0.025, 0.030)   # head, tail
PALM_OFFSET_Y = -0.013
PALM_CENTRE = (0.0, WRIST_BONE_Y[1] + PALM_OFFSET_Y, 0.0)

# --- Finger joint range-of-motion limits (human norms; degrees) --------------
# Ported from the guitar/bass fret hands (their FINGER_ROT_LIMIT), which cage
# every phalanx the way build_guitarist/build_bassist cage the elbow and knee:
# real hinges that flex freely but must never bend BACKWARD past their small
# natural extension, and (for the inter-phalangeal joints) neither splay
# sideways nor twist. They are LOCAL LIMIT_ROTATION constraints applied at BUILD
# time, so they only ever GUARD future hand poses -- animate_hands keeps its
# keyed press/hover poses inside them (see FINGER_MCP_SPLAY there).
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
# 26 deg cap animate_hands applies to the IK) is what stops a finger reaching
# sideways for a distant key by swinging UNDER its neighbours; reaching across
# the keyboard is the WRIST's job. The one joint kept generous is the MID (PIP):
# this rig's closed-form IK lumps the distal phalanx into the middle link (see
# animate_hands._finger_ik), so f<n>_mid.x carries the COMBINED PIP+DIP fold
# (~190 deg in a real hand) while f<n>_dist.x holds a fixed natural flexion.
FINGER_ROT_LIMIT = {
    "prox": {"x": (-100.0, 35.0), "y": (-8.0, 8.0), "z": (-30.0, 30.0)},  # MCP
    "mid":  {"x": (-142.0, 5.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # PIP(+lumped DIP)
    "dist": {"x": (-95.0, 10.0),  "y": (-5.0, 5.0), "z": (-6.0, 6.0)},    # DIP: flex only
}

# The thumb is a different joint chain and gets its own, looser cage -- caging it
# like a finger is what would forbid the thumb-under of a scale run. Its bones
# here are metacarpal / proximal / distal, i.e. the CMC, MCP and IP joints:
#   * CMC (saddle joint): flexion/extension ~15-20 deg each way, but ~45-60 deg
#     of palmar/radial ABDUCTION -- the widest sideways range in the hand. This
#     rig builds the thumb pointing +y in the palm plane rather than rotated
#     palmarly out of it, so its prox bone lumps that palmar rotation in with
#     CMC flexion; the x range is correspondingly generous. Sideways it is the
#     one LOPSIDED bound in the hand: the wide swing is all abduction, away from
#     the palm, because zero here is already the thumb lying against the index
#     and a real one gets across the palm by rotating UNDER it rather than
#     swinging further into it (see animate_hands.THUMB_CMC_ADDUCT, whose 12 deg
#     this leaves a little headroom over; the 60 deg abduction cap it leaves room
#     for is the top of the range, because in this rig the yaw also pays for the
#     bearing the roll below costs). Written for a RIGHT hand -- rz is -yaw, so
#     abduction is +z there -- and flipped for the left by rot_limit.
#     The saddle also PRONATES the column about its own length, the other half
#     of opposition, and unlike a finger's few degrees of passive twist that is
#     a real 45-90 deg of active rotation. It is what turns the thumb's flexion
#     out of the vertical plane so the tip folds across the palm rather than
#     hooking into the keyboard (animate_hands.THUMB_ROLL, 35 deg, which this
#     leaves headroom over). Symmetric, so it needs no flip: the roll simply
#     changes sign with the hand, toward whichever side the palm is on.
#   * MCP: flexion ~55, extension ~0-10, a few degrees of passive splay. As with
#     the fingers' PIP, the IK folds the distal phalanx into this link, so the
#     bone carries the COMBINED MCP+IP flexion (~135 deg in a real thumb) and is
#     capped there rather than at the MCP's own 55.
#   * IP: flexion ~80, hyperextension ~15-20 (thumbs really do bend back).
THUMB_ROT_LIMIT = {
    "prox": {"x": (-75.0, 30.0), "y": (-45.0, 45.0), "z": (-17.0, 65.0)},   # CMC
    "mid":  {"x": (-120.0, 12.0), "y": (-10.0, 10.0), "z": (-12.0, 12.0)},  # MCP(+lumped IP)
    "dist": {"x": (-85.0, 20.0), "y": (-5.0, 5.0),   "z": (-6.0, 6.0)},     # IP
}


def rot_limit(finger, seg, mirror=1.0):
    """The ROM cage for one phalanx: the thumb chain has its own.

    `mirror` is the hand (-1 for a left), which matters only where a bound is
    lopsided - the thumb's sideways range, which opens toward whichever side of
    the hand its own thumb sits on. The fingers' bounds are symmetric, so the
    flip leaves them exactly as they are.
    """
    limit = (THUMB_ROT_LIMIT if finger == 1 else FINGER_ROT_LIMIT)[seg]
    if mirror > 0:
        return limit
    lo, hi = limit["z"]
    return {**limit, "z": (-hi, -lo)}


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


def _finger_cross(finger, seg):
    """(width, height) of finger `finger`'s `seg` phalanx box, from anthropometry."""
    return FINGER_CROSS[FINGER_PROFILE[finger]][seg]


def _limit_rot(pbone, limit):
    """Add a LOCAL-space LIMIT_ROTATION constraint (degrees) to a hand bone --
    the same soft cage the guitar/bass fret hands put on their phalanges."""
    lr = pbone.constraints.new('LIMIT_ROTATION')
    lr.owner_space = 'LOCAL'
    for axis in ("x", "y", "z"):
        setattr(lr, f"use_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(lr, f"min_{axis}", math.radians(lo))
        setattr(lr, f"max_{axis}", math.radians(hi))


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
    wrist.head = (0.0, WRIST_BONE_Y[0], 0.0)
    wrist.tail = (0.0, WRIST_BONE_Y[1], 0.0)

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

    # Cage each phalanx to its human range of motion (one-way flexion hinges).
    for f in FINGERS:
        for seg in ("prox", "mid", "dist"):
            _limit_rot(arm_obj.pose.bones[f"f{f}_{seg}"],
                       rot_limit(f, seg, mirror))

    for f, spec in FINGERS.items():
        for seg, length in zip(("prox", "mid", "dist"), spec["lengths"]):
            w, h = _finger_cross(f, seg)
            _bone_box(arm_obj, coll, mat, f"f{f}_{seg}",
                      (w, length * 0.92, h), -length / 2.0)

    # Palm spans from the wrist to the knuckle line (wrist bone tail at
    # y=0.030 is the parent-space origin).
    _bone_box(arm_obj, coll, mat, "wrist", PALM_SIZE, PALM_OFFSET_Y)

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
