"""Builds a blocky, faceless, minimalist humanoid pianist SEATED at the piano,
to stand in for a real character model later on.

Run inside Blender (after build_piano.py) to create a "Pianist" collection: a
single ``Pianist`` armature (average adult proportions) with rigid box meshes on
every bone, shoes on the feet, and a minimalist piano stool under it. Same
blocky stand-in family as ``guitar/build_guitarist.py``, ``bass_guitar/
build_bassist.py`` and ``drum_kit/build_drummer.py`` -- same anthropometry, same
box-per-bone treatment, same human range-of-motion cage -- but posed sitting,
because a pianist plays from a bench and the bench height is what puts the
forearms level with the keys.

The arms are two-bone IK (upper arm + forearm) reaching a ``Wrist_*`` empty with
an ``Elbow_*`` pole, ending in a short ``hand.*`` stub bone. That stub is the
attach point for the piano hand rigs (``build_hands.py``'s ``Hand_L`` /
``Hand_R``): ``animate_pianist.py`` points each ``Wrist_*`` empty at the matching
hand rig's wrist bone, so the arm IK carries the shoulder and elbow to wherever
the animated hand has glided along the keyboard.

The legs are two-bone IK as well (thigh + shin reaching an ``Ankle_*`` empty with
a ``Knee_*`` pole), so the feet stay planted while the body grooves -- and so a
future sustain-pedal pass has an ankle to drive.

COORDINATE CONVENTION -- note it differs from the standing players. The piano's
frame (``key_layout``) has x along the keyboard, y running from the player (0 at
the key fronts) toward the fallboard, and z up. So the pianist sits at NEGATIVE y
and faces +Y: their LEFT hand is at -x and their RIGHT at +x, which is exactly
how ``build_hands.REST_LOCATION`` places ``Hand_L`` / ``Hand_R``. The floor is at
``build_piano.FLOOR_Z`` (one keybed slab + one trestle leg below the keys), not
at z = 0.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_pianist", "/path/to/piano/build_pianist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_pianist()
"""

import math

import bpy
import bmesh
import mathutils

try:
    from . import key_layout
    from .build_piano import FLOOR_Z
    from .build_hands import REST_LOCATION, WRIST_BONE_Y
except ImportError:  # loaded as a loose script via importlib
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import key_layout
    from build_piano import FLOOR_Z
    from build_hands import REST_LOCATION, WRIST_BONE_Y

V = mathutils.Vector

# ---------------------------------------------------------------------------
# Body proportions (average adult, metres). Shared verbatim with the guitarist /
# bassist / drummer stand-ins so all five players read as the same human.
# ---------------------------------------------------------------------------
# Average adult proportions from the Drillis & Contini / Winter anthropometric
# tables (segment length as a fraction of stature H), for H = 1.75 m:
#   upper arm 0.186 H, forearm 0.146 H, thigh 0.245 H, shank 0.246 H,
#   foot length 0.152 H, biacromial (shoulder) breadth 0.259 H,
#   head height ~0.130 H.
_H = 1.75
ANTHRO = {
    "upper_arm": 0.186 * _H,      # 0.326  shoulder -> elbow
    "forearm": 0.146 * _H,        # 0.256  elbow -> wrist
    "hand": 0.10,                 # wrist -> attach stub (real fingers = hand rig)
    "thigh": 0.245 * _H,          # 0.429  hip -> knee
    "shin": 0.246 * _H,           # 0.431  knee -> ankle
    "foot": 0.152 * _H,           # 0.266  heel -> toe (shoe length)
    "shoulder_dx": 0.259 * _H / 2.0,   # 0.227  half biacromial breadth
    "hip_dx": 0.09,               # half hip-joint spacing
    "arm_thick": 0.06,
    "leg_thick": 0.10,
    # Slimmer than the standing players' 0.34: the hanging elbows sit ~0.30 m
    # off the centre line at this seat height, and a 0.34-wide chest leaves them
    # barely a centimetre of air. (The drummer's chest was slimmed for the same
    # reason -- see build_drummer's Torso_Chest.)
    "torso_w": 0.32, "torso_d": 0.17,
    "head": (0.155, 0.20, 0.130 * _H),  # breadth, depth, height (~0.228)
}

# ---------------------------------------------------------------------------
# Where the player sits. These four numbers place the whole figure; everything
# else in POSE is measured off them (or solved from ANTHRO), so moving the bench
# moves the pianist.
# ---------------------------------------------------------------------------
# Centred on middle C, the conventional seat -- and, with this demo's hands
# ranging x = -0.36 .. +0.26, roughly between them.
PLAYER_X = key_layout.key_x(60)
# A standard 50 cm piano bench. The white-key surface sits 0.71 m above the same
# floor, so the keys are ~0.21 m above the seat: the usual bench-to-keyboard
# drop, and what puts the shoulders high enough for the forearms to run level
# into the keys (see ELBOW_BEND).
SEAT_Z = FLOOR_Z + 0.50
SEAT_Y = -0.44                 # bench centre, well clear of the keybed
HIP_Y = -0.36                  # hip joints: sitting on the FRONT of the bench,
#                                which is how a pianist sits
_HIP_Z = SEAT_Z + 0.07         # hip joint above the compressed seat cushion

# How far the whole upper body is rotated BACK about the hip joint, relative to
# the +y walk the joint stack below was originally authored with. That stack was
# built ~6 deg forward of vertical (crown 0.090 m ahead of the hip over a 0.85 m
# rise), which read as a stoop rather than a lean once the blocky torso/head
# boxes were on it; trimming 3 deg off leaves ~3 deg -- still clearly inclined
# into the keyboard, but with the back straight. Raise it to sit up further,
# make it negative to lean in more. This is THE lean knob: every part of the
# figure above the hips (spine and neck bones, shoulders, and the torso/neck/
# head mesh boxes, which carry their own hand-tuned offsets) is placed through
# `_lean_y`, so they all stay in the relationship they were tuned in.
LEAN_TRIM = math.radians(3.0)


def _lean_y(dy, z):
    """Forward offset `dy` (from the hip, +y = toward the keyboard) of a part
    sitting `z` above the hip joint, trimmed back by LEAN_TRIM.

    A SHEAR, not a true rotation: the trim moves y only and leaves every height
    exactly as authored, which matters because the mesh stack below is tuned to
    close with no gaps (see _clothe) and a rotation would slide those seams."""
    return HIP_Y + dy - (z - _HIP_Z) * math.tan(LEAN_TRIM)


# Seated joint positions (world, metres). The figure faces +Y (the keyboard);
# left = -X, right = +X. Vertical stack from the seat: hip joint +0.07, upper
# spine +0.32, shoulders +0.59 (seated acromial height for H = 1.75), neck
# +0.62, crown +0.92 (seated stature). The small +y walk up the stack is a
# gentle forward lean into the instrument, not a hunch -- see LEAN_TRIM, which
# sets how much of the walk written here actually survives into the rig.
POSE = {
    "pelvis":     (PLAYER_X, HIP_Y, _HIP_Z),
    "chest":      (PLAYER_X, _lean_y(0.025, SEAT_Z + 0.32), SEAT_Z + 0.32),
    "neck":       (PLAYER_X, _lean_y(0.050, SEAT_Z + 0.62), SEAT_Z + 0.62),
    "head_top":   (PLAYER_X, _lean_y(0.090, SEAT_Z + 0.92), SEAT_Z + 0.92),
    "shoulder_y": _lean_y(0.030, SEAT_Z + 0.59), "shoulder_z": SEAT_Z + 0.59,
    # The wrists rest exactly where build_hands parks the hand rigs, so the arms
    # are BUILT in the pose they will play in: the hand armature's origin plus
    # the offset to its "wrist" bone head, which is what animate_pianist aims
    # the arm IK at.
    "wrist_L":    (REST_LOCATION["L"][0], REST_LOCATION["L"][1] + WRIST_BONE_Y[0],
                   REST_LOCATION["L"][2]),
    "wrist_R":    (REST_LOCATION["R"][0], REST_LOCATION["R"][1] + WRIST_BONE_Y[0],
                   REST_LOCATION["R"][2]),
}

# Which way is "outboard" for each arm/leg. The pianist faces +Y, so the LEFT
# side is at -x -- the mirror of the standing players, who face -Y.
OUT_X = {"L": -1.0, "R": 1.0}

# Direction the elbow bulges off the shoulder->wrist line, and the whole reason
# this rig sits where it does. Pulled DOWN and BACK, it hangs the upper arm
# almost vertically from the shoulder and leaves the forearm running nearly
# level (+y) into the keys -- classic piano posture, and the only way the wrist
# stays straight, because the hand rigs are axis-locked with their fingers along
# +y and cannot rotate to meet a steep forearm. With the elbow instead pulled up
# or forward the forearm dives onto the keys and the wrist breaks ~50 deg.
# The small outboard term keeps the elbows off the ribs without winging them.
# It is a real trade: this demo's right hand plays as far in as the player's own
# centre line, so its arm reaches across the body and the outboard push is what
# stops the forearm cutting into the waist -- but every degree of it also opens
# the wrist. Measured over the take, 0.10 leaves the arm ~12 mm inside the torso
# at the worst cross-body frame, 0.20 brings it to touching (~5 mm, i.e. the arm
# resting against the ribs, as it does in life) at 27 deg of wrist, and 0.30
# clears by 5 mm but at 30 deg of wrist and elbows flared 0.34 m off centre.
ELBOW_BEND = (0.20, -0.8, -0.6)
# Knees bulge forward and up (a seated thigh is near horizontal, so its hinge
# direction has to be stated or the solver may fold the leg the wrong way).
KNEE_BEND = (0.0, 0.6, 0.8)

# How far BACK from the knee the ankle is tucked. A seated player's shins slope
# back under the bench rather than dropping straight -- and here they must: the
# piano's stretcher bar spans the full keyboard width at y = 0.13..0.17, only a
# few centimetres off the floor, so feet planted straight down under the knees
# would stand in it. At 0.16 the toes stop at y = 0.06 and the shoe box (which
# overhangs the toe) ends at y = 0.105, clear of the bar by 2.5 cm.
FOOT_TUCK = 0.16
ANKLE_Z = FLOOR_Z + 0.085     # ankle height of a shod foot standing on the floor
TOE_REACH = 0.16              # ankle -> toe, running LEVEL: a seated player's
#                               foot is flat on the floor, not pitched down onto
#                               a toe like the standing players' (whose sloping
#                               foot bone sinks the shoe box's front corners
#                               below the floor plane -- invisible on them,
#                               obvious on a figure sat next to a piano leg)
SHOE_H = 0.06                 # shoe box height; its sole is set on FLOOR_Z

# Per-side IK pole angles are not constants here: Blender measures the pole
# angle from a reference frame that depends on the chain root's roll, so it is
# SOLVED against the built rest pose (see _solve_pole_angle) and the rig stays
# correct if ANTHRO or POSE change.
POLE_DISTANCE = 0.45          # how far the pole empty sits off the rest joint

# --- Joint range-of-motion limits (AAOS human norms; degrees) ----------------
# ELBOW and KNEE are true 1-DOF hinges: they flex freely but must never
# hyperextend, so they get real anatomical caps on the IK hinge (local Z), and
# no twist/sideways bend (locked X/Y). Everything here is the INTERIOR joint
# angle -- 180 deg at full extension -- and the caps are applied relative to the
# built rest pose's interior angle (_REST_BEND).
#
# SIGN, measured in-scene rather than assumed: on these chains local +Z FLEXES
# the joint (interior = rest - z), so the flexion floor becomes the bone's ik_MAX
# and full extension its ik_MIN. Written the other way round -- which reads more
# natural and is how the standing players' builders have it -- the cage is
# inverted: the elbow then folds past the anatomical floor to 15 deg while being
# blocked from straightening beyond 136. Verified after the fix by pulling the
# wrist target out of reach and hard in: the elbow stops at exactly 178.0 and
# 30.0 deg.
ELBOW_FLEX_MIN = 30.0     # tightest interior angle (~150 deg of flexion)
ELBOW_STRAIGHT = 178.0    # interior at full extension (no hyperextension)
KNEE_FLEX_MIN = 35.0      # tightest interior angle (~145 deg of flexion)
KNEE_STRAIGHT = 178.0     # interior at full extension (no hyperextension)

# SHOULDER (upper_arm) and HIP (thigh) are positioned by the IK target + pole,
# so their orientation is a consequence of where the hand/foot goes. Rather than
# tight symmetric caps we cage each in a human-plausible envelope (per side, in
# the bone's local IK frame) so the solver can never pick an anatomically
# impossible branch, while leaving the full natural range free. Degrees, (min,
# max) about local X, Y, Z.
SHOULDER_IK_LIMIT = {
    "L": {"x": (-90.0, 130.0), "y": (-90.0, 90.0), "z": (-100.0, 100.0)},
    "R": {"x": (-90.0, 130.0), "y": (-90.0, 90.0), "z": (-100.0, 100.0)},
}
# A seated hip is already folded ~90 deg, so the flexion end of the envelope is
# opened up accordingly; the rest matches the standing players.
HIP_IK_LIMIT = {
    "L": {"x": (-130.0, 45.0), "y": (-45.0, 45.0), "z": (-45.0, 60.0)},
    "R": {"x": (-130.0, 45.0), "y": (-45.0, 45.0), "z": (-60.0, 45.0)},
}

# Soft rotation envelopes (deg, local X/Y/Z) on the FK joints -- the spine, neck,
# head, ankles and wrists -- so hand-authored poses stay in a human range. These
# are LIMIT_ROTATION constraints (they clamp the final pose), not IK limits.
# The X ranges are MIRRORED from the standing players': flexion (bending toward
# the instrument, and looking down at the keys) is local -X on this +Y-facing
# figure, so it gets the generous end of the envelope and extension the tight
# one. Copying the standing bounds verbatim would have let the pianist lean 40
# deg backwards while capping the forward lean at 25.
SPINE_ROT_LIMIT = {"x": (-40.0, 25.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
NECK_ROT_LIMIT = {"x": (-45.0, 40.0), "y": (-60.0, 60.0), "z": (-50.0, 50.0)}
ANKLE_ROT_LIMIT = {"x": (-35.0, 40.0), "y": (-15.0, 15.0), "z": (-20.0, 20.0)}
WRIST_ROT_LIMIT = {"x": (-70.0, 70.0), "y": (-30.0, 30.0), "z": (-80.0, 80.0)}

# --- Stool ------------------------------------------------------------------
# A minimalist bench: one slab on four square posts, in the piano's own matte
# black, so it reads as part of the instrument rather than as furniture.
STOOL_SIZE = (0.62, 0.34, 0.05)   # width, depth, slab thickness
STOOL_POST = 0.04                 # square section of each leg

# Elbow/knee rest bends (degrees), filled in while building; used to place each
# hinge's IK flexion limit just short of straight so it cannot hyperextend.
_REST_BEND = {}


def shoulder(side):
    dx = ANTHRO["shoulder_dx"] * OUT_X[side]
    return (PLAYER_X + dx, POSE["shoulder_y"], POSE["shoulder_z"])


def hip(side):
    dx = ANTHRO["hip_dx"] * OUT_X[side]
    return (PLAYER_X + dx, POSE["pelvis"][1], POSE["pelvis"][2])


def _solve_mid(a, b, l1, l2, bend):
    """Place the middle joint of a two-bone chain so the two bones are exactly
    l1 (a->mid) and l2 (mid->b), with the joint bulging toward `bend` (the
    world-space direction the elbow/knee should point). Endpoints a and b are
    fixed. If the span is too long to bend, the joint lands on the a->b line."""
    A, B = V(a), V(b)
    d = (B - A).length
    if d >= l1 + l2 or d < 1e-6:                       # can't bend: point along line
        return tuple(A + (B - A) * (l1 / max(d, 1e-6)))
    x = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)         # along-axis distance to joint
    h = math.sqrt(max(0.0, l1 * l1 - x * x))            # perpendicular bulge
    axis = (B - A) / d
    bd = V(bend)
    perp = bd - bd.dot(axis) * axis                     # bend dir, made perpendicular
    perp = perp.normalized() if perp.length > 1e-6 else axis.orthogonal().normalized()
    return tuple(A + axis * x + perp * h)


def _bulge_dir(a, mid, b):
    """Unit direction the middle joint bulges off the a->b line -- where the IK
    pole empty is parked so it pulls the joint the way it is already built."""
    A, B, M = V(a), V(b), V(mid)
    axis = (B - A)
    axis = axis.normalized() if axis.length > 1e-6 else V((0.0, 0.0, 1.0))
    off = (M - A) - (M - A).dot(axis) * axis
    return off.normalized() if off.length > 1e-6 else axis.orthogonal().normalized()


def solve_seated_leg(hip_pos):
    """Ankle and toe of one seated leg, from the anthropometric segment lengths
    and the floor. The shin is slanted back by FOOT_TUCK (see above), which fixes
    the knee height; the thigh then reaches forward from the hip by whatever run
    is left over. Only the ANKLE is returned as an endpoint -- the knee is left to
    _solve_mid (with KNEE_BEND), which is what guarantees both bones keep their
    exact lengths."""
    drop = math.sqrt(max(0.0, ANTHRO["shin"] ** 2 - FOOT_TUCK ** 2))
    knee_z = ANKLE_Z + drop
    run = math.sqrt(max(0.0, ANTHRO["thigh"] ** 2 - (hip_pos[2] - knee_z) ** 2))
    ankle = (hip_pos[0], hip_pos[1] + run - FOOT_TUCK, ANKLE_Z)
    toe = (hip_pos[0], ankle[1] + TOE_REACH, ANKLE_Z)   # level: foot flat on the floor
    return ankle, toe


# ---------------------------------------------------------------------------
# Materials + meshes
# ---------------------------------------------------------------------------
def _mat(name, color, rough, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def _box(name, sx, sy, sz, bevel=0.006):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, affect='EDGES')
    bm.to_mesh(mesh); bm.free()
    return mesh


def _shoe_mesh(name):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.10, ANTHRO["foot"], SHOE_H), verts=bm.verts)
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.014, segments=2, affect='EDGES')
    bm.to_mesh(mesh); bm.free()
    return mesh


def _part(arm, coll, mat, bone, name, location, size=None, rotation=(0, 0, 0), mesh=None):
    """A rigid box (or supplied mesh) parented to `bone`. Bone-parent space has
    its origin at the bone tail with local -Y running toward the head."""
    if mesh is None:
        mesh = _box(name + "Mesh", *size)
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = bone
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(mat)
    return obj


def _bone_upright(arm, coll, mat, bone, name, center, size):
    """An upright, world-aligned box that FOLLOWS a bone -- so the torso/head
    rotate with the spine while still reading as square boxes (not bone-tilted).
    Parent to the bone, then stamp the desired world matrix; Blender back-computes
    the local transform against the bone's rest pose."""
    obj = bpy.data.objects.new(name, _box(name + "Mesh", *size))
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = mathutils.Matrix.Translation(V(center))
    return obj


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------
def _build_skeleton(coll):
    arm_data = bpy.data.armatures.new("PianistRig")
    arm = bpy.data.objects.new("Pianist", arm_data)
    coll.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = V(head), V(tail)
        if parent is not None:
            b.parent = parent
            b.use_connect = connect
        return b

    # The PELVIS is the root: the seated figure's spine rises from it and both
    # legs hang off it, so a lean or a hip shift carries the whole body.
    px, py, pz = POSE["pelvis"]
    pelvis = bone("pelvis", (px, py, pz), (px, py, pz + 0.09))
    spine = bone("spine", POSE["pelvis"], POSE["chest"], pelvis, False)
    chest = bone("chest", POSE["chest"], POSE["neck"], spine, True)
    nx, ny, nz = POSE["neck"]
    neck = bone("neck", (nx, ny, nz),
                (nx, _lean_y(0.070, nz + 0.08), nz + 0.08), chest, True)
    bone("head", neck.tail, POSE["head_top"], neck, True)

    rest = {}
    for side in ("L", "R"):
        s = V(shoulder(side))
        outx = OUT_X[side]
        wrist = V(POSE[f"wrist_{side}"])
        # Elbow: solved so upper_arm + forearm are exact, hanging down-and-back.
        elbow = V(_solve_mid(s, wrist, ANTHRO["upper_arm"], ANTHRO["forearm"],
                             (ELBOW_BEND[0] * outx,) + ELBOW_BEND[1:]))
        # Hand stub: continues along the fingers (+y, arching gently down) --
        # the frame the axis-locked piano hand rigs are authored in.
        hdir = V((0.0, 0.97, -0.25)).normalized()
        clav = bone(f"clav.{side}", POSE["neck"], tuple(s), chest, False)
        up = bone(f"upper_arm.{side}", tuple(s), tuple(elbow), clav, False)
        fore = bone(f"forearm.{side}", tuple(elbow), tuple(wrist), up, True)
        hand_b = bone(f"hand.{side}", tuple(wrist),
                      tuple(wrist + hdir * ANTHRO["hand"]), fore, True)
        # Roll BOTH hand stubs to the same reference (local Z up) so the external
        # Hand_L / Hand_R rigs attach with a consistent frame on each side.
        hand_b.align_roll(V((0.0, 0.0, 1.0)))
        # Roll the forearm so its local Z is the elbow's hinge axis (perpendicular
        # to the rest arm plane); the IK limits below then make the elbow a one-way
        # hinge that cannot twist or bend backwards.
        hinge = (elbow - s).cross(wrist - elbow)
        if hinge.length > 1e-6:
            fore.align_roll(hinge.normalized())
        # INTERIOR angle at the elbow (180 = straight), which is what the flexion
        # caps below are expressed in. `.angle()` returns the exterior turn, so
        # it is subtracted from 180 rather than used raw.
        _REST_BEND[f"arm_{side}"] = 180.0 - math.degrees((elbow - s).angle(wrist - elbow))
        rest[f"arm_{side}"] = (tuple(s), tuple(elbow), tuple(wrist))

        # Leg: hip -> knee -> ankle -> toe, solved for a seated pose.
        h = V(hip(side))
        ankle_p, toe_p = solve_seated_leg(tuple(h))
        knee = V(_solve_mid(tuple(h), ankle_p, ANTHRO["thigh"], ANTHRO["shin"],
                            KNEE_BEND))
        thigh = bone(f"thigh.{side}", tuple(h), tuple(knee), pelvis, False)
        shin = bone(f"shin.{side}", tuple(knee), ankle_p, thigh, True)
        foot = bone(f"foot.{side}", ankle_p, toe_p, shin, True)
        # Roll the shin so its local Z is the knee's hinge axis (one-way hinge).
        khinge = (knee - h).cross(V(ankle_p) - knee)
        if khinge.length > 1e-6:
            shin.align_roll(khinge.normalized())
        # Feet share a predictable local +Z-up frame, so one ankle-press sign
        # works on either side (and on a future sustain-pedal pass).
        foot.align_roll(V((0.0, 0.0, 1.0)))
        _REST_BEND[f"leg_{side}"] = 180.0 - math.degrees((knee - h).angle(V(ankle_p) - knee))
        rest[f"leg_{side}"] = (tuple(h), tuple(knee), ankle_p)

    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
    return arm, rest


def _solve_pole_angle(arm, mid_bone, ik, rest_mid, coarse=5.0, fine=0.5):
    """Find the IK pole angle that reproduces the built rest pose.

    Blender measures the pole angle from a reference frame that depends on the
    chain root's roll, so there is no closed form worth hand-tuning: instead spin
    the angle through a full turn, evaluate the rig, and keep whatever puts the
    middle joint (elbow / knee) back where _solve_mid placed it. That makes the
    rig correct by construction -- change ANTHRO, the seat height or ELBOW_BEND
    and the poles re-solve instead of silently twisting the limb about its own
    axis. Returns (angle_rad, residual_m).

    NOTE the sweep runs -180..180, not 0..360: Blender HARD-CLAMPS pole_angle to
    +/-pi, so a 0..360 sweep silently pins everything past 180 to the same pose
    and finds only whichever half-circle happens to contain 0."""
    pb = arm.pose.bones[mid_bone]
    target = V(rest_mid)

    def err(deg):
        ik.pole_angle = math.radians(deg)
        bpy.context.view_layer.update()
        return ((arm.matrix_world @ pb.head) - target).length

    steps = int(360.0 / coarse)
    best = min(((err(-180.0 + c * coarse), -180.0 + c * coarse)
                for c in range(steps + 1)))
    lo, hi = best[1] - coarse, best[1] + coarse
    best = min([best] + [(err(lo + k * fine), lo + k * fine)
                         for k in range(int((hi - lo) / fine) + 1)
                         if -180.0 <= lo + k * fine <= 180.0])
    ik.pole_angle = math.radians(best[1])
    bpy.context.view_layer.update()
    return math.radians(best[1]), best[0]


def _add_targets(coll, arm, rest):
    """Target empties + constraints. Arms and legs are each a two-bone IK chain
    reaching a wrist/ankle empty with an elbow/knee pole; the FK joints (spine,
    neck, head, ankles, wrists) get soft rotation limits. Each pole is parked in
    the direction its joint already bulges, then its angle is solved against the
    rest pose."""
    empties = {}

    def add_empty(name, pos):
        e = bpy.data.objects.new(name, None)
        e.empty_display_size = 0.05
        e.location = pos
        coll.objects.link(e)
        empties[name] = e
        return e

    residuals = {}
    for side in ("L", "R"):
        s, elbow, wrist = rest[f"arm_{side}"]
        h, knee, ankle = rest[f"leg_{side}"]
        add_empty(f"Wrist_{side}", wrist)
        add_empty(f"Elbow_{side}",
                  tuple(V(elbow) + _bulge_dir(s, elbow, wrist) * POLE_DISTANCE))
        add_empty(f"Ankle_{side}", ankle)
        add_empty(f"Knee_{side}",
                  tuple(V(knee) + _bulge_dir(h, knee, ankle) * POLE_DISTANCE))

    for side in ("L", "R"):
        # --- Arm IK: forearm chain (upper_arm + forearm) reaches the wrist. ----
        pbF = arm.pose.bones[f"forearm.{side}"]
        ik = pbF.constraints.new('IK')
        ik.target = empties[f"Wrist_{side}"]
        ik.pole_target = empties[f"Elbow_{side}"]
        ik.chain_count = 2
        # Elbow = one-way hinge: no twist (Y) or sideways bend (X); flexion (Z)
        # folds freely but stops just short of straight (no hyperextension).
        pbF.lock_ik_x = True
        pbF.lock_ik_y = True
        pbF.use_ik_limit_z = True
        rb = _REST_BEND[f"arm_{side}"]
        pbF.ik_min_z = math.radians(rb - ELBOW_STRAIGHT)
        pbF.ik_max_z = math.radians(rb - ELBOW_FLEX_MIN)
        # Shoulder: cage the IK rotation to a human-plausible envelope.
        _cage_ik(arm.pose.bones[f"upper_arm.{side}"], SHOULDER_IK_LIMIT[side])
        _, residuals[f"arm_{side}"] = _solve_pole_angle(
            arm, f"forearm.{side}", ik, rest[f"arm_{side}"][1])

        # --- Leg IK: shin chain (thigh + shin) reaches the ankle. -------------
        pbS = arm.pose.bones[f"shin.{side}"]
        lik = pbS.constraints.new('IK')
        lik.target = empties[f"Ankle_{side}"]
        lik.pole_target = empties[f"Knee_{side}"]
        lik.chain_count = 2
        pbS.lock_ik_x = True
        pbS.lock_ik_y = True
        pbS.use_ik_limit_z = True
        lb = _REST_BEND[f"leg_{side}"]
        pbS.ik_min_z = math.radians(lb - KNEE_STRAIGHT)
        pbS.ik_max_z = math.radians(lb - KNEE_FLEX_MIN)
        # Hip: cage the IK rotation to a human-plausible envelope.
        _cage_ik(arm.pose.bones[f"thigh.{side}"], HIP_IK_LIMIT[side])
        _, residuals[f"leg_{side}"] = _solve_pole_angle(
            arm, f"shin.{side}", lik, rest[f"leg_{side}"][1])

        # --- Soft FK limits on the ankle and wrist. ---------------------------
        _limit_rot(arm.pose.bones[f"foot.{side}"], ANKLE_ROT_LIMIT)
        _limit_rot(arm.pose.bones[f"hand.{side}"], WRIST_ROT_LIMIT)

    # Spine / neck / head soft rotation limits.
    _limit_rot(arm.pose.bones["spine"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["chest"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["neck"], NECK_ROT_LIMIT)
    return empties, residuals


def _cage_ik(pbone, limit):
    """Cap a bone's IK rotation to a per-axis envelope (degrees)."""
    for axis in ("x", "y", "z"):
        setattr(pbone, f"use_ik_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(pbone, f"ik_min_{axis}", math.radians(lo))
        setattr(pbone, f"ik_max_{axis}", math.radians(hi))


def _limit_rot(pbone, limit):
    """Add a LOCAL-space LIMIT_ROTATION constraint (degrees) to an FK bone."""
    lr = pbone.constraints.new('LIMIT_ROTATION')
    lr.owner_space = 'LOCAL'
    for axis in ("x", "y", "z"):
        setattr(lr, f"use_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(lr, f"min_{axis}", math.radians(lo))
        setattr(lr, f"max_{axis}", math.radians(hi))


# ---------------------------------------------------------------------------
# Body meshes
# ---------------------------------------------------------------------------
def _clothe(arm, coll):
    skin = _mat("PianistBody", (0.30, 0.33, 0.38), 0.6)   # neutral grey mock
    dark = _mat("PianistDark", (0.13, 0.14, 0.17), 0.6)
    rubber = _mat("PianistShoe", (0.04, 0.04, 0.05), 0.6)

    aw = ANTHRO["arm_thick"]
    lw = ANTHRO["leg_thick"]
    L1, fore = ANTHRO["upper_arm"], ANTHRO["forearm"]
    thigh, shin = ANTHRO["thigh"], ANTHRO["shin"]

    # Torso / head: upright boxes that follow the spine bones (so a lean/sway
    # carries them) while reading as square blocks. The pelvis block sits ON the
    # bench (its underside at SEAT_Z) and is deeper than the chest, the seated
    # figure's weight being back on the buttocks.
    _bone_upright(arm, coll, dark, "pelvis", "Torso_Pelvis",
                  (PLAYER_X, HIP_Y - 0.05, SEAT_Z + 0.10), (ANTHRO["torso_w"], 0.22, 0.20))
    # The torso is TWO boxes, not one slab, and the waist is the reason. A
    # pianist's forearm passes in front of the abdomen whenever that hand plays
    # near the body's centre line; with a single full-depth, full-width block
    # down to the waist the right forearm cut ~2 cm through its front corner on
    # every cross-body reach, and no elbow-pole angle fixes that (the elbow is
    # not what is clipping). A real waist is narrower and set back, which is
    # exactly the clearance the arm needs -- so the ribcage sits above z = 0.36
    # and a slimmer abdomen bridges it to the pelvis.
    _bone_upright(arm, coll, skin, "spine", "Torso_Waist",
                  (PLAYER_X, _lean_y(-0.010, SEAT_Z + 0.265), SEAT_Z + 0.265),
                  (0.26, 0.17, 0.15))
    # Stack closes with no gaps: waist 0.19..0.34 above the seat, chest
    # 0.33..0.60 (its top reaching the shoulder joints at 0.59, so the clavicle
    # bars sit ON it), neck 0.585..0.6925, head 0.6925..0.92.
    _bone_upright(arm, coll, skin, "spine", "Torso_Chest",
                  (PLAYER_X, _lean_y(0.015, SEAT_Z + 0.465), SEAT_Z + 0.465),
                  (ANTHRO["torso_w"], ANTHRO["torso_d"], 0.27))
    _bone_upright(arm, coll, skin, "neck", "Neck",
                  (PLAYER_X, _lean_y(0.055, SEAT_Z + 0.639), SEAT_Z + 0.639),
                  (0.10, 0.10, 0.108))
    _head_z = SEAT_Z + 0.92 - ANTHRO["head"][2] / 2.0
    _bone_upright(arm, coll, skin, "head", "Head",
                  (PLAYER_X, _lean_y(0.050, _head_z), _head_z),
                  ANTHRO["head"])   # crown at seated stature

    for side in ("L", "R"):
        # Clavicle bar from the base of the neck out to the shoulder joint --
        # the shoulders sit 0.227 m off the centre line but the chest is only
        # 0.16 m half-wide, so without it each arm hangs off nothing.
        clav_len = arm.pose.bones[f"clav.{side}"].bone.length
        _part(arm, coll, skin, f"clav.{side}", f"Shoulder_{side}",
              (0, -clav_len / 2.0, 0), (aw * 1.15, clav_len * 0.9, aw * 0.95))
        _part(arm, coll, skin, f"upper_arm.{side}", f"UpperArm_{side}",
              (0, -L1 / 2.0, 0), (aw, L1 * 0.92, aw))
        _part(arm, coll, skin, f"forearm.{side}", f"Forearm_{side}",
              (0, -fore / 2.0, 0), (aw * 0.9, fore * 0.9, aw * 0.9))
        # Hand stub box: a fist-sized block at the wrist -- a placeholder the
        # piano hand rig sits over (animate_pianist hides it). Named Fist_*, NOT
        # Hand_* like the standing players': the piano's hand ARMATURES are
        # Hand_L / Hand_R, and Blender would rename whichever lost the race.
        _part(arm, coll, skin, f"hand.{side}", f"Fist_{side}",
              (0, -ANTHRO["hand"] / 2.0, 0), (aw * 1.1, ANTHRO["hand"] * 0.8, aw * 0.7))

        _part(arm, coll, dark, f"thigh.{side}", f"Thigh_{side}",
              (0, -thigh / 2.0, 0), (lw, thigh * 0.92, lw))
        _part(arm, coll, dark, f"shin.{side}", f"Shin_{side}",
              (0, -shin / 2.0, 0), (lw * 0.85, shin * 0.92, lw * 0.85))
        # The foot bone is level (see TOE_REACH) and rolled local-+Z-up, so the
        # shoe's local z offset drops it exactly onto the floor: the bone's tail
        # sits ANKLE_Z above FLOOR_Z, and the box is half its own height thick.
        _part(arm, coll, rubber, f"foot.{side}", f"Shoe_{side}",
              (0, -ANTHRO["foot"] / 2.0 + 0.045, -(ANKLE_Z - FLOOR_Z) + SHOE_H / 2.0),
              mesh=_shoe_mesh(f"Shoe_{side}Mesh"))


def _build_stool(coll):
    """A minimalist bench: one slab on four square posts, in the piano's matte
    black so it belongs to the instrument. Sized and placed so its front edge
    clears the sloping thighs and its legs clear the player's shins."""
    seat = _mat("PianoStool", (0.02, 0.02, 0.023), 0.4)
    w, d, t = STOOL_SIZE
    top = bpy.data.objects.new("Stool_Seat", _box("Stool_SeatMesh", w, d, t))
    top.location = (PLAYER_X, SEAT_Y, SEAT_Z - t / 2.0)
    top.data.materials.append(seat)
    coll.objects.link(top)

    post_h = (SEAT_Z - t) - FLOOR_Z
    for sx in (-1, 1):
        for sy in (-1, 1):
            name = f"Stool_Leg_{'L' if sx < 0 else 'R'}{'B' if sy < 0 else 'F'}"
            leg = bpy.data.objects.new(
                name, _box(name + "Mesh", STOOL_POST, STOOL_POST, post_h, bevel=0.004))
            leg.location = (PLAYER_X + sx * (w / 2.0 - STOOL_POST),
                            SEAT_Y + sy * (d / 2.0 - STOOL_POST),
                            FLOOR_Z + post_h / 2.0)
            leg.data.materials.append(seat)
            coll.objects.link(leg)


def build_pianist():
    scene = bpy.context.scene
    old = bpy.data.collections.get("Pianist")
    if old is not None:
        # Remove every object, not just the meshes: the IK target/pole EMPTIES
        # would otherwise survive and squat their own names on a rebuild.
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Pianist")
    scene.collection.children.link(coll)

    _REST_BEND.clear()
    arm, rest = _build_skeleton(coll)
    _, residuals = _add_targets(coll, arm, rest)
    _clothe(arm, coll)
    _build_stool(coll)

    bpy.context.view_layer.update()
    worst = max(residuals.values()) if residuals else 0.0
    print(f"Built pianist: armature '{arm.name}' with {len(arm.pose.bones)} bones, "
          f"IK arms + legs, shoes, stool; pole angles solved to "
          f"{worst * 1000:.2f} mm of the rest pose")
    return coll


if __name__ == "__main__":
    build_pianist()
