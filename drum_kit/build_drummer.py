"""Builds a blocky, faceless, minimalist humanoid drummer to sit in for a
real character model later on.

Run inside Blender AFTER build_drum_kit.py to create a "Drummer" collection:
a single ``Drummer`` armature (average adult proportions, seated at the kit)
with rigid box meshes on every bone, drumsticks held in the hands, shoes on
the feet, and a simple throne.

Why a body at all: the sticks must pivot about the drummer's shoulders (the
seat is the centre of lateral rotation), so a stick reaching the floor tom
swings out from the shoulder instead of staying pointed at the audience. Each
arm has a wrist joint: a two-bone IK chain (upper arm + forearm) reaches a
``Wrist_*`` empty, and a separate ``hand.*`` bone carrying the drumstick
Damped-Tracks the ``IK_Hand_*`` (stick-tip) empty. So the elbow hangs at a
natural height while the wrist angles the stick down onto the head -- without
the wrist joint the rigid stick-arm over-folds on close drums and the elbow
rides up at shoulder level. The animator moves the tip empty to each strike
point (and derives the wrist from it via ``wrist_target``); Blender's IK solves
the shoulder, elbow and wrist, so the whole arm reorients toward each target.

Because the reach is solved by IK, the animation is authored purely as
*target areas to hit at target velocities* (the planner's strike points and
MIDI velocities). Swap ``ANTHRO`` for a different body and the same targets
re-solve against the new proportions -- the point of a stand-in rig.

Feet: ``foot.L`` / ``foot.R`` bones carry the shoes and press the pedals; the
animator rotates the ankles (and the matching footboards) for kicks and hats.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_drummer", "/path/to/drum_kit/build_drummer.py")
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

V = mathutils.Vector

# ---------------------------------------------------------------------------
# Body proportions (average adult, metres). Swap this dict to re-shape the
# stand-in; the arms are IK so the drum targets re-solve automatically.
# ---------------------------------------------------------------------------
# Average adult proportions from the Drillis & Contini / Winter anthropometric
# tables (segment length as a fraction of stature H), for H = 1.75 m:
#   upper arm 0.186 H, forearm 0.146 H, thigh 0.245 H, shank 0.246 H,
#   biacromial (shoulder) breadth 0.259 H, head height ~0.130 H.
_H = 1.75
ANTHRO = {
    "upper_arm": 0.186 * _H,      # 0.326  shoulder -> elbow
    "forearm": 0.146 * _H,        # 0.256  elbow -> wrist
    "stick": 0.406,               # 16" (standard 5A drumstick)
    "thigh": 0.245 * _H,          # 0.429  hip -> knee
    "shin": 0.246 * _H,           # 0.431  knee -> ankle
    "shoulder_dx": 0.259 * _H / 2.0,   # 0.227  half biacromial breadth
    "hip_dx": 0.10,               # half hip-joint spacing
    "arm_thick": 0.06,
    "leg_thick": 0.10,
    "torso_w": 0.34, "torso_d": 0.21,
    "head": (0.155, 0.20, 0.130 * _H),  # breadth, depth, height (~0.228)
    "stick_r": 0.0075,            # ~15 mm dia, a real stick
}

# Seated joint positions (world, metres). The drummer sits at +Y facing -Y;
# left = +X, right = -X. Feet land on the kick (x=0) and hi-hat (x~0.55)
# pedals; hips sit behind on a throne; shoulders lean slightly in over the kit.
POSE = {
    # Seated high enough that the drums sit below the shoulders, so the arms
    # angle down to play (hands well below shoulder height, relaxed).
    "pelvis":     (0.0, 0.82, 0.60),
    "chest":      (0.0, 0.81, 1.12),   # upper spine essentially vertical over the
    "neck":       (0.0, 0.81, 1.21),   # hips so the drummer sits upright; the
    "head_top":   (0.0, 0.80, 1.44),   # forward lean is added only to reach (§torso)
    "shoulder_y": 0.72, "shoulder_z": 1.20,
    # Knees are solved from the anthropometric thigh/shin lengths (see
    # _solve_knee), so only the hip and the pedal-planted ankles are fixed here.
    "ankle_R":    (0.0, 0.42, 0.10),   # over the kick footboard
    "ankle_L":    (0.55, 0.48, 0.10),  # over the hi-hat footboard
    # Toes rest HIGHER than the ankles so the foot slopes up toward the toe,
    # sitting flush on the up-sloped pedal footboards (see PEDAL_SLOPE). The left
    # toe aims straight at the hi-hat stand centre (0.55, 0.32) so foot + pedal
    # line up with the stand.
    "toe_R":      (0.0, 0.24, 0.145),
    "toe_L":      (0.55, 0.32, 0.145),
    "throne":     (0.0, 0.90, 0.57),
    # Elbow pole targets. Real drummers keep the elbows LOW and TUCKED to the
    # sides, reaching with the forearm/wrist rather than lifting or winging the
    # elbow. Far-out poles (way off to each side) winged the elbows outboard --
    # the left splayed ~0.26 m past its shoulder on the backbeat, and the right
    # rode up on cross-body reaches. Poles pulled IN close to the body, forward
    # and just below, with the pole angles below, keep both elbows hanging at
    # the sides across the whole kit (measured worst case still below the
    # shoulder; mean outboard splay ~0). See RESEARCH.md "elbows hang / tuck".
    "pole_R":     (-0.40, 1.00, -0.30),
    "pole_L":     (0.20, 0.90, -0.30),
}

# Seat offset: shift the whole seated UPPER body (pelvis/spine/shoulders/head/
# throne) toward the hi-hat side (+X) and slightly forward (-Y) so the cross-body
# reaches shorten and the arms play with lower, more folded elbows (like a real
# player set close to the kit). The FEET stay planted on their pedals and the
# knees re-solve, so only the seat moves, not the kick/hi-hat pedal geometry.
SEAT_DX = 0.04
SEAT_DY = -0.02
# Yaw the seated TORSO (chest/shoulders/neck/head) about the vertical axis through
# the pelvis so the upper body faces the snare / hi-hat side (both on +X here)
# instead of squaring up to the kick -- the standard drum-ergonomics setup. Only
# the torso twists: the hips/legs stay square to the pedals (see _seat_sq), just
# like a real player who turns their upper body but keeps their feet planted.
SEAT_YAW = math.radians(18.0)
_SEAT_PIVOT_Y = 0.82          # pelvis Y: the vertical axis the torso turns about


def _seat_sq(p):
    """Lower-body seat map: the seat OFFSET only, no yaw -- so the pelvis, hips,
    legs and throne stay square to the pedals."""
    return (p[0] + SEAT_DX, p[1] + SEAT_DY, p[2])


def _seat(p):
    """Upper-body seat map: YAW the point about the vertical axis through the
    pelvis (so the torso/shoulders face the snare/hi-hat side), then apply the
    seat offset. The hips/feet use _seat_sq, so the torso twists over square legs."""
    dx, dy = p[0], p[1] - _SEAT_PIVOT_Y
    c, s = math.cos(SEAT_YAW), math.sin(SEAT_YAW)
    rx = dx * c - dy * s
    ry = dx * s + dy * c + _SEAT_PIVOT_Y
    return (rx + SEAT_DX, ry + SEAT_DY, p[2])

# Per-arm IK elbow pole angle. The arms reach different targets (the right
# crosses the body far more), so they are not mirror-symmetric; these angles
# roll each elbow down and in so it hangs tucked below the shoulder, and remove
# the frame-to-frame elbow "snap" the old raised pose caused mid-fill. The
# poles sit *behind* the shoulder (pole y > shoulder y) so the elbow hangs back,
# not forward across the chest (the over-flexed rest pose we saw at frame 1).
POLE_ANGLE = {"L": math.radians(45.0), "R": math.radians(130.0)}
# Knee pole angles (per leg) for the leg IK, tuned so the IK reproduces the
# square-to-pedal rest legs from _solve_knee.
LEG_POLE_ANGLE = {"L": math.radians(70.0), "R": math.radians(-10.0)}

# --- Joint range-of-motion limits (AAOS human norms; see RESEARCH.md) --------
# The ELBOW is a true 1-DOF hinge, so it gets real anatomical limits: flexion
# 0-150 deg (never past ~30 deg interior) and no hyperextension (stops just shy
# of straight). Applied on the forearm bone's IK hinge (local Z) below.
ELBOW_FLEX_MIN = 30.0     # tightest interior angle (150 deg flexion)
ELBOW_STRAIGHT = 178.0    # interior at full extension (0 deg, no hyperextension)

# The SHOULDER (upper_arm) is positioned by the IK target + pole, so its
# orientation is a consequence of where the drum is; the WRIST (hand) is aimed
# at the stick tip by a Damped Track. Neither is a free DOF, so instead of tight
# symmetric caps (which would forbid the large but real cross-body reach to the
# hi-hat) we cage each in a human-plausible envelope, per side, in the bone's
# local IK frame. These contain the natural seated + cross-arm motion with
# margin, so they block anatomically-impossible IK branches (e.g. the shoulder
# over-rotating the elbow across the chest, as at frame 1 before) without
# binding normal play. Degrees, (min, max) about local X, Y, Z.
SHOULDER_IK_LIMIT = {
    "L": {"x": (-91.0, 84.0), "y": (-90.0, 17.0), "z": (-16.0, 114.0)},
    "R": {"x": (-94.0, 122.0), "y": (-55.0, 177.0), "z": (-103.0, 100.0)},
}
WRIST_ROT_LIMIT = {
    "L": {"x": (-58.0, 22.0), "y": (-20.0, 45.0), "z": (26.0, 112.0)},
    "R": {"x": (-72.0, 14.0), "y": (-39.0, 43.0), "z": (-127.0, 87.0)},
}

# Elbow rest bend (degrees), filled in while building; used to place the IK
# flexion limit just short of straight so the elbow cannot hyperextend.
_REST_BEND = {}


def _yaw_left_foot():
    """Point the left foot along the ergonomically angled hi-hat pedal, so the
    shoe sits flat on it. The pedal is yawed about its TOE (the stand centre),
    so the foot follows by rotating the ANKLE about the toe -- the toe stays at
    the stand while the heel/ankle swing toward the drummer."""
    a = kit_layout.HIHAT_PEDAL_YAW
    tx, ty, tz = POSE["toe_L"]
    dx, dy = POSE["ankle_L"][0] - tx, POSE["ankle_L"][1] - ty
    POSE["ankle_L"] = (tx + dx * math.cos(a) - dy * math.sin(a),
                       ty + dx * math.sin(a) + dy * math.cos(a), POSE["ankle_L"][2])


_yaw_left_foot()


def shoulder(side):
    dx = ANTHRO["shoulder_dx"] * (1 if side == "L" else -1)
    return _seat((dx, POSE["shoulder_y"], POSE["shoulder_z"]))


def _solve_knee(hip, ankle, toe, thigh_len, shin_len):
    """Place the knee so the thigh and shin bones are exactly their
    anthropometric lengths for a seated leg (hip and ankle fixed, on the pedal),
    with the knee in the PEDAL'S VERTICAL PLANE -- the vertical plane through the
    ankle along the foot's heading. That makes the shin line up straight down the
    pedal (leg square to it) rather than the knee jutting off to the side, while
    the thigh is free to splay from the hip out to the pedal."""
    H, A, T = V(hip), V(ankle), V(toe)
    d = (A - H).length
    if d >= thigh_len + shin_len or d < 1e-6:      # can't bend: point along line
        return tuple(H + (A - H) * (thigh_len / max(d, 1e-6)))
    a = (d * d + thigh_len ** 2 - shin_len ** 2) / (2.0 * d)
    r = math.sqrt(max(0.0, thigh_len ** 2 - a ** 2))
    axis = (A - H) / d
    C = H + a * axis                                # centre of the knee circle
    tmp = V((0.0, 0.0, 1.0)) if abs(axis.z) < 0.9 else V((1.0, 0.0, 0.0))
    e1 = (tmp - tmp.dot(axis) * axis).normalized()  # circle plane basis
    e2 = axis.cross(e1)
    ped = V((T[0] - A[0], T[1] - A[1], 0.0))        # pedal heading (horizontal)
    if ped.length < 1e-6:
        ped = V((0.0, -1.0, 0.0))
    ped.normalize()
    n = V((-ped.y, ped.x, 0.0))                     # normal of the pedal's vert. plane
    # Knee on the circle with (knee - ankle).n == 0  ->  B1 cos + B2 sin = -A0
    A0 = (C - A).dot(n); B1 = r * e1.dot(n); B2 = r * e2.dot(n)
    Rm = math.hypot(B1, B2)
    if Rm > 1e-9 and abs(A0) <= Rm:
        phi = math.atan2(B2, B1); ac = math.acos(max(-1.0, min(1.0, -A0 / Rm)))
        cands = [C + r * (math.cos(t) * e1 + math.sin(t) * e2)
                 for t in (phi + ac, phi - ac)]
    else:                                           # plane out of reach: nearest bulge
        perp = (ped - ped.dot(axis) * axis)
        cands = [C + r * perp.normalized()] if perp.length > 1e-6 else [C + r * e1]
    aim = V((ped.x, ped.y, 0.0)) + V((0.0, 0.0, 1.2))   # prefer knee forward + up
    return tuple(max(cands, key=lambda k: (k - C).dot(aim)))


# --- wrist joint + grip ----------------------------------------------------
# The stick is NOT collinear with the forearm: the arm IK reaches the WRIST
# (the grip), and a separate hand/stick bone aims at the drum. The stick is held
# near its butt end, with only a tiny bit overhanging behind the fist, so almost
# the whole stick (grip->tip) is the working length that swings at the drum.
STICK_BUTT = 0.03                                 # grip -> butt: small overhang
STICK_LEN = ANTHRO["stick"] - STICK_BUTT          # grip -> tip (nearly all of it)
ARM_REACH = ANTHRO["upper_arm"] + ANTHRO["forearm"]
REACH_SOFT = ARM_REACH - 0.02                     # keep the wrist comfortably in reach
# The wrist sits BACK (+Y, toward the drummer) and a little UP from the tip, so
# the stick lies shallow across the surface (~20 deg) - a glancing stroke where
# the side of the tip grazes the head, not a straight-down stab.
WRIST_DIR = mathutils.Vector((0.0, 0.94, 0.34)).normalized()
# How much of the grip placement is the shallow back-and-up direction vs. the
# in-line "toward the shoulder" direction (0..1). Lower keeps the wrist nearer a
# natural cock; see wrist_target.
WRIST_BACK = 0.3


def wrist_target(shoulder_pos, tip_pos):
    """Where the wrist (grip) sits so the stick tip lands on `tip_pos`.

    Ideally the wrist is STICK_LEN back-and-up from the tip (WRIST_DIR), giving
    a shallow stick, and if that point is within reach we use it. If it is out
    of reach (a far cross-body target like the hi-hat), we pull the wrist IN
    along the tip->shoulder line instead of sliding it onto the far reach
    boundary: that is the closest the grip can get to the shoulder while still
    holding the stick, so |shoulder - wrist| is minimised and the elbow keeps a
    relaxed bend rather than locking straight to a distant grip. (Placing the
    wrist on the reach boundary instead left the arm ~97% extended -- a stiff,
    locked-out reach; pulling it in bends the elbow to a natural ~120 deg.)
    |wrist - tip| == STICK_LEN always, so the hand bone that aims from the wrist
    at the tip still lands exactly."""
    S = mathutils.Vector(shoulder_pos)
    P = mathutils.Vector(tip_pos)
    SP = S - P
    d = SP.length
    if d < 1e-6:
        return tuple(S)
    u = SP / d
    # Blend the shallow "back-and-up" grip direction (WRIST_DIR) with the
    # "toward the shoulder" direction (u). Pure back-and-up put the grip far
    # behind the tip, so on a close drum (the snare) the forearm doubled back
    # and the wrist hyper-bent (~140 deg). Mixing in the shoulder direction
    # keeps the grip roughly in line with the forearm, so the wrist sits at a
    # natural cock (~30 deg, like the reference) while the stick still lies
    # fairly shallow across the head.
    direction = (WRIST_BACK * WRIST_DIR + (1.0 - WRIST_BACK) * u).normalized()
    W = P + STICK_LEN * direction
    if (S - W).length <= REACH_SOFT:
        return tuple(W)
    return tuple(P + STICK_LEN * u)  # far target: pull grip toward shoulder


def _rest_tip(side):
    """Where each stick tip hovers at rest: over its convention home."""
    home = kit_layout.strike_point("hihat_closed" if side == "R" else "snare")
    return (home[0], home[1], home[2] + 0.10)


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


def _stick_mesh(name, length, r=0.011):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    c = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=12,
                              radius1=r, radius2=r * 0.55, depth=length)
    # cone is built along +Z; lay it along -Y (toward the tail/tip) and centre
    bmesh.ops.rotate(bm, verts=c["verts"],
                     matrix=mathutils.Matrix.Rotation(math.radians(90), 3, 'X'))
    bmesh.ops.translate(bm, verts=c["verts"], vec=(0, -length / 2.0, 0))
    bead = bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=r * 1.3)
    bmesh.ops.translate(bm, verts=bead["verts"], vec=(0, 0, 0))
    bm.to_mesh(mesh); bm.free()
    return mesh


def _shoe_mesh(name):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(0.09, 0.24, 0.06), verts=bm.verts)
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


# ---------------------------------------------------------------------------
# Armature
# ---------------------------------------------------------------------------
def _build_skeleton(coll):
    arm_data = bpy.data.armatures.new("DrummerRig")
    arm = bpy.data.objects.new("Drummer", arm_data)
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

    # A static PELVIS root carries both the (animated) spine and the legs, so the
    # torso can twist/lean without dragging the legs off the pedals -- rotation
    # lives entirely in the spine (upper body), the legs hang off the still hips.
    px, py, pz = _seat_sq(POSE["pelvis"])
    pelvis = bone("pelvis", (px, py, pz), (px, py, pz + 0.08))
    # Spine runs from the SQUARE pelvis up to the YAWED chest: the torso twists
    # toward the snare over hips that stay square to the pedals.
    spine = bone("spine", _seat_sq(POSE["pelvis"]), _seat(POSE["chest"]), pelvis, False)
    chest = bone("chest", _seat(POSE["chest"]), _seat(POSE["neck"]), spine, True)
    neck_head = _seat(POSE["neck"])
    neck = bone("neck", neck_head, (neck_head[0], neck_head[1] - 0.02,
                                    neck_head[2] + 0.06), chest, True)
    bone("head", neck.tail, _seat(POSE["head_top"]), neck, True)

    for side in ("L", "R"):
        s = V(shoulder(side))
        outx = 1.0 if side == "L" else -1.0
        tip = V(_rest_tip(side))
        wrist = V(wrist_target(tuple(s), tuple(tip)))
        # Rest elbow: down and out from the shoulder, exactly upper_arm long.
        edir = V((0.55 * outx, -0.10, -0.83)).normalized()
        elbow = s + edir * ANTHRO["upper_arm"]
        # forearm elbow->wrist (length ANTHRO forearm); hand wrist->tip (STICK_LEN)
        fdir = (wrist - elbow)
        fdir = fdir.normalized() if fdir.length > 1e-6 else V((0.0, -1.0, 0.0))
        wrist_bone = elbow + fdir * ANTHRO["forearm"]
        hdir = (tip - wrist_bone)
        hdir = hdir.normalized() if hdir.length > 1e-6 else V((0.0, -1.0, 0.0))
        clav = bone(f"clav.{side}", _seat(POSE["neck"]), tuple(s), chest, False)
        up = bone(f"upper_arm.{side}", tuple(s), tuple(elbow), clav, False)
        fore = bone(f"forearm.{side}", tuple(elbow), tuple(wrist_bone), up, True)
        bone(f"hand.{side}", tuple(wrist_bone), tuple(wrist_bone + hdir * STICK_LEN),
             fore, True)
        # Align the forearm roll so its local Z is the elbow's hinge axis
        # (perpendicular to the rest arm plane); the IK limits below then make
        # the elbow a one-way hinge that cannot twist or bend backwards.
        hinge = (elbow - s).cross(wrist_bone - elbow)
        if hinge.length > 1e-6:
            fore.align_roll(hinge.normalized())
        _REST_BEND[side] = math.degrees((elbow - s).angle(wrist_bone - elbow))

        hip = _seat_sq((ANTHRO["hip_dx"] * (1 if side == "L" else -1),
                        POSE["pelvis"][1], POSE["pelvis"][2]))
        knee = _solve_knee(hip, POSE[f"ankle_{side}"], POSE[f"toe_{side}"],
                           ANTHRO["thigh"], ANTHRO["shin"])
        thigh = bone(f"thigh.{side}", hip, knee, pelvis, False)  # off the STATIC pelvis
        shin = bone(f"shin.{side}", knee, POSE[f"ankle_{side}"], thigh, True)
        foot = bone(f"foot.{side}", POSE[f"ankle_{side}"], POSE[f"toe_{side}"], shin, True)
        # Align each foot's roll so local +Z points up. Both feet then share a
        # predictable frame (regardless of the pedal yaw), so a single ankle
        # press sign lowers the toe on either foot.
        foot.align_roll(V((0.0, 0.0, 1.0)))

    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
    return arm


def _add_targets(coll, arm):
    """Target empties and constraints: the forearm IK reaches the WRIST empty,
    and the hand bone Damped-Tracks the IK_Hand (stick tip) empty, so the stick
    always aims at the drum while the elbow hangs freely."""
    empties = {}
    for side in ("L", "R"):
        tip = _rest_tip(side)
        for tag, pos in (("IK_Hand", tip),
                         ("Wrist", wrist_target(shoulder(side), tip)),
                         ("Pole", POSE[f"pole_{side}"])):
            e = bpy.data.objects.new(f"{tag}_{side}", None)
            e.empty_display_size = 0.05
            e.location = pos
            coll.objects.link(e)
            empties[f"{tag}_{side}"] = e

    for side in ("L", "R"):
        pbF = arm.pose.bones[f"forearm.{side}"]
        ik = pbF.constraints.new('IK')
        ik.target = empties[f"Wrist_{side}"]
        ik.pole_target = empties[f"Pole_{side}"]
        ik.pole_angle = POLE_ANGLE[side]
        ik.chain_count = 2
        # Elbow = a one-way hinge: no twist (Y) or sideways bend (X); flexion
        # (local Z) may fold freely but stops ~3 deg short of straight so the
        # forearm can never bend backwards (+Z straightens; the rest is bent
        # _REST_BEND deg, so straight is at 180 - _REST_BEND).
        pbF.lock_ik_x = True
        pbF.lock_ik_y = True
        pbF.use_ik_limit_z = True
        # Human elbow: fold to 150 deg flexion (interior ELBOW_FLEX_MIN), extend
        # to just shy of straight (no hyperextension). Limits are relative to the
        # rest bend, so interior = limit + _REST_BEND.
        pbF.ik_min_z = math.radians(ELBOW_FLEX_MIN - _REST_BEND[side])
        pbF.ik_max_z = math.radians(ELBOW_STRAIGHT - _REST_BEND[side])

        # Shoulder: cage the IK rotation to a human-plausible envelope so it can
        # never over-rotate into an impossible pose (the pole is primary control).
        pbU = arm.pose.bones[f"upper_arm.{side}"]
        sl = SHOULDER_IK_LIMIT[side]
        for axis in ("x", "y", "z"):
            setattr(pbU, f"use_ik_limit_{axis}", True)
            lo, hi = sl[axis]
            setattr(pbU, f"ik_min_{axis}", math.radians(lo))
            setattr(pbU, f"ik_max_{axis}", math.radians(hi))

        dt = arm.pose.bones[f"hand.{side}"].constraints.new('DAMPED_TRACK')
        dt.target = empties[f"IK_Hand_{side}"]
        dt.track_axis = 'TRACK_Y'
        # Wrist: after the stick aims at the tip, clamp the hand's local rotation
        # to a human wrist envelope (flexion/extension/deviation) so it can't
        # bend past a real wrist.
        lr = arm.pose.bones[f"hand.{side}"].constraints.new('LIMIT_ROTATION')
        lr.owner_space = 'LOCAL'
        wl = WRIST_ROT_LIMIT[side]
        for axis in ("x", "y", "z"):
            setattr(lr, f"use_limit_{axis}", True)
            lo, hi = wl[axis]
            setattr(lr, f"min_{axis}", math.radians(lo))
            setattr(lr, f"max_{axis}", math.radians(hi))

    # --- Leg IK: the thigh+shin chain reaches an Ankle target so the knee can
    # drive up and down (the heel-up kick technique) while the foot stays planted
    # on its pedal. A Knee pole placed in the rest knee's bend direction keeps the
    # IK reproducing the square-to-pedal rest leg (see _solve_knee).
    for side in ("L", "R"):
        H = arm.matrix_world @ arm.pose.bones[f"thigh.{side}"].bone.head_local
        K = arm.matrix_world @ arm.pose.bones[f"shin.{side}"].bone.head_local
        A = arm.matrix_world @ arm.pose.bones[f"shin.{side}"].bone.tail_local
        axis = (A - H).normalized()
        bend = (K - H) - (K - H).dot(axis) * axis      # knee offset off the hip-ankle line
        bend = bend.normalized() if bend.length > 1e-6 else V((0.0, -1.0, 0.0))
        for tag, pos in (("Ankle", tuple(A)), ("Knee", tuple(K + bend * 0.45))):
            e = bpy.data.objects.new(f"{tag}_{side}", None)
            e.empty_display_size = 0.05
            e.location = pos
            coll.objects.link(e)
            empties[f"{tag}_{side}"] = e
        pbS = arm.pose.bones[f"shin.{side}"]
        ik = pbS.constraints.new('IK')
        ik.target = empties[f"Ankle_{side}"]
        ik.pole_target = empties[f"Knee_{side}"]
        ik.pole_angle = LEG_POLE_ANGLE[side]
        ik.chain_count = 2
    return empties


def _world_box(arm, coll, mat, name, center, size, rotation=(0, 0, 0)):
    """An axis-aligned world box parented (object-level) to the armature -
    for the static base (pelvis), which should not inherit a bone's tilt."""
    obj = bpy.data.objects.new(name, _box(name + "Mesh", *size))
    obj.location = center
    obj.rotation_euler = rotation
    coll.objects.link(obj)
    obj.parent = arm
    obj.matrix_parent_inverse = arm.matrix_world.inverted()
    obj.data.materials.append(mat)
    return obj


def _bone_upright(arm, coll, mat, bone, name, center, size):
    """An upright, world-aligned box that FOLLOWS a bone - so the torso/head
    rotate with the spine while still reading as square boxes (not bone-tilted).
    Parent to the bone, then stamp the desired upright world matrix; Blender
    back-computes the local transform against the bone's rest pose."""
    obj = bpy.data.objects.new(name, _box(name + "Mesh", *size))
    coll.objects.link(obj)
    obj.data.materials.append(mat)
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = bone
    bpy.context.view_layer.update()
    obj.matrix_world = (mathutils.Matrix.Translation(V(center))
                        @ mathutils.Matrix.Rotation(SEAT_YAW, 4, 'Z'))
    return obj


def _clothe(arm, coll):
    skin = _mat("DrummerBody", (0.30, 0.33, 0.38), 0.6)      # neutral grey mock
    dark = _mat("DrummerDark", (0.13, 0.14, 0.17), 0.6)
    wood = _mat("StickWood", (0.80, 0.66, 0.44), 0.45)
    rubber = _mat("ShoeRubber", (0.04, 0.04, 0.05), 0.6)

    aw = ANTHRO["arm_thick"]
    lw = ANTHRO["leg_thick"]
    L1 = ANTHRO["upper_arm"]
    fore, stick = ANTHRO["forearm"], ANTHRO["stick"]   # stick = total length

    # Pelvis is the static base; the chest/neck/head follow the spine so the
    # torso can twist toward the kit (they stay upright boxes via _bone_upright).
    _world_box(arm, coll, dark, "Torso_Pelvis", _seat_sq((0.0, 0.82, 0.66)), (0.34, 0.20, 0.32))
    # A slim chest, set back a touch: the cross-arm reach to the hi-hat passes
    # just in front of the torso, so a shallow (low y-depth), narrower chest
    # sitting slightly behind the shoulders keeps the crossing arm outside it.
    _bone_upright(arm, coll, skin, "spine", "Torso_Chest", _seat((0.0, 0.82, 0.98)), (0.32, 0.15, 0.34))
    _bone_upright(arm, coll, skin, "neck", "Neck", _seat((0.0, 0.81, 1.18)), (0.09, 0.09, 0.10))
    _bone_upright(arm, coll, skin, "head", "Head", _seat((0.0, 0.80, 1.31)), ANTHRO["head"])

    for side in ("L", "R"):
        _part(arm, coll, skin, f"upper_arm.{side}", f"UpperArm_{side}",
              (0, -L1 / 2.0, 0), (aw, L1 * 0.92, aw))
        # forearm bone: wrist(tail) -> elbow(head at -fore)
        _part(arm, coll, skin, f"forearm.{side}", f"Forearm_{side}",
              (0, -fore / 2.0, 0), (aw * 0.9, fore * 0.9, aw * 0.9))
        # hand bone: tip(tail) -> grip(head at -STICK_LEN). The fist grips a
        # third of the way up (butt overhangs behind it); the full stick runs
        # tip(0) -> butt(-stick) and aims at the drum via the Damped Track.
        _part(arm, coll, skin, f"hand.{side}", f"Hand_{side}",
              (0, -STICK_LEN, 0), (aw * 1.15, 0.05, aw * 1.15))
        _part(arm, coll, wood, f"hand.{side}", f"Stick_{side}",
              (0, 0, 0), mesh=_stick_mesh(f"Stick_{side}Mesh", stick, ANTHRO["stick_r"]))

        _part(arm, coll, dark, f"thigh.{side}", f"Thigh_{side}",
              (0, -ANTHRO["thigh"] / 2.0, 0), (lw, ANTHRO["thigh"] * 0.92, lw))
        _part(arm, coll, dark, f"shin.{side}", f"Shin_{side}",
              (0, -ANTHRO["shin"] / 2.0, 0), (lw * 0.85, ANTHRO["shin"] * 0.92, lw * 0.85))
        _part(arm, coll, rubber, f"foot.{side}", f"Shoe_{side}",
              (0, -0.09, -0.02), mesh=_shoe_mesh(f"Shoe_{side}Mesh"))


def _build_throne(coll):
    seat = _mat("Throne", (0.10, 0.10, 0.12), 0.5)
    tx, ty, tz = _seat_sq(POSE["throne"])
    top = bpy.data.objects.new("Throne_Seat", _box("Throne_SeatMesh", 0.32, 0.32, 0.05))
    top.location = (tx, ty, tz)
    top.data.materials.append(seat); coll.objects.link(top)
    post = bpy.data.objects.new("Throne_Post",
                                _box("Throne_PostMesh", 0.05, 0.05, tz - 0.05, bevel=0))
    post.location = (tx, ty, (tz - 0.02) / 2.0)
    post.data.materials.append(seat); coll.objects.link(post)


def build_drummer():
    scene = bpy.context.scene
    old = bpy.data.collections.get("Drummer")
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Drummer")
    scene.collection.children.link(coll)

    arm = _build_skeleton(coll)
    _add_targets(coll, arm)
    _clothe(arm, coll)
    _build_throne(coll)

    bpy.context.view_layer.update()
    print(f"Built drummer: armature '{arm.name}' with "
          f"{len(arm.pose.bones)} bones, IK arms, sticks, shoes, throne")
    return coll


if __name__ == "__main__":
    build_drummer()
