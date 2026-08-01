"""Builds a blocky, faceless, minimalist humanoid guitarist to stand in for a
real character model later on.

Run inside Blender (optionally after build_guitar.py) to create a "Guitarist"
collection: a single ``Guitarist`` armature (average adult proportions, standing
upright) with rigid box meshes on every bone and shoes on the feet.

Unlike the drummer (who sits bolted to the kit), the guitarist is a free-standing,
walk-capable figure. The rig is built so an animator can:

  * WALK it -- the pelvis is the root; translating/rotating it carries the whole
    body, while each leg is a two-bone IK chain (thigh + shin) reaching an
    ``Ankle_*`` empty with a ``Knee_*`` pole. Because the feet are driven by
    world-space targets (not parented to the moving hips), a foot can plant on the
    floor and stay put as the pelvis passes over it -- the basis of a walk cycle.
  * MOVE its limbs in a natural range of motion -- every joint carries human ROM
    limits (AAOS norms; see below). The elbows and knees are true one-way hinges
    (flex freely, never hyperextend); the shoulders and hips are caged in a
    human-plausible envelope; the spine, neck and wrists get soft rotation limits.
    So no pose the animator dials in can bend a joint the wrong way.

The arms are also two-bone IK (upper arm + forearm) reaching a ``Wrist_*`` empty
with an ``Elbow_*`` pole, ending in a short ``hand.*`` stub bone. That stub is the
attach point for the guitar animation pipeline's hand rigs (build_hands.py's
``FretHand`` / ``PickHand`` armatures): parent each hand armature to the matching
``hand.*`` bone later and the arm IK will carry the hands wherever the wrists go.
The hand stub is rolled to a predictable local-Z-up frame so that attachment is
consistent on both sides.

The figure stands at the world origin, feet on the z=0 floor, facing -Y (left =
+X, right = -X, up = +Z) -- the same axis convention as the drummer stand-in.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_guitarist", "/path/to/guitar/build_guitarist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_guitarist()
"""

import math

import bpy
import bmesh
import mathutils

V = mathutils.Vector

# ---------------------------------------------------------------------------
# Body proportions (average adult, metres). Swap this dict to re-shape the
# stand-in; the arms and legs are IK so hand/foot targets re-solve automatically.
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
    "torso_w": 0.34, "torso_d": 0.21,
    "head": (0.155, 0.20, 0.130 * _H),  # breadth, depth, height (~0.228)
}

# Standing joint positions (world, metres). The figure stands at the origin
# facing -Y; left = +X, right = -X, up = +Z. Feet rest on the z=0 floor.
#
# The vertical stack is chosen so the legs and arms are nearly -- but not quite --
# straight at rest: the knees carry a small forward bend and the elbows a small
# backward bend, so the IK solver knows which way each hinge folds (a dead-straight
# rest leaves the hinge direction ambiguous and the solver can pop the joint the
# wrong way). The exact knee/elbow points are solved from the segment lengths by
# ``_solve_mid`` so the bones are always their true anthropometric length.
POSE = {
    "pelvis":     (0.0, 0.0, 0.935),   # hip-joint height (~0.53 H)
    "chest":      (0.0, 0.0, 1.30),    # upper spine
    "neck":       (0.0, 0.0, 1.45),
    "head_top":   (0.0, 0.0, 1.75),    # crown at full stature H
    "shoulder_y": 0.0, "shoulder_z": 1.42,
    # Feet stand a little wider than the hips, toes forward (-Y), soles on z=0.
    "ankle_L":    (0.105, 0.0, 0.085),
    "ankle_R":    (-0.105, 0.0, 0.085),
    "toe_L":      (0.105, -0.16, 0.0),
    "toe_R":      (-0.105, -0.16, 0.0),
    # Wrists hang at the sides, a touch forward (-Y) and out, so the arms rest
    # nearly straight with the elbows just off the lock.
    "wrist_L":    (0.250, -0.03, 0.85),
    "wrist_R":    (-0.250, -0.03, 0.85),
    # IK pole targets. Elbows point back-and-out (pole behind, +Y); knees point
    # forward (pole ahead, -Y). Positions are refined per side below.
    "elbow_pole_L": (0.55, 0.55, 1.05),
    "elbow_pole_R": (-0.55, 0.55, 1.05),
    "knee_pole_L":  (0.11, -0.6, 0.52),
    "knee_pole_R":  (-0.11, -0.6, 0.52),
}

# Per-side IK pole angles (radians). Tuned so the two-bone IK reproduces the
# built rest pose without twisting the limb about its own axis.
ARM_POLE_ANGLE = {"L": math.radians(90.0), "R": math.radians(90.0)}
LEG_POLE_ANGLE = {"L": math.radians(-90.0), "R": math.radians(-90.0)}

# --- Joint range-of-motion limits (AAOS human norms; degrees) ----------------
# ELBOW and KNEE are true 1-DOF hinges: they flex freely but must never
# hyperextend, so they get real anatomical caps on the IK hinge (local Z), and
# no twist/sideways bend (locked X/Y). Interior angle = 180 deg at full
# extension; the cap stops just shy of straight.
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
HIP_IK_LIMIT = {
    "L": {"x": (-120.0, 45.0), "y": (-45.0, 45.0), "z": (-45.0, 60.0)},
    "R": {"x": (-120.0, 45.0), "y": (-45.0, 45.0), "z": (-60.0, 45.0)},
}

# Soft rotation envelopes (deg, local X/Y/Z) on the FK joints -- the spine, neck,
# head, ankles and wrists -- so hand-authored poses stay in a human range. These
# are LIMIT_ROTATION constraints (they clamp the final pose), not IK limits.
SPINE_ROT_LIMIT = {"x": (-25.0, 40.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
NECK_ROT_LIMIT = {"x": (-40.0, 45.0), "y": (-60.0, 60.0), "z": (-50.0, 50.0)}
ANKLE_ROT_LIMIT = {"x": (-35.0, 40.0), "y": (-15.0, 15.0), "z": (-20.0, 20.0)}
WRIST_ROT_LIMIT = {"x": (-70.0, 70.0), "y": (-30.0, 30.0), "z": (-80.0, 80.0)}

# Elbow/knee rest bends (degrees), filled in while building; used to place each
# hinge's IK flexion limit just short of straight so it cannot hyperextend.
_REST_BEND = {}


def shoulder(side):
    dx = ANTHRO["shoulder_dx"] * (1 if side == "L" else -1)
    return (dx, POSE["shoulder_y"], POSE["shoulder_z"])


def hip(side):
    dx = ANTHRO["hip_dx"] * (1 if side == "L" else -1)
    return (dx, POSE["pelvis"][1], POSE["pelvis"][2])


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
    bmesh.ops.scale(bm, vec=(0.10, ANTHRO["foot"], 0.06), verts=bm.verts)
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
    arm_data = bpy.data.armatures.new("GuitaristRig")
    arm = bpy.data.objects.new("Guitarist", arm_data)
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

    # The PELVIS is the movable root: translating/rotating it carries the whole
    # figure (the basis of a walk). The spine rises from it and both legs hang
    # off it, so a hip sway or a step swings the legs with the body.
    px, py, pz = POSE["pelvis"]
    pelvis = bone("pelvis", (px, py, pz), (px, py, pz + 0.09))
    spine = bone("spine", POSE["pelvis"], POSE["chest"], pelvis, False)
    chest = bone("chest", POSE["chest"], POSE["neck"], spine, True)
    neck_head = POSE["neck"]
    neck = bone("neck", neck_head, (neck_head[0], neck_head[1] - 0.02,
                                    neck_head[2] + 0.08), chest, True)
    bone("head", neck.tail, POSE["head_top"], neck, True)

    for side in ("L", "R"):
        s = V(shoulder(side))
        outx = 1.0 if side == "L" else -1.0
        wrist = V(POSE[f"wrist_{side}"])
        # Elbow: solved so upper_arm + forearm are exact, bulging back-and-out.
        elbow = V(_solve_mid(s, wrist, ANTHRO["upper_arm"], ANTHRO["forearm"],
                             (0.4 * outx, 0.9, -0.2)))
        # Hand stub: continues down-and-forward from the wrist (where the guitar
        # hand rig attaches later).
        hdir = V((0.1 * outx, -0.3, -0.95)).normalized()
        clav = bone(f"clav.{side}", POSE["neck"], tuple(s), chest, False)
        up = bone(f"upper_arm.{side}", tuple(s), tuple(elbow), clav, False)
        fore = bone(f"forearm.{side}", tuple(elbow), tuple(wrist), up, True)
        hand_b = bone(f"hand.{side}", tuple(wrist),
                      tuple(wrist + hdir * ANTHRO["hand"]), fore, True)
        # Roll BOTH hand stubs to the same reference (local Z up) so the external
        # FretHand / PickHand rigs attach with a consistent frame on each side.
        hand_b.align_roll(V((0.0, 0.0, 1.0)))
        # Roll the forearm so its local Z is the elbow's hinge axis (perpendicular
        # to the rest arm plane); the IK limits below then make the elbow a one-way
        # hinge that cannot twist or bend backwards.
        hinge = (elbow - s).cross(wrist - elbow)
        if hinge.length > 1e-6:
            fore.align_roll(hinge.normalized())
        _REST_BEND[f"arm_{side}"] = math.degrees((elbow - s).angle(wrist - elbow))

        # Leg: hip -> knee -> ankle -> toe. Knee solved to bulge forward (-Y).
        h = V(hip(side))
        ankle = V(POSE[f"ankle_{side}"])
        knee = V(_solve_mid(h, ankle, ANTHRO["thigh"], ANTHRO["shin"],
                            (0.05 * outx, -0.9, 0.0)))
        thigh = bone(f"thigh.{side}", tuple(h), tuple(knee), pelvis, False)
        shin = bone(f"shin.{side}", tuple(knee), tuple(ankle), thigh, True)
        foot = bone(f"foot.{side}", tuple(ankle), POSE[f"toe_{side}"], shin, True)
        # Roll the shin so its local Z is the knee's hinge axis (one-way hinge).
        khinge = (knee - h).cross(ankle - knee)
        if khinge.length > 1e-6:
            shin.align_roll(khinge.normalized())
        # Feet share a predictable local +Z-up frame, so one ankle-press sign
        # works on either side.
        foot.align_roll(V((0.0, 0.0, 1.0)))
        _REST_BEND[f"leg_{side}"] = math.degrees((knee - h).angle(ankle - knee))

    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
    return arm


def _add_targets(coll, arm):
    """Target empties + constraints. Arms and legs are each a two-bone IK chain
    reaching a wrist/ankle empty with an elbow/knee pole; the FK joints (spine,
    neck, head, ankles, wrists) get soft rotation limits."""
    empties = {}

    def add_empty(name, pos):
        e = bpy.data.objects.new(name, None)
        e.empty_display_size = 0.05
        e.location = pos
        coll.objects.link(e)
        empties[name] = e
        return e

    for side in ("L", "R"):
        add_empty(f"Wrist_{side}", POSE[f"wrist_{side}"])
        add_empty(f"Elbow_{side}", POSE[f"elbow_pole_{side}"])
        add_empty(f"Ankle_{side}", POSE[f"ankle_{side}"])
        add_empty(f"Knee_{side}", POSE[f"knee_pole_{side}"])

    for side in ("L", "R"):
        # --- Arm IK: forearm chain (upper_arm + forearm) reaches the wrist. ----
        pbF = arm.pose.bones[f"forearm.{side}"]
        ik = pbF.constraints.new('IK')
        ik.target = empties[f"Wrist_{side}"]
        ik.pole_target = empties[f"Elbow_{side}"]
        ik.pole_angle = ARM_POLE_ANGLE[side]
        ik.chain_count = 2
        # Elbow = one-way hinge: no twist (Y) or sideways bend (X); flexion (Z)
        # folds freely but stops just short of straight (no hyperextension).
        pbF.lock_ik_x = True
        pbF.lock_ik_y = True
        pbF.use_ik_limit_z = True
        rb = _REST_BEND[f"arm_{side}"]
        pbF.ik_min_z = math.radians(ELBOW_FLEX_MIN - rb)
        pbF.ik_max_z = math.radians(ELBOW_STRAIGHT - rb)
        # Shoulder: cage the IK rotation to a human-plausible envelope.
        pbU = arm.pose.bones[f"upper_arm.{side}"]
        _cage_ik(pbU, SHOULDER_IK_LIMIT[side])

        # --- Leg IK: shin chain (thigh + shin) reaches the ankle. -------------
        pbS = arm.pose.bones[f"shin.{side}"]
        lik = pbS.constraints.new('IK')
        lik.target = empties[f"Ankle_{side}"]
        lik.pole_target = empties[f"Knee_{side}"]
        lik.pole_angle = LEG_POLE_ANGLE[side]
        lik.chain_count = 2
        pbS.lock_ik_x = True
        pbS.lock_ik_y = True
        pbS.use_ik_limit_z = True
        lb = _REST_BEND[f"leg_{side}"]
        pbS.ik_min_z = math.radians(KNEE_FLEX_MIN - lb)
        pbS.ik_max_z = math.radians(KNEE_STRAIGHT - lb)
        # Hip: cage the IK rotation to a human-plausible envelope.
        pbT = arm.pose.bones[f"thigh.{side}"]
        _cage_ik(pbT, HIP_IK_LIMIT[side])

        # --- Soft FK limits on the ankle and wrist. ---------------------------
        _limit_rot(arm.pose.bones[f"foot.{side}"], ANKLE_ROT_LIMIT)
        _limit_rot(arm.pose.bones[f"hand.{side}"], WRIST_ROT_LIMIT)

    # Spine / neck / head soft rotation limits.
    _limit_rot(arm.pose.bones["spine"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["chest"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["neck"], NECK_ROT_LIMIT)
    return empties


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
    skin = _mat("GuitaristBody", (0.30, 0.33, 0.38), 0.6)   # neutral grey mock
    dark = _mat("GuitaristDark", (0.13, 0.14, 0.17), 0.6)
    rubber = _mat("GuitaristShoe", (0.04, 0.04, 0.05), 0.6)

    aw = ANTHRO["arm_thick"]
    lw = ANTHRO["leg_thick"]
    L1, fore = ANTHRO["upper_arm"], ANTHRO["forearm"]
    thigh, shin = ANTHRO["thigh"], ANTHRO["shin"]

    # Torso / head: upright boxes that follow the spine bones (so a lean/twist
    # carries them) while reading as square blocks.
    _bone_upright(arm, coll, dark, "pelvis", "Torso_Pelvis",
                  (0.0, 0.0, 0.88), (ANTHRO["torso_w"], ANTHRO["torso_d"], 0.20))
    _bone_upright(arm, coll, skin, "spine", "Torso_Chest",
                  (0.0, 0.0, 1.14), (ANTHRO["torso_w"], ANTHRO["torso_d"] * 0.85, 0.34))
    _bone_upright(arm, coll, skin, "neck", "Neck",
                  (0.0, 0.0, 1.40), (0.09, 0.09, 0.10))
    _bone_upright(arm, coll, skin, "head", "Head",
                  (0.0, -0.01, 1.635), ANTHRO["head"])   # crown reaches stature H

    for side in ("L", "R"):
        _part(arm, coll, skin, f"upper_arm.{side}", f"UpperArm_{side}",
              (0, -L1 / 2.0, 0), (aw, L1 * 0.92, aw))
        _part(arm, coll, skin, f"forearm.{side}", f"Forearm_{side}",
              (0, -fore / 2.0, 0), (aw * 0.9, fore * 0.9, aw * 0.9))
        # Hand stub box: a fist-sized block at the wrist -- a placeholder the
        # guitar hand rig will sit over/replace.
        _part(arm, coll, skin, f"hand.{side}", f"Hand_{side}",
              (0, -ANTHRO["hand"] / 2.0, 0), (aw * 1.1, ANTHRO["hand"] * 0.8, aw * 0.7))

        _part(arm, coll, dark, f"thigh.{side}", f"Thigh_{side}",
              (0, -thigh / 2.0, 0), (lw, thigh * 0.92, lw))
        _part(arm, coll, dark, f"shin.{side}", f"Shin_{side}",
              (0, -shin / 2.0, 0), (lw * 0.85, shin * 0.92, lw * 0.85))
        _part(arm, coll, rubber, f"foot.{side}", f"Shoe_{side}",
              (0, -ANTHRO["foot"] / 2.0 + 0.03, -0.02), mesh=_shoe_mesh(f"Shoe_{side}Mesh"))


def build_guitarist():
    scene = bpy.context.scene
    old = bpy.data.collections.get("Guitarist")
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Guitarist")
    scene.collection.children.link(coll)

    _REST_BEND.clear()
    arm = _build_skeleton(coll)
    _add_targets(coll, arm)
    _clothe(arm, coll)

    bpy.context.view_layer.update()
    print(f"Built guitarist: armature '{arm.name}' with "
          f"{len(arm.pose.bones)} bones, IK arms + legs (walk-ready), shoes")
    return coll


if __name__ == "__main__":
    build_guitarist()
