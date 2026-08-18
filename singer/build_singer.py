"""Builds a blocky, faceless, minimalist humanoid singer to stand in for a
real character model later on.

Run inside Blender to create a "Singer" collection: a single ``Singer``
armature (average adult proportions, standing upright) with rigid box meshes
on every bone and shoes on the feet -- the same house style as the
guitarist/drummer/pianist stand-ins (see guitar/build_guitarist.py, which this
is adapted from).

Like the guitarist, the rig is two-bone IK on both arms and both legs so an
animator can move the wrists/ankles as world-space targets and every joint
stays inside a human range of motion. The right ``hand.R`` stub is where
build_mic.py's handheld mic is mounted (held up to the mouth); the left arm
just rests at the singer's side.

The figure stands at the world origin, feet on the z=0 floor, facing -Y (left
= +X, right = -X, up = +Z) -- the same axis convention as every other
musician in this repo, and the same convention singer/mouth_shapes.py already
uses for the flat mouth card, so the mouth mounts straight onto the head with
no re-orientation.

Usage (Blender Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_singer", "/path/to/singer/build_singer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_singer()
"""

import math

import bpy
import bmesh
import mathutils

V = mathutils.Vector

# ---------------------------------------------------------------------------
# Body proportions (average adult, metres). Same anthropometric tables as
# guitar/build_guitarist.py (Drillis & Contini / Winter, H = 1.75 m).
# ---------------------------------------------------------------------------
_H = 1.75
ANTHRO = {
    "upper_arm": 0.186 * _H,
    "forearm": 0.146 * _H,
    "hand": 0.10,
    "thigh": 0.245 * _H,
    "shin": 0.246 * _H,
    "foot": 0.152 * _H,
    "shoulder_dx": 0.259 * _H / 2.0,
    "hip_dx": 0.09,
    "arm_thick": 0.06,
    "leg_thick": 0.10,
    "torso_w": 0.34, "torso_d": 0.21,
    "head": (0.155, 0.20, 0.130 * _H),
}

# Standing joint positions (world, metres). Figure faces -Y; left = +X, right
# = -X, up = +Z. Feet rest on the z=0 floor. Small knee/elbow rest bends (not
# dead straight) so the IK solver knows which way each hinge folds.
POSE = {
    "pelvis":     (0.0, 0.0, 0.935),
    "chest":      (0.0, 0.0, 1.30),
    "neck":       (0.0, 0.0, 1.45),
    "head_top":   (0.0, 0.0, 1.75),
    "shoulder_y": 0.0, "shoulder_z": 1.42,
    "ankle_L":    (0.105, 0.0, 0.085),
    "ankle_R":    (-0.105, 0.0, 0.085),
    "toe_L":      (0.105, -0.16, 0.0),
    "toe_R":      (-0.105, -0.16, 0.0),
    # Left wrist hangs relaxed at the side. Right wrist starts at the same
    # rest position -- animate_singer.py lifts it to hold the mic to the mouth.
    "wrist_L":    (0.250, -0.03, 0.85),
    "wrist_R":    (-0.250, -0.03, 0.85),
    "elbow_pole_L": (0.55, 0.55, 1.05),
    "elbow_pole_R": (-0.55, 0.55, 1.05),
    "knee_pole_L":  (0.11, -0.6, 0.52),
    "knee_pole_R":  (-0.11, -0.6, 0.52),
}

ARM_POLE_ANGLE = {"L": math.radians(90.0), "R": math.radians(90.0)}
LEG_POLE_ANGLE = {"L": math.radians(-90.0), "R": math.radians(-90.0)}

ELBOW_FLEX_MIN = 30.0
ELBOW_STRAIGHT = 178.0
KNEE_FLEX_MIN = 35.0
KNEE_STRAIGHT = 178.0

SHOULDER_IK_LIMIT = {
    "L": {"x": (-90.0, 130.0), "y": (-90.0, 90.0), "z": (-100.0, 100.0)},
    "R": {"x": (-90.0, 130.0), "y": (-90.0, 90.0), "z": (-100.0, 100.0)},
}
HIP_IK_LIMIT = {
    "L": {"x": (-120.0, 45.0), "y": (-45.0, 45.0), "z": (-45.0, 60.0)},
    "R": {"x": (-120.0, 45.0), "y": (-45.0, 45.0), "z": (-60.0, 45.0)},
}

SPINE_ROT_LIMIT = {"x": (-25.0, 40.0), "y": (-30.0, 30.0), "z": (-30.0, 30.0)}
NECK_ROT_LIMIT = {"x": (-40.0, 45.0), "y": (-60.0, 60.0), "z": (-50.0, 50.0)}
ANKLE_ROT_LIMIT = {"x": (-35.0, 40.0), "y": (-15.0, 15.0), "z": (-20.0, 20.0)}
WRIST_ROT_LIMIT = {"x": (-70.0, 70.0), "y": (-30.0, 30.0), "z": (-80.0, 80.0)}

_REST_BEND = {}


def shoulder(side):
    dx = ANTHRO["shoulder_dx"] * (1 if side == "L" else -1)
    return (dx, POSE["shoulder_y"], POSE["shoulder_z"])


def hip(side):
    dx = ANTHRO["hip_dx"] * (1 if side == "L" else -1)
    return (dx, POSE["pelvis"][1], POSE["pelvis"][2])


def _solve_mid(a, b, l1, l2, bend):
    A, B = V(a), V(b)
    d = (B - A).length
    if d >= l1 + l2 or d < 1e-6:
        return tuple(A + (B - A) * (l1 / max(d, 1e-6)))
    x = (d * d + l1 * l1 - l2 * l2) / (2.0 * d)
    h = math.sqrt(max(0.0, l1 * l1 - x * x))
    axis = (B - A) / d
    bd = V(bend)
    perp = bd - bd.dot(axis) * axis
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
    arm_data = bpy.data.armatures.new("SingerRig")
    arm = bpy.data.objects.new("Singer", arm_data)
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
        elbow = V(_solve_mid(s, wrist, ANTHRO["upper_arm"], ANTHRO["forearm"],
                             (0.4 * outx, 0.9, -0.2)))
        hdir = V((0.1 * outx, -0.3, -0.95)).normalized()
        clav = bone(f"clav.{side}", POSE["neck"], tuple(s), chest, False)
        up = bone(f"upper_arm.{side}", tuple(s), tuple(elbow), clav, False)
        fore = bone(f"forearm.{side}", tuple(elbow), tuple(wrist), up, True)
        hand_b = bone(f"hand.{side}", tuple(wrist),
                      tuple(wrist + hdir * ANTHRO["hand"]), fore, True)
        hand_b.align_roll(V((0.0, 0.0, 1.0)))
        hinge = (elbow - s).cross(wrist - elbow)
        if hinge.length > 1e-6:
            fore.align_roll(hinge.normalized())
        _REST_BEND[f"arm_{side}"] = math.degrees((elbow - s).angle(wrist - elbow))

        h = V(hip(side))
        ankle = V(POSE[f"ankle_{side}"])
        knee = V(_solve_mid(h, ankle, ANTHRO["thigh"], ANTHRO["shin"],
                            (0.05 * outx, -0.9, 0.0)))
        thigh = bone(f"thigh.{side}", tuple(h), tuple(knee), pelvis, False)
        shin = bone(f"shin.{side}", tuple(knee), tuple(ankle), thigh, True)
        foot = bone(f"foot.{side}", tuple(ankle), POSE[f"toe_{side}"], shin, True)
        khinge = (knee - h).cross(ankle - knee)
        if khinge.length > 1e-6:
            shin.align_roll(khinge.normalized())
        foot.align_roll(V((0.0, 0.0, 1.0)))
        _REST_BEND[f"leg_{side}"] = math.degrees((knee - h).angle(ankle - knee))

    bpy.ops.object.mode_set(mode='OBJECT')
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ'
    return arm


def _add_targets(coll, arm):
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
        pbF = arm.pose.bones[f"forearm.{side}"]
        ik = pbF.constraints.new('IK')
        ik.target = empties[f"Wrist_{side}"]
        ik.pole_target = empties[f"Elbow_{side}"]
        ik.pole_angle = ARM_POLE_ANGLE[side]
        ik.chain_count = 2
        pbF.lock_ik_x = True
        pbF.lock_ik_y = True
        pbF.use_ik_limit_z = True
        rb = _REST_BEND[f"arm_{side}"]
        pbF.ik_min_z = math.radians(ELBOW_FLEX_MIN - rb)
        pbF.ik_max_z = math.radians(ELBOW_STRAIGHT - rb)
        pbU = arm.pose.bones[f"upper_arm.{side}"]
        _cage_ik(pbU, SHOULDER_IK_LIMIT[side])

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
        pbT = arm.pose.bones[f"thigh.{side}"]
        _cage_ik(pbT, HIP_IK_LIMIT[side])

        _limit_rot(arm.pose.bones[f"foot.{side}"], ANKLE_ROT_LIMIT)
        _limit_rot(arm.pose.bones[f"hand.{side}"], WRIST_ROT_LIMIT)

    _limit_rot(arm.pose.bones["spine"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["chest"], SPINE_ROT_LIMIT)
    _limit_rot(arm.pose.bones["neck"], NECK_ROT_LIMIT)
    return empties


def _cage_ik(pbone, limit):
    for axis in ("x", "y", "z"):
        setattr(pbone, f"use_ik_limit_{axis}", True)
        lo, hi = limit[axis]
        setattr(pbone, f"ik_min_{axis}", math.radians(lo))
        setattr(pbone, f"ik_max_{axis}", math.radians(hi))


def _limit_rot(pbone, limit):
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
    skin = _mat("SingerBody", (0.30, 0.33, 0.38), 0.6)
    dark = _mat("SingerDark", (0.13, 0.14, 0.17), 0.6)
    rubber = _mat("SingerShoe", (0.04, 0.04, 0.05), 0.6)

    aw = ANTHRO["arm_thick"]
    lw = ANTHRO["leg_thick"]
    L1, fore = ANTHRO["upper_arm"], ANTHRO["forearm"]
    thigh, shin = ANTHRO["thigh"], ANTHRO["shin"]

    _bone_upright(arm, coll, dark, "pelvis", "Torso_Pelvis",
                  (0.0, 0.0, 0.88), (ANTHRO["torso_w"], ANTHRO["torso_d"], 0.20))
    _bone_upright(arm, coll, skin, "spine", "Torso_Chest",
                  (0.0, 0.0, 1.14), (ANTHRO["torso_w"], ANTHRO["torso_d"] * 0.85, 0.34))
    _bone_upright(arm, coll, skin, "neck", "Neck",
                  (0.0, 0.0, 1.40), (0.09, 0.09, 0.10))
    _bone_upright(arm, coll, skin, "head", "Head",
                  (0.0, -0.01, 1.635), ANTHRO["head"])

    for side in ("L", "R"):
        _part(arm, coll, skin, f"upper_arm.{side}", f"UpperArm_{side}",
              (0, -L1 / 2.0, 0), (aw, L1 * 0.92, aw))
        _part(arm, coll, skin, f"forearm.{side}", f"Forearm_{side}",
              (0, -fore / 2.0, 0), (aw * 0.9, fore * 0.9, aw * 0.9))
        _part(arm, coll, skin, f"hand.{side}", f"Hand_{side}",
              (0, -ANTHRO["hand"] / 2.0, 0), (aw * 1.1, ANTHRO["hand"] * 0.8, aw * 0.7))

        _part(arm, coll, dark, f"thigh.{side}", f"Thigh_{side}",
              (0, -thigh / 2.0, 0), (lw, thigh * 0.92, lw))
        _part(arm, coll, dark, f"shin.{side}", f"Shin_{side}",
              (0, -shin / 2.0, 0), (lw * 0.85, shin * 0.92, lw * 0.85))
        _part(arm, coll, rubber, f"foot.{side}", f"Shoe_{side}",
              (0, -ANTHRO["foot"] / 2.0 + 0.03, -0.02), mesh=_shoe_mesh(f"Shoe_{side}Mesh"))


def build_singer():
    scene = bpy.context.scene
    old = bpy.data.collections.get("Singer")
    if old is not None:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)

    coll = bpy.data.collections.new("Singer")
    scene.collection.children.link(coll)

    _REST_BEND.clear()
    arm = _build_skeleton(coll)
    _add_targets(coll, arm)
    _clothe(arm, coll)

    bpy.context.view_layer.update()
    print(f"Built singer: armature '{arm.name}' with "
          f"{len(arm.pose.bones)} bones, IK arms + legs")
    return coll


if __name__ == "__main__":
    build_singer()
