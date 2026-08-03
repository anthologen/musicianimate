"""Animates a full standing bassist PLAYING the bass -- the piece that joins the
bass pipeline together end to end (the counterpart to guitar/animate_guitarist.py).

The bass pipeline already produces, independently:

  * build_bass_guitar.py -> a "BassGuitar" collection (lies flat, face +Z, neck +Y)
  * build_hands.py        -> a "FretHand" plus a right-hand rig that plays that flat
                             bass: "PluckHand" (fingerstyle, the default) or, with
                             --style pick, "PickHand"
  * animate_hands.py      -> keyframes those two hand rigs from a fingering.json
                             (the fret hand glides the neck; the pluck/pick hand
                             sounds the strings near the bridge)
  * build_bassist.py      -> a standing, walk-capable humanoid "Bassist" rig with
                             IK arms + legs and a hand.* stub at each wrist

This module stitches them into one shot:

  1. (Re)build the bass, the bassist, and the hands (matching the fingering's style).
  2. Run animate_hands() so the FretHand / PluckHand play the fingering.json in the
     bass's own flat authoring frame.
  3. RIGIDLY lift the whole "bass + playing hands" assembly off the floor and into
     playing position across the standing bassist's torso (neck up to the player's
     left, body over the right hip, strings facing forward). Because the bass mesh
     AND both hand rigs are re-parented under one holder, the hands keep playing the
     bass exactly as authored -- the entire assembly just moves together. The holder
     is BONE-PARENTED to the torso (COUPLE_BONE), so the bass rides the body: when the
     player sways, the instrument (and the hands on it) sway with it.
  4. Strap the bass ON: a strap band is swept from the bass's lower end-pin, around
     the player's back, over the left shoulder and down the front to the upper strap
     button, then parented to the same torso bone so it moves with the body.
  5. Make the bassist's arms FOLLOW the hands: each arm's IK wrist target copies the
     matching hand rig's wrist, so the two-bone arm IK reaches from the shoulder out to
     wherever the fretting / plucking hand goes. The bassist's own blocky hand stubs
     are hidden so the detailed hand rigs read instead.
  6. Give the body a gentle performance -- a slow torso sway and head bob over the take
     -- while the feet stay planted (leg IK). Because the bass, strap and shoulders all
     ride the same torso bone, the whole upper assembly sways as one and the hands stay
     glued to the strings.

Usage (inside Blender / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "animate_bassist", "/path/to/bass_guitar/animate_bassist.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_bassist("/path/to/bass_guitar/fingering.json")
"""

import importlib
import json
import math
import os
import sys

import bpy
import mathutils

V = mathutils.Vector
M = mathutils.Matrix
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # repo root, so `bass_guitar` / `piano` import
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(name):
    """Import a bass_guitar/*.py module as a proper ``bass_guitar`` package
    submodule, so its package-relative imports (and the shared piano helpers)
    resolve."""
    return importlib.import_module("bass_guitar." + name)


fret_layout = _load("fret_layout")

# ---------------------------------------------------------------------------
# Playing-position placement of the flat bass+hands assembly on the standing
# bassist. The flat bass has local +Y = up the neck, +Z = out the face,
# +X = toward the treble side. We map it into the world so:
#   * the face (+Z) points FORWARD  (world -Y, the way the bassist faces),
#   * the neck (+Y) rises up and to the player's LEFT (+X) at NECK_ELEV,
#   * a chosen anchor point on the strings lands at ANCHOR_WORLD on the torso.
# These few constants are the whole "how the bass is worn" tuning surface.
# ---------------------------------------------------------------------------
NECK_ELEV = math.radians(30.0)     # neck rises this far above horizontal
# ^ Also the main comfort dial for the FRETTING WRIST. The fret hand's orientation
# is locked to the bass (it wraps the neck), so the only way to relax the left wrist
# is to present that hand to the incoming forearm at a straighter angle. The bass
# neck is long (34" scale), so the fret hand sits well up and out to the player's
# left; angling the neck up rotates the whole bass+hand assembly about the plucking
# anchor (all fret/pluck contacts are preserved -- the assembly is rigid), lifting
# the fret hand so the forearm rises UP to meet it and the wrist folds at a relaxed
# angle rather than an acute one. A modest ~30 deg reads as a natural worn-low bass.
# Local point on the bass (near the plucking spot, over the bridge pickup) pinned onto
# the body, and the world spot on the bassist's front-right torso it is pinned to.
# Height is a real posture tradeoff for the PLUCKING arm (see _wire_arms /
# ELBOW_POLE_OVERRIDE):
#   * worn very high -> the near-straight-down forearm has to reach UP, the elbow
#     wings up and out (chicken wing);
#   * worn very low  -> the hand sits at almost full arm's reach, so the elbow locks
#     out straight and CANNOT rotate; every stroke then telescopes the whole rigid arm.
# This MODERATE height puts the plucking hand at ~80% of reach, leaving the elbow
# comfortably bent with slack for the forearm to drape over the top of the body while
# the fingers rest by the bridge pickup (standard fingerstyle anchor).
ANCHOR_LOCAL = (0.0, fret_layout.PLUCK_Y, fret_layout.STRING_Z)
# Wear height of the plucking anchor on the torso. Worn LOW at the waist (z=1.05, bass
# slung low like the reference photo) -- the plucking elbow rides up by the ribs and the
# forearm descends to the bridge at a natural ~115 deg bend, elbow ~11 cm above the wrist.
# Wear height is the key structural dial for the plucking arm: it sets the shoulder->
# pluck-point reach and thus how bent the arm is. Go much lower (z~1.00) and the arm
# stretches toward full extension (~143 deg), which reads as a stiff arch; this z=1.05 is
# the low-slung sweet spot that keeps a relaxed bend. The WRIST is then kept straight
# independently by the pick-hand orientation (below), re-solved to this exact geometry.
# MOUNT-COUPLED: the pick hand (build_hands PICK_HAND_ROT / PICK_TIP_LOCAL /
# PICK_MESH_ROT) and the R elbow pole below are solved for THIS height; re-solve if it
# changes much.
ANCHOR_WORLD = (-0.02, -0.195, 1.05)

HOLDER_NAME = "BassRig"
STRAP_NAME = "BassStrap"

# Plucking-arm elbow pole. Two jobs. (1) With the bassist's default rest pole (which
# points BEHIND the player) the two-bone arm IK swings the elbow behind the thin bass
# body and the forearm spears straight through it; this pole pulls the elbow to the
# FRONT so the forearm instead lies over the front of the body toward the strings.
# (2) It picks WHERE on the arm's elbow circle the joint sits: at the low waist wear
# (ANCHOR_WORLD z=1.05) the arm carries a ~115 deg bend, and this pole tucks the elbow
# back toward the player's side (clear of the chest by ~10 cm) so it rides up by the ribs
# with the forearm descending to the bridge -- the low-slung pick-bassist posture from the
# reference photo, wrist kept straight by the hand orientation. Keyed per
# side. The fretting (L) arm gets a gentler one that
# TUCKS the elbow down and slightly forward so the forearm rises up the neck to the
# hand, keeping the wrist relaxed (see NECK_ELEV). Overrides only the playing shot;
# build_bassist's free-standing rest pose is left untouched.
ELBOW_POLE_OVERRIDE = {"R": (-0.4, 0.1, 0.4), "L": (0.55, 0.1, 0.55)}
# The bass, strap and (via the chest, which passively inherits it) the shoulders all
# ride this torso bone, so the whole upper assembly sways as one unit.
COUPLE_BONE = "spine"

# Strap band cross-section (metres).
STRAP_WIDTH = 0.05
STRAP_THICK = 0.008

# Left arm follows the fret hand; the right arm follows whichever right-hand rig the
# fingering's style produced (fingerstyle PluckHand by default, or PickHand).
_RIGHT_HAND = {"finger": "PluckHand", "pick": "PickHand"}

# --- Wrist range-of-motion guard --------------------------------------------
# The playing hands are SEPARATE armatures whose world orientation is AUTHORED
# (build_hands' PICK_HAND_ROT / PLUCK_ROT / fret wrap), not solved by a limited
# joint -- so unlike the body's caged shoulders/elbows (build_bassist), a bad hand
# orientation slips past every IK limit. _check_wrist_pose closes that gap: after the
# arms are wired it validates each wrist against human ROM, so an over-bent or
# palm-flipped re-solve fails loudly at build time instead of shipping a broken wrist.
WRIST_BEND_MAX = math.radians(90.0)   # max angle the hand's long axis may sit off the
#                                       forearm -- the combined flexion(~80)+deviation(~30)
#                                       envelope (a neck-wrapping fret hand rides near it)
PALM_TO_STRINGS_MIN = 0.20            # the picking palm must face the STRINGS (into the
#                                      bass face), never out toward the audience


# Yaw the whole bass about the vertical axis so the NECK angles toward the audience
# (instead of the body facing straight-on / the neck straight across the player). A
# negative angle swings the neck (+X, player's left) forward toward the audience (-Y);
# it also relaxes the right wrist by turning the string plane to meet the plucking hand.
BODY_YAW = math.radians(-18.0)


def _play_transform():
    """4x4 world matrix that carries the flat bass frame into playing pose."""
    e = NECK_ELEV
    ey = V((math.cos(e), 0.0, math.sin(e)))     # neck axis -> up-and-left
    ez = V((0.0, -1.0, 0.0))                    # face -> forward
    ex = ey.cross(ez)                           # treble side (keeps it right-handed)
    rot = M((ex, ey, ez)).transposed().to_4x4()  # columns = ex, ey, ez
    rot = M.Rotation(BODY_YAW, 4, 'Z') @ rot     # yaw the bass so the neck faces audience
    t = V(ANCHOR_WORLD) - (rot @ V(ANCHOR_LOCAL))
    return M.Translation(t) @ rot


# ---------------------------------------------------------------------------
# Build / reset
# ---------------------------------------------------------------------------
def _reset_assembly():
    """Undo a previous placement so the run is idempotent: unparent the bass +
    hands from the holder (snapping them back to the flat authoring frame), delete
    the holder and the strap, so animate_hands can re-author in the flat frame."""
    holder = bpy.data.objects.get(HOLDER_NAME)
    if holder is not None:
        for child in list(holder.children):
            child.parent = None
            child.matrix_parent_inverse = M.Identity(4)
        bpy.data.objects.remove(holder, do_unlink=True)
    strap = bpy.data.objects.get(STRAP_NAME)
    if strap is not None:
        bpy.data.objects.remove(strap, do_unlink=True)


def _ensure_built(style):
    """(Re)build the bass, the bassist and the hands in an order that is safe
    against build_bass_guitar.clear_scene (which wipes every mesh): bass first,
    then the bassist body, then the hand rigs (matching the fingering's style)."""
    _load("build_bass_guitar").build_bass_guitar()
    _load("build_bassist").build_bassist()
    _load("build_hands").build_hands(style=style)


def _require(*names):
    missing = [n for n in names if bpy.data.objects.get(n) is None]
    if missing:
        raise RuntimeError("missing rig object(s): " + ", ".join(missing)
                           + " -- run with build=True or build them first")


# ---------------------------------------------------------------------------
# Placement + wiring
# ---------------------------------------------------------------------------
def _place_assembly(arm):
    """Parent the bass meshes and both hand rigs under one holder and stamp the
    playing-pose transform on it, so the already-animated assembly moves as a
    rigid whole into position on the bassist. The holder is itself bone-parented
    to the torso, so the mounted bass then rides the body's motion."""
    holder = bpy.data.objects.new(HOLDER_NAME, None)
    holder.empty_display_size = 0.08
    bpy.context.scene.collection.objects.link(holder)
    # Ride the torso: parent to the spine bone (still at rest here -- the body
    # groove is keyed later -- so stamping the world playing-pose lands it exactly).
    holder.parent = arm
    holder.parent_type = 'BONE'
    holder.parent_bone = COUPLE_BONE
    bpy.context.view_layer.update()
    holder.matrix_world = _play_transform()
    bpy.context.view_layer.update()

    children = []
    bass_coll = bpy.data.collections.get("BassGuitar")
    if bass_coll is not None:
        children += list(bass_coll.objects)
    children += [o for o in (bpy.data.objects.get("FretHand"),
                             bpy.data.objects.get("PluckHand"),
                             bpy.data.objects.get("PickHand")) if o is not None]

    for obj in children:
        obj.parent = holder
        # Keep each child's own (flat-frame, possibly animated) local transform and
        # simply apply the holder's transform on top: world = holder @ local.
        obj.matrix_parent_inverse = M.Identity(4)
    return holder


def _wire_arms(right_hand):
    """Point each bassist arm's IK wrist target at the matching hand rig so the
    arm reaches out to wherever that hand plays. Left arm (+X, the player's fret
    side) follows the FretHand; right arm follows the pluck/pick hand. Hide the
    blocky bassist hand stubs so the detailed hand rigs show instead."""
    pairs = {"L": "FretHand", "R": right_hand}
    for side, hand_name in pairs.items():
        target = bpy.data.objects.get(f"Wrist_{side}")
        hand = bpy.data.objects.get(hand_name)
        if target is None or hand is None:
            continue
        for c in list(target.constraints):
            target.constraints.remove(c)
        con = target.constraints.new('COPY_LOCATION')
        con.target = hand
        con.subtarget = "wrist"          # the hand rig's wrist bone (the joint)
        # Re-aim this arm's elbow pole for the playing pose where needed, so the
        # forearm drapes over the bass body instead of intersecting it.
        pole_pos = ELBOW_POLE_OVERRIDE.get(side)
        pole = bpy.data.objects.get(f"Elbow_{side}")
        if pole_pos is not None and pole is not None:
            pole.location = pole_pos
        # Hide the bassist's placeholder fist -- the hand rig replaces it.
        stub = bpy.data.objects.get(f"Hand_{side}")
        if stub is not None:
            stub.hide_viewport = True
            stub.hide_render = True


def _check_wrist_pose(arm, right_hand, frames):
    """Validate each playing wrist against human range of motion, sampled over
    `frames` (so the pluck/pick swing extremes are covered too). The hand rigs are
    oriented by AUTHORED matrices, not by a limited joint, so this is the only thing
    standing between a bad build_hands re-solve and a broken wrist. Raises
    RuntimeError -- loud at build time -- on:
      * a wrist bent more than WRIST_BEND_MAX off the incoming forearm, or
      * a PICKING/PLUCKING palm that faces AWAY from the strings (out toward the
        audience) instead of onto them.
    The fret hand's palm direction is left free (it wraps the neck); only its bend
    is capped."""
    scene = bpy.context.scene
    holder = bpy.data.objects.get(HOLDER_NAME)
    into_face = None
    if holder is not None:
        # Strings sit on the bass face; its inward normal is where a picking palm
        # must point. Holder local +Z is OUT of the face, so -Z points into it.
        into_face = (holder.matrix_world.to_quaternion().to_matrix()
                     @ V((0.0, 0.0, -1.0))).normalized()
    pairs = {"L": "FretHand", "R": right_hand}
    saved = scene.frame_current
    worst_bend = {}   # side -> max wrist bend (rad), on the base finger axis
    worst_face = {}   # side -> min palm.into_face, on the POSED palm
    for f in frames:
        scene.frame_set(int(round(f)))
        bpy.context.view_layer.update()
        for side, hand_name in pairs.items():
            fore = arm.pose.bones.get(f"forearm.{side}")
            hand = bpy.data.objects.get(hand_name)
            if fore is None or hand is None:
                continue
            e = (arm.matrix_world @ fore.tail
                 - arm.matrix_world @ fore.head).normalized()
            # BEND = authored wrist straightness: forearm vs the hand's BASE finger
            # axis (object +Y), not the transient pluck swing.
            finger = (hand.matrix_world.to_3x3() @ V((0.0, 1.0, 0.0))).normalized()
            bend = e.angle(finger)
            if side not in worst_bend or bend > worst_bend[side]:
                worst_bend[side] = bend
            # PALM facing must be read on the POSED wrist bone (obj @ pose): the pick
            # swing ROLLS the palm during play, so the object palm alone is not what
            # actually faces the strings. PALM is the hand's local -Z (fingers curl
            # toward -z, the pick is pinched there); +Z is the back of the hand.
            wb = hand.pose.bones.get("wrist")
            Rp = ((hand.matrix_world @ wb.matrix).to_3x3() if wb is not None
                  else hand.matrix_world.to_3x3())
            face = (Rp @ V((0.0, 0.0, -1.0))).normalized().dot(into_face) \
                if into_face is not None else 1.0
            if side not in worst_face or face < worst_face[side]:
                worst_face[side] = face
    scene.frame_set(saved)
    for side, hand_name in pairs.items():
        if worst_bend.get(side, 0.0) > WRIST_BEND_MAX:
            raise RuntimeError(
                f"{hand_name} wrist is bent {math.degrees(worst_bend[side]):.0f} deg "
                f"off the forearm (human max ~{math.degrees(WRIST_BEND_MAX):.0f}); the "
                f"hand orientation in build_hands is anatomically impossible -- re-solve.")
        if side == "R" and into_face is not None \
                and worst_face.get(side, 1.0) < PALM_TO_STRINGS_MIN:
            raise RuntimeError(
                f"{hand_name} palm faces AWAY from the strings (posed palm.into-face="
                f"{worst_face[side]:+.2f}, need >= {PALM_TO_STRINGS_MIN}); the picking "
                f"wrist is rotated the wrong way -- fix PICK orientation in build_hands "
                f"so the palm turns onto the strings.")


# ---------------------------------------------------------------------------
# Bass strap
# ---------------------------------------------------------------------------
def _strap_material():
    mat = bpy.data.materials.get("BassStrap") or bpy.data.materials.new("BassStrap")
    mat.use_nodes = True
    b = mat.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.05, 0.05, 0.06, 1.0)  # black nylon
    b.inputs["Roughness"].default_value = 0.75
    return mat


def _sweep_tube(points, width, thick, out_hints):
    """A rectangular-section band swept through world-space `points`. Each point
    carries an OUTWARD hint (the direction its thin face should face -- away from
    the body); the band's flat (wide) faces then stay against/away from the body
    with no accumulated twist, and the thin edge rolls over the shoulder. Returns
    a mesh whose verts are in world space."""
    pts = [V(p) for p in points]
    m = len(pts)
    tans = []
    for i in range(m):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == m - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        tans.append(t.normalized() if t.length > 1e-9 else V((0.0, 0.0, 1.0)))

    def _perp(vec, t):
        v = vec - vec.dot(t) * t
        return v.normalized() if v.length > 1e-9 else t.orthogonal().normalized()

    normals = [_perp(V(h), tans[i]) for i, h in enumerate(out_hints)]

    verts, faces = [], []
    hw, ht = width / 2.0, thick / 2.0
    for i in range(m):
        P, t, n = pts[i], tans[i], normals[i]
        s = t.cross(n).normalized()
        verts += [tuple(P + s * hw + n * ht), tuple(P - s * hw + n * ht),
                  tuple(P - s * hw - n * ht), tuple(P + s * hw - n * ht)]
    for i in range(m - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, a + k2, b + k2, b + k))
    faces.append((0, 1, 2, 3))                          # end caps
    a = 4 * (m - 1)
    faces.append((a + 3, a + 2, a + 1, a + 0))
    mesh = bpy.data.meshes.new(STRAP_NAME + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def _build_strap(arm):
    """Sweep a strap from the bass's lower end-pin, around the player's back and
    over the left shoulder, down to the upper strap button, and parent it to the
    torso so it moves with the body (like the mounted bass)."""
    bg = _load("build_bass_guitar")
    T = _play_transform()

    def gw(local):                                      # bass-local -> world
        return tuple(T @ V(local))

    # Strap buttons on the bass body edges (bottom end-pin + upper-bout button).
    lower = gw((0.0, bg.BODY_Y0 + 0.006, -bg.BODY_THICK / 2.0))
    upper = gw((-0.10, bg.BODY_Y1 - 0.02, -bg.BODY_THICK / 2.0))
    # Left shoulder joint, in world.
    sL = arm.matrix_world @ arm.pose.bones["upper_arm.L"].bone.head_local
    back = 0.14   # how far the strap stands off the back of the torso (+Y = behind)

    # Control path: end-pin -> round the right hip to the back -> up the back -> over
    # the left shoulder (back, top, front) -> down the front to the upper button. The
    # end-pin already sits out at the right hip, so the first waypoint curves INWARD
    # toward the back rather than further out to the right.
    pts = [
        lower,
        (lower[0] + 0.09, lower[1] + 0.16, lower[2] + 0.05),         # round the right hip back
        (-0.12, back, 0.5 * (lower[2] + sL[2]) - 0.05),             # up the back, right of centre
        (0.06, back, sL[2] - 0.02),                                 # up the back, nearing the shoulder
        (sL[0] - 0.02, back * 0.6, sL[2] + 0.05),                   # back of the left shoulder
        (sL[0] + 0.02, -0.03, sL[2] + 0.08),                        # over the top of the shoulder
        (sL[0] - 0.02, -0.12, sL[2] - 0.04),                        # front of the shoulder
        (0.5 * (sL[0] + upper[0]), upper[1] - 0.03,                  # down the front
         0.5 * (sL[2] + upper[2]) + 0.03),
        upper,
    ]
    # Outward hint per waypoint (where the strap's thin face points -- away from the
    # body): front waypoints face forward (-Y), back waypoints face back (+Y), the
    # right-hip wrap faces right (-X), the shoulder-top faces up (+Z).
    hints = [
        (0.0, -1.0, -0.3),      # lower button: front/down
        (-0.7, 0.5, 0.0),       # right hip: out to the right, curling back
        (-0.3, 1.0, 0.0),       # back, right of centre
        (0.1, 1.0, 0.0),        # back
        (0.2, 0.8, 0.4),        # back of shoulder
        (0.2, 0.0, 1.0),        # over the top of the shoulder: up
        (0.2, -0.9, 0.2),       # front of shoulder
        (0.05, -1.0, 0.0),      # down the front
        (0.0, -1.0, -0.2),      # upper button: front
    ]
    mesh = _sweep_tube(pts, STRAP_WIDTH, STRAP_THICK, hints)
    obj = bpy.data.objects.new(STRAP_NAME, mesh)
    obj.data.materials.append(_strap_material())
    coll = bpy.data.collections.get("Bassist") or bpy.context.scene.collection
    coll.objects.link(obj)
    # Move with the torso, like the bass (verts are already in world space).
    obj.parent = arm
    obj.parent_type = 'BONE'
    obj.parent_bone = COUPLE_BONE
    bpy.context.view_layer.update()
    obj.matrix_world = M.Identity(4)
    return obj


# ---------------------------------------------------------------------------
# Body performance
# ---------------------------------------------------------------------------
BODY_SWAY_PERIOD = 3.2     # s per sway cycle -- a slow, relaxed groove
BODY_LEAN = math.radians(2.5)   # forward lean amplitude (into the instrument)
BODY_TWIST = math.radians(3.5)  # torso twist amplitude
HEAD_NOD = math.radians(3.0)    # head bob amplitude
BODY_SAMPLE = 0.2          # s between body keyframes


def _animate_body(arm, duration, fps, frame_start):
    """Slow torso sway + head bob over the take. Feet stay planted (leg IK) and
    the arms re-solve to keep the hands on the strings, so the body can groove
    without pulling the hands off the bass."""
    if duration <= 0.0:
        return frame_start
    spine, head = arm.pose.bones.get("spine"), arm.pose.bones.get("head")
    sp_path = 'pose.bones["spine"].rotation_euler'
    hd_path = 'pose.bones["head"].rotation_euler'

    t, last = 0.0, frame_start
    while t <= duration + 1e-6:
        ph = 2.0 * math.pi * t / BODY_SWAY_PERIOD
        frame = frame_start + t * fps
        if spine is not None:
            # X leans forward (breath), Y twists about the vertical bone axis.
            lean = BODY_LEAN * (0.5 - 0.5 * math.cos(2.0 * ph))
            spine.rotation_euler = (lean, BODY_TWIST * math.sin(ph), 0.0)
            arm.keyframe_insert(data_path=sp_path, frame=frame)
        if head is not None:
            head.rotation_euler = (HEAD_NOD * math.sin(2.0 * ph), 0.0,
                                   -0.4 * BODY_TWIST * math.sin(ph))
            arm.keyframe_insert(data_path=hd_path, frame=frame)
        last = frame
        t += BODY_SAMPLE

    act = arm.animation_data.action if arm.animation_data else None
    if act is not None:
        from piano.piano_midi_animator import _iter_action_fcurves
        for fc in _iter_action_fcurves(act):
            if fc.data_path in (sp_path, hd_path):
                for kp in fc.keyframe_points:
                    kp.interpolation = 'SINE'
    return last


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def _setup_camera(scene):
    """Frame the standing bassist from a front-3/4 angle."""
    cam = bpy.data.objects.get("Camera")
    if cam is None or cam.type != 'CAMERA':
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(cam)
    cam.location = (0.8, -3.15, 1.05)
    look = V((0.05, 0.0, 0.88))
    cam.rotation_euler = (look - V(cam.location)).to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 42
    scene.camera = cam


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def animate_bassist(fingering_json=None, fps=24, frame_start=1, build=True,
                    camera=True):
    """Build (optionally), animate the hands, mount the bass on the standing
    bassist, make the arms follow the hands, and add a body groove."""
    if fingering_json is None:
        fingering_json = os.path.join(_HERE, "fingering.json")

    with open(fingering_json) as fh:
        data = json.load(fh)
    style = data.get("style", "finger")
    right_hand = _RIGHT_HAND.get(style, "PluckHand")

    scene = bpy.context.scene
    _reset_assembly()
    if build:
        _ensure_built(style)
    _require("Bassist", "FretHand", right_hand)

    arm = bpy.data.objects["Bassist"]

    # 1. Author the hand performance in the bass's flat frame.
    animate_hands = _load("animate_hands").animate_hands
    hand_result = animate_hands(fingering_json, fps=fps, frame_start=frame_start)

    # 2. Lift the whole bass+hands assembly into playing position (bone-parented
    #    to the torso so it rides the body), 3. strap it on, 4. wire the bassist's
    #    arms to follow the hands.
    _place_assembly(arm)
    _build_strap(arm)
    _wire_arms(right_hand)

    # 4b. Guard: the wrists are authored, not joint-limited -- verify they sit
    #     within human ROM (sampled across the take) before going further.
    fend = hand_result["frame_end"]
    _check_wrist_pose(arm, right_hand,
                      [frame_start + t * (fend - frame_start) for t in
                       (0.0, 0.25, 0.5, 0.75, 1.0)])

    # 5. Body groove for the length of the take.
    duration = max((n["end"] for n in data["notes"]), default=0.0)
    _animate_body(arm, duration, fps, frame_start)

    if camera:
        _setup_camera(scene)

    scene.frame_start = frame_start
    scene.frame_end = max(scene.frame_end, hand_result["frame_end"])
    scene.frame_set(frame_start)

    return {"bassist": arm.name, "frame_end": scene.frame_end, "fps": fps,
            **hand_result}


if __name__ == "__main__":
    animate_bassist()
