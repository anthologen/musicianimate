"""Scan the FretHand animation for inter-finger collisions.

A regression tool for the bass animation engine: rebuilds the bass scene
and hands, animates bass_guitar/fingering.json, then for every frame
measures the minimum axis distance between the bone segments
(prox/mid/dist capsules) of each pair of fretting fingers on the
evaluated rig. Finger boxes are ~12 mm square, so axis distances around
12 mm mean the meshes sit flush and below ~10 mm they visibly
interpenetrate.

Usage::

    blender --background --python bass_guitar/scan_finger_collisions.py
"""
import os
import sys

import bpy
import mathutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from bass_guitar import build_bass_guitar, build_hands, animate_hands  # noqa: E402

build_bass_guitar.build_bass_guitar()
build_hands.build_hands()
animate_hands.animate_hands(os.path.join(ROOT, "bass_guitar/fingering.json"))

PENETRATE = 0.010  # visibly intersecting

scene = bpy.context.scene
arm = bpy.data.objects["FretHand"]
segs = [(f, seg) for f in (1, 2, 3, 4) for seg in ("prox", "mid", "dist")]


def seg_dist(p1, q1, p2, q2):
    """Min distance between segments p1-q1 and p2-q2."""
    r1, r2 = mathutils.geometry.intersect_line_line(p1, q1, p2, q2) or (None, None)
    best = None
    if r1 is not None:
        def clamp(pt, a, b):
            ab = b - a
            t = max(0.0, min(1.0, (pt - a).dot(ab) / max(ab.length_squared, 1e-12)))
            return a + t * ab
        c1 = clamp(r1, p1, q1)
        c2 = clamp(r2, p2, q2)
        c2 = clamp(c2, p2, q2)
        c1 = clamp(mathutils.geometry.intersect_point_line(c2, p1, q1)[0], p1, q1)
        best = (c1 - c2).length
    for a, b, c, d in ((p1, q1, p2, q2), (p2, q2, p1, q1)):
        for pt in (c, d):
            close, _ = mathutils.geometry.intersect_point_line(pt, a, b)
            ab = b - a
            t = max(0.0, min(1.0, (close - a).dot(ab) / max(ab.length_squared, 1e-12)))
            dist = ((a + t * ab) - pt).length
            best = dist if best is None else min(best, dist)
    return best


windows = {}  # (fa, fb) -> list of (start, end, min_dist, min_frame)
for frame in range(1, scene.frame_end + 1):
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ae = arm.evaluated_get(dg)
    mw = ae.matrix_world
    pts = {}
    for f, seg in segs:
        pb = ae.pose.bones[f"f{f}_{seg}"]
        pts[(f, seg)] = (mw @ pb.head, mw @ pb.tail)
    for fa in (1, 2, 3):
        for fb in range(fa + 1, 5):
            dmin = min(seg_dist(*pts[(fa, sa)], *pts[(fb, sb)])
                       for sa in ("prox", "mid", "dist")
                       for sb in ("prox", "mid", "dist"))
            if dmin < PENETRATE:
                key = (fa, fb)
                w = windows.setdefault(key, [])
                if w and w[-1][1] == frame - 1:
                    old = w[-1]
                    w[-1] = ((old[0], frame) +
                             ((dmin, frame) if dmin < old[2]
                              else (old[2], old[3])))
                else:
                    w.append((frame, frame, dmin, frame))

fps = scene.render.fps
print("=== collision windows (axis dist < %.0f mm) ===" % (PENETRATE * 1000))
if not windows:
    print("none")
for (fa, fb), spans in sorted(windows.items()):
    for s, e, d, mf in spans:
        if e - s >= 1:  # ignore single-frame grazes
            print(f"fingers {fa}+{fb}: frames {s}-{e}  min {d*1000:.1f} mm "
                  f"at frame {mf} (t {(s-1)/fps:.2f}-{(e-1)/fps:.2f}s)")
print("SCAN DONE")
