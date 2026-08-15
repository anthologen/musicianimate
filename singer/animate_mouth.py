"""Keyframes the 2D vector mouth from a phoneme timeline.

Reads the timeline.json produced by singer/timeline.py and drives the shape
keys built by singer/build_mouth.py, so the mouth interpolates from one
viseme position to the next in sync with the sung syllables.

The crossfade, which is the whole animation model:

    event i-1            event i              event i+1
    ......____           ____________           ____......
            \\ b        / b        \\ b        / b
             \\________/            \\________/
                  |<-- holds at 1 -->|

Each event's shape key ramps 0 -> 1 across its opening boundary, holds at 1
for the body of the event, and ramps 1 -> 0 across its closing boundary.
Both events sharing a boundary use the SAME ramp width, so their weights
always sum to 1 - and because Blender evaluates shape keys as
``basis + sum(weight * offset)``, a pair of weights summing to 1 is exactly
the interpolation between those two mouth positions.  Nothing else is ever
keyed, so the mouth can never drift toward a shape that is not a viseme.

Ramp width scales with the shorter of the two events (BLEND_FRAC), capped at
BLEND_MAX: a held vowel eases, a 50 ms consonant snaps.

Usage (inside Blender / MCP)::

    import singer.build_mouth as bm, singer.animate_mouth as am
    bm.build_mouth()
    am.animate_mouth("singer/timeline.json")
"""

import json
import os
import sys

import bpy

try:
    from . import mouth_shapes
except ImportError:  # loaded as a loose script via importlib, not a package
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import mouth_shapes

BLEND_FRAC = 0.40      # ramp width as a share of the shorter adjacent event
BLEND_MAX = 0.060      # seconds; long holds still transition briskly
BLEND_MIN = 0.004      # keep the two ramp keys on distinct sub-frames

MOUTH_NAME = "Mouth"


# ---------------------------------------------------------------------------
# Blender 4.4+ moved fcurves onto Action layers/strips/channelbags.
# ---------------------------------------------------------------------------
def _iter_action_fcurves(action):
    if action is None:
        return
    if getattr(action, "is_action_legacy", True):
        for fcurve in action.fcurves:
            yield fcurve
        return
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in getattr(strip, "channelbags", ()):
                for fcurve in channelbag.fcurves:
                    yield fcurve


# ---------------------------------------------------------------------------
# Timeline preparation
# ---------------------------------------------------------------------------
def _merge_runs(events):
    """Collapse neighbouring events that want the same mouth position.

    Two events on the same viseme AND loudness would otherwise both try to
    keyframe the same shape key at the shared boundary - once to 0 and once
    to 1, at the same frame.  Musically they are one held position anyway.
    """
    merged = []
    for event in events:
        key = (event["viseme"], event["level"])
        if merged and (merged[-1]["viseme"], merged[-1]["level"]) == key:
            merged[-1] = dict(merged[-1], end=event["end"])
        else:
            merged.append(dict(event))
    return merged


def _blend_widths(events):
    """Ramp half-width at each boundary, shared by the events either side."""
    widths = [0.0] * (len(events) + 1)
    for index in range(1, len(events)):
        span = min(events[index - 1]["end"] - events[index - 1]["start"],
                   events[index]["end"] - events[index]["start"])
        widths[index] = max(BLEND_MIN, min(BLEND_MAX, BLEND_FRAC * span))
    return widths


# ---------------------------------------------------------------------------
# Keyframing
# ---------------------------------------------------------------------------
def animate_mouth(timeline_json, mouth=None, fps=24, frame_start=1,
                  set_range=True):
    """Keyframe the mouth's shape keys from a timeline.json.  Returns
    (frame_start, frame_end)."""
    with open(timeline_json, "r", encoding="utf-8") as f:
        timeline = json.load(f)

    mouth = mouth or bpy.data.objects.get(MOUTH_NAME)
    if mouth is None:
        raise RuntimeError(f"no {MOUTH_NAME!r} object - run build_mouth first")
    shape_keys = mouth.data.shape_keys
    if shape_keys is None:
        raise RuntimeError(f"{mouth.name} has no shape keys")
    blocks = shape_keys.key_blocks

    events = _merge_runs(timeline["events"])
    widths = _blend_widths(events)

    # Everything starts closed and stays closed unless an event says otherwise.
    shape_keys.animation_data_clear()
    for block in blocks:
        block.value = 0.0

    def frame_of(seconds):
        return frame_start + seconds * fps

    def key(block, seconds, value):
        block.value = value
        block.keyframe_insert(data_path="value", frame=frame_of(seconds))

    used = set()
    for index, event in enumerate(events):
        name = mouth_shapes.key_name(event["viseme"], event["level"])
        block = blocks.get(name)
        if block is None:
            raise RuntimeError(f"shape key {name!r} missing - rebuild the mouth")
        used.add(name)
        b_in, b_out = widths[index], widths[index + 1]
        start, end = event["start"], event["end"]
        if b_in:
            key(block, start - b_in, 0.0)
        key(block, start + b_in, 1.0)
        key(block, end - b_out, 1.0)
        if b_out:
            key(block, end + b_out, 0.0)

    for block in blocks:
        block.value = 0.0
    if events:
        blocks[mouth_shapes.key_name(events[0]["viseme"],
                                     events[0]["level"])].value = 1.0

    # Smooth, symmetric ramps: auto-clamped handles on a monotone key pair
    # give an S-curve and its exact complement, so the pair still sums to 1.
    for fcurve in _iter_action_fcurves(shape_keys.animation_data.action):
        for point in fcurve.keyframe_points:
            point.interpolation = 'BEZIER'
            point.handle_left_type = point.handle_right_type = 'AUTO_CLAMPED'
        fcurve.update()

    frame_end = int(round(frame_of(timeline["duration"]))) + 1
    scene = bpy.context.scene
    if set_range:
        scene.render.fps = fps
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        scene.frame_set(frame_start)
    print(f"{timeline['source']}: {len(timeline['events'])} events "
          f"({len(events)} after merging holds) -> frames "
          f"{frame_start}-{frame_end} @ {fps}fps, {len(used)} visemes used")
    return frame_start, frame_end


# ---------------------------------------------------------------------------
# Verification: the crossfade's promise is that the weights sum to 1 at every
# frame.  If they ever do not, the mouth is drifting toward the closed basis
# shape mid-transition, which reads as a flicker.
# ---------------------------------------------------------------------------
def check_weights(mouth=None, frame_start=None, frame_end=None):
    """Return (worst_deviation, frame) of sum(shape key weights) from 1.0."""
    mouth = mouth or bpy.data.objects.get(MOUTH_NAME)
    action = mouth.data.shape_keys.animation_data.action
    curves = list(_iter_action_fcurves(action))
    scene = bpy.context.scene
    frame_start = frame_start if frame_start is not None else scene.frame_start
    frame_end = frame_end if frame_end is not None else scene.frame_end

    worst, worst_frame = 0.0, frame_start
    for frame in range(int(frame_start), int(frame_end) + 1):
        for sub in (0.0, 0.5):
            total = sum(c.evaluate(frame + sub) for c in curves)
            if abs(total - 1.0) > worst:
                worst, worst_frame = abs(total - 1.0), frame + sub
    return worst, worst_frame


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[-1] if sys.argv[-1].endswith(".json") \
        else os.path.join(here, "timeline.json")
    animate_mouth(path)
    print("max weight-sum error: %.4f at frame %.1f" % check_weights())
