"""MIDI-driven piano key animation for Blender.

Parses a Standard MIDI File (no external dependencies) and keyframes the key
objects produced by the piano builder script: objects named ``WhiteKey_<midi>``
/ ``BlackKey_<midi>`` inside a collection (default "Piano"), where ``<midi>``
is the standard MIDI note number (21 = A0 ... 108 = C8).

Note-on velocity controls press speed, not press depth: a loud note reaches
the bottom of its key travel in fewer frames (a sharp attack), a soft note
takes longer to get there (a gentle attack) - matching how velocity works on
a real keyboard.

Usage (inside Blender's Python console / MCP execute_blender_code)::

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "piano_midi_animator", "/path/to/piano_midi_animator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.animate_piano("/path/to/song.mid")
"""

import struct


# ---------------------------------------------------------------------------
# Minimal Standard MIDI File (SMF) reader
# ---------------------------------------------------------------------------

def _read_vlq(data, i):
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, i


def parse_midi(path):
    """Return (ticks_per_beat, tempo_map, events).

    tempo_map: sorted list of (tick, microseconds_per_beat).
    events: list of dicts {tick, type: 'on'|'off', note, velocity}.
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[0:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd header)")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:8 + header_len])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    ticks_per_beat = division

    pos = 8 + header_len
    events = []
    tempo_map = []

    for _track_idx in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("Malformed track chunk")
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track_end = pos + 8 + track_len
        i = pos + 8
        tick = 0
        running_status = None
        while i < track_end:
            delta, i = _read_vlq(data, i)
            tick += delta
            status = data[i]
            if status < 0x80:
                status = running_status  # running status: byte already consumed
            else:
                i += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                meta_type = data[i]
                i += 1
                length, i = _read_vlq(data, i)
                meta_data = data[i:i + length]
                i += length
                if meta_type == 0x51:  # Set Tempo
                    us_per_beat = struct.unpack(">I", b"\x00" + meta_data)[0]
                    tempo_map.append((tick, us_per_beat))
            elif status in (0xF0, 0xF7):
                length, i = _read_vlq(data, i)
                i += length
            else:
                event_type = status & 0xF0
                if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1, d2 = data[i], data[i + 1]
                    i += 2
                    if event_type == 0x90 and d2 > 0:
                        events.append({"tick": tick, "type": "on", "note": d1,
                                       "velocity": d2, "track": _track_idx})
                    elif event_type == 0x90 or event_type == 0x80:
                        events.append({"tick": tick, "type": "off", "note": d1,
                                       "velocity": d2, "track": _track_idx})
                elif event_type in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
        pos = track_end

    if not tempo_map:
        tempo_map = [(0, 500000)]  # default 120 BPM
    tempo_map.sort(key=lambda t: t[0])

    return ticks_per_beat, tempo_map, events


def _tick_to_seconds(tick, ticks_per_beat, tempo_map):
    seconds = 0.0
    last_tick = 0
    last_tempo = tempo_map[0][1]
    for t, tempo in tempo_map:
        if t >= tick:
            break
        seconds += (t - last_tick) * (last_tempo / 1_000_000.0) / ticks_per_beat
        last_tick = t
        last_tempo = tempo
    seconds += (tick - last_tick) * (last_tempo / 1_000_000.0) / ticks_per_beat
    return seconds


# ---------------------------------------------------------------------------
# Blender animation
# ---------------------------------------------------------------------------

def _iter_action_fcurves(action):
    """Yield fcurves from an Action, supporting both the legacy fcurve-list
    API and Blender 4.4+'s layered Action (layers -> strips -> channelbags)."""
    if action is None:
        return
    if getattr(action, "is_action_legacy", True):
        for fc in action.fcurves:
            yield fc
        return
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type != 'KEYFRAME':
                continue
            for channelbag in strip.channelbags:
                for fc in channelbag.fcurves:
                    yield fc

def animate_piano(
    midi_path,
    collection_name="Piano",
    fps=24,
    press_depth_white=0.0075,
    press_depth_black=0.0045,
    min_attack_frames=1.0,
    max_attack_frames=9.0,
    release_frames=5.0,
    frame_start=1,
    clear_existing=True,
):
    """Keyframe the piano's key objects to match a MIDI file's note events."""
    import bpy

    scene = bpy.context.scene
    scene.render.fps = fps
    scene.render.fps_base = 1.0

    ticks_per_beat, tempo_map, raw_events = parse_midi(midi_path)

    coll = bpy.data.collections.get(collection_name)
    if coll is None:
        raise RuntimeError(f"Collection '{collection_name}' not found")

    key_objs = {}
    for obj in coll.objects:
        if obj.name.startswith("WhiteKey_") or obj.name.startswith("BlackKey_"):
            midi_note = int(obj.name.split("_")[1])
            key_objs[midi_note] = obj

    # Pair each note-on with the next note-off for the same note number
    # (FIFO), so overlapping/retriggered notes are handled sensibly.
    open_notes = {}
    notes = []  # (on_tick, off_tick, note, velocity)
    for ev in sorted(raw_events, key=lambda e: e["tick"]):
        note = ev["note"]
        if ev["type"] == "on":
            open_notes.setdefault(note, []).append((ev["tick"], ev["velocity"]))
        else:
            stack = open_notes.get(note)
            if stack:
                on_tick, vel = stack.pop(0)
                notes.append((on_tick, ev["tick"], note, vel))

    # Cache each key's rest Z the first time it's animated, as a custom
    # property on the object. Re-running this function later (with the key
    # mid-animation, at an arbitrary current frame) must not mistake a
    # currently-pressed position for the rest position.
    rest_z = {}
    for note, obj in key_objs.items():
        if "piano_rest_z" in obj:
            z = obj["piano_rest_z"]
        else:
            z = obj.location.z
            obj["piano_rest_z"] = z
        rest_z[note] = z
        if clear_existing:
            obj.animation_data_clear()
        obj.location.z = z

    def tick_to_frame(tick):
        return frame_start + _tick_to_seconds(tick, ticks_per_beat, tempo_map) * fps

    def insert(obj, frame, z):
        obj.location.z = z
        obj.keyframe_insert(data_path="location", index=2, frame=frame)

    last_frame = frame_start
    animated = 0
    for on_tick, off_tick, note, vel in notes:
        obj = key_objs.get(note)
        if obj is None:
            continue  # note outside the 88-key range, skip

        depth = press_depth_black if obj.name.startswith("BlackKey_") else press_depth_white
        base_z = rest_z[note]
        pressed_z = base_z - depth

        t = max(0, min(127, vel)) / 127.0
        attack_frames = max_attack_frames - t * (max_attack_frames - min_attack_frames)

        on_frame = tick_to_frame(on_tick)
        off_frame = max(on_frame, tick_to_frame(off_tick))
        attack_start_frame = max(frame_start, on_frame - attack_frames)

        insert(obj, attack_start_frame, base_z)
        insert(obj, on_frame, pressed_z)
        insert(obj, off_frame, pressed_z)
        insert(obj, off_frame + release_frames, base_z)

        last_frame = max(last_frame, off_frame + release_frames)
        animated += 1

    for obj in key_objs.values():
        for fcurve in _iter_action_fcurves(obj.animation_data and obj.animation_data.action):
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'SINE'

    scene.frame_start = frame_start
    scene.frame_end = int(round(last_frame)) + fps
    scene.frame_set(frame_start)

    return {"notes_animated": animated, "frame_end": scene.frame_end, "fps": fps}
