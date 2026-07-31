"""Generate a short demo MIDI file to exercise piano_midi_animator.py and
piano/fingering.py.

Plays a one-octave C major scale ascending from soft to loud (velocity 30
to 127), then a soft C major chord followed by the same chord played loud -
enough contrast to clearly show velocity driving key-press speed. A final
two-hand section (walking bass under a two-octave right-hand scale run,
closing on chords in both hands) exercises the fingering engine's hand
splitting and thumb-under crossings.
"""

import struct
import sys


def write_midi(path, ticks_per_beat, tempo_us_per_beat, notes, channel=0):
    """notes: list of (start_tick, duration_tick, note, velocity).

    `channel` (0-15) is OR'd into the note-on/off status nibble; it defaults
    to 0 so existing callers are unchanged. Pass channel=9 for General MIDI
    percussion ("channel 10"). The animator's parse_midi masks the channel
    away, so this only affects how the file auditions in a DAW.
    """
    events = [(0, "tempo", tempo_us_per_beat, None)]
    for start, dur, note, vel in notes:
        events.append((start, "on", note, vel))
        events.append((start + dur, "off", note, 0))

    order = {"tempo": 0, "off": 1, "on": 2}
    events.sort(key=lambda e: (e[0], order[e[1]]))

    def vlq(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(out)

    track = bytearray()
    last_tick = 0
    for tick, etype, a, b in events:
        track += vlq(tick - last_tick)
        last_tick = tick
        if etype == "tempo":
            track += bytes([0xFF, 0x51, 0x03]) + struct.pack(">I", a)[1:]
        elif etype == "on":
            track += bytes([0x90 | channel, a, b])
        elif etype == "off":
            track += bytes([0x80 | channel, a, b])
    track += vlq(0) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    track_chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)

    with open(path, "wb") as f:
        f.write(header)
        f.write(track_chunk)


def build_demo_notes(tpb):
    notes = []

    # Lead-in silence so the first note's (soft) attack has room to ramp
    # instead of being clamped against the start of the timeline.
    lead_in = tpb

    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 D4 E4 F4 G4 A4 B4 C5
    velocities = [30, 45, 60, 75, 90, 105, 115, 127]
    for i, (note, vel) in enumerate(zip(scale, velocities)):
        notes.append((lead_in + i * tpb, int(tpb * 0.85), note, vel))
    scale_end = lead_in + len(scale) * tpb

    # All five black keys of the octave, soft to loud, so accidentals get
    # exercised just like the white-key scale did.
    black_run = [61, 63, 66, 68, 70]  # C#4 D#4 F#4 G#4 A#4
    black_vel = [40, 60, 80, 100, 120]
    black_start = scale_end + tpb
    for i, (note, vel) in enumerate(zip(black_run, black_vel)):
        notes.append((black_start + i * tpb, int(tpb * 0.85), note, vel))
    black_end = black_start + len(black_run) * tpb

    chord_soft_start = black_end + tpb
    chord_dur = int(tpb * 1.7)
    for note in (60, 64, 67):
        notes.append((chord_soft_start, chord_dur, note, 40))

    # Loud dominant-7 chord: the added B-flat (70) is a black key.
    chord_loud_start = chord_soft_start + tpb * 2
    for note in (60, 64, 67, 70):
        notes.append((chord_loud_start, chord_dur, note, 122))

    # Two-hand finale: a walking bass in the left hand under a two-octave
    # C major run in eighth notes (forcing thumb-under crossings), closing
    # on a C major chord in both hands.
    duet_start = chord_loud_start + tpb * 3
    bass = [36, 43, 48, 43, 36, 43, 48, 43]  # C2 G2 C3 G2 ...
    for i, note in enumerate(bass):
        notes.append((duet_start + i * tpb, int(tpb * 0.9), note, 85))

    run = [60, 62, 64, 65, 67, 69, 71,
           72, 74, 76, 77, 79, 81, 83, 84]  # C4 .. C6
    eighth = tpb // 2
    for i, note in enumerate(run):
        notes.append((duet_start + i * eighth, int(eighth * 0.85), note, 95))
    duet_end = duet_start + len(bass) * tpb

    final_start = duet_end + tpb
    final_dur = tpb * 3
    for note in (36, 43):          # left hand: C2 + G2
        notes.append((final_start, final_dur, note, 100))
    for note in (60, 64, 67, 72):  # right hand: C4 E4 G4 C5
        notes.append((final_start, final_dur, note, 100))

    total_ticks = final_start + final_dur
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "demo.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, ~{seconds:.2f}s at 120 BPM")
