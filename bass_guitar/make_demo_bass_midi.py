"""Generate a short demo MIDI file to exercise bass_guitar/fingering.py.

Sections chosen to stress every part of the bass engine: an ascending
walking line low on the neck (Simandl 1-2-4 fingering + open strings), a
descending run across the strings (right-hand raking), a high phrase that
forces a position shift up the neck and back (one-finger-per-fret), a few
double-stops (a fifth and octaves), and one out-of-range low note
(exercises octave folding). The whole thing stays in the E1..G3 register.
"""

import os
import sys

try:
    from piano.make_demo_piano_midi import write_midi
except ImportError:  # running as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.make_demo_piano_midi import write_midi


def build_demo_notes(tpb):
    notes = []
    t = tpb  # one-beat lead-in

    def add(midi, beats, vel=88, advance=True):
        nonlocal t
        dur = int(tpb * beats * 0.9)
        notes.append((t, dur, midi, vel))
        if advance:
            t += int(tpb * beats)

    # Ascending walking line low on the neck: open E, up through first
    # position (Simandl 1-2-4), using open A/D strings as pivots.
    for midi in (28, 31, 33, 35, 36, 38, 40, 43):  # E1 G1 A1 B1 C2 D2 E2 G2
        add(midi, 0.5, vel=84)

    # Descending run across the strings - each step drops to a thicker
    # string, so the plucking hand rakes with one finger.
    for midi in (43, 40, 38, 33, 31, 28):          # G2 E2 D2 A1 G1 E1
        add(midi, 0.25, vel=92)

    t += tpb // 2  # small rest -> re-anchors plucking to the index finger

    # Double-stops: a root+fifth, then two octaves (index + ring/pinky).
    for root, upper, beats in ((28, 35, 1.0),   # E1 + B1 (a fifth)
                               (33, 45, 1.0),   # A1 + A2 (an octave)
                               (36, 48, 1.0)):  # C2 + C3 (an octave)
        dur = int(tpb * beats * 0.9)
        notes.append((t, dur, root, 96))
        notes.append((t, dur, upper, 90))
        t += int(tpb * beats)

    t += tpb // 2

    # High phrase up the neck and back: one-finger-per-fret territory.
    for midi in (48, 50, 52, 53, 55, 53, 52, 50, 48, 43):
        add(midi, 0.5, vel=90)

    t += tpb // 2

    # An out-of-range low note (below E1) - exercises octave folding.
    add(21, 1.0, vel=100)  # A0, folds up an octave into range

    # Closing low E.
    add(28, 2.0, vel=104)

    total_ticks = t
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "bass_demo.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, ~{seconds:.2f}s at 120 BPM")
