"""Generate a short demo MIDI file to exercise guitar/fingering.py.

Sections: an open-position C major scale, a chromatic run up the treble
strings, small open-chord grips (C, Am, and E fragments), barre chords
(F major and B minor), a high phrase that forces a position shift up the
neck and back, and two deliberate stress cases - a five-note cluster
(exercises chord simplification) and an out-of-range bass note
(exercises octave folding).
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
    lead_in = tpb

    # Open-position C major scale, soft to loud.
    scale = [48, 50, 52, 53, 55, 57, 59, 60]  # C3 .. C4
    velocities = [60, 68, 76, 84, 92, 100, 108, 116]
    for i, (note, vel) in enumerate(zip(scale, velocities)):
        notes.append((lead_in + i * tpb, int(tpb * 0.85), note, vel))
    scale_end = lead_in + len(scale) * tpb

    # Chromatic run G3..E4 in eighths: smooth position shifting.
    eighth = tpb // 2
    chrom_start = scale_end + tpb
    for i in range(10):
        notes.append((chrom_start + i * eighth, int(eighth * 0.85), 55 + i, 90))
    chrom_end = chrom_start + 10 * eighth

    # Small open-chord grips: C (five voices, three fretted), Am, E fragment.
    chord_dur = int(tpb * 1.7)
    c_start = chrom_end + tpb
    for note in (48, 52, 55, 60, 64):   # open C grip
        notes.append((c_start, chord_dur, note, 95))
    am_start = c_start + tpb * 2
    for note in (45, 52, 57, 60):       # Am grip
        notes.append((am_start, chord_dur, note, 90))
    e_start = am_start + tpb * 2
    for note in (40, 47, 52):           # E chord bottom
        notes.append((e_start, chord_dur, note, 100))

    # Barre chords: F major (E-shape barre, fret 1) and B minor (A-shape
    # barre, fret 2).
    f_start = e_start + tpb * 2
    for note in (41, 48, 53, 57, 60, 65):
        notes.append((f_start, chord_dur, note, 100))
    bm_start = f_start + tpb * 2
    for note in (47, 54, 59, 62, 66):
        notes.append((bm_start, chord_dur, note, 95))
    chords_end = bm_start + tpb * 2

    # High phrase up the neck and back down to open position.
    phrase = [64, 67, 69, 71, 72, 71, 69, 67, 64, 60, 55]
    phrase_start = chords_end + tpb
    for i, note in enumerate(phrase):
        notes.append((phrase_start + i * eighth, int(eighth * 0.85), note, 95))
    phrase_end = phrase_start + len(phrase) * eighth

    # Stress cases: an unplayable five-note cluster, then a bass note
    # below the guitar's range.
    cluster_start = phrase_end + tpb
    for note in (48, 50, 52, 53, 55):
        notes.append((cluster_start, tpb * 2, note, 85))
    low_start = cluster_start + tpb * 3
    notes.append((low_start, tpb, 30, 100))

    # Closing E chord.
    final_start = low_start + tpb * 2
    final_dur = tpb * 3
    for note in (40, 47, 52):
        notes.append((final_start, final_dur, note, 105))

    total_ticks = final_start + final_dur
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "guitar_demo.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, ~{seconds:.2f}s at 120 BPM")
