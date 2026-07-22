"""Generate a MIDI groove that shows off the fingerstyle right hand.

Unlike make_demo_bass_midi.py (which stresses the left-hand tab engine),
this one keeps the left hand simple - mostly open strings and first
position - so the *plucking* hand is the star. Four sections, each
isolating one fingerstyle behaviour:

  A. Steady eighth-note pedal on one string  -> pure i-m-i-m alternation.
  B. Octave riff (root + octave, two strings) -> alternation across an
     ascending string crossing, then a rake coming back down.
  C. Descending open-string run G->D->A->E    -> a textbook rake (the
     same finger drags across each thicker string).
  D. Fast sixteenth burst on one string       -> rapid i-m alternation.

Renders/animates via the normal pipeline; drives fingerstyle_demo.blend.
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
    eighth = tpb // 2
    sixteenth = tpb // 4

    def add(midi, dur_ticks, vel, at):
        notes.append((at, int(dur_ticks), midi, vel))

    # --- A. Eighth-note pedal on the open A string: pure i-m alternation.
    for i in range(8):
        add(33, eighth * 0.9, 82 + (i % 2) * 8, t + i * eighth)
    t += 8 * eighth
    t += eighth  # breath

    # --- B. Octave riff: A1 (open A) <-> A2 (G string, fret 2), funk feel.
    octave = [33, 45, 33, 45, 33, 45, 45, 33]
    for i, midi in enumerate(octave):
        add(midi, eighth * 0.85, 88, t + i * eighth)
    t += len(octave) * eighth
    t += eighth

    # --- C. Descending open-string run G2->D2->A1->E1: raking across the
    # thicker strings, twice.
    for _ in range(2):
        for midi in (43, 38, 33, 28):
            add(midi, eighth * 0.9, 92, t)
            t += eighth
        t += eighth

    # --- D. Fast sixteenth burst on the open E string: rapid alternation.
    for i in range(8):
        add(28, sixteenth * 0.85, 80 + (i % 2) * 10, t + i * sixteenth)
    t += 8 * sixteenth

    # Closing sustained low E.
    t += eighth
    add(28, tpb * 2, 100, t)
    t += tpb * 2

    return notes, t


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "fingerstyle_demo.mid"
    TPB = 480
    TEMPO_US = 600000  # 100 BPM, groovier

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, ~{seconds:.2f}s at 100 BPM")
