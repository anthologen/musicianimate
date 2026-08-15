"""Generate a MIDI that drives the pianist to the two EXTREME ENDS of the
88-key board - a reach test rather than a musical demo.

Everything else in piano/ has been exercised around the middle of the
keyboard (make_demo_piano_midi.py never leaves C2..C6), so the arm IK, the
wrist range-of-motion guard in animate_pianist.py and the head-follows-hands
gaze have never been asked for a full-span reach. This piece is built to
ask for exactly that:

  1. Bottom edge  - the lowest four keys (A0 A#0 B0 C1) one at a time, then
     the A0+A1 octave held, so the left hand parks on the very last key.
  2. Top edge     - the same at the other end (C8 B7 A#7 A7, then C7+C8),
     so the right hand parks on the very first key it can reach no further
     than.
  3. Extreme alternation - A0 and C8 traded back and forth, first slowly
     then in eighths, holding both hands at opposite ends of the case while
     the body has to keep both arms out there.
  4. Contrary-motion sweep - both hands start together at middle C and walk
     outward in octave leaps to A0 / C8, then back in. This is the part that
     makes the wrists traverse the whole 1.2 m span continuously, so the
     glide, the gaze and the shoulders all get sampled everywhere in between.
  5. Finale - a low A0+E1 under a high G7+C8, the widest chord the board
     allows, held loud.

Usage::

    python -m piano.make_reach_piano_midi piano/piano_reach.mid
"""

import os
import sys

try:
    from piano.make_demo_piano_midi import write_midi
except ImportError:  # running as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.make_demo_piano_midi import write_midi

try:
    from piano import key_layout
except ImportError:  # loose script
    import key_layout


LOW, HIGH = key_layout.START_MIDI, key_layout.END_MIDI   # A0 = 21, C8 = 108


def build_reach_notes(tpb):
    notes = []

    # Lead-in silence: animate_hands glides the wrist into place before the
    # first press, and the first press here is at the very end of the board.
    t = tpb

    # 1. Bottom edge: the lowest four keys, one at a time, then the octave.
    for i, note in enumerate((LOW, LOW + 1, LOW + 2, LOW + 3)):  # A0 A#0 B0 C1
        notes.append((t + i * tpb, int(tpb * 0.85), note, 80))
    t += 4 * tpb
    for note in (LOW, LOW + 12):                                 # A0 + A1
        notes.append((t, int(tpb * 1.8), note, 95))
    t += tpb * 3

    # 2. Top edge: the same at the treble end.
    for i, note in enumerate((HIGH, HIGH - 1, HIGH - 2, HIGH - 3)):  # C8 B7 A#7 A7
        notes.append((t + i * tpb, int(tpb * 0.85), note, 80))
    t += 4 * tpb
    for note in (HIGH - 12, HIGH):                               # C7 + C8
        notes.append((t, int(tpb * 1.8), note, 95))
    t += tpb * 3

    # 3. Extreme alternation - both hands held at opposite ends of the case.
    #    Four slow trades, then eight fast ones (the hands cannot help each
    #    other here, so each stays parked at its own end).
    for i in range(4):
        notes.append((t + i * tpb, int(tpb * 0.9), LOW if i % 2 == 0 else HIGH, 90))
    t += 4 * tpb
    eighth = tpb // 2
    for i in range(8):
        notes.append((t + i * eighth, int(eighth * 0.85),
                      LOW if i % 2 == 0 else HIGH, 100))
    t += 8 * eighth + tpb

    # 4. Contrary-motion sweep: from middle C outward to both ends in octaves,
    #    then back in. The left hand descends 59 -> 21 and the right climbs
    #    67 -> 108, so between them the wrists cross the entire keyboard. The
    #    lines start a fifth apart, straddling middle C rather than meeting on
    #    it: two hands on one key is not a reach test (and a unison is one key
    #    for one finger), and starting both above C4 lets the splitter hand the
    #    whole first event to the right hand.
    down = [59, 48, 36, 24, LOW]           # B3 C3 C2 C1 A0
    up = [67, 79, 91, 103, HIGH]           # G4 G5 G6 G7 C8
    out_and_back = list(zip(down, up)) + list(zip(down[-2::-1], up[-2::-1]))
    for i, (lo, hi) in enumerate(out_and_back):
        start = t + i * tpb
        notes.append((start, int(tpb * 0.9), lo, 85))
        notes.append((start, int(tpb * 0.9), hi, 85))
    t += len(out_and_back) * tpb + tpb

    # 5. Finale: the widest chord on the board, held.
    final_dur = tpb * 4
    for note in (LOW, LOW + 7, HIGH - 5, HIGH):   # A0 + E1 under G7 + C8
        notes.append((t, final_dur, note, 110))
    t += final_dur

    return notes, t


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "piano_reach.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    reach_notes, total_ticks = build_reach_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, reach_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    pitches = [n[2] for n in reach_notes]
    print(f"Wrote {out_path}: {len(reach_notes)} notes, "
          f"MIDI {min(pitches)}..{max(pitches)}, ~{seconds:.2f}s at 120 BPM")
