"""Generate a demo MIDI of ONE rhythmic strumming pattern (companion to
make_demo_guitar_midi.py, which stays a varied fingering test bench).

A G - Em - C - D open-chord progression (one bar each, played through twice)
strummed with the single most common folk/pop rhythm, "D D U U D U" -- down
on beat 1, down-up on beat 2, up on the '&' of 3, down-up on beat 4. Every
onset is a full open chord, so the right-hand animator reads each as a STRUM
(4+ strings) and the down/up direction falls out of the beat position (down
on the beat, up off it).

The eight bars swell from soft to loud and back (see BAR_LEVELS) so the
velocity-scaled strum motion is on display: quiet bars barely move the hand,
the loud middle bars throw big sweeps, and it settles back down. Within each
bar the downstrokes are accented a touch louder than the upstrokes for groove.

    python guitar/make_strum_guitar_midi.py guitar/strum_demo.mid
"""

import os
import sys

try:
    from piano.make_demo_piano_midi import write_midi
except ImportError:  # running as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.make_demo_piano_midi import write_midi


# Open-chord voicings (MIDI note numbers), one per bar of the progression.
G_MAJOR = [43, 47, 50, 55, 59, 67]   # 320003  G B D G B G
E_MINOR = [40, 47, 52, 55, 59, 64]   # 022000  E B E G B E
C_MAJOR = [48, 52, 55, 60, 64]       # x32010  C E G C E
D_MAJOR = [50, 57, 62, 66]           # xx0232  D A D F#
PROGRESSION = [G_MAJOR, E_MINOR, C_MAJOR, D_MAJOR] * 2

# The one strumming pattern, as eighth-note slots within a 4/4 bar and the
# stroke direction. Slot 0 = beat 1, slot 7 = the '&' of beat 4. The animator
# derives direction itself from the beat, but we record it here to accent the
# downstrokes. Pattern: 1(D) 2(D) &(U) (3) &(U) 4(D) &(U)  ==  D D U U D U.
PATTERN = [(0, True), (2, True), (3, False),
           (5, False), (6, True), (7, False)]
# Peak (downstroke) MIDI velocity per bar -- a swell from pp up to ff and back,
# so the strum motion visibly grows then shrinks with the volume.
BAR_LEVELS = [38, 58, 82, 118, 118, 82, 58, 38]
ACCENT = 16   # downstrokes hit this much harder than the upstrokes


def build_demo_notes(tpb):
    notes = []
    eighth = tpb // 2          # one strum slot
    bar = tpb * 4              # 4/4 bar
    lead_in = tpb              # a one-beat count-in before the groove

    # Lay out every strum onset first, so each chord can ring right up to the
    # next strum (each stroke re-strikes the strings cleanly, no overlap).
    strums = []                # (onset_tick, chord, velocity)
    for b, chord in enumerate(PROGRESSION):
        bar_start = lead_in + b * bar
        level = BAR_LEVELS[b]
        for slot, is_down in PATTERN:
            onset = bar_start + slot * eighth
            vel = level if is_down else max(28, level - ACCENT)
            strums.append((onset, chord, vel))

    for i, (onset, chord, vel) in enumerate(strums):
        nxt = strums[i + 1][0] if i + 1 < len(strums) else onset + bar
        dur = max(eighth // 2, nxt - onset - 10)   # ring until just before next
        for midi in chord:
            notes.append((onset, dur, midi, vel))

    total_ticks = lead_in + len(PROGRESSION) * bar + bar // 2
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "strum_demo.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, ~{seconds:.2f}s at 120 BPM")
