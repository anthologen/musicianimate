"""Generate a second demo MIDI drum track that showcases the full DYNAMIC
RANGE - soft to loud - on every piece of the kit, one at a time, so the
velocity -> wind-up/strike-speed scaling in animate_drums.py reads clearly
on each voice in isolation before anything overlaps.

Written on MIDI channel 10 (index 9), same GM percussion notes as
make_demo_drum_midi.py. Structure (each solo crescendo runs ppp -> fff,
velocity ~16 -> 127):

  1.  Kick        - 7-hit crescendo, quarter-note spacing.
  2.  Snare       - 7-hit crescendo, quarter-note spacing.
  3.  Side stick  - 5-hit crescendo, eighth-note spacing.
  4.  Toms        - hi, mid, floor each get their own 5-hit crescendo.
  5.  Hi-hat      - closed crescendo (foot down), then a closed/open
      soft/loud contrast pattern, then a pedal-chick crescendo (foot only).
  6.  Ride        - bow crescendo, then bell crescendo.
  7.  Crash / Crash2 / China / Splash - each a 4-hit crescendo, spaced out
      (2 beats) so every hit rings before the next.
  8.  Finale      - a sixteenth-note wave across kick/snare/hi-hat that
      swells ppp -> fff and back down, closing on a single unison fff hit
      (kick + snare + crash + ride bell) left to ring.
"""

import os
import sys

try:
    from piano.make_demo_piano_midi import write_midi
except ImportError:  # running as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.make_demo_piano_midi import write_midi

DRUM_CHANNEL = 9  # MIDI "channel 10" - General MIDI percussion

# GM percussion note numbers (same map as make_demo_drum_midi.py).
KICK = 36
SNARE = 38
SIDE_STICK = 37
HAT_CLOSED = 42
HAT_PEDAL = 44
HAT_OPEN = 46
TOM_HI = 50
TOM_MID = 45
TOM_FLOOR = 41
CRASH = 49
CRASH2 = 57
SPLASH = 55
CHINA = 52
RIDE = 51
RIDE_BELL = 53

# Dynamic staircases: ppp -> fff.
DYN_8 = [16, 33, 49, 64, 80, 96, 112, 127]
DYN_7 = [16, 34, 52, 70, 88, 106, 127]
DYN_5 = [16, 40, 64, 96, 127]
DYN_4 = [22, 58, 94, 127]


def build_demo_notes(tpb):
    """Return (notes, total_ticks). notes: (start_tick, dur_tick, note, vel)."""
    beat = tpb
    eighth = tpb // 2
    sixteenth = tpb // 4
    hit = tpb // 8  # short, fixed duration - drum hits are transient
    notes = []

    def add(tick, note, vel, dur=None):
        notes.append((tick, dur or hit, note, vel))

    t = beat  # lead-in

    def crescendo(note, dyns, step):
        nonlocal t
        for v in dyns:
            add(t, note, v)
            t += step
        t += beat  # let the last hit settle before the next section

    # --- 1. Kick --------------------------------------------------------
    crescendo(KICK, DYN_7, beat)

    # --- 2. Snare --------------------------------------------------------
    crescendo(SNARE, DYN_7, beat)

    # --- 3. Side stick -----------------------------------------------------
    crescendo(SIDE_STICK, DYN_5, eighth)

    # --- 4. Toms: hi, mid, floor, each their own crescendo -----------------
    crescendo(TOM_HI, DYN_5, beat)
    crescendo(TOM_MID, DYN_5, beat)
    crescendo(TOM_FLOOR, DYN_5, beat)

    # --- 5a. Hi-hat closed crescendo (foot holds the pedal down) -----------
    crescendo(HAT_CLOSED, DYN_7, eighth)

    # --- 5b. Closed/open, soft/loud contrast: the pedal rocks the top
    #     cymbal up for each open hit and back down for the next closed one,
    #     each pair louder than the last. ------------------------------------
    for closed_v, open_v in ((24, 70), (40, 96), (60, 116), (80, 127)):
        add(t, HAT_CLOSED, closed_v)
        t += eighth
        add(t, HAT_OPEN, open_v)
        t += eighth
    t += beat

    # --- 5c. Pedal-chick crescendo (foot only, no stick) --------------------
    crescendo(HAT_PEDAL, DYN_5, eighth)

    # --- 6. Ride: bow crescendo, then bell crescendo ------------------------
    crescendo(RIDE, DYN_7, beat)
    crescendo(RIDE_BELL, DYN_5, beat)

    # --- 7. Crash family: each a 4-hit crescendo, spaced 2 beats apart so
    #     every strike rings out before the next. ----------------------------
    for note in (CRASH, CRASH2, CHINA, SPLASH):
        for v in DYN_4:
            add(t, note, v, dur=beat)
            t += 2 * beat
        t += beat

    # --- 8. Finale: a sixteenth-note wave across kick/snare/hi-hat, swelling
    #     ppp -> fff and back down to ppp, then a final unison fff hit left
    #     to ring. -----------------------------------------------------------
    wave_notes = [KICK, HAT_CLOSED, SNARE, HAT_CLOSED] * 8  # 32 sixteenths
    n = len(wave_notes)
    for i, note in enumerate(wave_notes):
        # Triangular envelope: ramps 16 -> 127 across the first half,
        # 127 -> 16 across the second half.
        half = n / 2.0
        phase = i if i < half else (n - 1 - i)
        v = int(round(16 + (127 - 16) * (phase / (half - 1))))
        add(t, note, v)
        t += sixteenth
    t += beat
    add(t, KICK, 127, dur=beat * 2)
    add(t, SNARE, 127, dur=beat * 2)
    add(t, CRASH, 127, dur=beat * 2)
    add(t, RIDE_BELL, 127, dur=beat * 2)
    total_ticks = t + beat * 2

    notes.sort(key=lambda n: n[0])
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "drum_demo2.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes, channel=DRUM_CHANNEL)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} hits, ~{seconds:.2f}s at 120 BPM")
