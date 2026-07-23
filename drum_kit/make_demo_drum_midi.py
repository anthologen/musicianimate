"""Generate a short demo MIDI drum track to exercise drum_kit/fingering.py
and animate_drums.py.

Written on MIDI channel 10 (index 9) using General MIDI percussion note
numbers, so it auditions correctly in any GM player and drives the whole
drum pipeline. The arrangement is built to touch every piece of the kit at
least once while staying a musical groove:

  1-2. Basic rock beat  - kick on 1 & 3, snare on 2 & 4, closed hats in
       eighths: the core groove the sticking convention should nail.
  3.   Dynamics bar      - the same beat with a hi-hat crescendo and an
       accented backbeat, so wind-up visibly scales with velocity.
  4.   Tom fill          - a descending snare -> hi -> mid -> floor fill in
       sixteenths, exercising nearest-hand assignment across the kit.
  5.   Sixteenth burst   - fast single strokes between hats and snare, to
       trigger left/right stick alternation (a single-stroke roll).
  6.   Ride section      - ride pattern with the bell on downbeats, the
       hi-hat worked by the foot (pedal chick) on 2 & 4, plus an open hat.
  7.   Cymbal tags       - crash, second crash, splash, china and a
       side-stick, guaranteeing every remaining voice fires.
  8.   Ending            - a crash + kick accent left to ring.

Standard GM percussion notes used: 36 kick, 38 snare, 37 side-stick,
42 closed hat, 44 pedal hat, 46 open hat, 50/45/41 toms, 49/57 crashes,
55 splash, 52 china, 51 ride, 53 ride bell.
"""

import os
import sys

try:
    from piano.make_demo_piano_midi import write_midi
except ImportError:  # running as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.make_demo_piano_midi import write_midi

DRUM_CHANNEL = 9   # MIDI "channel 10" - General MIDI percussion

# GM percussion note numbers.
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


def build_demo_notes(tpb):
    """Return (notes, total_ticks). notes: (start_tick, dur_tick, note, vel)."""
    beat = tpb
    eighth = tpb // 2
    sixteenth = tpb // 4
    bar = 4 * beat
    hit = tpb // 8            # short, fixed duration - drum hits are transient
    notes = []

    def add(tick, note, vel, dur=None):
        notes.append((tick, dur or hit, note, vel))

    lead_in = beat
    bar_start = lead_in

    # --- Bars 1-2: basic rock beat -----------------------------------------
    for _ in range(2):
        add(bar_start + 0 * beat, KICK, 100)
        add(bar_start + 2 * beat, KICK, 100)
        add(bar_start + 1 * beat, SNARE, 105)
        add(bar_start + 3 * beat, SNARE, 105)
        for e in range(8):
            vel = 88 if e % 2 == 0 else 66   # accent the downbeats
            add(bar_start + e * eighth, HAT_CLOSED, vel)
        bar_start += bar

    # --- Bar 3: dynamics - hi-hat crescendo + accented backbeat ------------
    add(bar_start + 0 * beat, KICK, 105)
    add(bar_start + 2 * beat, KICK, 105)
    add(bar_start + 1 * beat, SNARE, 90)
    add(bar_start + 3 * beat, SNARE, 124)    # big accent to show a tall wind-up
    for e in range(8):
        add(bar_start + e * eighth, HAT_CLOSED, 40 + e * 11)  # 40 -> 117
    bar_start += bar

    # --- Bar 4: descending tom fill ----------------------------------------
    add(bar_start + 0 * beat, KICK, 100)
    add(bar_start + 0 * beat, HAT_CLOSED, 85)
    add(bar_start + 1 * beat, SNARE, 100)
    add(bar_start + 1 * beat, HAT_CLOSED, 70)
    fill = [SNARE, SNARE, TOM_HI, TOM_HI, TOM_MID, TOM_MID, TOM_FLOOR, TOM_FLOOR]
    for i, note in enumerate(fill):
        add(bar_start + 2 * beat + i * sixteenth, note, 96 + (i % 2) * 12)
    bar_start += bar

    # --- Bar 5: sixteenth single-stroke burst (hats + snare) ---------------
    for i in range(16):
        note = SNARE if i % 4 == 2 else HAT_CLOSED
        vel = 110 if note == SNARE else (80 if i % 2 == 0 else 60)
        add(bar_start + i * sixteenth, note, vel)
    add(bar_start + 0 * beat, KICK, 100)
    add(bar_start + 2 * beat, KICK, 100)
    bar_start += bar

    # --- Bar 6: ride pattern, hat foot on 2 & 4, an open hat ---------------
    add(bar_start + 0 * beat, KICK, 100)
    add(bar_start + 2 * beat, KICK, 100)
    add(bar_start + 1 * beat, SNARE, 105)
    add(bar_start + 3 * beat, SNARE, 105)
    for e in range(8):
        note = RIDE_BELL if e % 4 == 0 else RIDE   # bell on the beat
        add(bar_start + e * eighth, note, 80 if e % 2 == 0 else 62)
    add(bar_start + 1 * beat, HAT_PEDAL, 70)       # left-foot chick on 2 & 4
    add(bar_start + 3 * beat, HAT_PEDAL, 70)
    add(bar_start + 3 * beat + eighth, HAT_OPEN, 96)  # open hat on the last "and"
    bar_start += bar

    # --- Bar 7: cymbal + side-stick tags -----------------------------------
    add(bar_start + 0 * beat, CRASH, 112)
    add(bar_start + 0 * beat, KICK, 100)
    add(bar_start + 1 * beat, SIDE_STICK, 80)
    add(bar_start + 2 * beat, SPLASH, 100)
    add(bar_start + 2 * beat, KICK, 100)
    add(bar_start + 3 * beat, CHINA, 106)
    add(bar_start + 3 * beat + eighth, CRASH2, 108)
    bar_start += bar

    # --- Bar 8: ending crash + kick, left to ring --------------------------
    add(bar_start + 0 * beat, CRASH, 120, dur=beat * 2)
    add(bar_start + 0 * beat, KICK, 118)
    total_ticks = bar_start + bar

    notes.sort(key=lambda n: n[0])
    return notes, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "drum_demo.mid"
    TPB = 480
    TEMPO_US = 500000  # 120 BPM

    demo_notes, total_ticks = build_demo_notes(TPB)
    write_midi(out_path, TPB, TEMPO_US, demo_notes, channel=DRUM_CHANNEL)

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} hits, ~{seconds:.2f}s at 120 BPM")
