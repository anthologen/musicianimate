"""Automatic guitar tablature and left-hand fingering from a MIDI file.

Given any Standard MIDI File, decides which string and fret sounds each
note and which left-hand finger (1 = index .. 4 = pinky, 0 = open string)
frets it, producing a timeline that a hand rig can be animated from.

The approach is the classic ergonomic dynamic-programming method for
string instruments (Sayegh 1989's optimum-path paradigm; Radisavljevic &
Driessen 2004; Hori 2021) - see guitar/RESEARCH.md for the survey:

  1. Notes are clustered into onset events (chords), reusing the piano
     engine's MIDI ingestion.
  2. Every feasible (string, fret, finger) assignment of an event's
     sounding notes becomes a search state: distinct strings, at most
     MAX_FRETTED fretted notes, fretted extent within physical reach,
     and fingers ordered index-to-pinky up the frets. Index-finger barre
     grips are also generated: finger 1 may flatten across every fretted
     note at the grip's lowest fret (freeing fingers 2-4 for up to three
     higher frets), provided no open string rings on or above the
     bass-most barred string.
  3. A beam-searched Viterbi pass picks the cheapest state sequence.
     Static costs favour open strings, low positions, and compact
     grips; transition costs penalize hand-position shifts (the
     dominant term), same-finger hops, and string changes.

Hand position follows Hori's form model: with the index finger at fret
P, finger f naturally covers fret P + f - 1, so a grip's position is
mean(fret - (finger - 1)) over its fretted notes.

Fingertip target coordinates come from guitar.fret_layout, so they line
up exactly with the meshes built by build_guitar.py.

Usage::

    python -m guitar.fingering demo.mid -o fingering.json
    python -m guitar.fingering --selftest
"""

import argparse
import itertools
import json
import os
import sys

try:
    from . import fret_layout
    from piano.fingering import load_notes, group_events, _time_factor
    from piano.piano_midi_animator import parse_midi
except ImportError:  # running as a loose script, not a package
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    import fret_layout
    from piano.fingering import load_notes, group_events, _time_factor
    from piano.piano_midi_animator import parse_midi


# ---------------------------------------------------------------------------
# Ergonomic model parameters
# ---------------------------------------------------------------------------

HAND_SIZE_FACTOR = {
    "XXS": 0.80, "XS": 0.85, "S": 0.90, "M": 1.00,
    "L": 1.05, "XL": 1.10, "XXL": 1.15,
}

MAX_FRETTED = 4       # non-barre grips: one finger per fretted note
REACH_COMF_M = 0.065  # comfortable index..pinky extent along the neck (size M)
REACH_MAX_M = 0.095   # practical limit; both scale with the hand-size factor
START_POSITION = 2.0  # the hand starts hovering near open position
BEAM_WIDTH = 200

WEIGHTS = {
    # static (per event state)
    "fretted_note": 0.10,        # per fretted note; opens are intrinsically cheaper
    "high_position": 0.07,       # per fret of the grip's mean fretted fret
    "span_comf": 30.0,           # per metre of extent beyond the comfortable reach
    "finger_fret_mismatch": 0.75,  # per |dfinger - dfret| between adjacent fretted notes
    "weak_ring": 0.15,           # per use of finger 3
    "weak_pinky": 0.30,          # per use of finger 4
    "index_not_lowest": 0.20,    # chord whose lowest fret isn't finger 1 or 2
    "barre": 0.85,               # flattening the index across strings at all
    "barre_string": 0.12,        # per string the barre spans
    # transition (scaled by _time_factor(dt))
    "shift": 1.0,                # per fret of hand-position movement - the dominant term
    "same_finger": 1.0,          # same finger re-pressed on a different fret
    "finger_string_hop": 0.15,   # per string a re-used finger crosses
    "string_change": 0.10,       # per string step between consecutive melody notes
    "substitution": 6.0,         # changing the fretting finger on a held note
}


# ---------------------------------------------------------------------------
# Scope enforcement: range folding and chord simplification
# ---------------------------------------------------------------------------

def sanitize_notes(notes, warnings):
    """Fold out-of-range pitches into the guitar's range, in place."""
    for n in notes:
        midi = n["midi"]
        if not fret_layout.MIN_MIDI <= midi <= fret_layout.MAX_MIDI:
            folded = midi
            while folded < fret_layout.MIN_MIDI:
                folded += 12
            while folded > fret_layout.MAX_MIDI:
                folded -= 12
            warnings.append(f"note {midi} at t={n['start']:.2f}s is out of "
                            f"range, folded to {folded}")
            n["midi"] = folded
            n["octave_shifted"] = True
    return notes


def simplify_chords(events, warnings):
    """Enforce playability scope on each onset event, in place.

    Merges unison duplicates and trims chords with more notes than
    strings down to the bass note plus the top three. Trimmed notes are
    tagged note["dropped"] and removed from the event. (Chords that turn
    out unplayable even with a barre are thinned later, at assign time.)
    """
    for ev in events:
        seen = {}
        for n in ev["notes"]:
            if n["midi"] in seen:
                keep = seen[n["midi"]]
                keep["end"] = max(keep["end"], n["end"])
                keep["velocity"] = max(keep["velocity"], n["velocity"])
                n["dropped"] = "unison"
            else:
                seen[n["midi"]] = n
        kept = [n for n in ev["notes"] if not n.get("dropped")]
        if len(kept) > fret_layout.NUM_STRINGS:
            keep_ids = {id(n) for n in [kept[0]] + kept[-3:]}
            for n in kept:
                if id(n) not in keep_ids:
                    n["dropped"] = "chord_simplified"
                    warnings.append(f"chord at t={ev['t']:.2f}s too dense, "
                                    f"dropped note {n['midi']}")
        ev["notes"] = [n for n in ev["notes"] if not n.get("dropped")]


# ---------------------------------------------------------------------------
# The beam-searched Viterbi pass
# ---------------------------------------------------------------------------

def _event_states(sounding, factor):
    """All feasible (string, fret, finger) grips for the sounding notes,
    as tuples aligned with `sounding`.

    Feasibility: distinct strings, fretted extent within the scaled
    physical reach, and either the one-finger-per-note model (at most
    MAX_FRETTED notes, strictly increasing fingers up the frets, fret
    ties giving the bass-side note the lower finger) or an index barre
    (see _barre_states).
    """
    reach = REACH_MAX_M * factor
    cand = [fret_layout.candidate_positions(n["midi"]) for n in sounding]
    states = []
    for combo in itertools.product(*cand):
        strings = [s for s, _ in combo]
        if len(set(strings)) != len(strings):
            continue
        fretted = [(f, s, idx) for idx, (s, f) in enumerate(combo) if f > 0]
        if not fretted:
            states.append(tuple((s, f, 0) for s, f in combo))
            continue
        frets = [f for f, _, _ in fretted]
        if fret_layout.press_y(min(frets)) - fret_layout.press_y(max(frets)) > reach:
            continue
        fretted.sort()
        if len(fretted) <= MAX_FRETTED:
            for fingers in itertools.combinations((1, 2, 3, 4), len(fretted)):
                asg = [0] * len(combo)
                for (f, s, idx), fg in zip(fretted, fingers):
                    asg[idx] = fg
                states.append(tuple((s, f, fg)
                                    for (s, f), fg in zip(combo, asg)))
        states.extend(_barre_states(combo, fretted))
    return states


def _barre_states(combo, fretted):
    """Index-barre grips for one position combo: finger 1 flattens across
    every fretted note at the lowest fret, fingers 2-4 take up to three
    higher-fret notes (increasing up the frets). Physically the index
    lies across the neck from its bass-most string to the treble edge, so
    no open string may sound on or above that string.

    `fretted` is (fret, string, combo-index) triples sorted ascending.
    """
    f_barre = fretted[0][0]
    barred = [x for x in fretted if x[0] == f_barre]
    if len(barred) < 2:
        return []
    rest = fretted[len(barred):]
    if len(rest) > 3:
        return []
    s_lowest = min(s for _, s, _ in barred)
    for s, f in combo:
        if s >= s_lowest and f < f_barre:  # open or lower fret under the bar
            return []
    states = []
    for fingers in itertools.combinations((2, 3, 4), len(rest)):
        asg = [0] * len(combo)
        for _, _, idx in barred:
            asg[idx] = 1
        for (f, s, idx), fg in zip(rest, fingers):
            asg[idx] = fg
        states.append(tuple((s, f, fg) for (s, f), fg in zip(combo, asg)))
    return states


def _state_cost(state, factor):
    """Static (within-event) ergonomic cost of one grip."""
    fretted = sorted((f, s, fg) for s, f, fg in state if f > 0)
    cost = WEIGHTS["fretted_note"] * len(fretted)
    if not fretted:
        return cost
    cost += WEIGHTS["high_position"] * (
        sum(f for f, _, _ in fretted) / len(fretted))
    extent = (fret_layout.press_y(fretted[0][0])
              - fret_layout.press_y(fretted[-1][0]))
    over = extent - REACH_COMF_M * factor
    if over > 0:
        cost += WEIGHTS["span_comf"] * over

    # A barre collapses to a single finger-1 entry for the weakness and
    # lowest-fret checks, and is left out of the finger-ordering check
    # entirely: a flattened bar isn't a curled finger, and both the
    # E-shape (bar + next fret) and A-shape (bar + two frets up) are
    # standard forms.
    barred = [x for x in fretted if x[2] == 1]
    if len(barred) >= 2:
        strings = [s for _, s, _ in barred]
        cost += (WEIGHTS["barre"]
                 + WEIGHTS["barre_string"] * (max(strings) - min(strings) + 1))
        curled = sorted(x for x in fretted if x[2] != 1)
        entries = sorted([barred[0]] + curled)
    else:
        curled = fretted
        entries = fretted
    # Fingers should climb with the frets; stacking adjacent fingers on
    # one fret (Am, E, A-shape grips) is natural, so a fret tie expects a
    # finger step of 1, not 0.
    for (f1, _, g1), (f2, _, g2) in zip(curled, curled[1:]):
        cost += (WEIGHTS["finger_fret_mismatch"]
                 * abs((g2 - g1) - max(f2 - f1, 1)))
    for _, _, g in entries:
        if g == 3:
            cost += WEIGHTS["weak_ring"]
        elif g == 4:
            cost += WEIGHTS["weak_pinky"]
    if len(entries) >= 2 and entries[0][2] not in (1, 2):
        cost += WEIGHTS["index_not_lowest"]
    return cost


def _hand_pos(state, prev_pos):
    """Index-finger fret implied by a grip (Hori's form model); all-open
    grips keep the previous position, making opens free pivots."""
    fretted = [(f, fg) for _, f, fg in state if f > 0]
    if not fretted:
        return prev_pos
    return sum(f - (fg - 1) for f, fg in fretted) / len(fretted)


def _transition_cost(prev_sounding, prev_state, sounding, state, new_ids,
                     ppos, dt):
    """Cost of moving between grips; None if the transition is impossible
    (a ringing note cannot change string or fret without re-striking)."""
    tf = _time_factor(dt)
    prev_by_id = {id(n): a for n, a in zip(prev_sounding, prev_state)}

    cost = 0.0
    cur_fingers = {}  # finger -> bass-most (string, fret) among new notes
    for note, (s, f, fg) in zip(sounding, state):
        prev = prev_by_id.get(id(note))
        if prev is not None and id(note) not in new_ids:
            if (prev[0], prev[1]) != (s, f):
                return None
            if f > 0 and prev[2] != fg:
                cost += WEIGHTS["substitution"]
        elif f > 0:
            if fg not in cur_fingers or s < cur_fingers[fg][0]:
                cur_fingers[fg] = (s, f)

    pos = _hand_pos(state, ppos)
    cost += WEIGHTS["shift"] * abs(pos - ppos) * tf

    # Same finger re-pressed on a new note: fret hops hurt, string hops
    # cost a little. Barres collapse to their bass-most press so a barred
    # transition is charged once, not per string.
    prev_fingers = {}
    for s, f, fg in prev_state:
        if f > 0 and (fg not in prev_fingers or s < prev_fingers[fg][0]):
            prev_fingers[fg] = (s, f)
    for fg, (ns, nf) in cur_fingers.items():
        if fg in prev_fingers:
            s, f = prev_fingers[fg]
            if (ns, nf) != (s, f):
                if nf != f:
                    cost += WEIGHTS["same_finger"] * tf
                cost += WEIGHTS["finger_string_hop"] * abs(ns - s)

    # Light legato preference for staying on a string in melody lines.
    if len(prev_sounding) == 1 and len(sounding) == 1:
        cost += WEIGHTS["string_change"] * abs(state[0][0] - prev_state[0][0])
    return cost


def assign_positions(events, hand_size="M", warnings=None):
    """Beam-searched Viterbi over per-event grips; writes note["string"],
    note["fret"], and note["finger"] at each note's onset.

    Held notes keep their string and fret across events; when an event
    admits no grip at all, holds are released oldest-first, then the
    chord itself is thinned from the middle out (tagging drops).
    """
    if warnings is None:
        warnings = []
    if not events:
        return
    factor = HAND_SIZE_FACTOR[hand_size]

    # The Markov state is (grip, hand position): a fretted grip implies its
    # position, but an all-open grip inherits it from the predecessor, so
    # the position must be part of the beam key or cheap open events would
    # collapse every candidate position into one.
    beams = []      # per event: {(state, pos_key): (cost, back_key, hand_pos)}
    soundings = []  # per event: note list each state tuple aligns with
    for i, ev in enumerate(events):
        new_notes = list(ev["notes"])
        new_ids = {id(n) for n in new_notes}
        held = []
        if i > 0:
            held = [n for n in soundings[-1]
                    if n["end"] > ev["t"] + 1e-6 and id(n) not in new_ids]
            held.sort(key=lambda n: n["start"])

        while True:
            sounding = held + new_notes
            dt = max(ev["t"] - events[i - 1]["t"], 1e-3) if i else 0.0
            states = {}
            for st in _event_states(sounding, factor):
                vcost = _state_cost(st, factor)
                if i == 0:
                    pos = _hand_pos(st, START_POSITION)
                    key = (st, round(pos, 3))
                    if key not in states or vcost < states[key][0]:
                        states[key] = (vcost, None, pos)
                    continue
                for pkey, (pcost, _, ppos) in beams[-1].items():
                    tcost = _transition_cost(soundings[-1], pkey[0], sounding,
                                             st, new_ids, ppos, dt)
                    if tcost is None:
                        continue
                    pos = _hand_pos(st, ppos)
                    key = (st, round(pos, 3))
                    cost = pcost + vcost + tcost
                    if key not in states or cost < states[key][0]:
                        states[key] = (cost, pkey, pos)
            if states:
                break
            if held:
                held.pop(0)  # release the oldest hold and retry
            elif len(new_notes) > 1:
                victim = sorted(new_notes, key=lambda n: n["midi"])[1]
                new_notes.remove(victim)
                victim["dropped"] = "unplayable"
                warnings.append(f"dropped unplayable note {victim['midi']} "
                                f"at t={victim['start']:.2f}s")
            else:
                # A lone in-range note always has a grip; this only fires
                # if the event emptied out entirely.
                break

        if not states:
            if beams:  # pass an empty event through the beam unchanged
                pkey = min(beams[-1], key=lambda k: beams[-1][k][0])
                pcost, _, ppos = beams[-1][pkey]
                states = {((), pkey[1]): (pcost, pkey, ppos)}
                sounding = []
            else:
                states = {((), START_POSITION): (0.0, None, START_POSITION)}
                sounding = []
        if len(states) > BEAM_WIDTH:
            keep = sorted(states, key=lambda k: states[k][0])[:BEAM_WIDTH]
            states = {k: states[k] for k in keep}
        beams.append(states)
        soundings.append(sounding)

    key = min(beams[-1], key=lambda k: beams[-1][k][0])
    for i in range(len(events) - 1, -1, -1):
        onset_ids = {id(n) for n in events[i]["notes"]}
        n_index = sum(1 for _, f, fg in key[0] if f > 0 and fg == 1)
        for note, (s, f, fg) in zip(soundings[i], key[0]):
            if id(note) in onset_ids:
                note["string"], note["fret"], note["finger"] = s, f, fg
                note["barre"] = f > 0 and fg == 1 and n_index >= 2
        key = beams[i][key][1]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def tab_notes(notes, hand_size="M"):
    """Sanitize, simplify, and assign a loaded note list.

    Returns (kept_notes, warnings): the notes that received a
    string/fret/finger, plus human-readable warnings for anything that
    had to be folded or dropped along the way.
    """
    warnings = []
    if not notes:
        return notes, warnings
    sanitize_notes(notes, warnings)
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    events = group_events(notes)
    simplify_chords(events, warnings)
    assign_positions(events, hand_size, warnings)
    kept = [n for n in notes if not n.get("dropped") and "string" in n]
    return kept, warnings


def _smoothed_samples(samples, rate, smooth_window):
    half = max(1, int(smooth_window * rate / 2))
    out = []
    for i in range(len(samples)):
        window = samples[max(0, i - half):i + half + 1]
        out.append(sum(window) / len(window))
    return out


def hand_position_curve(notes, t_end, rate=30.0, smooth_window=0.3):
    """Sampled left-hand position over time, box-smoothed like piano's
    wrist_curve: the index-finger fret plus the matching neck y."""
    samples = []
    pos = START_POSITION
    n_samples = int(t_end * rate) + 1
    for i in range(n_samples):
        t = i / rate
        active = [n for n in notes
                  if n["start"] <= t < n["end"] and n["fret"] > 0]
        if active:
            pos = sum(n["fret"] - (n["finger"] - 1) for n in active) / len(active)
        samples.append(pos)
    out = []
    for i, p in enumerate(_smoothed_samples(samples, rate, smooth_window)):
        y = fret_layout.press_y(max(1.0, min(float(fret_layout.NUM_FRETS), p)))
        out.append({"t": round(i / rate, 4),
                    "fret": round(p, 3), "y": round(y, 5)})
    return out


def pick_curve(notes, t_end, rate=30.0, smooth_window=0.15):
    """Sampled pick x: the most recently struck string's pluck point,
    lightly smoothed. y and z are the constants PLUCK_Y / STRING_Z."""
    starts = sorted(notes, key=lambda n: n["start"])
    if not starts:
        return []
    samples = []
    cur = fret_layout.pluck_point(starts[0]["string"])[0]
    j = 0
    n_samples = int(t_end * rate) + 1
    for i in range(n_samples):
        t = i / rate
        while j < len(starts) and starts[j]["start"] <= t:
            cur = fret_layout.pluck_point(starts[j]["string"])[0]
            j += 1
        samples.append(cur)
    return [{"t": round(i / rate, 4), "x": round(v, 5)}
            for i, v in enumerate(_smoothed_samples(samples, rate, smooth_window))]


def _beat_mapper(midi_path):
    """Return beats_at(seconds) -> fractional beats from the song start,
    inverting the MIDI tempo map. The right-hand animator uses beat
    positions to decide down- vs up-strokes (down on the beat, up on the
    off-beat). Constant tempo collapses to beats = seconds / sec_per_beat.
    """
    ticks_per_beat, tempo_map, _ = parse_midi(midi_path)
    # (seconds, beats) breakpoint at each tempo segment start.
    pts = []
    cum_sec, prev_tick, prev_tempo = 0.0, 0, tempo_map[0][1]
    for tick, tempo in tempo_map:
        cum_sec += (tick - prev_tick) * (prev_tempo / 1e6) / ticks_per_beat
        pts.append((cum_sec, tick / ticks_per_beat))
        prev_tick, prev_tempo = tick, tempo
    last_spb = prev_tempo / 1e6  # seconds per beat past the final breakpoint

    def beats_at(seconds):
        prev_s, prev_b = pts[0]
        for s, b in pts[1:]:
            if seconds < s:
                span = s - prev_s
                frac = (seconds - prev_s) / span if span > 1e-9 else 0.0
                return prev_b + frac * (b - prev_b)
            prev_s, prev_b = s, b
        return prev_b + (seconds - prev_s) / last_spb

    return beats_at


def compute_fingering(midi_path, hand_size="M"):
    """Full pipeline: MIDI file -> tablature/fingering timeline dict."""
    raw = load_notes(midi_path)
    total = len(raw)
    notes, warnings = tab_notes(raw, hand_size)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    beats_at = _beat_mapper(midi_path)

    t_end = max((n["end"] for n in notes), default=0.0)
    out_notes = []
    for n in notes:
        x, y, z = fret_layout.fingertip(n["string"], n["fret"])
        px, py, _ = fret_layout.pluck_point(n["string"])
        out_notes.append({
            "start": round(n["start"], 5),
            "end": round(n["end"], 5),
            "beat": round(beats_at(n["start"]), 4),
            "midi": n["midi"],
            "velocity": n["velocity"],
            "string": n["string"],
            "fret": n["fret"],
            "finger": n["finger"],
            "is_open": n["fret"] == 0,
            "barre": n.get("barre", False),
            "x": round(x, 5),
            "y": round(y, 5),
            "z": round(z, 5),
            "pluck_x": round(px, 5),
            "pluck_y": round(py, 5),
        })

    return {
        "source": os.path.basename(midi_path),
        "instrument": "guitar",
        "tuning": list(fret_layout.TUNING),
        "hand_size": hand_size,
        "note_count": len(out_notes),
        "dropped_count": total - len(out_notes),
        "notes": out_notes,
        "hands": {
            "fret": {"position": hand_position_curve(notes, t_end)},
            "pick": {"x": pick_curve(notes, t_end)},
        },
    }


# ---------------------------------------------------------------------------
# Self-test: golden tablature patterns
# ---------------------------------------------------------------------------

def _mknotes(seq, dur=0.22, step=0.25):
    """Build a note list from (start_beat, midi) or (start_beat, midi, dur)."""
    notes = []
    for item in seq:
        beat, midi = item[0], item[1]
        d = item[2] if len(item) > 2 else dur
        notes.append({"start": beat * step, "end": beat * step + d,
                      "midi": midi, "velocity": 80, "track": 0})
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    return notes


def selftest():
    failures = []

    def check(name, cond, detail=""):
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            failures.append(name)

    def run(seq, dur=0.22, step=0.25):
        kept, warnings = tab_notes(_mknotes(seq, dur, step))
        kept.sort(key=lambda n: (n["start"], n["midi"]))
        return kept, warnings

    def pitches_ok(kept):
        return all(fret_layout.midi_of(n["string"], n["fret"]) == n["midi"]
                   for n in kept)

    print("C major scale, open position:")
    kept, _ = run(list(enumerate([48, 50, 52, 53, 55, 57, 59, 60])))
    sf = [(n["string"], n["fret"]) for n in kept]
    check("pitches preserved", pitches_ok(kept))
    check("open-position tab",
          sf == [(1, 3), (2, 0), (2, 2), (2, 3), (3, 0), (3, 2), (4, 0), (4, 1)],
          str(sf))
    check("finger = fret in first position",
          all(n["finger"] == n["fret"] for n in kept),
          str([n["finger"] for n in kept]))

    print("C major triad, held:")
    kept, _ = run([(0, 48, 1.0), (0, 52, 1.0), (0, 55, 1.0)])
    by = {n["midi"]: (n["string"], n["fret"], n["finger"]) for n in kept}
    check("open-C grip fragment",
          {m: v[:2] for m, v in by.items()} == {48: (1, 3), 52: (2, 2), 55: (3, 0)},
          str(by))
    check("finger order matches frets",
          by[55][2] == 0 and 0 < by[52][2] < by[48][2], str(by))
    check("no barre in an open grip", not any(n.get("barre") for n in kept))

    print("F major barre chord:")
    kept, _ = run([(0, m, 1.5) for m in (41, 48, 53, 57, 60, 65)])
    by = {n["midi"]: (n["string"], n["fret"], n["finger"]) for n in kept}
    check("classic E-shape grip at fret 1",
          by == {41: (0, 1, 1), 48: (1, 3, 3), 53: (2, 3, 4),
                 57: (3, 2, 2), 60: (4, 1, 1), 65: (5, 1, 1)},
          str(by))
    check("barre flags on the fret-1 notes",
          {n["midi"] for n in kept if n.get("barre")} == {41, 60, 65},
          str({n["midi"]: n.get("barre") for n in kept}))

    print("B minor barre chord:")
    kept, _ = run([(0, m, 1.5) for m in (47, 54, 59, 62, 66)])
    by = {n["midi"]: (n["string"], n["fret"], n["finger"]) for n in kept}
    check("A-shape grip at fret 2",
          by == {47: (1, 2, 1), 54: (2, 4, 3), 59: (3, 4, 4),
                 62: (4, 3, 2), 66: (5, 2, 1)},
          str(by))
    check("barre spans outer strings",
          {n["midi"] for n in kept if n.get("barre")} == {47, 66},
          str({n["midi"]: n.get("barre") for n in kept}))

    print("Chromatic run G3..E4 in eighths:")
    kept, _ = run([(i, 55 + i) for i in range(10)], dur=0.11, step=0.125)
    check("pitches preserved", pitches_ok(kept))
    check("no same-finger fret hops", all(
        not (a["finger"] and a["finger"] == b["finger"]
             and (a["string"], a["fret"]) != (b["string"], b["fret"]))
        for a, b in zip(kept, kept[1:])),
        str([(n["string"], n["fret"], n["finger"]) for n in kept]))
    poss = [n["fret"] - (n["finger"] - 1) for n in kept if n["fret"] > 0]
    check("smooth position shifts (<= 2 frets)",
          all(abs(b - a) <= 2 for a, b in zip(poss, poss[1:])), str(poss))

    print("High phrase E4..C5:")
    kept, _ = run(list(enumerate([64, 67, 69, 71, 72])))
    check("pitches preserved", pitches_ok(kept))
    check("leaves open position", max(n["fret"] for n in kept) >= 5,
          str([(n["string"], n["fret"]) for n in kept]))

    print("Repeated middle C:")
    kept, _ = run([(i, 60) for i in range(6)])
    sf = {(n["string"], n["fret"]) for n in kept}
    check("stable position for a repeated pitch", len(sf) == 1, str(sf))

    print("Out-of-range bass note:")
    kept, warnings = run([(0, 30)])
    check("folded into range and tabbed",
          len(kept) == 1 and kept[0]["midi"] == 42
          and (kept[0]["string"], kept[0]["fret"]) == (0, 2),
          str([(n["midi"], n["string"], n["fret"]) for n in kept]))
    check("fold warning emitted", any("folded" in w for w in warnings))

    print("Five-note cluster:")
    kept, warnings = run([(0, m, 1.0) for m in (48, 50, 52, 53, 55)])
    strings = [n["string"] for n in kept]
    check("simplified to a playable grip", 0 < len(kept) < 5,
          f"kept {len(kept)}")
    check("distinct strings", len(set(strings)) == len(strings), str(strings))
    check("pitches preserved", pitches_ok(kept))
    check("drop warning emitted",
          any("unplayable" in w or "dense" in w for w in warnings))

    print()
    if failures:
        raise SystemExit(f"selftest: {len(failures)} check(s) failed: {failures}")
    print("selftest: all checks passed")


# ---------------------------------------------------------------------------
# Benchmark: well-known chord grips
# ---------------------------------------------------------------------------

# (name, midi pitches, {midi: (string, fret, finger)}). finger None means
# the canonical fingering is ambiguous (players use several); only the
# tab is checked for those notes.
BENCHMARK_CHORDS = [
    ("C major  x32010", (48, 52, 55, 60, 64),
     {48: (1, 3, 3), 52: (2, 2, 2), 55: (3, 0, 0),
      60: (4, 1, 1), 64: (5, 0, 0)}),
    ("A minor  x02210", (45, 52, 57, 60, 64),
     {45: (1, 0, 0), 52: (2, 2, 2), 57: (3, 2, 3),
      60: (4, 1, 1), 64: (5, 0, 0)}),
    ("E major  022100", (40, 47, 52, 56, 59, 64),
     {40: (0, 0, 0), 47: (1, 2, 2), 52: (2, 2, 3),
      56: (3, 1, 1), 59: (4, 0, 0), 64: (5, 0, 0)}),
    ("E minor  022000", (40, 47, 52, 55, 59, 64),
     {40: (0, 0, 0), 47: (1, 2, None), 52: (2, 2, None),
      55: (3, 0, 0), 59: (4, 0, 0), 64: (5, 0, 0)}),
    ("D major  xx0232", (50, 57, 62, 66),
     {50: (2, 0, 0), 57: (3, 2, 1), 62: (4, 3, 3), 66: (5, 2, 2)}),
    ("D minor  xx0231", (50, 57, 62, 65),
     {50: (2, 0, 0), 57: (3, 2, 2), 62: (4, 3, 3), 65: (5, 1, 1)}),
    ("G major  320003", (43, 47, 50, 55, 59, 67),
     {43: (0, 3, 2), 47: (1, 2, 1), 50: (2, 0, 0),
      55: (3, 0, 0), 59: (4, 0, 0), 67: (5, 3, 3)}),
    ("A major  x02220", (45, 52, 57, 61, 64),
     {45: (1, 0, 0), 52: (2, 2, 1), 57: (3, 2, 2),
      61: (4, 2, 3), 64: (5, 0, 0)}),
    ("F major  133211 barre", (41, 48, 53, 57, 60, 65),
     {41: (0, 1, 1), 48: (1, 3, 3), 53: (2, 3, 4),
      57: (3, 2, 2), 60: (4, 1, 1), 65: (5, 1, 1)}),
    ("B minor  x24432 barre", (47, 54, 59, 62, 66),
     {47: (1, 2, 1), 54: (2, 4, 3), 59: (3, 4, 4),
      62: (4, 3, 2), 66: (5, 2, 1)}),
    ("B major  x24442 barre", (47, 54, 59, 63, 66),
     {47: (1, 2, 1), 54: (2, 4, 2), 59: (3, 4, 3),
      63: (4, 4, 4), 66: (5, 2, 1)}),
    ("G major  355433 barre", (43, 50, 55, 59, 62, 67),
     {43: (0, 3, 1), 50: (1, 5, 3), 55: (2, 5, 4),
      59: (3, 4, 2), 62: (4, 3, 1), 67: (5, 3, 1)}),
]


def _tab_string(got):
    """Render {string: fret} as the usual low-E-first tab word, e.g.
    x32010."""
    frets = {s: f for s, f, _ in got.values()}
    return "".join(str(frets[s]) if s in frets else "x" for s in range(6))


def benchmark():
    """Check the engine against well-known chord grips; exit non-zero if
    any tab (string/fret choice) or unambiguous fingering is wrong."""
    failures = []
    for name, midis, expected in BENCHMARK_CHORDS:
        kept, _ = tab_notes(_mknotes([(0, m, 1.5) for m in midis]))
        got = {n["midi"]: (n["string"], n["fret"], n["finger"])
               for n in kept}
        tab_ok = ({m: v[:2] for m, v in got.items()} ==
                  {m: v[:2] for m, v in expected.items()})
        fing_ok = all(v[2] is None or (m in got and got[m][2] == v[2])
                      for m, v in expected.items())
        fingers = "".join(str(got[m][2]) for m in sorted(
            got, key=lambda m: got[m][0])) if tab_ok else "-"
        status = "ok" if tab_ok and fing_ok else "FAIL"
        print(f"  [{status}] {name:<24} got {_tab_string(got):<7} "
              f"fingers {fingers}")
        if status == "FAIL":
            failures.append(name)
            print(f"         expected {expected}")
            print(f"         got      {got}")
    print()
    if failures:
        raise SystemExit(
            f"benchmark: {len(failures)}/{len(BENCHMARK_CHORDS)} chords "
            f"failed: {failures}")
    print(f"benchmark: all {len(BENCHMARK_CHORDS)} chords match their "
          f"canonical grips")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("midi_file", nargs="?", help="MIDI file path")
    parser.add_argument("-o", "--out", default="fingering.json",
                        help="Output JSON path (default: fingering.json)")
    parser.add_argument("--hand-size", default="M",
                        choices=sorted(HAND_SIZE_FACTOR),
                        help="Hand size preset (default: M)")
    parser.add_argument("--selftest", action="store_true",
                        help="Run golden-pattern checks and exit")
    parser.add_argument("--benchmark", action="store_true",
                        help="Check well-known chord grips and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.benchmark:
        benchmark()
        return
    if not args.midi_file:
        parser.error("midi_file is required unless --selftest is given")

    result = compute_fingering(args.midi_file, args.hand_size)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    opens = sum(1 for n in result["notes"] if n["is_open"])
    max_fret = max((n["fret"] for n in result["notes"]), default=0)
    print(f"{result['source']}: {result['note_count']} notes tabbed "
          f"({opens} open, {result['note_count'] - opens} fretted, "
          f"max fret {max_fret}, {result['dropped_count']} dropped) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
