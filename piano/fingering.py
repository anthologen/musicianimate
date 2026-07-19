"""Automatic piano fingering estimation from a MIDI file.

Given any Standard MIDI File, decides which hand plays each note and which
finger (1 = thumb .. 5 = pinky) presses each key, producing a timeline that
a hand/finger rig can be animated from.

The approach is a dependency-free implementation of the classic ergonomic
dynamic-programming method:

  1. Notes are clustered into onset events (chords).
  2. Events are split between the hands with a Viterbi pass over per-event
     split points (all notes below a boundary go to the left hand), scoring
     hand span, movement speed, and hand crossing - a simplified take on
     Nakamura et al.'s merged-output HMM for both-hand fingering.
  3. Each hand is fingered independently with a second Viterbi pass whose
     states are complete finger-to-key assignments of the sounding chord
     (Al Kasimi et al. 2007) and whose costs follow Parncutt et al.'s 1997
     ergonomic rules: finger-pair span tables, weak-finger use, thumb on
     black keys, thumb passing, position changes, and same-finger repeats.

Fingertip target coordinates come from piano.key_layout, so they line up
exactly with the key objects built by build_piano.py.

Usage::

    python -m piano.fingering demo.mid -o fingering.json
    python -m piano.fingering --selftest
"""

import argparse
import itertools
import json
import os

try:
    from .piano_midi_animator import parse_midi, _tick_to_seconds
    from . import key_layout
except ImportError:  # running as a loose script, not a package
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from piano_midi_animator import parse_midi, _tick_to_seconds
    import key_layout


# ---------------------------------------------------------------------------
# Ergonomic model parameters
# ---------------------------------------------------------------------------

# Finger-pair span table in semitones, right-hand orientation (ascending
# pitch = ascending finger number). Values follow Parncutt et al. (1997):
# (min_prac, min_comf, min_rel, max_rel, max_comf, max_prac), where the
# "rel" range is relaxed, "comf" is comfortable, "prac" is the practical
# limit. Negative minima allow crossings in *sequential* motion (thumb
# under); simultaneous notes in a chord always need a span >= 1.
SPANS = {
    (1, 2): (-5, -3, 1, 5, 8, 10),
    (1, 3): (-4, -2, 3, 7, 10, 12),
    (1, 4): (-3, -1, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 2, 3, 5),
    (2, 4): (1, 1, 3, 4, 5, 7),
    (2, 5): (2, 2, 5, 6, 8, 10),
    (3, 4): (1, 1, 1, 2, 2, 4),
    (3, 5): (1, 1, 3, 4, 5, 7),
    (4, 5): (1, 1, 1, 2, 3, 5),
}

# Multiplier applied to the span table's upper limits, like pianoplayer's
# hand-size presets.
HAND_SIZE_FACTOR = {
    "XXS": 0.80, "XS": 0.85, "S": 0.90, "M": 1.00,
    "L": 1.05, "XL": 1.10, "XXL": 1.15,
}

# Typical resting offset of each fingertip from the thumb, in semitones,
# used to estimate the hand's position on the keyboard from an assignment.
FINGER_OFFSET = {1: 0.0, 2: 2.0, 3: 4.0, 4: 5.0, 5: 7.0}

WEIGHTS = {
    "stretch_comf": 2.0,     # per semitone outside the comfortable span
    "stretch_rel": 1.0,      # per semitone outside the relaxed span
    "stretch_prac": 10.0,    # per semitone outside the practical span
    "weak_4": 0.4,           # per use of the ring finger
    "weak_5": 0.2,           # per use of the pinky
    "thumb_black": 1.0,      # thumb on a black key
    "pinky_black": 0.4,      # pinky on a black key
    "move": 0.35,            # per semitone of hand repositioning
    "thumb_pass": 1.0,       # thumb crossing under / finger over thumb
    "thumb_pass_black": 1.0, # extra if the crossing lands on a black key
    "same_finger": 3.0,      # same finger restruck on a different key
    "substitution": 6.0,     # changing fingers on a held note
    # hand-splitting weights
    "split_span": 10.0,      # per semitone beyond a hand's max reach
    "split_wide": 0.5,       # per semitone of span beyond a comfortable 9
    "split_move": 0.4,       # per semitone of hand-centroid movement
    "split_cross": 8.0,      # per semitone of hands overlapping in pitch
    "split_order": 1.0,      # per semitone the LH centroid sits above RH
    "split_range": 0.05,     # weak pull of LH below / RH above middle C
    "split_activation": 2.5, # engaging a hand that was idle
}

MAX_REACH = 14        # max simultaneous span of one hand, semitones (size M)
ONSET_EPS = 0.03      # seconds; onsets closer than this form one chord
BEAM_WIDTH = 200
LH_START_POS, RH_START_POS = 48.0, 67.0  # C3 / G4


def _time_factor(dt):
    """Scale a transition cost by urgency: fast moves hurt, slow ones don't."""
    return min(2.5, max(0.5, 0.35 / max(dt, 0.05)))


def _span_cost(span, pair, factor):
    """Ergonomic cost of a sequential interval `span` between finger `pair`."""
    min_prac, min_comf, min_rel, max_rel, max_comf, max_prac = SPANS[pair]
    max_rel, max_comf, max_prac = (
        max_rel * factor, max_comf * factor, max_prac * factor)
    cost = 0.0
    if span > max_rel:
        cost += WEIGHTS["stretch_rel"] * (span - max_rel)
    elif span < min_rel:
        cost += WEIGHTS["stretch_rel"] * (min_rel - span)
    if span > max_comf:
        cost += WEIGHTS["stretch_comf"] * (span - max_comf)
    elif span < min_comf:
        cost += WEIGHTS["stretch_comf"] * (min_comf - span)
    if span > max_prac:
        cost += WEIGHTS["stretch_prac"] * (span - max_prac)
    elif span < min_prac:
        cost += WEIGHTS["stretch_prac"] * (min_prac - span)
    return cost


# ---------------------------------------------------------------------------
# MIDI ingestion
# ---------------------------------------------------------------------------

def load_notes(midi_path):
    """Parse a MIDI file into a note list sorted by onset.

    Each note is a dict {start, end, midi, velocity, track} with times in
    seconds. Note-ons and -offs pair FIFO per (track, note number).
    """
    ticks_per_beat, tempo_map, events = parse_midi(midi_path)
    order = {"off": 0, "on": 1}
    events.sort(key=lambda e: (e["tick"], order[e["type"]]))

    open_notes = {}
    notes = []
    for ev in events:
        key = (ev.get("track", 0), ev["note"])
        if ev["type"] == "on":
            open_notes.setdefault(key, []).append(ev)
        else:
            stack = open_notes.get(key)
            if stack:
                on = stack.pop(0)
                notes.append({
                    "start": _tick_to_seconds(on["tick"], ticks_per_beat, tempo_map),
                    "end": _tick_to_seconds(ev["tick"], ticks_per_beat, tempo_map),
                    "midi": ev["note"],
                    "velocity": on["velocity"],
                    "track": key[0],
                })
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    return notes


def group_events(notes, eps=ONSET_EPS):
    """Cluster notes whose onsets are within eps into chord events."""
    events = []
    for note in notes:
        if events and note["start"] - events[-1]["t"] <= eps:
            events[-1]["notes"].append(note)
        else:
            events.append({"t": note["start"], "notes": [note]})
    for ev in events:
        ev["notes"].sort(key=lambda n: n["midi"])
    return events


# ---------------------------------------------------------------------------
# Hand assignment
# ---------------------------------------------------------------------------

def _split_hand_cost(pitches, sustained, max_reach):
    """Span cost of one hand sounding `pitches` plus `sustained` holds."""
    all_p = pitches + sustained
    if len(all_p) < 2:
        return 0.0
    span = max(all_p) - min(all_p)
    cost = WEIGHTS["split_wide"] * max(0.0, span - 9)
    cost += WEIGHTS["split_span"] * max(0.0, span - max_reach)
    return cost


def assign_hands(events, max_reach=MAX_REACH):
    """Viterbi over per-event split points; writes note["hand"] in place.

    A state at each event is k = how many of its lowest notes go to the left
    hand. Alongside the path cost we carry each hand's last known centroid,
    so movement cost works even across events where a hand rests.
    """
    if not events:
        return
    # states: k -> (cost, lh_pos, rh_pos, back_k)
    states = {}
    first = events[0]["notes"]
    for k in range(len(first) + 1):
        lo, hi = [n["midi"] for n in first[:k]], [n["midi"] for n in first[k:]]
        # tf=0: placing the hands before the first note is free.
        cost = _event_split_cost(lo, hi, [], [], LH_START_POS, RH_START_POS, 0.0)
        lh = sum(lo) / len(lo) if lo else LH_START_POS
        rh = sum(hi) / len(hi) if hi else RH_START_POS
        states[k] = (cost, lh, rh, None)

    history = [states]
    for i in range(1, len(events)):
        prev_ev, ev = events[i - 1], events[i]
        dt = max(ev["t"] - prev_ev["t"], 1e-3)
        tf = _time_factor(dt)
        new_states = {}
        for k in range(len(ev["notes"]) + 1):
            lo = [n["midi"] for n in ev["notes"][:k]]
            hi = [n["midi"] for n in ev["notes"][k:]]
            best = None
            for pk, (pcost, plh, prh, _) in history[-1].items():
                held = [n for n in prev_ev["notes"] if n["end"] > ev["t"] + 1e-6]
                held_lo = [n["midi"] for n in held if prev_ev["notes"].index(n) < pk]
                held_hi = [n["midi"] for n in held if prev_ev["notes"].index(n) >= pk]
                cost = pcost + _event_split_cost(
                    lo, hi, held_lo, held_hi, plh, prh, tf, max_reach,
                    lo_active=pk > 0, hi_active=pk < len(prev_ev["notes"]))
                if best is None or cost < best[0]:
                    lh = sum(lo) / len(lo) if lo else plh
                    rh = sum(hi) / len(hi) if hi else prh
                    best = (cost, lh, rh, pk)
            new_states[k] = best
        history.append(new_states)

    # Backtrack the cheapest path and label the notes.
    k = min(history[-1], key=lambda s: history[-1][s][0])
    for i in range(len(events) - 1, -1, -1):
        for j, note in enumerate(events[i]["notes"]):
            note["hand"] = "L" if j < k else "R"
        k = history[i][k][3]


def _event_split_cost(lo, hi, held_lo, held_hi, plh, prh, tf,
                      max_reach=MAX_REACH, lo_active=False, hi_active=False):
    cost = _split_hand_cost(lo, held_lo, max_reach)
    cost += _split_hand_cost(hi, held_hi, max_reach)
    lh = sum(lo) / len(lo) if lo else None
    rh = sum(hi) / len(hi) if hi else None
    if lh is not None:
        cost += WEIGHTS["split_move"] * abs(lh - plh) * tf
        cost += WEIGHTS["split_range"] * max(0.0, lh - 60)
        if not lo_active:
            cost += WEIGHTS["split_activation"]
    if rh is not None:
        cost += WEIGHTS["split_move"] * abs(rh - prh) * tf
        cost += WEIGHTS["split_range"] * max(0.0, 60 - rh)
        if not hi_active:
            cost += WEIGHTS["split_activation"]
    if lo and hi:
        overlap = max(lo) - min(hi)
        if overlap > 0:
            cost += WEIGHTS["split_cross"] * overlap
        if lh > rh:
            cost += WEIGHTS["split_order"] * (lh - rh)
    return cost


def assign_hands_from_tracks(notes):
    """Trust track structure: with exactly two note-bearing tracks, the one
    with the higher mean pitch is the right hand. Returns True on success."""
    tracks = {}
    for n in notes:
        tracks.setdefault(n["track"], []).append(n["midi"])
    if len(tracks) != 2:
        return False
    lo_track = min(tracks, key=lambda t: sum(tracks[t]) / len(tracks[t]))
    for n in notes:
        n["hand"] = "L" if n["track"] == lo_track else "R"
    return True


# ---------------------------------------------------------------------------
# Per-hand fingering
# ---------------------------------------------------------------------------

def _chord_states(sounding, factor):
    """All feasible finger assignments for the sounding notes of one hand.

    `sounding` is sorted ascending by hand-oriented pitch q; an assignment
    is a strictly increasing finger tuple of the same length. Feasibility:
    every finger pair's simultaneous span must be positive and within the
    (scaled) practical limit.
    """
    n = len(sounding)
    states = []
    for fingers in itertools.combinations((1, 2, 3, 4, 5), n):
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                span = sounding[j]["q"] - sounding[i]["q"]
                pair = (fingers[i], fingers[j])
                if span < 1 or span > SPANS[pair][5] * factor:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            states.append(fingers)
    return states


def _state_cost(sounding, fingers, factor):
    """Vertical (within-chord) ergonomic cost of one assignment."""
    cost = 0.0
    for i in range(len(sounding)):
        f, note = fingers[i], sounding[i]
        if f == 4:
            cost += WEIGHTS["weak_4"]
        elif f == 5:
            cost += WEIGHTS["weak_5"]
        if key_layout.is_black(note["midi"]):
            if f == 1:
                cost += WEIGHTS["thumb_black"]
            elif f == 5:
                cost += WEIGHTS["pinky_black"]
        for j in range(i + 1, len(sounding)):
            cost += _span_cost(sounding[j]["q"] - note["q"],
                               (f, fingers[j]), factor)
    return cost


def _hand_pos(sounding, fingers):
    """Estimated hand position (thumb anchor) in q-space."""
    return sum(n["q"] - FINGER_OFFSET[f]
               for n, f in zip(sounding, fingers)) / len(sounding)


def _transition_cost(prev_sounding, prev_fingers, sounding, fingers,
                     new_notes, dt, factor):
    tf = _time_factor(dt)
    cost = WEIGHTS["move"] * abs(
        _hand_pos(sounding, fingers) - _hand_pos(prev_sounding, prev_fingers)) * tf

    prev_by_id = {id(n): f for n, f in zip(prev_sounding, prev_fingers)}
    new_ids = {id(n) for n in new_notes}
    for note, f in zip(sounding, fingers):
        if id(note) in prev_by_id and id(note) not in new_ids:
            if prev_by_id[id(note)] != f:  # finger substitution on a hold
                cost += WEIGHTS["substitution"]

    # Same finger restruck on a different key.
    for pn, pf in zip(prev_sounding, prev_fingers):
        for note, f in zip(sounding, fingers):
            if id(note) in new_ids and f == pf and note["q"] != pn["q"]:
                cost += WEIGHTS["same_finger"] * tf

    # Thumb passing / crossing, meaningful for single-note melodic motion.
    if len(prev_sounding) == 1 and len(new_notes) == 1:
        pn, pf = prev_sounding[0], prev_fingers[0]
        note, f = new_notes[0], None
        for nn, ff in zip(sounding, fingers):
            if id(nn) == id(new_notes[0]):
                note, f = nn, ff
                break
        if f is not None and pf != f:
            crossing = ((f == 1 and pf > 1 and note["q"] > pn["q"]) or
                        (pf == 1 and f > 1 and note["q"] < pn["q"]))
            if crossing:
                cost += WEIGHTS["thumb_pass"]
                if key_layout.is_black(note["midi"]):
                    cost += WEIGHTS["thumb_pass_black"]
            else:
                lo, hi = min(pf, f), max(pf, f)
                span = note["q"] - pn["q"] if pf < f else pn["q"] - note["q"]
                cost += 0.25 * _span_cost(span, (lo, hi), factor)
    return cost


def finger_hand(hand_notes, hand, hand_size="M", beam=BEAM_WIDTH):
    """Assign a finger to every note of one hand; writes note["finger"]."""
    if not hand_notes:
        return
    factor = HAND_SIZE_FACTOR[hand_size]
    sign = 1 if hand == "R" else -1
    for n in hand_notes:
        n["q"] = sign * n["midi"]

    events = group_events(hand_notes)
    beams = []  # per event: {fingers: (cost, back_state)}
    soundings = []
    for i, ev in enumerate(events):
        held = []
        if i > 0:
            held = [n for n in soundings[-1]
                    if n["end"] > ev["t"] + 1e-6 and id(n) not in
                    {id(m) for m in ev["notes"]}]
        sounding = sorted(held + ev["notes"], key=lambda n: n["q"])
        if len(sounding) > 5:
            # More sounding notes than fingers: release the oldest holds.
            held.sort(key=lambda n: n["start"])
            for drop in held:
                if len(sounding) <= 5:
                    break
                sounding.remove(drop)
        soundings.append(sounding)

        candidates = _chord_states(sounding, factor)
        if not candidates:
            # Impossible stretch even at practical limits: release all holds.
            sounding = sorted(ev["notes"], key=lambda n: n["q"])[:5]
            soundings[-1] = sounding
            candidates = _chord_states(sounding, factor)

        states = {}
        for fingers in candidates:
            vcost = _state_cost(sounding, fingers, factor)
            if i == 0:
                states[fingers] = (vcost, None)
            else:
                dt = max(ev["t"] - events[i - 1]["t"], 1e-3)
                best = None
                for pf, (pcost, _) in beams[-1].items():
                    cost = pcost + vcost + _transition_cost(
                        soundings[-2], pf, sounding, fingers,
                        ev["notes"], dt, factor)
                    if best is None or cost < best[0]:
                        best = (cost, pf)
                states[fingers] = best
        if len(states) > beam:
            keep = sorted(states, key=lambda s: states[s][0])[:beam]
            states = {s: states[s] for s in keep}
        beams.append(states)

    # Backtrack, labeling each note the first time it appears (its onset).
    state = min(beams[-1], key=lambda s: beams[-1][s][0])
    for i in range(len(events) - 1, -1, -1):
        new_ids = {id(n) for n in events[i]["notes"]}
        for note, f in zip(soundings[i], state):
            if id(note) in new_ids:
                note["finger"] = f
        state = beams[i][state][1]

    # Notes dropped from over-full chords still need a plausible label.
    for ev in events:
        for note in ev["notes"]:
            if "finger" not in note:
                labeled = [n for n in ev["notes"] if "finger" in n]
                if labeled:
                    nearest = min(labeled, key=lambda n: abs(n["q"] - note["q"]))
                    note["finger"] = nearest["finger"]
                else:
                    note["finger"] = 3
    for n in hand_notes:
        del n["q"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def finger_notes(notes, hand_size="M", hands_from_tracks=False):
    """Run hand assignment + per-hand fingering on a loaded note list."""
    if not notes:
        return notes
    if not (hands_from_tracks and assign_hands_from_tracks(notes)):
        assign_hands(group_events(notes))
    for hand in ("L", "R"):
        finger_hand([n for n in notes if n["hand"] == hand], hand, hand_size)
    return notes


def wrist_curve(notes, t_end, rate=30.0, smooth_window=0.3):
    """Sampled wrist x-position: smoothed centroid of active fingertips."""
    if not notes:
        return []
    samples = []
    x = key_layout.key_x(notes[0]["midi"])
    n_samples = int(t_end * rate) + 1
    for i in range(n_samples):
        t = i / rate
        active = [n for n in notes if n["start"] <= t < n["end"]]
        if active:
            x = sum(key_layout.key_x(n["midi"]) for n in active) / len(active)
        samples.append(x)
    half = max(1, int(smooth_window * rate / 2))
    smoothed = []
    for i in range(n_samples):
        window = samples[max(0, i - half):i + half + 1]
        smoothed.append(sum(window) / len(window))
    return [{"t": round(i / rate, 4), "x": round(v, 5)}
            for i, v in enumerate(smoothed)]


def compute_fingering(midi_path, hand_size="M", hands_from_tracks=False):
    """Full pipeline: MIDI file -> fingering timeline dict."""
    notes = load_notes(midi_path)
    finger_notes(notes, hand_size, hands_from_tracks)

    t_end = max((n["end"] for n in notes), default=0.0)
    out_notes = [{
        "start": round(n["start"], 5),
        "end": round(n["end"], 5),
        "midi": n["midi"],
        "velocity": n["velocity"],
        "hand": n["hand"],
        "finger": n["finger"],
        "x": round(key_layout.key_x(n["midi"]), 5),
        "y": round(key_layout.fingertip_y(n["midi"]), 5),
        "is_black": key_layout.is_black(n["midi"]),
    } for n in notes]

    return {
        "source": os.path.basename(midi_path),
        "hand_size": hand_size,
        "note_count": len(out_notes),
        "notes": out_notes,
        "hands": {
            hand: {"wrist_x": wrist_curve(
                [n for n in notes if n["hand"] == hand], t_end)}
            for hand in ("L", "R")
        },
    }


# ---------------------------------------------------------------------------
# Self-test: golden fingering patterns
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

    print("C major scale, one octave ascending (RH):")
    notes = finger_notes(_mknotes([(i, m) for i, m in
                                   enumerate([60, 62, 64, 65, 67, 69, 71, 72])]))
    fingers = [n["finger"] for n in notes]
    check("all right hand", all(n["hand"] == "R" for n in notes))
    check("standard 1-2-3-1-2-3-4-5", fingers == [1, 2, 3, 1, 2, 3, 4, 5],
          str(fingers))

    print("C-E-G chord (RH):")
    notes = finger_notes(_mknotes([(0, 60, 1.0), (0, 64, 1.0), (0, 67, 1.0)]))
    fingers = {n["midi"]: n["finger"] for n in notes}
    check("1-3-5", fingers == {60: 1, 64: 3, 67: 5}, str(fingers))

    print("Octave C4-C5 (RH):")
    notes = finger_notes(_mknotes([(0, 60, 1.0), (0, 72, 1.0)]))
    fingers = {n["midi"]: n["finger"] for n in notes}
    check("1-5", fingers == {60: 1, 72: 5}, str(fingers))

    print("C major scale, two octaves ascending (RH):")
    scale = [60, 62, 64, 65, 67, 69, 71]
    pitches = scale + [p + 12 for p in scale] + [84]
    notes = finger_notes(_mknotes([(i, m) for i, m in enumerate(pitches)]))
    fingers = [n["finger"] for n in notes]
    crossings = [(fingers[i], fingers[i + 1]) for i in range(len(fingers) - 1)
                 if fingers[i + 1] <= fingers[i]]
    check("starts on thumb", fingers[0] == 1, str(fingers))
    check("ends on pinky", fingers[-1] == 5, str(fingers))
    check("no repeated finger", all(a != b for a, b in zip(fingers, fingers[1:])))
    check("descents are thumb-under crossings",
          all(b == 1 for _a, b in crossings), str(fingers))

    print("C major scale, one octave descending from C4 (LH):")
    pitches = [60, 59, 57, 55, 53, 52, 50, 48]
    notes = finger_notes(_mknotes([(i, m) for i, m in enumerate(pitches)]))
    notes.sort(key=lambda n: n["start"])
    fingers = [n["finger"] for n in notes]
    check("all left hand", all(n["hand"] == "L" for n in notes))
    check("standard 1-2-3-1-2-3-4-5", fingers == [1, 2, 3, 1, 2, 3, 4, 5],
          str(fingers))

    print("Bass line + melody split:")
    bass = [(i, m, 0.45) for i, m in enumerate([36, 43, 40, 43, 36, 43, 40, 43])]
    melody = [(i, m) for i, m in enumerate([72, 74, 76, 77, 79, 77, 76, 74])]
    notes = finger_notes(_mknotes(bass + melody))
    check("bass to LH", all(n["hand"] == "L" for n in notes if n["midi"] < 60))
    check("melody to RH", all(n["hand"] == "R" for n in notes if n["midi"] >= 60))

    print()
    if failures:
        raise SystemExit(f"selftest: {len(failures)} check(s) failed: {failures}")
    print("selftest: all checks passed")


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
    parser.add_argument("--hands-from-tracks", action="store_true",
                        help="Trust a 2-track MIDI's tracks as the two hands")
    parser.add_argument("--selftest", action="store_true",
                        help="Run golden-pattern checks and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.midi_file:
        parser.error("midi_file is required unless --selftest is given")

    result = compute_fingering(args.midi_file, args.hand_size,
                               args.hands_from_tracks)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    by_hand = {"L": 0, "R": 0}
    for n in result["notes"]:
        by_hand[n["hand"]] += 1
    print(f"{result['source']}: {result['note_count']} notes fingered "
          f"(L: {by_hand['L']}, R: {by_hand['R']}) -> {args.out}")


if __name__ == "__main__":
    main()
