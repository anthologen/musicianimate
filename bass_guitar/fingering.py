"""Automatic bass tablature, left-hand fingering, and right-hand plucking
from a MIDI file.

Given any Standard MIDI File, decides which string and fret sounds each
note, which left-hand finger (1 = index .. 4 = pinky, 0 = open string)
frets it, and which right-hand finger ("i" = index, "m" = middle) plucks
it, producing a timeline a hand rig can be animated from.

The left-hand tab is the classic ergonomic dynamic-programming method for
string instruments (Sayegh 1989's optimum-path paradigm; Radisavljevic &
Driessen 2004; Hori 2021) - see bass_guitar/RESEARCH.md - reusing the
guitar engine's structure with three bass-specific changes:

  1. **Four low strings, mostly monophonic.** Tuning is EADG an octave
     below the guitar's low four; barres never occur, so the guitar's
     barre grips are dropped and the search is a near-linear per-note
     (string, fret, finger) choice.
  2. **Position-dependent finger model.** On a 34" scale the low frets
     are physically huge, so below SIMANDL_MAX_FRET the hand uses the
     Simandl 1-2-4 system (fingers 1, 2, 4 over three frets; the 3rd
     finger only reinforces the 4th and is penalized when used alone),
     while higher up it switches to one-finger-per-fret (1-2-3-4). Both
     regimes fall out of a single ``_finger_offset`` table.
  3. **Fingerstyle right hand.** Notes are plucked with strict i-m
     alternation, raking (reusing the finger) when a line descends across
     to a thicker string. A ``--style pick`` mode instead emits the
     guitar's metric down/up pick strokes.

Fingertip target coordinates come from bass_guitar.fret_layout, so they
line up exactly with the meshes built by build_bass_guitar.py.

Usage::

    python -m bass_guitar.fingering demo.mid -o fingering.json
    python -m bass_guitar.fingering --selftest
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

MAX_FRETTED = 4       # one finger per fretted note
SIMANDL_MAX_FRET = 5  # at/below this fret the hand plays 1-2-4 (Simandl);
                      # above it, one-finger-per-fret (1-2-3-4)
REACH_COMF_M = 0.075  # comfortable index..pinky extent along the neck (size M)
REACH_MAX_M = 0.115   # practical limit: allows a low 1-2-4 or a high 1-2-3-4
                      # span, forbids a low four-fret stretch; scales with size
START_POSITION = 1.0  # the hand starts hovering near first position
BEAM_WIDTH = 200

# Right-hand plucking.
PLUCK_GAP_RESET = 0.35  # seconds of silence that re-anchor alternation to index

WEIGHTS = {
    # static (per event state)
    "fretted_note": 0.10,        # per fretted note; opens are intrinsically cheaper
    "high_position": 0.07,       # per fret of the grip's mean fretted fret
    "span_comf": 30.0,           # per metre of extent beyond the comfortable reach
    "pos_spread": 0.75,          # per fret the fingers disagree on the hand position
    "weak_ring": 0.15,           # per use of finger 3
    "weak_pinky": 0.30,          # per use of finger 4
    "weak_low_ring": 0.55,       # finger 3 alone in a low (Simandl) position
    "index_not_lowest": 0.20,    # chord whose lowest fret isn't finger 1 or 2
    # transition (scaled by _time_factor(dt))
    "shift": 1.0,                # per fret of hand-position movement - the dominant term
    "same_finger": 1.0,          # same finger re-pressed on a different fret
    "finger_string_hop": 0.15,   # per string a re-used finger crosses
    "string_change": 0.10,       # per string step between consecutive melody notes
    "substitution": 6.0,         # changing the fretting finger on a held note
}


def _finger_offset(finger, fret):
    """Fret offset of `finger` from the hand's index position. Below
    SIMANDL_MAX_FRET the pinky and ring share the +2 fret (Simandl 1-2-4,
    the 3rd finger reinforcing the 4th); higher up each finger owns its
    own fret (one-finger-per-fret)."""
    if fret <= SIMANDL_MAX_FRET:
        return (0, 0, 1, 2, 2)[finger]   # 1->0, 2->1, 3->2, 4->2
    return finger - 1                     # 1->0, 2->1, 3->2, 4->3


# ---------------------------------------------------------------------------
# Scope enforcement: range folding and chord simplification
# ---------------------------------------------------------------------------

def sanitize_notes(notes, warnings):
    """Fold out-of-range pitches into the bass's range, in place."""
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

    Merges unison duplicates and trims events with more notes than strings
    down to the bass note plus the top three. Trimmed notes are tagged
    note["dropped"] and removed from the event. Bass lines are near always
    monophonic, so this rarely fires."""
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

    Feasibility: distinct strings, at most MAX_FRETTED fretted notes,
    fretted extent within the scaled physical reach, and strictly
    increasing fingers up the frets (fret ties give the bass-side note
    the lower finger)."""
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
        if len(fretted) > MAX_FRETTED:
            continue
        frets = [f for f, _, _ in fretted]
        if fret_layout.press_y(min(frets)) - fret_layout.press_y(max(frets)) > reach:
            continue
        fretted.sort()
        for fingers in itertools.combinations((1, 2, 3, 4), len(fretted)):
            asg = [0] * len(combo)
            for (f, s, idx), fg in zip(fretted, fingers):
                asg[idx] = fg
            states.append(tuple((s, f, fg)
                                for (s, f), fg in zip(combo, asg)))
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

    # Every finger should agree on one hand position (index fret): a grip
    # where the implied positions disagree is a stretch or a misfingering.
    # This handles both regimes automatically via _finger_offset.
    positions = [f - _finger_offset(fg, f) for f, _, fg in fretted]
    cost += WEIGHTS["pos_spread"] * (max(positions) - min(positions))

    for f, _, g in fretted:
        if g == 3:
            cost += WEIGHTS["weak_ring"]
            if f <= SIMANDL_MAX_FRET:
                cost += WEIGHTS["weak_low_ring"]
        elif g == 4:
            cost += WEIGHTS["weak_pinky"]
    if len(fretted) >= 2 and fretted[0][2] not in (1, 2):
        cost += WEIGHTS["index_not_lowest"]
    return cost


def _hand_pos(state, prev_pos):
    """Index-finger fret implied by a grip; all-open grips keep the
    previous position, making opens free pivots."""
    fretted = [(f, fg) for _, f, fg in state if f > 0]
    if not fretted:
        return prev_pos
    return sum(f - _finger_offset(fg, f) for f, fg in fretted) / len(fretted)


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
    # cost a little.
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
    admits no grip at all, holds are released oldest-first, then the event
    itself is thinned from the middle out (tagging drops)."""
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
        for note, (s, f, fg) in zip(soundings[i], key[0]):
            if id(note) in onset_ids:
                note["string"], note["fret"], note["finger"] = s, f, fg
        key = beams[i][key][1]


# ---------------------------------------------------------------------------
# Right-hand plucking (fingerstyle) assignment
# ---------------------------------------------------------------------------

def assign_plucking(notes):
    """Write note["pluck_finger"] in {"i","m"} by strict index/middle
    alternation, with raking and post-rest re-anchoring.

    Rules (see RESEARCH.md): consecutive notes strictly alternate i-m-i-m;
    a line that descends across strings toward a thicker (lower-index)
    string *rakes* - the finger that just played drags across and repeats,
    rather than alternating; after a rest (>= PLUCK_GAP_RESET) alternation
    re-anchors to the index finger. In a double-stop the index takes the
    lower string and the middle the next up."""
    events = group_events([n for n in notes if "string" in n])
    last_finger = None      # 'i' or 'm'
    last_string = None
    prev_t = None
    for ev in events:
        gap = None if prev_t is None else ev["t"] - prev_t
        reset = last_finger is None or (gap is not None and gap >= PLUCK_GAP_RESET)
        ev_notes = sorted(ev["notes"], key=lambda n: n["string"])
        if len(ev_notes) > 1:
            for k, n in enumerate(ev_notes):
                n["pluck_finger"] = "i" if k % 2 == 0 else "m"
            last_finger = ev_notes[-1]["pluck_finger"]
            last_string = ev_notes[-1]["string"]
        else:
            n = ev_notes[0]
            s = n["string"]
            if reset:
                fin = "i"
            elif last_string is not None and s < last_string:
                fin = last_finger                      # rake to the thicker string
            else:
                fin = "m" if last_finger == "i" else "i"  # strict alternation
            n["pluck_finger"] = fin
            last_finger = fin
            last_string = s
        prev_t = ev["t"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def tab_notes(notes, hand_size="M"):
    """Sanitize, simplify, assign a loaded note list, and pluck it.

    Returns (kept_notes, warnings): the notes that received a
    string/fret/finger/pluck_finger, plus human-readable warnings for
    anything folded or dropped along the way."""
    warnings = []
    if not notes:
        return notes, warnings
    sanitize_notes(notes, warnings)
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    events = group_events(notes)
    simplify_chords(events, warnings)
    assign_positions(events, hand_size, warnings)
    kept = [n for n in notes if not n.get("dropped") and "string" in n]
    assign_plucking(kept)
    return kept, warnings


def _smoothed_samples(samples, rate, smooth_window):
    half = max(1, int(smooth_window * rate / 2))
    out = []
    for i in range(len(samples)):
        window = samples[max(0, i - half):i + half + 1]
        out.append(sum(window) / len(window))
    return out


def hand_position_curve(notes, t_end, rate=30.0, smooth_window=0.3):
    """Sampled left-hand position over time, box-smoothed: the index-finger
    fret plus the matching neck y."""
    samples = []
    pos = START_POSITION
    n_samples = int(t_end * rate) + 1
    for i in range(n_samples):
        t = i / rate
        active = [n for n in notes
                  if n["start"] <= t < n["end"] and n["fret"] > 0]
        if active:
            pos = sum(n["fret"] - _finger_offset(n["finger"], n["fret"])
                      for n in active) / len(active)
        samples.append(pos)
    out = []
    for i, p in enumerate(_smoothed_samples(samples, rate, smooth_window)):
        y = fret_layout.press_y(max(1.0, min(float(fret_layout.NUM_FRETS), p)))
        out.append({"t": round(i / rate, 4),
                    "fret": round(p, 3), "y": round(y, 5)})
    return out


def contact_curve(notes, t_end, rate=30.0, smooth_window=0.15):
    """Sampled plucking-hand x: the most recently struck string's contact
    point, lightly smoothed. y and z are the constants PLUCK_Y / STRING_Z."""
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
    inverting the MIDI tempo map. The pick animator uses beat positions to
    decide down- vs up-strokes. Constant tempo collapses to
    beats = seconds / sec_per_beat."""
    ticks_per_beat, tempo_map, _ = parse_midi(midi_path)
    pts = []
    cum_sec, prev_tick, prev_tempo = 0.0, 0, tempo_map[0][1]
    for tick, tempo in tempo_map:
        cum_sec += (tick - prev_tick) * (prev_tempo / 1e6) / ticks_per_beat
        pts.append((cum_sec, tick / ticks_per_beat))
        prev_tick, prev_tempo = tick, tempo
    last_spb = prev_tempo / 1e6

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


def compute_fingering(midi_path, hand_size="M", style="finger"):
    """Full pipeline: MIDI file -> tablature/fingering/plucking timeline dict."""
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
            "pluck_finger": n.get("pluck_finger", "i"),
            "is_open": n["fret"] == 0,
            "x": round(x, 5),
            "y": round(y, 5),
            "z": round(z, 5),
            "pluck_x": round(px, 5),
            "pluck_y": round(py, 5),
        })

    hands = {"fret": {"position": hand_position_curve(notes, t_end)}}
    if style == "pick":
        hands["pick"] = {"x": contact_curve(notes, t_end)}
    else:
        hands["pluck"] = {"x": contact_curve(notes, t_end)}

    return {
        "source": os.path.basename(midi_path),
        "instrument": "bass_guitar",
        "style": style,
        "tuning": list(fret_layout.TUNING),
        "hand_size": hand_size,
        "note_count": len(out_notes),
        "dropped_count": total - len(out_notes),
        "notes": out_notes,
        "hands": hands,
    }


# ---------------------------------------------------------------------------
# Self-test: golden bass patterns
# ---------------------------------------------------------------------------

def _mknotes(seq, dur=0.20, step=0.25):
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

    def run(seq, dur=0.20, step=0.25):
        kept, warnings = tab_notes(_mknotes(seq, dur, step))
        kept.sort(key=lambda n: (n["start"], n["midi"]))
        return kept, warnings

    def pitches_ok(kept):
        return all(fret_layout.midi_of(n["string"], n["fret"]) == n["midi"]
                   for n in kept)

    print("Open strings E A D G:")
    kept, _ = run(list(enumerate([28, 33, 38, 43])))
    sf = [(n["string"], n["fret"]) for n in kept]
    check("pitches preserved", pitches_ok(kept))
    check("all open on their own string",
          sf == [(0, 0), (1, 0), (2, 0), (3, 0)], str(sf))

    print("Simandl first position F F# G on the E string:")
    kept, _ = run(list(enumerate([29, 30, 31])), dur=0.11, step=0.125)
    sf = [(n["string"], n["fret"]) for n in kept]
    fingers = [n["finger"] for n in kept]
    check("all on the E string, frets 1-2-3",
          sf == [(0, 1), (0, 2), (0, 3)], str(sf))
    check("Simandl 1-2-4 fingering (skips the lone 3rd finger)",
          fingers == [1, 2, 4], str(fingers))

    print("High chromatic run E3..G3 leaves the Simandl zone:")
    # These pitches are only reachable high on the neck (G3 forces fret 12),
    # so the hand plays one-finger-per-fret 1-2-3-4 up the G string.
    kept, _ = run(list(enumerate([52, 53, 54, 55])), dur=0.11, step=0.125)
    check("pitches preserved", pitches_ok(kept))
    check("all above the Simandl zone",
          all(n["fret"] > SIMANDL_MAX_FRET for n in kept),
          str([(n["string"], n["fret"]) for n in kept]))
    check("one-finger-per-fret 1-2-3-4 up high",
          [n["finger"] for n in kept] == [1, 2, 3, 4],
          str([(n["string"], n["fret"], n["finger"]) for n in kept]))

    print("Walking-bass scale G1..G2, smooth position shifts:")
    kept, _ = run(list(enumerate([31, 33, 35, 36, 38, 40, 42, 43])),
                  dur=0.22, step=0.25)
    check("pitches preserved", pitches_ok(kept))
    poss = [n["fret"] - _finger_offset(n["finger"], n["fret"])
            for n in kept if n["fret"] > 0]
    check("no wild position jumps (<= 3 frets)",
          all(abs(b - a) <= 3 for a, b in zip(poss, poss[1:])), str(poss))

    print("Repeated A1, stable position:")
    kept, _ = run([(i, 33) for i in range(6)])
    sf = {(n["string"], n["fret"]) for n in kept}
    check("stable position for a repeated pitch", len(sf) == 1, str(sf))

    print("Out-of-range low note folds up:")
    kept, warnings = run([(0, 20)])
    check("folded into range and tabbed",
          len(kept) == 1 and kept[0]["midi"] == 32, str(
              [(n["midi"], n["string"], n["fret"]) for n in kept]))
    check("fold warning emitted", any("folded" in w for w in warnings))

    print("Plucking alternates on a repeated string:")
    kept, _ = run([(i, 33) for i in range(6)], dur=0.11, step=0.125)
    fins = [n["pluck_finger"] for n in kept]
    check("strict i-m alternation", fins == ["i", "m", "i", "m", "i", "m"],
          str(fins))

    print("Plucking rakes on a descending string crossing:")
    # G2 (G string, open) -> D2 (D string, open) -> A1 (A string, open):
    # each step drops to a thicker string, so the same finger rakes across.
    kept, _ = run(list(enumerate([43, 38, 33])), dur=0.11, step=0.125)
    kept.sort(key=lambda n: n["start"])
    fins = [n["pluck_finger"] for n in kept]
    check("same finger reused down the strings (rake)",
          fins[0] == fins[1] == fins[2], str(fins))

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
    parser.add_argument("--style", default="finger", choices=("finger", "pick"),
                        help="Right-hand technique (default: finger)")
    parser.add_argument("--pick", dest="style", action="store_const",
                        const="pick", help="Shorthand for --style pick")
    parser.add_argument("--selftest", action="store_true",
                        help="Run golden-pattern checks and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.midi_file:
        parser.error("midi_file is required unless --selftest is given")

    result = compute_fingering(args.midi_file, args.hand_size, args.style)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    opens = sum(1 for n in result["notes"] if n["is_open"])
    max_fret = max((n["fret"] for n in result["notes"]), default=0)
    print(f"{result['source']}: {result['note_count']} notes tabbed "
          f"({opens} open, {result['note_count'] - opens} fretted, "
          f"max fret {max_fret}, {result['dropped_count']} dropped, "
          f"{args.style}-style) -> {args.out}")


if __name__ == "__main__":
    main()
