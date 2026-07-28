"""Drum striking planner: assigns each MIDI drum hit to a limb.

The drum analog of the other instruments' fingering.py. There is no
pitch->position ambiguity (each General MIDI percussion note maps to exactly
one voice via kit_layout), so this is a *limb scheduler* rather than a
fingering search. It decides which of the two sticks (left/right hand) or
two feet plays every hit, and emits a timeline JSON that animate_drums.py
keyframes.

Sticking model - the standard right-handed convention with automatic
fallbacks (see RESEARCH.md):

  * Feet are deterministic: the kick is the right foot, the hi-hat pedal the
    left foot.
  * Hands default to the convention: right hand on the hi-hat / ride, left
    hand on the snare. Toms and crashes are "flex" voices with no fixed hand.
  * For rapid successions (inter-onset gap below FAST_GAP) the hands
    alternate - a single-stroke roll (R L R L) - so fast runs and fills never
    ask one hand to play faster than it can.
  * Otherwise each hit goes to the hand that minimizes arm travel from its
    last position, with a penalty for abandoning a voice's convention hand,
    so grooves keep the natural hand assignment while fills take the nearest
    stick.
  * Simultaneous hits are split across the two hands to minimize total travel.

Usage::

    python -m drum_kit.fingering drum_demo.mid -o fingering.json
    python -m drum_kit.fingering --selftest
"""

import argparse
import json
import math
import os
import sys

try:
    from .piano_midi_animator import parse_midi
    from . import kit_layout
except ImportError:  # running as a loose script, not a package
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(_HERE)
    sys.path.append(os.path.dirname(_HERE))
    from piano.piano_midi_animator import parse_midi
    import kit_layout

try:
    from piano.fingering import load_notes, group_events
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.fingering import load_notes, group_events


# ---------------------------------------------------------------------------
# Sticking parameters
# ---------------------------------------------------------------------------
FAST_GAP = 0.14        # s; below this, successive stick hits alternate hands
CONV_PENALTY = 0.6     # metres-equivalent cost for leaving a voice's
                       # convention hand (keeps grooves on the standard grip)
BUSY_GAP = 0.05        # s; a hand asked to move far within this is flagged
BUSY_DIST = 0.15       # m of travel that BUSY_GAP cannot physically cover
# TOMS are flex voices with no convention hand, so without a bias the planner
# picks purely by travel from the hand's LAST position -- which lets a hand that
# just played the near side "stick" to a tom across the kit that it then has to
# reach for across the body (the left hand to the right-side toms is a genuine
# over-extension past arm+stick length, which the IK can only satisfy by pulling
# the grip up toward the shoulder -- angling the bead DOWN). Toms sit low in
# front of the drummer, so a real player splits them down the middle: the LEFT
# hand takes the drummer's-left toms (+X here), the RIGHT hand the right-side
# toms (-X). So a tom hit on the far side FOR A HAND pays CROSS_PENALTY per metre
# past the centreline, enough to overcome ordinary travel savings and keep each
# stick on its own side. Only the SIDE of the target matters (kit geometry, x=0
# kick/body centreline), not the body's arm lengths, so striking stays body-
# agnostic. This is deliberately NOT applied to the crashes: they sit high and
# overhead, where the right-handed convention crosses the right hand freely over
# to the drummer's-left cymbals (its hi-hat home is already there), so a side
# rule would fight the idiomatic cross-over. Fast rolls still alternate (handled
# before _cost), so this only steers the slower tom fills.
CROSS_TOMS = {"tom_hi", "tom_mid", "tom_floor"}
CROSS_PENALTY = 3.0    # metres-equivalent cost per metre a tom hit lies on the
                       # far side of the kit centre for the assigning hand

# Far-side cymbals: the hi-hat sits at the player's far left, the ride at the
# far right, so only their convention hand reaches them without the other arm
# contorting across the body. These never alternate off that hand.
LOCKED_VOICES = {"hihat_closed", "hihat_open", "ride", "ride_bell"}


def _beat_mapper(midi_path):
    """Return beats_at(seconds) -> fractional beats, inverting the tempo map.
    Constant tempo collapses to beats = seconds / seconds_per_beat."""
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


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _other(hand):
    return "L" if hand == "R" else "R"


def _cost(state, note, hand):
    """Cost of playing `note` with `hand` from its last position `state`:
    arm travel plus a penalty for abandoning the voice's convention hand."""
    c = _dist(state["pt"], note["point"])
    conv = note["limb"]
    if conv in ("R", "L") and conv != hand:
        c += CONV_PENALTY
    elif conv is None and note["voice"] in CROSS_TOMS:   # keep each stick on its
        sx = note["point"][0]                # own side of the kit (+X left / -X right)
        if hand == "L" and sx < 0.0:
            c += CROSS_PENALTY * -sx
        elif hand == "R" and sx > 0.0:
            c += CROSS_PENALTY * sx
    return c


def _assign_hands(stick_notes, hand, t, prev_hand, prev_stick_t, prev_voice, warnings):
    """Return [(note, 'R'|'L'), ...] for the simultaneous stick notes."""
    if len(stick_notes) == 1:
        n = stick_notes[0]
        gap = (t - prev_stick_t) if prev_stick_t is not None else None
        fast = prev_hand is not None and gap is not None and gap < FAST_GAP
        conv = n["limb"]                                 # 'R', 'L', or None
        if conv in ("R", "L"):
            # A convention voice (hi-hat/ride -> R, snare -> L) keeps its hand,
            # so returning to a groove after a fill re-anchors the grip. A fast
            # repeat of the SAME central voice (e.g. a snare roll) may alternate,
            # but far-side cymbals stay locked - this stops blind alternation
            # from flinging a hand across the kit into a contorted reach.
            can_roll = (fast and prev_voice == n["voice"]
                        and n["voice"] not in LOCKED_VOICES)
            h = _other(prev_hand) if can_roll else conv
        elif fast:
            h = _other(prev_hand)                        # flex voices: roll/fill
        else:
            h = "R" if _cost(hand["R"], n, "R") <= _cost(hand["L"], n, "L") else "L"
        return [(n, h)]

    # Two (or more) simultaneous hits: pick the R/L split with least travel.
    a, b = stick_notes[0], stick_notes[1]
    cost_ab = _cost(hand["R"], a, "R") + _cost(hand["L"], b, "L")
    cost_ba = _cost(hand["L"], a, "L") + _cost(hand["R"], b, "R")
    if cost_ab <= cost_ba:
        pairs = [(a, "R"), (b, "L")]
    else:
        pairs = [(a, "L"), (b, "R")]
    # Rare flam clusters (>2 at one instant): alternate the leftovers.
    h = pairs[-1][1]
    for extra in stick_notes[2:]:
        h = _other(h)
        pairs.append((extra, h))
        warnings.append(f"{len(stick_notes)} simultaneous stick hits at "
                        f"{t:.3f}s; extra voices stacked on alternating hands")
    return pairs


def _emit(out, note, limb, beats_at):
    x, y, z = note["point"]
    rec = {
        "start": round(note["start"], 5),
        "end": round(note["end"], 5),
        "beat": round(beats_at(note["start"]), 4),
        "midi": note["midi"],
        "velocity": note["velocity"],
        "voice": note["voice"],
        "target": note["target"],
        "limb": limb,
        "strike": note["strike"],
        "x": round(x, 5), "y": round(y, 5), "z": round(z, 5),
    }
    if "open" in note:
        rec["open"] = note["open"]
    out.append(rec)


def plan_strikes(notes, beats_at):
    """Core planner over a note list (from load_notes). Returns
    (out_notes, hihat_events, dropped, warnings)."""
    out, warnings = [], []
    dropped = 0
    hand = {"R": {"t": -1e9, "pt": kit_layout.strike_point("hihat_closed")},
            "L": {"t": -1e9, "pt": kit_layout.strike_point("snare")}}
    hihat_events = []          # (t, 'open'|'closed')
    prev_hand = None
    prev_stick_t = None
    prev_voice = None

    for ev in group_events(notes):
        t = ev["t"]
        stick_notes = []
        for n in ev["notes"]:
            info = kit_layout.strike_info(n["midi"])
            if info is None:
                warnings.append(f"note {n['midi']} at {t:.3f}s has no voice; dropped")
                dropped += 1
                continue
            merged = dict(n)
            merged.update(info)              # voice, target, point, strike, limb[, open]
            if merged["strike"] == "kick":
                _emit(out, merged, "footR", beats_at)
            elif merged["strike"] == "hat_pedal":
                _emit(out, merged, "footL", beats_at)
                hihat_events.append((t, "closed"))
            else:                            # stick
                stick_notes.append(merged)
                if merged["voice"] in ("hihat_open", "hihat_closed"):
                    hihat_events.append((t, "open" if merged.get("open") else "closed"))

        if not stick_notes:
            continue

        assigned = _assign_hands(stick_notes, hand, t, prev_hand, prev_stick_t,
                                 prev_voice, warnings)
        for n, h in assigned:
            if t - hand[h]["t"] < BUSY_GAP and _dist(hand[h]["pt"], n["point"]) > BUSY_DIST:
                warnings.append(f"{h} hand asked to cross the kit within "
                                f"{BUSY_GAP*1000:.0f}ms at {t:.3f}s")
            _emit(out, n, h, beats_at)
            hand[h] = {"t": t, "pt": n["point"]}

        if len(assigned) == 1:
            prev_hand, prev_voice = assigned[0][1], assigned[0][0]["voice"]
        else:
            prev_hand, prev_voice = None, None
        prev_stick_t = t

    out.sort(key=lambda r: (r["start"], r["voice"]))
    hihat_events.sort(key=lambda e: e[0])
    return out, hihat_events, dropped, warnings


def compute_striking(midi_path):
    """Full pipeline: MIDI file -> drum striking timeline dict."""
    raw = load_notes(midi_path)
    beats_at = _beat_mapper(midi_path)
    out, hihat_events, dropped, warnings = plan_strikes(raw, beats_at)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    # Collapse consecutive identical hi-hat states into a compact timeline.
    hihat = []
    for t, state in hihat_events:
        if not hihat or hihat[-1]["state"] != state:
            hihat.append({"t": round(t, 5), "state": state})

    return {
        "source": os.path.basename(midi_path),
        "instrument": "drums",
        "note_count": len(out),
        "dropped_count": dropped,
        "voices": sorted({n["voice"] for n in out}),
        "notes": out,
        "hihat": hihat,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _mknotes(items, dur=0.05):
    """Build a note list from (start_sec, midi[, velocity]) tuples."""
    notes = []
    for it in items:
        start, midi = it[0], it[1]
        vel = it[2] if len(it) > 2 else 90
        notes.append({"start": start, "end": start + dur, "midi": midi,
                      "velocity": vel, "track": 0})
    notes.sort(key=lambda n: (n["start"], n["midi"]))
    return notes


def selftest():
    failures = []
    beats_at = lambda s: s * 2.0   # 120 BPM

    def check(name, cond, detail=""):
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not cond:
            failures.append(name)

    # Mapping + feet: unknown notes drop; kick/pedal go to the right/left foot.
    notes = _mknotes([(0.0, 36), (0.5, 44), (1.0, 999)])
    out, hh, dropped, warns = plan_strikes(notes, beats_at)
    limb = {n["voice"]: n["limb"] for n in out}
    check("unknown note dropped", dropped == 1)
    check("kick -> right foot", limb.get("kick") == "footR")
    check("hi-hat pedal -> left foot", limb.get("hihat_pedal") == "footL")

    # Basic groove: closed hat on the right, snare on the left (convention),
    # played at a comfortable eighth-note spacing so alternation does not fire.
    groove = _mknotes([(0.0, 42), (0.25, 42), (0.5, 38),
                       (0.75, 42), (1.0, 42), (1.25, 38)])
    out, _, _, _ = plan_strikes(groove, beats_at)
    hats = [n["limb"] for n in out if n["voice"] == "hihat_closed"]
    snares = [n["limb"] for n in out if n["voice"] == "snare"]
    check("convention: hats on right hand", all(h == "R" for h in hats), str(hats))
    check("convention: snare on left hand", all(s == "L" for s in snares), str(snares))

    # Fast single-stroke run on a central voice (snare) alternates hands; the
    # far-side hi-hat/ride stay locked to their convention hand instead.
    burst = _mknotes([(i * 0.125, 38) for i in range(8)])
    out, _, _, _ = plan_strikes(burst, beats_at)
    hands = [n["limb"] for n in out]
    alternating = all(hands[i] != hands[i + 1] for i in range(len(hands) - 1))
    check("rapid snare run alternates sticks", alternating, "".join(hands))
    hat_run = _mknotes([(i * 0.125, 42) for i in range(6)])
    out, _, _, _ = plan_strikes(hat_run, beats_at)
    check("fast hi-hat stays on the right hand",
          all(n["limb"] == "R" for n in out), "".join(n["limb"] for n in out))

    # Simultaneous snare + hat: the two hands split (never double-booked).
    both = _mknotes([(0.0, 38), (0.0, 42)])
    out, _, _, _ = plan_strikes(both, beats_at)
    used = sorted(n["limb"] for n in out)
    check("simultaneous hits use both hands", used == ["L", "R"], str(used))

    # Descending tom fill in sixteenths alternates across the kit.
    fill = _mknotes([(0.0, 38)] + [(0.125 + i * 0.125, m)
                                   for i, m in enumerate([50, 45, 41, 50, 45, 41])])
    out, _, _, _ = plan_strikes(fill, beats_at)
    fh = [n["limb"] for n in out]
    check("tom fill alternates", all(fh[i] != fh[i + 1] for i in range(len(fh) - 1)),
          "".join(fh))

    if failures:
        print(f"\nSELFTEST FAILED: {failures}")
        sys.exit(1)
    print("\nselftest: all checks passed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("midi_file", nargs="?", help="MIDI file path")
    parser.add_argument("-o", "--out", default="fingering.json",
                        help="Output JSON path (default: fingering.json)")
    parser.add_argument("--selftest", action="store_true",
                        help="Run sticking checks and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.midi_file:
        parser.error("midi_file is required unless --selftest is given")

    result = compute_striking(args.midi_file)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    feet = sum(1 for n in result["notes"] if n["limb"].startswith("foot"))
    print(f"{result['source']}: {result['note_count']} hits planned "
          f"({result['note_count'] - feet} stick, {feet} foot; "
          f"{len(result['voices'])} voices, {result['dropped_count']} dropped) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
