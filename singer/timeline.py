"""Plan the phoneme timeline for a sung line - the singer's fingering.py.

Takes a karaoke file (kar.load_kar), phonemises each syllable (g2p) and lays
the phonemes out in time, producing the list of timed mouth positions that
animate_mouth.py keyframes.  Output is timeline.json, the analogue of the
other instruments' fingering.json.

The timing model, which is what makes this look sung rather than spoken:

* **Vowels carry the note.**  A syllable's vowel is held for essentially the
  whole note; it is the only phoneme whose duration scales with the melody.
* **Consonants are quick and live at the edges.**  Onset consonants take a
  fixed ~55 ms each and are placed so that they land *before* the beat and
  the vowel arrives on it (``ONSET_LEAD``) - singers articulate ahead of the
  beat, which is why a sung "star" has its /s/ before the downbeat.  Coda
  consonants take the last few tens of milliseconds of the note.
* **Diphthongs glide late.**  /aɪ/ in "high" is sung as a long /ɑ/ with the
  /ɪ/ tucked into the end of the note, not as two equal halves.
* **Melisma sustains.**  A note with no syllable of its own carries the
  previous vowel onward (re-stated, so its own velocity can change the
  loudness variant).
* **Silence closes the mouth.**  Any gap longer than ``MIN_SIL`` becomes an
  explicit SIL event, so the mouth is shut whenever nothing is being sung.

Every event also carries a loudness variant (soft/medium/loud) taken from the
note's MIDI velocity, selecting between the three mouth positions each viseme
provides.

Usage::

    python -m singer.timeline singer/twinkle_demo.kar -o singer/timeline.json
    python -m singer.timeline --selftest
"""

import argparse
import json
import os
import sys

try:
    from . import g2p, kar, mouth_shapes
except ImportError:  # loaded as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from singer import g2p, kar, mouth_shapes

# ---------------------------------------------------------------------------
# Timing constants (seconds)
# ---------------------------------------------------------------------------
ONSET_DUR = 0.055      # nominal hold for one onset consonant
CODA_DUR = 0.050       # nominal hold for one coda consonant
ONSET_LEAD = 0.70      # share of the onset sung BEFORE the beat
MIN_VOWEL = 0.070      # a vowel never gets squeezed below this
GLIDE_FRAC = 0.30      # share of a diphthong spent on its second element
GLIDE_MAX = 0.14
LEGATO_GAP = 0.12      # notes closer than this are sung joined-up
MIN_SIL = 0.10         # shorter silences are not worth closing the mouth for
LEAD_SIL = 0.40        # closed mouth held before the first note
TAIL_SIL = 0.60        # ... and after the last one


def _phones_for(note, cache):
    """IPA phones for this note's syllable, in the context of its whole word.

    The word is phonemised once and then split into exactly as many syllables
    as the lyric hyphenated it into - the lyric's syllabification is
    authoritative ("dia-mond" is two notes even though the dictionary says
    three), our syllabifier only decides where the consonants fall.
    """
    word = note["word"]
    if not word:
        return []
    if note["syllable_index"] == 0 or word not in cache:
        cache.clear()
        cache[word] = g2p.split_into(g2p.word_to_phones(word),
                                     note["syllable_count"])
    parts = cache[word]
    index = min(note["syllable_index"], len(parts) - 1)
    return list(parts[index])


def _split_syllable(phones):
    """(onset, nucleus, coda) - consonants before the vowel, the vowel run,
    and whatever follows it."""
    first = next((i for i, p in enumerate(phones)
                  if mouth_shapes.is_vowel(p)), None)
    if first is None:
        return list(phones), [], []
    last = first
    while last + 1 < len(phones) and mouth_shapes.is_vowel(phones[last + 1]):
        last += 1
    return phones[:first], phones[first:last + 1], phones[last + 1:]


def _spread(start, end, phones, kind, level, out):
    """Lay `phones` out evenly across [start, end]."""
    if not phones or end <= start:
        return
    step = (end - start) / len(phones)
    for i, phone in enumerate(phones):
        out.append(_event(start + i * step, start + (i + 1) * step,
                          phone, kind, level))


def _event(start, end, phone, kind, level):
    return {"start": start, "end": end, "phone": phone,
            "viseme": mouth_shapes.viseme_for(phone), "kind": kind,
            "level": level}


def _vowel_events(start, end, vowel, level, out):
    """A vowel, with a late glide if it is a diphthong."""
    if end <= start:
        return
    parts = mouth_shapes.DIPHTHONGS.get(vowel)
    if not parts:
        out.append(_event(start, end, vowel, "vowel", level))
        return
    glide = min(GLIDE_MAX, GLIDE_FRAC * (end - start))
    out.append(_event(start, end - glide, parts[0], "vowel", level))
    out.append(_event(end - glide, end, parts[1], "glide", level))


def plan_events(song):
    """[event] for a song loaded by kar.load_kar."""
    notes = song["notes"]
    events = []
    cache = {}
    held_vowel = None

    for index, note in enumerate(notes):
        start, level = note["start"], mouth_shapes.level_for_velocity(
            note["velocity"])
        next_start = notes[index + 1]["start"] if index + 1 < len(notes) else None
        sung_end = note["end"]
        if next_start is not None and next_start - sung_end < LEGATO_GAP:
            sung_end = next_start      # legato: hold right up to the next note

        if note["melisma"] or not note["word"]:
            # No new syllable: carry the vowel already being sung.
            _vowel_events(start, sung_end, held_vowel or "ə", level, events)
            continue

        onset, nucleus, coda = _split_syllable(_phones_for(note, cache))
        if not nucleus:
            # A syllable with no vowel at all (rare - a stray consonant):
            # let it fill the note rather than dropping the note silently.
            _spread(start, sung_end, onset or ["ə"], "coda", level, events)
            continue
        held_vowel = nucleus[-1]

        onset_total = len(onset) * ONSET_DUR
        coda_total = len(coda) * CODA_DUR
        # Short notes: shrink the consonants rather than the vowel, but never
        # past a quarter of their nominal length or they stop reading.
        span = sung_end - start + onset_total * ONSET_LEAD
        if onset_total + coda_total + MIN_VOWEL > span:
            room = max(0.0, span - MIN_VOWEL)
            scale = min(1.0, room / (onset_total + coda_total)) \
                if onset_total + coda_total else 1.0
            scale = max(scale, 0.25)
            onset_total *= scale
            coda_total *= scale

        # Singing ahead of the beat lets an onset bite into the tail of the
        # vowel before it, but not into that syllable's own coda consonants -
        # otherwise a legato "twin-kle" loses its /ŋ/ to the following /k/.
        onset_start = start - onset_total * ONSET_LEAD
        for previous in reversed(events):
            if previous["kind"] in ("onset", "coda"):
                onset_start = max(onset_start, previous["end"])
                break
            if previous["end"] < onset_start:
                break
        vowel_start = onset_start + onset_total
        coda_start = max(vowel_start + MIN_VOWEL * 0.5, sung_end - coda_total)
        coda_start = min(coda_start, sung_end)

        _spread(onset_start, vowel_start, onset, "onset", level, events)
        _vowel_events(vowel_start, coda_start, nucleus[0], level, events)
        for extra in nucleus[1:]:                 # e.g. the /i ɛ/ of "quiet"
            events.append(_event(coda_start, coda_start, extra, "vowel", level))
        _spread(coda_start, sung_end, coda, "coda", level, events)

    return _normalise(events)


def _normalise(events):
    """Trim overlaps, drop slivers, and close the mouth in every gap."""
    events.sort(key=lambda e: (e["start"], e["end"]))
    for a, b in zip(events, events[1:]):
        # An onset consonant is allowed to bite into the tail of the vowel
        # before it (that is what singing ahead of the beat means); a gap too
        # short to be worth closing the mouth for is absorbed instead.
        a["end"] = b["start"] if b["start"] - a["end"] < MIN_SIL \
            else min(a["end"], b["start"])
    events = [e for e in events if e["end"] - e["start"] > 1e-4]
    if not events:
        return events

    silence = mouth_shapes.SILENCE
    out = [_event(max(0.0, events[0]["start"] - LEAD_SIL), events[0]["start"],
                  silence, "silence", "medium")]
    for index, event in enumerate(events):
        out.append(event)
        nxt = events[index + 1]["start"] if index + 1 < len(events) else None
        if nxt is not None and nxt - event["end"] >= MIN_SIL:
            out.append(_event(event["end"], nxt, silence, "silence", "medium"))
    out.append(_event(events[-1]["end"], events[-1]["end"] + TAIL_SIL,
                      silence, "silence", "medium"))
    return [e for e in out if e["end"] - e["start"] > 1e-4]


def build_timeline(path):
    """The full timeline.json payload for a karaoke file."""
    song = kar.load_kar(path)
    events = plan_events(song)
    return {
        "source": os.path.basename(path),
        "title": song["title"],
        "duration": events[-1]["end"] if events else 0.0,
        "visemes": list(mouth_shapes.VISEME_ORDER),
        "levels": list(mouth_shapes.LEVELS),
        "event_count": len(events),
        "events": [
            {"start": round(e["start"], 5), "end": round(e["end"], 5),
             "phone": e["phone"], "viseme": e["viseme"], "level": e["level"],
             "kind": e["kind"]}
            for e in events],
        "notes": [
            {"start": round(n["start"], 5), "end": round(n["end"], 5),
             "note": n["note"], "velocity": n["velocity"],
             "syllable": n["syllable"], "word": n["word"],
             "melisma": n["melisma"]}
            for n in song["notes"]],
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def selftest():
    """Check the timing model's promises on the demo file."""
    here = os.path.dirname(os.path.abspath(__file__))
    demo = os.path.join(here, "twinkle_demo.kar")
    if not os.path.exists(demo):
        raise SystemExit(f"run make_demo_kar.py first: {demo} is missing")

    song = kar.load_kar(demo)
    events = plan_events(song)
    failures = []

    def check(label, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {label}{'' if ok else '  ' + detail}")
        if not ok:
            failures.append(label)

    # 1. Monotonic, gap-free, non-overlapping.
    gaps = [b["start"] - a["end"] for a, b in zip(events, events[1:])]
    check("events tile the timeline", all(abs(g) < 1e-6 for g in gaps),
          f"worst gap {max(map(abs, gaps)):.4f}s")

    # 2. Vowels dominate: they should hold most of the singing time.
    sung = [e for e in events if e["kind"] != "silence"]
    vowel_time = sum(e["end"] - e["start"] for e in sung
                     if e["kind"] in ("vowel", "glide"))
    total = sum(e["end"] - e["start"] for e in sung)
    check("vowels carry the notes", vowel_time / total > 0.75,
          f"{vowel_time / total:.0%} of sung time")

    # 3. Consonants stay quick.
    cons = [e["end"] - e["start"] for e in sung
            if e["kind"] in ("onset", "coda")]
    check("consonants stay quick", max(cons) <= ONSET_DUR + 1e-6,
          f"longest {max(cons):.3f}s")

    # 4. Onsets anticipate the beat: the vowel lands on (or just after) the
    #    note, the consonant before it.
    first = song["notes"][0]
    onset = next(e for e in events if e["kind"] == "onset")
    check("onsets lead the beat", onset["start"] < first["start"],
          f"onset {onset['start']:.3f} vs note {first['start']:.3f}")

    # 5. The mouth is closed during the rests between phrases.
    sil = [e for e in events if e["kind"] == "silence"]
    check("rests close the mouth", len(sil) >= len(kar.load_kar(demo)["notes"]) // 8,
          f"{len(sil)} silences")
    check("silence is the SIL viseme",
          all(e["viseme"] == "SIL" for e in sil))

    # 6. All three loudness variants are exercised by the demo's dynamics.
    levels = {e["level"] for e in sung}
    check("all three loudness variants used", levels == set(mouth_shapes.LEVELS),
          f"got {sorted(levels)}")

    # 7. The melisma note re-states the held vowel rather than dropping out.
    melisma = [n for n in song["notes"] if n["melisma"]]
    if melisma:
        during = [e for e in events
                  if e["start"] >= melisma[0]["start"] - 1e-6
                  and e["end"] <= melisma[0]["end"] + 0.2]
        check("melisma sustains a vowel",
              any(e["kind"] in ("vowel", "glide") for e in during))

    print("all passed" if not failures else f"{len(failures)} failure(s)")
    return len(failures)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kar_file", nargs="?", help=".kar or .mid with lyrics")
    parser.add_argument("-o", "--out", default="timeline.json",
                        help="Output JSON path (default: timeline.json)")
    parser.add_argument("--selftest", action="store_true",
                        help="Run timing-model checks on the demo and exit")
    parser.add_argument("--print", dest="show", action="store_true",
                        help="Print the planned events")
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(1 if selftest() else 0)
    if not args.kar_file:
        parser.error("kar_file is required unless --selftest is given")

    result = build_timeline(args.kar_file)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    if args.show:
        for event in result["events"]:
            print(f"  {event['start']:7.3f} -> {event['end']:7.3f}  "
                  f"{event['phone']:3s} {event['viseme']:4s} "
                  f"{event['level']:6s} {event['kind']}")

    sung = [e for e in result["events"] if e["kind"] != "silence"]
    print(f"{result['source']}: {len(result['notes'])} notes -> "
          f"{result['event_count']} mouth positions "
          f"({len(sung)} sung, {result['event_count'] - len(sung)} silent), "
          f"{result['duration']:.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
