"""Read a karaoke MIDI file (.kar, or a .mid with lyric events) and pair each
lyric syllable with the note it is sung on.

A .kar file is an ordinary Standard MIDI File with the words carried in meta
events.  Two conventions exist and both are handled here:

* **Lyric meta events (0xFF 0x05)** - the General MIDI way.
* **Text meta events (0xFF 0x01)** - the older Soft Karaoke way, where the
  first track also carries ``@KMIDI KARAOKE FILE``, ``@T`` title lines and
  other ``@``-prefixed tags that are information, not words.

Within either convention one event is one *syllable*, and syllables are joined
into words by markers: a leading space starts a new word (Soft Karaoke), or a
trailing ``-`` says "this syllable continues into the next" (the hyphen
style).  ``\\`` and ``/`` at the start of a syllable are line/paragraph breaks.
Which convention a file uses is detected from the file itself.

The lyrics usually live on their own track with no notes, so the melody track
is chosen by *agreement*: the note track whose onsets line up with the most
lyric events wins.  Notes that get no syllable are melisma - the same vowel
carried on over more notes - and are marked as such for the timing planner.

Usage::

    python -m singer.kar path/to/song.kar
"""

import os
import struct
import sys

try:
    from piano.piano_midi_animator import _tick_to_seconds
except ImportError:  # loaded as a loose script, not a package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from piano.piano_midi_animator import _tick_to_seconds

# A lyric event may sit a few ticks either side of its note-on; anything
# within this fraction of a beat is considered the same moment.
ALIGN_BEATS = 0.30


# ---------------------------------------------------------------------------
# SMF reading (lyrics included - piano's parse_midi keeps only notes + tempo)
# ---------------------------------------------------------------------------
def _read_vlq(data, i):
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, i


def parse_smf(path):
    """Return {ticks_per_beat, tempo_map, tracks}, tracks being a list of
    {name, notes, lyrics, texts} with ticks left in MIDI ticks."""
    with open(path, "rb") as f:
        data = f.read()

    if data[0:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd header)")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:8 + header_len])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")

    pos = 8 + header_len
    tempo_map = []
    tracks = []

    for _track_idx in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("Malformed track chunk")
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track_end = pos + 8 + track_len
        i = pos + 8
        tick = 0
        running_status = None
        track = {"name": "", "notes": [], "lyrics": [], "texts": []}
        sounding = {}

        while i < track_end:
            delta, i = _read_vlq(data, i)
            tick += delta
            status = data[i]
            if status < 0x80:
                status = running_status  # running status: byte already consumed
            else:
                i += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                meta_type = data[i]
                i += 1
                length, i = _read_vlq(data, i)
                meta = data[i:i + length]
                i += length
                if meta_type == 0x51:                       # Set Tempo
                    tempo_map.append((tick, struct.unpack(">I", b"\x00" + meta)[0]))
                elif meta_type == 0x03:                     # Track Name
                    track["name"] = meta.decode("latin-1")
                elif meta_type == 0x01:                     # Text
                    track["texts"].append((tick, meta.decode("latin-1")))
                elif meta_type == 0x05:                     # Lyric
                    track["lyrics"].append((tick, meta.decode("latin-1")))
            elif status in (0xF0, 0xF7):
                length, i = _read_vlq(data, i)
                i += length
            else:
                event_type = status & 0xF0
                if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1, d2 = data[i], data[i + 1]
                    i += 2
                    if event_type == 0x90 and d2 > 0:
                        sounding.setdefault(d1, []).append((tick, d2))
                    elif event_type in (0x80, 0x90):
                        stack = sounding.get(d1)
                        if stack:
                            start, vel = stack.pop(0)
                            track["notes"].append({"tick": start, "end_tick": tick,
                                                   "note": d1, "velocity": vel})
                elif event_type in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1

        for note, stack in sounding.items():                # unterminated notes
            for start, vel in stack:
                track["notes"].append({"tick": start, "end_tick": tick,
                                       "note": note, "velocity": vel})
        track["notes"].sort(key=lambda n: (n["tick"], n["note"]))
        tracks.append(track)
        pos = track_end

    if not tempo_map:
        tempo_map = [(0, 500000)]  # default 120 BPM
    tempo_map.sort(key=lambda t: t[0])
    return {"ticks_per_beat": division, "tempo_map": tempo_map,
            "tracks": tracks}


# ---------------------------------------------------------------------------
# Lyric extraction
# ---------------------------------------------------------------------------
def _collect_lyric_events(tracks):
    """[(tick, raw_text)] from whichever meta channel this file uses."""
    events = []
    for track in tracks:
        events.extend(track["lyrics"])
    if events:
        return sorted(events)
    # Soft Karaoke: words live in text events, mixed with @-tagged info lines.
    for track in tracks:
        events.extend((t, s) for t, s in track["texts"]
                      if not s.startswith("@"))
    return sorted(events)


def _title(tracks):
    for track in tracks:
        for _tick, text in track["texts"]:
            if text.startswith("@T"):
                return text[2:].strip()
    for track in tracks:
        if track["name"] and not track["name"].startswith("@"):
            return track["name"]
    return ""


def split_syllables(events):
    """[(tick, syllable, new_word, new_line)] from raw lyric events.

    Detects which word-boundary convention the file uses before splitting, so
    a leading-space file and a trailing-hyphen file both come out right.
    """
    cleaned = []
    for tick, raw in events:
        text = raw.replace("\r", "\n")
        new_line = False
        while text[:1] in ("\\", "/", "\n"):
            new_line = True
            text = text[1:]
        if not text.strip():
            continue
        cleaned.append((tick, text, new_line))

    uses_space = any(t.startswith(" ") for _k, t, _n in cleaned[1:])
    uses_hyphen = any(t.rstrip().endswith("-") for _k, t, _n in cleaned)

    out = []
    continues = False
    for index, (tick, text, new_line) in enumerate(cleaned):
        leading_space = text.startswith(" ")
        syllable = text.strip()
        hyphenated = syllable.endswith("-")
        if hyphenated:
            syllable = syllable[:-1]
        if index == 0 or new_line:
            new_word = True
        elif uses_space:
            new_word = leading_space
        elif uses_hyphen:
            new_word = not continues
        else:
            new_word = True
        out.append((tick, syllable, new_word, new_line))
        continues = hyphenated if uses_hyphen else False
    return out


# ---------------------------------------------------------------------------
# Pairing syllables with notes
# ---------------------------------------------------------------------------
def _pick_melody_track(tracks, lyric_ticks, tolerance):
    """The note track whose onsets agree best with the lyric ticks."""
    best, best_score = None, -1.0
    for track in tracks:
        if not track["notes"]:
            continue
        onsets = sorted({n["tick"] for n in track["notes"]})
        hits = 0
        for tick in lyric_ticks:
            lo = min(onsets, key=lambda o: abs(o - tick))
            hits += abs(lo - tick) <= tolerance
        score = hits / len(lyric_ticks) if lyric_ticks else 0.0
        # Ties (or no lyrics at all) go to the busier track.
        score += 1e-6 * len(onsets)
        if score > best_score:
            best, best_score = track, score
    return best


def load_kar(path):
    """Parse a .kar/.mid and return the sung line as timed, worded notes.

    Each note is {start, end, note, velocity, syllable, word, syllable_index,
    syllable_count, line, melisma}; ``syllable`` is None for a melisma note,
    which carries the previous syllable's vowel onward.
    """
    smf = parse_smf(path)
    tpb, tempo_map, tracks = (smf["ticks_per_beat"], smf["tempo_map"],
                              smf["tracks"])
    tolerance = ALIGN_BEATS * tpb

    events = _collect_lyric_events(tracks)
    syllables = split_syllables(events)
    lyric_ticks = [s[0] for s in syllables]
    melody = _pick_melody_track(tracks, lyric_ticks, tolerance)
    if melody is None:
        raise ValueError(f"{os.path.basename(path)} contains no notes")

    notes = [dict(n) for n in melody["notes"]]
    # One syllable per note: walk both in time order and consume the nearest
    # unclaimed syllable.  A note that claims none is a melisma.
    for note in notes:
        note["syllable"] = None
    used = [False] * len(syllables)
    for note in notes:
        best, best_gap = None, tolerance + 1
        for index, (tick, _syl, _nw, _nl) in enumerate(syllables):
            if used[index]:
                continue
            gap = abs(tick - note["tick"])
            if gap < best_gap:
                best, best_gap = index, gap
        if best is not None and best_gap <= tolerance:
            used[best] = True
            note["syllable_event"] = best

    # Words: a run of syllables between "new word" markers.
    word_of = []
    words = []
    for tick, syllable, new_word, _new_line in syllables:
        if new_word or not words:
            words.append([])
        words[-1].append(syllable)
        word_of.append(len(words) - 1)
    line_of, line = [], 0
    for _tick, _syllable, _new_word, new_line in syllables:
        line += new_line
        line_of.append(line)

    out = []
    for note in notes:
        index = note.get("syllable_event")
        start = _tick_to_seconds(note["tick"], tpb, tempo_map)
        end = _tick_to_seconds(note["end_tick"], tpb, tempo_map)
        entry = {"start": start, "end": end, "note": note["note"],
                 "velocity": note["velocity"], "syllable": None, "word": None,
                 "syllable_index": 0, "syllable_count": 1, "line": 0,
                 "melisma": index is None}
        if index is not None:
            word_index = word_of[index]
            word_syllables = words[word_index]
            first = word_of.index(word_index)
            entry.update(syllable=syllables[index][1],
                         word="".join(word_syllables),
                         syllable_index=index - first,
                         syllable_count=len(word_syllables),
                         line=line_of[index])
        elif out:
            entry["line"] = out[-1]["line"]
        out.append(entry)

    out.sort(key=lambda n: (n["start"], n["note"]))
    return {"path": path, "title": _title(tracks),
            "track": melody["name"] or "?", "notes": out,
            "syllable_count": len(syllables)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        sys.exit(1)
    song = load_kar(sys.argv[1])
    print(f"{os.path.basename(song['path'])}: {song['title'] or '(untitled)'} "
          f"- melody track {song['track']!r}, {len(song['notes'])} notes, "
          f"{song['syllable_count']} syllables")
    for note in song["notes"]:
        label = note["syllable"] if not note["melisma"] else "(melisma)"
        print(f"  {note['start']:7.3f}s -> {note['end']:7.3f}s  "
              f"midi {note['note']:3d} vel {note['velocity']:3d}  {label}")
