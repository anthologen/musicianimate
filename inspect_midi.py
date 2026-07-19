#!/usr/bin/env python3
"""Quickly visualize a MIDI file as a piano roll.

Renders note events as horizontal bars (time on the x-axis, pitch on the
y-axis) using matplotlib - no MuseScore / music21 required. Notes are colored
by track, and a summary of every track is printed to the terminal.

Examples::

    ./inspect_midi.py demo.mid                # show the whole file
    ./inspect_midi.py demo.mid -track 2       # show only track 2
    ./inspect_midi.py demo.mid -o roll.png    # save to PNG (no window)
    ./inspect_midi.py demo.mid --fingering    # color by hand, label fingers
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from piano.piano_midi_animator import _read_vlq, _tick_to_seconds, parse_midi


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi_note):
    """Return a human-readable name like 'A0' or 'C#4' for a MIDI note number."""
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def parse_tracks(path):
    """Parse a MIDI file into per-track notes.

    Returns (ticks_per_beat, tempo_map, tracks) where ``tracks`` is a list of
    per-track note lists, each note a dict with tick timing already resolved to
    seconds: {track, note, velocity, start, end}.
    """
    import struct

    with open(path, "rb") as f:
        data = f.read()

    if data[0:4] != b"MThd":
        raise ValueError("Not a Standard MIDI File (missing MThd header)")
    header_len = struct.unpack(">I", data[4:8])[0]
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:8 + header_len])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    ticks_per_beat = division

    # First pass via the shared parser gives us the tempo map for tick->seconds.
    _tpb, tempo_map, _events = parse_midi(path)

    pos = 8 + header_len
    tracks = []
    for track_idx in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("Malformed track chunk")
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track_end = pos + 8 + track_len
        i = pos + 8
        tick = 0
        running_status = None
        open_notes = {}  # note -> [(on_tick, velocity), ...]
        notes = []
        while i < track_end:
            delta, i = _read_vlq(data, i)
            tick += delta
            status = data[i]
            if status < 0x80:
                status = running_status
            else:
                i += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:
                i += 1  # meta type
                length, i = _read_vlq(data, i)
                i += length
            elif status in (0xF0, 0xF7):
                length, i = _read_vlq(data, i)
                i += length
            else:
                event_type = status & 0xF0
                if event_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1, d2 = data[i], data[i + 1]
                    i += 2
                    if event_type == 0x90 and d2 > 0:
                        open_notes.setdefault(d1, []).append((tick, d2))
                    elif event_type in (0x80, 0x90):
                        stack = open_notes.get(d1)
                        if stack:
                            on_tick, vel = stack.pop(0)
                            notes.append({
                                "track": track_idx,
                                "note": d1,
                                "velocity": vel,
                                "start": _tick_to_seconds(on_tick, ticks_per_beat, tempo_map),
                                "end": _tick_to_seconds(tick, ticks_per_beat, tempo_map),
                            })
                elif event_type in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
        tracks.append(notes)
        pos = track_end

    return ticks_per_beat, tempo_map, tracks


HAND_COLORS = {"L": "tab:blue", "R": "tab:red"}


def visualize_fingering(midi_file, out_path=None):
    """Render a piano roll colored by hand, with finger numbers in the bars."""
    from piano.fingering import compute_fingering

    result = compute_fingering(midi_file)
    notes = result["notes"]
    if not notes:
        raise SystemExit("No notes to visualize.")
    by_hand = {"L": 0, "R": 0}
    for n in notes:
        by_hand[n["hand"]] += 1
    print(f"Fingered {len(notes)} notes (L: {by_hand['L']}, R: {by_hand['R']})")

    fig, ax = plt.subplots(figsize=(14, 7))
    for n in notes:
        width = max(n["end"] - n["start"], 0.02)
        ax.add_patch(Rectangle(
            (n["start"], n["midi"] - 0.4), width, 0.8,
            facecolor=HAND_COLORS[n["hand"]], edgecolor="white",
            linewidth=0.3, alpha=0.75,
        ))
        ax.text(n["start"] + width / 2, n["midi"], str(n["finger"]),
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold")

    _decorate_roll(ax, notes, midi_file, key="midi")
    ax.legend([Rectangle((0, 0), 1, 1, color=HAND_COLORS["L"]),
               Rectangle((0, 0), 1, 1, color=HAND_COLORS["R"])],
              ["Left hand", "Right hand"], loc="upper right", fontsize="small")
    _finish(fig, out_path)


def _decorate_roll(ax, notes, midi_file, key="note"):
    """Axis limits, C-labeled pitch ticks, and title for a piano roll."""
    pitches = [n[key] for n in notes]
    lo, hi = min(pitches), max(pitches)
    end = max(n["end"] for n in notes)
    ax.set_xlim(0, end * 1.02)
    ax.set_ylim(lo - 1, hi + 1)
    cs = [p for p in range(lo - 1, hi + 2) if p % 12 == 0]
    ax.set_yticks(cs)
    ax.set_yticklabels([note_name(p) for p in cs])
    ax.grid(axis="y", which="major", alpha=0.2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pitch")
    ax.set_title(os.path.basename(midi_file))


def _finish(fig, out_path):
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
        print(f"Saved piano roll to {out_path}")
    else:
        plt.show()


def visualize(midi_file, track=None, out_path=None):
    """Render the MIDI file (or a single track) as a piano roll."""
    _tpb, _tempo, tracks = parse_tracks(midi_file)

    print(f"Found {len(tracks)} track(s):")
    for idx, notes in enumerate(tracks):
        if notes:
            pitches = [n["note"] for n in notes]
            span = f"{note_name(min(pitches))}-{note_name(max(pitches))}"
        else:
            span = "empty"
        print(f"  Track #{idx}: {len(notes):4d} notes ({span})")

    if track is not None:
        if not (0 <= track < len(tracks)):
            raise SystemExit(f"Track #{track} out of range (0-{len(tracks) - 1})")
        selected = [(track, tracks[track])]
        print(f"\nVisualizing track #{track}")
    else:
        selected = list(enumerate(tracks))
        print("\nVisualizing all tracks")

    all_notes = [n for _idx, notes in selected for n in notes]
    if not all_notes:
        raise SystemExit("No notes to visualize.")

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, notes in selected:
        color = cmap(idx % 10)
        for n in notes:
            alpha = 0.35 + 0.65 * (n["velocity"] / 127.0)
            ax.add_patch(Rectangle(
                (n["start"], n["note"] - 0.4),
                max(n["end"] - n["start"], 0.02), 0.8,
                facecolor=color, edgecolor="none", alpha=alpha,
            ))

    _decorate_roll(ax, all_notes, midi_file, key="note")
    if track is not None:
        ax.set_title(f"{os.path.basename(midi_file)} - track {track}")

    if len(selected) > 1:
        handles = [Rectangle((0, 0), 1, 1, color=cmap(idx % 10))
                   for idx, notes in selected if notes]
        labels = [f"Track {idx}" for idx, notes in selected if notes]
        ax.legend(handles, labels, loc="upper right", fontsize="small")

    _finish(fig, out_path)


def file_path(path):
    if os.path.isfile(path):
        return path
    raise argparse.ArgumentTypeError(f"File not found: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("midi_file", type=file_path, help="MIDI file path")
    parser.add_argument("-track", type=int, help="Only show this track number")
    parser.add_argument("-o", "--out", help="Save to an image file instead of "
                                            "opening a window")
    parser.add_argument("--fingering", action="store_true",
                        help="Color notes by hand and label finger numbers "
                             "(computed by piano/fingering.py)")
    args = parser.parse_args()

    if args.fingering:
        if args.track is not None:
            parser.error("--fingering shows both hands; -track is not supported")
        visualize_fingering(args.midi_file, args.out)
    else:
        visualize(args.midi_file, args.track, args.out)
