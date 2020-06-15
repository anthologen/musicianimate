#!/usr/bin/env python3

import argparse
import os

from music21 import converter

def visualize_track(midi_file, track):
    """
    Show contents of the selected track number in the specified midi_file.
    If no track number is specified, the entire score is shown.
    show('musicxml') launches MuseScore which has playback capabilities.
    """
    score = converter.parse(midi_file)
    part_stream = list(score.parts.stream())
    print("Found the following tracks:")
    for idx, part in enumerate(part_stream):
        print(f"Part #{idx} (Instrument '{part.getInstrument()}')")

    if not track:
        print("Track not selected - visualizing entire score")
        score.show('musicxml')
    elif 0 <= track and track < len(part_stream):
        print(f"Track #{track} selected to be visualized")
        part_stream[track].show('musicxml')
    else:
        print(f"Unknown track #{track} - visualizing entire score")
        score.show('musicxml')

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)

if __name__ == "__main__":
    desc_string = (
        "Shows MIDI file contents in MuseScore."
        " Visualize a track if specified, otherwise show the whole score. \n"
        f"Example: {__file__} example.mid          -> show entire score\n"
        f"Example: {__file__} example.mid -track 2 -> show third track\n"
    )
    parser = argparse.ArgumentParser(description=desc_string,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('midi_file', type=file_path, help="MIDI file path")
    parser.add_argument('-track', type=int, help="Track number to examine")
    parsed_args = parser.parse_args()

    visualize_track(parsed_args.midi_file, parsed_args.track)
