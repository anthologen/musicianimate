"""Generate a short demo karaoke file to exercise the singer pipeline.

Writes a genuine Soft Karaoke (.kar) file: a format-1 SMF whose first track
carries the ``@``-tagged header and the lyric syllables as text meta events,
and whose second track carries the sung melody.

The song is "Twinkle Twinkle Little Star", chosen because four short phrases
cover everything the mouth animator has to handle:

    phrase 1  medium   "Twin-kle twin-kle lit-tle star"   consonant clusters
                                                          (tw-, st-) and the
                                                          -nkl- coda
    phrase 2  soft     "How I won-der what you are"        diphthongs (aʊ, aɪ)
    phrase 3  loud     "Up a-bove the world so high"       the same visemes at
                                                          full volume
    phrase 4  medium   "Like a dia-mond in the sky"        a two-note MELISMA
                                                          on "sky", and a word
                                                          ("dia-mond") whose
                                                          lyric syllabification
                                                          disagrees with the
                                                          dictionary's

Rests of a beat or more between phrases give the mouth time to close, which is
the other thing worth seeing in the animation.
"""

import struct
import sys


def write_kar(path, ticks_per_beat, tempo_us_per_beat, notes, lyrics,
              title="Untitled", channel=0):
    """notes: [(start_tick, duration_tick, note, velocity)].
    lyrics: [(tick, syllable_text)] - Soft Karaoke conventions apply, i.e. a
    leading space starts a new word and a leading "\\" starts a new line."""

    def vlq(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(out)

    def meta(track, delta, meta_type, payload):
        track += vlq(delta) + bytes([0xFF, meta_type]) + vlq(len(payload))
        return track + payload

    # Track 0: tempo, karaoke header, lyrics.
    words = bytearray()
    words = meta(words, 0, 0x03, b"Soft Karaoke")
    words = meta(words, 0, 0x51, struct.pack(">I", tempo_us_per_beat)[1:])
    for tag in (b"@KMIDI KARAOKE FILE", b"@V0100",
                b"@T" + title.encode("latin-1"), b"@Isinger/make_demo_kar.py"):
        words = meta(words, 0, 0x01, tag)
    last = 0
    for tick, text in sorted(lyrics):
        words = meta(words, tick - last, 0x01, text.encode("latin-1"))
        last = tick
    words += vlq(0) + bytes([0xFF, 0x2F, 0x00])

    # Track 1: the melody.
    events = []
    for start, dur, note, vel in notes:
        events.append((start, "on", note, vel))
        events.append((start + dur, "off", note, 0))
    events.sort(key=lambda e: (e[0], e[1] == "on"))

    melody = bytearray()
    melody = meta(melody, 0, 0x03, b"Melody")
    last = 0
    for tick, etype, note, vel in events:
        melody += vlq(tick - last)
        last = tick
        melody += bytes([(0x90 if etype == "on" else 0x80) | channel, note, vel])
    melody += vlq(0) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, ticks_per_beat)
    with open(path, "wb") as f:
        f.write(header)
        for track in (words, melody):
            f.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))


# ---------------------------------------------------------------------------
# The song.  Each phrase is (velocity, [(syllable, midi, beats)]); a syllable
# of None is a melisma - another note on the vowel already being sung.
# ---------------------------------------------------------------------------
PHRASES = [
    (80, [("Twin", 60, 1), ("kle", 60, 1), (" twin", 67, 1), ("kle", 67, 1),
          (" lit", 69, 1), ("tle", 69, 1), (" star", 67, 2)]),
    (45, [("How", 65, 1), (" I", 65, 1), (" won", 64, 1), ("der", 64, 1),
          (" what", 62, 1), (" you", 62, 1), (" are", 60, 2)]),
    (110, [("Up", 67, 1), (" a", 67, 1), ("bove", 65, 1), (" the", 65, 1),
           (" world", 64, 1), (" so", 64, 1), (" high", 62, 2)]),
    (88, [("Like", 67, 1), (" a", 67, 1), (" dia", 65, 1), ("mond", 65, 1),
          (" in", 64, 1), (" the", 64, 1), (" sky", 62, 1), (None, 60, 1)]),
]

REST_BEATS = 1.5      # silence between phrases, so the mouth closes
LEAD_IN_BEATS = 1.0


def build_demo(tpb):
    notes, lyrics = [], []
    beat = LEAD_IN_BEATS
    for phrase_index, (velocity, syllables) in enumerate(PHRASES):
        for index, (text, midi, beats) in enumerate(syllables):
            tick = int(round(beat * tpb))
            notes.append((tick, int(beats * tpb * 0.92), midi, velocity))
            if text is not None:
                prefix = "\\" if index == 0 and phrase_index else ""
                lyrics.append((tick, prefix + text))
            beat += beats
        beat += REST_BEATS
    total_ticks = int(round(beat * tpb))
    return notes, lyrics, total_ticks


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "twinkle_demo.kar"
    TPB = 480
    TEMPO_US = 545455  # 110 BPM - a comfortable singing tempo

    demo_notes, demo_lyrics, total_ticks = build_demo(TPB)
    write_kar(out_path, TPB, TEMPO_US, demo_notes, demo_lyrics,
              title="Twinkle Twinkle Little Star")

    seconds = total_ticks / TPB * (TEMPO_US / 1_000_000.0)
    print(f"Wrote {out_path}: {len(demo_notes)} notes, "
          f"{len(demo_lyrics)} syllables, ~{seconds:.2f}s")
