# singer

Turns lyrics into mouth animation: text → IPA phonemes, karaoke MIDI (`.kar`)
→ a timed phoneme track, and a 2D vector mouth in Blender that interpolates
between one mouth position and the next.

Deliberately **2D for now**. The mouth is a flat card of parametric outline
geometry, so the phonetics, the viseme set and the timing model can be worked
out before any of it is spent on a 3D head. Everything except the two Blender
scripts is plain, dependency-free Python.

## Pipeline

```
your_song.kar  (MIDI + lyric meta events)
    │
    ▼
python -m singer.timeline song.kar -o timeline.json   # outside Blender
    │      kar.py      pairs each lyric syllable with the note it is sung on
    │      g2p.py      turns each word into IPA phonemes
    │      timeline.py lays those phonemes out in time
    ▼
build_mouth.build_mouth()                     # mouth mesh + 45 shape keys  ┐
animate_mouth.animate_mouth("timeline.json")  # crossfades the shape keys   ┘ inside Blender
    │
    ▼
bpy.ops.wm.save_as_mainfile(filepath="animated_mouth.blend")
```

`build_mouth.py` and `animate_mouth.py` need `bpy` and must run inside Blender
(Scripting tab, Text Editor "Run Script", or the Blender MCP
`execute_blender_code` tool). Everything else runs as ordinary Python.

## Files

| File | Role |
|---|---|
| `mouth_shapes.py` | bpy-free source of truth: the 15 visemes, their three loudness variants, the IPA→viseme map, and the parametric 2D mouth outline. The only file to touch to restyle the mouth. |
| `g2p.py` | English text → IPA. A ~300-word pronunciation lexicon over an ordered, context-sensitive letter-to-sound rule set (NRL style). Optional CMUdict drop-in for higher accuracy. |
| `kar.py` | Reads `.kar` / `.mid` lyric meta events, joins syllables into words, picks the melody track by agreement with the lyric ticks, and pairs each syllable with its note. |
| `timeline.py` | The timing planner (the singer's `fingering.py`): consonants quick and at the note edges, vowels holding the note, late diphthong glides, melisma, mouth closed in the rests. Emits `timeline.json`. |
| `build_mouth.py` | Builds the `Mouth` collection in Blender: one flat mesh with a shape key per viseme per loudness (45), plus a face-card backdrop, ortho camera and flat vector-art materials. |
| `animate_mouth.py` | Keyframes the shape keys from `timeline.json` as a chain of crossfades whose weights always sum to 1. |
| `make_demo_kar.py` | Writes `twinkle_demo.kar`, a Soft Karaoke test file covering consonant clusters, diphthongs, all three dynamics and a melisma. |
| `RESEARCH.md` | Design notes: viseme choice, the g2p approach, and the sung-timing model. |

## Quick start

```bash
# 1. Generate a demo karaoke file (or point at your own .kar)
python -m singer.make_demo_kar singer/twinkle_demo.kar

# 2. Plan the phoneme timeline (outside Blender)
python -m singer.timeline singer/twinkle_demo.kar -o singer/timeline.json

# 3. Inspect what it will sing
python -m singer.g2p "Twinkle twinkle little star"
python -m singer.kar singer/twinkle_demo.kar
python -m singer.timeline singer/twinkle_demo.kar -o /tmp/t.json --print
```

Then, inside Blender:

```python
import sys; sys.path.insert(0, "/path/to/musicianimate")
import singer.build_mouth as bm, singer.animate_mouth as am
bm.build_mouth()
am.animate_mouth("/path/to/musicianimate/singer/timeline.json", fps=24)
```

## Checks

```bash
python -m singer.g2p --selftest        # lexicon, rules, inflections, syllables
python -m singer.timeline --selftest   # the timing model's promises
python -m singer.mouth_shapes          # every viseme's aperture and width
```

Inside Blender, `animate_mouth.check_weights()` reports the worst deviation of
the summed shape-key weights from 1.0 across the take. It should be 0 — see
"Crossfade" below.

## How it works

### Visemes, not phonemes

A camera cannot tell /p/ from /b/ from /m/, so the 40-odd English phonemes
collapse to 15 **visemes**. Each viseme is a set of parameters (aperture,
width, rounding, upper lift, visible teeth, tongue height, lower-edge
tuck) that `mouth_shapes.outline()` turns into a mouth outline. There are no
lips: the mouth is a flat opening cut straight into the face.

Each viseme has three loudness variants — soft / medium / loud — chosen from
the note's MIDI velocity. Loudness multiplies the aperture rather than adding
to it, so a sealed consonant stays sealed however loudly it is sung.

### Consonants are quick, vowels carry the note

This is what makes it read as singing rather than speech:

* Onset consonants take a fixed ~55 ms each and are placed so the **vowel**
  lands on the beat and the consonants precede it.
* The vowel holds everything left of the note.
* Coda consonants take the last few tens of milliseconds.
* Diphthongs glide late: "high" is a long /ɑ/ with the /ɪ/ tucked into the
  end of the note.
* A note with no syllable of its own (melisma) carries the previous vowel
  onward.
* Any gap longer than 100 ms becomes an explicit silence and the mouth shuts.

On the demo, vowels take 86% of the sung time.

### Crossfade

Every event ramps its own shape key 0 → 1 across its opening boundary, holds
at 1, and ramps 1 → 0 across its closing boundary. Both events sharing a
boundary use the *same* ramp width, so their weights always sum to 1 — and
because Blender evaluates shape keys as `basis + Σ weight × offset`, weights
summing to 1 are exactly the interpolation between those two mouth positions.
Nothing else is ever keyed, so the mouth can never drift toward a shape that
is not a viseme.

## Scope

Handles: English lyrics, `.kar` and lyric-bearing `.mid`, both karaoke
word-boundary conventions (leading space and trailing hyphen), melisma,
per-note dynamics.

Not yet: languages other than English (the viseme set is largely
language-neutral; `g2p.py` is not), coarticulation between neighbouring
visemes, jaw/head motion, breath, a 3D head.
