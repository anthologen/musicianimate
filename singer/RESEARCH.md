# Singer research notes

Background for the design decisions in `singer/`. Three problems had to be
settled before any code: which mouth positions to model, how to get phonemes
out of English text without a dependency, and how sung timing differs from
spoken timing.

## 1. Visemes

A **viseme** is the visual equivalent of a phoneme: the set of sounds that
look the same from outside the mouth. The mapping is many-to-one and lossy on
purpose — /p/, /b/ and /m/ are one closed-lips position, and no amount of
animation detail will distinguish them, because the difference is voicing and
nasal airflow, neither of which is visible.

Standard viseme sets range from Preston Blair's classic 10 (hand-drawn
animation) through Disney's 12 to the ~20 used in modern lip-sync systems.
The choice here is 15, sized to what the parameters can actually express:

| Viseme | Phonemes (IPA) | Distinguishing feature |
|---|---|---|
| SIL | (silence) | lips closed, at rest |
| MBP | p b m | lips closed, slightly pressed |
| FV | f v | lower lip tucked under the upper teeth |
| TH | θ ð | tongue between the teeth |
| DD | t d n l | tongue tip up, small aperture |
| SS | s z | teeth nearly together, corners spread |
| CH | ʃ ʒ tʃ dʒ | pursed and pushed forward |
| KK | k g ŋ h | neutral open, tongue back |
| RR | ɹ ɝ ɚ | slightly rounded, tongue bunched |
| WW | w u ʊ | tightly rounded |
| OH | ɔ o | rounded, mid-open |
| AA | ɑ ɒ | wide open jaw |
| AE | æ | open and spread |
| EH | ɛ ʌ ə | mid-open neutral |
| IY | i ɪ e j | narrow and spread |

Two design notes:

* **The vowels are the ones worth spending resolution on.** They are held
  long enough to be seen, while a consonant is gone in 50 ms. That is why
  6 of the 15 are vowels while all the stops share KK/DD/MBP.
* **Diphthongs are two visemes, not one.** /aɪ/ is AA gliding into IY. They
  are stored in `DIPHTHONGS` and split by the timing planner, not given
  positions of their own.

### Loudness variants

Singing louder drops the jaw further and opens the aperture: the same vowel at
*pp* and at *ff* is visibly a different mouth. Each viseme therefore has three
variants keyed off MIDI velocity (< 55 soft, < 95 medium, else loud).

The gain **multiplies** the aperture rather than adding to it, which has one
important consequence: a closed consonant (aperture 0) stays closed at every
dynamic. Loud singing does not unseal an /m/. Corner spread and rounding are
articulatory rather than dynamic, so they move much less (a 0.65 + 0.35·gain
blend), and visible teeth scale mildly with the jaw drop.

### Why a mesh, not a curve

"2D vector mouth" suggests a Bezier curve object. Shape keys are the reason
not to: the animation is *interpolation between mouth positions*, and Blender
evaluates a shape-keyed mesh as `basis + Σ weight × offset`. Two keys at
weights summing to 1 therefore give exactly the linear blend of those two
shapes — precise, cheap and impossible to get wrong. Curve control points
have no equivalent. So the outline is generated as a fixed-topology polygon
mesh (140 vertices) and every viseme × loudness is one absolute shape key.

The parts (lip band, dark cavity, upper teeth, tongue) are stacked a few
millimetres apart along the camera axis for draw order. Two geometry rules
earned themselves comments in the source:

* The **tongue's lower edge is the lower aperture edge**, not a free ellipse.
  A free ellipse spills below the lip when raised; making it the floor of the
  mouth means it can never leave the aperture.
* The **teeth taper to nothing at the corners** and their top edge is tucked
  just inside the upper lip (including under the cupid's bow). Without the
  taper the teeth read as a white rectangle pasted over the mouth.

## 2. Grapheme-to-phoneme

The repo is deliberately dependency-free, which rules out `g2p_en`,
`phonemizer` and espeak. Two layers replace them:

**A lexicon.** English orthography is irregular exactly where it is frequent
— *the, of, one, love, heart, said, women*. About 300 entries, weighted
toward song vocabulary, cover most of the lyrics anyone will feed this.
Regular inflections of a lexicon stem (`dream` → `dreaming`, `dreams`) are
derived, with the -s/-ed voicing rule applied from the stem's final phone.

**Letter-to-sound rules** for everything else: an ordered, context-sensitive
rule set in the style of Elovitz et al., *Automatic Translation of English
Text to Phonetics by Means of Letter-to-Sound Rules* (NRL, 1976) — the
approach every rule-based English TTS front end has used since. Each rule is
`left_context | letters | right_context -> phones`, contexts written with
class wildcards (`#` vowels, `^` a consonant, `%` a suffix, `$` a word
boundary, ...), first match wins.

Rule ordering is the whole game, and three orderings cost real debugging time:

* **Magic-e outranks the generic vowel rules.** `|i|^%` (i, one consonant, a
  suffix) must be tried before `#:^|i|^+`, or *rise* and *shine* come out with
  /ɪ/.
* **A closed syllable outranks the consonant-class rules.** `|u|^^ -> ʌ` must
  precede `@|u| -> u`, or *sunrise* starts with /su/.
* **Word-initial unstressed `a-`** (`$|a|^# -> ə`) is worth a rule of its own:
  *about, above, away, alone, amaze* all take schwa. It misfires on *agent*
  and *acorn*, which is the right trade at this frequency — and both are one
  lexicon line away from being fixed.

The accuracy path is deliberately not "more rules": drop a CMUdict file next
to the module and call `load_cmudict()`. The lexicon grows, the rules stay as
the out-of-vocabulary fallback, nothing else in the pipeline changes.

**Syllables.** Lyrics are already hyphenated by the karaoke file, and that
hyphenation is authoritative — *dia-mond* is two notes even though the
dictionary says three. So `split_into(phones, n)` forces the phone list into
exactly the number of syllables the lyric asked for, using the sonority-based
syllabifier (maximal onset principle, restricted to legal English onsets)
only to decide *where* the consonants fall.

## 3. Sung timing

The interesting part, and the difference between this looking like singing
and looking like a talking head.

**Vowels carry the note.** In speech a syllable's phonemes get roughly
comparable durations. In singing the vowel is the note: it is the only
phoneme whose length scales with the melody, and it absorbs essentially all
of a whole note. On the demo, vowels take 86% of the sung time.

**Consonants are quick and live at the edges.** A consonant takes about the
same 50–60 ms whether the note is a semiquaver or a semibreve. Their placement
matters more than their length:

* **Onsets anticipate the beat.** Singers articulate *ahead* of the beat so
  that the vowel — the part that is actually pitched — lands on it. A sung
  "star" has its /s/ and /t/ before the downbeat, not on it. `ONSET_LEAD =
  0.70` puts 70% of the onset before the note starts.
* **Codas close the note**, taking its last few tens of milliseconds.
* An onset may bite into the tail of the vowel before it, but **not into that
  syllable's own coda**. Without this rule a legato "twin-kle" loses its /ŋ/
  to the following /k/ — the two consonants compete for the same boundary and
  the coda gets trimmed to 11 ms, which at 24 fps is a quarter of a frame.

**Diphthongs glide late.** /aɪ/ in "high" is not two equal halves; it is a
long /ɑ/ with the /ɪ/ tucked into the end of the note (30% of the vowel,
capped at 140 ms). Singers hold the first element and glide at the last
moment, because the first element is the one that sustains.

**Melisma.** A note with no syllable of its own carries the previous vowel
onward. It is re-stated rather than merely held, so its own velocity can
change the loudness variant — a swelling held note visibly opens up.

**Legato.** Notes closer together than 120 ms are sung joined-up: the first
note's sung end is extended to the second's start rather than closing and
reopening.

**Silence.** Any gap longer than 100 ms becomes an explicit SIL event and the
mouth shuts. Shorter gaps are absorbed into the neighbouring event — closing
the mouth for 60 ms reads as a flicker, not as a breath.

## 4. The crossfade

Each event ramps its shape key 0 → 1 across its opening boundary, holds at 1,
and ramps 1 → 0 across its closing boundary; the ramp half-width is 40% of
the shorter adjacent event, capped at 60 ms, so a held vowel eases and a
50 ms consonant snaps.

The invariant that makes this correct: **both events sharing a boundary use
the same ramp width**, so their weights sum to 1 throughout the transition,
and `basis + Σ weight × offset` is then exactly the interpolation between the
two mouth positions. If the sum ever drifted below 1 the mouth would sag
toward the closed basis shape mid-transition, which reads as a flicker.

Handle type matters here. Auto-clamped Bezier handles on a monotone key pair
produce an S-curve and its exact complement, so the pair still sums to 1 —
smooth *and* exact. `animate_mouth.check_weights()` verifies it across the
whole take; on the demo the worst deviation is 0.00000.

One case needs a pre-pass: two neighbouring events that want the *same*
viseme and loudness (a coda /t/ followed by an onset /d/, or a melisma
restating its vowel at the same dynamic) would both keyframe the same shape
key at the shared boundary — once to 0 and once to 1, on the same frame.
`_merge_runs()` collapses them into one held position, which is what they are
musically anyway. On the demo this merges 86 events into 78.

## Possible next steps

* Coarticulation: blend a viseme toward its neighbours (a /s/ before /u/ is
  already rounded), which is the single biggest remaining realism gap.
* Jaw and head motion driven by pitch and dynamics; breath intakes in the
  rests.
* A 3D head, at which point `mouth_shapes.py` becomes a driver for a real
  mouth rig rather than an outline generator.
* Other languages: the viseme set is largely language-neutral, `g2p.py` is
  not. A second g2p module per language, same interface.
* Vibrato and sustained-note motion; consonant-cluster simplification for
  very fast passages.
