# Piano Fingering: Research Notes

How to turn a MIDI file into a realistic piano fingering — which hand plays
each note and which finger (1 = thumb .. 5 = pinky) presses each key — so
the result can drive an animated hand rig. This is the background for
`piano/fingering.py`.

## The problem

Unlike guitar, a piano pitch is unambiguous: one key per note. The
difficulty is elsewhere. First, a raw MIDI file usually doesn't say which
hand plays what, so notes must be split between two hands that share the
keyboard and sometimes cross. Second, each hand's finger assignment is
combinatorial: a chord maps several simultaneous keys onto five fingers,
and melodic lines chain those choices over time — a fingering that is
comfortable now may force an impossible stretch three notes later, which is
why the choice must be optimized globally, not greedily. There is no single
"correct" answer: professional pianists agree with each other on only about
70% of notes, so ~60–65% agreement with any one human annotation is
effectively the ceiling for an algorithm — and for animation, *plausible
and physically consistent* matters more than matching one pianist's habits.

## The consensus method: ergonomic cost + dynamic programming

The literature converges on a minimum-cost path search over per-event
finger assignments:

- **[Parncutt, Sloboda, Clarke, Raekallio & Desain 1997, "An Ergonomic
  Model of Keyboard Fingering for Parsimonious Fingering of Melodic
  Fragments"](https://www.semanticscholar.org/paper/An-Ergonomic-Model-of-Keyboard-Fingering-for-Parncutt-Sloboda/1df73cbfa091a54f5760d4dab9bf508b03f1b98d)
  — the canonical model. Defines per-finger-pair **span tables** (minimum /
  comfortable / relaxed / practical interval each finger pair can cover, in
  semitones, with negative minima for crossings) plus ~12 weighted
  difficulty rules: stretch beyond comfort, weak fingers 4/5, thumb on a
  black key, position changes, thumb passing, same-finger repetition.
  Minimum-total-difficulty fingerings are found by dynamic programming.
  Later refined by Jacobs (1) toward modern pedagogical practice.
- **[Al Kasimi, Nichols & Raphael 2007, "A Simple Algorithm for Automatic
  Generation of Polyphonic Piano
  Fingerings"](https://www.semanticscholar.org/paper/A-Simple-Algorithm-for-Automatic-Generation-of-Kasimi-Nichols/2d1e519c087fc3ec9d4104427140697720052b0c)
  — extends the DP to chords by making each DP state a *complete
  finger-to-key assignment of the sounding chord*, with a vertical
  (within-chord) cost plus a horizontal (transition) cost between
  consecutive states. This is the shape our engine takes.
- **Balliauw, Herremans, Palhazi Cuervo & Sörensen** — variable
  neighborhood search over the same kind of ergonomic cost, including both
  hands; confirms the cost-model design but a Viterbi/beam DP is simpler
  and adequate at this scale.
- **[Zhao et al. 2022, pitch-difference fingering match model
  (EURASIP)](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-022-00237-8)**
  — emphasizes *playability* constraints (what a hand physically can do)
  over stylistic preference, a useful framing for animation work.

## Statistical and neural alternatives (not used)

- **[Nakamura, Saito & Yoshii 2020, "Statistical Learning and Estimation of
  Piano Fingering"](https://arxiv.org/abs/1904.10237)** with the **[PIG
  dataset](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/)**
  (150 classical pieces, fingerings annotated by pianists; Bach/Mozart/
  Chopin subsets have ≥4 annotators per piece). Second/third-order HMMs
  decode fingering at ~65% match to human annotations — the published
  state of the art, beating the DNN/LSTM baselines they compare against.
- **[Ramoneda et al. 2022, "Automatic Piano Fingering from Partially
  Annotated Scores using Autoregressive Neural
  Networks"](https://dl.acm.org/doi/10.1145/3503161.3548372)** — modern
  neural take; can complete partial human annotations.

Rejected here for the same reasons as on the guitar side: the PIG data is
licensed for academic research, learned models need training corpora and ML
dependencies in a deliberately stdlib-only repo, and the accuracy gap over
a tuned ergonomic DP (~60% vs ~65%, against a ~70% human ceiling) buys
little for animation, where a hand-tuned interpretable cost model is easier
to steer ("less thumb on black keys" is one weight).

## Hand separation

Most fingering literature assumes the score is already split into hands;
arbitrary MIDI isn't. Reference methods:

- **[Nakamura, Ono & Sagayama 2014, "Merged-Output HMM for Piano Fingering
  of Both Hands"
  (ISMIR)](https://eita-nakamura.github.io/articles/Nakamura_etal_MergedOutputHMMForPianoFingering_ISMIR2014.pdf)**
  — models the note stream as the merged output of two per-hand Markov
  processes and decodes hand attribution and fingering jointly; error rates
  1.4–18.7% depending on the piece.
- **[Hadjakos, Waloschek & Leemhuis 2019, "Detecting Hands in Piano MIDI
  Data"](http://www.cemfi.de/wp-content/papercite-data/pdf/hadjakos-2019-detectinghands.pdf)**
  — surveys rule-based, HMM, and neural hand-separation approaches.

Our engine borrows the *idea* (two hand processes competing for notes,
decoded globally) in simplified DP form — see below.

## Existing libraries (not used)

- **[pianoplayer (marcomusy)](https://github.com/marcomusy/pianoplayer)** —
  MIT, music21-based; minimizes finger-travel "effort" over a 5–9 note
  lookahead with hand-size presets. Heavy dependencies, raw-MIDI input is
  its weakest path, and we'd still need an adapter from score annotations
  back to a timed animation timeline.
- **[piano_fingering (Kanma)](https://github.com/Kanma/piano_fingering)** —
  lighter, same trade-offs.

## What the animation literature says about output format

Piano hand-motion synthesis systems — from **Handrix** (ElKoura & Singh,
SCA 2003) through **[PianoMotion10M 2024](https://arxiv.org/html/2406.09326v1)**,
**[FürElise (SIGGRAPH Asia 2024)](https://arxiv.org/abs/2410.05791)** and
**[Tipiano](https://arxiv.org/html/2604.09692v1)** — all share the pipeline
*fingering / fingertip targets → wrist trajectory → hand pose (IK or
learned)*. So a fingering engine for animation must emit more than
note→finger labels: it needs fingertip key targets in instrument-local
coordinates and a smooth per-hand wrist curve. That drove the design of
`fingering.json` (per-note `x/y/is_black` from `key_layout.py` plus 30 Hz
`wrist_x` curves), which `animate_hands.py` consumes.

## What `piano/fingering.py` implements

A dependency-free three-stage pipeline (the guitar engine in
`guitar/fingering.py` later mirrored this structure):

1. **Onset clustering** — notes starting within 30 ms form one chord event.
2. **Hand assignment** — Viterbi over per-event split points (bottom *k*
   notes → left hand), carrying each hand's last centroid in the state.
   Costs: span beyond a hand's reach, centroid movement scaled by urgency
   (`_time_factor`), crossed hands, a weak pull of LH below / RH above
   middle C, and an **activation cost** for engaging an idle hand — the
   term that stops the DP from splitting one-hand material (an octave, a
   scale) across two conveniently-placed hands. Placing both hands before
   the first event is free. A `hands_from_tracks` mode trusts a two-track
   MIDI instead.
3. **Per-hand fingering** — Al Kasimi-style Viterbi where a state is a
   strictly-monotonic finger assignment of the sounding chord (left hand
   handled by mirroring pitches, `q = −midi`, so one orientation of the
   span tables serves both hands). Chord feasibility and stretch costs come
   from the Parncutt span tables scaled by a hand-size preset (XXS–XXL);
   within-chord costs add weak-finger and thumb/pinky-on-black penalties;
   transition costs cover hand repositioning (thumb-anchor position via
   `FINGER_OFFSET`), thumb passing (extra onto a black key), same-finger
   restrikes, and finger substitution on held notes (allowed, heavily
   penalized). Fingers holding sustained notes are pinned; impossible
   sustain pileups (>5 sounding notes) release the oldest holds. Exact
   Viterbi with a beam of 200 as a safety valve.

Weights live in `WEIGHTS` and were hand-tuned against golden selftest
cases (`python -m piano.fingering --selftest`): C-major scale must produce
1-2-3-1-2-3-4-5 in both hands, C-E-G → 1-3-5, octave → 1-5 in one hand,
two-octave runs must cross thumb-under, and a bass line + melody must split
cleanly. Debug visualization: `./inspect_midi.py <file> --fingering`.

## Out of scope (future work)

- **Learned cost weights** from the PIG data (would need its academic
  license and a fitting step; the DP structure would not change).
- **Interdependence of the hands** beyond splitting (voice sharing,
  hand-over-hand passages) and phrase-boundary awareness — the known gaps
  Nakamura et al. list for all current models.
- Deliberate **finger substitution on held notes** as a technique (organ
  style) — currently only an emergency with a heavy penalty.
- Pedaling (would relax sustain pinning), ornaments, glissandi, and
  fingered octaves/double notes idioms (1-4/1-5 alternation in fast
  octave passages).
- Style profiles (editions differ: Czerny vs Chopin fingering habits could
  be alternate `WEIGHTS` presets).
