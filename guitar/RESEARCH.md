# Guitar Fingering: Research Notes

How to turn a MIDI file into a realistic guitar tab — which string and fret
sounds each note, and which left-hand finger presses it — so the result can
drive an animated hand rig. This is the background for `guitar/fingering.py`.

## The problem

Unlike a piano key, a guitar pitch is ambiguous: most notes can be played at
1–6 different (string, fret) positions (e.g. E4 = open 1st string, 5th fret
B string, 9th fret G string, ...), and each fretted position can be taken by
any of four fingers. A "tab" is a choice among these alternatives for every
note; a *realistic* tab is the choice a human guitarist would make — the one
minimizing mechanical difficulty across time.

## The consensus method: cost-based dynamic programming

The literature converges on modelling this as a minimum-cost path search:

- **Stages** are note-change events (a note or chord onset).
- **States** at each stage are all feasible fingering alternatives: matrices
  of one `(string, fret, finger)` row per sounding note, pruned by physical
  constraints (distinct strings, bounded fret span, sensible finger order).
- **Costs** split into *static* costs of a grip and *transition* costs
  between consecutive grips; Viterbi/DP finds the globally cheapest path.

Key sources:

- **Sayegh 1989**, "Fingering for String Instruments with the Optimum Path
  Paradigm", *Computer Music Journal* 13(3) — the original formulation of
  fingering as an optimal path over transition costs.
- **[Radisavljevic & Driessen 2004, "Path Difference Learning for Guitar
  Fingering Problem" (ICMC)](https://www.mistic.ece.uvic.ca/publications/2004_icmc_pdl.pdf)**
  — the clearest DP formulation. Introduces the static + transition cost
  split and expresses both as weighted feature vectors. Example transition
  features: "number of frets traversed by a specific finger", "finger changes
  from used to unused". Example static features: "number of frets between
  consecutive fingers", "average fret location", "number of empty strings".
  Weights are learned from published tablatures by gradient descent on the
  difference between the desired and DP-optimal paths.
- **[Hori 2021, "Three-Level Model for Fingering Decision of String
  Instruments" (CMMR)](https://cmmr2021.github.io/proceedings/pdffiles/cmmr2021_11.pdf)**
  (and Hori, Kameoka & Sagayama's earlier input-output HMM work, 2013) —
  inserts an explicit **hand form/position** level between the tablature and
  the fingers: with the index finger at position P, finger f naturally
  covers fret P + f − 1. Decoding is Viterbi over positions and forms.
- **[Burlet & Fujinaga 2013, Robotaba (ISMIR)](https://archives.ismir.net/ismir2013/paper/000217.pdf)**
  — A* over a weighted DAG of string-fret combinations; biomechanical edge
  weights: fret-to-fret movement, chord finger span, and penalties for
  positions past ~fret 7.
- **[Tuohy & Potter, genetic-algorithm tablature (UGA)](https://www.ai.uga.edu/sites/default/files/inline-files/tuohy_daniel.pdf)**
  — GA alternative; fitness rewards staying in one hand position and grips
  that "cover" the passage maximally.
- **[noahbaculi/guitar-tab-generator](https://github.com/noahbaculi/guitar-tab-generator)**
  — practical open-source implementation: candidate positions per pitch,
  cartesian product for chords (pruned to distinct strings), Yen's
  k-shortest-paths over Dijkstra with **movement / span / position** weights.

### Why not a neural model?

Recent transformer approaches outperform DP on accuracy against published
tabs — [Fretting-Transformer 2025](https://arxiv.org/pdf/2506.14223),
[MIDI-to-Tab 2024](https://arxiv.org/html/2408.05024v1), and
[a supervised ML baseline comparison](https://arxiv.org/pdf/2510.10619) —
but they need training corpora and ML dependencies. This repo is
stdlib-only, the piano engine is already a hand-tuned ergonomic DP, and DP
with explicit costs is interpretable and tunable per style, so DP wins here.

## What `guitar/fingering.py` implements

The engine mirrors `piano/fingering.py`'s beam-searched Viterbi:

- **States**: tuples of `(string, fret, finger)` per sounding note.
  Feasibility pruning: distinct strings; ≤ 4 fretted notes (one finger per
  note — the no-barre grip model); fretted extent along the neck within a
  *metric* reach limit (metres, not frets, so reach naturally loosens up
  the neck where frets shrink); fingers strictly increasing up the frets.
- **Hand position** (Hori's form model): `mean(fret − (finger − 1))` over
  fretted notes; all-open grips inherit the previous position, making open
  strings free pivots for shifts.
- **Static costs**: per-fretted-note base (opens are cheaper), mean-fret
  height, span beyond comfort, finger/fret-order mismatch, weak ring/pinky,
  index-not-lowest in chords.
- **Transition costs** (urgency-scaled by the shared `_time_factor`):
  hand-position shift (dominant), same-finger fret hops, finger string
  hops, melody string changes, finger substitution on held notes. A held
  note may never change its string or fret (hard constraint).
- Out-of-range pitches are octave-folded with a warning; chords that can
  never fit the grip model are thinned to bass + top three with a warning.

## Out of scope (future work)

- **Barre chords** and 5–6 string strums (needs a one-finger-many-strings
  form model and richer state space).
- Techniques: bends, slides, hammer-ons/pull-offs, vibrato, harmonics.
- **Right-hand fingerstyle** (p-i-m-a assignment); the engine only emits a
  pick-position curve.
- Capo and alternate tunings (`fret_layout.TUNING` is the single knob).
- **Learned cost weights** (Path Difference Learning) — weights here are
  hand-tuned against golden selftest cases, like the piano engine.
- Tone-based string preference (same pitch sounds different per string).
