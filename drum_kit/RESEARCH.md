# Drum animation research

Notes behind `fingering.py` (the striking planner) and `animate_drums.py`
(the keyframer). The goal: drive a two-handed, two-footed drummer from a
General MIDI drum track, with motion that reads as real playing and a kit
whose strike targets can be swapped out (`kit_layout.py`).

## 1. Prior work on animating drummers from MIDI

- **IK-driven virtual drummers.** Sporka et al.'s *Drum Set Tutorial System*
  and *An Animated Virtual Drummer* convert a note description (essentially a
  MIDI drum track) into a seated human animation by (a) assigning each stroke
  to a hand and (b) solving inverse kinematics so the stick tip reaches the
  struck surface, inferring a drumstick trajectory between hits. The hard part
  they identify is the **hand-to-stroke assignment** and keeping the arms out
  of each other's way on a kit of arbitrary layout.
- **Learned trajectories (DRUMS, SIGGRAPH MIG 2025).** A more recent system
  predicts fine-grained 3-D hand trajectories from raw MIDI with a
  bi-directional LSTM, **parents the sticks to the hands**, matches upper-body
  and facial motion to the MIDI, and adds a **procedural foot module** for the
  pedals. The takeaways that generalise to a lightweight procedural rig: sticks
  are rigid children of the hands (so you animate the hand and the tip
  follows), and feet are handled separately from hands.

Our approach is the lightweight, deterministic version of the IK papers: no
learned model and no runtime IK. The sticks are rigid (tip at a fixed point in
the hand's local frame), so placing the hand *is* placing the tip — we solve
for the hand location in closed form and bake plain keyframes, exactly the
pattern the guitar pick-hand already uses in this repo.

## 2. The General MIDI percussion map (channel 10)

GM fixes a note-number -> drum-sound table (notes 35-81) on MIDI channel 10,
so we route entirely off **note number**, never channel (the repo's
`parse_midi` masks the channel away anyway). The demo writes on channel 10 for
correct DAW playback. The subset folded onto this 5-piece kit:

| Note | GM sound | Voice | Note | GM sound | Voice |
|-----:|----------|-------|-----:|----------|-------|
| 35/36 | Bass Drum | `kick` | 46 | Open Hi-Hat | `hihat_open` |
| 37 | Side Stick | `side_stick` | 49 | Crash 1 | `crash` |
| 38/40 | Snare | `snare` | 51/59 | Ride | `ride` |
| 41/43 | Floor Tom | `tom_floor` | 52 | Chinese | `china` |
| 45/47 | Low/Low-mid Tom | `tom_mid` | 53 | Ride Bell | `ride_bell` |
| 48/50 | Hi-mid/High Tom | `tom_hi` | 55 | Splash | `splash` |
| 42 | Closed Hi-Hat | `hihat_closed` | 57 | Crash 2 | `crash2` |
| 44 | Pedal Hi-Hat | `hihat_pedal` | | | |

Voices with no dedicated object fold onto an existing one (china/splash/crash2
-> `Crash`, ride bell -> `Ride`, side stick -> `Snare`). Unmapped notes are
dropped with a warning. All of this — the note map, each voice's physical
target, strike point, mechanism and default limb — lives in `kit_layout.py`,
the one file to edit when swapping kits.

## 3. Sticking heuristics (which limb plays each hit)

Drumming sticking is a well-worn craft; the rules we encode:

- **Standard right-handed convention.** Right hand keeps time on the hi-hat /
  ride, left hand plays the snare, right foot the kick, left foot the hi-hat
  pedal. On most kits the hi-hat sits to the player's left, so the right hand
  **crosses over** the left to reach it (cross-arm playing) — the default we
  use. (Open-handed playing avoids the cross; it's a valid alternative the
  `kit_layout` conventions could be flipped to.)
- **Feet are deterministic.** Kick -> right foot, hi-hat pedal -> left foot,
  with no interaction with the hands. A pedal hi-hat note ("chick") also
  drives the hi-hat open/closed state.
- **Alternating single strokes for speed.** Fast runs and fills are played
  hand-to-hand (R L R L) so no single hand has to move faster than it can. We
  trigger alternation whenever the inter-onset gap drops below `FAST_GAP`
  (0.14 s ≈ sixteenths at 120 BPM); slower passages keep the convention hand.
- **Minimise arm travel on flex voices.** Toms and crashes have no fixed hand;
  each goes to the hand whose *last position* is nearest the target, with a
  penalty (`CONV_PENALTY`) for abandoning a voice's convention hand — so a
  groove keeps its natural grip while a fill grabs the closest stick.
- **Split simultaneous hits.** A chord of hits (e.g. snare + hi-hat on the
  backbeat) is divided across the two hands to minimise total travel, so one
  hand is never asked to be in two places at once.

These map directly to the planner: convention comes from
`kit_layout.VOICES[...]['limb']`, alternation is the `FAST_GAP` branch, and
travel minimisation is a nearest-hand cost with the convention penalty.

## 4. Motion model — wind-up proportional to volume

A real stroke is *anticipation -> contact -> rebound*: the stick lifts to a
backswing apex, drops to the head at the note onset, and bounces back. Louder
notes are struck from a **taller backswing dropped faster**. We reuse the
guitar pick-hand's velocity model:

    v = velocity / 127
    lift   = LIFT_MIN + v·(LIFT_MAX − LIFT_MIN)     # apex height above the head
    strike = STRIKE_SLOW − v·(STRIKE_SLOW − STRIKE_FAST)  # apex→contact seconds

so a hit at velocity 127 winds up ~14 cm and drops in ~45 ms, while a soft hit
winds up ~3 cm and drifts down over ~120 ms. Between strikes the hand glides
from one target toward the next, which reads as natural arm travel and honours
the "minimise movement" heuristic visually.

Feet follow the same idea: the kick beater cocks **further back when louder**
before swinging into the batter head (rest −10° -> +6° at contact), and the
footboards press in step (carrying the parented shoes). The hi-hat foot holds
the top cymbal open or closed along the planner's hi-hat timeline, with a
quick pedal chick on each left-foot note. Struck cymbals get a short,
velocity-scaled wobble that decays back to rest.

## 5. What this kit does / doesn't cover

- Two hands + two feet; four independent limbs, cross-arm hi-hat by default.
- Not modelled: buzz/press rolls, flams as grace-note offsets (a >2-hit
  cluster is stacked on alternating hands and warned about), moving the seated
  body/arms above the wrists (the hands float, matching the repo's other
  minimalist hand rigs), and remote/auxiliary hi-hats.

## Sources

- DRUMS: Drummer Reconstruction Using MIDI Sequences — ACM SIGGRAPH MIG 2025.
  https://dl.acm.org/doi/10.1145/3769047.3769066
- Sporka et al., *The Drum Set Tutorial System by Means of Inverse Kinematics*
  (CEMVRC 2005); *An Animated Virtual Drummer*.
  https://www.researchgate.net/publication/228976881_An_animated_virtual_drummer
- General MIDI percussion key map (notes 35-81).
  https://soundprogramming.net/file-formats/general-midi-drum-note-numbers/
- Cross-arm vs open-handed drumming / hi-hat ergonomics — Modern Drummer,
  Drummer Cafe, Wikipedia "Open-handed drumming".
