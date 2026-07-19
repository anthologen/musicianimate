"""Guitar fretboard geometry and tuning - the bpy-free source of truth.

Mirrors piano/key_layout.py's role: build_guitar.py (inside Blender) and
fingering.py (plain Python) both import this module, so fingering target
coordinates computed outside Blender land exactly on the meshes built
inside it.

Coordinate frame matches build_guitar.py: the guitar lies face-up (+Z),
neck pointing +Y, bridge at y=0, nut at y=SCALE_LEN. String 0 is the low
E on the bass (-x) side, matching the Guitar_String_<i> object names.
All dimensions are in metres.
"""

# ---------------------------------------------------------------------------
# Shared with build_guitar.py (single source of truth)
# ---------------------------------------------------------------------------
SCALE_LEN = 0.648          # nut-to-bridge speaking length (25.5")
NUM_FRETS = 22
NUM_STRINGS = 6

BRIDGE_Y = 0.0             # bridge saddles sit here, on the body
NUT_Y = BRIDGE_Y + SCALE_LEN

NECK_THICK = 0.021         # wood behind the fretboard
FB_THICK = 0.006           # fretboard
FRET_H = 0.0012
STRING_Z = NECK_THICK + FB_THICK + 0.004   # string plane above the fretboard

BRIDGE_SPAN = 0.052        # outer-string spacing at the bridge
NUT_SPAN = 0.035           # outer-string spacing at the nut

# ---------------------------------------------------------------------------
# Musical model
# ---------------------------------------------------------------------------
TUNING = (40, 45, 50, 55, 59, 64)   # EADGBE; index matches Guitar_String_<i>
MIN_MIDI = TUNING[0]                # E2
MAX_MIDI = TUNING[-1] + NUM_FRETS   # D6

PRESS_FRACTION = 0.30    # fingertip sits this far up the slot behind the wire
PRESS_Z = NECK_THICK + FB_THICK + FRET_H + 0.0006  # string pressed to the crown
PLUCK_Y = 0.14           # pick contact point, between the two pickups


def fret_y(n):
    """y of fret wire n (equal-tempered); fret_y(0) is the nut. Accepts
    fractional n for interpolated hand positions."""
    return NUT_Y - SCALE_LEN * (1.0 - 2.0 ** (-n / 12.0))


def fret_width(n):
    """Length of the slot behind fret n (between wires n-1 and n)."""
    return fret_y(n - 1) - fret_y(n)


def string_x(string, y):
    """x of a string at neck coordinate y. Strings are straight, fanning
    linearly from BRIDGE_SPAN at the bridge to NUT_SPAN at the nut."""
    f = (string - (NUM_STRINGS - 1) / 2.0) / (NUM_STRINGS - 1)
    return f * (BRIDGE_SPAN + (NUT_SPAN - BRIDGE_SPAN) * (y / NUT_Y))


def press_y(fret):
    """Fingertip y for a fretted note: just behind the wire, nut side."""
    return fret_y(fret) + PRESS_FRACTION * fret_width(fret)


def fingertip(string, fret):
    """(x, y, z) target for the fretting fingertip. For an open string
    (fret 0) there is no press; the nut-side reference point is returned
    so hand-position math still has a coordinate to work with."""
    if fret == 0:
        return (string_x(string, NUT_Y), NUT_Y, STRING_Z)
    py = press_y(fret)
    return (string_x(string, py), py, PRESS_Z)


def pluck_point(string):
    """(x, y, z) where the pick strikes this string."""
    return (string_x(string, PLUCK_Y), PLUCK_Y, STRING_Z)


def candidate_positions(midi):
    """All (string, fret) pairs that sound this pitch, low string first."""
    return [(s, midi - open_midi)
            for s, open_midi in enumerate(TUNING)
            if 0 <= midi - open_midi <= NUM_FRETS]


def midi_of(string, fret):
    """Inverse of candidate_positions, for checks and tests."""
    return TUNING[string] + fret
