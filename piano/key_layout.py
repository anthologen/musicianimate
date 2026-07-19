"""Keyboard geometry shared by the Blender piano builder and the fingering
engine.

This module is deliberately bpy-free so it can be imported both inside
Blender (by build_piano.py, which instantiates the key meshes) and outside
it (by piano/fingering.py, which needs fingertip target coordinates that
line up exactly with the key objects in the scene).

Coordinate conventions match the objects created by build_piano.build_keys():
x runs along the keyboard (0 at its center), y runs from the player (0 at
the key fronts) toward the fallboard, z is up.
"""

# ---------------------------------------------------------------------------
# Key geometry parameters (meters)
# ---------------------------------------------------------------------------

WHITE_W = 0.023
WHITE_GAP = 0.001
WHITE_PITCH = WHITE_W + WHITE_GAP
WHITE_LEN = 0.15
WHITE_H = 0.02

BLACK_W = 0.013
BLACK_LEN = 0.095
BLACK_H = 0.012

# Per-semitone x-offset within an octave, in units of WHITE_PITCH (0..7).
# A widely used approximation for laying out black keys realistically
# relative to their white-key neighbors.
NOTE_OFFSETS = {
    0: 0.0, 1: 0.65, 2: 1.0, 3: 1.75, 4: 2.0, 5: 3.0,
    6: 3.6, 7: 4.0, 8: 4.65, 9: 5.0, 10: 5.7, 11: 6.0,
}
WHITE_NOTES = {0, 2, 4, 5, 7, 9, 11}

START_MIDI, END_MIDI = 21, 108  # A0 .. C8 (88 keys)

WHITE_REST_Z = WHITE_H / 2.0
BLACK_REST_Z = WHITE_H + BLACK_H / 2.0


def is_black(midi):
    return midi % 12 not in WHITE_NOTES


def _raw_x(midi):
    """Left edge of the key's octave-relative slot, before centering."""
    group = midi // 12
    anchor_x = (group - 1) * 7 * WHITE_PITCH
    return anchor_x + NOTE_OFFSETS[midi % 12] * WHITE_PITCH


# Centering shift so the 88-key span is symmetric around x = 0, identical to
# the computation in build_piano.build_keys().
_MIN_X = min(_raw_x(m) for m in range(START_MIDI, END_MIDI + 1))
_MAX_X = max(_raw_x(m) for m in range(START_MIDI, END_MIDI + 1)) + WHITE_W
TOTAL_WIDTH = _MAX_X - _MIN_X
_SHIFT = -_MIN_X - TOTAL_WIDTH / 2.0


def key_x(midi):
    """Center x of the key object for this MIDI note (matches the scene)."""
    return _raw_x(midi) + _SHIFT + WHITE_W / 2.0


def key_y(midi):
    """Center y of the key object for this MIDI note (matches the scene)."""
    return WHITE_LEN - BLACK_LEN / 2.0 if is_black(midi) else WHITE_LEN / 2.0


def fingertip_y(midi):
    """Where a fingertip lands along the key: black keys are struck near
    their front edge (as pianists do - the raised center is a long reach),
    white keys on the exposed front portion."""
    if is_black(midi):
        return (WHITE_LEN - BLACK_LEN) + 0.012
    return (WHITE_LEN - BLACK_LEN) / 2.0


def rest_z(midi):
    return BLACK_REST_Z if is_black(midi) else WHITE_REST_Z
