"""Drum-kit geometry, the General MIDI drum map, and per-voice strike
targets - the bpy-free source of truth.

Mirrors the role of guitar/fret_layout.py and piano/key_layout.py:
build_drum_kit.py (inside Blender) imports the geometry constants here, and
fingering.py / animate_drums.py (plain Python, run outside Blender) import
the GM map and strike points. Because both sides read the same numbers, the
strike coordinates computed outside Blender land exactly on the meshes built
inside it.

To swap in a *different* kit model, this is the only file to touch: adjust
the drum/cymbal specifications (position, size, tilt), and - if the physical
object names or the note routing change - the ``VOICES`` targets and
``GM_DRUM_MAP``. Everything downstream (demo MIDI, striking planner,
animator) is driven from here.

Coordinate convention matches build_drum_kit.py (metres, +Z up, floor at
z = 0): the drummer sits at +Y looking toward -Y, so the drummer's LEFT hand
points +X and their RIGHT points -X. Tilt is a lean toward the drummer,
applied as a rotation about X.
"""

import math

INCH = 0.0254

# ---------------------------------------------------------------------------
# Kit specification (shared with build_drum_kit.py). Sizes are (diameter,
# depth) in inches; positions are the drum/cymbal centre in metres.
# ---------------------------------------------------------------------------
KICK_DIA, KICK_DEPTH = 22 * INCH, 18 * INCH
KICK_R = KICK_DIA / 2.0
KICK_CENTER = (0.0, 0.0, KICK_R)          # rests on the floor: centre = radius
KICK_BATTER_Y = KICK_DEPTH / 2.0          # batter head plane (drummer side, +Y)

# (name, dia_in, depth_in, center, tilt_deg)  -- vertical-axis drums.
TOMS = [
    ("Tom1",     12, 8,  (0.20, 0.01, 0.69), -13.0),    # small rack tom, left
    ("Tom2",     13, 9,  (-0.22, 0.01, 0.72), -13.0),   # mid rack tom, right
    ("FloorTom", 16, 16, (-0.38, 0.33, 0.44),  -4.0),   # floor tom, right (pulled
    # in toward the seat so the (back-seated, snare-angled) drummer reaches the
    # toms without over-reaching or crossing)
]

SNARE = ("Snare", 14, 5.5, (0.24, 0.40, 0.57), -6.0)   # in front of the kick

# (name, dia_in, center, tilt_deg, stand_base_xy) -- cymbals on boom stands.
CRASH = ("Crash", 16, (0.40, -0.10, 1.15), -16.0, (0.58, -0.46))
RIDE = ("Ride", 20, (-0.46, 0.06, 1.02), -14.0, (-0.68, -0.44))
HIHAT_DIA = 14
HIHAT_CENTER = (0.55, 0.32, 0.86)   # ~34" at the cymbals, just above snare
HIHAT_BASE_XY = (0.55, 0.32)
HIHAT_TOP_GAP = 0.045               # rest gap of HiHat_Top above HiHat_Bottom
# The hi-hat pedal's toe end sits at the stand centre (the pull-rod point) and
# the footboard is yawed about that toe so it points back toward the DRUMMER
# (heel swings toward the body, -X) -- the ergonomic angle -- while the toe stays
# aligned with the stand. The foot and the stand's leg gap follow this angle.
# Positive = heel swings toward the drummer (-X); the toe stays put.
HIHAT_PEDAL_YAW = math.radians(30.0)


# ---------------------------------------------------------------------------
# Strike-point geometry: where a stick tip / beater actually contacts each
# playing surface, derived from the kit spec so it tracks any layout change.
# ---------------------------------------------------------------------------
def _rot_x(vec, deg):
    """Rotate a vector about the world X axis by `deg` (the drum tilt)."""
    a = math.radians(deg)
    x, y, z = vec
    return (x, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a))


def _head_center(dia_in, depth_in, center, tilt_deg):
    """Centre of the batter (top) head of a vertical-axis drum, in metres.

    The batter head sits at local +z = depth/2 above the drum centre; the
    drum's tilt rotates that offset toward the drummer.
    """
    depth = depth_in * INCH
    ox, oy, oz = _rot_x((0.0, 0.0, depth / 2.0), tilt_deg)
    return (center[0] + ox, center[1] + oy, center[2] + oz)


def _cymbal_bow_point(dia_in, center, tilt_deg, frac=0.5):
    """A point out on the bow toward the drummer (+Y), where a stick strikes."""
    r = dia_in * INCH / 2.0
    ox, oy, oz = _rot_x((0.0, r * frac, 0.0), tilt_deg)
    return (center[0] + ox, center[1] + oy, center[2] + oz)


def _hihat_point(frac=0.45):
    """Top-cymbal edge toward the drummer, where the stick plays the hats."""
    r = HIHAT_DIA * INCH / 2.0
    cx, cy, cz = HIHAT_CENTER
    return (cx, cy + r * frac, cz + HIHAT_TOP_GAP + 0.01)


def _ride_bell_point():
    """The ride bell, near the cymbal centre, raised onto the dome."""
    cx, cy, cz = RIDE[2]
    return (cx, cy, cz + 0.02)


# ---------------------------------------------------------------------------
# Voices: the abstract sounds the planner reasons about. Each maps to a
# physical object built by build_drum_kit.py, a world strike point, a strike
# mechanism, and the default limb of the standard right-handed convention:
#   right hand -> hi-hat / ride,  left hand -> snare,
#   right foot -> kick,           left foot -> hi-hat pedal.
# A `limb` of None is a "flex" voice (toms, crashes) whose hand the planner
# chooses by proximity / alternation. `open` (hi-hat voices only) records the
# resulting hi-hat state so the animator can drive the top cymbal's gap.
# ---------------------------------------------------------------------------
def _build_voices():
    t1, t2, ft = TOMS[0], TOMS[1], TOMS[2]
    _, sd, sp, sc, st = SNARE
    snare_pt = _head_center(sd, sp, sc, st)
    # A crash is struck out on the EDGE (frac ~0.85), not mid-bow: that is real
    # crash technique (a glancing edge hit), and it also lets the natural steep
    # bead-up crash stroke clear the disc. Struck mid-bow (frac 0.5) the low grip
    # of a +28 deg up-pitched stick rakes the shaft ~0.058 m DOWN THROUGH the
    # tilted disc to reach the tip inboard; landing the bead near the rim instead
    # puts the whole shaft out over open air (see build_drummer stick_pitch). The
    # ride is different -- it is a bow instrument, so it keeps its mid-bow strike
    # and clears via a shallower pitch instead.
    crash_pt = _cymbal_bow_point(CRASH[1], CRASH[2], CRASH[3], frac=0.85)
    ride_pt = _cymbal_bow_point(RIDE[1], RIDE[2], RIDE[3])
    hat_pt = _hihat_point()

    def stick(target, point, limb, **extra):
        d = {"target": target, "point": point, "strike": "stick", "limb": limb}
        d.update(extra)
        return d

    return {
        "kick":        {"target": "Kick", "point": (0.0, KICK_BATTER_Y, KICK_R),
                        "strike": "kick", "limb": "footR"},
        "snare":       stick("Snare", snare_pt, "L"),
        "side_stick":  stick("Snare", snare_pt, "L"),
        "tom_hi":      stick("Tom1", _head_center(t1[1], t1[2], t1[3], t1[4]), None),
        "tom_mid":     stick("Tom2", _head_center(t2[1], t2[2], t2[3], t2[4]), None),
        "tom_floor":   stick("FloorTom", _head_center(ft[1], ft[2], ft[3], ft[4]), None),
        "hihat_closed": stick("HiHat_Top", hat_pt, "R", open=False),
        "hihat_open":   stick("HiHat_Top", hat_pt, "R", open=True),
        "hihat_pedal": {"target": "HiHat_Top", "point": hat_pt,
                        "strike": "hat_pedal", "limb": "footL", "open": False},
        "crash":       stick("Crash", crash_pt, None),
        "crash2":      stick("Crash", crash_pt, None),
        "china":       stick("Crash", crash_pt, None),
        "splash":      stick("Crash", crash_pt, None),
        "ride":        stick("Ride", ride_pt, "R"),
        "ride_bell":   stick("Ride", _ride_bell_point(), "R"),
    }


VOICES = _build_voices()

# General MIDI percussion note -> voice, folded onto the built 5-piece kit.
# (See RESEARCH.md for the full GM key map, notes 35-81.)
GM_DRUM_MAP = {
    35: "kick", 36: "kick",            # acoustic / bass drum 1
    37: "side_stick",
    38: "snare", 40: "snare", 39: "snare",   # acoustic / electric snare, clap
    41: "tom_floor", 43: "tom_floor",  # low / high floor tom
    45: "tom_mid", 47: "tom_mid",      # low / low-mid tom
    48: "tom_hi", 50: "tom_hi",        # hi-mid / high tom
    42: "hihat_closed", 44: "hihat_pedal", 46: "hihat_open",
    49: "crash", 57: "crash2", 55: "splash", 52: "china",
    51: "ride", 59: "ride", 53: "ride_bell",
}


def voice_of(midi):
    """The voice a GM percussion note plays, or None if unmapped."""
    return GM_DRUM_MAP.get(midi)


def strike_info(midi):
    """Full strike descriptor for a GM note: a copy of its voice spec plus a
    ``voice`` key. Returns None for notes this kit does not cover."""
    voice = GM_DRUM_MAP.get(midi)
    if voice is None:
        return None
    info = dict(VOICES[voice])
    info["voice"] = voice
    return info


def strike_point(voice):
    """(x, y, z) contact point for a voice."""
    return VOICES[voice]["point"]
