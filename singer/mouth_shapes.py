"""Viseme table and 2D mouth outline geometry - the bpy-free source of truth.

Mirrors piano/key_layout.py and guitar/fret_layout.py's role: build_mouth.py
(inside Blender) and timeline.py (plain Python) both import this module, so the
mouth shapes planned outside Blender are exactly the ones built inside it.

Two things live here:

1. **VISEMES** - the mouth position for each viseme (a group of phonemes that
   look alike from the outside; /p/ /b/ /m/ are one viseme because a camera
   cannot tell them apart).  ``PHONEME_VISEME`` maps IPA symbols onto them.
   Every viseme has three loudness variants - ``soft`` / ``medium`` / ``loud``
   - produced by ``shape_for(viseme, level)``.  Singing louder drops the jaw
   and opens the aperture; it does *not* unseal a closed consonant, which is
   why the gain multiplies the aperture rather than being added to it.

2. **outline(shape)** - a parametric 2D outline of the mouth in that shape,
   returned as a flat list of (x, z) points in a FIXED canonical order, and
   ``FACES`` describing how they connect.  Fixed order + fixed topology is the
   whole point: two different shapes give two vertex lists that correspond
   one-to-one, so Blender can store them as shape keys and interpolate between
   them.  (A Bezier curve object cannot blend control points nearly as
   cleanly, so the "vector" mouth is drawn as a flat polygon mesh.)

Coordinate frame: a flat card in the X-Z plane, facing -Y (the camera / the
audience), matching the way the other musicians in this repo face.  +X is the
viewer's right, +Z is up, the origin is the centre of the mouth.  All lengths
are fractions of a neutral mouth width (1.0), so the whole mouth can be scaled
onto a face of any size later.

Shape parameters (all 0..1 unless noted):
    open    aperture height, 0 = lips together (0.55 is a full open "ah")
    width   corner-to-corner width, 1.0 = neutral, >1 spread, <1 pursed
    round   lip rounding/protrusion; pulls the corners in and fattens the lips
    upper   share of the aperture taken above the lip line (upper-lip lift)
            vs below it (jaw drop); 0.5 is even
    corner  corner height offset (+ smile, - frown), in outline units
    teeth   how far the upper teeth show down into the aperture
    tongue  tongue height; 0 hides it behind the lower lip, 1 puts the tip
            up between the teeth (for /th/)
    tuck    lower lip drawn back under the upper teeth (for /f/ /v/)
"""

import math

# ---------------------------------------------------------------------------
# Outline geometry constants (fractions of a neutral mouth width of 1.0)
# ---------------------------------------------------------------------------
EDGE_SAMPLES = 17          # points along one lip edge, corner to corner
MIN_OPEN = 0.012           # a "closed" mouth is still a visible dark slit
LIP_T_UPPER = 0.055        # upper lip band thickness
LIP_T_LOWER = 0.072        # the lower lip is fuller than the upper
CORNER_OUT = 0.020         # how far the lip band extends past the corners
ARCH_IN = 0.60             # aperture edge fullness (lower = flatter/wider)
ARCH_OUT = 0.35            # outer lip edge fullness
CUPID_BOW = 0.28           # dip in the middle of the upper lip
ROUND_NARROW = 0.32        # how much full rounding narrows the mouth
ROUND_FATTEN = 0.55        # how much full rounding thickens the lips

TEETH_HALF_W = 0.78        # upper teeth width, as a fraction of the aperture
TEETH_SEGMENTS = 11        # the teeth row is an arc, not a flat bar
TONGUE_HALF_W = 0.80       # tongue width, likewise
TONGUE_SEGMENTS = 11

# Vertex-block layout inside the flat outline list.  build_mouth.py uses these
# to slice the mesh into material groups and to push each group back in Y so
# the lips read in front of the teeth, tongue and dark interior.
_INNER = 2 * EDGE_SAMPLES - 2                 # 32 aperture points
_OUTER = _INNER                               # 32 outer lip points
INNER_START, INNER_END = 0, _INNER
OUTER_START, OUTER_END = _INNER, _INNER + _OUTER
TEETH_START, TEETH_END = OUTER_END, OUTER_END + 2 * TEETH_SEGMENTS
TONGUE_START, TONGUE_END = TEETH_END, TEETH_END + 2 * TONGUE_SEGMENTS
VERT_COUNT = TONGUE_END


# ---------------------------------------------------------------------------
# Visemes.  Values are the MEDIUM (mezzo-forte) variant; soft and loud are
# derived by shape_for(), unless a viseme overrides them explicitly.
# ---------------------------------------------------------------------------
VISEMES = {
    # --- silence / closures -------------------------------------------------
    "SIL": dict(open=0.00, width=1.00, round=0.00, upper=0.50, corner=0.0,
                teeth=0.00, tongue=0.10, tuck=0.0),
    "MBP": dict(open=0.00, width=0.96, round=0.08, upper=0.50, corner=0.0,
                teeth=0.00, tongue=0.20, tuck=0.0),
    # --- other consonants ---------------------------------------------------
    "FV":  dict(open=0.07, width=0.94, round=0.10, upper=0.80, corner=0.0,
                teeth=0.75, tongue=0.15, tuck=0.85),
    "TH":  dict(open=0.14, width=1.00, round=0.05, upper=0.55, corner=0.0,
                teeth=0.50, tongue=0.95, tuck=0.0),
    "DD":  dict(open=0.17, width=1.02, round=0.05, upper=0.55, corner=0.0,
                teeth=0.50, tongue=0.85, tuck=0.0),
    "SS":  dict(open=0.09, width=1.08, round=0.00, upper=0.55, corner=0.01,
                teeth=0.85, tongue=0.55, tuck=0.0),
    "CH":  dict(open=0.17, width=0.80, round=0.78, upper=0.50, corner=0.0,
                teeth=0.55, tongue=0.45, tuck=0.0),
    "KK":  dict(open=0.23, width=1.00, round=0.05, upper=0.45, corner=0.0,
                teeth=0.25, tongue=0.35, tuck=0.0),
    "RR":  dict(open=0.21, width=0.88, round=0.45, upper=0.50, corner=0.0,
                teeth=0.20, tongue=0.55, tuck=0.0),
    "WW":  dict(open=0.17, width=0.60, round=1.00, upper=0.50, corner=0.0,
                teeth=0.00, tongue=0.25, tuck=0.0),
    # --- vowels -------------------------------------------------------------
    "OH":  dict(open=0.35, width=0.74, round=0.72, upper=0.45, corner=0.0,
                teeth=0.10, tongue=0.15, tuck=0.0),
    "AA":  dict(open=0.55, width=1.00, round=0.15, upper=0.35, corner=0.0,
                teeth=0.22, tongue=0.10, tuck=0.0),
    "AE":  dict(open=0.44, width=1.12, round=0.00, upper=0.45, corner=0.02,
                teeth=0.45, tongue=0.15, tuck=0.0),
    "EH":  dict(open=0.30, width=1.05, round=0.05, upper=0.45, corner=0.01,
                teeth=0.30, tongue=0.25, tuck=0.0),
    "IY":  dict(open=0.18, width=1.18, round=0.00, upper=0.50, corner=0.02,
                teeth=0.60, tongue=0.45, tuck=0.0),
}

VISEME_ORDER = ("SIL", "MBP", "FV", "TH", "DD", "SS", "CH", "KK", "RR", "WW",
                "OH", "AA", "AE", "EH", "IY")

LEVELS = ("soft", "medium", "loud")
LEVEL_GAIN = {"soft": 0.62, "medium": 1.00, "loud": 1.38}

# Aperture ceiling.  A sung "ah" drops the jaw, but an opening that approaches
# the mouth's own width reads as a cartoon gape rather than a singer.  The
# aperture is therefore soft-clipped rather than simply scaled down: below
# OPEN_KNEE nothing changes (the small consonant apertures keep their exact
# sizes and relationships), and above it the remaining travel is compressed
# asymptotically toward OPEN_MAX - so a loud "ah" still reads as more open
# than a soft one, but no shape ever gapes past the ceiling.
OPEN_KNEE = 0.28
OPEN_MAX = 0.46


def _limit_open(value):
    """Soft-clip an aperture to OPEN_MAX (see above)."""
    if value <= OPEN_KNEE:
        return value
    span = OPEN_MAX - OPEN_KNEE
    return OPEN_KNEE + span * (1.0 - math.exp(-(value - OPEN_KNEE) / span))

# Velocity thresholds for picking a variant from a MIDI note.
LEVEL_VELOCITY = ((55, "soft"), (95, "medium"), (128, "loud"))


def level_for_velocity(velocity):
    """Loudness variant for a MIDI note-on velocity."""
    for limit, level in LEVEL_VELOCITY:
        if velocity < limit:
            return level
    return LEVELS[-1]


# ---------------------------------------------------------------------------
# IPA -> viseme.  Anything unmapped falls back to EH (a neutral open mouth),
# which is the right failure mode: an unknown sound still moves the jaw.
# ---------------------------------------------------------------------------
PHONEME_VISEME = {
    # stops and nasals
    "p": "MBP", "b": "MBP", "m": "MBP",
    "t": "DD", "d": "DD", "n": "DD", "l": "DD",
    "k": "KK", "g": "KK", "ŋ": "KK", "h": "KK",
    # fricatives and affricates
    "f": "FV", "v": "FV",
    "θ": "TH", "ð": "TH",
    "s": "SS", "z": "SS",
    "ʃ": "CH", "ʒ": "CH", "tʃ": "CH", "dʒ": "CH",
    # approximants
    "ɹ": "RR", "r": "RR", "ɝ": "RR", "ɚ": "RR",
    "w": "WW", "j": "IY",
    # vowels
    "i": "IY", "ɪ": "IY", "e": "IY",
    "ɛ": "EH", "ʌ": "EH", "ə": "EH",
    "æ": "AE",
    "ɑ": "AA", "ɒ": "AA",
    "ɔ": "OH", "o": "OH",
    "ʊ": "WW", "u": "WW",
}

# Diphthongs are two mouth positions, not one: the first element carries the
# note and the second is a late glide (see timeline.py).
DIPHTHONGS = {
    "eɪ": ("e", "ɪ"),
    "aɪ": ("ɑ", "ɪ"),
    "ɔɪ": ("ɔ", "ɪ"),
    "aʊ": ("ɑ", "ʊ"),
    "oʊ": ("o", "ʊ"),
    "ju": ("j", "u"),
}

VOWELS = set("iɪeɛæɑɒɔoʊuʌəɝɚ") | set(DIPHTHONGS)


def is_vowel(phone):
    return phone in VOWELS or phone in DIPHTHONGS


# The timeline uses this pseudo-phone for "nothing is being sung".
SILENCE = "sil"


def viseme_for(phone):
    """Viseme name for an IPA phone (diphthongs resolve to their first
    element; call ``DIPHTHONGS`` yourself if you want the glide too)."""
    if not phone or phone == SILENCE:
        return "SIL"
    if phone in DIPHTHONGS:
        phone = DIPHTHONGS[phone][0]
    return PHONEME_VISEME.get(phone, "EH")


def shape_for(viseme, level="medium"):
    """The concrete mouth-shape parameters for one viseme at one loudness."""
    base = VISEMES[viseme]
    gain = LEVEL_GAIN[level]
    shape = dict(base)
    # The aperture scales with loudness - so a sealed consonant (open == 0)
    # stays sealed however loudly it is sung.
    shape["open"] = base["open"] * gain
    # Corner spread and rounding are articulatory, not dynamic, so they move
    # only a little; teeth show more when the jaw is lower.
    shape["width"] = 1.0 + (base["width"] - 1.0) * (0.65 + 0.35 * gain)
    shape["round"] = min(1.0, base["round"] * (1.0 + 0.10 * (gain - 1.0)))
    shape["teeth"] = min(1.0, base["teeth"] * (0.85 + 0.15 * gain))
    shape.update(base.get(level, {}))
    # Last word on the aperture, so an explicit per-level override is capped too.
    shape["open"] = _limit_open(shape["open"])
    return shape


def all_variants():
    """[(key_name, viseme, level, shape)] for every viseme x loudness, in the
    order build_mouth.py creates shape keys."""
    out = []
    for viseme in VISEME_ORDER:
        for level in LEVELS:
            out.append((f"V_{viseme}_{level}", viseme, level,
                        shape_for(viseme, level)))
    return out


def key_name(viseme, level):
    return f"V_{viseme}_{level}"


# ---------------------------------------------------------------------------
# Outline generation
# ---------------------------------------------------------------------------
def _arch(s, exponent):
    """Lip-edge profile across the mouth: 1 at the centre (s=0), 0 at the
    corners (s=+-1)."""
    v = 1.0 - s * s
    return v ** exponent if v > 0.0 else 0.0


def _edge(half_w, corner_z, height, sign):
    """One aperture edge, left corner to right corner."""
    pts = []
    for i in range(EDGE_SAMPLES):
        s = -1.0 + 2.0 * i / (EDGE_SAMPLES - 1)
        pts.append((s * half_w, corner_z + sign * height * _arch(s, ARCH_IN)))
    return pts


def outline(shape):
    """Flat list of (x, z) points for one mouth shape, in canonical order:

        [0 : 32]  aperture (inner) loop, counter-clockwise from the left
                  corner over the top and back under the bottom
        [32: 64]  outer lip loop, same winding, one-to-one with the inner loop
        [64: 68]  upper teeth quad
        [68: 80]  tongue

    The order and the count never change, which is what makes these usable as
    Blender shape keys.
    """
    rnd = shape["round"]
    aperture = max(shape["open"], MIN_OPEN)
    half_w = 0.5 * shape["width"] * (1.0 - ROUND_NARROW * rnd)
    corner_z = shape["corner"]
    upper_h = aperture * shape["upper"]
    lower_h = aperture * (1.0 - shape["upper"])

    # The lower lip tuck (f/v) lifts the lower edge and thins the lower lip.
    tuck = shape["tuck"]
    lower_h = max(lower_h * (1.0 - 0.55 * tuck), MIN_OPEN * 0.5)
    t_u = LIP_T_UPPER * (1.0 + ROUND_FATTEN * rnd)
    t_l = LIP_T_LOWER * (1.0 + ROUND_FATTEN * rnd) * (1.0 - 0.45 * tuck)

    inner_upper = _edge(half_w, corner_z, upper_h, +1.0)
    inner_lower = _edge(half_w, corner_z, lower_h, -1.0)
    inner = inner_upper + inner_lower[EDGE_SAMPLES - 2:0:-1]

    # Outer loop: same parameterisation pushed out in X at the corners and in
    # Z by the lip thickness, so it can never cross the inner loop.
    half_w_out = half_w + CORNER_OUT

    def outer_upper_z(s):
        band = 0.12 + 0.88 * _arch(s, ARCH_OUT)
        bow = 1.0 - CUPID_BOW * max(0.0, 1.0 - (s / 0.30) ** 2)
        return corner_z + upper_h * _arch(s, ARCH_IN) + t_u * band * bow

    outer_upper, outer_lower = [], []
    for i in range(EDGE_SAMPLES):
        s = -1.0 + 2.0 * i / (EDGE_SAMPLES - 1)
        band = 0.12 + 0.88 * _arch(s, ARCH_OUT)
        zl = corner_z - lower_h * _arch(s, ARCH_IN) - t_l * band
        outer_upper.append((s * half_w_out, outer_upper_z(s)))
        outer_lower.append((s * half_w_out, zl))
    # Pin the two corner vertices level with the aperture corners so the lip
    # band closes to a point instead of a blunt end.
    outer_upper[0] = (-half_w_out, corner_z)
    outer_upper[-1] = (half_w_out, corner_z)
    outer = outer_upper + outer_lower[EDGE_SAMPLES - 2:0:-1]

    # Upper teeth: a band hanging from behind the upper lip.  Both its edges
    # follow the aperture arch and the drop tapers to nothing at the corners,
    # so the row is a lens that fades away under the lip instead of ending in
    # a hard rectangle - and at teeth == 0 it is hidden entirely.
    aperture_h = upper_h + lower_h
    drop = shape["teeth"] * 0.70 * aperture_h
    floor_z = corner_z - lower_h * 0.55
    teeth_top, teeth_bottom = [], []
    for i in range(TEETH_SEGMENTS):
        u = -1.0 + 2.0 * i / (TEETH_SEGMENTS - 1)
        s = u * TEETH_HALF_W
        edge_z = corner_z + upper_h * _arch(s, ARCH_IN)
        # Tuck the top of the band just inside the upper lip, including under
        # the cupid's bow, so no sliver of white shows above the mouth.
        teeth_top.append((s * half_w,
                          min(edge_z + t_u * 0.9, outer_upper_z(s) - 0.002)))
        teeth_bottom.append((s * half_w,
                             max(edge_z - drop * _arch(u, 0.30), floor_z)))
    teeth = teeth_bottom + teeth_top[::-1]

    # Tongue: the floor of the mouth, not a free-floating blob.  Its lower
    # edge IS the lower aperture edge, so the tongue can never spill outside
    # the lips however high it is raised; the `tongue` parameter only domes
    # it up toward the teeth (fully raised for /th/).
    dome = min((0.06 + 0.88 * shape["tongue"]) * aperture_h, 0.95 * aperture_h)
    tongue_low, tongue_dome = [], []
    for i in range(TONGUE_SEGMENTS):
        u = -1.0 + 2.0 * i / (TONGUE_SEGMENTS - 1)
        s = u * TONGUE_HALF_W
        low_z = corner_z - lower_h * _arch(s, ARCH_IN)
        tongue_low.append((s * half_w, low_z))
        tongue_dome.append((s * half_w, low_z + dome * _arch(u, 0.45)))
    tongue = tongue_low + tongue_dome[::-1]

    return inner + outer + teeth + tongue


# ---------------------------------------------------------------------------
# Mesh form: the flat outline lifted into 3D.
#
# The parts are stacked a little way apart along +Y (away from the camera at
# -Y) so the lips read in front of everything and the dark mouth cavity
# behind it.  The tongue sits in FRONT of the teeth: that is what makes the
# /th/ viseme read, the tongue tip showing between them.  The cavity needs
# its own copy of the aperture ring - the ring itself belongs to the lip band
# at y = 0 - so the mesh has _INNER more vertices than the 2D outline.
# ---------------------------------------------------------------------------
LAYER_Y = {"lips": 0.0, "tongue": 0.004, "teeth": 0.008, "interior": 0.012}

CAVITY_START = VERT_COUNT
CAVITY_END = CAVITY_START + _INNER
MESH_VERT_COUNT = CAVITY_END

_LAYERS = ([LAYER_Y["lips"]] * (_INNER + _OUTER)
           + [LAYER_Y["teeth"]] * (2 * TEETH_SEGMENTS)
           + [LAYER_Y["tongue"]] * (2 * TONGUE_SEGMENTS)
           + [LAYER_Y["interior"]] * _INNER)


def mesh_vertices(shape):
    """The mouth as 3D vertices (x, y, z) for one shape, ready for Blender.

    Always MESH_VERT_COUNT points in the same order, so two shapes differ
    only in position - the requirement for shape keys.
    """
    points = outline(shape)
    points = points + points[INNER_START:INNER_END]      # cavity copy
    return [(x, y, z) for (x, z), y in zip(points, _LAYERS)]


def _build_faces():
    """Face index lists per material group; topology is shape-independent."""
    lips = []
    for i in range(_INNER):
        j = (i + 1) % _INNER
        lips.append((INNER_START + i, INNER_START + j,
                     OUTER_START + j, OUTER_START + i))
    return {"lips": lips,
            "interior": [tuple(range(CAVITY_START, CAVITY_END))],
            "teeth": [tuple(range(TEETH_START, TEETH_END))],
            "tongue": [tuple(range(TONGUE_START, TONGUE_END))]}


FACES = _build_faces()


if __name__ == "__main__":
    print(f"{len(VISEMES)} visemes x {len(LEVELS)} levels "
          f"= {len(all_variants())} shape keys, {MESH_VERT_COUNT} verts each")
    for name, viseme, level, shape in all_variants():
        pts = outline(shape)
        assert len(pts) == VERT_COUNT, (name, len(pts))
        assert len(mesh_vertices(shape)) == MESH_VERT_COUNT, name
        zs = [p[1] for p in pts[INNER_START:INNER_END]]
        xs = [p[0] for p in pts[OUTER_START:OUTER_END]]
        if level == "medium":
            print(f"  {viseme:4s} aperture {max(zs) - min(zs):.3f} "
                  f"width {max(xs) - min(xs):.3f}")
