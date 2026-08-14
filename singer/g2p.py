"""English text -> IPA phonemes, with no external dependencies.

The repo is deliberately dependency-free (own SMF parser, no mido/music21), so
this is a self-contained grapheme-to-phoneme engine rather than a wrapper
around espeak/g2p_en/phonemizer.  It has two layers:

1. **LEXICON** - a hand-written pronunciation dictionary of the ~300 most
   common English (and song-lyric) words.  English spelling is irregular
   exactly where it is frequent - "the", "of", "one", "love", "heart" - so a
   small lexicon buys most of the accuracy.
2. **Letter-to-sound rules** - an ordered, context-sensitive rule set in the
   style of Elovitz et al.'s NRL rules (1976), which is what every rule-based
   English TTS front end has used since.  Each rule is
   ``(left_context, letters, right_context, phones)``; the first rule whose
   contexts match at the current position wins and consumes ``letters``.

   Context pattern characters:
       ``#``  one or more vowels          ``:``  zero or more consonants
       ``^``  one consonant               ``.``  one voiced consonant
       ``+``  one front vowel (e/i/y)     ``%``  a suffix (e, er, es, ed, ing, ely)
       ``&``  one sibilant                ``@``  a consonant after which
       ``$``  a word boundary                    long-u is /u/, not /ju/

For higher accuracy, drop a CMUdict file next to this module and call
``load_cmudict(path)`` - the lexicon simply grows and the rules stay as the
fallback for out-of-vocabulary words.  Nothing else in the pipeline changes.

Output is IPA, using the General American inventory that mouth_shapes.py maps
to visemes.  Stress is not marked: the mouth does not show it, and singing
overrides lexical stress with the melody anyway.

Usage::

    python -m singer.g2p "Twinkle twinkle little star"
    python -m singer.g2p --selftest
"""

import os
import re
import sys

VOWEL_LETTERS = set("aeiouy")
CONSONANT_LETTERS = set("bcdfghjklmnpqrstvwxz")
VOICED_LETTERS = set("bdgjlmnrvwz")
FRONT_VOWELS = set("eiy")
SIBILANTS = set("scgzxj")
# Consonants after which orthographic "u" is /u/ rather than /ju/.
PLAIN_U_AFTER = set("tsrdlznj")
SUFFIXES = ("ing", "ely", "er", "es", "ed", "e")


# ---------------------------------------------------------------------------
# Pronunciation lexicon.  One word per line: "word ipa ipa ipa ...".
# ---------------------------------------------------------------------------
_LEXICON_TEXT = """
a ə
about ə b aʊ t
after æ f t ɝ
again ə g ɛ n
against ə g ɛ n s t
ah ɑ
ain't eɪ n t
all ɔ l
alone ə l oʊ n
above ə b ʌ v
along ə l ɔ ŋ
already ɔ l ɹ ɛ d i
also ɔ l s oʊ
always ɔ l w eɪ z
am æ m
an æ n
and æ n d
angel eɪ n dʒ ə l
another ə n ʌ ð ɝ
any ɛ n i
anymore ɛ n i m ɔ ɹ
anything ɛ n i θ ɪ ŋ
are ɑ ɹ
arms ɑ ɹ m z
around ɝ aʊ n d
as æ z
ask æ s k
at æ t
away ə w eɪ
baby b eɪ b i
back b æ k
be b i
beautiful b ju t ə f ə l
because b ɪ k ʌ z
been b ɪ n
before b ɪ f ɔ ɹ
begin b ɪ g ɪ n
behind b ɪ h aɪ n d
believe b ɪ l i v
beside b ɪ s aɪ d
between b ɪ t w i n
blue b l u
body b ɑ d i
both b oʊ θ
boy b ɔɪ
break b ɹ eɪ k
breath b ɹ ɛ θ
breathe b ɹ i ð
bright b ɹ aɪ t
bring b ɹ ɪ ŋ
burn b ɝ n
but b ʌ t
by b aɪ
call k ɔ l
came k eɪ m
can k æ n
can't k æ n t
cannot k æ n ɑ t
care k ɛ ɹ
'cause k ʌ z
change tʃ eɪ n dʒ
child tʃ aɪ l d
close k l oʊ s
cold k oʊ l d
come k ʌ m
comes k ʌ m z
coming k ʌ m ɪ ŋ
could k ʊ d
couldn't k ʊ d ə n t
cry k ɹ aɪ
dance d æ n s
dark d ɑ ɹ k
day d eɪ
days d eɪ z
dear d ɪ ɹ
deep d i p
diamond d aɪ ə m ə n d
did d ɪ d
didn't d ɪ d ə n t
die d aɪ
do d u
does d ʌ z
doesn't d ʌ z ə n t
done d ʌ n
don't d oʊ n t
door d ɔ ɹ
down d aʊ n
dream d ɹ i m
dreams d ɹ i m z
each i tʃ
earth ɝ θ
easy i z i
end ɛ n d
enough ɪ n ʌ f
even i v ə n
ever ɛ v ɝ
every ɛ v ɹ i
everything ɛ v ɹ i θ ɪ ŋ
eye aɪ
eyes aɪ z
face f eɪ s
fall f ɔ l
falling f ɔ l ɪ ŋ
far f ɑ ɹ
feel f i l
feeling f i l ɪ ŋ
few f ju
find f aɪ n d
fire f aɪ ɝ
first f ɝ s t
fly f l aɪ
follow f ɑ l oʊ
for f ɔ ɹ
forever f ɝ ɛ v ɝ
forget f ɝ g ɛ t
free f ɹ i
friend f ɹ ɛ n d
from f ɹ ʌ m
gave g eɪ v
get g ɛ t
girl g ɝ l
give g ɪ v
go g oʊ
goes g oʊ z
going g oʊ ɪ ŋ
gold g oʊ l d
gone g ɔ n
gonna g ə n ə
good g ʊ d
got g ɑ t
great g ɹ eɪ t
green g ɹ i n
had h æ d
hand h æ n d
hands h æ n d z
happy h æ p i
hard h ɑ ɹ d
has h æ z
have h æ v
he h i
head h ɛ d
hear h ɪ ɹ
heart h ɑ ɹ t
heaven h ɛ v ə n
held h ɛ l d
hello h ə l oʊ
help h ɛ l p
her h ɝ
here h ɪ ɹ
hey h eɪ
high h aɪ
him h ɪ m
his h ɪ z
hold h oʊ l d
home h oʊ m
hope h oʊ p
hour aʊ ɝ
how h aʊ
i aɪ
if ɪ f
i'll aɪ l
i'm aɪ m
in ɪ n
inside ɪ n s aɪ d
into ɪ n t u
is ɪ z
isn't ɪ z ə n t
it ɪ t
it's ɪ t s
its ɪ t s
i've aɪ v
just dʒ ʌ s t
keep k i p
kiss k ɪ s
knew n u
know n oʊ
known n oʊ n
la l ɑ
lady l eɪ d i
last l æ s t
laugh l æ f
learn l ɝ n
leave l i v
let l ɛ t
life l aɪ f
light l aɪ t
like l aɪ k
listen l ɪ s ə n
little l ɪ t ə l
live l ɪ v
long l ɔ ŋ
look l ʊ k
lose l u z
lost l ɔ s t
love l ʌ v
made m eɪ d
make m eɪ k
man m æ n
many m ɛ n i
may m eɪ
me m i
mind m aɪ n d
mine m aɪ n
moment m oʊ m ə n t
money m ʌ n i
moon m u n
more m ɔ ɹ
morning m ɔ ɹ n ɪ ŋ
most m oʊ s t
mother m ʌ ð ɝ
much m ʌ tʃ
music m ju z ɪ k
must m ʌ s t
my m aɪ
na n ɑ
name n eɪ m
near n ɪ ɹ
need n i d
never n ɛ v ɝ
new n u
night n aɪ t
no n oʊ
not n ɑ t
nothing n ʌ θ ɪ ŋ
now n aʊ
of ʌ v
off ɔ f
oh oʊ
old oʊ l d
on ɑ n
once w ʌ n s
one w ʌ n
only oʊ n l i
ooh u
open oʊ p ə n
or ɔ ɹ
other ʌ ð ɝ
our aʊ ɝ
out aʊ t
over oʊ v ɝ
own oʊ n
people p i p ə l
place p l eɪ s
play p l eɪ
please p l i z
put p ʊ t
rain ɹ eɪ n
reason ɹ i z ə n
remember ɹ ɪ m ɛ m b ɝ
right ɹ aɪ t
river ɹ ɪ v ɝ
road ɹ oʊ d
run ɹ ʌ n
said s ɛ d
same s eɪ m
say s eɪ
sea s i
see s i
she ʃ i
should ʃ ʊ d
show ʃ oʊ
sing s ɪ ŋ
sky s k aɪ
sleep s l i p
slow s l oʊ
smile s m aɪ l
so s oʊ
some s ʌ m
someone s ʌ m w ʌ n
something s ʌ m θ ɪ ŋ
song s ɔ ŋ
soul s oʊ l
sound s aʊ n d
star s t ɑ ɹ
stars s t ɑ ɹ z
stay s t eɪ
still s t ɪ l
stop s t ɑ p
strong s t ɹ ɔ ŋ
summer s ʌ m ɝ
sun s ʌ n
sweet s w i t
take t eɪ k
talk t ɔ k
tears t ɪ ɹ z
tell t ɛ l
than ð æ n
thank θ æ ŋ k
that ð æ t
that's ð æ t s
the ð ə
their ð ɛ ɹ
them ð ɛ m
then ð ɛ n
there ð ɛ ɹ
these ð i z
they ð eɪ
they're ð ɛ ɹ
thing θ ɪ ŋ
things θ ɪ ŋ z
think θ ɪ ŋ k
this ð ɪ s
those ð oʊ z
though ð oʊ
thought θ ɔ t
three θ ɹ i
through θ ɹ u
time t aɪ m
to t u
today t ə d eɪ
together t ə g ɛ ð ɝ
tonight t ə n aɪ t
too t u
touch t ʌ tʃ
true t ɹ u
try t ɹ aɪ
turn t ɝ n
twinkle t w ɪ ŋ k ə l
two t u
under ʌ n d ɝ
until ʌ n t ɪ l
up ʌ p
upon ə p ɑ n
us ʌ s
very v ɛ ɹ i
voice v ɔɪ s
wait w eɪ t
walk w ɔ k
wanna w ɑ n ə
want w ɑ n t
warm w ɔ ɹ m
was w ʌ z
watch w ɑ tʃ
water w ɔ t ɝ
way w eɪ
we w i
we're w ɪ ɹ
were w ɝ
what w ʌ t
when w ɛ n
where w ɛ ɹ
which w ɪ tʃ
while w aɪ l
white w aɪ t
who h u
whole h oʊ l
why w aɪ
will w ɪ l
wind w ɪ n d
with w ɪ ð
without w ɪ ð aʊ t
woman w ʊ m ə n
won't w oʊ n t
wonder w ʌ n d ɝ
word w ɝ d
words w ɝ d z
work w ɝ k
world w ɝ l d
would w ʊ d
yeah j ɛ ə
year j ɪ ɹ
yes j ɛ s
you j u
young j ʌ ŋ
your j ɔ ɹ
you're j ʊ ɹ
"""


def _parse_lexicon(text):
    lex = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            lex[parts[0]] = parts[1:]
    return lex


LEXICON = _parse_lexicon(_LEXICON_TEXT)


# ---------------------------------------------------------------------------
# Letter-to-sound rules
# ---------------------------------------------------------------------------
def _r(rules_text):
    """Parse the compact rule notation ``left|letters|right -> p h o n e s``."""
    rules = []
    for line in rules_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        context, _, phones = line.partition("->")
        left, letters, right = context.split("|")
        rules.append((left.strip(), letters, right.strip(), phones.split()))
    return rules


# Rules are grouped by their first letter; within a group order is priority.
# An empty context matches anything; "$" is a word boundary.
_RULES_TEXT = {
"a": """
$|are|$-> ɑ ɹ
$|ar|o -> ə ɹ
|arr|$-> ɑ ɹ
$|arr| -> ə ɹ
|arr| -> æ ɹ
|air| -> ɛ ɹ
|ar|# -> ɛ ɹ
|ar| -> ɑ ɹ
|a|wa -> ə
|aw| -> ɔ
|au| -> ɔ
|ai| -> eɪ
|ay| -> eɪ
|a|^+# -> eɪ
|a|^% -> eɪ
|alk| -> ɔ k
#:|al|$-> ə l
$|al|# -> ə l
|al|^ -> ɔ l
// unstressed initial a-: about, above, away, alone, amaze
$|a|^# -> ə
|ang|+ -> eɪ n dʒ
|a|^^ -> æ
|a|^# -> eɪ
|a| -> æ
""",
"b": """
$|be|^# -> b ɪ
|being| -> b i ɪ ŋ
|buil| -> b ɪ l
|bb| -> b
|b| -> b
""",
"c": """
$|ch|^ -> k
|ch| -> tʃ
|ci|a -> ʃ
|ci|o -> ʃ
|ci|en -> ʃ
|ck| -> k
|cc|+ -> k s
|cc| -> k
|c|+ -> s
|c| -> k
""",
"d": """
|dd| -> d
t|ed|$-> ɪ d
d|ed|$-> ɪ d
p|ed|$-> t
k|ed|$-> t
f|ed|$-> t
s|ed|$-> t
x|ed|$-> t
h|ed|$-> t
#:^|ed|$-> d
|de|^# -> d ɪ
|du|a -> dʒ u
|d| -> d
""",
"e": """
#:|e|$->
#:|es|$-> z
#:&|es|$-> ɪ z
$:|e|$-> i
|ever| -> ɛ v ɝ
|ee| -> i
|eigh| -> eɪ
|ei| -> i
|earn| -> ɝ n
$|ear|^ -> ɝ
|ead| -> ɛ d
|ea|su -> ɛ
|ea| -> i
|ew| -> u
|eu| -> ju
|ey| -> i
#:|ely|$-> l i
#:|ement| -> m ə n t
|eful| -> f ʊ l
|er|# -> ɛ ɹ
|er| -> ɝ
|e|^% -> i
|e|o -> i
|e| -> ɛ
""",
"f": """
|full| -> f ʊ l
|ff| -> f
|f| -> f
""",
"g": """
|giv| -> g ɪ v
$|g|i^ -> g
|gg| -> g
|gh|$->
|gh|^ ->
|g|+ -> dʒ
|g| -> g
""",
"h": """
|h|# -> h
|h| ->
""",
"i": """
$|i|$-> aɪ
|igh| -> aɪ
|ild| -> aɪ l d
|ign|$-> aɪ n
|ign|^ -> aɪ n
|ier| -> i ɝ
|ied|$-> aɪ d
|ien| -> i ɛ n
|ie|$-> i
|ie| -> i
// i before another vowel is its own syllable: lion, quiet, trial
|i|# -> aɪ
// magic e: rise, shine, time
|i|^% -> aɪ
|ir|# -> aɪ ɹ
|ir| -> ɝ
|ique| -> i k
|i|^+:# -> ɪ
#:^|i|^+ -> ɪ
|i|^+ -> aɪ
|i|^^ -> ɪ
|i|^$-> ɪ
|i| -> ɪ
""",
"j": """
|j| -> dʒ
""",
"k": """
$|k|n ->
|kk| -> k
|k| -> k
""",
"l": """
|ll| -> l
#:^|l|e$-> ə l
|l| -> l
""",
"m": """
|mm| -> m
|m| -> m
""",
"n": """
e|ng|+ -> n dʒ
|ngl|$-> ŋ g ə l
|ng|# -> ŋ g
|ng| -> ŋ
|nk| -> ŋ k
|nn| -> n
|n| -> n
""",
"o": """
|oo|k -> ʊ
|oo|d -> ʊ
|oo| -> u
|ough|t -> ɔ
|ough| -> ʌ f
|oul|d -> ʊ
|our| -> ɔ ɹ
|ous| -> ə s
|ou| -> aʊ
|ow| -> oʊ
|oy| -> ɔɪ
|oi| -> ɔɪ
|oa| -> oʊ
|oe|$-> oʊ
#:|or|$-> ɝ
|or| -> ɔ ɹ
|ol|d -> oʊ l
|o|^% -> oʊ
|o|^en -> oʊ
|o|ng -> ɔ
|o|ss$-> ɔ
|o|^^ -> ɑ
$:^|o|$-> oʊ
|o|$-> oʊ
|o|e -> oʊ
|o| -> ɑ
""",
"p": """
|ph| -> f
|peop| -> p i p
|pp| -> p
|p| -> p
""",
"q": """
|quar| -> k w ɔ ɹ
|qu| -> k w
|q| -> k
""",
"r": """
$|re|^# -> ɹ i
|rr| -> ɹ
|r| -> ɹ
""",
"s": """
|sh| -> ʃ
|sch| -> s k
#|sion| -> ʒ ə n
|sion| -> ʃ ə n
#|sur|# -> ʒ ɝ
|sur|# -> ʃ ɝ
#|su|# -> ʒ u
|ss| -> s
#|sed|$-> z d
|s|$-> z
#|s|# -> z
$|sm|$-> z ə m
|s| -> s
""",
"t": """
|th|$-> θ
|the|$-> ð ə
|th| -> θ
#:|ted|$-> t ɪ d
s|ti|#n -> tʃ
|tion| -> ʃ ə n
|ti|o -> ʃ
|ti|a -> ʃ
|tur|# -> tʃ ɝ
|tu|a -> tʃ u
|tt| -> t
|t| -> t
""",
"u": """
$|un|i -> j u n
$|un| -> ʌ n
@|ur|# -> ʊ ɹ
|ur|# -> j ʊ ɹ
|ur| -> ɝ
|uy| -> aɪ
g|u|# -> w
g|u|% ->
|u|^^ -> ʌ
|u|^$-> ʌ
#n|u| -> ju
@|u| -> u
|u| -> ju
""",
"v": """
|view| -> v ju
|v| -> v
""",
"w": """
|who| -> h u
|wh| -> w
|wr| -> ɹ
|wa|s -> w ɑ
|wa|t -> w ɑ
|war| -> w ɔ ɹ
|wor|^ -> w ɝ
|w| -> w
""",
"x": """
$|x| -> z
|x| -> k s
""",
"y": """
$|y| -> j
|ying| -> aɪ ɪ ŋ
$:^|y|$-> aɪ
#:^|y|$-> i
#:^|y|# -> i
|y|$-> i
|y|^^ -> ɪ
|y| -> aɪ
""",
"z": """
|zz| -> z
|z| -> z
""",
}

RULES = {letter: _r(text) for letter, text in _RULES_TEXT.items()}


# ---------------------------------------------------------------------------
# Context matching
# ---------------------------------------------------------------------------
def _is_suffix(word, i):
    rest = word[i:].rstrip()
    for suffix in SUFFIXES:
        if rest.startswith(suffix) and rest[len(suffix):].strip() == "":
            return len(suffix)
    return 0


def _match_right(pattern, word, i):
    """Does `pattern` match the word starting at index i?"""
    for ch in pattern:
        if ch == "#":
            if i >= len(word) or word[i] not in VOWEL_LETTERS:
                return False
            while i < len(word) and word[i] in VOWEL_LETTERS:
                i += 1
        elif ch == ":":
            while i < len(word) and word[i] in CONSONANT_LETTERS:
                i += 1
        elif ch == "^":
            if i >= len(word) or word[i] not in CONSONANT_LETTERS:
                return False
            i += 1
        elif ch == ".":
            if i >= len(word) or word[i] not in VOICED_LETTERS:
                return False
            i += 1
        elif ch == "+":
            if i >= len(word) or word[i] not in FRONT_VOWELS:
                return False
            i += 1
        elif ch == "&":
            if i >= len(word) or word[i] not in SIBILANTS:
                return False
            i += 1
        elif ch == "@":
            if i >= len(word) or word[i] not in PLAIN_U_AFTER:
                return False
            i += 1
        elif ch == "%":
            n = _is_suffix(word, i)
            if not n:
                return False
            i += n
        elif ch == "$":
            if i >= len(word) or word[i] != " ":
                return False
            i += 1
        else:
            if i >= len(word) or word[i] != ch:
                return False
            i += 1
    return True


def _match_left(pattern, word, i):
    """Does `pattern` match the word ending just before index i?"""
    j = i - 1
    for ch in reversed(pattern):
        if ch == "#":
            if j < 0 or word[j] not in VOWEL_LETTERS:
                return False
            while j >= 0 and word[j] in VOWEL_LETTERS:
                j -= 1
        elif ch == ":":
            while j >= 0 and word[j] in CONSONANT_LETTERS:
                j -= 1
        elif ch == "^":
            if j < 0 or word[j] not in CONSONANT_LETTERS:
                return False
            j -= 1
        elif ch == ".":
            if j < 0 or word[j] not in VOICED_LETTERS:
                return False
            j -= 1
        elif ch == "+":
            if j < 0 or word[j] not in FRONT_VOWELS:
                return False
            j -= 1
        elif ch == "&":
            if j < 0 or word[j] not in SIBILANTS:
                return False
            j -= 1
        elif ch == "@":
            if j < 0 or word[j] not in PLAIN_U_AFTER:
                return False
            j -= 1
        elif ch == "$":
            if j < 0 or word[j] != " ":
                return False
            j -= 1
        else:
            if j < 0 or word[j] != ch:
                return False
            j -= 1
    return True


def rules_to_phones(word):
    """Apply the letter-to-sound rules to a bare lowercase word."""
    padded = " " + word + " "
    phones = []
    i = 1
    while i < len(padded) - 1:
        letter = padded[i]
        matched = False
        for left, letters, right, out in RULES.get(letter, ()):
            if not padded.startswith(letters, i):
                continue
            if not _match_left(left, padded, i):
                continue
            if not _match_right(right, padded, i + len(letters)):
                continue
            phones.extend(out)
            i += len(letters)
            matched = True
            break
        if not matched:
            i += 1  # unknown character (digit, stray symbol): skip silently
    return phones


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z']+")


def word_to_phones(word):
    """IPA phones for one word.  Lexicon first, then the rules."""
    key = word.lower().strip()
    key = key.strip(".,!?;:\"()[]")
    if not key:
        return []
    if key in LEXICON:
        return list(LEXICON[key])
    # Regular inflections of a known stem: "dreaming" -> "dream" + ing.
    for suffix, extra in (("ing", ["ɪ", "ŋ"]), ("s", None), ("'s", None),
                          ("ed", None), ("in'", ["ɪ", "n"])):
        if key.endswith(suffix) and key[:-len(suffix)] in LEXICON:
            stem = list(LEXICON[key[:-len(suffix)]])
            if extra is not None:
                return stem + extra
            if suffix in ("s", "'s"):
                last = stem[-1] if stem else ""
                if last in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ"):
                    return stem + ["ɪ", "z"]
                return stem + (["s"] if last in ("p", "t", "k", "f", "θ")
                               else ["z"])
            last = stem[-1] if stem else ""
            if last in ("t", "d"):
                return stem + ["ɪ", "d"]
            return stem + (["t"] if last in ("p", "k", "f", "s", "ʃ", "tʃ", "θ")
                           else ["d"])
    return rules_to_phones(key.replace("'", ""))


def text_to_phones(text):
    """[(word, [phones])] for a line of text."""
    return [(w, word_to_phones(w)) for w in _WORD_RE.findall(text.lower())]


def phonemize(text):
    """Flat phone list for a line of text (word boundaries dropped)."""
    out = []
    for _word, phones in text_to_phones(text):
        out.extend(phones)
    return out


# ---------------------------------------------------------------------------
# Syllables
# ---------------------------------------------------------------------------
# Consonant clusters that may begin an English syllable; used by the maximal
# onset principle when splitting a word between two vowels.
LEGAL_ONSETS = {
    "st", "sp", "sk", "sm", "sn", "sl", "sw", "tw", "kw", "dw", "gw",
    "pl", "bl", "kl", "gl", "fl", "sl", "pr", "br", "tr", "dr", "kr", "gr",
    "fr", "θr", "ʃr", "spr", "str", "skr", "spl", "skw",
    "pɹ", "bɹ", "tɹ", "dɹ", "kɹ", "gɹ", "fɹ", "θɹ", "ʃɹ", "spɹ", "stɹ", "skɹ",
    "hj", "kj", "bj", "fj", "mj", "pj", "vj",
}

_VOWELISH = None


def _is_nucleus(phone):
    global _VOWELISH
    if _VOWELISH is None:
        from . import mouth_shapes  # local import keeps this module standalone
        _VOWELISH = mouth_shapes.VOWELS
    return phone in _VOWELISH


def syllabify(phones):
    """Split a phone list into syllables (one vowel nucleus each), applying
    the maximal onset principle to the consonants between nuclei."""
    nuclei = [i for i, p in enumerate(phones) if _is_nucleus(p)]
    if len(nuclei) <= 1:
        return [list(phones)] if phones else []

    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phones[a + 1:b]
        if not cluster:
            onset = 0
        elif len(cluster) == 1:
            onset = 1
        else:
            onset = 1
            for size in range(min(3, len(cluster)), 1, -1):
                if "".join(cluster[-size:]) in LEGAL_ONSETS:
                    onset = size
                    break
        cuts.append(b - onset)

    syllables, start = [], 0
    for cut in cuts:
        syllables.append(phones[start:cut])
        start = cut
    syllables.append(phones[start:])
    return [s for s in syllables if s]


def split_into(phones, count):
    """Force a phone list into exactly `count` syllables - the lyric's
    hyphenation is authoritative, our syllabifier is only an estimate."""
    syllables = syllabify(phones)
    if not syllables:
        return [[] for _ in range(count)]
    while len(syllables) > count:          # merge the shortest neighbour pair
        i = min(range(len(syllables) - 1),
                key=lambda k: len(syllables[k]) + len(syllables[k + 1]))
        syllables[i:i + 2] = [syllables[i] + syllables[i + 1]]
    while len(syllables) < count:          # no vowels left to split on: hold
        nucleus = [p for p in syllables[-1] if _is_nucleus(p)]
        syllables.append(nucleus[-1:] or syllables[-1][-1:])
    return syllables


# ---------------------------------------------------------------------------
# Optional CMUdict
# ---------------------------------------------------------------------------
ARPABET_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def load_cmudict(path=None):
    """Merge a CMUdict-format file into LEXICON.  Returns the entries added.

    Unstressed schwa is kept as written ("AH0" -> ʌ); the mouth cannot tell
    ʌ and ə apart anyway, they share the EH viseme.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cmudict.txt")
    if not os.path.exists(path):
        return 0
    added = 0
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if word.endswith(")"):        # alternate pronunciation: skip
                continue
            phones = [ARPABET_IPA.get(p.rstrip("012"), "") for p in parts[1:]]
            phones = [p for p in phones if p]
            if phones:
                LEXICON.setdefault(word, phones)
                added += 1
    return added


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
_SELFTEST = [
    # (word, expected IPA)  - covers the lexicon, the rules, and inflections.
    ("the", "ð ə"),
    ("love", "l ʌ v"),
    ("heart", "h ɑ ɹ t"),
    ("twinkle", "t w ɪ ŋ k ə l"),
    ("star", "s t ɑ ɹ"),
    ("how", "h aʊ"),
    ("what", "w ʌ t"),
    ("you", "j u"),
    ("dreaming", "d ɹ i m ɪ ŋ"),      # inflection of a lexicon stem
    ("dreams", "d ɹ i m z"),
    ("sings", "s ɪ ŋ z"),
    ("wanted", "w ɑ n t ɪ d"),
    # rule-driven, out of lexicon
    ("bright", "b ɹ aɪ t"),
    ("shine", "ʃ aɪ n"),
    ("sunrise", "s ʌ n ɹ aɪ z"),
    ("little", "l ɪ t ə l"),
    ("gentle", "dʒ ɛ n t ə l"),
    ("lion", "l aɪ ɑ n"),
]


def selftest():
    failures = 0
    for word, expected in _SELFTEST:
        got = " ".join(word_to_phones(word))
        ok = got == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {word:12s} {got}"
              + ("" if ok else f"   (expected {expected})"))
    # Syllabification must survive a round trip.
    for word, count in (("twinkle", 2), ("forever", 3), ("love", 1),
                        ("beautiful", 3)):
        syls = split_into(word_to_phones(word), count)
        joined = [p for s in syls for p in s]
        ok = len(syls) == count and joined == word_to_phones(word)
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {word:12s} -> "
              + " / ".join(" ".join(s) for s in syls))
    print("all passed" if not failures else f"{failures} failure(s)")
    return failures


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(1 if selftest() else 0)
    text = " ".join(sys.argv[1:]) or "Twinkle twinkle little star"
    for word, phones in text_to_phones(text):
        syls = syllabify(phones)
        print(f"{word:14s} {' '.join(phones):28s} "
              f"[{' / '.join(' '.join(s) for s in syls)}]")
