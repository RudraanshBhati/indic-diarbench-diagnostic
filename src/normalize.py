"""Text normalization, script detection, and code-mixing measurement (§3.6, §5.2).

The dataset's transcripts are native-script with Roman-script English inline, and
carry annotator tags such as ``<noise>``. Getting normalization wrong silently
inflates WER — see CLAUDE.md "Pitfalls".
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from functools import lru_cache

from constants import SCRIPT_RANGES

# Bracketed spans serve two distinct purposes in this corpus, and conflating them
# corrupts WER. Measured over six languages:
#
#   [ ]  12 tag types / 7,930 occurrences   vs  434 other types / 852 occurrences
#        ...of which 851 of 852 are Latin script — i.e. spoken English words that
#        annotators bracketed to mark code-switching.
#   < >  almost purely tags, plus misspellings of "unintelligible".
#
# Stripping every bracket therefore deletes real English words from the reference,
# precisely in the code-mixed regions this study is about. So: bracketed spans
# matching the tag vocabulary are removed; everything else is unwrapped and kept,
# and doubles as a human-annotated code-switch label (§3.6).
BRACKET_PATTERN = re.compile(r"[<\[\{]([^<>\[\]\{\}]{0,60})[>\]\}]")

CANONICAL_TAGS = {
    "laughter", "laugh", "noise", "unintelligible", "inaudible", "vocalization",
    "stutters", "stutter", "coughing", "cough", "gasps", "gasp", "clears throat",
    "background speech", "background_speech", "music", "silence", "sighs", "sigh",
    "sneeze", "sneezes", "breathes", "breath", "whispers", "cry", "crying",
    "applause", "beep", "static", "unk",
}

# Typos are common in the tag vocabulary; matched by similarity rather than
# enumerated. Cutoff is deliberately high so content words are never absorbed.
_TAG_SIMILARITY_CUTOFF = 0.82


@lru_cache(maxsize=8192)
def is_tag(content: str) -> bool:
    """Whether a bracketed span is an annotator tag rather than spoken content."""
    norm = re.sub(r"[\s_]+", " ", content.strip().lower())
    if not norm:
        return True
    if norm in CANONICAL_TAGS:
        return True
    return bool(difflib.get_close_matches(norm, CANONICAL_TAGS, n=1,
                                          cutoff=_TAG_SIMILARITY_CUTOFF))

# Punctuation to drop before word-level scoring. Includes Indic danda/double danda
# and the Arabic-script punctuation used by Urdu/Kashmiri/Sindhi.
_PUNCT_EXTRA = "।॥،؛؟۔–—‘’“”"


@lru_cache(maxsize=4096)
def _char_script(ch: str) -> str | None:
    """Return the script name for a character, or None if it carries no script."""
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def strip_tags(text: str) -> tuple[str, list[str]]:
    """Remove annotator tags, keeping bracketed *content* words.

    Returns ``(clean_text, tags_removed)``. A bracketed span is dropped only if
    :func:`is_tag` recognizes it; otherwise the brackets are unwrapped and the
    word survives into the transcript.
    """
    found: list[str] = []

    def repl(match: re.Match) -> str:
        inner = match.group(1)
        if is_tag(inner):
            found.append(match.group(0))
            return " "
        return f" {inner} "

    return BRACKET_PATTERN.sub(repl, text), found


def code_switch_spans(text: str) -> list[str]:
    """Bracketed spans the annotators marked as spoken content (§3.6).

    These are human-labelled code-switch tokens — overwhelmingly English — and
    are a stronger signal than script detection alone.
    """
    return [m.group(1).strip() for m in BRACKET_PATTERN.finditer(text)
            if not is_tag(m.group(1)) and m.group(1).strip()]


def normalize(text: str, *, drop_punct: bool = True) -> str:
    """Normalization applied identically to reference and hypothesis before WER.

    NFC-normalize, strip annotator tags, drop punctuation, lowercase Latin,
    collapse whitespace. Digits are left as-is; the dataset already normalizes
    numerals to Arabic digits.
    """
    if not text:
        return ""
    out, _ = strip_tags(text)
    out = unicodedata.normalize("NFC", out)
    if drop_punct:
        chars = []
        for ch in out:
            if ch in _PUNCT_EXTRA:
                chars.append(" ")
            elif unicodedata.category(ch).startswith("P"):
                chars.append(" ")
            else:
                chars.append(ch)
        out = "".join(chars)
    out = out.lower()
    return re.sub(r"\s+", " ", out).strip()


def token_script(token: str) -> str | None:
    """Dominant script of a token, or None for script-less tokens (digits, symbols).

    Script-less tokens are 'language-independent' in the CMI sense and are
    excluded from the code-mixing denominator.
    """
    counts: dict[str, int] = {}
    for ch in token:
        s = _char_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def code_mixing_index(text: str, native_script: str) -> tuple[float, int, int, int]:
    """Code-Mixing Index over {native script, Latin} (§3.6).

    Follows Gambäck & Das (2014) reduced to two languages::

        CMI = 100 * (1 - max(w_native, w_latin) / (n - u))

    where ``u`` is the count of language-independent tokens, which are excluded
    from the denominator. Returns (cmi, n_native, n_latin, n_neutral).

    CMI is 0 for monolingual text and rises toward 50 as the two scripts
    approach parity.
    """
    clean = normalize(text)
    if not clean:
        return 0.0, 0, 0, 0

    n_native = n_latin = n_neutral = 0
    for tok in clean.split():
        s = token_script(tok)
        if s is None:
            n_neutral += 1
        elif s == "Latin":
            n_latin += 1
        elif s == native_script:
            n_native += 1
        else:
            # A third script (e.g. stray Devanagari in a Tamil clip). Count it as
            # native-side so it cannot masquerade as English code-mixing.
            n_native += 1

    denom = n_native + n_latin
    if denom == 0:
        return 0.0, n_native, n_latin, n_neutral
    cmi = 100.0 * (1.0 - max(n_native, n_latin) / denom)
    return cmi, n_native, n_latin, n_neutral
