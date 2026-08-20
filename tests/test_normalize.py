"""Normalization and code-mixing tests (§3.6, §5.2).

Normalization mismatch is the documented way to silently inflate WER, so the
rules are pinned here rather than left implicit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from normalize import (  # noqa: E402
    code_mixing_index, normalize, strip_tags, token_script,
)


def test_strip_tags_removes_annotator_markers():
    clean, found = strip_tags("hello <noise> world [inaudible] ok")
    assert "noise" not in clean and "inaudible" not in clean
    assert set(found) == {"<noise>", "[inaudible]"}


def test_normalize_drops_punctuation_and_lowercases():
    assert normalize("Hello, World!") == "hello world"


def test_normalize_handles_danda():
    assert normalize("नमस्ते। कैसे हैं॥") == "नमस्ते कैसे हैं"


def test_normalize_collapses_whitespace():
    assert normalize("  a   b \n c ") == "a b c"


def test_normalize_empty():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_normalize_strips_tag_and_punct_together():
    assert normalize("यह <noise> है, ठीक।") == "यह है ठीक"


@pytest.mark.parametrize("token,expected", [
    ("hello", "Latin"),
    ("नमस्ते", "Devanagari"),
    ("বাংলা", "Bengali"),
    ("தமிழ்", "Tamil"),
    ("ಕನ್ನಡ", "Kannada"),
    ("اردو", "Arabic"),
    ("123", None),
    ("!!", None),
])
def test_token_script(token, expected):
    assert token_script(token) == expected


def test_cmi_monolingual_is_zero():
    cmi, nat, lat, _ = code_mixing_index("नमस्ते कैसे हैं आप", "Devanagari")
    assert cmi == pytest.approx(0.0)
    assert nat == 4 and lat == 0


def test_cmi_parity_is_fifty():
    # 2 native + 2 Latin -> 100*(1 - 2/4) = 50
    cmi, nat, lat, _ = code_mixing_index("नमस्ते कैसे hello world", "Devanagari")
    assert cmi == pytest.approx(50.0)
    assert nat == 2 and lat == 2


def test_cmi_light_mixing():
    # 3 native + 1 Latin -> 100*(1 - 3/4) = 25
    cmi, _, _, _ = code_mixing_index("यह मेरा specific काम", "Devanagari")
    assert cmi == pytest.approx(25.0)


def test_cmi_digits_are_language_independent():
    """Digits must not enter the CMI denominator."""
    cmi, nat, lat, neutral = code_mixing_index("नमस्ते 123 456", "Devanagari")
    assert neutral == 2
    assert nat == 1 and lat == 0
    assert cmi == pytest.approx(0.0)


def test_cmi_ignores_tags():
    cmi, nat, lat, _ = code_mixing_index("नमस्ते <noise> कैसे", "Devanagari")
    assert nat == 2 and lat == 0
    assert cmi == pytest.approx(0.0)


def test_cmi_third_script_counts_native_side():
    """A stray third script must not masquerade as English code-mixing."""
    cmi, nat, lat, _ = code_mixing_index("தமிழ் नमस्ते", "Tamil")
    assert lat == 0 and nat == 2
    assert cmi == pytest.approx(0.0)


def test_cmi_empty_text():
    assert code_mixing_index("", "Devanagari")[0] == 0.0


# --------------------------------------------------------------------------- #
# Bracket disambiguation.
#
# The corpus uses [ ] for BOTH annotator tags and code-switched English words.
# Measured over six languages: 12 tag types / 7,930 occurrences, versus 434
# other types / 852 occurrences of which 851 were Latin script. Stripping every
# bracket deletes real spoken English from the reference, biasing WER exactly on
# the code-mixed content the study is about.
# --------------------------------------------------------------------------- #

from normalize import code_switch_spans, is_tag  # noqa: E402


@pytest.mark.parametrize("content", [
    "laughter", "noise", "unintelligible", "vocalization", "stutters",
    "coughing", "gasps", "clears throat", "background speech",
    "background_speech", "sneeze", "  LAUGHTER  ", "",
])
def test_is_tag_accepts_annotation_vocabulary(content):
    assert is_tag(content) is True


@pytest.mark.parametrize("content", [
    "unintellegible", "unitelligible", "unintelligble",
])
def test_is_tag_tolerates_typos(content):
    """Annotators misspell tags; those are still tags, not spoken words."""
    assert is_tag(content) is True


@pytest.mark.parametrize("content", [
    "hello", "show", "confirm", "production", "heroine", "ready", "style",
    "modeling", "shoot", "doubt", "two", "city of temples",
])
def test_is_tag_rejects_spoken_english(content):
    assert is_tag(content) is False


def test_bracketed_english_survives_normalization():
    """The bug this guards: [confirm] is a spoken word, not an annotation."""
    assert normalize("मैंने [confirm] किया") == "मैंने confirm किया"


def test_bracketed_tag_is_removed():
    assert normalize("मैंने [laughter] किया") == "मैंने किया"


def test_mixed_tags_and_content_in_one_utterance():
    assert normalize("यह [noise] बहुत [style] है <unintelligible>") == "यह बहुत style है"


def test_code_switch_spans_extracts_only_content():
    spans = code_switch_spans("यह [noise] बहुत [style] और [confirm] है [laughter]")
    assert spans == ["style", "confirm"]


def test_code_switch_spans_empty_when_only_tags():
    assert code_switch_spans("यह <noise> [laughter] है") == []


def test_cmi_counts_bracketed_english():
    """Previously the brackets were stripped before CMI ran, understating
    code-mixing on exactly the utterances that had human-marked switches."""
    cmi, native, latin, _ = code_mixing_index("यह मेरा [specific] काम", "Devanagari")
    assert latin == 1 and native == 3
    assert cmi == pytest.approx(25.0)
