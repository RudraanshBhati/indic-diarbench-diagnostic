"""Metric tests (§5), with answers derived by hand before the code was run.

WDER has no reference implementation in any library — it is implemented from
El Shafey et al. (2019), the definition the benchmark paper cites as [18]. That
makes it the most dangerous code in the repo, so it is pinned hardest here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rttm import Segment  # noqa: E402
from score import (  # noqa: E402
    cp_wer, diarization_error_rate, score_clip, word_diarization_error_rate,
    word_error_rate, words_from_segments,
)

# --------------------------------------------------------------------------- #
# The worked example.
#
#   reference                              hypothesis
#   [0-2] Alice: we should ship on friday  [0-2] SPK1: we should ship on friday
#   [2-4] Bob:   i disagree lets wait      [2-4] SPK1: i disagree lets wait
#
# Every word is transcribed correctly, but the system heard only one speaker.
# By hand:
#   WER   = 0/9                    -> 0.0    (words are perfect)
#   DER   = 2s confusion / 4s      -> 0.5    (one speaker's time misattributed)
#   WDER  = 4 of 9 correct words carrying the wrong speaker -> 0.444
# This gap between WER and WDER is the phenomenon the whole study is about.
# --------------------------------------------------------------------------- #

REF = [
    Segment("Alice", 0.0, 2.0, "we should ship on friday"),
    Segment("Bob", 2.0, 4.0, "i disagree lets wait"),
]
HYP_MERGED = [
    Segment("SPK1", 0.0, 2.0, "we should ship on friday"),
    Segment("SPK1", 2.0, 4.0, "i disagree lets wait"),
]


def test_worked_example_wer_is_zero():
    assert word_error_rate(
        "we should ship on friday i disagree lets wait",
        "we should ship on friday i disagree lets wait",
    )["wer"] == pytest.approx(0.0)


def test_worked_example_der_is_half_all_confusion():
    d = diarization_error_rate(REF, HYP_MERGED)
    assert d["der"] == pytest.approx(0.5)
    assert d["confusion"] == pytest.approx(0.5)
    assert d["miss"] == pytest.approx(0.0)
    assert d["false_alarm"] == pytest.approx(0.0)


def test_worked_example_wder_is_four_ninths():
    cp = cp_wer(REF, HYP_MERGED)
    w = word_diarization_error_rate(
        words_from_segments(REF), words_from_segments(HYP_MERGED), cp["assignment"]
    )
    assert w["n_scored"] == 9
    assert w["n_correct"] == 9
    assert w["n_correct_wrong_spk"] == 4
    assert w["wder"] == pytest.approx(4 / 9)


def test_worked_example_end_to_end():
    """The headline contrast: perfect words, badly broken attribution."""
    s = score_clip(REF, HYP_MERGED).flat()
    assert s["wer"] == pytest.approx(0.0)
    assert s["der"] == pytest.approx(0.5)
    assert s["wder"] == pytest.approx(4 / 9)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

def test_der_components_sum_to_der():
    """The invariant score_clip asserts; the paper's Table 3 also satisfies it."""
    hyp = [Segment("X", 0.0, 1.5, "we should ship"), Segment("Y", 2.5, 4.5, "lets wait")]
    d = diarization_error_rate(REF, hyp)
    assert d["miss"] + d["false_alarm"] + d["confusion"] == pytest.approx(d["der"])


def test_perfect_system_scores_zero_everywhere():
    s = score_clip(REF, list(REF)).flat()
    assert s["der"] == pytest.approx(0.0)
    assert s["wer"] == pytest.approx(0.0)
    assert s["cpwer"] == pytest.approx(0.0)
    assert s["wder"] == pytest.approx(0.0)


def test_consistent_relabeling_is_free():
    """Renaming Alice->X and Bob->Y consistently is not an error.

    All three speaker-aware metrics resolve labels by optimal permutation, so a
    consistent relabeling must cost nothing. Only *inconsistent* attribution is
    a real error. If this fails, the permutation logic is broken.
    """
    hyp = [
        Segment("X", 0.0, 2.0, "we should ship on friday"),
        Segment("Y", 2.0, 4.0, "i disagree lets wait"),
    ]
    s = score_clip(REF, hyp).flat()
    assert s["der"] == pytest.approx(0.0)
    assert s["cpwer"] == pytest.approx(0.0)
    assert s["wder"] == pytest.approx(0.0)


def test_asr_errors_with_correct_speakers_leave_wder_clean():
    """WER and WDER must be able to move independently — that separation is the
    entire premise of the study."""
    hyp = [
        Segment("Alice", 0.0, 2.0, "we should skip on friday"),   # ship -> skip
        Segment("Bob", 2.0, 4.0, "i disagree lets wait"),
    ]
    s = score_clip(REF, hyp).flat()
    assert s["wer"] > 0.0
    assert s["der"] == pytest.approx(0.0)
    assert s["wder"] == pytest.approx(0.0)  # substitution, but speaker is right


# --------------------------------------------------------------------------- #
# DER components in isolation
# --------------------------------------------------------------------------- #

def test_pure_missed_speech():
    ref = [Segment("A", 0.0, 10.0, "a b c d")]
    hyp = [Segment("A", 0.0, 5.0, "a b")]
    d = diarization_error_rate(ref, hyp)
    assert d["miss"] == pytest.approx(0.5)
    assert d["false_alarm"] == pytest.approx(0.0)
    assert d["confusion"] == pytest.approx(0.0)


def test_pure_false_alarm():
    ref = [Segment("A", 0.0, 5.0, "a b")]
    hyp = [Segment("A", 0.0, 10.0, "a b c d")]
    d = diarization_error_rate(ref, hyp)
    assert d["false_alarm"] == pytest.approx(1.0)
    assert d["miss"] == pytest.approx(0.0)


def test_der_counts_overlapped_speech():
    """Paper §4: no collar, overlapping speech included. A system that emits only
    one of two simultaneous speakers must be penalised."""
    ref = [Segment("A", 0.0, 10.0, "x"), Segment("B", 0.0, 10.0, "y")]
    hyp = [Segment("A", 0.0, 10.0, "x")]
    d = diarization_error_rate(ref, hyp)
    assert d["der"] == pytest.approx(0.5)
    assert d["miss"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# WDER specifics
# --------------------------------------------------------------------------- #

def test_wder_excludes_deletions_and_insertions():
    """A word that was never recognized cannot be misattributed, so it is
    outside WDER's denominator."""
    ref_w = words_from_segments([Segment("A", 0.0, 4.0, "alpha beta gamma delta")])
    hyp_w = words_from_segments([Segment("X", 0.0, 2.0, "alpha beta")])
    w = word_diarization_error_rate(ref_w, hyp_w, {"A": "X"})
    assert w["n_scored"] == 2       # gamma/delta deleted, not scored
    assert w["wder"] == pytest.approx(0.0)


def test_wder_all_wrong_speaker():
    ref_w = words_from_segments([Segment("A", 0.0, 2.0, "one two")])
    hyp_w = words_from_segments([Segment("X", 0.0, 2.0, "one two")])
    w = word_diarization_error_rate(ref_w, hyp_w, {"A": "Y"})  # A maps to Y, not X
    assert w["wder"] == pytest.approx(1.0)
    assert w["n_correct_wrong_spk"] == 2


def test_wder_substitutions_are_scored():
    """Substituted words still count — the speaker label is still checkable."""
    ref_w = words_from_segments([Segment("A", 0.0, 2.0, "one two")])
    hyp_w = words_from_segments([Segment("X", 0.0, 2.0, "one three")])
    w = word_diarization_error_rate(ref_w, hyp_w, {"A": "X"})
    assert w["n_correct"] == 1 and w["n_sub"] == 1
    assert w["n_scored"] == 2
    assert w["wder"] == pytest.approx(0.0)


def test_wder_empty_reference():
    assert word_diarization_error_rate([], [], {})["wder"] == 0.0


def test_words_from_segments_orders_by_time_and_keeps_speaker():
    words = words_from_segments([
        Segment("B", 5.0, 6.0, "second"),
        Segment("A", 0.0, 1.0, "first"),
    ])
    assert [w.text for w in words] == ["first", "second"]
    assert [w.speaker for w in words] == ["A", "B"]


def test_words_from_segments_normalizes():
    words = words_from_segments([Segment("A", 0.0, 1.0, "Hello, <noise> World!")])
    assert [w.text for w in words] == ["hello", "world"]


# --------------------------------------------------------------------------- #
# Scored region (UEM)
# --------------------------------------------------------------------------- #

def test_false_alarm_outside_reference_extent_is_penalised():
    """Without an explicit scored region, pyannote approximates it from the data,
    so speech invented past the last reference segment can escape penalty.
    Passing the clip duration fixes the region to the whole clip."""
    ref = [Segment("A", 0.0, 5.0, "a b")]
    hyp = [Segment("A", 0.0, 5.0, "a b"), Segment("A", 6.0, 10.0, "invented")]
    d = diarization_error_rate(ref, hyp, duration=10.0)
    assert d["false_alarm"] == pytest.approx(4.0 / 5.0)
    assert d["der"] == pytest.approx(4.0 / 5.0)


def test_duration_does_not_change_a_clean_result():
    d_no = diarization_error_rate(REF, HYP_MERGED)
    d_yes = diarization_error_rate(REF, HYP_MERGED, duration=4.0)
    assert d_no["der"] == pytest.approx(d_yes["der"])
    assert d_yes["confusion"] == pytest.approx(0.5)


def test_leading_silence_does_not_inflate_denominator():
    """DER normalizes by reference *speech*, not clip length, so silence at the
    start must not dilute the rate."""
    ref = [Segment("A", 20.0, 30.0, "a b")]
    hyp = [Segment("B", 20.0, 30.0, "a b")]
    d = diarization_error_rate(ref, hyp, duration=60.0)
    assert d["total_ref_sec"] == pytest.approx(10.0)
    assert d["der"] == pytest.approx(0.0)  # consistent relabeling
