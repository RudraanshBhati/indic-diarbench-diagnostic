"""Hand-computed cases for the overlap logic behind the Stage 0 gate.

Every DER number in the study depends on segment/timeline handling being right,
so these fix the semantics with cases whose answers can be verified by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rttm import Segment, is_overlapping, overlap_stats, segments_from_record  # noqa: E402


def test_no_overlap_two_speakers():
    segs = [Segment("A", 0.0, 10.0), Segment("B", 10.0, 20.0)]
    st = overlap_stats(segs)
    assert st["speech_union"] == pytest.approx(20.0)
    assert st["speech_sum"] == pytest.approx(20.0)
    assert st["overlap_duration"] == pytest.approx(0.0)
    assert st["overlap_ratio"] == pytest.approx(0.0)


def test_simple_overlap():
    # A 0-10, B 5-15 -> union 15, sum 20, overlap 5
    segs = [Segment("A", 0.0, 10.0), Segment("B", 5.0, 15.0)]
    st = overlap_stats(segs)
    assert st["speech_union"] == pytest.approx(15.0)
    assert st["speech_sum"] == pytest.approx(20.0)
    assert st["overlap_duration"] == pytest.approx(5.0)
    assert st["overlap_ratio"] == pytest.approx(5.0 / 15.0)
    assert st["overlap_ratio_sum"] == pytest.approx(5.0 / 20.0)


def test_same_speaker_overlap_is_not_overlap():
    """A speaker overlapping themselves is an annotation artifact, not overlap."""
    segs = [Segment("A", 0.0, 10.0), Segment("A", 5.0, 15.0)]
    st = overlap_stats(segs)
    assert st["overlap_duration"] == pytest.approx(0.0)
    assert st["speech_union"] == pytest.approx(15.0)
    assert st["speech_sum"] == pytest.approx(15.0)  # self-union, not 20
    assert st["n_speakers_seen"] == 1


def test_three_way_overlap_counted_once():
    """Depth 3 is still overlap, not counted twice."""
    segs = [Segment("A", 0.0, 10.0), Segment("B", 0.0, 10.0), Segment("C", 0.0, 10.0)]
    st = overlap_stats(segs)
    assert st["speech_union"] == pytest.approx(10.0)
    assert st["overlap_duration"] == pytest.approx(10.0)
    assert st["overlap_ratio"] == pytest.approx(1.0)
    assert st["speech_sum"] == pytest.approx(30.0)


def test_nested_segment():
    # B fully inside A -> overlap is B's duration
    segs = [Segment("A", 0.0, 20.0), Segment("B", 5.0, 8.0)]
    st = overlap_stats(segs)
    assert st["speech_union"] == pytest.approx(20.0)
    assert st["overlap_duration"] == pytest.approx(3.0)


def test_disjoint_islands_with_gap():
    segs = [Segment("A", 0.0, 5.0), Segment("B", 100.0, 105.0)]
    st = overlap_stats(segs)
    assert st["speech_union"] == pytest.approx(10.0)
    assert st["overlap_duration"] == pytest.approx(0.0)


def test_empty():
    st = overlap_stats([])
    assert st["speech_union"] == 0.0
    assert st["overlap_ratio"] == 0.0


def test_is_overlapping():
    segs = [Segment("A", 0.0, 10.0), Segment("B", 5.0, 15.0), Segment("C", 50.0, 60.0)]
    assert is_overlapping(segs[0], segs) is True
    assert is_overlapping(segs[1], segs) is True
    assert is_overlapping(segs[2], segs) is False


def test_segments_from_record_drops_degenerate():
    rec = {"annotated_transcript": [
        {"speaker_id": "A", "transcript": "hi", "start_time": 0.0, "end_time": 1.0},
        {"speaker_id": "B", "transcript": "x", "start_time": 5.0, "end_time": 5.0},   # zero
        {"speaker_id": "C", "transcript": "y", "start_time": 9.0, "end_time": 8.0},   # negative
    ]}
    segs = segments_from_record(rec)
    assert len(segs) == 1
    assert segs[0].speaker == "A"


def test_segments_sorted_by_start():
    rec = {"annotated_transcript": [
        {"speaker_id": "B", "transcript": "", "start_time": 10.0, "end_time": 12.0},
        {"speaker_id": "A", "transcript": "", "start_time": 1.0, "end_time": 2.0},
    ]}
    segs = segments_from_record(rec)
    assert [s.start for s in segs] == [1.0, 10.0]


def test_segments_from_record_numpy_array_column():
    """pandas hands back a numpy object array, not a list — must not raise."""
    import numpy as np
    rec = {"annotated_transcript": np.array([
        {"speaker_id": "A", "transcript": "hi", "start_time": 0.0, "end_time": 1.0},
        {"speaker_id": "B", "transcript": "yo", "start_time": 0.5, "end_time": 2.0},
    ], dtype=object)}
    segs = segments_from_record(rec)
    assert len(segs) == 2
    assert overlap_stats(segs)["overlap_duration"] == pytest.approx(0.5)


def test_segments_from_record_empty_array():
    import numpy as np
    assert segments_from_record({"annotated_transcript": np.array([], dtype=object)}) == []
    assert segments_from_record({"annotated_transcript": None}) == []
    assert segments_from_record({}) == []
