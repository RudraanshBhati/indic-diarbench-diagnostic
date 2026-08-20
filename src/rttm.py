"""Reference/hypothesis timeline handling: overlap computation and RTTM I/O.

The dataset ships no RTTM (see CLAUDE.md). Every DER number downstream depends on
this module being correct, which is what the Stage 0 overlap gate validates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def segments_from_record(record: dict) -> list[Segment]:
    """Build segments from a dataset row's ``annotated_transcript`` list.

    The field arrives as a numpy object array when the row comes from pandas, so
    emptiness is tested by length rather than truthiness.
    """
    raw = record.get("annotated_transcript")
    if raw is None or len(raw) == 0:
        return []

    segs = []
    for s in raw:
        start, end = float(s["start_time"]), float(s["end_time"])
        if end <= start:
            continue  # zero/negative-length annotation artifact
        segs.append(Segment(str(s["speaker_id"]), start, end, s.get("transcript") or ""))
    return sorted(segs, key=lambda x: (x.start, x.end))


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals."""
    if not intervals:
        return []
    out = [list(iv) for iv in sorted(intervals)]
    merged = [out[0]]
    for s, e in out[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _total(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in intervals)


def overlap_stats(segments: list[Segment]) -> dict[str, float]:
    """Compute speech and overlap durations (§3.5).

    Overlap means *two or more distinct speakers* active at the same instant.
    Same-speaker segments are unioned first, so a speaker's own adjacent or
    slightly-overlapping annotations never count as overlap.

    Two denominators are reported because the docx formula ("overlapping speech /
    total speech duration") is ambiguous:

    - ``speech_union``: wall-clock time with >=1 speaker active
    - ``speech_sum``: sum of per-speaker speaking time (double-counts overlap)

    ``overlap_ratio`` uses ``speech_union``; ``overlap_ratio_sum`` uses
    ``speech_sum``. The Stage 0 gate decides which reproduces the published
    figures.
    """
    if not segments:
        return dict(speech_union=0.0, speech_sum=0.0, overlap_duration=0.0,
                    overlap_ratio=0.0, overlap_ratio_sum=0.0, n_speakers_seen=0)

    by_spk: dict[str, list[tuple[float, float]]] = {}
    for s in segments:
        by_spk.setdefault(s.speaker, []).append((s.start, s.end))

    per_spk = {k: _merge(v) for k, v in by_spk.items()}
    speech_sum = sum(_total(v) for v in per_spk.values())

    all_iv = [iv for v in per_spk.values() for iv in v]
    union = _merge(all_iv)
    speech_union = _total(union)

    # Sweep line over per-speaker (already self-unioned) intervals; regions where
    # depth >= 2 are multi-speaker overlap.
    events: list[tuple[float, int]] = []
    for v in per_spk.values():
        for s, e in v:
            events.append((s, 1))
            events.append((e, -1))
    events.sort()

    overlap = 0.0
    depth = 0
    prev = events[0][0]
    for t, delta in events:
        if depth >= 2:
            overlap += t - prev
        depth += delta
        prev = t

    return dict(
        speech_union=speech_union,
        speech_sum=speech_sum,
        overlap_duration=overlap,
        overlap_ratio=overlap / speech_union if speech_union > 0 else 0.0,
        overlap_ratio_sum=overlap / speech_sum if speech_sum > 0 else 0.0,
        n_speakers_seen=len(per_spk),
    )


def is_overlapping(seg: Segment, segments: list[Segment]) -> bool:
    """Whether ``seg`` shares any time with a *different* speaker (§4.3)."""
    for other in segments:
        if other.speaker == seg.speaker:
            continue
        if seg.start < other.end and other.start < seg.end:
            return True
    return False


def write_rttm(segments: list[Segment], uri: str, path: Path) -> Path:
    """Write NIST RTTM. One SPEAKER line per segment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in segments:
            fh.write(
                f"SPEAKER {uri} 1 {s.start:.3f} {s.duration:.3f} "
                f"<NA> <NA> {s.speaker} <NA> <NA>\n"
            )
    return path


def to_pyannote(segments: list[Segment]):
    """Convert to a ``pyannote.core.Annotation`` for DER scoring (§5.1)."""
    from pyannote.core import Annotation, Segment as PSegment

    ann = Annotation()
    for i, s in enumerate(segments):
        ann[PSegment(s.start, s.end), i] = s.speaker
    return ann
