"""Metric computation (§5). Per-clip, never only aggregate.

All four metrics use reference implementations where one exists:

- DER + components : ``pyannote.metrics``
- WER              : ``jiwer``
- cpWER            : ``meeteval``
- WDER             : implemented here — no library provides it. Follows
  El Shafey, Soltau & Shafran (Interspeech 2019), the definition the benchmark
  paper cites as [18].

DER is configured to match the benchmark paper's §4 exactly: **no forgiveness
collar, overlapping speech included**. These are set explicitly rather than left
to library defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jiwer

from normalize import normalize
from rttm import Segment, to_pyannote

# Benchmark paper §4: "DER computed without a forgiveness collar and including
# overlapping speech." Do not change without changing the paper.
DER_COLLAR = 0.0
DER_SKIP_OVERLAP = False


@dataclass
class Word:
    """A word with its speaker label, and optionally its time."""
    text: str
    speaker: str
    start: float | None = None
    end: float | None = None


# --------------------------------------------------------------------------- #
# §5.1 Diarization Error Rate
# --------------------------------------------------------------------------- #

def diarization_error_rate(
    ref: list[Segment],
    hyp: list[Segment],
    duration: float | None = None,
) -> dict[str, float]:
    """DER with its Miss / False Alarm / Speaker Confusion decomposition (§4.2).

    Rates are normalized by total reference speech time, so
    ``miss + false_alarm + confusion == der`` up to floating point. Callers
    should assert that invariant — the benchmark paper's Table 3 satisfies it.

    ``duration`` sets the scored region (UEM) to the whole clip, ``[0, duration]``.
    Pass it whenever the clip length is known. Without it pyannote approximates
    the scored region as the union of reference and hypothesis extents, which
    lets a system escape false-alarm penalties for speech it invents outside the
    reference's span, and makes the denominator depend on the hypothesis.
    """
    from pyannote.core import Segment as PSegment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=DER_COLLAR, skip_overlap=DER_SKIP_OVERLAP)
    uem = Timeline([PSegment(0.0, duration)]) if duration and duration > 0 else None
    components = metric(to_pyannote(ref), to_pyannote(hyp), uem=uem, detailed=True)

    total = float(components["total"])
    miss_s = float(components["missed detection"])
    fa_s = float(components["false alarm"])
    conf_s = float(components["confusion"])

    if total <= 0:
        return dict(der=0.0, miss=0.0, false_alarm=0.0, confusion=0.0,
                    miss_sec=miss_s, false_alarm_sec=fa_s, confusion_sec=conf_s,
                    total_ref_sec=0.0)

    return dict(
        der=(miss_s + fa_s + conf_s) / total,
        miss=miss_s / total,
        false_alarm=fa_s / total,
        confusion=conf_s / total,
        miss_sec=miss_s,
        false_alarm_sec=fa_s,
        confusion_sec=conf_s,
        total_ref_sec=total,
    )


# --------------------------------------------------------------------------- #
# §5.2 Word Error Rate
# --------------------------------------------------------------------------- #

def word_error_rate(ref_text: str, hyp_text: str) -> dict[str, float]:
    """Speaker-agnostic WER over the whole clip, with S/D/I counts.

    Both sides pass through the same normalization — asymmetric normalization is
    the documented way to silently inflate WER.
    """
    ref, hyp = normalize(ref_text), normalize(hyp_text)
    if not ref:
        return dict(wer=0.0, n_sub=0, n_del=0, n_ins=0, n_ref_words=0, n_hyp_words=len(hyp.split()))

    out = jiwer.process_words(ref, hyp)
    n_ref = len(ref.split())
    return dict(
        wer=float(out.wer),
        n_sub=int(out.substitutions),
        n_del=int(out.deletions),
        n_ins=int(out.insertions),
        n_ref_words=n_ref,
        n_hyp_words=len(hyp.split()),
    )


# --------------------------------------------------------------------------- #
# §5.3 Concatenated minimum-permutation WER
# --------------------------------------------------------------------------- #

def _by_speaker(segments: list[Segment]) -> dict[str, str]:
    """Concatenate each speaker's normalized text, in time order."""
    out: dict[str, list[str]] = {}
    for s in sorted(segments, key=lambda x: x.start):
        text = normalize(s.text)
        if text:
            out.setdefault(s.speaker, []).append(text)
    return {k: " ".join(v) for k, v in out.items()}


def cp_wer(ref: list[Segment], hyp: list[Segment]) -> dict:
    """cpWER plus the optimal reference→hypothesis speaker assignment.

    The assignment is reused for WDER and for §4.3 segment alignment, so speaker
    mapping is decided **once per clip**, globally — never per segment.
    """
    from meeteval.wer import cp_word_error_rate

    ref_by, hyp_by = _by_speaker(ref), _by_speaker(hyp)
    if not ref_by:
        return dict(cpwer=0.0, assignment={}, n_ref_speakers=0, n_hyp_speakers=len(hyp_by))
    if not hyp_by:
        hyp_by = {"__empty__": ""}

    res = cp_word_error_rate(ref_by, hyp_by)
    mapping = {r: h for r, h in res.assignment if r is not None and h is not None}
    return dict(
        cpwer=float(res.error_rate),
        assignment=mapping,
        n_ref_speakers=len(ref_by),
        n_hyp_speakers=len(hyp_by),
        cp_substitutions=int(res.substitutions),
        cp_deletions=int(res.deletions),
        cp_insertions=int(res.insertions),
        missed_speaker=int(res.missed_speaker),
        falarm_speaker=int(res.falarm_speaker),
    )


# --------------------------------------------------------------------------- #
# §5.4 Word Diarization Error Rate
# --------------------------------------------------------------------------- #

def words_from_segments(segments: list[Segment]) -> list[Word]:
    """Flatten segments into a time-ordered word sequence carrying speaker labels."""
    words: list[Word] = []
    for s in sorted(segments, key=lambda x: x.start):
        toks = normalize(s.text).split()
        if not toks:
            continue
        # Distribute time uniformly across the segment's words. Only used when
        # true word-level timestamps are unavailable.
        step = s.duration / len(toks) if s.duration > 0 else 0.0
        for i, tok in enumerate(toks):
            words.append(Word(tok, s.speaker, s.start + i * step, s.start + (i + 1) * step))
    return words


def word_diarization_error_rate(
    ref_words: list[Word],
    hyp_words: list[Word],
    assignment: dict[str, str],
) -> dict[str, float]:
    """WDER — of the words the ASR got right, how many carry the wrong speaker?

    Following El Shafey et al. (2019)::

        WDER = (S_IS + C_IS) / (S + C)

    where ``C`` is correctly-recognized words and ``S`` substitutions; the ``_IS``
    variants are those same words whose speaker attribution is wrong. Deletions
    and insertions are excluded — a word that was never recognized cannot be
    misattributed.

    ``assignment`` maps reference speaker → hypothesis speaker and must be the
    single global permutation from :func:`cp_wer`.
    """
    ref_tokens = [w.text for w in ref_words]
    hyp_tokens = [w.text for w in hyp_words]

    if not ref_tokens:
        return dict(wder=0.0, n_correct=0, n_sub=0, n_correct_wrong_spk=0,
                    n_sub_wrong_spk=0, n_scored=0)

    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))

    n_c = n_s = n_c_is = n_s_is = 0
    for chunk in out.alignments[0]:
        if chunk.type not in ("equal", "substitute"):
            continue  # deletions/insertions are outside WDER's scope
        span = chunk.ref_end_idx - chunk.ref_start_idx
        for k in range(span):
            ri = chunk.ref_start_idx + k
            hi = chunk.hyp_start_idx + k
            if ri >= len(ref_words) or hi >= len(hyp_words):
                continue
            expected = assignment.get(ref_words[ri].speaker)
            wrong = expected is None or hyp_words[hi].speaker != expected
            if chunk.type == "equal":
                n_c += 1
                n_c_is += wrong
            else:
                n_s += 1
                n_s_is += wrong

    scored = n_c + n_s
    return dict(
        wder=(n_c_is + n_s_is) / scored if scored else 0.0,
        n_correct=n_c,
        n_sub=n_s,
        n_correct_wrong_spk=n_c_is,
        n_sub_wrong_spk=n_s_is,
        n_scored=scored,
    )


# --------------------------------------------------------------------------- #
# Clip-level driver
# --------------------------------------------------------------------------- #

@dataclass
class ClipScores:
    der: dict = field(default_factory=dict)
    wer: dict = field(default_factory=dict)
    cpwer: dict = field(default_factory=dict)
    wder: dict = field(default_factory=dict)

    def flat(self) -> dict:
        out = {}
        out.update({k: v for k, v in self.der.items()})
        out.update({k: v for k, v in self.wer.items()})
        out["cpwer"] = self.cpwer.get("cpwer")
        out["n_ref_speakers"] = self.cpwer.get("n_ref_speakers")
        out["n_hyp_speakers"] = self.cpwer.get("n_hyp_speakers")
        out["wder"] = self.wder.get("wder")
        out["n_scored_words"] = self.wder.get("n_scored")
        return out


def score_clip(
    ref: list[Segment],
    hyp: list[Segment],
    hyp_words: list[Word] | None = None,
    duration: float | None = None,
    strict: bool = True,
) -> ClipScores:
    """Compute all four metrics for one clip.

    Pass ``hyp_words`` when the system provides true word-level timestamps; the
    fallback distributes word times uniformly within each segment, which is less
    accurate for WDER. Pass ``duration`` so DER is scored over the whole clip.
    """
    der = diarization_error_rate(ref, hyp, duration=duration)
    if strict:
        parts = der["miss"] + der["false_alarm"] + der["confusion"]
        assert abs(parts - der["der"]) < 1e-6, (
            f"DER decomposition broken: {parts} != {der['der']}"
        )

    ref_text = " ".join(s.text for s in sorted(ref, key=lambda x: x.start))
    hyp_text = " ".join(s.text for s in sorted(hyp, key=lambda x: x.start))

    cp = cp_wer(ref, hyp)
    return ClipScores(
        der=der,
        wer=word_error_rate(ref_text, hyp_text),
        cpwer=cp,
        wder=word_diarization_error_rate(
            words_from_segments(ref),
            hyp_words if hyp_words is not None else words_from_segments(hyp),
            cp["assignment"],
        ),
    )
