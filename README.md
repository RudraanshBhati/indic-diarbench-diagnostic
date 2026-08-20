# Diarization-to-ASR Error Propagation in Indian Languages

A diagnostic study on [`sarvamai/indic-diarbench`](https://huggingface.co/datasets/sarvamai/indic-diarbench):
how do speaker-diarization failures turn into speaker-attributed ASR failures, and
which conditions amplify that propagation?

The benchmark paper ([arXiv:2607.23808](https://arxiv.org/abs/2607.23808), Interspeech 2026)
reports duration-weighted aggregates across 8 systems. This work goes below that
level — segment-by-segment — to ask *which* diarization error component propagates,
and whether the reported relationships survive controlling for confounds.

- Build spec: [CLAUDE.md](CLAUDE.md)
- Paper section plan: `Indic_DiarBench_Diagnostic_Paper_Structure.docx`
- Deferred LLM-downstream study: [docs/deferred_downstream_study.md](docs/deferred_downstream_study.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is pinned — torch and
pyannote do not support 3.14.

```bash
uv sync                  # core: scoring, stats, plots (works on macOS)
uv sync --extra gpu      # add on the RTX 4070 box for inference
```

Create `.env` at the repo root with a Hugging Face token. It is gitignored, and
the value is never logged:

```
HF_TOKEN=hf_...
```

The token is optional for the dataset (public, CC-BY-4.0) but **required** for
`pyannote/speaker-diarization-3.1`, which is gated — accept its terms on the Hub
first.

## Download the corpus

~11.8 GB across 22 per-language parquet files. Resumable; re-run to continue.

```bash
uv run python scripts/download_dataset.py --list
uv run python scripts/download_dataset.py --languages Hindi Kashmiri
uv run python scripts/download_dataset.py --all --yes
uv run python scripts/download_dataset.py --all --yes --no-xet   # if Xet transfer fails
```

Stage 0 does **not** need this — parquet is columnar, so the audit reads only
metadata columns and never fetches the embedded audio.

## Run

```bash
uv run pytest tests/ -q             # scoring/parsing unit tests
uv run python src/corpus_stats.py   # Stage 0: audit + parsing gate
```

## Layout

| Path | Role |
|---|---|
| `src/constants.py` | fixed bins, thresholds, script ranges — all set a priori |
| `src/env.py` | `.env` / token loading (never logs the value) |
| `src/data.py` | dataset access; prefers the local HF cache |
| `src/normalize.py` | text normalization, script detection, code-mixing index |
| `src/rttm.py` | segment timelines, overlap computation, RTTM I/O |
| `src/corpus_stats.py` | Stage 0 audit and gate |
| `data/reference_stats.csv` | published per-language stats, used as the gate target |
| `tests/` | hand-computed cases pinning overlap and normalization semantics |

## Stage gates

Two hard gates, both designed to fail loudly before wasted compute:

- **Stage 0** — our per-language overlap ratios must reproduce the values published
  in the dataset README. This validates segment/timestamp parsing *before any model
  runs*, catching the class of bug that would otherwise silently corrupt every DER.
- **Stage 1** — aggregate DER/cpWER/WDER from the joint gate-calibration system must
  land in the ballpark of the paper's published rows.

Metrics use reference implementations (`pyannote.metrics`, `jiwer`, `meeteval`),
never hand-rolled definitions. DER is computed with **no forgiveness collar and
overlapping speech included**, matching the benchmark paper's §4.

## Status

Stage 0 written, not yet run end-to-end. Nothing downstream has executed.
