# CLAUDE.md — Diarization-to-ASR Error Propagation in Indian Languages

Read this fully before writing any code.

**Paper**: *Beyond Benchmark Scores: Analyzing Diarization-to-ASR Error Propagation across Indian Languages* — a **diagnostic / analytical journal paper**. Full section plan lives in `Indic_DiarBench_Diagnostic_Paper_Structure.docx` (repo root); section numbers below (§N) refer to it. That docx is the authority on paper structure; this file is the authority on how we build it.

**Central claim**: aggregate DER/cpWER/WDER hide *how* diarization failures become speaker-attributed ASR failures. We decompose DER into Miss / False Alarm / Speaker Confusion, align errors at segment level, and quantify which components propagate downstream and which conditions amplify them.

**The novelty is the framework (§4), not a new recognition architecture.** The baseline stays frozen across all analyses. Never "improve the model" to make numbers look better — that destroys the study.

## Deferred work — do not build now

`docs/deferred_downstream_study.md` holds an earlier spec for an LLM-downstream study (ASR errors → summarization/QA/action-item failures). It is **deferred, not cancelled** — we may return to it as an application-facing extension once the diagnostic paper lands. Two notes for whoever resumes it:

- Its Track D (native vs. Romanized script) is **dead as specced**. The Romanized transcripts described in the dataset paper's annotation pipeline were **not released** — the published schema has exactly one transcript field, verified across Kashmiri, Hindi, Bengali, Tamil and Santali.
- Do not interleave it with the current work. The diagnostic paper needs a frozen baseline and a clean analysis chain; bolting downstream LLM tasks on mid-stream muddies both.

## Dataset — verified, not assumed

`sarvamai/indic-diarbench` on HF. Public, not gated, CC-BY-4.0. Paper: arXiv:2607.23808 (Interspeech 2026).

**Real layout** (confirmed against the HF API, not the paper text):
- One **parquet config per language**, 22 configs, split `test` only. Path `<Language>/test-00000-of-00001.parquet`. ~12.06 GB total.
- Sizes range 120 MB (Kashmiri) to 1.38 GB (Bengali). Audio blobs are embedded in the parquet.
- **1,164 clips from only 590 distinct source recordings.** 47 recordings contribute >1 clip; one contributes 21.

**Schema** (identical across all 22 configs):

| Field | Type | Notes |
|---|---|---|
| `audio` | struct<bytes, path> | WAV 16 kHz mono, embedded |
| `recording_id` | string | `<language>_<nf\|ff\|itw>_<nnn>` |
| `language` | string | full name, e.g. `Hindi` |
| `annotated_transcript` | list<struct<speaker_id, transcript, start_time, end_time>> | the reference; **there is no RTTM file** |
| `dataset_type` | string | `Near field` / `Far field` / `In the wild` |
| `sample_id` | string | unique across all 1,164 clips — **the join key** |
| `num_speakers` | int32 | |
| `num_segments` | int32 | |
| `duration_seconds` | double | |

**Consequences you must respect:**
- **No RTTM ships.** Build reference RTTM from `annotated_transcript` in `src/rttm.py`. Every DER number depends on this being right — hence the Stage 0 gate.
- **Clips are not independent.** Group by `recording_id` for any split, aggregate, or CI. Clips from one recording share speakers and channel. `recording_id` is a required column everywhere.
- **Transcripts contain inline `<noise>`-style tags.** Strip in normalization or WER inflates silently.
- **Transcripts are native-script with Roman-script English inline** — this is the code-mixing signal for §3.6/§6.8. Detect via Unicode block.

**Coverage is unbalanced.** Near-field: all 22 languages (~53 h). Far-field: top 8 only (~27 h). In-the-wild: 10 only (~28 h). Low-resource languages are near-field-only at ~1.1–1.6 h. **Cross-language comparison (§6.7) must be near-field-only** or condition mix is confounded with language.

## Published reference numbers (use as gates, do not re-derive)

Per-language overlap ratios and durations are in the dataset README — fetch and store as `data/reference_stats.csv`. Corpus overlap 12.8%; range 6.1% (Punjabi) to 24.7% (Maithili).

Duration-weighted baselines, benchmark paper Table 3 — **note there is no published WER**, so our WER is uncalibrated:

| Model | DER % | cpWER % | WDER % | Miss | FA | Conf |
|---|---|---|---|---|---|---|
| Sarvam | 16.0 | 38.8 | 33.1 | 6.3 | 3.9 | 5.9 |
| AWS Transcribe | 23.5 | 43.7 | 34.3 | 13.1 | 3.1 | 7.4 |
| Deepgram Nova-3 | 32.0 | 63.2 | 39.3 | 18.3 | 5.4 | 8.3 |
| Azure STT | 34.8 | 60.8 | 39.5 | 24.4 | 1.7 | 8.7 |
| ElevenLabs Scribe | 35.0 | 58.3 | 40.7 | 13.6 | 6.2 | 15.3 |
| AssemblyAI | 40.5 | 88.6 | 43.7 | 25.5 | 5.5 | 9.6 |
| GPT-4o | 36.2 | 83.1 | 40.4 | 17.7 | 3.8 | 14.7 |
| Gemini 3 Pro | 74.0 | 58.9 | 33.0 | 41.7 | 5.8 | 26.5 |

Facts from this table that shape the work:
- **WDER floors around 33% even for the best system** — the phenomenon exists at corpus level. Our job is explaining *how* it arises, not proving that it does.
- **Gemini 3 Pro inverts the ordering**: worst DER (74.0), joint-best WDER (33.0). Broken timeline, decent per-word attribution. The opening argument that DER is a poor proxy.
- Components sum to DER (6.3+3.9+5.9 = 16.1 ≈ 16.0), confirming the invariant `score.py` must assert.

### What the benchmark paper already published (do not claim as contribution)

Read `2607.23808v1.pdf` before writing any results section. Their analysis is **entirely duration-weighted aggregate, n=8 systems, never below language level**. Already published:
- DER decomposition into Miss/FA/Confusion (Table 3) — the docx lists this as a "major contribution"; it is not.
- Overlap ratio vs DER/WDER/cpWER (Figure 3), per-language cpWER/WDER heatmaps (Figure 2).
- "Dravidian near-field cpWER ~5pp above Indo-Aryan at comparable DER"; near-field-only languages worse than multi-condition ones.

Still open, and where our contribution must live:
- **Segment-level anything** (§4.3, §4.4) — they have none.
- **RQ2**: they report components *and* downstream metrics but never regress one on the other. Impossible at n=8; feasible at n=1,164 clips.
- **Confound control** (§4.6) — only marginal correlations published. Their own §5 confounds overlap with speaker count ("Telugu... highest overlap and 4.5 speakers per recording").
- **Code-mixing** (§6.8) — in their title and curation pipeline, never quantified or related to error.
- **Cascaded systems** — §4: "diarization-only models (e.g., Pyannote) are excluded." They benchmarked only joint systems.

**Positioning is deferred until we have our own numbers.** Do not rewrite contribution claims on speculation; produce Stage 0–3 results first, then revisit.

### Resolved: the source paper's Telugu figure
§5 calls Telugu the highest-overlap language at 24.7% while Table 2 gives it 20.4%. **Both are correct** — §5 silently reports near-field only. Stage 0 measures near-field Telugu at 24.59% overlap with 4.5 speakers/recording, matching their §5 figures; all-conditions is 20.29%, matching Table 2. Useful confirmation that their language-level analysis is near-field-only, which is why §6.7 here must be too.

### Stage 0 measured facts (use these, they are verified)
- 1,164 clips / 590 recordings / 107.7 h. Near-field 781 clips (22 langs), far-field 173 (8), in-the-wild 210 (10).
- Overlap denominator is `speech_union`, confirmed against published values (MAE 0.20pp).
- **Recording concentration**: the top 10 recordings hold 17.1% of all clips; `hindi_nf_001` alone has 21. Effective n is nearer 590 than 1,164.
- Clip duration: median 201 s, mean 333 s, range 3.8–3,622 s. Only 2% under 60 s, so per-clip WER is stable — but always duration-weight aggregates.
- **Three clips have 1 speaker** (`marathi_030`, `odia_011`, `sanskrit_004`, all <12 s), contradicting the README's "2-9". Exclude them from speaker-confusion analyses — a single-speaker clip cannot be misattributed.
- The `[ ]` bracket convention is **not corpus-wide**: only 71/1,164 clips use it, 831 of 957 spans are Tamil, 865 of 957 in-the-wild. Treat annotator code-switch marks as a per-team artifact, not a label; the Unicode-script CMI is the primary measure.

### Supplementary unavailable
Their `DatasetSummary.csv` (referenced in §5 for language-wise near-vs-far-field results) is **not** in the HF repo — checked, 404. Recompute if needed.

## Systems (§3.7)

**Frozen primary baseline** — every headline number comes from this, unchanged after Stage 1:
- Diarization: `pyannote/speaker-diarization-3.1` (gated — accept terms on HF, needs `HF_TOKEN`)
- ASR: `ai4bharat/indic-conformer-600m-multilingual` — the only open model covering all 22 scheduled languages
- Word timestamps via forced alignment

**Robustness system** — retires §9 limitation 1 ("results may depend on the selected baseline"), never mixed into headline numbers:
- WhisperX (`large-v3` + pyannote + wav2vec2 alignment), ~13 of 22 languages. Report its language coverage gap explicitly.

**Gate-calibration system** — a *joint* diarization+ASR system, architecturally comparable to the paper's benchmarked rows:
- NeMo Sortformer (+ Canary/Parakeet ASR). Needed because the benchmark paper **excluded cascaded pipelines**, so neither our baseline nor WhisperX has a comparable published row. This system is what makes the Stage 1 gate a real reproduction check rather than a range check. It also supports a cascaded-vs-joint contrast in §8.2.

### Scoring configuration — must match the paper exactly (their §4)
- **DER: no forgiveness collar, overlapping speech included.** `DiarizationErrorRate(collar=0.0, skip_overlap=False)`. Set explicitly; never rely on library defaults.
- All systems receive the **same single-channel mixed audio**, including near-field (which was *recorded* multi-channel but released as a downmix). Do not attempt per-speaker channel tricks.
- **WDER requires word-level timestamps.** Whisper's native segment timestamps are too coarse. Forced alignment is mandatory, not optional — this constraint drives the whole pipeline design.

## Repo layout

```
src/
  data.py           # HF loading, clip metadata, segment extraction
  rttm.py           # reference + hypothesis RTTM construction, overlap computation
  normalize.py      # Indic text normalization, tag stripping, script detection
  corpus_stats.py   # §3.2-3.6: distributions, overlap ratio, code-mixing index
  asr_diarize.py    # §3.7: run baseline + robustness systems
  score.py          # §5: DER(+Miss/FA/Confusion), WER, cpWER, WDER — per clip
  align.py          # §4.3 segment-level alignment, §4.4 E1-E4 taxonomy
  stats.py          # §4.5-4.6: correlation, mixed-effects regression, bootstrap CIs
  figures.py        # all figures
data/
  reference_stats.csv       # published per-language stats, for gates
  clip_meta.parquet         # one row per clip, no model involved
  clip_scores.parquet       # one row per (clip, system)
  segment_alignment.parquet # one row per (clip, system, reference segment)
notebooks/
reports/figures/
```

## Schemas

### `data/clip_meta.parquet` — one row per clip, model-independent

`sample_id`, `recording_id`, `language`, `language_family`, `dataset_type`, `num_speakers`, `speaker_bin` (`2` / `3-4` / `5+`), `num_segments`, `duration_seconds`, `speech_duration`, `overlap_duration`, `overlap_ratio`, `overlap_bin` (`<5%` / `5-10%` / `10-20%` / `>20%`), `cmi`, `cmi_bin` (`none` / `low` / `moderate` / `high`)

### `data/clip_scores.parquet` — one row per (clip, system)

All `clip_meta` columns, plus:

| Column | Notes |
|---|---|
| `system` | `baseline` / `whisperx` |
| `der`, `miss`, `false_alarm`, `confusion` | rates; **must sum to `der`** — assert this |
| `miss_sec`, `false_alarm_sec`, `confusion_sec` | absolute seconds, for duration-weighting |
| `wer`, `n_sub`, `n_del`, `n_ins`, `n_ref_words` | §5.2 |
| `cpwer` | §5.3, via `meeteval` |
| `wder` | §5.4, needs word timestamps |

### `data/segment_alignment.parquet` — §4.3, the basis of all propagation analysis

One row per **reference** segment: `sample_id`, `recording_id`, `system`, `seg_idx`, `start_time`, `end_time`, `duration`, `ref_speaker`, `hyp_speaker`, `speaker_correct`, `ref_text`, `hyp_text`, `seg_wer`, `asr_correct`, `error_class` (`E1`–`E4`), `is_overlap`, and clip-level context (`language`, `dataset_type`, `num_speakers`, `overlap_bin`).

`hyp_speaker` is assigned **after** the global optimal speaker permutation for that clip — the same mapping cpWER uses. Never map speakers per-segment; that would define away speaker confusion.

## Methodological decisions (underspecified in the docx — resolved here)

1. **"ASR correct" in §4.4 is undefined.** Define `asr_correct = seg_wer <= τ` with **τ = 0.30 fixed before looking at results**, and report the E1–E4 distribution across τ ∈ {0.1, 0.2, 0.3, 0.5} as a sensitivity table. The taxonomy must not be an artifact of one arbitrary cut.
2. **§6.3 collinearity.** Miss + FA + Confusion sum to DER by construction. Regress WDER/cpWER on **the three components only** — never components *and* DER in the same model. Report standardized coefficients so components are rankable.
3. **§6.2 mechanical correlation.** DER and cpWER share speaker-assignment error by construction, so their correlation is inflated for uninteresting reasons. The honest headline is the **contrast**: DER→WER (should be weak; WER ignores speakers) vs DER→WDER (should be strong). Lead §6.2 with that contrast, not the raw scatter.
4. **§4.6 must be mixed-effects.** Random intercept for `recording_id` nested in `language`. 1,164 clips from 590 recordings makes independence false. Fixed effects: DER components, overlap ratio, speaker count, condition. Test DER×Overlap and DER×Condition interactions.
5. **RQ5 is underpowered per-language.** Kashmiri is 20 clips. Report per-language results with bootstrap CIs (resampled by `recording_id`), and make **language-family** the level at which claims are made. Full 22-language table goes to supplementary.

## Build order

### Stage 0 — Audit + parsing validation (no models)
1. Load metadata columns only from all 22 configs — parquet is columnar, so **do not download the 12 GB of audio for this**. Read via `HfFileSystem` + `pyarrow`.
2. Build `clip_meta.parquet`: distributions by language, condition, speaker count, duration (§3.2–3.4).
3. Compute per-clip overlap ratio from segment timestamps (§3.5); compute code-mixing index via Unicode-block token classification (§3.6).
4. **GATE**: per-language overlap ratios must reproduce the README's published values (corpus 12.8%, Punjabi 6.1%, Maithili 24.7%, …) within tolerance. This validates segment/timestamp parsing **before any model runs** and catches the RTTM bugs that would silently poison every DER. Log pass/fail per language.

### Stage 1 — Baseline inference + scoring (§3.7, §5, §6.1)
1. Run the frozen baseline over the corpus. Dev-loop on Mac with a handful of clips; full run on the 4070.
2. Compute per-clip DER(+components), WER, cpWER, WDER → `clip_scores.parquet`.
3. Assert `miss + false_alarm + confusion == der` per clip.
4. **GATE**: duration-weighted aggregate DER/cpWER/WDER from the **joint gate-calibration system** must land in the ballpark of the published rows — that is the only architecturally comparable comparison, since the paper excluded cascades. The cascaded baseline and WhisperX are additionally required to fall inside the spread spanned by the eight published systems. Wildly off ⇒ debug normalization, DER collar/overlap config, or RTTM parsing before proceeding. WER has no published reference; treat it as uncalibrated and say so in the paper.
5. Deliverable: §6.1 aggregate table.

### Stage 2 — Segment alignment + taxonomy (§4.3, §4.4)
Build `segment_alignment.parquet`; produce the E1–E4 distribution plus the τ sensitivity table. **This is the methodological heart of the paper — it is what distinguishes the manuscript from the original benchmark.** No gate, but it must exist before any Stage 3 claim.

### Stage 3 — Propagation analysis (§4.5, §4.6, §6.2, §6.3)
Correlations (Pearson + Spearman), the DER→WER vs DER→WDER contrast, mixed-effects models, bootstrap CIs, component effect-size ranking. **Do not assume speaker confusion dominates — that is the hypothesis under test, and a null result is publishable.**

### Stage 4 — Conditioning factors (§6.4–6.8)
Overlap bins, acoustic condition, speaker count, cross-lingual/family, code-mixing. Each subsection maps to one RQ.

### Stage 5 — Robustness + case studies (§7, §9)
WhisperX re-run to show conclusions survive a baseline swap; 6–10 qualitative cases drawn from `segment_alignment.parquet`, one per failure mechanism in §7.

## Pitfalls
- Text normalization mismatch (punctuation, casing, digits, noise tags) silently inflates WER. Validate against the published baselines first.
- Treating clips as independent — 1,164 clips, 590 recordings. Always group by `recording_id`.
- Comparing languages across different condition mixes — restrict §6.7 to near-field.
- **Overlap is strongly confounded with speaker count** — measured in Stage 0: r = +0.45 (p≈1e-40, near-field, n=781); mean overlap 9.1% / 15.3% / 20.3% for 2 / 3-4 / 5+ speakers. The benchmark paper's §5 treats these as separate difficulty factors ("highest overlap **and** 4.5 speakers"); they are not separable without a multivariable model. Never attribute an effect to overlap without controlling speaker count, or vice versa.
- Overlap vs *resource level* is a **weak, non-significant** trend, not an established confound: Spearman(hours, near-field overlap) = -0.383, p = 0.078, n = 22; single-condition languages average 17.5% overlap vs 16.1% for three-condition ones. Do not claim a resource-overlap relationship on this evidence — quoting the Punjabi 6.1% / Maithili 24.7% extremes is cherry-picking.
- Regressing on DER *and* its components together (singular by construction).
- Per-segment speaker mapping instead of one global permutation per clip — defines away the effect being measured.
- Choosing τ, overlap bins, or speaker bins after seeing which cut tells the best story. Fix them first; bins are already specified above.
- Claiming causal propagation from correlation alone. Segment-level alignment is what licenses the causal language — use it.
- Running all 22 languages before the pipeline is validated on a small subset.

## Hardware

| Machine | Role |
|---|---|
| MacBook Air M4, 16 GB | Dev loop, Stage 0 (metadata-only, no audio download), dry runs |
| RTX 4070, 12 GB | Stage 1+ inference at scale |

## Environment
`uv` manages everything, Python 3.11. `uv sync` for core (Mac: scoring, stats, plots); `uv sync --extra gpu` on the 4070. Never `pip install` into the venv directly.

Scoring uses reference implementations — `pyannote.metrics` (DER + components), `jiwer` (WER), `meeteval` (cpWER). Do not hand-roll metric definitions.

## Definition of done
Stage 0 and Stage 1 have hard gates; do not proceed past either until it explicitly passes and the result is logged. Later stages have no gate but must produce their stated deliverable regardless of outcome — a null result (e.g. speaker confusion does *not* dominate propagation) is a valid and reportable finding.
