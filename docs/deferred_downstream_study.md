# CLAUDE.md — Indic DiarBench Downstream Degradation Study

Read this fully before writing any code. This i
s the build spec for a project studying whether ASR and speaker-diarization errors in Indian-language conversational speech propagate into downstream LLM task failures — and specifically whether *speaker misattribution* (correct words, wrong speaker) breaks downstream tasks independently of word-level transcription errors.

## Dataset

- **Source**: `sarvamai/indic-diarbench` on Hugging Face (audio, RTTM speaker annotations, human-verified transcripts, evaluation protocol). Paper: arXiv:2607.23808 (Mehendale, Mehndiratta, Rathi, Bhogale, Khapra — Interspeech 2026).
- **Size**: ~108 hours total — ~53h near-field meetings (one close-mic per speaker), ~27h far-field meetings, ~28h in-the-wild YouTube. 22 scheduled Indian languages. 2–9 speakers per recording. 485 unique speakers / 189 districts (meetings) + ~750 speakers (in-the-wild).
- **Two transcript formats per session**: native Indic script, and Romanized (English words/numerals in Roman script) — both accepted for WER scoring. This is your Track D data, already built in.
- **Format**: audio + time-aligned, speaker-attributed transcripts in RTTM format.
- Confirm exact file layout and per-file sizes by browsing the HF repo's "Files and Versions" tab before writing loader code — do not assume a directory structure from the paper text alone.

## Metrics glossary (implement or reuse existing scoring code, don't hand-roll definitions from memory)

| Metric | What it measures |
|---|---|
| WER | Word Error Rate — transcription accuracy alone |
| DER | Diarization Error Rate — speaker-timeline accuracy alone (missed speech, false alarm, wrong speaker) |
| cpWER | Concatenated minimum-permutation WER — are the right words attached to the right speaker |
| WDER | Word Diarization Error Rate — how often a *correctly transcribed* word gets tagged with the *wrong* speaker |

**Track C depends on WER and DER being computed and stored separately per clip** — this is what lets you filter for "words right, speaker wrong" cases. Do not discard per-clip granularity in favor of only corpus-level aggregates.

## Hardware

| Machine | Role |
|---|---|
| RTX 4070, 12GB | ASR/diarization inference at scale, full-precision LLM downstream-task runs |
| Macs (1B/small models) | Dev loop, dry runs |
| Phone (Bonsai 27B or GGUF) | Stretch track only: quantized on-device downstream-task runs |

## Repo layout

```
indic-diarbench-degradation/
  src/
    data.py            # HF dataset loading, per-clip metadata (language, subset, speaker count)
    asr_diarize.py      # run ASR/diarization systems, produce transcripts + RTTM
    score.py             # WER / DER / cpWER / WDER computation, per-clip
    downstream.py       # summarization / QA / action-item extraction tasks
    misattribution.py   # Track C: filter clips by (low WER, high DER), run downstream, score attribution
    romanized_compare.py # Track D
  data/
    clip_scores.parquet   # one row per clip: language, subset, WER, DER, cpWER, WDER
    downstream_results.parquet  # one row per (clip, task, transcript_source): output, correctness score
  notebooks/
    trackA_heatmap.ipynb
    trackB_degradation.ipynb
    trackC_misattribution.ipynb
    trackD_script_comparison.ipynb
  README.md
```

## Trial schema (`data/downstream_results.parquet`)

| Column | Type | Description |
|---|---|---|
| `clip_id` | string | unique clip identifier from the dataset |
| `language` | string | one of 22 scheduled languages |
| `subset` | string | `near_field` / `far_field` / `in_the_wild` |
| `n_speakers` | int | speaker count for this clip |
| `transcript_source` | string | `gold` / `asr_system_name` |
| `script` | string | `native` / `romanized` |
| `wer` | float | word error rate for this clip/system |
| `der` | float | diarization error rate for this clip/system |
| `cpwer` | float | |
| `wder` | float | |
| `task` | string | `summarization` / `qa_who_said` / `action_items` |
| `output` | string | model's downstream output |
| `task_score` | float | task-specific correctness (see per-task scoring below) |
| `speaker_misattributed` | bool | only populated for Track C trials |

## Build order

### Stage 0 — Environment + data audit
1. Load the HF dataset, confirm audio + RTTM + both transcript formats are all accessible and aligned for a handful of clips.
2. Print language distribution, subset distribution, speaker-count distribution — confirm your assumptions about the corpus match reality before designing sampling.
3. Pick your language subset for the study: at minimum one high-resource (e.g. Hindi or Bengali) and one lower-resource language, ideally 4-6 total spanning the resource spectrum, to support the Track A heatmap without needing to process all 22.

### Stage 1 — Track A: baseline reproduction + language-disparity heatmap (do this first — cheap, guaranteed deliverable)
1. Run 2-3 ASR/diarization systems (can include a system from the paper's own baselines for a sanity check, or your own) on your language subset.
2. Compute WER/DER/cpWER/WDER per clip, store in `clip_scores.parquet`.
3. **Gate**: your numbers should be in the same ballpark as the paper's reported baselines for any system you can match — if wildly off, debug the scoring implementation before proceeding (most likely culprit: text normalization mismatch between your WER scorer and theirs, or RTTM parsing error).
4. Deliverable: heatmap, languages x metric, sorted by resource level.

### Stage 2 — Track B: downstream task degradation
1. Define the three downstream tasks concretely before writing prompts:
   - Summarization: fixed-format summary (who attended, key points, decisions)
   - QA: "who said/decided/proposed X" — generate these questions from the gold transcript so you have a checkable answer key
   - Action items: structured list (who, what, by when)
2. Run each task on both `gold` and each ASR system's transcript, same clip, same task, so every comparison is a matched pair.
3. Score: for QA, exact/fuzzy match on the named speaker; for summarization and action items, use a rubric-based or NLI-style factual-consistency check against the gold transcript — not just a generic LLM-judge quality score, since the point is *correctness*, not fluency.
4. **Gate**: confirm gold-transcript task scores are high (the LLM can do these tasks well when given clean input) before attributing any degradation to transcript errors rather than task difficulty.
5. Deliverable: grouped bar chart, task type x (gold vs. ASR-transcript score), faceted by language.

### Stage 3 — Track C: speaker misattribution propagation (the core novel contribution)
1. From `clip_scores.parquet`, filter clips where WER is low (words are basically right) but DER is high (speaker labels are wrong) — set explicit thresholds and justify them (e.g., WER < 15%, DER > 30%), don't cherry-pick post hoc.
2. Run the QA task ("who said X") on these filtered clips using the ASR system's transcript.
3. For each, check: does the model's answer name the *correct* speaker (per gold) or the *misattributed* speaker (per the ASR system's wrong diarization)?
4. **This is the headline number**: of N clips with correct words but wrong speaker labels, in what percentage does the downstream answer follow the wrong speaker label instead of getting it right some other way?
5. Deliverable: the headline percentage, plus 2-3 concrete worked examples (input transcript -> gold answer -> wrong answer caused by misattribution) as qualitative proof alongside the number.

### Stage 4 — Track D: native script vs. Romanized comparison
1. Reuses Track B's downstream pipeline — same clips, same tasks, but run once on the native-script transcript and once on the Romanized transcript of the identical audio.
2. Deliverable: paired/dumbbell plot, same clips, two bars per clip (native vs. Romanized task score).

### Stretch — Quantization interaction (only after Stages 0-4 are complete)
1. Repeat Track B's downstream step using a quantized on-device model (Bonsai 27B or a GGUF model) instead of full precision, on the same {gold, ASR-transcript} inputs.
2. Compute the 2x2 grid: {gold, ASR-transcript} x {full precision, quantized}.
3. The interesting result is whether the ASR-transcript + quantized cell degrades *more* than the two individual degradations would predict added together — an interaction effect, not just two stacked penalties.

## Pitfalls
- Text normalization mismatches (punctuation, casing, digit formats) will silently inflate WER if your scorer doesn't match the paper's normalization — verify against their reported baseline numbers first.
- Don't discard per-clip WER/DER granularity — Track C is impossible without it.
- Confounding transcript errors with task difficulty — always run the gold-transcript condition as a control, per task, per language.
- Cherry-picking the WER/DER thresholds for Track C after seeing which threshold produces the best story — set them before looking at downstream results.
- Comparing native-script and Romanized results without confirming they're actually the same underlying audio/content (verify via clip ID alignment, not just running both formats independently).
- Running the full 22-language corpus before validating your pipeline on a small subset first — expensive and unnecessary for the core result.

## Definition of done per stage
Don't move to the next stage until the current one's gate (if any) explicitly passes and is logged. Stage 1 and Stage 2 both have hard gates; Stage 3 and Stage 4 have no gate but must produce their stated deliverable (headline number + examples; paired plot) regardless of outcome — a null result on Track C (misattribution doesn't actually break the downstream task) is still a valid, reportable finding.
