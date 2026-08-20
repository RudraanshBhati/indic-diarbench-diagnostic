"""Fixed constants for the study.

Every bin edge and threshold here is chosen BEFORE looking at results (see CLAUDE.md
"Pitfalls"). Do not tune them to improve a finding.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIGURE_DIR = REPO_ROOT / "reports" / "figures"

HF_DATASET = "sarvamai/indic-diarbench"
HF_SPLIT_FILE = "test-00000-of-00001.parquet"

# The 22 scheduled languages, as the dataset names its configs.
LANGUAGES = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
    "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi",
    "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil",
    "Telugu", "Urdu",
]

# Columns that carry no audio — Stage 0 reads only these so the 12 GB of audio
# blobs are never fetched (parquet is columnar).
META_COLUMNS = [
    "sample_id", "recording_id", "language", "dataset_type",
    "num_speakers", "num_segments", "duration_seconds", "annotated_transcript",
]

DATASET_TYPES = ["Near field", "Far field", "In the wild"]
CONDITION_SHORT = {"Near field": "nf", "Far field": "ff", "In the wild": "itw"}

# --- §3.4 speaker-count bins (fixed a priori) ---
SPEAKER_BINS = [(2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")]

# --- §3.5 overlap-intensity bins (fixed a priori) ---
OVERLAP_BIN_EDGES = [0.0, 0.05, 0.10, 0.20, 1.01]
OVERLAP_BIN_LABELS = ["<5%", "5-10%", "10-20%", ">20%"]

# --- §3.6 code-mixing bins (fixed a priori) ---
CMI_BIN_EDGES = [-0.01, 0.0, 10.0, 25.0, 101.0]
CMI_BIN_LABELS = ["none", "low", "moderate", "high"]

# --- §4.4 segment-level "ASR correct" threshold ---
# Primary value; the sensitivity table sweeps ASR_CORRECT_TAU_SWEEP.
ASR_CORRECT_TAU = 0.30
ASR_CORRECT_TAU_SWEEP = [0.10, 0.20, 0.30, 0.50]

# --- Stage 0 gate tolerance ---
# Our overlap ratio must reproduce the README's published per-language values.
# The published figures are rounded to 1 dp and computed over full recordings
# rather than clips, so an exact match is not expected.
OVERLAP_GATE_ABS_TOL_PP = 3.0   # percentage points, per language
OVERLAP_GATE_CORPUS_TOL_PP = 1.5  # tighter for the corpus-level aggregate

# Unicode ranges per script. Used for code-mixing detection (§3.6).
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Oriya": [(0x0B00, 0x0B7F)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "OlChiki": [(0x1C50, 0x1C7F)],
    "MeeteiMayek": [(0xABC0, 0xABFF), (0xAAE0, 0xAAFF)],
    "Latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}

# Expected native script per language — used to sanity-check script detection.
LANGUAGE_SCRIPT = {
    "Assamese": "Bengali", "Bengali": "Bengali", "Bodo": "Devanagari",
    "Dogri": "Devanagari", "Gujarati": "Gujarati", "Hindi": "Devanagari",
    "Kannada": "Kannada", "Kashmiri": "Arabic", "Konkani": "Devanagari",
    "Maithili": "Devanagari", "Malayalam": "Malayalam", "Manipuri": "Bengali",
    "Marathi": "Devanagari", "Nepali": "Devanagari", "Odia": "Oriya",
    "Punjabi": "Gurmukhi", "Sanskrit": "Devanagari", "Santali": "OlChiki",
    "Sindhi": "Arabic", "Tamil": "Tamil", "Telugu": "Telugu", "Urdu": "Arabic",
}


def speaker_bin(n: int) -> str:
    for lo, hi, label in SPEAKER_BINS:
        if lo <= n <= hi:
            return label
    return "2" if n < 2 else "5+"
