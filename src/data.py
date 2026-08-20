"""Dataset access (§3.1).

Stage 0 needs metadata only. Parquet is columnar, so reading just the metadata
columns over ``HfFileSystem`` avoids fetching the ~12 GB of embedded audio blobs.
Audio is pulled lazily, per language, only when Stage 1 inference needs it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from constants import HF_DATASET, HF_SPLIT_FILE, LANGUAGES, META_COLUMNS


def _remote_path(language: str) -> str:
    return f"datasets/{HF_DATASET}/{language}/{HF_SPLIT_FILE}"


def local_path(language: str) -> Path | None:
    """Path to a downloaded parquet in the HF cache, or None if not present.

    Reading locally avoids re-fetching column chunks over the network once
    `scripts/download_dataset.py` has run.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"datasets--{HF_DATASET.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return None
    for snapshot in repo_dir.iterdir():
        candidate = snapshot / language / HF_SPLIT_FILE
        if candidate.exists():
            return candidate
    return None


def load_language_meta(language: str, fs=None, prefer_local: bool = True) -> pd.DataFrame:
    """Read one language's metadata columns (no audio) into a DataFrame.

    Uses the local HF cache when available, otherwise streams the needed column
    chunks from the Hub. Parquet is columnar, so the remote path still never
    fetches the embedded audio blobs.
    """
    if prefer_local:
        path = local_path(language)
        if path is not None:
            return pq.ParquetFile(path).read(columns=META_COLUMNS).to_pandas()

    from huggingface_hub import HfFileSystem

    fs = fs or HfFileSystem()
    with fs.open(_remote_path(language), "rb") as fh:
        table = pq.ParquetFile(fh).read(columns=META_COLUMNS)
    return table.to_pandas()


def load_all_meta(languages: list[str] | None = None, verbose: bool = True,
                  prefer_local: bool = True) -> pd.DataFrame:
    """Read metadata for every language config."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    frames = []
    for lang in languages or LANGUAGES:
        src = "local" if (prefer_local and local_path(lang)) else "remote"
        df = load_language_meta(lang, fs=fs, prefer_local=prefer_local)
        if verbose:
            print(f"  {lang:12s} {len(df):4d} clips  [{src}]")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_language_audio(language: str, split: str = "test"):
    """Load a language config *with* audio, for Stage 1 inference.

    Downloads and caches the full per-language parquet — 120 MB to 1.4 GB. Do not
    call this from Stage 0.
    """
    from datasets import load_dataset

    return load_dataset(HF_DATASET, language, split=split)
