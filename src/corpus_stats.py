"""Stage 0 — corpus audit and parsing validation (§3.2–3.6). No models involved.

Builds ``data/clip_meta.parquet`` and runs the Stage 0 GATE: our per-language
overlap ratios must reproduce the dataset README's published values. That
validates segment/timestamp parsing before any inference runs, catching the
kind of bug that would silently poison every DER downstream.

Run:  uv run python src/corpus_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import (  # noqa: E402
    CMI_BIN_EDGES, CMI_BIN_LABELS, DATA_DIR, LANGUAGE_SCRIPT,
    OVERLAP_BIN_EDGES, OVERLAP_BIN_LABELS, OVERLAP_GATE_ABS_TOL_PP,
    OVERLAP_GATE_CORPUS_TOL_PP, speaker_bin,
)
from data import load_all_meta  # noqa: E402
from normalize import code_mixing_index, code_switch_spans, strip_tags  # noqa: E402
from rttm import overlap_stats, segments_from_record  # noqa: E402

pd.set_option("display.width", 160)


def build_clip_meta(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Per-clip metadata: overlap (§3.5) and code-mixing (§3.6)."""
    ref = pd.read_csv(DATA_DIR / "reference_stats.csv")
    family = dict(zip(ref.language, ref.language_family))

    tag_inventory: dict[str, int] = {}
    rows = []
    for rec in df_raw.to_dict("records"):
        segs = segments_from_record(rec)
        ov = overlap_stats(segs)
        lang = rec["language"]
        native = LANGUAGE_SCRIPT.get(lang, "Devanagari")

        full_text = " ".join(s.text for s in segs)
        for tag in strip_tags(full_text)[1]:
            key = tag.strip().lower()
            tag_inventory[key] = tag_inventory.get(key, 0) + 1
        cmi, n_native, n_latin, n_neutral = code_mixing_index(full_text, native)
        # Annotator-marked code switches — a human label, stronger than script
        # detection alone (§3.6).
        cs_spans = code_switch_spans(full_text)

        rows.append({
            "sample_id": rec["sample_id"],
            "recording_id": rec["recording_id"],
            "language": lang,
            "language_family": family.get(lang, "Unknown"),
            "dataset_type": rec["dataset_type"],
            "num_speakers": int(rec["num_speakers"]),
            "speaker_bin": speaker_bin(int(rec["num_speakers"])),
            "num_segments": int(rec["num_segments"]),
            "duration_seconds": float(rec["duration_seconds"]),
            "speech_duration": ov["speech_union"],
            "speech_sum": ov["speech_sum"],
            "overlap_duration": ov["overlap_duration"],
            "overlap_ratio": ov["overlap_ratio"],
            "overlap_ratio_sum": ov["overlap_ratio_sum"],
            "n_speakers_seen": ov["n_speakers_seen"],
            "cmi": cmi,
            "n_tok_native": n_native,
            "n_tok_latin": n_latin,
            "n_tok_neutral": n_neutral,
            "n_code_switch_spans": len(cs_spans),
            "n_code_switch_words": sum(len(x.split()) for x in cs_spans),
        })

    meta = pd.DataFrame(rows)
    meta["overlap_bin"] = pd.cut(meta.overlap_ratio, bins=OVERLAP_BIN_EDGES,
                                 labels=OVERLAP_BIN_LABELS, include_lowest=True)
    meta["cmi_bin"] = pd.cut(meta.cmi, bins=CMI_BIN_EDGES,
                             labels=CMI_BIN_LABELS, include_lowest=True)
    return meta, tag_inventory


def _weighted_overlap(g: pd.DataFrame, col: str) -> float:
    """Duration-weighted overlap ratio: total overlap / total speech, not a mean
    of per-clip ratios (which would over-weight short clips)."""
    denom = g["speech_duration"].sum() if col == "overlap_ratio" else g["speech_sum"].sum()
    return 100.0 * g["overlap_duration"].sum() / denom if denom > 0 else 0.0


def run_overlap_gate(meta: pd.DataFrame) -> tuple[bool, pd.DataFrame, str]:
    """STAGE 0 GATE — reproduce the README's published per-language overlap %."""
    ref = pd.read_csv(DATA_DIR / "reference_stats.csv").set_index("language")

    recs = []
    for lang, g in meta.groupby("language"):
        recs.append({
            "language": lang,
            "n_clips": len(g),
            "hours_ours": g.duration_seconds.sum() / 3600.0,
            "hours_published": ref.at[lang, "total_hours"] if lang in ref.index else float("nan"),
            "overlap_union_pct": _weighted_overlap(g, "overlap_ratio"),
            "overlap_sum_pct": _weighted_overlap(g, "overlap_ratio_sum"),
            "overlap_published_pct": ref.at[lang, "overlap_pct"] if lang in ref.index else float("nan"),
        })
    cmp_df = pd.DataFrame(recs).sort_values("language").reset_index(drop=True)

    for name in ("union", "sum"):
        cmp_df[f"delta_{name}"] = cmp_df[f"overlap_{name}_pct"] - cmp_df.overlap_published_pct

    # Pick whichever denominator the published figures actually used.
    mae = {n: cmp_df[f"delta_{n}"].abs().mean() for n in ("union", "sum")}
    chosen = min(mae, key=mae.get)

    corpus_ours = 100.0 * meta.overlap_duration.sum() / (
        meta.speech_duration.sum() if chosen == "union" else meta.speech_sum.sum())
    corpus_published = 12.8

    per_lang_ok = (cmp_df[f"delta_{chosen}"].abs() <= OVERLAP_GATE_ABS_TOL_PP)
    corpus_ok = abs(corpus_ours - corpus_published) <= OVERLAP_GATE_CORPUS_TOL_PP
    cmp_df["gate_pass"] = per_lang_ok

    passed = bool(per_lang_ok.all() and corpus_ok)
    report = (
        f"denominator chosen: speech_{chosen} (MAE union={mae['union']:.2f}pp, "
        f"sum={mae['sum']:.2f}pp)\n"
        f"corpus overlap: ours={corpus_ours:.2f}%  published={corpus_published:.1f}%  "
        f"delta={corpus_ours - corpus_published:+.2f}pp  "
        f"[{'PASS' if corpus_ok else 'FAIL'} @ tol {OVERLAP_GATE_CORPUS_TOL_PP}pp]\n"
        f"per-language: {int(per_lang_ok.sum())}/{len(cmp_df)} within "
        f"{OVERLAP_GATE_ABS_TOL_PP}pp"
    )
    return passed, cmp_df, report


def main() -> int:
    print("=" * 78)
    print("STAGE 0 — corpus audit + parsing validation (metadata only, no audio)")
    print("=" * 78)

    print("\n[1/4] Loading metadata for 22 language configs...")
    raw = load_all_meta()
    print(f"      total clips: {len(raw)}")

    print("\n[2/4] Computing overlap (§3.5) and code-mixing (§3.6)...")
    meta, tags = build_clip_meta(raw)

    print("\n[3/4] Corpus structure")
    print(f"      clips              : {len(meta)}")
    print(f"      distinct recordings: {meta.recording_id.nunique()}")
    print(f"      total duration     : {meta.duration_seconds.sum() / 3600:.1f} h")
    print(f"      speakers/clip      : {meta.num_speakers.min()}-{meta.num_speakers.max()}")

    dup = meta.groupby("recording_id").size()
    print(f"      recordings w/ >1 clip: {(dup > 1).sum()}  (max {dup.max()} clips)")

    mism = meta[meta.num_speakers != meta.n_speakers_seen]
    print(f"      num_speakers vs segments mismatch: {len(mism)} clips")

    print("\n      -- §3.3 by condition --")
    print(meta.groupby("dataset_type").agg(
        clips=("sample_id", "size"),
        hours=("duration_seconds", lambda s: round(s.sum() / 3600, 1)),
        languages=("language", "nunique")).to_string())

    print("\n      -- §3.4 by speaker bin --")
    print(meta.groupby("speaker_bin", observed=True).agg(
        clips=("sample_id", "size"),
        hours=("duration_seconds", lambda s: round(s.sum() / 3600, 1))).to_string())

    print("\n      -- §3.5 by overlap bin --")
    print(meta.groupby("overlap_bin", observed=True).agg(
        clips=("sample_id", "size"),
        hours=("duration_seconds", lambda s: round(s.sum() / 3600, 1))).to_string())

    print("\n      -- §3.6 by code-mixing bin --")
    print(meta.groupby("cmi_bin", observed=True).agg(
        clips=("sample_id", "size"),
        mean_cmi=("cmi", lambda s: round(s.mean(), 2))).to_string())

    print(f"\n      -- annotator tag inventory ({len(tags)} distinct) --")
    for tag, n in sorted(tags.items(), key=lambda kv: -kv[1])[:15]:
        print(f"         {n:6d}  {tag}")

    print("\n[4/4] STAGE 0 GATE — overlap ratio vs published values")
    passed, cmp_df, report = run_overlap_gate(meta)
    print(cmp_df[["language", "n_clips", "hours_ours", "hours_published",
                  "overlap_union_pct", "overlap_sum_pct", "overlap_published_pct",
                  "gate_pass"]].round(2).to_string(index=False))
    print("\n" + report)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(DATA_DIR / "clip_meta.parquet", index=False)
    cmp_df.to_csv(DATA_DIR / "stage0_gate.csv", index=False)
    print(f"\nwrote {DATA_DIR / 'clip_meta.parquet'}  ({len(meta)} rows)")
    print(f"wrote {DATA_DIR / 'stage0_gate.csv'}")

    print("\n" + "=" * 78)
    print(f"STAGE 0 GATE: {'PASS' if passed else 'FAIL'}")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
