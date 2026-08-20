"""Download the Indic DiarBench corpus from Hugging Face.

Resumable: re-running skips files already present in the HF cache. Downloads into
the standard HF cache (override with HF_HOME) so `load_dataset` finds them with
no extra config.

  uv run python scripts/download_dataset.py --list
  uv run python scripts/download_dataset.py --languages Hindi Kashmiri
  uv run python scripts/download_dataset.py --all
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from constants import HF_DATASET, HF_SPLIT_FILE, LANGUAGES  # noqa: E402
from env import get_hf_token, token_status  # noqa: E402

# Per-language parquet sizes in MB, from the HF file listing.
SIZES_MB = {
    "Assamese": 161.4, "Bengali": 1381.5, "Bodo": 168.8, "Dogri": 159.1,
    "Gujarati": 1233.8, "Hindi": 1217.3, "Kannada": 956.0, "Kashmiri": 120.7,
    "Konkani": 181.1, "Maithili": 147.0, "Malayalam": 428.0, "Manipuri": 162.8,
    "Marathi": 1170.9, "Nepali": 144.9, "Odia": 355.8, "Punjabi": 1200.5,
    "Sanskrit": 175.5, "Santali": 172.1, "Sindhi": 168.3, "Tamil": 1126.8,
    "Telugu": 1054.4, "Urdu": 169.2,
}


def human(mb: float) -> str:
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def cmd_list() -> int:
    print(f"{HF_DATASET} — 22 language configs, split 'test'\n")
    print(f"{'Language':<12} {'Size':>10}")
    print("-" * 24)
    for lang in sorted(SIZES_MB, key=lambda k: -SIZES_MB[k]):
        print(f"{lang:<12} {human(SIZES_MB[lang]):>10}")
    print("-" * 24)
    print(f"{'TOTAL':<12} {human(sum(SIZES_MB.values())):>10}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", metavar="LANG",
                    help="specific language configs to fetch")
    ap.add_argument("--all", action="store_true", help="fetch all 22 languages (~12 GB)")
    ap.add_argument("--list", action="store_true", help="show sizes and exit")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--retries", type=int, default=4,
                    help="attempts per language before giving up (default 4)")
    ap.add_argument("--no-xet", action="store_true",
                    help="disable Xet transfer and use plain HTTP (slower, more reliable)")
    args = ap.parse_args()

    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)

    if args.list:
        return cmd_list()

    if args.all:
        langs = list(LANGUAGES)
    elif args.languages:
        langs, unknown = [], []
        for name in args.languages:
            match = next((L for L in LANGUAGES if L.lower() == name.lower()), None)
            (langs if match else unknown).append(match or name)
        if unknown:
            print(f"ERROR: unknown language(s): {', '.join(unknown)}")
            print(f"Valid: {', '.join(LANGUAGES)}")
            return 2
    else:
        ap.print_help()
        return 2

    total_mb = sum(SIZES_MB.get(L, 0) for L in langs)
    free_gb = shutil.disk_usage(Path.home()).free / 1024**3

    print(f"\n{token_status()}")
    print(f"dataset : {HF_DATASET}")
    print(f"configs : {len(langs)} — {', '.join(langs)}")
    print(f"download: {human(total_mb)}   free disk: {free_gb:.1f} GB")
    print("cache   : " + os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))

    if free_gb < (total_mb / 1024) * 1.3:
        print("\nERROR: not enough free disk space (want ~30% headroom).")
        return 1

    if not args.yes:
        try:
            if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("aborted.")
                return 1
        except EOFError:
            print("\nnon-interactive; re-run with --yes")
            return 1

    from huggingface_hub import hf_hub_download

    token = get_hf_token(required=False)  # public dataset; token used if present

    print("\ndownloading (resumable — re-run to continue after an interruption)")
    print("one language at a time; a failure is retried and does not abort the rest\n")

    ok: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, lang in enumerate(langs, 1):
        label = f"[{i}/{len(langs)}] {lang:<11} {human(SIZES_MB.get(lang, 0)):>9}"
        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                fp = hf_hub_download(
                    repo_id=HF_DATASET,
                    repo_type="dataset",
                    filename=f"{lang}/{HF_SPLIT_FILE}",
                    token=token,
                )
                mb = Path(fp).stat().st_size / 1024**2
                dt = time.time() - t0
                rate = f"{mb / dt:.1f} MB/s" if dt > 0.5 else "cached"
                print(f"{label}  ok   {rate}")
                ok.append(lang)
                break
            except Exception as exc:  # noqa: BLE001 - report and continue
                msg = str(exc).split("\n")[0][:110]
                if attempt < args.retries:
                    wait = 5 * attempt
                    print(f"{label}  retry {attempt}/{args.retries - 1} in {wait}s — {msg}")
                    time.sleep(wait)
                else:
                    print(f"{label}  FAIL after {args.retries} attempts — {msg}")
                    failed.append((lang, msg))

    print(f"\ndownloaded {len(ok)}/{len(langs)} configs")
    if failed:
        print("\nfailed:")
        for lang, msg in failed:
            print(f"  {lang:<12} {msg}")
        print("\nre-run to resume. If Xet transfer keeps failing, retry with:")
        print("  HF_HUB_DISABLE_XET=1 uv run python scripts/download_dataset.py --all --yes")
        return 1

    print("all requested configs present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
