"""One-time deep-clean of the local_transcript column and full JSONL rebuild.

Applies cleaning passes that the original add_transcripts.py missed:
  - HTML entity unescaping  (&gt; &amp; &#39; etc.)
  - VTT Kind:/Language: header tokens leaked into transcript text
  - VTT cue settings (align:start  position:0%  line:90% etc.)
  - Leftover empty angle brackets  <>
  - Consecutive duplicate word sequences from VTT sliding-window cue overlap

After cleaning the parquet the script re-runs prepare_transcript_jobs.py to
regenerate all three JSONL bundles (production + test-v1 + test-v2) so every
downstream artifact stays in sync.

Usage (run from the Data_Cleaned directory):
    python reclean_and_rebuild_jobs.py
    python reclean_and_rebuild_jobs.py --parquet path/to/other.parquet  --dry-run
"""

from __future__ import annotations

import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── cleaning helpers ──────────────────────────────────────────────────────────

_VTT_HEADER_RE  = re.compile(r"\b(?:Kind|Language)\s*:\s*\S+", re.IGNORECASE)
_CUE_SETTING_RE = re.compile(
    r"\b(?:align|position|line|size)\s*:\s*[0-9a-zA-Z%.-]+", re.IGNORECASE
)
_EMPTY_ANGLE_RE = re.compile(r"<>")
_WHITESPACE_RE  = re.compile(r"\s{2,}")


def _dedup_vtt_overlaps(text: str, min_len: int = 4) -> str:
    """Collapse repeated word sequences produced by VTT sliding-window cues."""
    words = text.split()
    if len(words) < min_len * 2:
        return text
    out = words[:min_len]
    i = min_len
    while i < len(words):
        max_w, skipped = min(30, len(words) - i), False
        for w in range(max_w, min_len - 1, -1):
            if len(out) >= w and words[i : i + w] == out[-w:]:
                i += w
                skipped = True
                break
        if not skipped:
            out.append(words[i])
            i += 1
    return " ".join(out)


def deep_clean(text: object) -> object:
    if not isinstance(text, str) or not text.strip():
        return text
    text = html.unescape(text)
    text = _VTT_HEADER_RE.sub("", text)
    text = _CUE_SETTING_RE.sub("", text)
    text = _EMPTY_ANGLE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _dedup_vtt_overlaps(text)


# ── JSONL rebuild config ──────────────────────────────────────────────────────

JSONL_RUNS = [
    dict(
        label="production",
        output="gcp_jobs/transcript_jobs.jsonl",
        manifest="gcp_jobs/transcript_jobs_manifest.json",
        rejects="gcp_jobs/transcript_rejects.jsonl",
        max_tokens="1800", overlap_tokens="150",
        tokenizer="cl100k_base", dedupe=False,
    ),
    dict(
        label="test-v2",
        output="gcp_jobs/test_transcript_jobs_v2.jsonl",
        manifest="gcp_jobs/test_transcript_jobs_manifest_v2.json",
        rejects="gcp_jobs/test_transcript_rejects_v2.jsonl",
        max_tokens="400", overlap_tokens="50",
        tokenizer="cl100k_base", dedupe=True,
    ),
    dict(
        label="test-v1",
        output="gcp_jobs/test_transcript_jobs.jsonl",
        manifest="gcp_jobs/test_transcript_jobs_manifest.json",
        rejects="gcp_jobs/test_transcript_rejects.jsonl",
        max_tokens="400", overlap_tokens="50",
        tokenizer="cl100k_base", dedupe=True,
    ),
]


def rebuild_jsonl(parquet_path: Path, dry_run: bool) -> None:
    base = [sys.executable, "pipeline_tools/prepare_transcript_jobs.py"]
    for run in JSONL_RUNS:
        cmd = base + [
            "--input",    str(parquet_path),
            "--output",   run["output"],
            "--manifest", run["manifest"],
            "--rejects",  run["rejects"],
            "--max-tokens",      run["max_tokens"],
            "--overlap-tokens",  run["overlap_tokens"],
            "--tokenizer-name",  run["tokenizer"],
            "--prompt-version",  "v1",
        ]
        if run["dedupe"]:
            cmd.append("--dedupe")

        print(f"\n[{run['label']}]  {' '.join(cmd)}")
        if dry_run:
            print("  (dry-run — skipped)")
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        status = "OK" if result.returncode == 0 else "FAILED"
        print(f"  [{status}]  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr.strip()[:400]}")
            sys.exit(1)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        default="yt_videos_with_local_transcripts.parquet",
        help="Path to the parquet file (default: yt_videos_with_local_transcripts.parquet)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Clean and report stats but do not write any files",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        sys.exit(f"Parquet file not found: {parquet_path}")

    # ── Step 1: clean the parquet ─────────────────────────────────────────────
    print(f"Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    transcript_col = df.columns[-1]
    print(f"  rows: {len(df)}   transcript column: '{transcript_col}'")
    print(f"  non-null transcripts: {df[transcript_col].notna().sum()}")

    before_mean = df[transcript_col].dropna().str.len().mean()
    df[transcript_col] = df[transcript_col].apply(deep_clean)
    after_mean = df[transcript_col].dropna().str.len().mean()

    print(f"\nCleaning complete.")
    print(f"  Avg length: {before_mean:.0f} -> {after_mean:.0f} chars  (delta {after_mean - before_mean:+.0f})")

    if not args.dry_run:
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved -> {parquet_path}")
    else:
        print("  (dry-run — parquet not written)")

    # ── Step 2: rebuild JSONL bundles ─────────────────────────────────────────
    print("\nRebuilding JSONL bundles …")
    rebuild_jsonl(parquet_path, dry_run=args.dry_run)

    print("\nAll done.")


if __name__ == "__main__":
    main()
