"""Build prompt-ready transcript job bundles locally for GCP inference.

This script performs the deterministic work on the workstation so the cloud only
receives compact, validated job records:

- transcript cleanup
- deduplication by normalized content hash
- token estimation
- chunking with overlap
- manifest and reject-file generation

Usage:
    python Data_Cleaned/pipeline_tools/prepare_transcript_jobs.py \
        --input Data_Cleaned/yt_videos_with_local_transcripts.parquet \
        --output Data_Cleaned/gcp_jobs/transcript_jobs.jsonl \
        --manifest Data_Cleaned/gcp_jobs/transcript_jobs_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    from langdetect import detect_langs  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    detect_langs = None


WEBVTT_RE = re.compile(r"(?im)^WEBVTT.*$")
TIMESTAMP_RE = re.compile(
    r"\d{1,2}:\d{2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}:\d{2}[\.,]\d+"
)
SHORT_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}[\.,]\d+")
STANDALONE_TS_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?[\.,]\d+\b")
SRT_INDEX_RE = re.compile(r"(?m)^\s*\d+\s*$")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
VTT_HEADER_RE = re.compile(r"\b(?:Kind|Language)\s*:\s*\S+", re.IGNORECASE)
CUE_SETTING_RE = re.compile(
    r"\b(?:align|position|line|size)\s*:\s*[0-9a-zA-Z%.-]+", re.IGNORECASE
)
EMPTY_ANGLE_RE = re.compile(r"<>")


def detect_id_column(df: pd.DataFrame) -> str:
    candidates = ["videoId", "video_id", "id", "videoID", "yt_id", "youtube_id"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise RuntimeError(f"Could not detect a video id column. Available columns: {list(df.columns)}")


def detect_title_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ["title", "video_title", "name"]:
        if candidate in df.columns:
            return candidate
    return None


def deduplicate_vtt_overlaps(text: str, min_len: int = 4) -> str:
    """Collapse repeated word sequences produced by VTT sliding-window cues."""
    words = text.split()
    if len(words) < min_len * 2:
        return text
    output: List[str] = words[:min_len]
    i = min_len
    while i < len(words):
        max_w = min(30, len(words) - i)
        skipped = False
        for w in range(max_w, min_len - 1, -1):
            if len(output) >= w and words[i : i + w] == output[-w:]:
                i += w
                skipped = True
                break
        if not skipped:
            output.append(words[i])
            i += 1
    return " ".join(output)


def clean_transcript(text: object) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = WEBVTT_RE.sub("", text)
    text = VTT_HEADER_RE.sub("", text)
    text = TIMESTAMP_RE.sub("", text)
    text = SHORT_TIMESTAMP_RE.sub("", text)
    text = STANDALONE_TS_RE.sub("", text)
    text = SRT_INDEX_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = EMPTY_ANGLE_RE.sub("", text)
    text = CUE_SETTING_RE.sub("", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = WHITESPACE_RE.sub(" ", " ".join(lines)).strip()
    return deduplicate_vtt_overlaps(text)


def normalize_for_hash(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.lower()).strip()


def estimate_tokens(text: str, encoder=None) -> int:
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, math.ceil(len(text) / 4))


def get_tokenizer(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def detect_language(text: str) -> str:
    if detect_langs is None or not text:
        return "unknown"
    try:
        detected = detect_langs(text)
        return detected[0].lang if detected else "unknown"
    except Exception:
        return "unknown"


def chunk_text(text: str, max_tokens: int, overlap_tokens: int, encoder=None) -> List[str]:
    if not text:
        return []

    if encoder is None:
        words = text.split()
        if not words:
            return []
        max_words = max(1, int(max_tokens / 1.25))
        overlap_words = min(max_words - 1, int(overlap_tokens / 1.25)) if max_words > 1 else 0
        step = max(1, max_words - overlap_words)
        chunks: List[str] = []
        start = 0
        while start < len(words):
            slice_words = words[start:start + max_words]
            chunk = " ".join(slice_words).strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    tokens = encoder.encode(text)
    if not tokens:
        return []

    step = max(1, max_tokens - overlap_tokens)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        token_slice = tokens[start:start + max_tokens]
        chunk = encoder.decode(token_slice).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_job_records(
    df: pd.DataFrame,
    id_col: str,
    title_col: Optional[str],
    transcript_col: str,
    max_tokens: int,
    overlap_tokens: int,
    prompt_version: str,
    model_name: str,
    dedupe: bool = False,
) -> tuple[List[dict], list[dict]]:
    encoder = get_tokenizer(model_name)
    seen_hashes: set[str] = set() if dedupe else set()
    job_records: List[dict] = []
    rejected: list[dict] = []

    for row_index, row in df.iterrows():
        raw_text = clean_transcript(row.get(transcript_col, ""))
        video_id = str(row.get(id_col, "")).strip().lstrip("_")
        title = str(row.get(title_col, "")) if title_col else ""

        if not video_id:
            rejected.append({"row_index": int(row_index), "reason": "missing_video_id"})
            continue

        if not raw_text:
            rejected.append({"video_id": video_id, "reason": "empty_transcript"})
            continue

        normalized = normalize_for_hash(raw_text)
        source_hash = stable_hash(normalized)
        if dedupe:
            if source_hash in seen_hashes:
                rejected.append({"video_id": video_id, "reason": "duplicate_transcript", "source_hash": source_hash})
                continue
            seen_hashes.add(source_hash)

        chunks = chunk_text(raw_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, encoder=encoder)
        if not chunks:
            rejected.append({"video_id": video_id, "reason": "no_chunk_generated", "source_hash": source_hash})
            continue

        chunk_total = len(chunks)
        language = detect_language(raw_text)
        for chunk_idx, chunk in enumerate(chunks):
            token_count = estimate_tokens(chunk, encoder)
            if token_count == 0:
                continue
            job_id = f"{video_id}_{chunk_idx:03d}"
            job_records.append(
                {
                    "job_id": job_id,
                    "video_id": video_id,
                    "title": title,
                    "chunk_idx": chunk_idx,
                    "chunk_total": chunk_total,
                    "chunk_text": chunk,
                    "estimated_input_tokens": token_count,
                    "language": language,
                    "source_hash": source_hash,
                    "prompt_version": prompt_version,
                }
            )

    return job_records, rejected


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(
    manifest_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    prompt_version: str,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
    total_rows: int,
    transcript_rows: int,
    emitted_jobs: int,
    rejected_rows: int,
    job_records: list[dict],
    rejected_records: list[dict],
) -> None:
    total_tokens = sum(int(item["estimated_input_tokens"]) for item in job_records)
    manifest = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "prompt_version": prompt_version,
        "tokenizer_name": tokenizer_name,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "total_rows": total_rows,
        "transcript_rows": transcript_rows,
        "emitted_jobs": emitted_jobs,
        "rejected_rows": rejected_rows,
        "estimated_input_tokens_total": total_tokens,
        "average_input_tokens_per_job": round(total_tokens / emitted_jobs, 2) if emitted_jobs else 0,
        "unique_video_ids": len({item["video_id"] for item in job_records}),
        "sample_job_ids": [item["job_id"] for item in job_records[:5]],
        "reject_reasons": {},
    }

    for item in rejected_records:
        reason = item.get("reason", "unknown")
        manifest["reject_reasons"][reason] = manifest["reject_reasons"].get(reason, 0) + 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local transcript job bundles for GCP inference.")
    parser.add_argument("--input", default="Data_Cleaned/yt_videos_with_local_transcripts.parquet")
    parser.add_argument("--output", default="Data_Cleaned/gcp_jobs/transcript_jobs.jsonl")
    parser.add_argument("--manifest", default="Data_Cleaned/gcp_jobs/transcript_jobs_manifest.json")
    parser.add_argument("--rejects", default="Data_Cleaned/gcp_jobs/transcript_rejects.jsonl")
    parser.add_argument("--transcript-column", default="local_transcript")
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--overlap-tokens", type=int, default=150)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--tokenizer-name", default="cl100k_base")
    parser.add_argument("--dedupe", action="store_true", default=False, help="Enable transcript deduplication by content hash")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    reject_path = Path(args.rejects)

    df = pd.read_parquet(input_path)
    id_col = detect_id_column(df)
    title_col = detect_title_column(df)

    transcript_rows = df[args.transcript_column].notna().sum() if args.transcript_column in df.columns else 0
    job_records, rejected_records = build_job_records(
        df,
        id_col=id_col,
        title_col=title_col,
        transcript_col=args.transcript_column,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        prompt_version=args.prompt_version,
        model_name=args.tokenizer_name,
        dedupe=args.dedupe,
    )

    write_jsonl(job_records, output_path)
    write_jsonl(rejected_records, reject_path)
    write_manifest(
        manifest_path,
        input_path=input_path,
        output_path=output_path,
        prompt_version=args.prompt_version,
        tokenizer_name=args.tokenizer_name,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        total_rows=len(df),
        transcript_rows=int(transcript_rows),
        emitted_jobs=len(job_records),
        rejected_rows=len(rejected_records),
        job_records=job_records,
        rejected_records=rejected_records,
    )

    print(f"Wrote {len(job_records)} job records to {output_path}")
    print(f"Wrote {len(rejected_records)} rejected rows to {reject_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
