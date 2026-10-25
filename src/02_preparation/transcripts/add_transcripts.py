"""Merge local transcripts into `yt_videos_cleaned.parquet`.

This script discovers transcripts under `transcripts/` and `transcripts/cloud_transcripts/`,
strips timing markers from VTT/SRT/TXT, and merges cleaned text into the dataframe as
`local_transcript`.

Usage:
    python Data_Cleaned/pipeline_tools/add_transcripts.py --input Data_Cleaned/yt_videos_cleaned.parquet --output Data_Cleaned/yt_videos_with_local_transcripts.parquet

Optional flags:
    --inplace    Overwrite the input parquet file.

Dependencies: pandas, pyarrow
"""
import argparse
import html
import os
import re
from pathlib import Path
import pandas as pd

_VTT_HEADER_RE = re.compile(r"\b(?:Kind|Language)\s*:\s*\S+", re.IGNORECASE)
_CUE_SETTING_RE = re.compile(
    r"\b(?:align|position|line|size)\s*:\s*[0-9a-zA-Z%.-]+", re.IGNORECASE
)
_EMPTY_ANGLE_RE = re.compile(r"<>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _deduplicate_vtt_overlaps(text: str, min_len: int = 4) -> str:
    words = text.split()
    if len(words) < min_len * 2:
        return text
    output = words[:min_len]
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


def strip_timestamps_and_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"(?i)webvtt(:?.*)?\n", "", text)
    text = _VTT_HEADER_RE.sub("", text)
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"\d{1,2}:\d{2}[\.,]\d+\s*-->\s*\d{1,2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"\d{1,2}:\d{2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"\d{1,2}:\d{2}[\.,]\d+", "", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = _EMPTY_ANGLE_RE.sub("", text)
    text = _CUE_SETTING_RE.sub("", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not re.match(r"^\s*$", ln)]
    text = " ".join(lines).strip()
    return _deduplicate_vtt_overlaps(text)


def discover_transcripts(transcripts_root: str):
    transcripts = {}
    root = Path(transcripts_root)
    if not root.exists():
        return transcripts
    for p in root.rglob('*'):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in ['.vtt', '.srt', '.txt', '.empty'] or p.parent.name == 'cloud_transcripts':
                try:
                    text = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    text = ''
                cleaned = strip_timestamps_and_tags(text)
                # derive video id from filename: remove leading underscores and known language tags
                name = p.name.lstrip('_')
                # remove language/extra suffixes like .it, .en before extension
                name = re.split(r"\.(?=[^\.]+$)", name)[0]
                # sometimes names include dots: take first token
                vid = name.split('.')[0]
                transcripts[vid] = cleaned
    return transcripts


def detect_id_column(df: pd.DataFrame):
    candidates = ['video_id', 'id', 'videoId', 'videoID', 'yt_id', 'youtube_id']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try to find any column that looks like video ids (length ~11)
    for c in df.columns:
        sample = df[c].dropna().astype(str)
        if not sample.empty and sample.iloc[0] and 5 <= len(sample.iloc[0]) <= 50:
            return c
    return None


def main(args):
    df = pd.read_parquet(args.input)
    id_col = detect_id_column(df)
    if id_col is None:
        raise RuntimeError('Could not detect a video id column in the dataframe. Please inspect columns: ' + ','.join(df.columns))

    transcripts_map = discover_transcripts(os.path.join(Path(args.input).parents[0].as_posix().replace('\\', '/'), '..', 'transcripts'))
    # fallback: also search workspace top-level transcripts folder
    if not transcripts_map:
        transcripts_map = discover_transcripts(os.path.join(Path(args.input).parents[1].as_posix().replace('\\', '/'), 'transcripts'))
    # simpler: also try './transcripts'
    if not transcripts_map:
        transcripts_map = discover_transcripts('transcripts')

    local_texts = []

    for _, row in df.iterrows():
        vid = str(row[id_col]).lstrip('_')
        # sometimes video ids are longer with prefixes; accept first token
        vid = vid.split('.')[0]
        # pick local transcript if available
        text = transcripts_map.get(vid, '')
        local_texts.append(text if text else pd.NA)

    df['local_transcript'] = local_texts

    out_path = args.output
    if args.inplace:
        out_path = args.input
    df.to_parquet(out_path, index=False)
    print(f'Wrote updated parquet to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input parquet file path')
    parser.add_argument('--output', required=False, help='Output parquet path', default='Data_Cleaned/yt_videos_with_local_transcripts.parquet')
    parser.add_argument('--inplace', action='store_true', help='Overwrite input file')
    args = parser.parse_args()
    main(args)
