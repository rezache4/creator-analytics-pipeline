"""Semantic chapter embedding pipeline for YouTube transcripts.

Processing strategy
-------------------
Short videos (is_short=True):
    Embed the full transcript as a single vector -> chapter_index=0.

Long videos (is_short=False):
    1.  Split transcript into overlapping BASE_CHUNK_WORDS-word base chunks.
    2.  Embed all base chunks in batches via Vertex AI.
    3.  Detect chapter breaks where sequential cosine similarity drops below
        SIMILARITY_THRESHOLD.
    4.  Mean-pool base-chunk vectors within each detected chapter to produce
        one "chapter vector" per semantic segment.

Output schema
-------------
Parquet with columns: video_id, is_short, chapter_index, embedding_vector

Downstream note (Phase 2 / SAE): the SAE trains on chapter-level vectors.
To reconstruct per-video sparse features for Phase 4 regression, max-pool or
mean-pool the chapter activations back to video level after SAE encoding.

Model choice
------------
intfloat/multilingual-e5-base — runs entirely locally via sentence-transformers,
no API calls, no quota limits, no cost. 768 dims, excellent Italian support.
Vertex AI text-multilingual-embedding-002 was the original target but requires a
paid GCP account for usable quotas.

Usage (from Data_Cleaned directory)
------------------------------------
    python embed_transcripts_semantic.py
    python embed_transcripts_semantic.py --project my-proj --output dense_vectors_semantic.parquet
    python embed_transcripts_semantic.py --resume       # continue from checkpoint
    python embed_transcripts_semantic.py --dry-run      # validate input, no API calls
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID   = "intfloat/multilingual-e5-base"   # 768 dims, Italian support, local CPU
BATCH_SIZE = 64   # sentence-transformers encode batch size

WORDS_SHORT_THRESHOLD = 300   # fallback when is_short flag is absent
BASE_CHUNK_WORDS      = 100   # words per base chunk for long videos
BASE_CHUNK_OVERLAP    = 10    # word overlap between consecutive base chunks
SIMILARITY_THRESHOLD  = 0.75  # cosine drop below this registers a chapter break

GROUP_SIZE = 100   # videos checkpointed together after each group

DEFAULT_INPUT   = "yt_videos_with_local_transcripts.parquet"
DEFAULT_OUTPUT  = "dense_vectors_semantic.parquet"
CHECKPOINT_FILE = ".embed_checkpoint.jsonl"

# ── Text utilities ────────────────────────────────────────────────────────────

def word_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-count-based chunks."""
    words = text.split()
    step  = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            out.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return out


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def detect_chapters(
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """
    Group consecutive base-chunk indices into semantic chapters.
    A new chapter starts wherever cosine_sim(chunk[i-1], chunk[i]) < threshold.
    """
    if len(embeddings) <= 1:
        return [list(range(len(embeddings)))]
    mats = [np.array(e) for e in embeddings]
    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(mats)):
        if cosine_sim(mats[i - 1], mats[i]) < threshold:
            groups.append(current)
            current = [i]
        else:
            current.append(i)
    groups.append(current)
    return groups


# ── Local embedding helpers ───────────────────────────────────────────────────

def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Embed texts locally — no API, no quota, no cost."""
    # intfloat/multilingual-e5-* models expect "passage: " prefix for documents
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict[str, list[dict]]:
    """Return {video_id: [row, ...]} for every video already written."""
    completed: dict[str, list[dict]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        completed[entry["video_id"]] = entry["rows"]
    return completed


def append_group_checkpoint(path: Path, group: dict[str, list[dict]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for video_id, rows in group.items():
            f.write(json.dumps({"video_id": video_id, "rows": rows}, ensure_ascii=False) + "\n")


# ── Core transform ────────────────────────────────────────────────────────────

def chunks_to_chapter_rows(
    chunk_records: list[tuple[str, int, str, bool]],
    embeddings: list[list[float]],
    threshold: float,
) -> dict[str, list[dict]]:
    """
    Given aligned (chunk_records, embeddings), reconstruct per-video chapter rows.
    chunk_records entries: (video_id, local_chunk_idx, text, is_short)
    """
    by_video: dict[str, list[tuple[int, list[float], bool]]] = defaultdict(list)
    for (vid, idx, _, is_short), emb in zip(chunk_records, embeddings):
        by_video[vid].append((idx, emb, is_short))

    results: dict[str, list[dict]] = {}
    for video_id, entries in by_video.items():
        entries.sort(key=lambda x: x[0])
        is_short = entries[0][2]
        embs     = [e[1] for e in entries]

        if is_short:
            # Pool all base chunks → single chapter-0 vector for shorts
            pooled = np.mean(embs, axis=0).tolist() if len(embs) > 1 else embs[0]
            results[video_id] = [
                {"video_id": video_id, "is_short": True,
                 "chapter_index": 0, "embedding_vector": pooled}
            ]
        elif len(embs) == 1:
            results[video_id] = [
                {"video_id": video_id, "is_short": False,
                 "chapter_index": 0, "embedding_vector": embs[0]}
            ]
        else:
            rows = []
            for ch_idx, chunk_indices in enumerate(detect_chapters(embs, threshold)):
                pooled = np.mean([embs[i] for i in chunk_indices], axis=0).tolist()
                rows.append(
                    {
                        "video_id": video_id,
                        "is_short": is_short,
                        "chapter_index": ch_idx,
                        "embedding_vector": pooled,
                    }
                )
            results[video_id] = rows

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet",   default=DEFAULT_INPUT,
                        help="Input parquet (default: %(default)s)")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,
                        help="Output parquet (default: %(default)s)")
    parser.add_argument("--model",     default=MODEL_ID,
                        help="sentence-transformers model ID (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Cosine-sim chapter-break threshold (default: %(default)s)")
    parser.add_argument("--resume",    action="store_true",
                        help="Resume from existing checkpoint file")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Validate input and print stats; skip all API calls")
    args = parser.parse_args()

    parquet_path    = Path(args.parquet)
    output_path     = Path(args.output)
    checkpoint_path = Path(CHECKPOINT_FILE)

    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}")

    # ── Load parquet ──────────────────────────────────────────────────────────
    print(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    valid = df["local_transcript"].notna()
    print(f"  {len(df):,} rows  |  non-null transcripts: {valid.sum():,}")
    df = df[valid].copy()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    completed: dict[str, list[dict]] = {}
    if args.resume:
        completed = load_checkpoint(checkpoint_path)
        print(f"  Resuming: {len(completed):,} videos already embedded")

    # ── Build base chunks for pending videos ──────────────────────────────────
    all_chunks: list[tuple[str, int, str, bool]] = []
    for _, row in df.iterrows():
        video_id = str(row["videoId"])
        if video_id in completed:
            continue
        text     = row["local_transcript"]
        is_short = bool(row.get("is_short", len(text.split()) < WORDS_SHORT_THRESHOLD))
        # Use the same 100-word chunking for all videos — avoids large single-text
        # payloads for short videos that would exceed the 20K token batch limit.
        # Shorts: all base chunks are pooled to a single chapter-0 vector.
        for i, chunk in enumerate(word_chunks(text, BASE_CHUNK_WORDS, BASE_CHUNK_OVERLAP)):
            all_chunks.append((video_id, i, chunk, is_short))

    pending_videos = list(dict.fromkeys(r[0] for r in all_chunks))
    n_short_chunks = sum(1 for r in all_chunks if r[3])
    n_long_chunks  = len(all_chunks) - n_short_chunks

    print(f"\nPending: {len(pending_videos):,} videos  |  "
          f"{len(all_chunks):,} base chunks  "
          f"({n_short_chunks} short, {n_long_chunks} long)")

    if args.dry_run:
        print("\n[dry-run] No model loaded. Exiting.")
        return

    if not pending_videos:
        print("Nothing to do — all videos already in checkpoint.")
    else:
        # ── Load local model ──────────────────────────────────────────────────
        print(f"\nLoading local model: {args.model}")
        st_model = SentenceTransformer(args.model)
        print(f"  Embedding dim: {st_model.get_sentence_embedding_dimension()}\n")

        # Build a per-video lookup so we can slice groups efficiently
        video_chunk_idx: dict[str, list[int]] = defaultdict(list)
        for i, (vid, *_) in enumerate(all_chunks):
            video_chunk_idx[vid].append(i)

        total_groups = (len(pending_videos) + GROUP_SIZE - 1) // GROUP_SIZE
        for g, group_start in enumerate(range(0, len(pending_videos), GROUP_SIZE)):
            group_vids = pending_videos[group_start : group_start + GROUP_SIZE]
            group_chunk_indices = sorted(
                idx for vid in group_vids for idx in video_chunk_idx[vid]
            )
            group_records = [all_chunks[i] for i in group_chunk_indices]

            print(f"Group {g + 1}/{total_groups}  "
                  f"({len(group_vids)} videos, {len(group_records)} chunks)", flush=True)
            group_embs    = embed_texts(st_model, [r[2] for r in group_records])
            group_results = chunks_to_chapter_rows(group_records, group_embs, args.threshold)

            append_group_checkpoint(checkpoint_path, group_results)
            completed.update(group_results)

            n_ch = sum(len(v) for v in group_results.values())
            print(f"  -> {n_ch} chapter rows  ({n_ch / len(group_vids):.1f} avg/video)")

    # ── Write output parquet ──────────────────────────────────────────────────
    all_rows = [row for rows in completed.values() for row in rows]
    out_df   = pd.DataFrame(all_rows)
    out_df.to_parquet(output_path, index=False)

    n_videos   = out_df["video_id"].nunique()
    n_chapters = len(out_df)
    print(f"\nSaved -> {output_path}")
    print(f"  {n_videos:,} videos  |  {n_chapters:,} chapter rows  "
          f"|  {n_chapters / max(n_videos, 1):.2f} avg chapters/video")
    if checkpoint_path.exists():
        print(f"  Checkpoint: {checkpoint_path}  (delete after verifying output)")


if __name__ == "__main__":
    main()
