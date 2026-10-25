"""Named-entity extraction for transcripts and YouTube comments.

Runs spaCy (it_core_news_lg) over both sources and saves per-video entity
summaries so downstream analysis can answer: which entities discussed in the
video were also mentioned in its comments?

Output files
------------
entities_transcripts.parquet   -- one row per video
    video_id, entities  (list of {text, label, count} dicts, sorted by count desc)

entities_comments.parquet      -- one row per video
    video_id, entities  (aggregated across all comments for that video)

entity_resonance.parquet       -- joined view
    video_id, entity_text, entity_label,
    in_transcript (bool), transcript_count (int),
    in_comments (bool), comment_count (int)

Setup (run once)
----------------
    pip install spacy
    python -m spacy download it_core_news_lg

Usage (from Data_Cleaned directory)
-------------------------------------
    python extract_entities.py
    python extract_entities.py --transcripts-only
    python extract_entities.py --comments-only
    python extract_entities.py --dry-run        # process 50 rows each, no write
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

# ── Configuration ─────────────────────────────────────────────────────────────

SPACY_MODEL          = "it_core_news_lg"
# Resolve paths relative to the script file so the script can be run
# from the repo root or any other working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
TRANSCRIPT_PARQUET   = SCRIPT_DIR / "yt_videos_with_local_transcripts.parquet"
COMMENT_FILES        = sorted(SCRIPT_DIR.glob("yt_comments_*_cleaned.parquet"))
OUT_TRANSCRIPTS      = SCRIPT_DIR / "entities_transcripts.parquet"
OUT_COMMENTS         = SCRIPT_DIR / "entities_comments.parquet"
OUT_RESONANCE        = SCRIPT_DIR / "entity_resonance.parquet"

# Entity label filter — keep only types meaningful for engagement analysis
KEEP_LABELS = {
    "PER",   # persons
    "ORG",   # organisations / brands / channels
    "LOC",   # locations
    "GPE",   # geo-political entities
    "MISC",  # events, works of art, etc.
    "PROD",  # products (not in all spaCy Italian models; included if present)
}

# Minimum character length to avoid noise tokens
MIN_ENTITY_LEN = 2

# spaCy batch size (texts per call to nlp.pipe)
NLP_BATCH_SIZE = 64

# ── Core NER helpers ──────────────────────────────────────────────────────────

def extract_entities_from_text(nlp: spacy.Language, text: str) -> list[dict]:
    """Return [{text, label, count}] for one document, sorted by count desc."""
    doc     = nlp(text)
    counter = Counter(
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in KEEP_LABELS and len(ent.text.strip()) >= MIN_ENTITY_LEN
    )
    return [
        {"text": t, "label": lbl, "count": cnt}
        for (t, lbl), cnt in counter.most_common()
    ]


def extract_entities_batch(
    nlp: spacy.Language,
    texts: list[str],
    ids: list[str],
) -> list[dict]:
    """
    Run nlp.pipe over texts and return rows ready for a DataFrame.
    rows: [{"video_id": ..., "entities": [...]}, ...]
    """
    rows = []
    for doc, vid in zip(
        nlp.pipe(texts, batch_size=NLP_BATCH_SIZE, disable=["parser"]),
        ids,
    ):
        counter = Counter(
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.label_ in KEEP_LABELS and len(ent.text.strip()) >= MIN_ENTITY_LEN
        )
        entities = [
            {"text": t, "label": lbl, "count": cnt}
            for (t, lbl), cnt in counter.most_common()
        ]
        rows.append({"video_id": vid, "entities": json.dumps(entities, ensure_ascii=False)})
    return rows


def aggregate_comment_entities(
    nlp: spacy.Language,
    df_comments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract entities from all comments, then aggregate counts per video.
    Returns DataFrame: video_id, entities (JSON string of [{text,label,count}]).
    """
    texts   = df_comments["textOriginal"].fillna("").astype(str).tolist()
    vid_ids = df_comments["videoId"].astype(str).tolist()

    # Accumulate per-video entity counters
    vid_counters: dict[str, Counter] = {}
    total = len(texts)
    for i, (doc, vid) in enumerate(
        zip(nlp.pipe(texts, batch_size=NLP_BATCH_SIZE, disable=["parser"]), vid_ids)
    ):
        if vid not in vid_counters:
            vid_counters[vid] = Counter()
        for ent in doc.ents:
            t = ent.text.strip()
            if ent.label_ in KEEP_LABELS and len(t) >= MIN_ENTITY_LEN:
                vid_counters[vid][(t, ent.label_)] += 1

        if (i + 1) % 5000 == 0:
            print(f"  comments processed: {i + 1:>7}/{total}", end="\r", flush=True)

    print()
    rows = []
    for vid, counter in vid_counters.items():
        entities = [
            {"text": t, "label": lbl, "count": cnt}
            for (t, lbl), cnt in counter.most_common()
        ]
        rows.append({"video_id": vid, "entities": json.dumps(entities, ensure_ascii=False)})

    return pd.DataFrame(rows)


# ── Resonance join ────────────────────────────────────────────────────────────

def build_resonance(
    df_trans: pd.DataFrame,
    df_comm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join transcript and comment entity summaries into a flat resonance table.
    One row per (video_id, entity_text, entity_label).
    """
    def explode_entities(df: pd.DataFrame, count_col: str) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            vid = row["video_id"]
            ents = json.loads(row["entities"]) if isinstance(row["entities"], str) else row["entities"]
            for e in ents:
                records.append(
                    {
                        "video_id":    vid,
                        "entity_text":  e["text"],
                        "entity_label": e["label"],
                        count_col:      e["count"],
                    }
                )
        return pd.DataFrame(records)

    t = explode_entities(df_trans, "transcript_count")
    c = explode_entities(df_comm,  "comment_count")

    merged = pd.merge(
        t, c,
        on=["video_id", "entity_text", "entity_label"],
        how="outer",
    )
    merged["transcript_count"] = merged["transcript_count"].fillna(0).astype(int)
    merged["comment_count"]    = merged["comment_count"].fillna(0).astype(int)
    merged["in_transcript"]    = merged["transcript_count"] > 0
    merged["in_comments"]      = merged["comment_count"] > 0
    return merged.sort_values(["video_id", "comment_count"], ascending=[True, False])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--transcript-parquet", type=str,
                        help="Path to transcript parquet file (overrides default)")
    parser.add_argument("--gcs-bucket", type=str, default=None,
                        help="GCS bucket name. Reads from gs://{bucket}/input/ and writes "
                             "to gs://{bucket}/output/ner/. Overrides local paths. "
                             "Can also be set via the GCS_BUCKET env var.")
    parser.add_argument("--transcripts-only", action="store_true")
    parser.add_argument("--comments-only",    action="store_true")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Process only 50 rows per source; do not write output files")
    args = parser.parse_args()

    do_transcripts = not args.comments_only
    do_comments    = not args.transcripts_only

    # ── Resolve paths (local vs GCS) ──────────────────────────────────────────
    gcs_bucket = args.gcs_bucket or os.environ.get("GCS_BUCKET")
    if gcs_bucket:
        bucket = gcs_bucket.strip("/")
        _in    = f"gs://{bucket}/input"
        _out   = f"gs://{bucket}/output/ner"
        import gcsfs
        _fs = gcsfs.GCSFileSystem()
        transcript_path = f"{_in}/yt_videos_with_local_transcripts.parquet"
        comment_paths   = sorted(f"gs://{p}" for p in
                                 _fs.glob(f"{bucket}/input/yt_comments_*_cleaned.parquet"))
        out_t = f"{_out}/entities_transcripts.parquet"
        out_c = f"{_out}/entities_comments.parquet"
        out_r = f"{_out}/entity_resonance.parquet"
        print(f"GCS mode  bucket={bucket}")
        print(f"  input : {_in}/")
        print(f"  output: {_out}/")
    else:
        transcript_path = (
            Path(args.transcript_parquet).expanduser().resolve()
            if args.transcript_parquet else TRANSCRIPT_PARQUET
        )
        comment_paths = sorted(SCRIPT_DIR.glob("yt_comments_*_cleaned.parquet"))
        out_t = OUT_TRANSCRIPTS
        out_c = OUT_COMMENTS
        out_r = OUT_RESONANCE

    print(f"Loading spaCy model: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)
    print(f"  Pipeline: {nlp.pipe_names}")

    # ── Transcripts ───────────────────────────────────────────────────────────
    df_trans: pd.DataFrame | None = None
    if do_transcripts:
        print(f"\nLoading {transcript_path} ...")
        df_vid = pd.read_parquet(transcript_path)
        df_vid = df_vid[df_vid["local_transcript"].notna()].copy()

        if args.dry_run:
            df_vid = df_vid.head(50)
            print("  [dry-run] limited to 50 videos")

        texts = df_vid["local_transcript"].astype(str).tolist()
        ids   = df_vid["videoId"].astype(str).tolist()
        print(f"  Extracting entities from {len(texts):,} transcripts ...")

        rows      = extract_entities_batch(nlp, texts, ids)
        df_trans  = pd.DataFrame(rows)
        print(f"  Done: {len(df_trans):,} rows")

        if not args.dry_run:
            df_trans.to_parquet(out_t, index=False)
            print(f"  Saved -> {out_t}")
        else:
            print("  [dry-run] transcript output not written")

    # ── Comments ──────────────────────────────────────────────────────────────
    df_comm: pd.DataFrame | None = None
    if do_comments:
        if not comment_paths:
            print("\nNo yt_comments_*_cleaned.parquet files found — skipping comments.")
        else:
            frames = [pd.read_parquet(p) for p in comment_paths]
            df_c   = pd.concat(frames, ignore_index=True)
            df_c   = df_c[df_c["textOriginal"].notna()].copy()

            if args.dry_run:
                df_c = df_c.head(50)
                print("\n  [dry-run] limited to 50 comments")

            print(f"\nExtracting entities from {len(df_c):,} comments ...")
            df_comm = aggregate_comment_entities(nlp, df_c)
            print(f"  Done: {len(df_comm):,} video rows")

            if not args.dry_run:
                df_comm.to_parquet(out_c, index=False)
                print(f"  Saved -> {out_c}")
            else:
                print("  [dry-run] comment output not written")

    # ── Resonance ─────────────────────────────────────────────────────────────
    if do_transcripts and do_comments and df_trans is not None and df_comm is not None:
        print("\nBuilding entity resonance table ...")
        df_res = build_resonance(df_trans, df_comm)
        print(f"  {len(df_res):,} (video, entity) pairs")

        if not args.dry_run:
            df_res.to_parquet(out_r, index=False)
            print(f"  Saved -> {out_r}")
        else:
            print("  [dry-run] resonance output not written")
            print(df_res.head(10).to_string())

    print("\nAll done.")


if __name__ == "__main__":
    main()
