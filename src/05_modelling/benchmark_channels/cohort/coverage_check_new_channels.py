"""
coverage_check_new_channels.py

Data-coverage audit for channels newly admitted to the cohort after the
semantic gate was dropped (2026-06-12).

For each new channel, reports % of its long-form post-2022 videos present
in every source table the pipeline reads. Near-zero coverage in any required
table is surfaced explicitly.

New channels (previously failed cosine gate only):
  ComparatorChannel11, ComparatorChannel6, AlicelikeAudrey, ComparatorChannel10, ComparatorChannel9, Davide Zambelli

Output: printed report + coverage_check_new_channels.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent               # cohort/
PROJ = BASE.parent.parent                             # 
sys.path.insert(0, str(BASE.parent / "features"))
from cohort_config import COHORT, TARGET, POST2022

ORIG_COHORT = {"ComparatorChannel4", "ComparatorChannel1", "ComparatorChannel2", "ComparatorChannel3"}
NEW_CHANNELS = [c for c in COHORT if c not in ORIG_COHORT and c != TARGET]

DENSE    = PROJ / "embeddings"      / "dense_vectors_semantic.parquet"
VIDEOS   = PROJ / "clean_data"      / "yt_videos_with_local_transcripts.parquet"
VPROF    = PROJ / "topic_clustering"/ "video_profiles.parquet"
THUMB    = PROJ / "clean_data"      / "thumbnail_features.parquet"
COMMENTS = PROJ / "comments"        / "Preped_Comments_comments_ml.parquet"
SENT     = BASE.parent / "features" / "comment_features" / "video_comment_features.parquet"
SPEECH   = BASE.parent / "features" / "speech_rate.parquet"
VTT_DIR  = PROJ / "transcripts" / "vtt_files"


def baseline_ids() -> set[str]:
    """Long-form post-2022 video IDs for NEW channels."""
    dv   = pd.read_parquet(DENSE, columns=["video_id", "is_short"])
    vid  = dv.groupby("video_id")["is_short"].first().reset_index()
    meta = pd.read_parquet(VIDEOS, columns=["videoId", "channelTitle", "publishedAt"]).rename(
        columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")
    v = vid.merge(meta, on="video_id")
    v = v[(v.channelTitle.isin(NEW_CHANNELS)) & (v.publishedAt >= POST2022) & (~v.is_short)]
    return v


def coverage_pct(ids: set[str], present: set[str]) -> float:
    if not ids:
        return float("nan")
    return round(len(ids & present) / len(ids) * 100, 1)


def main():
    print("=" * 72)
    print(f"Coverage check — {len(NEW_CHANNELS)} newly-admitted channels")
    print("  " + ", ".join(NEW_CHANNELS))
    print("=" * 72)

    v = baseline_ids()
    print(f"\nBaseline (long-form post-2022): {len(v)} videos across new channels")
    print(v.groupby("channelTitle").size().to_string())

    # per-channel ID sets
    ch_ids = {ch: set(v[v.channelTitle == ch]["video_id"].astype(str))
              for ch in NEW_CHANNELS}
    all_ids = set(v["video_id"].astype(str))

    rows = []

    # --- embeddings ---
    emb_ids = set(pd.read_parquet(DENSE, columns=["video_id"])["video_id"].astype(str))
    for ch in NEW_CHANNELS:
        rows.append({"channel": ch, "table": "embeddings",
                     "n_videos": len(ch_ids[ch]),
                     "coverage_pct": coverage_pct(ch_ids[ch], emb_ids)})

    # --- video meta / transcripts ---
    meta_ids = set(pd.read_parquet(VIDEOS, columns=["videoId"])["videoId"].astype(str))
    for ch in NEW_CHANNELS:
        rows.append({"channel": ch, "table": "yt_videos_meta",
                     "n_videos": len(ch_ids[ch]),
                     "coverage_pct": coverage_pct(ch_ids[ch], meta_ids)})

    # --- topic clusters ---
    vp_ids = set(pd.read_parquet(VPROF, columns=["video_id"])["video_id"].astype(str))
    for ch in NEW_CHANNELS:
        rows.append({"channel": ch, "table": "video_profiles (topic)",
                     "n_videos": len(ch_ids[ch]),
                     "coverage_pct": coverage_pct(ch_ids[ch], vp_ids)})

    # --- thumbnails ---
    th_ids = set(pd.read_parquet(THUMB, columns=["video_id"])["video_id"].astype(str))
    for ch in NEW_CHANNELS:
        rows.append({"channel": ch, "table": "thumbnail_features",
                     "n_videos": len(ch_ids[ch]),
                     "coverage_pct": coverage_pct(ch_ids[ch], th_ids)})

    # --- raw comments (comment_quality outcomes) ---
    c = pd.read_parquet(COMMENTS, columns=["media_id", "platform"])
    c = c[c["platform"] == "youtube"]
    com_ids = set(c["media_id"].astype(str))
    for ch in NEW_CHANNELS:
        rows.append({"channel": ch, "table": "raw_comments",
                     "n_videos": len(ch_ids[ch]),
                     "coverage_pct": coverage_pct(ch_ids[ch], com_ids)})

    # --- LLM sentiment ---
    if SENT.exists():
        sent_ids = set(pd.read_parquet(SENT, columns=["media_id"])["media_id"].astype(str))
        for ch in NEW_CHANNELS:
            rows.append({"channel": ch, "table": "sentiment (LLM)",
                         "n_videos": len(ch_ids[ch]),
                         "coverage_pct": coverage_pct(ch_ids[ch], sent_ids)})
    else:
        for ch in NEW_CHANNELS:
            rows.append({"channel": ch, "table": "sentiment (LLM)",
                         "n_videos": len(ch_ids[ch]),
                         "coverage_pct": 0.0})

    # --- speech rate (VTT) ---
    if SPEECH.exists():
        sr_ids = set(pd.read_parquet(SPEECH, columns=["video_id"])["video_id"].astype(str))
        for ch in NEW_CHANNELS:
            rows.append({"channel": ch, "table": "speech_rate (VTT)",
                         "n_videos": len(ch_ids[ch]),
                         "coverage_pct": coverage_pct(ch_ids[ch], sr_ids)})
    else:
        for ch in NEW_CHANNELS:
            rows.append({"channel": ch, "table": "speech_rate (VTT)",
                         "n_videos": len(ch_ids[ch]),
                         "coverage_pct": float("nan")})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="channel", columns="table", values="coverage_pct")

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}%".format)
    print("\nCoverage % (long-form post-2022 videos):\n")
    print(pivot.to_string())

    print("\n--- FLAGS (coverage < 50%) ---")
    low = df[df["coverage_pct"] < 50].sort_values("coverage_pct")
    if len(low):
        print(low.to_string(index=False))
    else:
        print("  (none — all tables >= 50% for all new channels)")

    out = BASE / "coverage_check_new_channels.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
