"""
check_chapter_coverage.py

Quantify topic_volatility coverage: how many long-form cohort videos have
>= 2 chapters (the minimum for topic_volatility to be defined). Decides whether
topic_volatility is worth keeping as a feature.
"""
from pathlib import Path
import pandas as pd
from cohort_config import COHORT, POST2022

BASE = Path(__file__).resolve().parent
PROJ = BASE.parent.parent  #  (project data root)

vp = pd.read_parquet(PROJ / "topic_clustering" / "video_profiles.parquet",
                     columns=["video_id", "n_chapters"])
dv = pd.read_parquet(PROJ / "embeddings" / "dense_vectors_semantic.parquet",
                     columns=["video_id", "is_short"])
vid = dv.groupby("video_id")["is_short"].first().reset_index()
meta = pd.read_parquet(PROJ / "clean_data" / "yt_videos_with_local_transcripts.parquet",
                       columns=["videoId", "channelTitle", "publishedAt"]).rename(columns={"videoId": "video_id"})
meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")

v = vid.merge(meta, on="video_id").merge(vp, on="video_id", how="left")
v = v[(v.channelTitle.isin(COHORT)) & (v.publishedAt >= POST2022) & (~v.is_short)]

print(f"long-form cohort videos: {len(v)}")
ge2 = (v.n_chapters >= 2)
print(f"n_chapters >= 2 (topic_volatility computable): {int(ge2.sum())} ({ge2.mean()*100:.1f}%)")
print(f"exactly 1 chapter: {(v.n_chapters == 1).mean()*100:.1f}%")
print("\nn_chapters describe:")
print(v.n_chapters.describe().round(2).to_string())
print("\nper-channel median / mean n_chapters / count:")
print(v.groupby("channelTitle").n_chapters.agg(["median", "mean", "count"]).round(2).to_string())
