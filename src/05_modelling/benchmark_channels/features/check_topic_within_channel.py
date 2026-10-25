"""
check_topic_within_channel.py

Question: do topic clusters vary WITHIN the cohort channels, or is topic
essentially constant within a channel (absorbed by channel fixed effects)?

If within-channel topic variation is negligible, the K-1 cluster dummies are
redundant with C(channelTitle) and should be dropped (they only spend df).
If meaningful, they earn their place as controls.

Unit: long-form post-2022 cohort videos (COHORT from cohort_config). Uses dominant_cluster
from video_profiles.parquet.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cohort_config import COHORT, POST2022

BASE = Path(__file__).resolve().parent
PROJ = BASE.parent.parent  #  (project data root)

dv = pd.read_parquet(PROJ / "embeddings" / "dense_vectors_semantic.parquet",
                     columns=["video_id", "is_short"])
vid = dv.groupby("video_id")["is_short"].first().reset_index()
meta = pd.read_parquet(PROJ / "clean_data" / "yt_videos_with_local_transcripts.parquet",
                       columns=["videoId", "channelTitle", "publishedAt"]).rename(columns={"videoId": "video_id"})
meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")
vp = pd.read_parquet(PROJ / "topic_clustering" / "video_profiles.parquet",
                     columns=["video_id", "dominant_cluster"])

v = vid.merge(meta, on="video_id").merge(vp, on="video_id", how="left")
v = v[(v.channelTitle.isin(COHORT)) & (v.publishedAt >= POST2022) & (~v.is_short)]

print(f"N = {len(v)} long-form post-2022 cohort videos\n")

# dominant-cluster distribution per channel
print("Dominant-cluster distribution per channel (row-normalised %):")
tab = pd.crosstab(v.channelTitle, v.dominant_cluster, normalize="index") * 100
print(tab.round(1).to_string())

# within-channel entropy / concentration: what share is the channel's top cluster?
print("\nPer-channel topic concentration:")
for ch, g in v.groupby("channelTitle"):
    vc = g.dominant_cluster.value_counts(normalize=True)
    top_share = vc.iloc[0] * 100
    n_clusters = (vc > 0.02).sum()  # clusters with >2% of the channel's videos
    print(f"  {ch:<18} top cluster = {top_share:5.1f}% of videos | "
          f"{n_clusters} clusters >2% | n_distinct = {g.dominant_cluster.nunique()}")

# overall: how much topic variation is BETWEEN vs WITHIN channel?
# eta-squared style: variance of cluster explained by channel
overall_top = v.dominant_cluster.value_counts(normalize=True).iloc[0] * 100
print(f"\nPooled: top cluster = {overall_top:.1f}% of all {len(v)} videos")
print("\nVerdict guide:")
print("  - If each channel sits ~entirely in one cluster -> topic ~ constant within")
print("    channel -> C(channelTitle) absorbs it -> DROP cluster dummies.")
print("  - If channels span several clusters across their videos -> within-channel")
print("    topic variation exists -> cluster dummies earn their place as controls.")
