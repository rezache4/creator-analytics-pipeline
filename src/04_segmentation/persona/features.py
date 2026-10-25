"""Stage 0 feature engineering + clustering feature selection.

Ports the notebook's "Stage 0 - Feature Engineering" and "Feature Selection for
Clustering" cells. ``FeatureEngineer.build`` aggregates per-author behavioural
features (with optional room-vibe enrichment and a parquet cache) and attaches a
representative comment sample. ``select_numeric_features`` returns the clustering
feature list, filtered to columns that actually exist.
"""
from __future__ import annotations

import logging
import os

from .config import PipelineConfig

_log = logging.getLogger("persona_pipeline")

# Candidate clustering features (mirrors SELECTED_NUMERIC_FEATURES in the notebook).
CANDIDATE_NUMERIC_FEATURES = [
    "total_comments",
    "unique_posts_commented",
    "activity_span_days",
    # "total_replies_made",      # REDUNDANT: use reply_ratio
    "reply_ratio",
    "mean_hours_to_comment",
    # "median_hours_to_comment", # REDUNDANT: use mean
    "pct_comments_under_1h",
    # "pct_comments_under_24h",  # OPTIONAL: tight subset of 1h signal
    "mean_word_count",
    "mean_mention_count",
    "emoji_usage_rate",
    "question_rate",
    "exclamation_rate",
    "post_concentration_ratio",
]


class FeatureEngineer:
    def __init__(self, config: PipelineConfig):
        self.config = config

    # ---------------------- room-vibe bridge load ------------------------
    def _load_post_vibes(self, path: str):
        import pandas as pd

        if not os.path.exists(path):
            _log.warning(
                "POST_VIBES_PATH not found (%s) - room-vibe features will be absent. "
                "Run sentiment_pipeline retrieve step first.",
                path,
            )
            return None
        df = pd.read_parquet(
            path,
            columns=["media_id", "room_vibe", "room_consensus", "room_sponsorship_alignment"],
        )
        df["media_id"] = df["media_id"].astype(str)
        _log.info("post_vibes loaded: %d posts", len(df))
        return df

    # ----------------------- per-user aggregation ------------------------
    @staticmethod
    def _build_user_feature_matrix(df, post_vibes=None):
        import numpy as np
        import pandas as pd

        grp = df.groupby("author_id")
        agg = {
            "total_comments": grp.size(),
            "unique_posts_commented": grp["media_id"].nunique(),
            "total_replies_made": grp["is_reply"].sum(),
            "reply_ratio": grp["is_reply"].mean(),
            "mean_hours_to_comment": grp["hours_to_comment"].mean(),
            "median_hours_to_comment": grp["hours_to_comment"].median(),
            "pct_comments_under_1h": grp["hours_to_comment"].apply(lambda x: (x < 1).mean()),
            "pct_comments_under_24h": grp["hours_to_comment"].apply(lambda x: (x < 24).mean()),
            "activity_span_days": grp["timestamp"].apply(
                lambda x: (x.max() - x.min()).days if x.notna().any() else 0
            ),
            "mean_word_count": grp["word_count"].mean(),
            "mean_mention_count": grp["mention_count"].mean(),
            "emoji_usage_rate": grp["has_emoji"].mean(),
            "question_rate": grp["has_question"].mean(),
            "exclamation_rate": grp["has_exclaim"].mean(),
        }
        feat = pd.DataFrame(agg).reset_index()
        feat["post_concentration_ratio"] = (
            feat["unique_posts_commented"] / feat["total_comments"]
        ).clip(upper=1.0)
        for c in [
            "mean_hours_to_comment",
            "median_hours_to_comment",
            "pct_comments_under_1h",
            "pct_comments_under_24h",
        ]:
            feat[c] = feat[c].fillna(0)

        if post_vibes is not None:
            enriched = df[["author_id", "media_id"]].merge(post_vibes, on="media_id", how="left")
            vibe_grp = enriched.groupby("author_id")
            feat = feat.merge(
                vibe_grp["room_consensus"].mean().rename("mean_engaged_consensus"),
                on="author_id", how="left",
            )
            feat = feat.merge(
                vibe_grp["room_sponsorship_alignment"].mean().rename("mean_sponsorship_tolerance"),
                on="author_id", how="left",
            )
            dominant_vibe = (
                vibe_grp["room_vibe"]
                .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else "neutral")
                .rename("dominant_room_vibe")
            )
            feat = feat.merge(dominant_vibe, on="author_id", how="left")
            feat["dominant_room_vibe"] = feat["dominant_room_vibe"].fillna("neutral").astype(str)
            n_enriched = feat["mean_engaged_consensus"].notna().sum()
            _log.info("room-vibe features attached for %d / %d users", n_enriched, len(feat))
            print(f"  room-vibe enrichment: {n_enriched:,}/{len(feat):,} users have vibe data")
        else:
            feat["mean_engaged_consensus"] = np.nan
            feat["mean_sponsorship_tolerance"] = np.nan
            feat["dominant_room_vibe"] = "neutral"
            print("  room-vibe enrichment skipped (post_vibes not loaded)")
        return feat

    # ----------------------------- public --------------------------------
    def build(self, ig_comments, ig_media):
        """Return user_features. Uses a parquet cache when present.

        ``ig_comments`` is mutated in place with derived flag/temporal columns,
        matching the notebook (so downstream sample fallbacks can re-join on it).
        """
        import numpy as np
        import pandas as pd

        cfg = self.config

        # 1) binary flags
        ig_comments["has_emoji"] = (ig_comments["emoji_count"] > 0).astype(int)
        ig_comments["has_question"] = (ig_comments["question_count"] > 0).astype(int)
        ig_comments["has_exclaim"] = (ig_comments["exclamation_count"] > 0).astype(int)
        ig_comments["is_reply"] = ig_comments["reply_to_comment_id"].notna().astype(int)
        print("Binary flags derived.")

        # 2) temporal features vs post time
        ig_comments["timestamp"] = pd.to_datetime(
            ig_comments["timestamp"], errors="coerce", utc=True
        )
        if cfg.attach_media and ig_media is not None and "timestamp" in ig_media.columns:
            pm = ig_media[["media_id", "timestamp"]].rename(columns={"timestamp": "post_timestamp"})
            ig_comments.drop(
                columns=[c for c in ["post_timestamp", "hours_to_comment"] if c in ig_comments.columns],
                inplace=True,
            )
            ig_comments = ig_comments.merge(pm, on="media_id", how="left")
            ig_comments["post_timestamp"] = pd.to_datetime(
                ig_comments["post_timestamp"], errors="coerce", utc=True
            )
            ig_comments["hours_to_comment"] = (
                (ig_comments["timestamp"] - ig_comments["post_timestamp"]).dt.total_seconds() / 3600
            ).clip(lower=0)
        else:
            ig_comments["hours_to_comment"] = np.nan
            print("   (no post metadata -> hours_to_comment = NaN)")

        # cached aggregation
        cache = cfg.user_features_cache_path
        if os.path.exists(cache):
            print(f"Loading cached user_features from {cache} ...")
            user_features = pd.read_parquet(cache)
            print(f"  cached user_features loaded: {len(user_features):,} users")
            return ig_comments, user_features

        print("Computing user_features (no cache found) ...")
        post_vibes = self._load_post_vibes(cfg.post_vibes_path)
        user_features = self._build_user_feature_matrix(ig_comments, post_vibes=post_vibes)
        print(f"User feature matrix built: {len(user_features):,} users.")

        # representative comment text
        print(f"Loading comment text from {cfg.comments_llm_path} ...")
        text_chunks = []
        for chunk in pd.read_json(cfg.comments_llm_path, lines=True, chunksize=200_000):
            if "platform" in chunk.columns:
                chunk = chunk[chunk["platform"] == cfg.platform]
            if len(chunk):
                text_chunks.append(chunk[["comment_id", "text"]])
        llm_text = (
            pd.concat(text_chunks, ignore_index=True)
            if text_chunks
            else pd.DataFrame(columns=["comment_id", "text"])
        )
        txt = ig_comments[["comment_id", "author_id"]].merge(llm_text, on="comment_id", how="inner")
        top_comments = (
            txt.groupby("author_id").head(5)
            .groupby("author_id")["text"]
            .apply(lambda ts: " ||| ".join(ts.astype(str).tolist()))
            .reset_index()
            .rename(columns={"text": "top_comments_sample"})
        )
        user_features = user_features.merge(top_comments, on="author_id", how="left")
        print(f"Representative comment history attached for {top_comments.shape[0]:,} users.")

        user_features.to_parquet(cache, index=False)
        print(f"  cached user_features to {cache}")
        return ig_comments, user_features

    @staticmethod
    def select_numeric_features(user_features) -> list:
        selected = [c for c in CANDIDATE_NUMERIC_FEATURES if c in user_features.columns]
        print(f"Selected {len(selected)} numeric features:")
        for f in selected:
            print(f"   - {f}")
        return selected
