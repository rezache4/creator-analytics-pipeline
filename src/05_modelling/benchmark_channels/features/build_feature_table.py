"""
build_feature_table.py

Build the regression-ready feature table for the long-form cohort and write it to
parquet. Cohort is loaded from cohort_config.py (reads cohort_selection.csv).
N=1453 as of 2026-06-12 (10 comparator channels + TargetCreator as FE level).

Applies:
  - topic-cluster dummies (K-1, cluster 4 excluded; included from 2026-06-12 because
    clusters now span channels in the expanded 10-channel cohort)
  - FEATURE_LIST_FINAL.md SS D2 transforms (log outcomes; log1p the 5 skewed predictors)
  - video_age_days: continuous view-accumulation control (API-call proxy 2026-03-14)
  - standardised copies of continuous predictors (for scale-fair Lasso)
  - topic_volatility imputation (channel median)

Outputs (in benchmark_analysis/features/):
  - feature_table.parquet        one row per video, all predictors + controls + outcomes
  - feature_table_manifest.csv   column inventory: role, transform, missing%

Run: python build_feature_table.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from cohort_config import COHORT, TARGET, POST2022

BASE = Path(__file__).resolve().parent       # features/
PROJ = BASE.parent.parent                     #   (project data root)
OUT_PARQUET = BASE / "feature_table.parquet"
OUT_MANIFEST = BASE / "feature_table_manifest.csv"
MUSIC_PLACEHOLDER_CLUSTER = 4  # excluded from topic-cluster control (SS B)

# Proxy for "data collection date": newest comment in the dataset (2026-03-14).
# Used to compute video_age_days = days from publishedAt to this date, replacing
# upload_year as the view-accumulation control (continuous, no binning artefacts).
API_CALL_DATE = pd.Timestamp("2026-03-14", tz="UTC")

DENSE  = PROJ / "embeddings" / "dense_vectors_semantic.parquet"
VIDEOS = PROJ / "clean_data" / "yt_videos_with_local_transcripts.parquet"
VPROF  = PROJ / "topic_clustering" / "video_profiles.parquet"
THUMB  = PROJ / "clean_data" / "thumbnail_features.parquet"
COMMENTS = PROJ / "comments" / "Preped_Comments_comments_ml.parquet"
SPEECH = BASE / "speech_rate.parquet"                       # from extract_speech_rate.py
SENTIMENT = BASE / "comment_features" / "video_comment_features.parquet"  # LLM sentiment outcomes

# Thumbnail scalars from FEATURE_LIST_FINAL.md SS A (available without new model calls).
THUMB_KEEP = [
    "brightness_mean", "saturation_mean", "text_present", "text_area_ratio",
    "text_word_count", "text_has_number", "clip_title_align", "face_present",
    "face_count", "face_area_ratio", "face_position_x", "face_position_y",
    "face_center_distance", "face_yaw_deg", "face_pitch_deg", "face_is_frontal",
    "mouth_openness", "eye_openness_mean",
    "contrast", "warm_cool_ratio", "dominant_hue_entropy", "edge_density",
    "visual_complexity", "rule_of_thirds_score", "left_right_asymmetry",
    "top_bottom_asymmetry",
]

# SS D2: the 5 predictors with |skew| >= 2 -> log1p (all have zeros).
LOG1P_PREDICTORS = [
    "days_since_last_video", "text_area_ratio", "text_word_count",
    "face_area_ratio", "warm_cool_ratio",
]

# Continuous prescriptive + thumbnail predictors that get a standardised copy for Lasso.
# Binary/categorical (is_weekend, text_present, face_present, ... , hour_band) are excluded.
BINARY_OR_CAT = {
    "is_weekend", "hour_band", "text_present", "text_has_number",
    "face_present", "face_is_frontal", "face_pose_available",
}


def build_modeling_set():
    """Assemble the long-form post-2022 cohort set (COHORT from cohort_config)."""
    dv = pd.read_parquet(DENSE, columns=["video_id", "is_short"])
    vid = dv.groupby("video_id")["is_short"].first().reset_index()
    meta = pd.read_parquet(VIDEOS, columns=[
        "videoId", "channelTitle", "publishedAt", "title",
        "viewCount", "likeCount", "commentCount", "duration_seconds"]).rename(
        columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")

    v = vid.merge(meta, on="video_id", how="inner")
    v = v[(v.channelTitle.isin(COHORT)) & (v.publishedAt >= POST2022) & (~v.is_short)].copy()

    # --- structural features (n_chapters + topic cluster control) ---
    vp = pd.read_parquet(VPROF, columns=["video_id", "n_chapters", "dominant_cluster"])
    v = v.merge(vp, on="video_id", how="left")
    v = v.merge(topic_volatility(), on="video_id", how="left")

    # --- derived cadence features (publish dates, cohort-wide) ---
    v = v.sort_values(["channelTitle", "publishedAt"])
    v["days_since_last_video"] = (
        v.groupby("channelTitle")["publishedAt"].diff().dt.total_seconds() / 86400)
    v["format_posting_freq_60d"] = posting_freq_60d(v)

    # --- thumbnail features ---
    tf = pd.read_parquet(THUMB)
    tf = tf[["video_id"] + [c for c in THUMB_KEEP if c in tf.columns]]
    v = v.merge(tf, on="video_id", how="left")

    # --- derived basics + outcomes ---
    v["duration_minutes"] = pd.to_numeric(v["duration_seconds"], errors="coerce") / 60.0
    v["upload_year"] = v["publishedAt"].dt.year
    # video_age_days: continuous view-accumulation control (days from publish to data
    # collection proxy 2026-03-14 = newest comment date in the dataset).
    v["video_age_days"] = (API_CALL_DATE - v["publishedAt"]).dt.total_seconds() / 86400
    v["is_weekend"] = v["publishedAt"].dt.dayofweek.isin([5, 6]).astype(int)
    v["hour_band"] = pd.cut(v["publishedAt"].dt.hour, [-1, 5, 11, 17, 23],
                            labels=["night", "morning", "afternoon", "evening"])
    # day x hour publication interaction (creator timing choice): cyclical-free
    # categorical cross of weekday-bucket and hour-band. Modelled as a single
    # categorical so OLS makes its own dummies; reference = most common cell.
    v["dow_hour"] = (v["publishedAt"].dt.dayofweek.map(
        {0: "wd", 1: "wd", 2: "wd", 3: "wd", 4: "wd", 5: "we", 6: "we"})
        + "_" + v["hour_band"].astype(str))

    # --- title features (pre-publication creator choices; no LLM/embeddings) ---
    v = add_title_features(v)

    for c in ["viewCount", "likeCount", "commentCount"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["comment_view_rate"] = v["commentCount"] / v["viewCount"].replace(0, np.nan)
    v["like_view_rate"] = v["likeCount"] / v["viewCount"].replace(0, np.nan)

    # face_pose_available flag (FaceMesh pose-failure caveat, SS D)
    if "face_yaw_deg" in v and "face_is_frontal" in v:
        v["face_pose_available"] = (~((v["face_present"] == 1) &
                                      (v["face_yaw_deg"] == 0) &
                                      (v["face_is_frontal"] == False))).astype(int)
    return v


def add_title_features(v):
    """Title string features -- all creator-controlled, pre-publish, no LLM/embeddings.
    Italian titles; counts are on the raw title string. Operate on a plain python-str
    Series (object dtype) so plain-Python ops work regardless of arrow-backed source."""
    t = v["title"].fillna("").astype("object").map(str)

    def _emoji(s):  # any pictographic char above the BMP Latin/symbol ranges
        return int(any(ord(ch) >= 0x2600 and not ch.isspace() and
                       (0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF)
                       for ch in s))

    v["title_char_len"] = t.map(len)
    v["title_word_count"] = t.map(lambda s: len(s.split()))
    v["title_has_question"] = t.map(lambda s: int("?" in s))
    v["title_has_number"] = t.map(lambda s: int(any(ch.isdigit() for ch in s)))
    v["title_exclam_count"] = t.map(lambda s: s.count("!"))
    v["title_has_mention"] = t.map(lambda s: int("@" in s))
    # ALL-CAPS intensity: share of alphabetic chars that are uppercase (0 when no letters)
    letters = t.map(lambda s: sum(ch.isalpha() for ch in s))
    caps = t.map(lambda s: sum(ch.isupper() for ch in s))
    v["title_caps_ratio"] = np.where(letters.values > 0,
                                     caps.values / np.where(letters.values > 0, letters.values, 1),
                                     0.0)
    v["title_has_emoji"] = t.map(_emoji)
    return v


def topic_volatility():
    """Mean cosine distance between consecutive chapter vectors, per video.

    RAW 768-dim embeddings (not UMAP) -- UMAP distorts cosine distance.
    """
    dv = pd.read_parquet(DENSE, columns=["video_id", "chapter_index", "embedding_vector"])
    rows = []
    for vid, g in dv.sort_values("chapter_index").groupby("video_id"):
        m = np.vstack(g["embedding_vector"].values).astype(np.float32)
        rows.append((vid, _vol(m)))
    return pd.DataFrame(rows, columns=["video_id", "topic_volatility"])


def _vol(m):
    if len(m) < 2:
        return np.nan
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    cos = (mn[:-1] * mn[1:]).sum(axis=1)
    return float((1 - cos).mean())


def posting_freq_60d(v):
    """# videos by same channel in the 60 days before each video."""
    out = np.zeros(len(v), dtype=float)
    arr = v[["channelTitle", "publishedAt"]].reset_index(drop=True)
    for ch, g in arr.groupby("channelTitle"):
        times = g["publishedAt"].values
        idx = g.index.values
        for i, t in zip(idx, times):
            lo = t - np.timedelta64(60, "D")
            out[i] = ((times >= lo) & (times < t)).sum()
    return out


# --------------------------------------------------------------------------- #
#  Transforms, imputation, controls, standardisation (the build steps).
# --------------------------------------------------------------------------- #
def add_transformed_outcomes(v):
    """SS D2 outcomes: log(viewCount); log(comment/view); like/view raw."""
    v["log_viewCount"] = np.log(v["viewCount"].replace(0, np.nan))
    v["log_comment_view_rate"] = np.log(v["comment_view_rate"].replace(0, np.nan))
    # like_view_rate kept raw (skew 0.44, already symmetric) -- carried as-is.
    return v


def add_comment_quality_outcomes(v):
    """Two engagement-QUALITY outcomes from the per-comment ML fields, ADDITIVE to
    the existing volume outcome (comment/view). PCA (comment_signal_explore.py)
    showed depth and affect are near-orthogonal (PC1 only 31%) -> two outcomes, not
    a single index:
      - depth  : log(avg_comment_word_count)         (skew 3.06 -> log)
      - affect : log1p(avg_emoji_count)              (emoji reaction; orthogonal to depth)
    Also carries n_comments (engagement-quality models share the low-comment caveat;
    use as an optional min-comment filter / control downstream).
    """
    c = pd.read_parquet(COMMENTS, columns=["media_id", "platform", "word_count", "emoji_count"])
    ids = set(v["video_id"].astype(str))
    c = c[(c["platform"] == "youtube") & (c["media_id"].astype(str).isin(ids))].copy()
    c["media_id"] = c["media_id"].astype(str)
    g = c.groupby("media_id")
    cq = pd.DataFrame({
        "n_comments": g.size(),
        "avg_comment_word_count": g["word_count"].mean(),
        "avg_comment_emoji_count": g["emoji_count"].mean(),
    }).reset_index().rename(columns={"media_id": "video_id"})

    v = v.merge(cq, on="video_id", how="left")
    v["log_comment_depth"] = np.log(v["avg_comment_word_count"].replace(0, np.nan))
    v["log_comment_affect"] = np.log1p(v["avg_comment_emoji_count"].clip(lower=0))
    return v


def add_sentiment_outcomes(v):
    """Audience-reception OUTCOMES from the LLM sentiment table (comment_features/).
    These are audience-generated (post-publish) -> valid as OUTCOMES, never predictors.
    Replaces the dropped emoji-affect model with substantive sentiment constructs:
      - vibe_score   : mean audience sentiment (-1..+1), authoritative room-vibe pass.
      - controversy  : 4*pos*neg, peaks when the room is evenly split (divisiveness).
      - room_state   : categorical {united/mixed/divided} -- carried as metadata.
    """
    if not SENTIMENT.exists():
        return v
    s = pd.read_parquet(SENTIMENT, columns=["media_id", "vibe_score", "controversy",
                                            "polarization", "toxicity_rate", "room_state"])
    s = s.rename(columns={"media_id": "video_id"})
    s["video_id"] = s["video_id"].astype(str)
    s = s.drop_duplicates("video_id")
    v = v.merge(s, on="video_id", how="left")
    # vibe_score is ~symmetric in (-1,1) -> model raw. controversy/polarization in [0,1].
    return v


def add_speech_rate(v):
    """Merge VTT speech-pacing predictors (articulation_wpm, speech_density) and impute
    the ~20% without a VTT by channel median, carrying a speech_rate_available flag
    (same convention as face_pose_available / topic_volatility_imputed)."""
    if not SPEECH.exists():
        return v
    sr = pd.read_parquet(SPEECH, columns=["video_id", "articulation_wpm",
                                          "speech_density", "speech_rate_available"])
    sr["video_id"] = sr["video_id"].astype(str)
    v = v.merge(sr, on="video_id", how="left")
    v["speech_rate_available"] = v["speech_rate_available"].fillna(0).astype(int)
    for col in ["articulation_wpm", "speech_density"]:
        med = v.groupby("channelTitle")[col].transform("median")
        v[col] = v[col].fillna(med).fillna(v[col].median())
    return v


def impute_topic_volatility(v):
    """Channel-median impute the ~10% 1-chapter NaNs; flag the imputed rows."""
    v["topic_volatility_imputed"] = v["topic_volatility"].isna().astype(int)
    med = v.groupby("channelTitle")["topic_volatility"].transform("median")
    v["topic_volatility"] = v["topic_volatility"].fillna(med)
    # fallback to global median if a channel had no valid value at all
    v["topic_volatility"] = v["topic_volatility"].fillna(v["topic_volatility"].median())
    return v


def add_log1p_predictors(v):
    """SS D2: log1p the 5 skewed predictors. Keep raw cols too (suffix _log1p)."""
    for c in LOG1P_PREDICTORS:
        if c in v:
            v[f"{c}_log1p"] = np.log1p(v[c].clip(lower=0))
    return v


def add_topic_cluster_dummies(v):
    """Topic-cluster control: INCLUDED as dummies (decision updated 2026-06-12).

    With the expanded 10-channel cohort the 4-channel verdict (95% cluster 0 ->
    nested in channel FE) no longer holds. Pooled top-cluster share dropped to 67%
    and several channels span 2-3 clusters within-channel (AlicelikeAudrey 3,
    ComparatorChannel11 3, ComparatorChannel6 2). Clusters now vary both between AND within channels,
    so the topic dummies are no longer collinear with channel FE.
    Cluster 4 (music/ASR placeholder, SS B) excluded; reference = cluster 0 (majority).
    See check_topic_within_channel.py for the reproducible justification.
    """
    if "dominant_cluster" not in v:
        return v
    valid = v["dominant_cluster"].dropna()
    clusters = sorted(c for c in valid.unique()
                      if c != MUSIC_PLACEHOLDER_CLUSTER and c != 0)
    for c in clusters:
        v[f"topic_cluster_{int(c)}"] = (v["dominant_cluster"] == c).astype(int)
    return v


def standardise_for_lasso(v):
    """Add z-scored copies (suffix _z) of continuous predictors for scale-fair Lasso.

    Uses the transformed version where one exists (e.g. days_since_last_video_log1p),
    so the penalty acts on the modelled scale.
    """
    cont = []
    for c in PRESCRIPTIVE_CONT + [t for t in THUMB_KEEP if t not in BINARY_OR_CAT]:
        col = f"{c}_log1p" if c in LOG1P_PREDICTORS else c
        if col in v and pd.api.types.is_numeric_dtype(v[col]):
            cont.append(col)
    for col in cont:
        s = v[col].astype(float)
        sd = s.std(ddof=0)
        v[f"{col}_z"] = (s - s.mean()) / sd if sd and not np.isnan(sd) else 0.0
    v.attrs["standardised_cols"] = cont
    return v


PRESCRIPTIVE_CONT = [
    "duration_minutes", "days_since_last_video", "format_posting_freq_60d",
    "n_chapters", "topic_volatility",
    # speech pacing (VTT) + title features (new prescriptive seams)
    "articulation_wpm", "speech_density",
    "title_char_len", "title_word_count", "title_exclam_count", "title_caps_ratio",
    # view-accumulation control (continuous; replaces/supplements upload_year)
    "video_age_days",
]


def build_manifest(v):
    """Column inventory: role / transform / missing% — a quick audit sheet."""
    outcomes = {"viewCount", "likeCount", "commentCount", "comment_view_rate",
                "like_view_rate", "log_viewCount", "log_comment_view_rate",
                "avg_comment_word_count", "avg_comment_emoji_count",
                "log_comment_depth", "log_comment_affect"}
    controls = {"channelTitle", "upload_year", "video_age_days", "dominant_cluster",
                "dow_hour"} | {
        c for c in v.columns if c.startswith("topic_cluster_")}
    rows = []
    for c in v.columns:
        if c in ("video_id", "publishedAt", "duration_seconds", "is_short",
                 "n_comments", "title", "room_state", "speech_rate_available"):
            role = "id/meta"
        elif c in outcomes:
            role = "outcome"
        elif c in controls:
            role = "control"
        elif c.endswith("_z"):
            role = "predictor (standardised)"
        elif c.endswith("_log1p"):
            role = "predictor (log1p)"
        else:
            role = "predictor"
        rows.append({
            "column": c, "role": role, "dtype": str(v[c].dtype),
            "missing_pct": round(v[c].isna().mean() * 100, 1),
            "n_unique": int(v[c].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def main():
    v = build_modeling_set()
    print(f"Modeling set: {len(v)} long-form post-2022 cohort videos")
    print(v.groupby("channelTitle").size().to_string())

    v = add_transformed_outcomes(v)
    v = add_comment_quality_outcomes(v)
    # Sentiment outcomes DROPPED 2026-06-13 (weak + cover only the old 5 channels).
    # add_sentiment_outcomes() kept below for provenance but no longer called.
    v = add_speech_rate(v)
    v = impute_topic_volatility(v)
    v = add_log1p_predictors(v)
    v = add_topic_cluster_dummies(v)
    v = standardise_for_lasso(v)

    # drop the now-redundant raw duration_seconds; keep duration_minutes
    v = v.drop(columns=[c for c in ["duration_seconds"] if c in v])

    v.to_parquet(OUT_PARQUET, index=False)
    manifest = build_manifest(v)
    manifest.to_csv(OUT_MANIFEST, index=False, encoding="utf-8-sig")

    print(f"\nWrote {OUT_PARQUET.name}: {v.shape[0]} rows x {v.shape[1]} cols")
    topic_dummy_cols = [c for c in v.columns if c.startswith("topic_cluster_")]
    print(f"Topic-cluster dummies: INCLUDED ({len(topic_dummy_cols)} dummies: "
          f"{topic_dummy_cols}). 10-channel cohort spans clusters within channels; "
          "see check_topic_within_channel.py.")
    print(f"topic_volatility imputed rows: {int(v['topic_volatility_imputed'].sum())} "
          f"({v['topic_volatility_imputed'].mean()*100:.1f}%)")
    print("\nKey missingness (post-build):")
    miss = manifest[manifest.missing_pct > 0].sort_values("missing_pct", ascending=False)
    print(miss.to_string(index=False) if len(miss) else "  (none)")
    print(f"\nManifest -> {OUT_MANIFEST.name}")


if __name__ == "__main__":
    main()
