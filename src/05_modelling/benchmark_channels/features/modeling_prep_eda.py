"""
modeling_prep_eda.py

Targeted modeling-prep EDA on the long-form cohort modeling set (COHORT from cohort_config).

Purpose (NOT a data-tour): validate the three unverified modeling assumptions in
FEATURE_LIST_FINAL.md before building the regression table:
  1. Skew -> which predictors/outcomes actually need a transform (decide log/none).
  2. Degenerate / near-constant features -> flag features that add no signal.
  3. Zero-inflation / missingness -> confirm the face_pose_available caveat and any
     NaN-heavy features.

Scope: long-form, post-2022, cohort + TargetCreator. Same unit as cohort_sample_size.py.

Outputs (in eda_modeling_prep/):
  - feature_summary.csv     per-feature: n, missing%, n_unique, top-value share,
                            mean/std/min/max, skew, zero-share, transform recommendation
  - distributions.png       small-multiples histograms of continuous features
  - outcomes.png            the 3 outcome variables, raw vs log
  - README printed to stdout with the decisions
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cohort_config import COHORT, POST2022

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "eda_modeling_prep"
OUTDIR.mkdir(exist_ok=True)

PROJ   = BASE.parent.parent  #  (project data root)
DENSE  = PROJ / "embeddings" / "dense_vectors_semantic.parquet"
VIDEOS = PROJ / "clean_data" / "yt_videos_with_local_transcripts.parquet"
VPROF  = PROJ / "topic_clustering" / "video_profiles.parquet"
CHAP   = PROJ / "topic_clustering" / "chapter_clusters.parquet"
THUMB  = PROJ / "clean_data" / "thumbnail_features.parquet"

# Features from FEATURE_LIST_FINAL.md that are AVAILABLE without new model calls.
# (hook_cluster is pending; format_posting_freq_60d / days_since_last_video derived below.)
THUMB_KEEP = [
    "brightness_mean", "saturation_mean", "text_present", "text_area_ratio",
    "text_word_count", "text_has_number", "clip_title_align", "face_present",
    "face_count", "face_area_ratio", "face_position_x", "face_position_y",
    "face_center_distance", "face_yaw_deg", "face_pitch_deg", "face_is_frontal",
    "mouth_openness", "eye_openness_mean",
    # new CV features (now present in the parquet):
    "contrast", "warm_cool_ratio", "dominant_hue_entropy", "edge_density",
    "visual_complexity", "rule_of_thirds_score", "left_right_asymmetry",
    "top_bottom_asymmetry",
]
DROP_THUMB = ["face_roll_deg", "gaze_off_camera"]  # documented unreliable


def build_modeling_set():
    dv = pd.read_parquet(DENSE, columns=["video_id", "is_short"])
    vid = dv.groupby("video_id")["is_short"].first().reset_index()
    meta = pd.read_parquet(VIDEOS, columns=[
        "videoId", "channelTitle", "publishedAt",
        "viewCount", "likeCount", "commentCount", "duration_seconds"]).rename(columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")

    v = vid.merge(meta, on="video_id", how="inner")
    v = v[(v.channelTitle.isin(COHORT)) & (v.publishedAt >= POST2022) & (~v.is_short)].copy()

    # --- structural features ---
    vp = pd.read_parquet(VPROF, columns=["video_id", "n_chapters"])
    v = v.merge(vp, on="video_id", how="left")
    v = v.merge(topic_volatility(), on="video_id", how="left")

    # --- derived cadence features (publish dates, cohort-wide) ---
    v = v.sort_values(["channelTitle", "publishedAt"])
    v["days_since_last_video"] = v.groupby("channelTitle")["publishedAt"].diff().dt.total_seconds() / 86400
    v["format_posting_freq_60d"] = posting_freq_60d(v)

    # --- thumbnail features ---
    tf = pd.read_parquet(THUMB)
    tf = tf[["video_id"] + [c for c in THUMB_KEEP if c in tf.columns]]
    v = v.merge(tf, on="video_id", how="left")

    # --- derived basics + outcomes ---
    v["duration_minutes"] = pd.to_numeric(v["duration_seconds"], errors="coerce") / 60.0
    v["upload_year"] = v["publishedAt"].dt.year
    v["is_weekend"] = v["publishedAt"].dt.dayofweek.isin([5, 6]).astype(int)
    v["hour_band"] = pd.cut(v["publishedAt"].dt.hour, [-1, 5, 11, 17, 23],
                            labels=["night", "morning", "afternoon", "evening"])
    for c in ["viewCount", "likeCount", "commentCount"]:
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["comment_view_rate"] = v["commentCount"] / v["viewCount"].replace(0, np.nan)
    v["like_view_rate"] = v["likeCount"] / v["viewCount"].replace(0, np.nan)
    # face_pose_available flag (FaceMesh pose-failure caveat)
    if "face_yaw_deg" in v and "face_is_frontal" in v:
        v["face_pose_available"] = ~((v["face_present"] == 1) &
                                     (v["face_yaw_deg"] == 0) &
                                     (v["face_is_frontal"] == False))
    return v


def topic_volatility():
    """Mean cosine distance between consecutive chapter vectors, per video.

    Uses the RAW 768-dim embeddings (not UMAP) — UMAP distorts distances, so
    cosine distance there is not the intended pacing construct.
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


def summarize(v, feature_cols):
    rows = []
    n = len(v)
    for c in feature_cols:
        s = v[c]
        nun = s.nunique(dropna=True)
        vc = s.value_counts(dropna=True)
        top_share = (vc.iloc[0] / s.notna().sum()) if len(vc) else np.nan
        miss = s.isna().mean()
        is_num = pd.api.types.is_numeric_dtype(s)
        sk = float(s.dropna().skew()) if is_num and nun > 2 else np.nan
        zero_share = float((s == 0).mean()) if is_num else np.nan
        rec = transform_rec(c, is_num, nun, sk, top_share, miss)
        rows.append({
            "feature": c, "missing_pct": round(miss * 100, 1), "n_unique": nun,
            "top_value_share": round(top_share, 3) if pd.notna(top_share) else np.nan,
            "skew": round(sk, 2) if pd.notna(sk) else np.nan,
            "zero_share": round(zero_share, 3) if pd.notna(zero_share) else np.nan,
            "mean": round(float(s.dropna().mean()), 3) if is_num else np.nan,
            "std": round(float(s.dropna().std()), 3) if is_num else np.nan,
            "recommendation": rec,
        })
    return pd.DataFrame(rows)


def transform_rec(c, is_num, nun, sk, top_share, miss):
    if miss > 0.5:
        return "DROP? >50% missing"
    if nun <= 1:
        return "DROP constant"
    if pd.notna(top_share) and top_share > 0.97 and nun <= 3:
        return "DROP? near-constant (>97% one value)"
    if not is_num or nun <= 2:
        return "keep (categorical/binary)"
    if pd.notna(sk) and abs(sk) >= 2:
        return "log/transform (|skew|>=2)"
    if pd.notna(sk) and abs(sk) >= 1:
        return "consider transform (|skew|>=1)"
    return "keep as-is"


def plot_distributions(v, cont_cols):
    cols = [c for c in cont_cols if c in v and v[c].notna().sum() > 5]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 2.4))
    for ax, c in zip(axes.ravel(), cols):
        ax.hist(v[c].dropna(), bins=30, color="#3498db", edgecolor="white", linewidth=0.3)
        ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"Modeling-set feature distributions (long-form, post-2022, N={len(v)})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTDIR / "distributions.png", dpi=130, bbox_inches="tight")
    plt.close()


def plot_outcomes(v):
    outs = [("viewCount", "Reach"), ("comment_view_rate", "Comment/view"),
            ("like_view_rate", "Like/view")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for j, (c, lab) in enumerate(outs):
        s = v[c].replace([np.inf, -np.inf], np.nan).dropna()
        axes[0, j].hist(s, bins=40, color="#e67e22", edgecolor="white", linewidth=0.3)
        axes[0, j].set_title(f"{lab} (raw)", fontsize=9)
        ls = np.log1p(s[s >= 0])
        axes[1, j].hist(ls, bins=40, color="#27ae60", edgecolor="white", linewidth=0.3)
        axes[1, j].set_title(f"{lab} (log1p)", fontsize=9)
    fig.suptitle("Outcome variables: raw vs log", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTDIR / "outcomes.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    v = build_modeling_set()
    print(f"Modeling set: {len(v)} long-form post-2022 cohort videos")
    print(v.groupby("channelTitle").size().to_string())

    prescriptive = [
        "duration_minutes", "is_weekend", "hour_band", "days_since_last_video",
        "format_posting_freq_60d", "n_chapters", "topic_volatility", "clip_title_align",
    ]
    feature_cols = prescriptive + [c for c in THUMB_KEEP if c != "clip_title_align" and c in v]
    outcome_cols = ["viewCount", "likeCount", "commentCount",
                    "comment_view_rate", "like_view_rate"]

    summ = summarize(v, [c for c in feature_cols + outcome_cols if c in v])
    summ.to_csv(OUTDIR / "feature_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print("\n" + "=" * 90)
    print("FEATURE SUMMARY (transform / drop recommendations)")
    print("=" * 90)
    print(summ.to_string(index=False))

    flags = summ[summ["recommendation"].str.contains("DROP|transform", case=False)]
    print("\n" + "-" * 90)
    print("ACTION ITEMS:")
    for _, r in flags.iterrows():
        print(f"  {r['feature']:<26} -> {r['recommendation']}")

    cont = [c for c in feature_cols if c in v and pd.api.types.is_numeric_dtype(v[c])
            and v[c].nunique() > 5]
    plot_distributions(v, cont)
    plot_outcomes(v)
    print(f"\nSaved: feature_summary.csv, distributions.png, outcomes.png  (in {OUTDIR.name}/)")


if __name__ == "__main__":
    main()
