"""
Per-video comment feature extraction for the 5 cohort YouTube channels.
Reads sentiment_youtube_ch.parquet + room_vibe_youtube_ch.parquet (already computed).
Writes video_comment_features.parquet to this folder.

Run from <REPO_ROOT>:
    python benchmark_analysis/features/comment_features/extract_comment_features.py
"""
import os, sys
import numpy as np
import pandas as pd

FEAT_DIR   = "benchmark_analysis/features/comment_features"
SENT_PATH  = "comments/outputs/sentiment_youtube_ch.parquet"
ROOM_PATH  = "comments/outputs/room_vibe_youtube_ch.parquet"
TRANS_PATH = "clean_data/yt_videos_with_local_transcripts.parquet"
OUT_PATH   = f"{FEAT_DIR}/video_comment_features.parquet"

# ── Load ──────────────────────────────────────────────────────────────────
sent = pd.read_parquet(SENT_PATH)
room = pd.read_parquet(ROOM_PATH)

sent["sentiment_score"] = pd.to_numeric(sent["sentiment_score"], errors="coerce")
sent["timestamp"]       = pd.to_datetime(sent["timestamp"], utc=True, errors="coerce")
sent["sarcasm"]         = sent["sarcasm"].fillna(False)
sent["toxicity"]        = sent["toxicity"].fillna("none")

# Video publish dates for trajectory features
trans = pd.read_parquet(TRANS_PATH, columns=["videoId", "publishedAt"])
trans["videoId"]     = trans["videoId"].astype(str)
trans["publishedAt"] = pd.to_datetime(trans["publishedAt"], utc=True, errors="coerce")
pub_date = dict(zip(trans["videoId"], trans["publishedAt"]))

# ── Helpers ───────────────────────────────────────────────────────────────
LOG2_N = np.log2(max(sent["emotion"].nunique(), 2))

def emotion_entropy(series):
    vc = series.value_counts(normalize=True)
    return float(-(vc * np.log2(vc)).sum()) / LOG2_N if len(vc) > 1 else 0.0

def score_slope(ts_series, score_series):
    """Linear slope of sentiment_score over comment timestamp (per second). NaN if <3 points."""
    df = pd.DataFrame({"t": ts_series, "s": score_series}).dropna()
    if len(df) < 3:
        return np.nan
    t_sec = (df["t"] - df["t"].min()).dt.total_seconds().values
    if t_sec.max() == 0:
        return np.nan
    return float(np.polyfit(t_sec, df["s"].values, 1)[0])

def early_late_delta(ts_series, score_series, frac=0.2):
    """Mean score of first frac% of comments minus last frac% by timestamp."""
    df = pd.DataFrame({"t": ts_series, "s": score_series}).dropna().sort_values("t")
    n = len(df)
    if n < 5:
        return np.nan
    k = max(1, int(n * frac))
    return float(df["s"].iloc[:k].mean() - df["s"].iloc[-k:].mean())

# ── Per-video extraction ──────────────────────────────────────────────────
rows = []
for vid_id, g in sent.groupby("media_id"):
    n = len(g)
    s = g["sentiment_score"]

    # ---- Sentiment signals -----------------------------------------------
    mean_score   = float(s.mean()) if s.notna().any() else np.nan
    score_std    = float(s.std(ddof=0)) if s.notna().sum() > 1 else 0.0
    lab          = g["sentiment_cat"].fillna("neutral")
    pos_frac     = float((lab == "positive").mean())
    neg_frac     = float((lab == "negative").mean())
    neu_frac     = float(max(0.0, 1 - pos_frac - neg_frac))

    slope        = score_slope(g["timestamp"], s)
    early_late   = early_late_delta(g["timestamp"], s)

    # ---- Audience dynamics (recompute here; room_vibe joined below) ------
    # (controversy, consensus etc. come from room_vibe join — no duplication)

    # ---- Intent rates ----------------------------------------------------
    intent       = g["intent"].fillna("other")
    praise_rate      = float((intent == "praise").mean())
    criticism_rate   = float((intent == "criticism").mean())

    # ---- Emotion ---------------------------------------------------------
    emotion      = g["emotion"].dropna()
    top_emotion  = emotion.mode().iloc[0] if len(emotion) else None
    em_entropy   = emotion_entropy(emotion)

    rows.append({
        "media_id":         vid_id,
        # sentiment signals
        "mean_score":       mean_score,
        "score_std":        score_std,
        "pos_frac":         pos_frac,
        "neg_frac":         neg_frac,
        "neu_frac":         neu_frac,
        "score_slope":      slope,        # positive = warming up over time, negative = souring
        "early_late_delta": early_late,   # positive = early comments more positive than late
        # intent
        "praise_rate":      praise_rate,
        "criticism_rate":   criticism_rate,
        # emotion
        "top_emotion":      top_emotion,
        "emotion_entropy":  em_entropy,   # 0 = one dominant emotion, 1 = evenly spread
        "n_comments":       n,
    })

feat = pd.DataFrame(rows)

# ── Join audience dynamics from room_vibe (already per-video) ─────────────
room_cols = [
    "media_id", "channelTitle",
    "vibe_score", "dispersion", "stance_alignment",
    "polarization", "controversy", "consensus",
    "sarcasm_rate", "toxicity_rate", "room_state",
    "llm_vibe", "dominant_stance", "split_axis", "room_summary",
    "llm_consensus", "llm_alignment", "llm_controversy",
]
feat = feat.merge(room[room_cols], on="media_id", how="left")

# ── Save ──────────────────────────────────────────────────────────────────
feat.to_parquet(OUT_PATH, index=False)
print(f"Saved {len(feat):,} videos x {len(feat.columns)} features -> {OUT_PATH}")
print("\nColumns:", feat.columns.tolist())
print("\nSample (5 rows):")
print(feat[["media_id", "channelTitle", "mean_score", "controversy",
            "praise_rate", "criticism_rate", "top_emotion", "emotion_entropy",
            "score_slope", "early_late_delta"]].head().to_string(index=False))
