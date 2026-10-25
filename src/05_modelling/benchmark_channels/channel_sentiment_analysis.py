"""
Per-channel sentiment aggregation.
Reads the already-computed sentiment_youtube.parquet and room_vibe_youtube.parquet,
joins channelTitle from the transcripts parquet, writes enriched parquets, and
produces per-channel summary tables + plots.

Run from <REPO_ROOT>:
    python benchmark_analysis/channel_sentiment_analysis.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
OUT_DIR        = "comments/outputs"
SENT_PATH      = f"{OUT_DIR}/sentiment_youtube.parquet"
ROOM_PATH      = f"{OUT_DIR}/room_vibe_youtube.parquet"
TRANSCRIPTS    = "clean_data/yt_videos_with_local_transcripts.parquet"
FIG_DIR        = f"{OUT_DIR}/figs_channel"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────
sent = pd.read_parquet(SENT_PATH)
room = pd.read_parquet(ROOM_PATH)

# Build video_id -> channelTitle lookup from transcripts
vid_meta = pd.read_parquet(TRANSCRIPTS, columns=["videoId", "channelTitle", "publishedAt"])
vid_meta["videoId"] = vid_meta["videoId"].astype(str)
vid_meta["publishedAt"] = pd.to_datetime(vid_meta["publishedAt"], utc=True, errors="coerce")
id_to_channel = dict(zip(vid_meta["videoId"], vid_meta["channelTitle"]))

# Join channelTitle
sent["channelTitle"] = sent["media_id"].map(id_to_channel)
room["channelTitle"] = room["media_id"].map(id_to_channel)

unmapped_sent = sent["channelTitle"].isna().sum()
unmapped_room = room["channelTitle"].isna().sum()
if unmapped_sent or unmapped_room:
    print(f"WARNING: {unmapped_sent} comments / {unmapped_room} videos could not be mapped to a channel.")

# Write enriched parquets (adds channelTitle column, everything else unchanged)
sent.to_parquet(f"{OUT_DIR}/sentiment_youtube_ch.parquet", index=False)
room.to_parquet(f"{OUT_DIR}/room_vibe_youtube_ch.parquet", index=False)
print(f"Saved enriched parquets -> {OUT_DIR}/sentiment_youtube_ch.parquet")
print(f"                        -> {OUT_DIR}/room_vibe_youtube_ch.parquet")

# ── Per-channel comment-level aggregation ─────────────────────────────────
sent["sentiment_score"] = pd.to_numeric(sent["sentiment_score"], errors="coerce")
sent["timestamp"] = pd.to_datetime(sent["timestamp"], utc=True, errors="coerce")

NEUTRAL_BAND = 0.15
LOG2_3 = np.log2(3)

def norm_entropy(pos, neg, neu):
    p = np.array([pos, neg, neu]); p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) / LOG2_3 if len(p) else 0.0

ch_rows = []
for ch, g in sent.groupby("channelTitle"):
    n = len(g)
    s = g["sentiment_score"].dropna()
    lab = g["sentiment_cat"].fillna("neutral")
    pos = float((lab == "positive").mean())
    neg = float((lab == "negative").mean())
    neu = float(max(0.0, 1 - pos - neg))
    ch_rows.append({
        "channelTitle":     ch,
        "n_comments":       n,
        "n_videos":         g["media_id"].nunique(),
        "mean_score":       float(s.mean()) if len(s) else np.nan,
        "score_std":        float(s.std(ddof=0)) if len(s) > 1 else 0.0,
        "pos_frac":         pos,
        "neg_frac":         neg,
        "neu_frac":         neu,
        "polarization":     2 * min(pos, neg),
        "controversy":      4 * pos * neg,
        "consensus":        1 - norm_entropy(pos, neg, neu),
        "sarcasm_rate":     float(g["sarcasm"].fillna(False).mean()),
        "toxicity_rate":    float((g["toxicity"].fillna("none") != "none").mean()),
        "top_emotion":      g["emotion"].mode().iloc[0] if g["emotion"].notna().any() else None,
        "top_intent":       g["intent"].mode().iloc[0] if g["intent"].notna().any() else None,
    })

ch_summary = pd.DataFrame(ch_rows).sort_values("mean_score", ascending=False)
ch_summary.to_parquet(f"{OUT_DIR}/channel_sentiment_summary.parquet", index=False)
ch_summary.to_csv(f"{OUT_DIR}/channel_sentiment_summary.csv", index=False)
print(f"\nSaved channel summary -> {OUT_DIR}/channel_sentiment_summary.parquet/.csv")

print("\n=== Per-channel sentiment summary ===")
print(ch_summary.to_string(index=False))

# ── Per-channel video-level aggregation ───────────────────────────────────
room["vibe_score"] = pd.to_numeric(room["vibe_score"], errors="coerce")

ch_room = (
    room.groupby("channelTitle")
    .agg(
        n_videos        =("media_id", "count"),
        mean_vibe       =("vibe_score", "mean"),
        mean_controversy=("controversy", "mean"),
        mean_consensus  =("consensus", "mean"),
        mean_polarization=("polarization","mean"),
        mean_sarcasm    =("sarcasm_rate", "mean"),
        mean_toxicity   =("toxicity_rate", "mean"),
        pct_divided     =("room_state", lambda x: (x == "divided").mean()),
        pct_united      =("room_state", lambda x: (x == "united").mean()),
    )
    .reset_index()
    .sort_values("mean_vibe", ascending=False)
)
ch_room.to_parquet(f"{OUT_DIR}/channel_room_summary.parquet", index=False)
ch_room.to_csv(f"{OUT_DIR}/channel_room_summary.csv", index=False)
print(f"\nSaved channel room summary -> {OUT_DIR}/channel_room_summary.parquet/.csv")

print("\n=== Per-channel video-level vibe summary ===")
print(ch_room.to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
POS_C, NEU_C, NEG_C = "#2e9e5b", "#b0b0b0", "#d1495b"
channels = ch_summary["channelTitle"].tolist()
x = np.arange(len(channels))

def save(fig, name):
    p = f"{FIG_DIR}/{name}.png"
    fig.savefig(p, bbox_inches="tight")
    print(f"  saved {p}")

# 1) Stacked sentiment fractions per channel
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, ch_summary["pos_frac"], label="positive", color=POS_C)
ax.bar(x, ch_summary["neu_frac"], bottom=ch_summary["pos_frac"], label="neutral", color=NEU_C)
ax.bar(x, ch_summary["neg_frac"],
       bottom=ch_summary["pos_frac"] + ch_summary["neu_frac"], label="negative", color=NEG_C)
ax.set_xticks(x); ax.set_xticklabels(channels, rotation=20, ha="right")
ax.set_ylabel("fraction of comments"); ax.set_ylim(0, 1)
ax.set_title("Sentiment composition per channel")
ax.legend(loc="upper right")
save(fig, "ch_01_sentiment_stack")
plt.close(fig)

# 2) Mean sentiment score + std per channel
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x, ch_summary["mean_score"],
       color=[POS_C if v > NEUTRAL_BAND else NEG_C if v < -NEUTRAL_BAND else NEU_C
              for v in ch_summary["mean_score"]])
ax.errorbar(x, ch_summary["mean_score"], yerr=ch_summary["score_std"],
            fmt="none", color="black", capsize=4, linewidth=1)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(channels, rotation=20, ha="right")
ax.set_ylabel("mean sentiment score (±1σ)")
ax.set_title("Mean sentiment score per channel")
save(fig, "ch_02_mean_score")
plt.close(fig)

# 3) Controversy vs consensus scatter
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(ch_room["mean_controversy"], ch_room["mean_consensus"],
                s=ch_room["n_videos"] * 2, alpha=0.8,
                c=ch_room["mean_vibe"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
for _, r in ch_room.iterrows():
    ax.annotate(r["channelTitle"], (r["mean_controversy"], r["mean_consensus"]),
                fontsize=8, ha="left", va="bottom")
plt.colorbar(sc, ax=ax, label="mean vibe score")
ax.set_xlabel("mean controversy (0=one-sided, 1=evenly split)")
ax.set_ylabel("mean consensus (1=everyone agrees)")
ax.set_title("Channel audience: controversy vs consensus\n(bubble size = n_videos)")
save(fig, "ch_03_controversy_consensus")
plt.close(fig)

# 4) Sarcasm + toxicity rates per channel
fig, ax = plt.subplots(figsize=(10, 4))
w = 0.35
ax.bar(x - w/2, ch_summary["sarcasm_rate"],  width=w, label="sarcasm rate",  color="#e07b39")
ax.bar(x + w/2, ch_summary["toxicity_rate"], width=w, label="toxicity rate", color="#c44e52")
ax.set_xticks(x); ax.set_xticklabels(channels, rotation=20, ha="right")
ax.set_ylabel("rate"); ax.set_title("Sarcasm & toxicity rates per channel")
ax.legend()
save(fig, "ch_04_sarcasm_toxicity")
plt.close(fig)

# 5) Sentiment over time per channel (monthly mean score)
sent_ts = sent.dropna(subset=["timestamp", "sentiment_score", "channelTitle"]).copy()
sent_ts["month"] = sent_ts["timestamp"].dt.to_period("M")
ts_ch = (
    sent_ts.groupby(["channelTitle", "month"])["sentiment_score"]
    .mean().reset_index()
)
ts_ch["month_dt"] = ts_ch["month"].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(13, 5))
for ch, grp in ts_ch.groupby("channelTitle"):
    ax.plot(grp["month_dt"], grp["sentiment_score"], marker=".", label=ch, linewidth=1.5)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("month"); ax.set_ylabel("mean sentiment score")
ax.set_title("Monthly mean sentiment score per channel")
ax.legend(fontsize=8)
save(fig, "ch_05_score_over_time")
plt.close(fig)

print("\nDone.")
