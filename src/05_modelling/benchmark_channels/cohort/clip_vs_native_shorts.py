"""
clip_vs_native_shorts.py

Classify each SHORT as a *derived clip* (a segment / stitched set of segments cut
from a recent long-form video of the same channel, posted as promo) vs a
*native short* (content made to be short, with no long-form parent).

Motivation
----------
The raw `is_short` flag conflates two structurally different content decisions.
TargetCreator's shorts are overwhelmingly clips of her long videos, so counting them
as independent "short-form" content inflates her shorts ratio and contaminates
any format-based benchmark selection. We need a clip-vs-native label.

Method (fast profile, 2026 only)
--------------------------------
For each short S (2026):
  1. Candidate parents = same-channel LONG videos published in the 30 days
     BEFORE S (strictly before S.publishedAt).
  2. For every chapter c of S, compute max cosine to any chapter of any
     candidate parent. A chapter is "matched" if that max >= MATCH_COS.
  3. matched_share(S) = (# matched chapters) / (# chapters of S).
  4. Label S as 'derived_clip' if matched_share >= SHARE_CUTOFF, else
     'native_short'. Shorts with no candidate parent -> 'native_short'
     (no long-form to be derived from).

Stitched multi-segment clips are handled because each segment is scored
independently at the chapter level; a high share of chapters matching prior
long content flags a clip even when no single long video contains all of it.

Outputs
-------
  - clip_vs_native_shorts.csv             per-short labels + matched_share
  - clip_vs_native_channel_profile.csv    per-channel clip rate
  - clip_vs_native_channel_profile.png    ONE presentation plot
  - prints the matched_share distribution for threshold calibration
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DENSE_VECTORS_PATH = Path(r"d:\creator-analytics\embeddings\dense_vectors_semantic.parquet")
VIDEOS_PATH        = Path(r"d:\creator-analytics\clean_data\yt_videos_with_local_transcripts.parquet")
OUT_SHORTS  = BASE_DIR / "clip_vs_native_shorts.csv"
OUT_PROFILE = BASE_DIR / "clip_vs_native_channel_profile.csv"
OUT_PLOT    = BASE_DIR / "clip_vs_native_channel_profile.png"

# --- tunable thresholds (calibrate from printed distribution) ---
WINDOW_DAYS  = 30      # look-back window for candidate parent long videos
MATCH_COS    = 0.85    # a short's chapter is "matched" if max cosine to a parent chapter >= this
SHARE_CUTOFF = 0.50    # short is a derived clip if >= this share of its chapters are matched
YEAR_SCOPE   = 2026    # if SCOPE=="year": profile channels on this year only (fast)
SCOPE        = "post2022"  # "year" -> YEAR_SCOPE only;  "post2022" -> all videos >= 2022-01-01
POST2022_CUT = pd.Timestamp("2022-01-01", tz="UTC")
MIN_SHORTS_PLOT = 5    # only plot channels with at least this many shorts


def load_chapter_matrix():
    """Return per-chapter df with float32 L2-normalised vectors + metadata join."""
    dv = pd.read_parquet(DENSE_VECTORS_PATH)
    mat = np.vstack(dv["embedding_vector"].values).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms  # normalise so dot product == cosine

    ch = dv[["video_id", "is_short", "chapter_index"]].copy().reset_index(drop=True)
    ch["row"] = np.arange(len(ch))

    meta = pd.read_parquet(VIDEOS_PATH, columns=["videoId", "channelTitle", "publishedAt"])
    meta = meta.rename(columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")
    ch = ch.merge(meta, on="video_id", how="inner")
    return ch, mat


def make_plot(prof):
    """One presentation plot: per-channel clip rate, TargetCreator highlighted."""
    p = prof[prof["n_shorts"] >= MIN_SHORTS_PLOT].copy()
    p = p.sort_values("clip_rate")
    colors = ["#e74c3c" if c == "TargetCreator" else "#3498db" for c in p["channelTitle"]]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(p))))
    bars = ax.barh(p["channelTitle"], p["clip_rate"] * 100, color=colors,
                   edgecolor="white", linewidth=0.8)
    for bar, (_, r) in zip(bars, p.iterrows()):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{r['clip_rate']*100:.0f}%  (n={int(r['n_shorts'])})",
                va="center", fontsize=8, color="#333333")

    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of 2026 shorts that are DERIVED CLIPS of recent long videos (%)")
    ax.set_title(
        "Not all shorts are the same content decision\n"
        "Clip-publishers (shorts cut from long videos) vs. native-shorts channels",
        fontsize=12, pad=12)
    ax.axvline(50, color="#888888", ls="--", lw=1, alpha=0.7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e74c3c", label="TargetCreator (target)"),
                       Patch(color="#3498db", label="Comparable channels")],
              loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", ls="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {OUT_PLOT.name}")


def main():
    print("=" * 64)
    print("Clip-vs-native short classifier")
    print(f"  window={WINDOW_DAYS}d  match_cos={MATCH_COS}  share_cutoff={SHARE_CUTOFF}  year={YEAR_SCOPE}")

    ch, mat = load_chapter_matrix()
    ch["year"] = ch["publishedAt"].dt.year
    if SCOPE == "post2022":
        ch26 = ch[ch["publishedAt"] >= POST2022_CUT].copy()
        print(f"  scope=post2022 (>= {POST2022_CUT.date()})")
    else:
        ch26 = ch[ch["year"] == YEAR_SCOPE].copy()
        print(f"  scope=year {YEAR_SCOPE}")

    # video-level table for the year (one row per video)
    vids = (ch26.groupby("video_id")
                 .agg(is_short=("is_short", "first"),
                      channelTitle=("channelTitle", "first"),
                      publishedAt=("publishedAt", "first"))
                 .reset_index())

    # precompute chapter rows per video once
    rows_by_vid = ch26.groupby("video_id")["row"].apply(lambda s: s.values).to_dict()

    results = []
    for ch_name, vg in vids.groupby("channelTitle"):
        shorts = vg[vg["is_short"]]
        longs  = vg[~vg["is_short"]]
        for _, s in shorts.iterrows():
            lo = s["publishedAt"] - pd.Timedelta(days=WINDOW_DAYS)
            cand = longs[(longs["publishedAt"] < s["publishedAt"]) &
                         (longs["publishedAt"] >= lo)]
            s_rows = rows_by_vid[s["video_id"]]
            n_chap = len(s_rows)
            if len(cand) == 0 or n_chap == 0:
                results.append((s["video_id"], ch_name, n_chap, len(cand),
                                np.nan, 0.0, "native_short"))
                continue
            cand_rows = np.concatenate([rows_by_vid[v] for v in cand["video_id"]])
            S = mat[s_rows]          # (n_chap, 768)
            L = mat[cand_rows]       # (n_cand_chap, 768)
            sims = S @ L.T           # cosines
            max_per_chap = sims.max(axis=1)
            matched = int((max_per_chap >= MATCH_COS).sum())
            share = matched / n_chap
            label = "derived_clip" if share >= SHARE_CUTOFF else "native_short"
            results.append((s["video_id"], ch_name, n_chap, len(cand),
                            float(max_per_chap.mean()), float(share), label))

    res = pd.DataFrame(results, columns=[
        "video_id", "channelTitle", "n_chapters", "n_candidate_parents",
        "mean_max_cos", "matched_share", "label"])
    res.to_csv(OUT_SHORTS, index=False, encoding="utf-8-sig")

    # ---- distribution for calibration ----
    print("\n" + "=" * 64)
    print("matched_share distribution (shorts that HAD a candidate parent):")
    has_parent = res[res["n_candidate_parents"] > 0]
    if len(has_parent):
        print(has_parent["matched_share"].describe().round(3).to_string())
        bins = [0, .1, .25, .5, .75, .9, 1.0001]
        print("\n histogram of matched_share:")
        print(pd.cut(has_parent["matched_share"], bins, include_lowest=True)
              .value_counts().sort_index().to_string())
    print(f"\n shorts with NO candidate parent (auto native): "
          f"{(res['n_candidate_parents'] == 0).sum()}")

    # ---- per-channel profile ----
    prof = (res.groupby("channelTitle")
               .agg(n_shorts=("video_id", "count"),
                    n_clip=("label", lambda x: (x == "derived_clip").sum()),
                    n_with_parent=("n_candidate_parents", lambda x: (x > 0).sum()))
               .reset_index())
    prof["clip_rate"] = prof["n_clip"] / prof["n_shorts"]
    prof = prof.sort_values("clip_rate", ascending=False)
    prof.to_csv(OUT_PROFILE, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 64)
    print("Per-channel clip rate (share of 2026 shorts that are derived clips):")
    print(prof.to_string(index=False))

    cami = prof[prof["channelTitle"] == "TargetCreator"]
    if len(cami):
        print(f"\n>>> TargetCreator clip_rate = {cami.iloc[0]['clip_rate']:.3f} "
              f"({int(cami.iloc[0]['n_clip'])}/{int(cami.iloc[0]['n_shorts'])} shorts are clips)")

    make_plot(prof)
    print(f"\nSaved: {OUT_SHORTS.name}, {OUT_PROFILE.name}, {OUT_PLOT.name}")


if __name__ == "__main__":
    main()
