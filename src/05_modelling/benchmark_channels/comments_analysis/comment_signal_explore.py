"""
comment_signal_explore.py

Decide the response-variable design for comment-based engagement: ONE integrated
index vs. a FEW separate outcome models. This is a diagnostic, not the final build.

Aggregates the per-comment ML fields to video level for the full cohort (N driven by
feature_table.parquet; 1453 as of 2026-06-12), then asks:
  1. How many distinct constructs are in these signals? (correlation heatmap + PCA
     scree -> if 1 component dominates, an index is justified; if 2-3, use separate
     outcomes.)
  2. Are the candidate outcomes well-behaved (skew, coverage, n_comments support)?

Outputs (in benchmark_analysis/comment_signals/):
  - comment_video_aggregates.parquet   per-video comment summaries
  - comment_signal_corr.png            correlation heatmap of the video-level signals
  - comment_signal_pca.png             PCA scree + PC1/PC2 loadings
  - comment_signal_summary.csv         per-signal coverage / skew / mean

Run: python comment_signal_explore.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent       # comments_analysis/
PROJ = BASE.parent.parent                     #  (project data root)
OUTDIR = BASE / "comment_signals"
OUTDIR.mkdir(exist_ok=True)

COMMENTS = PROJ / "comments" / "Preped_Comments_comments_ml.parquet"
TABLE = BASE.parent / "features" / "feature_table.parquet"   # produced by features/build_feature_table.py

# Per-comment fields worth aggregating into video-level engagement-quality signals.
PER_COMMENT = [
    "text_length", "word_count", "avg_word_length", "emoji_count",
    "emoji_entropy", "emoji_per_word_ratio", "question_count",
    "exclamation_count", "mention_count", "has_links", "has_numbers",
]


def build_video_aggregates():
    ft = pd.read_parquet(TABLE, columns=["video_id", "channelTitle"])
    ids = set(ft["video_id"].astype(str))

    c = pd.read_parquet(COMMENTS, columns=["media_id", "platform"] + PER_COMMENT)
    c = c[(c["platform"] == "youtube") & (c["media_id"].astype(str).isin(ids))].copy()
    c["media_id"] = c["media_id"].astype(str)

    g = c.groupby("media_id")
    agg = pd.DataFrame({
        "n_comments": g.size(),
        "avg_comment_word_count": g["word_count"].mean(),
        "avg_comment_text_length": g["text_length"].mean(),
        "avg_word_length": g["avg_word_length"].mean(),
        "avg_emoji_count": g["emoji_count"].mean(),
        "avg_emoji_entropy": g["emoji_entropy"].mean(),
        "avg_emoji_per_word": g["emoji_per_word_ratio"].mean(),
        "question_share": g["question_count"].apply(lambda s: (s > 0).mean()),
        "exclamation_share": g["exclamation_count"].apply(lambda s: (s > 0).mean()),
        "mention_share": g["mention_count"].apply(lambda s: (s > 0).mean()),
        "link_share": g["has_links"].mean(),
    }).reset_index().rename(columns={"media_id": "video_id"})

    agg = ft.merge(agg, on="video_id", how="left")
    agg.to_parquet(OUTDIR / "comment_video_aggregates.parquet", index=False)
    return agg


# Signals that describe engagement QUALITY (exclude n_comments = volume, already
# captured by comment/view in the main models).
SIGNALS = [
    "avg_comment_word_count", "avg_comment_text_length", "avg_word_length",
    "avg_emoji_count", "avg_emoji_entropy", "avg_emoji_per_word",
    "question_share", "exclamation_share", "mention_share", "link_share",
]


def corr_heatmap(agg):
    sig = [s for s in SIGNALS if s in agg]
    corr = agg[sig].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sig))); ax.set_xticklabels(sig, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sig))); ax.set_yticklabels(sig, fontsize=8)
    for i in range(len(sig)):
        for j in range(len(sig)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.values[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Video-level comment-quality signals: correlation structure\n"
                 "(tight blocks -> one index; orthogonal -> separate outcomes)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTDIR / "comment_signal_corr.png", dpi=130, bbox_inches="tight")
    plt.close()
    return corr


def pca_view(agg):
    sig = [s for s in SIGNALS if s in agg]
    X = agg[sig].dropna()
    Xs = StandardScaler().fit_transform(X.values)
    p = PCA().fit(Xs)
    evr = p.explained_variance_ratio_

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar(range(1, len(evr) + 1), evr * 100, color="#3498db")
    a1.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, "o-", color="#e67e22")
    a1.set_xlabel("Principal component"); a1.set_ylabel("% variance")
    a1.set_title(f"Scree — PC1 {evr[0]*100:.0f}%, PC1+2 {(evr[:2].sum())*100:.0f}%", fontsize=10)
    a1.axhline(100 / len(sig), ls="--", color="grey", lw=0.8, label="1/p (noise floor)")
    a1.legend(fontsize=8)

    load = pd.DataFrame(p.components_[:2].T, index=sig, columns=["PC1", "PC2"])
    y = np.arange(len(sig))
    a2.barh(y - 0.2, load["PC1"], height=0.4, label="PC1", color="#2c7fb8")
    a2.barh(y + 0.2, load["PC2"], height=0.4, label="PC2", color="#7fcdbb")
    a2.set_yticks(y); a2.set_yticklabels(sig, fontsize=8)
    a2.axvline(0, color="black", lw=0.6); a2.legend(fontsize=8)
    a2.set_title("PC1 / PC2 loadings (which signals move together)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTDIR / "comment_signal_pca.png", dpi=130, bbox_inches="tight")
    plt.close()
    return evr, load


def summary_table(agg):
    sig = [s for s in SIGNALS if s in agg] + ["n_comments"]
    rows = []
    for s in sig:
        col = agg[s]
        rows.append({
            "signal": s, "coverage_pct": round(col.notna().mean() * 100, 1),
            "mean": round(float(col.mean()), 3), "median": round(float(col.median()), 3),
            "skew": round(float(col.dropna().skew()), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "comment_signal_summary.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    agg = build_video_aggregates()
    print(f"Aggregated comments for {agg['video_id'].nunique()} videos "
          f"({agg['n_comments'].notna().sum()} have >=1 comment).")
    print(f"Median comments/video: {agg['n_comments'].median():.0f}  "
          f"(min {agg['n_comments'].min():.0f}, max {agg['n_comments'].max():.0f})")

    summ = summary_table(agg)
    print("\nPer-signal summary:")
    print(summ.to_string(index=False))

    corr = corr_heatmap(agg)
    evr, load = pca_view(agg)
    print(f"\nPCA: PC1 explains {evr[0]*100:.1f}% of variance; "
          f"PC1+PC2 = {evr[:2].sum()*100:.1f}%.")
    print("\nStrongest correlations (|r| >= 0.6, off-diagonal):")
    cc = corr.where(~np.eye(len(corr), dtype=bool))
    pairs = cc.abs().stack().sort_values(ascending=False)  # stack() drops the NaN diagonal
    seen = set()
    for (a, b), r in pairs.items():
        if r < 0.6 or (b, a) in seen:
            continue
        seen.add((a, b))
        print(f"  {a:<26} {b:<26} r={cc.loc[a, b]:+.2f}")

    print("\nDECISION GUIDE:")
    print("  PC1 >> 50% and a single tight block -> ONE integrated quality index.")
    print("  2-3 separable blocks / PC1 modest   -> that many SEPARATE outcome models.")
    print(f"\nSaved aggregates + 2 plots + summary -> {OUTDIR.name}/")


if __name__ == "__main__":
    main()
