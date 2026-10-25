"""
Instagram sentiment / toxicity exploratory analysis.

Object-oriented EDA over the comment-level sentiment dataframe produced by the
sentiment pipeline (sentiment_instagram.parquet). The analyser produces a set of
summary tables and figures, with a dedicated section drilling into toxicity and
negative sentiment.

Usage:
    python sentiment_analysis.py
    python sentiment_analysis.py --data path/to/sentiment_instagram.parquet --out outputs/sentiment
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, no display needed
import matplotlib.pyplot as plt
import pandas as pd

# Make stdout tolerant of emoji-laden comment text on Windows code pages.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repo-relative defaults so the script works regardless of CWD.
_HERE = Path(__file__).resolve().parent
DEFAULT_DATA = _HERE / "outputs" / "sentiment_instagram.parquet"
DEFAULT_OUT = _HERE / "outputs" / "sentiment_eda"

# A consistent ordering / palette for the three coarse sentiment buckets.
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
TOXICITY_ORDER = ["none", "mild", "severe", "spam_promo"]
TOXICITY_COLORS = {
    "none": "#cfcfcf",
    "mild": "#f4a259",
    "severe": "#d62728",
    "spam_promo": "#9467bd",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@dataclass
class SentimentDataset:
    """Thin wrapper around the comment-level sentiment dataframe.

    Owns loading and the handful of derived columns the analysis relies on
    (parsed timestamps, a negativity flag, a toxic flag), so downstream code can
    assume those exist.
    """

    path: Path
    df: pd.DataFrame = field(default=None, repr=False)

    NEGATIVE_TOXICITIES = ("mild", "severe")

    @classmethod
    def load(cls, path: Path) -> "SentimentDataset":
        df = pd.read_parquet(path)
        ds = cls(path=path, df=df)
        ds._derive()
        return ds

    def _derive(self) -> None:
        df = self.df
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
        # Coarse buckets are the reliable ones; the fine-grained `sentiment`
        # column has a long noisy tail, so analyse on sentiment_cat.
        df["is_negative"] = df["sentiment_cat"].eq("negative")
        df["is_toxic"] = df["toxicity"].isin(self.NEGATIVE_TOXICITIES)
        df["is_severe"] = df["toxicity"].eq("severe")

    def __len__(self) -> int:
        return len(self.df)


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
class TableReporter:
    """Builds tidy summary tables and writes them to CSV + console."""

    def __init__(self, ds: SentimentDataset, out_dir: Path):
        self.df = ds.df
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _emit(self, name: str, table: pd.DataFrame) -> pd.DataFrame:
        path = self.out_dir / f"{name}.csv"
        table.to_csv(path)
        print(f"\n### {name}  ->  {path.name}")
        print(table.to_string())
        return table

    def overview(self) -> pd.DataFrame:
        df = self.df
        rows = {
            "comments": len(df),
            "unique_authors": df["author_id"].nunique(),
            "unique_media": df["media_id"].nunique(),
            "date_min": df["timestamp"].min(),
            "date_max": df["timestamp"].max(),
            "pct_negative": round(100 * df["is_negative"].mean(), 2),
            "pct_toxic": round(100 * df["is_toxic"].mean(), 2),
            "pct_severe": round(100 * df["is_severe"].mean(), 4),
            "pct_sarcastic": round(100 * df["sarcasm"].mean(), 2),
            "mean_sentiment_score": round(df["sentiment_score"].mean(), 3),
        }
        return self._emit("00_overview", pd.Series(rows, name="value").to_frame())

    def categorical_breakdown(self, col: str, top: int = 15) -> pd.DataFrame:
        vc = self.df[col].value_counts(dropna=False).head(top)
        tbl = pd.DataFrame({"count": vc, "pct": (100 * vc / len(self.df)).round(2)})
        return self._emit(f"cat_{col}", tbl)

    def toxicity_by_target(self) -> pd.DataFrame:
        """Where toxicity lands: toxic-rate per comment target, ranked."""
        g = self.df.groupby("target").agg(
            comments=("comment_id", "size"),
            toxic=("is_toxic", "sum"),
            severe=("is_severe", "sum"),
            negative=("is_negative", "sum"),
        )
        g["toxic_rate_%"] = (100 * g["toxic"] / g["comments"]).round(2)
        g["negative_rate_%"] = (100 * g["negative"] / g["comments"]).round(2)
        g = g[g["comments"] >= 50].sort_values("toxic_rate_%", ascending=False)
        return self._emit("tox_by_target", g)

    def negative_emotion_mix(self) -> pd.DataFrame:
        """Emotion composition within negative comments vs. overall."""
        neg = self.df[self.df["is_negative"]]
        overall = self.df["emotion"].value_counts(normalize=True)
        neg_mix = neg["emotion"].value_counts(normalize=True)
        tbl = pd.DataFrame(
            {
                "neg_share_%": (100 * neg_mix).round(2),
                "overall_share_%": (100 * overall).round(2),
            }
        ).fillna(0.0)
        # Lift = over-representation of an emotion within negative comments.
        # Guard against div-by-zero (emotions absent from the overall mix).
        denom = tbl["overall_share_%"].replace(0, float("nan"))
        tbl["lift"] = (tbl["neg_share_%"] / denom).round(2)
        tbl = tbl.sort_values("neg_share_%", ascending=False).head(12)
        return self._emit("neg_emotion_mix", tbl)

    def toxic_authors(self, top: int = 20) -> pd.DataFrame:
        """Authors with the most toxic comments (min activity threshold)."""
        g = self.df.groupby("author_id").agg(
            comments=("comment_id", "size"),
            toxic=("is_toxic", "sum"),
            severe=("is_severe", "sum"),
            negative=("is_negative", "sum"),
        )
        g = g[g["toxic"] > 0].copy()
        g["toxic_rate_%"] = (100 * g["toxic"] / g["comments"]).round(1)
        g = g.sort_values(["toxic", "severe"], ascending=False).head(top)
        return self._emit("toxic_authors", g)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
class SentimentPlotter:
    """Renders the figure set. Each method writes one PNG and returns its path."""

    def __init__(self, ds: SentimentDataset, out_dir: Path, dpi: int = 130):
        self.df = ds.df
        self.out_dir = out_dir
        self.dpi = dpi
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> Path:
        path = self.out_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  figure -> {path.name}")
        return path

    def overview(self) -> Path:
        df = self.df
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Sentiment distribution
        sc = df["sentiment_cat"].value_counts().reindex(SENTIMENT_ORDER)
        axes[0, 0].bar(sc.index, sc.values, color=[SENTIMENT_COLORS[s] for s in sc.index])
        axes[0, 0].set_title("Sentiment distribution")
        axes[0, 0].set_ylabel("comments")

        # Sentiment score histogram
        axes[0, 1].hist(df["sentiment_score"], bins=40, color="#1f77b4", alpha=0.85)
        axes[0, 1].axvline(0, color="k", lw=0.8, ls="--")
        axes[0, 1].set_title("Sentiment score distribution")
        axes[0, 1].set_xlabel("score (-1 .. +1)")

        # Emotion top-10
        em = df["emotion"].value_counts().head(10)[::-1]
        axes[1, 0].barh(em.index, em.values, color="#9467bd")
        axes[1, 0].set_title("Top emotions")

        # Toxicity (log scale — heavily imbalanced)
        tx = df["toxicity"].value_counts().reindex(TOXICITY_ORDER).dropna()
        axes[1, 1].bar(tx.index, tx.values, color=[TOXICITY_COLORS[t] for t in tx.index])
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Toxicity distribution (log scale)")
        axes[1, 1].set_ylabel("comments (log)")

        fig.suptitle("Instagram comment sentiment — overview", fontweight="bold")
        return self._save(fig, "01_overview")

    def timeseries(self) -> Path:
        """Monthly volume + negative/toxic rate over time."""
        df = self.df.dropna(subset=["month"])
        monthly = df.groupby("month").agg(
            comments=("comment_id", "size"),
            neg_rate=("is_negative", "mean"),
            tox_rate=("is_toxic", "mean"),
        )
        # Trim ultra-sparse months so rates aren't dominated by noise.
        monthly = monthly[monthly["comments"] >= 30]

        fig, ax1 = plt.subplots(figsize=(13, 5))
        ax1.bar(monthly.index, monthly["comments"], width=20, color="#c7d3e0",
                label="comments")
        ax1.set_ylabel("comments / month", color="#5a6b7b")
        ax1.set_xlabel("month")

        ax2 = ax1.twinx()
        ax2.plot(monthly.index, 100 * monthly["neg_rate"], color="#d62728",
                 marker="o", ms=3, lw=1.5, label="% negative")
        ax2.plot(monthly.index, 100 * monthly["tox_rate"], color="#7b2cbf",
                 marker="s", ms=3, lw=1.5, label="% toxic")
        ax2.set_ylabel("% of monthly comments")
        ax2.legend(loc="upper left")
        ax1.set_title("Comment volume & negativity / toxicity over time", fontweight="bold")
        return self._save(fig, "02_timeseries")

    def toxicity_focus(self, reporter: TableReporter) -> Path:
        """Dedicated 4-panel deep dive on toxicity + negativity."""
        df = self.df
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Toxic rate by target
        tbt = reporter.toxicity_by_target().head(10).sort_values("toxic_rate_%")
        axes[0, 0].barh(tbt.index, tbt["toxic_rate_%"], color="#f4a259")
        axes[0, 0].set_title("Toxic-comment rate by target (%)")
        axes[0, 0].set_xlabel("% toxic")

        # 2. Negative-emotion lift vs overall
        mix = reporter.negative_emotion_mix().head(8).sort_values("neg_share_%")
        axes[0, 1].barh(mix.index, mix["neg_share_%"], color="#d62728", alpha=0.8,
                        label="within negative")
        axes[0, 1].barh(mix.index, mix["overall_share_%"], color="#999999", alpha=0.5,
                        label="overall")
        axes[0, 1].set_title("Emotion mix: negative comments vs overall")
        axes[0, 1].set_xlabel("% share")
        axes[0, 1].legend()

        # 3. Sentiment score for toxic vs non-toxic
        non_tox = df.loc[~df["is_toxic"], "sentiment_score"]
        tox = df.loc[df["is_toxic"], "sentiment_score"]
        axes[1, 0].hist(non_tox, bins=30, density=True, alpha=0.55, color="#2ca02c",
                        label="non-toxic")
        axes[1, 0].hist(tox, bins=30, density=True, alpha=0.65, color="#d62728",
                        label="toxic")
        axes[1, 0].set_title("Sentiment score: toxic vs non-toxic")
        axes[1, 0].set_xlabel("sentiment score")
        axes[1, 0].legend()

        # 4. Sarcasm interaction with sentiment
        ct = pd.crosstab(df["sarcasm_label"], df["sentiment_cat"],
                         normalize="index").reindex(columns=SENTIMENT_ORDER).fillna(0)
        bottom = pd.Series(0.0, index=ct.index)
        for s in SENTIMENT_ORDER:
            axes[1, 1].bar(ct.index, 100 * ct[s], bottom=100 * bottom,
                           color=SENTIMENT_COLORS[s], label=s)
            bottom += ct[s]
        axes[1, 1].set_title("Sentiment composition by sarcasm")
        axes[1, 1].set_ylabel("% of comments")
        axes[1, 1].legend()

        fig.suptitle("Toxicity & negative-sentiment deep dive", fontweight="bold")
        return self._save(fig, "03_toxicity_focus")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
class SentimentAnalysis:
    """Top-level driver: load -> tables -> figures."""

    def __init__(self, data_path: Path, out_dir: Path):
        self.ds = SentimentDataset.load(data_path)
        self.out_dir = out_dir
        self.tables = TableReporter(self.ds, out_dir)
        self.plots = SentimentPlotter(self.ds, out_dir / "figs")

    def run(self) -> None:
        print(f"Loaded {len(self.ds):,} comments from {self.ds.path}")

        print("\n" + "=" * 70 + "\n  SUMMARY TABLES\n" + "=" * 70)
        self.tables.overview()
        for col in ["sentiment_cat", "emotion", "intensity", "toxicity",
                    "intent", "target"]:
            self.tables.categorical_breakdown(col)

        print("\n" + "=" * 70 + "\n  TOXICITY / NEGATIVE FOCUS\n" + "=" * 70)
        self.tables.toxicity_by_target()
        self.tables.negative_emotion_mix()
        self.tables.toxic_authors()

        print("\n" + "=" * 70 + "\n  FIGURES\n" + "=" * 70)
        self.plots.overview()
        self.plots.timeseries()
        self.plots.toxicity_focus(self.tables)

        print(f"\nDone. Tables + figures written under: {self.out_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA,
                   help="path to sentiment_instagram.parquet")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="output directory for tables + figures")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    SentimentAnalysis(args.data, args.out).run()


if __name__ == "__main__":
    main()
