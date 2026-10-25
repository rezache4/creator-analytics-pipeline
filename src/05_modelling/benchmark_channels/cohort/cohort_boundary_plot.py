"""
cohort_boundary_plot.py

Poster figure: the cohort was selected to resemble TargetCreator on two
format-wise axes (NOT topic): how much of the channel is long-form, and
how few of its shorts are *native* (original) shorts rather than clips of
long-form content.

x = long_share          (share of content that is long-form)
y = native_short_share  (share of content that is original/native shorts)

The 10 selected channels separate cleanly from the rest on the y-axis
(all <= 0.163; every excluded channel >= 0.205). We draw a boundary band
around the selected region and highlight the cohort + TargetCreator.

Run: python cohort_boundary_plot.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE    = Path(__file__).resolve().parent
SEL     = BASE / "cohort_selection.csv"
PROFILE = BASE / "clip_vs_native_channel_profile.csv"
VIDEOS  = Path(r"D:\creator-analytics\clean_data\yt_videos_with_local_transcripts.parquet")
OUT     = BASE / "cohort_boundary.svg"

# Per-channel label offsets (in points) to avoid overlaps; default is (5, 4).
# Davide Zambelli and ComparatorChannel4 sit almost on top of each other, so we
# push ComparatorChannel4's label below its point.

# Threshold that cleanly separates the cohort (gap is 0.163 -> 0.205).
NATIVE_THRESH = 0.18

def TargetCreator_coords():
    """TargetCreator is the reference (not in the candidate CSV); derive its two
    axes from the raw video counts + the clip/native profile."""
    vid = pd.read_parquet(VIDEOS, columns=["channelTitle", "publishedAt", "is_short"])
    vid = vid[(vid.channelTitle == "TargetCreator") & (vid.publishedAt > "2022-01-01")]
    n_long  = int((~vid.is_short).sum())
    n_short = int(vid.is_short.sum())
    prof = pd.read_csv(PROFILE).set_index("channelTitle")
    n_clip = int(prof.loc["TargetCreator", "n_clip"])      # shorts that are clips of long-form
    n_native_short = n_short - n_clip                   # original/native shorts
    # Manual correction: the clip detector missed one derivative short, so the
    # reference channel actually has no genuinely-native shorts. Drop one to
    # reflect this (floored at 0).
    n_native_short = max(0, n_native_short - 1)
    total = n_long + n_short
    return n_long / total, n_native_short / total


def main():
    d = pd.read_csv(SEL)
    d["is_cami"] = False

    # Append TargetCreator as the reference point.
    cx, cy = TargetCreator_coords()
    d = pd.concat([d, pd.DataFrame([{
        "channelTitle": "TargetCreator", "long_share": cx,
        "native_short_share": cy, "include": False, "is_cami": True,
    }])], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    # --- boundary band: the selected region (y <= threshold) ---
    ax.axhspan(-0.05, NATIVE_THRESH, color="#1a9850", alpha=0.08, zorder=0)
    ax.axhline(NATIVE_THRESH, color="#1a9850", lw=1.4, ls="--", zorder=1)
    ax.text(0.015, NATIVE_THRESH + 0.012,
            f"selection boundary  (native-short share = {NATIVE_THRESH:.2f})",
            color="#157040", fontsize=8, style="italic")

    inc  = d[d.include & ~d.is_cami]
    exc  = d[~d.include & ~d.is_cami]
    cami = d[d.is_cami]

    # excluded channels
    ax.scatter(exc.long_share, exc.native_short_share, s=42,
               c="#b0b0b0", edgecolor="white", linewidth=0.6,
               label="Other channels (excluded)", zorder=3)
    # selected cohort
    ax.scatter(inc.long_share, inc.native_short_share, s=70,
               c="#1a9850", edgecolor="black", linewidth=0.7,
               label="Selected cohort (10)", zorder=4)
    # TargetCreator (reference)
    ax.scatter(cami.long_share, cami.native_short_share, s=170, marker="*",
               c="#d62728", edgecolor="black", linewidth=0.8,
               label="TargetCreator (reference)", zorder=5)

    # Deterministic manual label placement.
    # value = (dx, dy in points, ha, va, draw_leader)
    PLACE = {
        "ComparatorChannel11":     (-6,  0, "right", "center", False),
        "Davide Zambelli":  (-6, -6, "right", "center", False),
        "ComparatorChannel4":  (-3,-11, "right", "top",    True),
        "ComparatorChannel6":        (-6,  0, "right", "center", False),
        "ComparatorChannel2":            (-6,  0, "right", "center", False),
        "ComparatorChannel9":         (0, -11, "center","top",    True),
        "ComparatorChannel1":   (6,  20, "left",  "bottom", True),
        "ComparatorChannel10":(6,  12, "left",  "bottom", True),
        "AlicelikeAudrey":  (-6,-11, "right", "top",    True),
        "ComparatorChannel3":     (0,  14, "center","bottom", True),
        "TargetCreator":        (-8,  0, "right", "center", False),
    }
    for _, r in pd.concat([inc, cami]).iterrows():
        dx, dy, ha, va, leader = PLACE[r.channelTitle]
        ax.annotate(
            r.channelTitle, (r.long_share, r.native_short_share),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=6.5, ha=ha, va=va,
            color=("#d62728" if r.is_cami else "black"),
            fontweight=("bold" if r.is_cami else "normal"),
            arrowprops=(dict(arrowstyle="-", color="0.5", lw=0.5,
                             shrinkA=0, shrinkB=3) if leader else None),
        )

    ax.set_xlabel("Long-form share of content", fontsize=11)
    ax.set_ylabel("Native (original) shorts share of content", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    # No default grid; just one reference gridline at 0.5 (boundary is at 0.18).
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.axhline(0.5, color="0.8", lw=0.8, zorder=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
