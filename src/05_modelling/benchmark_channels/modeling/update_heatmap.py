"""
update_heatmap.py

Regenerates std_coef_heatmap_updated.png with:
  - Blue (negative) -> White -> Green (positive) diverging colormap
  - Narrow figure + tilted feature labels (less horizontal space)
  - Compact colorbar (sign endpoints only, no axis label)
  - Pruned feature rows (drop non-significant, non-intuitive features)

Standalone (no statsmodels import) so it runs under the creator-analytics-kernel env.
Run: python update_heatmap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

BASE      = Path(__file__).resolve().parent
OLSDIR    = BASE / "ols_baseline"
PLOTDIR   = OLSDIR / "plots"
TABLE     = BASE.parent / "features" / "feature_table.parquet"
COEF_FILE = OLSDIR / "coef_summary.csv"

PRESCRIPTIVE = ['duration_minutes', 'days_since_last_video_log1p', 'format_posting_freq_60d',
                'n_chapters', 'topic_volatility', 'is_weekend', 'clip_title_align',
                'articulation_wpm', 'speech_density', 'title_word_count', 'title_has_question',
                'title_has_number', 'title_exclam_count', 'title_caps_ratio', 'title_has_emoji']
THUMB = ['brightness_mean', 'saturation_mean', 'text_present', 'text_area_ratio_log1p',
         'text_word_count_log1p', 'text_has_number', 'face_present', 'face_count',
         'face_area_ratio_log1p', 'face_position_x', 'face_position_y', 'face_center_distance',
         'face_yaw_deg', 'face_pitch_deg', 'face_is_frontal', 'mouth_openness',
         'eye_openness_mean', 'contrast', 'warm_cool_ratio_log1p', 'dominant_hue_entropy',
         'edge_density', 'rule_of_thirds_score', 'left_right_asymmetry', 'top_bottom_asymmetry']
PRESCRIPTIVE_AND_THUMB = PRESCRIPTIVE + THUMB

OUTCOMES = ['log_viewCount', 'log_comment_view_rate', 'like_view_rate', 'log_comment_depth']
LABELS = {'log_viewCount': 'Reach: log(views)',
          'log_comment_view_rate': 'Engagement volume: log(comment/view)',
          'like_view_rate': 'Approval: like/view (raw)',
          'log_comment_depth': 'Engagement depth: log(avg comment words)'}

# Short column labels for the (now narrower) outcome columns.
ABBREV = {'log_viewCount': 'Reach',
          'log_comment_view_rate': 'Eng.Vol',
          'like_view_rate': 'Appr.',
          'log_comment_depth': 'Eng.Dep'}

# Drop rows neither significant in any outcome nor intuitively informative.
# face_pitch_deg / face_yaw_deg KEPT (intuitively meaningful framing cues).
DROP = {
    "rule_of_thirds_score", "left_right_asymmetry", "top_bottom_asymmetry",
    "saturation_mean", "face_center_distance",
    "dominant_hue_entropy", "warm_cool_ratio_log1p", "contrast",
    "mouth_openness", "eye_openness_mean",
    "n_chapters", "title_has_number",
}

# Shorter display names for the few long features, so horizontal y-labels
# stay compact and the figure can keep a narrow width.
SHORT = {
    "days_since_last_video_log1p": "days_since_last_video",
    "format_posting_freq_60d":     "posting_freq_60d",
    "text_area_ratio_log1p":       "text_area_ratio",
    "text_word_count_log1p":       "text_word_count",
    "face_area_ratio_log1p":       "face_area_ratio",
}


def main():
    v = pd.read_parquet(TABLE)
    coefs = pd.read_csv(COEF_FILE)

    terms = [t for t in PRESCRIPTIVE_AND_THUMB if t in v.columns and t not in DROP]

    # Step size per predictor: 1 for binary (on/off), +1 SD for continuous.
    def is_binary(col):
        u = pd.unique(v[col].dropna())
        return set(np.unique(u)).issubset({0, 1}) or len(u) <= 2

    LOGGED = {o: o.startswith("log_") for o in OUTCOMES}

    # Cell value = % effect over one step for logged outcomes,
    #              100*(exp(beta*step) - 1); raw change beta*step otherwise.
    M = pd.DataFrame(index=terms, columns=OUTCOMES, dtype=float)
    P = pd.DataFrame(index=terms, columns=OUTCOMES, dtype=float)

    for outcome in OUTCOMES:
        sub = coefs[coefs.outcome == outcome].set_index("term")
        for t in terms:
            if t in sub.index:
                b = sub.loc[t, "coef"]
                step = 1.0 if is_binary(t) else v[t].std()
                M.loc[t, outcome] = (100.0 * (np.exp(b * step) - 1.0)
                                     if LOGGED[outcome] else b * step)
                P.loc[t, outcome] = sub.loc[t, "p"]
    M = M.astype(float)

    # Keep only features that are significant (p<.05) in at least one outcome.
    keep = [t for t in terms if (P.loc[t] < 0.05).any()]
    terms = keep
    M = M.loc[terms]
    P = P.loc[terms]
    print(f"Significant-in->=1-outcome features kept: {len(terms)}")

    # Vertical (tall) layout: features are the rows, outcomes the 4 columns.
    fig, ax = plt.subplots(figsize=(3.2, 0.34 * len(terms) + 1.5))
    vmax = np.nanpercentile(np.abs(M.values), 98)
    cmap = LinearSegmentedColormap.from_list("BlueGreen", ["#2166ac", "#ffffff", "#1a9850"])
    im = ax.imshow(M.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    # Outcome abbreviations along the x-axis.
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([ABBREV[o] for o in OUTCOMES],
                       rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    # Horizontal feature labels on the y-axis; long names shortened via SHORT.
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([SHORT.get(t, t) for t in terms], fontsize=6.5)

    # Table-style cells: % effect (logged outcomes) or raw rate change (approval),
    # with an asterisk where the effect is significant (p<.05, robust SEs).
    for i, t in enumerate(terms):
        for j, o in enumerate(OUTCOMES):
            val = M.loc[t, o]
            p   = P.loc[t, o]
            if pd.isna(val):
                continue
            star = "*" if (pd.notna(p) and p < 0.05) else ""
            # Dark text on pale cells, white text on saturated cells.
            txt_color = "white" if abs(val) > 0.6 * vmax else "black"
            label = f"{val:+.0f}%{star}" if LOGGED[o] else f"{val:+.3f}{star}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=6, color=txt_color,
                    fontweight=("bold" if star else "normal"))

    # Thin white gridlines to give it a table appearance.
    ax.set_xticks(np.arange(-0.5, len(OUTCOMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(terms), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels(["−", "0", "+"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    out = PLOTDIR / "std_coef_heatmap_updated.svg"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved: {out}  ({len(terms)} features)")


if __name__ == "__main__":
    main()
