"""
generate_pdps.py

Partial Dependence Plots for the top prescriptive levers:
  - title_has_question
  - articulation_wpm
  - duration_minutes

Outcomes shown: log_comment_depth, log_viewCount
(depth is the best-fit model and the one these levers most robustly drive;
views as a second outcome for articulation_wpm and duration_minutes.)

Method: sklearn PartialDependenceDisplay on a Random Forest fitted
to the same predictor pool used in baseline_ols.py (with channel
dummies as one-hot controls so the RF partial dependence is
conditional on channel effects).

Outputs saved to: modeling/ols_baseline/plots/pdp_<feature>_<outcome>.png

Run: python generate_pdps.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler

BASE   = Path(__file__).resolve().parent
TABLE  = BASE.parent / "features" / "feature_table.parquet"
OUTDIR = BASE / "ols_baseline" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)

LEVER_FEATURES = ["title_has_question", "articulation_wpm", "duration_minutes"]

OUTCOMES = {
    "log_comment_depth":    "Engagement Depth: log(avg comment words)",
    "log_viewCount":        "Reach: log(views)",
}

# Same pool as baseline_ols.py
PRESCRIPTIVE = [
    "duration_minutes", "days_since_last_video_log1p", "format_posting_freq_60d",
    "n_chapters", "topic_volatility", "is_weekend", "clip_title_align",
    "articulation_wpm", "speech_density",
    "title_word_count", "title_has_question", "title_has_number",
    "title_exclam_count", "title_caps_ratio", "title_has_emoji",
]
THUMB = [
    "brightness_mean", "saturation_mean", "text_present", "text_area_ratio_log1p",
    "text_word_count_log1p", "text_has_number", "face_present", "face_count",
    "face_area_ratio_log1p", "face_position_x", "face_position_y",
    "face_center_distance", "face_yaw_deg", "face_pitch_deg", "face_is_frontal",
    "mouth_openness", "eye_openness_mean", "contrast", "warm_cool_ratio_log1p",
    "dominant_hue_entropy", "edge_density",
    "rule_of_thirds_score", "left_right_asymmetry", "top_bottom_asymmetry",
]

NICE_NAMES = {
    "title_has_question": "Title Has Question Mark",
    "articulation_wpm":   "Articulation Rate (wpm)",
    "duration_minutes":   "Video Duration (minutes)",
    "log_comment_depth":  "Engagement Depth\n[log(avg comment words)]",
    "log_viewCount":      "Reach [log(views)]",
}


def load():
    v = pd.read_parquet(TABLE)
    for c in ["face_is_frontal", "text_present", "text_has_number", "face_present",
              "is_weekend", "title_has_question", "title_has_number", "title_has_emoji"]:
        if c in v and v[c].dtype == bool:
            v[c] = v[c].astype(int)
    v["channelTitle"] = v["channelTitle"].astype("category")
    return v


def build_matrix(v, outcome, view_control):
    pool = [c for c in PRESCRIPTIVE + THUMB if c in v.columns]
    topic = [c for c in v.columns if c.startswith("topic_cluster_")]
    ch_dummies = pd.get_dummies(v["channelTitle"], prefix="ch", drop_first=True).astype(int)
    controls = topic + ["upload_year", "video_age_days"]
    if view_control:
        controls += ["log_viewCount"]
    controls = [c for c in controls if c in v.columns]
    feats = pool + controls
    Xdf = pd.concat([v[feats], ch_dummies], axis=1).copy()
    all_cols = list(Xdf.columns)
    # Drop NaN rows jointly on outcome + predictors
    mask = v[outcome].notna() & Xdf.notna().all(axis=1)
    Xdf = Xdf.loc[mask]
    y   = v.loc[mask, outcome].values
    return Xdf, y, pool, all_cols


def make_pdp(v, lever, outcome, view_control, label):
    print(f"  Fitting GBM for {outcome} ...")
    Xdf, y, pool, all_cols = build_matrix(v, outcome, view_control)
    feature_names = list(Xdf.columns)

    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=0
    )
    gbm.fit(Xdf.values, y)

    if lever not in feature_names:
        print(f"    WARNING: '{lever}' not in features for {outcome}; skipping.")
        return None

    feat_idx = feature_names.index(lever)
    fig, ax = plt.subplots(figsize=(6, 4))

    disp = PartialDependenceDisplay.from_estimator(
        gbm, Xdf.values, features=[feat_idx],
        feature_names=feature_names,
        kind="average",
        ax=ax,
        line_kw={"color": "#1f77b4", "linewidth": 2.5},
    )
    ax.set_xlabel(NICE_NAMES.get(lever, lever), fontsize=12)
    ax.set_ylabel(f"Partial effect on\n{NICE_NAMES.get(outcome, outcome)}", fontsize=11)
    ax.set_title(
        f"Partial Dependence: {NICE_NAMES.get(lever, lever)}\non {NICE_NAMES.get(outcome, outcome)}",
        fontsize=12, fontweight="bold"
    )
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"pdp_{lever}_{outcome}.png"
    fpath = OUTDIR / fname
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def main():
    v = load()
    print(f"Loaded {len(v)} rows.\n")

    view_control_map = {
        "log_comment_depth": True,
        "log_viewCount":     False,
    }

    saved = []
    for outcome, label in OUTCOMES.items():
        vc = view_control_map[outcome]
        print(f"\n{'='*60}")
        print(f"OUTCOME: {label}")
        for lever in LEVER_FEATURES:
            path = make_pdp(v, lever, outcome, vc, label)
            if path:
                saved.append(path)

    print(f"\n\nAll PDP plots saved to: {OUTDIR}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
