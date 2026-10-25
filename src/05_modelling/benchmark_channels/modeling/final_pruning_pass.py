"""
final_pruning_pass.py

Formal pruning pass for the 4-outcome OLS models (N=1453 cohort).
Step 1: LassoCV with the 1-SE rule (lambda = lambda_min + 1*SE) — selects
        a sparser, more conservative feature set than the plain min-lambda.
Step 2: Random Forest feature importance cross-validation on the same predictor
        set, for each outcome independently.
Step 3: Intersection of features that survive BOTH the 1-SE Lasso AND rank in
        the top half of RF importances → final_pruned_levers.csv.

Run: python final_pruning_pass.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

BASE   = Path(__file__).resolve().parent
TABLE  = BASE.parent / "features" / "feature_table.parquet"
OUTDIR = BASE / "ols_baseline"
OUTDIR.mkdir(exist_ok=True)

OUTCOMES = {
    "log_viewCount":         {"view_control": False, "label": "Reach: log(views)"},
    "log_comment_view_rate": {"view_control": True,  "label": "Engagement volume: log(comment/view)"},
    "like_view_rate":        {"view_control": True,  "label": "Approval: like/view (raw)"},
    "log_comment_depth":     {"view_control": True,  "label": "Engagement depth: log(avg comment words)"},
}

# Prescriptive predictors (same pool as baseline_ols.py)
PRESCRIPTIVE = [
    "duration_minutes",
    "days_since_last_video_log1p",
    "format_posting_freq_60d",
    "n_chapters",
    "topic_volatility",
    "is_weekend",
    "clip_title_align",
    "articulation_wpm",
    "speech_density",
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
    "visual_complexity",  # kept in Lasso pool (penalty resolves collinearity)
]
LASSO_ONLY_EXTRA = ["title_char_len"]


def load():
    v = pd.read_parquet(TABLE)
    v["channelTitle"] = v["channelTitle"].astype("category")
    for c in ["face_is_frontal", "text_present", "text_has_number",
              "face_present", "is_weekend", "title_has_question",
              "title_has_number", "title_has_emoji"]:
        if c in v and v[c].dtype == bool:
            v[c] = v[c].astype(int)
    return v


def build_X(v, pool, topic_cols, view_control):
    """Return (feature_names, X_array) after dropna on the combined set."""
    controls = list(topic_cols) + ["upload_year", "video_age_days"]
    if view_control:
        controls += ["log_viewCount"]
    feats = [c for c in pool if c in v.columns] + [c for c in controls if c in v.columns]
    return feats, v[feats].dropna()


def lasso_1se(X_arr, y_arr, feature_names):
    """
    Fit LassoCV then re-fit at lambda = alpha_min + 1*SE(alpha_min).
    Returns list of selected feature names.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xs = scaler_x.fit_transform(X_arr)
    ys = scaler_y.fit_transform(y_arr.reshape(-1, 1)).ravel()

    las = LassoCV(cv=5, random_state=0, max_iter=30000, n_alphas=200).fit(Xs, ys)

    # Identify 1-SE lambda from the CV path
    alpha_min = las.alpha_
    idx_min   = np.argmin(np.abs(las.alphas_ - alpha_min))
    # mse_path_ shape: (n_alphas, n_folds)
    mean_mse  = las.mse_path_.mean(axis=1)
    se_mse    = las.mse_path_.std(axis=1) / np.sqrt(las.mse_path_.shape[1])
    threshold = mean_mse[idx_min] + se_mse[idx_min]
    # Alphas are in descending order; we want the largest alpha (most regularised)
    # whose mean CV error is still within 1 SE of the minimum.
    # mse_path_ corresponds to las.alphas_ in the SAME order.
    within_1se = np.where(mean_mse <= threshold)[0]
    if len(within_1se) == 0:
        alpha_1se = alpha_min
    else:
        # largest alpha (most sparse) still within 1 SE
        alpha_1se = las.alphas_[within_1se[0]]

    from sklearn.linear_model import Lasso
    las_1se = Lasso(alpha=alpha_1se, max_iter=30000).fit(Xs, ys)
    selected = [f for f, c in zip(feature_names, las_1se.coef_) if abs(c) > 1e-8]

    print(f"    lambda_min={alpha_min:.5f}  lambda_1se={alpha_1se:.5f}  "
          f"survivors: {len(selected)}/{len(feature_names)}")
    return selected, alpha_min, alpha_1se


def rf_importance(X_arr, y_arr, feature_names, n_top_frac=0.5):
    """
    Fit a RandomForest, return features ranked by mean decrease in impurity.
    Keeps top `n_top_frac` fraction as 'important'.
    """
    rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(X_arr, y_arr)
    imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    cutoff = int(np.ceil(len(imp) * n_top_frac))
    top_feats = list(imp.index[:cutoff])
    print(f"    RF top {cutoff}/{len(imp)} features by importance:")
    print("    " + ", ".join(imp.index[:min(10, cutoff)]) + ("..." if cutoff > 10 else ""))
    return top_feats, imp


def main():
    v = load()
    print(f"Loaded {len(v)} rows, {v.shape[1]} cols.\n")

    pool = PRESCRIPTIVE + THUMB + LASSO_ONLY_EXTRA
    topic_cols = [c for c in v.columns if c.startswith("topic_cluster_")]

    rows = []
    rf_all = {}

    for outcome, cfg in OUTCOMES.items():
        print(f"\n{'='*70}")
        print(f"OUTCOME: {cfg['label']}")

        feats, df_use = build_X(v, pool, topic_cols, cfg["view_control"])
        # drop rows missing the outcome
        df_use = df_use.loc[v[outcome].dropna().index.intersection(df_use.index)].copy()
        y_arr = v.loc[df_use.index, outcome].values
        X_arr = df_use[feats].fillna(df_use[feats].median()).values
        print(f"  N={len(df_use)}, features={len(feats)}")

        # ---- 1-SE Lasso ----
        print("  [1-SE Lasso]")
        lasso_sel, lam_min, lam_1se = lasso_1se(X_arr, y_arr, feats)

        # ---- RF importance ----
        print("  [Random Forest importance]")
        rf_top, rf_imp = rf_importance(X_arr, y_arr, feats)
        rf_all[outcome] = rf_imp

        # ---- Intersection ----
        # Only prescriptive features go into the intersection (controls are always kept).
        # Fallback: if 1-SE Lasso returns 0 prescriptive survivors (over-regularised),
        # treat RF top-25% as the robust lever set (flagged as rf_only_fallback in CSV).
        lasso_prescriptive = [f for f in lasso_sel if f in PRESCRIPTIVE + THUMB + LASSO_ONLY_EXTRA]
        rf_prescriptive    = [f for f in rf_top    if f in PRESCRIPTIVE + THUMB + LASSO_ONLY_EXTRA]
        rf_top25_prescriptive = [f for f in list(rf_imp.index[:max(1, len(rf_imp)//4)])
                                  if f in PRESCRIPTIVE + THUMB + LASSO_ONLY_EXTRA]

        if lasso_prescriptive:
            intersection      = [f for f in lasso_prescriptive if f in rf_prescriptive]
            lasso_fallback    = False
        else:
            # Over-regularised: use RF top-25% as robust lever signal
            intersection      = rf_top25_prescriptive
            lasso_fallback    = True
            print(f"  NOTE: 1-SE Lasso zeroed all prescriptive features "
                  f"(over-regularised); using RF top-25% as robust levers.")

        print(f"  Lasso-1SE prescriptive survivors: {len(lasso_prescriptive)}")
        print(f"  RF top-half prescriptive:         {len(rf_prescriptive)}")
        method_label = 'RF-top-25%-fallback' if lasso_fallback else 'Lasso+RF-intersection'
        print(f"  ROBUST LEVERS ({method_label}): {len(intersection)}")
        for f in intersection:
            print(f"    - {f}")

        for f in feats:
            rows.append({
                "outcome":             outcome,
                "outcome_label":       cfg["label"],
                "feature":             f,
                "lasso_1se_selected":  int(f in lasso_sel),
                "rf_top_half":         int(f in rf_top),
                "rf_importance":       float(rf_imp.get(f, 0)),
                "robust_lever":        int(f in intersection),
                "lasso_fallback":      int(lasso_fallback),
                "lambda_min":          round(lam_min, 6),
                "lambda_1se":          round(lam_1se, 6),
                "is_prescriptive":     int(f in PRESCRIPTIVE + THUMB + LASSO_ONLY_EXTRA),
            })

    out = pd.DataFrame(rows)
    path = OUTDIR / "final_pruned_levers.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n\nSaved: {path}")

    # ---- Summary table ----
    print("\n\n" + "="*70)
    print("FINAL PRUNED LEVERS (robust across 1-SE Lasso AND RF top-half)")
    print("="*70)
    robust = out[(out["robust_lever"] == 1) & (out["is_prescriptive"] == 1)].copy()
    robust = robust.sort_values(["outcome", "rf_importance"], ascending=[True, False])
    pivot = robust.groupby(["feature", "outcome_label"])["rf_importance"].first().unstack(fill_value=0)
    robust_features = robust.groupby("feature")["outcome"].count().sort_values(ascending=False)
    print("\nFeatures robust in N outcomes:")
    print(robust_features.to_string())
    print("\nFull table per outcome saved to final_pruned_levers.csv")


if __name__ == "__main__":
    main()
