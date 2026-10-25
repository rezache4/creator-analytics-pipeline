"""
baseline_ols.py

Baseline regression on the long-form cohort (feature_table.parquet).
Cohort expanded 4->10 channels 2026-06-12 (semantic gate dropped; N 740->1453).

  Outcomes (4) -- sentiment models dropped 2026-06-13 (see OUTCOMES note):
    1. log_viewCount               (reach)
    2. log_comment_view_rate       (engagement volume / community activation)
    3. like_view_rate (raw)        (passive approval)
    4. log_comment_depth           (engagement depth = avg comment length)

  Predictors (prescriptive + thumbnail), log1p versions where SS D2 specifies.
  Controls: C(channelTitle) fixed effects, upload_year, video_age_days, and
            log_viewCount as a control in the rate models.
            Topic-cluster dummies are INCLUDED (decision updated 2026-06-12):
            with 10 channels the pooled top-cluster share dropped from 95% to 67%;
            several channels span 2-3 clusters within-channel, so clusters are no
            longer nested in the channel FE. See check_topic_within_channel.py.
  Estimation: statsmodels OLS with HC1 robust standard errors.

  Diagnostics:
    - VIF table (n_chapters x duration_minutes collinearity check, SS D).
    - Lasso (5-fold CV-lambda) on standardised predictors for scale-fair pruning;
      reports the non-zero (surviving) prescriptive levers per outcome.

Outputs (in benchmark_analysis/ols_baseline/):
  - ols_<outcome>.txt        full statsmodels summary + robust coef table
  - vif.csv                  variance inflation factors
  - lasso_selection.csv      surviving predictors + standardised coefs per outcome
  - coef_summary.csv         tidy OLS coef/se/p for all outcomes, one sheet

Run: python baseline_ols.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent                       # modeling/
TABLE = BASE.parent / "features" / "feature_table.parquet"   # produced by features/build_feature_table.py
OUTDIR = BASE / "ols_baseline"
OUTDIR.mkdir(exist_ok=True)

# Prescriptive predictors on their MODELLED scale (log1p where SS D2 says so).
PRESCRIPTIVE = [
    "duration_minutes",
    "days_since_last_video_log1p",
    "format_posting_freq_60d",
    "n_chapters",
    "topic_volatility",
    "is_weekend",
    "clip_title_align",
    # speech pacing (VTT, gaps>3s = silence) -- delivery levers
    "articulation_wpm",
    "speech_density",
    # title features (pre-publish creator choices; no LLM/embeddings).
    # title_char_len dropped from OLS: VIF ~7 with title_word_count (same length
    # construct); word_count kept as the interpretable lever. char_len stays in Lasso.
    "title_word_count", "title_has_question", "title_has_number",
    "title_exclam_count", "title_caps_ratio", "title_has_emoji",
]
# Thumbnail predictors (log1p the 4 flagged in SS D2; rest raw).
# NOTE: visual_complexity is dropped from the OLS predictor set -- it correlates
# 0.993 with `contrast` (VIF ~110-122), the same "busyness" construct measured
# twice. Keeping both destabilises their robust SEs. We retain `contrast` (the
# more interpretable design lever) for OLS; the Lasso path still sees both (the
# penalty resolves the collinearity on its own).
THUMB = [
    "brightness_mean", "saturation_mean", "text_present", "text_area_ratio_log1p",
    "text_word_count_log1p", "text_has_number", "face_present", "face_count",
    "face_area_ratio_log1p", "face_position_x", "face_position_y",
    "face_center_distance", "face_yaw_deg", "face_pitch_deg", "face_is_frontal",
    "mouth_openness", "eye_openness_mean", "contrast", "warm_cool_ratio_log1p",
    "dominant_hue_entropy", "edge_density",
    "rule_of_thirds_score", "left_right_asymmetry", "top_bottom_asymmetry",
]
# Carried in the Lasso predictor pool only (excluded from OLS for the VIF reason above).
LASSO_ONLY = ["visual_complexity", "title_char_len"]

# `extra`: outcome-specific extra RHS terms (e.g. justified interactions).
# FINAL model set = 4 outcomes (2026-06-13):
#   - Sentiment models (vibe_score, controversy) DROPPED: weak (adjR2~0.11) and the
#     LLM sentiment only covers the old 5 channels (733/1453) -> would run on a
#     different cohort than the rest. Not extending the Vertex batch; dropped outright.
#   - duration x face_count interaction DROPPED: RESET-justified on the OLD 740
#     (p=.005) but only p=.093 on the expanded 1453 cohort -> no longer earns its place;
#     the cohort expansion + video_age_days already lifted the views model to adjR2~0.49.
OUTCOMES = {
    "log_viewCount":          {"label": "Reach: log(views)",                     "view_control": False},
    "log_comment_view_rate":  {"label": "Engagement volume: log(comment/view)",  "view_control": True},
    "like_view_rate":         {"label": "Approval: like/view (raw)",             "view_control": True},
    "log_comment_depth":      {"label": "Engagement depth: log(avg comment words)", "view_control": True},
}


def load():
    v = pd.read_parquet(TABLE)
    v["channelTitle"] = v["channelTitle"].astype("category")
    # statsmodels formula needs valid identifiers; bool -> int
    for c in ["face_is_frontal", "text_present", "text_has_number", "face_present"]:
        if c in v and v[c].dtype == bool:
            v[c] = v[c].astype(int)
    return v


def predictor_cols(v):
    cont = [c for c in PRESCRIPTIVE + THUMB if c in v.columns]
    # Topic-cluster dummies now INCLUDED (decision updated 2026-06-12; see module docstring).
    topic = [c for c in v.columns if c.startswith("topic_cluster_")]
    return cont, topic


def run_ols(v, outcome, cont, topic, view_control, extra=None):
    cols = cont + topic + ["upload_year", "video_age_days"]
    if view_control:
        cols = cols + ["log_viewCount"]
    # day x hour publication interaction enters as a categorical control (its own dummies)
    rhs = " + ".join(cols) + " + C(channelTitle) + C(dow_hour)"
    # outcome-specific extra terms (e.g. duration_minutes:face_count for views)
    extra = extra or []
    if extra:
        rhs = rhs + " + " + " + ".join(extra)
    formula = f"{outcome} ~ {rhs}"
    # interaction terms reference base cols already in `cols`, so dropna on cols covers them
    df = v.dropna(subset=[outcome] + cols + ["dow_hour"]).copy()
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    return model, df, formula


def vif_table(v, cont):
    df = v[cont].dropna().copy()
    X = sm.add_constant(df)
    rows = []
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        rows.append({"feature": c, "VIF": round(variance_inflation_factor(X.values, i), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def lasso_select(v, outcome, cont, topic, view_control):
    """Scale-fair Lasso on standardised predictors; report survivors.

    The Lasso pool includes LASSO_ONLY features that OLS drops for collinearity --
    the L1 penalty resolves the redundancy, so it is safe to let it choose.
    """
    pool = cont + [c for c in LASSO_ONLY if c in v.columns]
    feats = [f"{c}_z" if f"{c}_z" in v.columns else c for c in pool]
    feats = [c for c in feats if c in v.columns]
    extra = list(topic) + ["upload_year"] + (["log_viewCount"] if view_control else [])
    use = feats + [c for c in extra if c in v.columns]
    df = v.dropna(subset=[outcome] + use).copy()
    X = StandardScaler().fit_transform(df[use].values)
    # Standardise y too so the CV-chosen lambda is comparable across outcomes.
    # (Without this, the raw like/view outcome -- scale ~0.02 -- drives lambda to
    # ~2e-5, i.e. no penalty, and "selects" everything. Z-scoring y fixes that.)
    y = StandardScaler().fit_transform(df[[outcome]].values).ravel()
    las = LassoCV(cv=5, random_state=0, max_iter=20000).fit(X, y)
    coefs = pd.DataFrame({"predictor": use, "lasso_coef_std": las.coef_})
    coefs["outcome"] = outcome
    coefs["selected"] = (coefs["lasso_coef_std"].abs() > 1e-8).astype(int)
    coefs["lambda"] = round(float(las.alpha_), 5)
    return coefs.sort_values("lasso_coef_std", key=lambda s: s.abs(), ascending=False)


def tidy_coefs(model, outcome):
    t = pd.DataFrame({
        "outcome": outcome,
        "term": model.params.index,
        "coef": model.params.values,
        "robust_se": model.bse.values,
        "z": model.tvalues.values,
        "p": model.pvalues.values,
    })
    t["sig"] = pd.cut(t["p"], [-1, .001, .01, .05, .1, 1],
                      labels=["***", "**", "*", ".", ""])
    return t


def main():
    v = load()
    cont, topic = predictor_cols(v)
    print(f"Loaded {len(v)} rows. {len(cont)} continuous/binary predictors, "
          f"{len(topic)} topic-cluster dummies.")

    # Clear stale per-outcome summaries so removed models can't leave orphan files
    # (e.g. a dropped outcome's ols_*.txt) that would contradict the current OUTCOMES.
    keep = {f"ols_{o}.txt" for o in OUTCOMES}
    for f in OUTDIR.glob("ols_*.txt"):
        if f.name not in keep:
            f.unlink()

    # --- VIF (collinearity diagnostic) ---
    vif = vif_table(v, cont)
    vif.to_csv(OUTDIR / "vif.csv", index=False, encoding="utf-8-sig")
    print("\nTop VIFs (watch n_chapters x duration_minutes):")
    print(vif.head(8).to_string(index=False))

    all_coefs, all_lasso = [], []
    for outcome, cfg in OUTCOMES.items():
        model, df, formula = run_ols(v, outcome, cont, topic, cfg["view_control"],
                                     extra=cfg.get("extra"))
        with open(OUTDIR / f"ols_{outcome}.txt", "w", encoding="utf-8") as f:
            f.write(f"OUTCOME: {cfg['label']}\nFORMULA: {formula}\n")
            f.write(f"N used: {int(model.nobs)}  (robust SE: HC1)\n\n")
            f.write(str(model.summary()))
        print(f"\n{'='*78}\n{cfg['label']}  |  N={int(model.nobs)}  "
              f"R2={model.rsquared:.3f}  adjR2={model.rsquared_adj:.3f}")
        tc = tidy_coefs(model, outcome)
        all_coefs.append(tc)
        # show prescriptive levers only (skip FE / cluster dummies / intercept) at a glance
        show = tc[tc.term.isin(cont)].copy()
        print(show[["term", "coef", "robust_se", "p", "sig"]].to_string(index=False))

        lz = lasso_select(v, outcome, cont, topic, cfg["view_control"])
        all_lasso.append(lz)
        surv = lz[lz.selected == 1]
        print(f"  Lasso (lambda={lz['lambda'].iloc[0]}): {surv.selected.sum()} predictors survive.")

    pd.concat(all_coefs, ignore_index=True).to_csv(
        OUTDIR / "coef_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_lasso, ignore_index=True).to_csv(
        OUTDIR / "lasso_selection.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved OLS summaries, vif.csv, coef_summary.csv, lasso_selection.csv "
          f"-> {OUTDIR.name}/")


if __name__ == "__main__":
    main()
