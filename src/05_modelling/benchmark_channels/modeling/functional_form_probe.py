"""
functional_form_probe.py

The RESET test flagged log_viewCount and log_comment_depth as having missing
nonlinear/interaction terms. This probe QUANTIFIES which specific term to add, so the
CCPR plots' visual hints become concrete model edits.

For the two flagged outcomes it tests, on top of the baseline spec:
  (1) a quadratic term for each key continuous lever (duration_minutes, n_chapters,
      face_count, days_since_last_video_log1p) -- is the squared term significant
      (robust) and does adj-R2 improve?
  (2) a duration_minutes x face_count interaction -- the two dominant levers; do they
      moderate each other?

Reports robust p-values and adj-R2 deltas vs the baseline. Decision rule printed.

Run: python functional_form_probe.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent
import importlib.util
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)

FLAGGED = ["log_viewCount", "log_comment_depth"]
QUAD_CANDIDATES = ["duration_minutes", "n_chapters", "face_count",
                   "days_since_last_video_log1p"]


def baseline_formula(v, outcome, cont, topic, view_control):
    cols = cont + topic + ["upload_year"] + (["log_viewCount"] if view_control else [])
    return f"{outcome} ~ " + " + ".join(cols) + " + C(channelTitle)", cols


def fit(formula, df):
    return smf.ols(formula, data=df).fit(cov_type="HC1")


def main():
    v = bo.load()
    cont, topic = bo.predictor_cols(v)
    rows = []
    for outcome in FLAGGED:
        cfg = bo.OUTCOMES[outcome]
        base_f, cols = baseline_formula(v, outcome, cont, topic, cfg["view_control"])
        df = v.dropna(subset=[outcome] + cols).copy()
        base = fit(base_f, df)
        base_adj = base.rsquared_adj

        # (1) quadratic terms, one at a time
        for c in QUAD_CANDIDATES:
            if c not in df.columns:
                continue
            f2 = f"{base_f} + I({c}**2)"
            m = fit(f2, df)
            term = f"I({c} ** 2)"
            p = m.pvalues.get(term, np.nan)
            rows.append({
                "outcome": outcome, "test": f"{c}^2", "robust_p": round(float(p), 4),
                "adj_r2_base": round(base_adj, 4), "adj_r2_new": round(m.rsquared_adj, 4),
                "adj_r2_delta": round(m.rsquared_adj - base_adj, 4),
                "verdict": "ADD" if (pd.notna(p) and p < 0.05 and m.rsquared_adj > base_adj)
                           else "no",
            })

        # (2) duration x face_count interaction
        if {"duration_minutes", "face_count"} <= set(df.columns):
            fi = f"{base_f} + duration_minutes:face_count"
            m = fit(fi, df)
            term = "duration_minutes:face_count"
            p = m.pvalues.get(term, np.nan)
            rows.append({
                "outcome": outcome, "test": "duration x face_count",
                "robust_p": round(float(p), 4),
                "adj_r2_base": round(base_adj, 4), "adj_r2_new": round(m.rsquared_adj, 4),
                "adj_r2_delta": round(m.rsquared_adj - base_adj, 4),
                "verdict": "ADD" if (pd.notna(p) and p < 0.05 and m.rsquared_adj > base_adj)
                           else "no",
            })

    res = pd.DataFrame(rows)
    out = BASE / "diagnostics" / "functional_form_probe.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("Functional-form probe (robust p; adj-R2 vs baseline):\n")
    print(res.to_string(index=False))
    print("\nVerdict = ADD when the term is robust-significant (p<.05) AND adj-R2 rises.")
    adds = res[res.verdict == "ADD"]
    if len(adds):
        print("\nTerms to add in the next model iteration:")
        for _, r in adds.iterrows():
            print(f"  [{r['outcome']}]  {r['test']}  (p={r['robust_p']}, +{r['adj_r2_delta']} adjR2)")
    else:
        print("\nNo single quadratic/interaction term clears the bar — RESET signal is "
              "likely diffuse (many tiny nonlinearities) rather than one missing term.")
    print(f"\nSaved -> {out.relative_to(BASE.parent)}")


if __name__ == "__main__":
    main()
