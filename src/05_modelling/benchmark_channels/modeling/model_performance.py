"""
model_performance.py

Standard performance testimonies for the 6 OLS models -- the consolidated pieces that
the per-model diagnostics don't already give:

  1. model_metrics.csv / .md  -- one table: N, k (params), R2, adj-R2, AIC, BIC,
     log-likelihood, F-stat + p, RMSE, condition number, and post-fit Breusch-Pagan p
     and Ramsey RESET p (heteroscedasticity / functional-form flags) per outcome.
  2. pred_vs_actual.png        -- predicted vs observed scatter per model with the 45deg
     line and R2 annotated (the classic fit testimony).
  3. metrics_bar.png           -- adj-R2 + RMSE bars across models at a glance.

Residual panels, scale-location, Q-Q, CCPR, resid-by-channel already exist in
diagnostics/ (resid_panel_*, ccpr_*, resid_vs_channel_*) -- not duplicated here.

Reuses the exact baseline spec via baseline_ols.py.

Run: python model_performance.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "ols_baseline"
PLOTDIR = OUTDIR / "plots"
PLOTDIR.mkdir(parents=True, exist_ok=True)

import importlib.util
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)


def fit_all():
    v = bo.load()
    cont, topic = bo.predictor_cols(v)
    out = {}
    for outcome, cfg in bo.OUTCOMES.items():
        model, df, _ = bo.run_ols(v, outcome, cont, topic, cfg["view_control"],
                                  extra=cfg.get("extra"))
        out[outcome] = (model, df, cfg["label"])
    return out


def metrics_row(outcome, model, label):
    resid = model.resid
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    try:
        bp_p = float(het_breuschpagan(resid, model.model.exog)[1])
    except Exception:
        bp_p = np.nan
    try:
        reset_p = float(linear_reset(model, power=2, use_f=True).pvalue)
    except Exception:
        reset_p = np.nan
    return {
        "outcome": outcome, "label": label,
        "N": int(model.nobs), "k_params": int(model.df_model) + 1,
        "R2": round(model.rsquared, 4), "adj_R2": round(model.rsquared_adj, 4),
        "AIC": round(model.aic, 1), "BIC": round(model.bic, 1),
        "logLik": round(model.llf, 1),
        "F": round(float(model.fvalue), 2), "F_p": _fmt_p(model.f_pvalue),
        "RMSE": round(rmse, 4),
        "cond_no": round(float(model.condition_number), 1),
        "BP_p": round(bp_p, 4), "RESET_p": round(reset_p, 4) if pd.notna(reset_p) else np.nan,
        "heterosced": "yes" if (pd.notna(bp_p) and bp_p < .05) else "no",
        "nonlinear": "yes" if (pd.notna(reset_p) and reset_p < .05) else "no",
    }


def _fmt_p(p):
    return "<1e-4" if (pd.notna(p) and p < 1e-4) else (round(float(p), 4) if pd.notna(p) else np.nan)


def _to_md(df):
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def pred_vs_actual(fits):
    n = len(fits)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4, nrow * 3.6), squeeze=False)
    for ax, (outcome, (model, df, label)) in zip(axes.ravel(), fits.items()):
        y = model.model.endog
        yhat = model.fittedvalues
        ax.scatter(yhat, y, s=10, alpha=0.4, color="#2980b9")
        lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
        ax.plot(lims, lims, color="#c0392b", lw=1.2, ls="--")
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("actual", fontsize=8)
        ax.set_title(f"{label}\nR2={model.rsquared:.3f}  adjR2={model.rsquared_adj:.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Predicted vs actual (45deg = perfect)  —  OLS + channel FE, HC1", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOTDIR / "pred_vs_actual.png", dpi=130, bbox_inches="tight")
    plt.close()


def metrics_bar(tab):
    t = tab.sort_values("adj_R2")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.barh(t["label"], t["adj_R2"], color="#2980b9")
    for i, val in enumerate(t["adj_R2"]):
        a1.text(val + .005, i, f"{val:.3f}", va="center", fontsize=8)
    a1.set_xlabel("adj R2"); a1.set_title("Fit (adj R2)", fontsize=10); a1.set_xlim(0, t["adj_R2"].max() * 1.15)
    a2.barh(t["label"], t["RMSE"], color="#8e44ad")
    for i, val in enumerate(t["RMSE"]):
        a2.text(val, i, f" {val:.3f}", va="center", fontsize=8)
    a2.set_xlabel("RMSE (model scale)"); a2.set_title("Error (RMSE)", fontsize=10)
    a2.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(PLOTDIR / "metrics_bar.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    fits = fit_all()
    rows = [metrics_row(o, m, lab) for o, (m, df, lab) in fits.items()]
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTDIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    (OUTDIR / "model_metrics.md").write_text(
        "# Model performance metrics\n\n" + _to_md(tab), encoding="utf-8")

    pred_vs_actual(fits)
    metrics_bar(tab)

    pd.set_option("display.width", 240)
    show = tab[["label", "N", "k_params", "R2", "adj_R2", "AIC", "BIC", "RMSE",
                "F", "F_p", "heterosced", "nonlinear"]]
    print(show.to_string(index=False))
    print("\nNote: AIC/BIC/RMSE/logLik are comparable ONLY across models of the SAME "
          "outcome (different y => different scale). Use them to compare model VARIANTS "
          "(e.g. views with vs without an interaction), not across different outcomes.")
    print(f"\nWrote model_metrics.csv/.md + pred_vs_actual.png + metrics_bar.png")


if __name__ == "__main__":
    main()
