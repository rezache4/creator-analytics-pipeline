"""
residual_diagnostics.py

Residual diagnostics for the 5 baseline OLS models, to guide TWO modelling decisions:

  (A) VARIANCE STRUCTURE -- is the error variance constant?
      * Residuals-vs-fitted and scale-location plots (visual).
      * Breusch-Pagan test (formal).  -> if heteroscedastic, HC1 robust SE is the
        right call (already used); strong patterns may also argue for a transform/WLS.
      * Q-Q plot for residual normality (affects small-sample inference).

  (B) FUNCTIONAL FORM -- are we missing curvature or interactions?
      * Component+residual (partial-residual / CCPR) plots for the key continuous
        levers -> curvature there => add a polynomial (e.g. duration^2).
      * Residuals vs predictor with LOWESS -> same, model-agnostic view.
      * Ramsey RESET test (formal) -> overall "missing nonlinear terms?" flag.
      * Residuals vs channelTitle (box) and residual interaction scan
        (duration x face_count) -> suggests interaction terms.

Reuses the EXACT baseline spec by importing baseline_ols.py (same predictors,
controls, HC1). Refits per outcome to obtain residuals/fitted.

Outputs -> modeling/diagnostics/
  - resid_panel_<outcome>.png     4-up: resid-vs-fitted, scale-location, Q-Q, hist
  - ccpr_<outcome>.png            partial-residual plots for the key continuous levers
  - resid_vs_channel_<outcome>.png  residual boxplots by channel (interaction hint)
  - functional_form_tests.csv     Breusch-Pagan + RESET per outcome (the summary sheet)

Run: python residual_diagnostics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as ss
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "diagnostics"
OUTDIR.mkdir(exist_ok=True)

# Reuse the baseline spec verbatim (predictor lists, controls, HC1, OUTCOMES).
import importlib.util
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)

# Continuous levers most worth a functional-form look (interpretable, central levers).
KEY_CONT = ["duration_minutes", "n_chapters", "face_count", "topic_volatility",
            "days_since_last_video_log1p"]


def fit_all():
    v = bo.load()
    cont, topic = bo.predictor_cols(v)
    fits = {}
    for outcome, cfg in bo.OUTCOMES.items():
        model, df, formula = bo.run_ols(v, outcome, cont, topic, cfg["view_control"],
                                        extra=cfg.get("extra"))
        fits[outcome] = (model, df)
    return v, cont, fits


def resid_panel(outcome, model, df):
    fitted = model.fittedvalues
    resid = model.resid
    std_resid = resid / resid.std()
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (1) residuals vs fitted -- nonlinearity + heteroscedasticity
    ax[0, 0].scatter(fitted, resid, s=10, alpha=0.4, color="#2980b9")
    lo = lowess(resid, fitted, frac=0.6, return_sorted=True)
    ax[0, 0].plot(lo[:, 0], lo[:, 1], color="#c0392b", lw=1.6)
    ax[0, 0].axhline(0, color="grey", lw=0.8, ls="--")
    ax[0, 0].set_xlabel("Fitted"); ax[0, 0].set_ylabel("Residual")
    ax[0, 0].set_title("Residuals vs fitted (curve => nonlinearity)", fontsize=9)

    # (2) scale-location -- heteroscedasticity (rising spread)
    sl = np.sqrt(np.abs(std_resid))
    ax[0, 1].scatter(fitted, sl, s=10, alpha=0.4, color="#16a085")
    lo2 = lowess(sl, fitted, frac=0.6, return_sorted=True)
    ax[0, 1].plot(lo2[:, 0], lo2[:, 1], color="#c0392b", lw=1.6)
    ax[0, 1].set_xlabel("Fitted"); ax[0, 1].set_ylabel("sqrt(|std resid|)")
    ax[0, 1].set_title("Scale-location (slope => heteroscedasticity)", fontsize=9)

    # (3) Q-Q -- residual normality
    ss.probplot(resid, dist="norm", plot=ax[1, 0])
    ax[1, 0].get_lines()[0].set_markersize(3.5)
    ax[1, 0].get_lines()[0].set_alpha(0.5)
    ax[1, 0].set_title("Normal Q-Q", fontsize=9)

    # (4) residual histogram
    ax[1, 1].hist(resid, bins=40, color="#8e44ad", edgecolor="white", linewidth=0.3)
    ax[1, 1].set_title("Residual distribution", fontsize=9)

    fig.suptitle(f"{bo.OUTCOMES[outcome]['label']}  —  residual diagnostics (N={int(model.nobs)})",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTDIR / f"resid_panel_{outcome}.png", dpi=130, bbox_inches="tight")
    plt.close()


def ccpr_panel(outcome, model, df):
    """Component + residual (partial-residual) plots for the key continuous levers.
    Curvature in the LOWESS vs the OLS line => that predictor wants a polynomial term.
    """
    cols = [c for c in KEY_CONT if c in model.params.index]
    if not cols:
        return
    ncol = min(3, len(cols))
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.6), squeeze=False)
    X = df
    for ax, c in zip(axes.ravel(), cols):
        b = model.params[c]
        xv = X[c].to_numpy()
        comp_resid = model.resid.to_numpy() + b * xv  # component + residual
        ax.scatter(xv, comp_resid, s=9, alpha=0.35, color="#34495e")
        order = np.argsort(xv)
        xs = xv[order]
        ax.plot(xs, b * xs + (comp_resid.mean() - b * xv.mean()),
                color="#2980b9", lw=1.4, label="OLS slope")
        lo = lowess(comp_resid, xv, frac=0.6, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color="#c0392b", lw=1.6, label="LOWESS")
        ax.set_xlabel(c, fontsize=8); ax.set_title(f"CCPR: {c}", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle(f"{bo.OUTCOMES[outcome]['label']}  —  partial-residual plots "
                 f"(LOWESS bends from line => add polynomial)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTDIR / f"ccpr_{outcome}.png", dpi=130, bbox_inches="tight")
    plt.close()


def resid_vs_channel(outcome, model, df):
    """Residual spread by channel -> uneven spread/level => channel interactions or
    channel-specific variance (argues for clustered SE or interaction terms)."""
    d = df.copy()
    d["_resid"] = model.resid
    chans = list(d["channelTitle"].cat.categories) if hasattr(d["channelTitle"], "cat") \
        else sorted(d["channelTitle"].unique())
    data = [d.loc[d["channelTitle"] == ch, "_resid"].values for ch in chans]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=chans, showmeans=True)
    ax.axhline(0, color="#c0392b", lw=0.9, ls="--")
    ax.set_ylabel("Residual"); ax.tick_params(axis="x", labelsize=8, rotation=15)
    ax.set_title(f"{bo.OUTCOMES[outcome]['label']} — residuals by channel\n"
                 "(uneven spread => heteroscedasticity by channel / interactions)", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"resid_vs_channel_{outcome}.png", dpi=130, bbox_inches="tight")
    plt.close()


def formal_tests(outcome, model):
    """Breusch-Pagan (heteroscedasticity) + Ramsey RESET (functional form)."""
    bp_lm, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    row = {"outcome": outcome,
           "breusch_pagan_LM": round(bp_lm, 2), "bp_p": round(bp_p, 4),
           "heteroscedastic_p05": bp_p < 0.05}
    try:
        reset = linear_reset(model, power=2, use_f=True)
        row["reset_F"] = round(float(reset.fvalue), 2)
        row["reset_p"] = round(float(reset.pvalue), 4)
        row["nonlinear_p05"] = float(reset.pvalue) < 0.05
    except Exception as e:
        row["reset_F"] = np.nan; row["reset_p"] = np.nan; row["nonlinear_p05"] = None
    return row


def main():
    v, cont, fits = fit_all()
    rows = []
    for outcome, (model, df) in fits.items():
        resid_panel(outcome, model, df)
        ccpr_panel(outcome, model, df)
        resid_vs_channel(outcome, model, df)
        rows.append(formal_tests(outcome, model))

    tests = pd.DataFrame(rows)
    tests.to_csv(OUTDIR / "functional_form_tests.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("Formal diagnostics (Breusch-Pagan heteroscedasticity; Ramsey RESET form):")
    print(tests.to_string(index=False))
    print("\nReading guide:")
    print("  bp_p < .05         -> heteroscedastic: keep HC1 robust SE (already used).")
    print("  reset_p < .05      -> missing nonlinear/interaction terms: inspect ccpr_*.png.")
    print("  ccpr LOWESS bends  -> add that predictor's square (e.g. duration_minutes^2).")
    print("  uneven channel box -> consider channel-clustered SE or channel interactions.")
    pngs = sorted(p.name for p in OUTDIR.glob("*.png"))
    print(f"\nWrote {len(pngs)} plots + functional_form_tests.csv -> {OUTDIR.relative_to(BASE.parent)}/")


if __name__ == "__main__":
    main()
