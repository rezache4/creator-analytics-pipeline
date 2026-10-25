"""
variance_function_refit.py

The residual diagnostics found ALL models heteroscedastic (Breusch-Pagan p<.05), and
for depth the variance is clearly CHANNEL-linked (residual-by-channel boxplots). HC1
already gives honest SEs; this script asks the next question: can we MODEL the variance
and refit by FGLS/WLS to (a) gain efficiency (tighter SEs / better fit) and (b) remove
the residual heteroscedasticity?

Two standard variance functions per outcome, compared against the HC1 OLS baseline:

  (A) Harvey multiplicative form -- Var(e) = exp(g0 + g1 * fitted):
      regress log(e^2) on the OLS fitted values, take fitted variances as weights,
      refit WLS. Targets variance that scales with the mean.

  (B) Groupwise (channel) form -- Var(e) = sigma_c^2 per channel:
      weight each obs by 1 / (channel residual variance), refit WLS. Targets the
      between-channel variance the boxplots showed (esp. for depth).

Outputs (modeling/diagnostics/):
  - variance_function_comparison.csv   per outcome x method: weighted/robust SE summary,
        post-refit Breusch-Pagan p, adj-R2 (pseudo for WLS), mean |SE change| vs HC1.
  - vfunc_resid_<outcome>.png           before/after scale-location for the better method.

Decision printed: for each outcome, whether a variance function HELPS (homoscedastic
post-weighting + tighter SEs) or whether HC1 OLS remains the pragmatic choice.

Run: python variance_function_refit.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

BASE = Path(__file__).resolve().parent
OUTDIR = BASE / "diagnostics"
WLS_OUTDIR = BASE / "wls_models"  # New directory for WLS model summaries
WLS_OUTDIR.mkdir(exist_ok=True) # Ensure WLS output directory exists once

OUTDIR.mkdir(exist_ok=True)

import importlib.util
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)


def fit_ols(v, outcome, cont, topic, cfg):
    model, df, formula = bo.run_ols(v, outcome, cont, topic, cfg["view_control"],
                                    extra=cfg.get("extra"))
    return model, df, formula


def bp_p(model):
    try:
        return float(het_breuschpagan(model.resid, model.model.exog)[1])
    except Exception:
        return np.nan


def harvey_weights(model):
    """Variance function A: Var(e)=exp(g0+g1*fitted). Returns 1/var weights."""
    fitted = model.fittedvalues.values
    log_e2 = np.log(np.maximum(model.resid.values ** 2, 1e-8))
    X = sm.add_constant(fitted)
    g = sm.OLS(log_e2, X).fit()
    var_hat = np.exp(g.predict(X))
    return 1.0 / np.maximum(var_hat, 1e-8)


def group_weights(model, df, group="channelTitle"):
    """Variance function B: Var(e)=sigma_c^2 per channel. Returns 1/var weights."""
    resid = pd.Series(model.resid.values, index=df.index)
    gv = resid.groupby(df[group]).transform("var")
    return (1.0 / np.maximum(gv.values, 1e-8))


def refit_wls(df, formula, weights):
    return smf.wls(formula, data=df, weights=weights).fit()


def scale_location(ax, model, title):
    fitted = np.asarray(model.fittedvalues)
    sr = np.sqrt(np.abs(model.resid / model.resid.std()))
    ax.scatter(fitted, sr, s=9, alpha=0.4, color="#16a085")
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lo = lowess(sr, fitted, frac=0.6, return_sorted=True)
    ax.plot(lo[:, 0], lo[:, 1], color="#c0392b", lw=1.6)
    ax.set_title(title, fontsize=9); ax.set_xlabel("fitted"); ax.set_ylabel("sqrt|std resid|")


def main():
    v = bo.load()
    cont, topic = bo.predictor_cols(v)
    rows = []
    for outcome, cfg in bo.OUTCOMES.items():
        ols, df, formula = fit_ols(v, outcome, cont, topic, cfg)
        ols_bp = bp_p(ols)
        base_se = ols.bse

        hc1_se = pd.Series(ols.get_robustcov_results("HC1").bse, index=ols.bse.index)
        models = {
            "OLS_HC1": ols,                                   # BP read from raw resid
            "WLS_Harvey": refit_wls(df, formula, harvey_weights(ols)),
            "WLS_channel": refit_wls(df, formula, group_weights(ols, df)),
        }
        for method, m in models.items():
            bpp = ols_bp if method == "OLS_HC1" else bp_p(m)
            # median SE ratio vs HC1 (on shared terms): <1 = tighter (efficiency gain)
            common = m.bse.index.intersection(hc1_se.index)
            se_ratio = float((m.bse[common] / hc1_se[common]).median()) if len(common) else np.nan

            if method.startswith("WLS"): # Save WLS model summaries to their dedicated folder
                with open(WLS_OUTDIR / f"{method.lower()}_{outcome}.txt", "w", encoding="utf-8") as f:
                    f.write(f"OUTCOME: {cfg['label']}\nMETHOD: {method}\n")
                    f.write(f"N used: {int(m.nobs)}\n\n")
                    f.write(str(m.summary()))
            rows.append({
                "outcome": outcome, "method": method,
                "adj_r2": round(float(getattr(m, "rsquared_adj", np.nan)), 4),
                "post_BP_p": round(bpp, 4) if pd.notna(bpp) else np.nan,
                "homoscedastic_now": (pd.notna(bpp) and bpp > 0.05),
                "median_SE_vs_HC1": round(se_ratio, 3),
            })

        # before/after scale-location for channel-weighted (the motivated one)
        wls_b = refit_wls(df, formula, group_weights(ols, df))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        scale_location(ax[0], ols, f"{cfg['label']}\nOLS (BP p={ols_bp:.3f})")
        scale_location(ax[1], wls_b, f"WLS channel-weighted (BP p={bp_p(wls_b):.3f})")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"vfunc_resid_{outcome}.png", dpi=120, bbox_inches="tight")
        plt.close()

    res = pd.DataFrame(rows)
    res.to_csv(OUTDIR / "variance_function_comparison.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("Variance-function comparison (vs HC1 OLS baseline):\n")
    print(res.to_string(index=False))
    print("\nReading: post_BP_p > .05 => that weighting REMOVED heteroscedasticity.")
    print("median_SE_vs_HC1 < 1 => weighting tightened SEs (efficiency gain); ~1 => no gain.")
    print("\nVerdict per outcome:")
    for outcome in bo.OUTCOMES:
        sub = res[res.outcome == outcome]
        fixed = sub[(sub.method != "OLS_HC1") & (sub.homoscedastic_now)]
        if len(fixed):
            best = fixed.sort_values("median_SE_vs_HC1").iloc[0]
            print(f"  {outcome:24} -> {best['method']} removes heterosced. "
                  f"(SE x{best['median_SE_vs_HC1']}); worth adopting.")
        else:
            print(f"  {outcome:24} -> no variance function removes it; KEEP HC1 OLS.")
    print(f"\nSaved comparison + 6 before/after plots -> {OUTDIR.relative_to(BASE.parent)}/")


if __name__ == "__main__":
    main()
