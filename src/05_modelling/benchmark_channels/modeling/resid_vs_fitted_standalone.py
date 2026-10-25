"""
resid_vs_fitted_standalone.py

Generates the residuals-vs-fitted plot ON ITS OWN (one file per model),
extracted from the 4-up quartet produced by residual_diagnostics.py.
Same model spec, same residuals/fitted, same LOWESS overlay.

Outputs -> modeling/diagnostics/resid_vs_fitted_<outcome>.png

Run: python resid_vs_fitted_standalone.py
"""
from pathlib import Path
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

BASE   = Path(__file__).resolve().parent
OUTDIR = BASE / "diagnostics"
OUTDIR.mkdir(exist_ok=True)

# Reuse the baseline spec verbatim (predictors, controls, HC1, OUTCOMES).
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)


def resid_vs_fitted(outcome, model):
    fitted = model.fittedvalues
    resid  = model.resid

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(fitted, resid, s=10, alpha=0.4, color="#2980b9")
    lo = lowess(resid, fitted, frac=0.6, return_sorted=True)
    ax.plot(lo[:, 0], lo[:, 1], color="#c0392b", lw=1.6)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residual")
    ax.set_title(f"{bo.OUTCOMES[outcome]['label']}\nResiduals vs fitted "
                 f"(N={int(model.nobs)})", fontsize=10)
    fig.tight_layout()
    out = OUTDIR / f"resid_vs_fitted_{outcome}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    v = bo.load()
    cont, topic = bo.predictor_cols(v)
    for outcome, cfg in bo.OUTCOMES.items():
        model, df, formula = bo.run_ols(v, outcome, cont, topic,
                                        cfg["view_control"], extra=cfg.get("extra"))
        resid_vs_fitted(outcome, model)


if __name__ == "__main__":
    main()
