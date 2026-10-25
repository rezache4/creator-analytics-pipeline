"""
baseline_plots.py

Report-ready visualisations for the baseline regression stage (5 outcomes).
Reads the artefacts written by baseline_ols.py (coef_summary.csv, lasso_selection.csv)
and feature_table.parquet. Produces:

  1. coef_forest_<outcome>.png   per-outcome coefficient forest plot (robust 95% CI),
                                 prescriptive levers only, sorted, sig-coloured.
  2. std_coef_heatmap.png        standardised OLS coefficients, predictors x outcomes,
                                 one glance at "what drives what" across all 5 models.
  3. outcome_distributions.png   the 5 modelled outcomes (post-transform) + n_comments.
  4. model_fit.png               adj-R2 bar across the 5 outcomes.
  5. lasso_survivors.png         # predictors surviving Lasso per outcome.

Outputs -> benchmark_analysis/ols_baseline/plots/

Run: python baseline_plots.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OLSDIR = BASE / "ols_baseline"
PLOTDIR = OLSDIR / "plots"
PLOTDIR.mkdir(parents=True, exist_ok=True)
TABLE = BASE.parent / "features" / "feature_table.parquet"

# Import the predictor lists + outcome labels from the OLS script (single source).
import importlib.util
spec = importlib.util.spec_from_file_location("baseline_ols", BASE / "baseline_ols.py")
bo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bo)

PRESCRIPTIVE_AND_THUMB = bo.PRESCRIPTIVE + bo.THUMB
OUTCOMES = bo.OUTCOMES
LABELS = {k: v["label"] for k, v in OUTCOMES.items()}
SIG_COLORS = {"***": "#1a5276", "**": "#2874a6", "*": "#5499c7", ".": "#aed6f1", "": "#d5d8dc"}


def sig_of(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .1 else ""


def coef_forest(coefs):
    """One forest plot per outcome: prescriptive-lever coefs with robust 95% CI."""
    for outcome, cfg in OUTCOMES.items():
        d = coefs[(coefs.outcome == outcome) & (coefs.term.isin(PRESCRIPTIVE_AND_THUMB))].copy()
        if d.empty:
            continue
        d["sig"] = d["p"].map(sig_of)
        d["lo"] = d["coef"] - 1.96 * d["robust_se"]
        d["hi"] = d["coef"] + 1.96 * d["robust_se"]
        d = d.sort_values("coef")
        y = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(8.5, max(5, 0.30 * len(d))))
        colors = d["sig"].map(SIG_COLORS)
        ax.hlines(y, d["lo"], d["hi"], color=colors, lw=2.2)
        ax.scatter(d["coef"], y, color=colors, s=34, zorder=3)
        ax.axvline(0, color="#7b241c", lw=0.9, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(d["term"], fontsize=8)
        ax.set_xlabel("Coefficient (robust 95% CI)", fontsize=9)
        ax.set_title(f"{cfg['label']}\nPrescriptive-lever coefficients (HC1)", fontsize=10)
        handles = [plt.Line2D([0], [0], color=SIG_COLORS[s], lw=3,
                              label={"***": "p<.001", "**": "p<.01", "*": "p<.05",
                                     ".": "p<.1", "": "ns"}[s])
                   for s in ["***", "**", "*", ".", ""]]
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(PLOTDIR / f"coef_forest_{outcome}.png", dpi=130, bbox_inches="tight")
        plt.close()


def std_coef_heatmap(v, coefs):
    """Standardised OLS coefs (predictor SD * coef / outcome SD) -> comparable cells."""
    terms = [t for t in PRESCRIPTIVE_AND_THUMB if t in v.columns]
    M = pd.DataFrame(index=terms, columns=list(OUTCOMES), dtype=float)
    P = pd.DataFrame(index=terms, columns=list(OUTCOMES), dtype=float)
    for outcome in OUTCOMES:
        sd_y = v[outcome].std()
        sub = coefs[coefs.outcome == outcome].set_index("term")
        for t in terms:
            if t in sub.index:
                M.loc[t, outcome] = sub.loc[t, "coef"] * v[t].std() / sd_y
                P.loc[t, outcome] = sub.loc[t, "p"]
    M = M.astype(float)
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(terms) + 2))
    vmax = np.nanpercentile(np.abs(M.values), 98)
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([LABELS[o].split(":")[0] for o in OUTCOMES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(terms))); ax.set_yticklabels(terms, fontsize=7.5)
    for i, t in enumerate(terms):
        for j, o in enumerate(OUTCOMES):
            p = P.loc[t, o]
            if pd.notna(p) and p < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=10,
                        color="black", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="std. coefficient")
    ax.set_title("Standardised OLS coefficients across outcomes\n"
                 "(* = robust p<.05; cell = SD change in outcome per 1 SD of predictor)", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTDIR / "std_coef_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()


def outcome_distributions(v):
    cols = list(OUTCOMES) + ["n_comments"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), cols):
        s = v[c].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(s, bins=40, color="#16a085", edgecolor="white", linewidth=0.3)
        title = LABELS.get(c, c) if c in LABELS else c
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.axvline(s.median(), color="#c0392b", lw=1, ls="--")
    fig.suptitle(f"Modelled outcomes (post-transform) + comment support (N={len(v)})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(PLOTDIR / "outcome_distributions.png", dpi=130, bbox_inches="tight")
    plt.close()


def model_fit(coefs):
    """adj-R2 per outcome, read from the saved ols_<outcome>.txt summaries."""
    fits = {}
    for outcome in OUTCOMES:
        txt = (OLSDIR / f"ols_{outcome}.txt").read_text(encoding="utf-8")
        for line in txt.splitlines():
            if "Adj. R-squared" in line:
                fits[outcome] = float(line.split(":")[-1].strip())
                break
    s = pd.Series(fits).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh([LABELS[o] for o in s.index], s.values, color="#2980b9")
    for b, val in zip(bars, s.values):
        ax.text(val + 0.005, b.get_y() + b.get_height() / 2, f"{val:.3f}",
                va="center", fontsize=8)
    ax.set_xlabel("Adjusted R²", fontsize=9)
    ax.set_title("Baseline model fit by outcome (OLS + channel FE, HC1)", fontsize=10)
    ax.set_xlim(0, max(s.values) * 1.15)
    plt.tight_layout()
    plt.savefig(PLOTDIR / "model_fit.png", dpi=130, bbox_inches="tight")
    plt.close()


def lasso_survivors(lasso):
    surv = (lasso[lasso.selected == 1].groupby("outcome").size()
            .reindex(list(OUTCOMES)).fillna(0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([LABELS[o] for o in surv.index], surv.values, color="#8e44ad")
    for i, val in enumerate(surv.values):
        ax.text(val + 0.3, i, f"{int(val)}", va="center", fontsize=8)
    ax.set_xlabel("# predictors surviving Lasso (CV-λ, standardised X & y)", fontsize=9)
    ax.set_title("Lasso sparsity by outcome", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTDIR / "lasso_survivors.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    v = pd.read_parquet(TABLE)
    coefs = pd.read_csv(OLSDIR / "coef_summary.csv")
    lasso = pd.read_csv(OLSDIR / "lasso_selection.csv")

    coef_forest(coefs)
    std_coef_heatmap(v, coefs)
    outcome_distributions(v)
    model_fit(coefs)
    lasso_survivors(lasso)

    pngs = sorted(p.name for p in PLOTDIR.glob("*.png"))
    print(f"Wrote {len(pngs)} plots -> {PLOTDIR.relative_to(BASE)}/")
    for p in pngs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
