# benchmark_analysis — layout

Engagement/performance driver analysis for format-similar Italian YouTubers
(TargetCreator is one panel member; cohort expanded to 10 channels 2026-06-12).
Start with `docs/SESSION_HANDOFF.md`.

## Folders

| Folder | Stage | Status |
|---|---|---|
| `docs/` | Handoff, findings, locked feature spec, plans | reference |
| `cohort/` | Cohort selection, sample size, clip-vs-native classifier, embedding cross-check | **done** (cohort LOCKED — 10 channels; `cohort_config.py` is the single source of truth) |
| `features/` | `build_feature_table.py` → **`feature_table.parquet`** (the canonical modeling table) + manifest; modeling-prep EDA + decision checks | **live** (build) + done (checks) |
| `comments_analysis/` | Comment-signal PCA/EDA that justified the depth/affect outcomes | **live** |
| `modeling/` | `baseline_ols.py` (**4-outcome** OLS, HC1) + `baseline_plots.py`; outputs in `ols_baseline/` | **live** |
| `_legacy/` | Pre-cohort per-channel exploratory work (ComparatorChannel4, ComparatorChannel7, affinity index) | archival |

## Live pipeline (re-runnable, in order)

```
features/extract_speech_rate.py        # writes features/speech_rate.parquet (VTT pacing); run BEFORE build
features/build_feature_table.py        # writes features/feature_table.parquet (1453x117, 6 outcomes)
comments_analysis/comment_signal_explore.py
modeling/baseline_ols.py               # 6-outcome OLS (HC1); reads features/feature_table.parquet
modeling/baseline_plots.py             # report plots; reads ols_baseline/ + the parquet
modeling/residual_diagnostics.py       # residual panels, CCPR, BP+RESET tests
modeling/functional_form_probe.py      # quantifies poly/interaction terms to add
modeling/variance_function_refit.py    # FGLS/WLS vs HC1 comparison (verdict: keep HC1)
```

Dependency note: `extract_speech_rate.py` must run before `build_feature_table.py`
(the build merges `speech_rate.parquet`). The sentiment outcomes come from
`features/comment_features/video_comment_features.parquet` (built separately).

Run with `python <script>`. These scripts were
path-fixed after the reorg: `PROJ = BASE.parent.parent` points at `` for raw data;
the shared `feature_table.parquet` lives in `features/` and is referenced via
`BASE.parent / "features"`.

## Archival scripts (NOT re-run; internal paths may be stale after the move)

`cohort/principled_analysis/*.py` and the `features/check_*.py` /
`features/modeling_prep_eda.py` diagnostics produced their CSV/PNG outputs and fed the
**locked** decisions in `docs/`. They are kept for provenance; `check_*.py` scripts now
import COHORT from `features/cohort_config.py` and can be re-run safely.
