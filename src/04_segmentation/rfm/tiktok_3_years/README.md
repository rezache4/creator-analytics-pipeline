# RFM_TK_3_years

This repository contains the analysis for TikTok RFM clustering across non-overlapping 3-year periods.

The code used for the analysis is available in the notebook:
- `tk_non_overlapping_3y_periods_new_definitions.ipynb`

The repository also includes the following outputs:
- Distribution plots of the RFM variables across the 3-year periods: `tk_nonoverlapping_3y_period_distributions.png`
- The threshold summary used to define the period-level segmentation: `tk_nonoverlapping_3y_threshold_table.csv`
- User-level assignments across periods: `tk_user_nonoverlapping_3y_periods_new_definitions.csv`
- Classification results and diagnostics for the clustering workflow: `tk_classification_nonoverlapping_3y.csv`, `tk_kmedoids_k_search_nonoverlapping_3y.csv`, and `tk_kmedoids_diagnostics_nonoverlapping_3y.png`
- Cluster summaries and period-level comparisons: `tk_macro_cluster_summary_nonoverlapping_3y.csv`, `tk_macro_cluster_counts_by_period_nonoverlapping_3y.csv`, `tk_macro_cluster_shares_by_period_nonoverlapping_3y.csv`, `tk_macro_cluster_bar_by_period_nonoverlapping_3y.png`, and `tk_macro_cluster_pie_by_period_nonoverlapping_3y.png`
- Cluster assignment details: `tk_macro_cluster_assignment_nonoverlapping_3y.csv`
- Radar chart outputs showing how cluster profiles evolve over time in `cluster_radar_plots_nonoverlapping_3y/`
