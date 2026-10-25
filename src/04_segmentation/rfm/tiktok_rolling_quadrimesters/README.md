# RFM_TK_rolling_quadrimesters

This repository contains the analysis for TikTok RFM/RCEGT clustering based on rolling quadrimesters.

The code used for the analysis is available in the notebook:
- `tk_rolling_quadrimesters.ipynb`

The repository also includes the following outputs:
- Distribution plots of the RFM/RCEGT variables across the rolling quadrimester windows: `tk_rolling_quadrimesters_2025_first5_distributions.png`
- The threshold summary used to define the rolling quadrimester segmentation: `tk_rolling_quadrimesters_threshold_table.csv`
- User-level rolling quadrimester assignments: `tk_user_rolling_quadrimesters.csv`
- Classification outputs and user frequency profiles: `tk_RCEGT_classification.csv` and `tk_RCEGT_user_frequency_profiles.csv`
- K-means cluster search, diagnostics, assignments, and summaries: `tk_RCEGT_kmeans_k_search.csv`, `tk_RCEGT_kmeans_silhouette_k_search.png`, `tk_RCEGT_kmeans_k3_user_clusters.csv`, and `tk_RCEGT_kmeans_k3_cluster_summary.csv`
- Rolling cluster matrices and transition outputs: `tk_RCEGT_k3_monthly_cluster_matrix.csv` and `tk_RCEGT_k3_user_transitions.csv`
- Sankey diagram materials in `Sankey diagram RCEGT/`
- Transition matrices in `Transition matrixes 4m/`
- Summary tables in `Summary tables/`
