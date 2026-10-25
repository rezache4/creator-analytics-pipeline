# Data

No data are included in this repository, and none should be committed.

The pipeline expects the following kinds of input, which you must supply yourself:

| Stage | Input |
|---|---|
| Ingestion | A table of YouTube video IDs (one per row) used to fill the Pub/Sub transcript queue |
| Preparation | Platform exports of posts and comments for Instagram, TikTok and YouTube |
| NLP features | Prepared transcripts and comments (Parquet / JSON Lines) |
| Segmentation / modelling | Per-user comment histories and the feature tables produced by the earlier stages |

Scripts contain `<DATA_DIR>`, `<GCS_BUCKET>`, `<GCP_PROJECT_ID>` and similar placeholders. Replace them with your own locations before running anything.

User-level comment data and platform exports contain personal data and must stay out of version control. The root `.gitignore` excludes common data and output formats.
