# Entity Extraction & Resonance

We run Named Entity Recognition (NER) over both video transcripts and YouTube comments to identify which entities discussed in a video also appear in audience reactions. This cross-source overlap is called **entity resonance**.

## Model

`spaCy it_core_news_lg` — large Italian pipeline. The dataset is primarily Italian-language content, so this model provides accurate entity spans for persons, organisations, locations, and named works.

```bash
pip install spacy
python -m spacy download it_core_news_lg
```

## Entity Types Retained

| Label | Meaning |
|-------|---------|
| `PER` | Persons |
| `ORG` | Organisations, brands, channels |
| `LOC` | Locations |
| `GPE` | Geo-political entities |
| `MISC` | Events, works of art, other named things |
| `PROD` | Products (when present in model) |

Only entities with `len(text.strip()) >= 2` characters are kept to filter noise tokens.

## Output Files

| File | Granularity | Description |
|------|-------------|-------------|
| `entities_transcripts.parquet` | one row per video | `{video_id, entities}` — JSON list of `{text, label, count}` sorted by count desc |
| `entities_comments.parquet` | one row per video | Same structure, counts aggregated across all comments for that video |
| `entity_resonance.parquet` | one row per (video, entity) | Flat outer-join of transcript and comment entities |

## Entity Resonance Table

The resonance table is the key analytical artifact. Each row has:

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | YouTube video ID |
| `entity_text` | str | Surface form of the entity |
| `entity_label` | str | spaCy label (PER, ORG, …) |
| `in_transcript` | bool | Entity appears in the transcript |
| `transcript_count` | int | Mention count in transcript |
| `in_comments` | bool | Entity appears in comments |
| `comment_count` | int | Aggregated mention count across comments |

An entity is **resonant** when both `in_transcript` and `in_comments` are True. High `comment_count` resonant entities are the topics the audience latched onto — a useful signal for engagement modelling.

## Export Formats

`export_entity_data.py` converts the flat resonance parquet into three formats suited to different downstream uses:

| Output file | Format | Best for |
|-------------|--------|----------|
| `entity_data_per_video.jsonl` | JSONL | Streaming pipeline processing, line-by-line parsing |
| `entity_data_per_video.json` | JSON array | Full dataset queries, nested structure exploration |
| `entity_data_per_video.csv` | CSV | Spreadsheet analysis, filtering by `entity_type` or label |

Each per-video record groups entities into three lists:
- `transcript_entities` — transcript-only, sorted by `count` desc
- `comment_entities` — comments-only, sorted by `count` desc
- `resonant_entities` — both sources, sorted by `comment_count` desc

## Usage

```bash
# From the Data_Cleaned directory
python extract_entities.py                     # transcripts + comments + resonance
python extract_entities.py --transcripts-only  # skip comments
python extract_entities.py --comments-only     # skip transcripts
python extract_entities.py --dry-run           # process 50 rows each, no file writes

# GCS mode (reads from gs://{bucket}/input/, writes to gs://{bucket}/output/ner/)
GCS_BUCKET=my-bucket python extract_entities.py

# Export resonance to JSONL / JSON / CSV
python export_entity_data.py
```
