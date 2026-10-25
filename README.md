# Creator Analytics Pipeline

A multi-platform social media analytics pipeline for studying how a creator's community behaves and evolves across **YouTube, Instagram and TikTok**. It was built by a student team as an M.Sc. course project at Politecnico di Milano, with an industry partner from the creator-economy sector.

This repository contains the **code only**. No source data, derived datasets, results or reports are included.

## Pipeline

```
 YouTube / Instagram / TikTok exports
              │
   01 ingestion          queue-driven YouTube transcript retrieval (GCP Pub/Sub + Cloud Run workers, yt-dlp)
              │
   02 preparation        comment cleaning and normalisation, transcript merging, JSONL job bundles
              │
   03 NLP features       transcript embeddings, topic clustering, entity extraction,
              │          thumbnail features, LLM-based sentiment (Vertex AI batch inference)
              │
   04 segmentation       commenter personas, RFM-style clustering, user journeys across periods
              │
   05 modelling          community transition models (Beta-Binomial GLMM in R), topic affinity,
                         benchmark-channel models
```

## Repository layout

| Path | Contents |
|---|---|
| `src/01_ingestion/` | Pub/Sub queue trigger, worker, foreman and retry/reconciliation scripts for YouTube transcripts |
| `src/02_preparation/` | Comment parsing and standardisation, final dataset builder, transcript merge and job preparation |
| `src/03_nlp_features/` | Embeddings, topic clustering, NER (Cloud Run), thumbnail features (Cloud Build), sentiment |
| `src/04_segmentation/` | Persona package, RFM segmentation notebooks (Instagram, TikTok), user-journey analysis |
| `src/05_modelling/` | Transition models (R), topic affinity index, benchmark-channel feature and model scripts |
| `docs/design_notes/` | Design notes on batch formats, hashing and deduplication, and local deterministic compute |
| `environment.yml` | Conda environment used for the Python pipeline |

## Data

Input data are not distributed. See [`data/README.md`](data/README.md) for the expected inputs. Names of the creator and of comparator channels were replaced with placeholders (`TargetCreator`, `ComparatorChannelN`), and local paths, cloud project IDs and service accounts were replaced with `<...>` placeholders. Notebooks are committed **without outputs**.

## Contributors

Built by:

- [@ShahrokhSp](https://github.com/ShahrokhSp)
- [@michelepollice](https://github.com/michelepollice)
- [@rezache4](https://github.com/rezache4)
