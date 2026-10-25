# Persona Pipeline (OO port of `persona_pipeline.ipynb`)

Object-oriented, CLI-driven version of the Show Reel community **persona
pipeline**: a multimodal, two-stage Vertex AI **batch** workflow that discovers an
audience-persona taxonomy from a stratified comment sample and classifies every
user into it.

The package preserves the notebook's **async submit/retrieve** model: each batch
stage is submitted as a non-blocking Vertex job, the handle is persisted to
`outputs/<stage>_job.json`, and you retrieve later from a fresh process.

## Install

```bash
pip install -r persona/requirements.txt   # umap-learn/hdbscan/scikit-learn only needed for Pathway B / Stage 3
```

Auth is **Application Default Credentials** (ADC) — no key files. Run from the
`Ali/` directory so relative paths (`outputs/`, `multimodal_dataset_fixed/`)
resolve as in the notebook.

## Architecture

| Module | Class | Notebook origin |
|--------|-------|-----------------|
| `config.py` | `PipelineConfig` | CONFIG cell (all knobs + derived paths) |
| `batch.py` | `BatchClient` | Vertex client + batch infra (write/upload/submit/poll/retrieve/record) |
| `data.py` | `DataLoader` | Data Loading cell |
| `features.py` | `FeatureEngineer` | Stage 0 feature engineering + feature selection |
| `media.py` | `MediaContextBuilder` | Multimodal Media Context (local-disk transcript fast path) |
| `stages.py` | `Stage1Discovery`, `TaxonomyConsolidator`, `Stage2Classifier`, `PersonaMapExporter` | Stage 1 / 1.5 / Pathway A / Stage 2 / map / validation |
| `clustering.py` | `PathwayBClusterer`, `LegacyStage3` | Pathway B (alt taxonomy) + optional post-Stage-2 view |
| `pipeline.py` | `PersonaPipeline` | run-order facade |
| `__main__.py` | — | argparse CLI |

Both consolidation pathways write the **identical 5-key `final_taxonomy`**, so
Stage 2 is pathway-agnostic. Stage 2 has a **hard stop**: it refuses to run unless
`taxonomy.json` has `status="APPROVED"`.

## Usage

### Pathway A (LLM consolidation — default)

```bash
python -m persona submit-stage1
# wait for the Vertex job, then:
python -m persona retrieve-stage1
python -m persona consolidate            # one Gemini call -> final_taxonomy
# edit outputs/taxonomy.json, set status="APPROVED"
python -m persona submit-stage2
python -m persona retrieve-stage2
python -m persona export-map
python -m persona validate
```

### Pathway B (UMAP + HDBSCAN clustering — alternative)

```bash
python -m persona submit-stage1
python -m persona retrieve-stage1
python -m persona pathway-b --pathway B_CLUSTER   # clusters the sample -> final_taxonomy
# review/approve taxonomy.json, then submit-stage2 ... as above
```

### Orchestrator (matches notebook `run_pipeline`)

```bash
python -m persona run --mode SAMPLE   # submit Stage 1, pause for approval
python -m persona run --mode FULL     # submit Stage 2 (needs APPROVED taxonomy)
python -m persona run --mode ALL      # both
```

### Global overrides

`--platform`, `--local-dir`, `--pathway`, `--sample-n`, `--target-personas`.

## Programmatic use

```python
from persona import PipelineConfig, PersonaPipeline

pipe = PersonaPipeline(PipelineConfig(platform="instagram", sample_n_users=20000))
pipe.submit_stage1()        # later, fresh process: pipe.retrieve_stage1()
```

## Notes

- Transcripts are read from the local `multimodal_dataset_fixed/` mirror when
  present (no per-post GCS round-trip in the build loop); image `fileData` URIs
  stay `gs://` because Vertex fetches them server-side.
- `user_features` is cached to `outputs/user_features_<platform>_cache.parquet`;
  delete it to force recomputation.
