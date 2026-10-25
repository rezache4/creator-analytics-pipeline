# Semantic Embedding

We generate chapter-level embedding vectors for all transcripts locally using `intfloat/multilingual-e5-base` via `sentence-transformers`. No API calls or GCP quota are required.

## Model Choice

| Candidate | Dims | Notes |
|-----------|------|-------|
| `intfloat/multilingual-e5-base` | 768 | **Chosen** — local CPU/GPU, strong Italian support, no cost |
| Vertex AI `text-multilingual-embedding-002` | 768 | Original target; rejected due to GCP quota constraints on free tier |

The `e5` model expects a `"passage: "` prefix prepended to each document text at inference time.

## Processing Strategy

Short and long videos are handled differently because their transcripts differ in length and structure.

### Short Videos (`is_short=True`)
1. Chunk the transcript into overlapping 100-word base chunks.
2. Embed all base chunks in a single batch.
3. **Mean-pool** all chunk vectors into one chapter-0 vector per video.

### Long Videos (`is_short=False`)
1. Chunk the transcript into overlapping 100-word base chunks.
2. Embed all base chunks in batches of 64.
3. **Detect chapter breaks**: a new chapter starts where cosine similarity between consecutive chunk embeddings drops below `SIMILARITY_THRESHOLD = 0.75`.
4. **Mean-pool** the base-chunk embeddings within each detected chapter into a single chapter vector.

The same 100-word chunking is used for both formats to avoid oversized single-text payloads; the difference is only in how the chunk vectors are aggregated.

## Output Schema

`dense_vectors_semantic.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | YouTube video ID |
| `is_short` | bool | Short video flag |
| `chapter_index` | int | 0-based chapter index (always 0 for shorts) |
| `embedding_vector` | list[float] | 768-dimensional unit vector |

Long videos yield multiple rows (one per detected chapter). To aggregate back to video level for downstream modelling (e.g. SAE regression), max-pool or mean-pool the chapter vectors after sparse encoding.

## Checkpointing

The script appends completed video results to `.embed_checkpoint.jsonl` after every 100-video group. Pass `--resume` to continue an interrupted run without re-embedding already-processed videos.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_CHUNK_WORDS` | 100 | Words per base chunk |
| `BASE_CHUNK_OVERLAP` | 10 | Word overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | 0.75 | Cosine-sim floor for chapter continuity |
| `GROUP_SIZE` | 100 | Videos per checkpoint flush |
| `BATCH_SIZE` | 64 | Texts per `sentence-transformers` encode call |

## Usage

```bash
# From the Data_Cleaned directory
python embed_transcripts_semantic.py                    # full run
python embed_transcripts_semantic.py --resume           # continue from checkpoint
python embed_transcripts_semantic.py --dry-run          # validate input, no model loaded
python embed_transcripts_semantic.py --threshold 0.70   # looser chapter detection
```
