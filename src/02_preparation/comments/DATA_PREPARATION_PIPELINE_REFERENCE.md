# Data Preparation Pipeline Reference
**Project**: Show Reel Media Group — Cross-Platform Community Evolution  
**Step**: 2 of 6 (Data Cleaning & Preparation)  
**Notebook**: `Data_Preparation_Pipeline.ipynb`  
**Dataset Size**: 987,115 raw comments (Instagram: 573,377 | Facebook: 394,084 | TikTok: 19,654)  
**Data collected up to**: March 2026

---

## What the Pipeline Does

Takes raw, platform-specific comment records from Instagram, Facebook, and TikTok and outputs three clean, paradigm-optimized files for downstream analytical use.

---

## Input Format

Each platform uses different field names. The pipeline normalizes these automatically.

| Field Meaning     | Instagram/Facebook key | TikTok key  | Normalized to            |
|-------------------|------------------------|-------------|--------------------------|
| Author identifier | `from_id`              | `uid`       | `author_id`              |
| Media identifier  | `media_id`             | `video_id`  | `media_id`               |
| Reply reference   | `parent_id`            | `reply_id`  | `reply_to_comment_id`    |
| Comment text      | `text`                 | `text`      | `comment_text`           |
| Timestamp         | `timestamp`            | `timestamp` | `timestamp`              |

### Minimal Valid Record Examples

**Instagram / Facebook:**
```json
{
  "from_id": "user_123",
  "media_id": "post_456",
  "text": "Great post! ❤️🔥",
  "parent_id": null,
  "timestamp": "2026-03-01T10:00:00Z"
}
```

**TikTok:**
```json
{
  "uid": "user_789",
  "video_id": "video_101",
  "text": "This is fire! 😂🎉",
  "reply_id": "comment_xyz",
  "timestamp": "2026-03-03T08:00:00Z"
}
```

---

## Output Files

All outputs are written to `./output/` (or `./test_output/` during testing).  
Each format is written as **one file per platform**, e.g. `comments_llm_instagram.jsonl`, `comments_ml_facebook.parquet`, `comments_gml_tiktok.parquet`.

### 1. `comments_llm_{platform}.jsonl`
**Purpose**: LLM token-efficient input. Retains raw unstripped text.  
**Format**: JSONL (one JSON object per line), UTF-8 encoded.

```json
{
  "comment_id": "instagram_user_123_0_0",
  "text": "Great post! ❤️🔥",
  "author_id": "user_123",
  "platform": "instagram",
  "timestamp": "2026-03-01T10:00:00Z"
}
```

**Use for**: Sentiment analysis, LLM fine-tuning, embedding generation, summarization.

---

### 2. `comments_ml_{platform}.parquet`
**Purpose**: Columnar numerical feature matrix for tabular ML frameworks (XGBoost, LightGBM, scikit-learn).  
**Format**: Parquet with Snappy compression. No nested arrays or embedded dicts.

| Column | Type | Description |
|---|---|---|
| `comment_id` | str | Unique ID: `{platform}_{author_id}_{global_idx}_{batch_idx}` |
| `author_id` | str | Normalized author identifier |
| `media_id` | str | Normalized media/post/video identifier |
| `platform` | str | `instagram`, `facebook`, or `tiktok` |
| `text_length` | int | Total character count of raw text |
| `word_count` | int | Word count (after emoji removal) |
| `emoji_count` | int | Total emoji instances (ZWJ-safe extraction) |
| `unique_emoji_count` | float | Count of distinct emojis |
| `emoji_entropy` | float | Shannon Entropy of emoji distribution (0 = repetitive, high = diverse) |
| `emoji_variety_ratio` | float | `unique_emoji_count / emoji_count` |
| `emoji_per_word_ratio` | float | `emoji_count / word_count` |
| `url_count` | int | Number of URLs detected |
| `mention_count` | int | Number of @mentions |
| `hashtag_count` | int | Number of #hashtags |
| `exclamation_count` | int | Number of `!` characters |
| `question_count` | int | Number of `?` characters |
| `avg_word_length` | float | Mean character length of words |
| `has_numbers` | int | Binary: 1 if any digit present |
| `has_links` | int | Binary: 1 if any URL present |
| `timestamp` | str | ISO 8601 timestamp string |

**Use for**: Engagement prediction, community classification, virality scoring, anomaly detection.

---

### 3. `comments_gml_{platform}.parquet`
**Purpose**: Directed heterogeneous edge list for Graph Neural Networks (PyTorch Geometric, DGL).  
**Format**: Parquet with Snappy compression. No text payloads.

| Column | Type | Description |
|---|---|---|
| `comment_id` | str | Unique comment node identifier |
| `author_id` | str | User node identifier |
| `media_id` | str | Media/post node identifier |
| `reply_to_comment_id` | str or None | Target comment node for reply edge (null if top-level) |
| `platform` | str | Platform identifier |
| `timestamp` | str | ISO 8601 timestamp string |

**Directed Edge Topology:**
```
(User) --[AUTHORED_ON]--> (Media)
(User) --[WROTE]---------> (Comment)
(Comment) --[REPLIED_TO]--> (Comment)   # only when reply_to_comment_id is not null
```

**Use for**: Community graph construction, influence propagation, reply chain modeling, GNN training.

---

## Core Engineering Decisions

### Emoji Extraction (ZWJ-Safe)
All emoji extraction uses the `emoji` library's native method to prevent corruption of composite sequences (skin tones, gender-neutral glyphs, coupled characters):
```python
# CORRECT — used in this pipeline
[e['emoji'] for e in emoji.emoji_list(text)]

# FORBIDDEN — corrupts ZWJ sequences
[c for c in text if c in emoji_set]
```

### Emoji Taxonomy
Emojis are pre-mapped to four semantic categories for downstream categorical features:

| Category | Example Emojis |
|---|---|
| `love` | ❤️ 💕 😍 🥰 👍 💯 |
| `celebration` | 🎉 🔥 ✨ 👏 🌟 🥳 |
| `humor` | 😂 🤣 😆 🤪 💀 😜 |
| `inquiry` | 🤔 ❓ 🧐 😕 💭 🙄 |

### Regex Pre-compilation
All pattern matching uses pre-compiled regex objects to achieve O(n) linear scaling across 10⁶+ rows. Patterns compiled once at module load, reused per record.

### Error Handling
- Individual record failures do not halt the batch job.
- Errors are logged via Python's `logging` module (not `print`).
- Failed records increment `pipeline.error_count` and are stored in `pipeline.errors` for audit.
- Records with missing `author_id` or `media_id` are silently skipped.

---

## Dependencies

```
emoji
numpy
pandas
pyarrow
scipy
```

Install:
```bash
pip install emoji numpy pandas pyarrow scipy
```

---

## Running the Pipeline

### Test Mode (mock data)
Run Section 6 of the notebook. Outputs go to `./test_output/`.

### Production Mode
Replace mock records with your actual data loader, pass real records to `pipeline.process_batch()`, and set `output_dir` to your target path:

```python
pipeline = UnifiedPipeline(output_dir=Path("/data/processed"))

for platform, records in [("instagram", ig_records), ("facebook", fb_records), ("tiktok", tk_records)]:
    ml, gml, llm = pipeline.process_batch(records, platform)
    pipeline.export_llm_jsonl(llm, platform)   # → comments_llm_{platform}.jsonl
    pipeline.export_ml_parquet(ml, platform)   # → comments_ml_{platform}.parquet
    pipeline.export_gml_parquet(gml, platform) # → comments_gml_{platform}.parquet
```

---

## What Downstream Steps Should Expect

| Downstream Step | Reads From | Key Assumption |
|---|---|---|
| LLM Sentiment / Embedding | `comments_llm_{platform}.jsonl` | Text is raw and unstripped; no cleaning applied |
| Tabular ML Training | `comments_ml_{platform}.parquet` | All columns are scalar numerics or strings; no nested types |
| GNN Graph Construction | `comments_gml_{platform}.parquet` | `reply_to_comment_id` is null for root comments; build edge index from non-null rows only |
