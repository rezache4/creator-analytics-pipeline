# JSONL & Cloud Batch Processing

We structure our local job bundles as **JSONL (JSON Lines)** files. Each line in a JSONL file is a self-contained, valid JSON object representing a single chunk of a transcript ready for inference.

## Is JSONL Suitable for Feeding LLMs?

Yes, JSONL is the industry standard for bulk offline data ingestion and batch LLM processing for several reasons:

### 1. Memory and Storage Efficiency
Unlike a standard JSON array (`[...]`), which requires parsing the entire file into memory at once, a JSONL file is processed line-by-line.
- **For local assembly**: We can append new jobs to the file sequentially without loading millions of rows into RAM.
- **For cloud workers**: The worker reads one line at a time, processes the request, writes the response to an output file, and discards the memory.

### 2. Native Google Cloud Vertex AI Integration
Vertex AI's Batch Prediction API natively expects GCS input files to be in JSONL format. 
Each line contains a standard payload matching the model's expected API schema. For example:
```json
{
  "contents": {
    "role": "user",
    "parts": [
      {"text": "Summarize this: [transcript chunk here...]"}
    ]
  }
}
```
Vertex AI processes these lines in parallel across managed infrastructure and outputs a corresponding JSONL file containing the responses.

### 3. Metadata Retention
JSONL allows us to bind metadata directly to the text payload. A single row in our job bundle contains:
```json
{
  "job_id": "eNU5FcmEwTc_000",
  "video_id": "eNU5FcmEwTc",
  "chunk_idx": 0,
  "chunk_total": 1,
  "chunk_text": "...",
  "estimated_input_tokens": 1856,
  "language": "it",
  "source_hash": "...",
  "prompt_version": "v1"
}
```
By keeping metadata in the same record, we avoid the risk of losing correlation between the LLM output and the source video. When the inference is completed, the output database uses the `job_id` to join the generated summaries or embeddings back to the main video records.
