# Local Deterministic Compute

In our transcript processing pipeline, we perform all **deterministic compute** locally on the workstation before uploading anything to the cloud.

## What is Deterministic Compute?

An operation is "deterministic" if it always yields the exact same output for a given input. Unlike LLM generation (which has probabilistic outputs based on temperature and top-p parameters), text cleaning, token counting, and chunk splitting are 100% rule-based.

## Operations Performed Locally

1. **Junk & Subtitle Metadata Removal**:
   - Stripping the `WEBVTT` protocol header.
   - Removing SRT/VTT time-range lines (e.g., `00:01:20.000 --> 00:01:24.000`).
   - Removing standalone timestamps.
   - Stripping SRT index numbers.
   - Stripping HTML/VTT style tags (e.g., `<c>`).
   
2. **Whitespace Normalization**:
   - Collapsing consecutive spaces, tabs, and line breaks into single spaces.
   
3. **Token-Aware Chunking**:
   - Splitting text based on token counts using a local tokenizer (`tiktoken` running the `cl100k_base` vocabulary).
   - Enforcing chunk sizes (e.g., maximum 1,800 tokens per chunk) with a rolling overlap (e.g., 150 tokens) to preserve semantic context across chunk boundaries.

## Why Run This Locally?

1. **Cost Minimization**:
   - Stripping timestamps and formatting characters reduces the overall character size of the transcript. Shipping only prompt-ready text means we do not pay GCP to ingest formatting noise.
   - Splitting transcripts into chunks locally ensures we only submit payloads that fit the model's target processing limits, preventing out-of-memory errors on Vertex AI.
   
2. **Operational Simplicity**:
   - Local preprocessing avoids having to write, debug, and run custom python data pipelines (like Dataflow or Apache Spark) inside GCP. We utilize the local CPU power of the workstation for simple, fast string manipulations.
   
3. **Pre-Flight Validation**:
   - It allows us to generate a complete manifest of the jobs *before* uploading them. We know exactly how many jobs we are about to run, how many tokens we will consume, and which videos were rejected (e.g., due to being empty).
