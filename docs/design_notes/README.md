# LLM Ingestion Pipeline Wiki

Welcome to the Creator Analytics Wiki. This directory documents the design, data structures, and reasoning behind our transcript preprocessing, embedding, entity analysis, and GCP batch ingestion pipeline.

## Wiki Contents

- [Deterministic Preprocessing](local_deterministic_compute.md): Why we clean and chunk transcripts locally and what steps are executed.
- [JSONL & Cloud Batch Processing](jsonl_batch_format.md): Why JSONL is chosen, how it aligns with Vertex AI Batch APIs, and how chunk metadata is preserved.
- [Hashing & Content Tracking](hashing_and_deduplication.md): The role of SHA-256 content hashes, idempotency, and audit checks.
- [Semantic Embedding](semantic_embedding.md): Local chapter-level embedding with `intfloat/multilingual-e5-base`, chapter-break detection strategy, and output schema.
- [Entity Extraction & Resonance](entity_extraction.md): NER with spaCy over transcripts and comments, and the resonance analysis that links what a video discusses to what its audience reacts to.
- [Glossary](glossary.md): Definitions for recurring terms — soft membership vector, hook, mean-pooling, cosine similarity, git staging, idempotency, tokenizer fertility.

## Pipeline Architecture Overview

```
[Local Parquet Data]
       │
       ▼ (Deterministic Compute)
[Text Clean -> Token Chunk -> SHA256 Hash]
       │
       ├──▶ (GCS Upload — Vertex AI path)
       │    [JSONL Job Bundle + Manifest]
       │           │
       │           ▼
       │    [GCP Vertex AI Batch Inference]
       │
       ├──▶ (Local Embedding)
       │    [intfloat/multilingual-e5-base]
       │    [Shorts: 1 vector | Long: chapter detection + mean-pool]
       │           │
       │           ▼
       │    [dense_vectors_semantic.parquet]
       │
       └──▶ (Entity Extraction)
            [spaCy it_core_news_lg — transcripts + comments]
                   │
                   ▼
            [entity_resonance.parquet -> JSONL / JSON / CSV export]
```
