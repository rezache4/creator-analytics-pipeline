# Hashing & Content Tracking

Our job assembly script generates a stable `source_hash` for each cleaned transcript using the SHA-256 algorithm.

## How the Hash is Generated

1. The raw transcript text has its timestamps, tags, and formatting stripped.
2. The remaining text is lowercased, and all consecutive whitespaces are collapsed into single spaces.
3. We hash this normalized string to produce a unique 64-character hexadecimal representation.

```python
def normalize_for_hash(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.lower()).strip()

def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

## The Role of Content Hashes

Even when deduplication is disabled, content hashes are highly valuable for:

### 1. Verification and Change Tracking (Auditing)
If the upstream video database or local subtitles are updated, we can re-generate the hashes. By comparing the new hash with the old hash, we can instantly tell if the actual content of the transcript changed, without needing to do complex line-by-line diffs.

### 2. Idempotency (Preventing Duplicates)
In the database (BigQuery or Parquet), the hash is used as an integrity check. If we run a batch worker and it outputs results for a video, we write it alongside the `source_hash`. If we update the prompt or rerun the job, the database can check if a result for that specific `source_hash` and `prompt_version` already exists to prevent duplicate writes or redundant billing.

### 3. Optional Deduplication (Skipped by default)
If deduplication is turned on, the script keeps a running set of seen hashes. If another video has the exact same content (e.g., an identical intro clip or silent video containing no spoken audio), it is marked as a duplicate and skipped. This saves cost, but is disabled by default to ensure we preserve a record for every single video row.
