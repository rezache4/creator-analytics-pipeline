#!/usr/bin/env python3
"""
merge_thumbnail_features.py — local post-processing step.

Run this on your laptop after the Cloud Run Job finishes to consolidate
the per-task parquet shards into a single thumbnail_features.parquet.

Usage:
    python Data_Cleaned/merge_thumbnail_features.py

Optional env vars:
    GCS_BUCKET      (default "socialmediaanalyticsproject")
    OUTPUT_PREFIX   (default "output/thumbnail_features/")
    LOCAL_OUT       (default "thumbnail_features.parquet")
"""
import os
import sys

import pandas as pd
from google.cloud import storage

BUCKET     = os.environ.get("GCS_BUCKET", "socialmediaanalyticsproject")
OUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "output/thumbnail_features/")
LOCAL_OUT  = os.environ.get("LOCAL_OUT", "thumbnail_features.parquet")


def main() -> None:
    gcs    = storage.Client()
    bucket = gcs.bucket(BUCKET)

    blobs = sorted(
        b.name
        for b in bucket.list_blobs(prefix=OUT_PREFIX)
        if b.name.endswith(".parquet")
    )
    if not blobs:
        print(f"No parquet shards found under gs://{BUCKET}/{OUT_PREFIX}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(blobs)} shards — downloading and merging...")
    frames = []
    for name in blobs:
        data = bucket.blob(name).download_as_bytes()
        import io
        frames.append(pd.read_parquet(io.BytesIO(data)))
        print(f"  {name}  ({len(frames[-1])} rows)")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(LOCAL_OUT, index=False)
    print(f"\nMerged {len(blobs)} shards → {len(merged)} rows → {LOCAL_OUT}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
