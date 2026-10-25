#!/usr/bin/env python3
"""
fetch_thumbnails.py

Reads thumbnail URLs from the metadata parquet, downloads each image from
YouTube CDN, and uploads it directly to GCS. Nothing is written to local disk.
Already-uploaded files are skipped so the script is safe to re-run.

Usage (from repo root):
    python thumbnail_cv/fetch_thumbnails.py

Optional env vars:
    META_PARQUET    local parquet path  (default: clean_data/yt_videos_with_local_transcripts.parquet)
    GCS_BUCKET      target bucket       (default: socialmediaanalyticsproject)
    GCS_PREFIX      GCS key prefix      (default: thumbnails/)
    THUMB_URL_COL   URL column name     (default: thumbnail_high_url)
    WORKERS         parallel threads    (default: 8)
"""
import logging
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

META_PARQUET  = os.environ.get("META_PARQUET",  "clean_data/yt_videos_with_local_transcripts.parquet")
GCS_BUCKET    = os.environ.get("GCS_BUCKET",    "socialmediaanalyticsproject")
GCS_PREFIX    = os.environ.get("GCS_PREFIX",    "thumbnails/")
THUMB_URL_COL = os.environ.get("THUMB_URL_COL", "thumbnail_high_url")
WORKERS       = int(os.environ.get("WORKERS",   "8"))

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_and_upload(bucket, video_id: str, url: str) -> str:
    blob = bucket.blob(f"{GCS_PREFIX}{video_id}.jpg")
    if blob.exists():
        return "skip"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    blob.upload_from_string(data, content_type="image/jpeg")
    return "ok"


def main() -> None:
    log.info("Reading %s", META_PARQUET)
    df = pd.read_parquet(META_PARQUET, columns=["videoId", "publishedAt", THUMB_URL_COL])
    df = df.dropna(subset=[THUMB_URL_COL])
    df = df[pd.to_datetime(df["publishedAt"]).dt.year >= 2022].reset_index(drop=True)
    log.info("%d videos published from 2022 onward with thumbnail URLs", len(df))

    gcs    = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)

    ok = skipped = failed = 0
    total = len(df)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_fetch_and_upload, bucket, str(row["videoId"]), str(row[THUMB_URL_COL])): str(row["videoId"])
            for _, row in df.iterrows()
        }
        for i, future in enumerate(as_completed(futures), 1):
            video_id = futures[future]
            try:
                result = future.result()
                if result == "skip":
                    skipped += 1
                else:
                    ok += 1
            except urllib.error.HTTPError as e:
                log.warning("HTTP %d — %s (%s)", e.code, video_id, e.reason)
                failed += 1
            except Exception as e:
                log.warning("FAIL — %s: %s", video_id, e)
                failed += 1

            if i % 200 == 0 or i == total:
                log.info(
                    "  %d / %d  (uploaded=%d  skipped=%d  failed=%d)",
                    i, total, ok, skipped, failed,
                )

    log.info("Done. uploaded=%d  skipped=%d  failed=%d", ok, skipped, failed)
    if failed:
        log.warning("%d failures — re-run to retry (already-uploaded files are skipped)", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
