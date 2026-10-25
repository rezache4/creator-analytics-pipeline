#!/usr/bin/env python3
"""
thumbnail_features_supplementary.py

Supplementary numpy pass: computes 8 colour/composition features that were
not included in the original Colab extraction run, then merges them into
the existing thumbnail_features.parquet.

New features:
  contrast              std dev of LAB L channel
  warm_cool_ratio       red+yellow pixel mass / blue+green pixel mass
  dominant_hue_entropy  Shannon entropy of the hue histogram (256 bins)
  edge_density          fraction of Canny edge pixels
  visual_complexity     std dev of grayscale pixel intensities (normalised)
  rule_of_thirds_score  edge mass within ±10% of rule-of-thirds lines
  left_right_asymmetry  |mean brightness left half - right half|
  top_bottom_asymmetry  |mean brightness top half - bottom half|

Usage (from repo root, creator-analytics-kernel env):
    python thumbnail_cv/thumbnail_features_supplementary.py

Output: clean_data/thumbnail_features.parquet updated in place.
"""

import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GCS_BUCKET    = "socialmediaanalyticsproject"
THUMB_PREFIX  = "thumbnails/"
LOCAL_PARQUET = "clean_data/thumbnail_features.parquet"
WORKERS       = 8
LOG_EVERY     = 200


# ── Feature functions ─────────────────────────────────────────────────────────

def _warm_cool_ratio(bgr: np.ndarray) -> float:
    r = bgr[:, :, 2].astype(float)
    g = bgr[:, :, 1].astype(float)
    b = bgr[:, :, 0].astype(float)
    warm = r + np.maximum(r - b, 0)         # red + orange-yellow contribution
    cool = b + g
    denom = cool.sum()
    return float(warm.sum() / denom) if denom > 0 else 0.0


def _hue_entropy(hsv: np.ndarray) -> float:
    h = hsv[:, :, 0].ravel()
    counts, _ = np.histogram(h, bins=256, range=(0, 180))
    p = counts / (counts.sum() + 1e-9)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _rule_of_thirds(edges: np.ndarray, h: int, w: int) -> float:
    total = edges.sum()
    if total == 0:
        return 0.0
    margin_h = max(1, int(h * 0.10))
    margin_w = max(1, int(w * 0.10))
    thirds_h = [h // 3, 2 * h // 3]
    thirds_w = [w // 3, 2 * w // 3]
    mass = 0
    for y in thirds_h:
        mass += edges[max(0, y - margin_h):y + margin_h, :].sum()
    for x in thirds_w:
        mass += edges[:, max(0, x - margin_w):x + margin_w].sum()
    return float(mass / total)


def compute_features(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    return dict(
        contrast             = float(lab[:, :, 0].std()),
        warm_cool_ratio      = _warm_cool_ratio(bgr),
        dominant_hue_entropy = _hue_entropy(hsv),
        edge_density         = float(edges.mean() / 255.0),
        visual_complexity    = float(gray.std() / 255.0),
        rule_of_thirds_score = _rule_of_thirds(edges, h, w),
        left_right_asymmetry = float(abs(gray[:, :w//2].mean() - gray[:, w//2:].mean()) / 255.0),
        top_bottom_asymmetry = float(abs(gray[:h//2].mean() - gray[h//2:].mean()) / 255.0),
    )


# ── GCS fetch + compute ───────────────────────────────────────────────────────

def process_blob(bucket, video_id: str) -> dict | None:
    try:
        data = bucket.blob(f"{THUMB_PREFIX}{video_id}.jpg").download_as_bytes()
        arr  = np.frombuffer(data, np.uint8)
        bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return {"video_id": video_id, **compute_features(bgr)}
    except Exception as e:
        log.warning("SKIP %s: %s", video_id, e)
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading %s", LOCAL_PARQUET)
    df = pd.read_parquet(LOCAL_PARQUET)

    new_cols = ["contrast", "warm_cool_ratio", "dominant_hue_entropy",
                "edge_density", "visual_complexity", "rule_of_thirds_score",
                "left_right_asymmetry", "top_bottom_asymmetry"]

    already_done = set()
    if new_cols[0] in df.columns:
        already_done = set(df.loc[df[new_cols[0]].notna(), "video_id"].tolist())
        log.info("%d rows already have supplementary features — skipping", len(already_done))

    video_ids = [v for v in df["video_id"].tolist() if v not in already_done]
    log.info("%d thumbnails to process", len(video_ids))

    if not video_ids:
        log.info("Nothing to do.")
        return

    gcs    = storage.Client()
    bucket = gcs.bucket(GCS_BUCKET)

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_blob, bucket, vid): vid for vid in video_ids}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row:
                results.append(row)
            if i % LOG_EVERY == 0 or i == len(video_ids):
                log.info("  %d / %d", i, len(video_ids))

    log.info("Merging %d new rows into parquet", len(results))
    supp = pd.DataFrame(results)
    df   = df.merge(supp, on="video_id", how="left")
    df.to_parquet(LOCAL_PARQUET, index=False)
    log.info("Saved → %s", LOCAL_PARQUET)


if __name__ == "__main__":
    main()
