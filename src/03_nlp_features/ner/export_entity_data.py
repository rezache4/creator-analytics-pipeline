"""Export entity resonance data in structured nested format.

Converts flat entity_resonance.parquet into per-video nested JSON objects:
  {
    "video_id": "xyz",
    "transcript_entities": [
      {"text": "entity_name", "label": "PER", "count": 5},
      ...
    ],
    "comment_entities": [
      {"text": "entity_name", "label": "ORG", "count": 10},
      ...
    ],
    "resonant_entities": [
      {
        "text": "entity_name",
        "label": "LOC",
        "transcript_count": 3,
        "comment_count": 7
      },
      ...
    ]
  }

Output files:
  - entity_data_per_video.jsonl  (one JSON object per line, one video per line)
  - entity_data_per_video.json   (single JSON array of all videos)
  - entity_data_per_video.csv    (flattened CSV for spreadsheet analysis)

Usage (from Data_Cleaned directory):
    python export_entity_data.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load the resonance data
print("Loading entity_resonance.parquet ...")
df_res = pd.read_parquet("entity_resonance.parquet")

# Group by video_id
print("Structuring data per video ...")
videos_data = defaultdict(lambda: {
    "transcript_entities": [],
    "comment_entities": [],
    "resonant_entities": [],
})

for _, row in df_res.iterrows():
    vid = row["video_id"]
    ent = {
        "text": row["entity_text"],
        "label": row["entity_label"],
    }

    # Transcript-only entities
    if row["in_transcript"] and not row["in_comments"]:
        ent["count"] = row["transcript_count"]
        videos_data[vid]["transcript_entities"].append(ent)

    # Comment-only entities
    elif row["in_comments"] and not row["in_transcript"]:
        ent["count"] = row["comment_count"]
        videos_data[vid]["comment_entities"].append(ent)

    # Resonant entities (both sources)
    elif row["in_transcript"] and row["in_comments"]:
        videos_data[vid]["resonant_entities"].append({
            "text": ent["text"],
            "label": ent["label"],
            "transcript_count": int(row["transcript_count"]),
            "comment_count": int(row["comment_count"]),
        })

# Sort entities within each video by count descending
for vid in videos_data:
    videos_data[vid]["transcript_entities"] = sorted(
        videos_data[vid]["transcript_entities"],
        key=lambda x: x.get("count", 0),
        reverse=True,
    )
    videos_data[vid]["comment_entities"] = sorted(
        videos_data[vid]["comment_entities"],
        key=lambda x: x.get("count", 0),
        reverse=True,
    )
    videos_data[vid]["resonant_entities"] = sorted(
        videos_data[vid]["resonant_entities"],
        key=lambda x: x["comment_count"],
        reverse=True,
    )

# ── 1. JSONL export (one video per line) ──────────────────────────────────────

print("Writing entity_data_per_video.jsonl ...")
with open("entity_data_per_video.jsonl", "w", encoding="utf-8") as f:
    for vid, data in sorted(videos_data.items()):
        obj = {
            "video_id": vid,
            **data,
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"  {len(videos_data):,} videos written")

# ── 2. JSON export (full array) ────────────────────────────────────────────────

print("Writing entity_data_per_video.json ...")
all_videos = [
    {"video_id": vid, **data}
    for vid, data in sorted(videos_data.items())
]
with open("entity_data_per_video.json", "w", encoding="utf-8") as f:
    json.dump(all_videos, f, ensure_ascii=False, indent=2)

print(f"  {len(all_videos):,} videos written")

# ── 3. CSV export (flattened for spreadsheet analysis) ────────────────────────

print("Writing entity_data_per_video.csv ...")
records = []
for vid, data in sorted(videos_data.items()):
    # One row per resonant entity (the most useful for analysis)
    for ent in data["resonant_entities"]:
        records.append({
            "video_id": vid,
            "entity_text": ent["text"],
            "entity_label": ent["label"],
            "transcript_count": ent["transcript_count"],
            "comment_count": ent["comment_count"],
            "entity_type": "resonant",
        })
    # Also include transcript-only and comment-only for completeness
    for ent in data["transcript_entities"]:
        records.append({
            "video_id": vid,
            "entity_text": ent["text"],
            "entity_label": ent["label"],
            "transcript_count": ent.get("count", 0),
            "comment_count": 0,
            "entity_type": "transcript_only",
        })
    for ent in data["comment_entities"]:
        records.append({
            "video_id": vid,
            "entity_text": ent["text"],
            "entity_label": ent["label"],
            "transcript_count": 0,
            "comment_count": ent.get("count", 0),
            "entity_type": "comment_only",
        })

df_export = pd.DataFrame(records)
df_export.to_csv("entity_data_per_video.csv", index=False)

print(f"  {len(df_export):,} rows written")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)
print(f"\nFiles created:")
print(f"  ✓ entity_data_per_video.jsonl  ({len(videos_data):,} videos)")
print(f"  ✓ entity_data_per_video.json   ({len(all_videos):,} videos)")
print(f"  ✓ entity_data_per_video.csv    ({len(df_export):,} entity rows)")

print(f"\nData structure per video (JSONL/JSON):")
print(json.dumps(all_videos[0] if all_videos else {}, indent=2, ensure_ascii=False)[:400] + "...")

print(f"\nUse cases:")
print(f"  - JSONL: streaming/pipeline processing, line-by-line parsing")
print(f"  - JSON:  full dataset analysis, nested structure")
print(f"  - CSV:   spreadsheet analysis, filtering by entity_type or label")
