"""
Match cloud-done videos with .txt transcript files and re-pool them as successful.
Searches entire workspace for .txt files matching the 2782 cloud_done_but_vtt_missing video IDs.
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

# Workspace paths
workspace = Path(r"<DATA_DIR>")
data_cleaned = workspace / "Data_Cleaned"
pooled_missing_csv = data_cleaned / "pooled_missing_transcripts_final_check.csv"
pooled_success_csv = data_cleaned / "pooled_successful_transcripts.csv"
cloud_match_output = data_cleaned / "cloud_matched_transcripts.csv"

print("=" * 80)
print("STEP 1: Load cloud_done_but_vtt_missing video IDs")
print("=" * 80)

# Load missing transcripts data
missing_df = pd.read_csv(pooled_missing_csv)
print(f"Total missing videos: {len(missing_df)}")

# Filter for cloud_done_but_vtt_missing
cloud_done_videos = missing_df[missing_df['final_check_status'] == 'cloud_done_but_vtt_missing'].copy()
cloud_done_ids = set(cloud_done_videos['videoId'].values)
print(f"Cloud_done_but_vtt_missing videos: {len(cloud_done_ids)}")
print(f"Sample IDs: {list(cloud_done_ids)[:5]}")

print("\n" + "=" * 80)
print("STEP 2: Search workspace for .txt files matching cloud_done video IDs")
print("=" * 80)

# Build a map of video IDs to .txt file paths found in workspace
txt_matches = defaultdict(list)

# Search entire workspace recursively for .txt files
print(f"Searching workspace: {workspace}")
for root, dirs, files in os.walk(workspace):
    for file in files:
        if file.endswith('.txt'):
            # Extract video ID from filename (format: VIDEOID.txt or VIDEOID.something.txt)
            basename = file.replace('.txt', '')
            # Try to match against known cloud_done IDs
            if basename in cloud_done_ids:
                full_path = Path(root) / file
                relative_path = full_path.relative_to(workspace)
                txt_matches[basename].append(str(relative_path))
                print(f"  ✓ Found: {basename} → {relative_path}")

print(f"\nTotal .txt files matched to cloud_done videos: {len(txt_matches)}")
print(f"Total matched video IDs: {sum(1 for vid_ids in txt_matches.values() if vid_ids)}")

print("\n" + "=" * 80)
print("STEP 3: Generate cloud matched transcripts CSV")
print("=" * 80)

# Create output for matched cloud transcripts
cloud_matched_list = []
for video_id, txt_paths in txt_matches.items():
    cloud_matched_list.append({
        'videoId': video_id,
        'transcript_count': len(txt_paths),
        'transcript_paths': '; '.join(txt_paths),
        'source': 'cloud'
    })

cloud_matched_df = pd.DataFrame(cloud_matched_list)
cloud_matched_df.to_csv(cloud_match_output, index=False)
print(f"Cloud matched transcripts saved: {cloud_match_output}")
print(f"Matched videos count: {len(cloud_matched_df)}")
if len(cloud_matched_df) > 0:
    print("\nSample matched entries:")
    print(cloud_matched_df.head(10))

print("\n" + "=" * 80)
print("STEP 4: Update pooled_successful_transcripts with cloud matches")
print("=" * 80)

# Load existing successful transcripts
success_df = pd.read_csv(pooled_success_csv)
print(f"Current successful transcripts: {len(success_df)}")

# Add cloud-matched videos to success pool
cloud_matched_df_with_source = cloud_matched_df.copy()
combined_success = pd.concat([success_df, cloud_matched_df_with_source], ignore_index=True)
combined_success = combined_success.drop_duplicates(subset=['videoId'], keep='first')

combined_success.to_csv(pooled_success_csv, index=False)
print(f"Updated successful transcripts CSV: {len(combined_success)} videos")
print(f"New videos added from cloud matches: {len(cloud_matched_df)}")

print("\n" + "=" * 80)
print("STEP 5: Update pooled_missing_transcripts (remove cloud matches)")
print("=" * 80)

# Remove cloud-matched videos from missing pool
matched_video_ids = set(cloud_matched_df['videoId'].values)
remaining_missing = missing_df[~missing_df['videoId'].isin(matched_video_ids)].copy()

remaining_missing.to_csv(pooled_missing_csv, index=False)
print(f"Updated missing transcripts CSV: {len(remaining_missing)} videos (was {len(missing_df)})")
print(f"Videos removed (now counted as successful): {len(missing_df) - len(remaining_missing)}")

# Breakdown of remaining missing
if len(remaining_missing) > 0:
    print("\nRemaining missing videos by status:")
    status_counts = remaining_missing['final_check_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Cloud-done videos searched: {len(cloud_done_ids)}")
print(f"Cloud-done videos with .txt files found: {len(cloud_matched_df)}")
print(f"Successful transcripts (updated): {len(combined_success)}")
print(f"Missing transcripts (updated): {len(remaining_missing)}")
print(f"Total accounted for: {len(combined_success) + len(remaining_missing)} (should be {len(missing_df) + len(success_df)})")

if len(remaining_missing) > 0:
    print(f"\nTrue unprocessed videos remaining: {len(remaining_missing)}")
    print("Categories:")
    for status, count in remaining_missing['final_check_status'].value_counts().items():
        print(f"  - {status}: {count}")
