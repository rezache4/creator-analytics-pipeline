"""
Generate clean retry queue for the 1037 unprocessed videos.
Prioritizes videos that failed (transient errors, not "no transcript available").
"""

import pandas as pd
from pathlib import Path

data_cleaned = Path("Data_Cleaned")
missing_df = pd.read_csv(data_cleaned / "pooled_missing_transcripts_final_check.csv")

print("=" * 80)
print("GENERATING RETRY QUEUE")
print("=" * 80)

# Sort by priority:
# 1. attempted_failed (transient errors, should retry)
# 2. cloud_done_but_vtt_missing (marked done but no local file, edge case)
# 3. attempted_no_transcript (YouTube has no transcript, low priority but include for completeness)

priority_order = {
    'attempted_failed': 1,
    'cloud_done_but_vtt_missing': 2,
    'attempted_no_transcript': 3
}

missing_df['priority'] = missing_df['final_check_status'].map(priority_order)
retry_queue = missing_df.sort_values('priority')[['videoId', 'final_check_status', 'assigned_to']].reset_index(drop=True)

print(f"\nTotal videos for retry: {len(retry_queue)}")
print(f"  Priority 1 (failed attempts): {len(retry_queue[retry_queue['final_check_status'] == 'attempted_failed'])}")
print(f"  Priority 2 (cloud done, no file): {len(retry_queue[retry_queue['final_check_status'] == 'cloud_done_but_vtt_missing'])}")
print(f"  Priority 3 (no transcript on YouTube): {len(retry_queue[retry_queue['final_check_status'] == 'attempted_no_transcript'])}")

# Save retry queue
retry_queue.to_csv(data_cleaned / "retry_queue_1037_unprocessed.csv", index=False)
print(f"\nRetry queue saved: retry_queue_1037_unprocessed.csv")

# Show sample
print("\nSample (first 10 rows):")
print(retry_queue.head(10).to_string())

print("\n" + "=" * 80)
print("READY FOR RETRY RUN")
print("=" * 80)
print(f"Execute: worker_script.py --queue {data_cleaned / 'retry_queue_1037_unprocessed.csv'}")
