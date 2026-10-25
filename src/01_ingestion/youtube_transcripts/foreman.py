import pandas as pd
import random
from pathlib import Path

# --- CONFIGURATION ---
CWD = Path(__file__).parent
PARQUET_FILE = CWD / "yt_videos_cleaned_with_transcripts.parquet"
TXT_FILE = CWD / "completed_video_ids.txt"
COLUMN_NAME = "videoId"
TOTAL_WANTED = 3000                          # Adjust this based on your model needs
WORKERS = 5

# 1. Read the raw text file and extract ONLY the video IDs
with open(TXT_FILE, 'r') as f:
    txt_lines = f.readlines()
completed_ids = [line.strip() for line in txt_lines if line.strip()]

print(f"Found {len(completed_ids)} completed videos.")

# 2. Load dataset
df = pd.read_parquet(PARQUET_FILE)
all_ids = df[COLUMN_NAME].dropna().unique().tolist()

# 3. Filter out completed
remaining_ids = [vid for vid in all_ids if vid not in completed_ids]
print(f"Videos left to process: {len(remaining_ids)}")

# 4. Randomly sample to avoid temporal bias
sample_size = min(TOTAL_WANTED, len(remaining_ids))
sampled_ids = random.sample(remaining_ids, sample_size)

# 5. Split into 5 chunks
chunk_size = len(sampled_ids) // WORKERS
for i in range(WORKERS):
    start = i * chunk_size
    end = None if i == WORKERS - 1 else (i + 1) * chunk_size 
    
    chunk = sampled_ids[start:end]
    pd.DataFrame({COLUMN_NAME: chunk}).to_csv(f"worker_{i+1}_tasks.csv", index=False)
    print(f"Created worker_{i+1}_tasks.csv with {len(chunk)} videos.")