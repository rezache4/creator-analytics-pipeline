import pandas as pd
import time
import os
import random
import glob
import yt_dlp
import argparse
import sys
from pathlib import Path
from yt_dlp.networking.impersonate import ImpersonateTarget

# --- CONFIGURATION ---
COOKIE_FILE = "<DATA_DIR>\\youtube.com_cookies.txt"  
OUTPUT_DIR = "<DATA_DIR>\\transcripts"
COLUMN_NAME = "videoId"
CWD = Path(__file__).parent


def load_retry_queue(queue_csv):
    """Load videos from retry queue CSV file."""
    queue_path = Path(queue_csv)
    if not queue_path.exists():
        print(f"Error: Queue file not found: {queue_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(queue_path)
        if COLUMN_NAME not in df.columns:
            print(f"Error: Queue file missing '{COLUMN_NAME}' column")
            sys.exit(1)
        video_ids = [str(vid).strip() for vid in df[COLUMN_NAME].dropna().tolist() if str(vid).strip()]
        print(f"Loaded {len(video_ids)} videos from {queue_path.name}")
        return video_ids
    except Exception as e:
        print(f"Error loading queue CSV: {e}")
        sys.exit(1)


def check_already_done(video_id, output_dir, force_retry=False):
    """Check if a video was already processed. Returns True if should skip."""
    # Always skip if has .vtt file (successful transcript)
    if glob.glob(os.path.join(output_dir, f"{video_id}.*.vtt")):
        return True, "already_has_transcript"
    
    # Always skip if has .empty marker (no transcript on YouTube)
    if os.path.exists(os.path.join(output_dir, f"{video_id}.empty")):
        return True, "no_transcript_available"
    
    # Skip .failed only if NOT force_retry
    if not force_retry and os.path.exists(os.path.join(output_dir, f"{video_id}.failed")):
        return True, "previously_failed_skipped"
    
    return False, None


def is_transient_download_error(exc):
    """Check if error is transient (network issue, not a permanent failure)."""
    error_text = f"{type(exc).__name__}: {exc}".lower()
    transient_markers = [
        "connection reset",
        "connection aborted",
        "connection closed",
        "remote end closed connection",
        "timed out",
        "timeout",
        "temporary failure",
        "network is unreachable",
        "no route to host",
        "name or service not known",
        "unable to download webpage",
        "unable to download api page",
        "http error 502",
        "http error 503",
        "http error 504",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
    ]
    return any(marker in error_text for marker in transient_markers)


if __name__ == "__main__":
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(
        description="Retry YouTube transcript downloads for specific videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retry 1037 unprocessed videos
  python worker_script.py --queue Data_Cleaned/retry_queue_1037_unprocessed.csv
  
  # Force retry (ignore previous .failed markers)
  python worker_script.py --queue Data_Cleaned/retry_queue_1037_unprocessed.csv --force-retry
        """
    )
    parser.add_argument(
        "--queue",
        type=str,
        required=True,
        help="Path to CSV file with videoId column (required)"
    )
    parser.add_argument(
        "--force-retry",
        action="store_true",
        help="Ignore .failed markers and retry previously failed videos"
    )
    
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    video_ids = load_retry_queue(args.queue)

    if not video_ids:
        print("No videos to process.")
        sys.exit(0)

    print(f"Starting retry extraction for {len(video_ids)} videos...")
    print(f"Force retry: {args.force_retry}")
    print()

    ydl_opts = {
        'skip_download': True,           
        'writesubtitles': True,          
        'writeautomaticsub': True,       
        'subtitleslangs': ['it'],
        'cookiefile': COOKIE_FILE,       
        'impersonate': ImpersonateTarget.from_str('chrome'),
        'quiet': True,
        'no_warnings': True,
        'outtmpl': os.path.join(OUTPUT_DIR, '%(id)s'), 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for count, vid in enumerate(video_ids, start=1):
            
            # Check if already processed
            should_skip, skip_reason = check_already_done(vid, OUTPUT_DIR, force_retry=args.force_retry)
            if should_skip:
                print(f"[{count}/{len(video_ids)}] SKIP: {vid} ({skip_reason})")
                continue

            url = f"https://www.youtube.com/watch?v={vid}"
            try:
                ydl.download([url])
                
                if glob.glob(os.path.join(OUTPUT_DIR, f"{vid}.*.vtt")):
                    print(f"[{count}/{len(video_ids)}] SUCCESS: {vid}")
                else:
                    open(os.path.join(OUTPUT_DIR, f"{vid}.empty"), 'w').close()
                    print(f"[{count}/{len(video_ids)}] NO TRANSCRIPT: {vid}")
                    
            except Exception as e:
                error_msg = str(e).split('\n')[0][:80]
                print(f"[{count}/{len(video_ids)}] FAILED: {vid} | {error_msg}")

                if is_transient_download_error(e):
                    print(f"[{count}/{len(video_ids)}] TRANSIENT ERROR: stopping to retry later")
                    break
                
                # Mark as failed to avoid retry without --force-retry
                open(os.path.join(OUTPUT_DIR, f"{vid}.failed"), 'w').close()
            
            # Delay between requests
            time.sleep(random.uniform(11, 15)) 