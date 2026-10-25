from pathlib import Path
from collections import defaultdict
import pandas as pd


DATA_CLEANED_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DATA_CLEANED_DIR.parent
PARQUET_FILE = DATA_CLEANED_DIR / "yt_videos_cleaned_with_transcripts.parquet"
COMPLETED_CLOUD_FILE = DATA_CLEANED_DIR / "completed_video_ids.txt"
COLLEAGUE_TASK_FILES = [
    WORKSPACE_ROOT / "Ali.csv",
    WORKSPACE_ROOT / "Meenal.csv",
    WORKSPACE_ROOT / "Pollice.csv",
    WORKSPACE_ROOT / "Zanoni.csv",
]

SUCCESS_OUTPUT_FILE = DATA_CLEANED_DIR / "pooled_successful_transcripts.csv"
MISSING_OUTPUT_FILE = DATA_CLEANED_DIR / "pooled_missing_transcripts_final_check.csv"
SUMMARY_OUTPUT_FILE = DATA_CLEANED_DIR / "pooled_transcript_summary.txt"


def extract_video_id(file_path: Path) -> str:
    return file_path.name.split(".")[0]


def load_video_ids_from_csv(csv_file: Path):
    if not csv_file.exists():
        return []

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return []

    if "videoId" not in df.columns:
        return []

    return [str(v).strip() for v in df["videoId"].dropna().tolist() if str(v).strip()]


def load_completed_cloud_ids(txt_file: Path):
    if not txt_file.exists():
        return set()

    with open(txt_file, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def main():
    df_master = pd.read_parquet(PARQUET_FILE)
    master_ids = [str(v).strip() for v in df_master["videoId"].dropna().tolist() if str(v).strip()]
    master_set = set(master_ids)

    # Scan the whole workspace to satisfy "present in the workspace".
    transcript_paths_by_id = defaultdict(list)
    for vtt_file in WORKSPACE_ROOT.rglob("*.vtt"):
        vid = extract_video_id(vtt_file)
        if vid in master_set:
            transcript_paths_by_id[vid].append(vtt_file)

    successful_ids = set(transcript_paths_by_id.keys())
    missing_ids = [vid for vid in master_ids if vid not in successful_ids]

    empty_marker_ids = set()
    for empty_file in WORKSPACE_ROOT.rglob("*.empty"):
        vid = extract_video_id(empty_file)
        if vid in master_set:
            empty_marker_ids.add(vid)

    failed_marker_ids = set()
    for failed_file in WORKSPACE_ROOT.rglob("*.failed"):
        vid = extract_video_id(failed_file)
        if vid in master_set:
            failed_marker_ids.add(vid)

    completed_cloud_ids = load_completed_cloud_ids(COMPLETED_CLOUD_FILE)

    assignments = defaultdict(list)
    for csv_file in COLLEAGUE_TASK_FILES:
        for vid in load_video_ids_from_csv(csv_file):
            if vid in master_set:
                assignments[vid].append(csv_file.stem)

    success_rows = []
    for vid in master_ids:
        paths = transcript_paths_by_id.get(vid, [])
        if not paths:
            continue

        rel_paths = [str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/") for p in paths]
        success_rows.append(
            {
                "videoId": vid,
                "transcript_count": len(rel_paths),
                "transcript_paths": " | ".join(sorted(rel_paths)),
            }
        )

    missing_rows = []
    for vid in missing_ids:
        in_completed_cloud = vid in completed_cloud_ids
        has_empty = vid in empty_marker_ids
        has_failed = vid in failed_marker_ids
        assigned_to = sorted(assignments.get(vid, []))

        if has_empty:
            final_check_status = "attempted_no_transcript"
        elif has_failed:
            final_check_status = "attempted_failed"
        elif in_completed_cloud:
            final_check_status = "cloud_done_but_vtt_missing"
        elif assigned_to:
            final_check_status = "assigned_unattempted_or_missing"
        else:
            final_check_status = "unassigned_unattempted"

        missing_rows.append(
            {
                "videoId": vid,
                "in_completed_video_ids_txt": in_completed_cloud,
                "has_empty_marker": has_empty,
                "has_failed_marker": has_failed,
                "assigned_to": "|".join(assigned_to),
                "final_check_status": final_check_status,
            }
        )

    pd.DataFrame(success_rows).to_csv(SUCCESS_OUTPUT_FILE, index=False)
    pd.DataFrame(missing_rows).to_csv(MISSING_OUTPUT_FILE, index=False)

    status_counts = pd.DataFrame(missing_rows)["final_check_status"].value_counts().to_dict() if missing_rows else {}

    summary_lines = [
        f"Master dataset videos: {len(master_ids)}",
        f"Videos with transcript present in workspace: {len(success_rows)}",
        f"Videos without transcript in workspace: {len(missing_rows)}",
        "",
        "Final check status counts for missing group:",
    ]
    for status_name in sorted(status_counts):
        summary_lines.append(f"- {status_name}: {status_counts[status_name]}")

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("Pooling completed.")
    print(f"- Success pool: {SUCCESS_OUTPUT_FILE}")
    print(f"- Missing pool + final check: {MISSING_OUTPUT_FILE}")
    print(f"- Summary: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
