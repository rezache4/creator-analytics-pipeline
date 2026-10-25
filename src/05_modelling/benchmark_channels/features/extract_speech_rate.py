"""
extract_speech_rate.py

Speech-pacing features from the word-timestamped VTT captions, for the 740 long-form
cohort videos. Two PRESCRIPTIVE delivery levers (creator-controlled pre-publish):

  - articulation_wpm : words / SPEAKING time. Speaking time = total talk span minus
                       inter-word gaps > SILENCE_GAP_S (dramatic pauses, music, B-roll,
                       silent gags). This is true delivery tempo, silence removed.
  - speech_density   : speaking time / video duration. The talk-vs-silence ratio
                       itself -- "how much dead air / visual-only time" -- a lever.

Silence rule (locked with user): a gap > 3.0 s between consecutive word timestamps is
silence and is subtracted from speaking time. Gaps <= 3 s are normal speech rhythm.

VTT word timing comes from inline tags like `parola<00:00:00.510>`. YouTube auto-captions
use a rolling window (each cue repeats prior words), so words are counted by the set of
UNIQUE inline onset timestamps -- one per spoken word -- NOT by displayed tokens (which
would overcount ~3x). See parse_vtt() for detail.

Coverage: ~597/740 have a VTT (80.7%); the rest are .txt (no timestamps) -> NaN here.
Imputation/flagging happens in build_feature_table.py (channel-median + available flag),
matching the face_pose / topic_volatility convention.

Output: features/speech_rate.parquet  [video_id, articulation_wpm, speech_density,
        n_words_vtt, speaking_seconds, speech_rate_available]

Run: python extract_speech_rate.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent           # features/
PROJ = BASE.parent.parent                          # 
VTT_DIR = PROJ / "transcripts" / "vtt_files"
VIDEOS = PROJ / "clean_data" / "yt_videos_with_local_transcripts.parquet"
TABLE = BASE / "feature_table.parquet"
OUT = BASE / "speech_rate.parquet"

SILENCE_GAP_S = 3.0   # inter-word gap above this = silence (subtracted from speaking time)

# Regexes: inline word-onset timestamp <hh:mm:ss.mmm>; word tokens are non-space,
# non-angle-bracket runs (after stripping the timestamp + <c> caption-styling tags).
TS = re.compile(r"<(\d{2}):(\d{2}):(\d{2}\.\d{3})>")


def _sec(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(path):
    """Return (n_words, speaking_seconds) from word-level timestamps, or None.

    IMPORTANT — YouTube auto-caption VTTs use a ROLLING window: each cue repeats the
    previous cue's words and appends new ones, so the same word's text appears on
    several consecutive lines. Counting displayed tokens therefore massively
    overcounts (gives ~530 "wpm"). Each spoken word, however, carries exactly ONE
    inline onset tag `<hh:mm:ss.mmm>` the first time it is voiced. So we count words
    by the set of UNIQUE onset timestamps -- that is both an honest word count and the
    timing source, keeping numerator and denominator internally consistent.

    n_words       = number of unique inline word-onset timestamps.
    speaking_secs = (last onset - first onset) MINUS the sum of consecutive inter-word
                    gaps > SILENCE_GAP_S (removes silence/pauses, keeps normal rhythm).
    Returns None when too little timed content to be reliable (<5 unique onsets).
    """
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    onsets = set()   # unique inline word-onset times (dedupes the rolling-window repeats)
    for line in txt.splitlines():
        if "-->" in line or line.strip() in ("", "WEBVTT") or line.startswith(("Kind:", "Language:")):
            continue
        for mt in TS.finditer(line):
            onsets.add(round(_sec(*mt.groups()), 3))
    times = np.array(sorted(onsets))
    n_words = len(times)
    if n_words < 5:
        return None
    gaps = np.diff(times)
    total_span = times[-1] - times[0]
    silence = gaps[gaps > SILENCE_GAP_S].sum()              # only long gaps = silence
    speaking = max(total_span - silence, 1e-6)
    return n_words, float(speaking)


def build_index():
    """Map video_id -> vtt path. Filenames are `<id>.it.vtt` or `<title> [<id>].it.vtt`.
    The 11-char YouTube id is taken from a trailing [..] group or the leading token."""
    idx = {}
    for f in VTT_DIR.glob("*.vtt"):
        m = re.search(r"\[([A-Za-z0-9_-]{11})\]\.", f.name) or re.match(r"^([A-Za-z0-9_-]{11})\.", f.name)
        if m:
            idx.setdefault(m.group(1), f)
    return idx


def main():
    ft = pd.read_parquet(TABLE, columns=["video_id", "channelTitle"])
    dur = pd.read_parquet(VIDEOS, columns=["videoId", "duration_seconds"]).rename(
        columns={"videoId": "video_id"})
    ft = ft.merge(dur, on="video_id", how="left")
    ft["duration_seconds"] = pd.to_numeric(ft["duration_seconds"], errors="coerce")

    idx = build_index()
    rows = []
    for _, r in ft.iterrows():
        vid = str(r["video_id"])
        path = idx.get(vid)
        res = parse_vtt(path) if path else None
        if res is None:
            rows.append((vid, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        n_words, speaking = res
        artic = n_words / (speaking / 60.0)                 # words per speaking-minute
        dens = speaking / r["duration_seconds"] if r["duration_seconds"] else np.nan
        # clamp density to (0,1]: caption timing noise can push marginally over 1.0
        dens = float(min(dens, 1.0)) if pd.notna(dens) else np.nan
        rows.append((vid, artic, dens, n_words, speaking, 1))

    out = pd.DataFrame(rows, columns=[
        "video_id", "articulation_wpm", "speech_density",
        "n_words_vtt", "speaking_seconds", "speech_rate_available"])
    out.to_parquet(OUT, index=False)

    cov = out["speech_rate_available"].mean()
    ok = out[out.speech_rate_available == 1]
    print(f"VTT speech-rate: {int(out.speech_rate_available.sum())}/{len(out)} "
          f"({cov*100:.1f}%) parsed.")
    print("\narticulation_wpm:")
    print(ok["articulation_wpm"].describe().round(1).to_string())
    print("\nspeech_density:")
    print(ok["speech_density"].describe().round(3).to_string())
    print(f"\nWrote -> {OUT.name}  (silence gap threshold = {SILENCE_GAP_S}s)")


if __name__ == "__main__":
    main()
