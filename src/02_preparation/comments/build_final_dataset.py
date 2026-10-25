"""
Build the FINAL dataset: one row per folder in multimodal_dataset_fixed/, joined
to the enriched features from the other chat (ig_posts_multimodal_enriched.parquet,
keyed by shortcode), plus per-folder media/processing audit columns.

Output: Output/ig_multimodal_final.parquet  (+ .csv for quick inspection)

Guarantees a PERFECT 1:1 with the folder set. Posts whose features could not be
fetched (HTTP 400 at fetch time, or added after the enrichment run) keep null
features but are explained by `features_status` so no null is silent/accidental.
"""
import json
import re
from pathlib import Path

import pandas as pd

BASE  = Path(__file__).parent
FIXED = BASE / "multimodal_dataset_fixed"
ENR   = BASE / "Output" / "ig_posts_multimodal_enriched.parquet"
MF    = BASE / "Output" / "ig_media_features.parquet"   # carries fetch_status for errors
OUT_PARQUET = BASE / "Output" / "ig_multimodal_final.parquet"
OUT_CSV     = BASE / "Output" / "ig_multimodal_final.csv"

VIDEO_FORMS = {"reel", "feed", "carousel_video"}


def nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def audit_folder(sc: str, form: str, d: Path) -> dict:
    """Per-folder media + processing audit."""
    jpgs = sorted(d.glob("*.jpg"))
    mp4s = sorted(d.glob("*.mp4"))
    info = {
        "shortcode": sc,
        "content_form": form,
        "folder": str(d.relative_to(BASE)),
        "n_jpg": len(jpgs),
        "n_mp4": len(mp4s),
    }

    if form in ("reel", "feed"):
        # single-video layout: <sc>.mp4 + frames/ + transcription.txt
        frames = list((d / "frames").glob("*.jpg")) if (d / "frames").exists() else []
        tr = d / "transcription.txt"
        info.update({
            "n_slides": 1,
            "has_frames": len(frames) > 0,
            "n_frames": len(frames),
            "has_transcript": tr.exists(),
            "transcript_nonempty": nonempty(tr),
        })
    elif form == "carousel_video":
        # multi-video: slide_NN/frames + slide_NN/transcription.txt
        slide_dirs = sorted(d.glob("slide_*"))
        nf = sum(len(list((sd / "frames").glob("*.jpg"))) for sd in slide_dirs)
        trs = [sd / "transcription.txt" for sd in slide_dirs]
        info.update({
            "n_slides": len(mp4s),
            "has_frames": all((sd / "frames").exists() and any((sd / "frames").glob("*.jpg")) for sd in slide_dirs) if slide_dirs else False,
            "n_frames": nf,
            "has_transcript": all(t.exists() for t in trs) if trs else False,
            "transcript_nonempty": all(nonempty(t) for t in trs) if trs else False,
        })
    elif form == "carousel":
        info.update({
            "n_slides": len(jpgs),
            "has_frames": None, "n_frames": 0,
            "has_transcript": None, "transcript_nonempty": None,
        })
    else:  # image
        info.update({
            "n_slides": 1,
            "has_frames": None, "n_frames": 0,
            "has_transcript": None, "transcript_nonempty": None,
        })
    return info


def main():
    rows = []
    for form in ("reel", "feed", "image", "carousel", "carousel_video"):
        sub = FIXED / form
        if not sub.exists():
            continue
        for d in sorted(sub.iterdir()):
            if d.is_dir() and (any(d.glob("*.jpg")) or any(d.glob("*.mp4"))):
                rows.append(audit_folder(d.name, form, d))
    audit = pd.DataFrame(rows)
    print(f"Folders audited: {len(audit)}")

    # ---- join enriched features ----
    enr = pd.read_parquet(ENR)
    # avoid clobbering audit's content columns; enriched has media_type etc.
    enr_feat_cols = [c for c in enr.columns if c not in ("shortcode",)]
    merged = audit.merge(enr, on="shortcode", how="left", suffixes=("", "_enr"))

    # ---- backfill base metadata from cleaned for rows missing an enriched row ----
    # (e.g. posts whose feature-fetch errored): guarantees media_id/timestamp/counts
    # are always correct so downstream joins like comments don't silently miss.
    clean = pd.read_parquet(BASE / "../Data/ig_posts_cleaned.parquet")
    clean["shortcode"] = clean["permalink"].astype(str).str.rstrip("/").str.split("/").str[-1]
    base_cols = [c for c in ("media_id", "caption", "comments_count", "like_count",
                             "media_product_type", "media_type", "permalink", "timestamp",
                             "reach", "saved", "views", "total_interactions")
                 if c in clean.columns and c in merged.columns]
    clean_idx = clean.drop_duplicates("shortcode").set_index("shortcode")
    for col in base_cols:
        fill = merged["shortcode"].map(clean_idx[col])
        merged[col] = merged[col].where(merged[col].notna(), fill)

    # ---- features_status from media_features fetch_status ----
    status_map = {}
    if MF.exists():
        mf = pd.read_parquet(MF)
        for _, r in mf.iterrows():
            status_map[r["shortcode"]] = r.get("fetch_status")
    have_feat = set(enr["shortcode"])
    def feat_status(sc):
        if sc in have_feat:
            return "ok"
        s = status_map.get(sc)
        if isinstance(s, str) and s.startswith("error"):
            return "fetch_error"
        return "not_fetched"
    merged["features_status"] = merged["shortcode"].map(feat_status)

    # ---- sanity ----
    assert len(merged) == len(audit), "row count changed on merge!"
    print(f"Final rows: {len(merged)}  | with features: {(merged['features_status']=='ok').sum()}"
          f" | fetch_error: {(merged['features_status']=='fetch_error').sum()}"
          f" | not_fetched: {(merged['features_status']=='not_fetched').sum()}")

    merged.to_parquet(OUT_PARQUET, index=False)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved:\n  {OUT_PARQUET}\n  {OUT_CSV}")
    print(f"Columns ({len(merged.columns)}):", list(merged.columns))


if __name__ == "__main__":
    main()
