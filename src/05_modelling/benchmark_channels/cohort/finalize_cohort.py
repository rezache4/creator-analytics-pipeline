"""
finalize_cohort.py

Finalise the benchmark comparator cohort for the format-similar Italian YouTuber
prescriptive analysis, using ONLY existing artefacts (no ratio recomputation).

Selection rule (updated 2026-06-12; see PLAN_expand_cohort_drop_semantic_gate.md):
  - FORMAT gate (strict): native_short_share <= 0.20  (TargetCreator's defining trait)
  - Cosine similarity is computed and retained as INFORMATIONAL only; it is NO LONGER
    part of the inclusion rule. Rationale: the two-stage mean-pool embedding encodes
    "informal spoken Italian, one person to camera" rather than subject matter, so high
    cosine reflects shared language/register, not shared topic. The gate was therefore
    not a trustworthy comparability certificate.

Output: cohort_selection.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DENSE = Path(r"d:\creator-analytics\embeddings\dense_vectors_semantic.parquet")
VIDEOS = Path(r"d:\creator-analytics\clean_data\yt_videos_with_local_transcripts.parquet")
COS_CSV  = BASE_DIR / "principled_analysis" / "channel_similarity_crosscheck.csv"
PROF_CSV = BASE_DIR / "clip_vs_native_channel_profile.csv"   # post-2022 clip_rate
OUT_CSV  = BASE_DIR / "cohort_selection.csv"

COS_GATE   = 0.90   # informational threshold only — NOT used in include (see module docstring)
FORMAT_PP  = 0.20   # (retained for profile_l1 reporting)
NATIVE_MAX = 0.20   # strict gate: the ONLY inclusion criterion (TargetCreator = 0.09)
POST2022   = pd.Timestamp("2022-01-01", tz="UTC")


def main():
    cos = pd.read_csv(COS_CSV, encoding="utf-8-sig")[["channelTitle", "cosine_similarity"]]
    prof = pd.read_csv(PROF_CSV, encoding="utf-8-sig")[["channelTitle", "clip_rate", "n_shorts"]]

    # post-2022 long-share per channel (raw is_short) + TargetCreator reference
    dv = pd.read_parquet(DENSE, columns=["video_id", "is_short"])
    vid = dv.groupby("video_id")["is_short"].first().reset_index()
    meta = pd.read_parquet(VIDEOS, columns=["videoId", "channelTitle", "publishedAt"]) \
             .rename(columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")
    v = vid.merge(meta, on="video_id", how="inner")
    v22 = v[v["publishedAt"] >= POST2022]
    ls = (v22.groupby("channelTitle")
              .agg(n_post2022=("video_id", "nunique"),
                   long_share=("is_short", lambda s: (~s).mean()))
              .reset_index())

    # --- three-way format profile (pure arithmetic, no recompute) ---
    # Each channel's post-2022 videos split into 3 mutually-exclusive shares
    # that sum to 1. The native/derived distinction is ONLY about shorts, so we
    # split the shorts share into clip vs native and keep long-form separate.
    #   long_share          (from raw is_short)
    #   shorts_share        = 1 - long_share
    #   clip_share          = shorts_share * clip_rate
    #   native_short_share  = shorts_share * (1 - clip_rate)
    def add_profile(d):
        d = d.copy()
        d["clip_rate"] = d["clip_rate"].fillna(0.0)   # no shorts / no clips -> 0
        shorts_share = 1 - d["long_share"]
        d["clip_share"]         = shorts_share * d["clip_rate"]
        d["native_short_share"] = shorts_share * (1 - d["clip_rate"])
        return d

    PROFILE_COLS = ["long_share", "clip_share", "native_short_share"]

    ls_prof = add_profile(ls.merge(prof, on="channelTitle", how="left"))

    # TargetCreator reference profile (3-vector), from full tables
    cami = ls_prof[ls_prof["channelTitle"] == "TargetCreator"].iloc[0]
    cami_vec = cami[PROFILE_COLS].astype(float).values

    # comparators = channels in the cosine CSV (TargetCreator already excluded there)
    comp = cos.merge(ls_prof, on="channelTitle", how="left").copy()

    # Selection rule (2026-06-12): include = native_ok ONLY.
    # cos_ok is retained as an INFORMATIONAL column documenting what we dropped.
    # Rationale in PLAN_expand_cohort_drop_semantic_gate.md §1:
    # the channel centroid averaging washes out topical structure; high cosine
    # reflects shared language/register, not topic. Not a trustworthy gate.
    comp_vecs = comp[PROFILE_COLS].astype(float).values
    comp["profile_l1"] = np.abs(comp_vecs - cami_vec).sum(axis=1)
    comp["cos_ok"]     = comp["cosine_similarity"] >= COS_GATE   # informational only
    comp["native_ok"]  = comp["native_short_share"] <= NATIVE_MAX
    comp["include"]    = comp["native_ok"]                        # cosine gate DROPPED
    comp = comp.sort_values(["include", "cosine_similarity"],
                            ascending=[False, False]).reset_index(drop=True)

    pd.set_option("display.width", 220)
    print("=" * 78)
    print("TargetCreator format profile (post-2022):")
    print(f"  long={cami_vec[0]:.3f}  clip={cami_vec[1]:.3f}  native_short={cami_vec[2]:.3f}")
    print(f"Gate: native_short_share <= {NATIVE_MAX} (ONLY criterion; cosine informational)")
    print(f"  cos_ok (>={COS_GATE}) retained as a column but NOT used for inclusion.")
    print("=" * 78)
    cols = ["channelTitle", "cosine_similarity", "long_share", "clip_share",
            "native_short_share", "cos_ok", "native_ok", "include"]
    print(comp[cols].round(3).to_string(index=False))

    cohort = comp[comp["include"]]
    print("\n" + "-" * 78)
    print(f"COHORT (native_short_share<={NATIVE_MAX} only, cosine gate dropped): "
          f"{len(cohort)} channels")
    print("  " + ", ".join(cohort["channelTitle"]) if len(cohort) else "  (none)")

    # sensitivity: how cohort size changes with the native-short threshold (all channels)
    print("\nSensitivity of cohort size to NATIVE_MAX (all channels, no cosine filter):")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30):
        sel = comp[comp["native_short_share"] <= thr]
        print(f"  native<= {thr:.2f}: {len(sel)} channels  -> "
              + ", ".join(sel["channelTitle"].tolist()))

    comp.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUT_CSV.name}")


if __name__ == "__main__":
    main()
