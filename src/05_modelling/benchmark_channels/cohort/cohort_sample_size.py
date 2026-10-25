"""
cohort_sample_size.py

Given the finalised cohort, report the usable sample size (N) under several
content-scope definitions, and translate that into a feature/df budget for the
within-cohort regression.

Scope rules established in FINDINGS.md:
  - post-2022 only (COVID exclusion)
  - derived clips are promo artefacts -> EXCLUDE from regression
  - native shorts are off-strategy for long-form prescriptive levers
  - => primary regression unit is LONG-FORM videos

Cohort: loaded from cohort_selection.csv via features/cohort_config.py.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent / "features"))
from cohort_config import COHORT as _FULL_COHORT, TARGET, POST2022

DENSE  = Path(r"d:\creator-analytics\embeddings\dense_vectors_semantic.parquet")
VIDEOS = Path(r"d:\creator-analytics\clean_data\yt_videos_with_local_transcripts.parquet")
CLIPS  = BASE_DIR / "clip_vs_native_shorts.csv"          # post-2022 clip labels

# Comparators only (TARGET = TargetCreator, kept separate for the N-table below)
COHORT = [c for c in _FULL_COHORT if c != TARGET]


def main():
    # video-level table, post-2022, with kind = long / native_short / derived_clip
    dv = pd.read_parquet(DENSE, columns=["video_id", "is_short"])
    vid = dv.groupby("video_id")["is_short"].first().reset_index()
    meta = pd.read_parquet(VIDEOS, columns=["videoId", "channelTitle", "publishedAt"]) \
             .rename(columns={"videoId": "video_id"})
    meta["publishedAt"] = pd.to_datetime(meta["publishedAt"], utc=True, errors="coerce")
    v = vid.merge(meta, on="video_id", how="inner")
    v = v[v["publishedAt"] >= POST2022].copy()

    clips = pd.read_csv(CLIPS, encoding="utf-8-sig")
    clip_ids = set(clips.loc[clips["label"] == "derived_clip", "video_id"])
    v["kind"] = np.where(~v["is_short"], "long",
                np.where(v["video_id"].isin(clip_ids), "derived_clip", "native_short"))

    keep = [TARGET] + COHORT
    vk = v[v["channelTitle"].isin(keep)]

    tab = (vk.groupby(["channelTitle", "kind"]).size()
              .unstack(fill_value=0)
              .reindex(keep))
    for c in ["long", "native_short", "derived_clip"]:
        if c not in tab: tab[c] = 0
    tab["total_post2022"] = tab[["long", "native_short", "derived_clip"]].sum(axis=1)
    tab["long_plus_native"] = tab["long"] + tab["native_short"]

    pd.set_option("display.width", 160)
    print("=" * 72)
    print("Post-2022 video counts by channel x kind")
    print("=" * 72)
    print(tab[["long", "derived_clip", "native_short",
               "long_plus_native", "total_post2022"]].to_string())

    # ---- N under each scope definition ----
    comp = tab.loc[COHORT]
    N_long_comp   = int(comp["long"].sum())
    N_lpn_comp    = int(comp["long_plus_native"].sum())
    N_all_comp    = int(comp["total_post2022"].sum())
    cami = tab.loc[TARGET]
    cami_long = int(cami["long"]); cami_lpn = int(cami["long_plus_native"])

    n_channels = len(COHORT) + 1  # + TargetCreator as a fixed effect level

    print("\n" + "=" * 72)
    print("Usable N and feature/df budget (pooled, channel fixed effects)")
    print("=" * 72)
    scopes = [
        ("LONG-FORM only (recommended)",
         N_long_comp, cami_long),
        ("LONG + NATIVE shorts (clips still excluded)",
         N_lpn_comp, cami_lpn),
        ("ALL post-2022 (clips included — NOT recommended)",
         N_all_comp, int(cami["total_post2022"])),
    ]
    print(f"\n{'Scope':<46}{'N_comp':>8}{'N_cami':>8}{'N_tot':>8}{'df_budget':>11}")
    print("-" * 81)
    for name, nc, ncam in scopes:
        N = nc + ncam
        # rule of thumb: keep params <= N/10 to N/15; subtract channel FE dummies
        fe = n_channels - 1
        budget_10 = max(0, N // 10 - fe)
        budget_15 = max(0, N // 15 - fe)
        print(f"{name:<46}{nc:>8}{ncam:>8}{N:>8}{f'{budget_15}-{budget_10}':>11}")

    print(f"\n  channel fixed-effect dummies absorbed: {n_channels - 1} "
          f"({n_channels} channels incl. TargetCreator)")
    print("  df_budget = (N/15 .. N/10) minus FE dummies  [predictors before "
          "controls like year/topic]")

    # ---- TargetCreator-only supplementary model ----
    print("\n" + "-" * 72)
    print(f"TargetCreator-only supplementary model: N_long = {cami_long} "
          f"-> max ~{max(1, cami_long // 8)}-{max(1, cami_long // 5)} predictors "
          f"(validation/direction check only)")

    out = BASE_DIR / "cohort_sample_size.csv"
    tab.to_csv(out, encoding="utf-8-sig")
    print(f"\nSaved: {out.name}")


if __name__ == "__main__":
    main()
