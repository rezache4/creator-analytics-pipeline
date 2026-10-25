"""Phase 2A — Chapter-level topic clustering + per-video compositional profiles.

Rationale
---------
Each video was embedded as one vector per semantic *chapter*
(embed_transcripts_semantic.py). Mean-pooling those chapters back to a single
video vector would blur a "60% cooking / 40% travel" video into a meaningless
midpoint. Instead we cluster at the chapter level and describe each video by how
its chapters distribute across the discovered topics — a compositional profile.

Pipeline
--------
1.  Load chapter-level dense vectors  -> one row per (video_id, chapter_index).
2.  UMAP 768 -> N_CLUSTER_DIMS (clustering space) and 768 -> 2 (plotting).
3.  HDBSCAN on the cluster space, with prediction_data so we can read *soft*
    membership vectors (each chapter gets a probability distribution over the K
    discovered topics, not just a hard label).
4.  Hard label per chapter = HDBSCAN label, with noise (-1) reassigned to the
    nearest cluster centroid. Raw label kept for auditing the noise share.
5.  Per-video compositional profile = mean of its chapters' soft membership
    vectors -> length-K distribution. dominant_cluster = argmax(profile).
6.  Outputs:
      chapter_clusters.parquet
          video_id, chapter_index, is_short,
          chapter_cluster, chapter_cluster_raw,
          umap_vector (list[float], length N_CLUSTER_DIMS), umap_x, umap_y
      video_profiles.parquet
          video_id, is_short, n_chapters,
          dominant_cluster, profile (list[float], length K)
      cluster_examples.json   nearest-centroid chapters per cluster (+ snippet),
                              ready input for the Gemini labelling step
      chapter_clusters_umap.png

Labels (topic names) are NOT assigned here — clustering is unsupervised. The
Gemini labelling step (Phase 3) reads cluster_examples.json and names each
cluster after the fact.

Usage (from the Data_Cleaned directory)
---------------------------------------
    python topic_clustering.py
    python topic_clustering.py --cluster-dims 20 --min-cluster-size 150
    python topic_clustering.py --no-plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# All file paths resolve relative to this script's directory, so the script
# runs the same whether launched from the Data_Cleaned dir, the repo root, or
# the IDE "Run" button (whose working directory is the workspace root).
BASE_DIR = Path(__file__).resolve().parent

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_INPUT      = "dense_vectors_semantic.parquet"
CHAPTER_OUTPUT     = "chapter_clusters.parquet"
PROFILE_OUTPUT     = "video_profiles.parquet"
EXAMPLES_OUTPUT    = "cluster_examples.json"
PLOT_OUTPUT        = "chapter_clusters_umap.png"

# Transcript source for example snippets (joined on videoId -> video_id).
# NOTE: dense_vectors stores no text, so snippets are the *whole-video*
# transcript of the chapter's video, not the exact chapter span — adequate for
# naming a cluster, since the nearest-centroid chapters dominate the video.
TRANSCRIPT_PARQUET = "yt_videos_with_local_transcripts.parquet"
TRANSCRIPT_ID_COL  = "videoId"
TRANSCRIPT_TXT_COL = "local_transcript"

N_CLUSTER_DIMS     = 20     # UMAP target dim for clustering
UMAP_NEIGHBORS     = 30     # larger -> more global structure
UMAP_MIN_DIST      = 0.0    # 0.0 packs clusters tightly (best for HDBSCAN)
MIN_CLUSTER_SIZE   = 150    # HDBSCAN: smallest admissible topic (chapter-level)
MIN_SAMPLES        = 10     # HDBSCAN: how conservative noise-labelling is
N_EXAMPLES         = 8      # nearest-centroid chapters saved per cluster
SNIPPET_CHARS      = 500    # transcript chars per example snippet
RANDOM_STATE       = 42


def _resolve(p: str | Path) -> Path:
    """Resolve a path against the script directory unless it's already absolute."""
    p = Path(p)
    return p if p.is_absolute() else BASE_DIR / p


# ── Step 1: load chapter vectors ───────────────────────────────────────────────

def load_chapters(input_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (chapter_df, matrix) where matrix is (n_chapters, dim)."""
    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    df = df.sort_values(["video_id", "chapter_index"]).reset_index(drop=True)
    matrix = np.stack(df["embedding_vector"].apply(np.asarray).to_numpy())
    print(f"  {len(df):,} chapters  |  {df['video_id'].nunique():,} videos  "
          f"|  dim={matrix.shape[1]}")
    return df, matrix


# ── Step 2: UMAP projections ────────────────────────────────────────────────────

def run_umap(matrix: np.ndarray, n_components: int) -> np.ndarray:
    import umap  # umap-learn

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",          # cosine suits normalized text embeddings
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(matrix)


# ── Step 3–4: HDBSCAN, soft memberships, noise reassignment ────────────────────

def cluster_chapters(
    cluster_space: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    min_samples: int = MIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (hard_labels, raw_labels, soft_memberships, cluster_ids).

    hard_labels       : per-chapter cluster id, noise reassigned to nearest centroid
    raw_labels        : per-chapter HDBSCAN label (-1 = noise), kept for auditing
    soft_memberships  : (n_chapters, K) probability distribution over clusters
    cluster_ids       : sorted list of the K real cluster ids (columns of soft)
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",       # UMAP output space is Euclidean-friendly
        prediction_data=True,     # needed for soft membership vectors
    )
    raw = clusterer.fit_predict(cluster_space)

    cluster_ids = sorted(c for c in set(raw) if c != -1)
    n_clusters  = len(cluster_ids)
    n_noise     = int((raw == -1).sum())
    print(f"  HDBSCAN: {n_clusters} clusters  |  "
          f"{n_noise:,} noise chapters ({n_noise / len(raw):.1%})")
    if n_clusters == 0:
        raise SystemExit("No clusters found — lower --min-cluster-size and retry.")

    # Soft membership vectors (probability over the K clusters).
    try:
        soft = hdbscan.all_points_membership_vectors(clusterer)
        soft = np.atleast_2d(soft)
        if soft.shape[1] != n_clusters:        # guard against shape quirks
            raise ValueError("membership width mismatch")
        print("  Soft membership vectors computed")
    except Exception as e:                      # fallback: one-hot of hard labels
        print(f"  Soft membership unavailable ({e}); falling back to one-hot")
        soft = None

    # Hard labels with noise reassigned to nearest centroid.
    labels = raw.copy()
    centroids = np.stack([cluster_space[raw == c].mean(axis=0) for c in cluster_ids])
    if n_noise:
        for i in np.where(raw == -1)[0]:
            d = np.linalg.norm(centroids - cluster_space[i], axis=1)
            labels[i] = cluster_ids[int(d.argmin())]
        print(f"  Reassigned {n_noise:,} noise chapters to nearest centroid")

    if soft is None:                            # one-hot fallback from hard labels
        soft = np.zeros((len(labels), n_clusters), dtype=float)
        id_to_col = {c: j for j, c in enumerate(cluster_ids)}
        for i, lab in enumerate(labels):
            soft[i, id_to_col[lab]] = 1.0

    return labels, raw, soft, cluster_ids


# ── Step 5: per-video compositional profiles ────────────────────────────────────

def build_profiles(
    chapter_df: pd.DataFrame,
    soft: np.ndarray,
    cluster_ids: list[int],
) -> pd.DataFrame:
    """Average each video's chapter membership vectors into one profile."""
    rows = []
    for video_id, grp in chapter_df.groupby("video_id", sort=False):
        idx     = grp.index.to_numpy()
        profile = soft[idx].mean(axis=0)
        profile = profile / profile.sum() if profile.sum() > 0 else profile
        rows.append(
            {
                "video_id": video_id,
                "is_short": bool(grp["is_short"].iloc[0]),
                "n_chapters": len(grp),
                "dominant_cluster": int(cluster_ids[int(profile.argmax())]),
                "profile": profile.tolist(),
            }
        )
    out = pd.DataFrame(rows)
    print(f"  Built profiles for {len(out):,} videos  (profile dim={len(cluster_ids)})")
    return out


# ── Step 6: representative chapters per cluster ─────────────────────────────────

def build_examples(
    chapter_df: pd.DataFrame,
    cluster_space: np.ndarray,
    hard_labels: np.ndarray,
    cluster_ids: list[int],
) -> dict:
    snippets: dict[str, str] = {}
    tpath = _resolve(TRANSCRIPT_PARQUET)
    if tpath.exists():
        tdf = pd.read_parquet(tpath, columns=[TRANSCRIPT_ID_COL, TRANSCRIPT_TXT_COL])
        snippets = {
            str(r[TRANSCRIPT_ID_COL]): (
                r[TRANSCRIPT_TXT_COL][:SNIPPET_CHARS]
                if pd.notna(r[TRANSCRIPT_TXT_COL]) else ""
            )
            for _, r in tdf.iterrows()
        }
    else:
        print(f"  (transcript parquet not found at {tpath}; snippets omitted)")

    vids = chapter_df["video_id"].to_numpy()
    chs  = chapter_df["chapter_index"].to_numpy()

    examples: dict = {}
    for c in cluster_ids:
        member_idx = np.where(hard_labels == c)[0]
        centroid   = cluster_space[member_idx].mean(axis=0)
        dists      = np.linalg.norm(cluster_space[member_idx] - centroid, axis=1)
        nearest    = member_idx[np.argsort(dists)[:N_EXAMPLES]]
        examples[str(int(c))] = {
            "n_chapters": int(len(member_idx)),
            "examples": [
                {
                    "video_id": str(vids[i]),
                    "chapter_index": int(chs[i]),
                    "snippet": snippets.get(str(vids[i]), ""),
                }
                for i in nearest
            ],
        }
    return examples


# ── plot ────────────────────────────────────────────────────────────────────────

def save_plot(xy: np.ndarray, labels: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=3, alpha=0.5, linewidths=0
    )
    ax.set_title("Chapters — UMAP projection coloured by topic cluster")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    legend = ax.legend(*scatter.legend_elements(num=None),
                       title="cluster", loc="best", fontsize=7, ncol=2)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Saved plot -> {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",            default=DEFAULT_INPUT)
    parser.add_argument("--cluster-dims",     type=int, default=N_CLUSTER_DIMS)
    parser.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples",      type=int, default=MIN_SAMPLES)
    parser.add_argument("--no-plot",          action="store_true")
    parser.add_argument("--examples-only",    action="store_true",
                        help="Skip UMAP/HDBSCAN; rebuild cluster_examples.json "
                             "from the already-saved chapter_clusters.parquet")
    args = parser.parse_args()

    input_path = _resolve(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # ── examples-only shortcut ────────────────────────────────────────────────
    if args.examples_only:
        chapter_out_path = _resolve(CHAPTER_OUTPUT)
        if not chapter_out_path.exists():
            raise SystemExit(f"chapter_clusters.parquet not found: {chapter_out_path}")
        print(f"--examples-only: loading {CHAPTER_OUTPUT} ...")
        chapter_out = pd.read_parquet(chapter_out_path)
        chapter_df  = chapter_out[["video_id", "chapter_index", "is_short"]].copy()
        cluster_space = np.stack(chapter_out["umap_vector"].apply(np.asarray).to_numpy())
        hard          = chapter_out["chapter_cluster"].to_numpy()
        cluster_ids   = sorted(set(hard.tolist()))
        examples = build_examples(chapter_df, cluster_space, hard, cluster_ids)
        _resolve(EXAMPLES_OUTPUT).write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved -> {EXAMPLES_OUTPUT}  ({len(examples)} clusters)")
        return

    # 1 ─ load chapters
    chapter_df, matrix = load_chapters(input_path)

    # 2 ─ UMAP (cluster space + 2-D viz)
    print(f"\nUMAP 768 -> {args.cluster_dims} (clustering) ...")
    cluster_space = run_umap(matrix, args.cluster_dims)
    print("UMAP 768 -> 2 (visualisation) ...")
    xy = run_umap(matrix, 2)

    # 3–4 ─ cluster + soft memberships
    print("\nClustering chapters ...")
    hard, raw, soft, cluster_ids = cluster_chapters(
        cluster_space, args.min_cluster_size, args.min_samples
    )

    # 6a ─ chapter-level output
    chapter_out = pd.DataFrame(
        {
            "video_id":            chapter_df["video_id"].to_numpy(),
            "chapter_index":       chapter_df["chapter_index"].to_numpy(),
            "is_short":            chapter_df["is_short"].to_numpy(),
            "chapter_cluster":     hard.astype(int),
            "chapter_cluster_raw": raw.astype(int),
            "umap_vector":         [v.tolist() for v in cluster_space],
            "umap_x":              xy[:, 0],
            "umap_y":              xy[:, 1],
        }
    )
    chapter_out.to_parquet(_resolve(CHAPTER_OUTPUT), index=False)
    print(f"\nSaved -> {CHAPTER_OUTPUT}  ({len(chapter_out):,} chapters)")

    # 5 / 6b ─ per-video profiles
    profiles = build_profiles(chapter_df, soft, cluster_ids)
    profiles.to_parquet(_resolve(PROFILE_OUTPUT), index=False)
    print(f"Saved -> {PROFILE_OUTPUT}  ({len(profiles):,} videos)")

    # 6c ─ examples for labelling
    examples = build_examples(chapter_df, cluster_space, hard, cluster_ids)
    _resolve(EXAMPLES_OUTPUT).write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved -> {EXAMPLES_OUTPUT}  ({len(examples)} clusters)")

    # summaries
    print("\nChapter cluster sizes:")
    for cid, n in chapter_out["chapter_cluster"].value_counts().sort_index().items():
        print(f"  cluster {cid:>3}: {n:>6,} chapters")
    print("\nDominant-cluster distribution across videos:")
    for cid, n in profiles["dominant_cluster"].value_counts().sort_index().items():
        print(f"  cluster {cid:>3}: {n:>6,} videos")

    if not args.no_plot:
        save_plot(xy, hard, _resolve(PLOT_OUTPUT))


if __name__ == "__main__":
    main()
