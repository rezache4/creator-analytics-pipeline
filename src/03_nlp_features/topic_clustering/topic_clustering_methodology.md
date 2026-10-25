# Topic Clustering Methodology

**Phase:** 2A of the Transcript-to-Causal-Features Pipeline  
**Date:** June 2026  
**Input:** `dense_vectors_semantic.parquet` — 35,857 chapter-level transcript embeddings across 11,796 videos  
**Outputs:** `chapter_clusters.parquet`, `video_profiles.parquet`, `cluster_examples.json`

---

## 1. Motivation

Each video transcript was previously segmented into semantic *chapters* using cosine-similarity boundary detection (Phase 1), then each chapter was embedded as a 768-dimensional dense vector using `intfloat/multilingual-e5-base`. Clustering was performed at the chapter level rather than the video level for the following reason: mean-pooling all chapters into a single per-video vector before clustering would destroy compositional structure. A video that is 60% cooking content and 40% lifestyle vlog would be mapped to a vague midpoint vector that belongs clearly to neither topic. Clustering chapters directly preserves this information and allows each video to carry a graded membership across topics — its *compositional profile*.

---

## 2. Pipeline Overview

```
35,857 chapter vectors (768-dim)
    │
    ▼
UMAP: 768 → 20 dims          (clustering space)
UMAP: 768 → 2 dims           (visualisation only)
    │
    ▼
HDBSCAN on 20-dim space
    │
    ├── hard cluster label per chapter
    ├── soft membership vector per chapter  (K-dim probability distribution)
    │
    ▼
Per-video compositional profile
    = mean of its chapters' soft membership vectors
    dominant_cluster = argmax(profile)
```

---

## 3. Dimensionality Reduction: UMAP

**UMAP** (Uniform Manifold Approximation and Projection, McInnes et al. 2018) is a non-linear dimensionality reduction algorithm. It learns a low-dimensional representation of the data that preserves the topological structure of the high-dimensional space — points that are neighbours in 768 dimensions remain neighbours in the reduced space.

UMAP was applied twice, both times from the original 768-dimensional embeddings:

- **768 → 20 dimensions**: the space on which HDBSCAN clusters. 20 dimensions retain substantially more structure than 2, while compressing the space enough for density-based clustering to be effective. Running HDBSCAN directly on the 768-dimensional embeddings was deliberately avoided: in very high-dimensional spaces, a mathematical phenomenon known as distance concentration causes all pairwise distances to become approximately equal, eliminating the density variation that HDBSCAN requires to detect cluster boundaries. UMAP first estimates the lower-dimensional manifold on which the embeddings actually reside — text embeddings do not fill 768-dimensional space uniformly — then maps it to a space where Euclidean distances are geometrically meaningful. HDBSCAN then operates on a well-conditioned input rather than a distance-degenerate one.
- **768 → 2 dimensions**: a 2-D projection used only for the scatter plot visualisation, produced independently from the 20-dimensional reduction. The 20-dimensional clustering space was optimised with `min_dist=0.0` (see parameter table), which packs cluster members as tightly as possible at the expense of topological fidelity. Compressing those already-distorted coordinates a second time (20 → 2) would amplify that distortion and produce a visualisation with artificially crisp, well-separated blobs. Running the visualisation reduction independently from the original 768 dimensions gives a layout that more faithfully represents the manifold structure of the corpus.

### UMAP parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `metric` | `cosine` | Text embeddings from sentence-transformers are L2-normalised. Cosine distance measures the angle between vectors and is invariant to vector magnitude, making it the appropriate similarity measure for this embedding space. |
| `n_neighbors` | 30 | Controls the balance between local and global structure. Each point's neighbourhood is estimated using its 30 nearest neighbours. A larger value causes UMAP to give more weight to the global topology of the corpus; a smaller value emphasises fine-grained local structure. The value 30 was chosen to capture broad topical groupings rather than per-creator micro-clusters. |
| `min_dist` | 0.0 | The minimum distance between points in the low-dimensional embedding. Setting this to 0.0 allows UMAP to pack cluster members as tightly as possible, which improves HDBSCAN's ability to detect density peaks. A non-zero value would spread the layout for aesthetic purposes but would obscure cluster boundaries. |
| `n_components` | 20 (clustering) / 2 (viz) | Dimensionality of the output space. |
| `random_state` | 42 | **Reproducibility seed.** UMAP initialises the low-dimensional positions randomly before optimising. Without a fixed seed, two runs on identical data produce different (though topologically equivalent) layouts. Setting `random_state=42` makes every run produce bit-identical output, which is necessary for a published result. The trade-off is that seeding forces single-threaded execution, making UMAP slower than an unseeded run. |

---

## 4. Clustering: HDBSCAN

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise, Campello et al. 2013) identifies clusters as regions of high point density separated by lower-density regions. Unlike k-means, it does not require the number of clusters to be specified in advance, and it handles clusters of arbitrary shape. Points in sparse regions are labelled as noise rather than forced into a cluster.

HDBSCAN was applied to the 20-dimensional UMAP output using Euclidean distance. The switch from cosine (used in UMAP) to Euclidean (used in HDBSCAN) is deliberate and correct: UMAP encodes the cosine-neighbour relationships into the geometry of the low-dimensional embedding, so Euclidean distance in that space faithfully approximates the original cosine similarity structure. HDBSCAN's internal space-partitioning data structures (KD-tree, Ball-tree) do not natively support cosine distance, and re-applying cosine on UMAP output coordinates would be geometrically inappropriate.

### HDBSCAN parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_cluster_size` | 150 | The minimum number of chapters required to form a cluster. Groups smaller than this are either absorbed into a neighbouring cluster or labelled as noise. This was set to 150 (out of 35,857 chapters, approximately 0.4% of the corpus) to suppress micro-clusters from single creators while allowing genuine niche content categories to emerge. |
| `min_samples` | 10 | Controls the conservativeness of the noise-labelling decision. A point is considered a core point (unambiguously inside a cluster) only if it has at least `min_samples` neighbours within its core distance. Higher values make HDBSCAN more conservative — it labels more points as noise and produces tighter, more homogeneous clusters. Lower values are more permissive. |
| `prediction_data` | `True` | Instructs HDBSCAN to retain intermediate data structures after fitting so that soft membership vectors can be extracted. This has no effect on the hard cluster assignments. |
| `metric` | `euclidean` | See rationale above. |

### Noise reassignment

HDBSCAN labelled 1,379 chapters (3.8% of the corpus) as noise (label −1). These points were not discarded; each was reassigned to its nearest cluster centroid in the 20-dimensional UMAP space (Euclidean distance). The original noise label is preserved in the `chapter_cluster_raw` column of `chapter_clusters.parquet` so the noise share remains auditable.

---

## 5. Soft Membership Vectors

A *hard* cluster assignment places each chapter into exactly one topic. A *soft* membership vector gives each chapter a probability distribution over all K discovered clusters, reflecting the degree to which it belongs to each.

Formally, for chapter $i$, the soft membership vector is $\mathbf{m}_i \in [0,1]^K$ with $\sum_k m_{ik} = 1$. These are extracted from HDBSCAN's condensed cluster tree using `hdbscan.all_points_membership_vectors()`. Points near a cluster core receive near-unit probability for that cluster; points in transitional regions between clusters receive split probabilities.

### Per-video compositional profile

For a video with $N$ chapters, its compositional profile is:

$$\mathbf{p}_{\text{video}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{m}_i$$

This is a K-dimensional vector (K = 10 in this run) that describes how the video distributes across topics. For example, a video with profile `[0.02, 0.0, 0.71, 0.0, 0.0, 0.0, 0.0, 0.27, 0.0, 0.0]` is predominantly cooking content (cluster 2) with a significant share of cosmetics review content (cluster 7).

The `dominant_cluster` column in `video_profiles.parquet` is simply $\text{argmax}(\mathbf{p}_{\text{video}})$ and provides a single discrete topic label per video for cross-tabulations. For regression, the full profile vector is more informative.

---

## 6. Results

10 clusters were discovered. Cluster sizes are chapter-level counts; the dominant-cluster distribution shows how many videos are primarily associated with each topic.

| ID | Label | Chapters | Dominant-topic videos | Confidence |
|----|-------|----------|-----------------------|------------|
| 0 | Personal Vlogs and Creator Updates | 30,936 | — (see note) | High |
| 1 | Home Fitness and Pilates Workouts | 175 | | High |
| 2 | Food Recipes and Refreshing Drinks | 269 | | High |
| 3 | Home Cleaning and Appliance Maintenance | 210 | | High |
| 4 | Music and Audio Cue Transcriptions | 202 | | High |
| 5 | Makeup Tutorials and Beauty Looks | 183 | | High |
| 6 | Cinema, Animation, and Literature | 633 | | Medium |
| 7 | Gift Ideas, Unboxings, and Toy Reviews | 527 | | High |
| 8 | Cosmetics Reviews and Product Recommendations | 816 | | High |
| 9 | Entertainment Challenges and Creative Sketches | 1,906 | | Medium |

Labels were assigned by Gemini 2.5 Flash (Vertex AI) presented with the 3 nearest-centroid transcript excerpts per cluster.

### Note on cluster 0 (dominant cluster)

Cluster 0 contains 86% of all chapters. This is a known behaviour of HDBSCAN on corpora with one dominant, diffuse category: the algorithm correctly identifies that the bulk of the data occupies a single broad region of the embedding space (general lifestyle/vlog content) while separating tighter, more distinctive niches into smaller clusters. This is not a failure of the method — it reflects the actual composition of the corpus, which is dominated by general-purpose creator content.

For modelling purposes, cluster 0 functions as the reference category. The soft membership profile is a more nuanced representation: even within cluster 0, videos carry non-zero probability on other clusters (e.g., a lifestyle video with a cooking segment will have elevated probability on cluster 2), and these fractional memberships are the features that enter the regression.

### Note on cluster 4 (artefact)

Cluster 4 consists exclusively of automated transcription placeholders (`[Musica]`, `[Applausi]`) produced by the ASR system when no intelligible speech was present. These chapters carry no semantic content. They are excluded from all downstream regression analyses. Their presence in the embedding space as a distinct cluster confirms that the model correctly separated speech content from non-speech segments.

---

## 7. Output Schema

### `chapter_clusters.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | YouTube video identifier |
| `chapter_index` | int | Chapter position within video (0-indexed) |
| `is_short` | bool | Whether the video is classified as a YouTube Short |
| `chapter_cluster` | int | Hard cluster label (noise reassigned to nearest centroid) |
| `chapter_cluster_raw` | int | Original HDBSCAN label (−1 = noise) |
| `umap_vector` | list[float] | 20-dimensional UMAP coordinates in clustering space |
| `umap_x`, `umap_y` | float | 2-D UMAP coordinates for visualisation |

### `video_profiles.parquet`
| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | YouTube video identifier |
| `is_short` | bool | |
| `n_chapters` | int | Number of semantic chapters in this video |
| `dominant_cluster` | int | argmax of the compositional profile |
| `profile` | list[float] | K-dimensional soft membership profile (sums to 1) |
