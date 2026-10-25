"""Central configuration for the persona pipeline.

A single ``PipelineConfig`` dataclass replaces the notebook's CONFIG cell. All
derived paths are computed in ``__post_init__`` so callers only set the few
knobs that actually change between runs (platform, sample size, caps...).

The defaults mirror the original notebook exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ----------------------------- platform ------------------------------
    platform: str = "instagram"            # instagram | youtube | tiktok | facebook

    # ----------------------------- GCP / Vertex --------------------------
    gcp_project_id: str = "gen-lang-client-0792749758"
    gcp_location: str = "us-central1"
    gcs_bucket: str = "<GCS_BUCKET>"

    # --------------------------- models / determinism --------------------
    model_stage1_exploratory: str = "gemini-2.5-flash"
    model_stage2_classify: str = "gemini-2.5-pro"
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens_stage1: int = 8192
    max_output_tokens_stage2: int = 2048
    thinking_budget_stage1: int = 1024
    thinking_budget_stage2: int = 512

    # ---------------------------- pipeline mode --------------------------
    pipeline_mode: str = "SAMPLE"          # SAMPLE | FULL | ALL

    # --------------------------- sampling / batching ---------------------
    sample_n_users: int = 20000
    sample_seed: int = 42
    stage1_users_per_request: int = 3
    stage2_max_users: int | None = None

    # ---------------------- multimodal attachment caps -------------------
    max_media_posts_per_user: int = 4
    max_images_per_post: int = 3
    include_transcript: bool = True
    max_transcript_chars: int = 1500

    # ----------------------------- batch I/O -----------------------------
    batch_input_prefix: str = "persona_batch/input/"
    batch_output_prefix: str = "persona_batch/output/"
    poll_interval_seconds: int = 60

    # --------------------------- local artifacts -------------------------
    local_dir: str = "outputs"

    # ----------------------- stratified sampling -------------------------
    stratify_by_sentiment: bool = True
    sentiment_strata_col: str = "sentiment_cat"

    # ----------------------- consolidation pathway -----------------------
    consolidation_pathway: str = "A_LLM"   # "A_LLM" | "B_CLUSTER"
    target_personas: int = 12
    pathway_b_target_clusters: int = 12

    # ----------------------- Pathway B UMAP/HDBSCAN ----------------------
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_n_components_cluster: int = 15
    umap_metric: str = "euclidean"
    umap_random_state: int = 42
    hdbscan_min_samples: int = 10
    hdbscan_cluster_selection: str = "eom"
    hdbscan_cluster_epsilon: float = 0.0
    model_stage3_naming: str = "gemini-2.5-flash"
    max_sample_users_per_cluster: int = 30

    # ----------------------- legacy Stage 3 knobs ------------------------
    stage3_hdbscan_min_cluster_size: int = 2000
    stage3_hdbscan_min_samples: int = 100

    # -------- derived (filled in __post_init__; not user-set) ------------
    comments_ml_path: str = field(init=False)
    comments_llm_path: str = field(init=False)
    edges_replies_path: str = field(init=False)
    media_index_path: str = field(init=False)
    media_gcs_prefix: str = field(init=False)
    media_local_prefix: str = field(init=False)
    attach_media: bool = field(init=False)
    taxonomy_json_path: str = field(init=False)
    results_path: str = field(init=False)
    sentiment_path: str = field(init=False)
    post_vibes_path: str = field(init=False)
    persona_map_path: str = field(init=False)
    cluster_persona_map_path: str = field(init=False)
    user_features_cache_path: str = field(init=False)
    macro_persona_path: str = field(init=False)
    cluster_names_path: str = field(init=False)

    def __post_init__(self) -> None:
        b = self.gcs_bucket
        self.comments_ml_path = f"gs://{b}/Preped_Comments/comments_ml.parquet"
        self.comments_llm_path = f"gs://{b}/Preped_Comments/comments_llm.jsonl"
        self.edges_replies_path = f"gs://{b}/HeteroGraph/edges_replies_to.parquet"
        self.media_index_path = f"gs://{b}/ig_multimodal_final.parquet"
        self.media_gcs_prefix = "multimodal_dataset_fixed/"
        self.media_local_prefix = "multimodal_dataset_fixed"
        self.attach_media = self.platform == "instagram"

        ld = self.local_dir
        os.makedirs(ld, exist_ok=True)
        self.taxonomy_json_path = f"{ld}/taxonomy.json"
        self.results_path = f"{ld}/user_personas.parquet"
        self.sentiment_path = f"{ld}/sentiment_{self.platform}.parquet"
        self.post_vibes_path = f"{ld}/post_vibes_{self.platform}.parquet"
        self.persona_map_path = f"{ld}/persona_map_{self.platform}.parquet"
        self.cluster_persona_map_path = f"{ld}/cluster_persona_map_{self.platform}.json"
        self.user_features_cache_path = f"{ld}/user_features_{self.platform}_cache.parquet"
        self.macro_persona_path = f"{ld}/user_macro_personas.parquet"
        self.cluster_names_path = f"{ld}/macro_persona_names.json"

    # convenience derived paths used by stages
    @property
    def stage1_sample_users_path(self) -> str:
        return f"{self.local_dir}/stage1_sample_users_{self.platform}.parquet"

    @property
    def pathway_b_assignments_path(self) -> str:
        return f"{self.local_dir}/pathway_b_assignments_{self.platform}.parquet"

    def gen_config_dict(self, max_tokens: int, thinking_budget: int | None = None) -> dict:
        """Per-request generationConfig (camelCase REST keys) for each JSONL line."""
        cfg = {
            "temperature": self.temperature,
            "topP": self.top_p,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        }
        if thinking_budget is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        return cfg

    def summary(self) -> str:
        return (
            f"Platform: {self.platform}  (media context: "
            f"{'ON' if self.attach_media else 'OFF'})\n"
            f"Bucket:   gs://{self.gcs_bucket}\n"
            f"Mode:     {self.pipeline_mode}"
        )
