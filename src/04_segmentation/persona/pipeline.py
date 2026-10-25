"""PersonaPipeline facade: wires every component together and exposes one method
per CLI subcommand. Mirrors the notebook's run-order and submit/retrieve split.

Data (ig_comments, user_features) is loaded lazily and cached on the instance, so
a single process invocation does the minimum work for the requested step.
"""
from __future__ import annotations

from .batch import BatchClient
from .clustering import LegacyStage3, PathwayBClusterer
from .config import PipelineConfig
from .data import DataLoader
from .features import FeatureEngineer
from .media import MediaContextBuilder
from .stages import (
    PersonaMapExporter,
    Stage1Discovery,
    Stage2Classifier,
    TaxonomyConsolidator,
    load_approved_taxonomy,
    validate_classification_output,
)


class PersonaPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        print("Configuration loaded.")
        print(self.config.summary())
        self._batch: BatchClient | None = None
        self._ig_comments = None
        self._media_index = None
        self._user_features = None
        self._selected_features: list | None = None
        self._media: MediaContextBuilder | None = None

    # ----------------------------- lazy deps -----------------------------
    @property
    def batch(self) -> BatchClient:
        if self._batch is None:
            self._batch = BatchClient(self.config)
        return self._batch

    def _ensure_data(self):
        if self._ig_comments is None:
            self._ig_comments, self._media_index = DataLoader(self.config).load()
        return self._ig_comments, self._media_index

    def _ensure_features(self):
        if self._user_features is None:
            ig_comments, media_index = self._ensure_data()
            fe = FeatureEngineer(self.config)
            self._ig_comments, self._user_features = fe.build(ig_comments, media_index)
            self._selected_features = fe.select_numeric_features(self._user_features)
        return self._user_features

    def _ensure_media(self) -> MediaContextBuilder:
        if self._media is None:
            self._ensure_features()
            self._media = MediaContextBuilder(self.config)
            self._media.build(self._ig_comments, self._media_index, self._user_features)
        return self._media

    # ------------------------------ commands -----------------------------
    def connectivity_test(self):
        self.batch.connectivity_test()

    def submit_stage1(self):
        uf = self._ensure_features()
        media = self._ensure_media()
        s1 = Stage1Discovery(self.config, self.batch, media)
        return s1.submit(uf, ig_comments=self._ig_comments)

    def retrieve_stage1(self):
        media = self._ensure_media()  # not strictly needed, but keeps deps consistent
        s1 = Stage1Discovery(self.config, self.batch, media)
        return s1.retrieve()

    def pathway_b(self):
        uf = self._ensure_features()
        media = self._ensure_media()
        s1 = Stage1Discovery(self.config, self.batch, media)
        pb = PathwayBClusterer(self.config, self.batch, self._selected_features)
        return pb.run(uf, stage1=s1, ig_comments=self._ig_comments)

    def consolidate(self, method="llm", top_k=15):
        self._ensure_data()  # ig_comments not needed, but keeps order; consolidation reads taxonomy.json
        tc = TaxonomyConsolidator(self.config, self.batch)
        if method == "draft":
            return tc.draft_from_candidates(top_k=top_k)
        final = tc.llm_consolidate()
        tc.write_pathway_a_wrapper(final)
        return final

    def submit_stage2(self):
        uf = self._ensure_features()
        media = self._ensure_media()
        taxonomy = load_approved_taxonomy(self.config.taxonomy_json_path)  # hard stop if not APPROVED
        s2 = Stage2Classifier(self.config, self.batch, media)
        return s2.submit(uf, taxonomy)

    def retrieve_stage2(self):
        uf = self._ensure_features()
        media = self._ensure_media()
        s2 = Stage2Classifier(self.config, self.batch, media)
        return s2.retrieve(uf)

    def export_map(self):
        return PersonaMapExporter(self.config).export()

    def validate(self):
        return validate_classification_output(self.config.results_path)

    def stage3(self):
        uf = self._ensure_features()
        s3 = LegacyStage3(self.config, self.batch, self._selected_features)
        return s3.run(uf)

    # ----------------------------- orchestrator --------------------------
    def run(self, mode=None):
        """Submit-only driver, mirroring the notebook run_pipeline.

        SAMPLE -> submit Stage 1 (then pause for human review/approval).
        FULL   -> submit Stage 2 (needs an APPROVED taxonomy).
        ALL    -> submit Stage 1 then Stage 2 (Stage 2 still needs APPROVED).
        """
        mode = mode or self.config.pipeline_mode
        print(f"\n{'#'*60}\n  SHOW REEL PERSONA PIPELINE (BATCH submit) - MODE: {mode}\n{'#'*60}")
        if mode in ("SAMPLE", "ALL"):
            self.submit_stage1()
            if mode == "SAMPLE":
                return None
        if mode in ("FULL", "ALL"):
            uf = self._ensure_features()
            media = self._ensure_media()
            taxonomy = load_approved_taxonomy(self.config.taxonomy_json_path)
            Stage2Classifier(self.config, self.batch, media).submit(uf, taxonomy)
        return None
