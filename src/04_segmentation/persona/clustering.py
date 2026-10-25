"""Clustering-based taxonomy discovery.

``PathwayBClusterer`` is the ALTERNATIVE to LLM consolidation: it clusters the
stratified Stage-1 sample (UMAP + HDBSCAN), then Gemini labels each cluster into
the identical 5-key ``final_taxonomy`` schema Stage 2 consumes. It runs BEFORE
Stage 2.

``LegacyStage3`` ports the original post-Stage-2 macro-clustering VIEW (clusters
classified users by their micro-persona label). It is optional and writes its own
``macro_persona_names.json`` / ``user_macro_personas.parquet`` - it does NOT feed
Stage 2.
"""
from __future__ import annotations

import json
import os

from .batch import BatchClient, strip_fences
from .config import PipelineConfig

FINAL_TAXONOMY_KEYS = ["codename", "label", "description", "quantitative_signals", "example_comments"]

PB_NAMING_SYSTEM_TMPL = (
    "You are a strategic audience analyst for an Italian Instagram influencer agency.\n"
    "Commenters were clustered into behavioural macro-segments (UMAP + HDBSCAN). Each cluster is\n"
    "described by mean behavioural stats, a dominant room-vibe mix, and representative comments.\n"
    "Turn EACH cluster into ONE audience persona. Personas must be MUTUALLY EXCLUSIVE.\n"
    "Return AT MOST {target} personas (one per cluster), ordered by audience share.\n"
    "For each persona output exactly these 5 keys: codename (UPPER_SNAKE_CASE), label (short Title Case),\n"
    "description (1-2 sentences), quantitative_signals (3-5 distinguishing behavioural signals as strings),\n"
    "example_comments (2-3 verbatim fragments drawn from the cluster's sample comments).\n"
    "Output ONLY a valid JSON array of objects with exactly those 5 keys. No preamble, no markdown fences."
)


class PathwayBClusterer:
    def __init__(self, config: PipelineConfig, batch: BatchClient, selected_numeric_features: list):
        self.config = config
        self.batch = batch
        self.numeric_features = list(selected_numeric_features)

    # ---- 1) working frame: the stratified Stage-1 sample ----
    def build_frame(self, user_features, stage1=None, ig_comments=None):
        import pandas as pd

        cfg = self.config
        path = cfg.stage1_sample_users_path
        if os.path.exists(path):
            sample_ids = set(pd.read_parquet(path)["author_id"].astype(str))
            pb_df = user_features[user_features["author_id"].astype(str).isin(sample_ids)].copy()
            print(f"Loaded Stage-1 sample users from {path}")
        else:
            print("stage1 sample file missing - drawing a fresh stratified sample live.")
            if stage1 is None:
                raise RuntimeError("sample file missing and no Stage1Discovery provided for live fallback.")
            pb_df = stage1.stratified_user_sample(user_features, ig_comments=ig_comments)
        pb_df = pb_df.reset_index(drop=True)
        print(f"Pathway B working set: {len(pb_df):,} users (the Stage-1 stratified sample).")
        return pb_df

    # ---- 2) feature matrix ----
    def build_matrix(self, pb_df):
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        feats = self.numeric_features
        print(f"Numeric features ({len(feats)}): {feats}")
        num_data = pb_df[feats].fillna(pb_df[feats].median())
        x_num = StandardScaler().fit_transform(num_data)
        if "dominant_room_vibe" in pb_df.columns and pb_df["dominant_room_vibe"].nunique() > 1:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            x_vibe = ohe.fit_transform(pb_df[["dominant_room_vibe"]])
            print(f"OHE dominant_room_vibe dims: {x_vibe.shape[1]} -> {ohe.categories_[0].tolist()}")
            X = np.hstack([x_num, x_vibe])
        else:
            print("dominant_room_vibe: single value or absent - numeric-only matrix.")
            X = x_num
        print(f"Pathway B feature matrix: {X.shape}  (rows=sample users, cols=numeric+OHE)")
        return X

    # ---- 3) UMAP ----
    def umap_embed(self, X):
        import umap

        cfg = self.config
        print(f"Running UMAP ({cfg.umap_n_components_cluster}-D) on {len(X):,} sample users ...")
        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components_cluster,
            metric=cfg.umap_metric,
            random_state=cfg.umap_random_state,
            low_memory=True,
            verbose=True,
        )
        emb = reducer.fit_transform(X)
        print(f"UMAP embedding: {emb.shape}")
        return emb

    # ---- 4) HDBSCAN auto-tuned toward target ----
    def cluster(self, emb):
        import hdbscan

        cfg = self.config
        target = cfg.pathway_b_target_clusters

        def near_target(e):
            best = None
            for frac in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
                mcs = max(5, int(len(e) * frac))
                lab = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=cfg.hdbscan_min_samples,
                    cluster_selection_method=cfg.hdbscan_cluster_selection,
                    cluster_selection_epsilon=cfg.hdbscan_cluster_epsilon,
                ).fit_predict(e)
                k = len(set(lab[lab >= 0]))
                if best is None or abs(k - target) < abs(best[1] - target):
                    best = (mcs, k, lab)
                if k <= target:
                    break
            return best

        mcs, k, labels = near_target(emb)
        unique = sorted(set(labels[labels >= 0]))
        n_noise = int((labels == -1).sum())
        print(
            f"Pathway B: min_cluster_size={mcs} -> {k} clusters | "
            f"noise={n_noise:,} ({100*n_noise/len(labels):.1f}%)"
        )
        for c in unique:
            cnt = int((labels == c).sum())
            print(f"   Cluster {c:>2}: {cnt:>6,} users ({100*cnt/len(labels):.1f}%)")
        return labels, unique

    # ---- 5) per-cluster summaries (no persona_codename/justification) ----
    def build_cluster_summary(self, cluster_id, df):
        cfg = self.config
        sub = df[df["macro_cluster"] == cluster_id]
        n = len(sub)
        stat_cols = [c for c in [
            "total_comments", "activity_span_days", "mean_hours_to_comment",
            "pct_comments_under_1h", "reply_ratio", "mean_word_count",
            "emoji_usage_rate", "question_rate", "exclamation_rate",
        ] if c in sub.columns]
        mean_stats = sub[stat_cols].mean().round(3).to_dict()
        vibe_mix = {}
        if "dominant_room_vibe" in sub.columns:
            vibe_mix = (
                sub["dominant_room_vibe"].value_counts(normalize=True)
                .mul(100).round(1).head(5).to_dict()
            )
        samples = []
        if "top_comments_sample" in sub.columns:
            picks = sub["top_comments_sample"].dropna()
            if len(picks):
                picks = picks.sample(
                    n=min(cfg.max_sample_users_per_cluster, len(picks)), random_state=42
                )
            for txt in picks:
                frags = [f.strip() for f in str(txt).split("|||") if f.strip()]
                samples.extend(frags[:2])
                if len(samples) >= 30:
                    break
        return {
            "cluster_id": int(cluster_id),
            "n_users": n,
            "pct_audience": round(100 * n / max(len(df[df["macro_cluster"] >= 0]), 1), 1),
            "mean_behavioral_stats": mean_stats,
            "dominant_room_vibe_mix": vibe_mix,
            "sample_comments": samples[:30],
        }

    # ---- 6) LLM-label each cluster, write final_taxonomy ----
    def label_and_write(self, pb_df, unique_clusters, summaries):
        from google.genai import types

        cfg = self.config

        def block(s):
            stats = "  ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(s["mean_behavioral_stats"].items())[:7]
            )
            vibe = "  ".join(f"{k}: {v}%" for k, v in s["dominant_room_vibe_mix"].items())
            comments = "\n    - ".join(s["sample_comments"][:8])
            return (
                f"CLUSTER {s['cluster_id']}  (n={s['n_users']:,}, {s['pct_audience']}% of clustered users)\n"
                f"  Mean behavioural stats: {stats}\n"
                f"  Dominant room-vibe mix: {vibe}\n"
                f"  Sample comments:\n    - {comments}"
            )

        system = PB_NAMING_SYSTEM_TMPL.format(target=cfg.target_personas)
        prompt = system + "\n\n=== CLUSTER DATA ===\n\n" + "\n\n".join(block(s) for s in summaries)
        config_obj = types.GenerateContentConfig(
            temperature=0.0, top_p=1.0, max_output_tokens=8192,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
        )
        print(f"Labelling {len(summaries)} clusters via {cfg.model_stage3_naming} ...")
        resp = self.batch.client.models.generate_content(
            model=cfg.model_stage3_naming, contents=prompt, config=config_obj
        )
        text = strip_fences(resp.text or "")
        try:
            personas = json.loads(text)
        except Exception as e:  # noqa: BLE001
            print("Could not parse LLM output:", str(e)[:120], "\n", text[:400])
            personas = []
        personas = [
            {k: p.get(k, "" if k in ("codename", "label", "description") else []) for k in FINAL_TAXONOMY_KEYS}
            for p in personas if isinstance(p, dict)
        ][: cfg.target_personas]
        print(f"{len(personas)} personas from Pathway B:")
        for p in personas:
            print(f"   {str(p['codename']):<28} | {p['label']}")

        # write taxonomy.json (same contract as Pathway A) + provenance + cluster maps
        if os.path.exists(cfg.taxonomy_json_path):
            data = json.load(open(cfg.taxonomy_json_path, encoding="utf-8"))
        else:
            data = {"status": "PENDING_HUMAN_REVIEW", "raw_candidates": []}
        data["final_taxonomy"] = personas
        data["status"] = "PENDING_HUMAN_REVIEW"
        data["pathway"] = "B_CLUSTER"
        data["instructions"] = (
            "Pathway B (UMAP+HDBSCAN) taxonomy. Review/merge, ensure MECE, set status='APPROVED' "
            "before Stage 2. Set CONSOLIDATION_PATHWAY='B_CLUSTER' so the Pathway A wrapper does not overwrite this."
        )
        with open(cfg.taxonomy_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Written -> {cfg.taxonomy_json_path} (pathway=B_CLUSTER).")

        cluster_map = {
            int(cid): personas[i]["codename"]
            for i, cid in enumerate(unique_clusters) if i < len(personas)
        }
        with open(cfg.cluster_persona_map_path, "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, ensure_ascii=False, indent=2)
        pb_df[["author_id", "macro_cluster"]].to_parquet(cfg.pathway_b_assignments_path, index=False)
        print(
            f"Cluster->persona map -> {cfg.cluster_persona_map_path}; "
            f"assignments -> {cfg.pathway_b_assignments_path}"
        )
        return personas

    # ---- orchestration ----
    def run(self, user_features, stage1=None, ig_comments=None):
        pb_df = self.build_frame(user_features, stage1=stage1, ig_comments=ig_comments)
        X = self.build_matrix(pb_df)
        emb = self.umap_embed(X)
        labels, unique = self.cluster(emb)
        pb_df["macro_cluster"] = labels
        summaries = [self.build_cluster_summary(c, pb_df) for c in unique]
        print(f"Built summaries for {len(summaries)} clusters.")
        return self.label_and_write(pb_df, unique, summaries)


class LegacyStage3:
    """Optional post-Stage-2 macro-clustering VIEW (clusters classified users by
    their Stage-2 micro-persona label). Does not feed Stage 2."""

    STAGE3_NAMING_SYSTEM = (
        "You are a strategic audience analyst for an Italian influencer marketing agency.\n"
        "You have clustered Instagram commenters into macro audience segments using UMAP + HDBSCAN.\n"
        "Each cluster is described by its dominant micro-persona mix, mean behavioral stats, "
        "and representative comment samples.\n\n"
        "For EACH cluster produce ONE macro persona with exactly these fields:\n"
        "  cluster_id        : (integer - echo back exactly as provided)\n"
        "  codename          : UPPER_SNAKE_CASE concise name (e.g. PASSIONATE_LOYALISTS)\n"
        "  label             : Short Title Case marketing label (3-5 words)\n"
        "  description       : 2-3 sentences - who they are, how and why they engage\n"
        "  key_traits        : JSON list of 4-6 behavioural / attitudinal bullet strings\n"
        "  marketing_insight : 1 actionable sentence for the brand or agency\n\n"
        "Output ONLY a valid JSON array of objects with exactly those 6 keys. "
        "No preamble, no markdown fences, no extra keys."
    )

    def __init__(self, config: PipelineConfig, batch: BatchClient, selected_numeric_features: list):
        self.config = config
        self.batch = batch
        self.numeric_features = list(selected_numeric_features)

    def run(self, user_features, plot=False):
        import numpy as np
        import pandas as pd
        import umap
        import hdbscan
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        cfg = self.config
        if not os.path.exists(cfg.results_path):
            raise FileNotFoundError(
                f"Stage 2 results not found at '{cfg.results_path}'. Run retrieve-stage2 first."
            )
        results_df = pd.read_parquet(cfg.results_path)
        results_df["author_id"] = results_df["author_id"].astype(str)
        print(
            f"Stage 2 results : {len(results_df):,} users  |  "
            f"classified: {results_df['persona_codename'].notna().sum():,}"
        )
        extra_cols = [
            "author_id", "unique_posts_commented", "total_replies_made",
            "pct_comments_under_24h", "emoji_usage_rate", "question_rate",
            "exclamation_rate", "mean_mention_count", "post_concentration_ratio",
            "top_comments_sample",
        ]
        available = [c for c in extra_cols if c in user_features.columns]
        feat_ext = user_features[available].copy()
        feat_ext["author_id"] = feat_ext["author_id"].astype(str)
        stage3_df = results_df.merge(feat_ext, on="author_id", how="left")
        stage3_df = stage3_df[
            stage3_df["persona_codename"].notna()
            & (stage3_df["persona_codename"] != "CLASSIFICATION_ERROR")
        ].copy().reset_index(drop=True)
        print(f"Stage 3 working set: {len(stage3_df):,} users with valid micro-persona labels")

        feats = self.numeric_features
        num_data = stage3_df[feats].fillna(stage3_df[feats].median())
        x_num = StandardScaler().fit_transform(num_data)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        x_ohe = ohe.fit_transform(stage3_df[["persona_codename"]])
        if "dominant_room_vibe" in stage3_df.columns and stage3_df["dominant_room_vibe"].nunique() > 1:
            ohe_v = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            x_vibe = ohe_v.fit_transform(stage3_df[["dominant_room_vibe"]])
            X = np.hstack([x_num, x_ohe * 2.0, x_vibe])
        else:
            X = np.hstack([x_num, x_ohe * 2.0])
        print(f"Feature matrix: {X.shape}")

        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors, min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components_cluster, metric=cfg.umap_metric,
            random_state=cfg.umap_random_state, low_memory=True, verbose=True,
        )
        x_umap = reducer.fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.stage3_hdbscan_min_cluster_size,
            min_samples=cfg.stage3_hdbscan_min_samples,
            cluster_selection_method=cfg.hdbscan_cluster_selection,
            cluster_selection_epsilon=cfg.hdbscan_cluster_epsilon,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(x_umap)
        stage3_df["macro_cluster"] = labels
        unique = sorted(set(labels[labels >= 0]))
        n_noise = int((labels == -1).sum())
        print(f"HDBSCAN: {len(unique)} clusters | noise {n_noise:,} ({100*n_noise/len(labels):.1f}%)")

        summaries = [self._summary(c, stage3_df) for c in unique]
        macro = self._name_clusters(summaries, unique, n_noise)
        self._save(stage3_df, macro)
        return macro

    def _summary(self, cluster_id, df):
        cfg = self.config
        sub = df[df["macro_cluster"] == cluster_id].copy()
        n = len(sub)
        persona_dist = (
            sub["persona_codename"].value_counts(normalize=True).mul(100).round(1).head(5).to_dict()
        )
        stat_cols = [c for c in [
            "total_comments", "activity_span_days", "mean_hours_to_comment",
            "pct_comments_under_1h", "reply_ratio", "mean_word_count",
            "emoji_usage_rate", "question_rate", "exclamation_rate",
        ] if c in sub.columns]
        mean_stats = sub[stat_cols].mean().round(3).to_dict()
        samples = []
        if "top_comments_sample" in sub.columns:
            for txt in sub["top_comments_sample"].dropna().sample(
                n=min(cfg.max_sample_users_per_cluster, n), random_state=42
            ):
                frags = [f.strip() for f in str(txt).split("|||") if f.strip()]
                samples.extend(frags[:2])
                if len(samples) >= 30:
                    break
        justifications = []
        if "justification" in sub.columns:
            for j in sub["justification"].dropna().sample(n=min(10, n), random_state=42):
                justifications.append(str(j)[:200])
        return {
            "cluster_id": cluster_id, "n_users": n,
            "pct_audience": round(100 * n / max(len(df[df["macro_cluster"] >= 0]), 1), 1),
            "dominant_micro_personas": persona_dist, "mean_behavioral_stats": mean_stats,
            "sample_comments": samples[:30], "sample_justifications": justifications[:10],
        }

    def _name_clusters(self, summaries, unique, n_noise):
        from google.genai import types as genai_types

        cfg = self.config
        blocks = []
        for s in summaries:
            stats = "  ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(s["mean_behavioral_stats"].items())[:7]
            )
            personas = "  ".join(f"{p}: {pct}%" for p, pct in s["dominant_micro_personas"].items())
            comments = "\n    - ".join(s["sample_comments"][:8])
            justs = s.get("sample_justifications", [])
            just_str = ("\n  LLM justification samples: " + " | ".join(justs[:3])) if justs else ""
            blocks.append(
                f"CLUSTER {s['cluster_id']}  (n={s['n_users']:,}, {s['pct_audience']}% of clustered audience)\n"
                f"  Dominant micro-personas: {personas}\n"
                f"  Mean behavioral stats:   {stats}{just_str}\n"
                f"  Sample comments:\n    - {comments}"
            )
        prompt = self.STAGE3_NAMING_SYSTEM + "\n\n=== CLUSTER DATA ===\n\n" + "\n\n".join(blocks)
        cfg_obj = genai_types.GenerateContentConfig(
            temperature=0.0, top_p=1.0, max_output_tokens=4096,
            response_mime_type="application/json",
            thinking_config=genai_types.ThinkingConfig(thinking_budget=2048),
        )
        print(f"Naming {len(summaries)} macro clusters via {cfg.model_stage3_naming} ...")
        resp = self.batch.client.models.generate_content(
            model=cfg.model_stage3_naming, contents=prompt, config=cfg_obj
        )
        raw = strip_fences(resp.text or "")
        try:
            macro = json.loads(raw)
        except json.JSONDecodeError as e:
            print("JSON parse error:", e, "\n", raw[:800])
            macro = []
        if not isinstance(macro, list):
            macro = []
        with open(cfg.cluster_names_path, "w", encoding="utf-8") as f:
            json.dump(
                {"status": "COMPLETE", "n_clusters": len(unique),
                 "noise_users": int(n_noise), "macro_personas": macro},
                f, ensure_ascii=False, indent=2,
            )
        print(f"Macro persona definitions saved -> {cfg.cluster_names_path}")
        return macro

    def _save(self, stage3_df, macro):
        cfg = self.config
        id_to_macro = {}
        for mp in macro:
            cid = mp.get("cluster_id")
            if cid is not None:
                id_to_macro[int(cid)] = {
                    "macro_persona_codename": mp.get("codename", f"CLUSTER_{cid}"),
                    "macro_persona_label": mp.get("label", ""),
                }
        stage3_df["macro_persona_codename"] = stage3_df["macro_cluster"].map(
            lambda c: id_to_macro.get(c, {}).get("macro_persona_codename", "NOISE")
        )
        stage3_df["macro_persona_label"] = stage3_df["macro_cluster"].map(
            lambda c: id_to_macro.get(c, {}).get("macro_persona_label", "Noise / Unclustered")
        )
        output_cols = [c for c in [
            "author_id", "persona_codename", "confidence",
            "macro_cluster", "macro_persona_codename", "macro_persona_label",
        ] if c in stage3_df.columns]
        stage3_df[output_cols].to_parquet(cfg.macro_persona_path, index=False)
        print(f"Macro persona assignments saved -> {cfg.macro_persona_path}")
