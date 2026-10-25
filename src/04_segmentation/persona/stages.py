"""LLM stages: Stage 1 discovery, taxonomy consolidation (Pathway A), Stage 2
classification, persona-map export, and the shared approved-taxonomy loader.

Ports the notebook's Stage 1 / Stage 1.5 / LLM-consolidation / Stage 2 / persona
map / load_approved_taxonomy cells. Pathway B (clustering) lives in clustering.py.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from .batch import BatchClient, strip_fences
from .config import PipelineConfig
from .media import MediaContextBuilder

STAGE1_SYSTEM_PROMPT = (
    "You are an expert community analyst for a major Italian influencer agency.\n"
    "Identify distinct audience persona archetypes from Instagram commenter behaviour.\n"
    "Each user profile carries quantitative metrics, a sample of their comments, AND the media\n"
    "(images / video frames + transcript) of the posts they engage with most.\n"
    "Use BOTH how they comment and WHAT content they engage with.\n"
    "For each recurring pattern output a candidate persona with: a short codename (e.g. SUPERFAN),\n"
    "a 1-sentence behavioural description, key quantitative signals, and 2-3 verbatim comment fragments.\n"
    "Output ONLY a valid JSON array. No preamble, no markdown fences.\n"
    'Schema: [{"codename": str, "description": str, "signals": [str], "examples": [str]}]'
)

FINAL_TAXONOMY_KEYS = ["codename", "label", "description", "quantitative_signals", "example_comments"]


def _coerce_persona(p: dict) -> dict:
    return {k: p.get(k, "" if k in ("codename", "label", "description") else []) for k in FINAL_TAXONOMY_KEYS}


# ============================ helpers (free fns) ===========================
def load_approved_taxonomy(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "APPROVED":
        raise ValueError(f"Taxonomy at '{path}' is not approved. Review and set status='APPROVED'.")
    taxonomy = data.get("final_taxonomy", [])
    if not taxonomy:
        raise ValueError("final_taxonomy is empty.")
    print(f"Approved taxonomy loaded: {len(taxonomy)} personas.")
    return taxonomy


# ============================== Stage 1 ====================================
class Stage1Discovery:
    def __init__(self, config: PipelineConfig, batch: BatchClient, media: MediaContextBuilder):
        self.config = config
        self.batch = batch
        self.media = media

    # ---- profile / request builders ----
    @staticmethod
    def format_user_profile(row) -> str:
        return (
            f"USER: {row['author_id']} | "
            f"Total comments: {int(row['total_comments'])} | "
            f"Unique posts: {int(row['unique_posts_commented'])} | "
            f"Activity span: {int(row['activity_span_days'])} days | "
            f"Avg hrs to comment: {row['mean_hours_to_comment']:.1f}h | "
            f"Early commenter (<1h): {row['pct_comments_under_1h']:.0%} | "
            f"Reply ratio: {row['reply_ratio']:.0%} | "
            f"Avg word count: {row['mean_word_count']:.0f} | "
            f"Emoji rate: {row['emoji_usage_rate']:.0%} | "
            f"Question rate: {row['question_rate']:.0%} | "
            f"Sample comments: {str(row.get('top_comments_sample', ''))[:400]}"
        )

    def build_line(self, group_df) -> dict:
        cfg = self.config
        parts = [{"text": STAGE1_SYSTEM_PROMPT + "\n\n--- USER PROFILES BATCH (with engaged-post media) ---"}]
        for _, row in group_df.iterrows():
            parts.append({"text": "\n" + self.format_user_profile(row)})
            parts.extend(self.media.build_user_media_parts(row["author_id"]))
        parts.append({"text": "\nIdentify all distinct behavioural archetypes in this batch. Output ONLY the JSON array."})
        return {
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": cfg.gen_config_dict(
                    cfg.max_output_tokens_stage1, cfg.thinking_budget_stage1
                ),
            }
        }

    # ---- stratified sampling ----
    def stratified_user_sample(self, user_features, ig_comments=None, n_sample=None, seed=None):
        import pandas as pd

        cfg = self.config
        n_sample = cfg.sample_n_users if n_sample is None else n_sample
        seed = cfg.sample_seed if seed is None else seed

        if not cfg.stratify_by_sentiment or not os.path.exists(cfg.sentiment_path):
            if cfg.stratify_by_sentiment:
                print(f"{cfg.sentiment_path} not found - falling back to PLAIN RANDOM sample.")
            return user_features.sample(
                n=min(n_sample, len(user_features)), random_state=seed
            ).reset_index(drop=True)

        sent = pd.read_parquet(cfg.sentiment_path)
        col = cfg.sentiment_strata_col
        if "author_id" in sent.columns:
            sa = sent[["author_id", col]].dropna().copy()
        else:
            if ig_comments is None:
                raise ValueError("sentiment file lacks author_id and no ig_comments given to join.")
            sa = (
                sent[["comment_id", col]].dropna()
                .merge(ig_comments[["comment_id", "author_id"]], on="comment_id", how="inner")
            )
        sa["author_id"] = sa["author_id"].astype(str)
        author_strata = (
            sa.groupby("author_id")[col]
            .agg(lambda x: x.value_counts().idxmax())
            .rename("author_sentiment")
        )
        uf = user_features.copy()
        uf["author_id"] = uf["author_id"].astype(str)
        uf = uf.merge(author_strata, on="author_id", how="left")
        uf["author_sentiment"] = uf["author_sentiment"].fillna("neutral")

        n = min(n_sample, len(uf))
        frac = n / len(uf) if len(uf) else 0.0
        parts = []
        for _, grp in uf.groupby("author_sentiment"):
            k = max(1, int(round(len(grp) * frac)))
            parts.append(grp.sample(n=min(k, len(grp)), random_state=seed))
        out = (
            pd.concat(parts)
            .sample(frac=1.0, random_state=seed)
            .head(n).reset_index(drop=True)
        )
        print(
            f"Stratified Stage-1 sample (user-level dominant {col}): "
            f"{len(out):,} users | strata mix (%):"
        )
        print(out["author_sentiment"].value_counts(normalize=True).mul(100).round(1).to_string())
        return out

    # ---- submit / retrieve ----
    def submit(self, user_features, ig_comments=None, n_sample=None, group_size=None, seed=None):
        from tqdm import tqdm

        cfg = self.config
        n_sample = cfg.sample_n_users if n_sample is None else n_sample
        group_size = cfg.stage1_users_per_request if group_size is None else group_size
        seed = cfg.sample_seed if seed is None else seed
        print(
            f"\n{'='*60}\nSTAGE 1 (batch submit) - Taxonomy Discovery\n"
            f"  Sample: {n_sample} users | {group_size}/request | "
            f"Model: {cfg.model_stage1_exploratory}\n{'='*60}"
        )
        sample_df = self.stratified_user_sample(
            user_features, ig_comments=ig_comments, n_sample=n_sample, seed=seed
        )
        sample_df[["author_id"]].to_parquet(cfg.stage1_sample_users_path, index=False)
        groups = [sample_df.iloc[i : i + group_size] for i in range(0, len(sample_df), group_size)]
        lines = [self.build_line(g) for g in tqdm(groups, desc="Build Stage 1 requests")]
        in_uri = self.batch.upload_to_gcs(
            self.batch.write_jsonl(lines, f"{cfg.local_dir}/stage1_input.jsonl"),
            cfg.batch_input_prefix + "stage1_input.jsonl",
        )
        out_uri = f"gs://{cfg.gcs_bucket}/{cfg.batch_output_prefix}stage1/"
        job = self.batch.submit_batch_job(in_uri, out_uri, cfg.model_stage1_exploratory)
        self.batch.record_batch_job("stage1", job, out_uri)
        print(f"\nStage 1 submitted ({len(lines)} requests). Safe to close the laptop.")
        print("   When it finishes, run:  retrieve-stage1   -> writes taxonomy.json for review")
        return job

    def save_taxonomy_for_review(self, candidates) -> None:
        cfg = self.config
        out = {
            "status": "PENDING_HUMAN_REVIEW",
            "instructions": (
                "Consolidate the candidates into a MECE taxonomy. Each final persona needs a "
                "unique 'codename' (plus 'label', 'description', 'quantitative_signals', "
                "'example_comments'). Set status='APPROVED' before Stage 2."
            ),
            "raw_candidates": candidates,
            "final_taxonomy": [],
        }
        with open(cfg.taxonomy_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Raw candidates saved -> {cfg.taxonomy_json_path}")
        print("   HUMAN ACTION: review, consolidate, set status='APPROVED'.")

    def retrieve(self, save=True):
        cfg = self.config
        job = self.batch.get_recorded_job("stage1")
        if job.state != self.batch.JobState.JOB_STATE_SUCCEEDED:
            print(f"Stage 1 not ready (state={job.state}). Re-run later.")
            return None
        candidates = []
        for t in self.batch.retrieve_response_texts(job):
            if not t:
                continue
            try:
                c = json.loads(strip_fences(t))
                if isinstance(c, list):
                    candidates.extend(c)
            except Exception as e:  # noqa: BLE001
                print("   parse error:", str(e)[:80])
        print(f"Stage 1 complete. Raw candidates: {len(candidates)}")
        if save:
            self.save_taxonomy_for_review(candidates)
        return candidates


# ===================== Taxonomy consolidation (Pathway A) ==================
class TaxonomyConsolidator:
    def __init__(self, config: PipelineConfig, batch: BatchClient):
        self.config = config
        self.batch = batch

    # --- string-normalised draft (Stage 1.5) ---
    def draft_from_candidates(self, top_k=15, min_count=1, write=True):
        cfg = self.config
        data = json.load(open(cfg.taxonomy_json_path, encoding="utf-8"))
        raw = [c for c in data.get("raw_candidates", []) if isinstance(c, dict) and c.get("codename")]
        if not raw:
            print("No raw_candidates - run/retrieve Stage 1 first.")
            return []

        def norm(c):
            c = re.sub(r"[^A-Z0-9 ]", " ", c.upper().replace("_", " "))
            c = re.sub(r"\s+", " ", c).strip()
            return c[4:].strip() if c.startswith("THE ") else c

        groups = defaultdict(list)
        for c in raw:
            groups[norm(c["codename"])].append(c)
        ranked = [
            (k, v)
            for k, v in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
            if len(v) >= min_count
        ][:top_k]

        def merge_list(cands, key, limit=6):
            seen, out = set(), []
            for c in cands:
                for item in c.get(key) or []:
                    s = str(item).strip()
                    if s and s.lower() not in seen:
                        seen.add(s.lower())
                        out.append(s)
            return out[:limit]

        draft = []
        for theme, cands in ranked:
            canon = Counter(
                c["codename"].strip().upper().replace(" ", "_") for c in cands
            ).most_common(1)[0][0]
            draft.append({
                "codename": canon,
                "label": theme.title(),
                "description": max((c.get("description", "") for c in cands), key=len, default=""),
                "quantitative_signals": merge_list(cands, "signals"),
                "example_comments": merge_list(cands, "examples"),
                "_n_candidates": len(cands),
            })
        covered = sum(d["_n_candidates"] for d in draft)
        print(
            f"{len(raw)} candidates -> {len(groups)} themes. Drafted top {len(draft)} "
            f"(cover {covered}/{len(raw)} = {100*covered/len(raw):.0f}% of candidates):"
        )
        for d in draft:
            print(f"  {d['codename']:<28} merged {d['_n_candidates']:>3} | {d['label']}")
        if write:
            data["final_taxonomy"] = draft
            with open(cfg.taxonomy_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\nDraft written to {cfg.taxonomy_json_path} -> final_taxonomy.")
            print("    Next: MERGE similar personas, trim to MECE, remove '_n_candidates', set status='APPROVED'.")
        return draft

    # --- one-shot LLM consolidation (Pathway A) ---
    def llm_consolidate(self, target_personas=None, model=None, write=True):
        from google.genai import types

        cfg = self.config
        target_personas = cfg.target_personas if target_personas is None else target_personas
        model = cfg.model_stage2_classify if model is None else model

        data = json.load(open(cfg.taxonomy_json_path, encoding="utf-8"))
        raw = [c for c in data.get("raw_candidates", []) if isinstance(c, dict) and c.get("codename")]
        if not raw:
            print("No raw_candidates - run/retrieve Stage 1 first.")
            return []
        items = [{
            "codename": c.get("codename", ""),
            "description": str(c.get("description", ""))[:220],
            "signals": (c.get("signals") or [])[:3],
            "examples": (c.get("examples") or [])[:2],
        } for c in raw]

        system = (
            f"You are consolidating {len(items)} NOISY candidate audience-persona archetypes - discovered "
            "batch-by-batch from an Italian Instagram influencer's comment community, with many near-duplicate "
            "names and heavy semantic overlap - into ONE clean, MECE taxonomy.\n"
            f"Merge synonyms and overlapping archetypes into AT MOST {target_personas} distinct, non-overlapping "
            "personas that together cover the candidates. Order them by how common/important the archetype is.\n"
            "For each final persona output: codename (UPPER_SNAKE_CASE), label (short Title Case), "
            "description (1-2 sentences), quantitative_signals (3-5 distinguishing behavioural signals), "
            "example_comments (2-3 verbatim fragments drawn from the candidates).\n"
            "Output ONLY a valid JSON array of objects with exactly those 5 keys. No preamble, no markdown fences."
        )
        prompt = system + "\n\n=== CANDIDATES ===\n" + json.dumps(items, ensure_ascii=False)
        config_obj = types.GenerateContentConfig(
            temperature=0.0, top_p=1.0, max_output_tokens=8192,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
        )
        print(f"Consolidating {len(items)} candidates -> <= {target_personas} personas via {model} ...")
        resp = self.batch.client.models.generate_content(model=model, contents=prompt, config=config_obj)
        text = strip_fences(resp.text or "")
        try:
            final = json.loads(text)
        except Exception as e:  # noqa: BLE001
            print("Could not parse LLM output:", str(e)[:120], "\n", text[:400])
            return []
        if not isinstance(final, list):
            print("Expected a JSON array, got", type(final).__name__)
            return []
        final = [_coerce_persona(p) for p in final if isinstance(p, dict)]
        print(f"{len(final)} consolidated personas:")
        for p in final:
            print(f"  {str(p['codename']):<28} | {p['label']}")
        if write:
            data["final_taxonomy"] = final
            data["pathway"] = "A_LLM"
            with open(cfg.taxonomy_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\nWritten to {cfg.taxonomy_json_path} -> final_taxonomy. Review/tweak, set status='APPROVED'.")
        return final

    def write_pathway_a_wrapper(self, final):
        """Guarded wrapper write: only when pathway is A_LLM and we have a result."""
        cfg = self.config
        if cfg.consolidation_pathway == "A_LLM" and final:
            wrapper = {
                "status": "PENDING_HUMAN_REVIEW",
                "pathway": "A_LLM",
                "instructions": (
                    "Consolidate the candidates into a MECE taxonomy. Each final persona needs a unique "
                    "'codename' (plus 'label', 'description', 'quantitative_signals', 'example_comments'). "
                    "Set status='APPROVED' before Stage 2."
                ),
                "final_taxonomy": final,
            }
            with open(cfg.taxonomy_json_path, "w", encoding="utf-8") as f:
                json.dump(wrapper, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(final)} personas to {cfg.taxonomy_json_path}")
        else:
            print("Skipping Pathway A wrapper (Pathway B taxonomy already written, or no result).")


# ============================== Stage 2 ====================================
class Stage2Classifier:
    def __init__(self, config: PipelineConfig, batch: BatchClient, media: MediaContextBuilder):
        self.config = config
        self.batch = batch
        self.media = media

    @staticmethod
    def build_system_prompt(taxonomy: list) -> str:
        taxonomy_text = ""
        for p in taxonomy:
            taxonomy_text += (
                f"\nPERSONA: {p['codename']} - {p.get('label', '')}\n"
                f"  Description: {p.get('description', '')}\n"
                f"  Quantitative signals: {'; '.join(p.get('quantitative_signals', []))}\n"
                f"  Example comments: {' | '.join(p.get('example_comments', []))}\n"
            )
        return (
            "You are a deterministic community analyst for Show Reel Media Group.\n"
            "Classify the Instagram commenter into exactly ONE persona from the approved taxonomy,\n"
            "using their behavioural metrics, comment samples, AND the attached post media (images /\n"
            "video frames + transcript) of the content they engage with most.\n\n"
            "=== APPROVED PERSONA TAXONOMY ===\n"
            f"{taxonomy_text}"
            "=================================\n\n"
            "RULES:\n"
            "1. Assign exactly ONE persona - the closest match.\n"
            "2. Output a confidence score between 0.0 and 1.0.\n"
            "3. Cite a specific comment fragment or media detail as justification.\n"
            "4. If data is insufficient, assign the most probable persona with confidence <= 0.4.\n"
            "5. Echo the author_id exactly as given.\n"
            "6. Output ONLY a single valid JSON object. No preamble, no markdown fences.\n\n"
            'Schema: {"author_id": str, "persona_codename": str, "confidence": float, "justification": str}'
        )

    @staticmethod
    def format_user_profile(row) -> dict:
        return {
            "author_id": str(row["author_id"]),
            "total_comments": int(row["total_comments"]),
            "unique_posts": int(row["unique_posts_commented"]),
            "activity_span_days": int(row["activity_span_days"]),
            "pct_comments_under_1h": round(float(row["pct_comments_under_1h"]), 2),
            "pct_comments_under_24h": round(float(row["pct_comments_under_24h"]), 2),
            "reply_ratio": round(float(row["reply_ratio"]), 2),
            "mean_mention_count": round(float(row["mean_mention_count"]), 2),
            "mean_word_count": round(float(row["mean_word_count"]), 1),
            "emoji_usage_rate": round(float(row["emoji_usage_rate"]), 2),
            "question_rate": round(float(row["question_rate"]), 2),
            "exclamation_rate": round(float(row["exclamation_rate"]), 2),
            "post_concentration_ratio": round(float(row["post_concentration_ratio"]), 2),
            "sample_comments": str(row.get("top_comments_sample", ""))[:500],
        }

    def build_line(self, row, system_prompt: str) -> dict:
        cfg = self.config
        profile = json.dumps(self.format_user_profile(row), ensure_ascii=False)
        parts = [{
            "text": system_prompt + "\n\n=== USER TO CLASSIFY ===\n" + profile
            + "\n\nThe images/frames and transcripts below are sample posts this user engaged with most."
        }]
        parts.extend(self.media.build_user_media_parts(row["author_id"]))
        parts.append({"text": "\nClassify THIS user into exactly one persona. Output ONLY one JSON object."})
        return {
            "request": {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": cfg.gen_config_dict(
                    cfg.max_output_tokens_stage2, cfg.thinking_budget_stage2
                ),
            }
        }

    def submit(self, user_features, taxonomy, max_users=None):
        from tqdm import tqdm

        cfg = self.config
        max_users = cfg.stage2_max_users if max_users is None else max_users
        df = user_features if max_users is None else user_features.head(max_users)
        print(
            f"\n{'='*60}\nSTAGE 2 (batch submit) - Classification\n"
            f"  Users: {len(df):,} (1/request) | Model: {cfg.model_stage2_classify}\n{'='*60}"
        )
        system_prompt = self.build_system_prompt(taxonomy)
        lines = [
            self.build_line(row, system_prompt)
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Build Stage 2 requests")
        ]
        in_uri = self.batch.upload_to_gcs(
            self.batch.write_jsonl(lines, f"{cfg.local_dir}/stage2_input.jsonl"),
            cfg.batch_input_prefix + "stage2_input.jsonl",
        )
        out_uri = f"gs://{cfg.gcs_bucket}/{cfg.batch_output_prefix}stage2/"
        job = self.batch.submit_batch_job(in_uri, out_uri, cfg.model_stage2_classify)
        self.batch.record_batch_job("stage2", job, out_uri)
        print(f"\nStage 2 submitted ({len(lines)} requests). Safe to close the laptop.")
        print("   When it finishes, run:  retrieve-stage2   -> writes user_personas.parquet")
        return job

    def retrieve(self, user_features, max_users=None, output_path=None):
        import pandas as pd

        cfg = self.config
        max_users = cfg.stage2_max_users if max_users is None else max_users
        output_path = cfg.results_path if output_path is None else output_path
        job = self.batch.get_recorded_job("stage2")
        if job.state != self.batch.JobState.JOB_STATE_SUCCEEDED:
            print(f"Stage 2 not ready (state={job.state}). Re-run later.")
            return None
        df = user_features if max_users is None else user_features.head(max_users)

        results = []
        for t in self.batch.retrieve_response_texts(job):
            if not t:
                continue
            try:
                obj = json.loads(strip_fences(t))
                if isinstance(obj, list):
                    results.extend(obj)
                elif isinstance(obj, dict):
                    results.append(obj)
            except Exception as e:  # noqa: BLE001
                print("   parse error:", str(e)[:80])

        results_df = pd.DataFrame(results)
        if "author_id" in results_df.columns:
            results_df["author_id"] = results_df["author_id"].astype(str)
        summary_cols = [
            "author_id", "total_comments", "activity_span_days",
            "mean_hours_to_comment", "pct_comments_under_1h",
            "reply_ratio", "mean_word_count",
        ]
        base = df.copy()
        base["author_id"] = base["author_id"].astype(str)
        final_df = base[summary_cols].merge(results_df, on="author_id", how="left")
        final_df.to_parquet(output_path, index=False)

        matched = (
            final_df["persona_codename"].notna().sum()
            if "persona_codename" in final_df.columns else 0
        )
        print(f"\nStage 2 complete. Classified {matched:,}/{len(final_df):,} users -> {output_path}")
        if "persona_codename" in final_df.columns:
            dist = final_df["persona_codename"].value_counts(normalize=True).mul(100).round(1)
            print("\n   Persona Distribution:")
            for persona, pct in dist.items():
                print(f"      {str(persona):<35} {pct:.1f}%")
        return final_df


# ============================ Persona map ==================================
class PersonaMapExporter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def export(self, results_path=None, out_path=None):
        import pandas as pd

        cfg = self.config
        results_path = cfg.results_path if results_path is None else results_path
        out_path = cfg.persona_map_path if out_path is None else out_path
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"{results_path} missing - run retrieve-stage2 first.")
        df = pd.read_parquet(results_path)
        df["author_id"] = df["author_id"].astype(str)
        cols = ["author_id", "persona_codename"] + (["confidence"] if "confidence" in df.columns else [])
        pmap = (
            df[cols].dropna(subset=["persona_codename"])
            .drop_duplicates("author_id").reset_index(drop=True)
        )
        pmap.to_parquet(out_path, index=False)
        print(f"Persona map: {len(pmap):,} authors -> {out_path}")
        print("   Attach downstream via:")
        print(
            "     ig_comments_clean = ig_comments_clean.merge("
            "pd.read_parquet(PERSONA_MAP_PATH), on='author_id', how='left')"
        )
        return pmap


# ============================ Validation ===================================
def validate_classification_output(results_path: str):
    import pandas as pd

    df = pd.read_parquet(results_path)
    classified = df["persona_codename"].notna() & (df["persona_codename"] != "CLASSIFICATION_ERROR")
    coverage_pct = classified.sum() / len(df) * 100
    print("\nCLASSIFICATION QA REPORT")
    print(f"   Total users             : {len(df):,}")
    print(f"   Successfully classified : {classified.sum():,} ({coverage_pct:.1f}%)")
    print(f"   Unclassified / errors   : {(~classified).sum():,}")
    sub = df[classified].copy()
    sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
    print(
        f"\n   Confidence  mean: {sub['confidence'].mean():.3f}  "
        f"median: {sub['confidence'].median():.3f}  "
        f"<0.40: {(sub['confidence'] < 0.4).sum():,}"
    )
    dupes = df["author_id"].duplicated().sum()
    print(f"\n   MECE Check: {dupes} duplicate assignments ({'FAIL' if dupes > 0 else 'PASS'})")
    print(df["persona_codename"].value_counts().to_frame("count").to_string())
    return df
