"""CLI for the persona pipeline.

The subcommands mirror the notebook's async submit/retrieve flow: each is a
separate process invocation, and batch job handles persist via the
outputs/<stage>_job.json files written by BatchClient.record_batch_job.

Typical sequence (Pathway A, the default):

    python -m persona submit-stage1
    # ... wait for the Vertex batch job to finish ...
    python -m persona retrieve-stage1
    python -m persona consolidate                 # Pathway A: LLM consolidation
    # review outputs/taxonomy.json, set status="APPROVED"
    python -m persona submit-stage2
    python -m persona retrieve-stage2
    python -m persona export-map
    python -m persona validate

Pathway B (alternative taxonomy discovery) replaces `consolidate`:

    python -m persona submit-stage1
    python -m persona retrieve-stage1
    python -m persona pathway-b --pathway B_CLUSTER
    # review/approve taxonomy.json, then submit-stage2 as above
"""
from __future__ import annotations

import argparse
import sys

from .config import PipelineConfig
from .pipeline import PersonaPipeline


def _build_config(args) -> PipelineConfig:
    overrides = {}
    if args.platform:
        overrides["platform"] = args.platform
    if args.local_dir:
        overrides["local_dir"] = args.local_dir
    if args.pathway:
        overrides["consolidation_pathway"] = args.pathway
    if args.sample_n is not None:
        overrides["sample_n_users"] = args.sample_n
    if args.target_personas is not None:
        overrides["target_personas"] = args.target_personas
    if getattr(args, "mode", None):
        overrides["pipeline_mode"] = args.mode
    return PipelineConfig(**overrides)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(prog="persona", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # global config overrides
    parser.add_argument("--platform", help="instagram | youtube | tiktok | facebook")
    parser.add_argument("--local-dir", dest="local_dir", help="output directory (default: outputs)")
    parser.add_argument("--pathway", choices=["A_LLM", "B_CLUSTER"],
                        help="authoritative consolidation pathway")
    parser.add_argument("--sample-n", dest="sample_n", type=int, help="Stage 1 sample size")
    parser.add_argument("--target-personas", dest="target_personas", type=int,
                        help="soft persona cap (default 12)")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("connectivity-test", help="online ping both models")
    sub.add_parser("submit-stage1", help="stratified sample + submit Stage 1 batch job")
    sub.add_parser("retrieve-stage1", help="parse Stage 1 output -> taxonomy.json (raw_candidates)")
    sub.add_parser("pathway-b", help="UMAP+HDBSCAN on the sample -> final_taxonomy (B_CLUSTER)")

    p_cons = sub.add_parser("consolidate", help="Pathway A: LLM consolidation -> final_taxonomy")
    p_cons.add_argument("--method", choices=["llm", "draft"], default="llm",
                        help="llm (one Gemini call) or draft (string-normalised)")
    p_cons.add_argument("--top-k", dest="top_k", type=int, default=15,
                        help="draft method: number of themes to draft")

    sub.add_parser("submit-stage2", help="classify ALL users (needs APPROVED taxonomy)")
    sub.add_parser("retrieve-stage2", help="parse Stage 2 output -> user_personas.parquet")
    sub.add_parser("export-map", help="author_id -> persona map parquet")
    sub.add_parser("validate", help="classification QA report")
    sub.add_parser("stage3", help="optional post-Stage-2 macro-clustering view")

    p_run = sub.add_parser("run", help="orchestrator (SAMPLE | FULL | ALL)")
    p_run.add_argument("--mode", choices=["SAMPLE", "FULL", "ALL"], help="override PIPELINE_MODE")

    args = parser.parse_args(argv)
    config = _build_config(args)
    pipe = PersonaPipeline(config)

    cmd = args.command
    if cmd == "connectivity-test":
        pipe.connectivity_test()
    elif cmd == "submit-stage1":
        pipe.submit_stage1()
    elif cmd == "retrieve-stage1":
        pipe.retrieve_stage1()
    elif cmd == "pathway-b":
        pipe.pathway_b()
    elif cmd == "consolidate":
        pipe.consolidate(method=args.method, top_k=args.top_k)
    elif cmd == "submit-stage2":
        pipe.submit_stage2()
    elif cmd == "retrieve-stage2":
        pipe.retrieve_stage2()
    elif cmd == "export-map":
        pipe.export_map()
    elif cmd == "validate":
        pipe.validate()
    elif cmd == "stage3":
        pipe.stage3()
    elif cmd == "run":
        pipe.run(mode=args.mode)
    else:  # pragma: no cover - argparse enforces choices
        parser.error(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
