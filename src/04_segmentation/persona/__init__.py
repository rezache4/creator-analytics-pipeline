"""Show Reel Community Persona Pipeline (Vertex AI Batch + Multimodal).

Object-oriented port of Ali/persona_pipeline.ipynb. Public surface:

    from persona import PipelineConfig, PersonaPipeline

    cfg = PipelineConfig(platform="instagram")
    pipe = PersonaPipeline(cfg)
    pipe.submit_stage1()        # -> later: pipe.retrieve_stage1()

See ``python -m persona --help`` for the CLI (submit/retrieve subcommands).
"""
from .config import PipelineConfig
from .pipeline import PersonaPipeline

__all__ = ["PipelineConfig", "PersonaPipeline"]
