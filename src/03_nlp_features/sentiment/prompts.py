"""Central prompt registry for ALL Vertex AI / Gemini calls in this project.

╔══════════════════════════════════════════════════════════════════════════╗
║  EDIT YOUR LLM PROMPTS HERE — this is the single source of truth.          ║
║  Nothing else in the pipelines hard-codes prompt text; they import from    ║
║  this module. To tune a prompt, change the constant below and re-run.      ║
╚══════════════════════════════════════════════════════════════════════════╝

Each constant is a ``str.format(...)`` template. The ``{placeholders}`` each
template expects are documented next to it. The ``PROMPTS`` dict at the bottom
lists every prompt by name for quick discovery (e.g. ``python -c "import
prompts; print(list(prompts.PROMPTS))"``).

Consumers
---------
    community_vibe_pipeline.py
        POST_CONTEXT_PROMPT      → Phase 1 (text-only post summary, Flash)
        COMMUNITY_VIBE_PROMPT    → Phase 3 (community vibe over comments, Pro)

    instagram_multimodal_pipeline.py
        MULTIMODAL_POST_PROMPT   → Phase 1 (frames + transcript + caption)
        CAG_SYSTEM_INSTRUCTION   → Phase 3 cache system instruction (CAG)
        CAG_COMMENT_CHUNK_PROMPT → Phase 3 per-chunk comment scoring (CAG)
"""

from __future__ import annotations

from typing import Dict

# --------------------------------------------------------------------------- #
# Phase 1 — Post-Level Context Summarizer (TEXT-ONLY, Gemini 2.5 Flash)
#   placeholders: {platform} {post_text} {transcript} {formats} {tones}
# --------------------------------------------------------------------------- #
POST_CONTEXT_PROMPT = (
    "Sei un analista di media digitali. Analizza il seguente post pubblicato "
    "su {platform}. Combina didascalia e trascrizione (se presente) per "
    "dedurre il contesto creativo.\n\n"
    "--- DIDASCALIA / DESCRIZIONE ---\n{post_text}\n\n"
    "--- TRASCRIZIONE (se disponibile) ---\n{transcript}\n\n"
    "Restituisci ESCLUSIVAMENTE un oggetto JSON conforme allo schema, senza "
    "testo aggiuntivo e senza wrapper markdown. Campi:\n"
    "- format_type: uno tra {formats}\n"
    "- primary_topic: argomento principale conciso (max 6 parole)\n"
    "- intended_emotional_tone: uno tra {tones}\n"
    "- brand_entities: elenco di marchi/aziende citati esplicitamente "
    "(vuoto se nessuno)."
)

# --------------------------------------------------------------------------- #
# Phase 3 — Community Vibe Aggregator (TEXT-ONLY, Gemini 2.5 Pro)
#   placeholders: {platform} {topic} {tone} {comments}
# --------------------------------------------------------------------------- #
COMMUNITY_VIBE_PROMPT = (
    "Sei un sociologo delle community online. Di seguito un campione di "
    "commenti (di primo livello e risposte) sotto un singolo post su "
    "{platform}. Contesto del post: argomento='{topic}', tono='{tone}'.\n\n"
    "--- COMMENTI CAMPIONE ---\n{comments}\n\n"
    "Valuta il 'vibe' collettivo e restituisci ESCLUSIVAMENTE JSON conforme "
    "allo schema (nessun markdown):\n"
    "- sentiment_polarization_index: numero 0.0-1.0 (0.0=consenso totale, "
    "1.0=frammentazione/polarizzazione marcata)\n"
    "- dominant_community_emotion: emozione modale del pubblico\n"
    "- community_noun_phrases: frasi nominali salienti su cui converge il "
    "pubblico."
)

# --------------------------------------------------------------------------- #
# Phase 1 (Instagram) — MULTIMODAL Post Summarizer (frames + transcript + caption)
#   placeholders: {subtype} {caption} {transcript} {formats} {tones}
# --------------------------------------------------------------------------- #
MULTIMODAL_POST_PROMPT = (
    "Analizza questo post Instagram di tipo '{subtype}'. Ti fornisco i FRAME "
    "estratti dal contenuto, la trascrizione audio e la didascalia.\n\n"
    "--- DIDASCALIA ---\n{caption}\n\n"
    "--- TRASCRIZIONE ---\n{transcript}\n\n"
    "Osserva i frame e combinali con il testo. Restituisci SOLO JSON conforme "
    "allo schema (nessun markdown):\n"
    "- format_type: uno tra {formats}\n"
    "- primary_topic: argomento principale (max 6 parole)\n"
    "- intended_emotional_tone: uno tra {tones}\n"
    "- brand_entities: marchi/aziende visibili o citati (vuoto se nessuno)\n"
    "- visual_summary: cosa si vede nei frame (1-2 frasi)\n"
    "- on_screen_text: testo a schermo / sticker (vuoto se nessuno)\n"
    "- visual_setting: ambientazione (es. cucina, esterno città, studio)."
)

# --------------------------------------------------------------------------- #
# Phase 3 (Instagram) — Cache-Augmented Generation
#   CAG_SYSTEM_INSTRUCTION  : system instruction stored in the CachedContent
#                             (no placeholders)
#   CAG_COMMENT_CHUNK_PROMPT : per-chunk query against the cache
#                             placeholders: {comments}
# --------------------------------------------------------------------------- #
CAG_SYSTEM_INSTRUCTION = (
    "Sei un sociologo delle community. Le immagini (frame del post), "
    "la trascrizione e la didascalia forniti sono la VERITÀ DI BASE "
    "del post Instagram. Userai SOLO questo contesto per valutare i "
    "commenti che ti verranno passati."
)

CAG_COMMENT_CHUNK_PROMPT = (
    "Sulla base del contesto del post fornito (frame, trascrizione, "
    "didascalia in cache), valuta SOLO questi commenti e restituisci JSON:\n"
    "{comments}\n\n"
    "- sentiment_polarization_index: 0.0 (consenso) → 1.0 (frammentazione)\n"
    "- dominant_community_emotion: emozione modale\n"
    "- community_noun_phrases: frasi nominali salienti dei commenti\n"
    "- visual_reference_ratio: frazione (0-1) di commenti che fanno "
    "riferimento a ciò che si VEDE nei frame del post."
)


# --------------------------------------------------------------------------- #
# Discovery registry — every prompt by name.
# --------------------------------------------------------------------------- #
PROMPTS: Dict[str, str] = {
    "post_context": POST_CONTEXT_PROMPT,
    "community_vibe": COMMUNITY_VIBE_PROMPT,
    "multimodal_post": MULTIMODAL_POST_PROMPT,
    "cag_system_instruction": CAG_SYSTEM_INSTRUCTION,
    "cag_comment_chunk": CAG_COMMENT_CHUNK_PROMPT,
}


if __name__ == "__main__":  # quick listing / sanity check
    for _name, _text in PROMPTS.items():
        print(f"\n=== {_name} ===\n{_text}")
