"""LLM semantic evaluation for ATS readiness (Phase 4).

OpenRouter judges what rules cannot: bullet clarity, impact,
specificity, relevance, vague statements, achievement quality,
and technical descriptions. Returns structured ATSSemanticSignals;
scoring/blending lives in app.services.ats_analyzer.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.ats_analysis import ATSSemanticSignals
from app.schemas.resume_profile import ResumeProfile

ATS_SEMANTIC_INSTRUCTIONS = """You are an ATS (applicant tracking system) resume reviewer. Evaluate ONLY the resume text below for semantic quality.

Rules:
- Judge ONLY what is explicitly in the resume text. NEVER invent facts, scores must reflect the text.
- Score each dimension 0-100 based on the text: bullet clarity, impact, specificity, relevance.
- Quote short vague statements verbatim in vague_statements (max 5). Empty list if none.
- Put brief observations in achievement_quality_notes and technical_description_notes (max 4 each).
- List concrete strengths, weaknesses, missing_keywords (role-relevant terms absent from the text), and recommendations (max 6 each).
- Keep every item short (one sentence).

Resume text:
---
{resume_text}
---
"""


def _coerce_to_signals(raw: Any) -> ATSSemanticSignals:
    if isinstance(raw, ATSSemanticSignals):
        return raw
    if isinstance(raw, dict):
        try:
            return ATSSemanticSignals.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned invalid ATS semantic signals: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected ATS semantic signals or a dictionary."
    )


def evaluate_resume_semantics(
    resume_text: str,
    profile: ResumeProfile | None = None,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> ATSSemanticSignals:
    """Evaluate resume semantics via OpenRouter structured output.

    Args:
        resume_text: Extracted plain text.
        profile: Optional structured profile for context (summarized, not required).
        model: Optional pre-built LangChain chat model (tests inject a mock).
        api_key: Optional API key override; defaults to settings.
        model_name: Gemini model name when building the default client.

    Raises:
        ValueError: Empty resume text, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume before ATS analysis.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = ATS_SEMANTIC_INSTRUCTIONS.format(resume_text=resume_text.strip())

    raw = invoke_structured(
        chat_model,
        ATSSemanticSignals,
        prompt,
        invalid_label="OpenRouter returned invalid ATS semantic signals",
        failure_label="ATS semantic evaluation failed",
    )

    return _coerce_to_signals(raw)
