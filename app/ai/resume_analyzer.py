"""AI resume-profile extraction via OpenRouter through LangChain.

All OpenRouter/LangChain logic lives here. Streamlit code must call
:func:`analyze_resume` and never construct LLM clients directly.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    get_gemini_model,
    invoke_structured,
)
from app.schemas.resume_profile import ResumeProfile

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_OPENROUTER_MODEL",
    "EXTRACTION_INSTRUCTIONS",
    "analyze_resume",
    "get_chat_model",
    "get_gemini_model",
]

EXTRACTION_INSTRUCTIONS = """You extract a structured resume profile from the resume text below.

Rules:
- Extract ONLY information explicitly present in the resume text.
- NEVER invent, guess, or infer missing details.
- If a scalar field (name, email, phone, location, summary) is not present, return null for it.
- If a list section (education, experience, projects, skills, certifications, achievements) has no entries, return an empty list.
- Keep descriptions concise and copied or closely paraphrased from the source text.

Resume text:
---
{resume_text}
---
"""


def _coerce_to_profile(raw: Any) -> ResumeProfile:
    """Validate raw structured output into a ResumeProfile."""
    if isinstance(raw, ResumeProfile):
        return raw
    if isinstance(raw, dict):
        try:
            return ResumeProfile.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned an invalid resume profile: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected a resume profile object or dictionary."
    )


def analyze_resume(
    resume_text: str,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> ResumeProfile:
    """Convert resume text into a validated ResumeProfile via OpenRouter.

    Args:
        resume_text: Extracted plain text (ResumeDocument.text).
        model: Optional pre-built LangChain chat model (used by tests
            to inject a mock instead of calling OpenRouter).
        api_key: Optional API key override; defaults to settings.
        model_name: OpenRouter model name when building the default client.

    Raises:
        ValueError: Empty resume text, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume before analyzing.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = EXTRACTION_INSTRUCTIONS.format(resume_text=resume_text.strip())

    raw = invoke_structured(
        chat_model,
        ResumeProfile,
        prompt,
        invalid_label="OpenRouter returned an invalid resume profile",
        failure_label="Resume analysis failed",
    )

    return _coerce_to_profile(raw)
