"""Job description extraction via OpenRouter through LangChain (Phase 5).

All OpenRouter/LangChain logic lives here. Streamlit code must call
:func:`analyze_job` and never construct LLM clients directly.
No resume matching happens here.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.job import JobProfile

JOB_EXTRACTION_INSTRUCTIONS = """You extract a structured job profile from the job description below.

Rules:
- Extract ONLY information explicitly present in the job description.
- NEVER invent, guess, or infer missing details.
- If job_title or company is not present, return null for it.
- If a list section (required_skills, preferred_skills, responsibilities, qualifications, experience_requirements, keywords) has no entries, return an empty list.
- required_skills are explicitly marked as required; preferred_skills are marked as nice-to-have, preferred, or bonus.
- keywords are role-relevant terms appearing in the description (skills, tools, technologies).
- Keep every item short (one line).

Job description:
---
{job_description}
---
"""


def _coerce_to_job_profile(raw: Any) -> JobProfile:
    """Validate raw structured output into a JobProfile."""
    if isinstance(raw, JobProfile):
        return raw
    if isinstance(raw, dict):
        try:
            return JobProfile.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned an invalid job profile: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected a job profile object or dictionary."
    )


def analyze_job(
    job_description: str,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> JobProfile:
    """Convert a pasted job description into a validated JobProfile via OpenRouter.

    Args:
        job_description: Raw pasted job description text.
        model: Optional pre-built LangChain chat model (used by tests
            to inject a mock instead of calling OpenRouter).
        api_key: Optional API key override; defaults to settings.
        model_name: Gemini model name when building the default client.

    Raises:
        ValueError: Empty input, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty. Paste a job description before analyzing.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = JOB_EXTRACTION_INSTRUCTIONS.format(job_description=job_description.strip())

    raw = invoke_structured(
        chat_model,
        JobProfile,
        prompt,
        invalid_label="OpenRouter returned an invalid job profile",
        failure_label="Job analysis failed",
    )

    return _coerce_to_job_profile(raw)
