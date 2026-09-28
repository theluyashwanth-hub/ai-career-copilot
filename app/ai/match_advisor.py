"""LLM semantic comparison for resume-vs-job matching (Phase 6).

OpenRouter judges what rules cannot: experience relevance,
responsibility alignment, project relevance, and transferable
skills. Returns structured JobMatchSemanticSignals; scoring and
blending live in app.services.job_matcher.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchSemanticSignals
from app.schemas.resume_profile import ResumeProfile

MATCH_SEMANTIC_INSTRUCTIONS = """You compare a candidate resume profile against a job profile for fit.

Rules:
- Use ONLY the information in the two profiles below. NEVER invent experience, skills, or facts.
- NEVER claim the candidate has a skill that does not appear in the resume profile.
- matching_experience must quote or closely paraphrase resume experience relevant to the job (max 6).
- transferable_skills must be skills present in the resume profile that apply to this job (max 8).
- Score each dimension 0-100: experience relevance, responsibility alignment, project relevance, transferable skills.
- List concrete strengths, gaps (job needs with little resume evidence), and recommendations (max 6 each).
- Keep every item short (one sentence).

Resume profile:
---
{resume_json}
---

Job profile:
---
{job_json}
---
"""


def _coerce_to_signals(raw: Any) -> JobMatchSemanticSignals:
    if isinstance(raw, JobMatchSemanticSignals):
        return raw
    if isinstance(raw, dict):
        try:
            return JobMatchSemanticSignals.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned invalid job match signals: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected job match signals or a dictionary."
    )


def evaluate_match_semantics(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> JobMatchSemanticSignals:
    """Compare profiles semantically via OpenRouter structured output.

    Args:
        resume_profile: Analyzed resume (must not be None).
        job_profile: Analyzed job description (must not be None).
        model: Optional pre-built LangChain chat model (tests inject a mock).
        api_key: Optional API key override; defaults to settings.
        model_name: Gemini model name when building the default client.

    Raises:
        ValueError: Missing profiles, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if resume_profile is None:
        raise ValueError("Resume profile is missing. Analyze a resume on the Resume page first.")
    if job_profile is None:
        raise ValueError(
            "Job profile is missing. Analyze a job description on the Job Description page first."
        )

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = MATCH_SEMANTIC_INSTRUCTIONS.format(
        resume_json=resume_profile.model_dump_json(),
        job_json=job_profile.model_dump_json(),
    )

    raw = invoke_structured(
        chat_model,
        JobMatchSemanticSignals,
        prompt,
        invalid_label="OpenRouter returned invalid job match signals",
        failure_label="Job match evaluation failed",
    )

    return _coerce_to_signals(raw)
