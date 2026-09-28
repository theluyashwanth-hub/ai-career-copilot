"""Career SWOT analysis via OpenRouter through LangChain (Phase 8).

Builds a SWOT grounded entirely in the user's existing data:
ResumeProfile, ATSAnalysis, and JobMatchAnalysis when available.
All OpenRouter/LangChain logic lives here. Streamlit code must call
:func:`analyze_swot` and never construct LLM clients directly.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.schemas.swot import SWOTAnalysis

SWOT_INSTRUCTIONS = """You write a career SWOT analysis grounded ENTIRELY in the supplied data below.

Rules:
- Do NOT invent achievements, experience, skills, or opportunities. Every point must trace to the evidence given.
- Strengths: proven capabilities with resume evidence (e.g. "Strong experience with React and Node.js projects.").
- Weaknesses: areas where the resume shows limited or no evidence (e.g. "Resume contains limited evidence of automated testing.").
- Opportunities: target-job needs the resume already supports (e.g. "Target job requires technologies already demonstrated in projects."). Only use job-match data when provided.
- Threats: target-job needs absent from the resume (e.g. "Target job requires a technology absent from the resume."). Only use job-match data when provided.
- End each point with its evidence in parentheses, citing the source, e.g. "(Skills: Python, SQL; Experience: Backend Engineer at Acme)".
- Each list holds at most 6 short items (one to two sentences). Empty list when nothing is supported by evidence.

Resume profile:
---
{resume_json}
---

ATS analysis:
---
{ats_json}
---

Job match analysis:
---
{match_json}
---
"""


def _coerce_to_swot(raw: Any) -> SWOTAnalysis:
    """Validate raw structured output into a SWOTAnalysis."""
    if isinstance(raw, SWOTAnalysis):
        return raw
    if isinstance(raw, dict):
        try:
            return SWOTAnalysis.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned an invalid SWOT analysis: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected a SWOT analysis object or dictionary."
    )


def analyze_swot(
    resume_profile: ResumeProfile | None,
    ats_analysis: ATSAnalysis | None = None,
    job_match: JobMatchAnalysis | None = None,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> SWOTAnalysis:
    """Generate an evidence-grounded SWOT via OpenRouter structured output.

    Args:
        resume_profile: Analyzed resume (required).
        ats_analysis: ATS readiness result (optional but recommended).
        job_match: Resume-vs-job comparison (optional; opportunities and
            threats draw on it only when provided).
        model: Optional pre-built LangChain chat model (used by tests
            to inject a mock instead of calling OpenRouter).
        api_key: Optional API key override; defaults to settings.
        model_name: Gemini model name when building the default client.

    Raises:
        ValueError: Missing resume profile, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if resume_profile is None:
        raise ValueError("Resume profile is missing. Analyze a resume on the Resume page first.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = SWOT_INSTRUCTIONS.format(
        resume_json=resume_profile.model_dump_json(),
        ats_json=ats_analysis.model_dump_json() if ats_analysis is not None else "Not available.",
        match_json=job_match.model_dump_json() if job_match is not None else "Not available.",
    )

    raw = invoke_structured(
        chat_model,
        SWOTAnalysis,
        prompt,
        invalid_label="OpenRouter returned an invalid SWOT analysis",
        failure_label="SWOT analysis failed",
    )

    return _coerce_to_swot(raw)
