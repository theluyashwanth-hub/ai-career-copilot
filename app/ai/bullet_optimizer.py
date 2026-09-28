"""Single-bullet resume optimization via OpenRouter through LangChain (Phase 7).

All OpenRouter/LangChain logic lives here. Streamlit code must call
:func:`optimize_bullet` and never construct LLM clients directly.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.bullet import OptimizedBullet

BULLET_OPTIMIZATION_INSTRUCTIONS = """You improve a single resume bullet. Rewrite it for clarity, specificity, impact, strong action verbs, technical precision, ATS relevance, and conciseness (one to two lines).

CRITICAL RULE — NEVER invent: numbers, percentages, performance improvements, user counts, revenue, technologies, responsibilities, or achievements. Use ONLY what the original bullet states. If metrics are missing, do NOT add any; instead list in missing_metrics the kinds of real metrics the user could supply to strengthen the bullet (e.g. "latency reduction percentage", "number of users served").

Rules:
- improvements: short explanations of what changed and why (max 6).
- action_verbs: strong verbs used in the optimized bullet.
- xyz_breakdown: accomplishment (what was done), measurement (quantified result, or null when the original has none), method (how it was done, or null).
- Keep the optimized bullet truthful to the original scope.
{job_section}
Original bullet:
---
{bullet}
---
"""

JOB_SECTION_TEMPLATE = """- A job description is provided below. Adapt wording toward its relevant keywords where truthful, without adding experience the bullet does not state.

Job description:
---
{job_description}
---
"""


def _coerce_to_bullet(raw: Any) -> OptimizedBullet:
    """Validate raw structured output into an OptimizedBullet."""
    if isinstance(raw, OptimizedBullet):
        return raw
    if isinstance(raw, dict):
        try:
            return OptimizedBullet.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned an invalid optimized bullet: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected an optimized bullet object or dictionary."
    )


def optimize_bullet(
    bullet: str,
    job_description: str | None = None,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> OptimizedBullet:
    """Optimize a single resume bullet via OpenRouter structured output.

    Args:
        bullet: Original resume bullet text.
        job_description: Optional job description for keyword adaptation.
        model: Optional pre-built LangChain chat model (used by tests
            to inject a mock instead of calling OpenRouter).
        api_key: Optional API key override; defaults to settings.
        model_name: Gemini model name when building the default client.

    Raises:
        ValueError: Empty bullet, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not bullet or not bullet.strip():
        raise ValueError("Bullet is empty. Enter a resume bullet before optimizing.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    job_section = ""
    if job_description and job_description.strip():
        job_section = JOB_SECTION_TEMPLATE.format(job_description=job_description.strip())
    prompt = BULLET_OPTIMIZATION_INSTRUCTIONS.format(
        bullet=bullet.strip(),
        job_section=job_section,
    )

    raw = invoke_structured(
        chat_model,
        OptimizedBullet,
        prompt,
        invalid_label="OpenRouter returned an invalid optimized bullet",
        failure_label="Bullet optimization failed",
    )

    return _coerce_to_bullet(raw)
