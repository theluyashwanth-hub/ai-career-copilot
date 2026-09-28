"""Typed agent state (Phase 11).

Only the information the workflow needs. Nothing executable:
no code, no file handles, no credentials beyond an optional
API-key override (tests inject mock models instead).
"""

from typing import Any, Literal

from typing_extensions import TypedDict

from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile

Intent = Literal["analyze_resume", "match_job", "improve", "prepare_role", "general"]


class CopilotState(TypedDict, total=False):
    """LangGraph state for the Career Copilot workflow."""

    user_request: str
    intent: Intent
    needs_job: bool
    resume_profile: ResumeProfile | None
    resume_text: str | None
    job_profile: JobProfile | None
    ats_analysis: ATSAnalysis | None
    job_match: JobMatchAnalysis | None
    recommendations: list[str]
    final_response: str
    error: str | None
    # Test/runtime injection slots for LangChain chat models (None = real Gemini).
    ats_model: Any | None
    match_model: Any | None
    api_key: str | None
