"""Career SWOT schemas (Phase 8).

SWOTAnalysis is grounded entirely in the user's existing data
(ResumeProfile, ATSAnalysis, optional JobMatchAnalysis). Each point
carries its evidence inline in parentheses. Nothing is invented.
"""

from pydantic import BaseModel, Field


class SWOTAnalysis(BaseModel):
    """Evidence-grounded career SWOT analysis."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
