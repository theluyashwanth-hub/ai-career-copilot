"""ATS analysis schemas (Phase 4).

ATSAnalysis is the combined result of deterministic checks and
LLM semantic evaluation. Scores are 0-100 integers computed by the
service layer, never invented by the LLM alone.
"""

from pydantic import BaseModel, Field


class ATSSemanticSignals(BaseModel):
    """Structured semantic evaluation returned by the LLM.

    All scores are 0-100. Lists are grounded in the resume text;
    the LLM must not invent content.
    """

    bullet_clarity_score: int = Field(default=50, ge=0, le=100)
    impact_score: int = Field(default=50, ge=0, le=100)
    specificity_score: int = Field(default=50, ge=0, le=100)
    relevance_score: int = Field(default=50, ge=0, le=100)
    vague_statements: list[str] = Field(default_factory=list)
    achievement_quality_notes: list[str] = Field(default_factory=list)
    technical_description_notes: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class ATSAnalysis(BaseModel):
    """Combined ATS readiness analysis."""

    overall_score: int = Field(ge=0, le=100)
    formatting_score: int = Field(ge=0, le=100)
    keyword_score: int = Field(ge=0, le=100)
    experience_score: int = Field(ge=0, le=100)
    skills_score: int = Field(ge=0, le=100)
    projects_score: int = Field(ge=0, le=100)
    education_score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
