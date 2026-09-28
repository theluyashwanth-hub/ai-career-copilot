"""Resume-vs-job match schemas (Phase 6).

JobMatchAnalysis combines deterministic skill/keyword comparison
with LLM semantic comparison. The LLM must never fabricate
experience or attribute skills absent from the resume.
"""

from pydantic import BaseModel, Field


class JobMatchSemanticSignals(BaseModel):
    """Structured semantic comparison returned by the LLM.

    All scores are 0-100 and grounded in the supplied profiles.
    """

    experience_relevance_score: int = Field(default=50, ge=0, le=100)
    responsibility_alignment_score: int = Field(default=50, ge=0, le=100)
    project_relevance_score: int = Field(default=50, ge=0, le=100)
    transferable_skills_score: int = Field(default=50, ge=0, le=100)
    matching_experience: list[str] = Field(default_factory=list)
    transferable_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class JobMatchAnalysis(BaseModel):
    """Combined resume-vs-job match result."""

    match_score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    matching_experience: list[str] = Field(default_factory=list)
    missing_experience: list[str] = Field(default_factory=list)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
