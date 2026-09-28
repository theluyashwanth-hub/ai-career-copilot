"""Job description schemas (Phase 5).

All fields are optional (null) or empty lists when the job
description does not contain the information. The model must
never invent data.
"""

from pydantic import BaseModel, Field


class JobProfile(BaseModel):
    """Structured profile extracted from a job description."""

    job_title: str | None = Field(default=None)
    company: str | None = Field(default=None)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)
    experience_requirements: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
