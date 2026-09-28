"""Structured resume profile schemas (Phase 3).

All fields are optional (null) or empty lists when the resume
does not contain the information. The model must never invent data.
"""

from pydantic import BaseModel, Field


class Education(BaseModel):
    """A single education entry."""

    institution: str | None = Field(default=None)
    degree: str | None = Field(default=None)
    field_of_study: str | None = Field(default=None)
    start_date: str | None = Field(default=None)
    end_date: str | None = Field(default=None)
    description: str | None = Field(default=None)


class Experience(BaseModel):
    """A single work-experience entry."""

    company: str | None = Field(default=None)
    role: str | None = Field(default=None)
    location: str | None = Field(default=None)
    start_date: str | None = Field(default=None)
    end_date: str | None = Field(default=None)
    description: str | None = Field(default=None)


class Project(BaseModel):
    """A single project entry."""

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    technologies: list[str] = Field(default_factory=list)
    url: str | None = Field(default=None)


class ResumeProfile(BaseModel):
    """Structured profile extracted from resume text."""

    name: str | None = Field(default=None)
    email: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    location: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
