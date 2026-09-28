"""Interview coach schemas (Phase 9).

InterviewSession tracks a stateful practice interview: one question
at a time, with answers and per-answer feedback. Pure workflow
transitions live in app.services.interview_coach; LLM calls live
in app.ai.interview_coach.
"""

from typing import Literal

from pydantic import BaseModel, Field

QuestionCategory = Literal["behavioral", "technical", "resume-specific", "role-specific"]


class InterviewQuestion(BaseModel):
    """A single interview question grounded in resume/job data."""

    question: str = Field(min_length=1)
    category: QuestionCategory
    difficulty: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    related_resume_area: str | None = Field(default=None)


class AnswerFeedback(BaseModel):
    """Feedback on a single interview answer."""

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    clarity: str = Field(min_length=1)
    relevance: str = Field(min_length=1)
    suggested_improvement: str = Field(min_length=1)


class InterviewSession(BaseModel):
    """Stateful practice interview session."""

    questions: list[InterviewQuestion] = Field(min_length=1)
    current_question: int = Field(default=0, ge=0)
    answers: list[str] = Field(default_factory=list)
    feedback: list[AnswerFeedback] = Field(default_factory=list)
