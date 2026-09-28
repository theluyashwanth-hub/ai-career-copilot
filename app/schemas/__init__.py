"""Pydantic schemas."""

from app.schemas.ats_analysis import ATSAnalysis, ATSSemanticSignals
from app.schemas.bullet import OptimizedBullet, XYZBreakdown
from app.schemas.interview import (
    AnswerFeedback,
    InterviewQuestion,
    InterviewSession,
)
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis, JobMatchSemanticSignals
from app.schemas.rag import RAGAnswer, RAGSource, ResumeChunk, RetrievedChunk
from app.schemas.resume_profile import (
    Education,
    Experience,
    Project,
    ResumeProfile,
)
from app.schemas.swot import SWOTAnalysis

__all__ = [
    "ATSAnalysis",
    "ATSSemanticSignals",
    "AnswerFeedback",
    "Education",
    "Experience",
    "InterviewQuestion",
    "InterviewSession",
    "JobMatchAnalysis",
    "JobMatchSemanticSignals",
    "JobProfile",
    "OptimizedBullet",
    "Project",
    "RAGAnswer",
    "RAGSource",
    "ResumeChunk",
    "ResumeProfile",
    "RetrievedChunk",
    "SWOTAnalysis",
    "XYZBreakdown",
]
