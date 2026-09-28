"""ORM models (Phase 12).

Tables mirror the existing Pydantic domain without changing it:

- ``users`` — owner of every row. No authentication in this phase;
  repositories use a single default user (see ``UserRepository``).
- ``resumes`` — one row per uploaded file (``ResumeDocument``).
- ``resume_profiles`` — one structured ``ResumeProfile`` per resume,
  kept both as queryable scalar columns and full ``profile_json``.
- ``job_descriptions`` — raw pasted text (not otherwise stored today).
- ``job_profiles`` — one structured ``JobProfile`` per job description.
- ``analyses`` — ATS / match / SWOT / bullet / copilot results with
  ``result_json`` payloads plus the headline ``score``.
- ``interview_sessions`` + ``interview_questions`` + ``interview_answers`` —
  one row-set per ``InterviewSession`` workflow.
- ``rag_documents`` — per-chunk *metadata* (section/source/text, optional
  embedding payload). The FAISS index itself stays in
  ``app.rag.vector_store`` and can be rebuilt from these rows.

All JSON payloads use portable ``sqlalchemy.JSON`` (JSONB on Postgres
is not required), and embeddings stay out of any vector extension so
FAISS remains the only vector backend.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _created_at() -> Mapped[dt.datetime]:
    return mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class User(Base):
    """Application user. Single default row until auth arrives."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[dt.datetime] = _created_at()

    resumes: Mapped[list[Resume]] = relationship(
        "Resume", back_populates="user", cascade="all, delete-orphan"
    )
    job_descriptions: Mapped[list[JobDescription]] = relationship(
        "JobDescription", back_populates="user", cascade="all, delete-orphan"
    )


class Resume(Base):
    """One uploaded resume file (``ResumeDocument``)."""

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Stable deduplication key, e.g. "name:size" (mirrors UI file ids).
    file_key: Mapped[str] = mapped_column(String(768), nullable=False, unique=True)
    created_at: Mapped[dt.datetime] = _created_at()

    user: Mapped[User] = relationship("User", back_populates="resumes")
    profile: Mapped[ResumeProfile | None] = relationship(
        "ResumeProfile",
        back_populates="resume",
        cascade="all, delete-orphan",
        uselist=False,
    )
    rag_documents: Mapped[list[RagDocument]] = relationship(
        "RagDocument", back_populates="resume", cascade="all, delete-orphan"
    )


class ResumeProfile(Base):
    """Structured ``ResumeProfile`` for one resume (one-to-one)."""

    __tablename__ = "resume_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(80), nullable=True)
    location: Mapped[str | None] = mapped_column(String(300), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full validated profile payload (education/experience/projects/...).
    profile_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = _created_at()

    resume: Mapped[Resume] = relationship("Resume", back_populates="profile")


class JobDescription(Base):
    """Raw pasted job description text."""

    __tablename__ = "job_descriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional stable key of the analyzed input (mirrors UI caching keys).
    input_key: Mapped[str | None] = mapped_column(String(768), nullable=True)
    created_at: Mapped[dt.datetime] = _created_at()

    user: Mapped[User] = relationship("User", back_populates="job_descriptions")
    profile: Mapped[JobProfile | None] = relationship(
        "JobProfile",
        back_populates="job_description",
        cascade="all, delete-orphan",
        uselist=False,
    )


class JobProfile(Base):
    """Structured ``JobProfile`` for one job description (one-to-one)."""

    __tablename__ = "job_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_description_id: Mapped[int] = mapped_column(
        ForeignKey("job_descriptions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    job_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    company: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # Full validated profile payload (skills/responsibilities/keywords/...).
    profile_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[dt.datetime] = _created_at()

    job_description: Mapped[JobDescription] = relationship(
        "JobDescription", back_populates="profile"
    )


class Analysis(Base):
    """A persisted analysis result.

    ``kind`` is one of ``ats``, ``match``, ``swot``, ``bullet``,
    ``copilot``. The validated Pydantic result lives in ``result_json``;
    ``score`` holds the headline score (overall/match) when applicable
    and ``recommendations`` the merged recommendation list.
    """

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[int | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    job_description_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_descriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Stable cache key of the inputs (e.g. UI match_key / swot_key).
    input_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    recommendations: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = _created_at()


class InterviewSession(Base):
    """One interview-coach workflow run."""

    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[int | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    job_description_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_descriptions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    current_question: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[dt.datetime] = _created_at()
    updated_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    questions: Mapped[list[InterviewQuestion]] = relationship(
        "InterviewQuestion",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.idx",
    )
    answers: Mapped[list[InterviewAnswer]] = relationship(
        "InterviewAnswer",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewAnswer.question_idx",
    )


class InterviewQuestion(Base):
    """One generated question inside a session (ordered by ``idx``)."""

    __tablename__ = "interview_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    related_resume_area: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped[InterviewSession] = relationship(
        "InterviewSession", back_populates="questions"
    )

    __table_args__ = (UniqueConstraint("session_id", "idx", name="uq_question_idx"),)


class InterviewAnswer(Base):
    """One submitted answer plus its ``AnswerFeedback`` payload."""

    __tablename__ = "interview_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = _created_at()

    session: Mapped[InterviewSession] = relationship(
        "InterviewSession", back_populates="answers"
    )

    __table_args__ = (
        UniqueConstraint("session_id", "question_idx", name="uq_answer_idx"),
    )


class RagDocument(Base):
    """One resume chunk's metadata (FAISS index stays separate)."""

    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resume_id: Mapped[int] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False)
    section: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    source: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional embedding payload for future backends; FAISS is untouched.
    embedding_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = _created_at()

    resume: Mapped[Resume] = relationship("Resume", back_populates="rag_documents")

    __table_args__ = (
        UniqueConstraint("resume_id", "chunk_id", name="uq_rag_chunk"),
    )
