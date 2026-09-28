"""Repository / data-access services (Phase 12).

This module is the *only* place (besides ``app.db.models``) allowed to
issue SQLAlchemy queries. Application services, AI modules, RAG code,
and UI pages must call these repository classes instead of querying
the ORM directly.

Each repository takes a ``sqlalchemy.orm.Session`` and exposes
plain CRUD methods returning ORM model instances.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Analysis,
    InterviewAnswer,
    InterviewQuestion,
    InterviewSession,
    JobDescription,
    JobProfile,
    RagDocument,
    Resume,
    ResumeProfile,
    User,
)

ANALYSIS_KINDS = frozenset({"ats", "match", "swot", "bullet", "copilot"})

DEFAULT_USER_EMAIL = "local@ai-career-copilot"


class UserRepository:
    """Users. No authentication in Phase 12 — one default local user."""

    def __init__(self, session: Session):
        self.session = session

    def get(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        return self.session.scalar(select(User).where(User.email == email))

    def get_or_create_default(self) -> User:
        """Return the default local user, creating it on first use."""
        user = self.get_by_email(DEFAULT_USER_EMAIL)
        if user is None:
            user = User(email=DEFAULT_USER_EMAIL, display_name="Local user")
            self.session.add(user)
            self.session.flush()
        return user

    def create(self, *, email: str | None = None, display_name: str | None = None) -> User:
        user = User(email=email, display_name=display_name)
        self.session.add(user)
        self.session.flush()
        return user


class ResumeRepository:
    """Resumes and their structured profiles."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        *,
        user_id: int,
        filename: str,
        file_type: str,
        file_size: int,
        text: str,
        character_count: int = 0,
        word_count: int = 0,
        file_key: str,
    ) -> Resume:
        resume = Resume(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            text=text,
            character_count=character_count,
            word_count=word_count,
            file_key=file_key,
        )
        self.session.add(resume)
        self.session.flush()
        return resume

    def get(self, resume_id: int) -> Resume | None:
        return self.session.get(Resume, resume_id)

    def get_by_file_key(self, file_key: str) -> Resume | None:
        return self.session.scalar(select(Resume).where(Resume.file_key == file_key))

    def list_for_user(self, user_id: int, *, limit: int = 50) -> list[Resume]:
        return list(
            self.session.scalars(
                select(Resume)
                .where(Resume.user_id == user_id)
                .order_by(Resume.id.desc())
                .limit(limit)
            )
        )

    def save_profile(
        self,
        resume_id: int,
        *,
        name: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        location: str | None = None,
        summary: str | None = None,
        profile_json: dict | None = None,
    ) -> ResumeProfile:
        """Insert or replace the structured profile for ``resume_id``."""
        existing = self.session.scalar(
            select(ResumeProfile).where(ResumeProfile.resume_id == resume_id)
        )
        if existing is not None:
            existing.name = name
            existing.email = email
            existing.phone = phone
            existing.location = location
            existing.summary = summary
            existing.profile_json = profile_json or {}
            self.session.flush()
            return existing
        profile = ResumeProfile(
            resume_id=resume_id,
            name=name,
            email=email,
            phone=phone,
            location=location,
            summary=summary,
            profile_json=profile_json or {},
        )
        self.session.add(profile)
        self.session.flush()
        return profile

    def get_profile(self, resume_id: int) -> ResumeProfile | None:
        return self.session.scalar(
            select(ResumeProfile).where(ResumeProfile.resume_id == resume_id)
        )


class JobRepository:
    """Job descriptions and their structured profiles."""

    def __init__(self, session: Session):
        self.session = session

    def create_description(
        self, *, user_id: int, raw_text: str, input_key: str | None = None
    ) -> JobDescription:
        job = JobDescription(user_id=user_id, raw_text=raw_text, input_key=input_key)
        self.session.add(job)
        self.session.flush()
        return job

    def get_description(self, job_id: int) -> JobDescription | None:
        return self.session.get(JobDescription, job_id)

    def list_for_user(self, user_id: int, *, limit: int = 50) -> list[JobDescription]:
        return list(
            self.session.scalars(
                select(JobDescription)
                .where(JobDescription.user_id == user_id)
                .order_by(JobDescription.id.desc())
                .limit(limit)
            )
        )

    def save_profile(
        self,
        job_description_id: int,
        *,
        job_title: str | None = None,
        company: str | None = None,
        profile_json: dict | None = None,
    ) -> JobProfile:
        """Insert or replace the structured profile for ``job_description_id``."""
        existing = self.session.scalar(
            select(JobProfile).where(
                JobProfile.job_description_id == job_description_id
            )
        )
        if existing is not None:
            existing.job_title = job_title
            existing.company = company
            existing.profile_json = profile_json or {}
            self.session.flush()
            return existing
        profile = JobProfile(
            job_description_id=job_description_id,
            job_title=job_title,
            company=company,
            profile_json=profile_json or {},
        )
        self.session.add(profile)
        self.session.flush()
        return profile

    def get_profile(self, job_description_id: int) -> JobProfile | None:
        return self.session.scalar(
            select(JobProfile).where(
                JobProfile.job_description_id == job_description_id
            )
        )


class AnalysisRepository:
    """ATS / match / SWOT / bullet / copilot analysis results."""

    def __init__(self, session: Session):
        self.session = session

    def record(
        self,
        *,
        user_id: int,
        kind: str,
        result_json: dict | None = None,
        resume_id: int | None = None,
        job_description_id: int | None = None,
        input_key: str | None = None,
        score: int | None = None,
        recommendations: list | None = None,
        error: str | None = None,
    ) -> Analysis:
        if kind not in ANALYSIS_KINDS:
            raise ValueError(
                f"Unknown analysis kind: {kind!r}. Expected one of {sorted(ANALYSIS_KINDS)}."
            )
        analysis = Analysis(
            user_id=user_id,
            resume_id=resume_id,
            job_description_id=job_description_id,
            kind=kind,
            input_key=input_key,
            score=score,
            result_json=result_json or {},
            recommendations=recommendations or [],
            error=error,
        )
        self.session.add(analysis)
        self.session.flush()
        return analysis

    def get(self, analysis_id: int) -> Analysis | None:
        return self.session.get(Analysis, analysis_id)

    def list_for_user(
        self, user_id: int, *, kind: str | None = None, limit: int = 50
    ) -> list[Analysis]:
        stmt = (
            select(Analysis)
            .where(Analysis.user_id == user_id)
            .order_by(Analysis.id.desc())
            .limit(limit)
        )
        if kind is not None:
            stmt = (
                select(Analysis)
                .where(Analysis.user_id == user_id, Analysis.kind == kind)
                .order_by(Analysis.id.desc())
                .limit(limit)
            )
        return list(self.session.scalars(stmt))

    def latest(
        self,
        *,
        user_id: int,
        kind: str,
        resume_id: int | None = None,
        job_description_id: int | None = None,
    ) -> Analysis | None:
        stmt = (
            select(Analysis)
            .where(Analysis.user_id == user_id, Analysis.kind == kind)
            .order_by(Analysis.id.desc())
            .limit(1)
        )
        if resume_id is not None:
            stmt = stmt.where(Analysis.resume_id == resume_id)
        if job_description_id is not None:
            stmt = stmt.where(Analysis.job_description_id == job_description_id)
        return self.session.scalar(stmt)

    def count_for_user(self, user_id: int) -> int:
        return (
            self.session.scalar(
                select(func.count()).select_from(Analysis).where(
                    Analysis.user_id == user_id
                )
            )
            or 0
        )


class InterviewRepository:
    """Interview sessions, questions, and answers."""

    def __init__(self, session: Session):
        self.session = session

    def create_session(
        self,
        *,
        user_id: int,
        resume_id: int | None = None,
        job_description_id: int | None = None,
    ) -> InterviewSession:
        interview = InterviewSession(
            user_id=user_id,
            resume_id=resume_id,
            job_description_id=job_description_id,
        )
        self.session.add(interview)
        self.session.flush()
        return interview

    def get_session(self, session_id: int) -> InterviewSession | None:
        return self.session.get(InterviewSession, session_id)

    def list_for_user(self, user_id: int, *, limit: int = 50) -> list[InterviewSession]:
        return list(
            self.session.scalars(
                select(InterviewSession)
                .where(InterviewSession.user_id == user_id)
                .order_by(InterviewSession.id.desc())
                .limit(limit)
            )
        )

    def add_question(
        self,
        session_id: int,
        *,
        idx: int,
        question: str,
        category: str,
        difficulty: str = "",
        reason: str = "",
        related_resume_area: str | None = None,
    ) -> InterviewQuestion:
        row = InterviewQuestion(
            session_id=session_id,
            idx=idx,
            question=question,
            category=category,
            difficulty=difficulty,
            reason=reason,
            related_resume_area=related_resume_area,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def record_answer(
        self,
        session_id: int,
        *,
        question_idx: int,
        answer: str,
        feedback_json: dict | None = None,
    ) -> InterviewAnswer:
        row = InterviewAnswer(
            session_id=session_id,
            question_idx=question_idx,
            answer=answer,
            feedback_json=feedback_json,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def advance(self, session_id: int, *, status: str | None = None) -> InterviewSession:
        interview = self.session.get(InterviewSession, session_id)
        if interview is None:
            raise ValueError(f"Interview session {session_id} not found.")
        interview.current_question += 1
        if status is not None:
            interview.status = status
        self.session.flush()
        return interview

    def complete(self, session_id: int) -> InterviewSession:
        return self.advance(session_id, status="complete")

    def delete_session(self, session_id: int) -> None:
        self.session.execute(
            delete(InterviewSession).where(InterviewSession.id == session_id)
        )


class RagDocumentRepository:
    """Resume chunk metadata (the FAISS index itself stays separate)."""

    def __init__(self, session: Session):
        self.session = session

    def upsert_chunks(
        self,
        resume_id: int,
        chunks: list[dict],
    ) -> list[RagDocument]:
        """Insert or replace chunk rows for ``resume_id``.

        Each chunk dict has ``chunk_id``, ``text``, ``section``,
        ``source`` and optional ``embedding`` (list of floats).
        """
        rows: list[RagDocument] = []
        for chunk in chunks:
            existing = self.session.scalar(
                select(RagDocument).where(
                    RagDocument.resume_id == resume_id,
                    RagDocument.chunk_id == chunk["chunk_id"],
                )
            )
            if existing is not None:
                existing.section = chunk.get("section", "")
                existing.source = chunk.get("source", "")
                existing.text = chunk["text"]
                if "embedding" in chunk:
                    existing.embedding_json = chunk["embedding"]
                rows.append(existing)
            else:
                row = RagDocument(
                    resume_id=resume_id,
                    chunk_id=chunk["chunk_id"],
                    section=chunk.get("section", ""),
                    source=chunk.get("source", ""),
                    text=chunk["text"],
                    embedding_json=chunk.get("embedding"),
                )
                self.session.add(row)
                rows.append(row)
        self.session.flush()
        return rows

    def list_for_resume(self, resume_id: int) -> list[RagDocument]:
        return list(
            self.session.scalars(
                select(RagDocument)
                .where(RagDocument.resume_id == resume_id)
                .order_by(RagDocument.id)
            )
        )

    def clear_for_resume(self, resume_id: int) -> int:
        result = self.session.execute(
            delete(RagDocument).where(RagDocument.resume_id == resume_id)
        )
        return result.rowcount or 0

    def count_for_resume(self, resume_id: int) -> int:
        return (
            self.session.scalar(
                select(func.count()).select_from(RagDocument).where(
                    RagDocument.resume_id == resume_id
                )
            )
            or 0
        )
