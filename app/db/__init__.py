"""Persistent storage layer (Phase 12: PostgreSQL + SQLAlchemy + Alembic).

Only modules inside ``app.db`` may import SQLAlchemy. Application services,
AI modules, RAG code, and Streamlit pages must use the repository classes
re-exported here and never issue ORM queries directly. The FAISS vector
store (``app.rag.vector_store``) stays separate; only chunk *metadata*
lives in the ``rag_documents`` table.
"""

from app.db.base import Base
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
from app.db.repositories import (
    AnalysisRepository,
    InterviewRepository,
    JobRepository,
    RagDocumentRepository,
    ResumeRepository,
    UserRepository,
)
from app.db.session import create_engine_for_url, db_session, get_engine, init_db

__all__ = [
    "Analysis",
    "AnalysisRepository",
    "Base",
    "InterviewAnswer",
    "InterviewQuestion",
    "InterviewRepository",
    "InterviewSession",
    "JobDescription",
    "JobProfile",
    "JobRepository",
    "RagDocument",
    "RagDocumentRepository",
    "Resume",
    "ResumeProfile",
    "ResumeRepository",
    "User",
    "UserRepository",
    "create_engine_for_url",
    "db_session",
    "get_engine",
    "init_db",
]
