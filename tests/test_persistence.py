"""Tests for Phase 12 persistent storage (test database).

The test database is ``settings.test_database_url`` (in-memory SQLite by
default, so no live PostgreSQL server is needed). Point
``TEST_DATABASE_URL`` at a real Postgres database to run this same suite
against the production backend.
"""

import pathlib

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db import (
    AnalysisRepository,
    InterviewRepository,
    JobRepository,
    RagDocumentRepository,
    Resume,
    ResumeRepository,
    UserRepository,
    create_engine_for_url,
    db_session,
    get_engine,
    init_db,
)
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.resume_profile import ResumeProfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture()
def engine():
    url = settings.test_database_url
    engine = create_engine_for_url(url)
    init_db(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def session(engine):
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture()
def user(session):
    repo = UserRepository(session)
    user = repo.get_or_create_default()
    session.commit()
    return user


def _resume(session, user_id, file_key="resume.pdf:1234"):
    repo = ResumeRepository(session)
    resume = repo.create(
        user_id=user_id,
        filename="resume.pdf",
        file_type="pdf",
        file_size=1234,
        text="Jane Doe\nPython engineer",
        character_count=26,
        word_count=4,
        file_key=file_key,
    )
    session.commit()
    return resume


# --- Users ---------------------------------------------------------------


def test_default_user_created_once(session):
    repo = UserRepository(session)
    first = repo.get_or_create_default()
    session.commit()
    second = repo.get_or_create_default()
    assert first.id == second.id
    assert first.email == "local@ai-career-copilot"


# --- Resumes + profiles --------------------------------------------------


def test_resume_crud_and_file_key_lookup(session, user):
    repo = ResumeRepository(session)
    resume = _resume(session, user.id)
    assert repo.get(resume.id).filename == "resume.pdf"
    assert repo.get_by_file_key("resume.pdf:1234").id == resume.id
    assert repo.get_by_file_key("missing") is None
    assert [r.id for r in repo.list_for_user(user.id)] == [resume.id]


def test_resume_file_key_unique(session, user):
    _resume(session, user.id)
    with pytest.raises(IntegrityError):
        _resume(session, user.id)
    session.rollback()


def test_resume_profile_save_replace_and_round_trip(session, user):
    repo = ResumeRepository(session)
    resume = _resume(session, user.id)
    profile = ResumeProfile(name="Jane Doe", email="jane@example.com", skills=["Python"])
    saved = repo.save_profile(
        resume.id,
        name=profile.name,
        email=profile.email,
        profile_json=profile.model_dump(),
    )
    session.commit()
    assert saved.id is not None
    # Replace path keeps a single profile row per resume.
    repo.save_profile(resume.id, name="Jane D.", profile_json={"name": "Jane D."})
    session.commit()
    fetched = repo.get_profile(resume.id)
    assert fetched.name == "Jane D."
    assert ResumeProfile.model_validate(fetched.profile_json).name == "Jane D."
    # Full payload round-trips through JSON storage unchanged.
    repo.save_profile(
        resume.id,
        name=profile.name,
        email=profile.email,
        profile_json=profile.model_dump(),
    )
    session.commit()
    assert ResumeProfile.model_validate(repo.get_profile(resume.id).profile_json) == profile


# --- Jobs ----------------------------------------------------------------


def test_job_description_and_profile(session, user):
    repo = JobRepository(session)
    job = repo.create_description(
        user_id=user.id, raw_text="Senior Python role", input_key="abc"
    )
    session.commit()
    saved = repo.save_profile(
        job.id,
        job_title="Senior Python Engineer",
        company="Acme",
        profile_json={"job_title": "Senior Python Engineer"},
    )
    session.commit()
    assert saved.job_title == "Senior Python Engineer"
    assert repo.get_profile(job.id).company == "Acme"
    assert repo.get_description(job.id).raw_text == "Senior Python role"
    # Replace path (one profile per description).
    repo.save_profile(job.id, job_title="Staff Engineer", profile_json={})
    session.commit()
    assert repo.get_profile(job.id).job_title == "Staff Engineer"


# --- Analyses ------------------------------------------------------------


def test_analyses_record_list_latest_count(session, user):
    resume = _resume(session, user.id)
    job = JobRepository(session).create_description(user_id=user.id, raw_text="JD")
    session.commit()
    repo = AnalysisRepository(session)
    ats = ATSAnalysis(
        overall_score=80,
        formatting_score=85,
        keyword_score=70,
        experience_score=75,
        skills_score=72,
        projects_score=68,
        education_score=90,
    )
    repo.record(
        user_id=user.id,
        kind="ats",
        resume_id=resume.id,
        input_key="resume.pdf:1234",
        score=ats.overall_score,
        result_json=ats.model_dump(),
        recommendations=["Add metrics"],
    )
    repo.record(
        user_id=user.id,
        kind="match",
        resume_id=resume.id,
        job_description_id=job.id,
        score=65,
        result_json={"match_score": 65},
        recommendations=[],
    )
    session.commit()

    assert repo.count_for_user(user.id) == 2
    assert len(repo.list_for_user(user.id)) == 2
    assert len(repo.list_for_user(user.id, kind="ats")) == 1
    latest = repo.latest(user_id=user.id, kind="ats", resume_id=resume.id)
    assert latest.score == 80
    assert ATSAnalysis.model_validate(latest.result_json).overall_score == 80
    assert repo.latest(user_id=user.id, kind="swot") is None


def test_analyses_reject_unknown_kind(session, user):
    with pytest.raises(ValueError, match="Unknown analysis kind"):
        AnalysisRepository(session).record(user_id=user.id, kind="nope")


# --- Interviews ----------------------------------------------------------


def test_interview_lifecycle(session, user):
    resume = _resume(session, user.id)
    repo = InterviewRepository(session)
    interview = repo.create_session(user_id=user.id, resume_id=resume.id)
    session.commit()
    assert interview.status == "active"

    repo.add_question(
        interview.id,
        idx=0,
        question="Tell me about yourself.",
        category="behavioral",
        difficulty="easy",
        reason="Warm-up.",
    )
    repo.add_question(
        interview.id,
        idx=1,
        question="Explain Python GIL.",
        category="technical",
        difficulty="medium",
        reason="Role fit.",
        related_resume_area="Skills: Python",
    )
    feedback = repo.record_answer(
        interview.id,
        question_idx=0,
        answer="I am an engineer.",
        feedback_json={"clarity": "Clear."},
    )
    session.commit()
    assert feedback.feedback_json == {"clarity": "Clear."}

    fetched = repo.get_session(interview.id)
    assert [q.idx for q in fetched.questions] == [0, 1]
    assert fetched.answers[0].answer == "I am an engineer."

    repo.advance(interview.id)
    session.commit()
    assert repo.get_session(interview.id).current_question == 1
    repo.complete(interview.id)
    session.commit()
    assert repo.get_session(interview.id).status == "complete"

    repo.delete_session(interview.id)
    session.commit()
    assert repo.get_session(interview.id) is None
    # Questions/answers cascade with the session.
    assert repo.list_for_user(user.id) == []


def test_interview_advance_missing_rejected(session):
    with pytest.raises(ValueError, match="not found"):
        InterviewRepository(session).advance(9999)


# --- RAG documents -------------------------------------------------------


def test_rag_documents_upsert_list_clear(session, user):
    resume = _resume(session, user.id)
    repo = RagDocumentRepository(session)
    repo.upsert_chunks(
        resume.id,
        [
            {
                "chunk_id": "resume.pdf#skills:0",
                "section": "skills",
                "source": "resume.pdf",
                "text": "Python, SQL",
                "embedding": [0.1, 0.2],
            },
            {
                "chunk_id": "resume.pdf#experience:0",
                "section": "experience",
                "source": "resume.pdf",
                "text": "Backend engineer",
            },
        ],
    )
    session.commit()
    assert repo.count_for_resume(resume.id) == 2
    # Upsert replaces text for an existing chunk id (no duplicate rows).
    repo.upsert_chunks(
        resume.id,
        [
            {
                "chunk_id": "resume.pdf#skills:0",
                "section": "skills",
                "source": "resume.pdf",
                "text": "Python, SQL, Docker",
            }
        ],
    )
    session.commit()
    rows = repo.list_for_resume(resume.id)
    assert len(rows) == 2
    assert rows[0].text == "Python, SQL, Docker"
    assert rows[0].embedding_json == [0.1, 0.2]
    assert repo.clear_for_resume(resume.id) == 2
    session.commit()
    assert repo.count_for_resume(resume.id) == 0


# --- Session helper ------------------------------------------------------


def test_db_session_commits_and_rolls_back(engine, user):
    with db_session(engine) as session:
        repo = ResumeRepository(session)
        resume = repo.create(
            user_id=user.id,
            filename="a.pdf",
            file_type="pdf",
            file_size=10,
            text="text",
            file_key="a.pdf:10",
        )
        resume_id = resume.id
    with Session(engine) as check:
        assert check.get(Resume, resume_id) is not None
    with pytest.raises(RuntimeError, match="boom"):
        with db_session(engine) as session:
            ResumeRepository(session).create(
                user_id=user.id,
                filename="b.pdf",
                file_type="pdf",
                file_size=10,
                text="text",
                file_key="b.pdf:10",
            )
            raise RuntimeError("boom")
    with Session(engine) as check:
        assert ResumeRepository(check).get_by_file_key("b.pdf:10") is None


def test_get_engine_uses_test_url_when_provided():
    engine = get_engine("sqlite+pysqlite:///:memory:")
    try:
        init_db(engine)
        with Session(engine) as session:
            assert UserRepository(session).get_or_create_default().id is not None
    finally:
        engine.dispose()


# --- Migration -----------------------------------------------------------


def test_alembic_migration_creates_all_tables(tmp_path):
    """The committed Alembic migration builds the full schema from scratch."""
    from alembic import command
    from alembic.config import Config

    db_file = tmp_path / "migrated.db"
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_file}")
    command.upgrade(cfg, "head")
    try:
        engine = create_engine_for_url(f"sqlite+pysqlite:///{db_file}")
        try:
            tables = set(inspect(engine).get_table_names())
        finally:
            engine.dispose()
    finally:
        command.downgrade(cfg, "base")
    expected = {
        "users",
        "resumes",
        "resume_profiles",
        "job_descriptions",
        "job_profiles",
        "analyses",
        "interview_sessions",
        "interview_questions",
        "interview_answers",
        "rag_documents",
        "alembic_version",
    }
    assert expected <= tables


# --- Separation guard ----------------------------------------------------


def test_business_logic_has_no_sqlalchemy_dependency():
    """App services/AI/RAG/agent/UI must use repositories, never SQLAlchemy."""
    roots = ["app/services", "app/ai", "app/rag", "app/agent", "app/ui"]
    offenders = []
    for root in roots:
        for path in pathlib.Path(root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "sqlalchemy" in text or "from app.db" in text or "import app.db" in text:
                offenders.append(str(path))
    assert offenders == []
