"""Tests for Phase 9 interview coach (Gemini mocked, workflow pure)."""

import pytest

from app.ai.interview_coach import (
    QUESTION_GENERATION_INSTRUCTIONS,
    evaluate_interview_answer,
    generate_interview_questions,
)
from app.schemas.interview import AnswerFeedback, InterviewQuestion, InterviewSession
from app.schemas.job import JobProfile
from app.schemas.resume_profile import ResumeProfile
from app.services.interview_coach import (
    advance,
    create_session,
    get_current_question,
    has_pending_feedback,
    is_complete,
    record_answer,
)

RESUME = ResumeProfile(name="Jane Doe", skills=["Python", "SQL"])
JOB = JobProfile(job_title="Backend Engineer", required_skills=["Python"])

QUESTIONS_DICTS = [
    {
        "question": "Tell me about a challenging bug you fixed.",
        "category": "behavioral",
        "difficulty": "medium",
        "reason": "Probes debugging experience relevant to backend work.",
        "related_resume_area": "Backend Engineer",
    },
    {
        "question": "How would you optimize a slow SQL query?",
        "category": "technical",
        "difficulty": "medium",
        "reason": "Tests SQL depth from the resume.",
        "related_resume_area": "SQL",
    },
    {
        "question": "Walk me through your payments API project.",
        "category": "resume-specific",
        "difficulty": "easy",
        "reason": "Grounds discussion in stated experience.",
        "related_resume_area": "Projects",
    },
    {
        "question": "How would you design our ingestion pipeline? (role requirement)",
        "category": "role-specific",
        "difficulty": "hard",
        "reason": "Role requirement marked for the target job (role requirement).",
        "related_resume_area": None,
    },
]

FEEDBACK_DICT = {
    "strengths": ["Concrete example with outcome."],
    "weaknesses": ["Missing structure."],
    "missing_points": ["Quantified result."],
    "clarity": "Mostly clear but rambles at the end.",
    "relevance": "Directly answers the question.",
    "suggested_improvement": "Use the STAR format with a metric.",
}


class _FakeStructured:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return self._result


class _FakeModel:
    def __init__(self, result=None, error=None):
        self._structured = _FakeStructured(result=result, error=error)
        self.schemas: list[type] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self._structured


def _make_questions():
    return [InterviewQuestion.model_validate(q) for q in QUESTIONS_DICTS]


# --- Question generation ---


def test_generate_questions_covers_all_categories():
    fake = _FakeModel(result={"questions": [dict(q) for q in QUESTIONS_DICTS]})
    questions = generate_interview_questions(RESUME, JOB, model=fake)
    assert len(questions) == 4
    assert {q.category for q in questions} == {
        "behavioral",
        "technical",
        "resume-specific",
        "role-specific",
    }
    assert fake.schemas == [InterviewSession]
    assert "Jane Doe" in fake._structured.prompts[0]


def test_generate_questions_without_job():
    fake = _FakeModel(result={"questions": [dict(QUESTIONS_DICTS[0])] * 4})
    questions = generate_interview_questions(RESUME, None, num_questions=4, model=fake)
    assert len(questions) == 4
    assert "Not available" in fake._structured.prompts[0]


def test_generate_questions_requires_minimum_count():
    fake = _FakeModel(result={"questions": []})
    with pytest.raises(ValueError, match="At least 4 questions"):
        generate_interview_questions(RESUME, JOB, num_questions=2, model=fake)


def test_generate_questions_missing_resume_rejected():
    fake = _FakeModel(result={"questions": []})
    with pytest.raises(ValueError, match="Resume profile is missing"):
        generate_interview_questions(None, JOB, model=fake)
    assert fake._structured.prompts == []


def test_generate_questions_malformed_rejected():
    fake = _FakeModel(result="not questions at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        generate_interview_questions(RESUME, JOB, model=fake)


def test_generate_questions_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        generate_interview_questions(RESUME, JOB, api_key="")


def test_generate_questions_api_error_wrapped():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="invalid interview questions failed"):
        generate_interview_questions(RESUME, JOB, model=fake)


def test_generation_prompt_guards_grounding_and_scope():
    assert "Do NOT invent experience" in QUESTION_GENERATION_INSTRUCTIONS
    assert "(role requirement)" in QUESTION_GENERATION_INSTRUCTIONS
    assert "medical" in QUESTION_GENERATION_INSTRUCTIONS


# --- Answer feedback ---


def test_evaluate_answer_returns_feedback():
    question = _make_questions()[0]
    fake = _FakeModel(result=dict(FEEDBACK_DICT))
    feedback = evaluate_interview_answer(question, "I fixed a race condition.", model=fake)
    assert isinstance(feedback, AnswerFeedback)
    assert feedback.strengths == ["Concrete example with outcome."]
    assert feedback.missing_points == ["Quantified result."]
    assert "STAR" in feedback.suggested_improvement


def test_evaluate_empty_answer_rejected():
    question = _make_questions()[0]
    fake = _FakeModel(result=dict(FEEDBACK_DICT))
    with pytest.raises(ValueError, match="Answer is empty"):
        evaluate_interview_answer(question, "   ", model=fake)
    assert fake._structured.prompts == []


def test_evaluate_malformed_rejected():
    question = _make_questions()[0]
    fake = _FakeModel(result="not feedback")
    with pytest.raises(ValueError, match="unexpected response type"):
        evaluate_interview_answer(question, "Some answer.", model=fake)


# --- Pure workflow ---


def test_full_workflow_start_answer_advance_complete():
    session = create_session(_make_questions())
    assert get_current_question(session).category == "behavioral"
    assert not is_complete(session)

    feedback = AnswerFeedback.model_validate(FEEDBACK_DICT)
    record_answer(session, "My answer.", feedback)
    assert has_pending_feedback(session)
    assert session.answers == ["My answer."]

    advance(session)
    assert get_current_question(session).category == "technical"

    for _ in range(3):
        record_answer(session, "Answer.", AnswerFeedback.model_validate(FEEDBACK_DICT))
        advance(session)
    assert is_complete(session)
    assert get_current_question(session) is None


def test_create_session_requires_questions():
    with pytest.raises(ValueError, match="without questions"):
        create_session([])


def test_double_answer_rejected():
    session = create_session(_make_questions())
    feedback = AnswerFeedback.model_validate(FEEDBACK_DICT)
    record_answer(session, "First.", feedback)
    with pytest.raises(ValueError, match="already answered"):
        record_answer(session, "Second.", feedback)


def test_advance_without_answer_rejected():
    session = create_session(_make_questions())
    with pytest.raises(ValueError, match="Answer the current question"):
        advance(session)


def test_record_after_complete_rejected():
    session = create_session(_make_questions()[:1])
    feedback = AnswerFeedback.model_validate(FEEDBACK_DICT)
    record_answer(session, "Answer.", feedback)
    advance(session)
    with pytest.raises(ValueError, match="already complete"):
        record_answer(session, "Late.", feedback)
    with pytest.raises(ValueError, match="already complete"):
        advance(session)
