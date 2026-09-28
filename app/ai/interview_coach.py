"""AI interview coach via OpenRouter through LangChain (Phase 9).

Generates interview questions from ResumeProfile (+ optional
JobProfile) and evaluates answers. All OpenRouter/LangChain logic
lives here. Streamlit code must call the public functions and
never construct LLM clients directly. No LangGraph here.
"""

from typing import Any

from pydantic import ValidationError

from app.ai.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    get_chat_model,
    invoke_structured,
)
from app.schemas.interview import AnswerFeedback, InterviewQuestion, InterviewSession
from app.schemas.job import JobProfile
from app.schemas.resume_profile import ResumeProfile

QUESTION_GENERATION_INSTRUCTIONS = """You prepare interview questions from the candidate data below. Return an interview session with exactly {num_questions} questions covering all four categories: behavioral, technical, resume-specific, role-specific.

Rules:
- Base every question on actual resume or job data. Do NOT invent experience for the candidate.
- Do NOT ask about technologies unrelated to the resume/job, unless the technology is an explicit job requirement — then mark it clearly in reason as "(role requirement)".
- related_resume_area names the resume area the question probes (e.g. a skill, role, or project), or null for pure role-requirement questions.
- reason explains in one sentence why this question matters for this candidate and role.
- difficulty is one of: easy, medium, hard.
- Do NOT make medical, psychological, or personality claims or assessments about the candidate.

Resume profile:
---
{resume_json}
---

Job profile:
---
{job_json}
---
"""

ANSWER_FEEDBACK_INSTRUCTIONS = """You review a candidate's interview answer. Give honest, specific feedback on the answer content.

Rules:
- Judge ONLY the answer given for the question below. Do NOT invent facts about the candidate.
- strengths: what the answer did well. weaknesses: where it fell short.
- missing_points: concrete points the answer should have covered.
- clarity: one-sentence assessment of how clearly the answer was expressed.
- relevance: one-sentence assessment of how directly the answer addressed the question.
- suggested_improvement: one concrete way to improve the answer.
- Do NOT make medical, psychological, or personality claims or assessments about the candidate.

Question ({category}, {difficulty}): {question_text}

Candidate answer:
---
{answer}
---
"""


def _coerce_questions(raw: Any) -> list[InterviewQuestion]:
    if isinstance(raw, InterviewSession):
        return raw.questions
    if isinstance(raw, dict):
        try:
            return InterviewSession.model_validate(raw).questions
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned invalid interview questions: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected interview questions or a dictionary."
    )


def _coerce_feedback(raw: Any) -> AnswerFeedback:
    if isinstance(raw, AnswerFeedback):
        return raw
    if isinstance(raw, dict):
        try:
            return AnswerFeedback.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"OpenRouter returned invalid answer feedback: {exc}") from exc
    raise ValueError(
        f"OpenRouter returned an unexpected response type: {type(raw).__name__}. "
        "Expected answer feedback or a dictionary."
    )


def _invoke_structured(chat_model, schema, prompt: str, error_label: str):
    return invoke_structured(
        chat_model,
        schema,
        prompt,
        invalid_label=f"OpenRouter returned {error_label}",
        failure_label=f"{error_label} failed",
    )


def generate_interview_questions(
    resume_profile: ResumeProfile | None,
    job_profile: JobProfile | None = None,
    *,
    num_questions: int = 8,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> list[InterviewQuestion]:
    """Generate grounded interview questions via OpenRouter structured output.

    Raises:
        ValueError: Missing resume profile, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if resume_profile is None:
        raise ValueError("Resume profile is missing. Analyze a resume on the Resume page first.")
    if num_questions < 4:
        raise ValueError("At least 4 questions are required (one per category).")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = QUESTION_GENERATION_INSTRUCTIONS.format(
        num_questions=num_questions,
        resume_json=resume_profile.model_dump_json(),
        job_json=job_profile.model_dump_json() if job_profile is not None else "Not available.",
    )
    return _coerce_questions(
        _invoke_structured(chat_model, InterviewSession, prompt, "invalid interview questions")
    )


def evaluate_interview_answer(
    question: InterviewQuestion,
    answer: str,
    *,
    model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> AnswerFeedback:
    """Evaluate a single interview answer via OpenRouter structured output.

    Raises:
        ValueError: Empty answer, missing API key, or malformed output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    if not answer or not answer.strip():
        raise ValueError("Answer is empty. Write an answer before submitting.")

    chat_model = model if model is not None else get_chat_model(api_key=api_key, model_name=model_name)
    prompt = ANSWER_FEEDBACK_INSTRUCTIONS.format(
        category=question.category,
        difficulty=question.difficulty,
        question_text=question.question,
        answer=answer.strip(),
    )
    return _coerce_feedback(
        _invoke_structured(chat_model, AnswerFeedback, prompt, "invalid answer feedback")
    )
