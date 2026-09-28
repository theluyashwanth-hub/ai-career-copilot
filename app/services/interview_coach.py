"""Interview session workflow (Phase 9).

Plain-Python state transitions over InterviewSession. No LLM calls
and no LangGraph here; the Streamlit page drives these functions
with session_state.
"""

from app.schemas.interview import AnswerFeedback, InterviewQuestion, InterviewSession


def create_session(questions: list[InterviewQuestion]) -> InterviewSession:
    """Start a new interview session from generated questions."""
    if not questions:
        raise ValueError("Cannot start an interview without questions.")
    return InterviewSession(questions=list(questions))


def get_current_question(session: InterviewSession) -> InterviewQuestion | None:
    """Return the question awaiting an answer, or None when complete."""
    if session.current_question >= len(session.questions):
        return None
    return session.questions[session.current_question]


def is_complete(session: InterviewSession) -> bool:
    """True once the flow advanced past the final question."""
    return session.current_question >= len(session.questions)


def has_pending_feedback(session: InterviewSession) -> bool:
    """True when the current question was answered but not yet advanced."""
    return len(session.feedback) > session.current_question


def record_answer(
    session: InterviewSession,
    answer: str,
    feedback: AnswerFeedback,
) -> InterviewSession:
    """Record the answer and feedback for the current question."""
    if is_complete(session):
        raise ValueError("Interview is already complete.")
    if has_pending_feedback(session):
        raise ValueError("Current question already answered. Continue to the next question.")
    if not answer or not answer.strip():
        raise ValueError("Answer is empty.")
    session.answers.append(answer.strip())
    session.feedback.append(feedback)
    return session


def advance(session: InterviewSession) -> InterviewSession:
    """Move to the next question after the current one is answered."""
    if is_complete(session):
        raise ValueError("Interview is already complete.")
    if not has_pending_feedback(session):
        raise ValueError("Answer the current question before continuing.")
    session.current_question += 1
    return session
