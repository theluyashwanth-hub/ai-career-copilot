"""Interview Coach page (Phase 9).

Thin Streamlit wrapper around app.ai.interview_coach and
app.services.interview_coach. Plain functions plus session_state
drive the flow: start, answer, feedback, next. No LLM logic and
no LangGraph here.
"""

import streamlit as st

from app.ai.interview_coach import evaluate_interview_answer, generate_interview_questions
from app.core.config import settings
from app.schemas.interview import AnswerFeedback, InterviewSession
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

INTERVIEW_STATE_KEYS = ("interview_session", "interview_error")


def _render_feedback(feedback: AnswerFeedback) -> None:
    st.subheader("Feedback")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths**")
        if feedback.strengths:
            for item in feedback.strengths:
                st.markdown(f"- ✅ {item}")
        else:
            st.write("—")
    with col2:
        st.markdown("**Weaknesses**")
        if feedback.weaknesses:
            for item in feedback.weaknesses:
                st.markdown(f"- ⚠️ {item}")
        else:
            st.write("—")
    st.markdown("**Missing points**")
    if feedback.missing_points:
        for item in feedback.missing_points:
            st.markdown(f"- {item}")
    else:
        st.write("None — good coverage.")
    st.markdown(f"**Clarity:** {feedback.clarity}")
    st.markdown(f"**Relevance:** {feedback.relevance}")
    st.markdown(f"**Suggested improvement:** {feedback.suggested_improvement}")


def _render_summary(session: InterviewSession) -> None:
    st.subheader("Interview Complete 🎉")
    st.write(f"You answered {len(session.answers)} of {len(session.questions)} questions.")
    for index, (question, answer, feedback) in enumerate(
        zip(session.questions, session.answers, session.feedback), start=1
    ):
        with st.expander(f"Q{index} ({question.category}): {question.question[:80]}", expanded=False):
            st.markdown(f"**Question:** {question.question}")
            st.markdown(f"**Your answer:** {answer}")
            _render_feedback(feedback)


def _clear_state() -> None:
    for key in INTERVIEW_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("interview_answer_input", None)


def render_interview_page() -> None:
    """Render the Interview Coach page."""
    st.header("Interview Coach")
    st.write("Practice with questions generated from your resume and target job.")

    resume_profile = st.session_state.get("resume_profile")
    job_profile = st.session_state.get("job_profile")
    has_resume = isinstance(resume_profile, ResumeProfile)
    has_job = isinstance(job_profile, JobProfile)

    col1, col2 = st.columns(2)
    with col1:
        if has_resume:
            st.success("Resume profile ready.")
        else:
            st.warning("No resume profile. Analyze one on the **Resume** page.")
    with col2:
        if has_job:
            st.success("Job profile ready (role-specific questions enabled).")
        else:
            st.info("No job profile. Optional — analyze one on the **Job Description** page.")

    if not has_resume:
        st.info("Start an interview once a resume profile is available.")
        return

    session = st.session_state.get("interview_session")
    if not isinstance(session, InterviewSession):
        if st.button("Start Interview", type="primary"):
            if not settings.openrouter_api_key:
                st.session_state["interview_error"] = (
                    "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
                )
            else:
                with st.spinner("Generating interview questions with OpenRouter..."):
                    try:
                        questions = generate_interview_questions(
                            resume_profile, job_profile if has_job else None
                        )
                        st.session_state["interview_session"] = create_session(questions)
                        st.session_state["interview_error"] = None
                    except (ValueError, RuntimeError) as exc:
                        st.session_state["interview_error"] = str(exc)
                    except Exception as exc:  # defensive: never crash the page
                        st.session_state["interview_error"] = f"Could not start interview: {exc}"
                st.session_state.pop("interview_answer_input", None)
                st.rerun()
        error = st.session_state.get("interview_error")
        if error:
            st.error(error)
        return

    total = len(session.questions)
    if is_complete(session):
        _render_summary(session)
        if st.button("Restart Interview"):
            _clear_state()
            st.rerun()
        return

    question = get_current_question(session)
    st.caption(f"Question {session.current_question + 1} of {total}")
    st.progress((session.current_question) / total)
    st.subheader(question.question)
    st.caption(f"{question.category} · {question.difficulty} — {question.reason}")

    if has_pending_feedback(session):
        _render_feedback(session.feedback[session.current_question])
        if session.current_question + 1 < total:
            if st.button("Next Question", type="primary"):
                try:
                    advance(session)
                except ValueError as exc:
                    st.session_state["interview_error"] = str(exc)
                st.rerun()
        else:
            if st.button("Finish Interview", type="primary"):
                try:
                    advance(session)
                except ValueError as exc:
                    st.session_state["interview_error"] = str(exc)
                st.rerun()
    else:
        answer = st.text_area(
            "Your answer",
            height=150,
            key="interview_answer_input",
            placeholder="Write your answer here...",
        )
        if st.button("Submit Answer", type="primary"):
            if not answer or not answer.strip():
                st.session_state["interview_error"] = (
                    "Answer is empty. Write an answer before submitting."
                )
            elif not settings.openrouter_api_key:
                st.session_state["interview_error"] = (
                    "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
                )
            else:
                with st.spinner("Evaluating your answer with OpenRouter..."):
                    try:
                        feedback = evaluate_interview_answer(question, answer)
                        record_answer(session, answer, feedback)
                        st.session_state["interview_error"] = None
                    except (ValueError, RuntimeError) as exc:
                        st.session_state["interview_error"] = str(exc)
                    except Exception as exc:  # defensive: never crash the page
                        st.session_state["interview_error"] = f"Could not evaluate answer: {exc}"
            st.rerun()

    error = st.session_state.get("interview_error")
    if error:
        st.error(error)

    with st.expander("Previous questions", expanded=False):
        for index in range(session.current_question):
            q = session.questions[index]
            st.markdown(f"**Q{index + 1} ({q.category}):** {q.question}")

    if st.button("End Interview"):
        _clear_state()
        st.rerun()
