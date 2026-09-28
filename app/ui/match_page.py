"""Job Match page (Phase 6).

Thin Streamlit wrapper around app.services.job_matcher.
Reuses the ResumeProfile (Resume page) and JobProfile
(Job Description page) from session state. No parsing,
extraction, or LLM logic lives here.
"""

import streamlit as st

from app.core.config import settings
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.services.job_matcher import analyze_job_match

MATCH_STATE_KEYS = ("match_result", "match_error", "match_key")


def _score_label(score: int) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Partial match"
    return "Weak match"


def _render_list_section(title: str, items: list[str], empty_message: str) -> None:
    st.subheader(title)
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.write(empty_message)


def _render_match(result: JobMatchAnalysis) -> None:
    st.subheader("Match Score")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Match", f"{result.match_score}/100")
    with col2:
        st.progress(result.match_score / 100)
        st.caption(_score_label(result.match_score))

    st.subheader("Skills")
    left, right = st.columns(2)
    with left:
        st.markdown("**Matched skills**")
        if result.matched_skills:
            for skill in result.matched_skills:
                st.markdown(f"- ✅ {skill}")
        else:
            st.write("None matched.")
    with right:
        st.markdown("**Missing skills**")
        if result.missing_skills:
            for skill in result.missing_skills:
                st.markdown(f"- ❌ {skill}")
        else:
            st.write("None missing.")

    _render_list_section(
        "Matching Experience", result.matching_experience, "No directly matching experience found."
    )
    _render_list_section(
        "Missing Experience", result.missing_experience, "No experience gaps detected."
    )

    st.subheader("Keywords")
    left, right = st.columns(2)
    with left:
        st.markdown("**Matched keywords**")
        st.write(", ".join(f"`{kw}`" for kw in result.matched_keywords) or "None matched.")
    with right:
        st.markdown("**Missing keywords**")
        st.write(", ".join(f"`{kw}`" for kw in result.missing_keywords) or "None missing.")

    _render_list_section("Strengths", result.strengths, "No strengths identified yet.")
    _render_list_section("Gaps", result.gaps, "No gaps identified.")
    _render_list_section("Recommendations", result.recommendations, "No recommendations.")


def render_match_page() -> None:
    """Render the Job Match section."""
    st.header("Job Match")
    st.write("Compare your analyzed resume against an analyzed job description.")

    resume_profile = st.session_state.get("resume_profile")
    job_profile = st.session_state.get("job_profile")
    has_resume = isinstance(resume_profile, ResumeProfile)
    has_job = isinstance(job_profile, JobProfile)

    left, right = st.columns(2)
    with left:
        if has_resume:
            name = resume_profile.name or "resume"
            st.success(f"Resume ready ({name}).")
        else:
            st.warning("No analyzed resume. Upload and analyze one on the **Resume** page.")
    with right:
        if has_job:
            title = job_profile.job_title or "job description"
            st.success(f"Job ready ({title}).")
        else:
            st.warning("No analyzed job. Paste and analyze one on the **Job Description** page.")

    if not (has_resume and has_job):
        st.info("Run match analysis once both steps above are complete.")
        return

    resume_part = st.session_state.get("resume_file_id", "resume")
    job_part = st.session_state.get("job_analyzed_input", "job")
    match_key = f"{resume_part}::{job_part}"

    if st.button("Run Match Analysis", type="primary"):
        if not settings.openrouter_api_key:
            st.session_state["match_result"] = None
            st.session_state["match_key"] = match_key
            st.session_state["match_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Comparing resume and job..."):
                try:
                    result = analyze_job_match(resume_profile, job_profile)
                except ValueError as exc:
                    st.session_state["match_result"] = None
                    st.session_state["match_key"] = match_key
                    st.session_state["match_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["match_result"] = None
                    st.session_state["match_key"] = match_key
                    st.session_state["match_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["match_result"] = None
                    st.session_state["match_key"] = match_key
                    st.session_state["match_error"] = f"Job match failed: {exc}"
                else:
                    st.session_state["match_result"] = result
                    st.session_state["match_key"] = match_key
                    st.session_state["match_error"] = None

    result = st.session_state.get("match_result")
    error = st.session_state.get("match_error")
    stored_key = st.session_state.get("match_key")
    if stored_key != match_key:
        for key in MATCH_STATE_KEYS:
            st.session_state.pop(key, None)
        st.info("Resume or job changed since the last match. Click **Run Match Analysis** again.")
        return
    if error:
        st.error(error)
    elif isinstance(result, JobMatchAnalysis):
        _render_match(result)
