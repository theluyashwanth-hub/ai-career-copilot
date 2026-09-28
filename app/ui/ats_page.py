"""ATS Analysis page (Phase 4).

Thin Streamlit wrapper around app.services.ats_analyzer.
No scoring or LLM logic lives here; this module only handles
session state and display.
"""

import streamlit as st

from app.core.config import settings
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.services.ats_analyzer import analyze_ats
from app.services.resume_parser import ResumeDocument, parse_resume

ATS_STATE_KEYS = ("ats_result", "ats_error", "ats_file_id")


def _score_label(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Needs work"
    return "Weak"


def _render_score_row(label: str, score: int) -> None:
    st.write(f"**{label}** — {score}/100 ({_score_label(score)})")
    st.progress(score / 100)


def _render_analysis(result: ATSAnalysis) -> None:
    st.subheader("Overall Score")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("ATS Readiness", f"{result.overall_score}/100")
    with col2:
        st.progress(result.overall_score / 100)
        st.caption(_score_label(result.overall_score))

    st.subheader("Category Scores")
    left, right = st.columns(2)
    with left:
        _render_score_row("Formatting", result.formatting_score)
        _render_score_row("Keywords", result.keyword_score)
        _render_score_row("Experience", result.experience_score)
    with right:
        _render_score_row("Skills", result.skills_score)
        _render_score_row("Projects", result.projects_score)
        _render_score_row("Education", result.education_score)

    st.subheader("Strengths")
    if result.strengths:
        for item in result.strengths:
            st.markdown(f"- ✅ {item}")
    else:
        st.write("No strengths identified yet.")

    st.subheader("Weaknesses")
    if result.weaknesses:
        for item in result.weaknesses:
            st.markdown(f"- ⚠️ {item}")
    else:
        st.write("No weaknesses identified.")

    st.subheader("Missing Keywords")
    if result.missing_keywords:
        st.write(", ".join(f"`{kw}`" for kw in result.missing_keywords))
    else:
        st.write("No missing keywords identified.")

    st.subheader("Recommendations")
    if result.recommendations:
        for index, item in enumerate(result.recommendations, start=1):
            st.markdown(f"{index}. {item}")
    else:
        st.write("No recommendations.")


def _ensure_resume() -> ResumeDocument | None:
    """Return the session resume, offering an inline upload fallback."""
    doc = st.session_state.get("resume_document")
    if isinstance(doc, ResumeDocument):
        return doc
    st.info("Upload a resume to run ATS analysis. You can also upload on the Resume page.")
    uploaded = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        help="Supported formats: PDF (.pdf), Word (.docx).",
        key="ats_uploader",
    )
    if uploaded is None:
        return None
    try:
        doc = parse_resume(uploaded.getvalue(), uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        return None
    st.session_state["resume_file_id"] = f"{uploaded.name}:{uploaded.size}"
    st.session_state["resume_document"] = doc
    st.session_state["resume_error"] = None
    return doc


def render_ats_page() -> None:
    """Render the ATS Analysis page."""
    st.header("ATS Analysis")
    st.write(
        "Checks your resume with deterministic ATS rules plus "
        "OpenRouter semantic review, combined into one readiness score."
    )

    doc = _ensure_resume()
    if doc is None:
        return
    st.caption(f"Analyzing: {doc.filename} ({doc.word_count} words)")

    file_id = st.session_state.get("resume_file_id", doc.filename)
    profile = st.session_state.get("resume_profile")
    if not isinstance(profile, ResumeProfile):
        profile = None

    if st.button("Analyze ATS", type="primary"):
        if not settings.openrouter_api_key:
            st.session_state["ats_result"] = None
            st.session_state["ats_file_id"] = file_id
            st.session_state["ats_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Running ATS analysis..."):
                try:
                    result = analyze_ats(doc.text, profile)
                except ValueError as exc:
                    st.session_state["ats_result"] = None
                    st.session_state["ats_file_id"] = file_id
                    st.session_state["ats_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["ats_result"] = None
                    st.session_state["ats_file_id"] = file_id
                    st.session_state["ats_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["ats_result"] = None
                    st.session_state["ats_file_id"] = file_id
                    st.session_state["ats_error"] = f"ATS analysis failed: {exc}"
                else:
                    st.session_state["ats_result"] = result
                    st.session_state["ats_file_id"] = file_id
                    st.session_state["ats_error"] = None

    result = st.session_state.get("ats_result")
    error = st.session_state.get("ats_error")
    result_file_id = st.session_state.get("ats_file_id")
    if result_file_id != file_id:
        for key in ATS_STATE_KEYS:
            st.session_state.pop(key, None)
        return
    if error:
        st.error(error)
    elif isinstance(result, ATSAnalysis):
        _render_analysis(result)
