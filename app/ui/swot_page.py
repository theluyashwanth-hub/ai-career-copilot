"""SWOT Analysis dashboard (Phase 8).

Thin Streamlit wrapper around app.ai.swot_analyzer.
Reuses ResumeProfile, ATSAnalysis, and JobMatchAnalysis from
session state. No LLM logic lives here.
"""

import streamlit as st

from app.ai.swot_analyzer import analyze_swot
from app.core.config import settings
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.schemas.swot import SWOTAnalysis

SWOT_STATE_KEYS = ("swot_result", "swot_error", "swot_key")


def _render_quadrant(title: str, icon: str, items: list[str], empty_message: str) -> None:
    st.subheader(f"{icon} {title}")
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.write(empty_message)


def _render_swot(result: SWOTAnalysis) -> None:
    top_left, top_right = st.columns(2)
    with top_left:
        _render_quadrant("Strengths", "💪", result.strengths, "No strengths identified.")
    with top_right:
        _render_quadrant("Weaknesses", "🔍", result.weaknesses, "No weaknesses identified.")
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        _render_quadrant("Opportunities", "🚀", result.opportunities, "No opportunities identified.")
    with bottom_right:
        _render_quadrant("Threats", "⚠️", result.threats, "No threats identified.")


def render_swot_page() -> None:
    """Render the SWOT Analysis dashboard."""
    st.header("SWOT Analysis")
    st.write(
        "A career SWOT grounded entirely in your existing data. "
        "Each point cites the evidence behind it; nothing is invented."
    )

    resume_profile = st.session_state.get("resume_profile")
    ats_analysis = st.session_state.get("ats_result")
    job_match = st.session_state.get("match_result")
    has_resume = isinstance(resume_profile, ResumeProfile)
    has_ats = isinstance(ats_analysis, ATSAnalysis)
    has_match = isinstance(job_match, JobMatchAnalysis)

    col1, col2, col3 = st.columns(3)
    with col1:
        if has_resume:
            st.success("Resume profile ready.")
        else:
            st.warning("No resume profile. Analyze one on the **Resume** page.")
    with col2:
        if has_ats:
            st.success("ATS analysis ready.")
        else:
            st.info("No ATS analysis. Optional — run it on the **ATS Analysis** page.")
    with col3:
        if has_match:
            st.success("Job match ready.")
        else:
            st.info("No job match. Optional — run it on the **Job Match** page.")

    if not has_resume:
        st.info("Generate a SWOT once a resume profile is available.")
        return

    swot_key = (
        f"{st.session_state.get('resume_file_id', 'resume')}"
        f"::{st.session_state.get('ats_file_id', 'no-ats')}"
        f"::{st.session_state.get('match_key', 'no-match')}"
    )

    if st.button("Generate SWOT", type="primary"):
        if not settings.openrouter_api_key:
            st.session_state["swot_result"] = None
            st.session_state["swot_key"] = swot_key
            st.session_state["swot_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Generating SWOT analysis with OpenRouter..."):
                try:
                    result = analyze_swot(
                        resume_profile,
                        ats_analysis if has_ats else None,
                        job_match if has_match else None,
                    )
                except ValueError as exc:
                    st.session_state["swot_result"] = None
                    st.session_state["swot_key"] = swot_key
                    st.session_state["swot_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["swot_result"] = None
                    st.session_state["swot_key"] = swot_key
                    st.session_state["swot_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["swot_result"] = None
                    st.session_state["swot_key"] = swot_key
                    st.session_state["swot_error"] = f"SWOT analysis failed: {exc}"
                else:
                    st.session_state["swot_result"] = result
                    st.session_state["swot_key"] = swot_key
                    st.session_state["swot_error"] = None

    result = st.session_state.get("swot_result")
    error = st.session_state.get("swot_error")
    stored_key = st.session_state.get("swot_key")
    if stored_key != swot_key:
        for key in SWOT_STATE_KEYS:
            st.session_state.pop(key, None)
        if stored_key is not None:
            st.info("Your data changed since the last SWOT. Click **Generate SWOT** again.")
        return
    if error:
        st.error(error)
    elif isinstance(result, SWOTAnalysis):
        _render_swot(result)
