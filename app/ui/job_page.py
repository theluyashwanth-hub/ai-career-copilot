"""Job Description page (Phase 5).

Thin Streamlit wrapper around app.ai.job_analyzer.
No LLM logic lives here; this module only handles input,
session state, and display. No resume matching here.
"""

import streamlit as st

from app.ai.job_analyzer import analyze_job
from app.core.config import settings
from app.schemas.job import JobProfile

JOB_STATE_KEYS = ("job_profile", "job_error", "job_analyzed_input")


def _render_list_section(title: str, items: list[str], empty_message: str) -> None:
    st.subheader(title)
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.write(empty_message)


def _render_job_profile(profile: JobProfile) -> None:
    st.subheader("Job Title")
    st.write(profile.job_title if profile.job_title else "—")

    st.subheader("Company")
    st.write(profile.company if profile.company else "—")

    _render_list_section("Required Skills", profile.required_skills, "No required skills found.")
    _render_list_section("Preferred Skills", profile.preferred_skills, "No preferred skills found.")
    _render_list_section("Responsibilities", profile.responsibilities, "No responsibilities found.")
    _render_list_section("Qualifications", profile.qualifications, "No qualifications found.")
    _render_list_section(
        "Experience Requirements",
        profile.experience_requirements,
        "No experience requirements found.",
    )

    st.subheader("Keywords")
    if profile.keywords:
        st.write(", ".join(profile.keywords))
    else:
        st.write("No keywords found.")


def render_job_page() -> None:
    """Render the Job Description paste-and-analyze page."""
    st.header("Job Description")
    st.write("Paste a job description below to extract its structured profile.")

    job_input = st.text_area(
        "Job description",
        height=220,
        placeholder="Paste the full job description here...",
        help="The text stays in this session; only analysis calls use Gemini.",
    )

    if st.button("Analyze Job", type="primary"):
        if not job_input or not job_input.strip():
            st.session_state["job_profile"] = None
            st.session_state["job_analyzed_input"] = None
            st.session_state["job_error"] = (
                "Job description is empty. Paste a job description before analyzing."
            )
        elif not settings.openrouter_api_key:
            st.session_state["job_profile"] = None
            st.session_state["job_analyzed_input"] = job_input.strip()
            st.session_state["job_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Analyzing job description with OpenRouter..."):
                try:
                    profile = analyze_job(job_input)
                except ValueError as exc:
                    st.session_state["job_profile"] = None
                    st.session_state["job_analyzed_input"] = job_input.strip()
                    st.session_state["job_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["job_profile"] = None
                    st.session_state["job_analyzed_input"] = job_input.strip()
                    st.session_state["job_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["job_profile"] = None
                    st.session_state["job_analyzed_input"] = job_input.strip()
                    st.session_state["job_error"] = f"Job analysis failed: {exc}"
                else:
                    st.session_state["job_profile"] = profile
                    st.session_state["job_analyzed_input"] = job_input.strip()
                    st.session_state["job_error"] = None

    profile = st.session_state.get("job_profile")
    error = st.session_state.get("job_error")
    analyzed_input = st.session_state.get("job_analyzed_input")

    if error and (analyzed_input is None or analyzed_input == (job_input or "").strip()):
        st.error(error)
    elif isinstance(profile, JobProfile) and analyzed_input == (job_input or "").strip():
        _render_job_profile(profile)
    elif isinstance(profile, JobProfile) and analyzed_input is not None:
        st.info("The input changed since the last analysis. Click **Analyze Job** again.")
        _render_job_profile(profile)

    if st.button("Clear job analysis"):
        for key in JOB_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
