"""Streamlit entry point for AI Career Copilot (Phase 11: copilot agent)."""

import streamlit as st

from app.core.config import settings
from app.ui.ats_page import render_ats_page
from app.ui.chat_page import render_chat_page
from app.ui.components import render_header, render_placeholder, render_sidebar
from app.ui.copilot_page import render_copilot_page
from app.ui.job_page import render_job_page
from app.ui.match_page import render_match_page
from app.ui.interview_page import render_interview_page
from app.ui.optimizer_page import render_optimizer_page
from app.ui.resume_page import render_resume_page
from app.ui.swot_page import render_swot_page


def main() -> None:
    """Render the application shell and route to the selected section."""
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="💼",
        layout="centered",
    )

    render_header()
    selection = render_sidebar()
    if selection == "Copilot":
        render_copilot_page()
    elif selection == "Resume":
        render_resume_page()
    elif selection == "ATS Analysis":
        render_ats_page()
    elif selection == "Job Description":
        render_job_page()
    elif selection == "Job Match":
        render_match_page()
    elif selection == "Resume Optimizer":
        render_optimizer_page()
    elif selection == "SWOT Analysis":
        render_swot_page()
    elif selection == "Interview Coach":
        render_interview_page()
    elif selection == "Career Chat":
        render_chat_page()
    else:
        render_placeholder(selection)

    st.divider()
    if settings.openrouter_api_key:
        st.caption("OPENROUTER_API_KEY is configured.")
    else:
        st.caption("OPENROUTER_API_KEY is not configured yet. See .env.example.")


if __name__ == "__main__":
    main()
