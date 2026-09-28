"""Reusable Streamlit UI pieces (Phase 1: placeholders only)."""

import streamlit as st

APP_TITLE = "AI Career Copilot"
APP_SUBTITLE = "Your AI-powered career assistant."

NAV_ITEMS: list[str] = [
    "Copilot",
    "Resume",
    "ATS Analysis",
    "Job Description",
    "Job Match",
    "Resume Optimizer",
    "SWOT Analysis",
    "Interview Coach",
    "Career Chat",
]


def get_nav_items() -> list[str]:
    """Return the sidebar navigation placeholders (a copy)."""
    return list(NAV_ITEMS)


def render_header() -> None:
    """Render the main title and subtitle."""
    st.title(APP_TITLE)
    st.write(APP_SUBTITLE)


def render_sidebar() -> str:
    """Render sidebar navigation placeholders and return the selection."""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio(
        "Go to",
        get_nav_items(),
        index=0,
    )
    return selection


def render_placeholder(section: str) -> None:
    """Render a placeholder body for a future feature section."""
    st.header(section)
    st.info(
        f"The **{section}** section is coming in a future phase. "
        "This is currently a navigation placeholder."
    )
