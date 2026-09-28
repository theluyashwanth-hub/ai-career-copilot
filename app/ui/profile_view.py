"""Structured profile display (Phase 3). No LLM calls here."""

import streamlit as st

from app.schemas.resume_profile import ResumeProfile


def _value(text: str | None) -> str:
    return text if text else "—"


def render_profile(profile: ResumeProfile) -> None:
    """Render the eight structured profile sections."""
    st.subheader("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {_value(profile.name)}")
        st.write(f"**Email:** {_value(profile.email)}")
    with col2:
        st.write(f"**Phone:** {_value(profile.phone)}")
        st.write(f"**Location:** {_value(profile.location)}")

    st.subheader("Summary")
    st.write(_value(profile.summary))

    st.subheader("Education")
    if not profile.education:
        st.write("No education entries found.")
    for edu in profile.education:
        title = " / ".join(p for p in (edu.degree, edu.field_of_study) if p) or "Education"
        st.markdown(f"- **{title}** — {_value(edu.institution)}")
        dates = " – ".join(p for p in (edu.start_date, edu.end_date) if p)
        if dates:
            st.caption(dates)
        if edu.description:
            st.caption(edu.description)

    st.subheader("Experience")
    if not profile.experience:
        st.write("No experience entries found.")
    for exp in profile.experience:
        st.markdown(f"- **{_value(exp.role)}** at {_value(exp.company)}")
        meta = ", ".join(p for p in (exp.location, exp.start_date, exp.end_date) if p)
        if meta:
            st.caption(meta)
        if exp.description:
            st.caption(exp.description)

    st.subheader("Projects")
    if not profile.projects:
        st.write("No project entries found.")
    for proj in profile.projects:
        st.markdown(f"- **{_value(proj.name)}**")
        if proj.description:
            st.caption(proj.description)
        if proj.technologies:
            st.caption("Technologies: " + ", ".join(proj.technologies))
        if proj.url:
            st.caption(proj.url)

    st.subheader("Skills")
    if profile.skills:
        st.write(", ".join(profile.skills))
    else:
        st.write("No skills found.")

    st.subheader("Certifications")
    if profile.certifications:
        for cert in profile.certifications:
            st.markdown(f"- {cert}")
    else:
        st.write("No certifications found.")

    st.subheader("Achievements")
    if profile.achievements:
        for item in profile.achievements:
            st.markdown(f"- {item}")
    else:
        st.write("No achievements found.")
