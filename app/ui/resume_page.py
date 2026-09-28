"""Resume upload + AI profile page (Phase 3).

Thin Streamlit wrapper around app.services.resume_parser and
app.ai.resume_analyzer. No parsing or LLM logic lives here; this
module only handles upload, session state, and display.
"""

import streamlit as st

from app.ai.resume_analyzer import analyze_resume
from app.core.config import settings
from app.schemas.resume_profile import ResumeProfile
from app.services.resume_parser import ResumeDocument, parse_resume
from app.ui.profile_view import render_profile

PROFILE_STATE_KEYS = ("resume_profile", "resume_profile_error", "resume_profile_file_id")


def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    return f"{num_bytes} bytes ({num_bytes / 1024:.1f} KB)"


def _render_document_summary(doc: ResumeDocument) -> None:
    st.success(f"Successfully extracted text from {doc.filename}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filename", doc.filename)
        st.metric("File type", doc.file_type)
    with col2:
        st.metric("File size", _format_file_size(doc.file_size))
        st.metric("Words / Characters", f"{doc.word_count} / {doc.character_count}")
    with st.expander("View extracted resume text", expanded=False):
        st.text(doc.text)


def _clear_profile_state() -> None:
    for key in PROFILE_STATE_KEYS:
        st.session_state.pop(key, None)


def _render_analysis_section(doc: ResumeDocument, file_id: str) -> None:
    """Render the Analyze Resume button, cached profile, and errors."""
    st.divider()
    st.subheader("AI Resume Profile")

    cached_profile = st.session_state.get("resume_profile")
    cached_file_id = st.session_state.get("resume_profile_file_id")
    cached_error = st.session_state.get("resume_profile_error")

    if st.button("Analyze Resume", type="primary"):
        if not doc.text or not doc.text.strip():
            st.session_state["resume_profile"] = None
            st.session_state["resume_profile_file_id"] = file_id
            st.session_state["resume_profile_error"] = "Resume text is empty. Upload a resume before analyzing."
        elif not settings.openrouter_api_key:
            st.session_state["resume_profile"] = None
            st.session_state["resume_profile_file_id"] = file_id
            st.session_state["resume_profile_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Analyzing resume with OpenRouter..."):
                try:
                    profile = analyze_resume(doc.text)
                except ValueError as exc:
                    st.session_state["resume_profile"] = None
                    st.session_state["resume_profile_file_id"] = file_id
                    st.session_state["resume_profile_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["resume_profile"] = None
                    st.session_state["resume_profile_file_id"] = file_id
                    st.session_state["resume_profile_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["resume_profile"] = None
                    st.session_state["resume_profile_file_id"] = file_id
                    st.session_state["resume_profile_error"] = f"Resume analysis failed: {exc}"
                else:
                    st.session_state["resume_profile"] = profile
                    st.session_state["resume_profile_file_id"] = file_id
                    st.session_state["resume_profile_error"] = None

    profile = st.session_state.get("resume_profile")
    error = st.session_state.get("resume_profile_error")
    profile_file_id = st.session_state.get("resume_profile_file_id")

    # Only show results belonging to the currently displayed resume.
    if profile_file_id == file_id:
        if error:
            st.error(error)
        elif isinstance(profile, ResumeProfile):
            render_profile(profile)
    elif cached_file_id is not None and cached_file_id != file_id:
        # Stale cache from a previous file; drop it silently.
        _clear_profile_state()


def render_resume_page() -> None:
    """Render the Resume upload and extraction page."""
    st.header("Resume")
    st.write("Upload your resume as a PDF or DOCX file to extract its text.")

    uploaded_file = st.file_uploader(
        "Drag and drop your resume here",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        help="Supported formats: PDF (.pdf), Word (.docx).",
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("resume_file_id") != file_id:
            # New file: parse once and cache the result in session state.
            with st.spinner(f"Extracting text from {uploaded_file.name}..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    doc = parse_resume(file_bytes, uploaded_file.name)
                except ValueError as exc:
                    st.session_state["resume_file_id"] = file_id
                    st.session_state["resume_document"] = None
                    st.session_state["resume_error"] = str(exc)
                else:
                    st.session_state["resume_file_id"] = file_id
                    st.session_state["resume_document"] = doc
                    st.session_state["resume_error"] = None
                    _clear_profile_state()

        error = st.session_state.get("resume_error")
        doc = st.session_state.get("resume_document")
        if error:
            st.error(error)
        elif isinstance(doc, ResumeDocument):
            st.info("Upload status: file preserved from session state." if st.session_state.get("resume_file_id") else "Upload status: complete.")
            _render_document_summary(doc)
            _render_analysis_section(doc, st.session_state.get("resume_file_id", file_id))

        if st.button("Clear uploaded resume"):
            for key in ("resume_file_id", "resume_document", "resume_error"):
                st.session_state.pop(key, None)
            _clear_profile_state()
            st.rerun()
    else:
        # No file widget value this run: keep showing the cached document.
        doc = st.session_state.get("resume_document")
        error = st.session_state.get("resume_error")
        if isinstance(doc, ResumeDocument):
            st.info("Showing previously uploaded resume stored in this session.")
            _render_document_summary(doc)
            _render_analysis_section(doc, st.session_state.get("resume_file_id", ""))
            if st.button("Clear uploaded resume"):
                for key in ("resume_file_id", "resume_document", "resume_error"):
                    st.session_state.pop(key, None)
                _clear_profile_state()
                st.rerun()
        elif error:
            st.error(error)
        else:
            st.info("Upload status: no file uploaded yet.")
