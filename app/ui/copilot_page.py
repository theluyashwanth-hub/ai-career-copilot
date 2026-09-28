"""Career Copilot chat interface (Phase 11).

Thin Streamlit wrapper around app.agent.graph.run_copilot.
The LangGraph workflow coordinates existing services; no
analysis logic lives here.
"""

import streamlit as st

from app.agent.graph import run_copilot
from app.core.config import settings
from app.schemas.job import JobProfile
from app.schemas.resume_profile import ResumeProfile

COPILOT_HISTORY_KEY = "copilot_history"

EXAMPLE_REQUESTS = [
    "Analyze my resume.",
    "How well do I match this job?",
    "What should I improve?",
    "Prepare me for this role.",
]


def _handle_request(user_request: str) -> None:
    """Run one copilot turn and append both messages to history."""
    history = st.session_state.setdefault(COPILOT_HISTORY_KEY, [])
    history.append({"role": "user", "content": user_request})

    resume_profile = st.session_state.get("resume_profile")
    job_profile = st.session_state.get("job_profile")
    resume_doc = st.session_state.get("resume_document")
    if not isinstance(resume_profile, ResumeProfile):
        resume_profile = None
    if not isinstance(job_profile, JobProfile):
        job_profile = None
    resume_text = resume_doc.text if hasattr(resume_doc, "text") else None

    if resume_profile is None:
        history.append(
            {
                "role": "assistant",
                "content": "Please upload and analyze a resume on the **Resume** page first.",
            }
        )
        return
    if not settings.openrouter_api_key:
        history.append(
            {
                "role": "assistant",
                "content": "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example.",
            }
        )
        return
    with st.spinner("Copilot is working..."):
        try:
            final = run_copilot(user_request, resume_profile, job_profile, resume_text)
        except Exception as exc:  # defensive: never crash the page
            history.append({"role": "assistant", "content": f"Sorry — copilot failed: {exc}"})
            return
    response = final.get("final_response") or "Sorry — I could not produce a response."
    history.append({"role": "assistant", "content": response})


def render_copilot_page() -> None:
    """Render the Career Copilot chat interface."""
    st.header("Career Copilot")
    st.write("Ask about your resume, job fit, improvements, or role preparation.")
    st.caption("Try an example or type your own request below.")

    cols = st.columns(2)
    for index, example in enumerate(EXAMPLE_REQUESTS):
        with cols[index % 2]:
            if st.button(example, key=f"copilot_example_{index}"):
                _handle_request(example)
                st.rerun()

    history = st.session_state.setdefault(COPILOT_HISTORY_KEY, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("e.g. Am I a good match for this job and what should I improve?")
    if prompt:
        _handle_request(prompt)
        st.rerun()

    if history and st.button("Clear copilot chat"):
        st.session_state.pop(COPILOT_HISTORY_KEY, None)
        st.rerun()
