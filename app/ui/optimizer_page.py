"""Resume Optimizer page (Phase 7).

Thin Streamlit wrapper around app.ai.bullet_optimizer.
No LLM logic lives here; this module only handles input,
session state, and display.
"""

import streamlit as st

from app.ai.bullet_optimizer import optimize_bullet
from app.core.config import settings
from app.schemas.bullet import OptimizedBullet

OPTIMIZER_STATE_KEYS = ("optimized_bullet", "optimizer_error")


def _render_result(result: OptimizedBullet) -> None:
    st.subheader("Original")
    st.write(result.original)

    st.subheader("Improved Bullet")
    st.success(result.optimized)

    st.subheader("Explanation")
    if result.improvements:
        for item in result.improvements:
            st.markdown(f"- {item}")
    else:
        st.write("No explanation provided.")

    st.subheader("Missing Metrics")
    if result.missing_metrics:
        st.info(
            "The original bullet has no metrics, so none were invented. "
            "Consider adding real numbers for:"
        )
        for item in result.missing_metrics:
            st.markdown(f"- {item}")
    else:
        st.write("No missing metrics identified.")

    st.subheader("Action Verbs")
    if result.action_verbs:
        st.write(", ".join(result.action_verbs))
    else:
        st.write("No action verbs identified.")

    st.subheader("X-Y-Z Breakdown")
    breakdown = result.xyz_breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Accomplishment (X)**")
        st.write(breakdown.accomplishment or "—")
    with col2:
        st.markdown("**Measurement (Y)**")
        st.write(breakdown.measurement or "—")
    with col3:
        st.markdown("**Method (Z)**")
        st.write(breakdown.method or "—")


def render_optimizer_page() -> None:
    """Render the Resume Optimizer page."""
    st.header("Resume Optimizer")
    st.write(
        "Paste a single resume bullet to get an improved version. "
        "Metrics are never invented — missing ones are flagged instead."
    )

    bullet_input = st.text_area(
        "Resume bullet",
        height=100,
        placeholder="e.g. Worked on the payments API...",
        help="One bullet at a time for focused feedback.",
    )
    job_input = st.text_area(
        "Job description (optional)",
        height=150,
        placeholder="Paste a job description to adapt wording toward its keywords...",
        help="Only adapts wording truthfully; no experience is added.",
    )

    if st.button("Optimize", type="primary"):
        if not bullet_input or not bullet_input.strip():
            st.session_state["optimized_bullet"] = None
            st.session_state["optimizer_error"] = (
                "Bullet is empty. Enter a resume bullet before optimizing."
            )
        elif not settings.openrouter_api_key:
            st.session_state["optimized_bullet"] = None
            st.session_state["optimizer_error"] = (
                "OPENROUTER_API_KEY is not configured. Add it to your .env file. See .env.example."
            )
        else:
            with st.spinner("Optimizing bullet with OpenRouter..."):
                try:
                    result = optimize_bullet(
                        bullet_input,
                        job_description=job_input if job_input and job_input.strip() else None,
                    )
                except ValueError as exc:
                    st.session_state["optimized_bullet"] = None
                    st.session_state["optimizer_error"] = str(exc)
                except RuntimeError as exc:
                    st.session_state["optimized_bullet"] = None
                    st.session_state["optimizer_error"] = str(exc)
                except Exception as exc:  # defensive: never crash the page
                    st.session_state["optimized_bullet"] = None
                    st.session_state["optimizer_error"] = f"Bullet optimization failed: {exc}"
                else:
                    st.session_state["optimized_bullet"] = result
                    st.session_state["optimizer_error"] = None

    result = st.session_state.get("optimized_bullet")
    error = st.session_state.get("optimizer_error")
    if error:
        st.error(error)
    elif isinstance(result, OptimizedBullet):
        _render_result(result)

    if st.button("Clear optimization"):
        for key in OPTIMIZER_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
