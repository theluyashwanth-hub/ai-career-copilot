"""Graph nodes (Phase 11).

Small single-purpose steps. Analysis nodes delegate to tools
(existing services); they contain no scoring or matching logic
of their own. Every node is a pure function of state, safe to
unit test with injected mock models.
"""

from app.agent.state import CopilotState
from app.agent.tools import (
    build_recommendations,
    classify_intent,
    compose_response,
    run_ats_analysis,
    run_job_match,
)

ATS_INTENTS = {"analyze_resume", "improve", "match_job"}
MATCH_INTENTS = {"match_job", "improve", "prepare_role"}


def intent_node(state: CopilotState) -> dict:
    """Understand the request: intent + whether a job profile is needed."""
    if not state.get("user_request") or not state["user_request"].strip():
        return {"error": "Request is empty. Ask something like 'How well do I match this job?'."}
    intent, needs_job = classify_intent(state["user_request"])
    return {"intent": intent, "needs_job": needs_job, "error": None}


def resume_node(state: CopilotState) -> dict:
    """Load check: a resume profile is required for every workflow."""
    if state.get("error"):
        return {}
    if state.get("resume_profile") is None:
        return {"error": "Resume profile is missing. Analyze a resume on the Resume page first."}
    return {}


def job_node(state: CopilotState) -> dict:
    """Load check: job-requiring intents need a job profile."""
    if state.get("error"):
        return {}
    if state.get("needs_job") and state.get("job_profile") is None:
        return {
            "error": (
                "Job profile is missing. Analyze a job description on the "
                "Job Description page first."
            )
        }
    return {}


def match_node(state: CopilotState) -> dict:
    """Run job matching for match/improve/prepare intents (existing service)."""
    if state.get("error"):
        return {}
    if state.get("intent") not in MATCH_INTENTS or state.get("job_profile") is None:
        return {}
    try:
        match = run_job_match(
            state.get("resume_profile"),
            state.get("job_profile"),
            semantic_model=state.get("match_model"),
            api_key=state.get("api_key"),
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    return {"job_match": match}


def ats_node(state: CopilotState) -> dict:
    """Run ATS analysis when useful (existing service, no duplicated logic)."""
    if state.get("error"):
        return {}
    if state.get("intent") not in ATS_INTENTS or not state.get("resume_text"):
        return {}
    try:
        ats = run_ats_analysis(
            state.get("resume_text"),
            state.get("resume_profile"),
            semantic_model=state.get("ats_model"),
            api_key=state.get("api_key"),
        )
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc)}
    return {"ats_analysis": ats}


def recommend_node(state: CopilotState) -> dict:
    """Identify gaps: merge service outputs into recommendations."""
    if state.get("error"):
        return {}
    recommendations = build_recommendations(state.get("ats_analysis"), state.get("job_match"))
    return {"recommendations": recommendations}


def response_node(state: CopilotState) -> dict:
    """Generate the final structured response (deterministic template)."""
    if state.get("error"):
        return {"final_response": f"Sorry — {state['error']}"}
    response = compose_response(
        state.get("intent", "general"),
        state.get("ats_analysis"),
        state.get("job_match"),
        state.get("recommendations", []),
    )
    return {"final_response": response}
