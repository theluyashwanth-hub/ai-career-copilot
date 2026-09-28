"""LangGraph wiring (Phase 11).

Fixed state machine: intent -> resume -> job -> (match?) ->
(ats?) -> recommend -> respond. Conditional edges skip analyses
the intent does not need. No loops, no dynamic tools.
"""

from langgraph.graph import END, StateGraph

from app.agent import nodes
from app.agent.state import CopilotState
from app.schemas.job import JobProfile
from app.schemas.resume_profile import ResumeProfile


def should_run_match(state: CopilotState) -> str:
    """Route after the job check: match only for relevant intents with a job."""
    if state.get("error"):
        return "recommend"
    if state.get("intent") in nodes.MATCH_INTENTS and state.get("job_profile") is not None:
        return "match"
    return "ats_branch"


def should_run_ats(state: CopilotState) -> str:
    """Route before ATS: only for useful intents with resume text."""
    if state.get("error"):
        return "recommend"
    if state.get("intent") in nodes.ATS_INTENTS and state.get("resume_text"):
        return "ats"
    return "recommend"


def build_graph():
    """Assemble the controlled Career Copilot workflow."""
    builder = StateGraph(CopilotState)
    builder.add_node("intent", nodes.intent_node)
    builder.add_node("resume", nodes.resume_node)
    builder.add_node("job", nodes.job_node)
    builder.add_node("match", nodes.match_node)
    builder.add_node("ats_branch", lambda state: {})
    builder.add_node("ats", nodes.ats_node)
    builder.add_node("recommend", nodes.recommend_node)
    builder.add_node("respond", nodes.response_node)

    builder.set_entry_point("intent")
    builder.add_edge("intent", "resume")
    builder.add_edge("resume", "job")
    builder.add_conditional_edges(
        "job", should_run_match, {"match": "match", "ats_branch": "ats_branch", "recommend": "recommend"}
    )
    builder.add_edge("match", "ats_branch")
    builder.add_conditional_edges(
        "ats_branch", should_run_ats, {"ats": "ats", "recommend": "recommend"}
    )
    builder.add_edge("ats", "recommend")
    builder.add_edge("recommend", "respond")
    builder.add_edge("respond", END)
    return builder.compile()


def run_copilot(
    user_request: str,
    resume_profile: ResumeProfile | None,
    job_profile: JobProfile | None = None,
    resume_text: str | None = None,
    *,
    ats_model=None,
    match_model=None,
    api_key: str | None = None,
) -> CopilotState:
    """Run the full workflow and return the final state.

    Tests inject mock models; the UI passes real session data with
    models left as None (Gemini via settings).
    """
    graph = build_graph()
    initial: CopilotState = {
        "user_request": user_request,
        "intent": "general",
        "needs_job": False,
        "resume_profile": resume_profile,
        "resume_text": resume_text,
        "job_profile": job_profile,
        "ats_analysis": None,
        "job_match": None,
        "recommendations": [],
        "final_response": "",
        "error": None,
        "ats_model": ats_model,
        "match_model": match_model,
        "api_key": api_key,
    }
    return graph.invoke(initial)
