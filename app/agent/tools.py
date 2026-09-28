"""Agent tools (Phase 11).

Thin, deterministic wrappers around existing services. The graph
may only call these — no arbitrary execution, no filesystem or
shell access. Pure helpers (intent, recommendations, response)
are fully deterministic; service calls reuse existing logic.
"""

from app.agent.state import Intent
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.services.ats_analyzer import analyze_ats
from app.services.job_matcher import analyze_job_match

MATCH_KEYWORDS = {"match", "fit", "suitable", "qualified", "chance", "good for"}
JOB_KEYWORDS = {"job", "role", "position", "opening", "vacancy", "posting"}
IMPROVE_KEYWORDS = {"improv", "better", "fix", "weakness", "weak", "gap", "recommend", "advice", "tips"}
PREPARE_KEYWORDS = {"prepar", "interview", "practice", "ready", "coach"}
RESUME_KEYWORDS = {"resume", "cv", "profile", "ats", "review", "summar"}

INTENT_LABELS = {
    "analyze_resume": "Resume analysis",
    "match_job": "Job matching",
    "improve": "Improvement plan",
    "prepare_role": "Role preparation",
    "general": "General career question",
}


def classify_intent(user_request: str) -> tuple[Intent, bool]:
    """Route a request to an intent plus whether a job profile is needed."""
    text = (user_request or "").lower()
    wants_job = any(word in text for word in JOB_KEYWORDS | {"this job", "the job"})
    if any(word in text for word in PREPARE_KEYWORDS):
        return "prepare_role", True
    if any(word in text for word in MATCH_KEYWORDS) and wants_job:
        return "match_job", True
    if wants_job and any(word in text for word in {"match", "fit", "good", "am i"}):
        return "match_job", True
    if any(word in text for word in IMPROVE_KEYWORDS):
        return "improve", wants_job
    if any(word in text for word in RESUME_KEYWORDS):
        return "analyze_resume", False
    if wants_job:
        return "match_job", True
    return "general", False


def run_ats_analysis(
    resume_text: str | None,
    profile: ResumeProfile | None,
    *,
    semantic_model=None,
    api_key: str | None = None,
) -> ATSAnalysis:
    """ATS analysis via the existing service (no duplicated logic)."""
    return analyze_ats(resume_text or "", profile, semantic_model=semantic_model, api_key=api_key)


def run_job_match(
    resume_profile: ResumeProfile | None,
    job_profile: JobProfile | None,
    *,
    semantic_model=None,
    api_key: str | None = None,
) -> JobMatchAnalysis:
    """Job matching via the existing service (no duplicated logic)."""
    return analyze_job_match(
        resume_profile, job_profile, semantic_model=semantic_model, api_key=api_key
    )


def build_recommendations(
    ats: ATSAnalysis | None,
    match: JobMatchAnalysis | None,
) -> list[str]:
    """Deterministically merge service recommendations (max 8, deduped)."""
    merged: list[str] = []
    seen: set[str] = set()
    pools: list[list[str]] = []
    if match is not None:
        if match.missing_skills:
            pools.append([f"Close skill gap (truthfully): {', '.join(match.missing_skills[:4])}."])
        pools.append(match.recommendations)
    if ats is not None:
        pools.append(ats.recommendations)
    for pool in pools:
        for item in pool:
            text = item.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                merged.append(text)
            if len(merged) >= 8:
                return merged
    return merged


def compose_response(
    intent: Intent,
    ats: ATSAnalysis | None,
    match: JobMatchAnalysis | None,
    recommendations: list[str],
) -> str:
    """Deterministically compose the final structured response."""
    lines = [f"**{INTENT_LABELS.get(intent, 'Career Copilot')}**"]
    if match is not None:
        lines.append(f"Match score: **{match.match_score}/100**.")
        if match.matched_skills:
            lines.append(f"Matched skills: {', '.join(match.matched_skills[:8])}.")
        if match.missing_skills:
            lines.append(f"Missing skills: {', '.join(match.missing_skills[:8])}.")
        if match.gaps:
            lines.append(f"Key gaps: {'; '.join(match.gaps[:4])}")
    if ats is not None:
        lines.append(f"ATS readiness: **{ats.overall_score}/100**.")
        if ats.weaknesses:
            lines.append(f"ATS weaknesses: {'; '.join(ats.weaknesses[:4])}")
    if intent == "prepare_role":
        lines.append("Next step: practice role-specific questions on the **Interview Coach** page.")
    if recommendations:
        lines.append("Recommendations:")
        lines.extend(f"{i}. {rec}" for i, rec in enumerate(recommendations, 1))
    else:
        lines.append("No specific recommendations — your materials already look solid.")
    return "\n".join(lines)
