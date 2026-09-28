"""Tests for Phase 11 copilot agent (LLM calls mocked, no API key)."""

import pytest

from app.agent import nodes
from app.agent.graph import build_graph, run_copilot, should_run_ats, should_run_match
from app.agent.state import CopilotState
from app.agent.tools import (
    build_recommendations,
    classify_intent,
    compose_response,
)
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile

RESUME = ResumeProfile(name="Jane Doe", skills=["Python", "SQL"])
JOB = JobProfile(job_title="Backend Engineer", required_skills=["Python"])
RESUME_TEXT = "Jane Doe\njane@example.com\n555-123-4567\nSummary:\nPython engineer.\nSkills:\nPython, SQL"

ATS_DICT = {
    "overall_score": 80,
    "formatting_score": 85,
    "keyword_score": 75,
    "experience_score": 80,
    "skills_score": 85,
    "projects_score": 70,
    "education_score": 90,
    "strengths": ["Good skills."],
    "weaknesses": ["Few metrics."],
    "missing_keywords": ["docker"],
    "recommendations": ["Add metrics to bullets."],
}

MATCH_SIGNALS_DICT = {
    "experience_relevance_score": 80,
    "responsibility_alignment_score": 75,
    "project_relevance_score": 60,
    "transferable_skills_score": 85,
    "matching_experience": [],
    "transferable_skills": ["Python"],
    "strengths": [],
    "gaps": ["No Docker evidence."],
    "recommendations": ["Add Docker evidence if truthful."],
}


class _FakeStructured:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def invoke(self, prompt):
        if self._error is not None:
            raise self._error
        return self._result


class _FakeModel:
    def __init__(self, result=None, error=None):
        self._structured = _FakeStructured(result=result, error=error)

    def with_structured_output(self, schema):
        self._schema = schema
        return self._structured


def _base_state(**overrides) -> CopilotState:
    state: CopilotState = {
        "user_request": "How well do I match this job?",
        "intent": "general",
        "needs_job": False,
        "resume_profile": RESUME,
        "resume_text": RESUME_TEXT,
        "job_profile": JOB,
        "ats_analysis": None,
        "job_match": None,
        "recommendations": [],
        "final_response": "",
        "error": None,
        "ats_model": None,
        "match_model": None,
        "api_key": None,
    }
    state.update(overrides)
    return state


# --- Routing ---


def test_classify_intent_match():
    intent, needs_job = classify_intent("Am I a good match for this job?")
    assert intent == "match_job"
    assert needs_job is True


def test_classify_intent_improve():
    intent, _ = classify_intent("What should I improve in my resume?")
    assert intent == "improve"


def test_classify_intent_prepare():
    intent, needs_job = classify_intent("Prepare me for this role.")
    assert intent == "prepare_role"
    assert needs_job is True


def test_classify_intent_analyze_resume():
    intent, needs_job = classify_intent("Analyze my resume.")
    assert intent == "analyze_resume"
    assert needs_job is False


def test_classify_intent_general():
    intent, _ = classify_intent("Hello, what can you do?")
    assert intent == "general"


def test_routing_helpers():
    assert should_run_match(_base_state(intent="match_job")) == "match"
    assert should_run_match(_base_state(intent="analyze_resume")) == "ats_branch"
    assert should_run_match(_base_state(intent="match_job", error="boom")) == "recommend"
    assert should_run_ats(_base_state(intent="analyze_resume")) == "ats"
    assert should_run_ats(_base_state(intent="prepare_role")) == "recommend"
    assert should_run_ats(_base_state(intent="analyze_resume", resume_text=None)) == "recommend"


# --- Nodes ---


def test_intent_node_sets_intent():
    update = nodes.intent_node(_base_state(user_request="How well do I match this job?"))
    assert update["intent"] == "match_job"
    assert update["needs_job"] is True


def test_intent_node_empty_request():
    update = nodes.intent_node(_base_state(user_request="   "))
    assert "empty" in update["error"].lower()


def test_resume_node_missing_profile():
    update = nodes.resume_node(_base_state(resume_profile=None))
    assert "Resume profile is missing" in update["error"]


def test_job_node_missing_job_when_needed():
    update = nodes.job_node(_base_state(needs_job=True, job_profile=None))
    assert "Job profile is missing" in update["error"]


def test_job_node_skips_when_not_needed():
    assert nodes.job_node(_base_state(needs_job=False, job_profile=None)) == {}


def test_match_node_runs_service_with_mock():
    fake = _FakeModel(result=dict(MATCH_SIGNALS_DICT))
    state = _base_state(intent="match_job", match_model=fake)
    update = nodes.match_node(state)
    assert update["job_match"].match_score >= 0
    assert "Python" in update["job_match"].matched_skills


def test_match_node_skips_for_resume_intent():
    state = _base_state(intent="analyze_resume", match_model=_FakeModel(result={}))
    assert nodes.match_node(state) == {}
    assert state.get("job_match") is None


def test_ats_node_runs_service_with_mock():
    fake = _FakeModel(result=dict(ATS_DICT))
    state = _base_state(intent="analyze_resume", ats_model=fake)
    update = nodes.ats_node(state)
    ats = update["ats_analysis"]
    assert isinstance(ats, ATSAnalysis)
    assert 0 <= ats.overall_score <= 100
    # Semantic signals merged in (not deterministic-only).
    assert "Good skills." in ats.strengths
    assert "Add metrics to bullets." in ats.recommendations


def test_ats_node_skips_without_text():
    state = _base_state(intent="analyze_resume", resume_text=None)
    assert nodes.ats_node(state) == {}


def test_recommend_node_merges_deterministically():
    ats = ATSAnalysis.model_validate(ATS_DICT)
    match = JobMatchAnalysis(
        match_score=70,
        matched_skills=["Python"],
        missing_skills=["Docker"],
        recommendations=["Add Docker evidence if truthful."],
    )
    update = nodes.recommend_node(_base_state(ats_analysis=ats, job_match=match))
    recs = update["recommendations"]
    assert any("Docker" in r for r in recs)
    assert any("metrics" in r for r in recs)
    assert len(recs) <= 8


def test_response_node_formats_scores():
    ats = ATSAnalysis.model_validate(ATS_DICT)
    match = JobMatchAnalysis(match_score=72, matched_skills=["Python"], missing_skills=["Docker"])
    update = nodes.response_node(
        _base_state(intent="match_job", ats_analysis=ats, job_match=match,
                    recommendations=["Do X."])
    )
    response = update["final_response"]
    assert "72/100" in response
    assert "80/100" in response
    assert "Do X." in response


def test_response_node_renders_error():
    update = nodes.response_node(_base_state(error="Something broke."))
    assert "Something broke." in update["final_response"]


# --- Complete workflow ---


def test_full_match_workflow_with_mocks():
    final = run_copilot(
        "Am I a good match for this job and what should I improve?",
        RESUME,
        JOB,
        RESUME_TEXT,
        ats_model=_FakeModel(result=dict(ATS_DICT)),
        match_model=_FakeModel(result=dict(MATCH_SIGNALS_DICT)),
    )
    assert final["intent"] == "match_job"
    assert final["error"] is None
    assert final["job_match"] is not None
    assert final["ats_analysis"] is not None
    assert final["recommendations"]
    assert "Match score" in final["final_response"]
    assert "ATS readiness" in final["final_response"]


def test_workflow_missing_resume_short_circuits():
    final = run_copilot(
        "Analyze my resume.",
        None,
        None,
        None,
        ats_model=_FakeModel(result=dict(ATS_DICT)),
        match_model=_FakeModel(result=dict(MATCH_SIGNALS_DICT)),
    )
    assert "Resume profile is missing" in final["error"]
    assert final["job_match"] is None
    assert final["ats_analysis"] is None
    assert "Resume profile is missing" in final["final_response"]


def test_workflow_missing_job_when_needed():
    final = run_copilot(
        "How well do I match this job?",
        RESUME,
        None,
        RESUME_TEXT,
        ats_model=_FakeModel(result=dict(ATS_DICT)),
        match_model=_FakeModel(result=dict(MATCH_SIGNALS_DICT)),
    )
    assert "Job profile is missing" in final["error"]
    assert "Job profile is missing" in final["final_response"]


def test_workflow_resume_only_skips_match():
    final = run_copilot(
        "Analyze my resume.",
        RESUME,
        None,
        RESUME_TEXT,
        ats_model=_FakeModel(result=dict(ATS_DICT)),
        match_model=_FakeModel(result=dict(MATCH_SIGNALS_DICT)),
    )
    assert final["intent"] == "analyze_resume"
    assert final["job_match"] is None
    assert final["ats_analysis"] is not None
    assert "ATS readiness" in final["final_response"]


def test_graph_builds_with_expected_nodes():
    graph = build_graph()
    assert graph is not None


def test_compose_response_prepare_points_to_interview():
    response = compose_response("prepare_role", None, None, [])
    assert "Interview Coach" in response
