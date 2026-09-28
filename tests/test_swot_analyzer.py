"""Tests for Phase 8 career SWOT analysis (OpenRouter mocked)."""

import pytest

from app.ai.swot_analyzer import SWOT_INSTRUCTIONS, analyze_swot
from app.schemas.ats_analysis import ATSAnalysis
from app.schemas.job_match import JobMatchAnalysis
from app.schemas.resume_profile import ResumeProfile
from app.schemas.swot import SWOTAnalysis

RESUME = ResumeProfile(name="Jane Doe", skills=["React", "Node.js"])
ATS = ATSAnalysis(
    overall_score=80,
    formatting_score=85,
    keyword_score=75,
    experience_score=80,
    skills_score=85,
    projects_score=70,
    education_score=90,
    strengths=["Strong skills coverage."],
    weaknesses=["Few quantified achievements."],
)
MATCH = JobMatchAnalysis(
    match_score=72,
    matched_skills=["React"],
    missing_skills=["Kubernetes"],
)

VALID_SWOT_DICT = {
    "strengths": ["Strong experience with React and Node.js projects (Skills: React, Node.js)."],
    "weaknesses": ["Resume contains limited evidence of automated testing (Skills: no testing entries)."],
    "opportunities": ["Target job requires technologies already demonstrated in projects (Job match: React)."],
    "threats": ["Target job requires a technology absent from the resume (Job match: Kubernetes)."],
}


class _FakeStructured:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return self._result


class _FakeModel:
    def __init__(self, result=None, error=None):
        self._structured = _FakeStructured(result=result, error=error)
        self.schemas: list[type] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self._structured


def test_valid_swot_from_mocked_openrouter():
    fake = _FakeModel(result=dict(VALID_SWOT_DICT))
    result = analyze_swot(RESUME, ATS, MATCH, model=fake)
    assert isinstance(result, SWOTAnalysis)
    assert result.strengths == VALID_SWOT_DICT["strengths"]
    assert result.weaknesses == VALID_SWOT_DICT["weaknesses"]
    assert result.opportunities == VALID_SWOT_DICT["opportunities"]
    assert result.threats == VALID_SWOT_DICT["threats"]
    assert fake.schemas == [SWOTAnalysis]
    prompt = fake._structured.prompts[0]
    assert "Jane Doe" in prompt
    assert "Kubernetes" in prompt


def test_job_match_optional():
    fake = _FakeModel(result={"strengths": ["Good skills (Skills: React)."]})
    result = analyze_swot(RESUME, ATS, None, model=fake)
    assert isinstance(result, SWOTAnalysis)
    assert result.opportunities == []
    assert result.threats == []
    assert "Not available" in fake._structured.prompts[0]


def test_ats_optional():
    fake = _FakeModel(result={"strengths": ["Good skills (Skills: React)."]})
    result = analyze_swot(RESUME, None, None, model=fake)
    assert isinstance(result, SWOTAnalysis)
    assert result.strengths == ["Good skills (Skills: React)."]


def test_profile_instance_passthrough():
    expected = SWOTAnalysis(strengths=["Strong (Skills: React)."])
    fake = _FakeModel(result=expected)
    assert analyze_swot(RESUME, ATS, MATCH, model=fake) is expected


def test_malformed_response_type_rejected():
    fake = _FakeModel(result="not a swot at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        analyze_swot(RESUME, ATS, MATCH, model=fake)


def test_malformed_response_values_rejected():
    fake = _FakeModel(result={"strengths": [{"not": "a-string"}]})
    with pytest.raises(ValueError, match="invalid SWOT analysis"):
        analyze_swot(RESUME, ATS, MATCH, model=fake)


def test_missing_resume_profile_rejected_without_calling_model():
    fake = _FakeModel(result=dict(VALID_SWOT_DICT))
    with pytest.raises(ValueError, match="Resume profile is missing"):
        analyze_swot(None, ATS, MATCH, model=fake)
    assert fake._structured.prompts == []


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        analyze_swot(RESUME, ATS, MATCH, api_key="")


def test_api_errors_wrapped_as_runtime_error():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="SWOT analysis failed"):
        analyze_swot(RESUME, ATS, MATCH, model=fake)


def test_prompt_demands_grounding():
    assert "Do NOT invent" in SWOT_INSTRUCTIONS
    assert "evidence" in SWOT_INSTRUCTIONS
