"""Tests for Phase 5 job description analysis (OpenRouter mocked)."""

import pytest

from app.ai.job_analyzer import JOB_EXTRACTION_INSTRUCTIONS, analyze_job
from app.schemas.job import JobProfile


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


VALID_JOB_DICT = {
    "job_title": "Senior Backend Engineer",
    "company": "Acme Corp",
    "required_skills": ["Python", "SQL"],
    "preferred_skills": ["Kubernetes"],
    "responsibilities": ["Build and maintain payments APIs."],
    "qualifications": ["B.Sc. in Computer Science."],
    "experience_requirements": ["5+ years of backend experience."],
    "keywords": ["Python", "SQL", "APIs"],
}

SAMPLE_JD = """Senior Backend Engineer at Acme Corp.
Must have Python and SQL. Bonus: Kubernetes.
You will build payments APIs. Requires 5+ years of experience."""


def test_valid_job_profile_from_mocked_openrouter():
    fake = _FakeModel(result=dict(VALID_JOB_DICT))
    profile = analyze_job(SAMPLE_JD, model=fake)
    assert isinstance(profile, JobProfile)
    assert profile.job_title == "Senior Backend Engineer"
    assert profile.company == "Acme Corp"
    assert profile.required_skills == ["Python", "SQL"]
    assert profile.preferred_skills == ["Kubernetes"]
    assert profile.responsibilities == ["Build and maintain payments APIs."]
    assert profile.qualifications == ["B.Sc. in Computer Science."]
    assert profile.experience_requirements == ["5+ years of backend experience."]
    assert profile.keywords == ["Python", "SQL", "APIs"]
    assert fake.schemas == [JobProfile]
    assert "Senior Backend Engineer" in fake._structured.prompts[0]


def test_missing_fields_default_to_null_or_empty():
    fake = _FakeModel(result={})
    profile = analyze_job("Some minimal job ad text", model=fake)
    assert profile.job_title is None
    assert profile.company is None
    assert profile.required_skills == []
    assert profile.preferred_skills == []
    assert profile.responsibilities == []
    assert profile.qualifications == []
    assert profile.experience_requirements == []
    assert profile.keywords == []


def test_profile_instance_passthrough():
    expected = JobProfile(job_title="Data Analyst")
    fake = _FakeModel(result=expected)
    assert analyze_job("Data analyst ad", model=fake) is expected


def test_malformed_response_type_rejected():
    fake = _FakeModel(result="not a profile at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        analyze_job(SAMPLE_JD, model=fake)


def test_malformed_response_values_rejected():
    fake = _FakeModel(result={"required_skills": [{"not": "a-string"}]})
    with pytest.raises(ValueError, match="invalid job profile"):
        analyze_job(SAMPLE_JD, model=fake)


def test_empty_input_rejected_without_calling_model():
    fake = _FakeModel(result=dict(VALID_JOB_DICT))
    with pytest.raises(ValueError, match="empty"):
        analyze_job("   ", model=fake)
    assert fake._structured.prompts == []


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        analyze_job(SAMPLE_JD, api_key="")


def test_api_errors_wrapped_as_runtime_error():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="Job analysis failed"):
        analyze_job(SAMPLE_JD, model=fake)


def test_prompt_forbids_invention():
    assert "NEVER invent" in JOB_EXTRACTION_INSTRUCTIONS
    assert "empty list" in JOB_EXTRACTION_INSTRUCTIONS
