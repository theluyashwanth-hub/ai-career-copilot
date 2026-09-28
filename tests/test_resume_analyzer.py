"""Tests for Phase 3 AI resume-profile extraction (OpenRouter mocked)."""

import pytest

from app.ai.resume_analyzer import (
    EXTRACTION_INSTRUCTIONS,
    analyze_resume,
    get_chat_model,
)
from app.schemas.resume_profile import ResumeProfile


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


VALID_PROFILE_DICT = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+1-555-0100",
    "location": "Berlin, Germany",
    "summary": "Software engineer with 5 years of Python experience.",
    "education": [
        {"institution": "TU Munich", "degree": "B.Sc.", "field_of_study": "Computer Science"}
    ],
    "experience": [
        {"company": "Acme Corp", "role": "Backend Engineer", "location": "Berlin"}
    ],
    "projects": [
        {"name": "Career Bot", "description": "AI assistant", "technologies": ["Python"]}
    ],
    "skills": ["Python", "SQL"],
    "certifications": ["AWS Certified"],
    "achievements": ["Shipped X to 1M users"],
}


def test_valid_profile_from_mocked_openrouter():
    fake = _FakeModel(result=dict(VALID_PROFILE_DICT))
    profile = analyze_resume("Jane Doe\nSoftware engineer", model=fake)
    assert isinstance(profile, ResumeProfile)
    assert profile.name == "Jane Doe"
    assert profile.email == "jane@example.com"
    assert profile.skills == ["Python", "SQL"]
    assert profile.education[0].institution == "TU Munich"
    assert profile.experience[0].company == "Acme Corp"
    assert profile.projects[0].technologies == ["Python"]
    # Structured output was requested with the ResumeProfile schema.
    assert fake.schemas == [ResumeProfile]
    # Resume text reaches the prompt.
    assert "Jane Doe" in fake._structured.prompts[0]


def test_missing_fields_default_to_null_or_empty():
    fake = _FakeModel(result={})
    profile = analyze_resume("Some minimal resume text", model=fake)
    assert profile.name is None
    assert profile.email is None
    assert profile.summary is None
    assert profile.education == []
    assert profile.experience == []
    assert profile.projects == []
    assert profile.skills == []
    assert profile.certifications == []
    assert profile.achievements == []


def test_profile_instance_passthrough():
    expected = ResumeProfile(name="John", skills=["Go"])
    fake = _FakeModel(result=expected)
    assert analyze_resume("John resume", model=fake) is expected


def test_malformed_response_type_rejected():
    fake = _FakeModel(result="not a profile at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        analyze_resume("Some resume text", model=fake)


def test_malformed_response_values_rejected():
    # skills must be a list of strings; a dict entry is invalid.
    fake = _FakeModel(result={"skills": [{"not": "a-string"}]})
    with pytest.raises(ValueError, match="invalid resume profile"):
        analyze_resume("Some resume text", model=fake)


def test_empty_resume_text_rejected_without_calling_model():
    fake = _FakeModel(result=dict(VALID_PROFILE_DICT))
    with pytest.raises(ValueError, match="empty"):
        analyze_resume("   ", model=fake)
    assert fake._structured.prompts == []


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        get_chat_model(api_key="")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        analyze_resume("Some resume text", api_key="")


def test_api_errors_wrapped_as_runtime_error():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="Resume analysis failed"):
        analyze_resume("Some resume text", model=fake)


def test_prompt_forbids_invention():
    assert "NEVER invent" in EXTRACTION_INSTRUCTIONS
    assert "empty list" in EXTRACTION_INSTRUCTIONS
