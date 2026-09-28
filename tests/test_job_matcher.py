"""Tests for Phase 6 resume-vs-job matching (deterministic + mocked LLM)."""

import pytest

from app.ai.match_advisor import MATCH_SEMANTIC_INSTRUCTIONS, evaluate_match_semantics
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis, JobMatchSemanticSignals
from app.schemas.resume_profile import Experience, ResumeProfile
from app.services.job_matcher import (
    _normalize_skill,
    analyze_job_match,
    analyze_match_deterministic,
    combine_job_match,
)

RESUME = ResumeProfile(
    name="Jane Doe",
    skills=["Python", "SQL", "Docker", "Git"],
    experience=[
        Experience(
            company="Acme Corp",
            role="Backend Engineer",
            description="Built Python payments APIs serving millions of users with SQL and Docker.",
        )
    ],
    projects=[],
    certifications=["AWS Certified"],
)

JOB = JobProfile(
    job_title="Senior Backend Engineer",
    company="Acme Corp",
    required_skills=["Python", "SQL", "Kubernetes"],
    preferred_skills=["Docker", "Go"],
    responsibilities=["Build payments APIs."],
    qualifications=["B.Sc. Computer Science."],
    experience_requirements=["5+ years building backend payments systems with Python."],
    keywords=["Python", "APIs", "Kubernetes"],
)


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


SEMANTIC_DICT = {
    "experience_relevance_score": 80,
    "responsibility_alignment_score": 75,
    "project_relevance_score": 60,
    "transferable_skills_score": 85,
    "matching_experience": ["Built Python payments APIs serving millions of users."],
    "transferable_skills": ["Python", "Docker"],
    "strengths": ["Direct payments API experience."],
    "gaps": ["No Kubernetes evidence."],
    "recommendations": ["Add Kubernetes project evidence if truthful."],
}


# --- Normalization ---


def test_normalize_skill_case_and_aliases():
    assert _normalize_skill("Python") == _normalize_skill("python")
    assert _normalize_skill("  SQL ") == "sql"
    assert _normalize_skill("K8s") == "kubernetes"
    assert _normalize_skill("JS") == "javascript"


# --- Deterministic matching ---


def test_exact_and_normalized_skill_matching():
    resume = ResumeProfile(skills=["python", "K8s"])
    job = JobProfile(required_skills=["Python", "Kubernetes"], preferred_skills=[])
    result = analyze_match_deterministic(resume, job)
    assert sorted(result.matched_skills) == ["Kubernetes", "Python"]
    assert result.missing_skills == []
    assert result.skills_coverage == pytest.approx(1.0)


def test_missing_skills_reported_required_first():
    result = analyze_match_deterministic(RESUME, JOB)
    assert "Kubernetes" in result.missing_skills
    assert "Go" in result.missing_skills
    # Required missing comes before preferred missing.
    assert result.missing_skills.index("Kubernetes") < result.missing_skills.index("Go")
    assert "Python" in result.matched_skills
    assert "Docker" in result.matched_skills


def test_keywords_matched_and_missing():
    result = analyze_match_deterministic(RESUME, JOB)
    assert "Python" in result.matched_keywords
    assert "Kubernetes" in result.missing_keywords


def test_experience_overlap_detection():
    result = analyze_match_deterministic(RESUME, JOB)
    assert result.matching_experience == ["5+ years building backend payments systems with Python."]
    assert result.missing_experience == []


def test_experience_gap_detection():
    job = JobProfile(experience_requirements=["10 years of embedded firmware design."])
    result = analyze_match_deterministic(RESUME, job)
    assert result.matching_experience == []
    assert result.missing_experience == ["10 years of embedded firmware design."]


def test_missing_profiles_rejected():
    with pytest.raises(ValueError, match="Resume profile is missing"):
        analyze_match_deterministic(None, JOB)
    with pytest.raises(ValueError, match="Job profile is missing"):
        analyze_match_deterministic(RESUME, None)


# --- Combine (no fabrication) ---


def test_invented_transferable_skills_filtered_out():
    det = analyze_match_deterministic(RESUME, JOB)
    semantic = JobMatchSemanticSignals(
        transferable_skills=["Python", "Rust", "Cobol"],
    )
    combined = combine_job_match(det, semantic, {"python", "sql", "docker", "git"})
    assert "Python" in combined.matched_skills
    assert "Rust" not in combined.matched_skills
    assert "Cobol" not in combined.matched_skills


def test_combine_score_is_deterministic_led():
    det = analyze_match_deterministic(RESUME, JOB)
    perfect = JobMatchSemanticSignals(
        experience_relevance_score=100,
        responsibility_alignment_score=100,
        project_relevance_score=100,
        transferable_skills_score=100,
    )
    combined = combine_job_match(det, perfect, {"python", "sql"})
    # Missing Kubernetes/Go keeps a perfect semantic score from reaching 100.
    assert combined.match_score < 100
    assert 0 <= combined.match_score <= 100


def test_perfect_match_scores_high():
    resume = ResumeProfile(skills=["Python", "SQL"])
    job = JobProfile(required_skills=["Python", "SQL"], keywords=["Python"])
    det = analyze_match_deterministic(resume, job)
    semantic = JobMatchSemanticSignals(
        experience_relevance_score=90,
        responsibility_alignment_score=90,
        project_relevance_score=90,
        transferable_skills_score=90,
    )
    combined = combine_job_match(det, semantic, {"python", "sql"})
    assert combined.match_score >= 90


# --- Orchestration with mocked LLM ---


def test_analyze_job_match_combines_both_signals():
    fake = _FakeModel(result=dict(SEMANTIC_DICT))
    result = analyze_job_match(RESUME, JOB, semantic_model=fake)
    assert isinstance(result, JobMatchAnalysis)
    assert fake.schemas == [JobMatchSemanticSignals]
    assert "Python" in result.matched_skills
    assert "Kubernetes" in result.missing_skills
    assert "Rust" not in result.matched_skills
    assert "No Kubernetes evidence." in result.gaps
    assert 0 <= result.match_score <= 100


def test_analyze_job_match_missing_profiles_rejected():
    fake = _FakeModel(result=dict(SEMANTIC_DICT))
    with pytest.raises(ValueError, match="Resume profile is missing"):
        analyze_job_match(None, JOB, semantic_model=fake)
    with pytest.raises(ValueError, match="Job profile is missing"):
        analyze_job_match(RESUME, None, semantic_model=fake)


def test_analyze_job_match_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        analyze_job_match(RESUME, JOB, api_key="")


def test_analyze_job_match_api_error_wrapped():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="Job match evaluation failed"):
        analyze_job_match(RESUME, JOB, semantic_model=fake)


def test_analyze_job_match_malformed_output_rejected():
    fake = _FakeModel(result="not structured at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        analyze_job_match(RESUME, JOB, semantic_model=fake)


def test_semantic_prompt_forbids_invention():
    assert "NEVER claim" in MATCH_SEMANTIC_INSTRUCTIONS
    fake = _FakeModel(result=dict(SEMANTIC_DICT))
    signals = evaluate_match_semantics(RESUME, JOB, model=fake)
    assert isinstance(signals, JobMatchSemanticSignals)
    assert "Acme Corp" in fake._structured.prompts[0]
