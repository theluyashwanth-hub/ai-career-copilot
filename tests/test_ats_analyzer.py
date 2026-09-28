"""Tests for Phase 4 ATS analysis (deterministic rules + mocked LLM)."""

import pytest

from app.schemas.ats_analysis import ATSAnalysis, ATSSemanticSignals
from app.schemas.resume_profile import ResumeProfile
from app.services.ats_analyzer import (
    analyze_ats,
    analyze_ats_deterministic,
    combine_ats_analysis,
)

STRONG_RESUME = """Jane Doe
jane.doe@example.com | +1-555-123-4567 | Berlin, Germany

Summary:
Software engineer with 5 years of experience building backend systems.

Experience:
Senior Backend Engineer at Acme Corp, Berlin, 2021 - Present
- Led a team of 5 engineers to rebuild the payments API
- Increased throughput by 40% and reduced latency by 25%
- Built automated testing pipeline serving 2 million users
- Mentored junior engineers and drove agile adoption

Education:
B.Sc. Computer Science
TU Munich, 2016 - 2020

Skills:
Python, SQL, Docker, Kubernetes, AWS, Git, Agile, Testing, APIs, Communication

Projects:
Career Bot - AI assistant built with Python and APIs
- Deployed with Docker to serve 10 thousand monthly users
- Improved response quality by 15% with better prompts
"""

WEAK_RESUME = """John
No contact info here
I did stuff. Responsible for things. Various duties as assigned.
"""


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
        self.schemas: list[type] = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self._structured


SEMANTIC_DICT = {
    "bullet_clarity_score": 80,
    "impact_score": 70,
    "specificity_score": 75,
    "relevance_score": 85,
    "vague_statements": [],
    "achievement_quality_notes": [],
    "technical_description_notes": [],
    "strengths": ["Clear technical bullets."],
    "weaknesses": [],
    "missing_keywords": [],
    "recommendations": ["Add more metrics to early roles."],
}


# --- Deterministic rules ---


def test_strong_resume_scores_high():
    result = analyze_ats_deterministic(STRONG_RESUME)
    assert result.overall_score >= 80
    assert result.formatting_score >= 85
    assert result.skills_score == 100
    assert result.education_score == 90
    assert result.experience_score >= 70
    assert any("email" in s.lower() for s in result.strengths)
    assert any("phone" in s.lower() for s in result.strengths)


def test_missing_contact_detected():
    result = analyze_ats_deterministic(WEAK_RESUME)
    assert any("email" in w.lower() for w in result.weaknesses)
    assert any("phone" in w.lower() for w in result.weaknesses)
    assert any("email" in r.lower() for r in result.recommendations)


def test_missing_sections_detected():
    result = analyze_ats_deterministic(WEAK_RESUME)
    for section in ("summary", "experience", "education", "skills"):
        assert any(section in w.lower() for w in result.weaknesses)


def test_skills_from_profile():
    profile = ResumeProfile(skills=["Python", "SQL", "Git", "Docker", "AWS", "Agile", "Testing", "Linux"])
    result = analyze_ats_deterministic("John Doe\nj@x.com\n555-123-4567", profile)
    assert result.skills_score == 100


def test_no_skills_scores_low():
    result = analyze_ats_deterministic(WEAK_RESUME)
    assert result.skills_score == 10
    assert any("skill" in w.lower() for w in result.weaknesses)


def test_measurable_achievements_rewarded():
    strong = analyze_ats_deterministic(STRONG_RESUME)
    weak = analyze_ats_deterministic(WEAK_RESUME)
    assert strong.experience_score > weak.experience_score
    assert any("quantified" in w.lower() for w in weak.weaknesses)


def test_action_verbs_detected():
    text = "Summary:\nLed the team. Built the API. Improved latency."
    result = analyze_ats_deterministic(f"a@b.com\n555-123-4567\n{text}")
    assert result.keyword_score > analyze_ats_deterministic(WEAK_RESUME).keyword_score


def test_excessive_pipes_penalized():
    text = STRONG_RESUME + "\n" + " | ".join(f"item{i}" for i in range(15))
    assert analyze_ats_deterministic(text).formatting_score < 100


def test_decorative_symbols_penalized():
    text = STRONG_RESUME + "\n◆◆◆◆◆◆◆\n"
    result = analyze_ats_deterministic(text)
    assert result.formatting_score < 100
    assert any("decorative" in w.lower() for w in result.weaknesses)


def test_emoji_penalized():
    text = STRONG_RESUME + "\n🚀 Rockstar developer 🚀\n"
    result = analyze_ats_deterministic(text)
    assert result.formatting_score < 100
    assert any("emoji" in w.lower() for w in result.weaknesses)


def test_empty_section_detected():
    text = "Summary:\nExperience:\nJohn Doe\nj@x.com\n555-123-4567\nDid work."
    result = analyze_ats_deterministic(text)
    assert any("empty" in w.lower() for w in result.weaknesses)


def test_excessively_long_section_detected():
    long_section = "Experience:\n" + "worked " * 500
    result = analyze_ats_deterministic(f"John\nj@x.com\n555-123-4567\n{long_section}")
    assert any("excessively long" in w.lower() for w in result.weaknesses)


def test_missing_keywords_reported():
    result = analyze_ats_deterministic(WEAK_RESUME)
    assert len(result.missing_keywords) > 0
    assert "python" in result.missing_keywords


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        analyze_ats_deterministic("   ")


def test_scores_within_range():
    for text in (STRONG_RESUME, WEAK_RESUME):
        result = analyze_ats_deterministic(text)
        for score in (
            result.overall_score,
            result.formatting_score,
            result.keyword_score,
            result.experience_score,
            result.skills_score,
            result.projects_score,
            result.education_score,
        ):
            assert 0 <= score <= 100


# --- Combine logic ---


def test_combine_is_deterministic_led():
    det = analyze_ats_deterministic(WEAK_RESUME)
    semantic = ATSSemanticSignals(
        bullet_clarity_score=100,
        impact_score=100,
        specificity_score=100,
        relevance_score=100,
    )
    combined = combine_ats_analysis(det, semantic)
    assert isinstance(combined, ATSAnalysis)
    # Deterministic carries 60-70% weight: weak text cannot reach 100.
    assert combined.overall_score < 100
    assert combined.formatting_score == round(0.7 * det.formatting_score + 0.3 * 100)


def test_combine_merges_lists_with_dedup_and_limits():
    det = analyze_ats_deterministic(STRONG_RESUME)
    semantic = ATSSemanticSignals(
        strengths=["Clear technical bullets.", "Clear technical bullets."],
        weaknesses=["Too few metrics in early roles."],
        missing_keywords=["kubernetes", "KUBERNETES "],
        recommendations=["Add more metrics to early roles."],
        vague_statements=["Did stuff"],
    )
    combined = combine_ats_analysis(det, semantic)
    assert "Clear technical bullets." in combined.strengths
    assert len(combined.strengths) <= 12
    assert any("Vague statement to tighten: Did stuff" in w for w in combined.weaknesses)
    assert len(combined.weaknesses) <= 12
    assert combined.missing_keywords.count("kubernetes") == 1
    assert len(combined.missing_keywords) <= 15
    assert len(combined.recommendations) <= 12


# --- Orchestration with mocked LLM ---


def test_analyze_ats_combines_both_signals():
    fake = _FakeModel(result=dict(SEMANTIC_DICT))
    result = analyze_ats(STRONG_RESUME, semantic_model=fake)
    assert isinstance(result, ATSAnalysis)
    assert fake.schemas == [ATSSemanticSignals]
    assert "Clear technical bullets." in result.strengths
    assert "Add more metrics to early roles." in result.recommendations
    assert 0 <= result.overall_score <= 100


def test_analyze_ats_empty_text_rejected():
    fake = _FakeModel(result=dict(SEMANTIC_DICT))
    with pytest.raises(ValueError, match="empty"):
        analyze_ats("  ", semantic_model=fake)


def test_analyze_ats_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        analyze_ats(STRONG_RESUME, api_key="")


def test_analyze_ats_api_error_wrapped():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="ATS semantic evaluation failed"):
        analyze_ats(STRONG_RESUME, semantic_model=fake)


def test_analyze_ats_malformed_llm_output_rejected():
    fake = _FakeModel(result="not structured at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        analyze_ats(STRONG_RESUME, semantic_model=fake)
