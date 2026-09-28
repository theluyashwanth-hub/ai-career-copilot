"""Resume-vs-job matching service (Phase 6).

Deterministic comparison covers exact/normalized skills and
keywords. Semantic comparison (experience, responsibilities,
project relevance, transferable skills) comes from the AI layer
and is combined here. Never claims a skill the resume lacks:
semantic transferable skills are filtered against the resume set.
"""

import re
from dataclasses import dataclass, field

from app.ai.openrouter_client import DEFAULT_OPENROUTER_MODEL
from app.schemas.job import JobProfile
from app.schemas.job_match import JobMatchAnalysis, JobMatchSemanticSignals
from app.schemas.resume_profile import ResumeProfile

SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "py": "python",
    "postgres": "postgresql",
    "ml": "machine learning",
}

STOPWORDS = {
    "and", "the", "with", "for", "our", "you", "your", "will",
    "are", "have", "has", "from", "into", "over", "per", "plus",
    "years", "year", "experience", "experienced", "strong", "ability",
    "work", "working", "team", "teams", "role", "new", "using", "use",
}

TOKEN_RE = re.compile(r"[a-z0-9+#/.]+")


def _normalize_skill(skill: str) -> str:
    """Normalize a skill label for comparison (case, punctuation, aliases)."""
    text = skill.strip().lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9+#/. ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return SKILL_ALIASES.get(text, text)


def _labeled_set(items: list[str]) -> dict[str, str]:
    """Map normalized value -> first-seen display label."""
    labeled: dict[str, str] = {}
    for item in items:
        label = item.strip()
        if not label:
            continue
        labeled.setdefault(_normalize_skill(label), label)
    return labeled


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in TOKEN_RE.findall(text.lower()):
        tok = raw.strip("./")
        if len(tok) >= 3 and tok not in STOPWORDS:
            tokens.add(tok)
    return tokens


@dataclass
class DeterministicMatchResult:
    """Pure deterministic comparison output."""

    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    matching_experience: list[str] = field(default_factory=list)
    missing_experience: list[str] = field(default_factory=list)
    skills_coverage: float = 0.0
    keyword_coverage: float = 0.0
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _resume_skill_set(profile: ResumeProfile) -> dict[str, str]:
    labeled = _labeled_set(profile.skills)
    for cert in profile.certifications:
        label = cert.strip()
        if label:
            labeled.setdefault(_normalize_skill(label), label)
    for project in profile.projects:
        for tech in project.technologies:
            label = tech.strip()
            if label:
                labeled.setdefault(_normalize_skill(label), label)
    return labeled


def _resume_keyword_set(profile: ResumeProfile, skill_labeled: dict[str, str]) -> dict[str, str]:
    labeled = dict(skill_labeled)
    for item in list(profile.achievements):
        for word in _tokens(item):
            labeled.setdefault(word, item.strip()[:60])
    return labeled


def analyze_match_deterministic(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
) -> DeterministicMatchResult:
    """Compare profiles with exact/normalized skill and keyword overlap."""
    if resume_profile is None:
        raise ValueError("Resume profile is missing. Analyze a resume on the Resume page first.")
    if job_profile is None:
        raise ValueError("Job profile is missing. Analyze a job description on the Job Description page first.")

    resume_skills = _resume_skill_set(resume_profile)
    resume_keywords = _resume_keyword_set(resume_profile, resume_skills)
    required = _labeled_set(job_profile.required_skills)
    preferred = _labeled_set(job_profile.preferred_skills)
    job_terms = _labeled_set(job_profile.keywords + job_profile.required_skills + job_profile.preferred_skills)

    matched_required = [label for norm, label in required.items() if norm in resume_skills]
    missing_required = [label for norm, label in required.items() if norm not in resume_skills]
    matched_preferred = [label for norm, label in preferred.items() if norm in resume_skills]
    missing_preferred = [label for norm, label in preferred.items() if norm not in resume_skills]

    matched_skills = matched_required + [s for s in matched_preferred if s not in matched_required]
    missing_skills = missing_required + missing_preferred

    matched_keywords = [label for norm, label in job_terms.items() if norm in resume_keywords]
    missing_keywords = [label for norm, label in job_terms.items() if norm not in resume_keywords]

    req_cov = len(matched_required) / len(required) if required else 1.0
    pref_cov = len(matched_preferred) / len(preferred) if preferred else 1.0
    skills_coverage = 0.7 * req_cov + 0.3 * pref_cov
    if not required and not preferred:
        skills_coverage = 1.0
    keyword_coverage = len(matched_keywords) / len(job_terms) if job_terms else 1.0

    # Experience-requirement overlap against resume experience text.
    exp_corpus = " ".join(
        " ".join(part for part in (exp.role, exp.company, exp.description) if part)
        for exp in resume_profile.experience
    )
    exp_tokens = _tokens(exp_corpus)
    matching_experience: list[str] = []
    missing_experience: list[str] = []
    for req in job_profile.experience_requirements:
        req_toks = _tokens(req)
        if not req_toks:
            continue
        overlap = len(req_toks & exp_tokens) / len(req_toks) if exp_tokens else 0.0
        (matching_experience if overlap >= 0.3 else missing_experience).append(req.strip())

    strengths: list[str] = []
    gaps: list[str] = []
    recommendations: list[str] = []
    if required:
        strengths.append(f"Matched {len(matched_required)} of {len(required)} required skills.")
    if matched_preferred:
        strengths.append(f"Matched {len(matched_preferred)} preferred skills.")
    if not required and not preferred and not job_terms:
        gaps.append("Job profile lists no explicit skills or keywords; comparison is semantic-only.")
    if missing_required:
        gaps.append(f"Missing required skills: {', '.join(missing_required[:6])}.")
        recommendations.append(
            "Close required-skill gaps with truthful evidence (projects, courses) before applying."
        )
    if missing_preferred:
        gaps.append(f"Missing preferred skills: {', '.join(missing_preferred[:6])}.")
    if matching_experience:
        strengths.append(f"Resume experience overlaps {len(matching_experience)} job experience requirement(s).")
    if missing_experience:
        gaps.append(f"{len(missing_experience)} experience requirement(s) with little resume overlap.")
        recommendations.append("Mirror overlapping experience with quantified bullets from the resume.")
    if matched_keywords:
        strengths.append(f"Matched {len(matched_keywords)} job keywords.")
    if missing_keywords:
        recommendations.append(
            "Add missing keywords only where truthful: " + ", ".join(missing_keywords[:5]) + "."
        )

    return DeterministicMatchResult(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
        matching_experience=matching_experience,
        missing_experience=missing_experience,
        skills_coverage=skills_coverage,
        keyword_coverage=keyword_coverage,
        strengths=strengths,
        gaps=gaps,
        recommendations=recommendations,
    )


def _merge_lists(*lists: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for items in lists:
        for item in items:
            text = item.strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                merged.append(text)
            if len(merged) >= limit:
                return merged
    return merged


def combine_job_match(
    deterministic: DeterministicMatchResult,
    semantic: JobMatchSemanticSignals,
    resume_skill_norms: set[str],
) -> JobMatchAnalysis:
    """Blend deterministic coverage with semantic scores.

    Semantic transferable skills are filtered against the resume set
    so the result never claims a skill absent from the resume.
    """
    semantic_avg = (
        semantic.experience_relevance_score
        + semantic.responsibility_alignment_score
        + semantic.project_relevance_score
        + semantic.transferable_skills_score
    ) / 400
    match_score = max(
        0,
        min(
            100,
            round(
                100
                * (
                    0.55 * deterministic.skills_coverage
                    + 0.15 * deterministic.keyword_coverage
                    + 0.30 * semantic_avg
                )
            ),
        ),
    )

    grounded_transferable = [
        skill
        for skill in semantic.transferable_skills
        if skill.strip() and _normalize_skill(skill) in resume_skill_norms
    ]
    matched_skills = _merge_lists(deterministic.matched_skills, grounded_transferable, limit=20)

    return JobMatchAnalysis(
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=list(deterministic.missing_skills[:20]),
        matching_experience=_merge_lists(
            deterministic.matching_experience, semantic.matching_experience, limit=8
        ),
        missing_experience=list(deterministic.missing_experience[:8]),
        matched_keywords=list(deterministic.matched_keywords[:20]),
        missing_keywords=list(deterministic.missing_keywords[:20]),
        strengths=_merge_lists(deterministic.strengths, semantic.strengths, limit=10),
        gaps=_merge_lists(deterministic.gaps, semantic.gaps, limit=10),
        recommendations=_merge_lists(
            deterministic.recommendations, semantic.recommendations, limit=10
        ),
    )


def analyze_job_match(
    resume_profile: ResumeProfile | None,
    job_profile: JobProfile | None,
    *,
    semantic_model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> JobMatchAnalysis:
    """Combine deterministic comparison with OpenRouter semantic comparison.

    Raises:
        ValueError: Missing profiles, missing API key, or malformed LLM output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    from app.ai.match_advisor import evaluate_match_semantics

    if resume_profile is None:
        raise ValueError("Resume profile is missing. Analyze a resume on the Resume page first.")
    if job_profile is None:
        raise ValueError(
            "Job profile is missing. Analyze a job description on the Job Description page first."
        )

    deterministic = analyze_match_deterministic(resume_profile, job_profile)
    semantic = evaluate_match_semantics(
        resume_profile,
        job_profile,
        model=semantic_model,
        api_key=api_key,
        model_name=model_name,
    )
    resume_norms = set(_resume_skill_set(resume_profile).keys())
    return combine_job_match(deterministic, semantic, resume_norms)
