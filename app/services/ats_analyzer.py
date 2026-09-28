"""ATS readiness analysis service (Phase 4).

Combines deterministic resume checks with LLM semantic signals into
a final ATSAnalysis. Deterministic scoring lives here; LLM calls
live in app.ai.ats_advisor and are injected, never constructed here.
"""

import re
from dataclasses import dataclass, field

from app.ai.openrouter_client import DEFAULT_OPENROUTER_MODEL
from app.schemas.ats_analysis import ATSAnalysis, ATSSemanticSignals
from app.schemas.resume_profile import ResumeProfile

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+\d[\d\s\-().]{6,}\d|\d{3}[\s\-.]\d{3}[\s\-.]\d{4}|\(\d{3}\)\s*\d{3}[\s\-.]\d{4})")
MEASUREMENT_RE = re.compile(
    r"(\d+\s?%|\$\s?\d[\d,]*|\d[\d,]*\s?(million|billion|thousand|k\b|\+|percent))",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"\d+")

ACTION_VERBS = [
    "led", "developed", "built", "designed", "implemented", "improved",
    "increased", "reduced", "created", "launched", "managed", "mentored",
    "automated", "optimized", "delivered", "drove", "owned", "spearheaded",
    "architected", "migrated", "scaled", "shipped", "streamlined", "tested",
    "deployed", "analyzed", "collaborated", "coordinated", "achieved", "accelerated",
]

SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "objective", "profile"},
    "experience": {"experience", "work experience", "employment", "work history", "professional experience"},
    "education": {"education", "academic background", "qualifications"},
    "skills": {"skills", "technical skills", "core skills", "competencies"},
    "projects": {"projects", "personal projects", "selected projects"},
    "certifications": {"certifications", "certificates", "licenses"},
}

COMMON_ATS_KEYWORDS = [
    "python", "sql", "communication", "teamwork", "leadership",
    "problem solving", "agile", "git", "data analysis", "testing",
    "documentation", "project management", "api", "cloud", "excel",
    "collaboration", "ownership", "ci/cd", "docker", "statistics",
]

DECORATIVE_CHARS = set("◆●■▪▸►★☆♥♦♣♠✓✔✦✧◉⬢⬣⬥⬧⬨⬩⬪⬫⬬⬭⬮⬯⯀⯁⯂⯃")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]"
)

MAX_WORDS = 1500
MAX_SECTION_WORDS = 400
MAX_LONG_LINE_CHARS = 200


@dataclass
class DeterministicResult:
    """Pure deterministic scoring output (all scores 0-100)."""

    formatting_score: int = 0
    keyword_score: int = 0
    experience_score: int = 0
    skills_score: int = 0
    projects_score: int = 0
    education_score: int = 0
    overall_score: int = 0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def _clamp(score: int) -> int:
    return max(0, min(100, int(round(score))))


def _split_sections(text: str) -> dict[str, list[str]]:
    """Group lines under standard section headings (heading -> content lines)."""
    alias_to_section = {alias: name for name, aliases in SECTION_ALIASES.items() for alias in aliases}
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.split("\n"):
        line = raw_line.strip().rstrip(":").strip()
        key = line.lower()
        if key in alias_to_section and len(line) <= 30:
            current = alias_to_section[key]
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(raw_line.strip())
    return sections


def _nonempty_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.strip()]


def analyze_ats_deterministic(
    resume_text: str,
    profile: ResumeProfile | None = None,
) -> DeterministicResult:
    """Run transparent rule-based ATS checks over resume text (+ optional profile)."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty. Upload a resume before ATS analysis.")
    text = resume_text.strip()
    lowered = text.lower()
    words = text.split()
    word_count = len(words)
    sections = _split_sections(text)

    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendations: list[str] = []

    # --- Contact information ---
    has_email = EMAIL_RE.search(text) is not None
    has_phone = PHONE_RE.search(text) is not None
    if has_email:
        strengths.append("Contact email present.")
    else:
        weaknesses.append("No email address detected.")
        recommendations.append("Add a contact email address near the top of the resume.")
    if has_phone:
        strengths.append("Contact phone number present.")
    else:
        weaknesses.append("No phone number detected.")
        recommendations.append("Add a contact phone number near the top of the resume.")

    # --- Standard sections ---
    core_sections = ["summary", "experience", "education", "skills"]
    found_core = [s for s in core_sections if s in sections]
    coverage = len(found_core) / len(core_sections)
    for missing in [s for s in core_sections if s not in sections]:
        weaknesses.append(f"Standard '{missing}' section not detected.")
        recommendations.append(f"Add a clearly labeled '{missing.capitalize()}' section.")
    if coverage == 1.0:
        strengths.append("All core resume sections detected (summary, experience, education, skills).")

    # --- Skills ---
    skill_items: list[str] = list(profile.skills) if profile and profile.skills else []
    if not skill_items and "skills" in sections:
        for line in _nonempty_lines(sections["skills"]):
            skill_items.extend([p.strip() for p in re.split(r"[,•\-|]", line) if p.strip()])
    skill_count = len({s.lower() for s in skill_items if s})
    if skill_count >= 8:
        skills_score = 100
        strengths.append(f"Strong skills coverage ({skill_count} distinct skills).")
    elif skill_count >= 5:
        skills_score = 80
        strengths.append(f"Good skills coverage ({skill_count} distinct skills).")
    elif skill_count >= 3:
        skills_score = 60
    elif skill_count >= 1:
        skills_score = 40
        weaknesses.append("Skills section looks thin.")
        recommendations.append("Expand the skills section with role-relevant tools and technologies.")
    else:
        skills_score = 10
        weaknesses.append("No skills detected.")
        recommendations.append("Add a skills section listing role-relevant tools and technologies.")

    # --- Education ---
    edu_entries = list(profile.education) if profile and profile.education else []
    edu_lines = _nonempty_lines(sections.get("education", []))
    if edu_entries or len(edu_lines) >= 2:
        education_score = 90
        strengths.append("Education section present with details.")
    elif "education" in sections:
        education_score = 55
        weaknesses.append("Education section looks sparse.")
        recommendations.append("Add institution, degree, and dates to the education section.")
    else:
        education_score = 15

    # --- Measurable achievements ---
    measure_hits = len(MEASUREMENT_RE.findall(text))
    number_hits = len(NUMBER_RE.findall(text))
    if measure_hits >= 4:
        measurable_score = 100
        strengths.append(f"Strong use of measurable achievements ({measure_hits} quantified results).")
    elif measure_hits == 3:
        measurable_score = 85
    elif measure_hits == 2:
        measurable_score = 70
    elif measure_hits == 1:
        measurable_score = 55
        recommendations.append("Quantify more achievements with numbers, percentages, or scale.")
    else:
        measurable_score = 30
        weaknesses.append("No quantified achievements detected (numbers, %, scale).")
        recommendations.append("Add measurable results (e.g. percentages, counts, scale) to experience bullets.")

    # --- Action verbs ---
    found_verbs = sorted({v for v in ACTION_VERBS if re.search(rf"\b{v}\b", lowered)})
    if len(found_verbs) >= 8:
        verbs_score = 100
        strengths.append("Strong action-verb usage across bullets.")
    elif len(found_verbs) >= 5:
        verbs_score = 85
    elif len(found_verbs) >= 3:
        verbs_score = 70
    elif len(found_verbs) >= 1:
        verbs_score = 50
    else:
        verbs_score = 25
        weaknesses.append("Few action verbs detected.")
        recommendations.append("Start bullets with strong action verbs (e.g. Led, Built, Improved).")

    # --- Experience / projects presence ---
    exp_entries = list(profile.experience) if profile and profile.experience else []
    proj_entries = list(profile.projects) if profile and profile.projects else []
    if len(exp_entries) >= 2 or ("experience" in sections and len(_nonempty_lines(sections["experience"])) >= 4):
        exp_base = 90
        strengths.append("Work experience section present with substance.")
    elif exp_entries or "experience" in sections:
        exp_base = 60
    else:
        exp_base = 15
        weaknesses.append("No work experience detected.")
        recommendations.append("Add a work experience section with role, company, and dates.")
    if len(proj_entries) >= 2 or ("projects" in sections and len(_nonempty_lines(sections["projects"])) >= 3):
        proj_base = 90
        strengths.append("Projects section present with substance.")
    elif proj_entries or "projects" in sections:
        proj_base = 60
    else:
        proj_base = 30
        recommendations.append("Consider adding a projects section to showcase applied skills.")
    experience_score = _clamp(0.5 * exp_base + 0.3 * measurable_score + 0.2 * verbs_score)
    projects_score = _clamp(0.7 * proj_base + 0.3 * measurable_score)

    # --- Formatting: excessive indicators, symbols, long/empty sections ---
    formatting = 100
    pipe_count = text.count("|")
    if pipe_count > 10:
        formatting -= 20
        weaknesses.append("Heavy use of '|' separators, which can confuse ATS parsers.")
        recommendations.append("Replace '|' separators with simple line breaks or commas.")
    decorative_hits = sum(1 for ch in text if ch in DECORATIVE_CHARS)
    if decorative_hits > 5:
        formatting -= 15
        weaknesses.append("Decorative symbols detected (e.g. ◆ ● ■ ★).")
        recommendations.append("Remove decorative symbols; use plain bullets ('-' or '•').")
    emoji_hits = len(EMOJI_RE.findall(text))
    if emoji_hits > 0:
        formatting -= min(30, 10 * emoji_hits)
        weaknesses.append("Emoji or pictograph symbols detected.")
        recommendations.append("Remove emojis; ATS parsers often drop or misread them.")
    long_lines = sum(1 for line in text.split("\n") if len(line) > MAX_LONG_LINE_CHARS)
    if long_lines:
        formatting -= min(20, 5 * long_lines)
        weaknesses.append(f"{long_lines} very long line(s) detected (> {MAX_LONG_LINE_CHARS} chars).")
        recommendations.append("Break long lines into short, scannable bullets (1-2 lines each).")
    if word_count > MAX_WORDS:
        formatting -= 15
        weaknesses.append(f"Resume is very long ({word_count} words).")
        recommendations.append("Trim the resume toward 1-2 pages; keep only relevant content.")
    empty_sections = [name for name, lines in sections.items() if not _nonempty_lines(lines)]
    for name in empty_sections:
        formatting -= 10
        weaknesses.append(f"Section '{name}' appears empty.")
        recommendations.append(f"Fill in or remove the empty '{name}' section.")
    formatting = _clamp(formatting)
    if formatting >= 85:
        strengths.append("Clean, ATS-friendly formatting.")
    long_sections = [
        name for name, lines in sections.items() if len(" ".join(lines).split()) > MAX_SECTION_WORDS
    ]
    for name in long_sections:
        weaknesses.append(f"Section '{name}' is excessively long.")
        recommendations.append(f"Condense the '{name}' section to the most relevant bullets.")

    # --- Keyword score + missing keywords ---
    keyword_score = _clamp(50 * coverage + 0.3 * verbs_score + (20 if "skills" in sections else 0))
    missing_keywords = [kw for kw in COMMON_ATS_KEYWORDS if kw not in lowered][:10]
    if missing_keywords:
        recommendations.append(
            "Consider adding these commonly scanned keywords if truthful: "
            + ", ".join(missing_keywords[:5]) + "."
        )
    if number_hits == 0 and measure_hits == 0:
        recommendations.append("Add at least one quantified result per role.")

    overall = _clamp(
        (formatting + keyword_score + experience_score + skills_score + projects_score + education_score) / 6
    )

    return DeterministicResult(
        formatting_score=formatting,
        keyword_score=keyword_score,
        experience_score=experience_score,
        skills_score=skills_score,
        projects_score=projects_score,
        education_score=education_score,
        overall_score=overall,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_keywords=missing_keywords,
        recommendations=recommendations,
    )


def _merge_lists(*lists: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for items in lists:
        for item in items:
            key = item.strip().lower()
            if item.strip() and key not in seen:
                seen.add(key)
                merged.append(item.strip())
            if len(merged) >= limit:
                return merged
    return merged


def combine_ats_analysis(
    deterministic: DeterministicResult,
    semantic: ATSSemanticSignals,
) -> ATSAnalysis:
    """Blend deterministic scores with LLM semantic scores (deterministic-led)."""
    formatting = _clamp(0.7 * deterministic.formatting_score + 0.3 * semantic.bullet_clarity_score)
    keyword = _clamp(0.7 * deterministic.keyword_score + 0.3 * semantic.relevance_score)
    experience = _clamp(0.6 * deterministic.experience_score + 0.4 * semantic.impact_score)
    skills = _clamp(0.7 * deterministic.skills_score + 0.3 * semantic.relevance_score)
    projects = _clamp(0.6 * deterministic.projects_score + 0.4 * semantic.specificity_score)
    education = _clamp(deterministic.education_score)
    overall = _clamp((formatting + keyword + experience + skills + projects + education) / 6)

    vague = [f"Vague statement to tighten: {s}" for s in semantic.vague_statements[:5]]
    observations = (
        [f"Observation on achievement quality: {n}" for n in semantic.achievement_quality_notes[:4]]
        + [f"Observation on technical descriptions: {n}" for n in semantic.technical_description_notes[:4]]
    )

    return ATSAnalysis(
        overall_score=overall,
        formatting_score=formatting,
        keyword_score=keyword,
        experience_score=experience,
        skills_score=skills,
        projects_score=projects,
        education_score=education,
        strengths=_merge_lists(deterministic.strengths, semantic.strengths, limit=12),
        weaknesses=_merge_lists(deterministic.weaknesses, vague, semantic.weaknesses, limit=12),
        missing_keywords=_merge_lists(
            deterministic.missing_keywords, semantic.missing_keywords, limit=15
        ),
        recommendations=_merge_lists(
            deterministic.recommendations, semantic.recommendations, observations, limit=12
        ),
    )


def analyze_ats(
    resume_text: str,
    profile: ResumeProfile | None = None,
    *,
    semantic_model=None,
    api_key: str | None = None,
    model_name: str = DEFAULT_OPENROUTER_MODEL,
) -> ATSAnalysis:
    """Run deterministic checks plus OpenRouter semantic evaluation, then combine.

    Raises:
        ValueError: Empty resume text, missing API key, or malformed LLM output.
        RuntimeError: Underlying OpenRouter/LangChain API failure.
    """
    from app.ai.ats_advisor import evaluate_resume_semantics

    deterministic = analyze_ats_deterministic(resume_text, profile)
    semantic = evaluate_resume_semantics(
        resume_text,
        profile,
        model=semantic_model,
        api_key=api_key,
        model_name=model_name,
    )
    return combine_ats_analysis(deterministic, semantic)
