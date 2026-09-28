"""Tests for Phase 7 bullet optimization (OpenRouter mocked)."""

import pytest

from app.ai.bullet_optimizer import BULLET_OPTIMIZATION_INSTRUCTIONS, optimize_bullet
from app.schemas.bullet import OptimizedBullet

VALID_BULLET_DICT = {
    "original": "Worked on the payments API.",
    "optimized": "Built and maintained the payments API serving internal services.",
    "improvements": ["Replaced weak verb 'Worked' with 'Built'."],
    "missing_metrics": ["request volume or latency figures"],
    "action_verbs": ["Built"],
    "xyz_breakdown": {
        "accomplishment": "Payments API development",
        "measurement": None,
        "method": "Python service development",
    },
}

SAMPLE_BULLET = "Worked on the payments API."


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


def test_valid_optimized_bullet_from_mocked_openrouter():
    fake = _FakeModel(result=dict(VALID_BULLET_DICT))
    result = optimize_bullet(SAMPLE_BULLET, model=fake)
    assert isinstance(result, OptimizedBullet)
    assert result.original == "Worked on the payments API."
    assert "Built" in result.optimized
    assert result.improvements == ["Replaced weak verb 'Worked' with 'Built'."]
    assert result.missing_metrics == ["request volume or latency figures"]
    assert result.action_verbs == ["Built"]
    assert result.xyz_breakdown.accomplishment == "Payments API development"
    assert result.xyz_breakdown.measurement is None
    assert fake.schemas == [OptimizedBullet]
    assert SAMPLE_BULLET in fake._structured.prompts[0]


def test_job_description_included_in_prompt():
    fake = _FakeModel(result=dict(VALID_BULLET_DICT))
    optimize_bullet(SAMPLE_BULLET, job_description="Python backend role", model=fake)
    assert "Python backend role" in fake._structured.prompts[0]


def test_no_job_description_omits_job_section():
    fake = _FakeModel(result=dict(VALID_BULLET_DICT))
    optimize_bullet(SAMPLE_BULLET, model=fake)
    assert "Job description:" not in fake._structured.prompts[0]


def test_profile_instance_passthrough():
    expected = OptimizedBullet(original="Did X.", optimized="Delivered X.")
    fake = _FakeModel(result=expected)
    assert optimize_bullet("Did X.", model=fake) is expected


def test_malformed_response_type_rejected():
    fake = _FakeModel(result="not a bullet at all")
    with pytest.raises(ValueError, match="unexpected response type"):
        optimize_bullet(SAMPLE_BULLET, model=fake)


def test_malformed_response_values_rejected():
    fake = _FakeModel(result={"original": "", "optimized": ""})
    with pytest.raises(ValueError, match="invalid optimized bullet"):
        optimize_bullet(SAMPLE_BULLET, model=fake)


def test_empty_bullet_rejected_without_calling_model():
    fake = _FakeModel(result=dict(VALID_BULLET_DICT))
    with pytest.raises(ValueError, match="empty"):
        optimize_bullet("   ", model=fake)
    assert fake._structured.prompts == []


def test_missing_api_key_rejected():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        optimize_bullet(SAMPLE_BULLET, api_key="")


def test_api_errors_wrapped_as_runtime_error():
    fake = _FakeModel(error=ConnectionError("boom"))
    with pytest.raises(RuntimeError, match="Bullet optimization failed"):
        optimize_bullet(SAMPLE_BULLET, model=fake)


def test_prompt_forbids_invention():
    assert "NEVER invent" in BULLET_OPTIMIZATION_INSTRUCTIONS
    assert "missing_metrics" in BULLET_OPTIMIZATION_INSTRUCTIONS
