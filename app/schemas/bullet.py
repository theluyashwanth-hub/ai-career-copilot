"""Bullet optimization schemas (Phase 7).

OptimizedBullet is produced by Gemini from a single resume bullet
plus an optional job description. The model must never invent
metrics, technologies, or achievements.
"""

from pydantic import BaseModel, Field


class XYZBreakdown(BaseModel):
    """X-Y-Z decomposition of the optimized bullet."""

    accomplishment: str | None = Field(default=None)
    measurement: str | None = Field(default=None)
    method: str | None = Field(default=None)


class OptimizedBullet(BaseModel):
    """Optimized resume bullet with explanation."""

    original: str = Field(min_length=1)
    optimized: str = Field(min_length=1)
    improvements: list[str] = Field(default_factory=list)
    missing_metrics: list[str] = Field(default_factory=list)
    action_verbs: list[str] = Field(default_factory=list)
    xyz_breakdown: XYZBreakdown = Field(default_factory=XYZBreakdown)
