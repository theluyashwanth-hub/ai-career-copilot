"""Tests for Phase 1 UI placeholders."""

import app.main as main_module
from app.ui.components import (
    APP_SUBTITLE,
    APP_TITLE,
    NAV_ITEMS,
    get_nav_items,
)


EXPECTED_NAV_ITEMS = [
    "Copilot",
    "Resume",
    "ATS Analysis",
    "Job Description",
    "Job Match",
    "Resume Optimizer",
    "SWOT Analysis",
    "Interview Coach",
    "Career Chat",
]


def test_app_title_and_subtitle():
    assert APP_TITLE == "AI Career Copilot"
    assert APP_SUBTITLE == "Your AI-powered career assistant."


def test_nav_items_match_expected_placeholders():
    assert NAV_ITEMS == EXPECTED_NAV_ITEMS
    assert get_nav_items() == EXPECTED_NAV_ITEMS


def test_get_nav_items_returns_copy():
    items = get_nav_items()
    items.append("Fake")
    assert NAV_ITEMS == EXPECTED_NAV_ITEMS


def test_main_entrypoint_exists():
    assert callable(main_module.main)
