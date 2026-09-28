"""Tests for foundation configuration."""

from app.core.config import Settings, get_settings


def test_default_app_name_and_subtitle():
    settings = Settings(_env_file=None)
    assert settings.app_name == "AI Career Copilot"
    assert settings.app_subtitle == "Your AI-powered career assistant."


def test_gemini_api_key_defaults_to_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key == ""


def test_gemini_api_key_loaded_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    settings = Settings(_env_file=None)
    assert settings.openrouter_api_key == "test-key-123"


def test_get_settings_returns_settings_instance(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "AI Career Copilot"
