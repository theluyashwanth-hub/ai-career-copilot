"""Environment-backed application configuration (Phase 1 foundation).

Only foundation settings live here. No LLM clients, RAG, agents,
or feature-specific logic are introduced in this phase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Career Copilot"
    app_subtitle: str = "Your AI-powered career assistant."

    # OpenRouter (OpenAI-compatible) settings. Replaces Gemini.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"
    openrouter_embedding_model: str = "openai/text-embedding-3-small"
    openrouter_reasoning_enabled: bool = True

    # Deprecated: kept for backwards compatibility with existing .env files.
    # If OPENROUTER_API_KEY is empty, this fallback key is used.
    gemini_api_key: str = ""

    # Phase 12: persistent storage (PostgreSQL via SQLAlchemy).
    # No authentication in this phase; all rows belong to a default user.
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/ai_career_copilot"
    )
    # Used by the test suite. SQLite by default so tests run without a
    # live PostgreSQL server; point it at a Postgres database to test
    # against the real backend (e.g. .../ai_career_copilot_test).
    test_database_url: str = "sqlite+pysqlite:///:memory:"


def get_settings() -> Settings:
    """Return a fresh Settings instance (reads current environment)."""
    return Settings()


# Shared default instance used by the Streamlit app.
settings = get_settings()
