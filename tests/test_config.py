"""
Tests for src/config.py -- the Settings loader.

What these tests prove:
    - Settings loads successfully from the real .env (proves required
      variables are actually present -- this would fail loudly, "fail
      fast," if e.g. ANTHROPIC_API_KEY were missing).
    - get_settings() returns the same cached instance on repeated calls.
    - postgres_dsn assembles correctly, without asserting on the actual
      secret values (never print/compare real credentials in a test).
"""

from src.config import Settings, get_settings


def test_settings_loads_from_env():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.anthropic_api_key != ""
    assert settings.sec_user_agent != ""
    assert settings.anthropic_model != ""


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_postgres_dsn_has_expected_shape():
    settings = get_settings()
    dsn = settings.postgres_dsn
    assert dsn.startswith("postgresql://")
    assert f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}" in dsn
