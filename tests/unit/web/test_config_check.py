"""Tests for sherloc_pipeline.web.config_check."""

from __future__ import annotations

import os

import pytest

from sherloc_pipeline.web import config_check


_R2_ENV_KEYS = ("PHASE_TIER", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                 "AWS_ENDPOINT_URL", "AWS_REGION")


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every SHERLOC_* + R2-related var so each test starts clean."""
    for key in list(os.environ):
        if key.startswith("SHERLOC_"):
            monkeypatch.delenv(key, raising=False)
    for key in _R2_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_missing_db_is_reported(clean_env):
    errors = config_check.validate()
    assert any("SHERLOC_DB" in e for e in errors)


def test_memory_db_is_accepted(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    errors = config_check.validate()
    assert errors == []


def test_writable_existing_db_is_accepted(clean_env, tmp_path):
    db = tmp_path / "phase.db"
    db.touch()
    clean_env.setenv("SHERLOC_DB", str(db))
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    errors = config_check.validate()
    assert errors == []


def test_missing_db_parent_is_reported(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", str(tmp_path / "no_such_dir" / "phase.db"))
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    errors = config_check.validate()
    assert any("parent directory does not exist" in e for e in errors)


def test_unknown_auth_mode_is_reported(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "notarealmode")
    errors = config_check.validate()
    assert any("SHERLOC_AUTH_MODE" in e and "notarealmode" in e for e in errors)


def test_auth0_requires_domain_and_audience(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    errors = config_check.validate()
    assert any("SHERLOC_AUTH0_DOMAIN" in e for e in errors)
    assert any("SHERLOC_AUTH0_AUDIENCE" in e for e in errors)
    assert any("SHERLOC_AUTH0_SPA_CLIENT_ID" in e for e in errors)


def test_auth0_complete_config_is_accepted(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "sherloc.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://example/api")
    clean_env.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "spaclient_abc")
    clean_env.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", "https://example/claims/identity")
    # auth0 mode also requires R2 wiring per platform spec §3.9.2 (v4.1.7+).
    clean_env.setenv("PHASE_TIER", "team")
    clean_env.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    clean_env.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    clean_env.setenv("AWS_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    errors = config_check.validate()
    assert errors == []


def test_auth0_requires_r2_wiring(clean_env, tmp_path):
    """Auth0 (production) mode must reject missing PHASE_TIER + AWS_* per §3.9.2."""
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "sherloc.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://example/api")
    clean_env.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "spaclient_abc")
    errors = config_check.validate()
    assert any("PHASE_TIER" in e for e in errors)
    assert any("AWS_ACCESS_KEY_ID" in e for e in errors)
    assert any("AWS_SECRET_ACCESS_KEY" in e for e in errors)
    assert any("AWS_ENDPOINT_URL" in e for e in errors)


def test_auth0_rejects_invalid_phase_tier(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "sherloc.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://example/api")
    clean_env.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "spaclient_abc")
    clean_env.setenv("PHASE_TIER", "bogus")
    clean_env.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    clean_env.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    clean_env.setenv("AWS_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com")
    errors = config_check.validate()
    assert any("PHASE_TIER" in e and "bogus" in e for e in errors)


def test_dev_mode_does_not_require_r2_wiring(clean_env, tmp_path):
    """Dev mode is for unit tests + local dev; R2 vars are not required."""
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    errors = config_check.validate()
    assert errors == []


def test_cf_access_requires_team_domain_and_audience(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "cf-access")
    errors = config_check.validate()
    assert any("SHERLOC_CF_TEAM_DOMAIN" in e for e in errors)
    assert any("SHERLOC_CF_AUDIENCE" in e for e in errors)


def test_unknown_access_mode_is_reported(clean_env, tmp_path):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_ACCESS_MODE", "rogue")
    errors = config_check.validate()
    assert any("SHERLOC_ACCESS_MODE" in e and "rogue" in e for e in errors)


def test_main_returns_nonzero_on_errors(clean_env, capsys):
    rc = config_check.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "missing required variable: SHERLOC_DB" in captured.err


def test_main_returns_zero_when_ok(clean_env, tmp_path, capsys):
    clean_env.setenv("SHERLOC_DB", ":memory:")
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    rc = config_check.main()
    assert rc == 0
