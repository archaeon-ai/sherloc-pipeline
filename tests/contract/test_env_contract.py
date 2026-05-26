"""Contract test: config_check.py enforces the documented env-var surface.

Pins the always-required + mode-driven + access-mode + retired surface
documented in DEPLOYMENT_CONTRACT.md §5.

Failures here mean the contract surface drifted from the validator. To
move the contract surface, edit config_check.py AND this test together.
"""

from __future__ import annotations

import inspect
import os
import re

import pytest

from sherloc_pipeline.web import config_check


_R2_ENV_KEYS = (
    "PHASE_TIER",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "AWS_REGION",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every SHERLOC_*/PHASE_*/AWS_* var so each test starts clean."""
    for key in list(os.environ):
        if key.startswith("SHERLOC_") or key.startswith("PHASE_"):
            monkeypatch.delenv(key, raising=False)
    for key in _R2_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_valid_auth_modes_pin():
    assert config_check.VALID_AUTH_MODES == {"auth0", "cf-access", "dev"}


def test_valid_access_modes_pin():
    assert config_check.VALID_ACCESS_MODES == {"internal", "public"}


def test_valid_phase_tiers_pin():
    assert config_check.VALID_PHASE_TIERS == {"team", "public"}


def test_empty_env_reports_required_set(clean_env):
    """With nothing set, the validator must surface every always-required
    error: SHERLOC_DB plus the cf-access mode defaults (the validator
    defaults SHERLOC_AUTH_MODE to cf-access when unset)."""
    errors = config_check.validate()
    joined = "\n".join(errors)
    assert "missing required variable: SHERLOC_DB" in joined
    assert "missing required variable: SHERLOC_CF_TEAM_DOMAIN" in joined
    assert "missing required variable: SHERLOC_CF_AUDIENCE" in joined


def test_default_auth_mode_is_cf_access(clean_env, tmp_path):
    """SHERLOC_AUTH_MODE default = cf-access. Setting only the cf-access
    requireds + SHERLOC_DB should validate clean."""
    clean_env.setenv("SHERLOC_DB", str(tmp_path / "phase.db"))
    clean_env.setenv("SHERLOC_CF_TEAM_DOMAIN", "team.example.com")
    clean_env.setenv("SHERLOC_CF_AUDIENCE", "aud-id")
    assert config_check.validate() == []


@pytest.mark.parametrize(
    "missing_var",
    [
        "SHERLOC_AUTH0_DOMAIN",
        "SHERLOC_AUTH0_AUDIENCE",
        "SHERLOC_AUTH0_SPA_CLIENT_ID",
        "PHASE_TIER",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
    ],
)
def test_auth0_mode_required_vars(clean_env, missing_var):
    """Each var required under auth0 mode must produce a 'missing required
    variable: <NAME>' error when omitted."""
    auth0_set = {
        "SHERLOC_AUTH_MODE": "auth0",
        "SHERLOC_DB": ":memory:",
        "SHERLOC_AUTH0_DOMAIN": "tenant.auth0.com",
        "SHERLOC_AUTH0_AUDIENCE": "https://api.example.com",
        "SHERLOC_AUTH0_SPA_CLIENT_ID": "spa-client-id",
        "PHASE_TIER": "team",
        "AWS_ACCESS_KEY_ID": "AKIA-test",
        "AWS_SECRET_ACCESS_KEY": "secret-test",
        "AWS_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
    }
    auth0_set.pop(missing_var)
    for k, v in auth0_set.items():
        clean_env.setenv(k, v)
    errors = config_check.validate()
    assert any(f"missing required variable: {missing_var}" in e for e in errors), errors


@pytest.mark.parametrize(
    "missing_var", ["SHERLOC_CF_TEAM_DOMAIN", "SHERLOC_CF_AUDIENCE"]
)
def test_cf_access_mode_required_vars(clean_env, missing_var):
    cf_set = {
        "SHERLOC_AUTH_MODE": "cf-access",
        "SHERLOC_DB": ":memory:",
        "SHERLOC_CF_TEAM_DOMAIN": "team.example.com",
        "SHERLOC_CF_AUDIENCE": "aud-id",
    }
    cf_set.pop(missing_var)
    for k, v in cf_set.items():
        clean_env.setenv(k, v)
    errors = config_check.validate()
    assert any(f"missing required variable: {missing_var}" in e for e in errors), errors


def test_dev_mode_has_no_further_requirements(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_DB", ":memory:")
    assert config_check.validate() == []


def test_retired_role_claim_uri_var_not_in_source():
    """The legacy split-claim env var was retired in v4.1.0 and MUST NOT
    reappear in config_check.py. Building the literal at runtime keeps
    this test free of the retired name itself."""
    src = inspect.getsource(config_check)
    legacy_literal = "SHERLOC_AUTH0_" + "ROLE_CLAIM_URI"
    assert legacy_literal not in src, (
        f"Retired env var {legacy_literal!r} reappeared in config_check.py — "
        "see DEPLOYMENT_CONTRACT.md §5.5"
    )


# --- Q2 (v4.1.14): SHERLOC_AUTH0_IDENTITY_CLAIM_URI added to validator ---


def test_auth0_mode_requires_identity_claim_uri(clean_env):
    """Pre-v4.1.14, this URI was only checked at first request by
    web/auth.py — a misconfigured container booted clean and failed on
    first auth call. v4.1.14 moves the check to config_check so boot
    fails immediately."""
    for k, v in {
        "SHERLOC_AUTH_MODE": "auth0",
        "SHERLOC_DB": ":memory:",
        "SHERLOC_AUTH0_DOMAIN": "tenant.auth0.com",
        "SHERLOC_AUTH0_AUDIENCE": "https://api.example.com",
        "SHERLOC_AUTH0_SPA_CLIENT_ID": "spa-client-id",
        "PHASE_TIER": "team",
        "AWS_ACCESS_KEY_ID": "AKIA-test",
        "AWS_SECRET_ACCESS_KEY": "secret-test",
        "AWS_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
    }.items():
        clean_env.setenv(k, v)
    # IDENTITY_CLAIM_URI deliberately unset.
    errors = config_check.validate()
    assert any(
        "missing required variable: SHERLOC_AUTH0_IDENTITY_CLAIM_URI" in e
        for e in errors
    ), errors


def test_auth0_mode_with_identity_claim_uri_validates(clean_env):
    for k, v in {
        "SHERLOC_AUTH_MODE": "auth0",
        "SHERLOC_DB": ":memory:",
        "SHERLOC_AUTH0_DOMAIN": "tenant.auth0.com",
        "SHERLOC_AUTH0_AUDIENCE": "https://api.example.com",
        "SHERLOC_AUTH0_SPA_CLIENT_ID": "spa-client-id",
        "SHERLOC_AUTH0_IDENTITY_CLAIM_URI": "https://example.com/claims/identity",
        "PHASE_TIER": "team",
        "AWS_ACCESS_KEY_ID": "AKIA-test",
        "AWS_SECRET_ACCESS_KEY": "secret-test",
        "AWS_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com",
    }.items():
        clean_env.setenv(k, v)
    assert config_check.validate() == []


# --- Q1 (v4.1.14): PHASE_DATABASE_PATH mismatch fail-fast ---


def test_phase_database_path_mismatch_is_rejected(clean_env, tmp_path):
    """If both vars are set to different files, the validator fails. The
    dangerous shape is silent divergence between alembic (reads
    PHASE_DATABASE_PATH) and the app (reads SHERLOC_DB)."""
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_DB", str(tmp_path / "phase.db"))
    clean_env.setenv("PHASE_DATABASE_PATH", str(tmp_path / "different.db"))
    errors = config_check.validate()
    joined = " | ".join(errors)
    assert "PHASE_DATABASE_PATH" in joined
    assert "SHERLOC_DB" in joined
    assert "differ" in joined.lower(), joined


def test_phase_database_path_matching_is_accepted(clean_env, tmp_path):
    db = tmp_path / "phase.db"
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_DB", str(db))
    clean_env.setenv("PHASE_DATABASE_PATH", str(db))
    assert config_check.validate() == []


def test_phase_database_path_unset_is_accepted(clean_env, tmp_path):
    """Post-Q1 (v4.1.14): SHERLOC_DB alone is sufficient; the entrypoint
    exports PHASE_DATABASE_PATH from it. The validator must not require
    PHASE_DATABASE_PATH to be set independently."""
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_DB", str(tmp_path / "phase.db"))
    assert config_check.validate() == []
