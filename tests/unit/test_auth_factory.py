"""DevValidator + ``get_validator()`` selection tests (§13.1, §19.1 B.8).

Covers:
- DevValidator returns synthetic claims regardless of token shape (§13.5).
- ``get_validator()`` picks the right validator class for each
  ``SHERLOC_AUTH_MODE`` value, raising RuntimeError on misconfig.
- Validator is process-cached; ``_reset_validator_for_tests`` drops it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sherloc_pipeline.web.auth import (
    Auth0Validator,
    CFAccessValidator,
    DevValidator,
    TokenClaims,
    _reset_jwks_cache_for_tests,
    _reset_validator_for_tests,
    get_validator,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()
    yield
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every SHERLOC_* var to start from a known baseline."""
    import os as _os

    for key in list(_os.environ):
        if key.startswith("SHERLOC_"):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------
# DevValidator
# ---------------------------------------------------------------------------


def test_dev_validator_returns_phase_team_member_synthetic_claims():
    claims = DevValidator().validate("any-token-string")
    assert isinstance(claims, TokenClaims)
    assert claims.sub == "localhost-dev"
    assert claims.email == "dev@local"
    # §2.6.1 namespace: dev mode synthesises ``phase:team-member`` so a
    # local container passes the team-mode route gate without a real
    # Auth0 token. The legacy ``sherloc:internal`` was retired at v4.1.
    assert claims.roles == ["phase:team-member"]
    # Synthetic 24h expiry (allow some clock slop)
    assert claims.expires_at > datetime.now(timezone.utc)


def test_dev_validator_ignores_token_argument():
    claims_a = DevValidator().validate("foo")
    claims_b = DevValidator().validate("")
    assert claims_a.sub == claims_b.sub == "localhost-dev"
    assert claims_a.roles == claims_b.roles


# ---------------------------------------------------------------------------
# get_validator() selection by SHERLOC_AUTH_MODE
# ---------------------------------------------------------------------------


def test_get_validator_dev_mode(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    validator = get_validator()
    assert isinstance(validator, DevValidator)


def test_get_validator_cf_access_mode(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "cf-access")
    clean_env.setenv("SHERLOC_CF_TEAM_DOMAIN", "team.cloudflareaccess.com")
    clean_env.setenv("SHERLOC_CF_AUDIENCE", "cf-aud-tag")
    validator = get_validator()
    assert isinstance(validator, CFAccessValidator)
    assert validator.team_domain == "team.cloudflareaccess.com"
    assert validator.audience == "cf-aud-tag"


def test_get_validator_cf_access_missing_team_domain_raises(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "cf-access")
    clean_env.setenv("SHERLOC_CF_AUDIENCE", "cf-aud-tag")
    with pytest.raises(RuntimeError, match="SHERLOC_CF_TEAM_DOMAIN"):
        get_validator()


PHASE_IDENTITY_CLAIM_URI = "https://m2020-phase.net/claims/identity"


def _set_min_auth0_env(env) -> None:
    """Minimal env to construct an auth0 validator successfully.

    §2.6.1 requires ``SHERLOC_AUTH0_IDENTITY_CLAIM_URI`` (mandatory at
    v4.1 B.0 cutover); helper centralizes the baseline so individual
    tests focus on the variable they exercise.
    """
    env.setenv("SHERLOC_AUTH_MODE", "auth0")
    env.setenv("SHERLOC_AUTH0_DOMAIN", "tenant.us.auth0.com")
    env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://api/")
    env.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", PHASE_IDENTITY_CLAIM_URI)


def test_get_validator_auth0_mode(clean_env):
    _set_min_auth0_env(clean_env)
    validator = get_validator()
    assert isinstance(validator, Auth0Validator)
    assert validator.domain == "tenant.us.auth0.com"
    assert validator.audience == "https://api/"
    assert validator.identity_claim_uri == PHASE_IDENTITY_CLAIM_URI


def test_get_validator_auth0_known_spa_client_ids_parsed(clean_env):
    _set_min_auth0_env(clean_env)
    clean_env.setenv(
        "SHERLOC_AUTH0_KNOWN_SPA_CLIENT_IDS",
        "spa_one, spa_two , ,spa_three",
    )
    validator = get_validator()
    assert isinstance(validator, Auth0Validator)
    assert validator.known_spa_client_ids == ["spa_one", "spa_two", "spa_three"]


def test_get_validator_auth0_identity_claim_uri_env_routed(clean_env):
    """``SHERLOC_AUTH0_IDENTITY_CLAIM_URI`` plumbs through to the
    package's Auth0Validator (§2.6.1 contract)."""
    _set_min_auth0_env(clean_env)
    validator = get_validator()
    assert isinstance(validator, Auth0Validator)
    assert validator.identity_claim_uri == PHASE_IDENTITY_CLAIM_URI


def test_get_validator_auth0_missing_identity_claim_uri_raises(clean_env):
    """§2.6.1 makes the identity claim URI mandatory; v4.1 dropped the
    Phase A backward-compat path. A SHERLOC instance booted without
    ``SHERLOC_AUTH0_IDENTITY_CLAIM_URI`` must refuse to start (better
    than a silent role-name mismatch at runtime)."""
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "tenant.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://api/")
    with pytest.raises(RuntimeError, match="SHERLOC_AUTH0_IDENTITY_CLAIM_URI"):
        get_validator()


def test_get_validator_auth0_missing_domain_raises(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://api/")
    clean_env.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", PHASE_IDENTITY_CLAIM_URI)
    with pytest.raises(RuntimeError, match="SHERLOC_AUTH0_DOMAIN"):
        get_validator()


def test_get_validator_auth0_missing_audience_raises(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "tenant.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", PHASE_IDENTITY_CLAIM_URI)
    with pytest.raises(RuntimeError, match="SHERLOC_AUTH0_AUDIENCE"):
        get_validator()


def test_get_validator_unknown_mode_raises(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "nope")
    with pytest.raises(RuntimeError, match="Unrecognized SHERLOC_AUTH_MODE"):
        get_validator()


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------


def test_get_validator_returns_same_instance_within_process(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    a = get_validator()
    b = get_validator()
    assert a is b


def test_reset_validator_drops_singleton(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    a = get_validator()
    _reset_validator_for_tests()
    b = get_validator()
    assert a is not b


def test_singleton_does_not_re_read_env_after_first_build(clean_env):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    a = get_validator()
    assert isinstance(a, DevValidator)
    # Mutate env post-first-build; singleton should not change shape.
    clean_env.setenv("SHERLOC_AUTH_MODE", "cf-access")
    clean_env.setenv("SHERLOC_CF_TEAM_DOMAIN", "team.cloudflareaccess.com")
    clean_env.setenv("SHERLOC_CF_AUDIENCE", "tag")
    b = get_validator()
    assert b is a
    assert isinstance(b, DevValidator)
    # Operator must call _reset_validator_for_tests() to refresh.
    _reset_validator_for_tests()
    c = get_validator()
    assert isinstance(c, CFAccessValidator)
