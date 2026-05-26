"""Smoke test: SHERLOC consumes the §2.6.1 contract from ``phase_platform_auth``.

The conformance suite for the §2.6.1 Auth0Validator lives in the
``phase-platform-auth`` package itself (archaeon-ai/phase-platform-auth
``tests/``) — not duplicated here. This file just proves SHERLOC's
import surface still routes ``Auth0Validator``, ``AuthError``,
``JWKSUnavailableError``, ``TokenClaims``, and ``build_www_authenticate``
back to the package, and that ``_build_validator`` in auth0 mode hands
back a package-class instance.

If this test breaks, the v4.1 B.0 switchover regressed: SHERLOC has
either grown a local re-implementation again (forbidden) or imports
have drifted from the package surface.
"""

from __future__ import annotations

import phase_platform_auth as ppa
import pytest

import sherloc_pipeline.web.auth as auth


@pytest.fixture(autouse=True)
def _reset():
    auth._reset_jwks_cache_for_tests()
    auth._reset_validator_for_tests()
    yield
    auth._reset_jwks_cache_for_tests()
    auth._reset_validator_for_tests()


@pytest.fixture
def clean_env(monkeypatch):
    import os as _os

    for key in list(_os.environ):
        if key.startswith("SHERLOC_"):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


def test_auth0_validator_is_re_exported_from_package():
    """``sherloc_pipeline.web.auth.Auth0Validator`` IS the package class."""
    assert auth.Auth0Validator is ppa.Auth0Validator


def test_token_claims_is_re_exported_from_package():
    assert auth.TokenClaims is ppa.TokenClaims


def test_auth_error_types_are_re_exported_from_package():
    assert auth.AuthError is ppa.AuthError
    assert auth.JWKSUnavailableError is ppa.JWKSUnavailableError


def test_build_www_authenticate_is_re_exported_from_package():
    assert auth.build_www_authenticate is ppa.build_www_authenticate


def test_build_validator_auth0_returns_package_validator(clean_env):
    """Auth0 mode must build a ``phase_platform_auth.Auth0Validator``;
    SHERLOC no longer carries its own §2.6.1 implementation."""
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_AUTH0_DOMAIN", "tenant.us.auth0.com")
    clean_env.setenv("SHERLOC_AUTH0_AUDIENCE", "https://api.m2020-phase.net")
    clean_env.setenv(
        "SHERLOC_AUTH0_IDENTITY_CLAIM_URI",
        "https://m2020-phase.net/claims/identity",
    )
    validator = auth.get_validator()
    assert isinstance(validator, ppa.Auth0Validator)
    assert validator.domain == "tenant.us.auth0.com"
    assert (
        validator.identity_claim_uri == "https://m2020-phase.net/claims/identity"
    )
