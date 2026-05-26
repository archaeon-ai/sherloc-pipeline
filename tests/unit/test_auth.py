"""Tests for CF Access JWT validation.

Covers the 11 cases listed in
``docs/specs/PUBLIC_RELEASE_PREP_SPEC-revised2.md`` §4.9.2:

    1.  Valid JWT (signature + issuer + audience + non-expired)
    2.  Invalid signature
    3.  Wrong issuer
    4.  Wrong audience
    5.  Expired JWT
    6.  Missing ``Cf-Access-Jwt-Assertion`` header → 401
    7.  ``SHERLOC_AUTH_MODE=dev`` bypass
    8.  JWKS cached on second fetch within TTL
    9.  JWKS fetch fails, valid cached keys, in-grace window → reuse
    10. JWKS fetch fails, no cached keys → 503
    11. JWKS endpoint returns malformed body → fetch-failure paths
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from sherloc_pipeline.web import auth as auth_mod
from sherloc_pipeline.web.auth import (
    AuthError,
    JWKSUnavailableError,
    JWKS_GRACE_SECONDS,
    JWKS_TTL_SECONDS,
    _cf_jwks_url,
    _reset_jwks_cache_for_tests,
    _reset_validator_for_tests,
    fetch_cf_jwks,
    validate_cf_jwt,
)


TEAM_DOMAIN = "test-team.cloudflareaccess.com"
AUDIENCE = "test-aud-tag-abc123"
ISSUER = f"https://{TEAM_DOMAIN}"

# Only intercept Cloudflare Access JWKS requests; let any stray httpx
# calls from background threads in earlier tests pass through unmocked.
# Without this, unrelated PDS/web-route retries that fire after the
# httpx_mock fixture activates trip teardown asserts.
pytestmark = pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host == TEAM_DOMAIN,
)


# ---------------------------------------------------------------------------
# Key + JWKS helpers
# ---------------------------------------------------------------------------

def _gen_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _key_to_pem(private_key) -> bytes:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_jwk(private_key, kid: str) -> dict:
    """Build a JWK dict from a private RSA key (the public half)."""
    public_numbers = private_key.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        import base64
        length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()

    return {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64url_uint(public_numbers.n),
        "e": _b64url_uint(public_numbers.e),
    }


def _mint_token(
    private_key,
    *,
    kid: str,
    audience: str = AUDIENCE,
    issuer: str = ISSUER,
    email: str = "alice@example.com",
    expires_in: int = 3600,
) -> str:
    """Mint an RS256-signed JWT against the given key."""
    now = datetime.now(timezone.utc)
    payload = {
        "iss": issuer,
        "aud": audience,
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        "iat": int(now.timestamp()),
        "email": email,
        "sub": email,
    }
    return pyjwt.encode(
        payload,
        _key_to_pem(private_key),
        algorithm="RS256",
        headers={"kid": kid},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_jwks_cache():
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()
    yield
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()


@pytest.fixture()
def signing_key():
    return _gen_rsa_key()


@pytest.fixture()
def jwks(signing_key):
    return {"keys": [_public_jwk(signing_key, kid="test-kid-1")]}


@pytest.fixture()
def mock_jwks_ok(httpx_mock, jwks):
    """JWKS endpoint returns 200 + the test JWKS, once per call."""
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        json=jwks,
    )
    return httpx_mock


# ---------------------------------------------------------------------------
# 1. Valid JWT
# ---------------------------------------------------------------------------

def test_valid_jwt_returns_claims(signing_key, mock_jwks_ok):
    token = _mint_token(signing_key, kid="test-kid-1")
    claims = validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)
    assert claims["email"] == "alice@example.com"
    assert claims["aud"] == AUDIENCE
    assert claims["iss"] == ISSUER


# ---------------------------------------------------------------------------
# 2. Invalid signature
# ---------------------------------------------------------------------------

def test_invalid_signature_raises_autherror(signing_key, mock_jwks_ok):
    # Sign with a different key but advertise the trusted kid
    other_key = _gen_rsa_key()
    token = _mint_token(other_key, kid="test-kid-1")
    with pytest.raises(AuthError):
        validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)


# ---------------------------------------------------------------------------
# 3. Wrong issuer (different CF Access team)
# ---------------------------------------------------------------------------

def test_wrong_issuer_raises_autherror(signing_key, mock_jwks_ok):
    token = _mint_token(
        signing_key,
        kid="test-kid-1",
        issuer="https://attacker-team.cloudflareaccess.com",
    )
    with pytest.raises(AuthError):
        validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)


# ---------------------------------------------------------------------------
# 4. Wrong audience
# ---------------------------------------------------------------------------

def test_wrong_audience_raises_autherror(signing_key, mock_jwks_ok):
    token = _mint_token(signing_key, kid="test-kid-1", audience="some-other-app")
    with pytest.raises(AuthError):
        validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)


# ---------------------------------------------------------------------------
# 5. Expired JWT
# ---------------------------------------------------------------------------

def test_expired_jwt_raises_autherror(signing_key, mock_jwks_ok):
    token = _mint_token(signing_key, kid="test-kid-1", expires_in=-60)
    with pytest.raises(AuthError):
        validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)


# ---------------------------------------------------------------------------
# 6. Missing token → AuthError (handler maps to 401)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token", [None, ""])
def test_missing_jwt_raises_autherror(token):
    with pytest.raises(AuthError):
        validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)


# ---------------------------------------------------------------------------
# 7. Dev-mode escape hatch (route-level)
# ---------------------------------------------------------------------------

def test_dev_mode_bypasses_validation(monkeypatch, tmp_path):
    """SHERLOC_AUTH_MODE=dev short-circuits validation in routes/user.py."""
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")

    from fastapi.testclient import TestClient
    from sherloc_pipeline.database.connection import create_all_tables, get_engine
    from sherloc_pipeline.web.app import create_app

    db_path = tmp_path / "auth_dev.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)

    app = create_app(engine=engine)
    with TestClient(app) as client:
        # No CF JWT header — anonymous would normally return [].
        # Under dev mode this resolves to dev@local instead.
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 42},
        )
        assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# 8. JWKS cached on second fetch within TTL
# ---------------------------------------------------------------------------

def test_jwks_cached_within_ttl(httpx_mock, jwks):
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        json=jwks,
    )
    first = fetch_cf_jwks(TEAM_DOMAIN)
    second = fetch_cf_jwks(TEAM_DOMAIN)
    assert first is second  # same cached dict object
    # Only one request should have been issued
    requests = httpx_mock.get_requests()
    assert len(requests) == 1


# ---------------------------------------------------------------------------
# 9. Fetch fails, in-grace cache reused
# ---------------------------------------------------------------------------

def test_fetch_failure_within_grace_uses_cache(httpx_mock, jwks, signing_key, monkeypatch):
    # Prime the cache with a successful fetch
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        json=jwks,
    )
    fetch_cf_jwks(TEAM_DOMAIN)

    # Push the cache past TTL but still within grace window. The cache
    # is keyed by full JWKS URL after the §13.1 refactor so the Auth0
    # and CF Access entries cannot collide.
    cached = auth_mod._jwks_cache[_cf_jwks_url(TEAM_DOMAIN)]
    cached["fetched_at"] = time.monotonic() - (JWKS_TTL_SECONDS + 60)

    # Next fetch fails (5xx)
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        status_code=503,
    )
    keys = fetch_cf_jwks(TEAM_DOMAIN)
    assert keys == jwks  # served from cache

    # And validation still works against the cached keys
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        status_code=503,
    )
    cached["fetched_at"] = time.monotonic() - (JWKS_TTL_SECONDS + 60)
    token = _mint_token(signing_key, kid="test-kid-1")
    claims = validate_cf_jwt(token, audience=AUDIENCE, team_domain=TEAM_DOMAIN)
    assert claims["email"] == "alice@example.com"


# ---------------------------------------------------------------------------
# 10. Fetch fails with no cache → JWKSUnavailableError (handler maps to 503)
# ---------------------------------------------------------------------------

def test_fetch_failure_no_cache_raises_unavailable(httpx_mock):
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        status_code=503,
    )
    with pytest.raises(JWKSUnavailableError):
        fetch_cf_jwks(TEAM_DOMAIN)


def test_fetch_failure_no_cache_returns_503_at_route(monkeypatch, tmp_path, httpx_mock, signing_key):
    """End-to-end: a request with a JWT but no JWKS available returns 503."""
    monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", TEAM_DOMAIN)
    monkeypatch.setenv("SHERLOC_CF_AUDIENCE", AUDIENCE)
    monkeypatch.delenv("SHERLOC_AUTH_MODE", raising=False)

    from fastapi.testclient import TestClient
    from sherloc_pipeline.database.connection import create_all_tables, get_engine
    from sherloc_pipeline.web.app import create_app

    db_path = tmp_path / "auth_503.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    app = create_app(engine=engine)

    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        status_code=502,
    )
    token = _mint_token(signing_key, kid="test-kid-1")

    with TestClient(app) as client:
        resp = client.get(
            "/api/user/preferences",
            headers={"Cf-Access-Jwt-Assertion": token},
        )
        assert resp.status_code == 503, resp.text


# ---------------------------------------------------------------------------
# 11. Malformed JWKS body treated as fetch failure
# ---------------------------------------------------------------------------

def test_malformed_jwks_body_treated_as_failure(httpx_mock):
    # 200 OK but body has no 'keys' list
    httpx_mock.add_response(
        url=f"https://{TEAM_DOMAIN}/cdn-cgi/access/certs",
        json={"unexpected": "shape"},
    )
    with pytest.raises(JWKSUnavailableError):
        fetch_cf_jwks(TEAM_DOMAIN)
