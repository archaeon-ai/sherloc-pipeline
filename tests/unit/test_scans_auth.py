"""G18.11.* sub-gate coverage for /api/scans and other data routes.

The B.7+B.8 work landed auth on /api/user/preferences/* but did NOT extend
it to data routes. B.12 F1 closes that gap by applying
require_authenticated_request as a router-level dependency. These tests
exercise the new dep against /api/scans across all three auth modes
(dev / cf-access / auth0) and verify the G18.11.* status-code matrix
called out in spec §18.2.

The dev mode here is the operator-trust path used for local CLI testing
(§13.5). cf-access mode is the legacy production path. auth0
mode is the v4.0.0 VPS production path.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from sherloc_pipeline.database.connection import create_all_tables, get_engine
from sherloc_pipeline.web.app import create_app
from sherloc_pipeline.web.auth import (
    _reset_jwks_cache_for_tests,
    _reset_validator_for_tests,
)


AUTH0_DOMAIN = "sherloc-test.us.auth0.com"
AUTH0_AUDIENCE_INTERNAL = "https://sherloc-internal/api"
AUTH0_AUDIENCE_PUBLIC = "https://sherloc-public/api"
PHASE_IDENTITY_CLAIM_URI = "https://m2020-phase.net/claims/identity"
JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
EXPECTED_ISSUER = f"https://{AUTH0_DOMAIN}/"

pytestmark = pytest.mark.httpx_mock(
    should_mock=lambda request: request.url.host == AUTH0_DOMAIN,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()
    yield
    _reset_jwks_cache_for_tests()
    _reset_validator_for_tests()


@pytest.fixture
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _key_to_pem(key) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _public_jwk(key, kid: str) -> dict:
    pub = key.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()

    return {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64url_uint(pub.n),
        "e": _b64url_uint(pub.e),
    }


def _mint(
    key,
    *,
    audience: Any = None,
    identity: list[str] | None = None,
    expires_in: int = 3600,
    issuer: str = EXPECTED_ISSUER,
) -> str:
    """Mint a §2.6.1-shaped access token with the platform identity claim."""
    now = datetime.now(timezone.utc)
    # §2.6.1 requires array-shaped ``aud`` — default to a single-element
    # list mirroring Auth0's actual access-token shape.
    if audience is None:
        audience = [AUTH0_AUDIENCE_INTERNAL]
    elif isinstance(audience, str):
        audience = [audience]
    payload: dict[str, Any] = {
        "iss": issuer,
        "aud": audience,
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        "iat": int(now.timestamp()),
        "sub": "auth0|abc",
    }
    if identity is not None:
        payload[PHASE_IDENTITY_CLAIM_URI] = identity
    # ``typ=at+jwt`` (RFC 9068) matches what Auth0 emits for access
    # tokens and the §2.6.1 contract enforced by Auth0Validator.
    return pyjwt.encode(
        payload, _key_to_pem(key), algorithm="RS256",
        headers={"kid": "kid-1", "typ": "at+jwt"},
    )


def _set_auth0_env(monkeypatch, *, audience: str, access_mode: str = "internal") -> None:
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "auth0")
    monkeypatch.setenv("SHERLOC_AUTH0_DOMAIN", AUTH0_DOMAIN)
    monkeypatch.setenv("SHERLOC_AUTH0_AUDIENCE", audience)
    # §2.6.1 mandatory at v4.1 cutover.
    monkeypatch.setenv(
        "SHERLOC_AUTH0_IDENTITY_CLAIM_URI", PHASE_IDENTITY_CLAIM_URI
    )
    monkeypatch.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "spa_client_xyz")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", access_mode)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_AUDIENCE", raising=False)


def _build_app(tmp_path) -> TestClient:
    db_path = tmp_path / "scans_auth.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    return TestClient(create_app(engine=engine))


# ---------------------------------------------------------------------------
# G18.11.no-auth — no Authorization header → 401 on /api/scans
# ---------------------------------------------------------------------------


def test_auth0_mode_scans_no_header_returns_401(monkeypatch, tmp_path):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/scans")
    assert resp.status_code == 401, resp.text


def test_cf_access_mode_scans_no_header_returns_401(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "cf-access")
    monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", "test.cloudflareaccess.com")
    monkeypatch.setenv("SHERLOC_CF_AUDIENCE", "test-aud")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/scans")
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# G18.11.role-required — auth0 + correct role → 200 on /api/scans
# ---------------------------------------------------------------------------


def test_auth0_team_member_role_grants_scans_access(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=["phase:team-member"])
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 200, resp.text


def test_auth0_any_valid_token_grants_scans_access_in_public_mode(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    """§2.6.1 line 359: public-mode endpoints accept any valid token —
    no role requirement."""
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_PUBLIC, access_mode="public"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=[], audience=AUTH0_AUDIENCE_PUBLIC)
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# G18.11.role-required (negative) — auth0 + wrong role → 403 on /api/scans
# ---------------------------------------------------------------------------


def test_auth0_unrelated_role_on_internal_host_is_forbidden(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL, access_mode="internal"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=["phase:other"])
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# G18.11.no-role — empty identity claim → 403 on /api/scans
# ---------------------------------------------------------------------------


def test_auth0_empty_identity_is_forbidden_on_scans(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=[])
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# G18.11.audience-cross — internal-audience token sent to public-mode host → 401
# ---------------------------------------------------------------------------


def test_auth0_internal_audience_token_rejected_by_public_host(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    # Public host validates against PUBLIC audience but client sends a
    # token minted for INTERNAL audience. Per §13.3.4c the validator
    # rejects on audience mismatch (401, not 403).
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_PUBLIC, access_mode="public"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(
        signing_key,
        audience=AUTH0_AUDIENCE_INTERNAL,
        identity=["phase:team-member"],
    )
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# G18.11.expired — expired token → 401 on /api/scans
# ---------------------------------------------------------------------------


def test_auth0_expired_token_returns_401(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    # expires_in < 0 mints an already-expired token; pyjwt's leeway is 60s,
    # so we set expiry well past that.
    token = _mint(signing_key, identity=["phase:team-member"], expires_in=-3600)
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# Dev mode — passes by construction (DevValidator returns phase:team-member)
# ---------------------------------------------------------------------------


def test_dev_mode_allows_scans_without_token(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/scans")
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Public endpoints stay public — health + config must NOT 401
# ---------------------------------------------------------------------------


def test_health_endpoint_unauthenticated(monkeypatch, tmp_path):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text


def test_config_endpoint_unauthenticated(monkeypatch, tmp_path):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auth"]["auth_mode"] == "auth0"
