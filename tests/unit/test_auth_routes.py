"""Route-level auth integration tests against FastAPI TestClient.

Covers (per spec §13 / §18):

- G18.11.role-required — auth0 token with required role → 200; without → 403
- G18.11.audience-cross — internal-audience token sent to a public-mode
  container → 401 (the role-per-API isolation boundary)
- G18.11.no-role — auth0 token with empty roles → 403
- /api/config exposes the AuthConfig block expected by §13.4 (auth_mode +
  Auth0 fields populated only in auth0 mode).

Mocked JWKS via httpx_mock; no live Auth0 tenant.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Fixtures + helpers (kept local so this file stands alone)
# ---------------------------------------------------------------------------


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
    public_numbers = key.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
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


def _mint(
    key,
    *,
    audience: Any = None,
    identity: list[str] | None = None,
    email: Optional[str] = "alice@example.com",
    expires_in: int = 3600,
    extra_claims: Optional[dict] = None,
) -> str:
    """Mint a §2.6.1-shaped access token.

    ``identity`` populates the ``PHASE_IDENTITY_CLAIM_URI`` claim used
    by ``Auth0Validator`` for ``TokenClaims.roles``. The Phase A
    legacy ``{role_claim_uri}/roles`` path is gone at v4.1 — there is
    no longer a way to mint a token that exercises it.
    """
    now = datetime.now(timezone.utc)
    # §2.6.1 requires array-shaped ``aud``. Default to
    # ``[AUTH0_AUDIENCE_INTERNAL]`` to mirror Auth0's actual access-token
    # shape; tests can override (e.g., to inject the public audience for
    # a cross-deployment isolation check). String overrides are coerced
    # to a single-element list for ergonomic compatibility with existing
    # call-sites; explicit non-conforming-shape testing should pass a raw
    # scalar via ``payload["aud"] = ...`` outside this helper.
    if audience is None:
        audience = [AUTH0_AUDIENCE_INTERNAL]
    elif isinstance(audience, str):
        audience = [audience]
    payload: dict[str, Any] = {
        "iss": EXPECTED_ISSUER,
        "aud": audience,
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        "iat": int(now.timestamp()),
        "sub": "auth0|abc",
    }
    if email is not None:
        payload["email"] = email
    if identity is not None:
        payload[PHASE_IDENTITY_CLAIM_URI] = identity
    if extra_claims:
        payload.update(extra_claims)
    # ``typ=at+jwt`` (RFC 9068) matches what Auth0 emits for access
    # tokens and the §2.6.1 contract enforced by Auth0Validator.
    return pyjwt.encode(
        payload, _key_to_pem(key), algorithm="RS256",
        headers={"kid": "kid-1", "typ": "at+jwt"},
    )


def _set_auth0_env(monkeypatch, *, audience: str, access_mode: str = "internal") -> None:
    """§2.6.1 / Phase B auth0 env. Identity claim URI is mandatory at
    v4.1 — there is no longer a Phase A legacy mode."""
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "auth0")
    monkeypatch.setenv("SHERLOC_AUTH0_DOMAIN", AUTH0_DOMAIN)
    monkeypatch.setenv("SHERLOC_AUTH0_AUDIENCE", audience)
    monkeypatch.setenv(
        "SHERLOC_AUTH0_IDENTITY_CLAIM_URI", PHASE_IDENTITY_CLAIM_URI
    )
    monkeypatch.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "spa_client_xyz")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", access_mode)
    # Ensure no CF Access env leaks into the resolution path.
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_AUDIENCE", raising=False)


def _build_app(tmp_path) -> TestClient:
    db_path = tmp_path / "auth_routes.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    return TestClient(create_app(engine=engine))


# ---------------------------------------------------------------------------
# G18.11.role-required (positive): auth0 + correct role → 200
# ---------------------------------------------------------------------------


def test_auth0_with_team_member_role_grants_access(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=["phase:team-member"])

    with _build_app(tmp_path) as client:
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 7},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# G18.11.role-required (negative): auth0 + wrong role → 403
# ---------------------------------------------------------------------------


def test_auth0_with_unrelated_role_on_internal_host_is_forbidden(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL, access_mode="internal"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    # Token has identity entries, but none match phase:team-member.
    token = _mint(signing_key, identity=["phase:other"])

    with _build_app(tmp_path) as client:
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 7},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# G18.11.no-role: empty identity claim → 403
# ---------------------------------------------------------------------------


def test_auth0_with_empty_identity_is_forbidden(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=[])

    with _build_app(tmp_path) as client:
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 7},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 403, resp.text


# ---------------------------------------------------------------------------
# G18.11.audience-cross: internal-audience token to public-mode container → 401
# ---------------------------------------------------------------------------


def test_audience_cross_rejected_at_public_host(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    # Container env: auth0 mode, audience pinned to PUBLIC; access_mode=public.
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_PUBLIC, access_mode="public"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    # Operator presents a token minted for the INTERNAL audience.
    token = _mint(
        signing_key,
        audience=AUTH0_AUDIENCE_INTERNAL,
        identity=["phase:team-member"],
    )

    # Public mode requires phase_pds.db; build the app with one that
    # passes the create_app validator.
    db_path = tmp_path / "phase_pds.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    app = create_app(engine=engine, access_mode="public")

    with TestClient(app) as client:
        resp = client.get(
            "/api/user/preferences",
            headers={"Authorization": f"Bearer {token}"},
        )
    # Audience mismatch → AuthError → 401 (the cross-deployment isolation
    # boundary; the role gate at 403 never fires).
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# Anonymous (no Authorization header) in auth0 mode → 200, empty preferences
# ---------------------------------------------------------------------------


def test_auth0_anonymous_request_is_anonymous_not_rejected(
    monkeypatch, tmp_path
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with _build_app(tmp_path) as client:
        # No Authorization header at all.
        resp = client.get("/api/user/preferences")
    assert resp.status_code == 200
    assert resp.json()["preferences"] == []


# ---------------------------------------------------------------------------
# /api/config AuthConfig block (§13.4)
# ---------------------------------------------------------------------------


def test_api_config_in_auth0_mode_populates_auth_block(
    monkeypatch, tmp_path
):
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200, resp.text
    auth = resp.json()["auth"]
    assert auth["auth_mode"] == "auth0"
    assert auth["auth0_domain"] == AUTH0_DOMAIN
    assert auth["auth0_client_id"] == "spa_client_xyz"
    assert auth["auth0_audience"] == AUTH0_AUDIENCE_INTERNAL
    # role_claim_uri is no longer surfaced (the legacy
    # ``{role_claim_uri}/roles`` path is gone at v4.1); the schema field
    # remains on the AuthConfig DTO for SPA back-compat but always
    # serializes to None.
    assert auth.get("role_claim_uri") is None


def test_api_config_in_cf_access_mode_returns_null_auth0_fields(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "cf-access")
    monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", "team.cloudflareaccess.com")
    monkeypatch.setenv("SHERLOC_CF_AUDIENCE", "cf-aud-tag")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_AUTH0_AUDIENCE", raising=False)

    with _build_app(tmp_path) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200
    auth = resp.json()["auth"]
    assert auth["auth_mode"] == "cf-access"
    assert auth["auth0_domain"] is None
    assert auth["auth0_client_id"] is None
    assert auth["auth0_audience"] is None
    assert auth.get("role_claim_uri") is None


def test_api_config_in_dev_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    with _build_app(tmp_path) as client:
        resp = client.get("/api/config")
    assert resp.status_code == 200
    assert resp.json()["auth"]["auth_mode"] == "dev"


# ---------------------------------------------------------------------------
# B.12 F4 — sub is the identity key; email is optional metadata
# ---------------------------------------------------------------------------


def test_auth0_token_without_email_succeeds_on_user_route(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    """Per spec §13.1 + B.12 F4: Auth0 access tokens commonly omit email
    (it lives on the ID token). A token with sub + role but no email
    must still resolve a user record and authorise /api/user/* calls.
    """
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=["phase:team-member"], email=None)

    with _build_app(tmp_path) as client:
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 42},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# §2.6.1 D5 — WWW-Authenticate header on 401
# ---------------------------------------------------------------------------


def test_missing_token_401_includes_www_authenticate_header(
    monkeypatch, tmp_path
):
    """Per m2020-phase Revised++ §2.6.1: a missing-credential 401 on a
    protected endpoint MUST include ``WWW-Authenticate: Bearer
    realm="m2020-phase"`` (the contract literal). v4.1 retired the
    Phase A parallel-run ``sherloc`` realm default.
    """
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    monkeypatch.delenv("SHERLOC_AUTH_REALM", raising=False)
    with _build_app(tmp_path) as client:
        resp = client.get("/api/scans")  # no Authorization header
    assert resp.status_code == 401
    assert (
        resp.headers.get("WWW-Authenticate") == 'Bearer realm="m2020-phase"'
    )


def test_missing_token_401_realm_overridable_via_env(monkeypatch, tmp_path):
    """``SHERLOC_AUTH_REALM`` overrides the §2.6.1 default — useful for
    staging tenants or future platform renames without code changes."""
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    monkeypatch.setenv("SHERLOC_AUTH_REALM", "staging-phase")
    with _build_app(tmp_path) as client:
        resp = client.get("/api/scans")
    assert resp.status_code == 401
    assert (
        resp.headers.get("WWW-Authenticate")
        == 'Bearer realm="staging-phase"'
    )


def test_invalid_token_401_includes_www_authenticate_header(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    """Invalid-token 401 (signature/audience/expiry/etc.) carries the
    same WWW-Authenticate header so clients can re-auth uniformly."""
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL,
        json={"keys": [_public_jwk(signing_key, kid="kid-1")]},
    )
    # Token signed with a different key — signature won't verify.
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _mint(other_key, identity=["phase:team-member"])
    with _build_app(tmp_path) as client:
        resp = client.get(
            "/api/scans", headers={"Authorization": f"Bearer {token}"}
        )
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate", "").startswith("Bearer realm=")


# ---------------------------------------------------------------------------
# §2.6.1 D6 — successful validation logs sub claim only
# ---------------------------------------------------------------------------


def test_successful_validation_logs_sub_only(
    monkeypatch, tmp_path, httpx_mock, signing_key, caplog
):
    """Per §2.6.1 logging contract: on success the validator emits a
    DEBUG trace line carrying ``sub`` only — no email, no roles, no
    token contents. The line is the only authn log for a 2xx request.
    """
    import logging

    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(signing_key, identity=["phase:team-member"])
    with caplog.at_level(logging.DEBUG, logger="sherloc_pipeline.web.auth"):
        with _build_app(tmp_path) as client:
            resp = client.get(
                "/api/scans",
                headers={"Authorization": f"Bearer {token}"},
            )
    assert resp.status_code == 200
    success_logs = [
        rec for rec in caplog.records if "authn ok sub=" in rec.message
    ]
    assert success_logs, "expected at least one authn-ok log line"
    log_text = " ".join(rec.message for rec in success_logs)
    # sub appears; email and identity entries do not leak.
    assert "auth0|abc" in log_text
    assert "alice@example.com" not in log_text
    assert "phase:team-member" not in log_text


def test_failed_validation_logs_reason_code_no_token(
    monkeypatch, tmp_path, caplog
):
    """Failure logging uses a reason code, never token contents
    (§2.6.1)."""
    import logging

    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    with caplog.at_level(logging.WARNING, logger="sherloc_pipeline.web.auth"):
        with _build_app(tmp_path) as client:
            resp = client.get("/api/scans")  # no token
    assert resp.status_code == 401
    failure_logs = [
        rec for rec in caplog.records if "authn failed reason=" in rec.message
    ]
    assert failure_logs, "expected at least one authn-failed log"
    assert any(
        "reason=no_credential" in rec.message for rec in failure_logs
    )


def test_invalid_token_failure_log_omits_token_and_pyjwt_text(
    monkeypatch, tmp_path, httpx_mock, signing_key, caplog
):
    """Invalid-token failure path emits ``reason=invalid_token`` and
    must NOT carry token bytes, email, role values, or the underlying
    PyJWT exception text. §2.6.1: 'Validation failures logged with
    reason code, NOT with token contents'.
    """
    import logging

    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL,
        json={"keys": [_public_jwk(signing_key, kid="kid-1")]},
    )
    # Sign with a different key so PyJWT raises an InvalidSignature
    # exception during decode — exercises the AuthError-handling branch
    # in require_authenticated_request that emits ``reason=invalid_token``.
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _mint(
        other_key,
        identity=["phase:team-member"],
        email="alice@example.com",
    )
    with caplog.at_level(logging.WARNING, logger="sherloc_pipeline.web.auth"):
        with _build_app(tmp_path) as client:
            resp = client.get(
                "/api/scans", headers={"Authorization": f"Bearer {token}"}
            )
    assert resp.status_code == 401

    failure_logs = [
        rec for rec in caplog.records
        if "authn failed reason=" in rec.message
        and "auth.py" in (rec.pathname or "")
    ]
    assert failure_logs, "expected an invalid-token failure log"
    # Coarse reason code is present.
    assert any("reason=invalid_token" in rec.message for rec in failure_logs)
    log_text = " ".join(rec.message for rec in failure_logs)
    # Token bytes / sensitive claim values must not leak into the log.
    assert token not in log_text, "raw token leaked into failure log"
    assert "alice@example.com" not in log_text
    assert "phase:team-member" not in log_text
    # PyJWT-specific exception text is intentionally not propagated.
    for needle in ("InvalidSignatureError", "Signature verification failed"):
        assert needle not in log_text, f"PyJWT detail '{needle}' leaked"


def test_auth0_two_tokens_with_same_sub_resolve_same_user(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    """Two requests with the same sub must hit the SAME user row, even
    when one carries email and the other does not. Prevents a regression
    where Auth0 access tokens (sub-only) and CF Access tokens (sub +
    email) resolve to different user records during the parallel-run
    window.
    """
    _set_auth0_env(monkeypatch, audience=AUTH0_AUDIENCE_INTERNAL)
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )

    with _build_app(tmp_path) as client:
        # First request: token WITH email; sets a preference.
        token_with_email = _mint(
            signing_key, identity=["phase:team-member"], email="alice@example.com"
        )
        r1 = client.put(
            "/api/user/preferences/k1",
            json={"value": 1},
            headers={"Authorization": f"Bearer {token_with_email}"},
        )
        assert r1.status_code == 200, r1.text

        # Second request: token WITHOUT email but same sub; must read the
        # SAME user's preferences (single record per sub).
        token_no_email = _mint(
            signing_key, identity=["phase:team-member"], email=None
        )
        r2 = client.get(
            "/api/user/preferences",
            headers={"Authorization": f"Bearer {token_no_email}"},
        )
        assert r2.status_code == 200, r2.text
        prefs = {p["key"]: p["value"] for p in r2.json()["preferences"]}
        assert prefs.get("k1") == 1, prefs


# ---------------------------------------------------------------------------
# §2.6.1 — public-mode route gate accepts any valid token
# ---------------------------------------------------------------------------


def test_public_mode_accepts_any_valid_token(
    monkeypatch, tmp_path, httpx_mock, signing_key
):
    """§2.6.1 line 359: public-mode endpoints accept any valid token —
    no role requirement. An empty identity array must NOT 403."""
    _set_auth0_env(
        monkeypatch, audience=AUTH0_AUDIENCE_PUBLIC, access_mode="public"
    )
    httpx_mock.add_response(
        url=JWKS_URL, json={"keys": [_public_jwk(signing_key, kid="kid-1")]}
    )
    token = _mint(
        signing_key,
        audience=AUTH0_AUDIENCE_PUBLIC,
        identity=[],
    )
    with _build_app(tmp_path) as client:
        resp = client.put(
            "/api/user/preferences/test_key",
            json={"value": 7},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert resp.status_code == 200, resp.text
