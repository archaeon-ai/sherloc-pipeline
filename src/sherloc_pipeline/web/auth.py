"""Token validation: ``TokenValidator`` Protocol + three concrete validators.

§13.1 abstraction: routes depend only on ``TokenValidator.validate(token)``;
swapping auth backends is a DI change, not a route rewrite.

Concrete validators:
  - ``Auth0Validator``    — re-exported from ``phase_platform_auth``.
                            Single source of truth for the §2.6.1
                            contract (archaeon-ai/phase-platform-auth
                            >= 0.1.0). v4.1 B.0 switchover landed this
                            cutover; SHERLOC no longer carries a local
                            §2.6.1 implementation.
  - ``CFAccessValidator`` — Cloudflare Access JWT (legacy; retained at
                            v4.1 for the parallel-run with legacy host —
                            §13.6).
  - ``DevValidator``      — synthetic claims for ``SHERLOC_AUTH_MODE=dev``
                            local development (§13.5; G3).

JWKS caching is per-process and lives in ``phase_platform_auth.jwks``.
The CF Access path reuses the same shared cache via the package's
``fetch_jwks`` / ``find_signing_key`` helpers, so the two HTTP providers
share the eviction policy without colliding (cache is keyed by JWKS URL).

``get_validator()`` reads ``SHERLOC_AUTH_MODE`` and returns a
process-cached validator instance.

Spec contract: §2.6.1 of the PHASE Platform validator surface
(re-exported from ``phase_platform_auth``).
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

import jwt
from fastapi import HTTPException, Request
from jwt.exceptions import InvalidTokenError, PyJWKClientError

# §2.6.1 contract: imported from the package, not redefined here.
from phase_platform_auth import (
    Auth0Validator,
    AuthError,
    JWKSUnavailableError,
    TokenClaims,
    build_www_authenticate,
)
from phase_platform_auth.jwks import (
    _jwks_cache,
    _jwks_lock,
    fetch_jwks,
    find_signing_key,
    reset_cache_for_tests as _reset_jwks_cache_for_tests,
)

logger = logging.getLogger(__name__)


# Legacy CF Access cache window (preserved for parity with prior behavior).
JWKS_TTL_SECONDS = 3600
JWKS_GRACE_SECONDS = 86_400


# ---------------------------------------------------------------------------
# Re-exports for back-compat with existing SHERLOC imports.
# ---------------------------------------------------------------------------

# Tests and a handful of consumers import these names from
# ``sherloc_pipeline.web.auth``; the package owns the implementation but the
# import surface stays stable across the v4.0 → v4.1 cutover.
__all__ = [
    "Auth0Validator",
    "AuthError",
    "CFAccessValidator",
    "DevValidator",
    "JWKSUnavailableError",
    "JWKS_GRACE_SECONDS",
    "JWKS_TTL_SECONDS",
    "TokenClaims",
    "TokenValidator",
    "build_www_authenticate",
    "fetch_cf_jwks",
    "get_validator",
    "required_role_for_access_mode",
    "require_authenticated_request",
    "validate_cf_jwt",
]


# ---------------------------------------------------------------------------
# Cloudflare Access helpers (legacy interface; preserved for callers/tests)
# ---------------------------------------------------------------------------


def _cf_jwks_url(team_domain: str) -> str:
    return f"https://{team_domain}/cdn-cgi/access/certs"


def fetch_cf_jwks(team_domain: str) -> dict:
    """Fetch (and cache) Cloudflare Access JWKs for ``team_domain``.

    Thin shim over the package's ``fetch_jwks`` that preserves the
    prior dict-only return signature for existing tests and external
    callers operating at the JWKS layer rather than through
    ``CFAccessValidator``. CF Access has been treating any
    kid-not-found as 401 since pre-B.1, and the F5 outage-vs-invalid-
    token distinction is an Auth0-only refinement that arrived with
    §2.6.1; CF Access sunset removes this surface entirely in a later
    v4.x.
    """
    jwks, _source = fetch_jwks(
        _cf_jwks_url(team_domain),
        ttl_seconds=JWKS_TTL_SECONDS,
        max_stale_seconds=JWKS_GRACE_SECONDS,
    )
    return jwks


def validate_cf_jwt(
    token: Optional[str], audience: str, team_domain: str
) -> dict:
    """Validate a Cloudflare Access JWT and return the raw claims dict.

    Returns the raw decoded claims (not ``TokenClaims``) so existing
    callers and tests that read ``claims["email"]`` / ``["aud"]`` /
    ``["iss"]`` keep working. New code should call
    ``CFAccessValidator(...).validate(token)`` for normalized claims.
    """
    if not token:
        raise AuthError("Missing JWT")

    expected_issuer = f"https://{team_domain}"
    jwks = fetch_cf_jwks(team_domain)
    try:
        signing_key = find_signing_key(token, jwks)
    except (PyJWKClientError, InvalidTokenError) as exc:
        raise AuthError(f"Could not resolve signing key: {exc}") from exc

    try:
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=audience,
            issuer=expected_issuer,
            options={"require": ["exp", "iss", "aud"]},
        )
    except InvalidTokenError as exc:
        raise AuthError(f"JWT validation failed: {exc}") from exc

    return claims


# ---------------------------------------------------------------------------
# TokenValidator + concrete validators (§13.1)
# ---------------------------------------------------------------------------


class TokenValidator(Protocol):
    """Validates an opaque token and returns ``TokenClaims`` or raises."""

    def validate(self, token: str) -> TokenClaims: ...


class CFAccessValidator:
    """Cloudflare Access JWT validator (§13.6).

    Retained at v4.1 for the parallel-run window; deletion is a later
    v4.x follow-up after legacy web-UI sunset (§1.4 operator-discretion).
    """

    def __init__(self, team_domain: str, audience: str) -> None:
        self.team_domain = team_domain
        self.audience = audience

    @property
    def jwks_url(self) -> str:
        return _cf_jwks_url(self.team_domain)

    @property
    def expected_issuer(self) -> str:
        return f"https://{self.team_domain}"

    def validate(self, token: str) -> TokenClaims:
        claims = validate_cf_jwt(
            token, audience=self.audience, team_domain=self.team_domain
        )
        email = claims.get("email")
        sub = claims.get("sub") or email or ""
        return TokenClaims(
            sub=str(sub),
            email=email,
            roles=[],  # CF Access has no role concept in this deployment
            expires_at=datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc),
        )


class DevValidator:
    """Synthetic-claims validator for ``SHERLOC_AUTH_MODE=dev`` (§13.5).

    Returns a fixed identity regardless of the token. Localhost-only —
    NEVER expose a dev-mode container to public networks (no token
    signature check; any request authenticates as the synthetic
    identity). Synthetic ``roles`` carry the §2.6.1 ``phase:team-member``
    identity so dev-mode requests pass the Phase B route gate without
    needing a real Auth0 token.
    """

    DEV_SUB = "localhost-dev"
    DEV_EMAIL = "dev@local"

    def validate(self, token: str) -> TokenClaims:
        return TokenClaims(
            sub=self.DEV_SUB,
            email=self.DEV_EMAIL,
            roles=["phase:team-member"],
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )


# ---------------------------------------------------------------------------
# Validator selection
# ---------------------------------------------------------------------------


_validator_singleton: Optional[TokenValidator] = None
_validator_lock = threading.Lock()


def get_validator() -> TokenValidator:
    """Return the process-wide validator for the current ``SHERLOC_AUTH_MODE``.

    Modes (per spec §13):
      - ``"auth0"``     → ``Auth0Validator`` from ``phase_platform_auth``
        (reads ``SHERLOC_AUTH0_*``; ``SHERLOC_AUTH0_IDENTITY_CLAIM_URI``
        is mandatory per §2.6.1).
      - ``"cf-access"`` → ``CFAccessValidator`` (reads
        ``SHERLOC_CF_TEAM_DOMAIN`` / ``SHERLOC_CF_AUDIENCE``).
      - ``"dev"``       → ``DevValidator``.

    Cached across requests; ``_reset_validator_for_tests()`` drops the
    instance after env mutation in tests.
    """
    global _validator_singleton
    if _validator_singleton is not None:
        return _validator_singleton
    with _validator_lock:
        if _validator_singleton is None:
            _validator_singleton = _build_validator()
        return _validator_singleton


def _build_validator() -> TokenValidator:
    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")
    if mode == "dev":
        return DevValidator()
    if mode == "auth0":
        domain = _require_env("SHERLOC_AUTH0_DOMAIN")
        audience = _require_env("SHERLOC_AUTH0_AUDIENCE")
        # §2.6.1 requires the identity claim URI; the package's
        # Auth0Validator constructor enforces this at the type level
        # (no longer Optional). Make startup fail loud and explicit.
        identity_claim_uri = _require_env("SHERLOC_AUTH0_IDENTITY_CLAIM_URI")
        ttl = int(
            os.environ.get(
                "SHERLOC_AUTH0_JWKS_TTL_SECONDS", "600"
            )
        )
        max_stale = int(
            os.environ.get(
                "SHERLOC_AUTH0_JWKS_MAX_STALE_SECONDS",
                "86400",
            )
        )
        spa_ids_env = os.environ.get("SHERLOC_AUTH0_KNOWN_SPA_CLIENT_IDS", "")
        spa_ids = [s.strip() for s in spa_ids_env.split(",") if s.strip()]
        expected_azp = os.environ.get("SHERLOC_AUTH0_EXPECTED_AZP") or None
        return Auth0Validator(
            domain=domain,
            audience=audience,
            identity_claim_uri=identity_claim_uri,
            jwks_cache_ttl=ttl,
            jwks_max_stale_seconds=max_stale,
            expected_azp=expected_azp,
            known_spa_client_ids=spa_ids,
        )
    if mode == "cf-access":
        team_domain = _require_env("SHERLOC_CF_TEAM_DOMAIN")
        audience = _require_env("SHERLOC_CF_AUDIENCE")
        return CFAccessValidator(team_domain=team_domain, audience=audience)
    raise RuntimeError(
        f"Unrecognized SHERLOC_AUTH_MODE={mode!r}; "
        f"expected one of auth0, cf-access, dev."
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Auth misconfigured: {name} must be set for the current SHERLOC_AUTH_MODE."
        )
    return value


def _reset_validator_for_tests() -> None:
    """Test-only helper to drop the cached validator (after env mutation)."""
    global _validator_singleton
    with _validator_lock:
        _validator_singleton = None


# ---------------------------------------------------------------------------
# Request-level authentication dependency (§13.3, §13.3.7)
# ---------------------------------------------------------------------------


def _extract_request_token(request: Request, mode: str) -> Optional[str]:
    """Pick the credential header for the active auth mode.

    - ``cf-access``: ``Cf-Access-Jwt-Assertion`` (Cloudflare-injected; the
      ``Authorization`` header is stripped at the edge).
    - ``auth0`` / anything else: ``Authorization: Bearer <jwt>``.
    """
    if mode == "cf-access":
        return request.headers.get("Cf-Access-Jwt-Assertion")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def required_role_for_access_mode(access_mode: str) -> Optional[str]:
    """Resolve the required role-per-API for the given access mode.

    m2020-phase Revised++ §2.6.1: team-mode endpoints require
    ``phase:team-member``; public-mode endpoints accept any valid token
    (no role requirement — §2.6.1 line 359).

    Returns ``None`` when the access mode does not impose a role
    requirement (public mode, or unrecognized access mode). Callers
    must short-circuit the role check on ``None``.
    """
    return {
        "internal": "phase:team-member",
        "public": None,
    }.get(access_mode)


def _www_authenticate_header() -> dict[str, str]:
    """Build the ``WWW-Authenticate`` header per m2020-phase §2.6.1.

    Reads ``SHERLOC_AUTH_REALM`` per call so deployment-time env changes
    take effect without re-importing the module. The default realm is
    ``m2020-phase`` (the §2.6.1 contract literal); deployments may
    override (staging, future platform renames).
    """
    realm = os.environ.get("SHERLOC_AUTH_REALM", "m2020-phase")
    return build_www_authenticate(realm=realm)


def require_authenticated_request(request: Request) -> TokenClaims:
    """FastAPI dependency: enforce per-request authentication + role-per-API.

    Apply at the router level on every data-bearing API surface
    (``scans``, ``spectra``, ``plots``, ``images``, ``pds``,
    ``processing``, ``jobs``, ``map``, ``user``). Per spec §13.3 the
    backend must validate the credential on every request; per §13.3.7
    the role-per-API gate ensures a token issued for one deployment's
    audience cannot grant access to another's API.

    Public endpoints (``/api/health``, ``/api/config``) MUST NOT depend
    on this — health checks are unauthenticated by load-balancer
    convention, and the SPA fetches ``/api/config`` BEFORE login to
    learn how to bootstrap auth (§13.4).

    Behavior matrix:

    +---------------+-------------------------------------------+--------+
    | mode          | flow                                      | result |
    +===============+===========================================+========+
    | dev           | ``DevValidator.validate("")`` returns     | 200    |
    |               | synthetic claims with                     |        |
    |               | ``phase:team-member``; role-per-API check |        |
    |               | passes by construction.                   |        |
    +---------------+-------------------------------------------+--------+
    | cf-access     | Validate ``Cf-Access-Jwt-Assertion``.     | 200    |
    |               | CF Access has no role concept (§13.6);    |        |
    |               | role-per-API enforcement is SKIPPED.      |        |
    |               | Removal paired with v4.x CF Access        |        |
    |               | sunset.                                   |        |
    +---------------+-------------------------------------------+--------+
    | auth0         | Validate ``Authorization: Bearer`` via    | 200    |
    |               | ``phase_platform_auth.Auth0Validator``;   |        |
    |               | enforce role-per-API against              |        |
    |               | ``SHERLOC_ACCESS_MODE``                   |        |
    |               | (internal→phase:team-member,              |        |
    |               | public→no role required).                 |        |
    +---------------+-------------------------------------------+--------+

    Status codes match G18.11.* sub-gates per spec §18.2:

    - Missing credential header                   → 401 (G18.11.no-auth)
    - Token signature/issuer/audience/expiry fail → 401 (G18.11.signature/audience/expired)
    - Wrong audience (cross-deployment isolation) → 401 (G18.11.audience-cross)
    - JWKS unreachable + no usable cache          → 503
    - Auth misconfigured (missing required env)   → 500
    - Token valid + missing required role         → 403 (G18.11.role-required, .no-role)
    """
    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")

    if mode == "dev":
        # DevValidator ignores the token; synthetic claims always pass.
        try:
            claims = get_validator().validate("")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        logger.debug("authn ok sub=%s", claims.sub)
        return claims

    token = _extract_request_token(request, mode)
    if not token:
        # m2020-phase Revised++ §2.6.1: missing-credential 401 must
        # surface a ``WWW-Authenticate: Bearer realm=...`` header.
        logger.warning("authn failed reason=no_credential")
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers=_www_authenticate_header(),
        )

    try:
        validator = get_validator()
        claims = validator.validate(token)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except JWKSUnavailableError:
        # No reason-coded log for the auth failure here — the
        # JWKS-refresh failure path already logs at WARNING with
        # endpoint context (see ``phase_platform_auth.jwks.fetch_jwks``);
        # duplicating would be noise.
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable",
        )
    except AuthError:
        # §2.6.1: failures logged with a reason code, never with
        # token contents. The reason here is a coarse "invalid_token"
        # bucket; the underlying AuthError message (which may include
        # the PyJWT exception text) is intentionally NOT propagated to
        # logs to keep token-bearing diagnostics out of the log stream.
        logger.warning("authn failed reason=invalid_token")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers=_www_authenticate_header(),
        )

    # §13.3.7 role-per-API enforcement; cf-access exempt per §13.6.
    if mode != "cf-access":
        access_mode = os.environ.get("SHERLOC_ACCESS_MODE", "internal")
        required_role = required_role_for_access_mode(access_mode)
        if required_role and required_role not in claims.roles:
            logger.warning(
                "authn ok but authz failed reason=missing_role sub=%s",
                claims.sub,
            )
            raise HTTPException(
                status_code=403,
                detail=f"Required role missing: {required_role}",
            )

    # §2.6.1: successful validation logs the sub claim only.
    logger.debug("authn ok sub=%s", claims.sub)
    return claims
