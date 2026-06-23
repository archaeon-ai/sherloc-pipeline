"""User preference and classification profile endpoints.

Identity: Cloudflare Zero Trust JWT → email.
Public mode: anonymous (no server persistence, client uses localStorage).
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sherloc_pipeline.database.models import (
    ClassificationProfileORM,
    UserORM,
    UserPreferenceORM,
)
from sherloc_pipeline.services.classification import (
    ClassificationProfile,
    get_default_classification_rules,
)
from sherloc_pipeline.web.auth import (
    AuthError,
    DevValidator,
    JWKSUnavailableError,
    get_validator,
    required_role_for_access_mode,
)
from sherloc_pipeline.web.data_access import DataAccessService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/user", tags=["user"])


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class PreferenceValue(BaseModel):
    value: Any


class PreferenceItem(BaseModel):
    key: str
    value: Any
    updated_at: str


class PreferencesResponse(BaseModel):
    preferences: list[PreferenceItem]


class ProfileListItem(BaseModel):
    id: str
    name: str
    base: str
    created_at: str
    updated_at: str


class ProfileListResponse(BaseModel):
    profiles: list[ProfileListItem]


class ProfileBody(BaseModel):
    id: Optional[str] = None
    name: str
    base: str = "default"
    overrides: list[dict] = []


class ProfileResponse(BaseModel):
    id: str
    name: str
    base: str
    overrides: list[dict]
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_or_create_user(
    session: Session, sub: str, email: Optional[str] = None
) -> UserORM:
    """Get existing user or auto-create on first request.

    ``sub`` is the stable identity key per spec §13.1 + B.12 F4.
    ``email`` is optional metadata — backfilled onto an existing record
    if the validator surfaces an email and we don't already have one.
    """
    user = session.query(UserORM).filter(UserORM.sub == sub).first()
    if user is None:
        now = _utc_now()
        user = UserORM(sub=sub, email=email, created_at=now, last_seen_at=now)
        session.add(user)
        session.flush()
    else:
        user.last_seen_at = _utc_now()
        if email and not user.email:
            user.email = email
    return user


def _extract_token(request: Request, mode: str) -> Optional[str]:
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


def _resolve_user(request: Request) -> tuple[Session, Optional[UserORM]]:
    """Resolve the request identity through the configured ``TokenValidator``.

    Returns ``(session, None)`` when no token is presented (anonymous).
    Returns ``(session, UserORM)`` when the token validates.

    Raises ``HTTPException`` 401 on token validation failure, 503 on
    JWKS unavailability, or 500 if the deployment is missing required
    auth env vars.

    Dev escape hatch: ``SHERLOC_AUTH_MODE=dev`` returns a synthetic
    identity from ``DevValidator`` without inspecting the token.
    """
    session = request.state.db
    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")

    if mode == "dev":
        try:
            claims = get_validator().validate("")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return session, _get_or_create_user(
            session, sub=claims.sub, email=claims.email or DevValidator.DEV_EMAIL
        )

    token = _extract_token(request, mode)
    if not token:
        return session, None

    try:
        validator = get_validator()
        claims = validator.validate(token)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except JWKSUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Authentication service unavailable",
        )
    except AuthError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    # §13.3.7 role-per-API enforcement: a token without the role for
    # this deployment must be rejected (403). CF Access is exempt
    # because CF Access has no role concept in this deployment (§13.6
    # / B.1: CFAccessValidator returns roles=[]); enforcing it would
    # 403 every existing legacy request. Role mapping is shared with
    # require_authenticated_request via required_role_for_access_mode
    # so Phase B namespace (phase:team-member, optional public-mode no
    # role required) is honored uniformly across data routes and
    # /api/user/*.
    if mode != "cf-access":
        access_mode = os.environ.get("SHERLOC_ACCESS_MODE", "internal")
        required_role = required_role_for_access_mode(access_mode)
        if required_role and required_role not in claims.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Required role missing: {required_role}",
            )

    # Per spec §13.1 + B.12 F4: ``sub`` is the stable identity key.
    # ``email`` is optional metadata (Auth0 access tokens commonly omit
    # it; CF Access always sets sub == email so cf-access continues to
    # resolve the same user records).
    if not claims.sub:
        raise HTTPException(status_code=401, detail="Token missing sub claim")
    return session, _get_or_create_user(
        session, sub=claims.sub, email=claims.email
    )


def _require_user(request: Request) -> tuple[Session, UserORM]:
    """Resolve user or raise 403 for anonymous requests."""
    session, user = _resolve_user(request)
    if user is None:
        raise HTTPException(
            status_code=403,
            detail="Authentication required. Anonymous users cannot save preferences.",
        )
    return session, user


def _profile_orm_to_response(orm: ClassificationProfileORM) -> ProfileResponse:
    try:
        parsed = json.loads(orm.profile_json)
    except Exception:
        parsed = {}
    return ProfileResponse(
        id=orm.id,
        name=orm.name,
        base=parsed.get("base", "default"),
        overrides=parsed.get("overrides", []),
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


# ---------------------------------------------------------------------------
# Preference endpoints
# ---------------------------------------------------------------------------


@router.get("/preferences", response_model=PreferencesResponse)
def get_preferences(request: Request) -> PreferencesResponse:
    """Return all preferences for the current user.

    Returns 200 with empty list for anonymous users (client manages state
    via localStorage in that case).
    """
    session, user = _resolve_user(request)
    if user is None:
        return PreferencesResponse(preferences=[])

    prefs = (
        session.query(UserPreferenceORM)
        .filter(UserPreferenceORM.user_id == user.id)
        .all()
    )
    items = []
    for pref in prefs:
        try:
            value = json.loads(pref.value)
        except Exception:
            value = pref.value
        items.append(PreferenceItem(key=pref.key, value=value, updated_at=pref.updated_at))
    return PreferencesResponse(preferences=items)


@router.put("/preferences/{key}", response_model=PreferenceItem)
def set_preference(request: Request, key: str, body: PreferenceValue) -> PreferenceItem:
    """Set a preference value for the current user."""
    session, user = _require_user(request)

    pref = (
        session.query(UserPreferenceORM)
        .filter(UserPreferenceORM.user_id == user.id, UserPreferenceORM.key == key)
        .first()
    )
    now = _utc_now()
    encoded = json.dumps(body.value)

    if pref is None:
        pref = UserPreferenceORM(
            user_id=user.id,
            key=key,
            value=encoded,
            updated_at=now,
        )
        session.add(pref)
    else:
        pref.value = encoded
        pref.updated_at = now

    session.flush()
    return PreferenceItem(key=key, value=body.value, updated_at=now)


@router.delete("/preferences/{key}", status_code=204)
def delete_preference(request: Request, key: str) -> Response:
    """Reset (delete) a preference, reverting to the client default."""
    session, user = _require_user(request)

    deleted = (
        session.query(UserPreferenceORM)
        .filter(UserPreferenceORM.user_id == user.id, UserPreferenceORM.key == key)
        .delete()
    )
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Preference '{key}' not found")
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Classification profile endpoints
# ---------------------------------------------------------------------------


@router.get("/profiles", response_model=ProfileListResponse)
def list_profiles(request: Request) -> ProfileListResponse:
    """List all classification profiles for the current user.

    Returns 200 with empty list for anonymous users.
    """
    session, user = _resolve_user(request)
    if user is None:
        return ProfileListResponse(profiles=[])

    profiles = (
        session.query(ClassificationProfileORM)
        .filter(ClassificationProfileORM.user_id == user.id)
        .order_by(ClassificationProfileORM.created_at)
        .all()
    )
    items = [
        ProfileListItem(
            id=p.id,
            name=p.name,
            base=json.loads(p.profile_json).get("base", "default"),
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in profiles
    ]
    return ProfileListResponse(profiles=items)


@router.post("/profiles", response_model=ProfileResponse, status_code=201)
def create_profile(request: Request, body: ProfileBody) -> ProfileResponse:
    """Create a new classification profile."""
    session, user = _require_user(request)

    profile_id = body.id or str(uuid.uuid4())
    now = _utc_now()

    # Validate no duplicate id
    existing = (
        session.query(ClassificationProfileORM)
        .filter(ClassificationProfileORM.id == profile_id)
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Profile '{profile_id}' already exists")

    profile_json = json.dumps({
        "id": profile_id,
        "name": body.name,
        "base": body.base,
        "overrides": body.overrides,
    })
    orm = ClassificationProfileORM(
        id=profile_id,
        user_id=user.id,
        name=body.name,
        profile_json=profile_json,
        created_at=now,
        updated_at=now,
    )
    session.add(orm)
    session.flush()

    return ProfileResponse(
        id=profile_id,
        name=body.name,
        base=body.base,
        overrides=body.overrides,
        created_at=now,
        updated_at=now,
    )


@router.put("/profiles/{profile_id}", response_model=ProfileResponse)
def update_profile(request: Request, profile_id: str, body: ProfileBody) -> ProfileResponse:
    """Update an existing classification profile."""
    session, user = _require_user(request)

    orm = (
        session.query(ClassificationProfileORM)
        .filter(
            ClassificationProfileORM.id == profile_id,
            ClassificationProfileORM.user_id == user.id,
        )
        .first()
    )
    if orm is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    now = _utc_now()
    orm.name = body.name
    orm.profile_json = json.dumps({
        "id": profile_id,
        "name": body.name,
        "base": body.base,
        "overrides": body.overrides,
    })
    orm.updated_at = now

    return ProfileResponse(
        id=profile_id,
        name=body.name,
        base=body.base,
        overrides=body.overrides,
        created_at=orm.created_at,
        updated_at=now,
    )


@router.delete("/profiles/{profile_id}", status_code=204)
def delete_profile(request: Request, profile_id: str) -> Response:
    """Delete a classification profile."""
    session, user = _require_user(request)

    deleted = (
        session.query(ClassificationProfileORM)
        .filter(
            ClassificationProfileORM.id == profile_id,
            ClassificationProfileORM.user_id == user.id,
        )
        .delete()
    )
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Profile not found")
    return Response(status_code=204)


@router.get("/profiles/{profile_id}/export")
def export_profile(request: Request, profile_id: str) -> Response:
    """Download a classification profile as JSON."""
    session, user = _require_user(request)

    orm = (
        session.query(ClassificationProfileORM)
        .filter(
            ClassificationProfileORM.id == profile_id,
            ClassificationProfileORM.user_id == user.id,
        )
        .first()
    )
    if orm is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    filename = f"classification_profile_{orm.name.replace(' ', '_')}.json"
    return Response(
        content=orm.profile_json,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/profiles/import", response_model=ProfileResponse, status_code=201)
def import_profile(request: Request, body: ProfileBody) -> ProfileResponse:
    """Upload a JSON profile body and create a new profile from it.

    Assigns a fresh UUID regardless of the id field in the payload to
    avoid conflicts with existing profiles.
    """
    session, user = _require_user(request)

    profile_id = str(uuid.uuid4())
    now = _utc_now()

    profile_json = json.dumps({
        "id": profile_id,
        "name": body.name,
        "base": body.base,
        "overrides": body.overrides,
    })
    orm = ClassificationProfileORM(
        id=profile_id,
        user_id=user.id,
        name=body.name,
        profile_json=profile_json,
        created_at=now,
        updated_at=now,
    )
    session.add(orm)
    session.flush()

    return ProfileResponse(
        id=profile_id,
        name=body.name,
        base=body.base,
        overrides=body.overrides,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Classification defaults (no auth required)
# ---------------------------------------------------------------------------


@router.get("/classification/defaults")
def get_classification_defaults(request: Request) -> dict:
    """Return the default classification rules.

    No authentication required — these are read-only reference data.
    """
    return get_default_classification_rules()
