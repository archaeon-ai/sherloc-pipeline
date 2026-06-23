"""Startup configuration validator for containerized deployments.

Invoked by ``docker-entrypoint.sh`` before Alembic migrations and the
uvicorn server start. Validates required environment variables and
fails fast (exit code 1) with explicit ``missing required variable: NAME``
messages so deployment misconfiguration is obvious in container logs.

Per PUBLIC_TOOLKIT_ARCHITECTURE_SPEC §7.3 and §14.1.

Variables validated:
    SHERLOC_DB         — must point to a writable path (or its parent
                         directory must exist and be writable so SQLite
                         can create the file on first migration)
    SHERLOC_AUTH_MODE  — one of {auth0, cf-access, dev}; defaults to
                         ``cf-access`` when unset (legacy
                         behavior). Auth0 mode wiring lands in B.1
                         and is accepted here so the variable can be
                         set in env files without tripping validation.
    SHERLOC_AUTH0_DOMAIN, SHERLOC_AUTH0_AUDIENCE,
    SHERLOC_AUTH0_SPA_CLIENT_ID
                       — required when SHERLOC_AUTH_MODE=auth0
                         (SPA_CLIENT_ID is surfaced via /api/config so
                         the SPA can bootstrap @auth0/auth0-spa-js
                         without rebuild — §13.4)
    SHERLOC_CF_TEAM_DOMAIN, SHERLOC_CF_AUDIENCE
                       — required when SHERLOC_AUTH_MODE=cf-access

Exit codes:
    0  all checks passed
    1  one or more required variables missing or invalid
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("sherloc_pipeline.web.config_check")

VALID_AUTH_MODES = {"auth0", "cf-access", "dev"}
VALID_ACCESS_MODES = {"internal", "public"}
VALID_PHASE_TIERS = {"team", "public"}


def _check_db_path(errors: List[str]) -> None:
    db = os.environ.get("SHERLOC_DB")
    if not db:
        errors.append("missing required variable: SHERLOC_DB")
        return

    # Q1 (v4.1.14): PHASE_DATABASE_PATH is an optional legacy alias.
    # docker-entrypoint.sh exports it from SHERLOC_DB when unset; if
    # the consumer sets both to different values, fail fast — alembic
    # reads PHASE_DATABASE_PATH while the app reads SHERLOC_DB, and a
    # silent divergence would migrate one DB while serving another.
    phase_db = os.environ.get("PHASE_DATABASE_PATH")
    if phase_db and phase_db != db:
        errors.append(
            f"PHASE_DATABASE_PATH and SHERLOC_DB differ "
            f"(PHASE_DATABASE_PATH={phase_db!r}, SHERLOC_DB={db!r}); "
            f"unset PHASE_DATABASE_PATH or set them to the same path"
        )

    if db == ":memory:":
        return

    path = Path(db)
    if path.exists():
        if not os.access(path, os.W_OK):
            errors.append(
                f"SHERLOC_DB path is not writable: {db} "
                f"(check container user/group ownership of bind mount)"
            )
        return

    parent = path.parent
    if not parent.exists():
        errors.append(
            f"SHERLOC_DB parent directory does not exist: {parent} "
            f"(bind mount target must be present before first start)"
        )
        return

    if not os.access(parent, os.W_OK):
        errors.append(
            f"SHERLOC_DB parent directory is not writable: {parent} "
            f"(check container user/group ownership of bind mount)"
        )


def _check_auth(errors: List[str]) -> None:
    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")
    if mode not in VALID_AUTH_MODES:
        errors.append(
            f"SHERLOC_AUTH_MODE has unrecognized value: {mode!r} "
            f"(expected one of {sorted(VALID_AUTH_MODES)})"
        )
        return

    if mode == "auth0":
        # Q2 (v4.1.14): SHERLOC_AUTH0_IDENTITY_CLAIM_URI is the namespace
        # URI auth.py uses to extract identity claims from the JWT. It
        # was previously required at first request via
        # web/auth.py::_require_env() — boot would succeed and the
        # container failed on first request. Boot-fail closes the
        # contract surface: a misconfigured container never reports
        # ready.
        for name in (
            "SHERLOC_AUTH0_DOMAIN",
            "SHERLOC_AUTH0_AUDIENCE",
            "SHERLOC_AUTH0_SPA_CLIENT_ID",
            "SHERLOC_AUTH0_IDENTITY_CLAIM_URI",
        ):
            if not os.environ.get(name):
                errors.append(f"missing required variable: {name}")
    elif mode == "cf-access":
        for name in ("SHERLOC_CF_TEAM_DOMAIN", "SHERLOC_CF_AUDIENCE"):
            if not os.environ.get(name):
                errors.append(f"missing required variable: {name}")


def _check_access_mode(errors: List[str]) -> None:
    mode = os.environ.get("SHERLOC_ACCESS_MODE", "internal")
    if mode not in VALID_ACCESS_MODES:
        errors.append(
            f"SHERLOC_ACCESS_MODE has unrecognized value: {mode!r} "
            f"(expected one of {sorted(VALID_ACCESS_MODES)})"
        )


def _check_r2(errors: List[str]) -> None:
    """Validate R2 storage env vars when running under auth0 (production) mode.

    Per m2020-phase platform spec §3.9.2, the SHERLOC backend reads ACI
    bytes from R2; the per-tier env file MUST set PHASE_TIER + AWS_*.
    A missing or mistyped PHASE_TIER will produce 500 'tier_unset' on
    every ACI request — fail loudly at boot instead.

    cf-access + dev modes skip this check (legacy parallel-run +
    unit-test paths).
    """
    if os.environ.get("SHERLOC_AUTH_MODE", "cf-access") != "auth0":
        return
    tier = os.environ.get("PHASE_TIER", "").strip().lower()
    if not tier:
        errors.append("missing required variable: PHASE_TIER")
    elif tier not in VALID_PHASE_TIERS:
        errors.append(
            f"PHASE_TIER has unrecognized value: {tier!r} "
            f"(expected one of {sorted(VALID_PHASE_TIERS)})"
        )
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ENDPOINT_URL",
    ):
        if not os.environ.get(name):
            errors.append(f"missing required variable: {name}")


def validate() -> List[str]:
    """Run all checks and return a list of error messages. Empty == OK."""
    errors: List[str] = []
    _check_db_path(errors)
    _check_auth(errors)
    _check_access_mode(errors)
    _check_r2(errors)
    return errors


def main() -> int:
    logging.basicConfig(level=os.environ.get("SHERLOC_LOG_LEVEL", "INFO"))
    errors = validate()
    if errors:
        for msg in errors:
            print(f"config_check: {msg}", file=sys.stderr)
        print(
            "config_check: aborting startup; fix the variables above "
            "in /etc/sherloc/<deployment>.env and restart.",
            file=sys.stderr,
        )
        return 1

    logger.info(
        "config_check: ok — db=%s, auth_mode=%s, access_mode=%s",
        os.environ.get("SHERLOC_DB"),
        os.environ.get("SHERLOC_AUTH_MODE", "cf-access"),
        os.environ.get("SHERLOC_ACCESS_MODE", "internal"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
