"""Shared R2 reader for v1.0-beta m2020-phase deployment.

Extracts the R2 client + per-tier config machinery from
``web/routes/images.py`` (v4.1.5–v4.1.8) into a shared module so non-image
consumers can reuse the same per-tier strip-prefix + bucket scoping +
error mapping. Current consumers:

- ``web/routes/images.py`` — ACI image bytes (existing since v4.1.5;
  refactored at v4.1.9 to import from this module).
- ``core/coordinates.py`` — Loupe-workspace ``spatial.csv`` / ``loupe.csv``
  (NEW at v4.1.9; unblocks ``/api/map/layers/<id>`` on the v1.0-beta VPS
  deployment per m2020-phase platform spec §3.9.8).

Contract: m2020-phase ``docs/PHASE_PLATFORM_v1.0_SPEC-revised.md`` §3.9
(ACI bytes) + §3.9.8 (companion files). Tier isolation is dual-enforced:
code-side (per-tier strip-prefix table + sherloc-aci/ re-root) +
credential-side (R2 bucket-scoped read tokens; slot 1 team / slot 3 public).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from fastapi import HTTPException

# Pure key-derivation primitives + tier-config constants live in core/ so
# core/coordinates.py can derive diagnostic R2 keys (spec §3.9.8.3) without
# violating the cli/->api/->services/->core/->models/ layering rule. These
# names are re-exported here for backward compat with existing importers
# (web/routes/{images,map}.py, tests/unit/web/test_r2_reader.py).
from sherloc_pipeline.core.r2_keys import (
    R2_KEY_PREFIX,
    TIER_TO_BUCKET,
    TIER_TO_STRIP_PREFIX,
    WORKSPACE_FILENAMES,
    derive_r2_key,
    get_strip_prefix_for_current_tier,
)
from sherloc_pipeline.core.r2_keys import (
    derive_workspace_key as _derive_workspace_key_core,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level cached client + tier config
# ---------------------------------------------------------------------------

_r2_client: Any = None
_r2_config: Optional[dict] = None
_r2_init_attempted = False
_r2_init_error: Optional[str] = None


def get_r2_client_and_config() -> tuple[Any, dict]:
    """Return ``(boto3 S3 client, tier config dict)`` — lazily initialized.

    Tier config: ``{"tier", "strip_prefix", "bucket"}``. Boto3 client is
    cached after first build. Tests inject via
    :func:`set_r2_client_for_tests`.

    Raises ``HTTPException(500, "tier_unset")`` if ``PHASE_TIER`` or the
    ``AWS_*`` env vars are not set (defense-in-depth; container startup
    config_check should reject this at boot in production).
    """
    global _r2_client, _r2_config, _r2_init_attempted, _r2_init_error
    if _r2_client is not None and _r2_config is not None:
        return _r2_client, _r2_config
    if _r2_init_attempted and _r2_init_error:
        raise HTTPException(status_code=500, detail=_r2_init_error)
    _r2_init_attempted = True

    tier = os.environ.get("PHASE_TIER", "").strip().lower()
    if tier not in TIER_TO_STRIP_PREFIX:
        _r2_init_error = "tier_unset"
        raise HTTPException(status_code=500, detail="tier_unset")

    aws_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_sec = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    region = os.environ.get("AWS_REGION", "auto")
    if not (aws_id and aws_sec and endpoint):
        _r2_init_error = "tier_unset"
        raise HTTPException(status_code=500, detail="tier_unset")

    import boto3
    from botocore.config import Config as BotoConfig

    _r2_client = boto3.client(
        "s3",
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_sec,
        endpoint_url=endpoint,
        region_name=region,
        config=BotoConfig(
            connect_timeout=5,
            read_timeout=10,
            retries={"max_attempts": 2, "mode": "standard"},
            signature_version="s3v4",
        ),
    )
    _r2_config = {
        "tier": tier,
        "strip_prefix": TIER_TO_STRIP_PREFIX[tier],
        "bucket": TIER_TO_BUCKET[tier],
    }
    return _r2_client, _r2_config


def is_r2_mode() -> bool:
    """Return True iff ``PHASE_TIER`` is set in the environment.

    Per m2020-phase spec §3.9.6 + §3.9.8.5 the legacy FS fallback path is
    permitted only when ``PHASE_TIER`` is unset (local dev runtime /
    ``main`` worktree v3.0.0). A
    container that has ``PHASE_TIER`` set is a production / staging
    deployment and MUST route through the R2 resolver.

    This predicate does NOT validate the tier value or check the AWS_*
    env vars — those are inspected by :func:`get_r2_client_and_config`
    at first call. If ``PHASE_TIER`` is set but invalid or paired with
    missing AWS credentials, callers will hit
    ``HTTPException(500, "tier_unset")`` from the inner R2 init per
    spec §3.9.4 — which is the spec-correct failure mode, not a silent
    fallback to local FS.

    Returns True even for invalid tier values (e.g., ``PHASE_TIER=bogus``
    or ``PHASE_TIER=""``) so the route layer routes the request into the
    R2 path where the ``tier_unset`` 500 response is generated. The
    predicate uses ``"PHASE_TIER" in os.environ`` rather than a
    truthiness check on the value, because an env file that exports
    ``PHASE_TIER=`` (empty string) is a production misconfiguration —
    spec §3.9.4 says "unset OR not in {team, public}" both fail as
    ``tier_unset``; treating empty string as "unset" would silently
    select FS fallback.
    """
    return "PHASE_TIER" in os.environ


def set_r2_client_for_tests(client: Any, tier: str) -> None:
    """Test-only: inject a pre-built boto3 (or moto-backed) S3 client + tier."""
    global _r2_client, _r2_config, _r2_init_attempted, _r2_init_error
    if tier not in TIER_TO_STRIP_PREFIX:
        raise ValueError(f"invalid tier: {tier!r}")
    _r2_client = client
    _r2_config = {
        "tier": tier,
        "strip_prefix": TIER_TO_STRIP_PREFIX[tier],
        "bucket": TIER_TO_BUCKET[tier],
    }
    _r2_init_attempted = True
    _r2_init_error = None


def reset_r2_client_for_tests() -> None:
    """Test-only: clear cached client + tier config so next request re-inits."""
    global _r2_client, _r2_config, _r2_init_attempted, _r2_init_error
    _r2_client = None
    _r2_config = None
    _r2_init_attempted = False
    _r2_init_error = None


# ---------------------------------------------------------------------------
# R2 GET / HEAD primitives (spec §3.9.3 + §3.9.4). Key derivation lives in
# core/r2_keys.py and is re-exported above for back-compat.
# ---------------------------------------------------------------------------

def r2_get_bytes(key: str) -> bytes:
    """GET an object from R2; map S3 errors to ``HTTPException`` per spec §3.9.4.

    Status mapping (per the spec §3.9.4 table):

    - R2 404 / NoSuchKey            → 404 ``not_found``
    - R2 403 / AccessDenied         → 502 ``upstream_credential_error``
    - boto3 timeout (connect/read)  → 504 ``upstream_timeout``
    - any other boto3/botocore error → 500 ``upstream_error``
    """
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        ConnectTimeoutError,
        ReadTimeoutError,
    )

    client, cfg = get_r2_client_and_config()
    try:
        resp = client.get_object(Bucket=cfg["bucket"], Key=key)
        return resp["Body"].read()
    except ClientError as exc:
        err = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = err.get("Code", "")
        status = (
            exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if hasattr(exc, "response")
            else None
        )
        if code in ("NoSuchKey", "404") or status == 404:
            raise HTTPException(status_code=404, detail="not_found") from exc
        if code in ("AccessDenied", "403", "Forbidden") or status == 403:
            # Signals cross-tier credential misconfiguration; do NOT leak
            # the underlying credential identity to the response body.
            raise HTTPException(
                status_code=502, detail="upstream_credential_error"
            ) from exc
        logger.exception("R2 GET unexpected ClientError for key=%s", key)
        raise HTTPException(status_code=500, detail="upstream_error") from exc
    except (ReadTimeoutError, ConnectTimeoutError) as exc:
        logger.warning("R2 GET timeout for key=%s", key)
        raise HTTPException(status_code=504, detail="upstream_timeout") from exc
    except BotoCoreError as exc:
        logger.exception("R2 GET network/SDK error for key=%s", key)
        raise HTTPException(status_code=500, detail="upstream_error") from exc


def r2_head_exists(key: str) -> bool:
    """HEAD probe; True if key exists, False on 404. Other errors propagate as False."""
    from botocore.exceptions import BotoCoreError, ClientError

    client, cfg = get_r2_client_and_config()
    try:
        client.head_object(Bucket=cfg["bucket"], Key=key)
        return True
    except ClientError as exc:
        err = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
        code = err.get("Code", "")
        status = (
            exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if hasattr(exc, "response")
            else None
        )
        if code in ("404", "NoSuchKey", "Not Found", "NotFound") or status == 404:
            return False
        # 403 here likely means cross-tier credential mismatch; treat as
        # "not available" rather than 502 (opportunistic lookup).
        logger.warning("R2 HEAD unexpected error for key=%s code=%s", key, code)
        return False
    except BotoCoreError:
        logger.warning("R2 HEAD network/SDK error for key=%s", key, exc_info=True)
        return False


def find_colorized_key(base_key: str) -> Optional[str]:
    """Substitute ``sol_NNNN → sol_NNNN_colorized`` in the first matching segment.

    Returns the colorized key iff R2 HEAD finds it; else None. R2 analog
    of the legacy ``_find_colorized_path`` which probed the local FS for
    sibling ``sol_NNNN_colorized/`` directories.
    """
    parts = base_key.split("/")
    for i, part in enumerate(parts):
        if re.match(r"sol_\d+$", part):
            candidate_parts = list(parts)
            candidate_parts[i] = f"{part}_colorized"
            candidate = "/".join(candidate_parts)
            if r2_head_exists(candidate):
                return candidate
    return None


def colorized_variant_exists(file_path: str) -> bool:
    """Public predicate: True if R2 has a ``sol_NNNN_colorized/`` variant.

    Used by ``routes/map.py`` to decide whether to advertise the
    ``aci_colorized`` URL in the layer list. Returns False on any
    R2-side error (the result is opportunistic — a missing variant
    just hides the URL).
    """
    if not file_path or file_path.startswith("pds:"):
        return False
    try:
        _, cfg = get_r2_client_and_config()
        base_key = derive_r2_key(
            file_path, cfg["strip_prefix"], active_tier=cfg.get("tier")
        )
    except HTTPException:
        return False
    return find_colorized_key(base_key) is not None


# ---------------------------------------------------------------------------
# High-level fetchers (spec §3.9.3 + §3.9.8.2)
# ---------------------------------------------------------------------------

def get_working_file(file_path: str, filename: str) -> bytes:
    """Fetch a Loupe-workspace companion file from R2 (spec §3.9.8).

    The Loupe workspace directory is ``Path(file_path).parent.parent``
    (drops the ``img/<aci-product>.{PNG,IMG}`` suffix). The R2 key is
    constructed by :func:`derive_workspace_key` (re-exported from
    :mod:`core.r2_keys`) using the per-tier strip-prefix from the cached
    R2 config.

    Args:
        file_path: ``ContextImageORM.file_path`` value (absolute path to
            the ACI product under the per-tier strip-prefix root).
        filename: companion-file name within the workspace directory.
            MUST be one of :data:`WORKSPACE_FILENAMES` (``spatial.csv``
            or ``loupe.csv``) — anything else is rejected as
            ``misconfigured_path`` BEFORE any R2 call.

    Returns:
        Raw bytes of the requested file.

    Raises:
        HTTPException per spec §3.9.4 + §3.9.8.3 failure-mode table.
        Notable rows:
        - 500 ``misconfigured_path`` if ``filename`` is not in
          :data:`WORKSPACE_FILENAMES`, if ``file_path`` does not begin
          with the per-tier strip prefix, or if the derived key fails
          the path-traversal guard.
        - 404 ``not_found`` if R2 has no object at the derived key.
        - 502 ``upstream_credential_error`` on R2 403 (cross-tier creds).
    """
    key = derive_workspace_key(file_path, filename)
    return r2_get_bytes(key)


def derive_workspace_key(file_path: str, filename: str) -> str:
    """Backward-compat wrapper around :func:`core.r2_keys.derive_workspace_key`.

    Uses the per-tier strip-prefix from the cached R2 config (initialized
    via :func:`get_r2_client_and_config` or
    :func:`set_r2_client_for_tests`). This matches the v4.1.9 surface
    that ``web/routes/{images,map}.py`` and ``tests/unit/web/test_r2_reader.py``
    target — they don't pass ``strip_prefix`` explicitly.

    :mod:`core.coordinates` calls ``core.r2_keys.derive_workspace_key``
    directly (without going through this web-layer wrapper) so it can
    derive a diagnostic key in the R2-404 fallback without violating
    layering. That code path uses the env-based ``PHASE_TIER`` lookup
    (no cached cfg available in core/).
    """
    if filename not in WORKSPACE_FILENAMES:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    _, cfg = get_r2_client_and_config()
    return _derive_workspace_key_core(
        file_path,
        filename,
        strip_prefix=cfg["strip_prefix"],
        active_tier=cfg.get("tier"),
    )
