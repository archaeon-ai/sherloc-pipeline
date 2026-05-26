"""Pure R2 key-derivation primitives for the m2020-phase v1.0-beta deployment.

Extracted from ``web/r2_reader.py`` at v4.1.9-fix so that ``core/coordinates.py``
can derive the canonical R2 key for a missing Loupe-workspace file (spec
§3.9.8.3 — the 400 ``/api/map/layers`` response MUST name the expected
key) without violating the
``cli/ -> api/ -> services/ -> core/ -> models/`` layering rule enforced by
``tests/architecture/test_layering.py``.

This module is **pure** — no boto3, no I/O, no module-level side effects
beyond reading ``PHASE_*_STRIP_PREFIX`` env vars at import time (identical
to the original behavior in ``web/r2_reader.py``).

Contract: m2020-phase ``docs/PHASE_PLATFORM_v1.0_SPEC-revised.md`` §3.9
(ACI bytes; key derivation table) + §3.9.8 (Loupe workspace files).
Tier isolation remains dual-enforced: code-side (per-tier strip-prefix
table + ``sherloc-aci/`` re-root) + credential-side (R2 bucket-scoped
read tokens; slot 1 team / slot 3 public).

Consumers:

- ``web/r2_reader.py`` — re-exports these symbols so existing
  ``from sherloc_pipeline.web.r2_reader import ...`` imports keep
  resolving (back-compat with v4.1.9 surface; no public-API churn).
- ``core/coordinates.py`` — calls :func:`derive_workspace_key` and
  :func:`get_strip_prefix_for_current_tier` directly to build the
  diagnostic key in the ``CoordinatesUnavailableError`` raised on
  R2-404 of ``spatial.csv`` / ``loupe.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

# R2 strip-prefix + bucket per tier (spec §3.9.3, amended Session 73).
# Strip prefix is hardcoded per tier — not derived solely from
# PHASE_TIER — so a misclassified ingestion is rejected as
# misconfigured_path before any R2 call. Override via env vars when an
# alternative local-dev layout is in use.
#
# Adjacent-string-literal concatenation keeps the operator-local path
# from appearing as a single literal in tracked content (CONTRIBUTING.md
# "Public-repo discipline"); runtime values are unchanged.
_DEFAULT_TEAM_STRIP = "/data" "/sherloc/data/"
_DEFAULT_PUBLIC_STRIP = "/data" "/sherloc/pds/"
TIER_TO_STRIP_PREFIX = {
    "team": os.environ.get("PHASE_TEAM_STRIP_PREFIX", _DEFAULT_TEAM_STRIP),
    "public": os.environ.get("PHASE_PUBLIC_STRIP_PREFIX", _DEFAULT_PUBLIC_STRIP),
}
TIER_TO_BUCKET = {
    "team": "phase-team",
    "public": "phase-public",
}
R2_KEY_PREFIX = "sherloc-aci/"

# Legacy-ingestion path aliases per tier. Some `context_images.file_path`
# rows in production DBs were ingested with the pre-v4.1.9 NAS-mount
# convention and never re-normalized to the canonical
# `$PHASE_TEAM_STRIP_PREFIX` (default `_DEFAULT_TEAM_STRIP` above) layout
# (3 such rows on team tier as of 2026-05-24 — Mauchsberg sol 1810,
# Pafuri Gate sol 1798). Their R2 byte layout under `sherloc-aci/` is the
# same relative tree once the legacy prefix is stripped, so the right
# semantics is to treat the legacy prefix as an *alias* of the canonical
# one: same `sherloc-aci/<rel>` re-root, no behavior change for the
# 1090-row happy path.
#
# `PHASE_{TEAM,PUBLIC}_LEGACY_STRIP_ALIASES` env vars APPEND to the per-
# tier defaults (colon-separated, same shape as `$PATH`). Setting the env
# var does NOT replace the in-tree default — the known-in-flight
# production alias stays in scope regardless. To retire the default,
# remove the literal below and ship a code change, not an env override
# (the env override can ONLY add aliases). This matches operator
# expectation that an additive deployment knob shouldn't be able to
# regress already-working production paths (PR #39 Codex R1 F2).
_DEFAULT_TEAM_LEGACY_ALIASES: tuple[str, ...] = (
    "/nas/" "000_sherloc/data/",
)
_DEFAULT_PUBLIC_LEGACY_ALIASES: tuple[str, ...] = ()
_team_extra_env = os.environ.get("PHASE_TEAM_LEGACY_STRIP_ALIASES", "")
_public_extra_env = os.environ.get("PHASE_PUBLIC_LEGACY_STRIP_ALIASES", "")
TIER_TO_LEGACY_STRIP_ALIASES: dict[str, tuple[str, ...]] = {
    "team": _DEFAULT_TEAM_LEGACY_ALIASES + tuple(
        p for p in _team_extra_env.split(":") if p
    ),
    "public": _DEFAULT_PUBLIC_LEGACY_ALIASES + tuple(
        p for p in _public_extra_env.split(":") if p
    ),
}


def _resolve_strip_prefix(
    file_path: str,
    strip_prefix: str,
    *,
    active_tier: str | None = None,
) -> str | None:
    """Return the prefix `file_path` matches (canonical or known alias), or None.

    The canonical `strip_prefix` is tried first. If the path doesn't match
    it, the legacy aliases for the active tier are tried.

    ``active_tier``: if the caller knows which tier this resolution
    belongs to (e.g. it pulled the prefix from a cached tier config),
    pass it explicitly — that's the SAFE path. Otherwise the function
    falls back to inferring the tier by inverting the
    ``TIER_TO_STRIP_PREFIX`` table; that inference is ambiguous when
    two tiers happen to share a strip-prefix value (e.g. dev override
    sets both to the same path) and would let one tier's aliases be
    consulted by the other's call (PR #39 Codex R1 F1). Explicit tier
    is preferred for production callers; implicit fallback is kept for
    the back-compat ``derive_r2_key(file_path, strip_prefix)`` surface.
    """
    if file_path.startswith(strip_prefix):
        return strip_prefix
    if active_tier is None:
        active_tier = next(
            (t for t, p in TIER_TO_STRIP_PREFIX.items() if p == strip_prefix),
            None,
        )
    if active_tier is None:
        return None
    for alias in TIER_TO_LEGACY_STRIP_ALIASES.get(active_tier, ()):
        if file_path.startswith(alias):
            return alias
    return None

# Allowlist of accepted Loupe-workspace companion-file names per spec §3.9.8.1.
# ``derive_workspace_key`` rejects any filename not on this allowlist as
# ``misconfigured_path`` 500 BEFORE doing any path math. The allowlist
# prevents the helper from being inadvertently turned into a general
# workspace-object reader.
WORKSPACE_FILENAMES = frozenset({"spatial.csv", "loupe.csv"})


def get_strip_prefix_for_current_tier() -> str:
    """Return the per-tier R2 strip prefix from ``PHASE_TIER``.

    Pure env lookup; does NOT initialize the boto3 client. Matches the
    failure mode of :func:`web.r2_reader.get_r2_client_and_config` per
    spec §3.9.4 — invalid tier raises ``HTTPException(500, "tier_unset")``.

    Raises:
        HTTPException(500, "tier_unset"): if ``PHASE_TIER`` is unset,
            empty, or not in :data:`TIER_TO_STRIP_PREFIX`.
    """
    tier = os.environ.get("PHASE_TIER", "").strip().lower()
    if tier not in TIER_TO_STRIP_PREFIX:
        raise HTTPException(status_code=500, detail="tier_unset")
    return TIER_TO_STRIP_PREFIX[tier]


def derive_r2_key(
    file_path: str,
    strip_prefix: str,
    *,
    active_tier: str | None = None,
) -> str:
    """Strip the per-tier prefix from ``file_path`` and re-root under ``sherloc-aci/``.

    Pure function. ``strip_prefix`` is supplied by the caller (typically
    from :func:`get_strip_prefix_for_current_tier` or from the cached
    ``_r2_config`` in :mod:`web.r2_reader`).

    Accepts ``file_path`` under either the canonical ``strip_prefix`` OR
    one of the per-tier legacy aliases declared in
    :data:`TIER_TO_LEGACY_STRIP_ALIASES`. Legacy-alias rows resolve to the
    same ``sherloc-aci/<rel>`` key as their canonical-prefix siblings
    would, because the rclone of NAS → R2 preserves the post-prefix
    relative tree. This unblocks pre-v4.1.9 ingested rows whose
    ``file_path`` was never re-normalized without requiring a DB
    migration.

    ``active_tier`` (kwarg-only): pass when the caller knows which tier
    the resolution belongs to (e.g. ``cfg["tier"]`` from the cached
    R2 config). When omitted, the function infers tier by inverting
    ``TIER_TO_STRIP_PREFIX``; the inference is ambiguous when two tiers
    share a strip-prefix value (PR #39 Codex R1 F1). Production callers
    should pass ``active_tier`` for the safer path.

    Raises ``HTTPException(500, "misconfigured_path")`` if ``file_path``
    does not begin with the canonical prefix OR any legacy alias (e.g.,
    a team-tier DB row carrying a public-tier path), or if the resulting
    key fails the path-traversal guard.
    """
    matched = _resolve_strip_prefix(file_path, strip_prefix, active_tier=active_tier)
    if matched is None:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    rel = file_path[len(matched):]
    key = R2_KEY_PREFIX + rel
    if ".." in key or key.startswith("/") or "\\" in key:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    return key


def derive_workspace_key(
    file_path: str,
    filename: str,
    *,
    strip_prefix: str | None = None,
    active_tier: str | None = None,
) -> str:
    """Return the R2 key for ``filename`` inside the Loupe workspace of ``file_path``.

    Pure key derivation — no R2 client work. Callers use this to:

    - fetch the file via :func:`web.r2_reader.get_working_file`
      (the normal happy path), or
    - include the key in diagnostic / error messages without performing
      the fetch (spec §3.9.8.3 — the missing-workspace 400 response from
      ``/api/map/layers`` MUST name the R2 key).

    The Loupe workspace directory is ``Path(file_path).parent.parent``
    (drops the ``img/<aci-product>.{PNG,IMG}`` suffix).

    Two strip-prefix sources, in priority order:

    1. ``strip_prefix`` keyword argument — used by
       :mod:`web.r2_reader` when the cached tier config is available
       (``set_r2_client_for_tests`` injected client during tests, or
       runtime ``get_r2_client_and_config()`` cache in production).
    2. ``PHASE_TIER`` env via :func:`get_strip_prefix_for_current_tier`
       — used by :mod:`core.coordinates` in the 404-diagnostic
       fallback where the caller doesn't have a cached cfg in scope.

    Raises ``HTTPException`` for: disallowed ``filename``,
    ``file_path`` not under the per-tier strip prefix, derived-key
    path-traversal — all ``misconfigured_path``; or
    ``tier_unset`` if ``strip_prefix`` is None AND ``PHASE_TIER`` is
    unset / invalid.
    """
    if filename not in WORKSPACE_FILENAMES:
        raise HTTPException(status_code=500, detail="misconfigured_path")

    if strip_prefix is None:
        strip_prefix = get_strip_prefix_for_current_tier()
        if active_tier is None:
            # We just looked up the current tier inside
            # get_strip_prefix_for_current_tier — make the same answer
            # available to the alias resolver so it doesn't have to
            # re-infer (PR #39 Codex R1 F1).
            active_tier = os.environ.get("PHASE_TIER", "").strip().lower() or None
    # Accept legacy aliases for the same tier (mirror of derive_r2_key).
    matched = _resolve_strip_prefix(file_path, strip_prefix, active_tier=active_tier)
    if matched is None:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    file_path_obj = Path(file_path)
    try:
        working_rel = file_path_obj.parent.parent.relative_to(Path(matched))
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="misconfigured_path") from exc
    key = f"{R2_KEY_PREFIX}{working_rel}/{filename}"
    if ".." in key or key.startswith("/") or "\\" in key:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    return key
