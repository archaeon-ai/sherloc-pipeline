"""Pure R2 key-derivation primitives for the m2020-phase v1.0-beta deployment.

The object identity of a context image is its **relative locator**
(``context_images.r2_rel_key``): a POSIX-style relative path that is the
image's position in the tier's data tree. Physical locations derive from
it at the edges:

- R2 key   = ``sherloc-aci/`` + locator (:func:`derive_r2_key`)
- disk path = ``<data_root>/`` + locator (processing side; deployment
  config, not this module)

Unresolved PDS references carry the ``pds:<lidvid>`` scheme in the same
column (one scheme in the relative-locator family); they have no R2 key
until the download step resolves them, so :func:`derive_r2_key` rejects
them exactly like the pre-locator serve path did.

This module is **pure** — no boto3, no I/O, no env reads. That also
makes :func:`derive_workspace_key` usable from ``core/coordinates.py``
for the R2-404 diagnostic key (spec §3.9.8.3) without violating the
``cli/ -> api/ -> services/ -> core/ -> models/`` layering rule enforced
by ``tests/architecture/test_layering.py``.

History: through v5.3.x, ``file_path`` stored a machine-specific
absolute path and this module carried a per-tier strip-prefix table,
``PHASE_*_STRIP_PREFIX`` env fallbacks, legacy-ingestion aliases, and
tier inference to undo that at serve time. The stored-locator model
replaces all of it; the one-time translation now lives in the Alembic
backfill migration.

Contract: m2020-phase platform spec §3.9 (ACI bytes; key derivation)
+ §3.9.8 (Loupe workspace files). Tier isolation is credential-side
(R2 bucket-scoped read tokens; per-tier team/public credentials) plus
per-tier databases — the locator itself is tier-neutral.
"""

from __future__ import annotations

import re
from pathlib import PurePath, PurePosixPath

from fastapi import HTTPException

R2_KEY_PREFIX = "sherloc-aci/"

# Matches a bare ``sol_NNNN`` path segment (the colorized variant appends
# ``_colorized``: ``sol_1213`` → ``sol_1213_colorized``).
_SOL_SEGMENT_RE = re.compile(r"^sol_\d+$")

# Matches ``sol_NNNN`` or ``sol_NNNN_colorized`` when anchoring a
# relative locator inside an absolute ingestion path.
_SOL_ANCHOR_RE = re.compile(r"^sol_\d+(_colorized)?$")


def colorize_sol_segment(path: str) -> str | None:
    """Swap the first ``sol_NNNN`` path segment to ``sol_NNNN_colorized``.

    Pure string transform on a ``/``-delimited path (an R2 key *or* a
    relative locator — both carry the ``sol_NNNN`` workspace segment).
    Returns the rewritten path, or ``None`` when no bare ``sol_NNNN``
    segment is present (so callers can treat "no colorized variant
    derivable" distinctly from a successful swap).

    The colorized Loupe workspace mirrors the grayscale tree exactly with
    only this one segment renamed, so the same swap derives both the
    colorized image key (see :func:`web.r2_reader.find_colorized_key`) and
    the colorized ``spatial.csv`` / ``loupe.csv`` workspace locator (see
    :func:`core.coordinates._resolve_scanner_workspace`).
    """
    parts = path.split("/")
    for i, part in enumerate(parts):
        if _SOL_SEGMENT_RE.match(part):
            parts[i] = f"{part}_colorized"
            return "/".join(parts)
    return None


# Allowlist of accepted Loupe-workspace companion-file names per spec §3.9.8.1.
# ``derive_workspace_key`` rejects any filename not on this allowlist as
# ``misconfigured_path`` 500 BEFORE doing any path math. The allowlist
# prevents the helper from being inadvertently turned into a general
# workspace-object reader.
WORKSPACE_FILENAMES = frozenset({"spatial.csv", "loupe.csv"})


def _validate_key(key: str) -> str:
    """Shared path-traversal guard for derived R2 keys."""
    if ".." in key or key.startswith("/") or "\\" in key:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    return key


def derive_r2_key(rel_locator: str | None) -> str:
    """Return the R2 object key for a stored relative locator.

    Single concatenation: ``sherloc-aci/`` + locator. Raises
    ``HTTPException(500, "misconfigured_path")`` when the locator is
    missing (row predates the backfill or matched no known layout),
    carries the unresolved ``pds:`` scheme (broken ingestion at serve
    time, same rejection as pre-locator behavior), is absolute, or fails
    the path-traversal guard.
    """
    if (
        not rel_locator
        or rel_locator.startswith("pds:")
        or rel_locator.startswith("/")
    ):
        raise HTTPException(status_code=500, detail="misconfigured_path")
    return _validate_key(R2_KEY_PREFIX + rel_locator)


def derive_workspace_key(rel_locator: str | None, filename: str) -> str:
    """Return the R2 key for ``filename`` inside the Loupe workspace of a locator.

    Pure key derivation — no R2 client work. Callers use this to:

    - fetch the file via :func:`web.r2_reader.get_working_file`
      (the normal happy path), or
    - include the key in diagnostic / error messages without performing
      the fetch (spec §3.9.8.3 — the missing-workspace 400 response from
      ``/api/map/layers`` MUST name the R2 key).

    The Loupe workspace directory is the locator's grandparent
    (drops the ``img/<aci-product>.{PNG,IMG}`` suffix).

    Raises ``HTTPException(500, "misconfigured_path")`` for: disallowed
    ``filename``, a missing / ``pds:``-schemed / absolute locator, a
    locator too shallow to contain a ``<workspace>/img/<file>`` tree, or
    a derived key failing the path-traversal guard.
    """
    if filename not in WORKSPACE_FILENAMES:
        raise HTTPException(status_code=500, detail="misconfigured_path")
    if (
        not rel_locator
        or rel_locator.startswith("pds:")
        or rel_locator.startswith("/")
    ):
        raise HTTPException(status_code=500, detail="misconfigured_path")
    working_rel = PurePosixPath(rel_locator).parent.parent
    if str(working_rel) in (".", "/", ""):
        raise HTTPException(status_code=500, detail="misconfigured_path")
    return _validate_key(f"{R2_KEY_PREFIX}{working_rel}/{filename}")


def derive_rel_locator(file_path: str | PurePath) -> str | None:
    """Derive the relative locator from an absolute ingestion path, or None.

    Structural anchor on the ``sol_NNNN`` segment — the R2 tree layout is
    a platform convention, so the locator is derivable from the path
    shape regardless of which machine or mount the ingestion read from:

    - Loupe workspace tree (team tier):
      ``…/loupe/sol_NNNN/<scan>/<workspace>/img/<file>`` →
      ``loupe/sol_NNNN/<scan>/<workspace>/img/<file>``
    - PDS ACI cache tree (public tier):
      ``…/sol_NNNN/data_aci/<file>`` → ``sol_NNNN/data_aci/<file>``

    Returns ``None`` when the path matches neither convention (the row
    then has no derivable R2 identity; serving it fails the same way an
    unrecognized absolute path always has), or when the derived locator
    would fail :func:`derive_r2_key`'s traversal guard — a locator that
    cannot serve must not be persisted.
    """
    parts = PurePath(file_path).parts
    for i, part in enumerate(parts):
        if not _SOL_ANCHOR_RE.match(part):
            continue
        if i > 0 and parts[i - 1] == "loupe":
            rel = "/".join(("loupe", *parts[i:]))
        elif i + 1 < len(parts) and parts[i + 1] == "data_aci":
            rel = "/".join(parts[i:])
        else:
            continue
        # Mirror _validate_key: never persist a locator serving would
        # reject (same conservative substring semantics).
        if ".." in rel or "\\" in rel:
            return None
        return rel
    return None
