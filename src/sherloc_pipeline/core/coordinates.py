"""Coordinate frame resolution for Map Mode spatial overlays.

Resolves scan point coordinates to ACI pixel space for rendering,
with persistent caching in the map_display_coordinates table.

Two coordinate frames exist in the database:

- ``aci_pixel`` (PDS scans): x_pixel/y_pixel are already ACI image pixel
  coordinates. No transform needed; values are copied directly.

- ``scanner_workspace`` (Loupe scans): x/y are in scanner-relative units
  (±0.5 for detail scans, ±2.5 for surveys). Must be converted to ACI pixels
  via the Loupe calibration embedded in ``spatial.csv`` / ``loupe.csv``.

Results are written to ``map_display_coordinates`` on first call and reused
on subsequent calls (cache-aside pattern). Pass ``force_recompute=True`` to
invalidate the cache and recompute.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Type alias: a Loupe-workspace companion-file reader.
# Signature: ``r2_reader(file_path: str, filename: str) -> bytes``.
# Production implementation: ``sherloc_pipeline.web.r2_reader.get_working_file``.
# Tests inject a moto-backed callable. Legacy local-filesystem dev
# installs (no R2) pass None and fall back to local-FS reads.
#
# NOTE: the diagnostic re-derivation of the missing key on 404 imports
# from ``sherloc_pipeline.core.r2_keys`` (not ``web.r2_reader``) to
# preserve the cli/->api/->services/->core/->models/ layering rule
# enforced by ``tests/architecture/test_layering.py``.
WorkspaceReader = Callable[[str, str], bytes]


@dataclass
class DisplayCoordinate:
    """A single point's resolved ACI pixel coordinates."""

    scan_point_id: str
    point_index: int
    aci_x: float
    aci_y: float
    transform_method: str  # 'identity' | 'scanner_calibration'


class CoordinatesUnavailableError(Exception):
    """Raised when coordinates cannot be resolved for a scan.

    The message includes a human-readable reason to surface to the caller
    (e.g. missing spatial.csv, unknown coordinate frame, no scan points found).
    """


def resolve_display_coordinates(
    session: Session,
    scan_id: str,
    *,
    force_recompute: bool = False,
    workspace_reader: Optional[WorkspaceReader] = None,
    colorized: bool = False,
) -> list[DisplayCoordinate]:
    """Resolve ACI pixel coordinates for all points in a scan.

    Strategy:
    1. Check ``map_display_coordinates`` cache (unless *force_recompute*).
    2. If ``coordinate_frame == 'aci_pixel'``: identity — copy x_pixel/y_pixel
       directly as aci_x/aci_y.
    3. If ``coordinate_frame == 'scanner_workspace'``: find the Loupe workspace
       directory from ``ContextImageORM``, call ``load_spatial_table()``, and
       use xPix/yPix from the spatial table.
    4. If coordinate_frame is null or unrecognised: raise
       :class:`CoordinatesUnavailableError`.

    Computed results are written to ``map_display_coordinates`` before
    returning.  Existing cache rows for the scan (of the requested
    ``colorized`` variant) are deleted first when *force_recompute* is
    ``True``.

    Parameters
    ----------
    session:
        Active SQLAlchemy session.
    scan_id:
        UUID string of the scan whose points to resolve.
    force_recompute:
        When ``True``, delete any existing cache rows and recompute.
    workspace_reader:
        Optional callable ``(file_path, filename) -> bytes`` used to fetch
        Loupe workspace companion files (``spatial.csv``, ``loupe.csv``)
        from a non-FS source (v1.0-beta production: R2 via
        ``web/r2_reader.get_working_file``; tests: moto-backed mock).
        When ``None``, the legacy local-FS read path is used
        (local-filesystem dev installs without R2; production containers
        have NO local SHERLOC data mount and always supply an R2-backed
        reader).
    colorized:
        When ``True``, resolve coordinates against the **colorized** Loupe
        workspace (``sol_NNNN_colorized/``) instead of the grayscale base.
        The colorized ACI PNG is a pure crop of the grayscale image, and
        the colorized workspace ships its own ``spatial.csv`` / ``loupe.csv``
        whose per-point coordinates are re-solved for the cropped frame.
        Resolving against that dataset keeps the scan-point overlay
        registered when the web UI toggles Colorized (issue #8). Only
        ``scanner_workspace`` scans have a colorized variant; requesting
        ``colorized=True`` for an ``aci_pixel`` (PDS) scan raises
        :class:`CoordinatesUnavailableError`. Cached separately from the
        grayscale result (cache key ``(scan_point_id, colorized)``).

    Returns
    -------
    list[DisplayCoordinate]
        One entry per scan point, sorted by point_index.

    Raises
    ------
    CoordinatesUnavailableError
        If there are no scan points, the coordinate frame is unknown, the
        scanner workspace files cannot be found/read, or ``colorized=True``
        was requested for a scan with no colorized workspace.
    """
    from sherloc_pipeline.database.models import (
        ContextImageORM,
        MapDisplayCoordinateORM,
        ScanPointORM,
    )

    # ------------------------------------------------------------------
    # 1. Cache check
    # ------------------------------------------------------------------
    if not force_recompute:
        cached = _load_from_cache(session, scan_id, colorized=colorized)
        if cached:
            logger.debug(
                "resolve_display_coordinates: cache hit for scan %s "
                "(%d points, colorized=%s)",
                scan_id,
                len(cached),
                colorized,
            )
            return cached

    # ------------------------------------------------------------------
    # 2. Load scan points
    # ------------------------------------------------------------------
    points = (
        session.query(ScanPointORM)
        .filter(ScanPointORM.scan_id == scan_id)
        .order_by(ScanPointORM.point_index)
        .all()
    )
    if not points:
        raise CoordinatesUnavailableError(
            f"No scan points found for scan_id={scan_id!r}"
        )

    # Determine coordinate frame from first point (all points in a scan share
    # the same frame — they come from a single Loupe or PDS ingestion).
    frame = points[0].coordinate_frame  # raw string or None

    # ------------------------------------------------------------------
    # 3. Resolve coordinates
    # ------------------------------------------------------------------
    if frame == "aci_pixel":
        if colorized:
            # PDS aci_pixel scans have no colorized ACI variant (and no
            # Loupe workspace to re-solve against). The route layer gates
            # colorized requests on `colorized_variant_exists`, which is
            # False for these, so this is a defensive guard.
            raise CoordinatesUnavailableError(
                f"Scan {scan_id!r} uses the 'aci_pixel' coordinate frame, "
                "which has no colorized workspace variant."
            )
        coords = _resolve_identity(points)
    elif frame == "scanner_workspace":
        coords = _resolve_scanner_workspace(
            session,
            scan_id,
            points,
            workspace_reader=workspace_reader,
            colorized=colorized,
        )
    else:
        raise CoordinatesUnavailableError(
            f"Cannot resolve coordinates for scan {scan_id!r}: "
            f"coordinate_frame={frame!r} is not supported. "
            "Expected 'aci_pixel' or 'scanner_workspace'."
        )

    # ------------------------------------------------------------------
    # 4. Write to cache (delete-then-insert for force_recompute safety)
    # ------------------------------------------------------------------
    _write_cache(
        session, scan_id, coords, colorized=colorized, force_recompute=force_recompute
    )

    return coords


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_from_cache(
    session: Session, scan_id: str, *, colorized: bool = False
) -> list[DisplayCoordinate]:
    """Return cached DisplayCoordinate objects for the scan variant, or [] if none."""
    from sherloc_pipeline.database.models import MapDisplayCoordinateORM, ScanPointORM

    rows = (
        session.query(MapDisplayCoordinateORM, ScanPointORM.point_index)
        .join(ScanPointORM, ScanPointORM.id == MapDisplayCoordinateORM.scan_point_id)
        .filter(
            ScanPointORM.scan_id == scan_id,
            MapDisplayCoordinateORM.colorized == colorized,
        )
        .order_by(ScanPointORM.point_index)
        .all()
    )
    return [
        DisplayCoordinate(
            scan_point_id=row.scan_point_id,
            point_index=point_index,
            aci_x=row.aci_x,
            aci_y=row.aci_y,
            transform_method=row.transform_method,
        )
        for row, point_index in rows
    ]


def _resolve_identity(points: list) -> list[DisplayCoordinate]:
    """Identity transform: x_pixel/y_pixel are already ACI pixel coordinates."""
    missing = [p for p in points if p.x_pixel is None or p.y_pixel is None]
    if missing:
        raise CoordinatesUnavailableError(
            f"{len(missing)} scan point(s) have coordinate_frame='aci_pixel' but "
            "null x_pixel/y_pixel values (e.g. point_index="
            f"{missing[0].point_index}). Cannot produce display coordinates."
        )
    return [
        DisplayCoordinate(
            scan_point_id=pt.id,
            point_index=pt.point_index,
            aci_x=float(pt.x_pixel),
            aci_y=float(pt.y_pixel),
            transform_method="identity",
        )
        for pt in points
    ]


def _fetch_workspace_file(
    workspace_reader: WorkspaceReader,
    file_path: str,
    filename: str,
    scan_id: str,
) -> bytes:
    """Fetch one Loupe-workspace companion file via ``workspace_reader``.

    Wraps R2-404 errors as :class:`CoordinatesUnavailableError` with a
    message that names the specific missing file's R2 key per spec
    §3.9.8.3. Non-404 HTTPException-s propagate unchanged so the route
    layer preserves their status (502 / 504 / 500 per §3.9.4).
    """
    from fastapi import HTTPException

    try:
        return workspace_reader(file_path, filename)
    except HTTPException as exc:
        if exc.status_code == 404:
            # Diagnostic re-derivation per spec §3.9.8.3.
            # Imports from ``core.r2_keys`` — NOT ``web.r2_reader`` — so the
            # cli/->api/->services/->core/->models/ layering rule holds
            # (tests/architecture/test_layering.py). When the env-based tier
            # lookup fails (e.g., production tier config disappeared mid-flight
            # or this is the FS-fallback path), fall back to a placeholder so
            # the error message still has actionable context.
            from sherloc_pipeline.core.r2_keys import derive_workspace_key
            try:
                key = derive_workspace_key(file_path, filename)
            except HTTPException:
                key = f"<key-derivation-failed for {filename!r}>"
            raise CoordinatesUnavailableError(
                f"Loupe workspace file not found in R2 for scan {scan_id!r} "
                f"(file_path={file_path!r}, missing_file={filename!r}, "
                f"expected_key={key!r}, "
                f"upstream_detail={exc.detail!r}). "
                "Cannot compute scanner_workspace transform."
            ) from exc
        raise


def _resolve_scanner_workspace(
    session: Session,
    scan_id: str,
    points: list,
    *,
    workspace_reader: Optional[WorkspaceReader] = None,
    colorized: bool = False,
) -> list[DisplayCoordinate]:
    """Transform scanner_workspace coordinates to ACI pixels via Loupe spatial table.

    Two storage paths:

    - ``workspace_reader`` provided: fetch ``spatial.csv`` + ``loupe.csv``
      bytes via the reader (production: R2; tests: moto), materialize
      into a temp directory, and call ``load_spatial_table``. R2 errors
      from the reader (e.g., ``HTTPException(404, "not_found")``) are
      converted to ``CoordinatesUnavailableError`` so the route layer's
      existing 400 mapping (``map.py:get_map_layers``) applies.
    - ``workspace_reader=None``: legacy local-FS read (local-filesystem
      dev installs without R2).

    When ``colorized`` is ``True`` the ACI ``file_path`` is rewritten
    ``sol_NNNN → sol_NNNN_colorized`` (via
    :func:`core.r2_keys.colorize_sol_segment`) so the colorized
    workspace's own ``spatial.csv`` / ``loupe.csv`` drive the transform.
    The colorized workspace mirrors the grayscale tree with only that one
    segment renamed, so the same path math (R2 key derivation or
    ``Path(...).parent.parent`` for the FS fallback) resolves the
    colorized companion files. A missing colorized workspace surfaces as
    :class:`CoordinatesUnavailableError` (R2 404 or FS not-found), which
    the route layer turns into "no colorized point set" rather than an
    error — the grayscale overlay still renders.
    """
    from fastapi import HTTPException

    from sherloc_pipeline.core.r2_keys import colorize_sol_segment
    from sherloc_pipeline.database.models import ContextImageORM
    from sherloc_pipeline.core.spatial import load_spatial_table

    # Derive working directory from the ACI context image path.
    # Convention: ACI file lives at <workspace>/img/<aci-product>.{PNG,IMG},
    # so the workspace dir is two levels up from the file.
    aci = (
        session.query(ContextImageORM)
        .filter(
            ContextImageORM.scan_id == scan_id,
            ContextImageORM.image_type == "ACI",
        )
        .first()
    )
    if aci is None:
        raise CoordinatesUnavailableError(
            f"No ACI context image found for scan {scan_id!r}. "
            "Cannot locate Loupe workspace to compute scanner_workspace transform."
        )

    # Select the grayscale base workspace or its colorized sibling. The
    # swap is on the same file_path the grayscale path uses, so both the
    # R2 reader and the FS fallback below resolve the correct workspace.
    workspace_file_path = aci.file_path
    if colorized:
        colorized_file_path = colorize_sol_segment(aci.file_path or "")
        if colorized_file_path is None:
            raise CoordinatesUnavailableError(
                f"Cannot derive a colorized workspace for scan {scan_id!r}: "
                f"ACI file_path {aci.file_path!r} has no 'sol_NNNN' segment "
                "to rewrite to 'sol_NNNN_colorized'."
            )
        workspace_file_path = colorized_file_path

    if workspace_reader is not None:
        # v1.0-beta production path: fetch CSVs from R2 via the injected
        # reader, materialize to a temp dir, then call the unchanged
        # FS-bound load_spatial_table on that temp dir.
        #
        # Each file is fetched in its own try/except so a 404 on either
        # one produces an error message that names the specific missing
        # file's R2 key (spec §3.9.8.3).
        # 5xx errors (misconfigured_path, upstream_credential_error,
        # upstream_timeout, upstream_error) propagate to the route
        # handler which preserves their HTTP status (FastAPI re-raises
        # HTTPException without wrapping). The route-layer
        # CoordinatesUnavailableError → 400 mapping does NOT apply
        # to these; they surface their own status codes.
        spatial_bytes = _fetch_workspace_file(
            workspace_reader, workspace_file_path, "spatial.csv", scan_id
        )
        loupe_bytes = _fetch_workspace_file(
            workspace_reader, workspace_file_path, "loupe.csv", scan_id
        )
        with tempfile.TemporaryDirectory(prefix="loupe_ws_") as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "spatial.csv").write_bytes(spatial_bytes)
            (tmp_path / "loupe.csv").write_bytes(loupe_bytes)
            try:
                df = load_spatial_table(tmp_path)
            except Exception as exc:
                raise CoordinatesUnavailableError(
                    f"Failed to parse Loupe workspace files from R2 for scan "
                    f"{scan_id!r} (file_path={workspace_file_path!r}): {exc}"
                ) from exc
    else:
        # Legacy FS path: local-filesystem dev runtime without R2.
        # Production containers do NOT take this branch.
        working_dir = Path(workspace_file_path).parent.parent

        # Validate workspace files exist before calling load_spatial_table
        # so we can give a clearer error than a raw exception.
        loupe_csv = working_dir / "loupe.csv"
        spatial_csv = working_dir / "spatial.csv"
        if not spatial_csv.is_file() or not loupe_csv.is_file():
            raise CoordinatesUnavailableError(
                f"Loupe workspace files not found at {working_dir!r} "
                f"(spatial.csv present={spatial_csv.is_file()}, "
                f"loupe.csv present={loupe_csv.is_file()}). "
                "Cannot compute scanner_workspace transform."
            )

        try:
            df = load_spatial_table(working_dir)
        except Exception as exc:
            raise CoordinatesUnavailableError(
                f"Failed to load spatial table from {working_dir!r}: {exc}"
            ) from exc

    # Build a lookup from zero-based point index → (xPix, yPix).
    spatial_lookup: dict[int, tuple[float, float]] = {
        int(row["point"]): (float(row["xPix"]), float(row["yPix"]))
        for _, row in df.iterrows()
    }

    coords: list[DisplayCoordinate] = []
    missing_indices: list[int] = []
    for pt in points:
        pix = spatial_lookup.get(pt.point_index)
        if pix is None:
            missing_indices.append(pt.point_index)
            continue
        coords.append(
            DisplayCoordinate(
                scan_point_id=pt.id,
                point_index=pt.point_index,
                aci_x=pix[0],
                aci_y=pix[1],
                transform_method="scanner_calibration",
            )
        )

    if missing_indices:
        logger.warning(
            "resolve_display_coordinates: %d point(s) from scan %s not found in "
            "spatial table and will be skipped (point_indices=%s)",
            len(missing_indices),
            scan_id,
            missing_indices[:10],
        )

    if not coords:
        raise CoordinatesUnavailableError(
            f"Spatial table for scan {scan_id!r} contained no matching points "
            f"(spatial table has {len(df)} rows, scan has {len(points)} points)."
        )

    return coords


def _write_cache(
    session: Session,
    scan_id: str,
    coords: list[DisplayCoordinate],
    *,
    colorized: bool,
    force_recompute: bool,
) -> None:
    """Persist resolved coordinates to map_display_coordinates.

    Rows are keyed by ``(scan_point_id, colorized)``; the grayscale and
    colorized variants of a scan coexist. When *force_recompute* is True,
    deletes existing rows **of the same variant** first (so recomputing
    grayscale never evicts the cached colorized result and vice versa).
    """
    from sherloc_pipeline.database.models import MapDisplayCoordinateORM, ScanPointORM

    if force_recompute:
        # Delete existing cache rows for this scan's points, this variant.
        # Pass an explicit select() to in_() (not a .subquery()) so SQLAlchemy
        # 2.x doesn't emit the "Coercing Subquery into a select() for IN()"
        # warning.
        point_ids = select(ScanPointORM.id).where(ScanPointORM.scan_id == scan_id)
        session.query(MapDisplayCoordinateORM).filter(
            MapDisplayCoordinateORM.scan_point_id.in_(point_ids),
            MapDisplayCoordinateORM.colorized == colorized,
        ).delete(synchronize_session="fetch")

    now = datetime.now(timezone.utc)
    for coord in coords:
        row = MapDisplayCoordinateORM(
            scan_point_id=coord.scan_point_id,
            colorized=colorized,
            aci_x=coord.aci_x,
            aci_y=coord.aci_y,
            transform_method=coord.transform_method,
            computed_at=now,
        )
        session.merge(row)  # upsert: safe if row already exists from a prior run

    session.flush()
    logger.debug(
        "_write_cache: wrote %d coordinate rows for scan %s (colorized=%s)",
        len(coords),
        scan_id,
        colorized,
    )
