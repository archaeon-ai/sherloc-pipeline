"""Scan-related endpoints: list, detail, points."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from sherloc_pipeline.core.coordinates import (
    CoordinatesUnavailableError,
    resolve_display_coordinates,
)
from sherloc_pipeline.database.models import ScanORM, ScanPointORM
from sherloc_pipeline.web.adapters import (
    point_orm_to_dto,
    scan_orm_to_detail,
    scan_orm_to_list_item,
)
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.r2_reader import (
    colorized_variant_exists,
    get_working_file,
    is_r2_mode,
)
from sherloc_pipeline.web.routes.images import select_served_aci
from sherloc_pipeline.web.schemas import (
    PointsResponse,
    ScanDetailResponse,
    ScanListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["scans"])

VALID_SCAN_CLASSES = {"primary", "sub_scan", "composite"}
VALID_SCAN_TYPES = {"detail", "line", "hdr", "survey"}


def _get_session(request: Request) -> Session:
    return request.state.db


def _get_data_access(request: Request) -> DataAccessService:
    """Resolve the DataAccessService from app state."""
    access_mode = getattr(request.app.state, "access_mode", "internal")
    return DataAccessService(access_mode=access_mode)


@router.get("/scans", response_model=ScanListResponse)
def list_scans(
    request: Request,
    sol: Optional[int] = Query(None),
    target: Optional[str] = Query(None),
    scan_class: Optional[str] = Query(None),
    scan_type: Optional[str] = Query(None),
    processing_status: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> ScanListResponse:
    """List scans with optional filters and pagination."""
    if scan_class is not None and scan_class not in VALID_SCAN_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid scan_class: {scan_class}")

    session = _get_session(request)
    data_access = _get_data_access(request)

    q = session.query(ScanORM)

    # Apply access-mode filtering (excludes Loupe data in public mode)
    q = data_access.filter_scans_query(q)

    if sol is not None:
        q = q.filter(ScanORM.sol_number == sol)
    if target is not None:
        # Case-insensitive substring match, treating spaces/underscores as equivalent
        pattern = f"%{target.replace('_', ' ')}%"
        q = q.filter(func.replace(func.lower(ScanORM.target), '_', ' ').like(pattern.lower()))
    if scan_class is not None:
        q = q.filter(ScanORM.scan_class == scan_class)
    if scan_type is not None:
        q = q.filter(ScanORM.scan_type == scan_type)
    if processing_status is not None:
        if processing_status == "null":
            q = q.filter(ScanORM.processing_status.is_(None))
        else:
            q = q.filter(ScanORM.processing_status == processing_status)

    total = q.count()
    scans = q.order_by(ScanORM.sol_number, ScanORM.scan_name).offset(offset).limit(limit).all()

    # Per spec §12.2: when the DB is empty (zero scans visible to this
    # access mode) AND the caller applied no filters, surface a hint
    # instead of letting the SPA show a generic "no scans found" state.
    message: Optional[str] = None
    has_filters = any(
        v is not None
        for v in (sol, target, scan_class, scan_type, processing_status)
    )
    if total == 0 and not has_filters:
        message = "No data ingested yet. See 'sherloc init --help' to bootstrap."

    return ScanListResponse(
        scans=[scan_orm_to_list_item(s) for s in scans],
        total=total,
        offset=offset,
        limit=limit,
        message=message,
    )


@router.get("/scans/{scan_id}", response_model=ScanDetailResponse)
def get_scan(request: Request, scan_id: str) -> ScanDetailResponse:
    """Retrieve full metadata for a single scan."""
    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    detail = scan_orm_to_detail(scan)

    # Probe R2 for a sol_NNNN_colorized/ sibling so the Workbench AciViewer
    # can gate its "Colorized" button instead of silently falling back to
    # grayscale when the variant is missing. Mirrors routes/map.py:198
    # which uses the same predicate to decide whether to advertise the
    # aci_colorized map layer. One R2 HEAD per scan-detail fetch — same
    # cost as the map-layers endpoint already pays.
    #
    # The selection MUST match the served ACI: scans with multiple
    # context_images rows (e.g. base + angle-range _145-185 variants)
    # would otherwise see the button disagree with the actual image
    # bytes the route serves. PR #31 Codex Round 1 F1.
    aci_row = select_served_aci(session, scan_id)
    if aci_row is not None:
        detail.colorized_aci_available = colorized_variant_exists(aci_row.file_path)

    return ScanDetailResponse(scan=detail)


@router.get("/scans/{scan_id}/points", response_model=PointsResponse)
def get_scan_points(request: Request, scan_id: str) -> PointsResponse:
    """List all measurement points for a scan."""
    session = _get_session(request)
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    data_access = _get_data_access(request)
    data_access.validate_scan_access(scan)

    points = (
        session.query(ScanPointORM)
        .filter(ScanPointORM.scan_id == scan_id)
        .order_by(ScanPointORM.point_index)
        .all()
    )

    dtos = [point_orm_to_dto(p) for p in points]

    # Resolve ACI pixel coordinates so the Workbench overlay (issue #15) and
    # any other consumer of `/api/scans/{id}/points` sees the same coords as
    # Map mode. Delegates to core.coordinates.resolve_display_coordinates,
    # which (a) returns cached rows from map_display_coordinates when present
    # and (b) falls back to the R2 workspace reader for first-time Loupe
    # scanner_workspace resolution. CoordinatesUnavailableError is the
    # graceful-degradation path: the endpoint still returns raw x_pixel/y_pixel
    # so PDS aci_pixel scans render via the frontend's null-frame branch,
    # and Loupe scans that cannot be resolved (e.g. missing workspace files)
    # simply lack the overlay rather than erroring the entire endpoint.
    reader = get_working_file if is_r2_mode() else None
    try:
        coords = resolve_display_coordinates(session, scan_id, workspace_reader=reader)
    except CoordinatesUnavailableError as exc:
        logger.info(
            "ACI pixel resolution unavailable for scan %s: %s",
            scan_id,
            exc,
        )
        coords = []

    pixel_map = {c.point_index: (c.aci_x, c.aci_y) for c in coords}
    for dto in dtos:
        pix = pixel_map.get(dto.point_index)
        if pix is not None:
            dto.x_aci_pixel, dto.y_aci_pixel = pix

    return PointsResponse(
        scan_id=scan_id,
        points=dtos,
        n_points=len(dtos),
    )
