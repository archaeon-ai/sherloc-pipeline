"""Map Mode API routes — spatial overlay and fitting.

Endpoints:
    GET  /api/map/layers/{scan_id}  — point set, Voronoi, available layers
    GET  /api/map/data/{scan_id}    — scalar layer data for rendering
    GET  /api/map/jobs/{job_id}     — fit job status (REST fallback)
    POST /api/map/fit               — start fit job
"""
import asyncio
import hashlib
import json
import logging
import secrets
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy.orm import Session

from sherloc_pipeline.core.coordinates import (
    CoordinatesUnavailableError,
    resolve_display_coordinates,
)
from sherloc_pipeline.core.voronoi import compute_voronoi_geometry
from sherloc_pipeline.database.models import (
    ContextImageORM,
    FittedPeakORM,
    MapFitCacheORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
)
from sherloc_pipeline.web.data_access import DataAccessService
from sherloc_pipeline.web.r2_reader import (
    colorized_variant_exists,
    get_working_file,
    is_r2_mode,
)
from sherloc_pipeline.web.schemas import (
    MapCachedResultDTO,
    MapDataPointDTO,
    MapDataResponse,
    MapFitRequest,
    MapFitResponse,
    MapJobStatusResponse,
    MapLayerInfoDTO,
    MapLayersResponse,
    MapPointDTO,
    MapVoronoiDTO,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/map", tags=["map"])


def _get_session(request: Request) -> Session:
    return request.state.db


def _get_data_access(request: Request) -> DataAccessService:
    access_mode = getattr(request.app.state, "access_mode", "internal")
    return DataAccessService(access_mode=access_mode)


# ---------------------------------------------------------------------------
# GET /api/map/layers/{scan_id}
# ---------------------------------------------------------------------------


@router.get("/layers/{scan_id}", response_model=MapLayersResponse)
def get_map_layers(request: Request, scan_id: str) -> MapLayersResponse:
    """Return point set, Voronoi geometry, available overlay layers, and cached fit results."""
    session = _get_session(request)

    # Verify scan exists and check access
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    _get_data_access(request).validate_scan_access(scan)

    # Resolve display coordinates (raises 400 if unavailable). In v1.0-beta
    # production, inject the R2-backed Loupe-workspace reader so the resolver
    # can fetch spatial.csv + loupe.csv from R2 (spec §3.9.8). Legacy local
    # dev worktree (no PHASE_TIER set) falls
    # back to FS reads via the None-reader branch in coordinates.py.
    reader = get_working_file if is_r2_mode() else None
    try:
        coords = resolve_display_coordinates(session, scan_id, workspace_reader=reader)
    except CoordinatesUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not coords:
        raise HTTPException(
            status_code=400,
            detail=f"No display coordinates available for scan {scan_id}",
        )

    # Build point DTO list and numpy array for Voronoi
    point_dtos = [
        MapPointDTO(index=c.point_index, x=c.aci_x, y=c.aci_y) for c in coords
    ]
    pts_array = np.array([[c.aci_x, c.aci_y] for c in coords], dtype=np.float64)
    coordinate_source = coords[0].transform_method

    # Compute Voronoi geometry (returns None for degenerate point sets)
    voronoi_result = compute_voronoi_geometry(pts_array)
    if voronoi_result is not None:
        voronoi_dto = MapVoronoiDTO(
            vertices=voronoi_result.vertices,
            regions=voronoi_result.regions,
            boundary=voronoi_result.boundary,
            edge_mask=voronoi_result.edge_mask,
        )
    else:
        voronoi_dto = None

    # Query available layers: group fitted_peaks by fit_modality + mineral_assignment
    # Join: FittedPeakORM -> SpectrumORM -> ScanPointORM (filtered by scan_id)
    peak_rows = (
        session.query(
            FittedPeakORM.fit_modality,
            FittedPeakORM.mineral_assignment,
        )
        .join(SpectrumORM, SpectrumORM.id == FittedPeakORM.spectrum_id)
        .join(ScanPointORM, ScanPointORM.id == SpectrumORM.scan_point_id)
        .filter(ScanPointORM.scan_id == scan_id)
        .all()
    )

    # Group into: {source: {domain: {n_detections, classes}}}
    # Using a single source key "db" since peaks come from the fitted_peaks table.
    layer_counts: dict[str, dict[str, set]] = {}
    for fit_modality, mineral_assignment in peak_rows:
        domain_map = layer_counts.setdefault(fit_modality, {})
        if mineral_assignment:
            domain_map.setdefault(mineral_assignment, set()).add(mineral_assignment)

    # Build available_layers: {domain: {mineral_assignment: MapLayerInfoDTO}}
    # Structure: {source: {domain: MapLayerInfoDTO}}
    available_layers: dict[str, dict[str, dict]] = {}
    for fit_modality, assignments in layer_counts.items():
        source_key = "db"
        domain_entry = available_layers.setdefault(source_key, {})
        classes = sorted(assignments.keys())
        # Add g1_doublet composite if both group1a and group1b exist
        if fit_modality == "fluorescence" and "group1a" in classes and "group1b" in classes:
            classes = ["g1_doublet"] + classes
        domain_entry[fit_modality] = MapLayerInfoDTO(
            n_detections=sum(len(v) for v in assignments.values()),
            classes=classes,
        ).model_dump()

    # Query cached fit results for this scan
    cache_rows = (
        session.query(MapFitCacheORM)
        .filter(MapFitCacheORM.scan_id == scan_id)
        .order_by(MapFitCacheORM.created_at.desc())
        .all()
    )
    cached_results = []
    for row in cache_rows:
        try:
            domains = json.loads(row.domains) if isinstance(row.domains, str) else row.domains
        except (json.JSONDecodeError, TypeError):
            domains = []
        try:
            n_detections = (
                json.loads(row.n_detections_json)
                if isinstance(row.n_detections_json, str)
                else row.n_detections_json
            )
        except (json.JSONDecodeError, TypeError):
            n_detections = {}
        cached_results.append(
            MapCachedResultDTO(
                cache_id=row.id,
                domains=domains,
                profile_name=row.profile_name,
                profile_hash=row.profile_hash,
                created_at=row.created_at,
                n_points=row.n_points,
                n_detections=n_detections,
            )
        )

    base_images = [
        {"type": "aci", "url": f"/api/images/{scan_id}/aci"},
        {"type": "aci_enhanced", "url": f"/api/images/{scan_id}/aci?enhanced=true"},
    ]
    aci_row = (
        session.query(ContextImageORM)
        .filter(
            ContextImageORM.scan_id == scan_id,
            ContextImageORM.image_type == "ACI",
        )
        .first()
    )
    if aci_row is not None and colorized_variant_exists(aci_row.file_path):
        base_images.append(
            {"type": "aci_colorized", "url": f"/api/images/{scan_id}/aci?colorized=true"}
        )

    return MapLayersResponse(
        scan_id=scan_id,
        coordinate_source=coordinate_source,
        base_images=base_images,
        point_set={
            "points": [p.model_dump() for p in point_dtos],
            "voronoi": voronoi_dto.model_dump() if voronoi_dto is not None else None,
        },
        available_layers=available_layers,
        cached_results=cached_results,
    )


# ---------------------------------------------------------------------------
# GET /api/map/data/{scan_id}
# ---------------------------------------------------------------------------

# Minimum SNR to classify a peak as "measured"
_SNR_THRESHOLD = 3.0


@router.get("/data/{scan_id}", response_model=MapDataResponse)
def get_map_data(
    request: Request,
    scan_id: str,
    domain: str = Query(..., description="Fitting domain (minerals/organics/hydration/fluorescence)"),
    value_type: str = Query("snr", description="Scalar value to render (snr, amplitude, area)"),
    cache_id: Optional[str] = Query(None, description="Use a specific cached result"),
) -> MapDataResponse:
    """Return per-point scalar data for a given domain layer."""
    session = _get_session(request)

    # Verify scan exists and check access
    scan = session.query(ScanORM).filter(ScanORM.id == scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    _get_data_access(request).validate_scan_access(scan)

    # ------------------------------------------------------------------
    # Cache path: load from map_fit_cache if cache_id provided
    # ------------------------------------------------------------------
    if cache_id is not None:
        cache_row = (
            session.query(MapFitCacheORM)
            .filter(MapFitCacheORM.id == cache_id, MapFitCacheORM.scan_id == scan_id)
            .first()
        )
        if cache_row is None:
            raise HTTPException(status_code=404, detail="Cached result not found")

        try:
            results = json.loads(cache_row.results_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(
                status_code=500, detail="Malformed cache entry"
            ) from exc

        points = []
        for entry in results:
            if entry.get("domain") != domain:
                continue
            points.append(
                MapDataPointDTO(
                    index=entry["point_index"],
                    value=entry.get("value"),
                    status=entry.get("status", "missing"),
                    assignment=entry.get("assignment"),
                    center_cm1=entry.get("center_cm1"),
                    fwhm_cm1=entry.get("fwhm_cm1"),
                    center_nm=entry.get("center_nm"),
                )
            )
        return MapDataResponse(
            scan_id=scan_id,
            domain=domain,
            value_type=value_type,
            cache_id=cache_id,
            points=points,
        )

    # ------------------------------------------------------------------
    # Live path: query fitted_peaks for the scan filtered by domain
    # ------------------------------------------------------------------
    # Get all scan_points for this scan to ensure we cover every point (including those
    # with no peaks — they get status="missing").
    all_scan_points = (
        session.query(ScanPointORM.id, ScanPointORM.point_index)
        .filter(ScanPointORM.scan_id == scan_id)
        .order_by(ScanPointORM.point_index)
        .all()
    )
    if not all_scan_points:
        raise HTTPException(status_code=400, detail=f"No scan points found for scan {scan_id}")

    scan_point_id_to_index: dict[str, int] = {sp.id: sp.point_index for sp in all_scan_points}

    # Query peaks grouped by scan_point_id for this domain
    peak_rows = (
        session.query(
            ScanPointORM.id.label("scan_point_id"),
            ScanPointORM.point_index,
            FittedPeakORM.snr,
            FittedPeakORM.amplitude,
            FittedPeakORM.area,
            FittedPeakORM.mineral_assignment,
            FittedPeakORM.center_cm1,
            FittedPeakORM.fwhm_cm1,
            FittedPeakORM.center_nm,
        )
        .join(SpectrumORM, SpectrumORM.scan_point_id == ScanPointORM.id)
        .join(FittedPeakORM, FittedPeakORM.spectrum_id == SpectrumORM.id)
        .filter(
            ScanPointORM.scan_id == scan_id,
            FittedPeakORM.fit_modality == domain,
        )
        .all()
    )

    # Group peaks by point_index
    peaks_by_point: dict[int, list] = {}
    for row in peak_rows:
        peaks_by_point.setdefault(row.point_index, []).append(row)

    # Extract the scalar value from a peak row based on value_type
    def _extract_value(peak_row) -> Optional[float]:
        if value_type == "snr":
            return float(peak_row.snr) if peak_row.snr is not None else None
        if value_type == "amplitude":
            return float(peak_row.amplitude) if peak_row.amplitude is not None else None
        if value_type == "area":
            return float(peak_row.area) if peak_row.area is not None else None
        # Unknown value_type — return snr as fallback
        return float(peak_row.snr) if peak_row.snr is not None else None

    # Optional class_id filter from query params
    class_id = request.query_params.get("class_id")
    is_doublet = class_id == "g1_doublet"

    # Build per-point results
    points: list[MapDataPointDTO] = []
    for sp_id, point_index in all_scan_points:
        peak_list = peaks_by_point.get(point_index, [])

        # g1_doublet composite: measured only if BOTH group1a and group1b present
        if is_doublet:
            g1a = [p for p in peak_list if p.mineral_assignment == "group1a"]
            g1b = [p for p in peak_list if p.mineral_assignment == "group1b"]
            if g1a and g1b:
                # Value = min SNR of the pair (weaker detection limits confidence)
                snr_a = g1a[0].snr or 0.0
                snr_b = g1b[0].snr or 0.0
                min_snr = min(snr_a, snr_b)
                points.append(
                    MapDataPointDTO(
                        index=point_index,
                        value=min_snr,
                        status="measured" if min_snr >= _SNR_THRESHOLD else "below_threshold",
                        assignment="g1_doublet",
                        center_nm=round((g1a[0].center_nm + g1b[0].center_nm) / 2, 1)
                        if g1a[0].center_nm and g1b[0].center_nm
                        else None,
                    )
                )
            else:
                points.append(MapDataPointDTO(index=point_index, value=None, status="missing"))
            continue

        # Filter by class_id if specified
        if class_id:
            peak_list = [p for p in peak_list if p.mineral_assignment == class_id]

        if not peak_list:
            points.append(
                MapDataPointDTO(
                    index=point_index,
                    value=None,
                    status="missing",
                )
            )
            continue

        # Find the best peak: highest SNR
        best = max(peak_list, key=lambda p: p.snr or 0.0)
        best_snr = best.snr if best.snr is not None else 0.0

        if best_snr >= _SNR_THRESHOLD:
            status = "measured"
        else:
            status = "below_threshold"

        value = _extract_value(best)
        points.append(
            MapDataPointDTO(
                index=point_index,
                value=value,
                status=status,
                assignment=best.mineral_assignment,
                center_cm1=float(best.center_cm1) if best.center_cm1 is not None else None,
                fwhm_cm1=float(best.fwhm_cm1) if best.fwhm_cm1 is not None else None,
                center_nm=float(best.center_nm) if best.center_nm is not None else None,
            )
        )

    return MapDataResponse(
        scan_id=scan_id,
        domain=domain,
        value_type=value_type,
        cache_id=None,
        points=points,
    )


# ---------------------------------------------------------------------------
# POST /api/map/fit
# ---------------------------------------------------------------------------


def _config_to_dict(config) -> dict:
    """Convert a Config object to a plain dict for MapFitService."""
    result = {}
    for attr in ("fitting", "fluorescence_fitting", "preprocessing", "spectral_regions"):
        val = getattr(config, attr, None)
        if val is not None:
            if isinstance(val, dict):
                result[attr] = val
            elif hasattr(val, "__dict__"):
                result[attr] = vars(val)
            else:
                result[attr] = val
    return result


@router.post("/fit", response_model=MapFitResponse, status_code=202)
def start_map_fit(request: Request, body: MapFitRequest) -> MapFitResponse:
    """Start a map-mode fitting job.

    Creates a MapJobContext in the registry, submits fitting to the
    map executor thread, and returns 202 with the job_id and ws_url.
    """
    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.services.map_fitting import (
        MapFitService,
        compute_fit_signature,
    )
    from sherloc_pipeline.web.ws_map import MapJobRegistry, make_fitting_callbacks

    session = _get_session(request)

    # Verify scan exists and check access
    scan = session.query(ScanORM).filter(ScanORM.id == body.scan_id).first()
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")
    _get_data_access(request).validate_scan_access(scan)

    # Resolve coordinates to determine n_points and validate the scan is mappable.
    # Same R2/FS branch as get_map_layers above (spec §3.9.8).
    reader = get_working_file if is_r2_mode() else None
    try:
        coords = resolve_display_coordinates(
            session, body.scan_id, workspace_reader=reader
        )
    except CoordinatesUnavailableError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n_points = (
        len(body.point_indices) if body.point_indices is not None else len(coords)
    )

    # Build point_coords lookup: {point_index: (aci_x, aci_y)}
    point_coords = {c.point_index: (c.aci_x, c.aci_y) for c in coords}

    # Check for active jobs for this scan via the map registry
    registry: MapJobRegistry = request.app.state.map_registry
    existing = registry.find_active_for_scan(body.scan_id)
    if existing is not None:
        return MapFitResponse(
            job_id=existing.job_id,
            n_points=n_points,
            ws_url=f"/ws/map/{existing.job_id}",
        )

    # Create job
    job_id = f"mf_{secrets.token_hex(12)}"
    loop = request.app.state.event_loop
    ctx = registry.create(
        job_id=job_id,
        scan_id=body.scan_id,
        loop=loop,
        n_points=n_points,
    )

    # Create callbacks that bridge fitting thread -> asyncio queue
    on_point_fitted, on_progress, on_log = make_fitting_callbacks(ctx)

    # Submit fitting work to the dedicated map executor thread.
    # The fitting thread gets its own DB session (SQLAlchemy sessions are
    # not thread-safe).
    engine = request.app.state.engine
    config = request.app.state.config
    scan_id = body.scan_id
    domains = list(body.domains)
    point_indices = list(body.point_indices) if body.point_indices is not None else None

    def _run_fit() -> None:
        """Fitting thread entry point."""
        factory = get_session_factory(engine)
        fit_session = factory()
        try:
            ctx.set_status("running")
            service = MapFitService(config=config.to_dict() if hasattr(config, 'to_dict') else _config_to_dict(config))
            summary = service.run_map_fit(
                session=fit_session,
                scan_id=scan_id,
                domains=domains,
                point_indices=point_indices,
                point_coords=point_coords,
                on_point_fitted=on_point_fitted,
                on_progress=on_progress,
                on_log=on_log,
                cancel_event=ctx.cancel_event,
            )
            if ctx.cancel_event.is_set():
                ctx.set_status("cancelled")
            else:
                ctx.set_status("complete")
                on_point_fitted.send_complete({  # type: ignore[attr-defined]
                    "total_points": summary.total_points,
                    "detections": summary.detections,
                    "elapsed_s": summary.elapsed_s,
                })
        except Exception as exc:
            logger.exception("Map fit job %s failed", job_id)
            ctx.set_status("failed")
            on_point_fitted.send_error(str(exc))  # type: ignore[attr-defined]
        finally:
            fit_session.close()

    executor = request.app.state.map_executor
    executor.submit(_run_fit)

    return MapFitResponse(
        job_id=job_id,
        n_points=n_points,
        ws_url=f"/ws/map/{job_id}",
    )


# ---------------------------------------------------------------------------
# GET /api/map/jobs/{job_id}
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=MapJobStatusResponse)
def get_map_job_status(request: Request, job_id: str) -> MapJobStatusResponse:
    """Return fit job status (REST polling fallback when WebSocket is unavailable).

    Checks the map job registry first (for map fitting jobs), then falls back
    to the general job queue.
    """
    # Check map registry first (map fitting jobs use mf_ prefix)
    registry = getattr(request.app.state, "map_registry", None)
    if registry is not None:
        ctx = registry.get(job_id)
        if ctx is not None:
            status = ctx.get_status()
            # Map internal status to schema status
            status_map = {
                "queued": "running",
                "running": "running",
                "complete": "complete",
                "failed": "failed",
                "cancelled": "cancelled",
            }
            mapped_status = status_map.get(status, status)
            total = ctx.n_points
            # Estimate fitted from the message buffer (count point_fitted messages)
            fitted = sum(
                1 for m in ctx.message_buffer if m.get("type") == "point_fitted"
            )
            results_available = status == "complete"
            return MapJobStatusResponse(
                job_id=job_id,
                status=mapped_status,
                fitted=fitted,
                total=total,
                results_available=results_available,
            )

    # Fall back to general job queue
    job_queue = request.app.state.job_queue
    state = job_queue.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    snap = state.snapshot()
    status = snap["status"]

    # Map JobStatus values to the schema's expected strings
    status_map = {
        "queued": "running",
        "running": "running",
        "completed": "complete",
        "failed": "failed",
        "cancelled": "cancelled",
    }
    mapped_status = status_map.get(status, status)

    # Derive fitted/total from progress_pct if available
    progress_pct = snap.get("progress_pct") or 0
    # We don't have n_points stored on the job directly, so approximate from result
    result = snap.get("result") or {}
    total = result.get("n_points", 0)
    fitted = int(total * progress_pct / 100) if total > 0 else 0

    results_available = status in ("completed",) and bool(result)

    return MapJobStatusResponse(
        job_id=job_id,
        status=mapped_status,
        fitted=fitted,
        total=total,
        results_available=results_available,
    )
