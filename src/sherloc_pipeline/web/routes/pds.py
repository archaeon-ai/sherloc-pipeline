"""PDS archive endpoints: catalog, download+ingest, available-sols (legacy)."""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import distinct

from sherloc_pipeline.database.models import SolORM
from sherloc_pipeline.web.jobs import JobQueue
from sherloc_pipeline.web.routes.config import feature_pds_browser_enabled
from sherloc_pipeline.web.schemas import (
    API_SCHEMA_VERSION,
    PDSAvailableResponse,
    PDSCatalogResponse,
    PDSCatalogSolInfo,
    PDSDownloadRequest,
    PDSDownloadResponse,
    PDSSolInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["pds"])


def _require_pds_browser_enabled() -> None:
    """Raise 404 when the PDS Browser feature is env-disabled (issue #21).

    Pure defense in depth — the SPA already hides the nav-tab + route
    on the same env signal. A direct ``curl https://.../api/pds/...``
    after the SPA gate should still 404 so unauthenticated probes can't
    discover the disabled feature surface.
    """
    if not feature_pds_browser_enabled():
        raise HTTPException(status_code=404, detail="Not Found")


# ---------------------------------------------------------------------------
# Simple in-memory cache for PDS catalog (1-hour TTL)
# ---------------------------------------------------------------------------

_catalog_cache: Dict[str, Any] = {
    "sols": None,
    "timestamp": 0.0,
}
_CATALOG_TTL_SECONDS = 3600  # 1 hour


def _get_cached_pds_sols(config) -> Optional[List[int]]:
    """Return cached PDS sol list if still fresh, else None."""
    if _catalog_cache["sols"] is not None:
        age = time.time() - _catalog_cache["timestamp"]
        if age < _CATALOG_TTL_SECONDS:
            return _catalog_cache["sols"]
    return None


def _set_cached_pds_sols(sols: List[int]) -> None:
    """Store PDS sol list in the cache."""
    _catalog_cache["sols"] = sols
    _catalog_cache["timestamp"] = time.time()


def _clear_catalog_cache() -> None:
    """Clear the catalog cache (useful for tests)."""
    _catalog_cache["sols"] = None
    _catalog_cache["timestamp"] = 0.0


# ---------------------------------------------------------------------------
# GET /api/pds/catalog
# ---------------------------------------------------------------------------


@router.get("/pds/catalog", response_model=PDSCatalogResponse)
def pds_catalog(request: Request) -> PDSCatalogResponse:
    """List available PDS sols with caching and ingested-sol overlay.

    Fetches the PDS collection inventory via :class:`PDSDownloader` and
    caches the result for 1 hour.  Also queries the local database for
    already-ingested sols.
    """
    _require_pds_browser_enabled()
    config = request.app.state.config
    session = request.state.db

    # Try cache first
    cached_sols = _get_cached_pds_sols(config)
    if cached_sols is not None:
        available_sols = cached_sols
    else:
        # Fetch from PDS
        try:
            from sherloc_pipeline.core.pds_client import PDSDownloader

            downloader = PDSDownloader.from_config(getattr(config, "pds", None))
            available_sols = downloader.discover_available_sols()
            downloader.close()
            _set_cached_pds_sols(available_sols)
        except Exception as exc:
            logger.exception("Failed to fetch PDS catalog")
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch PDS catalog: {exc}",
            ) from exc

    # Get already-ingested sols from local DB
    ingested = [
        row[0]
        for row in session.query(distinct(SolORM.sol_number))
        .order_by(SolORM.sol_number)
        .all()
    ]

    return PDSCatalogResponse(
        available_sols=[PDSCatalogSolInfo(sol=s) for s in available_sols],
        total_available=len(available_sols),
        already_ingested=ingested,
    )


# ---------------------------------------------------------------------------
# GET /api/pds/available-sols (legacy stub)
# ---------------------------------------------------------------------------


@router.get("/pds/available-sols", response_model=PDSAvailableResponse)
def available_sols(request: Request) -> PDSAvailableResponse:
    """List available sols from PDS.

    This is a stub implementation that returns already-ingested sols.
    Full PDS crawling requires network access and is not implemented
    in the initial backend.
    """
    _require_pds_browser_enabled()
    session = request.state.db

    # Get already-ingested sols
    ingested = [
        row[0]
        for row in session.query(distinct(SolORM.sol_number)).order_by(SolORM.sol_number).all()
    ]

    return PDSAvailableResponse(
        sols=[],  # Real PDS crawl not implemented yet
        total=0,
        already_ingested=ingested,
    )


# ---------------------------------------------------------------------------
# POST /api/pds/download
# ---------------------------------------------------------------------------


def _download_aci_for_sol(
    sol: int,
    downloader,
    pds_db_path: Optional[str],
) -> int:
    """Download ACI images for a sol by resolving pds: references.

    Queries the PDS database for context image records whose ``file_path``
    starts with ``pds:`` (i.e. unresolved LIDVIDs), resolves them to
    download URLs via the PDS Search API, downloads the images, and
    updates the DB records to point at the local files.

    Args:
        sol: Mars sol number.
        downloader: An open :class:`PDSDownloader` instance.
        pds_db_path: Path to the PDS SQLite database, or None.

    Returns:
        Number of ACI images successfully downloaded.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import ContextImageORM, ScanORM

    if pds_db_path is None:
        return 0

    engine = create_engine(f"sqlite:///{pds_db_path}")
    factory = get_session_factory(engine)
    session = factory()

    try:
        # Find all pds: ACI references for this sol
        aci_records = (
            session.query(ContextImageORM)
            .join(ScanORM, ContextImageORM.scan_id == ScanORM.id)
            .filter(
                ScanORM.sol_number == sol,
                ContextImageORM.image_type == "ACI",
                ContextImageORM.file_path.like("pds:%"),
            )
            .all()
        )

        if not aci_records:
            logger.info("Sol %d: no unresolved pds: ACI references found", sol)
            return 0

        # Collect LIDVIDs (strip "pds:" prefix)
        lidvids = [r.file_path[4:] for r in aci_records]

        # Batch resolve download URLs
        url_map = downloader.resolve_aci_urls(lidvids)

        # Download each and update DB
        cache_dir = downloader.cache_dir
        dest_dir = cache_dir / f"sol_{sol:04d}" / "data_aci"
        dest_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for record in aci_records:
            lidvid = record.file_path[4:]
            url = url_map.get(lidvid)
            if url is None:
                continue

            # Extract filename from URL
            filename = url.rsplit("/", 1)[-1]
            dest = dest_dir / filename

            if downloader.download_aci_image(url, dest):
                record.file_path = str(dest)
                downloaded += 1

        session.commit()
        logger.info("Sol %d: downloaded %d ACI images", sol, downloaded)
        return downloaded
    except Exception as exc:
        logger.warning("ACI download failed for sol %d: %s", sol, exc)
        session.rollback()
        return 0
    finally:
        session.close()
        engine.dispose()


def _run_pds_download(
    sol: int,
    engine,
    config,
    job_state=None,
    force_reingest: bool = False,
) -> Dict[str, Any]:
    """Background PDS download + ingest task.

    Steps:
      1. Download processed CSV/XML products via PDSDownloader.
      2. Ingest into phase_pds.db via PDSIngestionService.
      3. Download ACI images by resolving pds: LIDVIDs via PDS Search API.
      4. Report progress through ``job_state``.
    """
    from sherloc_pipeline.core.pds_client import PDSDownloader

    # Phase 1: Download (0-70% of progress)
    if job_state is not None:
        job_state.update(
            phase="downloading",
            progress_pct=0.0,
            message=f"Starting download for sol {sol}",
        )

    def _on_download_progress(done: int, total: int) -> None:
        if job_state is not None:
            pct = 70.0 * (done / max(total, 1))
            job_state.update(
                phase="downloading",
                progress_pct=round(pct, 1),
                message=f"Sol {sol}: {done}/{total} files",
            )

    pds_config = getattr(config, "pds", None)
    downloader = PDSDownloader.from_config(pds_config)
    try:
        download_result = downloader.download_sol(
            sol,
            force=force_reingest,
            collections=["data_processed"],
            progress_callback=_on_download_progress,
        )

        # Phase 2: Ingest (70-85% of progress)
        if job_state is not None:
            job_state.update(
                phase="ingesting",
                progress_pct=70.0,
                message=f"Ingesting sol {sol} into database...",
            )

        from sherloc_pipeline.services.pds_ingestion import PDSIngestionService

        cache_dir = downloader.cache_dir
        sol_dir = cache_dir / f"sol_{sol:04d}" / "data_processed"

        n_scans = 0
        n_spectra = 0
        warnings: List[str] = []

        pds_db_path = (
            config.database.get("pds_path")
            if hasattr(config, "database") and isinstance(config.database, dict)
            else None
        )

        if sol_dir.exists():
            service = PDSIngestionService(pds_db_path=pds_db_path)
            ingest_result = service.ingest_sol(sol_dir, force=force_reingest)
            n_scans = getattr(ingest_result, "n_scans", 0)
            n_spectra = getattr(ingest_result, "n_spectra", 0)
        else:
            warnings.append(f"No data_processed directory found for sol {sol}")

        # Phase 3: Download ACI images via PDS Search API
        if job_state is not None:
            job_state.update(
                phase="aci_download",
                progress_pct=85.0,
                message=f"Downloading ACI images for sol {sol}...",
            )

        n_aci = _download_aci_for_sol(sol, downloader, pds_db_path)

        return {
            "sol": sol,
            "n_scans": n_scans,
            "n_spectra": n_spectra,
            "n_aci": n_aci,
            "n_downloaded": download_result.n_downloaded,
            "n_skipped": download_result.n_skipped,
            "warnings": warnings,
        }
    finally:
        downloader.close()


@router.post("/pds/download", status_code=202, response_model=PDSDownloadResponse)
def download_sol(request: Request, body: PDSDownloadRequest) -> PDSDownloadResponse:
    """Submit a PDS download+ingest job."""
    _require_pds_browser_enabled()
    # Public mode: PDS download is not available
    access_mode = getattr(request.app.state, "access_mode", "internal")
    if access_mode == "public":
        raise HTTPException(
            status_code=403,
            detail="PDS download is not available in public mode",
        )

    job_queue: JobQueue = request.app.state.job_queue
    engine = request.app.state.engine
    config = request.app.state.config

    # Single-flight: if a download for this sol is already active, return existing job
    existing_job = job_queue.find_active_by_scan_id(f"pds_sol_{body.sol}")
    if existing_job is not None:
        pos = job_queue.queue_position(existing_job.job_id) or 1
        return PDSDownloadResponse(
            job_id=existing_job.job_id,
            status=existing_job.status.value,
            queue_position=pos,
            sol=body.sol,
            submitter_token=existing_job.submitter_token,
            created_at=existing_job.created_at,
        )

    # Check if sol already ingested
    session = request.state.db
    existing = session.query(SolORM).filter(SolORM.sol_number == body.sol).first()
    if existing and not body.force_reingest:
        raise HTTPException(
            status_code=409,
            detail=f"Sol {body.sol} already ingested. Set force_reingest=true to re-ingest.",
        )

    try:
        state = job_queue.submit(
            scan_id=f"pds_sol_{body.sol}",
            func=_run_pds_download,
            sol=body.sol,
            engine=engine,
            config=config,
            force_reingest=body.force_reingest,
        )
    except ValueError:
        return JSONResponse(
            status_code=429,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": "queue_full",
                "retry_after_seconds": 60,
            },
            headers={"Retry-After": "60"},
        )

    pos = job_queue.queue_position(state.job_id) or 1

    return PDSDownloadResponse(
        job_id=state.job_id,
        status=state.status.value,
        queue_position=pos,
        sol=body.sol,
        submitter_token=state.submitter_token,
        created_at=state.created_at,
    )
