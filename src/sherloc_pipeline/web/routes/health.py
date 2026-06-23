"""GET /api/health endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter, Request
from sqlalchemy import func

from sherloc_pipeline.database.models import ScanORM, SpectrumORM
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION, HealthCheck, HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """System health check."""
    engine = request.app.state.engine
    config = request.app.state.config
    job_queue = request.app.state.job_queue

    checks = {}
    overall = "ok"

    # Database check -- use the per-request session from middleware
    try:
        session = request.state.db
        n_scans = session.query(func.count(ScanORM.id)).scalar() or 0
        n_spectra = session.query(func.count(SpectrumORM.id)).scalar() or 0
        db_path = str(engine.url).replace("sqlite:///", "")
        checks["database"] = HealthCheck(
            status="ok",
            path=db_path if db_path else None,
            n_scans=n_scans,
            n_spectra=n_spectra,
        )
    except Exception as exc:
        checks["database"] = HealthCheck(status="error", error=str(exc))
        overall = "error"

    # Config check
    try:
        import hashlib
        import json

        fitting_json = json.dumps(config.fitting, sort_keys=True, default=str)
        config_hash = f"sha256:{hashlib.sha256(fitting_json.encode()).hexdigest()[:12]}"
        checks["config"] = HealthCheck(status="ok", config_hash=config_hash)
    except Exception as exc:
        checks["config"] = HealthCheck(status="error", error=str(exc))
        if overall != "error":
            overall = "degraded"

    # Job queue check
    stats = job_queue.stats
    checks["job_queue"] = HealthCheck(
        status="ok",
        running=stats["running"],
        queued=stats["queued"],
        max_depth=stats["max_depth"],
    )

    from sherloc_pipeline import __version__

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        pipeline_version=__version__,
        checks=checks,
    )
