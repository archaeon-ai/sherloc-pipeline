"""GET /api/health endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func

from sherloc_pipeline.database.models import ScanORM, SpectrumORM
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION, HealthCheck, HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


def _authenticated(request: Request) -> bool:
    """True when the request carries a valid credential (always in dev mode).

    Monitoring probes send none; they are not run through the validator, so
    they leave no failed-auth log line.
    """
    import os

    from sherloc_pipeline.web.auth import require_authenticated_request

    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")
    has_credential = bool(
        request.headers.get("authorization") or request.headers.get("cf-access-jwt-assertion")
    )
    if mode != "dev" and not has_credential:
        return False
    try:
        require_authenticated_request(request)
    except HTTPException:
        return False
    return True


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """System health check.

    An anonymous caller gets status and version only. The per-check detail
    (counts, queue depth, config hash) needs an authenticated request; the
    database path is never returned.
    """
    detailed = _authenticated(request)

    config = request.app.state.config
    job_queue = request.app.state.job_queue

    checks = {}
    overall = "ok"

    # Database check -- use the per-request session from middleware
    try:
        session = request.state.db
        n_scans = session.query(func.count(ScanORM.id)).scalar() or 0
        n_spectra = session.query(func.count(SpectrumORM.id)).scalar() or 0
        checks["database"] = HealthCheck(
            status="ok",
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
        checks=checks if detailed else {},
    )
