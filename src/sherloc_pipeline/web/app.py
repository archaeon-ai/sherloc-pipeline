"""FastAPI application factory for the SHERLOC Web API."""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from sherloc_pipeline.web.jobs import JobQueue
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION
from sherloc_pipeline.web.ws_map import MapJobRegistry

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple sliding-window rate limiter per IP."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def is_allowed(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            # Lazy eviction of stale IPs
            stale_ips = [k for k, v in self._requests.items() if v and now - v[-1] > 300]
            for k in stale_ips:
                del self._requests[k]

            times = self._requests.setdefault(ip, [])
            times[:] = [t for t in times if now - t < self._window]
            if len(times) >= self._max:
                return False
            times.append(now)
            return True


def create_app(
    *,
    engine: Optional[Engine] = None,
    database_path: Optional[Union[str, Path]] = None,
    config=None,
    access_mode: str = "internal",
    frontend_dist: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: Pre-created SQLAlchemy engine (used in tests).
        database_path: Path to SQLite database (creates engine if ``engine`` is None).
            Falls back to the ``SHERLOC_DB`` environment variable when not provided.
        config: Pre-loaded Config object (used in tests).
        access_mode: ``"internal"`` (default) or ``"public"``. Controls
            whether Loupe-sourced data is accessible via the API.
            Falls back to the ``SHERLOC_ACCESS_MODE`` environment variable when
            the default ``"internal"`` value is in use.
        frontend_dist: Override path to the Svelte SPA build directory.
            Defaults to ``<this file>/frontend/dist``. Used by regression tests
            that need to drive the SPA catch-all against a controlled tree.

    Returns:
        Configured FastAPI instance.
    """
    # Resolve database_path and access_mode from env vars when not passed explicitly.
    # This allows uvicorn --factory invocations (which call create_app() with no args)
    # to pick up configuration from the environment. When a pre-built engine is passed
    # in (test path), skip the env-var lookup — the engine is authoritative and the
    # ambient SHERLOC_DB is irrelevant to which DB the app will actually use.
    if database_path is None and engine is None:
        database_path = os.environ.get("SHERLOC_DB")
    if access_mode == "internal":
        access_mode = os.environ.get("SHERLOC_ACCESS_MODE", "internal")

    if access_mode == "public":
        # Validate against whichever source the app will actually use.
        db_str = str(engine.url) if engine is not None else str(database_path or "")
        if "phase.db" in db_str and "phase_pds.db" not in db_str:
            raise ValueError(
                "Public mode must NOT use phase.db (contains Loupe data). "
                "Use phase_pds.db instead."
            )

    job_queue = JobQueue()
    map_registry = MapJobRegistry()
    map_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="map-fit")

    # Resolve engine eagerly so it's available before lifespan
    if engine is None:
        from sherloc_pipeline.database.connection import get_engine as _get_engine

        db_path = str(database_path) if database_path else None
        engine = _get_engine(db_path)

    # Resolve config eagerly
    if config is None:
        from sherloc_pipeline.config import get_config as _get_config

        config = _get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup/shutdown lifecycle."""
        import asyncio
        app.state.engine = engine
        app.state.config = config
        app.state.job_queue = job_queue
        app.state.map_registry = map_registry
        app.state.map_executor = map_executor
        app.state.access_mode = access_mode
        app.state.event_loop = asyncio.get_running_loop()
        logger.info("SHERLOC Web API started (access_mode=%s)", access_mode)
        if os.environ.get("SHERLOC_AUTH_MODE") == "dev":
            logger.warning(
                "SHERLOC_AUTH_MODE=dev: CF Access JWT validation is BYPASSED. "
                "All authenticated routes resolve to a hardcoded 'dev@local' "
                "identity. Do NOT enable this in production."
            )
        yield
        job_queue.shutdown()
        map_executor.shutdown(wait=False)
        logger.info("SHERLOC Web API stopped")

    app = FastAPI(
        title="SHERLOC Pipeline API",
        version=API_SCHEMA_VERSION,
        lifespan=lifespan,
    )

    # Eagerly set state so it's available even without lifespan (e.g. test clients)
    app.state.engine = engine
    app.state.config = config
    app.state.job_queue = job_queue
    app.state.map_registry = map_registry
    app.state.map_executor = map_executor
    app.state.access_mode = access_mode

    # CORS — origins are env-driven. Default empty (no cross-origin).
    # Set SHERLOC_CORS_ALLOWED_ORIGINS=https://example.com,https://other.example.com
    _cors_env = os.environ.get("SHERLOC_CORS_ALLOWED_ORIGINS", "")
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Public mode: rate limiting + compute guards
    # Register BEFORE db_session_middleware so it executes first (FastAPI middleware
    # runs in reverse registration order).
    if access_mode == "public":
        _rate_limiter = _RateLimiter(max_requests=10, window_seconds=60)

        @app.middleware("http")
        async def public_guards_middleware(request: Request, call_next):
            # Body size limit (256 KB)
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > 256 * 1024:
                return JSONResponse(
                    status_code=413,
                    content={
                        "schema_version": API_SCHEMA_VERSION,
                        "error": "payload_too_large",
                        "detail": "Request body exceeds 256 KB limit",
                    },
                )

            # Rate limit on processing endpoints
            if request.url.path.startswith("/api/process/"):
                client_ip = request.client.host if request.client else "unknown"
                if not _rate_limiter.is_allowed(client_ip):
                    return JSONResponse(
                        status_code=429,
                        content={
                            "schema_version": API_SCHEMA_VERSION,
                            "error": "rate_limited",
                            "detail": "Too many requests. Try again later.",
                        },
                        headers={"Retry-After": "60"},
                    )

            return await call_next(request)

    # Per-request DB session middleware
    @app.middleware("http")
    async def db_session_middleware(request: Request, call_next):
        """Attach a per-request DB session to request.state.db."""
        eng = request.app.state.engine
        from sherloc_pipeline.database.connection import get_session_factory

        factory = get_session_factory(eng)
        session = factory()
        request.state.db = session
        try:
            response = await call_next(request)
            session.commit()
            return response
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Error handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": "not_found",
                "detail": str(exc.detail) if hasattr(exc, "detail") else "Resource not found",
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        logger.exception("Internal server error")
        return JSONResponse(
            status_code=500,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": "internal_error",
                "detail": "An internal error occurred. Check server logs for details.",
            },
        )

    # Register route modules
    from sherloc_pipeline.web.routes import config as config_routes
    from sherloc_pipeline.web.routes import health as health_routes
    from sherloc_pipeline.web.routes import images as images_routes
    from sherloc_pipeline.web.routes import jobs as jobs_routes
    from sherloc_pipeline.web.routes import map as map_routes
    from sherloc_pipeline.web.routes import pds as pds_routes
    from sherloc_pipeline.web.routes import user as user_routes
    from sherloc_pipeline.web.routes import plots as plots_routes
    from sherloc_pipeline.web.routes import processing as processing_routes
    from sherloc_pipeline.web.routes import scans as scan_routes
    from sherloc_pipeline.web.routes import spectra as spectra_routes
    from sherloc_pipeline.web import ws as ws_module
    from sherloc_pipeline.web import ws_map as ws_map_module

    # Public endpoints — no per-request auth dependency.
    # /api/health is the load-balancer / monitoring probe; /api/config is
    # the SPA bootstrap call that runs BEFORE login to learn how to
    # initialise auth (§13.4).
    app.include_router(health_routes.router)
    app.include_router(config_routes.router)

    # Data API surface — every route enforces auth + role-per-API per
    # spec §13.3 + §13.3.7. CF Access mode is exempt from the role
    # check (CF Access has no role concept; §13.6); auth0/dev modes
    # both enforce. See web/auth.require_authenticated_request docstring
    # for the full G18.11.* sub-gate matrix.
    from fastapi import Depends
    from sherloc_pipeline.web.auth import require_authenticated_request

    auth_dep = [Depends(require_authenticated_request)]
    app.include_router(scan_routes.router, dependencies=auth_dep)
    app.include_router(spectra_routes.router, dependencies=auth_dep)
    app.include_router(processing_routes.router, dependencies=auth_dep)
    app.include_router(plots_routes.router, dependencies=auth_dep)
    app.include_router(pds_routes.router, dependencies=auth_dep)
    app.include_router(images_routes.router, dependencies=auth_dep)
    app.include_router(jobs_routes.router, dependencies=auth_dep)
    app.include_router(map_routes.router, dependencies=auth_dep)

    # user_routes does its own auth via _resolve_user; F4 follow-up
    # will unify it onto require_authenticated_request once the
    # sub-vs-email identity-key change lands.
    app.include_router(user_routes.router)

    # WebSocket routes use a per-job submitter_token (browsers cannot
    # set Authorization headers on the WS handshake). WS auth is a
    # separate concern and out of scope for the per-request HTTP dep.
    app.include_router(ws_module.router)
    app.include_router(ws_map_module.router)

    # Mount frontend static files (Svelte build output)
    _frontend_dist = (
        Path(frontend_dist)
        if frontend_dist is not None
        else Path(__file__).parent / "frontend" / "dist"
    )
    if _frontend_dist.is_dir():
        from fastapi.staticfiles import StaticFiles
        from starlette.responses import FileResponse

        # Serve static assets (JS, CSS, etc.)
        app.mount(
            "/assets",
            StaticFiles(directory=_frontend_dist / "assets"),
            name="frontend-assets",
        )

        # Resolve dist once; serve_spa uses it to enforce containment.
        _frontend_dist_resolved = _frontend_dist.resolve()
        _index_path = _frontend_dist / "index.html"
        _index_headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}

        # Catch-all: serve index.html for SPA routing.
        # Containment guard: %2e%2e-encoded `..` segments survive ASGI URL
        # decoding and reach this handler as literal `..`; Path joins do not
        # normalize them, so we resolve and verify the result stays inside
        # `_frontend_dist` before serving. Escape attempts fall through to
        # the SPA index — matching the existing "non-existent file" fallback.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            """Serve the Svelte SPA for all non-API routes."""
            if not full_path:
                return FileResponse(_index_path, headers=_index_headers)
            try:
                candidate = (_frontend_dist / full_path).resolve(strict=False)
                candidate.relative_to(_frontend_dist_resolved)
            except (ValueError, OSError):
                return FileResponse(_index_path, headers=_index_headers)
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(_index_path, headers=_index_headers)

    return app
