"""FastAPI application factory for the SHERLOC Web API."""

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine

from sherloc_pipeline.web.config_check import VALID_ACCESS_MODES, access_tier_pairing_error
from sherloc_pipeline.web.jobs import JobQueue
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION
from sherloc_pipeline.web.ws_map import MapJobRegistry

logger = logging.getLogger(__name__)

PUBLIC_PROCESS_RATE_LIMIT = 60


class _RateLimiter:
    """Simple sliding-window rate limiter per IP."""

    def __init__(
        self,
        max_requests: int = PUBLIC_PROCESS_RATE_LIMIT,
        window_seconds: int = 60,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def retry_after(self, ip: str) -> int:
        """Record an allowed request (0), or return seconds until a slot opens."""
        with self._lock:
            now = self._clock()
            # Lazy eviction of stale IPs
            stale_ips = [k for k, v in self._requests.items() if v and now - v[-1] > 300]
            for k in stale_ips:
                del self._requests[k]

            times = self._requests.setdefault(ip, [])
            times[:] = [t for t in times if now - t < self._window]
            if len(times) >= self._max:
                return max(1, math.ceil(times[0] + self._window - now))
            times.append(now)
            return 0


# Ten thousand 64-character template IDs plus bounded metadata fit in 1 MiB.
IDENTIFY_BODY_LIMIT = 1024 * 1024


class _IdentifyBodyLimit:
    """Bound actual ASGI bytes before request parsing in either access tier."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") != "/api/identify":
            return await self.app(scope, receive, send)

        async def reject():
            response = JSONResponse(
                status_code=413,
                headers={"Cache-Control": "private, no-store"},
                content={"schema_version": API_SCHEMA_VERSION, "error": "payload_too_large",
                         "detail": "Identification request body exceeds 1 MiB limit"},
            )
            await response(scope, receive, send)

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > IDENTIFY_BODY_LIMIT:
                        return await reject()
                except ValueError:
                    pass
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > IDENTIFY_BODY_LIMIT:
                return await reject()
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        pending = True

        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


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

    # Fail closed: an unknown mode (env typo or explicit argument) would
    # otherwise skip the public-tier restrictions and serve as internal.
    if access_mode not in VALID_ACCESS_MODES:
        raise ValueError(
            f"access mode has unrecognized value: {access_mode!r} "
            f"(expected one of {sorted(VALID_ACCESS_MODES)})"
        )

    pairing_error = access_tier_pairing_error(access_mode)
    if pairing_error:
        raise ValueError(pairing_error)

    from sherloc_pipeline.web.feature_policy import load_feature_policy, load_public_release_index

    feature_policy = load_feature_policy(access_mode)

    if access_mode == "public":
        # Validate against whichever source the app will actually use.
        db_str = str(engine.url) if engine is not None else str(database_path or "")
        if "phase.db" in db_str and "phase_pds.db" not in db_str:
            raise ValueError(
                "Public mode must NOT use phase.db (contains Loupe data). "
                "Use phase_pds.db instead."
            )

    public_release_index = load_public_release_index(access_mode, feature_policy)

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
        if access_mode == "public" and public_release_index is None:
            logger.warning("strict release-index check OFF (no index configured)")
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
        # Served below behind the auth dependency, not anonymously.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Eagerly set state so it's available even without lifespan (e.g. test clients)
    app.state.engine = engine
    app.state.config = config
    app.state.job_queue = job_queue
    app.state.map_registry = map_registry
    app.state.map_executor = map_executor
    app.state.access_mode = access_mode
    from sherloc_pipeline.services.identification_runtime import IdentificationRuntime, ResponseCache

    app.state.identification_runtime = IdentificationRuntime(feature_policy.identification.max_concurrency)
    app.state.reference_cache = ResponseCache(capacity=4)
    app.state.feature_policy = feature_policy
    app.state.public_release_index = public_release_index

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
        _rate_limiter = _RateLimiter(max_requests=PUBLIC_PROCESS_RATE_LIMIT, window_seconds=60)

        @app.middleware("http")
        async def public_guards_middleware(request: Request, call_next):
            # Body size limit (256 KB)
            content_length = request.headers.get("content-length")
            if request.url.path.rstrip("/") != "/api/identify" and content_length and int(content_length) > 256 * 1024:
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
                # uvicorn rewrites the peer only when FORWARDED_ALLOW_IPS trusts it.
                client_ip = request.client.host if request.client else "unknown"
                retry_after = _rate_limiter.retry_after(client_ip)
                if retry_after:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "schema_version": API_SCHEMA_VERSION,
                            "error": "rate_limited",
                            "detail": "Too many requests. Try again later.",
                        },
                        headers={"Retry-After": str(retry_after)},
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
            headers={"Cache-Control": "private, no-store"} if (
                request.url.path == "/api/identify" or request.url.path.startswith("/api/references")
            ) else None,
            content={
                "schema_version": API_SCHEMA_VERSION,
                "error": "internal_error",
                "detail": "An internal error occurred. Check server logs for details.",
            },
        )

    @app.middleware("http")
    async def evidence_no_store(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/api/identify" or request.url.path.startswith("/api/references"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    # Registered last so byte admission wraps DB, auth and JSON parsing.
    app.add_middleware(_IdentifyBodyLimit)

    # Register route modules
    from sherloc_pipeline.web.routes import config as config_routes
    from sherloc_pipeline.web.routes import health as health_routes
    from sherloc_pipeline.web.routes import images as images_routes
    from sherloc_pipeline.web.routes import jobs as jobs_routes
    from sherloc_pipeline.web.routes import map as map_routes
    from sherloc_pipeline.web.routes import pds as pds_routes
    from sherloc_pipeline.web.routes import plots as plots_routes
    from sherloc_pipeline.web.routes import processing as processing_routes
    from sherloc_pipeline.web.routes import scans as scan_routes
    from sherloc_pipeline.web.routes import spectra as spectra_routes
    from sherloc_pipeline.web.routes import references as references_routes
    from sherloc_pipeline.web.routes import identification as identification_routes
    from sherloc_pipeline.web import ws as ws_module
    from sherloc_pipeline.web import ws_map as ws_map_module

    # Public endpoints — no per-request auth dependency.
    # /api/health is the load-balancer / monitoring probe (status and
    # version only when anonymous); /api/config/bootstrap is the SPA call
    # that runs BEFORE login to learn how to initialise auth (§13.4).
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
    app.include_router(config_routes.protected_router, dependencies=auth_dep)

    # API docs and schema: authenticated like the data API.
    from fastapi import HTTPException
    from fastapi.openapi.docs import get_swagger_ui_html

    @app.get("/openapi.json", include_in_schema=False, dependencies=auth_dep)
    def openapi_schema():
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=auth_dep)
    def swagger_ui():
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")

    # The other docs variants (ReDoc, the Swagger OAuth2 redirect,
    # trailing-slash forms) are retired: a 404, never the SPA shell. A
    # browser cannot attach the Auth0 bearer token to a top-level
    # navigation, so in auth0 mode /docs and /openapi.json are for
    # API clients that send the token, not for browsing.
    @app.get("/redoc", include_in_schema=False)
    @app.get("/redoc/{rest:path}", include_in_schema=False)
    @app.get("/docs/{rest:path}", include_in_schema=False)
    @app.get("/openapi.json/", include_in_schema=False)
    def retired_docs():
        raise HTTPException(status_code=404, detail="Not Found")

    app.include_router(scan_routes.router, dependencies=auth_dep)
    app.include_router(spectra_routes.router, dependencies=auth_dep)
    app.include_router(references_routes.router, dependencies=auth_dep)
    app.include_router(identification_routes.router, dependencies=auth_dep)
    app.include_router(processing_routes.router, dependencies=auth_dep)
    app.include_router(plots_routes.router, dependencies=auth_dep)
    app.include_router(pds_routes.router, dependencies=auth_dep)
    app.include_router(images_routes.router, dependencies=auth_dep)
    app.include_router(jobs_routes.router, dependencies=auth_dep)
    app.include_router(map_routes.router, dependencies=auth_dep)

    # /api/user/* (preferences, classification profiles, classification
    # defaults) is retired: no UI called it, and every authenticated call
    # wrote a users row. A 404 for every method, never the SPA shell. The
    # users / user_preferences / classification_profiles tables stay.
    @app.api_route(
        "/api/user",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    @app.api_route(
        "/api/user/{rest:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def retired_user_routes():
        raise HTTPException(status_code=404, detail="Not Found")

    # WebSocket routes authenticate the handshake themselves
    # (web/auth.authenticate_websocket): browsers cannot set an
    # Authorization header on a WebSocket, so the token rides in the
    # subprotocol. The per-job submitter_token only authorizes cancel.
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
