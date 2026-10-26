"""Public processing admission through uvicorn's proxy trust boundary."""

import os
from functools import partial
from unittest.mock import Mock

import pytest
from fastapi import Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from sherloc_pipeline.web import app as app_module
from sherloc_pipeline.web.schemas import API_SCHEMA_VERSION


@pytest.fixture
def app_factory(monkeypatch, fake_config):
    for key in list(os.environ):
        if key.startswith("SHERLOC_") or key in ("PHASE_TIER", "PHASE_DATABASE_PATH"):
            monkeypatch.delenv(key, raising=False)
    clock = Mock(return_value=1000.0)
    monkeypatch.setattr(app_module, "_RateLimiter", partial(app_module._RateLimiter, clock=clock))
    engine = create_engine("sqlite:///:memory:")
    apps = []

    def make_app(access_mode="public", trusted_hosts="10.99.0.1"):
        app = app_module.create_app(engine=engine, config=fake_config, access_mode=access_mode)
        apps.append(app)

        @app.post("/api/process/rate-limit-probe")
        async def probe(request: Request):
            return {"client": request.client.host}

        return ProxyHeadersMiddleware(app, trusted_hosts=trusted_hosts)

    yield make_app, clock
    for app in apps:
        app.state.job_queue.shutdown()
        app.state.map_executor.shutdown(wait=False)
    engine.dispose()


def _client(app, peer="10.99.0.1"):
    return AsyncClient(
        transport=ASGITransport(app=app, client=(peer, 12345)), base_url="http://synthetic"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("trusted_hosts", ["10.99.0.1", "10.99.0.0/24"])
async def test_trusted_proxy_separates_visitors(app_factory, trusted_hosts):
    make_app, _ = app_factory
    async with _client(make_app(trusted_hosts=trusted_hosts)) as client:
        for _ in range(60):
            response = await client.post(
                "/api/process/rate-limit-probe", headers={"X-Forwarded-For": "192.0.2.1"}
            )
            assert response.status_code == 200
            assert response.json()["client"] == "192.0.2.1"
        assert (await client.post(
            "/api/process/rate-limit-probe", headers={"X-Forwarded-For": "192.0.2.1"}
        )).status_code == 429
        response = await client.post(
            "/api/process/rate-limit-probe", headers={"X-Forwarded-For": "192.0.2.2"}
        )
        assert response.status_code == 200
        assert response.json()["client"] == "192.0.2.2"


@pytest.mark.asyncio
async def test_untrusted_peer_cannot_spoof_separate_buckets(app_factory):
    make_app, _ = app_factory
    async with _client(make_app(), peer="10.98.0.1") as client:
        for _ in range(60):
            response = await client.post(
                "/api/process/rate-limit-probe", headers={"X-Forwarded-For": "192.0.2.1"}
            )
            assert response.status_code == 200
            assert response.json()["client"] == "10.98.0.1"
        for visitor in ("192.0.2.1", "192.0.2.2"):
            response = await client.post(
                "/api/process/rate-limit-probe", headers={"X-Forwarded-For": visitor}
            )
            assert response.status_code == 429


@pytest.mark.asyncio
async def test_retry_after_tracks_oldest_request_expiry(app_factory):
    make_app, clock = app_factory
    async with _client(make_app()) as client:
        assert (await client.post("/api/process/rate-limit-probe")).status_code == 200
        clock.return_value = 1000.75
        for _ in range(59):
            assert (await client.post("/api/process/rate-limit-probe")).status_code == 200

        for now, expected in ((1000.75, 60), (1017.2, 43), (1059.75, 1)):
            clock.return_value = now
            response = await client.post("/api/process/rate-limit-probe")
            assert response.status_code == 429
            assert response.json() == {
                "schema_version": API_SCHEMA_VERSION,
                "error": "rate_limited",
                "detail": "Too many requests. Try again later.",
            }
            assert response.headers["Retry-After"] == str(expected)
            assert 1 <= int(response.headers["Retry-After"]) <= 60

        # Only the oldest slot expires at 1060; the other 59 expire at 1060.75.
        clock.return_value = 1060.0
        assert (await client.post("/api/process/rate-limit-probe")).status_code == 200
        response = await client.post("/api/process/rate-limit-probe")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "1"


@pytest.mark.asyncio
async def test_internal_mode_has_no_processing_limit(app_factory):
    make_app, clock = app_factory
    async with _client(make_app(access_mode="internal")) as client:
        for _ in range(61):
            assert (await client.post("/api/process/rate-limit-probe")).status_code == 200
    clock.assert_not_called()
