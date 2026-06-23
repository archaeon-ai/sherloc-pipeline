"""PDS route tests (catalog, download, available-sols)."""

from unittest.mock import MagicMock, patch

import pytest

from sherloc_pipeline.web.routes.pds import _clear_catalog_cache


@pytest.fixture(autouse=True)
def _reset_catalog_cache():
    """Ensure PDS catalog cache is clean between tests."""
    _clear_catalog_cache()
    yield
    _clear_catalog_cache()


# ---------------------------------------------------------------------------
# GET /api/pds/available-sols (legacy stub)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_available_sols(client):
    """Legacy endpoint returns already-ingested sols."""
    resp = await client.get("/api/pds/available-sols")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["total"] == 0
    assert data["sols"] == []
    # The test fixture has sol 921 ingested
    assert 921 in data["already_ingested"]


# ---------------------------------------------------------------------------
# GET /api/pds/catalog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pds_catalog_success(client):
    """Catalog endpoint returns PDS sols and ingested overlay."""
    mock_downloader = MagicMock()
    mock_downloader.discover_available_sols.return_value = [100, 200, 300, 921]

    with patch(
        "sherloc_pipeline.core.pds_client.PDSDownloader"
    ) as MockCls:
        MockCls.from_config.return_value = mock_downloader
        resp = await client.get("/api/pds/catalog")

    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["total_available"] == 4
    assert len(data["available_sols"]) == 4
    assert data["available_sols"][0]["sol"] == 100
    assert 921 in data["already_ingested"]


@pytest.mark.asyncio
async def test_pds_catalog_uses_cache(client):
    """Second call uses cache, not a fresh PDS fetch."""
    mock_downloader = MagicMock()
    mock_downloader.discover_available_sols.return_value = [500, 600]

    with patch(
        "sherloc_pipeline.core.pds_client.PDSDownloader"
    ) as MockCls:
        MockCls.from_config.return_value = mock_downloader

        # First call — populates cache
        resp1 = await client.get("/api/pds/catalog")
        assert resp1.status_code == 200

        # Second call — should use cache
        resp2 = await client.get("/api/pds/catalog")
        assert resp2.status_code == 200

    # PDSDownloader should be constructed only once
    assert MockCls.from_config.call_count == 1


@pytest.mark.asyncio
async def test_pds_catalog_network_failure(client):
    """502 when PDS fetch fails."""
    mock_downloader = MagicMock()
    mock_downloader.discover_available_sols.side_effect = RuntimeError("network down")

    with patch(
        "sherloc_pipeline.core.pds_client.PDSDownloader"
    ) as MockCls:
        MockCls.from_config.return_value = mock_downloader
        resp = await client.get("/api/pds/catalog")

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /api/pds/download
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pds_download_submits_job(client):
    """Download endpoint returns 202 with a job_id."""
    # Sol 999 is not ingested, so it should succeed
    resp = await client.post(
        "/api/pds/download", json={"sol": 999}
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert "job_id" in data
    assert data["sol"] == 999
    assert data["status"] == "running"  # starts immediately (empty queue)
    assert "submitter_token" in data


@pytest.mark.asyncio
async def test_pds_download_conflict_existing_sol(client):
    """409 when sol is already ingested and force_reingest is false."""
    resp = await client.post(
        "/api/pds/download", json={"sol": 921, "force_reingest": False}
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_pds_download_force_reingest(client):
    """force_reingest=true bypasses the conflict check."""
    resp = await client.post(
        "/api/pds/download", json={"sol": 921, "force_reingest": True}
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["sol"] == 921


# ---------------------------------------------------------------------------
# Phase D new tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pds_download_returns_403_in_public_mode(monkeypatch):
    """Public mode blocks PDS download endpoint with 403."""
    from httpx import ASGITransport, AsyncClient
    from tests.unit.web.test_data_access import _make_public_client

    app, engine = _make_public_client(monkeypatch)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/pds/download", json={"sol": 100})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_job_status_endpoint(client):
    """GET /api/jobs/{job_id} returns status fields after submitting a download job."""
    resp = await client.post("/api/pds/download", json={"sol": 777})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status_resp = await client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert "schema_version" in data
    assert data["job_id"] == job_id
    assert "status" in data


@pytest.mark.asyncio
async def test_get_job_status_404(client):
    """GET /api/jobs/{nonexistent} returns 404."""
    resp = await client.get("/api/jobs/nonexistent_job_id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_single_flight_same_sol(client):
    """Submitting a download for the same sol twice returns the same job_id."""
    import threading

    blocker = threading.Event()

    # Use sol 888 (not yet ingested in fixture)
    resp1 = await client.post("/api/pds/download", json={"sol": 888})
    assert resp1.status_code == 202
    job_id_1 = resp1.json()["job_id"]

    # Second submit for same sol should return same job_id
    resp2 = await client.post("/api/pds/download", json={"sol": 888})
    assert resp2.status_code == 202
    job_id_2 = resp2.json()["job_id"]

    assert job_id_1 == job_id_2


# ---------------------------------------------------------------------------
# Feature flag: PDS endpoints 404 when env-disabled (issue #21)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pds_catalog_404_when_feature_disabled(client, monkeypatch):
    """Direct GET on /api/pds/catalog returns 404 when the env flag is off."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "disabled")
    resp = await client.get("/api/pds/catalog")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pds_available_sols_404_when_feature_disabled(client, monkeypatch):
    """Direct GET on /api/pds/available-sols returns 404 when the env flag is off."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "disabled")
    resp = await client.get("/api/pds/available-sols")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pds_download_404_when_feature_disabled(client, monkeypatch):
    """POST /api/pds/download returns 404 when the env flag is off
    (precedes the public-mode 403 check — feature absence beats access mode)."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "disabled")
    resp = await client.post("/api/pds/download", json={"sol": 999})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pds_catalog_still_works_when_feature_enabled(client, monkeypatch):
    """Sanity: explicit `enabled` leaves /api/pds/catalog reachable."""
    monkeypatch.setenv("SHERLOC_FEATURE_PDS_BROWSER", "enabled")
    mock_downloader = MagicMock()
    mock_downloader.discover_available_sols.return_value = [42]
    with patch("sherloc_pipeline.core.pds_client.PDSDownloader") as MockCls:
        MockCls.from_config.return_value = mock_downloader
        resp = await client.get("/api/pds/catalog")
    assert resp.status_code == 200
