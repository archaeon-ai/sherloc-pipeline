"""Health endpoint tests."""

import pytest
import pytest_asyncio


@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "pipeline_version" in data
    assert "database" in data["checks"]
    assert data["checks"]["database"]["status"] == "ok"
    assert data["checks"]["database"]["n_scans"] == 1


@pytest.mark.asyncio
async def test_health_has_job_queue(client):
    resp = await client.get("/api/health")
    data = resp.json()
    jq = data["checks"]["job_queue"]
    assert jq["status"] == "ok"
    assert jq["running"] == 0
    assert jq["queued"] == 0
    assert jq["max_depth"] == 3
