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


@pytest.mark.asyncio
async def test_health_n_spectra_counts_measured_spectra_only(client, test_engine):
    """Derived rows (dark_sub_despiked) are not counted as spectra."""
    import uuid

    from sherloc_pipeline.database.connection import get_session_factory
    from sherloc_pipeline.database.models import SpectrumORM

    session = get_session_factory(test_engine)()
    try:
        measured = session.query(SpectrumORM).count()
        parent = session.query(SpectrumORM).first()
        session.add(SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=parent.scan_point_id, region=parent.region,
            spectrum_type="dark_sub_despiked", processing_level="ml_despiked",
            intensities=parent.intensities,
        ))
        session.commit()
    finally:
        session.close()

    resp = await client.get("/api/health")
    assert resp.json()["checks"]["database"]["n_spectra"] == measured
