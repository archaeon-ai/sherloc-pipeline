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


def _grow_spectra(engine, n):
    """Copy the fixture's first spectrum n times (synthetic rows only)."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(
            "WITH RECURSIVE k(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM k WHERE i < :n) "
            "INSERT INTO spectra (id, scan_point_id, region, spectrum_type, processing_level, "
            "intensities, created_at) "
            "SELECT 'synthetic-' || k.i, s.scan_point_id, s.region, s.spectrum_type, "
            "s.processing_level, s.intensities, s.created_at "
            "FROM k, (SELECT * FROM spectra LIMIT 1) AS s"
        ), {"n": n})


@pytest_asyncio.fixture
async def anonymous_client(test_engine, fake_config, monkeypatch):
    """A cf-access app hit with no credential: what docker and the sync scripts send."""
    from httpx import ASGITransport, AsyncClient

    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    monkeypatch.setenv("SHERLOC_AUTH_MODE", "cf-access")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", "synthetic.cloudflareaccess.invalid")
    monkeypatch.setenv("SHERLOC_CF_AUDIENCE", "synthetic-aud")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    _reset_validator_for_tests()
    app = create_app(engine=test_engine, config=fake_config)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    _reset_validator_for_tests()


def _record_sql(engine):
    from sqlalchemy import event

    seen = []

    def _before(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _before)
    return seen, lambda: event.remove(engine, "before_cursor_execute", _before)


@pytest.mark.asyncio
async def test_anonymous_health_pings_without_counting(anonymous_client, test_engine):
    """Anonymous health is status + version from a one-row ping, never a table count,
    however big the spectra table is."""
    _grow_spectra(test_engine, 50_000)
    seen, stop = _record_sql(test_engine)
    try:
        resp = await anonymous_client.get("/api/health")
    finally:
        stop()
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok" and data["checks"] == {} and data["pipeline_version"]
    lowered = [s.lower() for s in seen]
    assert not [s for s in lowered if "count(" in s], seen
    assert not [s for s in lowered if "spectra" in s or "scans" in s], seen
    assert any("select 1" in s for s in lowered), seen


@pytest.mark.asyncio
async def test_anonymous_health_reports_a_dead_database(anonymous_client, test_engine, monkeypatch):
    from sqlalchemy.orm import Session

    def _fail(self, *args, **kwargs):
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(Session, "execute", _fail)
    resp = await anonymous_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "error"


@pytest.mark.asyncio
async def test_detailed_health_never_counts_spectra(client, test_engine):
    """Signed-in internal health keeps the cheap scans count the team
    DB-refresh runbooks read, and never counts spectra (#154)."""
    _grow_spectra(test_engine, 10)
    seen, stop = _record_sql(test_engine)
    try:
        resp = await client.get("/api/health")
    finally:
        stop()
    db = resp.json()["checks"]["database"]
    assert db["status"] == "ok" and db["n_scans"] == 1 and db.get("n_spectra") is None
    counts = [s.lower() for s in seen if "count(" in s.lower()]
    assert counts and all("from scans" in s and "spectra" not in s for s in counts), counts
