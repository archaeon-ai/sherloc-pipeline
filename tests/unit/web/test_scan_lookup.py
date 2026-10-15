"""Exact viewer identity lookup using invented scans in an in-memory database."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.database.connection import create_all_tables
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.web.app import create_app


@pytest.fixture
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)
    with Session(engine) as session:
        session.add_all([SolORM(sol_number=sol) for sol in (42424, 42425)])
        session.flush()
        for index, (name, sol, source) in enumerate([
            ("Testrock_detail", 42424, "loupe"),
            ("Testrock_detail", 42424, "pds4"),
            ("Testrock_detail", 42425, "pds4"),
            ("Testrock_detail_extra", 42424, "pds4"),
            ("testrock_detail", 42424, "pds4"),
            ("TestrockXdetail", 42424, "pds4"),
            ("Private_test_scan", 42424, "loupe"),
        ]):
            session.add(ScanORM(
                id=f"00000000-0000-0000-0000-{index + 1:012d}",
                scan_id=f"test-scan-{index}",
                scan_name=name,
                sol_number=sol,
                target="Testrock",
                data_source=source,
                n_points=3,
                sclk_start=100 + index,
                target_type="mars_target",
                scan_class="primary",
                scan_type="detail",
            ))
        session.commit()
    yield engine
    engine.dispose()


@pytest.mark.asyncio
async def test_lookup_returns_all_exact_matches(client):
    response = await client.get("/api/scans", params={"scan_name": "Testrock_detail", "sol": 42424})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [scan["scan_id"] for scan in body["scans"]] == ["test-scan-0", "test-scan-1"]


@pytest.mark.asyncio
async def test_lookup_by_other_sol_returns_one_match(client):
    response = await client.get("/api/scans", params={"scan_name": "Testrock_detail", "sol": 42425})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["scans"][0]["scan_id"] == "test-scan-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Absent_test_scan", "Testrock%", "Testrock", "TESTROCK_DETAIL"])
async def test_lookup_no_match_has_no_bootstrap_hint(client, name):
    response = await client.get("/api/scans", params={"scan_name": name})
    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["scans"] == []
    assert response.json()["message"] is None


@pytest.mark.asyncio
async def test_lookup_pagination_keeps_total_and_stable_order(client):
    pages = []
    for offset in (0, 1):
        response = await client.get("/api/scans", params={
            "scan_name": "Testrock_detail", "sol": 42424, "limit": 1, "offset": offset,
        })
        assert response.status_code == 200
        assert response.json()["total"] == 2
        pages.extend(response.json()["scans"])
    assert [scan["scan_id"] for scan in pages] == ["test-scan-0", "test-scan-1"]


@pytest.mark.asyncio
async def test_lookup_empty_name_is_invalid(client):
    response = await client.get("/api/scans", params={"scan_name": "", "sol": 42424})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_public_lookup_hides_private_matches(test_engine, fake_config, client):
    # The shared client fixture establishes isolated dev authentication.
    app = create_app(engine=test_engine, config=fake_config, access_mode="public")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as public:
        response = await public.get("/api/scans", params={"scan_name": "Testrock_detail", "sol": 42424})
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["scans"][0]["scan_id"] == "test-scan-1"
        response = await public.get("/api/scans", params={"scan_name": "Private_test_scan", "sol": 42424})
        assert response.status_code == 200
        assert response.json()["total"] == 0
        assert response.json()["scans"] == []
