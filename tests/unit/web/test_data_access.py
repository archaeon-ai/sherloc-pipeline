"""DataAccessService and access-mode endpoint tests."""

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sherloc_pipeline.web.data_access import DataAccessService
from tests.unit.web.conftest import SCAN_UUID, SOL_NUMBER
from tests.unit.public_release_helpers import synthetic_index


def synthetic_release_metadata(product_id, index):
    return {"version": "synthetic-v1", "source_sha256": "a" * 64, "public_release": {
        "schema_version": "public-release-1", "eligible": True,
        "product_id": product_id, "version_id": "synthetic-v1",
        "source_sha256": "a" * 64, "index_sha256": index.sha256,
        "release_id": "synthetic-release",
    }}


SCAN_SOURCE_CASES = [
    pytest.param({"data_source": "pds4"}, id="pds4"),
    pytest.param({"data_source": "loupe"}, id="loupe"),
    pytest.param({"data_source": None}, id="null"),
    pytest.param({}, id="missing"),
    pytest.param({"data_source": "PDS4"}, id="uppercase"),
    pytest.param({"data_source": "invented-private"}, id="unknown"),
]


# ---------------------------------------------------------------------------
# Unit tests for DataAccessService
# ---------------------------------------------------------------------------


class TestDataAccessService:
    """Pure unit tests (no HTTP, no DB)."""

    def test_default_mode_is_internal(self):
        svc = DataAccessService()
        assert svc.access_mode == "internal"
        assert not svc.is_public

    def test_public_mode(self):
        svc = DataAccessService(access_mode="public")
        assert svc.access_mode == "public"
        assert svc.is_public

    def test_internal_mode_does_not_filter(self):
        """In internal mode, filter_scans_query returns the query unchanged."""
        from unittest.mock import MagicMock

        svc = DataAccessService(access_mode="internal")
        mock_query = MagicMock()
        result = svc.filter_scans_query(mock_query)
        # Should return the same object — no .filter() called
        assert result is mock_query
        mock_query.filter.assert_not_called()

    def test_public_mode_applies_source_filter(self):
        """In public mode, a .filter() call is applied."""
        from unittest.mock import MagicMock

        svc = DataAccessService(access_mode="public")
        mock_query = MagicMock()
        result = svc.filter_scans_query(mock_query)
        mock_query.filter.assert_called_once()
        # Returns the chained result
        assert result is mock_query.filter.return_value


# ---------------------------------------------------------------------------
# Integration: access-mode endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_access_mode_default(client):
    """Default access_mode is 'internal'."""
    resp = await client.get("/api/config/access-mode")
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert data["access_mode"] == "internal"


# ---------------------------------------------------------------------------
# Integration: public mode blocks Loupe scans
# ---------------------------------------------------------------------------


def _make_public_client(monkeypatch, tmp_path):
    index = synthetic_index(tmp_path, monkeypatch, ["synthetic-public-product"])
    """Create an async client with access_mode='public'.

    Runs in dev auth mode so route-logic tests do not need to mint
    tokens; B.12 F1 router-level auth dependency rejects unauthenticated
    requests on every data route. The dedicated auth tests
    (test_scans_auth.py et al.) cover the auth-gating behaviour.
    """
    import pytest_asyncio
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    from sherloc_pipeline.database.connection import (
        create_all_tables,
        get_session_factory,
    )
    from sherloc_pipeline.database.models import ScanORM, SolORM
    from sherloc_pipeline.web.app import create_app
    from sherloc_pipeline.web.auth import _reset_validator_for_tests

    if monkeypatch is not None:
        monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
        monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
        monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
        monkeypatch.delenv("SHERLOC_CF_AUDIENCE", raising=False)
    _reset_validator_for_tests()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    create_all_tables(engine)

    factory = get_session_factory(engine)
    session = factory()
    try:
        # Sols
        session.add(SolORM(sol_number=100, data_source="loupe"))
        session.add(SolORM(sol_number=200, data_source="pds4"))
        session.flush()

        # Loupe scan
        session.add(
            ScanORM(
                id=str(uuid.uuid4()),
                sol_number=100,
                scan_name="detail_1",
                target="Target_A",
                scan_id="100_Target_A_detail_1",
                sclk_start=100000,
                n_points=5,
                n_channels=7,
                laser_wavelength_nm=211.0,
                data_source="loupe",
                target_type="mars_target",
                scan_class="primary",
            )
        )

        # PDS scan
        session.add(
            ScanORM(
                id=str(uuid.uuid4()),
                sol_number=200,
                scan_name="detail_2",
                target="Target_B",
                scan_id="synthetic-public-product",
                pds4_metadata=synthetic_release_metadata("synthetic-public-product", index),
                sclk_start=200000,
                n_points=3,
                n_channels=7,
                laser_wavelength_nm=211.0,
                data_source="pds4",
                target_type="mars_target",
                scan_class="primary",
            )
        )
        session.commit()
    finally:
        session.close()

    from tests.unit.test_tier_boundary import fake_config

    app = create_app(engine=engine, config=fake_config(), access_mode="public")
    return app, engine


@pytest.mark.asyncio
async def test_public_mode_hides_loupe_scans(monkeypatch, tmp_path):
    """In public mode, only PDS scans are visible."""
    from httpx import ASGITransport, AsyncClient

    app, engine = _make_public_client(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/scans")
        data = resp.json()
        assert data["total"] == 1
        assert data["scans"][0]["target"] == "Target_B"
        assert data["scans"][0]["sol_number"] == 200


@pytest.mark.asyncio
async def test_public_mode_access_mode_endpoint(monkeypatch, tmp_path):
    """access-mode endpoint reflects 'public'."""
    from httpx import ASGITransport, AsyncClient

    app, engine = _make_public_client(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/config/access-mode")
        data = resp.json()
        assert data["access_mode"] == "public"


@pytest.mark.asyncio
async def test_internal_mode_shows_all_scans(client):
    """In internal (default) mode, all scans are visible."""
    resp = await client.get("/api/scans")
    data = resp.json()
    # The default fixture has one Loupe scan
    assert data["total"] == 1
    assert data["scans"][0]["id"] == SCAN_UUID


# ---------------------------------------------------------------------------
# Phase D new tests: DataAccessService.validate_scan_access
# ---------------------------------------------------------------------------


class TestValidateScanAccess:
    @pytest.mark.parametrize("source_attributes", SCAN_SOURCE_CASES)
    def test_validate_scan_access_public_allowlist(self, source_attributes, tmp_path, monkeypatch):
        index = synthetic_index(tmp_path, monkeypatch, ["synthetic-product"])
        svc = DataAccessService(access_mode="public", public_release_index=index)
        scan = SimpleNamespace(
            scan_id="synthetic-product",
            pds4_metadata=synthetic_release_metadata("synthetic-product", index),
            **source_attributes,
        )
        if source_attributes.get("data_source") == "pds4":
            svc.validate_scan_access(scan)
            return

        with pytest.raises(HTTPException) as exc_info:
            svc.validate_scan_access(scan)
        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == "This resource is not available in public mode"

    @pytest.mark.parametrize("source_attributes", SCAN_SOURCE_CASES)
    def test_validate_scan_access_allows_all_in_internal(self, source_attributes):
        svc = DataAccessService(access_mode="internal")
        svc.validate_scan_access(SimpleNamespace(**source_attributes))


@pytest.fixture
def allowlist_session(tmp_path, monkeypatch):
    release_index = synthetic_index(tmp_path, monkeypatch, [f"invented-allowlist-{name}" for name in (
        "pds4", "loupe", "null", "missing", "uppercase", "unknown",
    )])
    """Metadata-only rows using the existing in-memory web-test DB helpers."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    from sherloc_pipeline.database.connection import (
        create_all_tables,
        get_session_factory,
    )
    from sherloc_pipeline.database.models import ScanORM, SolORM

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    create_all_tables(engine)
    session = get_session_factory(engine)()
    try:
        session.add(SolORM(sol_number=42424, data_source="pds4"))
        session.flush()
        for index, (name, source_attributes) in enumerate([
            ("pds4", {"data_source": "pds4"}),
            ("loupe", {"data_source": "loupe"}),
            ("null", {"data_source": None}),
            ("missing", {}),
            ("uppercase", {"data_source": "PDS4"}),
            ("unknown", {"data_source": "invented-private"}),
        ]):
            session.add(ScanORM(
                id=str(uuid.UUID(int=index + 1000)),
                sol_number=42424,
                scan_name=name,
                scan_id=f"invented-allowlist-{name}",
                pds4_metadata=synthetic_release_metadata(f"invented-allowlist-{name}", release_index),
                sclk_start=index,
                n_points=0,
                n_channels=1,
                laser_wavelength_nm=1.0,
                target_type="mars_target",
                scan_class="primary",
                **source_attributes,
            ))
        # An omitted ORM source becomes SQL NULL, just like explicit None.
        session.commit()
        yield session, release_index
    finally:
        session.close()
        engine.dispose()


@pytest.mark.parametrize("access_mode", ["public", "internal"])
def test_filter_scans_query_source_allowlist(allowlist_session, access_mode):
    from sherloc_pipeline.database.models import ScanORM

    allowlist_session, index = allowlist_session
    rows = allowlist_session.query(ScanORM).all()
    assert len(rows) == 6
    assert {row.scan_name for row in rows if row.data_source is None} == {"null", "missing"}

    query = allowlist_session.query(ScanORM)
    visible = DataAccessService(access_mode, index).filter_scans_query(query).all()
    expected = {"pds4"} if access_mode == "public" else {
        "pds4", "loupe", "null", "missing", "uppercase", "unknown",
    }
    assert {row.scan_name for row in visible} == expected


# ---------------------------------------------------------------------------
# Phase D new tests: public mode endpoint gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_public_mode_scan_detail_blocked(monkeypatch, tmp_path):
    """Public mode returns 403 for Loupe scan detail endpoint."""
    from httpx import ASGITransport, AsyncClient

    app, engine = _make_public_client(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Get the scan list to find the Loupe scan ID
        scans_resp = await ac.get("/api/scans?data_source=loupe")
        # In public mode, Loupe scans are hidden — fetch direct from DB instead
        # We need to look up the loupe scan's ID from the DB
        from sqlalchemy.orm import Session
        from sherloc_pipeline.database.models import ScanORM

        with Session(engine) as session:
            loupe_scan = session.query(ScanORM).filter(
                ScanORM.data_source == "loupe"
            ).first()
            loupe_id = loupe_scan.id

        resp = await ac.get(f"/api/scans/{loupe_id}")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_public_mode_scan_detail_pds_allowed(monkeypatch, tmp_path):
    """Public mode returns 200 for PDS scan detail endpoint."""
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.orm import Session
    from sherloc_pipeline.database.models import ScanORM

    app, engine = _make_public_client(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)

    with Session(engine) as session:
        pds_scan = session.query(ScanORM).filter(
            ScanORM.data_source == "pds4"
        ).first()
        pds_id = pds_scan.id

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(f"/api/scans/{pds_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_public_mode_download_blocked(monkeypatch, tmp_path):
    """Public mode blocks PDS download endpoint."""
    from httpx import ASGITransport, AsyncClient

    app, engine = _make_public_client(monkeypatch, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/pds/download", json={"sol": 100})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Phase D new tests: create_app validation
# ---------------------------------------------------------------------------


def test_create_app_rejects_phase_db_in_public():
    """create_app raises ValueError when public mode + phase.db path."""
    from sherloc_pipeline.web.app import create_app

    with pytest.raises(ValueError, match="phase.db"):
        create_app(database_path="./phase.db", access_mode="public")


def test_create_app_allows_phase_pds_db_in_public(tmp_path, monkeypatch):
    synthetic_index(tmp_path, monkeypatch)
    """create_app succeeds when public mode + phase_pds.db path."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sherloc_pipeline.database.connection import create_all_tables
    from sherloc_pipeline.web.app import create_app
    from tests.unit.test_tier_boundary import fake_config

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)

    # Should not raise
    app = create_app(
        database_path="./phase_pds.db",
        access_mode="public",
        engine=engine,
        config=fake_config(),
    )
    assert app is not None
