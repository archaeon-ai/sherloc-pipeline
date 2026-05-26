"""§12.2 empty-DB UX tests.

Per spec §12.2, when the SHERLOC database has zero scans visible to the
active access mode AND the caller applied no filters, the `/api/scans`
response includes a `message` field hinting at `sherloc init --help`.
This signal lets the SPA render an onboarding panel instead of the
generic "no scans found" empty-table state.

Filtered queries that return zero rows (e.g., `?sol=999` against a
populated DB with no sol 999) are NOT empty-DB and must NOT carry the
message — the caller filtered to empty intentionally.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sherloc_pipeline.database.connection import create_all_tables, get_engine
from sherloc_pipeline.database.models import ScanORM, SolORM
from sherloc_pipeline.web.app import create_app


EXPECTED_HINT_FRAGMENT = "sherloc init"


@pytest.fixture
def empty_db_client(tmp_path, monkeypatch):
    # Dev auth mode bypasses token validation per §13.5; B.12 F1 added
    # router-level auth on /api/scans, so these route-logic tests run
    # in dev mode (auth-gating itself is covered by test_scans_auth.py).
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    from sherloc_pipeline.web.auth import _reset_validator_for_tests
    _reset_validator_for_tests()
    db_path = tmp_path / "empty.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    with TestClient(create_app(engine=engine)) as client:
        yield client
    _reset_validator_for_tests()


@pytest.fixture
def populated_db_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SHERLOC_AUTH_MODE", "dev")
    monkeypatch.setenv("SHERLOC_ACCESS_MODE", "internal")
    monkeypatch.delenv("SHERLOC_AUTH0_DOMAIN", raising=False)
    monkeypatch.delenv("SHERLOC_CF_TEAM_DOMAIN", raising=False)
    from sherloc_pipeline.web.auth import _reset_validator_for_tests
    _reset_validator_for_tests()
    db_path = tmp_path / "populated.db"
    engine = get_engine(str(db_path))
    create_all_tables(engine)
    # Seed a single PDS-source scan so it's visible in both internal and
    # public access modes (Loupe-source rows would be filtered out in
    # public mode and confuse the test).
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.add(SolORM(sol_number=921, data_source="pds"))
        session.add(
            ScanORM(
                id="test-scan-1",
                sol_number=921,
                target="Quartier",
                scan_name="quartier_detail_1",
                scan_id="ss_0921_quartier_detail_1",
                sclk_start=789012345,
                n_points=100,
                n_channels=2148,
                laser_wavelength_nm=248.6,
                scan_class="primary",
                scan_type="detail",
                target_type="mars_target",
                data_source="pds",
            )
        )
        session.commit()
    with TestClient(create_app(engine=engine)) as client:
        yield client


def test_empty_db_unfiltered_includes_onboarding_message(empty_db_client):
    resp = empty_db_client.get("/api/scans")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scans"] == []
    assert body["total"] == 0
    assert body.get("message") is not None
    assert EXPECTED_HINT_FRAGMENT in body["message"]
    # Existing pagination shape preserved.
    assert body["offset"] == 0
    assert body["limit"] == 50
    assert "schema_version" in body


def test_empty_db_with_filter_omits_message(empty_db_client):
    # Spec §12.2 only fires on the unfiltered empty path; a filtered
    # zero-result is "no match", not "DB empty".
    resp = empty_db_client.get("/api/scans", params={"sol": 921})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scans"] == []
    assert body["total"] == 0
    assert body.get("message") is None


def test_populated_db_unfiltered_omits_message(populated_db_client):
    resp = populated_db_client.get("/api/scans")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] >= 1
    assert body.get("message") is None


def test_populated_db_filter_to_empty_omits_message(populated_db_client):
    # Filter that matches nothing in a populated DB: still no message
    # (caller filtered to empty intentionally).
    resp = populated_db_client.get("/api/scans", params={"sol": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["scans"] == []
    assert body["total"] == 0
    assert body.get("message") is None
