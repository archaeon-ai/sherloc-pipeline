"""Tests for ``web/routes/map.py`` — R2-mode failure-mode propagation.

v4.1.9 adds the Loupe-workspace R2 reader; ``get_map_layers`` +
``start_map_fit`` inject ``r2_reader.get_working_file`` when
``is_r2_mode()``. Per m2020-phase platform spec §3.9.8.3, the
``CoordinatesUnavailableError → 400`` mapping in the route handler
applies ONLY to R2-404 (the resolver wraps it as a coords-unavailable
error with the "Loupe workspace files not found in R2" message). All
other §3.9.4 failure modes (500 misconfigured_path, 502
upstream_credential_error, 504 upstream_timeout) propagate identically
through the route.

These tests force R2 mode + monkeypatch ``get_working_file`` to raise
the various HTTPException flavors, then assert the
``/api/map/layers/<id>`` response preserves the status code.

Reference: Codex PR #11 R1 F2 — closes the "future broad except Exception
in the route could silently regress propagation" gap by exercising the
contract end-to-end.
"""

from __future__ import annotations

import uuid
from typing import Optional

import pytest
from fastapi import HTTPException

from sherloc_pipeline.database.connection import get_session_factory
from sherloc_pipeline.database.models import (
    ContextImageORM,
    ScanPointORM,
)
from tests.unit.web.conftest import SCAN_UUID

# Same per-tier file_path convention as test_r2_reader.py / test_coordinates.py.
_TEAM_FILE_PATH = (
    "/data" "/sherloc/data/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)


@pytest.fixture()
def scanner_workspace_scan(test_engine):
    """Adapt the conftest's SCAN_UUID + 3 points to coordinate_frame='scanner_workspace'.

    The conftest's ``test_engine`` fixture creates points with x_pixel /
    y_pixel set but coordinate_frame=NULL (per the column default). For
    these tests we need the scan to route through
    ``_resolve_scanner_workspace`` so the injected workspace_reader gets
    called. This fixture mutates the existing points + adds an ACI row.
    """
    factory = get_session_factory(test_engine)
    session = factory()
    try:
        session.query(ScanPointORM).filter(
            ScanPointORM.scan_id == SCAN_UUID
        ).update(
            {"coordinate_frame": "scanner_workspace", "x_pixel": None, "y_pixel": None}
        )
        session.add(
            ContextImageORM(
                id=str(uuid.uuid4()),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=_TEAM_FILE_PATH,
            )
        )
        session.commit()
    finally:
        session.close()
    return test_engine


@pytest.fixture()
def force_r2_mode(monkeypatch):
    """Force ``is_r2_mode()`` True so the route layer injects ``get_working_file``."""
    monkeypatch.setenv("PHASE_TIER", "team")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://moto.invalid")


# ---------------------------------------------------------------------------
# /api/map/layers/<id> — failure-mode propagation in R2 mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_layers_propagates_404_as_400_via_coords_unavailable(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """R2-404 → CoordinatesUnavailableError → route 400 (the user-facing banner path).

    Per Codex PR #11 R2 F4 + spec §3.9.8.3 — the 400 response MUST name
    the derived R2 key so operators can distinguish an ingestion gap
    from a key-derivation issue.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=404, detail="not_found")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
    assert resp.status_code == 400
    body = resp.json()
    detail = body["detail"]
    assert "Loupe workspace file not found in R2" in detail
    # F4 — the derived R2 key + named missing file must appear.
    assert "expected_key=" in detail
    assert "sherloc-aci/" in detail
    assert "missing_file='spatial.csv'" in detail


@pytest.mark.asyncio
async def test_map_layers_preserves_502_credential_error(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """R2-403 → reader 502 → route preserves 502 (not wrapped as 400).

    Spec §3.9.8.3: misconfigured_path 500, 403→502, timeout→504 propagate
    identically; only R2-404 maps through CoordinatesUnavailableError.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=502, detail="upstream_credential_error")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "upstream_credential_error"


@pytest.mark.asyncio
async def test_map_layers_preserves_504_timeout(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """boto3 timeout → reader 504 → route preserves 504."""
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=504, detail="upstream_timeout")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
    assert resp.status_code == 504
    assert resp.json()["detail"] == "upstream_timeout"


@pytest.mark.asyncio
async def test_map_layers_preserves_500_misconfigured_path(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """Misconfigured per-tier strip-prefix → 500 misconfigured_path → route preserves 500."""
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=500, detail="misconfigured_path")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "misconfigured_path"


@pytest.mark.asyncio
async def test_map_layers_phase_tier_set_aws_missing_returns_tier_unset(
    client, scanner_workspace_scan, monkeypatch
):
    """PHASE_TIER set + AWS_* missing → /api/map/layers returns 500 tier_unset.

    Per Codex PR #11 R2 F3 + spec §3.9.4 + §3.9.8.5: any container that
    has PHASE_TIER set is a production deployment and MUST route through
    the R2 path. Missing AWS_* env in such a container is a production
    misconfiguration; the request surface MUST fail loudly as
    ``tier_unset`` — NOT silently fall back to local FS.

    The real ``get_working_file`` is invoked (no monkeypatch), so
    ``get_r2_client_and_config`` runs and raises the spec-defined error.
    """
    # Force the spec-defined misconfiguration: tier set, AWS empty.
    monkeypatch.setenv("PHASE_TIER", "team")
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(key, raising=False)
    # Ensure no stale cached client from a prior test.
    from sherloc_pipeline.web import r2_reader
    r2_reader.reset_r2_client_for_tests()

    resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
    assert resp.status_code == 500
    assert resp.json()["detail"] == "tier_unset"


# ---------------------------------------------------------------------------
# POST /api/map/fit — failure-mode propagation in R2 mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_map_fit_propagates_404_as_400_via_coords_unavailable(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """POST /api/map/fit — R2-404 → CoordinatesUnavailableError → route 400.

    Per Codex PR #11 R1 F2 + R2 closing remark: start_map_fit uses the
    same R2 branch as get_map_layers; its propagation contract is the
    same.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=404, detail="not_found")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.post(
        "/api/map/fit",
        json={"scan_id": SCAN_UUID, "domains": ["minerals"]},
    )
    assert resp.status_code == 400
    assert "Loupe workspace file not found in R2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_map_fit_preserves_502_credential_error(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """POST /api/map/fit — 502 upstream_credential_error preserves status."""
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=502, detail="upstream_credential_error")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.post(
        "/api/map/fit",
        json={"scan_id": SCAN_UUID, "domains": ["minerals"]},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"] == "upstream_credential_error"


@pytest.mark.asyncio
async def test_map_fit_preserves_504_timeout(
    client, scanner_workspace_scan, force_r2_mode, monkeypatch
):
    """POST /api/map/fit — 504 upstream_timeout preserves status."""
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=504, detail="upstream_timeout")

    monkeypatch.setattr(
        "sherloc_pipeline.web.routes.map.get_working_file", fake_reader
    )
    resp = await client.post(
        "/api/map/fit",
        json={"scan_id": SCAN_UUID, "domains": ["minerals"]},
    )
    assert resp.status_code == 504
    assert resp.json()["detail"] == "upstream_timeout"
