"""Tests for ``core/coordinates.py`` — Loupe-workspace R2 + FS resolution.

v4.1.9 added the optional ``workspace_reader`` injection to
``resolve_display_coordinates()`` so the production runtime can fetch
``spatial.csv`` / ``loupe.csv`` from R2 instead of the local filesystem
(per m2020-phase platform spec §3.9.8; unblocks ``/api/map/layers/<id>``
on the v1.0-beta VPS deployment).

Exercises:

- R2 path: inject a mock reader that returns synthetic Loupe CSV bytes,
  verify the resolver materializes them through a temp dir and produces
  ``DisplayCoordinate`` rows.
- FS path (legacy): no reader → resolver reads ``spatial.csv`` /
  ``loupe.csv`` from a real temp dir (mirrors local dev
  worktree at branch ``main`` v3.0.0).
- 404 path: reader raises ``HTTPException(404)`` → resolver wraps as
  ``CoordinatesUnavailableError`` with a clear message; the route layer
  (``web/routes/map.py:get_map_layers``) re-raises this as HTTP 400 per
  spec §3.9.8.3.

The R2-key-derivation logic lives in ``web/r2_reader.py`` and is covered
by ``tests/unit/web/test_r2_reader.py``; these tests only exercise the
resolver-side branch logic.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.core.coordinates import (
    CoordinatesUnavailableError,
    resolve_display_coordinates,
)
from sherloc_pipeline.database.connection import (
    create_all_tables,
    get_session_factory,
)
from sherloc_pipeline.database.models import (
    ContextImageORM,
    ScanORM,
    ScanPointORM,
    SolORM,
)

# Per-tier file_path conventions; the Loupe workspace is two levels up.
_TEAM_FILE_PATH = (
    "/data" "/sherloc/data/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)

# Minimal Loupe CSVs that produce a valid spatial table when load_spatial_table
# parses them. spatial.csv has the 'x,y' block per the Loupe convention; loupe.csv
# provides laser_x / laser_y for the calibration so the xPix/yPix derivation works.
_SAMPLE_SPATIAL_CSV = (
    b"x,y\n"
    b"0.0,0.0\n"
    b"0.1,0.1\n"
    b"0.2,0.2\n"
)
_SAMPLE_LOUPE_CSV = (
    b"laser_x,809.0\n"
    b"laser_y,664.0\n"
)


# ---------------------------------------------------------------------------
# In-memory SQLite DB with a scanner_workspace scan + ACI context image
# ---------------------------------------------------------------------------

SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
SOL_NUMBER = 921
N_POINTS = 3


@pytest.fixture()
def scan_session():
    """Build an in-memory DB with a scanner_workspace scan + 3 points + an ACI row.

    Yields a SQLAlchemy session. Tests get/commit through this session.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    create_all_tables(engine)
    session = get_session_factory(engine)()
    try:
        sol = SolORM(sol_number=SOL_NUMBER, data_source="loupe")
        session.add(sol)
        session.flush()

        scan = ScanORM(
            id=SCAN_UUID,
            sol_number=SOL_NUMBER,
            scan_name="detail_1",
            target="Test_Target",
            scan_id="0921_test_detail_1",
            sclk_start=730000000,
            sclk_stop=730001000,
            n_points=N_POINTS,
            n_channels=2148,
            shots_per_point=50,
            laser_wavelength_nm=248.5794,
            data_source="loupe",
            target_type="mars_target",
            scan_class="primary",
            scan_type="detail",
        )
        session.add(scan)
        session.flush()

        for i in range(N_POINTS):
            pt = ScanPointORM(
                id=str(uuid.UUID(f"00000000-0000-0000-0000-{10 + i:012d}")),
                scan_id=SCAN_UUID,
                point_index=i,
                # scanner_workspace points carry x/y in scanner-relative units, NOT pixels
                x_pixel=None,
                y_pixel=None,
                photodiode_mean=4800.0 + i,
                photodiode_std=12.0,
                coordinate_frame="scanner_workspace",
            )
            session.add(pt)

        session.add(
            ContextImageORM(
                id=str(uuid.uuid4()),
                scan_id=SCAN_UUID,
                image_type="ACI",
                file_path=_TEAM_FILE_PATH,
            )
        )
        session.commit()
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# R2 path (injected reader)
# ---------------------------------------------------------------------------

def test_resolve_via_workspace_reader_returns_coords(scan_session):
    """Inject a reader returning Loupe CSV bytes; resolver produces DisplayCoordinates."""
    calls: list[tuple[str, str]] = []

    def fake_reader(file_path: str, filename: str) -> bytes:
        calls.append((file_path, filename))
        if filename == "spatial.csv":
            return _SAMPLE_SPATIAL_CSV
        if filename == "loupe.csv":
            return _SAMPLE_LOUPE_CSV
        raise AssertionError(f"unexpected filename: {filename!r}")

    coords = resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=fake_reader
    )

    # Reader called for both spatial.csv and loupe.csv with the same file_path
    assert (_TEAM_FILE_PATH, "spatial.csv") in calls
    assert (_TEAM_FILE_PATH, "loupe.csv") in calls

    # Resolver produced 3 coords matching the 3 scan points
    assert len(coords) == N_POINTS
    for c in coords:
        assert c.transform_method == "scanner_calibration"
        # xPix/yPix should be valid floats derived from the Loupe calibration
        assert isinstance(c.aci_x, float)
        assert isinstance(c.aci_y, float)


def test_resolve_via_workspace_reader_spatial_404_names_spatial(scan_session):
    """spatial.csv 404 → CoordinatesUnavailableError naming spatial.csv key.

    The route layer's ``CoordinatesUnavailableError → HTTP 400`` mapping
    in ``map.py:get_map_layers`` produces the user-facing "Loupe
    workspace file not found in R2" banner per spec §3.9.8.3.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        # Both files 404; spatial.csv is fetched first so it raises first.
        raise HTTPException(status_code=404, detail="not_found")

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=fake_reader
        )
    msg = str(excinfo.value)
    assert "Loupe workspace file not found in R2" in msg
    assert SCAN_UUID in msg
    assert "file_path=" in msg
    assert "missing_file='spatial.csv'" in msg
    assert "expected_key=" in msg
    assert "spatial.csv" in msg


def test_resolve_via_workspace_reader_loupe_404_names_loupe(scan_session):
    """Only loupe.csv 404s (spatial.csv succeeds) → error names loupe.csv.

    Per Codex PR #11 R3 F4-residual: when loupe.csv is the actually-
    missing file, the error message MUST name loupe.csv, not spatial.csv.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        if filename == "spatial.csv":
            return _SAMPLE_SPATIAL_CSV
        if filename == "loupe.csv":
            raise HTTPException(status_code=404, detail="not_found")
        raise AssertionError(f"unexpected filename: {filename!r}")

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=fake_reader
        )
    msg = str(excinfo.value)
    assert "Loupe workspace file not found in R2" in msg
    assert "missing_file='loupe.csv'" in msg
    assert "expected_key=" in msg
    assert "loupe.csv" in msg
    # Negative: must NOT erroneously name spatial.csv as the missing file.
    assert "missing_file='spatial.csv'" not in msg


def test_resolve_via_workspace_reader_5xx_propagates(scan_session):
    """Reader raises HTTPException(500) → propagates unchanged (not wrapped as 400).

    Spec §3.9.8.3: 500 ``misconfigured_path`` + 502 ``upstream_credential_error``
    + 504 ``upstream_timeout`` surface their own status codes; only
    R2-404 maps through ``CoordinatesUnavailableError → 400``.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        raise HTTPException(status_code=502, detail="upstream_credential_error")

    with pytest.raises(HTTPException) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=fake_reader
        )
    assert excinfo.value.status_code == 502
    assert excinfo.value.detail == "upstream_credential_error"


def test_resolve_via_workspace_reader_malformed_csv_raises_coords_unavailable(
    scan_session,
):
    """Reader returns bytes that don't parse as Loupe CSV → CoordinatesUnavailableError.

    Defense-in-depth: even if the R2 object exists, malformed content
    surfaces as a clear error rather than an internal traceback.
    """
    def fake_reader(file_path: str, filename: str) -> bytes:
        return b"not, a, valid, loupe csv\n"

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=fake_reader
        )
    assert "Failed to parse Loupe workspace files from R2" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Legacy FS path (no reader)
# ---------------------------------------------------------------------------

def test_resolve_legacy_fs_path(scan_session, tmp_path, monkeypatch):
    """workspace_reader=None → resolver reads spatial.csv/loupe.csv from FS.

    The ACI file_path is rewritten to point at tmp_path so the parent.parent
    derivation lands in a real directory we control. Mirrors local dev
    worktree on branch ``main`` v3.0.0
    (no PHASE_TIER set; no R2 mode).
    """
    workspace = tmp_path / "loupe" / "sol_0921" / "detail_1" / "ws"
    workspace.mkdir(parents=True)
    (workspace / "spatial.csv").write_bytes(_SAMPLE_SPATIAL_CSV)
    (workspace / "loupe.csv").write_bytes(_SAMPLE_LOUPE_CSV)
    img_dir = workspace / "img"
    img_dir.mkdir()
    aci_file = img_dir / "test.PNG"
    aci_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny PNG header

    # Update ACI row to point at the temp location so .parent.parent == workspace.
    aci = scan_session.query(ContextImageORM).filter_by(scan_id=SCAN_UUID).first()
    aci.file_path = str(aci_file)
    scan_session.commit()

    coords = resolve_display_coordinates(scan_session, SCAN_UUID)
    assert len(coords) == N_POINTS
    for c in coords:
        assert c.transform_method == "scanner_calibration"


def test_resolve_legacy_fs_path_missing_files_raises(scan_session, tmp_path):
    """workspace_reader=None and FS files missing → CoordinatesUnavailableError."""
    workspace = tmp_path / "loupe" / "sol_0921" / "detail_1" / "ws"
    workspace.mkdir(parents=True)
    # Intentionally do NOT write spatial.csv / loupe.csv
    img_dir = workspace / "img"
    img_dir.mkdir()
    aci_file = img_dir / "test.PNG"
    aci_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    aci = scan_session.query(ContextImageORM).filter_by(scan_id=SCAN_UUID).first()
    aci.file_path = str(aci_file)
    scan_session.commit()

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(scan_session, SCAN_UUID)
    msg = str(excinfo.value)
    assert "Loupe workspace files not found at" in msg
    # FS-mode message names the directory; R2-mode message names the file_path
    assert "spatial.csv present=False" in msg
