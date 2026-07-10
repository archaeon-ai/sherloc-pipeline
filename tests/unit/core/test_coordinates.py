"""Tests for ``core/coordinates.py`` — Loupe-workspace R2 + FS resolution.

v4.1.9 added the optional ``workspace_reader`` injection to
``resolve_display_coordinates()`` so the production runtime can fetch
``spatial.csv`` / ``loupe.csv`` from R2 instead of the local filesystem
(unblocks ``/api/map/layers/<id>`` on R2-backed deployments).

Exercises:

- R2 path: inject a mock reader that returns synthetic Loupe CSV bytes,
  verify the resolver materializes them through a temp dir and produces
  ``DisplayCoordinate`` rows.
- FS path (legacy): no reader → resolver reads ``spatial.csv`` /
  ``loupe.csv`` from a real temp dir (mirrors local-filesystem dev
  installs without R2).
- 404 path: reader raises ``HTTPException(404)`` → resolver wraps as
  ``CoordinatesUnavailableError`` with a clear message; the route layer
  (``web/routes/map.py:get_map_layers``) re-raises this as HTTP 400.

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

# Canonical team-tier shapes; the Loupe workspace is two levels up. The
# R2 path keys off the relative locator (r2_rel_key); the FS fallback
# still uses the absolute file_path.
_TEAM_FILE_PATH = (
    "/data/sherloc/data/loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)
_TEAM_LOCATOR = (
    "loupe/sol_0921/detail_1/"
    "SrlcSpecSpecSohRaw_TEST_Loupe_working/img/SC3_0921_TEST.PNG"
)
# The colorized variant rewrites only the bare ``sol_NNNN`` segment
# (``sol_0921`` → ``sol_0921_colorized``); the ``_0921`` inside the PNG
# filename is NOT a bare sol segment and stays untouched (issue #8).
_TEAM_LOCATOR_COLORIZED = (
    "loupe/sol_0921_colorized/detail_1/"
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
# Colorized spatial.csv: the colorized ACI is a pure crop of grayscale, so
# the colorized workspace re-solves each point's x/y for the cropped frame.
# Distinct x/y values here ⇒ distinct xPix/yPix so the resolver's two
# variants are provably different (issue #8).
_SAMPLE_SPATIAL_CSV_COLORIZED = (
    b"x,y\n"
    b"0.3,0.1\n"
    b"0.4,0.2\n"
    b"0.5,0.3\n"
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
                r2_rel_key=_TEAM_LOCATOR,
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

    # Reader called for both spatial.csv and loupe.csv with the same locator
    assert (_TEAM_LOCATOR, "spatial.csv") in calls
    assert (_TEAM_LOCATOR, "loupe.csv") in calls

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
    assert "locator=" in msg
    assert "missing_file='spatial.csv'" in msg
    assert "expected_key=" in msg
    assert "spatial.csv" in msg


def test_resolve_via_workspace_reader_loupe_404_names_loupe(scan_session):
    """Only loupe.csv 404s (spatial.csv succeeds) → error names loupe.csv.

    When loupe.csv is the actually-missing file, the error message MUST
    name loupe.csv, not spatial.csv.
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

# FS-fallback locator: workspace two levels up from the ACI file. Kept
# distinct from the fixture's _TEAM_LOCATOR so each FS test lays its own
# tmp layout and resolves it via data_root + r2_rel_key (NOT file_path, #7).
_FS_LOCATOR = "loupe/sol_0921/detail_1/ws/img/test.PNG"


def test_resolve_legacy_fs_path(scan_session, tmp_path, monkeypatch):
    """workspace_reader=None → resolver reads spatial.csv/loupe.csv from FS.

    Disk reads derive from the ACI's stored locator (r2_rel_key) + the
    injected data_root (#7); .parent.parent lands in a real workspace dir
    we control. Mirrors local-filesystem dev installs (no R2 mode).
    """
    workspace = tmp_path / "loupe" / "sol_0921" / "detail_1" / "ws"
    workspace.mkdir(parents=True)
    (workspace / "spatial.csv").write_bytes(_SAMPLE_SPATIAL_CSV)
    (workspace / "loupe.csv").write_bytes(_SAMPLE_LOUPE_CSV)
    img_dir = workspace / "img"
    img_dir.mkdir()
    aci_file = img_dir / "test.PNG"
    aci_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny PNG header

    # Point the ACI locator at the tmp layout so
    # resolve_disk_path(locator, data_root=tmp_path).parent.parent == workspace.
    aci = scan_session.query(ContextImageORM).filter_by(scan_id=SCAN_UUID).first()
    aci.r2_rel_key = _FS_LOCATOR
    scan_session.commit()

    coords = resolve_display_coordinates(
        scan_session, SCAN_UUID, data_root=str(tmp_path)
    )
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
    aci.r2_rel_key = _FS_LOCATOR
    scan_session.commit()

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, data_root=str(tmp_path)
        )
    msg = str(excinfo.value)
    assert "Loupe workspace files not found at" in msg
    # FS-mode message names the resolved workspace directory.
    assert "spatial.csv present=False" in msg


def test_resolve_legacy_fs_path_null_locator_raises(scan_session, tmp_path):
    """workspace_reader=None and a NULL r2_rel_key → CoordinatesUnavailableError.

    A row that matched no known layout has a NULL locator; the FS fallback
    cannot resolve a disk path and fails with the missing-locator error
    (the same shape as today's missing-file failure).
    """
    aci = scan_session.query(ContextImageORM).filter_by(scan_id=SCAN_UUID).first()
    aci.r2_rel_key = None
    scan_session.commit()

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, data_root=str(tmp_path)
        )
    assert "Cannot resolve a Loupe workspace disk path" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Colorized variant resolution (issue #8)
# ---------------------------------------------------------------------------

def _variant_aware_reader(calls: list[tuple[str, str]]):
    """Reader returning the colorized spatial.csv iff the path is colorized.

    Records (file_path, filename) calls so a test can assert WHICH workspace
    (``sol_0921`` vs ``sol_0921_colorized``) the resolver fetched from.
    """
    def reader(file_path: str, filename: str) -> bytes:
        calls.append((file_path, filename))
        is_colorized = "_colorized" in file_path
        if filename == "spatial.csv":
            return _SAMPLE_SPATIAL_CSV_COLORIZED if is_colorized else _SAMPLE_SPATIAL_CSV
        if filename == "loupe.csv":
            return _SAMPLE_LOUPE_CSV
        raise AssertionError(f"unexpected filename: {filename!r}")

    return reader


def test_resolve_colorized_reads_colorized_workspace_and_differs(scan_session):
    """colorized=True fetches the sol_NNNN_colorized workspace and yields shifted coords."""
    calls: list[tuple[str, str]] = []
    reader = _variant_aware_reader(calls)

    grayscale = resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader
    )
    colorized = resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader, colorized=True
    )

    # Grayscale fetched the base workspace; colorized fetched sol_0921_colorized.
    assert (_TEAM_LOCATOR, "spatial.csv") in calls
    assert (_TEAM_LOCATOR_COLORIZED, "spatial.csv") in calls
    assert (_TEAM_LOCATOR_COLORIZED, "loupe.csv") in calls
    # The colorized variant must NEVER be fetched from the grayscale path.
    assert (_TEAM_LOCATOR, "spatial.csv") != (_TEAM_LOCATOR_COLORIZED, "spatial.csv")

    assert len(grayscale) == len(colorized) == N_POINTS
    # Same point indices, but the colorized coords are crop-shifted ⇒ differ.
    by_idx_gray = {c.point_index: c for c in grayscale}
    by_idx_color = {c.point_index: c for c in colorized}
    assert set(by_idx_gray) == set(by_idx_color)
    for idx in by_idx_gray:
        assert by_idx_gray[idx].aci_x != by_idx_color[idx].aci_x
        assert by_idx_gray[idx].transform_method == "scanner_calibration"
        assert by_idx_color[idx].transform_method == "scanner_calibration"


def test_resolve_colorized_and_grayscale_cached_separately(scan_session):
    """Both variants persist side by side keyed by (scan_point_id, colorized)."""
    from sherloc_pipeline.database.models import MapDisplayCoordinateORM

    calls: list[tuple[str, str]] = []
    reader = _variant_aware_reader(calls)

    resolve_display_coordinates(scan_session, SCAN_UUID, workspace_reader=reader)
    resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader, colorized=True
    )

    gray_rows = (
        scan_session.query(MapDisplayCoordinateORM)
        .filter(MapDisplayCoordinateORM.colorized.is_(False))
        .all()
    )
    color_rows = (
        scan_session.query(MapDisplayCoordinateORM)
        .filter(MapDisplayCoordinateORM.colorized.is_(True))
        .all()
    )
    assert len(gray_rows) == N_POINTS
    assert len(color_rows) == N_POINTS

    # A second colorized call is served from cache — the reader is NOT touched.
    calls_before = len(calls)
    again = resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader, colorized=True
    )
    assert len(again) == N_POINTS
    assert len(calls) == calls_before  # no new workspace fetches

    # The cached colorized coords match the colorized (not grayscale) values.
    cached_color = {c.point_index: c.aci_x for c in again}
    gray = resolve_display_coordinates(scan_session, SCAN_UUID, workspace_reader=reader)
    gray_x = {c.point_index: c.aci_x for c in gray}
    for idx in cached_color:
        assert cached_color[idx] != gray_x[idx]


def test_resolve_colorized_missing_workspace_raises(scan_session):
    """colorized=True with a missing colorized spatial.csv → CoordinatesUnavailableError.

    The route layer turns this into "no colorized point set" (grayscale
    overlay still renders) rather than erroring the endpoint.
    """
    def reader(file_path: str, filename: str) -> bytes:
        if "_colorized" in file_path:
            raise HTTPException(status_code=404, detail="not_found")
        if filename == "spatial.csv":
            return _SAMPLE_SPATIAL_CSV
        if filename == "loupe.csv":
            return _SAMPLE_LOUPE_CSV
        raise AssertionError(f"unexpected filename: {filename!r}")

    # Grayscale still resolves fine.
    assert len(resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader
    )) == N_POINTS

    # Colorized raises because the colorized workspace file is absent.
    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=reader, colorized=True
        )
    assert "missing_file='spatial.csv'" in str(excinfo.value)
    assert "sol_0921_colorized" in str(excinfo.value)


def test_resolve_colorized_on_aci_pixel_frame_raises(scan_session):
    """aci_pixel (PDS) scans have no colorized variant → colorized=True raises."""
    # Flip the scan's points to the aci_pixel frame with real pixel values.
    for pt in scan_session.query(ScanPointORM).filter_by(scan_id=SCAN_UUID).all():
        pt.coordinate_frame = "aci_pixel"
        pt.x_pixel = 100.0 + pt.point_index
        pt.y_pixel = 200.0 + pt.point_index
    scan_session.commit()

    # Grayscale identity resolution works.
    assert len(resolve_display_coordinates(scan_session, SCAN_UUID)) == N_POINTS

    with pytest.raises(CoordinatesUnavailableError) as excinfo:
        resolve_display_coordinates(scan_session, SCAN_UUID, colorized=True)
    assert "aci_pixel" in str(excinfo.value)


def test_resolve_force_recompute_rewrites_cache_quietly(scan_session):
    """force_recompute deletes + rewrites the variant's cache rows, no SAWarning.

    Covers the previously-untested force_recompute delete path and guards the
    fix that passes an explicit ``select()`` (not a ``.subquery()``) to
    ``in_()`` so SQLAlchemy 2.x does not emit the "Coercing Subquery into a
    select()" warning.
    """
    import warnings

    from sqlalchemy.exc import SAWarning

    from sherloc_pipeline.database.models import MapDisplayCoordinateORM

    reader = _variant_aware_reader([])
    first = resolve_display_coordinates(
        scan_session, SCAN_UUID, workspace_reader=reader, force_recompute=True
    )
    assert len(first) == N_POINTS

    with warnings.catch_warnings():
        warnings.simplefilter("error", SAWarning)
        again = resolve_display_coordinates(
            scan_session, SCAN_UUID, workspace_reader=reader, force_recompute=True
        )
    assert len(again) == N_POINTS

    # Delete-then-insert leaves exactly N_POINTS grayscale rows (no duplication).
    assert (
        scan_session.query(MapDisplayCoordinateORM)
        .filter(MapDisplayCoordinateORM.colorized.is_(False))
        .count()
        == N_POINTS
    )
