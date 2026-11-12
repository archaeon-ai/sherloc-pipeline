"""Scan fit status derived at read time from fitted_peaks.

``scans.processing_status`` has no writer, so the scan list and detail
report ``has_fitted_peaks`` instead: true when at least one
``fitted_peaks`` row belongs to the scan (fitted_peaks -> spectra ->
scan_points -> scans), else false. The list computes it in the same
query as the rows (no per-scan query), and the ``has_fitted_peaks``
filter applies the same rule. Reads never write the scans table.
"""

from contextlib import contextmanager

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from tests.unit.web.conftest import SCAN_UUID, SOL_NUMBER, SPECTRUM_UUID, _make_spectrum_bytes

UNFITTED_SCAN = "00000000-0000-0000-0000-00000000a001"


def _add_peak(session, *, peak_id: str, spectrum_id: str) -> None:
    from sherloc_pipeline.database.models import FittedPeakORM

    session.add(
        FittedPeakORM(
            id=peak_id,
            spectrum_id=spectrum_id,
            amplitude=1.0,
            center_cm1=1085.0,
            fit_modality="minerals",
        )
    )


def _add_scan_with_spectrum(session, *, scan_id: str, name: str, sol: int = SOL_NUMBER) -> str:
    """Seed a scan with one point and one spectrum; return the spectrum id."""
    from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SolORM, SpectrumORM

    if session.get(SolORM, sol) is None:
        session.add(SolORM(sol_number=sol, data_source="loupe"))
    session.add(
        ScanORM(
            id=scan_id,
            sol_number=sol,
            scan_name=name,
            target="Fixture_Rock",
            scan_id=f"{sol}_{name}",
            sclk_start=730000000 + sol,
            n_points=1,
            n_channels=2148,
            laser_wavelength_nm=248.5794,
            data_source="loupe",
            target_type="mars_target",
            scan_class="primary",
            scan_type="detail",
        )
    )
    point_id = f"p{scan_id[1:]}"
    spectrum_id = f"s{scan_id[1:]}"
    session.add(ScanPointORM(id=point_id, scan_id=scan_id, point_index=0))
    session.add(
        SpectrumORM(
            id=spectrum_id,
            scan_point_id=point_id,
            region="R1",
            spectrum_type="dark_subtracted",
            processing_level="dark_subtracted",
            intensities=_make_spectrum_bytes(),
        )
    )
    return spectrum_id


@pytest.fixture()
def fit_fixture(test_engine):
    """The conftest scan gets one fitted peak; a second scan has a spectrum and no peaks."""
    with Session(test_engine) as session:
        _add_peak(session, peak_id="00000000-0000-0000-0000-00000000f001", spectrum_id=SPECTRUM_UUID)
        _add_scan_with_spectrum(session, scan_id=UNFITTED_SCAN, name="detail_2")
        session.commit()
    return test_engine


@contextmanager
def _statements(engine):
    seen: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        yield seen
    finally:
        event.remove(engine, "before_cursor_execute", _record)


@pytest.mark.asyncio
async def test_list_reports_fit_status_per_scan(client, fit_fixture):
    resp = await client.get("/api/scans")
    assert resp.status_code == 200
    by_id = {s["id"]: s for s in resp.json()["scans"]}
    assert by_id[SCAN_UUID]["has_fitted_peaks"] is True
    assert by_id[UNFITTED_SCAN]["has_fitted_peaks"] is False


@pytest.mark.asyncio
async def test_detail_reports_fit_status(client, fit_fixture):
    fitted = (await client.get(f"/api/scans/{SCAN_UUID}")).json()["scan"]
    unfitted = (await client.get(f"/api/scans/{UNFITTED_SCAN}")).json()["scan"]
    assert fitted["has_fitted_peaks"] is True
    assert unfitted["has_fitted_peaks"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "expected"), [("true", [SCAN_UUID]), ("false", [UNFITTED_SCAN])])
async def test_filter_uses_the_same_rule(client, fit_fixture, value, expected):
    resp = await client.get("/api/scans", params={"has_fitted_peaks": value})
    assert resp.status_code == 200
    data = resp.json()
    assert [s["id"] for s in data["scans"]] == expected
    assert data["total"] == 1
    assert all(s["has_fitted_peaks"] is (value == "true") for s in data["scans"])


@pytest.mark.asyncio
async def test_list_query_count_does_not_grow_with_scans(client, fit_fixture):
    """N+1 guard: the list issues the same statements for 2 scans as for 8."""
    with _statements(fit_fixture) as small:
        resp = await client.get("/api/scans")
    assert resp.json()["total"] == 2

    with Session(fit_fixture) as session:
        for i in range(6):
            scan_id = f"00000000-0000-0000-0000-00000000b{i:03d}"
            spectrum_id = _add_scan_with_spectrum(session, scan_id=scan_id, name=f"detail_b{i}")
            if i % 2 == 0:
                _add_peak(session, peak_id=f"00000000-0000-0000-0000-00000000c{i:03d}", spectrum_id=spectrum_id)
        session.commit()

    with _statements(fit_fixture) as large:
        resp = await client.get("/api/scans")
    scans = resp.json()["scans"]
    assert len(scans) == 8
    assert sum(s["has_fitted_peaks"] for s in scans) == 4
    assert len(large) == len(small)
    assert sum("fitted_peaks" in s for s in large) == 1


@pytest.mark.asyncio
async def test_reads_leave_processing_status_untouched(client, fit_fixture):
    """The stored column is neither derived into nor written by these reads."""
    from sherloc_pipeline.database.models import ScanORM

    with _statements(fit_fixture) as seen:
        listed = (await client.get("/api/scans")).json()["scans"]
        detail = (await client.get(f"/api/scans/{SCAN_UUID}")).json()["scan"]
        await client.get("/api/scans", params={"has_fitted_peaks": "true"})
    assert all(s["processing_status"] is None for s in listed)
    assert detail["processing_status"] is None
    assert not [s for s in seen if s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))]
    with Session(fit_fixture) as session:
        assert {s.processing_status for s in session.query(ScanORM)} == {None}
