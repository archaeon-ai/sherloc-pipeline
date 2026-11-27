"""The scan list's ``has_fitted_peaks`` and the map's layers count a stored Raman component only
at or above the display threshold ``fitting.min_snr``. Components stored below it
(``fitting.store_min_snr`` lower than ``min_snr``) are not shown; other modalities and rows
without an SNR count as before."""

import pytest
from sqlalchemy.orm import Session

from sherloc_pipeline.database.models import FittedPeakORM
from sherloc_pipeline.web.peak_display import display_min_snr, shown_peak_clause
from tests.unit.web.conftest import SCAN_UUID, SPECTRUM_UUID

FAINT_SCAN = "00000000-0000-0000-0000-00000000b001"


def _peak(session, pid, spectrum_id, *, snr, modality="minerals", assignment="calcite"):
    session.add(FittedPeakORM(id=pid, spectrum_id=spectrum_id, amplitude=1.0, center_cm1=1085.0,
                              fit_modality=modality, mineral_assignment=assignment, snr=snr))


@pytest.fixture()
def faint_fixture(test_engine):
    """The conftest scan holds a minerals row at SNR 4 and one at SNR 2.5 (another class);
    a second scan holds only a minerals row at SNR 2.5."""
    from tests.unit.web.test_scan_fit_status import _add_scan_with_spectrum

    with Session(test_engine) as session:
        _peak(session, "00000000-0000-0000-0000-00000000c001", SPECTRUM_UUID, snr=4.0)
        _peak(session, "00000000-0000-0000-0000-00000000c002", SPECTRUM_UUID, snr=2.5, assignment="sulfate")
        spectrum = _add_scan_with_spectrum(session, scan_id=FAINT_SCAN, name="detail_3")
        _peak(session, "00000000-0000-0000-0000-00000000c003", spectrum, snr=2.5)
        session.commit()
    return test_engine


def _shown(engine, threshold):
    with Session(engine) as session:
        return sorted(p.id[-4:] for p in session.query(FittedPeakORM).filter(shown_peak_clause(threshold)))


def test_the_clause_hides_only_raman_rows_below_the_threshold(test_engine):
    with Session(test_engine) as session:
        _peak(session, "00000000-0000-0000-0000-00000000d001", SPECTRUM_UUID, snr=2.5)
        _peak(session, "00000000-0000-0000-0000-00000000d002", SPECTRUM_UUID, snr=2.5, modality="organics")
        _peak(session, "00000000-0000-0000-0000-00000000d003", SPECTRUM_UUID, snr=2.5, modality="hydration")
        _peak(session, "00000000-0000-0000-0000-00000000d004", SPECTRUM_UUID, snr=2.5, modality="fluorescence")
        _peak(session, "00000000-0000-0000-0000-00000000d005", SPECTRUM_UUID, snr=None)
        _peak(session, "00000000-0000-0000-0000-00000000d006", SPECTRUM_UUID, snr=3.0)
        session.commit()
    assert _shown(test_engine, 3.0) == ["d004", "d005", "d006"]
    assert _shown(test_engine, 2.0) == ["d001", "d002", "d003", "d004", "d005", "d006"]


def test_the_threshold_is_the_configured_min_snr(fake_config):
    assert display_min_snr(fake_config) == 3.0
    fake_config.fitting["min_snr"] = 2.0
    assert display_min_snr(fake_config) == 2.0
    assert display_min_snr(None) == 3.0


@pytest.mark.asyncio
async def test_a_scan_with_only_faint_rows_reads_as_not_fitted(client, faint_fixture):
    by_id = {s["id"]: s for s in (await client.get("/api/scans")).json()["scans"]}
    assert by_id[SCAN_UUID]["has_fitted_peaks"] is True
    assert by_id[FAINT_SCAN]["has_fitted_peaks"] is False
    detail = (await client.get(f"/api/scans/{FAINT_SCAN}")).json()["scan"]
    assert detail["has_fitted_peaks"] is False
    unfitted = (await client.get("/api/scans", params={"has_fitted_peaks": "false"})).json()["scans"]
    assert FAINT_SCAN in {s["id"] for s in unfitted}


@pytest.mark.asyncio
async def test_a_lower_display_threshold_shows_them(client, faint_fixture, fake_config):
    fake_config.fitting["min_snr"] = 2.0
    by_id = {s["id"]: s for s in (await client.get("/api/scans")).json()["scans"]}
    assert by_id[FAINT_SCAN]["has_fitted_peaks"] is True


@pytest.mark.asyncio
async def test_map_layers_leave_out_faint_classes(client, faint_fixture, fake_config, monkeypatch):
    from types import SimpleNamespace

    import sherloc_pipeline.web.routes.map as map_routes

    monkeypatch.setattr(map_routes, "resolve_display_coordinates",
                        lambda *a, **k: [SimpleNamespace(transform_method="fixture")])
    monkeypatch.setattr(map_routes, "_build_point_set_payload", lambda coords: {"points": [], "voronoi": None})

    async def layers():
        resp = await client.get(f"/api/map/layers/{SCAN_UUID}")
        assert resp.status_code == 200, resp.text
        return resp.json()["available_layers"]["db"]["minerals"]

    shown = await layers()
    assert shown["classes"] == ["calcite"] and shown["n_detections"] == 1
    fake_config.fitting["min_snr"] = 2.0
    assert (await layers())["classes"] == ["calcite", "sulfate"]
