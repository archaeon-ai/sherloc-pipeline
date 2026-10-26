"""Map mode must fit the already laser-normalized spectra ingested from PDS."""

import threading
import uuid
import zlib
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.database.connection import create_all_tables, get_session_factory
from sherloc_pipeline.database.models import ScanORM, ScanPointORM, SolORM, SpectrumORM
from sherloc_pipeline.services.map_fitting import (
    FLUOR_NOT_DESPIKED,
    MapFitService,
)
from sherloc_pipeline.services import map_fitting

N_CHANNELS = 2148
N_POINTS = 2


def _encode(values):
    return zlib.compress(np.asarray(values, dtype=np.float32).tobytes())


@pytest.fixture()
def pds_scan():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)
    session = get_session_factory(engine)()
    session.add(SolORM(sol_number=938, data_source="pds4"))
    scan = ScanORM(
        id=str(uuid.uuid4()), sol_number=938, scan_name="detail_1",
        scan_id="synthetic-pds-map", sclk_start=740000000, sclk_stop=740001000,
        n_points=N_POINTS, n_channels=N_CHANNELS, data_source="pds4",
    )
    session.add(scan)
    session.flush()
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    for i in range(N_POINTS):
        point = ScanPointORM(
            id=str(uuid.uuid4()), scan_id=scan.id, point_index=i,
            x_pixel=100.0 + i, y_pixel=200.0 + i, photodiode_mean=None,
        )
        session.add(point)
        session.flush()
        rng = np.random.RandomState(i)
        raman = 10.0 + 0.01 * np.arange(N_CHANNELS)
        for center, sigma, amplitude in ((1000, 20, 100), (1600, 40, 80), (3400, 70, 120)):
            raman += amplitude * np.exp(-0.5 * ((wavenumber - center) / sigma) ** 2)
        raman += rng.normal(0, 0.15, N_CHANNELS)
        fluor = 10.0 + 400.0 * np.exp(-0.5 * ((wavelength - 306.0) / 10.0) ** 2)
        fluor += rng.normal(0, 0.15, N_CHANNELS)
        for region in ("R1", "R2", "R3"):
            session.add(SpectrumORM(
                id=str(uuid.uuid4()), scan_point_id=point.id, region=region,
                spectrum_type="laser_normalized", processing_level="normalized",
                intensities=_encode(raman if region == "R1" else fluor),
                wavelengths=_encode(wavelength), wavelength_source="pds_embedded",
            ))
    session.commit()
    try:
        yield session, scan
    finally:
        session.close()
        engine.dispose()


@pytest.mark.parametrize("domains", [
    pytest.param(["minerals", "organics", "hydration"], id="raman"),
    pytest.param(["fluorescence"], id="fluorescence"),
])
def test_pds_map_fits_without_photodiode(pds_scan, domains):
    session, scan = pds_scan
    results = []
    summary = MapFitService(config={}).run_map_fit(
        session=session, scan_id=scan.id, domains=domains, point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=results.append, on_progress=lambda *a: None,
        on_log=lambda *a: None, cancel_event=threading.Event(),
    )

    assert summary.total_points == N_POINTS
    assert len(results) == N_POINTS
    assert summary.laser_norm_not_applied == []
    for point in results:
        for domain in domains:
            assert point.results[domain].status == "measured"
            assert point.results[domain].peaks
        assert point.notes == ([FLUOR_NOT_DESPIKED] if "fluorescence" in domains else [])


def test_pds_map_keeps_upstream_normalization_with_photodiodes(pds_scan, monkeypatch):
    """Even if photodiodes become available, neither domain is scaled twice."""
    session, scan = pds_scan
    points = session.query(ScanPointORM).order_by(ScanPointORM.point_index).all()
    for point, photodiode in zip(points, (1.0, 4.0)):
        point.photodiode_mean = photodiode
    session.commit()
    preprocess = Mock(side_effect=lambda x, y, *a, **kw: y)
    raman_fit = Mock(return_value=map_fitting.DomainResult(status="below_threshold"))
    fluor_fit = Mock(return_value=map_fitting.DomainResult(status="below_threshold"))
    monkeypatch.setattr(map_fitting, "_preprocess_r1_intensity", preprocess)
    monkeypatch.setattr(map_fitting, "_fit_raman_domain", raman_fit)
    monkeypatch.setattr(map_fitting, "_fit_fluorescence_domain", fluor_fit)
    results = []
    summary = MapFitService(config={"fluorescence_fitting": {"laser_normalize": True}}).run_map_fit(
        session=session, scan_id=scan.id, domains=["minerals", "fluorescence"],
        point_indices=None, point_coords={}, on_point_fitted=results.append,
        on_progress=lambda *a: None, on_log=lambda *a: None, cancel_event=threading.Event(),
    )

    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    r1_mask = map_fitting.get_region_wavelength_mask(wavelength, "R1")
    assert preprocess.call_count == N_POINTS
    assert fluor_fit.call_count == N_POINTS
    for i, point in enumerate(points):
        rows = session.query(SpectrumORM).filter(SpectrumORM.scan_point_id == point.id).all()
        arrays = {row.region: np.frombuffer(zlib.decompress(row.intensities), dtype=np.float32)
                  for row in rows}
        np.testing.assert_array_equal(preprocess.call_args_list[i].args[1], arrays["R1"][r1_mask])
        fluor_kwargs = fluor_fit.call_args_list[i].kwargs
        for region in ("R1", "R2", "R3"):
            np.testing.assert_array_equal(fluor_kwargs["preloaded_parts"][region], arrays[region])
        assert fluor_kwargs["saturation_scale"] == 1.0
    assert summary.laser_norm_not_applied == []
    assert all(point.notes == [FLUOR_NOT_DESPIKED] for point in results)


def test_pds_fluorescence_fallback_uses_normalized_rows(pds_scan):
    session, scan = pds_scan
    point = session.query(ScanPointORM).filter(ScanPointORM.scan_id == scan.id).first()
    result = map_fitting._fit_fluorescence_domain(
        session, point.id, {}, spectrum_type="laser_normalized"
    )
    assert result.status == "measured"
    assert result.peaks
