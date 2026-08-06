"""Tests for shared per-point preprocessing in Map Mode fitting (issue #6).

Despiking and the asPLS baseline depend only on a point's R1 spectrum,
not on the fitting domain, but ``_fit_raman_domain`` used to redo both
for every requested domain — 3x the most expensive preprocessing step
for the default minerals+organics+hydration selection. That is a large
part of why big scans appear to hang.

The hoist is only safe if it is behaviour-preserving, so these tests pin
both halves: identical fit results, and one preprocessing pass per point
regardless of how many Raman domains are selected.
"""

from __future__ import annotations

import threading
import uuid
import zlib

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from sherloc_pipeline.core.baseline import BaselineParams
from sherloc_pipeline.core.calibration import (
    calculate_loupe_wavelength_wavenumber,
    get_region_wavelength_mask,
)
from sherloc_pipeline.core.preprocessing import build_weight_vector_from_windows
from sherloc_pipeline.database.connection import create_all_tables, get_session_factory
from sherloc_pipeline.database.models import (
    ScanORM,
    ScanPointORM,
    SolORM,
    SpectrumORM,
)
from sherloc_pipeline.services import map_fitting
from sherloc_pipeline.services.map_fitting import (
    MapFitService,
    _fit_raman_domain,
    _preprocess_r1_intensity,
)

N_CHANNELS = 2148
N_POINTS = 3
SOL_NUMBER = 938
SCAN_UUID = str(uuid.UUID("00000000-0000-0000-0000-000000000901"))

_CONFIG = {
    "fitting": {
        "r1_fit_range": [700, 1200],
        "organics_fit_range": [1250, 1850],
        "hydration_fit_range": [2800, 3900],
        "max_peaks": 5,
        "min_snr": 3.0,
        "min_seed_snr": 2.0,
        "fit_fwhm_min_initial_cm1": 22,
        "filter_fwhm_min_cm1": 30,
        "fwhm_max_cm1": 90,
        "peak_separation_cm1": 25,
        "r_squared_min": 0.25,
        "noise_estimation": {"window": [2000.0, 2100.0]},
        "parsimony": {},
        "posthoc_filters": {},
    },
    "preprocessing": {"baseline": {}},
}


def _spectrum_bytes(seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    x = np.arange(N_CHANNELS, dtype=np.float64)
    spectrum = 500.0 + 140.0 * np.exp(-((x - 200) ** 2) / (2 * 30**2))
    spectrum += rng.normal(0, 5, size=N_CHANNELS)
    # A couple of cosmic-ray-like spikes so despiking is not a no-op.
    spectrum[311] += 4000.0
    spectrum[742] += 2500.0
    return zlib.compress(spectrum.astype(np.float32).tobytes())


@pytest.fixture()
def r1_arrays():
    """(wavenumber_r1, intensity_r1, baseline_params, baseline_weights)."""
    wavelength, wavenumber = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    r1_mask = get_region_wavelength_mask(wavelength, "R1")
    wavenumber_r1 = wavenumber[r1_mask]

    raw = np.frombuffer(zlib.decompress(_spectrum_bytes(7)), dtype=np.float32)
    intensity_r1 = raw[r1_mask]

    bl_params = BaselineParams(lam=1e6, asymmetric_coef=0.01, iters=10, diff_order=2, tol=1e-3)
    bl_weights = build_weight_vector_from_windows(
        wavenumber_r1,
        keep_windows=[(600.0, 1130.0), (1300.0, 1720.0), (3000.0, 3800.0)],
        default_weight=1.0,
        keep_weight=0.01,
    )
    return wavenumber_r1, intensity_r1, bl_params, bl_weights


@pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
def test_preprocessed_input_matches_inline_preprocessing(r1_arrays, domain):
    """Hoisting despike+baseline out of the domain fit changes nothing."""
    wavenumber_r1, intensity_r1, bl_params, bl_weights = r1_arrays

    inline = _fit_raman_domain(
        wavenumber_r1,
        intensity_r1,
        _CONFIG,
        domain,
        baseline_params=bl_params,
        baseline_weights=bl_weights,
    )

    prepped = _preprocess_r1_intensity(wavenumber_r1, intensity_r1, bl_params, bl_weights)
    hoisted = _fit_raman_domain(
        wavenumber_r1,
        prepped,
        _CONFIG,
        domain,
        baseline_params=bl_params,
        baseline_weights=bl_weights,
        preprocessed=True,
    )

    assert hoisted.status == inline.status
    assert hoisted.peaks == inline.peaks


def test_preprocess_changes_the_spectrum(r1_arrays):
    """Guard the equivalence test above against a no-op preprocessing step."""
    wavenumber_r1, intensity_r1, bl_params, bl_weights = r1_arrays

    prepped = _preprocess_r1_intensity(wavenumber_r1, intensity_r1, bl_params, bl_weights)

    assert prepped.shape == intensity_r1.shape
    assert not np.allclose(prepped, intensity_r1.astype(np.float64))


# ---------------------------------------------------------------------------
# run_map_fit: one preprocessing pass per point, not per (point, domain)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fit_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_all_tables(engine)
    session = get_session_factory(engine)()
    session.add(SolORM(sol_number=SOL_NUMBER, data_source="loupe"))
    session.flush()
    session.add(
        ScanORM(
            id=SCAN_UUID,
            sol_number=SOL_NUMBER,
            scan_name="detail_1",
            scan_id="0938_detail_1",
            sclk_start=740000000,
            sclk_stop=740001000,
            n_points=N_POINTS,
            n_channels=N_CHANNELS,
            data_source="loupe",
        )
    )
    session.flush()
    for i in range(N_POINTS):
        pt_id = str(uuid.UUID(f"00000000-0000-0000-0000-{910 + i:012d}"))
        session.add(
            ScanPointORM(
                id=pt_id,
                scan_id=SCAN_UUID,
                point_index=i,
                x_pixel=100.0 + i,
                y_pixel=200.0 + i,
                photodiode_mean=4800.0 + i,
            )
        )
        session.flush()
        session.add(
            SpectrumORM(
                id=str(uuid.UUID(f"00000000-0000-0000-0000-{950 + i:012d}")),
                scan_point_id=pt_id,
                region="R1",
                spectrum_type="dark_subtracted",
                processing_level="dark_subtracted",
                intensities=_spectrum_bytes(i),
            )
        )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_preprocessing_runs_once_per_point_across_domains(fit_session, monkeypatch):
    calls = {"n": 0}
    real = map_fitting._preprocess_r1_intensity

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(map_fitting, "_preprocess_r1_intensity", counting)

    domains = ["minerals", "organics", "hydration"]
    summary = MapFitService(config=_CONFIG).run_map_fit(
        session=fit_session,
        scan_id=SCAN_UUID,
        domains=domains,
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=lambda r: None,
        on_progress=lambda *a: None,
        on_log=lambda *a: None,
        cancel_event=threading.Event(),
    )

    assert summary.total_points == N_POINTS
    # Once per point — previously once per (point, domain) = 9.
    assert calls["n"] == N_POINTS


def test_run_map_fit_logs_scan_scale_before_fitting(fit_session):
    logged: list[tuple[int, str]] = []

    MapFitService(config=_CONFIG).run_map_fit(
        session=fit_session,
        scan_id=SCAN_UUID,
        domains=["minerals"],
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=lambda r: None,
        on_progress=lambda *a: None,
        on_log=lambda idx, msg: logged.append((idx, msg)),
        cancel_event=threading.Event(),
    )

    # A job-level line (point_index -1) naming the point count arrives
    # before any per-point line, so a slow scan says how big it is.
    assert logged
    assert logged[0][0] == -1
    assert f"{N_POINTS} points" in logged[0][1]


def test_cancel_event_stops_the_run_early(fit_session):
    """The stall fix relies on cancel actually being honoured mid-run."""
    cancel = threading.Event()
    cancel.set()

    summary = MapFitService(config=_CONFIG).run_map_fit(
        session=fit_session,
        scan_id=SCAN_UUID,
        domains=["minerals"],
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=lambda r: None,
        on_progress=lambda *a: None,
        on_log=lambda *a: None,
        cancel_event=cancel,
    )

    assert summary.total_points == 0


def test_cancel_before_start_skips_the_eager_spectrum_load(fit_session, monkeypatch):
    """A job cancelled while queued must not pay for the whole-scan load.

    Setup loads every point's spectra up front, which on a large scan is
    the dominant cost. The only cancel check used to be inside the
    per-point loop underneath it, so a job cancelled before the executor
    reached it still read the entire scan before noticing.
    """
    loaded: list[str] = []

    def spy(session, ids, region="R1"):
        loaded.append(region)
        return {}

    monkeypatch.setattr(map_fitting, "_load_point_spectra", spy)

    cancel = threading.Event()
    cancel.set()
    logged: list[tuple[int, str]] = []

    summary = MapFitService(config=_CONFIG).run_map_fit(
        session=fit_session,
        scan_id=SCAN_UUID,
        domains=["minerals", "fluorescence"],
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=lambda r: None,
        on_progress=lambda *a: None,
        on_log=lambda idx, msg: logged.append((idx, msg)),
        cancel_event=cancel,
    )

    assert loaded == []
    assert logged == []
    assert summary.total_points == 0
    assert summary.detections == {"minerals": 0, "fluorescence": 0}
