"""Tests for shared per-point preprocessing in Map Mode fitting (issue #6).

Despiking and the asPLS baseline depend only on a point's R1 spectrum,
not on the fitting domain, but ``_fit_raman_domain`` used to redo both
for every requested domain — 3x the most expensive preprocessing step
for the default minerals+organics+hydration selection. That is a large
part of why big scans appear to hang.

The hoist is only safe if it is behaviour-preserving, so these tests pin
both halves: identical fit results, and one preprocessing pass per point
regardless of how many Raman domains are selected (one per variant:
minerals are fitted non-despiked, organics and hydration despiked).
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
    """Hoisting despike+baseline out of the domain fit changes nothing.

    The hoisted input uses the same variant as the domain: minerals are not
    despiked, organics and hydration are.
    """
    wavenumber_r1, intensity_r1, bl_params, bl_weights = r1_arrays

    inline = _fit_raman_domain(
        wavenumber_r1,
        intensity_r1,
        _CONFIG,
        domain,
        baseline_params=bl_params,
        baseline_weights=bl_weights,
    )

    prepped = _preprocess_r1_intensity(
        wavenumber_r1, intensity_r1, bl_params, bl_weights,
        despike=domain not in map_fitting._NON_DESPIKED_DOMAINS,
    )
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
    # Once per point per variant (despiked for organics/hydration, not for
    # minerals) — previously once per (point, domain) = 9.
    assert calls["n"] == 2 * N_POINTS


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

    def spy(session, ids, region="R1", **kwargs):
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


# ---------------------------------------------------------------------------
# Batched loading must not turn one bad blob into a failed scan
# ---------------------------------------------------------------------------


def _point_id(i: int) -> str:
    return str(uuid.UUID(f"00000000-0000-0000-0000-{910 + i:012d}"))


@pytest.fixture()
def fluor_session(fit_session):
    """``fit_session`` plus R2/R3 spectra, with point 1's R2 blob corrupt.

    Fluorescence used to load its three regions per point, inside the
    per-point error handling, so an unreadable spectrum cost that point its
    fluorescence result and nothing else. Batching the load moved the
    decode to the top of the run, where it can take the whole scan with it.
    """
    for i in range(N_POINTS):
        for region, offset in (("R2", 960), ("R3", 970)):
            payload = _spectrum_bytes(i + offset)
            if i == 1 and region == "R2":
                payload = b"this is not a zlib stream"
            fit_session.add(
                SpectrumORM(
                    id=str(uuid.UUID(f"00000000-0000-0000-0000-{offset + i:012d}")),
                    scan_point_id=_point_id(i),
                    region=region,
                    spectrum_type="dark_subtracted",
                    processing_level="dark_subtracted",
                    intensities=payload,
                )
            )
    fit_session.commit()
    return fit_session


def test_batch_load_skips_an_unreadable_blob_rather_than_raising(fluor_session):
    from sherloc_pipeline.services.fluor_inputs import load_fluorescence_inputs

    loaded = load_fluorescence_inputs(
        fluor_session, [_point_id(i) for i in range(N_POINTS)]
    )

    # Only the affected point is dropped; the others keep all three regions.
    assert set(loaded) == {_point_id(0), _point_id(2)}
    assert set(loaded[_point_id(0)]["arrays"]) == {"R1", "R2", "R3"}


def test_one_corrupt_spectrum_costs_its_own_point_not_the_whole_scan(fluor_session):
    """Every other point still gets fitted, as it did before batching."""
    results = {}

    summary = MapFitService(config=_CONFIG).run_map_fit(
        session=fluor_session,
        scan_id=SCAN_UUID,
        domains=["minerals", "fluorescence"],
        point_indices=None,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
        on_point_fitted=lambda r: results.update({r.point_index: r}),
        on_progress=lambda *a: None,
        on_log=lambda *a: None,
        cancel_event=threading.Event(),
    )

    assert summary.total_points == N_POINTS
    assert sorted(results) == list(range(N_POINTS))
    # The unreadable spectrum is reported as a missing domain for that point,
    # exactly as an absent row is — and its Raman fit is untouched.
    assert results[1].results["fluorescence"].status == "missing"
    assert results[1].results["minerals"].status != "missing"


# ---------------------------------------------------------------------------
# ...but a corrupt R1 spectrum still fails the job, rather than being
# laundered into a point that merely looks unmeasured
# ---------------------------------------------------------------------------


@pytest.fixture()
def corrupt_r1_session(fit_session):
    """``fit_session`` with point 1's R1 blob replaced by garbage."""
    row = (
        fit_session.query(SpectrumORM)
        .filter(
            SpectrumORM.scan_point_id == _point_id(1),
            SpectrumORM.region == "R1",
        )
        .one()
    )
    row.intensities = b"this is not a zlib stream"
    fit_session.commit()
    return fit_session


def test_unreadable_r1_blob_raises_rather_than_dropping_the_point(corrupt_r1_session):
    """Corrupt Raman data must not leave the loader as a silent absence.

    ``missing`` is how this module reports a point that was never
    measured. A point whose stored spectrum is damaged is a different
    fact, and it is the one a reader of the map has to know about.
    """
    with pytest.raises(zlib.error):
        map_fitting._load_point_spectra(
            corrupt_r1_session, [_point_id(i) for i in range(N_POINTS)], region="R1"
        )


def test_unreadable_r1_blob_fails_the_fit_instead_of_reporting_missing(
    corrupt_r1_session,
):
    """The job fails loudly, as it did before the batched load was added.

    ``start_map_fit``'s fitting thread turns this into a ``failed`` status
    and an ``error`` frame, so the corruption reaches the operator instead
    of being served as a hole in an otherwise ordinary map.
    """
    results = {}

    with pytest.raises(zlib.error):
        MapFitService(config=_CONFIG).run_map_fit(
            session=corrupt_r1_session,
            scan_id=SCAN_UUID,
            domains=["minerals"],
            point_indices=None,
            point_coords={i: (float(i), float(i)) for i in range(N_POINTS)},
            on_point_fitted=lambda r: results.update({r.point_index: r}),
            on_progress=lambda *a: None,
            on_log=lambda *a: None,
            cancel_event=threading.Event(),
        )

    # And it fails before emitting anything, so no point was ever streamed
    # to the client carrying a status that hid the corruption.
    assert results == {}


def test_skipping_an_unreadable_blob_is_opt_in(corrupt_r1_session):
    """The tolerant path exists for the batched fluorescence load only."""
    loaded = map_fitting._load_point_spectra(
        corrupt_r1_session,
        [_point_id(i) for i in range(N_POINTS)],
        region="R1",
        skip_unreadable=True,
    )

    assert sorted(loaded) == [_point_id(0), _point_id(2)]


# ---------------------------------------------------------------------------
# Laser normalization must not be skipped silently (issue #124)
# ---------------------------------------------------------------------------


def _drop_photodiode(session, point_index):
    point = session.query(ScanPointORM).filter_by(scan_id=SCAN_UUID, point_index=point_index).one()
    point.photodiode_mean = None
    session.commit()


def _run(session, config, results=None, logs=None, point_indices=None):
    return MapFitService(config=config).run_map_fit(
        session=session,
        scan_id=SCAN_UUID,
        domains=["minerals"],
        point_indices=point_indices,
        point_coords={i: (float(i), float(i)) for i in range(N_POINTS + 3)},
        on_point_fitted=(results.append if results is not None else lambda r: None),
        on_progress=lambda *a: None,
        on_log=(lambda idx, msg: logs.append((idx, msg))) if logs is not None else (lambda *a: None),
        cancel_event=threading.Event(),
    )


def test_loupe_uses_dark_subtracted_and_applies_laser_normalization(fit_session, monkeypatch):
    """Normalized decoys must not replace Loupe's measured R1 spectra."""
    points = fit_session.query(ScanPointORM).order_by(ScanPointORM.point_index).all()
    for point, photodiode in zip(points, (1200.0, 4800.0, 2400.0)):
        point.photodiode_mean = photodiode
        fit_session.add(SpectrumORM(
            id=str(uuid.uuid4()), scan_point_id=point.id, region="R1",
            spectrum_type="laser_normalized", processing_level="normalized",
            intensities=zlib.compress(np.full(N_CHANNELS, 9999, dtype=np.float32).tobytes()),
        ))
    fit_session.commit()
    seen = []
    real = map_fitting._preprocess_r1_intensity

    def spy(x, y, *args, **kwargs):
        seen.append(y.copy())
        return real(x, y, *args, **kwargs)

    monkeypatch.setattr(map_fitting, "_preprocess_r1_intensity", spy)
    results = []
    summary = _run(fit_session, _CONFIG, results)

    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    mask = get_region_wavelength_mask(wavelength, "R1")
    assert len(seen) == N_POINTS
    for i, factor in enumerate((4.0, 1.0, 2.0)):
        raw = np.frombuffer(zlib.decompress(_spectrum_bytes(i)), dtype=np.float32)
        np.testing.assert_allclose(seen[i], raw[mask] * factor)
    assert summary.total_points == N_POINTS
    assert summary.laser_norm_not_applied == []
    assert all(point.notes == [] for point in results)


def test_a_point_without_photodiode_stops_the_map_fit(fit_session):
    _drop_photodiode(fit_session, 1)
    results = []
    with pytest.raises(ValueError, match="photodiode_mean > 0 for every requested point"):
        _run(fit_session, _CONFIG, results)
    assert results == []  # Raised before any point was fitted.


def test_the_opt_out_fits_unnormalized_points_and_records_each(fit_session):
    _drop_photodiode(fit_session, 1)
    config = {**_CONFIG, "preprocessing": {"baseline": {}, "laser_normalization": {"allow_missing_photodiode": True}}}
    results, logs = [], []
    summary = _run(fit_session, config, results, logs)

    assert summary.total_points == N_POINTS
    assert summary.laser_norm_not_applied == [1]
    notes = {r.point_index: r.notes for r in results}
    assert notes == {0: [], 1: [map_fitting.LASER_NORM_NOT_APPLIED], 2: []}
    assert (1, map_fitting.LASER_NORM_NOT_APPLIED) in logs
    prep = next(msg for idx, msg in logs if idx == -1 and msg.startswith("Preprocessing:"))
    assert "1 requested pts have no photodiode (fitted unnormalized)" in prep
    assert "background" not in prep


def test_every_point_with_photodiode_records_nothing(fit_session):
    results, logs = [], []
    summary = _run(fit_session, _CONFIG, results, logs)
    assert summary.laser_norm_not_applied == []
    assert all(r.notes == [] for r in results)
    assert not any("not applied" in msg for _, msg in logs)


def _add_point(session, i, photodiode, with_r1=True):
    pt_id = str(uuid.UUID(f"00000000-0000-0000-0000-{910 + i:012d}"))
    session.add(
        ScanPointORM(
            id=pt_id, scan_id=SCAN_UUID, point_index=i,
            x_pixel=100.0 + i, y_pixel=200.0 + i, photodiode_mean=photodiode,
        )
    )
    session.flush()
    if with_r1:
        session.add(
            SpectrumORM(
                id=str(uuid.UUID(f"00000000-0000-0000-0000-{950 + i:012d}")),
                scan_point_id=pt_id, region="R1", spectrum_type="dark_subtracted",
                processing_level="dark_subtracted", intensities=_spectrum_bytes(i),
            )
        )
    session.commit()


@pytest.fixture()
def subset_session(fit_session):
    """Points 0-2 as in fit_session, with point 1's photodiode dropped, plus:
    3 has the scan's highest photodiode and 4 has none (both outside the
    requested subset); 5 has no photodiode and no R1 spectrum."""
    _drop_photodiode(fit_session, 1)
    _add_point(fit_session, 3, 9000.0)
    _add_point(fit_session, 4, None)
    _add_point(fit_session, 5, None, with_r1=False)
    return fit_session


SUBSET = [0, 1, 5]


def test_only_requested_points_need_photodiode(subset_session):
    with pytest.raises(ValueError, match=r"2 of 3 have none \(points 1, 5\)"):
        _run(subset_session, _CONFIG, point_indices=SUBSET)


def test_the_opt_out_on_a_subset_keeps_the_scan_scale_and_notes_only_fitted_points(subset_session, monkeypatch):
    seen = []
    real = map_fitting._preprocess_r1_intensity

    def spy(wavenumber_r1, intensity_r1, *args, **kwargs):
        seen.append(np.array(intensity_r1, dtype=np.float64))
        return real(wavenumber_r1, intensity_r1, *args, **kwargs)

    monkeypatch.setattr(map_fitting, "_preprocess_r1_intensity", spy)
    config = {**_CONFIG, "preprocessing": {"baseline": {}, "laser_normalization": {"allow_missing_photodiode": True}}}
    results, logs = [], []
    summary = _run(subset_session, config, results, logs, point_indices=SUBSET)

    # Point 5 has no R1 spectrum, so nothing was fitted unnormalized there.
    assert summary.laser_norm_not_applied == [1]
    notes = {r.point_index: r.notes for r in results}
    assert notes == {0: [], 1: [map_fitting.LASER_NORM_NOT_APPLIED], 5: []}
    assert results[-1].results["minerals"].status == "missing"
    assert [idx for idx, msg in logs if msg == map_fitting.LASER_NORM_NOT_APPLIED] == [1]

    # The scale stays scan-wide: point 0 is scaled by the scan maximum
    # (point 3, outside the subset); point 1 is left unnormalized.
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=N_CHANNELS)
    r1_mask = get_region_wavelength_mask(wavelength, "R1")

    def raw(i):
        return np.frombuffer(zlib.decompress(_spectrum_bytes(i)), dtype=np.float32)[r1_mask].astype(np.float64)

    assert len(seen) == 2
    np.testing.assert_allclose(seen[0], raw(0) * (9000.0 / 4800.0), rtol=1e-5)
    np.testing.assert_allclose(seen[1], raw(1), rtol=1e-6)
