"""Despiked input for broad domains, non-despiked for minerals.

Ken's ruling (2026-10-21): organics, hydration and fluorescence fit the
despiked spectrum (broad, hard to confuse with cosmic rays); minerals stay
non-despiked because despiking can clip sharp mineral peaks. These tests pin
the hydration point path and map mode to that rule. All spectra are synthetic.
"""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sherloc_pipeline.config import load_config
from sherloc_pipeline.core.hydration_veto import HydrationVetoConfig
from sherloc_pipeline.services import fitting as fitting_service
from sherloc_pipeline.services import map_fitting
from sherloc_pipeline.services.fitting import FittingService, _fit_point_hydration
from sherloc_pipeline.services.runtime import RuntimeContext

SOL = "0921"
TARGET = "Test_Target"
SCAN = "detail_1"


def _axis(lo=1800.0, hi=4000.0, n=256):
    return np.linspace(lo, hi, n)


def _noise(x, seed=38):
    return 100.0 + np.random.default_rng(seed).normal(0.0, 4.0, size=x.shape)


def _with_spike(x, y, center, amplitude=6000.0):
    y = y.copy()
    idx = int(np.argmin(np.abs(x - center)))
    y[idx] += amplitude
    y[idx + 1] += amplitude * 0.9
    return y


# --- Point hydration (services.fitting.fit_hydration) ---------------------


def _write_r1(results_base: Path, level: str, y: np.ndarray, x: np.ndarray) -> None:
    results_base.mkdir(parents=True, exist_ok=True)
    path = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_{level}.csv"
    pd.DataFrame({"raman_shift": x, "0": y}).to_csv(path, index=False)


def _run_hydration(tmp_path, results_base, **fitting_overrides):
    cfg = copy.deepcopy(load_config())
    cfg.fitting = dict(cfg.fitting or {})
    cfg.fitting["parallel_workers"] = 1
    cfg.fitting.update(fitting_overrides)
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = results_base
    scan_ctx.results_dir = results_base
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_base
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion):
        return FittingService(context=context).fit_hydration(
            sol=SOL, target=TARGET, scan=SCAN
        )


def test_point_hydration_fits_the_despiked_branch(tmp_path):
    x = _axis()
    clean = _noise(x)
    base = tmp_path / "results" / SCAN
    # The non-despiked spectrum carries a cosmic ray the fit would accept;
    # the despiked branch does not.
    _write_r1(base, "normalized", _with_spike(x, clean, 3300.0), x)
    _write_r1(base, "normalized_despiked", clean, x)

    result = _run_hydration(tmp_path, base)

    assert result.metadata["total_accepted_peaks"] == 0


def test_point_hydration_without_despiked_branch_is_unchanged(tmp_path):
    x = _axis()
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", _with_spike(x, _noise(x), 3300.0), x)

    result = _run_hydration(tmp_path, base)

    assert result.metadata["total_accepted_peaks"] == 1


def test_point_hydration_baselined_fallback_prefers_its_despiked_branch(tmp_path):
    x = _axis()
    clean = _noise(x)
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized_baselined", _with_spike(x, clean, 3300.0), x)
    _write_r1(base, "normalized_despiked_baselined", clean, x)

    result = _run_hydration(tmp_path, base, hydration_use_prebaseline=False)

    assert result.metadata["total_accepted_peaks"] == 0


def test_point_hydration_veto_receives_the_non_despiked_column(tmp_path, monkeypatch):
    x = _axis()
    raw = _with_spike(x, _noise(x, seed=5), 3300.0)
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", raw, x)
    # A spike the despiker missed, so the fit has a candidate to veto.
    _write_r1(base, "normalized_despiked", _with_spike(x, _noise(x), 3300.0), x)
    seen = []
    real = fitting_service.despike_for_veto

    def spy(x_arr, y_arr, params):
        seen.append(np.asarray(y_arr))
        return real(x_arr, y_arr, params)

    monkeypatch.setattr(fitting_service, "despike_for_veto", spy)
    _run_hydration(
        tmp_path, base, hydration_cr_veto={"enabled": True, "action": "reject"}
    )

    assert len(seen) == 1
    written = pd.read_csv(base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized.csv")["0"]
    np.testing.assert_allclose(seen[0], written.to_numpy(float))


def test_hydration_veto_scores_the_non_despiked_spectrum(tmp_path, monkeypatch):
    x = _axis()
    y_fit = _with_spike(x, _noise(x), 3300.0)
    y_raw = _with_spike(x, _noise(x, seed=5), 3300.0)
    seen = []
    real = fitting_service.despike_for_veto

    def spy(x_arr, y_arr, params):
        seen.append(y_arr)
        return real(x_arr, y_arr, params)

    monkeypatch.setattr(fitting_service, "despike_for_veto", spy)
    oh_roi, oh_plot = (2800.0, 3900.0), (2600.0, 4000.0)
    _fit_point_hydration(
        {"point_idx": 1, "y": y_fit, "y_raw": y_raw},
        x=x,
        fit_cfg_oh={
            "max_peaks": 2, "fit_fwhm_min_initial_cm1": 50.0,
            "filter_fwhm_min_cm1": 50.0, "fwhm_max_cm1": 300.0,
            "min_snr": 3.0, "min_seed_snr": 3.0, "r_squared_min": 0.25,
            "peak_separation_cm1": 25,
            "noise_estimation": {"window": [2000.0, 2100.0]},
            "parsimony": {"model_selection": "ftest", "ftest_alpha": 0.01},
            "posthoc_filters": {},
        },
        oh_roi=oh_roi,
        oh_plot=oh_plot,
        oh_mask=(x >= oh_roi[0]) & (x <= oh_roi[1]),
        plot_mask_oh=(x >= oh_plot[0]) & (x <= oh_plot[1]),
        n_edge=5, min_snr=3.0, r2_min=0.25, center_lo=3000.0, center_hi=3900.0,
        out_dir=str(tmp_path), sol=SOL, target=TARGET, scan=SCAN,
        veto_cfg=HydrationVetoConfig.from_fitting_config(
            {"hydration_cr_veto": {"enabled": True, "action": "reject"}}
        ),
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], y_raw)


# --- Map mode (services.map_fitting._fit_raman_domain) --------------------


def _captured_fit_input(monkeypatch, domain, x, y):
    captured = []

    def fake_fit(x_arr, y_arr, cfg, roi=None):
        captured.append(np.asarray(y_arr).copy())
        return SimpleNamespace(peaks=[]), None

    monkeypatch.setattr(map_fitting, "fit_spectrum", fake_fit)
    map_fitting._fit_raman_domain(x, y, {"fitting": {}}, domain)
    return captured[0]


def test_map_minerals_fit_the_non_despiked_spectrum(monkeypatch):
    x = _axis(700.0, 1200.0)
    y = _with_spike(x, _noise(x), 900.0)  # outside the sulfate guard

    seen = _captured_fit_input(monkeypatch, "minerals", x, y)

    np.testing.assert_allclose(seen, y)


def test_map_organics_fit_the_despiked_spectrum(monkeypatch):
    x = _axis(700.0, 1200.0)
    y = _with_spike(x, _noise(x), 900.0)  # outside the sulfate guard

    seen = _captured_fit_input(monkeypatch, "organics", x, y)

    assert seen.max() < y.max() / 2
