"""The batch hydration pre-fit gate (``fitting.hydration_prefit_min_snr``) apart from the
``pass_snr`` threshold of hydration components (``fitting.hydration_min_snr``)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from sherloc_pipeline.core import fitting as core_fitting
from sherloc_pipeline.services import fitting as service_fitting
from sherloc_pipeline.services.fitting import _fit_point_hydration
from tests.unit.services.test_hydration_rerun_artifacts import (
    _run_fit,
    _service,
    results_base,  # noqa: F401  (pytest fixture)
)

CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[3] / "src/sherloc_pipeline/config.yaml").read_text())["fitting"]


def test_shipped_gate_values():
    assert CONFIG["hydration_prefit_min_snr"] == 2.0
    assert CONFIG["hydration_min_snr"] == 3.0
    assert CONFIG["hydration_r2_min"] == 0.25


def _point(tmp_path, prefit_min_snr, monkeypatch):
    """Run the worker on one point; return whether the fit was reached, and the point's
    pre-fit statistic."""
    x = np.arange(1900., 3901., 10.)
    y = np.cos(np.arange(x.size)) + core_fitting.gaussian(x, 3400., 0.5, 120.)
    roi = (2800., 3900.)
    mask = (x >= roi[0]) & (x <= roi[1])
    cfg = {"hydration_noise_window": (2000., 2100.)}
    seed = core_fitting.hydration_seed_signal(y[mask], 5)
    statistic = float(np.max(seed)) / core_fitting.hydration_noise(x, y, cfg["hydration_noise_window"], x[mask], seed)
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(1)
        return core_fitting.FitResult([], 0., 0., 0, []), np.zeros(int(mask.sum())), np.zeros_like(x)

    monkeypatch.setattr(core_fitting, "fit_hydration_spectrum", fake_fit)
    _fit_point_hydration({"point_idx": 0, "y": y}, x=x, fit_cfg_oh=cfg, oh_roi=roi, oh_plot=roi,
                         oh_mask=mask, plot_mask_oh=mask, n_edge=5,
                         prefit_min_snr=prefit_min_snr(statistic), r2_min=0.25,
                         center_lo=3000., center_hi=3900., out_dir=str(tmp_path),
                         sol="1", target="sample", scan="scan")
    return bool(calls), statistic


def test_a_point_below_the_gate_is_not_fitted_and_one_at_it_is(tmp_path, monkeypatch):
    reached_below, statistic = _point(tmp_path, lambda s: s + 0.01, monkeypatch)
    reached_at, _ = _point(tmp_path, lambda s: s, monkeypatch)
    assert 2.0 <= statistic < 3.0, statistic
    assert not reached_below
    assert reached_at
    # Between 2 and 3: fitted at the shipped gate, not at the earlier 3.0.
    assert _point(tmp_path, lambda s: CONFIG["hydration_prefit_min_snr"], monkeypatch)[0]
    assert not _point(tmp_path, lambda s: 3.0, monkeypatch)[0]


def _seen_gates(tmp_path, results_base, monkeypatch, overrides, drop=()):
    """The pre-fit gate and the components' pass_snr threshold fit_hydration hands the worker."""
    service = _service(tmp_path, **overrides)
    for key in drop:
        service.context.config.fitting.pop(key)
    seen = {}
    worker = service_fitting._fit_point_hydration

    def spy(point_data, **kwargs):
        seen["prefit_min_snr"] = kwargs["prefit_min_snr"]
        seen["min_snr"] = kwargs["fit_cfg_oh"]["min_snr"]
        return worker(point_data, **kwargs)

    monkeypatch.setattr(service_fitting, "_fit_point_hydration", spy)
    _run_fit(service, results_base)
    return seen


@pytest.mark.parametrize(("overrides", "drop", "expected"), [
    ({}, (), {"prefit_min_snr": 2.0, "min_snr": 3.0}),
    ({"hydration_prefit_min_snr": 2.5}, (), {"prefit_min_snr": 2.5, "min_snr": 3.0}),
    # Without the key the gate falls back to hydration_min_snr, as before the split.
    ({"hydration_min_snr": 4.0}, ("hydration_prefit_min_snr",), {"prefit_min_snr": 4.0, "min_snr": 4.0}),
])
def test_fit_hydration_passes_the_gate_and_the_pass_snr_threshold(tmp_path, results_base, monkeypatch,
                                                                   overrides, drop, expected):
    assert _seen_gates(tmp_path, results_base, monkeypatch, overrides, drop) == expected
