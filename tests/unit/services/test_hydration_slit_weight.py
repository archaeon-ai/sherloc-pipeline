"""The batch hydration fit carries no slit-width preference.

With no slit term, the hydration fit of a spectrum does not depend on the
spectrum's overall scale: the same spectrum multiplied by a constant gives the
same centers and widths, and amplitudes multiplied by that constant. These tests
drive the real ``FittingService.fit_hydration`` over a synthetic scan.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.config import load_config
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.services.runtime import RuntimeContext

SOL = "0921"
TARGET = "Test_Target"
SCAN = "detail_1"
SCALE = 50.0

# Tolerances for the scale test: centers and FWHM in cm-1, amplitude relative.
CENTER_TOL_CM1 = 0.01
FWHM_TOL_CM1 = 0.01
AMPLITUDE_REL_TOL = 1e-4


def _gaussian(x, center, amplitude, fwhm):
    return amplitude * np.exp(-4.0 * np.log(2.0) * (x - center) ** 2 / fwhm ** 2)


def _write_scan(results_base: Path) -> None:
    """Point 0: two overlapping O-H bands on an offset with noise. Point 1: point 0 times SCALE."""
    x = np.linspace(1800.0, 4000.0, 600)
    y = (
        100.0
        + _gaussian(x, 3300.0, 40.0, 220.0)
        + _gaussian(x, 3550.0, 25.0, 110.0)
        + np.random.default_rng(7).normal(0.0, 1.0, size=x.shape)
    )
    results_base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"raman_shift": x, "0": y, "1": SCALE * y}).to_csv(
        results_base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized.csv", index=False
    )


def _service(tmp_path) -> FittingService:
    cfg = copy.deepcopy(load_config())
    cfg.fitting = dict(cfg.fitting or {})
    cfg.fitting["parallel_workers"] = 1
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    return FittingService(context=context)


def _run_fit(tmp_path):
    """Run fit_hydration; return the config passed to the worker and the per-point results."""
    results_base = tmp_path / "results" / SCAN
    _write_scan(results_base)
    service = _service(tmp_path)

    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = results_base
    scan_ctx.results_dir = results_base
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_base

    captured = {}
    run_parallel = service._run_parallel

    def _capture(worker, inputs, n_workers, label):
        captured["fit_cfg_oh"] = worker.keywords["fit_cfg_oh"]
        captured["results"] = run_parallel(worker, inputs, n_workers, label)
        return captured["results"]

    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion), \
         patch.object(service, "_run_parallel", side_effect=_capture):
        service.fit_hydration(sol=SOL, target=TARGET, scan=SCAN)
    return captured


@pytest.fixture(scope="module")
def fit(tmp_path_factory):
    return _run_fit(tmp_path_factory.mktemp("hydration_slit"))


def test_batch_hydration_fit_config_has_slit_weight_zero(fit):
    assert load_config().fitting["slit_pref_weight"] > 0, "precondition: the mineral setting is nonzero"
    assert fit["fit_cfg_oh"]["slit_pref_weight"] == 0.0


def test_hydration_fit_does_not_depend_on_spectrum_scale(fit):
    by_point = {r["point_idx"]: r["accepted_peaks"] for r in fit["results"]}
    base = sorted(by_point[0], key=lambda r: r["center_cm1"])
    scaled = sorted(by_point[1], key=lambda r: r["center_cm1"])

    assert len(base) == 2, "precondition: both bands are accepted"
    assert len(scaled) == len(base)
    for b, s in zip(base, scaled):
        assert s["center_cm1"] == pytest.approx(b["center_cm1"], abs=CENTER_TOL_CM1)
        assert s["fwhm_cm1"] == pytest.approx(b["fwhm_cm1"], abs=FWHM_TOL_CM1)
        assert s["amplitude_a"] == pytest.approx(SCALE * b["amplitude_a"], rel=AMPLITUDE_REL_TOL)
