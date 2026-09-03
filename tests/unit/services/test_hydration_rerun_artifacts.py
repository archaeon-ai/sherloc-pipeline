"""Re-running hydration fitting into a populated results directory (#38).

The cosmic-ray veto is a config flag, so the realistic upgrade path is
"fit with it off, then fit again with it on, same results directory". Artifacts
are only written when peaks survive, so unless the fitter actively removes what
it no longer produces, the earlier run's per-point CSV survives — and
``persist_raman_peaks`` rediscovers exactly that CSV, restoring the peak the
veto rejected. These tests drive the real ``FittingService.fit_hydration`` end
to end over a synthetic scan to pin that down.
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


def _cosmic_ray_scan_csv(results_base: Path) -> Path:
    """An R1_normalized.csv whose single point contains only a cosmic ray."""
    x = np.linspace(1800.0, 4000.0, 256)
    y = 100.0 + np.random.default_rng(38).normal(0.0, 4.0, size=x.shape)
    idx = int(np.argmin(np.abs(x - 3300.0)))
    y[idx] += 6000.0
    y[idx + 1] += 5400.0

    results_base.mkdir(parents=True, exist_ok=True)
    csv_path = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized.csv"
    pd.DataFrame({"raman_shift": x, "0": y}).to_csv(csv_path, index=False)
    return csv_path


def _service(tmp_path, veto_enabled: bool) -> FittingService:
    cfg = copy.deepcopy(load_config())
    cfg.fitting = dict(cfg.fitting or {})
    cfg.fitting["parallel_workers"] = 1
    cfg.fitting["hydration_cr_veto"] = {
        "enabled": veto_enabled, "action": "reject",
    }
    context = RuntimeContext.bootstrap(
        data_dir=tmp_path / "data", results_dir=tmp_path / "results", config=cfg
    )
    return FittingService(context=context)


def _run_fit(service, results_base):
    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = results_base
    scan_ctx.results_dir = results_base

    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_base

    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion):
        return service.fit_hydration(sol=SOL, target=TARGET, scan=SCAN)


@pytest.fixture
def results_base(tmp_path):
    base = tmp_path / "results" / SCAN
    _cosmic_ray_scan_csv(base)
    return base


def _artifact_paths(results_base):
    fit_dir = results_base / "hydration_fit"
    return (
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_point0_hydration_peaks.csv",
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_point0_hydration_fit.png",
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_hydration_accepted_peaks.csv",
    )


def test_veto_off_then_on_leaves_no_hydration_artifacts(tmp_path, results_base):
    peaks_csv, png, acc_csv = _artifact_paths(results_base)

    off = _run_fit(_service(tmp_path, veto_enabled=False), results_base)
    assert off.metadata["total_accepted_peaks"] == 1
    assert peaks_csv.exists() and png.exists() and acc_csv.exists()

    on = _run_fit(_service(tmp_path, veto_enabled=True), results_base)
    assert on.metadata["total_accepted_peaks"] == 0
    assert not peaks_csv.exists(), "stale per-point CSV would be re-persisted"
    assert not png.exists(), "stale overlay would still show the vetoed fit"
    assert not acc_csv.exists(), "stale accepted-peaks summary would read as current"


def test_summary_csv_is_still_written_on_a_zero_result_run(tmp_path, results_base):
    """The scan-level summary is the evidence the fit actually ran.

    ``persist_raman_peaks`` uses it to tell "fitted, accepted nothing" (clear
    the domain's rows) from "never fitted" (an error), so it must survive a run
    that produced no peaks.
    """
    _run_fit(_service(tmp_path, veto_enabled=True), results_base)

    summary = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_hydration_summary.csv"
    assert summary.exists()
    rows = pd.read_csv(summary)
    assert bool(rows["oh_detected"].iloc[0]) is False
