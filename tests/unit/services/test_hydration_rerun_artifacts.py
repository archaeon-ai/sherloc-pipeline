"""Re-running fitting into a populated results directory (#38).

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
from sherloc_pipeline.services.fitting import (
    FittingService,
    StaleArtifactError,
    _fit_run_marker_path,
    _read_fit_run_marker,
)
from sherloc_pipeline.services.errors import FittingError
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
    """The scan-level summary still describes a run that accepted nothing."""
    _run_fit(_service(tmp_path, veto_enabled=True), results_base)

    summary = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_hydration_summary.csv"
    assert summary.exists()
    rows = pd.read_csv(summary)
    assert bool(rows["oh_detected"].iloc[0]) is False


def _marker_path(results_base):
    return _fit_run_marker_path(
        results_base / "hydration_fit", SOL, TARGET, SCAN, "R1", "hydration"
    )


def test_run_marker_records_this_runs_accepted_count(tmp_path, results_base):
    """The marker — not the summary — is what licenses clearing the database.

    ``persist_raman_peaks`` uses it to tell "fitted, accepted nothing" (clear
    the domain's rows) from "never fitted" or "interrupted" (errors), so a
    zero-result run must leave one saying zero, overwriting the positive marker
    the previous run left.
    """
    marker = _marker_path(results_base)

    _run_fit(_service(tmp_path, veto_enabled=False), results_base)
    assert _read_fit_run_marker(marker)["accepted_peaks"] == 1

    _run_fit(_service(tmp_path, veto_enabled=True), results_base)
    assert _read_fit_run_marker(marker)["accepted_peaks"] == 0


def test_failed_stale_artifact_removal_fails_the_fit(tmp_path, results_base):
    """A cleanup that could not delete must not report success.

    Reporting success would leave the previous run's peak CSV on disk for
    ``persist_raman_peaks`` to rediscover, restoring the vetoed peak.
    """
    peaks_csv, _png, _acc = _artifact_paths(results_base)
    _run_fit(_service(tmp_path, veto_enabled=False), results_base)
    assert peaks_csv.exists()

    real_unlink = Path.unlink

    def refuse_peaks_csv(self, *args, **kwargs):
        if self == peaks_csv:
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    with patch.object(Path, "unlink", refuse_peaks_csv):
        with pytest.raises(Exception) as exc_info:
            _run_fit(_service(tmp_path, veto_enabled=True), results_base)

    assert "failed to remove stale artifact" in str(exc_info.value)
    assert peaks_csv.exists(), "the undeletable CSV is exactly the risk"
    # The marker still records an in-progress run, so persistence refuses to
    # treat this as a zero result (and does not fall back to legacy behaviour).
    assert _read_fit_run_marker(_marker_path(results_base))["status"] == "running"


def test_remove_stale_artifacts_raises_on_unlink_failure(tmp_path):
    from sherloc_pipeline.services.fitting import _remove_stale_artifacts

    doomed = tmp_path / "leftover.csv"
    doomed.write_text("center_cm1\n3300.0\n")

    with patch.object(Path, "unlink", side_effect=OSError("read-only fs")):
        with pytest.raises(StaleArtifactError):
            _remove_stale_artifacts([doomed])

    # Missing files stay non-fatal: nothing to resurrect.
    _remove_stale_artifacts([tmp_path / "absent.csv"])


def _pointless_scan_csv(results_base: Path) -> Path:
    """An R1_normalized.csv with a shift axis but no numeric point columns."""
    x = np.linspace(1800.0, 4000.0, 64)
    results_base.mkdir(parents=True, exist_ok=True)
    csv_path = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized.csv"
    pd.DataFrame({"raman_shift": x, "mean": np.full_like(x, 100.0)}).to_csv(
        csv_path, index=False
    )
    return csv_path


def test_hydration_fit_with_no_point_columns_leaves_no_completed_marker(tmp_path):
    """A run that fits nothing must not complete.

    Completing would write an `accepted_peaks: 0` marker, which persistence
    would then read as "the fit found nothing" and use to clear peaks a real
    earlier run put in the database. The in-progress marker must survive: it is
    what stops persistence falling back to pre-marker behaviour.
    """
    base = tmp_path / "results" / SCAN
    _pointless_scan_csv(base)

    with pytest.raises(FittingError) as exc_info:
        _run_fit(_service(tmp_path, veto_enabled=False), base)
    assert "No point columns found to fit" in str(exc_info.value)

    marker = _fit_run_marker_path(
        base / "hydration_fit", SOL, TARGET, SCAN, "R1", "hydration",
    )
    assert _read_fit_run_marker(marker)["status"] == "running"


def test_organics_fit_with_no_point_columns_leaves_no_completed_marker(tmp_path):
    """Same guard on the organics path, which shares the persistence gate."""
    base = tmp_path / "results" / SCAN
    base.mkdir(parents=True, exist_ok=True)
    x = np.linspace(800.0, 2000.0, 64)
    pd.DataFrame({"raman_shift": x, "mean": np.full_like(x, 100.0)}).to_csv(
        base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_baselined.csv", index=False
    )

    service = _service(tmp_path, veto_enabled=False)
    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = base
    scan_ctx.results_dir = base
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = base

    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion), \
         pytest.raises(FittingError) as exc_info:
        service.fit_organics(sol=SOL, target=TARGET, scan=SCAN)
    assert "No point columns found to fit" in str(exc_info.value)

    marker = _fit_run_marker_path(
        base / "organics_fit", SOL, TARGET, SCAN, "R1", "organics",
    )
    assert _read_fit_run_marker(marker)["status"] == "running"


def test_accepted_summary_write_failure_fails_the_fit(tmp_path, results_base):
    """A positive rerun whose accepted-peaks export fails must not complete.

    Both runs accept a peak, so the previous run's summary is still on disk
    when the rewrite fails. Reporting success would leave those older R2 and
    centre values under a `completed` marker, and the assembler would read them
    as this run's result. The run must instead drop the stale file and fail.
    """
    _peaks_csv, _png, acc_csv = _artifact_paths(results_base)

    first = _run_fit(_service(tmp_path, veto_enabled=False), results_base)
    assert first.metadata["total_accepted_peaks"] == 1
    assert acc_csv.exists()

    real_to_csv = pd.DataFrame.to_csv

    def refuse_accepted_summary(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(acc_csv):
            raise OSError(28, "No space left on device")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    with patch.object(pd.DataFrame, "to_csv", refuse_accepted_summary):
        with pytest.raises(FittingError) as exc_info:
            _run_fit(_service(tmp_path, veto_enabled=False), results_base)

    assert "accepted-peaks summary" in str(exc_info.value)
    assert not acc_csv.exists(), "the previous run's R2 values must not survive"
    assert _read_fit_run_marker(_marker_path(results_base))["status"] == "running"


# --- Minerals ---------------------------------------------------------------
#
# The mineral path has the same shape as hydration: per-point peak tables that
# `assemble_accepted_peaks` globs, plus a scan-level AICc summary it reads R2
# from, all certified by one run marker.

def _mineral_scan_csv(results_base: Path) -> Path:
    """A baselined R1 CSV with one clean, fittable band."""
    x = np.linspace(900.0, 1300.0, 400)
    peak = 900.0 / (1.0 + ((x - 1090.0) / 6.0) ** 2)
    y = peak + np.random.default_rng(38).normal(0.0, 1.0, size=x.shape)

    results_base.mkdir(parents=True, exist_ok=True)
    csv_path = results_base / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_baselined.csv"
    pd.DataFrame({"raman_shift": x, "0": y}).to_csv(csv_path, index=False)
    return csv_path


def _run_mineral_fit(service, results_base):
    scan_ctx = MagicMock()
    scan_ctx.base_data_dir = results_base
    scan_ctx.results_dir = results_base

    ingestion = MagicMock()
    ingestion.get_results_path.return_value = results_base

    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion):
        return service.fit_minerals(sol=SOL, target=TARGET, scan=SCAN)


@pytest.fixture
def mineral_base(tmp_path):
    base = tmp_path / "results" / SCAN
    _mineral_scan_csv(base)
    return base


def _mineral_paths(results_base):
    fit_dir = results_base / "minerals_fit"
    return (
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_point0_fit_peaks.csv",
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_point0_fit.png",
        fit_dir / f"{SOL}_{TARGET}_{SCAN}_R1_accepted_peaks.csv",
    )


def _mineral_marker(results_base):
    return _fit_run_marker_path(
        results_base / "minerals_fit", SOL, TARGET, SCAN, "R1", "minerals"
    )


def test_mineral_point_failure_cleans_its_artifacts_and_blocks_completion(
    tmp_path, mineral_base
):
    """A point whose fit raises must not leave the earlier run's peak table.

    The table is what `assemble_accepted_peaks` globs, so a swallowed per-point
    failure under a `completed` marker would republish the previous run's peaks
    as this run's result.
    """
    peaks_csv, png, _acc = _mineral_paths(mineral_base)

    _run_mineral_fit(_service(tmp_path, veto_enabled=False), mineral_base)
    assert peaks_csv.exists()

    with patch("sherloc_pipeline.core.fitting.fit_spectrum",
               side_effect=RuntimeError("solver diverged")):
        with pytest.raises(FittingError) as exc_info:
            _run_mineral_fit(_service(tmp_path, veto_enabled=False), mineral_base)

    assert "Mineral fitting failed for point" in str(exc_info.value)
    assert not peaks_csv.exists(), "stale peak table would be re-persisted"
    assert not png.exists()
    assert _read_fit_run_marker(_mineral_marker(mineral_base))["status"] == "running"


def test_mineral_aicc_summary_write_failure_fails_the_fit(tmp_path, mineral_base):
    """The AICc summary supplies per-point R2 to persistence; it must be current."""
    aicc_csv = mineral_base / f"{SOL}_{TARGET}_{SCAN}_R1_fit_aicc_summary.csv"

    _run_mineral_fit(_service(tmp_path, veto_enabled=False), mineral_base)
    assert aicc_csv.exists()

    real_to_csv = pd.DataFrame.to_csv

    def refuse_aicc(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(aicc_csv):
            raise OSError(28, "No space left on device")
        return real_to_csv(self, path_or_buf, *args, **kwargs)

    with patch.object(pd.DataFrame, "to_csv", refuse_aicc):
        with pytest.raises(FittingError) as exc_info:
            _run_mineral_fit(_service(tmp_path, veto_enabled=False), mineral_base)

    assert "AICc summary" in str(exc_info.value)
    assert not aicc_csv.exists(), "the previous run's R2 values must not survive"
    assert _read_fit_run_marker(_mineral_marker(mineral_base))["status"] == "running"
