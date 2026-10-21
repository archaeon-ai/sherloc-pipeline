"""Despiked R1 outputs never outlive the run that wrote them (PR #130).

Hydration and organics fit the despiked branch whenever it exists, so
process-scan removes the scan's R1_normalized_despiked.csv and
R1_normalized_despiked_baselined.csv before the despike step and rewrites them
only when despiking runs and succeeds. Runs run_scan on the committed sol_0921
fixture workspace.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sherloc_pipeline.services.errors import PreprocessingError
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.services.preprocessing import PreprocessingService

SOL, SCAN = "0921", "detail_1"
TARGET = "Despike_Stale"
STALE = "stale despiked output from an earlier run\n"


def _scan_dir(tmp_path) -> Path:
    return tmp_path / "results" / TARGET / f"{SOL}_{SCAN}"


def _despiked_paths(tmp_path):
    scan_dir = _scan_dir(tmp_path)
    return [
        scan_dir / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked.csv",
        scan_dir / f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked_baselined.csv",
    ]


def _plant_stale(tmp_path):
    paths = _despiked_paths(tmp_path)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        path.write_text(STALE)
    return paths


def _run(tmp_path, fixtures_path, despike_method):
    return PreprocessingService().run_scan(
        sol=SOL, target=TARGET, scan=SCAN,
        data_dir=fixtures_path / "loupe", results_dir=tmp_path / "results",
        generate_plots=False, despike_method=despike_method,
    )


def test_despike_off_removes_earlier_despiked_outputs(tmp_path, fixtures_path):
    paths = _plant_stale(tmp_path)

    result = _run(tmp_path, fixtures_path, "none")

    assert not any(path.exists() for path in paths)
    assert result.metadata["despike_removed_stale"] == [path.name for path in paths]
    assert any("Despiking is off" in w for w in result.warnings)

    # Hydration then fits the non-despiked level and says so.
    scan_dir = _scan_dir(tmp_path)
    scan_ctx = MagicMock(base_data_dir=scan_dir, results_dir=scan_dir)
    ingestion = MagicMock()
    ingestion.get_results_path.return_value = scan_dir
    with patch("sherloc_pipeline.services.fitting.resolve_scan_context",
               return_value=scan_ctx), \
         patch("sherloc_pipeline.core.data_ingestion.DataIngestion",
               return_value=ingestion):
        fit = FittingService().fit_hydration(sol=SOL, target=TARGET, scan=SCAN)
    assert fit.metadata["hydration_input_despiked"] is False
    assert any("No despiked R1 branch found" in w for w in fit.warnings)


def test_a_failed_despike_leaves_no_despiked_outputs(tmp_path, fixtures_path, monkeypatch):
    paths = _plant_stale(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic despiker failure")

    monkeypatch.setattr(
        "sherloc_pipeline.core.preprocessing.despike_r1_dataframe", boom
    )
    with pytest.raises(PreprocessingError, match="despiking"):
        _run(tmp_path, fixtures_path, "modz")

    assert not any(path.exists() for path in paths)


def test_despike_on_writes_fresh_despiked_outputs(tmp_path, fixtures_path):
    paths = _plant_stale(tmp_path)

    result = _run(tmp_path, fixtures_path, "modz")

    for path in paths:
        assert path.exists()
        assert path.read_text() != STALE
    assert result.metadata["despike_removed_stale"] == [path.name for path in paths]
    assert not any("Despiking is off" in w for w in result.warnings)
