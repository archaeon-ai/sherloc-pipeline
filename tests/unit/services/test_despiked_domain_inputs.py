"""Despiked input for broad domains, non-despiked for minerals.

Organics, hydration and fluorescence fit the
despiked spectrum (broad, hard to confuse with cosmic rays); minerals stay
non-despiked because despiking can clip sharp mineral peaks. These tests pin
the hydration point path and map mode to that rule. All spectra are synthetic.
"""

from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from sherloc_pipeline.config import load_config
from sherloc_pipeline.services import map_fitting

from sherloc_pipeline.services.errors import FittingError
from sherloc_pipeline.services.fitting import (
    FittingService,
    _fit_run_marker_path,
    _read_fit_run_marker,
)
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


def _run_hydration(tmp_path, results_base, config_file=None, **fitting_overrides):
    cfg = load_config(config_file)
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


def test_hydration_result_rows_with_older_config(tmp_path):
    x = _axis()
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", _with_spike(x, _noise(x), 3300.0), x)
    accepted = base / "hydration_fit" / f"{SOL}_{TARGET}_{SCAN}_R1_hydration_accepted_peaks.csv"

    _run_hydration(tmp_path, base)
    expected = pd.read_csv(accepted)
    assert len(expected) == 1
    assert set(expected.columns) == {
        "point", "center_cm1", "amplitude_a", "fwhm_cm1", "snr", "band",
        "r2", "sharpness_ratio", "pass_sharpness",
    }

    # A config saved by an earlier version can request peak flags.
    settings = yaml.safe_load(files("sherloc_pipeline").joinpath("config.yaml").read_text())
    legacy_key = "_".join(("hydration", "cr", "veto"))
    settings["fitting"][legacy_key] = {"enabled": True, "action": "flag"}
    config_file = tmp_path / "older.yaml"
    config_file.write_text(yaml.safe_dump(settings))

    _run_hydration(tmp_path, base, config_file=config_file)
    pd.testing.assert_frame_equal(pd.read_csv(accepted), expected)


def _marker_input(results_base):
    marker = _fit_run_marker_path(
        results_base / "hydration_fit", SOL, TARGET, SCAN, "R1", "hydration"
    )
    return _read_fit_run_marker(marker)["input"]


def test_point_hydration_records_the_despiked_input(tmp_path):
    x = _axis()
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", _noise(x), x)
    _write_r1(base, "normalized_despiked", _noise(x), x)

    result = _run_hydration(tmp_path, base)

    name = f"{SOL}_{TARGET}_{SCAN}_R1_normalized_despiked.csv"
    assert result.metadata["hydration_input"] == name
    assert result.metadata["hydration_input_despiked"] is True
    assert not any("No despiked" in w for w in result.warnings)
    assert _marker_input(base) == {"file": name, "despiked": True}


def test_point_hydration_without_despiked_branch_says_so(tmp_path):
    x = _axis()
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", _noise(x), x)

    result = _run_hydration(tmp_path, base)

    name = f"{SOL}_{TARGET}_{SCAN}_R1_normalized.csv"
    assert result.metadata["hydration_input"] == name
    assert result.metadata["hydration_input_despiked"] is False
    assert any("No despiked R1 branch found" in w for w in result.warnings)
    assert _marker_input(base) == {"file": name, "despiked": False}


def test_point_hydration_refuses_mismatched_axes(tmp_path):
    x = _axis()
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized", _noise(x), x)
    _write_r1(base, "normalized_despiked", _noise(x), x + 1.0)

    with pytest.raises(FittingError, match="different Raman-shift axes"):
        _run_hydration(tmp_path, base)


def test_point_hydration_baselined_fallback_prefers_its_despiked_branch(tmp_path):
    x = _axis()
    clean = _noise(x)
    base = tmp_path / "results" / SCAN
    _write_r1(base, "normalized_baselined", _with_spike(x, clean, 3300.0), x)
    _write_r1(base, "normalized_despiked_baselined", clean, x)

    result = _run_hydration(tmp_path, base, hydration_use_prebaseline=False)

    assert result.metadata["total_accepted_peaks"] == 0


# --- Map mode (services.map_fitting._fit_raman_domain) --------------------


def _captured_fit_input(monkeypatch, domain, x, y):
    captured = []

    def fake_fit(x_arr, y_arr, cfg, roi=None, **_kwargs):
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
