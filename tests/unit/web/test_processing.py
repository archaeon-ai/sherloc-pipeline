"""Baseline, fitting, despike, and background route tests."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


def _make_test_spectrum():
    """Create a simple test spectrum with a peak."""
    wn = np.linspace(700, 1200, 100).tolist()
    # Gaussian peak at 1000 cm-1 on a sloping baseline
    x = np.array(wn)
    intensity = (
        500.0
        + 0.5 * (x - 700)
        + 200 * np.exp(-((x - 1000) ** 2) / (2 * 20**2))
        + np.random.RandomState(42).normal(0, 5, len(x))
    ).tolist()
    return wn, intensity


def _make_spiked_spectrum():
    """Create a test spectrum with artificial spikes for despike testing.

    Spike positions are chosen to avoid protected spectral regions
    (laser window ~600-700 cm-1, sulfate guard ~990-1050 cm-1).
    """
    rng = np.random.RandomState(42)
    wn = np.linspace(700, 1200, 200)
    # Smooth signal with a gentle peak away from spike locations
    intensity = 500.0 + 50.0 * np.exp(-((wn - 1000) ** 2) / (2 * 30**2))
    intensity += rng.normal(0, 3, len(wn))
    # Insert obvious spikes at positions that map to safe wavenumber regions:
    # idx 20 -> ~750 cm-1 (above laser window, below sulfate)
    # idx 170 -> ~1126 cm-1 (above sulfate guard)
    spike_indices = [20, 170]
    for idx in spike_indices:
        intensity[idx] += 2000.0  # massive spike
    return wn.tolist(), intensity.tolist(), spike_indices


@pytest.mark.asyncio
async def test_baseline(client):
    wn, intensity = _make_test_spectrum()
    resp = await client.post(
        "/api/process/baseline",
        json={"wavenumber": wn, "intensity": intensity},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert len(data["raw"]) == len(wn)
    assert len(data["baseline"]) == len(wn)
    assert len(data["corrected"]) == len(wn)
    assert data["params_used"]["method"] == "aspls"
    assert data["params_used"]["lam"] == 1_000_000.0


@pytest.mark.asyncio
async def test_baseline_custom_params(client):
    wn, intensity = _make_test_spectrum()
    resp = await client.post(
        "/api/process/baseline",
        json={
            "wavenumber": wn,
            "intensity": intensity,
            "params": {"method": "aspls", "lam": 5e5, "max_iter": 5},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["params_used"]["lam"] == 5e5
    assert data["params_used"]["max_iter"] == 5


@pytest.mark.asyncio
async def test_baseline_length_mismatch(client):
    resp = await client.post(
        "/api/process/baseline",
        json={"wavenumber": [1, 2, 3], "intensity": [1, 2]},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_baseline_not_monotonic(client):
    resp = await client.post(
        "/api/process/baseline",
        json={"wavenumber": [3, 2, 1], "intensity": [1, 2, 3]},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_baseline_unsupported_method(client):
    resp = await client.post(
        "/api/process/baseline",
        json={
            "wavenumber": [1, 2, 3],
            "intensity": [1, 2, 3],
            "params": {"method": "unknown"},
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_fit(client):
    wn, intensity = _make_test_spectrum()
    resp = await client.post(
        "/api/process/fit",
        json={"wavenumber": wn, "intensity": intensity},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert "peaks" in data
    assert data["n_peaks"] == len(data["peaks"])
    assert len(data["residual"]) == len(wn)
    assert len(data["baseline"]) == len(wn)
    assert len(data["corrected"]) == len(wn)
    assert data["provenance"]["calibration_version"] == "loupe_v5.1.5a"
    assert data["model_selection_method"] == "aicc"


@pytest.mark.asyncio
async def test_fit_length_mismatch(client):
    resp = await client.post(
        "/api/process/fit",
        json={"wavenumber": [1, 2, 3], "intensity": [1, 2]},
    )
    assert resp.status_code == 400


def _make_fluor_doublet_spectrum():
    """Synthetic Ce3+ doublet on a nm-axis (Berry Hollow signature).

    Two Gaussians: 1a ~304 nm, 1b ~325 nm.
    """
    rng = np.random.RandomState(42)
    nm = np.linspace(276.0, 355.0, 400)
    intensity = (
        100.0
        + 800.0 * np.exp(-((nm - 304.0) ** 2) / (2 * 6.0**2))
        + 600.0 * np.exp(-((nm - 325.0) ** 2) / (2 * 6.5**2))
        + rng.normal(0, 8, len(nm))
    )
    return nm.tolist(), intensity.tolist()


@pytest.mark.asyncio
async def test_fit_routes_fluorescence_to_fluor_fitter(client):
    """Regression: domain="fluorescence" must hit fit_fluorescence_spectrum,
    not the Raman fit_spectrum. Bug history: a previous router change sent
    fluorescence fits through the cm-1-space Raman fitter and produced
    single-peak fits across the Berry Hollow Ce3+ doublet.
    """
    nm, intensity = _make_fluor_doublet_spectrum()

    with (
        patch(
            "sherloc_pipeline.web.routes.processing.fit_fluorescence_spectrum",
            wraps=__import__(
                "sherloc_pipeline.core.fluor_fitting", fromlist=["fit_fluorescence_spectrum"]
            ).fit_fluorescence_spectrum,
        ) as fluor_spy,
        patch(
            "sherloc_pipeline.web.routes.processing.fit_spectrum",
            side_effect=AssertionError(
                "fit_spectrum (Raman) must not be invoked for domain=fluorescence"
            ),
        ) as raman_spy,
    ):
        resp = await client.post(
            "/api/process/fit",
            json={
                "wavenumber": nm,
                "intensity": intensity,
                "params": {
                    "fitting": {
                        "domain": "fluorescence",
                        "wavenumber_range": [276.0, 355.0],
                        "fwhm_bounds": [10.0, 40.0],
                        "max_peaks": 4,
                        "min_snr": 5.0,
                        "model_selection": "aicc",
                    },
                },
            },
        )

    assert resp.status_code == 200, resp.text
    assert fluor_spy.call_count == 1, "fit_fluorescence_spectrum was not called"
    assert raman_spy.call_count == 0, "Raman fit_spectrum was incorrectly called"

    data = resp.json()
    assert data["model_selection_method"] == "aicc"
    # All returned peaks must be tagged as fluorescence
    for peak in data["peaks"]:
        assert peak["fit_modality"] == "fluorescence"


# ---------------------------------------------------------------------------
# Quality classifier wiring (v4.1.12 launch blocker)
#
# These exercise the integration between /api/process/fit and
# services.quality.classify_fit_quality. Per-rule semantics live in
# tests/unit/services/test_quality_classifier.py; here we lock in that the
# route actually populates `quality` and that the two operator-supplied
# reproducer scenarios never render as "pass".
# ---------------------------------------------------------------------------


def _flat_baseline_spectrum(value: float = 1000.0, length: int = 200):
    """Featureless spectrum — Raman fitter is highly likely to produce a
    poor fit (negative or near-zero R²) when handed something with no real
    peak structure. Used as a tractable stand-in for the
    minerals_pass_negative_r2.png scenario.
    """
    wn = np.linspace(700, 1200, length).tolist()
    rng = np.random.RandomState(0)
    intensity = (np.full(length, value) + rng.normal(0, 5, length)).tolist()
    return wn, intensity


@pytest.mark.asyncio
async def test_fit_populates_quality_field(client):
    """Every peak in /api/process/fit responses carries a quality string."""
    wn, intensity = _make_test_spectrum()
    resp = await client.post(
        "/api/process/fit",
        json={"wavenumber": wn, "intensity": intensity},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["peaks"], "test spectrum should yield at least one peak"
    for peak in data["peaks"]:
        assert peak["quality"] in {"pass", "review", "fail"}, peak


@pytest.mark.asyncio
async def test_fit_calibration_target_caps_quality_at_review(client):
    """Reproducer scenario A: external_calibration scan (target_type=cal_target).
    Even a high-R² fit on a real Gaussian peak must NOT render as "pass" because
    a calibration target has no Mars-target ground truth to validate against.
    """
    wn, intensity = _make_test_spectrum()
    resp = await client.post(
        "/api/process/fit",
        json={"wavenumber": wn, "intensity": intensity, "target_type": "cal_target"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["peaks"]
    for peak in data["peaks"]:
        assert peak["quality"] != "pass", (
            "calibration scan must never auto-pass; got "
            f"{peak['quality']} for peak {peak}"
        )


@pytest.mark.asyncio
async def test_fit_negative_r_squared_yields_fail(client):
    """Reproducer scenario B (minerals_pass_negative_r2.png): when the
    fitter produces a poor fit (R² < 0), the route must surface "fail" — the
    legacy code surfaced "pass" via pass_sharpness only.

    Deterministically pinned by mocking the Raman fitter so the test stays
    green even if a future fitter change happens to dampen the anti-
    correlated convergence that produced the operator's screenshot. We
    don't want a CI silence-on-skip path on the launch-blocker regression.
    """
    from sherloc_pipeline.models.fitting import FitResult, PeakFit

    wn, intensity = _make_test_spectrum()
    forced_peak = PeakFit(
        m_cm1=745.1,
        a=518.3,
        fwhm=18.6,  # in minerals bounds, so FWHM gate doesn't mask the R² gate
        sigma=18.6 / 2.355,
        area=518.3 * 18.6 * 1.0645,
        snr=15.7,
        pass_snr=True,
        pass_fwhm=True,
        pass_r2=False,
        sharpness_ratio=1.0,
        pass_sharpness=True,
    )
    fake_result = FitResult(peaks=[forced_peak], r2=-4.487, rss=0.0, dof=1, warnings=[])
    fake_model = np.zeros_like(np.array(wn, dtype=float))

    with patch(
        "sherloc_pipeline.web.routes.processing.fit_spectrum",
        return_value=(fake_result, fake_model),
    ):
        resp = await client.post(
            "/api/process/fit",
            json={
                "wavenumber": wn,
                "intensity": intensity,
                "target_type": "mars_target",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["r_squared"] == pytest.approx(-4.487)
    assert data["peaks"], "mocked fitter returned one peak; route should surface it"
    for peak in data["peaks"]:
        assert peak["quality"] == "fail", (
            f"R²={data['r_squared']} must yield 'fail', got {peak['quality']}"
        )


# ---------------------------------------------------------------------------
# Despike endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_despike_basic(client):
    """Despike removes obvious spikes and returns correct structure."""
    wn, intensity, spike_indices = _make_spiked_spectrum()
    resp = await client.post(
        "/api/process/despike",
        json={"wavenumber": wn, "intensity": intensity},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert len(data["despiked"]) == len(wn)
    assert len(data["spike_mask"]) == len(wn)
    assert data["n_spikes"] >= 1  # at least some spikes detected
    # The spike locations should be flagged
    for idx in spike_indices:
        assert data["spike_mask"][idx] is True, f"Spike at index {idx} not detected"
    # Despiked values at spike locations should be lower than original
    for idx in spike_indices:
        assert data["despiked"][idx] < intensity[idx]


@pytest.mark.asyncio
async def test_despike_custom_params(client):
    """Despike accepts and reflects custom parameters."""
    wn, intensity, _ = _make_spiked_spectrum()
    resp = await client.post(
        "/api/process/despike",
        json={
            "wavenumber": wn,
            "intensity": intensity,
            "params": {
                "window_size": 9,
                "zscore_threshold": 4.0,
                "max_iterations": 2,
                "sulfate_guard": False,
            },
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["params_used"]["window_size"] == 9
    assert data["params_used"]["zscore_threshold"] == 4.0
    assert data["params_used"]["max_iterations"] == 2
    assert data["params_used"]["sulfate_guard"] is False


@pytest.mark.asyncio
async def test_despike_matches_direct_call(client):
    """Endpoint output matches direct call to despike_r1_spectrum."""
    from sherloc_pipeline.core.preprocessing import DespikeParams, despike_r1_spectrum

    wn, intensity, _ = _make_spiked_spectrum()

    # Direct call
    x = np.array(wn, dtype=np.float64)
    y = np.array(intensity, dtype=np.float64)
    series = pd.Series(y, index=x)
    params = DespikeParams(window_size=7, zscore_threshold=6.0, max_iterations=1)
    direct_despiked, direct_mask = despike_r1_spectrum(series, params, raman_shift=x)

    # API call
    resp = await client.post(
        "/api/process/despike",
        json={"wavenumber": wn, "intensity": intensity},
    )
    assert resp.status_code == 200
    data = resp.json()

    np.testing.assert_allclose(
        data["despiked"], direct_despiked.values.tolist(), rtol=1e-10
    )
    assert data["spike_mask"] == direct_mask.values.astype(bool).tolist()


@pytest.mark.asyncio
async def test_despike_length_mismatch(client):
    resp = await client.post(
        "/api/process/despike",
        json={"wavenumber": [1, 2, 3], "intensity": [1, 2]},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_despike_too_few_points(client):
    resp = await client.post(
        "/api/process/despike",
        json={"wavenumber": [1], "intensity": [1]},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_despike_even_window_rejected(client):
    """Even window_size should be rejected by schema validation."""
    wn, intensity, _ = _make_spiked_spectrum()
    resp = await client.post(
        "/api/process/despike",
        json={
            "wavenumber": wn,
            "intensity": intensity,
            "params": {"window_size": 8},
        },
    )
    assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Background subtraction endpoint tests
# ---------------------------------------------------------------------------


def _make_fake_background_df():
    """Create a fake background DataFrame for mocking."""
    wn = np.linspace(600, 1300, 500)
    intensity = 100.0 + 0.05 * (wn - 600)  # gentle slope
    return pd.DataFrame({"raman_shift": wn, "intensity": intensity})


@pytest.mark.asyncio
async def test_background_as(client):
    """Background subtraction with 'as' type returns correct structure."""
    wn = np.linspace(700, 1200, 100).tolist()
    intensity = (np.full(100, 500.0) + np.random.RandomState(42).normal(0, 5, 100)).tolist()

    fake_bg = _make_fake_background_df()
    with patch(
        "sherloc_pipeline.web.routes.processing._load_background_standalone",
        return_value=fake_bg,
    ):
        resp = await client.post(
            "/api/process/background",
            json={
                "wavenumber": wn,
                "intensity": intensity,
                "bg_type": "as",
                "scale": 1.0,
                "scan_ppp": 900,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == "1.0.0"
    assert len(data["subtracted"]) == len(wn)
    assert len(data["background_scaled"]) == len(wn)
    assert data["scale_used"] == 1.0
    assert data["bg_type"] == "as"


@pytest.mark.asyncio
async def test_background_fs(client):
    """Background subtraction with 'fs' type works."""
    wn = np.linspace(700, 1200, 100).tolist()
    intensity = np.full(100, 500.0).tolist()

    fake_bg = _make_fake_background_df()
    with patch(
        "sherloc_pipeline.web.routes.processing._load_background_standalone",
        return_value=fake_bg,
    ):
        resp = await client.post(
            "/api/process/background",
            json={
                "wavenumber": wn,
                "intensity": intensity,
                "bg_type": "fs",
                "scale": 0.5,
                "scan_ppp": 450,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bg_type"] == "fs"
    assert data["scale_used"] == 0.5
    # Subtracted should be intensity minus scaled background
    bg_scaled = np.array(data["background_scaled"])
    expected_subtracted = np.array(intensity) - bg_scaled
    np.testing.assert_allclose(data["subtracted"], expected_subtracted.tolist(), rtol=1e-10)


@pytest.mark.asyncio
async def test_background_auto_scale(client):
    """Auto scale computes scan_ppp / 900."""
    wn = np.linspace(700, 1200, 50).tolist()
    intensity = np.full(50, 300.0).tolist()

    fake_bg = _make_fake_background_df()
    with patch(
        "sherloc_pipeline.web.routes.processing._load_background_standalone",
        return_value=fake_bg,
    ):
        resp = await client.post(
            "/api/process/background",
            json={
                "wavenumber": wn,
                "intensity": intensity,
                "bg_type": "as",
                "scale": "auto",
                "scan_ppp": 450,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scale_used"] == pytest.approx(450 / 900.0)


@pytest.mark.asyncio
async def test_background_length_mismatch(client):
    resp = await client.post(
        "/api/process/background",
        json={
            "wavenumber": [1, 2, 3],
            "intensity": [1, 2],
            "bg_type": "as",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_background_invalid_bg_type(client):
    """Invalid bg_type should be rejected by schema validation."""
    resp = await client.post(
        "/api/process/background",
        json={
            "wavenumber": [1, 2, 3],
            "intensity": [1, 2, 3],
            "bg_type": "invalid",
        },
    )
    assert resp.status_code == 422  # Pydantic rejects invalid Literal


@pytest.mark.asyncio
async def test_background_file_not_found(client):
    """Missing background file returns 404."""
    wn = np.linspace(700, 1200, 50).tolist()
    intensity = np.full(50, 300.0).tolist()

    with patch(
        "sherloc_pipeline.web.routes.processing._load_background_standalone",
        side_effect=FileNotFoundError("Background file not found"),
    ):
        resp = await client.post(
            "/api/process/background",
            json={
                "wavenumber": wn,
                "intensity": intensity,
                "bg_type": "as",
                "scale": 1.0,
            },
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Direct tests for the resolver (`_load_background_standalone`). These cover
# the post-issue-#13 contract: package-data is primary, legacy FS is
# fallback, traversal-shaped filenames are rejected.
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _BgConfig:
    """Minimal config stub for `_load_background_standalone`."""
    preprocessing: Dict[str, Any] = field(default_factory=dict)


def test_load_background_standalone_uses_package_data():
    """Primary resolution path: package-data ships both default CSVs."""
    from sherloc_pipeline.web.routes.processing import _load_background_standalone

    df_as = _load_background_standalone(_BgConfig(), "as")
    assert list(df_as.columns) == ["raman_shift", "intensity"]
    assert len(df_as) > 0
    # arm-stowed reference covers the R1 wavenumber window (~600-1400 cm-1)
    assert df_as["raman_shift"].min() < 800
    assert df_as["raman_shift"].max() > 1200

    df_fs = _load_background_standalone(_BgConfig(), "fs")
    assert list(df_fs.columns) == ["raman_shift", "intensity"]
    assert len(df_fs) > 0


def test_load_background_standalone_falls_back_to_fs(tmp_path, monkeypatch):
    """When package-data lookup misses, the legacy FS search succeeds."""
    from sherloc_pipeline.web.routes.processing import _load_background_standalone

    # Build a fake background CSV at the legacy FS location and point the
    # config at it via a config-supplied filename that does NOT match any
    # shipped package-data file.
    bg_dir = tmp_path / "data" / "background"
    bg_dir.mkdir(parents=True)
    custom_csv = bg_dir / "custom_background.csv"
    custom_csv.write_text("raman_shift,intensity\n800,10\n1000,20\n1200,30\n")

    monkeypatch.chdir(tmp_path)

    cfg = _BgConfig(preprocessing={
        "background_subtraction": {
            "backgrounds": {
                "as": {"file": "custom_background.csv"},
            }
        }
    })

    df = _load_background_standalone(cfg, "as")
    assert list(df.columns) == ["raman_shift", "intensity"]
    assert len(df) == 3
    assert df["raman_shift"].iloc[0] == 800


def test_load_background_standalone_rejects_traversal_filename():
    """Filename with `..` or absolute path is rejected before any lookup."""
    from sherloc_pipeline.web.routes.processing import _load_background_standalone

    cfg = _BgConfig(preprocessing={
        "background_subtraction": {
            "backgrounds": {
                "as": {"file": "../../../etc/passwd"},
            }
        }
    })

    with pytest.raises(FileNotFoundError, match="rejected"):
        _load_background_standalone(cfg, "as")


def test_load_background_standalone_missing_filename_raises(tmp_path, monkeypatch):
    """Filename that exists in neither package-data nor FS raises FileNotFoundError."""
    from sherloc_pipeline.web.routes.processing import _load_background_standalone

    monkeypatch.chdir(tmp_path)  # empty cwd, no ./data/background/
    cfg = _BgConfig(preprocessing={
        "background_subtraction": {
            "backgrounds": {
                "as": {"file": "nonexistent_baseline.csv"},
            }
        }
    })

    with pytest.raises(FileNotFoundError, match="not found"):
        _load_background_standalone(cfg, "as")


@pytest.mark.asyncio
async def test_background_endpoint_e2e_with_package_data(client):
    """End-to-end: POST /api/process/background succeeds without mocking
    `_load_background_standalone` — exercises the real package-data
    resolution path."""
    wn = np.linspace(700, 1200, 50).tolist()
    intensity = np.full(50, 500.0).tolist()

    resp = await client.post(
        "/api/process/background",
        json={
            "wavenumber": wn,
            "intensity": intensity,
            "bg_type": "as",
            "scale": 1.0,
            "scan_ppp": 900,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["bg_type"] == "as"
    assert len(data["subtracted"]) == len(wn)
    assert len(data["background_scaled"]) == len(wn)


# ---------------------------------------------------------------------------
# Packaging contract tests for the wheel-installed artifact.
#
# Issue #13's underlying failure was a packaging-layer regression: the
# Workbench resolver only checked local FS, and the wheel ALSO shipped
# nothing under `data/background/`, so production containers had no path
# that resolved. The tests above exercise the runtime resolver against
# the source tree (editable install). These tests add the missing layer:
# the package-data contract in pyproject.toml + the built-wheel zip.
# Codex `/code-review` PR #25 R1 F1 surfaced the gap.
# ---------------------------------------------------------------------------


def test_pyproject_package_data_includes_all_background_csvs():
    """Setuptools-contract gate: every on-disk CSV in
    `src/sherloc_pipeline/data/background/` MUST be matched by a glob in
    ``[tool.setuptools.package-data].sherloc_pipeline`` in pyproject.toml.

    setuptools expands these globs at wheel-build time; glob coverage on
    the source tree is equivalent to wheel-content coverage. Catches the
    "added a new CSV but forgot to register it" regression class
    (issue #13's root failure mode) BEFORE any wheel is built.
    """
    import fnmatch
    import tomllib
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text())
    pkg_data = pyproject["tool"]["setuptools"]["package-data"]["sherloc_pipeline"]

    bg_dir = repo / "src" / "sherloc_pipeline" / "data" / "background"
    on_disk = sorted(bg_dir.glob("*.csv"))
    assert len(on_disk) >= 2, (
        f"Expected ≥2 background CSVs at {bg_dir}, got {[p.name for p in on_disk]}. "
        f"Issue #13 ships arm-stowed + fused-silica references."
    )

    src_root = repo / "src" / "sherloc_pipeline"
    for csv in on_disk:
        rel = csv.relative_to(src_root).as_posix()
        matched = any(fnmatch.fnmatch(rel, glob) for glob in pkg_data)
        assert matched, (
            f"On-disk CSV {rel!r} is NOT matched by any "
            f"[tool.setuptools.package-data].sherloc_pipeline glob "
            f"{pkg_data!r}; the built wheel will silently omit it. "
            f"Issue #13 was exactly this failure mode."
        )


def test_built_wheel_ships_background_csvs(tmp_path):
    """End-to-end production-artifact gate: build the wheel, inspect its
    contents, assert background CSVs ship inside it. Catches anything
    the cheap glob-contract test might miss (e.g., setuptools quirks,
    pyproject parse-vs-build divergence, future build-backend swap).

    Skipped if the test env lacks pip + setuptools (so editor-mode local
    runs against a slim venv don't fail; CI's `pip install -e .[dev]`
    brings both transparently).
    """
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]

    # Skip cleanly when build tooling isn't available in the test env.
    for mod in ("pip", "setuptools"):
        probe = subprocess.run(
            [sys.executable, "-c", f"import {mod}"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            pytest.skip(f"{mod} not available in test env; cannot build wheel")

    out = tmp_path / "wheel-out"
    out.mkdir()
    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps", "--no-build-isolation",
            "--wheel-dir", str(out),
            str(repo),
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"`pip wheel` failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    wheels = list(out.glob("sherloc_pipeline-*.whl"))
    assert len(wheels) == 1, f"Expected 1 wheel, got: {[w.name for w in wheels]}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())

    expected = {
        "sherloc_pipeline/data/background/Arm_Stowed_post-anomaly_900ppp_trimmed_mean_1266.csv",
        "sherloc_pipeline/data/background/Fused_Silica_Corning7980_Air_Subtracted-Bandwidth-35_SB-Pitt.csv",
    }
    missing = expected - names
    in_wheel_bg = sorted(n for n in names if "data/background" in n)
    assert not missing, (
        f"Built wheel {wheels[0].name} is MISSING expected package-data files: "
        f"{sorted(missing)}. Wheel contents under data/background/: "
        f"{in_wheel_bg}. This is the production-artifact regression "
        f"issue #13 was originally about."
    )
