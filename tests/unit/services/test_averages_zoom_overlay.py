"""Companion zoomed mineral-region fit overlays for average spectra (#30).

`fit_averages` emits the full-range (500-4000 cm^-1) overlay plus a zoomed
companion over the configured `r1_fit_range` for every average kind, with the
range tokens in the filename derived from the config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.models.fitting import FitResult, PeakFit
from sherloc_pipeline.services.fitting import FittingService
from sherloc_pipeline.visualization.fitting_plots import (
    format_range_token,
    plot_fit_overlay_zoomed,
    resolve_fit_zoom_range,
    zoomed_overlay_path,
)

SOL = "0921"
TARGET = "Amherst_Point"
SCAN = "detail_1"
REGION = "R1"

# The two average kinds `fit_averages` discovers (mean + trimmed mean); the
# trim label is read back off the input filename, hence `4p`.
AVERAGE_STEMS = (
    f"{SOL}_{TARGET}_{SCAN}_{REGION}_avg-mean_bkgsub",
    f"{SOL}_{TARGET}_{SCAN}_{REGION}_avg-4p_trim_mean_bkgsub",
)
AVERAGE_INPUTS = (
    f"{SOL}_{TARGET}_{SCAN}_{REGION}_raw-n_mean_bkgsub_baselined.csv",
    f"{SOL}_{TARGET}_{SCAN}_{REGION}_raw-n_4p_trim_mean_bkgsub_baselined.csv",
)


def _gaussian(x: np.ndarray, center: float, amp: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _synthetic_average() -> pd.DataFrame:
    """A baseline-subtracted average spanning the full 500-4000 cm^-1 plot range.

    Peaks sit in each segment `fit_averages` fits (laser / minerals / G-band /
    hydration) so the fit path is exercised realistically.
    """
    x = np.arange(500.0, 4000.5, 2.0)
    rng = np.random.RandomState(7)
    y = rng.normal(0.0, 4.0, x.size)
    for center, amp, fwhm in (
        (642.0, 160.0, 32.0),
        (850.0, 60.0, 40.0),
        (1090.0, 80.0, 45.0),
        (1590.0, 40.0, 90.0),
        (3200.0, 35.0, 120.0),
    ):
        y += _gaussian(x, center, amp, fwhm)
    return pd.DataFrame({"raman_shift": x, "intensity": y})


@pytest.fixture
def averages_tree(tmp_path: Path) -> Path:
    """A results tree holding the average CSVs `fit_averages` consumes."""
    base = tmp_path / "results" / TARGET / f"{SOL}_{SCAN}"
    base.mkdir(parents=True, exist_ok=True)
    spectrum = _synthetic_average()
    for name in AVERAGE_INPUTS:
        spectrum.to_csv(base / name, index=False)
    return base


def _fit_averages(context, fixtures_path: Path, tmp_path: Path):
    return FittingService(context=context).fit_averages(
        sol=SOL,
        target=TARGET,
        scan=SCAN,
        data_dir=fixtures_path / "loupe",
        results_dir=tmp_path / "results",
        region=REGION,
    )


def test_both_overlays_emitted_per_average_kind(
    test_context, fixtures_path: Path, tmp_path: Path, averages_tree: Path
):
    """Default config → full-range plot + the historical `_700-1200` companion."""
    result = _fit_averages(test_context, fixtures_path, tmp_path)

    out_dir = averages_tree / "averages_fit"
    emitted = {Path(artifact).name for artifact in result.artifacts}
    for stem in AVERAGE_STEMS:
        full_range = out_dir / f"{stem}_fit.png"
        zoomed = out_dir / f"{stem}_fit_700-1200.png"
        assert full_range.exists(), f"missing full-range overlay: {full_range}"
        assert zoomed.exists(), f"missing zoomed overlay: {zoomed}"
        assert full_range.name in emitted
        assert zoomed.name in emitted


def test_zoom_range_follows_config(
    test_context, fixtures_path: Path, tmp_path: Path, averages_tree: Path
):
    """A non-default `r1_fit_range` self-describes in the companion filename."""
    test_context.config.fitting["r1_fit_range"] = [650, 1150]

    result = _fit_averages(test_context, fixtures_path, tmp_path)

    out_dir = averages_tree / "averages_fit"
    emitted = {Path(artifact).name for artifact in result.artifacts}
    for stem in AVERAGE_STEMS:
        assert (out_dir / f"{stem}_fit_650-1150.png").exists()
        assert f"{stem}_fit_650-1150.png" in emitted
        assert not (out_dir / f"{stem}_fit_700-1200.png").exists()


def test_resolve_fit_zoom_range():
    assert resolve_fit_zoom_range({"r1_fit_range": [700, 1200]}) == (700.0, 1200.0)
    assert resolve_fit_zoom_range({}) == (700.0, 1200.0)  # config default
    assert resolve_fit_zoom_range(None) == (700.0, 1200.0)
    # Malformed / degenerate ranges skip the companion rather than plotting an
    # empty axis.
    assert resolve_fit_zoom_range({"r1_fit_range": [1200, 700]}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": [700]}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": "700,1200"}) is None


def test_range_token_and_path_naming(tmp_path: Path):
    assert format_range_token((700.0, 1200.0)) == "700-1200"
    assert format_range_token((700.5, 1200.0)) == "700.5-1200"

    full_range = tmp_path / "1934_La_Suize_line_R1_avg-4p_trim_mean_bkgsub_fit.png"
    assert zoomed_overlay_path(full_range, (700.0, 1200.0)) == (
        tmp_path / "1934_La_Suize_line_R1_avg-4p_trim_mean_bkgsub_fit_700-1200.png"
    )


def test_zoomed_overlay_is_render_only(tmp_path: Path):
    """The companion re-renders the same peaks; only R^2 is rescoped to the window."""
    spectrum = _synthetic_average()
    x = spectrum["raman_shift"].to_numpy(float)
    y = spectrum["intensity"].to_numpy(float)
    peak = PeakFit(
        m_cm1=1090.0, a=80.0, fwhm=45.0,
        sigma=45.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        area=1.0, snr=10.0,
        pass_snr=True, pass_fwhm=True, pass_r2=True,
    )
    y_model = _gaussian(x, peak.m_cm1, peak.a, peak.fwhm)
    result = FitResult(peaks=[peak], r2=0.4, rss=1.0, dof=1, warnings=[])

    full_range = tmp_path / "sol_fit.png"
    written = plot_fit_overlay_zoomed(
        x, y, result, y_model, full_range, (700.0, 1200.0),
        title_stem="sol 0921 Amherst_Point detail_1 R1 avg mean",
    )

    assert written == tmp_path / "sol_fit_700-1200.png"
    assert written.exists()
    # Render-only: the fit result the caller passed in is untouched.
    assert result.r2 == 0.4
    assert result.peaks == [peak]
    # No full-range plot is written by the companion call.
    assert not full_range.exists()


def test_zoomed_overlay_skips_empty_window(tmp_path: Path):
    x = np.arange(700.0, 1200.0, 2.0)
    y = np.zeros_like(x)
    result = FitResult(peaks=[], r2=0.0, rss=0.0, dof=0, warnings=[])

    written = plot_fit_overlay_zoomed(
        x, y, result, y, tmp_path / "sol_fit.png", (2000.0, 2500.0)
    )

    assert written is None
    assert not (tmp_path / "sol_fit_2000-2500.png").exists()
