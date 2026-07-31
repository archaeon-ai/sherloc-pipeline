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
    test_context, fixtures_path: Path, tmp_path: Path, averages_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A non-default `r1_fit_range` self-describes in the companion filename."""
    # setitem, not assignment: RuntimeContext.bootstrap() shares one config
    # object process-wide, so a bare mutation would leak into later tests.
    monkeypatch.setitem(test_context.config.fitting, "r1_fit_range", [650, 1150])

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
    assert resolve_fit_zoom_range({"r1_fit_range": [700, 1000, 1200]}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": "700,1200"}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": None}) is None
    # Non-finite bounds order fine but matplotlib rejects them as axis limits.
    assert resolve_fit_zoom_range({"r1_fit_range": [700, float("inf")]}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": [float("-inf"), 1200]}) is None
    assert resolve_fit_zoom_range({"r1_fit_range": [float("nan"), 1200]}) is None


def test_malformed_range_skips_companion_without_failing(
    test_context, fixtures_path: Path, tmp_path: Path, averages_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A bad configured range must not fail the run after artifacts are written."""
    monkeypatch.setitem(
        test_context.config.fitting, "r1_fit_range", [700, float("inf")]
    )

    result = _fit_averages(test_context, fixtures_path, tmp_path)

    out_dir = averages_tree / "averages_fit"
    for stem in AVERAGE_STEMS:
        assert (out_dir / f"{stem}_fit.png").exists()  # full-range still emitted
    assert not list(out_dir.glob("*_fit_*.png"))
    assert result.metadata["averages_fitted"] == 2


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


# --- averaged-mode spectral plot path (the second emission site) ------------ #


def _averaged_request(**kw):
    from sherloc_pipeline.services.spectral import SpectralPlotRequest

    params = dict(
        sol=SOL,
        target=TARGET,
        scan=SCAN,
        mode="averaged",
        avg_method="mean",
        background=None,
        baseline=True,
        fit=True,
        export="both",
    )
    params.update(kw)
    return SpectralPlotRequest(**params)


def _run_averaged(test_context, **kw):
    from sherloc_pipeline.services.spectral import SpectralService

    return SpectralService(context=test_context).process(_averaged_request(**kw))


def test_averaged_mode_fit_emits_zoomed_companion(test_context):
    """An averaged-mode fit plot ships with the mineral-region companion."""
    result = _run_averaged(test_context)

    pngs = [p for p in result.artifacts if p.suffix == ".png"]
    assert len(pngs) == 2
    full_range = next(p for p in pngs if not p.stem.endswith("_700-1200"))
    zoomed = next(p for p in pngs if p.stem.endswith("_700-1200"))
    assert zoomed == full_range.with_name(f"{full_range.stem}_700-1200.png")
    assert zoomed.exists() and full_range.exists()


def test_averaged_mode_zoom_follows_config(
    test_context, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(test_context.config.fitting, "r1_fit_range", [650, 1150])

    result = _run_averaged(test_context)

    pngs = {p.name for p in result.artifacts if p.suffix == ".png"}
    assert any(name.endswith("_650-1150.png") for name in pngs)
    assert not any(name.endswith("_700-1200.png") for name in pngs)


def test_averaged_mode_companion_r2_describes_its_own_window(test_context):
    """A fit range narrower than the companion window must not annotate it."""
    import json

    from sherloc_pipeline.visualization import fitting_plots

    captured: list[tuple] = []
    original = fitting_plots.plot_fit_overlay  # unused here; guard against drift

    from sherloc_pipeline.services import spectral as spectral_mod

    real_generate = spectral_mod.SpectralService._generate_plot

    def spy(self, spectrum_df, request, fit_result=None, model_array=None):
        if fit_result is not None:
            captured.append(
                (request.xlim, float(fit_result.r2), spectrum_df, list(fit_result.peaks))
            )
        return real_generate(
            self, spectrum_df, request, fit_result=fit_result, model_array=model_array
        )

    spectral_mod.SpectralService._generate_plot = spy
    try:
        # A fit range whose LOWER edge sits on a peak: the drawn Gaussian
        # tail extends well below it, where the API's model array is zero.
        result = _run_averaged(test_context, fit_range=(1080.0, 1200.0))
    finally:
        spectral_mod.SpectralService._generate_plot = real_generate

    assert original is fitting_plots.plot_fit_overlay
    full_xlim, full_r2, _, _ = captured[0]
    zoom_xlim, zoom_r2, spectrum, peaks = captured[1]
    assert full_xlim is None and zoom_xlim == (700.0, 1200.0)
    # The companion's R² is computed over what it displays, not over the
    # narrower fit range the full-range render reported.
    assert zoom_r2 != pytest.approx(full_r2)

    # …and it measures the model that is DRAWN (Gaussians reconstructed from
    # the fitted peaks over the whole domain), not the fitting API's model
    # array, which is zero-padded outside the 1000-1200 fit ROI. Recomputed
    # here independently.
    x = spectrum["raman_shift"].to_numpy(float)
    y = spectrum["intensity"].to_numpy(float)
    drawn = np.zeros_like(x)
    for peak in peaks:
        drawn += _gaussian(x, peak.m_cm1, peak.a, peak.fwhm)
    mask = (x >= 700.0) & (x <= 1200.0)
    ss_res = float(np.sum((y[mask] - drawn[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
    assert zoom_r2 == pytest.approx(1.0 - ss_res / ss_tot, abs=1e-9)
    # The peak's tail reaches below the 1080 fit edge, so scoring against the
    # zero-padded array would have reported a materially worse value.
    padded = np.zeros_like(x)
    in_roi = (x >= 1080.0) & (x <= 1200.0)
    padded[in_roi] = drawn[in_roi]
    padded_r2 = 1.0 - float(np.sum((y[mask] - padded[mask]) ** 2)) / ss_tot
    assert zoom_r2 - padded_r2 > 0.01, (zoom_r2, padded_r2)

    # And the JSON sidecar inventories the companion + its window (#30 F4).
    meta_path = next(p for p in result.artifacts if p.suffix == ".json")
    outputs = json.loads(meta_path.read_text())
    zoomed_name = outputs["outputs"]["files"]["png_zoomed"]
    assert zoomed_name.endswith("_700-1200.png")
    assert outputs["plot"]["zoomed_xlim"] == [700.0, 1200.0]
    assert (meta_path.parent / zoomed_name).exists()


def test_averaged_mode_no_companion_without_a_fit(test_context):
    """A bare spectrum plot has no fit overlay to zoom."""
    result = _run_averaged(test_context, fit=False)

    pngs = [p for p in result.artifacts if p.suffix == ".png"]
    assert len(pngs) == 1


def test_averaged_mode_explicit_xlim_governs(test_context):
    """An explicit window is the caller's choice — no second fixed-window plot."""
    result = _run_averaged(test_context, xlim=(900.0, 1400.0))

    pngs = [p for p in result.artifacts if p.suffix == ".png"]
    assert len(pngs) == 1
    assert not pngs[0].stem.endswith("_700-1200")


def test_averaged_mode_csv_only_export_writes_no_companion(test_context):
    result = _run_averaged(test_context, export="csv")

    assert not [p for p in result.artifacts if p.suffix == ".png"]


def test_zoomed_overlay_scores_the_drawn_model(tmp_path: Path):
    """A broad out-of-window peak is drawn inside the zoom, so it must be scored.

    `fit_averages` assembles `y_model_total` from per-segment fits, so it is
    zero outside each segment's ROI — while the renderer recomputes every peak's
    Gaussian across the window. The companion must not score a visible tail as
    nothing.
    """
    from sherloc_pipeline.core.fitting import compute_r2
    from sherloc_pipeline.visualization import fitting_plots

    x = np.arange(500.0, 4000.5, 2.0)
    hydration = PeakFit(  # wide band fitted in 3000-4000, tail reaching the zoom
        m_cm1=1400.0, a=50.0, fwhm=1200.0,
        sigma=1200.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))),
        area=1.0, snr=10.0, pass_snr=True, pass_fwhm=True, pass_r2=True,
    )
    drawn = _gaussian(x, hydration.m_cm1, hydration.a, hydration.fwhm)
    y = drawn + 5.0
    # The caller's array is zero below 1300 — the segment ROI boundary.
    segmented = np.where(x >= 1300.0, drawn, 0.0)
    result = FitResult(peaks=[hydration], r2=0.9, rss=1.0, dof=1, warnings=[])

    captured: dict = {}
    real = fitting_plots.plot_fit_overlay

    def spy(x_cm1, y_arr, mask, res, y_model_full, out, **kw):
        captured["r2"] = float(res.r2)
        return real(x_cm1, y_arr, mask, res, y_model_full, out, **kw)

    fitting_plots.plot_fit_overlay = spy
    try:
        fitting_plots.plot_fit_overlay_zoomed(
            x, y, result, segmented, tmp_path / "a_fit.png", (700.0, 1200.0)
        )
    finally:
        fitting_plots.plot_fit_overlay = real

    mask = (x >= 700.0) & (x <= 1200.0)
    assert captured["r2"] == pytest.approx(
        float(compute_r2(y[mask], drawn[mask])), abs=1e-9
    )
    # Scoring the zero-padded array instead would have been materially worse.
    assert captured["r2"] - float(compute_r2(y[mask], segmented[mask])) > 0.1


def test_zoomed_overlay_skips_empty_window(tmp_path: Path):
    x = np.arange(700.0, 1200.0, 2.0)
    y = np.zeros_like(x)
    result = FitResult(peaks=[], r2=0.0, rss=0.0, dof=0, warnings=[])

    written = plot_fit_overlay_zoomed(
        x, y, result, y, tmp_path / "sol_fit.png", (2000.0, 2500.0)
    )

    assert written is None
    assert not (tmp_path / "sol_fit_2000-2500.png").exists()
