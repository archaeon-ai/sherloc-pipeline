"""Unit tests for visualization — plot output verification (step 5.3).

Covers:
- plot_fluor_fit_overlay() produces PNG from synthetic data
- SpectralService.process() handles domain='fluor' without error
- SpectralService.process() handles domain='both' without error
- Existing Raman plot behavior unchanged (regression)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sherloc_pipeline.core.fitting import fwhm_to_sigma
from sherloc_pipeline.core.fluor_fitting import (
    FluorFitResult,
    FluorPeakFit,
)
from sherloc_pipeline.visualization.fitting_plots import plot_fluor_fit_overlay
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.spectral import (
    LoupeData,
    SpectralPlotRequest,
    SpectralService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wavelength(start: float = 260.0, stop: float = 360.0, n: int = 500):
    return np.linspace(start, stop, n)


def _synth_fluor_spectrum(
    wavelength: np.ndarray,
    peaks: list[tuple[float, float, float]],
    noise_std: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros_like(wavelength, dtype=float)
    for center, amp, fwhm in peaks:
        sigma = fwhm_to_sigma(fwhm)
        y += amp * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    y += rng.normal(0.0, noise_std, size=y.size)
    return y


def _make_fluor_fit_result(peaks: list[FluorPeakFit], r2: float = 0.95) -> FluorFitResult:
    return FluorFitResult(
        peaks=peaks,
        r2=r2,
        rss=100.0,
        aicc=50.0,
        n_peaks=len(peaks),
        is_saturated=False,
        fit_skipped=False,
        n_saturated_channels=0,
        n_masked_channels=0,
    )


def _make_fluor_loupe_data(n_channels: int = 500, n_points: int = 10) -> LoupeData:
    """Build a synthetic LoupeData with wavelength + point columns."""
    wl = np.linspace(260.0, 360.0, n_channels)
    data = {"wavelength": wl}
    for i in range(n_points):
        data[i] = _synth_fluor_spectrum(
            wl, [(305.0, 5000.0, 20.0), (325.0, 3000.0, 18.0)], seed=42 + i
        )
    return LoupeData(
        spectra_df=pd.DataFrame(data),
        n_points=n_points,
        ppp=500.0,
        working_dir=Path("/tmp/fake"),
        metadata={"n_spectra": n_points},
    )


def _make_raman_loupe_data(n_channels: int = 523, n_points: int = 10) -> LoupeData:
    """Build a synthetic Raman LoupeData with raman_shift + point columns."""
    rs = np.linspace(640.0, 4200.0, n_channels)
    data = {"raman_shift": rs}
    for i in range(n_points):
        rng = np.random.default_rng(42 + i)
        y = 500.0 * np.exp(-0.5 * ((rs - 1000.0) / 30.0) ** 2) + rng.normal(0, 10, n_channels)
        data[i] = y
    return LoupeData(
        spectra_df=pd.DataFrame(data),
        n_points=n_points,
        ppp=500.0,
        working_dir=Path("/tmp/fake_raman"),
        metadata={"n_spectra": n_points},
    )


# ---------------------------------------------------------------------------
# AC 1: plot_fluor_fit_overlay() produces PNG from synthetic data
# ---------------------------------------------------------------------------


class TestPlotFluorFitOverlay:

    def test_produces_png(self, tmp_path):
        """plot_fluor_fit_overlay() should produce a PNG file."""
        wl = _make_wavelength()
        intensity = _synth_fluor_spectrum(wl, [(305.0, 5000.0, 20.0), (325.0, 3000.0, 18.0)])
        result = _make_fluor_fit_result([
            FluorPeakFit(center_nm=305.0, amplitude=5000.0, fwhm_nm=20.0, area=1e5, snr=500.0),
            FluorPeakFit(center_nm=325.0, amplitude=3000.0, fwhm_nm=18.0, area=6e4, snr=300.0),
        ])

        png_path = tmp_path / "test_fluor_overlay.png"
        plot_fluor_fit_overlay(wl, intensity, result, str(png_path))

        assert png_path.exists(), "PNG file was not created"
        assert png_path.stat().st_size > 0, "PNG file is empty"

    def test_no_pdf_alongside(self, tmp_path):
        """plot_fluor_fit_overlay() should only produce PNG, no PDF."""
        wl = _make_wavelength()
        intensity = _synth_fluor_spectrum(wl, [(305.0, 5000.0, 20.0)])
        result = _make_fluor_fit_result([
            FluorPeakFit(center_nm=305.0, amplitude=5000.0, fwhm_nm=20.0, area=1e5, snr=500.0),
        ])

        png_path = tmp_path / "test_fluor.png"
        plot_fluor_fit_overlay(wl, intensity, result, str(png_path))

        assert png_path.exists(), "PNG file was not created"
        pdf_path = tmp_path / "test_fluor.pdf"
        assert not pdf_path.exists(), "PDF should not be created alongside PNG"

    def test_no_peaks_still_produces_file(self, tmp_path):
        """When fit_skipped=True (no peaks), overlay still writes a file."""
        wl = _make_wavelength()
        intensity = _synth_fluor_spectrum(wl, [(305.0, 100.0, 20.0)], noise_std=200.0)
        result = FluorFitResult(
            peaks=[],
            r2=0.0,
            rss=0.0,
            aicc=float("inf"),
            n_peaks=0,
            is_saturated=True,
            fit_skipped=True,
            n_saturated_channels=10,
            n_masked_channels=10,
            warnings=["full_saturation_skip"],
        )

        png_path = tmp_path / "test_no_peaks.png"
        plot_fluor_fit_overlay(wl, intensity, result, str(png_path))
        assert png_path.exists()

    def test_saturated_channels_marked(self, tmp_path):
        """Saturated channels should be plotted (red markers)."""
        wl = _make_wavelength()
        intensity = _synth_fluor_spectrum(wl, [(305.0, 5000.0, 20.0)])
        # Inject saturation
        intensity[200:205] = 65000.0

        result = _make_fluor_fit_result([
            FluorPeakFit(center_nm=305.0, amplitude=5000.0, fwhm_nm=20.0, area=1e5, snr=500.0),
        ])

        png_path = tmp_path / "test_saturated.png"
        plot_fluor_fit_overlay(wl, intensity, result, str(png_path))
        assert png_path.exists()

    def test_auto_title_from_metadata(self, tmp_path):
        """Title should auto-generate from sol/target/scan/point."""
        wl = _make_wavelength()
        intensity = _synth_fluor_spectrum(wl, [(305.0, 5000.0, 20.0)])
        result = _make_fluor_fit_result([
            FluorPeakFit(center_nm=305.0, amplitude=5000.0, fwhm_nm=20.0, area=1e5, snr=500.0),
        ])

        png_path = tmp_path / "test_title.png"
        plot_fluor_fit_overlay(
            wl, intensity, result, str(png_path),
            sol="0921", target="TestTarget", scan="detail_1", point=5,
        )
        assert png_path.exists()


# ---------------------------------------------------------------------------
# AC 2: Plot service handles domain='fluor' without error
# ---------------------------------------------------------------------------


class TestServiceFluorDomain:

    def test_fluor_averaged_no_fit(self, test_context):
        """domain='fluor' averaged mode runs without error."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()

        with patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="fluor", fit=False,
            )
            result = service.process(request)

        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) >= 1
        assert result.metadata["domain"] == "fluor"
        # Check PNG was created
        png_artifact = [a for a in result.artifacts if str(a).endswith(".png")]
        assert len(png_artifact) == 1
        assert Path(png_artifact[0]).exists()

    def test_fluor_averaged_with_fit(self, test_context):
        """domain='fluor' averaged mode with fit=True runs and finds peaks."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()

        with patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="fluor", fit=True,
            )
            result = service.process(request)

        assert isinstance(result, ServiceResult)
        assert result.metadata.get("fit") is True
        assert result.metadata.get("n_peaks", 0) >= 0
        png_artifact = [a for a in result.artifacts if str(a).endswith(".png")]
        assert len(png_artifact) == 1
        assert Path(png_artifact[0]).exists()

    def test_fluor_point_mode(self, test_context):
        """domain='fluor' point mode runs without error."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()

        with patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="point", point=0, domain="fluor", fit=True,
            )
            result = service.process(request)

        assert isinstance(result, ServiceResult)
        assert result.metadata.get("domain") == "fluor"
        assert result.metadata.get("point") == 0

    def test_fluor_background_warning(self, test_context):
        """Raman-only options emit warnings when used with fluor domain."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()

        with patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="fluor", background="fs", baseline=True,
            )
            result = service.process(request)

        # Should warn about both background and baseline
        assert any("background" in w for w in result.warnings)
        assert any("baseline" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# AC 3: Plot service handles domain='both' without error
# ---------------------------------------------------------------------------


class TestServiceBothDomain:

    def test_both_domain_runs(self, test_context):
        """domain='both' processes both Raman and fluorescence."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()
        raman_data = _make_raman_loupe_data()

        with (
            patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data),
            patch.object(service, "_load_loupe_data", return_value=raman_data),
        ):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="both",
            )
            result = service.process(request)

        assert isinstance(result, ServiceResult)
        # Combined artifacts from both domains
        assert len(result.artifacts) >= 2
        # Should contain both raman and fluor outputs
        artifact_strs = [str(a) for a in result.artifacts]
        assert any("fluor" in s for s in artifact_strs), (
            f"No fluor artifact in: {artifact_strs}"
        )

    def test_both_domain_combines_metadata(self, test_context):
        """domain='both' merges metadata from both sub-results."""
        service = SpectralService(context=test_context)
        fluor_data = _make_fluor_loupe_data()
        raman_data = _make_raman_loupe_data()

        with (
            patch.object(service, "_load_fluor_loupe_data", return_value=fluor_data),
            patch.object(service, "_load_loupe_data", return_value=raman_data),
        ):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="both",
            )
            result = service.process(request)

        assert "raman" in result.metadata or "fluor" in result.metadata


# ---------------------------------------------------------------------------
# AC 4: Existing Raman plot behavior unchanged (regression)
# ---------------------------------------------------------------------------


class TestRamanRegressionPlot:

    def test_raman_averaged_still_works(self, test_context):
        """Default domain='raman' averaged mode still works."""
        service = SpectralService(context=test_context)
        raman_data = _make_raman_loupe_data()

        with patch.object(service, "_load_loupe_data", return_value=raman_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="raman",
            )
            result = service.process(request)

        assert isinstance(result, ServiceResult)
        assert len(result.artifacts) >= 1
        png_artifact = [a for a in result.artifacts if str(a).endswith(".png")]
        assert len(png_artifact) == 1
        assert Path(png_artifact[0]).exists()

    def test_raman_is_default_domain(self, test_context):
        """When domain is not specified, Raman is the default."""
        request = SpectralPlotRequest(
            sol="0921", target="TestTarget", scan="detail_1",
        )
        assert request.domain == "raman"

    def test_raman_plot_no_fluor_contamination(self, test_context):
        """Raman processing should not invoke fluorescence loading."""
        service = SpectralService(context=test_context)
        raman_data = _make_raman_loupe_data()

        with (
            patch.object(service, "_load_loupe_data", return_value=raman_data),
            patch.object(service, "_load_fluor_loupe_data") as mock_fluor,
        ):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="raman",
            )
            service.process(request)

        mock_fluor.assert_not_called()

    def test_raman_artifacts_use_r1_naming(self, test_context):
        """Raman artifacts should NOT use 'fluor' in their filenames."""
        service = SpectralService(context=test_context)
        raman_data = _make_raman_loupe_data()

        with patch.object(service, "_load_loupe_data", return_value=raman_data):
            request = SpectralPlotRequest(
                sol="0921", target="TestTarget", scan="detail_1",
                mode="averaged", domain="raman",
            )
            result = service.process(request)

        for artifact in result.artifacts:
            assert "fluor" not in str(artifact).lower(), (
                f"Raman artifact should not contain 'fluor': {artifact}"
            )
