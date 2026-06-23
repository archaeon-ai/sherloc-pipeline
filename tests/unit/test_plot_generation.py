"""Unit tests for SpectralService plot generation.

Tests the _generate_plot() method that creates matplotlib figures for
spectral visualization with optional Gaussian fit overlay.

T1.8: Implement plot generation
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotRequest,
    FitResult,
    PeakFit,
    calculate_background_scale,
)


# Use non-interactive backend for tests
matplotlib.use('Agg')


class TestGeneratePlot:
    """Tests for SpectralService._generate_plot() method."""

    def test_generate_plot_returns_figure(self, test_context):
        """Plot generation should return a matplotlib Figure."""
        service = SpectralService(context=test_context)
        
        spectrum_df = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 1000),
            "intensity": np.random.randn(1000) * 10 + 100,
        })
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
        )
        
        fig = service._generate_plot(spectrum_df, request)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_generate_plot_without_fit(self, test_context):
        """Plot generation without fitting should show only spectrum."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(500, 4000, 1000)
        intensity = 500 * np.exp(-((raman_shift - 850) ** 2) / 5000) + 50
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            fit=False,
        )
        
        fig = service._generate_plot(spectrum_df, request)
        
        assert isinstance(fig, Figure)
        # Should have exactly one axes
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_generate_plot_with_fit_result(self, test_context):
        """Plot generation with fit should show spectrum and model overlay."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Create a mock fit result
        fit_result = FitResult(
            peaks=[
                PeakFit(
                    m_cm1=850.0, a=500.0, fwhm=40.0, sigma=17.0,
                    area=8500.0, snr=25.0,
                    pass_snr=True, pass_fwhm=True, pass_r2=True,
                )
            ],
            r2=0.95,
            rss=100.0,
            dof=50,
            warnings=[],
        )
        
        # Mock model array (same length as spectrum)
        model_array = peak
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            fit=True,
        )
        
        fig = service._generate_plot(spectrum_df, request, fit_result, model_array)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_generate_plot_respects_xlim(self, test_context):
        """Plot generation should respect xlim axis limits."""
        service = SpectralService(context=test_context)
        
        spectrum_df = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 1000),
            "intensity": np.random.randn(1000) * 10 + 100,
        })
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            xlim=(700, 1200),
        )
        
        fig = service._generate_plot(spectrum_df, request)
        ax = fig.axes[0]
        
        # Check that xlim was applied
        xlim = ax.get_xlim()
        assert xlim[0] == 700
        assert xlim[1] == 1200
        plt.close(fig)

    def test_generate_plot_respects_ylim(self, test_context):
        """Plot generation should respect ylim axis limits."""
        service = SpectralService(context=test_context)
        
        spectrum_df = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 1000),
            "intensity": np.random.randn(1000) * 10 + 100,
        })
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            ylim=(0, 500),
        )
        
        fig = service._generate_plot(spectrum_df, request)
        ax = fig.axes[0]
        
        # Check that ylim was applied
        ylim = ax.get_ylim()
        assert ylim[0] == 0
        assert ylim[1] == 500
        plt.close(fig)

    def test_generate_plot_missing_columns_raises_error(self, test_context):
        """Missing required columns should raise ValueError."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
        )
        
        # Missing intensity
        df_no_intensity = pd.DataFrame({
            "raman_shift": [700, 800, 900],
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            service._generate_plot(df_no_intensity, request)
        
        # Missing raman_shift
        df_no_raman = pd.DataFrame({
            "intensity": [100, 200, 300],
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            service._generate_plot(df_no_raman, request)


class TestBuildPlotTitle:
    """Tests for SpectralService._build_plot_title() method."""

    def test_basic_title_format(self, test_context):
        """Title should follow format: sol <sol> <target> <scan> R1 avg <method>."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="mean",
        )
        
        title = service._build_plot_title(request)
        
        assert "sol 0921" in title
        assert "Amherst_Point" in title
        assert "detail_1" in title
        assert "R1" in title
        assert "avg mean" in title

    def test_trim_mean_title_format(self, test_context):
        """Trim-mean should show percentage in title."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
        )
        
        title = service._build_plot_title(request)
        
        assert "2p_trim_mean" in title

    def test_title_with_background(self, test_context):
        """Title should include background type when applied."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
        )
        
        title = service._build_plot_title(request)
        
        assert "fs" in title

    def test_title_with_baseline(self, test_context):
        """Title should include 'baselined' when baseline correction applied."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
        )
        
        title = service._build_plot_title(request)
        
        assert "baselined" in title

    def test_title_with_fit(self, test_context):
        """Title should include 'fit' when fitting applied."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.9, rss=100.0, dof=50, warnings=[])
        
        title = service._build_plot_title(request, fit_result)
        
        assert "fit" in title

    def test_full_processing_title(self, test_context):
        """Title should show full processing chain."""
        service = SpectralService(context=test_context)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=True,
            fit=True,
        )
        
        fit_result = FitResult(peaks=[], r2=0.9, rss=100.0, dof=50, warnings=[])
        
        title = service._build_plot_title(request, fit_result)
        
        assert "sol 0921" in title
        assert "Amherst_Point" in title
        assert "2p_trim_mean" in title
        assert "fs" in title
        assert "baselined" in title
        assert "fit" in title


class TestPlotGenerationOnRealData:
    """Tests using real fixture data for plot generation."""

    def test_plot_amherst_point_averaged(self, test_context):
        """Generate plot from Amherst Point averaged spectrum."""
        service = SpectralService(context=test_context)
        
        # Load and process data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
        )
        
        fig = service._generate_plot(avg_df, request)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_with_background_subtraction(self, test_context):
        """Generate plot after background subtraction."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            background="fs",
        )
        
        fig = service._generate_plot(bg_subtracted, request)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_baseline_correction(self, test_context):
        """Generate plot after baseline correction."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        corrected = service._apply_baseline(avg_df)
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            baseline=True,
        )
        
        fig = service._generate_plot(corrected, request)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_full_pipeline_and_fitting(self, test_context):
        """Generate plot with full processing pipeline including fitting."""
        service = SpectralService(context=test_context)
        
        # Full pipeline
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        corrected = service._apply_baseline(bg_subtracted)
        fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            avg_method="trim-mean",
            trim_pct=2.0,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
        )
        
        fig = service._generate_plot(corrected, request, fit_result, model)
        
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_lake_haiyaha(self, test_context):
        """Generate plot from Lake Haiyaha (pure olivine)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0852", "Lake_Haiyaha", "detail_1")
        avg_df = service._compute_average(data, method="median")
        
        request = SpectralPlotRequest(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
            mode="averaged",
            avg_method="median",
        )
        
        fig = service._generate_plot(avg_df, request)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_stigbreen(self, test_context):
        """Generate plot from Stigbreen line scan."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("1634", "Stigbreen", "line_1")
        avg_df = service._compute_average(data, method="mean")
        
        request = SpectralPlotRequest(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
            mode="averaged",
            avg_method="mean",
        )
        
        fig = service._generate_plot(avg_df, request)
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPlotGenerationEdgeCases:
    """Edge case tests for plot generation."""

    def test_plot_with_custom_xlim_and_ylim(self, test_context):
        """Plot with both xlim and ylim specified."""
        service = SpectralService(context=test_context)
        
        spectrum_df = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 1000),
            "intensity": np.random.randn(1000) * 10 + 100,
        })
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            xlim=(700, 1300),
            ylim=(-50, 500),
        )
        
        fig = service._generate_plot(spectrum_df, request)
        ax = fig.axes[0]
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        assert xlim == (700, 1300)
        assert ylim == (-50, 500)
        plt.close(fig)

    def test_plot_with_multiple_peaks(self, test_context):
        """Plot with multiple fitted peaks in legend."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 500 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 400 * np.exp(-((raman_shift - 1000) ** 2) / 150)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Multiple peaks
        fit_result = FitResult(
            peaks=[
                PeakFit(m_cm1=850.0, a=500.0, fwhm=40.0, sigma=17.0,
                       area=8500.0, snr=25.0,
                       pass_snr=True, pass_fwhm=True, pass_r2=True),
                PeakFit(m_cm1=1000.0, a=400.0, fwhm=45.0, sigma=19.0,
                       area=7200.0, snr=20.0,
                       pass_snr=True, pass_fwhm=True, pass_r2=True),
            ],
            r2=0.95,
            rss=100.0,
            dof=50,
            warnings=[],
        )
        
        model_array = peak1 + peak2
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            fit=True,
        )
        
        fig = service._generate_plot(spectrum_df, request, fit_result, model_array)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_with_failing_peak(self, test_context):
        """Plot should show failing peaks in different style."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # One passing, one failing peak
        fit_result = FitResult(
            peaks=[
                PeakFit(m_cm1=850.0, a=500.0, fwhm=40.0, sigma=17.0,
                       area=8500.0, snr=25.0,
                       pass_snr=True, pass_fwhm=True, pass_r2=True),
                PeakFit(m_cm1=950.0, a=50.0, fwhm=20.0, sigma=8.5,
                       area=425.0, snr=2.0,
                       pass_snr=False, pass_fwhm=False, pass_r2=True),  # failing
            ],
            r2=0.9,
            rss=200.0,
            dof=45,
            warnings=[],
        )
        
        model_array = peak
        
        request = SpectralPlotRequest(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            mode="averaged",
            fit=True,
        )
        
        # Should not raise error
        fig = service._generate_plot(spectrum_df, request, fit_result, model_array)
        
        assert isinstance(fig, Figure)
        plt.close(fig)

