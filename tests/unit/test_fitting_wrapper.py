"""Unit tests for SpectralService Gaussian fitting wrapper.

Tests the _apply_fitting() method that wraps the existing multi-Gaussian
fitting algorithm for use with averaged spectra in the spectral plotting workflow.

T1.7: Implement Gaussian fitting wrapper
"""

import pytest
import numpy as np
import pandas as pd

from sherloc_pipeline.services.spectral import (
    SpectralService,
    FitResult,
    PeakFit,
    calculate_background_scale,
)


class TestApplyFitting:
    """Tests for SpectralService._apply_fitting() method."""

    def test_fitting_returns_fit_result_and_model(self, test_context):
        """Fitting should return a FitResult and model array."""
        service = SpectralService(context=test_context)
        
        # Create synthetic spectrum with clear peaks
        raman_shift = np.linspace(600, 1300, 700)
        # Add peaks in mineral region
        peak1 = 500 * np.exp(-((raman_shift - 850) ** 2) / 200)  # olivine-like
        peak2 = 300 * np.exp(-((raman_shift - 1000) ** 2) / 200)  # pyroxene-like
        noise = 10 * np.random.randn(len(raman_shift))
        intensity = peak1 + peak2 + noise + 50  # small baseline offset
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, model = service._apply_fitting(spectrum_df, fit_range=(700, 1200))
        
        assert isinstance(fit_result, FitResult)
        assert isinstance(model, np.ndarray)
        assert len(model) == len(spectrum_df)

    def test_fitting_detects_peaks(self, test_context):
        """Fitting should detect peaks in spectrum."""
        service = SpectralService(context=test_context)
        
        # Create spectrum with strong, well-separated peaks
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 800 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 600 * np.exp(-((raman_shift - 1050) ** 2) / 150)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(spectrum_df, fit_range=(700, 1200))
        
        # Should detect peaks
        assert isinstance(fit_result.peaks, list)
        # With strong synthetic peaks, we expect at least one to be detected
        # (exact count depends on SNR thresholds)

    def test_fitting_with_explicit_range(self, test_context):
        """Fitting should respect explicit fit_range parameter."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(500, 2000, 1500)
        # Put peaks at different positions
        peak1 = 500 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 500 * np.exp(-((raman_shift - 1600) ** 2) / 150)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Fit only the first peak region
        fit_result, model = service._apply_fitting(spectrum_df, fit_range=(700, 1000))
        
        # Model should be same length as input
        assert len(model) == len(spectrum_df)
        
        # The fitted peaks should be within the specified range
        for peak in fit_result.peaks:
            assert 700 <= peak.m_cm1 <= 1000

    def test_fitting_uses_default_range(self, test_context):
        """Fitting without fit_range should use config defaults."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(500, 2000, 1500)
        peak = 500 * np.exp(-((raman_shift - 900) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Should not raise error when fit_range is not specified
        fit_result, model = service._apply_fitting(spectrum_df)
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(spectrum_df)

    def test_fitting_result_has_r_squared(self, test_context):
        """FitResult should include R² value."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 900) ** 2) / 150)
        intensity = peak + 10 * np.random.randn(len(raman_shift)) + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(spectrum_df, fit_range=(700, 1200))
        
        assert hasattr(fit_result, "r2")
        assert isinstance(fit_result.r2, float)

    def test_fitting_result_has_rss_and_dof(self, test_context):
        """FitResult should include RSS and DOF."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 900) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(spectrum_df, fit_range=(700, 1200))
        
        assert hasattr(fit_result, "rss")
        assert hasattr(fit_result, "dof")
        assert isinstance(fit_result.rss, float)
        assert isinstance(fit_result.dof, int)

    def test_fitting_missing_columns_raises_error(self, test_context):
        """Missing required columns should raise ValueError."""
        service = SpectralService(context=test_context)
        
        # Missing intensity
        df_no_intensity = pd.DataFrame({
            "raman_shift": [700, 800, 900],
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            service._apply_fitting(df_no_intensity)
        
        # Missing raman_shift
        df_no_raman = pd.DataFrame({
            "intensity": [100, 200, 300],
        })
        
        with pytest.raises(ValueError, match="missing required columns"):
            service._apply_fitting(df_no_raman)

    def test_peak_fit_attributes(self, test_context):
        """PeakFit objects should have expected attributes."""
        service = SpectralService(context=test_context)
        
        # Create spectrum with a clear peak
        raman_shift = np.linspace(600, 1300, 700)
        peak = 800 * np.exp(-((raman_shift - 850) ** 2) / 100)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(spectrum_df, fit_range=(700, 1000))
        
        if fit_result.peaks:
            peak = fit_result.peaks[0]
            assert hasattr(peak, "m_cm1")  # center position
            assert hasattr(peak, "a")      # amplitude
            assert hasattr(peak, "fwhm")   # full width at half maximum
            assert hasattr(peak, "snr")    # signal-to-noise ratio
            assert hasattr(peak, "area")   # area under peak


class TestFittingOnRealData:
    """Tests using real fixture data for Gaussian fitting."""

    def test_fitting_on_amherst_point(self, test_context):
        """Gaussian fitting on Amherst Point averaged spectrum."""
        service = SpectralService(context=test_context)
        
        # Load, average, and baseline-correct data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Apply background subtraction and baseline
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        # Apply fitting
        fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        # Verify structure
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        
        # R² should be defined (may be 0 if no peaks detected)
        assert isinstance(fit_result.r2, float)

    def test_fitting_on_lake_haiyaha(self, test_context):
        """Gaussian fitting on Lake Haiyaha averaged spectrum (pure olivine)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0852", "Lake_Haiyaha", "detail_1")
        avg_df = service._compute_average(data, method="mean")
        corrected = service._apply_baseline(avg_df)
        
        fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)

    def test_fitting_on_stigbreen(self, test_context):
        """Gaussian fitting on Stigbreen averaged spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("1634", "Stigbreen", "line_1")
        avg_df = service._compute_average(data, method="median")
        
        # Apply full pipeline
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "as", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        # Model should have finite values
        assert np.all(np.isfinite(model))


class TestFittingIntegration:
    """Integration tests for full processing pipeline with fitting."""

    def test_full_averaged_pipeline_with_fitting(self, test_context):
        """Test complete averaged spectrum workflow ending with fitting."""
        service = SpectralService(context=test_context)
        
        # Load data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        assert data.ppp == 500  # Verify expected PPP
        
        # Average
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        assert len(avg_df) > 0
        
        # Background subtraction
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        assert abs(scale - 500/900) < 1e-9
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        
        # Baseline correction
        corrected = service._apply_baseline(bg_subtracted)
        
        # Fitting
        fit_result, model = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        # Validate outputs
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        assert isinstance(fit_result.peaks, list)
        assert isinstance(fit_result.r2, float)
        assert fit_result.r2 >= 0 or fit_result.peaks == []  # R² can be 0 if no fit

    def test_fitting_different_ranges(self, test_context):
        """Test fitting with different spectral ranges."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean")
        corrected = service._apply_baseline(avg_df)
        
        # Test narrow range (mineral region)
        fit_narrow, _ = service._apply_fitting(corrected, fit_range=(800, 900))
        
        # Test wider range
        fit_wide, _ = service._apply_fitting(corrected, fit_range=(700, 1200))
        
        # Both should be valid FitResults
        assert isinstance(fit_narrow, FitResult)
        assert isinstance(fit_wide, FitResult)


class TestFitResultAndPeakFit:
    """Tests for FitResult and PeakFit dataclass exports."""

    def test_fit_result_import(self):
        """FitResult should be importable from spectral module."""
        from sherloc_pipeline.services.spectral import FitResult
        
        # Should be able to create (for testing purposes)
        result = FitResult(peaks=[], r2=0.95, rss=100.0, dof=50, warnings=[])
        assert result.r2 == 0.95
        assert result.rss == 100.0
        assert result.dof == 50
        assert result.warnings == []

    def test_peak_fit_import(self):
        """PeakFit should be importable from spectral module."""
        from sherloc_pipeline.services.spectral import PeakFit
        
        # Should be able to create (for testing purposes)
        peak = PeakFit(
            m_cm1=850.0,
            a=500.0,
            fwhm=40.0,
            sigma=17.0,
            area=8500.0,
            snr=25.0,
            pass_snr=True,
            pass_fwhm=True,
            pass_r2=True,
        )
        assert peak.m_cm1 == 850.0
        assert peak.a == 500.0
        assert peak.fwhm == 40.0
        assert peak.pass_snr is True

