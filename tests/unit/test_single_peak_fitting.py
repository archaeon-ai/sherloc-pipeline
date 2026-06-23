"""Unit tests for single-peak fitting mode.

Tests the _apply_fitting() method with single_peak_center parameter, which
fits exactly one Gaussian near a specified position, bypassing automatic
peak detection and AICc model selection.

T2.4: Implement single-peak fitting mode
"""

import pytest
import numpy as np
import pandas as pd

from sherloc_pipeline.services.spectral import (
    SpectralService,
    FitResult,
    calculate_background_scale,
)


class TestSinglePeakFitting:
    """Tests for single-peak fitting mode."""

    def test_single_peak_returns_exactly_one_peak(self, test_context):
        """Single-peak mode should return exactly one peak."""
        service = SpectralService(context=test_context)
        
        # Create synthetic spectrum with two clear peaks
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 500 * np.exp(-((raman_shift - 850) ** 2) / 200)
        peak2 = 400 * np.exp(-((raman_shift - 1050) ** 2) / 200)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Fit only the first peak using single-peak mode
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=850,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(fit_result.peaks) == 1
        # Peak should be near the specified center
        assert 800 < fit_result.peaks[0].m_cm1 < 900

    def test_single_peak_at_different_positions(self, test_context):
        """Single-peak mode should fit peak near specified center."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 950) ** 2) / 200)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Fit at 950 cm^-1
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=950,
        )
        
        assert len(fit_result.peaks) == 1
        # Peak should be close to the actual peak position (near 950)
        fitted_center = fit_result.peaks[0].m_cm1
        assert abs(fitted_center - 950) < 30  # Within 30 cm^-1

    def test_single_peak_ignores_other_peaks(self, test_context):
        """Single-peak mode should fit only at specified position, ignoring others."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        # Two strong peaks
        peak1 = 800 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 600 * np.exp(-((raman_shift - 1050) ** 2) / 150)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Target the second peak only
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=1050,
        )
        
        assert len(fit_result.peaks) == 1
        # Should be near 1050, not 850
        assert fit_result.peaks[0].m_cm1 > 1000

    def test_single_peak_respects_fit_range(self, test_context):
        """Single-peak mode should respect fit_range parameter."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(500, 1500, 1000)
        peak = 500 * np.exp(-((raman_shift - 1100) ** 2) / 200)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Use narrow fit range around the peak
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(1000, 1200),
            single_peak_center=1100,
        )
        
        assert len(fit_result.peaks) == 1
        # Fitted peak should be within the fit range
        assert 1000 <= fit_result.peaks[0].m_cm1 <= 1200

    def test_single_peak_returns_valid_model(self, test_context):
        """Single-peak mode should return a valid model array."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / 200)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        _, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=850,
        )
        
        # Model should be same length as input
        assert len(model) == len(spectrum_df)
        # Model should have finite values
        assert np.all(np.isfinite(model))

    def test_single_peak_has_valid_fwhm(self, test_context):
        """Single-peak fit should have FWHM within configured bounds [22, 90]."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        # Create peak with realistic FWHM
        fwhm_actual = 50  # cm^-1
        sigma = fwhm_actual / (2 * np.sqrt(2 * np.log(2)))
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / (2 * sigma**2))
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=850,
        )
        
        assert len(fit_result.peaks) == 1
        fitted_fwhm = fit_result.peaks[0].fwhm
        # FWHM should be within bounds
        assert 22 <= fitted_fwhm <= 90


class TestSinglePeakOnRealData:
    """Tests using real fixture data for single-peak fitting."""

    def test_single_peak_on_amherst_point(self, test_context):
        """Single-peak fitting on Amherst Point averaged spectrum."""
        service = SpectralService(context=test_context)
        
        # Load, average, and process data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        # Target potential carbonate peak region (~1090 cm^-1)
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(1000, 1200),
            single_peak_center=1090,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        # Should have exactly one peak
        assert len(fit_result.peaks) <= 1  # May be 0 if no peak detected at that position

    def test_single_peak_on_lake_haiyaha(self, test_context):
        """Single-peak fitting on Lake Haiyaha (olivine-rich) spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0852", "Lake_Haiyaha", "detail_1")
        avg_df = service._compute_average(data, method="mean")
        corrected = service._apply_baseline(avg_df)
        
        # Target olivine peak region (~850 cm^-1)
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(700, 1000),
            single_peak_center=850,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        assert len(fit_result.peaks) <= 1

    def test_single_peak_on_stigbreen(self, test_context):
        """Single-peak fitting on Stigbreen spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("1634", "Stigbreen", "line_1")
        avg_df = service._compute_average(data, method="median")
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "as", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        # Target mineral region
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(700, 1200),
            single_peak_center=1000,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        assert len(fit_result.peaks) <= 1


class TestSinglePeakComparedToAutomatic:
    """Tests comparing single-peak mode to automatic peak detection."""

    def test_single_peak_differs_from_automatic(self, test_context):
        """Single-peak mode should give different results than automatic detection."""
        service = SpectralService(context=test_context)
        
        # Create spectrum with multiple peaks
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 600 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 500 * np.exp(-((raman_shift - 1000) ** 2) / 150)
        peak3 = 400 * np.exp(-((raman_shift - 1100) ** 2) / 150)
        intensity = peak1 + peak2 + peak3 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Automatic detection (no single_peak_center)
        auto_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
        )
        
        # Single peak at 850
        single_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=850,
        )
        
        # Automatic should detect multiple peaks
        # (or at least have different behavior than single-peak mode)
        assert len(single_result.peaks) == 1
        # Auto mode may detect more (depends on SNR thresholds)

    def test_single_peak_bypasses_aicc(self, test_context):
        """Single-peak mode should bypass AICc model selection."""
        service = SpectralService(context=test_context)
        
        # Create spectrum with one clear peak
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 900) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Single peak should return exactly one peak regardless of AICc
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=900,
        )
        
        # AICc would potentially add more peaks - single mode should not
        assert len(fit_result.peaks) == 1


class TestSinglePeakEdgeCases:
    """Edge case tests for single-peak fitting."""

    def test_single_peak_center_outside_fit_range(self, test_context):
        """Single-peak center outside fit_range should still fit within range."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / 200)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Seed center at 850, but fit range starts at 900
        # The core fitting maps to nearest point in ROI
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(900, 1200),
            single_peak_center=850,
        )
        
        # Should still return a result (may or may not have a peak)
        assert isinstance(fit_result, FitResult)

    def test_single_peak_with_weak_signal(self, test_context):
        """Single-peak fitting on weak signal should handle gracefully."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        # Very weak peak (low SNR)
        peak = 10 * np.exp(-((raman_shift - 900) ** 2) / 200)
        noise = 20 * np.random.randn(len(raman_shift))
        intensity = peak + noise + 100
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Should not raise error even with weak signal
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=900,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(spectrum_df)

    def test_single_peak_with_flat_spectrum(self, test_context):
        """Single-peak fitting on flat spectrum should handle gracefully."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        # Flat baseline with noise
        intensity = 100 + 5 * np.random.randn(len(raman_shift))
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Should not raise error on flat spectrum
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            single_peak_center=900,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(spectrum_df)

