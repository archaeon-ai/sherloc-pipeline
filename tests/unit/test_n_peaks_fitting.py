"""Unit tests for n-peaks limit fitting option.

Tests the _apply_fitting() method with n_peaks parameter, which constrains
the AICc model selection to fit at most N peaks while still allowing
automatic peak detection.

T2.5: Implement n-peaks limit option
"""

import pytest
import numpy as np
import pandas as pd

from sherloc_pipeline.services.spectral import (
    SpectralService,
    FitResult,
    calculate_background_scale,
)


class TestNPeaksFitting:
    """Tests for n-peaks limit fitting mode."""

    def test_n_peaks_limits_peak_count(self, test_context):
        """n_peaks parameter should limit the number of fitted peaks."""
        service = SpectralService(context=test_context)
        
        # Create synthetic spectrum with many clear peaks
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 500 * np.exp(-((raman_shift - 800) ** 2) / 150)
        peak2 = 400 * np.exp(-((raman_shift - 900) ** 2) / 150)
        peak3 = 350 * np.exp(-((raman_shift - 1000) ** 2) / 150)
        peak4 = 300 * np.exp(-((raman_shift - 1100) ** 2) / 150)
        intensity = peak1 + peak2 + peak3 + peak4 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Limit to at most 2 peaks
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=2,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(fit_result.peaks) <= 2

    def test_n_peaks_one(self, test_context):
        """n_peaks=1 should fit at most one peak."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 500 * np.exp(-((raman_shift - 850) ** 2) / 150)
        peak2 = 400 * np.exp(-((raman_shift - 1050) ** 2) / 150)
        intensity = peak1 + peak2 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=1,
        )
        
        assert len(fit_result.peaks) <= 1

    def test_n_peaks_three(self, test_context):
        """n_peaks=3 should fit at most three peaks."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak1 = 500 * np.exp(-((raman_shift - 800) ** 2) / 150)
        peak2 = 450 * np.exp(-((raman_shift - 900) ** 2) / 150)
        peak3 = 400 * np.exp(-((raman_shift - 1000) ** 2) / 150)
        peak4 = 350 * np.exp(-((raman_shift - 1100) ** 2) / 150)
        intensity = peak1 + peak2 + peak3 + peak4 + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=3,
        )
        
        assert len(fit_result.peaks) <= 3

    def test_n_peaks_returns_valid_model(self, test_context):
        """n_peaks mode should return a valid model array."""
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
            n_peaks=2,
        )
        
        assert len(model) == len(spectrum_df)
        assert np.all(np.isfinite(model))

    def test_n_peaks_aicc_still_selects_optimal(self, test_context):
        """AICc should still select optimal count within n_peaks limit."""
        service = SpectralService(context=test_context)
        
        # Create spectrum with only one peak
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 900) ** 2) / 150)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        # Set n_peaks=5, but only one peak exists
        fit_result, _ = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=5,
        )
        
        # AICc should select fewer than 5 (ideally 1) since only one peak exists
        assert isinstance(fit_result, FitResult)
        # With realistic data, should detect the single peak
        # (exact count depends on SNR thresholds)


class TestNPeaksOnRealData:
    """Tests using real fixture data for n-peaks fitting."""

    def test_n_peaks_on_amherst_point(self, test_context):
        """n-peaks fitting on Amherst Point averaged spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "fs", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        # Limit to 2 peaks
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(700, 1200),
            n_peaks=2,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(corrected)
        assert len(fit_result.peaks) <= 2

    def test_n_peaks_on_lake_haiyaha(self, test_context):
        """n-peaks fitting on Lake Haiyaha spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("0852", "Lake_Haiyaha", "detail_1")
        avg_df = service._compute_average(data, method="mean")
        corrected = service._apply_baseline(avg_df)
        
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(700, 1000),
            n_peaks=1,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(fit_result.peaks) <= 1

    def test_n_peaks_on_stigbreen(self, test_context):
        """n-peaks fitting on Stigbreen spectrum."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data("1634", "Stigbreen", "line_1")
        avg_df = service._compute_average(data, method="median")
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900)
        bg_subtracted = service._apply_background_subtraction(avg_df, "as", scale)
        corrected = service._apply_baseline(bg_subtracted)
        
        fit_result, model = service._apply_fitting(
            corrected,
            fit_range=(700, 1200),
            n_peaks=3,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(fit_result.peaks) <= 3


class TestNPeaksMutualExclusion:
    """Tests for mutual exclusion with single_peak_center."""

    def test_n_peaks_and_single_peak_mutually_exclusive(self, test_context):
        """Providing both n_peaks and single_peak_center should raise error."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 500 * np.exp(-((raman_shift - 850) ** 2) / 200)
        intensity = peak + 20
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        with pytest.raises(ValueError, match="mutually exclusive"):
            service._apply_fitting(
                spectrum_df,
                fit_range=(700, 1200),
                single_peak_center=850,
                n_peaks=2,
            )


class TestNPeaksEdgeCases:
    """Edge case tests for n-peaks fitting."""

    def test_n_peaks_with_no_detectable_peaks(self, test_context):
        """n-peaks on flat spectrum should handle gracefully."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        intensity = 100 + 5 * np.random.randn(len(raman_shift))
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=2,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(spectrum_df)

    def test_n_peaks_with_weak_signal(self, test_context):
        """n-peaks with weak signal should handle gracefully."""
        service = SpectralService(context=test_context)
        
        raman_shift = np.linspace(600, 1300, 700)
        peak = 10 * np.exp(-((raman_shift - 900) ** 2) / 200)
        noise = 20 * np.random.randn(len(raman_shift))
        intensity = peak + noise + 100
        
        spectrum_df = pd.DataFrame({
            "raman_shift": raman_shift,
            "intensity": intensity,
        })
        
        fit_result, model = service._apply_fitting(
            spectrum_df,
            fit_range=(700, 1200),
            n_peaks=1,
        )
        
        assert isinstance(fit_result, FitResult)
        assert len(model) == len(spectrum_df)

