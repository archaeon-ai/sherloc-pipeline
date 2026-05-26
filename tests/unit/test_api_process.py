"""
Unit tests for the Python API process_scan_average() and process_subset_average() functions.

Tests verify that the API functions correctly wrap SpectralService and return
clean DataFrames and FitResults.
"""

import pytest
import pandas as pd
import numpy as np

from sherloc_pipeline.api.spectral import process_scan_average, process_subset_average


class TestProcessScanAverage:
    """Tests for process_scan_average() API function."""
    
    def test_basic_return_types(self, fixtures_path):
        """Test that function returns (DataFrame, Optional[FitResult]) tuple."""
        df, fit_result = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert fit_result is None  # No fitting requested
    
    def test_dataframe_structure(self, fixtures_path):
        """Test that DataFrame has raman_shift and intensity columns."""
        df, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df.columns) == 2
    
    def test_dataframe_has_data(self, fixtures_path):
        """Test that DataFrame contains spectral data."""
        df, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            data_dir=fixtures_path / "loupe",
        )
        
        assert len(df) > 0
        # R1 spectra should span ~238 to ~4765 cm^-1
        assert df["raman_shift"].min() < 300
        assert df["raman_shift"].max() > 4000
    
    def test_with_background_subtraction(self, fixtures_path):
        """Test that background subtraction changes intensity values."""
        df_no_bg, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        df_with_bg, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        # Background subtraction should change values
        assert not np.allclose(
            df_no_bg["intensity"].values,
            df_with_bg["intensity"].values
        )
    
    def test_with_baseline_correction(self, fixtures_path):
        """Test that baseline correction changes intensity values."""
        df_no_bl, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        df_with_bl, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        # Baseline correction should change values
        assert not np.allclose(
            df_no_bl["intensity"].values,
            df_with_bl["intensity"].values
        )
    
    def test_with_fitting_returns_fit_result(self, fixtures_path):
        """Test that fitting returns FitResult with peaks."""
        df, fit_result = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        
        assert fit_result is not None
        assert hasattr(fit_result, "peaks")
        assert hasattr(fit_result, "r2")
        assert len(fit_result.peaks) > 0
        assert 0 <= fit_result.r2 <= 1
    
    def test_fit_result_peak_properties(self, fixtures_path):
        """Test that FitResult peaks have expected properties."""
        _, fit_result = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        
        for peak in fit_result.peaks:
            assert hasattr(peak, "m_cm1")  # Center position
            assert hasattr(peak, "a")      # Amplitude
            assert hasattr(peak, "fwhm")   # Full width at half max
            assert hasattr(peak, "snr")    # Signal-to-noise ratio
            # Position should be within fit range
            assert 700 <= peak.m_cm1 <= 1200
    
    def test_single_peak_fitting(self, fixtures_path):
        """Test single-peak fitting mode."""
        _, fit_result = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(1000, 1200),
            single_peak_center=1090,
            data_dir=fixtures_path / "loupe",
        )
        
        # Should fit exactly one peak
        assert fit_result is not None
        assert len(fit_result.peaks) == 1
        # Peak should be near the specified center
        assert abs(fit_result.peaks[0].m_cm1 - 1090) < 50
    
    def test_n_peaks_limit(self, fixtures_path):
        """Test n_peaks limit mode."""
        _, fit_result = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            n_peaks=2,
            data_dir=fixtures_path / "loupe",
        )
        
        # Should fit at most 2 peaks
        assert fit_result is not None
        assert len(fit_result.peaks) <= 2
    
    def test_averaging_methods(self, fixtures_path):
        """Test different averaging methods produce different results."""
        df_mean, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            avg_method="mean",
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        df_trim, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            avg_method="trim-mean",
            trim_pct=5.0,
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        # Methods should produce different results (outlier handling)
        # They may be close but not identical
        correlation = np.corrcoef(
            df_mean["intensity"].values,
            df_trim["intensity"].values
        )[0, 1]
        assert correlation > 0.8  # Highly correlated (real data has outliers)
        assert correlation < 1.0  # But not identical
    
    def test_all_fixture_datasets(self, loupe_datasets, fixtures_path):
        """Test that API works with all fixture datasets."""
        for sol, dataset in loupe_datasets.items():
            df, _ = process_scan_average(
                sol=dataset["sol"],
                target=dataset["target"],
                scan=dataset["scan"],
                background=None,
                baseline=False,
                fit=False,
                data_dir=fixtures_path / "loupe",
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


class TestProcessSubsetAverage:
    """Tests for process_subset_average() API function."""
    
    def test_basic_return_types(self, fixtures_path):
        """Test that function returns (DataFrame, Optional[FitResult]) tuple."""
        df, fit_result = process_subset_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            points=[0, 1, 2, 3, 4],
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert fit_result is None
    
    def test_dataframe_structure(self, fixtures_path):
        """Test that DataFrame has raman_shift and intensity columns."""
        df, _ = process_subset_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            points=[0, 5, 10, 15, 20],
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df.columns) == 2
    
    def test_subset_differs_from_full_average(self, fixtures_path):
        """Test that subset average differs from full scan average."""
        df_full, _ = process_scan_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        df_subset, _ = process_subset_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            points=[0, 1, 2],  # Just first 3 points
            background=None,
            baseline=False,
            fit=False,
            data_dir=fixtures_path / "loupe",
        )
        
        # Subset should differ from full average
        assert not np.allclose(
            df_full["intensity"].values,
            df_subset["intensity"].values
        )
    
    def test_requires_minimum_two_points(self, fixtures_path):
        """Test that subset requires at least 2 points."""
        with pytest.raises(ValueError, match="at least 2 points"):
            process_subset_average(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                points=[0],  # Only one point
                data_dir=fixtures_path / "loupe",
            )
    
    def test_invalid_point_indices(self, fixtures_path):
        """Test that invalid point indices raise error."""
        with pytest.raises(ValueError, match="Invalid point indices"):
            process_subset_average(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                points=[0, 1, 999],  # 999 is out of range
                data_dir=fixtures_path / "loupe",
            )
    
    def test_with_fitting(self, fixtures_path):
        """Test subset averaging with fitting."""
        df, fit_result = process_subset_average(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        
        assert fit_result is not None
        assert len(fit_result.peaks) > 0
    
    def test_empty_points_list(self, fixtures_path):
        """Test that empty points list raises error."""
        with pytest.raises(ValueError, match="at least 2 points"):
            process_subset_average(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                points=[],
                data_dir=fixtures_path / "loupe",
            )
    
    def test_negative_point_index(self, fixtures_path):
        """Test that negative point indices raise error."""
        with pytest.raises(ValueError, match="Invalid point indices"):
            process_subset_average(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                points=[-1, 0, 1],
                data_dir=fixtures_path / "loupe",
            )

