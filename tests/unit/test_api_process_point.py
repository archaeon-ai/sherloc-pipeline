"""
Unit tests for process_point() API function (T4.6).

Tests the public API function for processing single points from Loupe data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sherloc_pipeline.api.spectral import process_point


class TestProcessPointBasic:
    """Test basic process_point() functionality."""

    def test_function_exists_and_importable(self):
        """Verify process_point can be imported from the API."""
        from sherloc_pipeline.api.spectral import process_point
        assert callable(process_point)

    def test_basic_processing(self, fixtures_path):
        """Test basic point processing returns DataFrame and None fit."""
        df, fit = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            data_dir=fixtures_path / "loupe",
        )
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'raman_shift' in df.columns
        assert 'intensity' in df.columns
        assert len(df) > 0
        
        # No fit requested
        assert fit is None

    def test_returns_tuple(self, fixtures_path):
        """Test that function returns a tuple."""
        result = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestProcessPointValidation:
    """Test validation in process_point()."""

    def test_invalid_point_raises_error(self, fixtures_path):
        """Test that out-of-range point raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            process_point(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=9999,
                data_dir=fixtures_path / "loupe",
            )

    def test_negative_point_raises_error(self, fixtures_path):
        """Test that negative point raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            process_point(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=-1,
                data_dir=fixtures_path / "loupe",
            )

    def test_first_point_valid(self, fixtures_path):
        """Test that point 0 is valid."""
        df, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            data_dir=fixtures_path / "loupe",
        )
        assert len(df) > 0


class TestProcessPointProcessing:
    """Test processing options in process_point()."""

    def test_with_background_subtraction(self, fixtures_path):
        """Test point processing with background subtraction."""
        df, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_with_baseline_correction(self, fixtures_path):
        """Test point processing with baseline correction."""
        df, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            baseline=True,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_with_fitting(self, fixtures_path):
        """Test point processing with Gaussian fitting."""
        df, fit = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert fit is not None
        assert hasattr(fit, 'peaks')
        assert hasattr(fit, 'r2')

    def test_with_single_peak_fitting(self, fixtures_path):
        """Test point processing with single-peak fitting."""
        df, fit = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(1000, 1200),
            single_peak_center=1090.0,
            data_dir=fixtures_path / "loupe",
        )
        
        assert fit is not None
        # Single peak mode should produce exactly 1 peak
        assert len(fit.peaks) == 1

    def test_with_n_peaks_limit(self, fixtures_path):
        """Test point processing with n-peaks limit."""
        df, fit = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            n_peaks=2,
            data_dir=fixtures_path / "loupe",
        )
        
        assert fit is not None
        assert len(fit.peaks) <= 2

    def test_full_processing_chain(self, fixtures_path):
        """Test full processing chain: bg-sub + baseline + fit."""
        df, fit = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            baseline=True,
            fit=True,
            fit_range=(700, 1200),
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert fit is not None


class TestProcessPointComparison:
    """Test that process_point() produces expected results."""

    def test_different_points_produce_different_spectra(self, fixtures_path):
        """Test that different points produce different spectra."""
        df0, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            data_dir=fixtures_path / "loupe",
        )
        
        df5, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            data_dir=fixtures_path / "loupe",
        )
        
        # Different points should have different intensities
        assert not np.allclose(df0['intensity'].values, df5['intensity'].values)

    def test_processing_modifies_spectrum(self, fixtures_path):
        """Test that processing steps modify the spectrum."""
        # Raw spectrum
        df_raw, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            data_dir=fixtures_path / "loupe",
        )
        
        # Processed spectrum
        df_proc, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            background="fs",
            baseline=True,
            data_dir=fixtures_path / "loupe",
        )
        
        # Processing should change the spectrum
        assert not np.allclose(df_raw['intensity'].values, df_proc['intensity'].values)


class TestProcessPointAllDatasets:
    """Test process_point() on all fixture datasets."""

    @pytest.mark.parametrize("sol,target,scan,point", [
        ("0921", "Amherst_Point", "detail_1", 5),
        ("0852", "Lake_Haiyaha", "detail_1", 10),
        ("1634", "Stigbreen", "line_1", 15),
    ])
    def test_on_all_fixtures(self, sol, target, scan, point, fixtures_path):
        """Test process_point works on all fixture datasets."""
        df, _ = process_point(
            sol=sol,
            target=target,
            scan=scan,
            point=point,
            background="fs",
            baseline=True,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestProcessPointDataDir:
    """Test data_dir and results_dir overrides."""

    def test_data_dir_override(self, fixtures_path):
        """Test that data_dir override works."""
        df, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            data_dir=fixtures_path / "loupe",
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_results_dir_has_no_effect(self, fixtures_path, tmp_path):
        """Test that results_dir doesn't affect processing (no file output)."""
        df, _ = process_point(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path,
        )
        
        # Should work - results_dir not used in API (no file output)
        assert isinstance(df, pd.DataFrame)

