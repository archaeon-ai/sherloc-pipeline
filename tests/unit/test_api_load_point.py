"""
Unit tests for the Python API load_point_spectrum() function.

Tests verify that the API function correctly loads pre-processed spectra
from pipeline outputs.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


from sherloc_pipeline.api.spectral import load_point_spectrum


class TestLoadPointSpectrum:
    """Tests for load_point_spectrum() API function."""
    
    @pytest.fixture
    def pipeline_outputs_dir(self, fixtures_path):
        """Return path to pipeline output fixtures."""
        return fixtures_path / "pipeline_outputs"
    
    def test_basic_return_type(self, pipeline_outputs_dir):
        """Test that function returns a DataFrame."""
        df = load_point_spectrum(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            level="normalized",
            results_dir=pipeline_outputs_dir,
        )
        
        assert isinstance(df, pd.DataFrame)
    
    def test_dataframe_structure(self, pipeline_outputs_dir):
        """Test that DataFrame has raman_shift and intensity columns."""
        df = load_point_spectrum(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            level="normalized",
            results_dir=pipeline_outputs_dir,
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df.columns) == 2
    
    def test_dataframe_has_data(self, pipeline_outputs_dir):
        """Test that DataFrame contains spectral data."""
        df = load_point_spectrum(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=0,
            level="normalized",
            results_dir=pipeline_outputs_dir,
        )
        
        assert len(df) > 0
        # Should have raman shift values
        assert df["raman_shift"].min() > 0
    
    def test_all_processing_levels(self, pipeline_outputs_dir):
        """Test loading all three processing levels."""
        levels = ["normalized", "normalized_baselined", "normalized_despiked_baselined"]
        
        for level in levels:
            df = load_point_spectrum(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=0,
                level=level,
                results_dir=pipeline_outputs_dir,
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "raman_shift" in df.columns
            assert "intensity" in df.columns
    
    def test_different_points(self, pipeline_outputs_dir):
        """Test loading different point indices."""
        # Load points 0, 5, and 9 (fixture has 10 points)
        dfs = []
        for point in [0, 5, 9]:
            df = load_point_spectrum(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=point,
                level="normalized",
                results_dir=pipeline_outputs_dir,
            )
            dfs.append(df)
        
        # All should have same shape (same raman shift values)
        assert all(len(df) == len(dfs[0]) for df in dfs)
        
        # But intensity values should differ
        assert not np.allclose(dfs[0]["intensity"].values, dfs[1]["intensity"].values)
    
    def test_levels_produce_different_results(self, pipeline_outputs_dir):
        """Test that different processing levels produce different intensity values."""
        df_norm = load_point_spectrum(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            level="normalized",
            results_dir=pipeline_outputs_dir,
        )
        
        df_baselined = load_point_spectrum(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            point=5,
            level="normalized_baselined",
            results_dir=pipeline_outputs_dir,
        )
        
        # Same raman shift values
        np.testing.assert_array_almost_equal(
            df_norm["raman_shift"].values,
            df_baselined["raman_shift"].values
        )
        
        # But different intensities
        assert not np.allclose(
            df_norm["intensity"].values,
            df_baselined["intensity"].values
        )
    
    def test_invalid_level_raises_error(self, pipeline_outputs_dir):
        """Test that invalid level raises appropriate error."""
        from sherloc_pipeline.services.spectral import SpectralPlotError
        
        with pytest.raises(SpectralPlotError, match="Invalid processing level"):
            load_point_spectrum(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=0,
                level="invalid_level",  # Not a valid level
                results_dir=pipeline_outputs_dir,
            )
    
    def test_missing_file_raises_error(self, pipeline_outputs_dir):
        """Test that missing file raises appropriate error."""
        from sherloc_pipeline.services.spectral import SpectralPlotError
        
        with pytest.raises(SpectralPlotError, match="Pipeline output not found"):
            load_point_spectrum(
                sol="9999",  # Non-existent sol
                target="Fake_Target",
                scan="detail_1",
                point=0,
                level="normalized",
                results_dir=pipeline_outputs_dir,
            )
    
    def test_point_out_of_range_raises_error(self, pipeline_outputs_dir):
        """Test that out-of-range point index raises appropriate error."""
        from sherloc_pipeline.services.spectral import SpectralPlotError
        
        with pytest.raises(SpectralPlotError, match="not found"):
            load_point_spectrum(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=999,  # Out of range (fixture has 10 points)
                level="normalized",
                results_dir=pipeline_outputs_dir,
            )
    
    def test_consistent_raman_shift_across_points(self, pipeline_outputs_dir):
        """Test that all points share the same raman_shift values."""
        raman_shifts = []
        for point in range(3):  # Check first 3 points
            df = load_point_spectrum(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                point=point,
                level="normalized",
                results_dir=pipeline_outputs_dir,
            )
            raman_shifts.append(df["raman_shift"].values)
        
        # All should be identical
        for i in range(1, len(raman_shifts)):
            np.testing.assert_array_equal(raman_shifts[0], raman_shifts[i])


