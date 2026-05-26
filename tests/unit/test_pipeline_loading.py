"""
Unit tests for pipeline output loading in SpectralService.

T2.1: Tests for _load_pipeline_output() method that loads processed
spectra from existing pipeline results.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotError,
)


class TestLoadPipelineOutput:
    """Test _load_pipeline_output() method with fixture data."""
    
    @pytest.fixture
    def pipeline_fixtures_path(self, fixtures_path):
        """Path to pipeline output fixtures."""
        return fixtures_path / "pipeline_outputs"
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        # Override results_root to point to fixtures/pipeline_outputs
        from sherloc_pipeline.services.runtime import RuntimeContext
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_load_normalized_point_0(self, service_with_pipeline_fixtures):
        """Test loading point 0 from normalized level."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized",
            point=0,
        )
        
        # Check structure
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df.columns) == 2
        
        # Check data types
        assert df["raman_shift"].dtype in [np.float64, np.float32]
        assert df["intensity"].dtype in [np.float64, np.float32]
        
        # Check reasonable data range (fixture has 32 rows)
        assert len(df) == 32
        assert df["raman_shift"].iloc[0] == pytest.approx(238.393)
    
    def test_load_normalized_baselined_point_5(self, service_with_pipeline_fixtures):
        """Test loading point 5 from normalized_baselined level."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized_baselined",
            point=5,
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) == 32
    
    def test_load_normalized_despiked_baselined(self, service_with_pipeline_fixtures):
        """Test loading from normalized_despiked_baselined level."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized_despiked_baselined",
            point=3,
        )
        
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) == 32
    
    def test_different_points_have_different_intensities(self, service_with_pipeline_fixtures):
        """Verify that different points have different intensity values."""
        df0 = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921", target="Amherst_Point", scan="detail_1",
            level="normalized", point=0,
        )
        df1 = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921", target="Amherst_Point", scan="detail_1",
            level="normalized", point=1,
        )
        
        # Same raman_shift values
        np.testing.assert_array_almost_equal(
            df0["raman_shift"].values,
            df1["raman_shift"].values
        )
        
        # Different intensity values
        assert not np.allclose(df0["intensity"].values, df1["intensity"].values)
    
    def test_different_levels_have_different_intensities(self, service_with_pipeline_fixtures):
        """Verify that different processing levels produce different data."""
        df_norm = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921", target="Amherst_Point", scan="detail_1",
            level="normalized", point=0,
        )
        df_base = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921", target="Amherst_Point", scan="detail_1",
            level="normalized_baselined", point=0,
        )
        
        # Normalized should have higher values (includes baseline offset)
        assert df_norm["intensity"].mean() > df_base["intensity"].mean()


class TestLoadPipelineOutputErrors:
    """Test error handling in _load_pipeline_output()."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        from sherloc_pipeline.services.runtime import RuntimeContext
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_invalid_level_raises_error(self, service_with_pipeline_fixtures):
        """Test that invalid processing level raises SpectralPlotError."""
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures._load_pipeline_output(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                level="invalid_level",
                point=0,
            )
        
        assert "Invalid processing level" in str(excinfo.value)
        assert "invalid_level" in str(excinfo.value)
    
    def test_nonexistent_file_raises_error(self, service_with_pipeline_fixtures):
        """Test that missing file raises SpectralPlotError."""
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures._load_pipeline_output(
                sol="9999",
                target="Nonexistent_Target",
                scan="detail_1",
                level="normalized",
                point=0,
            )
        
        assert "Pipeline output not found" in str(excinfo.value)
    
    def test_point_out_of_range_raises_error(self, service_with_pipeline_fixtures):
        """Test that out-of-range point index raises SpectralPlotError."""
        # Fixture has 10 points (0-9)
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures._load_pipeline_output(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                level="normalized",
                point=100,  # Out of range
            )
        
        assert "Point 100 not found" in str(excinfo.value)
    
    def test_negative_point_raises_error(self, service_with_pipeline_fixtures):
        """Test that negative point index raises SpectralPlotError."""
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_pipeline_fixtures._load_pipeline_output(
                sol="0921",
                target="Amherst_Point",
                scan="detail_1",
                level="normalized",
                point=-1,
            )
        
        assert "Point -1 not found" in str(excinfo.value)


class TestLoadPipelineOutputAllLevels:
    """Parametrized tests for all valid processing levels."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        from sherloc_pipeline.services.runtime import RuntimeContext
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    @pytest.mark.parametrize("level", [
        "normalized",
        "normalized_baselined",
        "normalized_despiked_baselined",
    ])
    def test_load_each_level(self, service_with_pipeline_fixtures, level):
        """Test that each processing level can be loaded."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level=level,
            point=0,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert "raman_shift" in df.columns
        assert "intensity" in df.columns
        assert len(df) > 0
    
    @pytest.mark.parametrize("point", [0, 1, 5, 9])
    def test_load_various_points(self, service_with_pipeline_fixtures, point):
        """Test loading various point indices (fixture has 10 points)."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized",
            point=point,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLoadPipelineOutputIntegration:
    """Integration tests verifying pipeline loading works with real-like data."""
    
    @pytest.fixture
    def service_with_pipeline_fixtures(self, test_context, fixtures_path):
        """Service configured to use pipeline output fixtures as results."""
        from sherloc_pipeline.services.runtime import RuntimeContext
        context = RuntimeContext.bootstrap(
            data_dir=test_context.data_root,
            results_dir=fixtures_path / "pipeline_outputs",
        )
        return SpectralService(context=context)
    
    def test_loaded_spectrum_has_realistic_raman_range(self, service_with_pipeline_fixtures):
        """Verify loaded spectrum covers expected Raman shift range."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized",
            point=0,
        )
        
        # Fixture data covers ~238 to ~4765 cm^-1 (subset)
        assert df["raman_shift"].min() < 300
        assert df["raman_shift"].max() > 4000
    
    def test_loaded_spectrum_has_peaks(self, service_with_pipeline_fixtures):
        """Verify loaded spectrum has peak features."""
        df = service_with_pipeline_fixtures._load_pipeline_output(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
            level="normalized_baselined",
            point=0,
        )
        
        # Fixture has synthetic peaks around 850 and 1000 cm^-1
        # Check that there's intensity variation
        assert df["intensity"].std() > 0
        assert df["intensity"].max() > df["intensity"].min()

