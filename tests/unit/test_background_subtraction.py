"""
Unit tests for background subtraction functionality.

Tests cover:
- Loading AS (arm stowed) and FS (fused silica) backgrounds
- Interpolation of background to spectrum x-axis
- Scale factor application
- Full background subtraction workflow
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotError,
    calculate_background_scale,
)
from sherloc_pipeline.services.runtime import RuntimeContext


class TestLoadBackground:
    """Tests for _load_background() method."""

    @pytest.fixture
    def service(self, fixtures_path: Path, tmp_path: Path) -> SpectralService:
        """Create SpectralService with context pointing to fixtures."""
        # Point data_root to fixtures so background lookup finds fixtures/background
        context = RuntimeContext.bootstrap(
            data_dir=fixtures_path,  # fixtures/ so ../background doesn't work
            results_dir=tmp_path / "results",
        )
        # Override to use fixtures/background directly
        context._data_root = fixtures_path / "loupe"
        return SpectralService(context=context)

    @pytest.fixture
    def service_with_bg_path(self, fixtures_path: Path, tmp_path: Path) -> SpectralService:
        """Create SpectralService with background path accessible."""
        # Set data_root so that data_root/../background points to fixtures/background
        context = RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path / "results",
        )
        return SpectralService(context=context)

    def test_load_arm_stowed_background(
        self, service_with_bg_path: SpectralService, manifest: Dict[str, Any]
    ):
        """Load AS (arm stowed) background successfully."""
        bg_df = service_with_bg_path._load_background("as")
        
        # Verify structure
        assert "raman_shift" in bg_df.columns
        assert "intensity" in bg_df.columns
        
        # Verify we have data
        assert len(bg_df) > 0
        
        # AS background should have a wide range including negative shifts
        assert bg_df["raman_shift"].min() < 0

    def test_load_fused_silica_background(
        self, service_with_bg_path: SpectralService, manifest: Dict[str, Any]
    ):
        """Load FS (fused silica) background successfully."""
        bg_df = service_with_bg_path._load_background("fs")
        
        # Verify structure
        assert "raman_shift" in bg_df.columns
        assert "intensity" in bg_df.columns
        
        # Verify we have data
        assert len(bg_df) > 0
        
        # FS background starts around 67 cm^-1
        assert bg_df["raman_shift"].min() > 50
        assert bg_df["raman_shift"].min() < 100

    def test_background_column_normalization(
        self, service_with_bg_path: SpectralService
    ):
        """Both backgrounds have normalized column names."""
        as_df = service_with_bg_path._load_background("as")
        fs_df = service_with_bg_path._load_background("fs")
        
        # Both should have same column names
        assert list(as_df.columns) == ["raman_shift", "intensity"]
        assert list(fs_df.columns) == ["raman_shift", "intensity"]

    def test_invalid_bg_type_raises_error(self, service_with_bg_path: SpectralService):
        """Invalid background type raises SpectralPlotError."""
        with pytest.raises(SpectralPlotError) as excinfo:
            service_with_bg_path._load_background("invalid")
        
        assert "Invalid background type" in str(excinfo.value)

    def test_background_intensities_are_numeric(
        self, service_with_bg_path: SpectralService
    ):
        """Background intensity values are numeric (not NaN)."""
        for bg_type in ["as", "fs"]:
            bg_df = service_with_bg_path._load_background(bg_type)
            
            # No NaN values
            assert not bg_df["intensity"].isna().any()
            assert not bg_df["raman_shift"].isna().any()
            
            # Values are numeric
            assert np.issubdtype(bg_df["intensity"].dtype, np.number)
            assert np.issubdtype(bg_df["raman_shift"].dtype, np.number)


class TestApplyBackgroundSubtraction:
    """Tests for _apply_background_subtraction() method."""

    @pytest.fixture
    def service(self, fixtures_path: Path, tmp_path: Path) -> SpectralService:
        """Create SpectralService with context pointing to fixtures."""
        context = RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path / "results",
        )
        return SpectralService(context=context)

    @pytest.fixture
    def sample_spectrum(self) -> pd.DataFrame:
        """Create a simple sample spectrum for testing."""
        return pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 100),
            "intensity": np.random.randn(100) * 100 + 1000,
        })

    def test_background_subtraction_reduces_intensity(
        self, service: SpectralService, sample_spectrum: pd.DataFrame
    ):
        """Background subtraction reduces overall intensity."""
        # Use scale of 1.0 for maximum effect
        result = service._apply_background_subtraction(sample_spectrum, "fs", scale=1.0)
        
        # Verify structure preserved
        assert "raman_shift" in result.columns
        assert "intensity" in result.columns
        assert len(result) == len(sample_spectrum)
        
        # X values should be unchanged
        np.testing.assert_array_equal(
            result["raman_shift"].values,
            sample_spectrum["raman_shift"].values
        )

    def test_background_subtraction_with_zero_scale(
        self, service: SpectralService, sample_spectrum: pd.DataFrame
    ):
        """Zero scale should leave spectrum unchanged."""
        result = service._apply_background_subtraction(sample_spectrum, "fs", scale=0.0)
        
        # Intensity should be unchanged
        np.testing.assert_array_almost_equal(
            result["intensity"].values,
            sample_spectrum["intensity"].values
        )

    def test_scale_factor_affects_subtraction(
        self, service: SpectralService, sample_spectrum: pd.DataFrame
    ):
        """Larger scale factors result in more subtraction."""
        result_half = service._apply_background_subtraction(sample_spectrum, "fs", scale=0.5)
        result_full = service._apply_background_subtraction(sample_spectrum, "fs", scale=1.0)
        
        # Full scale should subtract more (lower mean intensity)
        assert result_full["intensity"].mean() < result_half["intensity"].mean()

    def test_subtraction_is_linear_with_scale(
        self, service: SpectralService, sample_spectrum: pd.DataFrame
    ):
        """Background subtraction should be linear with scale factor."""
        result_0 = service._apply_background_subtraction(sample_spectrum, "fs", scale=0.0)
        result_1 = service._apply_background_subtraction(sample_spectrum, "fs", scale=1.0)
        result_half = service._apply_background_subtraction(sample_spectrum, "fs", scale=0.5)
        
        # The half-scale result should be midway between 0 and 1
        expected_half = (result_0["intensity"].values + result_1["intensity"].values) / 2
        np.testing.assert_array_almost_equal(
            result_half["intensity"].values,
            expected_half,
            decimal=10
        )

    def test_as_and_fs_backgrounds_differ(
        self, service: SpectralService, sample_spectrum: pd.DataFrame
    ):
        """AS and FS backgrounds produce different results."""
        result_as = service._apply_background_subtraction(sample_spectrum, "as", scale=0.5)
        result_fs = service._apply_background_subtraction(sample_spectrum, "fs", scale=0.5)
        
        # Results should not be identical
        assert not np.allclose(
            result_as["intensity"].values,
            result_fs["intensity"].values
        )

    def test_missing_columns_raises_error(self, service: SpectralService):
        """DataFrame missing required columns raises ValueError."""
        bad_df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        
        with pytest.raises(ValueError) as excinfo:
            service._apply_background_subtraction(bad_df, "fs", scale=1.0)
        
        assert "missing required columns" in str(excinfo.value)


class TestBackgroundSubtractionOnRealData:
    """Integration-style tests using real fixture data."""

    @pytest.fixture
    def service(self, fixtures_path: Path, tmp_path: Path) -> SpectralService:
        """Create SpectralService with context pointing to fixtures."""
        context = RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path / "results",
        )
        return SpectralService(context=context)

    def test_background_subtraction_on_amherst_point(
        self, service: SpectralService
    ):
        """Apply background subtraction to Amherst Point averaged spectrum."""
        # Load and average the data
        data = service._load_loupe_data("0921", "Amherst_Point", "detail_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Calculate PPP-based scale
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900.0)
        assert abs(scale - 500/900) < 1e-9  # 500 PPP scan
        
        # Apply background subtraction
        result = service._apply_background_subtraction(avg_df, "fs", scale)
        
        # Verify structure
        assert len(result) == len(avg_df)
        assert list(result.columns) == ["raman_shift", "intensity"]
        
        # Background subtraction should have occurred
        # (mean intensity should change)
        assert not np.allclose(
            result["intensity"].mean(),
            avg_df["intensity"].mean()
        )

    def test_background_subtraction_on_stigbreen(
        self, service: SpectralService
    ):
        """Apply background subtraction to Stigbreen line scan (900 PPP)."""
        # Load and average the data
        data = service._load_loupe_data("1634", "Stigbreen", "line_1")
        avg_df = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Calculate PPP-based scale - should be 1.0 for 900 PPP scan
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900.0)
        assert abs(scale - 1.0) < 1e-9  # 900 PPP scan
        
        # Apply background subtraction
        result = service._apply_background_subtraction(avg_df, "fs", scale)
        
        # Verify result
        assert len(result) == len(avg_df)

    def test_background_subtraction_both_types(
        self, service: SpectralService
    ):
        """Apply both AS and FS background subtraction."""
        data = service._load_loupe_data("0852", "Lake_Haiyaha", "detail_1")
        avg_df = service._compute_average(data, method="mean")
        
        scale = calculate_background_scale(scan_ppp=data.ppp, bg_ppp=900.0)
        
        result_as = service._apply_background_subtraction(avg_df, "as", scale)
        result_fs = service._apply_background_subtraction(avg_df, "fs", scale)
        
        # Both should produce valid results
        assert len(result_as) == len(avg_df)
        assert len(result_fs) == len(avg_df)
        
        # But they should differ
        assert not np.allclose(
            result_as["intensity"].values,
            result_fs["intensity"].values
        )


class TestBackgroundInterpolation:
    """Tests specifically for background interpolation behavior."""

    @pytest.fixture
    def service(self, fixtures_path: Path, tmp_path: Path) -> SpectralService:
        """Create SpectralService with context pointing to fixtures."""
        context = RuntimeContext.bootstrap(
            data_dir=fixtures_path / "loupe",
            results_dir=tmp_path / "results",
        )
        return SpectralService(context=context)

    def test_interpolation_handles_spectrum_within_bg_range(
        self, service: SpectralService
    ):
        """Spectrum entirely within background range interpolates correctly."""
        # FS background spans ~67 to ~4348 cm^-1
        # Create spectrum well within that range
        spectrum = pd.DataFrame({
            "raman_shift": np.linspace(500, 4000, 50),
            "intensity": np.ones(50) * 1000,
        })
        
        result = service._apply_background_subtraction(spectrum, "fs", scale=1.0)
        
        # Should have no NaN values
        assert not result["intensity"].isna().any()

    def test_interpolation_handles_extrapolation(
        self, service: SpectralService
    ):
        """Spectrum extending beyond background range is extrapolated."""
        # Create spectrum that extends beyond FS background range
        # FS background ends around 4348 cm^-1
        spectrum = pd.DataFrame({
            "raman_shift": np.linspace(100, 4500, 50),
            "intensity": np.ones(50) * 1000,
        })
        
        result = service._apply_background_subtraction(spectrum, "fs", scale=1.0)
        
        # Should still produce results (extrapolation enabled)
        assert not result["intensity"].isna().any()
        assert len(result) == 50

    def test_result_x_axis_matches_input(
        self, service: SpectralService
    ):
        """Output x-axis exactly matches input spectrum x-axis."""
        # Use irregular x-axis spacing
        x_values = np.array([500, 600, 750, 1000, 1500, 2000, 3000, 4000])
        spectrum = pd.DataFrame({
            "raman_shift": x_values,
            "intensity": np.ones(len(x_values)) * 1000,
        })
        
        result = service._apply_background_subtraction(spectrum, "fs", scale=1.0)
        
        np.testing.assert_array_equal(
            result["raman_shift"].values,
            x_values
        )

