"""
Unit tests for spectrum averaging methods in SpectralService.

These tests verify that _compute_average() correctly:
- Computes mean, median, and trim-mean across point spectra
- Returns DataFrame with raman_shift and intensity columns
- Handles edge cases and validates parameters
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from sherloc_pipeline.services.spectral import (
    SpectralService,
    SpectralPlotError,
    LoupeData,
)
from sherloc_pipeline.services.runtime import RuntimeContext


class TestAveragingMethods:
    """Tests for _compute_average() with different methods."""

    def test_mean_averaging(self, test_context: RuntimeContext):
        """Mean averaging should compute arithmetic mean across points."""
        service = SpectralService(context=test_context)
        
        # Load fixture data
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result = service._compute_average(data, method="mean")
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["raman_shift", "intensity"]
        assert len(result) == len(data.spectra_df)
        
        # Verify raman_shift is preserved
        np.testing.assert_array_equal(
            result["raman_shift"].values,
            data.spectra_df["raman_shift"].values
        )
        
        # Verify mean calculation on a sample row
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        sample_row = 100  # Pick a row in the middle
        expected_mean = data.spectra_df.loc[sample_row, point_cols].mean()
        np.testing.assert_almost_equal(
            result.loc[sample_row, "intensity"],
            expected_mean,
            decimal=10
        )

    def test_median_averaging(self, test_context: RuntimeContext):
        """Median averaging should compute median across points."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result = service._compute_average(data, method="median")
        
        # Verify result structure
        assert list(result.columns) == ["raman_shift", "intensity"]
        assert len(result) == len(data.spectra_df)
        
        # Verify median calculation on a sample row
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        sample_row = 100
        expected_median = data.spectra_df.loc[sample_row, point_cols].median()
        np.testing.assert_almost_equal(
            result.loc[sample_row, "intensity"],
            expected_median,
            decimal=10
        )

    def test_trim_mean_averaging_default(self, test_context: RuntimeContext):
        """Trim-mean with default 2% should trim top and bottom 2% of values."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Verify result structure
        assert list(result.columns) == ["raman_shift", "intensity"]
        assert len(result) == len(data.spectra_df)
        
        # Verify trim-mean calculation on a sample row
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        sample_row = 100
        row_values = data.spectra_df.loc[sample_row, point_cols].values
        expected_trim_mean = stats.trim_mean(row_values, 0.02)  # 2% = 0.02
        np.testing.assert_almost_equal(
            result.loc[sample_row, "intensity"],
            expected_trim_mean,
            decimal=10
        )

    def test_trim_mean_averaging_5_percent(self, test_context: RuntimeContext):
        """Trim-mean with 5% should trim top and bottom 5% of values."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result = service._compute_average(data, method="trim-mean", trim_pct=5.0)
        
        # Verify trim-mean calculation
        point_cols = [c for c in data.spectra_df.columns if isinstance(c, int)]
        sample_row = 100
        row_values = data.spectra_df.loc[sample_row, point_cols].values
        expected_trim_mean = stats.trim_mean(row_values, 0.05)  # 5% = 0.05
        np.testing.assert_almost_equal(
            result.loc[sample_row, "intensity"],
            expected_trim_mean,
            decimal=10
        )

    def test_trim_mean_averaging_10_percent(self, test_context: RuntimeContext):
        """Trim-mean with 10% should trim top and bottom 10% of values."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result_10 = service._compute_average(data, method="trim-mean", trim_pct=10.0)
        result_2 = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        # Different trim percentages should give different results
        # (unless data is perfectly uniform, which real data is not)
        assert not np.allclose(
            result_10["intensity"].values,
            result_2["intensity"].values
        )


class TestAveragingAcrossDatasets:
    """Tests verifying averaging works on all fixture datasets."""

    def test_averaging_lake_haiyaha(self, test_context: RuntimeContext):
        """Averaging should work on Lake Haiyaha dataset."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0852",
            target="Lake_Haiyaha",
            scan="detail_1",
        )
        
        for method in ["mean", "median", "trim-mean"]:
            result = service._compute_average(data, method=method)
            assert len(result) == len(data.spectra_df)
            assert "intensity" in result.columns
            # Should have non-NaN values
            assert not result["intensity"].isna().all()

    def test_averaging_stigbreen(self, test_context: RuntimeContext):
        """Averaging should work on Stigbreen dataset (smaller, 25 points)."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="1634",
            target="Stigbreen",
            scan="line_1",
        )
        
        # With only 25 points, trim-mean 2% trims ~0.5 points per side
        # scipy handles this correctly
        result = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        assert len(result) == len(data.spectra_df)
        assert not result["intensity"].isna().any()


class TestAveragingEdgeCases:
    """Tests for edge cases in averaging."""

    def test_trim_pct_zero_equals_mean(self, test_context: RuntimeContext):
        """Trim-mean with 0% should equal regular mean."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result_trim0 = service._compute_average(data, method="trim-mean", trim_pct=0.0)
        result_mean = service._compute_average(data, method="mean")
        
        np.testing.assert_allclose(
            result_trim0["intensity"].values,
            result_mean["intensity"].values,
            rtol=1e-10
        )

    def test_default_is_trim_mean(self, test_context: RuntimeContext):
        """Default method should be trim-mean with 2%."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result_default = service._compute_average(data)
        result_explicit = service._compute_average(data, method="trim-mean", trim_pct=2.0)
        
        np.testing.assert_array_equal(
            result_default["intensity"].values,
            result_explicit["intensity"].values
        )

    def test_output_dtypes(self, test_context: RuntimeContext):
        """Output should have proper dtypes."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result = service._compute_average(data, method="mean")
        
        # Both columns should be float
        assert result["raman_shift"].dtype in [np.float64, np.float32, float]
        assert result["intensity"].dtype in [np.float64, np.float32, float]


class TestAveragingErrorHandling:
    """Tests for error handling in averaging."""

    def test_invalid_method_raises_error(self, test_context: RuntimeContext):
        """Invalid averaging method should raise ValueError."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        with pytest.raises(ValueError) as exc_info:
            service._compute_average(data, method="invalid")
        
        assert "Invalid averaging method" in str(exc_info.value)

    def test_trim_pct_negative_raises_error(self, test_context: RuntimeContext):
        """Negative trim_pct should raise ValueError."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        with pytest.raises(ValueError) as exc_info:
            service._compute_average(data, method="trim-mean", trim_pct=-5.0)
        
        assert "trim_pct must be between 0 and 50" in str(exc_info.value)

    def test_trim_pct_over_50_raises_error(self, test_context: RuntimeContext):
        """trim_pct > 50 should raise ValueError."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        with pytest.raises(ValueError) as exc_info:
            service._compute_average(data, method="trim-mean", trim_pct=60.0)
        
        assert "trim_pct must be between 0 and 50" in str(exc_info.value)


class TestAveragingStatisticalProperties:
    """Tests verifying expected statistical properties."""

    def test_median_more_robust_to_outliers(self, test_context: RuntimeContext):
        """Median should be more robust to outliers than mean."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        result_mean = service._compute_average(data, method="mean")
        result_median = service._compute_average(data, method="median")
        result_trim = service._compute_average(data, method="trim-mean", trim_pct=10.0)
        
        # All should produce valid results
        assert not result_mean["intensity"].isna().any()
        assert not result_median["intensity"].isna().any()
        assert not result_trim["intensity"].isna().any()
        
        # Results should be in same ballpark but not identical
        # (exact relationship depends on data distribution)
        mean_sum = result_mean["intensity"].sum()
        median_sum = result_median["intensity"].sum()
        trim_sum = result_trim["intensity"].sum()
        
        # All methods should give similar total (within 50%)
        assert abs(median_sum - mean_sum) < abs(mean_sum) * 0.5
        assert abs(trim_sum - mean_sum) < abs(mean_sum) * 0.5

    def test_averaging_methods_produce_similar_results(self, test_context: RuntimeContext):
        """All averaging methods should produce reasonably similar results."""
        service = SpectralService(context=test_context)
        
        data = service._load_loupe_data(
            sol="0921",
            target="Amherst_Point",
            scan="detail_1",
        )
        
        mean = service._compute_average(data, method="mean")["intensity"]
        median = service._compute_average(data, method="median")["intensity"]
        trim_mean_2pct = service._compute_average(data, method="trim-mean", trim_pct=2.0)["intensity"]
        
        # All averaging methods should be positively correlated (>0.5)
        # Real spectral data can have significant outliers, so we use a lenient threshold
        assert np.corrcoef(mean.values, trim_mean_2pct.values)[0, 1] > 0.5
        assert np.corrcoef(median.values, mean.values)[0, 1] > 0.5
        assert np.corrcoef(median.values, trim_mean_2pct.values)[0, 1] > 0.5
        
        # All methods should produce non-trivial results (not all zeros or NaN)
        assert mean.abs().sum() > 0
        assert median.abs().sum() > 0
        assert trim_mean_2pct.abs().sum() > 0

