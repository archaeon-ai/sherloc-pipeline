"""Unit tests for core/preprocessing.py -- cosmic ray removal and baseline orchestration.

Regression guards capturing current behavior before structural refactor.
Do NOT test plotting functions (those will move in R-006).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sherloc_pipeline.core.preprocessing import (
    DespikeParams,
    despike_r1_spectrum,
    despike_r1_dataframe,
    baseline_aspls,
    baseline_r1_dataframe,
    build_weight_vector_from_windows,
)
from sherloc_pipeline.core.baseline import BaselineParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_despike_params():
    return DespikeParams()


@pytest.fixture
def default_baseline_params():
    return BaselineParams()


@pytest.fixture
def clean_spectrum():
    """Smooth spectrum with no spikes."""
    rng = np.random.RandomState(42)
    x = np.linspace(800, 4000, 523)
    y = 200 + 50 * np.sin(2 * np.pi * x / 1000) + rng.normal(0, 2, size=len(x))
    return pd.Series(y, index=pd.RangeIndex(len(y)))


@pytest.fixture
def spiked_spectrum():
    """Spectrum with artificial spike at index 100."""
    rng = np.random.RandomState(42)
    x = np.linspace(800, 4000, 523)
    y = 200 + 50 * np.sin(2 * np.pi * x / 1000) + rng.normal(0, 2, size=len(x))
    y[100] = 5000.0  # large spike
    return pd.Series(y, index=pd.RangeIndex(len(y)))


@pytest.fixture
def raman_shift_array():
    """Raman shift values corresponding to 523-channel R1 spectrum."""
    return np.linspace(800, 4000, 523)


@pytest.fixture
def small_r1_dataframe(raman_shift_array):
    """Small R1 DataFrame with 3 point columns and raman_shift."""
    rng = np.random.RandomState(42)
    n = len(raman_shift_array)
    data = {"raman_shift": raman_shift_array}
    for col in range(3):
        data[col] = 200 + rng.normal(0, 5, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# despike_r1_spectrum tests
# ---------------------------------------------------------------------------


class TestDespikeR1Spectrum:

    def test_clean_spectrum_mostly_unchanged(self, clean_spectrum, default_despike_params):
        despiked, mask = despike_r1_spectrum(clean_spectrum, default_despike_params)
        # With RNG seed 42 and zscore_threshold=6, a few noise values may trigger
        # due to edge effects in the rolling window (min_periods=1).
        # Key property: the vast majority of the spectrum is untouched.
        spike_count = mask.sum()
        assert spike_count < 20, f"Expected <20 spikes in smooth+noise data, got {spike_count}"
        # Despiked output should still be close to original (most values untouched)
        unchanged = ~mask
        assert_allclose(despiked.values[unchanged], clean_spectrum.values[unchanged], atol=1e-10)

    def test_constant_high_amplitude_no_interior_spikes(self, default_despike_params):
        """Constant spectrum at high amplitude: no variance → no spikes."""
        spectrum = pd.Series(np.full(200, 5000.0))
        despiked, mask = despike_r1_spectrum(spectrum, default_despike_params)
        assert not mask.any()
        assert_allclose(despiked.values, 5000.0, atol=1e-10)

    def test_spike_removed(self, spiked_spectrum, default_despike_params):
        despiked, mask = despike_r1_spectrum(spiked_spectrum, default_despike_params)
        # The spike at index 100 should be detected
        assert mask.iloc[100], "Spike at index 100 should be detected"
        # After despiking, the value should be much lower than original spike
        assert despiked.iloc[100] < 1000.0, "Spike should be reduced after despiking"

    def test_output_shapes(self, clean_spectrum, default_despike_params):
        despiked, mask = despike_r1_spectrum(clean_spectrum, default_despike_params)
        assert despiked.shape == clean_spectrum.shape
        assert mask.shape == clean_spectrum.shape

    def test_type_error_on_non_series(self, default_despike_params):
        with pytest.raises(TypeError, match="intensity_series must be a pandas Series"):
            despike_r1_spectrum(np.array([1.0, 2.0]), default_despike_params)

    def test_invalid_window_size_even(self):
        with pytest.raises(ValueError, match="window_size must be an odd integer"):
            despike_r1_spectrum(
                pd.Series([1.0, 2.0, 3.0]),
                DespikeParams(window_size=4),
            )

    def test_invalid_window_size_too_small(self):
        with pytest.raises(ValueError, match="window_size must be an odd integer"):
            despike_r1_spectrum(
                pd.Series([1.0, 2.0, 3.0]),
                DespikeParams(window_size=1),
            )

    def test_with_raman_shift(self, spiked_spectrum, default_despike_params, raman_shift_array):
        despiked, mask = despike_r1_spectrum(
            spiked_spectrum, default_despike_params, raman_shift=raman_shift_array
        )
        assert despiked.shape == spiked_spectrum.shape

    def test_constant_spectrum_no_spikes(self, default_despike_params):
        """Constant spectrum should have no variance → no spikes."""
        spectrum = pd.Series(np.full(100, 42.0))
        despiked, mask = despike_r1_spectrum(spectrum, default_despike_params)
        assert not mask.any()
        assert_allclose(despiked.values, 42.0, atol=1e-10)


# ---------------------------------------------------------------------------
# despike_r1_dataframe tests
# ---------------------------------------------------------------------------


class TestDespikeR1Dataframe:

    def test_dataframe_output_shape(self, small_r1_dataframe, default_despike_params):
        despiked_df, mask_df = despike_r1_dataframe(small_r1_dataframe, default_despike_params)
        assert despiked_df.shape == small_r1_dataframe.shape
        assert "raman_shift" in despiked_df.columns
        assert mask_df.shape[0] == small_r1_dataframe.shape[0]

    def test_raman_shift_preserved(self, small_r1_dataframe, default_despike_params):
        despiked_df, _ = despike_r1_dataframe(small_r1_dataframe, default_despike_params)
        assert_allclose(
            despiked_df["raman_shift"].values,
            small_r1_dataframe["raman_shift"].values,
            atol=1e-10,
        )

    def test_missing_raman_shift_raises(self, default_despike_params):
        df = pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]})
        with pytest.raises(ValueError, match="Expected 'raman_shift' column"):
            despike_r1_dataframe(df, default_despike_params)

    def test_no_point_columns_raises(self, default_despike_params):
        df = pd.DataFrame({"raman_shift": [800.0, 900.0], "other": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No integer point columns found"):
            despike_r1_dataframe(df, default_despike_params)

    def test_point_columns_sorted(self, small_r1_dataframe, default_despike_params):
        despiked_df, mask_df = despike_r1_dataframe(small_r1_dataframe, default_despike_params)
        point_cols = [c for c in despiked_df.columns if isinstance(c, int)]
        assert point_cols == sorted(point_cols)


# ---------------------------------------------------------------------------
# baseline_aspls tests
# ---------------------------------------------------------------------------


class TestBaselineAspls:

    def test_output_shape(self, clean_spectrum, default_baseline_params):
        corrected, baseline = baseline_aspls(clean_spectrum, default_baseline_params)
        assert corrected.shape == clean_spectrum.shape
        assert baseline.shape == clean_spectrum.shape

    def test_type_error_on_non_series(self, default_baseline_params):
        with pytest.raises(TypeError, match="intensity_series must be a pandas Series"):
            baseline_aspls(np.array([1.0, 2.0, 3.0]), default_baseline_params)

    def test_corrected_plus_baseline_equals_input(self, clean_spectrum, default_baseline_params):
        corrected, baseline = baseline_aspls(clean_spectrum, default_baseline_params)
        reconstructed = corrected.values + baseline.values
        assert_allclose(reconstructed, clean_spectrum.values, atol=1e-10)


# ---------------------------------------------------------------------------
# baseline_r1_dataframe tests
# ---------------------------------------------------------------------------


class TestBaselineR1Dataframe:

    def test_output_shape(self, small_r1_dataframe, default_baseline_params):
        corrected_df, baseline_df = baseline_r1_dataframe(small_r1_dataframe, default_baseline_params)
        assert corrected_df.shape == small_r1_dataframe.shape
        assert baseline_df.shape == small_r1_dataframe.shape

    def test_raman_shift_preserved(self, small_r1_dataframe, default_baseline_params):
        corrected_df, baseline_df = baseline_r1_dataframe(small_r1_dataframe, default_baseline_params)
        assert_allclose(
            corrected_df["raman_shift"].values,
            small_r1_dataframe["raman_shift"].values,
            atol=1e-10,
        )

    def test_missing_raman_shift_raises(self, default_baseline_params):
        df = pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]})
        with pytest.raises(ValueError, match="Expected 'raman_shift' column"):
            baseline_r1_dataframe(df, default_baseline_params)

    def test_no_point_columns_raises(self, default_baseline_params):
        df = pd.DataFrame({"raman_shift": [800.0, 900.0], "other": [1.0, 2.0]})
        with pytest.raises(ValueError, match="No integer point columns found"):
            baseline_r1_dataframe(df, default_baseline_params)

    def test_with_weights(self, small_r1_dataframe, default_baseline_params):
        weights = np.ones(len(small_r1_dataframe))
        corrected_df, baseline_df = baseline_r1_dataframe(
            small_r1_dataframe, default_baseline_params, weights=weights
        )
        assert corrected_df.shape == small_r1_dataframe.shape


# ---------------------------------------------------------------------------
# build_weight_vector_from_windows tests
# ---------------------------------------------------------------------------


class TestBuildWeightVector:

    def test_output_shape(self):
        raman_shift = np.linspace(800, 1200, 200)
        weights = build_weight_vector_from_windows(raman_shift, [(900, 1000)])
        assert weights.shape == raman_shift.shape

    def test_default_weight_outside_windows(self):
        raman_shift = np.linspace(800, 1200, 200)
        weights = build_weight_vector_from_windows(raman_shift, [(900, 1000)])
        outside = (raman_shift < 900) | (raman_shift > 1000)
        assert_allclose(weights[outside], 1.0)

    def test_keep_weight_inside_windows(self):
        raman_shift = np.linspace(800, 1200, 200)
        weights = build_weight_vector_from_windows(raman_shift, [(900, 1000)])
        inside = (raman_shift >= 900) & (raman_shift <= 1000)
        assert_allclose(weights[inside], 0.01)

    def test_multiple_windows(self):
        raman_shift = np.linspace(800, 1200, 400)
        weights = build_weight_vector_from_windows(
            raman_shift, [(850, 870), (1050, 1100)], keep_weight=0.05
        )
        w1 = (raman_shift >= 850) & (raman_shift <= 870)
        w2 = (raman_shift >= 1050) & (raman_shift <= 1100)
        assert_allclose(weights[w1], 0.05)
        assert_allclose(weights[w2], 0.05)

    def test_custom_weights(self):
        raman_shift = np.linspace(800, 1200, 200)
        weights = build_weight_vector_from_windows(
            raman_shift, [(900, 1000)], default_weight=2.0, keep_weight=0.5
        )
        outside = (raman_shift < 900) | (raman_shift > 1000)
        inside = (raman_shift >= 900) & (raman_shift <= 1000)
        assert_allclose(weights[outside], 2.0)
        assert_allclose(weights[inside], 0.5)

    def test_empty_windows(self):
        raman_shift = np.linspace(800, 1200, 200)
        weights = build_weight_vector_from_windows(raman_shift, [])
        assert_allclose(weights, 1.0)
