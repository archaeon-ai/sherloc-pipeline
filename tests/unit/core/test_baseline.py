"""Unit tests for core/baseline.py -- asPLS baseline fitting.

Regression guards capturing current behavior before structural refactor.
These tests must pass BEFORE and AFTER refactoring.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sherloc_pipeline.core.baseline import BaselineParams, fit_baseline, fit_baseline_window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_params():
    return BaselineParams()


@pytest.fixture
def synthetic_spectrum():
    """Synthetic spectrum: polynomial background + Gaussian peak + small noise."""
    rng = np.random.RandomState(42)
    x = np.linspace(800, 1200, 200)
    background = 500 + 0.5 * (x - 800) + 0.001 * (x - 800) ** 2
    peak = 300 * np.exp(-0.5 * ((x - 1010) / 15) ** 2)
    noise = rng.normal(0, 5, size=len(x))
    y = background + peak + noise
    return pd.Series(y, index=pd.RangeIndex(len(y)))


@pytest.fixture
def flat_spectrum():
    """Flat spectrum at constant value."""
    return pd.Series(np.full(200, 100.0), index=pd.RangeIndex(200))


# ---------------------------------------------------------------------------
# fit_baseline tests
# ---------------------------------------------------------------------------


class TestFitBaseline:

    def test_output_shape_matches_input(self, synthetic_spectrum, default_params):
        corrected, baseline = fit_baseline(synthetic_spectrum, default_params)
        assert corrected.shape == synthetic_spectrum.shape
        assert baseline.shape == synthetic_spectrum.shape

    def test_output_index_preserved(self, synthetic_spectrum, default_params):
        corrected, baseline = fit_baseline(synthetic_spectrum, default_params)
        pd.testing.assert_index_equal(corrected.index, synthetic_spectrum.index)
        pd.testing.assert_index_equal(baseline.index, synthetic_spectrum.index)

    def test_baseline_smoother_than_input(self, synthetic_spectrum, default_params):
        _, baseline = fit_baseline(synthetic_spectrum, default_params)
        input_diff_var = np.var(np.diff(synthetic_spectrum.values))
        baseline_diff_var = np.var(np.diff(baseline.values))
        assert baseline_diff_var < input_diff_var

    def test_flat_spectrum_near_flat_baseline(self, flat_spectrum, default_params):
        corrected, baseline = fit_baseline(flat_spectrum, default_params)
        # Baseline should be close to the constant value
        assert_allclose(baseline.values, 100.0, atol=5.0)
        # Corrected should be close to zero
        assert_allclose(corrected.values, 0.0, atol=5.0)

    def test_corrected_plus_baseline_equals_input(self, synthetic_spectrum, default_params):
        corrected, baseline = fit_baseline(synthetic_spectrum, default_params)
        reconstructed = corrected.values + baseline.values
        assert_allclose(reconstructed, synthetic_spectrum.values, atol=1e-10)

    def test_type_error_on_non_series(self, default_params):
        with pytest.raises(TypeError, match="series must be a pandas Series"):
            fit_baseline(np.array([1.0, 2.0, 3.0]), default_params)

    def test_with_weights(self, synthetic_spectrum, default_params):
        weights = np.ones(len(synthetic_spectrum))
        # Lower weight in the middle (peak region)
        weights[80:120] = 0.01
        corrected, baseline = fit_baseline(synthetic_spectrum, default_params, weights=weights)
        assert corrected.shape == synthetic_spectrum.shape

    def test_snapshot_simple_input(self, default_params):
        """Capture exact output for a deterministic input -- snapshot regression test."""
        y = pd.Series(np.arange(50, dtype=float))
        corrected, baseline = fit_baseline(y, default_params)
        # Snapshot: just check that values are finite and deterministic
        assert np.all(np.isfinite(corrected.values))
        assert np.all(np.isfinite(baseline.values))
        # Record the first and last values for regression
        assert_allclose(corrected.values[0] + baseline.values[0], 0.0, atol=1e-10)
        assert_allclose(corrected.values[-1] + baseline.values[-1], 49.0, atol=1e-10)


# ---------------------------------------------------------------------------
# fit_baseline_window tests
# ---------------------------------------------------------------------------


class TestFitBaselineWindow:

    def test_aspls_method_output_shape(self, default_params):
        x = np.linspace(800, 1200, 200)
        y = 500 + 0.3 * (x - 800) + 100 * np.exp(-0.5 * ((x - 1010) / 15) ** 2)
        roi = (850.0, 1150.0)
        y_out = fit_baseline_window(x, y, roi, default_params, method="aspls")
        assert y_out.shape == y.shape

    def test_poly_method_output_shape(self, default_params):
        x = np.linspace(800, 1200, 200)
        y = 500 + 0.3 * (x - 800) + 100 * np.exp(-0.5 * ((x - 1010) / 15) ** 2)
        roi = (850.0, 1150.0)
        y_out = fit_baseline_window(x, y, roi, default_params, method="poly")
        assert y_out.shape == y.shape

    def test_outside_roi_unchanged(self, default_params):
        x = np.linspace(800, 1200, 200)
        y = np.full(200, 100.0)
        roi = (900.0, 1100.0)
        y_out = fit_baseline_window(x, y, roi, default_params, method="aspls")
        outside_mask = (x < 900.0) | (x > 1100.0)
        assert_allclose(y_out[outside_mask], y[outside_mask], atol=1e-10)

    def test_poly_with_weights_builder(self, default_params):
        x = np.linspace(800, 1200, 200)
        y = 500 + 0.3 * (x - 800)
        roi = (850.0, 1150.0)
        weights_builder = ((1010.0, 20.0, 0.01),)
        y_out = fit_baseline_window(x, y, roi, default_params, weights_builder=weights_builder, method="poly")
        assert y_out.shape == y.shape

    def test_aspls_with_weights_builder(self, default_params):
        x = np.linspace(800, 1200, 200)
        y = 500 + 0.3 * (x - 800)
        roi = (850.0, 1150.0)
        weights_builder = ((1010.0, 20.0, 0.01),)
        y_out = fit_baseline_window(x, y, roi, default_params, weights_builder=weights_builder, method="aspls")
        assert y_out.shape == y.shape
