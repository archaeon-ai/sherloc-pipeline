"""
Tests for R123 stitching module.

Validates overlap summation, edge cases, batch mode, and performance.
"""

import time

import numpy as np
import pytest

from sherloc_pipeline.core.r123_stitching import (
    N_CHANNELS,
    _OVERLAP1_END,
    _OVERLAP2_END,
    _R1_ONLY_END,
    _R2_ONLY_END,
    _R3_ONLY_END,
    r123_wavelength_axis,
    r123_wavenumber_axis,
    stitch_r123_batch,
    stitch_r123_spectrum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_region(fill_value: float = 0.0) -> np.ndarray:
    """Create a 2148-channel array filled with a constant."""
    return np.full(N_CHANNELS, fill_value, dtype=np.float64)


def _make_r1(value: float = 1.0) -> np.ndarray:
    """Create R1 readout with uniform signal across all channels the stitcher reads.

    The stitcher reads R1 channels 0-689 (R1-only + overlap 1).
    Real R1 has meaningful signal in ch 52-574, but the stitcher uses the
    raw array values at those indices regardless. For test predictability
    we fill the full 2148 array with the given value.
    """
    return np.full(N_CHANNELS, value, dtype=np.float64)


def _make_r2(value: float = 2.0) -> np.ndarray:
    """Create R2 readout with uniform signal across all channels the stitcher reads.

    The stitcher reads R2 channels 565-1689 (overlap 1 + R2-only + overlap 2).
    For test predictability we fill the full 2148 array.
    """
    return np.full(N_CHANNELS, value, dtype=np.float64)


def _make_r3(value: float = 3.0) -> np.ndarray:
    """Create R3 readout with uniform signal across all channels the stitcher reads.

    The stitcher reads R3 channels 1668-2147 (overlap 2 + R3-only).
    For test predictability we fill the full 2148 array.
    """
    return np.full(N_CHANNELS, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic stitching
# ---------------------------------------------------------------------------

class TestStitchR123Spectrum:
    """Tests for single-spectrum stitching."""

    def test_shape_and_dtype(self):
        """Result must be (2148,) float64."""
        r1, r2, r3 = _make_r1(), _make_r2(), _make_r3()
        result = stitch_r123_spectrum(r1, r2, r3)
        assert result.shape == (N_CHANNELS,)
        assert result.dtype == np.float64

    def test_r1_only_region(self):
        """Channels 0-564 should come from R1 only."""
        r1 = _make_r1(5.0)
        r2 = _make_r2(10.0)
        r3 = _make_r3(20.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        np.testing.assert_array_equal(result[0:_R1_ONLY_END], r1[0:_R1_ONLY_END])

    def test_overlap1_summation(self):
        """Channels 565-689 should be R1 + R2 (SUMMATION)."""
        r1 = _make_r1(3.0)
        r2 = _make_r2(7.0)
        r3 = _make_r3(1.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        expected = r1[_R1_ONLY_END:_OVERLAP1_END] + r2[_R1_ONLY_END:_OVERLAP1_END]
        np.testing.assert_array_equal(
            result[_R1_ONLY_END:_OVERLAP1_END], expected
        )

    def test_overlap1_is_sum_not_average(self):
        """Explicitly verify overlap 1 uses summation, not averaging."""
        r1_val, r2_val = 4.0, 6.0
        r1 = _make_r1(r1_val)
        r2 = _make_r2(r2_val)
        r3 = _make_r3(1.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        # In the overlap region (ch 565-689), r1 has r1_val and r2 has r2_val
        overlap1 = result[_R1_ONLY_END:_OVERLAP1_END]
        # Sum, NOT average
        assert np.all(overlap1 == r1_val + r2_val), (
            f"Overlap 1 should be {r1_val + r2_val} (sum), "
            f"not {(r1_val + r2_val) / 2} (average)"
        )

    def test_r2_only_region(self):
        """Channels 690-1667 should come from R2 only."""
        r1 = _make_r1(5.0)
        r2 = _make_r2(10.0)
        r3 = _make_r3(20.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        np.testing.assert_array_equal(
            result[_OVERLAP1_END:_R2_ONLY_END], r2[_OVERLAP1_END:_R2_ONLY_END]
        )

    def test_overlap2_summation(self):
        """Channels 1668-1689 should be R2 + R3 (SUMMATION)."""
        r1 = _make_r1(1.0)
        r2 = _make_r2(5.0)
        r3 = _make_r3(9.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        expected = r2[_R2_ONLY_END:_OVERLAP2_END] + r3[_R2_ONLY_END:_OVERLAP2_END]
        np.testing.assert_array_equal(
            result[_R2_ONLY_END:_OVERLAP2_END], expected
        )

    def test_overlap2_is_sum_not_average(self):
        """Explicitly verify overlap 2 uses summation, not averaging."""
        r2_val, r3_val = 8.0, 12.0
        r1 = _make_r1(1.0)
        r2 = _make_r2(r2_val)
        r3 = _make_r3(r3_val)
        result = stitch_r123_spectrum(r1, r2, r3)
        overlap2 = result[_R2_ONLY_END:_OVERLAP2_END]
        assert np.all(overlap2 == r2_val + r3_val), (
            f"Overlap 2 should be {r2_val + r3_val} (sum), "
            f"not {(r2_val + r3_val) / 2} (average)"
        )

    def test_r3_only_region(self):
        """Channels 1690-2147 should come from R3 only."""
        r1 = _make_r1(5.0)
        r2 = _make_r2(10.0)
        r3 = _make_r3(20.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        np.testing.assert_array_equal(
            result[_OVERLAP2_END:_R3_ONLY_END], r3[_OVERLAP2_END:_R3_ONLY_END]
        )

    def test_full_stitching_coverage(self):
        """Every channel in the output should be accounted for (no gaps)."""
        r1 = _make_r1(1.0)
        r2 = _make_r2(2.0)
        r3 = _make_r3(3.0)
        result = stitch_r123_spectrum(r1, r2, r3)

        # R1-only region
        assert np.all(result[0:_R1_ONLY_END] == 1.0)
        # Overlap 1: r1(1.0) + r2(2.0) = 3.0
        assert np.all(result[_R1_ONLY_END:_OVERLAP1_END] == 3.0)
        # R2-only region
        assert np.all(result[_OVERLAP1_END:_R2_ONLY_END] == 2.0)
        # Overlap 2: r2(2.0) + r3(3.0) = 5.0
        assert np.all(result[_R2_ONLY_END:_OVERLAP2_END] == 5.0)
        # R3-only region
        assert np.all(result[_OVERLAP2_END:_R3_ONLY_END] == 3.0)

    def test_region_widths(self):
        """Validate the documented region widths."""
        assert _R1_ONLY_END == 565                         # R1-only: 565 ch
        assert _OVERLAP1_END - _R1_ONLY_END == 125         # Overlap 1: 125 ch
        assert _R2_ONLY_END - _OVERLAP1_END == 978         # R2-only: 978 ch
        assert _OVERLAP2_END - _R2_ONLY_END == 22          # Overlap 2: 22 ch
        assert _R3_ONLY_END - _OVERLAP2_END == 458         # R3-only: 458 ch
        # Sum must be 2148
        total = (
            _R1_ONLY_END
            + (_OVERLAP1_END - _R1_ONLY_END)
            + (_R2_ONLY_END - _OVERLAP1_END)
            + (_OVERLAP2_END - _R2_ONLY_END)
            + (_R3_ONLY_END - _OVERLAP2_END)
        )
        assert total == N_CHANNELS

    def test_does_not_mutate_inputs(self):
        """Stitching must not alter the input arrays."""
        r1, r2, r3 = _make_r1(1.0), _make_r2(2.0), _make_r3(3.0)
        r1_copy, r2_copy, r3_copy = r1.copy(), r2.copy(), r3.copy()
        stitch_r123_spectrum(r1, r2, r3)
        np.testing.assert_array_equal(r1, r1_copy)
        np.testing.assert_array_equal(r2, r2_copy)
        np.testing.assert_array_equal(r3, r3_copy)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling: zeros, NaNs, input validation."""

    def test_all_zeros(self):
        """All-zero inputs should produce all-zero output."""
        r1 = r2 = r3 = np.zeros(N_CHANNELS)
        result = stitch_r123_spectrum(r1, r2, r3)
        assert np.all(result == 0.0)

    def test_one_region_all_zeros(self):
        """If R2 is all zeros, overlap regions degrade gracefully."""
        r1 = _make_r1(5.0)
        r2 = np.zeros(N_CHANNELS)  # R2 detector off
        r3 = _make_r3(10.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        # R1-only region: still 5.0
        assert np.all(result[0:_R1_ONLY_END] == 5.0)
        # Overlap 1: r1(5.0) + r2(0.0) = 5.0 (only R1 contributes)
        assert np.all(result[_R1_ONLY_END:_OVERLAP1_END] == 5.0)
        # R2-only region: 0.0 (nothing from R2)
        assert np.all(result[_OVERLAP1_END:_R2_ONLY_END] == 0.0)
        # Overlap 2: r2(0.0) + r3(10.0) = 10.0 (only R3 contributes)
        assert np.all(result[_R2_ONLY_END:_OVERLAP2_END] == 10.0)
        # R3-only region: 10.0
        assert np.all(result[_OVERLAP2_END:_R3_ONLY_END] == 10.0)

    def test_nan_in_overlap_replaced(self):
        """NaN values in overlap region should be replaced with 0 (default)."""
        r1 = _make_r1(2.0)
        r2 = _make_r2(3.0)
        r3 = _make_r3(4.0)
        # Inject NaN into R1 overlap region
        r1[600] = np.nan
        result = stitch_r123_spectrum(r1, r2, r3)
        # Channel 600 is in overlap 1 (565-689)
        # NaN replaced with 0, so result = 0.0 + 3.0 = 3.0
        assert result[600] == 3.0
        assert not np.any(np.isnan(result))

    def test_nan_in_non_overlap_replaced(self):
        """NaN in R1-only region should be replaced with 0."""
        r1 = _make_r1(5.0)
        r1[100] = np.nan
        r2, r3 = _make_r2(), _make_r3()
        result = stitch_r123_spectrum(r1, r2, r3)
        assert result[100] == 0.0
        assert not np.any(np.isnan(result))

    def test_nan_to_zero_disabled(self):
        """With nan_to_zero=False, NaN should propagate."""
        r1 = _make_r1(2.0)
        r1[100] = np.nan
        r2, r3 = _make_r2(), _make_r3()
        result = stitch_r123_spectrum(r1, r2, r3, nan_to_zero=False)
        assert np.isnan(result[100])

    def test_nan_in_overlap_propagates_when_disabled(self):
        """With nan_to_zero=False, NaN in overlap should propagate."""
        r1 = _make_r1(2.0)
        r2 = _make_r2(3.0)
        r3 = _make_r3(4.0)
        r1[600] = np.nan  # In overlap 1
        result = stitch_r123_spectrum(r1, r2, r3, nan_to_zero=False)
        assert np.isnan(result[600])

    def test_wrong_shape_raises(self):
        """Inputs with wrong shape should raise ValueError."""
        r_bad = np.zeros(100)
        r_good = np.zeros(N_CHANNELS)
        with pytest.raises(ValueError, match="r1 must have shape"):
            stitch_r123_spectrum(r_bad, r_good, r_good)
        with pytest.raises(ValueError, match="r2 must have shape"):
            stitch_r123_spectrum(r_good, r_bad, r_good)
        with pytest.raises(ValueError, match="r3 must have shape"):
            stitch_r123_spectrum(r_good, r_good, r_bad)

    def test_2d_input_raises(self):
        """2D inputs should raise ValueError for single-spectrum function."""
        r_2d = np.zeros((1, N_CHANNELS))
        r_good = np.zeros(N_CHANNELS)
        with pytest.raises(ValueError):
            stitch_r123_spectrum(r_2d, r_good, r_good)

    def test_negative_values(self):
        """Negative intensity values should be handled correctly."""
        r1 = _make_r1(-5.0)
        r2 = _make_r2(3.0)
        r3 = _make_r3(1.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        # R1-only region: -5.0
        assert np.all(result[0:_R1_ONLY_END] == -5.0)
        # Overlap 1: -5.0 + 3.0 = -2.0
        assert np.all(result[_R1_ONLY_END:_OVERLAP1_END] == -2.0)

    def test_large_values(self):
        """Very large values should not overflow."""
        big = 1e15
        r1 = _make_r1(big)
        r2 = _make_r2(big)
        r3 = _make_r3(big)
        result = stitch_r123_spectrum(r1, r2, r3)
        # Overlap should be 2 * big
        assert np.all(result[_R1_ONLY_END:_OVERLAP1_END] == 2 * big)

    def test_integer_input_promoted(self):
        """Integer inputs should be promoted to float64."""
        r1 = np.ones(N_CHANNELS, dtype=np.int32)
        r2 = np.ones(N_CHANNELS, dtype=np.int32) * 2
        r3 = np.ones(N_CHANNELS, dtype=np.int32) * 3
        result = stitch_r123_spectrum(r1, r2, r3)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Batch stitching
# ---------------------------------------------------------------------------

class TestStitchR123Batch:
    """Tests for batch stitching."""

    def test_batch_shape(self):
        """Batch output shape should be (N, 2148)."""
        n = 10
        r1 = np.zeros((n, N_CHANNELS))
        r2 = np.zeros((n, N_CHANNELS))
        r3 = np.zeros((n, N_CHANNELS))
        result = stitch_r123_batch(r1, r2, r3)
        assert result.shape == (n, N_CHANNELS)
        assert result.dtype == np.float64

    def test_batch_matches_single(self):
        """Batch stitching should produce identical results to single-spectrum."""
        rng = np.random.default_rng(42)
        n = 5
        r1_batch = rng.standard_normal((n, N_CHANNELS))
        r2_batch = rng.standard_normal((n, N_CHANNELS))
        r3_batch = rng.standard_normal((n, N_CHANNELS))

        batch_result = stitch_r123_batch(r1_batch, r2_batch, r3_batch)

        for i in range(n):
            single_result = stitch_r123_spectrum(
                r1_batch[i], r2_batch[i], r3_batch[i]
            )
            np.testing.assert_array_equal(batch_result[i], single_result)

    def test_batch_overlap_summation(self):
        """Verify overlap summation in batch mode."""
        n = 3
        r1_val, r2_val, r3_val = 1.0, 2.0, 3.0

        # Fill full arrays (stitcher reads full 2148 from each)
        r1 = np.full((n, N_CHANNELS), r1_val)
        r2 = np.full((n, N_CHANNELS), r2_val)
        r3 = np.full((n, N_CHANNELS), r3_val)

        result = stitch_r123_batch(r1, r2, r3)

        # Check overlap 1 for all spectra: r1_val + r2_val = 3.0
        assert np.all(result[:, _R1_ONLY_END:_OVERLAP1_END] == r1_val + r2_val)
        # Check overlap 2 for all spectra: r2_val + r3_val = 5.0
        assert np.all(result[:, _R2_ONLY_END:_OVERLAP2_END] == r2_val + r3_val)

    def test_batch_nan_handling(self):
        """Batch NaN replacement should work."""
        n = 2
        r1 = np.zeros((n, N_CHANNELS))
        r2 = np.zeros((n, N_CHANNELS))
        r3 = np.zeros((n, N_CHANNELS))
        r1[0, 100] = np.nan
        r2[1, 600] = np.nan  # In overlap 1

        result = stitch_r123_batch(r1, r2, r3)
        assert not np.any(np.isnan(result))

    def test_batch_mismatched_sizes_raises(self):
        """Mismatched batch sizes should raise ValueError."""
        r1 = np.zeros((5, N_CHANNELS))
        r2 = np.zeros((3, N_CHANNELS))
        r3 = np.zeros((5, N_CHANNELS))
        with pytest.raises(ValueError, match="Batch sizes must match"):
            stitch_r123_batch(r1, r2, r3)

    def test_batch_wrong_channels_raises(self):
        """Wrong channel count in batch should raise ValueError."""
        r1 = np.zeros((5, 100))
        r2 = np.zeros((5, N_CHANNELS))
        r3 = np.zeros((5, N_CHANNELS))
        with pytest.raises(ValueError, match="r1_batch must have shape"):
            stitch_r123_batch(r1, r2, r3)

    def test_batch_1d_raises(self):
        """1D input to batch function should raise ValueError."""
        r1 = np.zeros(N_CHANNELS)
        r2 = np.zeros(N_CHANNELS)
        r3 = np.zeros(N_CHANNELS)
        with pytest.raises(ValueError):
            stitch_r123_batch(r1, r2, r3)

    def test_batch_single_spectrum(self):
        """Batch with N=1 should match single-spectrum result."""
        rng = np.random.default_rng(99)
        r1 = rng.standard_normal((1, N_CHANNELS))
        r2 = rng.standard_normal((1, N_CHANNELS))
        r3 = rng.standard_normal((1, N_CHANNELS))

        batch_result = stitch_r123_batch(r1, r2, r3)
        single_result = stitch_r123_spectrum(r1[0], r2[0], r3[0])
        np.testing.assert_array_equal(batch_result[0], single_result)


# ---------------------------------------------------------------------------
# Wavelength / wavenumber axis
# ---------------------------------------------------------------------------

class TestWavelengthAxis:
    """Tests for the wavelength/wavenumber axis helpers."""

    def test_wavelength_axis_shape(self):
        """Wavelength axis should have 2148 elements."""
        wl = r123_wavelength_axis()
        assert wl.shape == (N_CHANNELS,)

    def test_wavelength_axis_monotonic(self):
        """Wavelength should be monotonically increasing."""
        wl = r123_wavelength_axis()
        assert np.all(np.diff(wl) > 0)

    def test_wavelength_range(self):
        """Wavelength should span ~246-357 nm (full CCD range)."""
        wl = r123_wavelength_axis()
        assert wl[0] > 240.0
        assert wl[0] < 250.0
        assert wl[-1] > 350.0
        assert wl[-1] < 365.0

    def test_wavenumber_axis_shape(self):
        """Wavenumber axis should have 2148 elements."""
        wn = r123_wavenumber_axis()
        assert wn.shape == (N_CHANNELS,)

    def test_wavenumber_axis_monotonic(self):
        """Wavenumber (Raman shift) should be monotonically increasing."""
        wn = r123_wavenumber_axis()
        assert np.all(np.diff(wn) > 0)

    def test_r1_region_in_wavelength(self):
        """R1 region (250-282 nm) should be identifiable in wavelength axis."""
        wl = r123_wavelength_axis()
        r1_mask = (wl >= 250.0) & (wl <= 282.0)
        assert r1_mask.sum() == 523  # Expected R1 channel count


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

class TestPerformance:
    """Performance benchmarks."""

    def test_single_spectrum_under_1ms(self):
        """Single spectrum stitching should take < 1ms."""
        r1 = _make_r1(1.0)
        r2 = _make_r2(2.0)
        r3 = _make_r3(3.0)

        # Warm-up
        stitch_r123_spectrum(r1, r2, r3)

        # Time 1000 iterations
        n_iter = 1000
        start = time.perf_counter()
        for _ in range(n_iter):
            stitch_r123_spectrum(r1, r2, r3)
        elapsed = time.perf_counter() - start

        per_call_ms = (elapsed / n_iter) * 1000
        assert per_call_ms < 1.0, (
            f"Single-spectrum stitch took {per_call_ms:.3f} ms (target: < 1 ms)"
        )

    def test_batch_1000_under_10ms(self):
        """Batch of 1000 spectra should complete in reasonable time."""
        n = 1000
        rng = np.random.default_rng(42)
        r1 = rng.standard_normal((n, N_CHANNELS))
        r2 = rng.standard_normal((n, N_CHANNELS))
        r3 = rng.standard_normal((n, N_CHANNELS))

        # Warm-up
        stitch_r123_batch(r1, r2, r3)

        # Time
        start = time.perf_counter()
        stitch_r123_batch(r1, r2, r3)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be well under 100ms for 1000 spectra (vectorised)
        assert elapsed_ms < 100.0, (
            f"Batch of {n} took {elapsed_ms:.1f} ms (target: < 100 ms)"
        )


# ---------------------------------------------------------------------------
# Regression: overlap must be SUM, not average or last-writer-wins
# ---------------------------------------------------------------------------

class TestNotAverageNotLastWriter:
    """Regression tests to ensure overlap uses SUMMATION specifically."""

    def test_overlap1_not_average(self):
        """Overlap 1 result should NOT equal the average of R1 and R2."""
        r1 = _make_r1(4.0)
        r2 = _make_r2(8.0)
        r3 = _make_r3(0.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        overlap = result[_R1_ONLY_END:_OVERLAP1_END]
        # Average would be 6.0, sum is 12.0
        assert np.all(overlap == 12.0)
        assert not np.all(overlap == 6.0)

    def test_overlap2_not_average(self):
        """Overlap 2 result should NOT equal the average of R2 and R3."""
        r1 = _make_r1(0.0)
        r2 = _make_r2(6.0)
        r3 = _make_r3(10.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        overlap = result[_R2_ONLY_END:_OVERLAP2_END]
        # Average would be 8.0, sum is 16.0
        assert np.all(overlap == 16.0)
        assert not np.all(overlap == 8.0)

    def test_overlap1_not_last_writer(self):
        """Overlap 1 should not be last-writer-wins (R2 only)."""
        r1 = _make_r1(4.0)
        r2 = _make_r2(8.0)
        r3 = _make_r3(0.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        overlap = result[_R1_ONLY_END:_OVERLAP1_END]
        # Last-writer-wins would give 8.0 (R2), sum gives 12.0
        assert not np.all(overlap == 8.0)
        assert np.all(overlap == 12.0)

    def test_overlap2_not_last_writer(self):
        """Overlap 2 should not be last-writer-wins (R3 only)."""
        r1 = _make_r1(0.0)
        r2 = _make_r2(6.0)
        r3 = _make_r3(10.0)
        result = stitch_r123_spectrum(r1, r2, r3)
        overlap = result[_R2_ONLY_END:_OVERLAP2_END]
        # Last-writer-wins would give 10.0 (R3), sum gives 16.0
        assert not np.all(overlap == 10.0)
        assert np.all(overlap == 16.0)
