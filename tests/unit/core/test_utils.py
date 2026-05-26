"""Tests for core.utils module."""

from unittest.mock import patch

import pytest
from scipy import stats

from sherloc_pipeline.core.utils import (
    format_trim_label,
    require_file,
    resolve_parallel_workers,
    resolve_trim_proportion,
)


class TestResolveParallelWorkers:
    """Tests for resolve_parallel_workers()."""

    def test_auto_mode_24_cores(self):
        """Auto mode (configured=0) with 24 cores → 12 workers."""
        with patch("os.cpu_count", return_value=24):
            assert resolve_parallel_workers(0, 100) == 12

    def test_auto_mode_8_cores(self):
        """Auto mode with 8 cores → 4 workers."""
        with patch("os.cpu_count", return_value=8):
            assert resolve_parallel_workers(0, 100) == 4

    def test_auto_mode_2_cores(self):
        """Auto mode with 2 cores → 1 worker."""
        with patch("os.cpu_count", return_value=2):
            assert resolve_parallel_workers(0, 100) == 1

    def test_auto_mode_1_core(self):
        """Auto mode with 1 core → 1 worker (sequential)."""
        with patch("os.cpu_count", return_value=1):
            assert resolve_parallel_workers(0, 100) == 1

    def test_auto_mode_cpu_count_none(self):
        """Auto mode when os.cpu_count() returns None → 1 worker."""
        with patch("os.cpu_count", return_value=None):
            assert resolve_parallel_workers(0, 100) == 1

    def test_explicit_workers(self):
        """Explicit configured=6 → 6 workers."""
        assert resolve_parallel_workers(6, 100) == 6

    def test_sequential_mode(self):
        """configured=1 → sequential (1 worker)."""
        assert resolve_parallel_workers(1, 100) == 1

    def test_cap_to_n_items(self):
        """Workers capped to n_items to avoid idle processes."""
        with patch("os.cpu_count", return_value=24):
            assert resolve_parallel_workers(0, 5) == 5

    def test_cap_explicit_to_n_items(self):
        """Explicit 12 workers but only 3 items → 3."""
        assert resolve_parallel_workers(12, 3) == 3

    def test_negative_configured_treated_as_1(self):
        """Negative configured value → clamped to 1."""
        assert resolve_parallel_workers(-1, 100) == 1

    def test_zero_items_returns_zero(self):
        """Zero items → 0 workers (min caps)."""
        with patch("os.cpu_count", return_value=24):
            assert resolve_parallel_workers(0, 0) == 0

    def test_single_item(self):
        """Single item → 1 worker regardless of config."""
        with patch("os.cpu_count", return_value=24):
            assert resolve_parallel_workers(0, 1) == 1


class TestResolveTrimProportion:
    """Tests for resolve_trim_proportion()."""

    def test_100_points_unchanged(self):
        """100-pt scan: baseline 0.02 dominates, m=2 per tail."""
        result = resolve_trim_proportion(100, 0.02)
        assert result == 0.02
        assert int(result * 100) == 2

    def test_50_points_functionally_unchanged(self):
        """50-pt scan: m=1 per tail (same as baseline 0.02)."""
        result = resolve_trim_proportion(50, 0.02)
        # (1+eps)/50 is marginally above 0.02 due to epsilon, but m is the same
        assert int(result * 50) == 1

    def test_25_points_adjusted(self):
        """25-pt scan: adjusted upward so m=1 (was m=0 at baseline)."""
        result = resolve_trim_proportion(25, 0.02)
        assert result > 0.02
        assert int(result * 25) >= 1

    def test_10_points_adjusted(self):
        """10-pt scan: adjusted to ~0.10 so m=1."""
        result = resolve_trim_proportion(10, 0.02)
        assert result > 0.02
        assert int(result * 10) >= 1

    def test_3_points_median(self):
        """3-pt scan: proportion ~0.33, takes median."""
        result = resolve_trim_proportion(3, 0.02)
        assert result > 0.3
        assert int(result * 3) >= 1

    def test_2_points_guard(self):
        """n=2: too few to trim, returns baseline (m=0 → plain mean)."""
        assert resolve_trim_proportion(2, 0.02) == 0.02

    def test_1_point_guard(self):
        """n=1: returns baseline unchanged."""
        assert resolve_trim_proportion(1, 0.02) == 0.02

    def test_0_points_guard(self):
        """n=0: defensive — returns baseline unchanged."""
        assert resolve_trim_proportion(0, 0.02) == 0.02

    def test_negative_points_guard(self):
        """n<0: defensive — returns baseline unchanged."""
        assert resolve_trim_proportion(-1, 0.02) == 0.02

    def test_zero_baseline_no_adjustment(self):
        """baseline_pct=0.0 means no trimming — never adjusted upward."""
        assert resolve_trim_proportion(25, 0.0) == 0.0
        assert resolve_trim_proportion(10, 0.0) == 0.0
        assert resolve_trim_proportion(3, 0.0) == 0.0

    def test_custom_baseline_dominates(self):
        """Higher baseline (0.05) dominates for large n."""
        result = resolve_trim_proportion(100, 0.05)
        assert result == 0.05

    def test_scipy_integration_25pts(self):
        """Integration: scipy actually trims >=1 point from 25-pt data."""
        import numpy as np
        n = 25
        ptc = resolve_trim_proportion(n, 0.02)
        m = int(ptc * n)
        assert m >= 1
        # Single cosmic-ray spike at one end; m=1 trims it
        data = np.ones(n)
        data[-1] = 1000.0  # spike at high tail
        result = stats.trim_mean(data, ptc)
        # With m>=1, the spike is trimmed; plain mean would be ~41
        assert result < 2.0

    def test_ieee754_n49_regression(self):
        """Regression: n=49 was first failure with naive 1/n (int(1/49*49)=0)."""
        ptc = resolve_trim_proportion(49, 0.02)
        assert int(ptc * 49) >= 1

    def test_ieee754_all_n_3_to_1000(self):
        """Guarantee int(ptc * n) >= 1 for all n in [3, 1000]."""
        for n in range(3, 1001):
            ptc = resolve_trim_proportion(n, 0.02)
            assert int(ptc * n) >= 1, f"Failed for n={n}: ptc={ptc}, m={int(ptc * n)}"

    def test_large_n_exact_baseline(self):
        """For n >= 51, baseline 0.02 already gives m >= 1."""
        for n in [51, 100, 200, 500, 1000]:
            assert resolve_trim_proportion(n, 0.02) == 0.02


class TestFormatTrimLabel:
    """Tests for format_trim_label()."""

    def test_large_scan_uses_baseline(self):
        """100-point scan at 2% config → '2p_trim_mean'."""
        assert format_trim_label(100, 0.02) == "2p_trim_mean"

    def test_25_point_scan(self):
        """25-point line scan → dynamic 4% → '4p_trim_mean'."""
        assert format_trim_label(25, 0.02) == "4p_trim_mean"

    def test_33_point_scan(self):
        """33-point HDR scan → dynamic ~3% → '3p_trim_mean'."""
        assert format_trim_label(33, 0.02) == "3p_trim_mean"

    def test_34_point_scan(self):
        """34-point HDR scan → dynamic ~2.9% → '2.9p_trim_mean'."""
        assert format_trim_label(34, 0.02) == "2.9p_trim_mean"

    def test_small_scan_2_points(self):
        """2-point scan → baseline pct unchanged → '2p_trim_mean'."""
        assert format_trim_label(2, 0.02) == "2p_trim_mean"

    def test_custom_baseline_5pct(self):
        """Custom 5% baseline on large scan → '5p_trim_mean'."""
        assert format_trim_label(100, 0.05) == "5p_trim_mean"
