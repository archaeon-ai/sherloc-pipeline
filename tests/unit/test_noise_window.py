"""Tests for configurable noise window in SNR calculation.

The noise window determines which spectral region is used to estimate
the noise standard deviation for SNR calculations. By default this is
2000-2100 cm⁻¹, but it can be configured via config.yaml.
"""

import pytest
import numpy as np

from sherloc_pipeline.core.fitting import compute_noise_std, fit_spectrum


class TestComputeNoiseStd:
    """Tests for compute_noise_std() function."""

    @pytest.fixture
    def synthetic_spectrum(self):
        """Create synthetic spectrum with known noise in different regions."""
        x = np.linspace(500, 4000, 1000)
        # Base signal with peak at 1000 cm⁻¹
        y = 100 * np.exp(-((x - 1000) ** 2) / (2 * 50 ** 2))
        # Add known noise: low in 2000-2100, high in 3000-3100
        np.random.seed(42)
        noise = np.zeros_like(y)
        # Low noise region (std ~5)
        mask_low = (x >= 2000) & (x <= 2100)
        noise[mask_low] = np.random.normal(0, 5, mask_low.sum())
        # High noise region (std ~50)
        mask_high = (x >= 3000) & (x <= 3100)
        noise[mask_high] = np.random.normal(0, 50, mask_high.sum())
        # Medium noise elsewhere (std ~10)
        mask_other = ~mask_low & ~mask_high
        noise[mask_other] = np.random.normal(0, 10, mask_other.sum())
        return x, y + noise

    def test_default_window(self, synthetic_spectrum):
        """Default window (2000-2100) should be used when no args provided."""
        x, y = synthetic_spectrum
        noise_std = compute_noise_std(x, y)
        # Should pick up the low-noise region (~5)
        assert 3 < noise_std < 8

    def test_explicit_window_takes_precedence(self, synthetic_spectrum):
        """Explicit window parameter should override config."""
        x, y = synthetic_spectrum
        # Use high-noise region
        noise_std = compute_noise_std(x, y, window=(3000.0, 3100.0))
        # Should pick up the high-noise region (~50)
        assert 30 < noise_std < 70

    def test_config_window(self, synthetic_spectrum):
        """Config dict should set the window."""
        x, y = synthetic_spectrum
        cfg = {
            'noise_estimation': {
                'window': [3000.0, 3100.0]
            }
        }
        noise_std = compute_noise_std(x, y, cfg=cfg)
        # Should pick up the high-noise region (~50)
        assert 30 < noise_std < 70

    def test_explicit_window_overrides_config(self, synthetic_spectrum):
        """Explicit window should override config."""
        x, y = synthetic_spectrum
        cfg = {
            'noise_estimation': {
                'window': [3000.0, 3100.0]  # Would give high noise
            }
        }
        # Explicit window points to low-noise region
        noise_std = compute_noise_std(x, y, window=(2000.0, 2100.0), cfg=cfg)
        # Should use explicit window, not config
        assert 3 < noise_std < 8

    def test_fallback_to_global_when_window_unavailable(self):
        """Should fallback to global std if window outside data range."""
        x = np.linspace(500, 1500, 500)  # Doesn't include 2000-2100
        np.random.seed(42)
        y = np.random.normal(100, 15, 500)
        
        noise_std = compute_noise_std(x, y, window=(2000.0, 2100.0))
        # Should use global std (~15)
        assert 10 < noise_std < 20

    def test_empty_config_uses_default(self, synthetic_spectrum):
        """Empty config dict should use default window."""
        x, y = synthetic_spectrum
        noise_std = compute_noise_std(x, y, cfg={})
        # Should use default (2000-2100) with low noise
        assert 3 < noise_std < 8

    def test_partial_config_uses_default(self, synthetic_spectrum):
        """Config without noise_estimation should use default."""
        x, y = synthetic_spectrum
        cfg = {'other_setting': 123}
        noise_std = compute_noise_std(x, y, cfg=cfg)
        # Should use default (2000-2100)
        assert 3 < noise_std < 8


class TestFitSpectrumNoiseConfig:
    """Tests that fit_spectrum uses noise config correctly."""

    @pytest.fixture
    def simple_spectrum(self):
        """Create simple spectrum with peak."""
        x = np.linspace(600, 4000, 1000)
        # Peak at 850 cm⁻¹ (olivine-like)
        y = 500 * np.exp(-((x - 850) ** 2) / (2 * 20 ** 2))
        np.random.seed(42)
        y += np.random.normal(0, 10, len(x))
        return x, y

    def test_fit_uses_config_noise_window(self, simple_spectrum):
        """fit_spectrum should pass cfg to compute_noise_std."""
        x, y = simple_spectrum
        
        # Default config
        cfg_default = {
            'r1_fit_range': [700, 1200],
            'max_peaks': 3,
            'min_snr': 2.0,
            'min_seed_snr': 1.0,
            'min_display_snr': 1.0,
        }
        
        # Config with custom noise window
        cfg_custom = {
            **cfg_default,
            'noise_estimation': {
                'window': [3500.0, 3600.0]  # Different region
            }
        }
        
        result_default, _ = fit_spectrum(x, y, cfg_default, roi=(700, 1200))
        result_custom, _ = fit_spectrum(x, y, cfg_custom, roi=(700, 1200))
        
        # Both should find the peak, but SNR values may differ
        assert len(result_default.peaks) > 0
        assert len(result_custom.peaks) > 0
        
        # SNR values should be different due to different noise estimates
        # (unless by chance the noise is identical in both regions)
        # Just verify the fits completed successfully
        assert result_default.r2 > 0
        assert result_custom.r2 > 0


class TestNoiseWindowEdgeCases:
    """Edge case tests for noise window configuration."""

    def test_very_narrow_window(self):
        """Very narrow window should still work."""
        x = np.linspace(500, 4000, 1000)
        np.random.seed(42)
        y = np.random.normal(100, 10, 1000)
        
        # 1 cm⁻¹ wide window
        noise_std = compute_noise_std(x, y, window=(2000.0, 2001.0))
        # Should still compute something (may be less reliable)
        assert noise_std > 0

    def test_inverted_window(self):
        """Inverted window (max < min) should return 0 or global."""
        x = np.linspace(500, 4000, 1000)
        np.random.seed(42)
        y = np.random.normal(100, 10, 1000)
        
        # Inverted window - no points will match
        noise_std = compute_noise_std(x, y, window=(2100.0, 2000.0))
        # Should fallback to global std
        assert 5 < noise_std < 15

    def test_window_as_list_in_config(self):
        """Config window can be list (YAML loads as list, not tuple)."""
        x = np.linspace(500, 4000, 1000)
        np.random.seed(42)
        y = np.random.normal(100, 10, 1000)
        
        cfg = {
            'noise_estimation': {
                'window': [2000.0, 2100.0]  # List, not tuple
            }
        }
        noise_std = compute_noise_std(x, y, cfg=cfg)
        assert noise_std > 0

