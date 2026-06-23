"""
Unit tests for PPP (pulses per point) background scaling.

These tests verify the calculate_background_scale function which determines
how to scale background spectra based on the ratio of scan PPP to background PPP.

Background context:
- SHERLOC background spectra (Arm Stowed and Fused Silica) are captured at 900 PPP
- Individual scans may use 500 PPP (most detail scans) or 900 PPP (some line scans)
- When subtracting background, scale factor = scan_ppp / bg_ppp
- Users can override with explicit scale factor for manual tuning
"""

import pytest

from sherloc_pipeline.services.spectral import calculate_background_scale


class TestPPPScaling:
    """Tests for calculate_background_scale function."""

    def test_ppp_scaling_500ppp_scan(self):
        """500 PPP scan should scale 900 PPP background by 500/900."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
        assert abs(scale - 500 / 900) < 1e-9

    def test_ppp_scaling_900ppp_scan(self):
        """900 PPP scan should have scale factor of 1.0."""
        scale = calculate_background_scale(scan_ppp=900, bg_ppp=900)
        assert abs(scale - 1.0) < 1e-9

    def test_ppp_scaling_override(self):
        """Explicit override should take precedence over auto-calculation."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900, override=0.5)
        assert abs(scale - 0.5) < 1e-9

    def test_ppp_scaling_default_bg_ppp(self):
        """Default bg_ppp should be 900."""
        scale = calculate_background_scale(scan_ppp=500)
        expected = 500 / 900
        assert abs(scale - expected) < 1e-9

    def test_ppp_scaling_custom_bg_ppp(self):
        """Custom bg_ppp should be used in calculation."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=1000)
        expected = 500 / 1000
        assert abs(scale - expected) < 1e-9

    def test_ppp_scaling_override_ignores_ppp_values(self):
        """When override is provided, scan_ppp and bg_ppp are ignored."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900, override=0.75)
        assert abs(scale - 0.75) < 1e-9

    def test_ppp_scaling_zero_override(self):
        """Override of 0.0 should return 0.0 (no background subtraction)."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900, override=0.0)
        assert abs(scale - 0.0) < 1e-9

    def test_ppp_scaling_negative_override(self):
        """Negative override is allowed (though unusual - adds background)."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900, override=-0.5)
        assert abs(scale - (-0.5)) < 1e-9


class TestPPPScalingEdgeCases:
    """Edge case tests for calculate_background_scale."""

    def test_ppp_scaling_very_small_scan_ppp(self):
        """Very small scan PPP should produce very small scale."""
        scale = calculate_background_scale(scan_ppp=50, bg_ppp=900)
        expected = 50 / 900
        assert abs(scale - expected) < 1e-9
        assert scale < 0.1  # Sanity check

    def test_ppp_scaling_large_scan_ppp(self):
        """Scan PPP larger than background PPP is valid (over-subtraction)."""
        scale = calculate_background_scale(scan_ppp=1200, bg_ppp=900)
        expected = 1200 / 900
        assert abs(scale - expected) < 1e-9
        assert scale > 1.0  # Scale > 1 means background is amplified

    def test_ppp_scaling_equal_ppp_values(self):
        """Equal scan and background PPP should yield scale of 1.0."""
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=500)
        assert abs(scale - 1.0) < 1e-9

    def test_ppp_scaling_float_values(self):
        """Float PPP values should work correctly."""
        scale = calculate_background_scale(scan_ppp=500.5, bg_ppp=900.0)
        expected = 500.5 / 900.0
        assert abs(scale - expected) < 1e-9


class TestPPPScalingRealWorldScenarios:
    """Tests based on actual SHERLOC mission data scenarios."""

    def test_amherst_point_scenario(self):
        """Amherst Point sol 0921 uses 500 PPP."""
        # From manifest: sol_0921 detail_1 has ppp=500
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
        assert abs(scale - 500 / 900) < 1e-9
        # Should be approximately 0.556
        assert 0.55 < scale < 0.56

    def test_stigbreen_scenario(self):
        """Stigbreen sol 1634 uses 900 PPP."""
        # From manifest: sol_1634 line_1 has ppp=900
        scale = calculate_background_scale(scan_ppp=900, bg_ppp=900)
        assert abs(scale - 1.0) < 1e-9

    def test_lake_haiyaha_scenario(self):
        """Lake Haiyaha sol 0852 uses 500 PPP."""
        # From manifest: sol_0852 detail_1 has ppp=500
        scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
        assert abs(scale - 500 / 900) < 1e-9

    def test_manual_correction_for_over_subtraction(self):
        """User might reduce scale to correct for over-subtraction at ~800 cm⁻¹."""
        # When instrument background causes anomalous dip, user may override
        auto_scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
        manual_scale = calculate_background_scale(scan_ppp=500, bg_ppp=900, override=0.4)
        
        # Manual scale should be less than auto scale
        assert manual_scale < auto_scale
        assert abs(manual_scale - 0.4) < 1e-9


class TestPPPScalingWarnings:
    """Tests for sanity check warnings in calculate_background_scale (T5.7)."""

    def test_zero_ppp_returns_fallback(self, caplog):
        """Zero scan_ppp should return 1.0 and log warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            scale = calculate_background_scale(scan_ppp=0, bg_ppp=900)
        
        assert scale == 1.0
        assert "invalid" in caplog.text.lower() or "0" in caplog.text

    def test_negative_ppp_returns_fallback(self, caplog):
        """Negative scan_ppp should return 1.0 and log warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            scale = calculate_background_scale(scan_ppp=-100, bg_ppp=900)
        
        assert scale == 1.0

    def test_scale_within_bounds_no_warning(self, caplog):
        """Normal scale values should not trigger warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            scale = calculate_background_scale(scan_ppp=500, bg_ppp=900)
        
        # Should not warn for scale ~0.556 which is within [0.1, 5.0]
        assert "outside expected bounds" not in caplog.text
        assert 0.5 < scale < 0.6

    def test_scale_below_lower_bound_warns(self, caplog):
        """Very low scale should trigger warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            # 50 PPP scan -> scale = 50/900 ≈ 0.056, below 0.1
            scale = calculate_background_scale(scan_ppp=50, bg_ppp=900)
        
        assert scale < 0.1
        assert "outside expected bounds" in caplog.text

    def test_scale_above_upper_bound_warns(self, caplog):
        """Very high scale should trigger warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            # 5000 PPP scan -> scale = 5000/900 ≈ 5.56, above 5.0
            scale = calculate_background_scale(scan_ppp=5000, bg_ppp=900)
        
        assert scale > 5.0
        assert "outside expected bounds" in caplog.text

    def test_custom_scale_bounds(self, caplog):
        """Custom scale_bounds should be respected."""
        import logging
        with caplog.at_level(logging.WARNING):
            # Scale = 500/900 ≈ 0.556, which is outside [0.6, 0.7]
            scale = calculate_background_scale(
                scan_ppp=500, bg_ppp=900, scale_bounds=(0.6, 0.7)
            )
        
        assert "outside expected bounds" in caplog.text

