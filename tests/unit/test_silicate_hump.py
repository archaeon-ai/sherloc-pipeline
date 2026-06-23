"""Tests for detect_silicate_hump() in services/pipeline.py."""

import pytest

from sherloc_pipeline.services.pipeline import detect_silicate_hump


class TestDetectSilicateHump:
    """Tests for broad silicate hump post-hoc detection."""

    def test_broad_single_peak_detected(self):
        """Single peak with FWHM >= 70 in range -> detected."""
        peaks = [{"center_cm1": 1000.0, "fwhm_cm1": 85.0, "mineral_assignment": "1050"}]
        result = detect_silicate_hump(peaks)
        assert result is not None
        assert result["detected"] is True
        assert len(result["contributing_peaks"]) == 1
        assert result["mean_center_cm1"] == 1000.0
        assert result["mean_fwhm_cm1"] == 85.0
        assert "silicate" in result["interpretation"].lower()

    def test_two_peaks_wide_envelope_detected(self):
        """Two peaks spanning >= 80 cm-1 combined envelope -> detected."""
        peaks = [
            {"center_cm1": 970.0, "fwhm_cm1": 45.0},
            {"center_cm1": 1040.0, "fwhm_cm1": 50.0},
        ]
        # lo_edge = 970 - 22.5 = 947.5, hi_edge = 1040 + 25 = 1065
        # span = 1065 - 947.5 = 117.5 >= 80
        result = detect_silicate_hump(peaks)
        assert result is not None
        assert result["detected"] is True
        assert len(result["contributing_peaks"]) == 2

    def test_single_narrow_pyroxene_rejected(self):
        """Single narrow peak (FWHM < 40) -> not detected."""
        peaks = [{"center_cm1": 990.0, "fwhm_cm1": 30.0, "mineral_assignment": "pyroxene"}]
        result = detect_silicate_hump(peaks)
        assert result is None

    def test_peaks_outside_range_rejected(self):
        """Peaks outside 950-1060 range -> not detected."""
        peaks = [
            {"center_cm1": 850.0, "fwhm_cm1": 80.0},
            {"center_cm1": 1100.0, "fwhm_cm1": 85.0},
        ]
        result = detect_silicate_hump(peaks)
        assert result is None

    def test_empty_peak_list(self):
        """Empty peak list -> not detected."""
        result = detect_silicate_hump([])
        assert result is None

    def test_fwhm_exactly_at_threshold(self):
        """Peak exactly at FWHM = 70 threshold -> detected."""
        peaks = [{"center_cm1": 1005.0, "fwhm_cm1": 70.0}]
        result = detect_silicate_hump(peaks)
        assert result is not None
        assert result["detected"] is True

    def test_fwhm_just_below_single_threshold_no_envelope(self):
        """Single peak with FWHM = 69 (below 70) and no envelope -> not detected."""
        peaks = [{"center_cm1": 1005.0, "fwhm_cm1": 69.0}]
        result = detect_silicate_hump(peaks)
        assert result is None

    def test_missing_keys_graceful(self):
        """Peaks with missing center_cm1 or fwhm_cm1 are ignored."""
        peaks = [
            {"center_cm1": None, "fwhm_cm1": 80.0},
            {"fwhm_cm1": 80.0},
            {"center_cm1": 1000.0, "fwhm_cm1": None},
        ]
        result = detect_silicate_hump(peaks)
        assert result is None

    def test_multiple_broad_peaks(self):
        """Multiple broad peaks all contribute."""
        peaks = [
            {"center_cm1": 990.0, "fwhm_cm1": 75.0},
            {"center_cm1": 1020.0, "fwhm_cm1": 80.0},
        ]
        result = detect_silicate_hump(peaks)
        assert result is not None
        assert len(result["contributing_peaks"]) == 2
        assert result["mean_center_cm1"] == pytest.approx(1005.0)
        assert result["mean_fwhm_cm1"] == pytest.approx(77.5)
