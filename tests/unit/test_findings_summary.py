"""Tests for summarize_findings() in services/pipeline.py."""

import pytest

from sherloc_pipeline.services.pipeline import summarize_findings


def _make_mineral_peak(point_idx, assignment, snr=15.0, center=1012.0, fwhm=35.0):
    return {
        "point_idx": point_idx,
        "mineral_assignment": assignment,
        "snr": snr,
        "center_cm1": center,
        "fwhm_cm1": fwhm,
        "fit_modality": "minerals",
    }


def _make_fluor_peak(point_idx, group):
    return {"point_idx": point_idx, "fluor_group": group}


def _make_organics_peak(point_idx, assignment="D_band", snr=5.0, center=1350.0):
    return {
        "point_idx": point_idx,
        "mineral_assignment": assignment,
        "snr": snr,
        "center_cm1": center,
        "fwhm_cm1": 120.0,
        "fit_modality": "organics",
    }


def _make_hydration_peak(point_idx, snr=8.0, center=3450.0):
    return {
        "point_idx": point_idx,
        "mineral_assignment": "OH_stretch",
        "snr": snr,
        "center_cm1": center,
        "fwhm_cm1": 150.0,
        "fit_modality": "hydration",
    }


class TestSummarizeFindings:
    """Tests for structured scientific summary generation."""

    def test_typical_scan(self):
        """Typical scan with multiple minerals and fluorescence."""
        raman = [
            _make_mineral_peak(i, "sulf1_v1", snr=20.0 + i)
            for i in range(10)
        ] + [
            _make_mineral_peak(i + 10, "olivine", snr=8.0)
            for i in range(3)
        ]
        fluor = [_make_fluor_peak(i, "group1a") for i in range(5)]
        result = summarize_findings(raman, fluor, n_total_points=100)

        assert len(result["minerals_detected"]) == 2
        sulf = [m for m in result["minerals_detected"] if m["assignment"] == "sulf1_v1"]
        assert len(sulf) == 1
        assert sulf[0]["n_points"] == 10
        assert result["co_occurrences"] is not None
        assert result["organics_detected"] is False
        assert result["hydration_detected"] is False

    def test_confidence_high(self):
        """SNR >= 10 AND >= 5 points -> high confidence."""
        raman = [_make_mineral_peak(i, "sulf1_v1", snr=15.0) for i in range(6)]
        result = summarize_findings(raman, [], n_total_points=100)
        sulf = [m for m in result["minerals_detected"] if m["assignment"] == "sulf1_v1"]
        assert sulf[0]["confidence"] == "high"

    def test_confidence_moderate_snr(self):
        """SNR >= 5 but < 10, < 3 points -> moderate."""
        raman = [_make_mineral_peak(0, "olivine", snr=7.0)]
        result = summarize_findings(raman, [], n_total_points=100)
        olv = [m for m in result["minerals_detected"] if m["assignment"] == "olivine"]
        assert olv[0]["confidence"] == "moderate"

    def test_confidence_moderate_points(self):
        """SNR < 5, >= 3 points -> moderate."""
        raman = [_make_mineral_peak(i, "lo-carb", snr=4.0) for i in range(4)]
        result = summarize_findings(raman, [], n_total_points=100)
        lc = [m for m in result["minerals_detected"] if m["assignment"] == "lo-carb"]
        assert lc[0]["confidence"] == "moderate"

    def test_confidence_low(self):
        """SNR < 5 and < 3 points -> low."""
        raman = [_make_mineral_peak(0, "hi-carb", snr=3.5)]
        result = summarize_findings(raman, [], n_total_points=100)
        hc = [m for m in result["minerals_detected"] if m["assignment"] == "hi-carb"]
        assert hc[0]["confidence"] == "low"

    def test_narrative_contains_dominant(self):
        """Narrative starts with dominant mineral."""
        raman = [_make_mineral_peak(i, "sulf1_v1", snr=20.0) for i in range(10)]
        result = summarize_findings(raman, [], n_total_points=100)
        assert "sulf1_v1" in result["narrative"]
        assert "10/100" in result["narrative"]

    def test_no_detections(self):
        """No detections -> appropriate empty result."""
        result = summarize_findings([], [], n_total_points=50)
        assert result["minerals_detected"] == []
        assert result["silicate_hump"] is None
        assert result["fluorescence_groups"] == []
        assert result["co_occurrences"] == []
        assert result["organics_detected"] is False
        assert result["hydration_detected"] is False
        assert "No mineral" in result["narrative"]

    def test_organics_detected_flag(self):
        """Organics peaks set organics_detected = True."""
        raman = [_make_organics_peak(0)]
        result = summarize_findings(raman, [], n_total_points=100)
        assert result["organics_detected"] is True
        assert "Organic" in result["narrative"]

    def test_hydration_detected_flag(self):
        """Hydration peaks set hydration_detected = True."""
        raman = [_make_hydration_peak(0)]
        result = summarize_findings(raman, [], n_total_points=100)
        assert result["hydration_detected"] is True
        assert "Hydration" in result["narrative"]

    def test_silicate_hump_included(self):
        """Silicate hump detected when broad peak present in minerals."""
        raman = [_make_mineral_peak(0, "1050", snr=12.0, center=1000.0, fwhm=85.0)]
        result = summarize_findings(raman, [], n_total_points=100)
        assert result["silicate_hump"] is not None
        assert result["silicate_hump"]["detected"] is True
        assert "silicate hump" in result["narrative"].lower()

    def test_co_occurrence_in_narrative(self):
        """Co-occurrence mention appears in narrative."""
        raman = [_make_mineral_peak(i, "sulf1_v1", snr=20.0) for i in range(5)]
        fluor = [_make_fluor_peak(i, "group1a") for i in range(5)]
        result = summarize_findings(raman, fluor, n_total_points=100)
        assert "Ca-sulfate" in result["narrative"]
        assert "confirmed" in result["narrative"]

    def test_fluorescence_groups_populated(self):
        """Fluorescence groups are listed with labels."""
        fluor = [
            _make_fluor_peak(0, "group1a"),
            _make_fluor_peak(1, "group1a"),
            _make_fluor_peak(2, "group3"),
        ]
        result = summarize_findings([], fluor, n_total_points=50)
        groups = {g["group"]: g for g in result["fluorescence_groups"]}
        assert "group1a" in groups
        assert groups["group1a"]["n_peaks"] == 2
        assert "group3" in groups
        assert groups["group3"]["n_peaks"] == 1
