"""Unit tests for fluorescence feature detection (core/fluor_detection.py).

Covers: feature scanning, candidate model enumeration, edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from sherloc_pipeline.core.fitting import fwhm_to_sigma
from sherloc_pipeline.core.fluor_detection import (
    CandidateModel,
    FeatureDetection,
    FeatureScanResult,
    enumerate_candidate_models,
    scan_fluorescence_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wavelength(start: float = 260.0, stop: float = 360.0, n: int = 500):
    return np.linspace(start, stop, n)


def _synth_spectrum(
    wavelength: np.ndarray,
    peaks: list[tuple[float, float, float]],
    noise_std: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros_like(wavelength, dtype=float)
    for center, amp, fwhm in peaks:
        sigma = fwhm_to_sigma(fwhm)
        y += amp * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    y += rng.normal(0.0, noise_std, size=y.size)
    return y


# ---------------------------------------------------------------------------
# Feature scan tests
# ---------------------------------------------------------------------------


class TestScanFluorescenceFeatures:

    def test_group3_detection(self):
        """Single peak at 280nm detected as group3."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(280.0, 3000.0, 15.0)], noise_std=10.0)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert result.group3_detected
        assert not result.group1_detected
        assert not result.group2_detected
        assert any(d.group == "group3" for d in result.detections)

    def test_group2_detection(self):
        """Single peak at 340nm detected as group2."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(340.0, 4000.0, 20.0)], noise_std=10.0)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert result.group2_detected
        assert not result.group3_detected
        g2 = [d for d in result.detections if d.group == "group2"]
        assert len(g2) == 1
        assert abs(g2[0].center_nm - 340.0) < 3.0

    def test_group1_doublet_detection(self):
        """Ce3+ doublet at 304+325nm detected as group1 (both 1a and 1b)."""
        wl = _make_wavelength(n=800)
        intensity = _synth_spectrum(
            wl, [(304.0, 5000.0, 10.0), (325.0, 3000.0, 10.0)], noise_std=10.0
        )
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert result.group1_detected
        assert result.group1a_detected
        assert result.group1b_detected

    def test_group1a_implies_doublet(self):
        """Only group1a detected → group1_detected is True (physical doublet implied)."""
        wl = _make_wavelength(n=800)
        intensity = _synth_spectrum(wl, [(304.0, 5000.0, 10.0)], noise_std=10.0)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        # Ce3+ 5d→2F5/2 implies the 5d→2F7/2 partner exists
        assert result.group1_detected
        assert result.group1a_detected
        assert not result.group1b_detected  # not independently detected, but implied

    def test_no_features_in_noise(self):
        """Pure noise → no features detected."""
        wl = _make_wavelength()
        rng = np.random.default_rng(42)
        intensity = rng.normal(0.0, 10.0, size=wl.size)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert not result.group3_detected
        assert not result.group1_detected
        assert not result.group2_detected
        assert len(result.detections) == 0

    def test_all_groups_present(self):
        """Three groups simultaneously detected."""
        wl = _make_wavelength(n=1000)
        intensity = _synth_spectrum(
            wl,
            [
                (280.0, 2000.0, 15.0),  # group3
                (304.0, 4000.0, 10.0),  # group1a
                (325.0, 3000.0, 10.0),  # group1b
                (340.0, 3500.0, 20.0),  # group2
            ],
            noise_std=10.0,
        )
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert result.group3_detected
        assert result.group1_detected
        assert result.group2_detected

    def test_low_prominence_peak_not_detected(self):
        """Peak with amplitude below prominence threshold is not detected."""
        wl = _make_wavelength()
        # amplitude = 20 with noise_std=10 → prominence ~20, threshold = 3*10 = 30
        intensity = _synth_spectrum(wl, [(340.0, 20.0, 15.0)], noise_std=10.0)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        assert not result.group2_detected

    def test_detection_center_estimate(self):
        """Detected center is close to true peak center."""
        wl = _make_wavelength(n=800)
        true_center = 340.0
        intensity = _synth_spectrum(wl, [(true_center, 5000.0, 18.0)], noise_std=10.0)
        result = scan_fluorescence_features(wl, intensity, noise_std=10.0)

        g2 = [d for d in result.detections if d.group == "group2"]
        assert len(g2) == 1
        assert abs(g2[0].center_nm - true_center) < 2.0


# ---------------------------------------------------------------------------
# Candidate model enumeration tests
# ---------------------------------------------------------------------------


class TestEnumerateCandidateModels:

    def test_always_includes_m0(self):
        """M0 (null baseline) is always included."""
        scan = FeatureScanResult()  # nothing detected
        models = enumerate_candidate_models(scan, y_max=5000.0)
        assert any(m.model_id == "M0" for m in models)
        m0 = [m for m in models if m.model_id == "M0"][0]
        assert m0.n_peaks == 0
        assert m0.bounds == []

    def test_nothing_detected_only_m0(self):
        """No features → only M0 returned."""
        scan = FeatureScanResult()
        models = enumerate_candidate_models(scan, y_max=5000.0)
        assert len(models) == 1
        assert models[0].model_id == "M0"

    def test_group3_only(self):
        """Group 3 only → M0 + M1."""
        scan = FeatureScanResult(group3_detected=True)
        models = enumerate_candidate_models(scan, y_max=5000.0)
        ids = {m.model_id for m in models}
        assert ids == {"M0", "M1"}
        m1 = [m for m in models if m.model_id == "M1"][0]
        assert m1.n_peaks == 1

    def test_group2_only(self):
        """Group 2 only → M0 + M2."""
        scan = FeatureScanResult(group2_detected=True)
        models = enumerate_candidate_models(scan, y_max=5000.0)
        ids = {m.model_id for m in models}
        assert ids == {"M0", "M2"}

    def test_group1_doublet(self):
        """Group 1 detected → M0 + M3."""
        scan = FeatureScanResult(group1_detected=True)
        scan.detections = [
            FeatureDetection("group1a", 304.0, 500.0, 10.0, 3000.0),
            FeatureDetection("group1b", 325.0, 400.0, 10.0, 2000.0),
        ]
        models = enumerate_candidate_models(scan, y_max=5000.0)
        ids = {m.model_id for m in models}
        assert "M3" in ids
        m3 = [m for m in models if m.model_id == "M3"][0]
        assert m3.n_peaks == 2

    def test_all_groups_generates_full_set(self):
        """All groups detected → M0 through M7."""
        scan = FeatureScanResult(
            group3_detected=True,
            group1_detected=True,
            group2_detected=True,
        )
        scan.detections = [
            FeatureDetection("group3", 280.0, 500.0, 15.0, 2000.0),
            FeatureDetection("group1a", 304.0, 500.0, 10.0, 3000.0),
            FeatureDetection("group1b", 325.0, 400.0, 10.0, 2000.0),
            FeatureDetection("group2", 340.0, 600.0, 20.0, 3500.0),
        ]
        models = enumerate_candidate_models(scan, y_max=5000.0)
        ids = {m.model_id for m in models}
        assert ids == {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"}

    def test_candidate_bounds_are_valid(self):
        """Each candidate has 3 bounds per peak (center, amp, fwhm)."""
        scan = FeatureScanResult(group3_detected=True, group2_detected=True)
        models = enumerate_candidate_models(scan, y_max=5000.0)
        for m in models:
            assert len(m.bounds) == m.n_peaks * 3
            for lo, hi in m.bounds:
                assert lo < hi

    def test_group2_constrained_to_single_peak(self):
        """Group 2 model (M2) has exactly 1 peak with FWHM >= 15 nm."""
        scan = FeatureScanResult(group2_detected=True)
        models = enumerate_candidate_models(scan, y_max=5000.0)
        m2 = [m for m in models if m.model_id == "M2"][0]
        assert m2.n_peaks == 1
        # FWHM bounds: 3rd bound tuple (index 2)
        fwhm_lo, fwhm_hi = m2.bounds[2]
        assert fwhm_lo >= 15.0
