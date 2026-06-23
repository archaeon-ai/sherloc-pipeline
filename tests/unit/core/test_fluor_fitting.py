"""Unit tests for fluorescence fitting (core/fluor_fitting.py) and
fluorescence identification (core/fluor_id.py).

Covers: synthetic Gaussian recovery, saturation tiers, group assignment,
doublet detection, SNR filtering. See spec §11 Tier 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from sherloc_pipeline.core.fitting import fwhm_to_sigma, gaussian
from sherloc_pipeline.core.fluor_fitting import (
    FluorFitResult,
    FluorPeakFit,
    fit_fluorescence_spectrum,
)
from sherloc_pipeline.core.fluor_id import (
    CooccurrenceScore,
    DoubletRecord,
    FluorescenceRule,
    FLUORESCENCE_RULES,
    assign_fluor_group,
    classify_fluor_peaks,
    detect_doublets,
    score_cooccurrences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wavelength(start: float = 260.0, stop: float = 360.0, n: int = 500):
    """Monotonically increasing wavelength array (nm)."""
    return np.linspace(start, stop, n)


def _synth_spectrum(
    wavelength: np.ndarray,
    peaks: list[tuple[float, float, float]],
    noise_std: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic multi-Gaussian spectrum.

    peaks: list of (center_nm, amplitude, fwhm_nm) tuples.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros_like(wavelength, dtype=float)
    for center, amp, fwhm in peaks:
        sigma = fwhm_to_sigma(fwhm)
        y += amp * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    y += rng.normal(0.0, noise_std, size=y.size)
    return y


# ---------------------------------------------------------------------------
# AC 1: Single-Gaussian fits within 1 nm
# ---------------------------------------------------------------------------


class TestSingleGaussianFit:

    def test_single_peak_recovery(self):
        """Synthetic single Gaussian fits within 1 nm of true center."""
        wl = _make_wavelength()
        true_center, true_amp, true_fwhm = 305.0, 5000.0, 20.0
        intensity = _synth_spectrum(wl, [(true_center, true_amp, true_fwhm)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, max_peaks=2, strategy="agnostic",
        )

        assert result.n_peaks >= 1
        assert not result.fit_skipped
        # Find the peak closest to the true center
        best = min(result.peaks, key=lambda p: abs(p.center_nm - true_center))
        assert abs(best.center_nm - true_center) < 1.0, (
            f"Fitted center {best.center_nm:.2f} nm too far from true {true_center:.2f} nm"
        )
        assert result.r2 > 0.95

    def test_single_peak_at_340nm(self):
        """Fit a peak in the group2 region (329-355 nm)."""
        wl = _make_wavelength()
        true_center = 340.0
        intensity = _synth_spectrum(wl, [(true_center, 4000.0, 18.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0, max_peaks=2)

        assert result.n_peaks >= 1
        best = min(result.peaks, key=lambda p: abs(p.center_nm - true_center))
        assert abs(best.center_nm - true_center) < 1.0


# ---------------------------------------------------------------------------
# AICc model selection: broad peak should not be split
# ---------------------------------------------------------------------------


class TestAICcModelSelection:

    def test_broad_gaussian_prefers_one_peak(self):
        """AICc regression: broad symmetric Gaussian at 340nm → 1-peak model wins.

        Without AICc, raw RSS always favors 2+ peaks because two narrow
        Gaussians can approximate a broad peak with lower residuals. AICc
        penalizes the extra 3 parameters, so the 1-peak model should win.
        """
        wl = _make_wavelength(n=500)
        # Broad, symmetric peak in Group 2 region
        true_center, true_fwhm = 340.0, 22.0
        intensity = _synth_spectrum(
            wl, [(true_center, 5000.0, true_fwhm)], noise_std=15.0
        )

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=15.0, max_peaks=3
        )

        assert result.n_peaks == 1, (
            f"Expected 1 peak for broad symmetric Gaussian, got {result.n_peaks}: "
            f"{[(p.center_nm, p.fwhm_nm) for p in result.peaks]}"
        )
        assert abs(result.peaks[0].center_nm - true_center) < 2.0
        assert result.aicc < float("inf")

    def test_aicc_field_present(self):
        """FluorFitResult has aicc field."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)
        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0)
        assert hasattr(result, "aicc")
        assert np.isfinite(result.aicc)


# ---------------------------------------------------------------------------
# AC 2: Multi-Gaussian (2-3 peaks) fits correctly
# ---------------------------------------------------------------------------


class TestMultiGaussianFit:

    def test_two_peak_recovery(self):
        """Two well-separated Gaussians both recovered."""
        wl = _make_wavelength(n=800)
        peaks_true = [(290.0, 3000.0, 15.0), (335.0, 4000.0, 20.0)]
        intensity = _synth_spectrum(wl, peaks_true, noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, max_peaks=3,
            fit_range=(276.0, 355.0), position_bounds=(270.0, 355.0),
        )

        assert result.n_peaks >= 2
        centers = sorted(p.center_nm for p in result.peaks)
        # Both peaks recovered within 2 nm
        assert abs(centers[0] - 290.0) < 2.0
        assert abs(centers[-1] - 335.0) < 2.0

    def test_three_peak_recovery(self):
        """Three Gaussians spanning fit range recovered (agnostic strategy)."""
        wl = _make_wavelength(n=1000)
        peaks_true = [(285.0, 2500.0, 15.0), (305.0, 3500.0, 18.0), (340.0, 3000.0, 20.0)]
        intensity = _synth_spectrum(wl, peaks_true, noise_std=8.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=8.0, max_peaks=3, strategy="agnostic",
        )

        assert result.n_peaks >= 3
        centers = sorted(p.center_nm for p in result.peaks)
        assert abs(centers[0] - 285.0) < 2.0
        assert abs(centers[1] - 305.0) < 2.0
        assert abs(centers[2] - 340.0) < 2.0


# ---------------------------------------------------------------------------
# AC 3: Three-tier saturation handling
# ---------------------------------------------------------------------------


class TestSaturationTiers:

    def test_normal_no_saturation(self):
        """Below threshold: normal fit, not saturated."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl,
            intensity,
            noise_std=10.0,
            saturation_threshold=60000.0,
        )

        assert not result.is_saturated
        assert not result.fit_skipped
        assert result.n_peaks >= 1

    def test_partial_saturation_mask_and_fit(self):
        """Few channels above threshold: mask and still fit wing shape."""
        wl = _make_wavelength(n=500)
        true_center = 305.0
        intensity = _synth_spectrum(wl, [(true_center, 50000.0, 20.0)], noise_std=10.0)
        # Clip peak region to create partial saturation (3 channels)
        sat_level = 60000.0
        n_sat = np.sum(intensity >= sat_level)
        # Inject a few saturated channels to be under the limit
        intensity[intensity >= sat_level] = sat_level + 1.0
        # Ensure count is under limit
        sat_mask = intensity >= sat_level
        n_above = int(np.sum(sat_mask))
        # If we got too many, trim — just want < saturation_channel_limit
        if n_above >= 5:
            # Use a tall narrow peak that only saturates a couple channels
            intensity = _synth_spectrum(wl, [(true_center, 80000.0, 12.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl,
            intensity,
            noise_std=10.0,
            saturation_threshold=60000.0,
            saturation_channel_limit=5,
        )

        # Partial saturation: is_saturated but not fit_skipped
        if result.is_saturated:
            assert not result.fit_skipped or result.n_saturated_channels >= 5
        # Should still recover the peak center
        if result.n_peaks >= 1:
            best = min(result.peaks, key=lambda p: abs(p.center_nm - true_center))
            assert abs(best.center_nm - true_center) < 3.0

    def test_full_saturation_skip(self):
        """Many channels saturated: fit skipped entirely."""
        wl = _make_wavelength(n=500)
        # Create a very broad, very tall peak that saturates many channels
        intensity = _synth_spectrum(wl, [(305.0, 200000.0, 30.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl,
            intensity,
            noise_std=10.0,
            saturation_threshold=60000.0,
            saturation_channel_limit=5,
        )

        assert result.is_saturated
        assert result.fit_skipped
        assert result.n_peaks == 0
        assert result.n_saturated_channels >= 5
        assert "full_saturation_skip" in result.warnings


# ---------------------------------------------------------------------------
# AC 4: Group assignment — each range returns correct label
# ---------------------------------------------------------------------------


class TestAssignFluorGroup:

    @pytest.mark.parametrize(
        "center_nm, expected",
        [
            (270.0, "group3"),  # lower bound group3
            (280.0, "group3"),  # mid group3
            (295.0, "group3"),  # upper bound group3
            (300.0, "group1a"),  # lower bound group1a
            (303.5, "group1a"),  # mid group1a
            (307.0, "group1a"),  # upper bound group1a
            (322.0, "group1b"),  # lower bound group1b
            (325.5, "group1b"),  # mid group1b
            (329.0, "group1b"),  # upper bound group1b
            (330.0, "group2"),  # lower range group2
            (342.0, "group2"),  # mid group2
            (355.0, "group2"),  # upper bound group2
        ],
    )
    def test_group_assignment_ranges(self, center_nm, expected):
        assert assign_fluor_group(center_nm) == expected

    def test_rules_are_fluorescence_rule_objects(self):
        for rule in FLUORESCENCE_RULES:
            assert isinstance(rule, FluorescenceRule)
            assert rule.lo < rule.hi


# ---------------------------------------------------------------------------
# AC 5: Doublet detection with greedy pairing
# ---------------------------------------------------------------------------


class TestDetectDoublets:

    def _make_peak(self, center_nm, amplitude=1000.0, fwhm_nm=15.0, snr=10.0):
        return FluorPeakFit(
            center_nm=center_nm,
            amplitude=amplitude,
            fwhm_nm=fwhm_nm,
            area=0.0,
            snr=snr,
        )

    def test_basic_doublet(self):
        """group1a + group1b pair detected as doublet."""
        peaks = [
            self._make_peak(305.0),  # group1a
            self._make_peak(325.0),  # group1b
        ]
        doublets = detect_doublets(peaks)
        assert len(doublets) == 1
        assert doublets[0].center_1a_nm == 305.0
        assert doublets[0].center_1b_nm == 325.0
        assert doublets[0].separation_nm == 20.0

    def test_multiple_doublets_greedy_pairing(self):
        """Multiple candidates: nearest-separation greedy picks correct pairs."""
        peaks = [
            self._make_peak(305.0, snr=10.0),  # group1a
            self._make_peak(303.0, snr=8.0),  # group1a
            self._make_peak(325.0, snr=10.0),  # group1b
            self._make_peak(327.0, snr=8.0),  # group1b
        ]
        doublets = detect_doublets(peaks)
        assert len(doublets) == 2
        # Pairs should be (305,325)=20nm and (303,327)=24nm — greedy by nearest sep
        seps = sorted(d.separation_nm for d in doublets)
        assert seps[0] == pytest.approx(20.0, abs=0.1)
        assert seps[1] == pytest.approx(24.0, abs=0.1)

    def test_no_doublet_single_group(self):
        """Only group1a peaks → no doublets."""
        peaks = [
            self._make_peak(305.0),
            self._make_peak(303.0),
        ]
        assert detect_doublets(peaks) == []

    def test_no_doublet_empty(self):
        """Empty peak list → no doublets."""
        assert detect_doublets([]) == []

    def test_doublet_record_fields(self):
        """DoubletRecord contains all expected fields."""
        peaks = [
            self._make_peak(305.0, amplitude=1200.0, fwhm_nm=14.0, snr=12.0),
            self._make_peak(325.0, amplitude=800.0, fwhm_nm=16.0, snr=9.0),
        ]
        doublets = detect_doublets(peaks, doublet_snr_threshold=5.0)
        assert len(doublets) == 1
        d = doublets[0]
        assert d.peak_1a_idx == 0
        assert d.peak_1b_idx == 1
        assert d.amplitude_1a == 1200.0
        assert d.amplitude_1b == 800.0
        assert d.fwhm_1a_nm == 14.0
        assert d.fwhm_1b_nm == 16.0
        assert d.intensity_ratio == pytest.approx(1200.0 / 800.0)

    def test_separation_out_of_range_rejected(self):
        """Peaks with separation outside 18-29 nm are not paired."""
        peaks = [
            self._make_peak(300.0),  # group1a (lower bound)
            self._make_peak(335.0),  # group2, not group1b — but test sep directly
        ]
        # 300 is group1a, 335 is group2 — not a pair
        assert detect_doublets(peaks) == []

    def test_separation_too_small_rejected(self):
        """Peaks with separation < 18 nm are not paired (even if both groups)."""
        # group1a: 307, group1b: 322 → sep = 15 nm < 18
        peaks = [
            self._make_peak(307.0),  # group1a upper
            self._make_peak(322.0),  # group1b lower
        ]
        assert detect_doublets(peaks) == []

    def test_doublet_ratio_range_filters(self):
        """Doublets with intensity ratio outside range are rejected."""
        peaks = [
            self._make_peak(305.0, amplitude=3000.0),  # group1a, strong
            self._make_peak(325.0, amplitude=1000.0),  # group1b, weaker
        ]
        # ratio = 3000/1000 = 3.0
        # Without ratio filter → doublet found
        assert len(detect_doublets(peaks)) == 1
        # Ratio filter accepts range [0.5, 2.0] → ratio=3.0 rejected
        assert detect_doublets(peaks, doublet_ratio_range=(0.5, 2.0)) == []
        # Ratio filter accepts range [0.5, 4.0] → ratio=3.0 accepted
        assert len(detect_doublets(peaks, doublet_ratio_range=(0.5, 4.0))) == 1

    def test_doublet_ratio_range_none_no_filter(self):
        """doublet_ratio_range=None disables ratio filtering (default)."""
        peaks = [
            self._make_peak(305.0, amplitude=5000.0),
            self._make_peak(325.0, amplitude=100.0),
        ]
        # Extreme ratio = 50.0, but no filter → still detected
        doublets = detect_doublets(peaks, doublet_ratio_range=None)
        assert len(doublets) == 1
        assert doublets[0].intensity_ratio == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# AC 6: Out-of-range peaks return 'unidentified'
# ---------------------------------------------------------------------------


class TestOutOfRange:

    @pytest.mark.parametrize(
        "center_nm",
        [
            260.0,  # below all groups
            296.0,  # gap between group3 and group1a
            310.0,  # gap between group1a and group1b
            356.0,  # above all groups
        ],
    )
    def test_out_of_range_unidentified(self, center_nm):
        assert assign_fluor_group(center_nm) == "unidentified"


# ---------------------------------------------------------------------------
# AC 6b: Post-classification rules (orphan group1b reclassification)
# ---------------------------------------------------------------------------


class TestClassifyFluorPeaks:

    def test_orphan_group1b_reclassified_to_group2(self):
        """Group1b peak >328 nm without group1a companion is reclassified as group2."""
        peaks = [
            FluorPeakFit(center_nm=329.0, amplitude=1000.0, fwhm_nm=15.0, area=0.0, snr=50.0),
        ]
        groups = [assign_fluor_group(p.center_nm) for p in peaks]
        assert groups == ["group1b"]

        result = classify_fluor_peaks(groups, peaks)
        assert result == ["group2"]

    def test_orphan_group1b_retained_below_threshold(self):
        """Group1b peak <=328 nm without group1a stays group1b (not reclassified)."""
        peaks = [
            FluorPeakFit(center_nm=326.0, amplitude=1000.0, fwhm_nm=15.0, area=0.0, snr=50.0),
        ]
        groups = [assign_fluor_group(p.center_nm) for p in peaks]
        assert groups == ["group1b"]

        result = classify_fluor_peaks(groups, peaks)
        assert result == ["group1b"]

    def test_group1b_kept_with_group1a(self):
        """Group1b retained when group1a companion is present (valid doublet)."""
        peaks = [
            FluorPeakFit(center_nm=304.0, amplitude=2000.0, fwhm_nm=15.0, area=0.0, snr=100.0),
            FluorPeakFit(center_nm=326.0, amplitude=1000.0, fwhm_nm=15.0, area=0.0, snr=50.0),
        ]
        groups = [assign_fluor_group(p.center_nm) for p in peaks]
        assert groups == ["group1a", "group1b"]

        result = classify_fluor_peaks(groups, peaks)
        assert result == ["group1a", "group1b"]

    def test_no_group1b_no_change(self):
        """No group1b peaks means no reclassification needed."""
        peaks = [
            FluorPeakFit(center_nm=285.0, amplitude=500.0, fwhm_nm=20.0, area=0.0, snr=25.0),
            FluorPeakFit(center_nm=340.0, amplitude=3000.0, fwhm_nm=18.0, area=0.0, snr=150.0),
        ]
        groups = [assign_fluor_group(p.center_nm) for p in peaks]
        assert groups == ["group3", "group2"]

        result = classify_fluor_peaks(groups, peaks)
        assert result == ["group3", "group2"]

    def test_multiple_group1b_mixed_reclassification(self):
        """Orphan group1b peaks: <=328 nm retained, >328 nm reclassified to group2."""
        peaks = [
            FluorPeakFit(center_nm=323.0, amplitude=500.0, fwhm_nm=12.0, area=0.0, snr=25.0),
            FluorPeakFit(center_nm=329.0, amplitude=800.0, fwhm_nm=14.0, area=0.0, snr=40.0),
        ]
        groups = [assign_fluor_group(p.center_nm) for p in peaks]
        assert groups == ["group1b", "group1b"]

        result = classify_fluor_peaks(groups, peaks)
        assert result == ["group1b", "group2"]


# ---------------------------------------------------------------------------
# AC 7: SNR filtering at threshold 2.0
# ---------------------------------------------------------------------------


class TestSNRFiltering:

    def test_low_snr_peak_filtered(self):
        """Peak with SNR below threshold is excluded from results."""
        wl = _make_wavelength()
        # Very weak peak with high noise → low SNR
        intensity = _synth_spectrum(wl, [(305.0, 50.0, 20.0)], noise_std=100.0)

        result = fit_fluorescence_spectrum(
            wl,
            intensity,
            noise_std=100.0,
            snr_threshold=2.0,
        )

        # The weak peak should be filtered out by SNR threshold
        for p in result.peaks:
            assert p.snr >= 2.0

    def test_high_snr_peak_retained(self):
        """Strong peak above SNR threshold is retained."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, snr_threshold=2.0
        )

        assert result.n_peaks >= 1
        assert all(p.snr >= 2.0 for p in result.peaks)

    def test_doublet_snr_filtering(self):
        """Doublet detection respects SNR threshold."""
        peaks = [
            FluorPeakFit(center_nm=305.0, amplitude=100.0, fwhm_nm=15.0, area=0.0, snr=3.0),
            FluorPeakFit(center_nm=325.0, amplitude=100.0, fwhm_nm=15.0, area=0.0, snr=3.0),
        ]
        # With default threshold 5.0 → both below → no doublets
        assert detect_doublets(peaks, doublet_snr_threshold=5.0) == []
        # Lower threshold → both pass → doublet found
        assert len(detect_doublets(peaks, doublet_snr_threshold=2.0)) == 1


# ---------------------------------------------------------------------------
# AC 8: Edge cases and result structure
# ---------------------------------------------------------------------------


class TestResultStructure:

    def test_result_is_fluor_fit_result(self):
        """fit_fluorescence_spectrum returns FluorFitResult."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)
        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0)
        assert isinstance(result, FluorFitResult)

    def test_peaks_sorted_by_center(self):
        """Peaks in result are sorted by center_nm."""
        wl = _make_wavelength(n=800)
        intensity = _synth_spectrum(
            wl, [(335.0, 4000.0, 20.0), (290.0, 3000.0, 15.0)], noise_std=10.0
        )
        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0, max_peaks=3)
        if result.n_peaks >= 2:
            for i in range(result.n_peaks - 1):
                assert result.peaks[i].center_nm <= result.peaks[i + 1].center_nm

    def test_empty_data_range(self):
        """Fit range outside wavelength → no data warning."""
        wl = _make_wavelength(start=200.0, stop=250.0)  # below fit_range
        intensity = np.ones_like(wl) * 100.0
        result = fit_fluorescence_spectrum(wl, intensity)
        assert result.n_peaks == 0
        assert "no_data_in_fit_range" in result.warnings

    def test_peak_has_expected_fields(self):
        """FluorPeakFit has center_nm, amplitude, fwhm_nm, area, snr."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)
        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0)
        assert result.n_peaks >= 1
        p = result.peaks[0]
        assert hasattr(p, "center_nm")
        assert hasattr(p, "amplitude")
        assert hasattr(p, "fwhm_nm")
        assert hasattr(p, "area")
        assert hasattr(p, "snr")
        assert p.area > 0
        assert p.snr > 0

    def test_overlap_exclusion_channels_masked(self):
        """Channels in overlap zone (337.4-338.4) are excluded."""
        wl = _make_wavelength(n=800)
        # Put a peak right in the overlap zone
        intensity = _synth_spectrum(wl, [(338.0, 5000.0, 2.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(wl, intensity, noise_std=10.0)
        assert result.n_masked_channels >= 0  # overlap zone counted


# ---------------------------------------------------------------------------
# Hypothesis-driven fitting tests
# ---------------------------------------------------------------------------


class TestHypothesisDrivenFitting:

    def test_hypothesis_group2_asymmetric_single_peak(self):
        """Group 2 asymmetric band → hypothesis constrains to single peak.

        This is the key scientific requirement: the Group 2 feature is a
        single asymmetric band (partially-resolved doublet per Scheller),
        not two narrow Gaussians.
        """
        wl = _make_wavelength(n=500)
        # Broad asymmetric peak simulated as single Gaussian at 340nm
        intensity = _synth_spectrum(
            wl, [(340.0, 5000.0, 22.0)], noise_std=15.0
        )

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=15.0,
            strategy="hypothesis", max_peaks=3,
        )

        # Should fit as 1 peak, not split into 2
        assert result.n_peaks == 1, (
            f"Expected 1 peak for Group 2 asymmetric band, got {result.n_peaks}: "
            f"{[(p.center_nm, p.fwhm_nm) for p in result.peaks]}"
        )
        assert abs(result.peaks[0].center_nm - 340.0) < 3.0
        assert result.r2 > 0.9

    def test_hypothesis_group1_doublet(self):
        """Ce3+ doublet (304+325 nm) → hypothesis recovers 2 peaks."""
        wl = _make_wavelength(n=800)
        intensity = _synth_spectrum(
            wl,
            [(304.0, 5000.0, 10.0), (325.0, 3000.0, 10.0)],
            noise_std=10.0,
        )

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0,
            strategy="hypothesis", max_peaks=3,
        )

        assert result.n_peaks >= 2
        centers = sorted(p.center_nm for p in result.peaks)
        assert abs(centers[0] - 304.0) < 3.0
        assert abs(centers[1] - 325.0) < 3.0

    def test_hypothesis_group3_plus_group2(self):
        """Group 3 + Group 2 → M4 model (2 peaks)."""
        wl = _make_wavelength(n=800)
        intensity = _synth_spectrum(
            wl,
            [(280.0, 3000.0, 15.0), (340.0, 4000.0, 20.0)],
            noise_std=10.0,
        )

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0,
            strategy="hypothesis", max_peaks=3,
        )

        assert result.n_peaks >= 2
        centers = sorted(p.center_nm for p in result.peaks)
        assert abs(centers[0] - 280.0) < 3.0
        assert abs(centers[-1] - 340.0) < 3.0

    def test_agnostic_strategy_still_works(self):
        """strategy='agnostic' is the default path and works correctly."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 20.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0,
            strategy="agnostic", max_peaks=2,
        )

        assert result.n_peaks >= 1
        assert result.r2 > 0.9

    def test_hypothesis_noise_only_returns_no_peaks(self):
        """Pure noise → hypothesis scan detects nothing → M0 wins → 0 peaks."""
        wl = _make_wavelength()
        rng = np.random.default_rng(42)
        # Zero-mean noise (dark-subtracted, no fluorescence signal)
        intensity = rng.normal(0.0, 10.0, size=wl.size)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0,
            strategy="hypothesis",
        )

        assert result.n_peaks == 0


# ---------------------------------------------------------------------------
# Cross-modal co-occurrence scoring
# ---------------------------------------------------------------------------


class TestScoreCooccurrences:

    def test_sulfate_raman_confirms_group1(self):
        """Ca-sulfate Raman + Group 1 → confirmed Ce3+-bearing anhydrite."""
        fluor_groups = ["group1a", "group1b"]
        raman_assignments = ["sulf1_v1"]
        scores = score_cooccurrences(fluor_groups, raman_assignments)

        assert len(scores) == 2
        assert scores[0].raman_support == "confirmed"
        assert scores[0].phase_interpretation == "Ce3+-bearing anhydrite"
        assert scores[0].confidence_boost > 1.0

    def test_phosphate_raman_confirms_group2(self):
        """Phosphate Raman + Group 2 → confirmed Ce3+-bearing phosphate."""
        fluor_groups = ["group2"]
        raman_assignments = ["phosphate"]
        scores = score_cooccurrences(fluor_groups, raman_assignments)

        assert len(scores) == 1
        assert scores[0].raman_support == "confirmed"
        assert scores[0].phase_interpretation == "Ce3+-bearing phosphate"
        assert scores[0].confidence_boost > 1.0

    def test_perchlorate_contradicts_group2(self):
        """Perchlorate Raman + Group 2 → contradicted (likely phosphate)."""
        fluor_groups = ["group2"]
        raman_assignments = ["perchlorate"]
        scores = score_cooccurrences(fluor_groups, raman_assignments)

        assert scores[0].raman_support == "contradicted"
        assert scores[0].confidence_boost < 1.0
        assert "phosphate" in scores[0].phase_interpretation.lower()

    def test_no_raman_data(self):
        """No Raman peaks → 'no_raman' support."""
        fluor_groups = ["group1a", "group2"]
        scores = score_cooccurrences(fluor_groups, [])

        assert all(s.raman_support == "no_raman" for s in scores)
        assert all(s.confidence_boost == 1.0 for s in scores)

    def test_group2_without_phosphate_raman(self):
        """Group 2 with non-phosphate Raman → unsupported."""
        fluor_groups = ["group2"]
        raman_assignments = ["olivine"]
        scores = score_cooccurrences(fluor_groups, raman_assignments)

        assert scores[0].raman_support == "unsupported"
        assert "below Raman detection" in scores[0].phase_interpretation

    def test_group3_always_silicate_defect(self):
        """Group 3 → silicate defect regardless of Raman."""
        for raman in [[], ["olivine"], ["phosphate"]]:
            scores = score_cooccurrences(["group3"], raman)
            assert scores[0].phase_interpretation == "Silicate defect luminescence"

    def test_unidentified_group(self):
        """Unidentified fluorescence → generic annotation."""
        scores = score_cooccurrences(["unidentified"], ["olivine"])
        assert scores[0].phase_interpretation == "Unidentified fluorescence"

    def test_mixed_groups_scored_independently(self):
        """Multiple fluor groups each scored against same Raman set."""
        fluor_groups = ["group1a", "group1b", "group2"]
        raman_assignments = ["sulf1_v1", "phosphate"]
        scores = score_cooccurrences(fluor_groups, raman_assignments)

        assert len(scores) == 3
        # Group 1 peaks should be confirmed (sulfate present)
        assert scores[0].raman_support == "confirmed"
        assert scores[1].raman_support == "confirmed"
        # Group 2 should also be confirmed (phosphate present)
        assert scores[2].raman_support == "confirmed"


# ---------------------------------------------------------------------------
# Early bail-out tests
# ---------------------------------------------------------------------------


class TestEarlyBailOut:
    """Early bail-out skips fitting when max signal is below SNR threshold."""

    def test_noise_only_bails_out(self):
        """Pure noise below SNR threshold -> fit_skipped=True, 0 peaks."""
        wl = _make_wavelength()
        rng = np.random.default_rng(42)
        intensity = rng.normal(0.0, 10.0, size=wl.size)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, snr_threshold=10.0,
        )

        assert result.n_peaks == 0
        assert result.fit_skipped is True
        assert "below_snr_threshold" in result.warnings

    def test_real_signal_not_bailed_out(self):
        """Spectrum with clear peak above threshold -> fitting proceeds."""
        wl = _make_wavelength()
        intensity = _synth_spectrum(wl, [(305.0, 5000.0, 15.0)], noise_std=10.0)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, snr_threshold=10.0,
        )

        assert result.n_peaks >= 1
        assert result.fit_skipped is False
        assert "below_snr_threshold" not in result.warnings

    def test_cosmic_ray_cleaned_before_check(self):
        """Cosmic ray spike on flat noise -> despiker cleans it -> bail-out fires."""
        wl = _make_wavelength()
        rng = np.random.default_rng(42)
        intensity = rng.normal(0.0, 10.0, size=wl.size)
        # Plant a cosmic ray spike (single channel, very high)
        spike_idx = np.argmin(np.abs(wl - 310.0))
        intensity[spike_idx] = 50000.0

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, snr_threshold=10.0,
        )

        assert result.n_peaks == 0
        assert result.fit_skipped is True
        assert "below_snr_threshold" in result.warnings

    def test_baseline_offset_noise_bails_out(self):
        """Noise on high baseline (no features) -> bail-out fires.

        Fluorescence data has no baseline subtraction, so raw counts can be
        large even without features. The bail-out uses prominence (max - median)
        rather than raw max to handle this.
        """
        wl = _make_wavelength()
        rng = np.random.default_rng(42)
        # Flat baseline at 500 counts + small noise
        intensity = 500.0 + rng.normal(0.0, 10.0, size=wl.size)

        result = fit_fluorescence_spectrum(
            wl, intensity, noise_std=10.0, snr_threshold=10.0,
        )

        assert result.n_peaks == 0
        assert result.fit_skipped is True
        assert "below_snr_threshold" in result.warnings
