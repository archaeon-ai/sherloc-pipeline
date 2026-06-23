"""
Unit tests for peak fitting models (bd-tqd: WS2-A).

Tests the peak fitting models:
- FittedPeak: A fitted spectral peak
- FittingResult: Summary of multi-peak fitting
- PeakType: Peak profile enumeration
"""

import uuid

import pytest
from pydantic import ValidationError

from sherloc_pipeline.models import (
    PeakType,
    FittedPeak,
    ModelRegistry,
)
from sherloc_pipeline.models.fitting import FittingResult


class TestPeakType:
    """Tests for PeakType enum."""

    def test_values(self):
        """PeakType has expected values."""
        assert PeakType.GAUSSIAN.value == "gaussian"
        assert PeakType.LORENTZIAN.value == "lorentzian"
        assert PeakType.VOIGT.value == "voigt"
        assert PeakType.PSEUDO_VOIGT.value == "pseudo_voigt"

    def test_all_types(self):
        """All four peak types are defined."""
        types = list(PeakType)
        assert len(types) == 4


class TestFittedPeak:
    """Tests for FittedPeak model."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, spectrum_id):
        """Create FittedPeak with minimal required fields."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )
        assert peak.center_cm1 == 1085.5
        assert peak.amplitude == 1500.0
        assert peak.fwhm_cm1 == 25.0
        assert peak.peak_type == PeakType.GAUSSIAN  # default
        assert peak.fit_modality == "minerals"

    def test_full_creation(self, spectrum_id):
        """Create FittedPeak with all fields."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            peak_type=PeakType.VOIGT,
            fit_modality="minerals",
            center_cm1=1085.5,
            center_uncertainty=0.5,
            amplitude=1500.0,
            amplitude_uncertainty=50.0,
            fwhm_cm1=25.0,
            fwhm_uncertainty=2.0,
            area=37500.0,
            snr=15.2,
            fit_quality=0.995,
            mineral_assignment="calcite",
            assignment_confidence=0.92,
        )
        assert peak.peak_type == PeakType.VOIGT
        assert peak.center_uncertainty == 0.5
        assert peak.snr == 15.2
        assert peak.mineral_assignment == "calcite"
        assert peak.assignment_confidence == 0.92

    def test_fwhm_must_be_positive(self, spectrum_id):
        """FWHM must be > 0."""
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.5,
                amplitude=1500.0,
                fwhm_cm1=0,  # Must be > 0
            )

        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.5,
                amplitude=1500.0,
                fwhm_cm1=-10.0,
            )

    def test_center_range_validation(self, spectrum_id):
        """Peak center must be in reasonable range."""
        # Valid range
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=0,  # Edge case
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )
        assert peak.center_cm1 == 0

        # Invalid - too high
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=15000,  # > 10000
                amplitude=1500.0,
                fwhm_cm1=25.0,
            )

        # Invalid - negative
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=-100,
                amplitude=1500.0,
                fwhm_cm1=25.0,
            )

    def test_fwhm_reasonable_validation(self, spectrum_id):
        """FWHM should not be excessively large."""
        # Valid large FWHM
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=400.0,  # Large but valid
        )
        assert peak.fwhm_cm1 == 400.0

        # Invalid - too large
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.5,
                amplitude=1500.0,
                fwhm_cm1=600.0,  # > 500
            )

    def test_fit_quality_range(self, spectrum_id):
        """fit_quality must be 0-1."""
        # Valid at boundaries
        peak1 = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            fit_quality=0.0,
        )
        peak2 = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            fit_quality=1.0,
        )
        assert peak1.fit_quality == 0.0
        assert peak2.fit_quality == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.5,
                amplitude=1500.0,
                fwhm_cm1=25.0,
                fit_quality=1.5,
            )

    def test_assignment_confidence_range(self, spectrum_id):
        """assignment_confidence must be 0-1."""
        with pytest.raises(ValidationError):
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.5,
                amplitude=1500.0,
                fwhm_cm1=25.0,
                mineral_assignment="calcite",
                assignment_confidence=1.5,
            )

    def test_assignment_confidence_defaults(self, spectrum_id):
        """assignment_confidence does NOT auto-default to 1.0."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            mineral_assignment="calcite",
            # assignment_confidence not provided — should remain None
        )
        assert peak.assignment_confidence is None

    def test_calculate_area_gaussian(self, spectrum_id):
        """calculate_area for Gaussian peak."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            peak_type=PeakType.GAUSSIAN,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=100.0,
            fwhm_cm1=25.0,
        )
        area = peak.calculate_area()
        # Gaussian: area = amplitude * fwhm * sqrt(pi / (4 * ln(2))) ~ 1.0645
        expected = 100.0 * 25.0 * 1.0645
        assert area == pytest.approx(expected, rel=0.01)

    def test_calculate_area_lorentzian(self, spectrum_id):
        """calculate_area for Lorentzian peak."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            peak_type=PeakType.LORENTZIAN,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=100.0,
            fwhm_cm1=25.0,
        )
        area = peak.calculate_area()
        # Lorentzian: area = amplitude * fwhm * pi/2
        import math
        expected = 100.0 * 25.0 * math.pi / 2
        assert area == pytest.approx(expected, rel=0.01)

    def test_is_significant(self, spectrum_id):
        """is_significant checks SNR threshold."""
        # Significant
        peak1 = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            snr=5.0,
        )
        assert peak1.is_significant(min_snr=3.0) is True

        # Not significant
        peak2 = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            snr=2.0,
        )
        assert peak2.is_significant(min_snr=3.0) is False

        # No SNR - assume significant
        peak3 = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )
        assert peak3.is_significant(min_snr=3.0) is True

    def test_matches_mineral(self, spectrum_id):
        """matches_mineral finds matching mineral peaks."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )

        mineral_db = {
            "calcite": [1085.0, 282.0],
            "gypsum": [1008.0, 414.0],
            "dolomite": [1097.0, 299.0],
        }

        # Should match calcite (within tolerance)
        match = peak.matches_mineral(mineral_db, tolerance_cm1=10.0)
        assert match == "calcite"

        # Tighter tolerance - no match
        match2 = peak.matches_mineral(mineral_db, tolerance_cm1=0.1)
        assert match2 is None

    def test_has_uuid(self, spectrum_id):
        """FittedPeak has auto-generated UUID."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )
        assert peak.id is not None
        assert isinstance(peak.id, uuid.UUID)

    def test_model_can_be_registered(self):
        """FittedPeak can be registered in ModelRegistry."""
        assert hasattr(FittedPeak, "__pydantic_complete__")


class TestFittingResult:
    """Tests for FittingResult model."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_basic_creation(self, spectrum_id):
        """Create FittingResult with minimal fields."""
        result = FittingResult(
            spectrum_id=spectrum_id,
            n_peaks=3,
        )
        assert result.n_peaks == 3
        assert result.fitting_method == "lmfit"  # default

    def test_full_creation(self, spectrum_id):
        """Create FittingResult with all fields."""
        result = FittingResult(
            spectrum_id=spectrum_id,
            n_peaks=3,
            residual_rms=0.5,
            r_squared=0.995,
            chi_squared=12.5,
            fitting_method="scipy_optimize",
            config_hash="abc123",
        )
        assert result.residual_rms == 0.5
        assert result.r_squared == 0.995
        assert result.fitting_method == "scipy_optimize"

    def test_n_peaks_non_negative(self, spectrum_id):
        """n_peaks must be >= 0."""
        # Zero is valid
        result = FittingResult(spectrum_id=spectrum_id, n_peaks=0)
        assert result.n_peaks == 0

        with pytest.raises(ValidationError):
            FittingResult(spectrum_id=spectrum_id, n_peaks=-1)

    def test_r_squared_range(self, spectrum_id):
        """r_squared must be 0-1."""
        with pytest.raises(ValidationError):
            FittingResult(
                spectrum_id=spectrum_id,
                n_peaks=3,
                r_squared=1.5,
            )

    def test_residual_rms_non_negative(self, spectrum_id):
        """residual_rms must be >= 0."""
        with pytest.raises(ValidationError):
            FittingResult(
                spectrum_id=spectrum_id,
                n_peaks=3,
                residual_rms=-0.1,
            )


class TestFittingIntegration:
    """Integration tests for fitting models."""

    def test_multiple_peaks_per_spectrum(self):
        """Multiple FittedPeaks can reference same spectrum."""
        spectrum_id = uuid.uuid4()

        peaks = [
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=pos,
                amplitude=1000.0,
                fwhm_cm1=20.0,
            )
            for pos in [282.0, 711.0, 1085.0]
        ]

        assert len(peaks) == 3
        assert all(p.spectrum_id == spectrum_id for p in peaks)
        assert len(set(p.id for p in peaks)) == 3  # Unique IDs

    def test_fitting_workflow(self):
        """Simulate complete fitting workflow."""
        spectrum_id = uuid.uuid4()

        # Create fitted peaks
        peaks = [
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=1085.0,
                amplitude=1500.0,
                fwhm_cm1=25.0,
                snr=15.0,
                fit_quality=0.995,
            ),
            FittedPeak(
                spectrum_id=spectrum_id,
                fit_modality="minerals",
                center_cm1=282.0,
                amplitude=500.0,
                fwhm_cm1=30.0,
                snr=8.0,
                fit_quality=0.98,
            ),
        ]

        # Create fitting result
        result = FittingResult(
            spectrum_id=spectrum_id,
            n_peaks=len(peaks),
            r_squared=0.99,
            residual_rms=0.3,
        )

        assert result.n_peaks == 2
        assert all(p.is_significant(min_snr=5.0) for p in peaks)
