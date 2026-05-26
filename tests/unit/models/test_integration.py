"""
Unit tests for Pydantic integration utilities (bd-17w: WS2-C).

Tests the conversion between core.fitting dataclasses and Pydantic models:
- PeakFit <-> FittedPeak
- FitResult <-> FittingResult
"""

import uuid
import math

import pytest

from sherloc_pipeline.models.fitting import PeakFit, FitResult
from sherloc_pipeline.models import (
    FittedPeak,
    FittingResult,
    PeakType,
    peak_fit_to_pydantic,
    peak_fits_to_pydantic,
    fit_result_to_pydantic,
    pydantic_to_peak_fit,
    pydantic_to_fit_result,
)


class TestPeakFitToPydantic:
    """Tests for core PeakFit -> Pydantic FittedPeak conversion."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_peak_fit(self):
        """Create a sample PeakFit dataclass."""
        return PeakFit(
            m_cm1=1085.5,
            a=1500.0,
            fwhm=25.0,
            sigma=25.0 / (2.0 * math.sqrt(2.0 * math.log(2.0))),
            area=39656.0,
            snr=15.2,
            pass_snr=True,
            pass_fwhm=True,
            pass_r2=True,
        )

    def test_basic_conversion(self, sample_peak_fit, spectrum_id):
        """Convert PeakFit to FittedPeak."""
        result = peak_fit_to_pydantic(sample_peak_fit, spectrum_id, "minerals")

        assert isinstance(result, FittedPeak)
        assert result.spectrum_id == spectrum_id
        assert result.center_cm1 == 1085.5
        assert result.amplitude == 1500.0
        assert result.fwhm_cm1 == 25.0
        assert result.snr == 15.2
        assert result.peak_type == PeakType.GAUSSIAN

    def test_conversion_with_r2(self, sample_peak_fit, spectrum_id):
        """Conversion includes R^2 as fit_quality."""
        result = peak_fit_to_pydantic(sample_peak_fit, spectrum_id, "minerals", r2=0.995)

        assert result.fit_quality == 0.995

    def test_converted_peak_has_uuid(self, sample_peak_fit, spectrum_id):
        """Converted peak gets its own UUID."""
        result = peak_fit_to_pydantic(sample_peak_fit, spectrum_id, "minerals")

        assert result.id is not None
        assert isinstance(result.id, uuid.UUID)

    def test_area_preserved(self, sample_peak_fit, spectrum_id):
        """Area value is preserved in conversion."""
        result = peak_fit_to_pydantic(sample_peak_fit, spectrum_id, "minerals")

        assert result.area == 39656.0


class TestPeakFitsToPydantic:
    """Tests for batch conversion of PeakFit list."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_peaks(self):
        """Create sample PeakFit list."""
        return [
            PeakFit(m_cm1=282.0, a=500.0, fwhm=30.0, sigma=12.7, area=15900.0, snr=8.0,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
            PeakFit(m_cm1=711.0, a=800.0, fwhm=28.0, sigma=11.9, area=23600.0, snr=10.0,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
            PeakFit(m_cm1=1085.5, a=1500.0, fwhm=25.0, sigma=10.6, area=39656.0, snr=15.2,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
        ]

    def test_batch_conversion(self, sample_peaks, spectrum_id):
        """Convert list of PeakFit to FittedPeak list."""
        results = peak_fits_to_pydantic(sample_peaks, spectrum_id, "minerals")

        assert len(results) == 3
        assert all(isinstance(p, FittedPeak) for p in results)
        assert all(p.spectrum_id == spectrum_id for p in results)

    def test_batch_preserves_order(self, sample_peaks, spectrum_id):
        """Batch conversion preserves peak order."""
        results = peak_fits_to_pydantic(sample_peaks, spectrum_id, "minerals")

        assert [p.center_cm1 for p in results] == [282.0, 711.0, 1085.5]

    def test_batch_with_r2(self, sample_peaks, spectrum_id):
        """Batch conversion includes R^2."""
        results = peak_fits_to_pydantic(sample_peaks, spectrum_id, "minerals", r2=0.98)

        assert all(p.fit_quality == 0.98 for p in results)


class TestFitResultToPydantic:
    """Tests for core FitResult -> Pydantic FittingResult conversion."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_fit_result(self):
        """Create a sample FitResult dataclass."""
        peaks = [
            PeakFit(m_cm1=1085.5, a=1500.0, fwhm=25.0, sigma=10.6, area=39656.0, snr=15.2,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
        ]
        return FitResult(
            peaks=peaks,
            r2=0.995,
            rss=123.45,
            dof=97,
            warnings=[],
        )

    def test_basic_conversion(self, sample_fit_result, spectrum_id):
        """Convert FitResult to FittingResult."""
        result = fit_result_to_pydantic(sample_fit_result, spectrum_id)

        assert isinstance(result, FittingResult)
        assert result.spectrum_id == spectrum_id
        assert result.n_peaks == 1
        assert result.r_squared == 0.995

    def test_fitting_method_set(self, sample_fit_result, spectrum_id):
        """Fitting method is set to scipy_leastsq."""
        result = fit_result_to_pydantic(sample_fit_result, spectrum_id)

        assert result.fitting_method == "scipy_leastsq"

    def test_residual_rms_calculated(self, sample_fit_result, spectrum_id):
        """Residual RMS is calculated from RSS and DOF."""
        result = fit_result_to_pydantic(sample_fit_result, spectrum_id)

        expected_rms = math.sqrt(123.45 / 97)
        assert result.residual_rms == pytest.approx(expected_rms, rel=0.01)


class TestPydanticToPeakFit:
    """Tests for Pydantic FittedPeak -> core PeakFit conversion."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_basic_conversion(self, spectrum_id):
        """Convert FittedPeak to PeakFit."""
        pydantic_peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=35.0,  # Above 30 threshold
            snr=15.2,
            fit_quality=0.995,
        )

        result = pydantic_to_peak_fit(pydantic_peak)

        assert isinstance(result, PeakFit)
        assert result.m_cm1 == 1085.5
        assert result.a == 1500.0
        assert result.fwhm == 35.0
        assert result.snr == 15.2

    def test_pass_flags_set_correctly(self, spectrum_id):
        """Pass flags are computed from thresholds."""
        # Good peak - passes all
        good_peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=35.0,
            snr=5.0,
            fit_quality=0.5,
        )
        good_result = pydantic_to_peak_fit(good_peak)
        assert good_result.pass_snr is True
        assert good_result.pass_fwhm is True
        assert good_result.pass_r2 is True

        # Bad peak - fails SNR and FWHM
        bad_peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=20.0,  # Below 30 threshold
            snr=2.0,  # Below 3.0 threshold
            fit_quality=0.1,  # Below 0.25 threshold
        )
        bad_result = pydantic_to_peak_fit(bad_peak)
        assert bad_result.pass_snr is False
        assert bad_result.pass_fwhm is False
        assert bad_result.pass_r2 is False

    def test_area_calculated_if_missing(self, spectrum_id):
        """Area is calculated if not provided."""
        peak = FittedPeak(
            spectrum_id=spectrum_id,
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=100.0,
            fwhm_cm1=35.0,
        )
        result = pydantic_to_peak_fit(peak)

        # Should have calculated area
        assert result.area > 0


class TestPydanticToFitResult:
    """Tests for Pydantic FittingResult -> core FitResult conversion."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_basic_conversion(self, spectrum_id):
        """Convert FittingResult and peaks to FitResult."""
        pydantic_result = FittingResult(
            spectrum_id=spectrum_id,
            n_peaks=2,
            r_squared=0.98,
            residual_rms=1.5,
        )
        pydantic_peaks = [
            FittedPeak(spectrum_id=spectrum_id, fit_modality="minerals", center_cm1=282.0, amplitude=500.0, fwhm_cm1=35.0),
            FittedPeak(spectrum_id=spectrum_id, fit_modality="minerals", center_cm1=1085.5, amplitude=1500.0, fwhm_cm1=35.0),
        ]

        result = pydantic_to_fit_result(pydantic_result, pydantic_peaks)

        assert isinstance(result, FitResult)
        assert len(result.peaks) == 2
        assert result.r2 == 0.98

    def test_peaks_converted(self, spectrum_id):
        """Peaks in result are PeakFit instances."""
        pydantic_result = FittingResult(
            spectrum_id=spectrum_id,
            n_peaks=1,
            r_squared=0.95,
        )
        pydantic_peaks = [
            FittedPeak(spectrum_id=spectrum_id, fit_modality="minerals", center_cm1=1085.5, amplitude=1500.0, fwhm_cm1=35.0),
        ]

        result = pydantic_to_fit_result(pydantic_result, pydantic_peaks)

        assert all(isinstance(p, PeakFit) for p in result.peaks)


class TestRoundTrip:
    """Tests for round-trip conversion (core -> pydantic -> core)."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_peak_round_trip(self, spectrum_id):
        """PeakFit survives round-trip conversion."""
        original = PeakFit(
            m_cm1=1085.5,
            a=1500.0,
            fwhm=35.0,
            sigma=35.0 / (2.0 * math.sqrt(2.0 * math.log(2.0))),
            area=55000.0,
            snr=15.2,
            pass_snr=True,
            pass_fwhm=True,
            pass_r2=True,
        )

        # Core -> Pydantic -> Core
        pydantic = peak_fit_to_pydantic(original, spectrum_id, "minerals", r2=0.99)
        recovered = pydantic_to_peak_fit(pydantic)

        # Key properties preserved
        assert recovered.m_cm1 == original.m_cm1
        assert recovered.a == original.a
        assert recovered.fwhm == original.fwhm
        assert recovered.snr == original.snr

    def test_fit_result_round_trip(self, spectrum_id):
        """FitResult survives round-trip conversion."""
        original_peaks = [
            PeakFit(m_cm1=282.0, a=500.0, fwhm=35.0, sigma=14.9, area=18500.0, snr=8.0,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
            PeakFit(m_cm1=1085.5, a=1500.0, fwhm=35.0, sigma=14.9, area=55000.0, snr=15.2,
                    pass_snr=True, pass_fwhm=True, pass_r2=True),
        ]
        original = FitResult(
            peaks=original_peaks,
            r2=0.98,
            rss=100.0,
            dof=94,
            warnings=[],
        )

        # Core -> Pydantic
        pydantic_result = fit_result_to_pydantic(original, spectrum_id)
        pydantic_peaks = peak_fits_to_pydantic(original.peaks, spectrum_id, "minerals", original.r2)

        # Pydantic -> Core
        recovered = pydantic_to_fit_result(pydantic_result, pydantic_peaks)

        # Key properties preserved
        assert recovered.r2 == original.r2
        assert len(recovered.peaks) == len(original.peaks)
        assert [p.m_cm1 for p in recovered.peaks] == [p.m_cm1 for p in original.peaks]


class TestEdgeCases:
    """Tests for edge cases in conversion."""

    @pytest.fixture
    def spectrum_id(self):
        """Provide a spectrum UUID."""
        return uuid.uuid4()

    def test_empty_peaks_list(self, spectrum_id):
        """Handle empty peaks list."""
        result = peak_fits_to_pydantic([], spectrum_id, "minerals")
        assert result == []

    def test_fit_result_no_peaks(self, spectrum_id):
        """Handle FitResult with no peaks."""
        original = FitResult(
            peaks=[],
            r2=0.0,
            rss=float('inf'),
            dof=0,
            warnings=["no_peaks_detected"],
        )

        pydantic = fit_result_to_pydantic(original, spectrum_id)

        assert pydantic.n_peaks == 0

    def test_zero_snr_peak(self, spectrum_id):
        """Handle peak with zero SNR."""
        peak = PeakFit(
            m_cm1=1085.5,
            a=1500.0,
            fwhm=35.0,
            sigma=14.9,
            area=55000.0,
            snr=0.0,
            pass_snr=False,
            pass_fwhm=True,
            pass_r2=True,
        )

        result = peak_fit_to_pydantic(peak, spectrum_id, "minerals")
        assert result.snr == 0.0

    def test_negative_amplitude_peak(self, spectrum_id):
        """Handle peak with negative amplitude (absorption feature)."""
        peak = PeakFit(
            m_cm1=1085.5,
            a=-500.0,
            fwhm=35.0,
            sigma=14.9,
            area=-18500.0,
            snr=5.0,
            pass_snr=True,
            pass_fwhm=True,
            pass_r2=True,
        )

        result = peak_fit_to_pydantic(peak, spectrum_id, "minerals")
        assert result.amplitude == -500.0
