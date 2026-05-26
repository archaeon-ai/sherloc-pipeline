"""
Peak fitting models for PHASE.

This module defines models for spectral peak fitting results:
- FittedPeak: A single fitted peak with position, amplitude, and quality metrics
- PeakType: Enumeration of supported peak profile types

Peak fitting is a key analysis step that identifies spectral features
and enables mineral identification through peak position matching.

Example:
    >>> from sherloc_pipeline.models.fitting import FittedPeak, PeakType
    >>>
    >>> peak = FittedPeak(
    ...     spectrum_id=spectrum.id,
    ...     peak_type=PeakType.GAUSSIAN,
    ...     center_cm1=1085.5,
    ...     amplitude=1500.0,
    ...     fwhm_cm1=25.0,
    ...     snr=15.2,
    ... )
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional
import uuid

from pydantic import Field, field_validator, model_validator

from sherloc_pipeline.models.base import (
    IdentifiableModel,
    ModelRegistry,
)


@dataclass
class PeakFit:
    m_cm1: float            # peak position/mean in cm^-1
    a: float                # amplitude (peak height)
    fwhm: float             # full width at half maximum (cm^-1)
    sigma: float            # standard deviation corresponding to fwhm
    area: float             # area under the Gaussian
    snr: float              # signal-to-noise ratio estimate
    pass_snr: bool
    pass_fwhm: bool
    pass_r2: bool
    sharpness_ratio: float = 1.0   # data_at_center / amplitude (>>1 → cosmic ray)
    pass_sharpness: bool = True    # False if sharpness_ratio exceeds threshold


@dataclass
class FitResult:
    peaks: List[PeakFit]
    r2: float
    rss: float
    dof: int
    warnings: List[str]


class PeakType(str, Enum):
    """Type of peak profile function.

    - gaussian: Gaussian (normal) peak shape
    - lorentzian: Lorentzian (Cauchy) peak shape
    - voigt: Voigt profile (Gaussian-Lorentzian convolution)
    - pseudo_voigt: Pseudo-Voigt (weighted sum of Gaussian and Lorentzian)
    """
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"


@ModelRegistry.register
class FittedPeak(IdentifiableModel):
    """A fitted peak from spectral analysis.

    FittedPeak stores the results of fitting a peak profile to a
    spectral feature. This includes the peak position, amplitude,
    width, and quality metrics like SNR and fit quality.

    Mineral assignments can be added based on peak position matching
    against known mineral spectral libraries.

    Attributes:
        spectrum_id: UUID of parent Spectrum
        peak_type: Type of peak profile (gaussian, lorentzian, etc.)
        center_cm1: Peak center position in cm^-1
        center_uncertainty: Uncertainty in peak center (optional)
        amplitude: Peak amplitude (height)
        amplitude_uncertainty: Uncertainty in amplitude (optional)
        fwhm_cm1: Full width at half maximum in cm^-1
        fwhm_uncertainty: Uncertainty in FWHM (optional)
        area: Integrated peak area (optional)
        snr: Signal-to-noise ratio (optional)
        fit_quality: Goodness of fit (R^2, 0-1) (optional)
        mineral_assignment: Identified mineral name (optional)
        assignment_confidence: Confidence in mineral assignment (0-1)

    Example:
        >>> peak = FittedPeak(
        ...     spectrum_id=spectrum.id,
        ...     peak_type=PeakType.GAUSSIAN,
        ...     center_cm1=1085.5,
        ...     amplitude=1500.0,
        ...     fwhm_cm1=25.0,
        ...     snr=15.2,
        ...     mineral_assignment="calcite",
        ...     assignment_confidence=0.92,
        ... )
        >>> peak.center_cm1
        1085.5
        >>> peak.mineral_assignment
        'calcite'
    """

    spectrum_id: uuid.UUID = Field(
        description="UUID of parent Spectrum"
    )
    peak_type: PeakType = Field(
        default=PeakType.GAUSSIAN,
        description="Type of peak profile function"
    )

    # Domain discriminator (no default — callers must always set explicitly)
    fit_modality: Literal["minerals", "organics", "hydration", "fluorescence"] = Field(
        description="Fit domain: minerals, organics, hydration, or fluorescence"
    )

    # Peak position (Raman: cm1 required; Fluorescence: nm required)
    center_cm1: Optional[float] = Field(
        default=None,
        description="Peak center position in cm^-1 (Raman peaks)"
    )
    center_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        description="Uncertainty in peak center"
    )
    center_nm: Optional[float] = Field(
        default=None,
        description="Peak center position in nm (fluorescence peaks)"
    )

    # Peak amplitude
    amplitude: float = Field(
        description="Peak amplitude (height)"
    )
    amplitude_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        description="Uncertainty in amplitude"
    )

    # Peak width (Raman: cm1 required; Fluorescence: nm required)
    fwhm_cm1: Optional[float] = Field(
        default=None,
        gt=0,
        description="Full width at half maximum in cm^-1 (Raman peaks)"
    )
    fwhm_uncertainty: Optional[float] = Field(
        default=None,
        ge=0,
        description="Uncertainty in FWHM"
    )
    fwhm_nm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Full width at half maximum in nm (fluorescence peaks)"
    )

    # Saturation flag (fluorescence)
    is_saturated: Optional[bool] = Field(
        default=None,
        description="Whether peak is saturated (fluorescence)"
    )

    # Derived quantities
    area: Optional[float] = Field(
        default=None,
        description="Integrated peak area"
    )
    snr: Optional[float] = Field(
        default=None,
        description="Signal-to-noise ratio"
    )
    fit_quality: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Goodness of fit (R^2, 0-1)"
    )

    # Mineral identification
    mineral_assignment: Optional[str] = Field(
        default=None,
        description="Identified mineral name"
    )
    assignment_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence in mineral assignment (0-1)"
    )

    @field_validator("center_cm1")
    @classmethod
    def validate_center_range(cls, v: Optional[float]) -> Optional[float]:
        """Validate that peak center is in reasonable Raman range.

        Only fires for Raman peaks (where center_cm1 is non-None).
        """
        if v is None:
            return v
        if v < 0 or v > 10000:
            raise ValueError(
                f"Peak center {v} cm^-1 is outside expected range (0-10000)"
            )
        return v

    @field_validator("fwhm_cm1")
    @classmethod
    def validate_fwhm_reasonable(cls, v: Optional[float]) -> Optional[float]:
        """Validate that FWHM is reasonable.

        Only fires for Raman peaks (where fwhm_cm1 is non-None).
        """
        if v is None:
            return v
        if v > 500:
            raise ValueError(
                f"FWHM {v} cm^-1 is unusually large (>500)"
            )
        return v

    @model_validator(mode="after")
    def validate_domain_fields(self) -> "FittedPeak":
        """Ensure correct unit fields are populated for each domain."""
        if self.fit_modality in ("minerals", "organics", "hydration"):
            if self.center_cm1 is None or self.fwhm_cm1 is None:
                raise ValueError(
                    f"Raman peaks ({self.fit_modality}) require center_cm1 and fwhm_cm1"
                )
        elif self.fit_modality == "fluorescence":
            if self.center_nm is None or self.fwhm_nm is None:
                raise ValueError(
                    "Fluorescence peaks require center_nm and fwhm_nm"
                )
        return self

    @model_validator(mode="after")
    def validate_assignment_with_confidence(self) -> "FittedPeak":
        """Validate mineral assignment and confidence pairing.

        No auto-default for assignment_confidence — callers must set
        it explicitly when providing a mineral_assignment.
        """
        return self

    def calculate_area(self) -> float:
        """Calculate integrated peak area.

        For a Gaussian peak:
            Area = amplitude * fwhm * sqrt(pi / (4 * ln(2)))
            Area approx amplitude * fwhm * 1.0645

        Returns:
            Integrated peak area
        """
        import math
        if self.peak_type == PeakType.GAUSSIAN:
            factor = math.sqrt(math.pi / (4 * math.log(2)))
        elif self.peak_type == PeakType.LORENTZIAN:
            factor = math.pi / 2
        else:
            # Approximate factor for Voigt-like profiles
            factor = 1.5
        return self.amplitude * self.fwhm_cm1 * factor

    def is_significant(self, min_snr: float = 3.0) -> bool:
        """Check if peak is statistically significant.

        Args:
            min_snr: Minimum SNR threshold (default 3.0)

        Returns:
            True if peak SNR exceeds threshold
        """
        if self.snr is None:
            return True  # Assume significant if SNR not calculated
        return self.snr >= min_snr

    def matches_mineral(
        self,
        mineral_peaks: dict,
        tolerance_cm1: float = 10.0,
    ) -> Optional[str]:
        """Check if peak matches any known mineral peak positions.

        Args:
            mineral_peaks: Dict mapping mineral names to lists of peak positions
            tolerance_cm1: Position tolerance for matching

        Returns:
            Matched mineral name, or None if no match

        Example:
            >>> mineral_db = {
            ...     "calcite": [1085.0, 282.0],
            ...     "gypsum": [1008.0, 414.0],
            ... }
            >>> peak.matches_mineral(mineral_db, tolerance_cm1=15.0)
            'calcite'
        """
        for mineral, positions in mineral_peaks.items():
            for pos in positions:
                if abs(self.center_cm1 - pos) <= tolerance_cm1:
                    return mineral
        return None


class FittingResult(IdentifiableModel):
    """Summary of a multi-peak fitting result.

    FittingResult captures metadata about a fitting session,
    including the number of peaks found and overall fit quality.

    Attributes:
        spectrum_id: UUID of the fitted Spectrum
        n_peaks: Number of peaks identified
        residual_rms: RMS of fit residuals
        r_squared: Overall R^2 of the fit
        chi_squared: Chi-squared statistic
        fitting_method: Algorithm used for fitting
        config_hash: Hash of fitting configuration

    Note:
        Individual peaks are stored as FittedPeak records linked
        to the same spectrum_id. This model provides summary stats.
    """

    spectrum_id: uuid.UUID = Field(
        description="UUID of the fitted Spectrum"
    )
    n_peaks: int = Field(
        ge=0,
        description="Number of peaks identified"
    )
    residual_rms: Optional[float] = Field(
        default=None,
        ge=0,
        description="RMS of fit residuals"
    )
    r_squared: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Overall R^2 of the fit"
    )
    chi_squared: Optional[float] = Field(
        default=None,
        ge=0,
        description="Chi-squared statistic"
    )
    fitting_method: str = Field(
        default="lmfit",
        description="Algorithm used for fitting"
    )
    config_hash: Optional[str] = Field(
        default=None,
        description="Hash of fitting configuration for reproducibility"
    )
