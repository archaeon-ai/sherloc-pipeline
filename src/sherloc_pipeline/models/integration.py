"""
Pydantic integration for existing pipeline fitting structures.

This module provides conversion utilities between the legacy dataclass-based
fitting results (PeakFit, FitResult) in core.fitting and the Pydantic models
(FittedPeak, FittingResult) in models.fitting.

The integration enables:
- Backward compatibility: existing code using dataclasses continues to work
- Database persistence: Pydantic models can be stored via SQLAlchemy ORM
- Validation: Pydantic models provide automatic validation
- Serialization: Pydantic models support JSON/dict serialization

Usage:
    # Convert from core types to Pydantic
    >>> from sherloc_pipeline.core.fitting import fit_spectrum, FitResult, PeakFit
    >>> from sherloc_pipeline.models.integration import (
    ...     peak_fit_to_pydantic,
    ...     fit_result_to_pydantic,
    ...     peak_fits_to_pydantic,
    ... )
    >>>
    >>> result, y_model = fit_spectrum(x, y, cfg)
    >>> pydantic_peaks = peak_fits_to_pydantic(result.peaks, spectrum_id)
    >>> pydantic_result = fit_result_to_pydantic(result, spectrum_id)

    # Convert from Pydantic back to core types
    >>> from sherloc_pipeline.models.integration import (
    ...     pydantic_to_peak_fit,
    ...     pydantic_to_fit_result,
    ... )
    >>>
    >>> core_peak = pydantic_to_peak_fit(pydantic_peak)
    >>> core_result = pydantic_to_fit_result(pydantic_result, pydantic_peaks)
"""

from typing import List, Literal, Optional
import uuid
import math

from sherloc_pipeline.models.fitting import (
    FittedPeak,
    FitResult,
    FittingResult,
    PeakFit,
    PeakType,
)


def fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to Gaussian sigma."""
    return float(fwhm) / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def peak_fit_to_pydantic(
    peak,  # PeakFit from core.fitting
    spectrum_id: uuid.UUID,
    fit_modality: Literal["minerals", "organics", "hydration", "fluorescence"],
    r2: Optional[float] = None,
) -> FittedPeak:
    """Convert a core.fitting.PeakFit dataclass to Pydantic FittedPeak.

    Args:
        peak: PeakFit dataclass instance from core.fitting
        spectrum_id: UUID of the parent Spectrum for database linking
        fit_modality: Domain discriminator (required, no default)
        r2: Optional R^2 value from the fitting result for fit_quality

    Returns:
        FittedPeak Pydantic model instance

    Example:
        >>> from sherloc_pipeline.core.fitting import fit_spectrum
        >>> result, _ = fit_spectrum(x, y, cfg)
        >>> for peak in result.peaks:
        ...     pydantic_peak = peak_fit_to_pydantic(peak, spectrum_id, "minerals", result.r2)
    """
    return FittedPeak(
        spectrum_id=spectrum_id,
        peak_type=PeakType.GAUSSIAN,  # core.fitting only supports Gaussian
        fit_modality=fit_modality,
        center_cm1=peak.m_cm1,
        amplitude=peak.a,
        fwhm_cm1=peak.fwhm,
        area=peak.area,
        snr=peak.snr,
        fit_quality=r2 if r2 is not None else None,
    )


def peak_fits_to_pydantic(
    peaks: List,  # List[PeakFit]
    spectrum_id: uuid.UUID,
    fit_modality: Literal["minerals", "organics", "hydration", "fluorescence"],
    r2: Optional[float] = None,
) -> List[FittedPeak]:
    """Convert a list of core.fitting.PeakFit to Pydantic FittedPeak list.

    Args:
        peaks: List of PeakFit dataclass instances from core.fitting
        spectrum_id: UUID of the parent Spectrum
        fit_modality: Domain discriminator (required, no default)
        r2: Optional R^2 value from the fitting result

    Returns:
        List of FittedPeak Pydantic model instances
    """
    return [peak_fit_to_pydantic(p, spectrum_id, fit_modality, r2) for p in peaks]


def fit_result_to_pydantic(
    result,  # FitResult from core.fitting
    spectrum_id: uuid.UUID,
) -> FittingResult:
    """Convert a core.fitting.FitResult dataclass to Pydantic FittingResult.

    Args:
        result: FitResult dataclass instance from core.fitting
        spectrum_id: UUID of the parent Spectrum

    Returns:
        FittingResult Pydantic model instance

    Note:
        Individual peaks should be converted separately using
        peak_fits_to_pydantic() and stored as FittedPeak records.
    """
    return FittingResult(
        spectrum_id=spectrum_id,
        n_peaks=len(result.peaks) if result.peaks else 0,
        r_squared=result.r2 if result.r2 is not None else None,
        residual_rms=math.sqrt(result.rss / max(result.dof, 1)) if result.rss > 0 and result.dof > 0 else None,
        fitting_method="scipy_leastsq",  # core.fitting uses scipy.optimize.least_squares
    )


def pydantic_to_peak_fit(peak: FittedPeak):
    """Convert a Pydantic FittedPeak to core.fitting.PeakFit dataclass.

    This enables round-trip conversion when loading from database.

    Args:
        peak: FittedPeak Pydantic model instance

    Returns:
        PeakFit dataclass instance compatible with core.fitting functions

    Note:
        Import is deferred to avoid circular imports with core.fitting.
    """
    if peak.center_cm1 is None or peak.fwhm_cm1 is None:
        raise ValueError(
            f"Cannot convert {peak.fit_modality} peak to PeakFit: "
            "center_cm1 and fwhm_cm1 are required for core PeakFit"
        )
    return PeakFit(
        m_cm1=peak.center_cm1,
        a=peak.amplitude,
        fwhm=peak.fwhm_cm1,
        sigma=fwhm_to_sigma(peak.fwhm_cm1),
        area=peak.area if peak.area is not None else peak.calculate_area(),
        snr=peak.snr if peak.snr is not None else 0.0,
        pass_snr=peak.snr is not None and peak.snr >= 3.0,
        pass_fwhm=peak.fwhm_cm1 >= 30.0,  # Default threshold
        pass_r2=peak.fit_quality is not None and peak.fit_quality >= 0.25,
    )


def pydantic_to_fit_result(
    result: FittingResult,
    peaks: List[FittedPeak],
):
    """Convert Pydantic FittingResult and FittedPeak list to core.fitting.FitResult.

    This enables round-trip conversion when loading from database.

    Args:
        result: FittingResult Pydantic model instance
        peaks: List of FittedPeak Pydantic model instances

    Returns:
        FitResult dataclass instance compatible with core.fitting functions

    Note:
        Import is deferred to avoid circular imports with core.fitting.
    """
    core_peaks = [pydantic_to_peak_fit(p) for p in peaks]

    return FitResult(
        peaks=core_peaks,
        r2=result.r_squared if result.r_squared is not None else 0.0,
        rss=(result.residual_rms ** 2 * result.n_peaks) if result.residual_rms is not None else 0.0,
        dof=result.n_peaks * 3 if result.n_peaks > 0 else 0,
        warnings=[],
    )


# Compatibility aliases for ease of migration
CoreToFittedPeak = peak_fit_to_pydantic
CoreToFittingResult = fit_result_to_pydantic
FittedPeakToCore = pydantic_to_peak_fit
FittingResultToCore = pydantic_to_fit_result


__all__ = [
    # Primary conversion functions
    "peak_fit_to_pydantic",
    "peak_fits_to_pydantic",
    "fit_result_to_pydantic",
    "pydantic_to_peak_fit",
    "pydantic_to_fit_result",
    # Aliases
    "CoreToFittedPeak",
    "CoreToFittingResult",
    "FittedPeakToCore",
    "FittingResultToCore",
    # Utility
    "fwhm_to_sigma",
]
