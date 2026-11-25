"""Workbench quality labels for fitted peaks."""
from __future__ import annotations


def classify_fit_quality(
    *,
    r_squared: float | None,
    fwhm: float | None,
    modality: str,
    fwhm_bounds: tuple[float, float] | None,
    sharpness_ratio: float | None = None,
    sharpness_max: float = 3.0,
) -> str:
    """Rate a peak using the fit R², sharpness, and the Raman fit's width bounds."""
    if r_squared is not None and r_squared < 0:
        return "fail"

    if sharpness_ratio is not None and sharpness_ratio >= sharpness_max:
        return "fail"

    if r_squared is None or r_squared < 0.5:
        return "review"

    if modality in {"minerals", "organics", "hydration"}:
        if fwhm is None or fwhm_bounds is None:
            return "review"
        lo, hi = fwhm_bounds
        if fwhm < lo or fwhm > hi:
            return "review"
    elif modality != "fluorescence":
        return "review"

    if r_squared >= 0.8:
        return "pass"

    return "review"
