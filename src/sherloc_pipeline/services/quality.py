"""Workbench fit-quality classifier.

Maps a single fitted peak's numerical descriptors (overall fit R², per-peak
FWHM, modality, scan target type, cosmic-ray sharpness) to a tri-state
display label: ``"pass"``, ``"review"``, or ``"fail"``.

Conservative-by-design: peaks default to ``"review"``; ``"pass"`` must be
earned by positive evidence (high R² + physically plausible FWHM + a
real-target scan context). See ``docs/specs/PUBLIC_RELEASE_PREP_SPEC-revised2.md``
and v4.1.12 launch-blocker seed
``_scratch/sessions/quality_classifier_fix_seed.md`` for rationale.

Why a separate module:
- The legacy ``pass_sharpness`` boolean (``core/fitting.py``) gates cosmic-ray
  contamination only. It does not catch unphysical FWHM or anti-correlated
  fits (R² < 0), which is the v4.1.12 launch-blocker failure mode.
- Centralizing the rule here lets both the Workbench ``/api/process/fit``
  response shaping and any future persisted-peak reader apply identical
  semantics.
"""
from __future__ import annotations

from typing import Optional


# Modality-specific FWHM plausibility bounds in cm⁻¹ (Raman) or nm (fluorescence).
# Conservative literature-grounded starting points for v4.1.12; refine in v1.0.1
# with real-target performance data per seed §54.
FWHM_BOUNDS_CM1: dict[str, tuple[float, float]] = {
    "minerals":  (5.0, 100.0),    # mineral Raman bands, e.g. carbonate ν1 ~3–10
    "organics":  (10.0, 60.0),    # D/G bands, CH stretches
    "hydration": (50.0, 600.0),   # broad OH stretch envelope
}
# Fluorescence is in nm space and the existing fluorescence fitter already
# enforces a min_fwhm_nm gate at fit time; we do not double-gate here.


# Calibration / engineering target types never auto-promote to "pass" because
# there is no Mars-target ground truth being measured. See seed §40 and
# models/spectra.py:TargetType.
_NON_SCIENCE_TARGET_TYPES = frozenset({"cal_target", "engineering"})


def classify_fit_quality(
    *,
    r_squared: Optional[float],
    fwhm: Optional[float],
    modality: str,
    target_type: Optional[str] = None,
    sharpness_ratio: Optional[float] = None,
    sharpness_max: float = 3.0,
) -> str:
    """Classify a single fitted peak as ``"pass"`` | ``"review"`` | ``"fail"``.

    Args:
        r_squared: Overall fit R². ``None`` → ``"review"`` (cannot prove pass).
            R² < 0 is anti-correlation with the data (worse than predicting
            the mean) — hard ``"fail"``.
        fwhm: Per-peak FWHM in cm⁻¹ for Raman modalities, nm for fluorescence.
        modality: One of ``"minerals"``, ``"organics"``, ``"hydration"``,
            ``"fluorescence"``. Selects the FWHM plausibility band.
        target_type: Optional scan target_type from ``models/spectra.py``.
            ``"cal_target"`` / ``"engineering"`` downgrade to ``"review"``
            regardless of fit quality (no Mars-target ground truth).
        sharpness_ratio: Per-peak cosmic-ray descriptor
            (``data_at_center / amplitude``). ``>= sharpness_max`` → ``"fail"``
            (matches legacy ``pass_sharpness`` semantics).
        sharpness_max: Threshold for the cosmic-ray fail gate. Defaults to
            the ``config.yaml`` ``posthoc_filters.sharpness_max`` value (3.0).

    Returns:
        ``"pass"``, ``"review"``, or ``"fail"``.

    Resolution order (highest priority first — first match wins):
      1. **fail** — ``r_squared < 0`` (anti-correlation, no defensible
         interpretation).
      2. **fail** — sharpness exceeds ``sharpness_max`` (cosmic-ray hit on
         the peak center).
      3. **review** — non-science target type (calibration/engineering).
      4. **review** — ``r_squared`` is unknown or ``< 0.5``.
      5. **review** — FWHM outside modality-specific plausibility band.
      6. **pass** — ``r_squared >= 0.8`` and all gates above passed.
      7. **review** — otherwise (i.e. ``0.5 <= r_squared < 0.8`` with no
         other failures).
    """
    if r_squared is not None and r_squared < 0:
        return "fail"

    if sharpness_ratio is not None and sharpness_ratio >= sharpness_max:
        return "fail"

    if target_type in _NON_SCIENCE_TARGET_TYPES:
        return "review"

    if r_squared is None or r_squared < 0.5:
        return "review"

    # FWHM-bound gate: when the modality has bounds (i.e. is Raman), absence
    # of an FWHM value is absence of positive evidence — seed §32 locks the
    # default to Review unless the fit earns pass. Fluorescence (and any
    # future bounds-exempt modality) skips this gate entirely; the existing
    # fluorescence fitter enforces min_fwhm_nm at fit time.
    if modality in FWHM_BOUNDS_CM1:
        bounds = FWHM_BOUNDS_CM1[modality]
        if fwhm is None:
            return "review"
        lo, hi = bounds
        if fwhm < lo or fwhm > hi:
            return "review"
    elif modality not in {"fluorescence"}:
        # Unknown modality: no evidence the FWHM is plausible. Default to
        # Review rather than silently letting it through on R² alone.
        return "review"

    if r_squared >= 0.8:
        return "pass"

    return "review"
