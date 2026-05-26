"""Unit tests for ``services.quality.classify_fit_quality``.

Covers the v4.1.12 launch-blocker rules locked in
``_scratch/sessions/quality_classifier_fix_seed.md``: negative R² → fail,
sub-threshold R² → review, modality FWHM bounds, calibration-scan default,
sharpness cosmic-ray gate.

The two reproducer screenshots end up encoded as the negative-R² minerals
case (`minerals_pass_negative_r2.png`) and the unphysical-FWHM organics case
(`organics_pass_unphysical_fwhm.png`); both must NOT classify as "pass".
"""
from __future__ import annotations

import pytest

from sherloc_pipeline.services.quality import classify_fit_quality


# ---------------------------------------------------------------------------
# Hard fail gates (highest priority)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r2", [-0.001, -1.677, -4.487, -100.0])
def test_negative_r_squared_is_fail(r2: float) -> None:
    """Reproducer #2: minerals fit with R² < 0 must be ``fail``.

    Matches the four-decimal R² values shown in
    ``minerals_pass_negative_r2.png`` (-4.487, -1.677, ...). The classifier
    must reject these regardless of all other inputs — anti-correlation
    with the data has no defensible interpretation as a valid fit.
    """
    assert (
        classify_fit_quality(
            r_squared=r2,
            fwhm=30.0,
            modality="minerals",
            target_type="mars_target",  # even on a real target
        )
        == "fail"
    )


def test_negative_r_squared_beats_calibration_default() -> None:
    """Fail gate takes priority over the calibration → review downgrade."""
    assert (
        classify_fit_quality(
            r_squared=-2.0,
            fwhm=30.0,
            modality="minerals",
            target_type="cal_target",
        )
        == "fail"
    )


def test_sharpness_exceeding_threshold_is_fail() -> None:
    """Sharpness >= sharpness_max → fail (cosmic-ray hit; legacy semantics)."""
    assert (
        classify_fit_quality(
            r_squared=0.99,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
            sharpness_ratio=3.5,
        )
        == "fail"
    )


def test_sharpness_below_threshold_does_not_force_fail() -> None:
    assert (
        classify_fit_quality(
            r_squared=0.95,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
            sharpness_ratio=1.2,
        )
        == "pass"
    )


# ---------------------------------------------------------------------------
# Calibration / engineering downgrade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_type", ["cal_target", "engineering"])
def test_non_science_target_caps_at_review(target_type: str) -> None:
    """Reproducer scan ``external_calibration sol 70 meteorite_detail_1`` is
    ``cal_target``. Even with a textbook fit, no ground truth → ``review``.
    """
    assert (
        classify_fit_quality(
            r_squared=0.99,
            fwhm=25.0,
            modality="minerals",
            target_type=target_type,
        )
        == "review"
    )


def test_unknown_target_type_assumed_science() -> None:
    """``target_type=None`` (legacy / Workbench calling without scan ctx)
    must not block ``pass`` — otherwise the classifier silently downgrades
    every real-target fit through legacy callers.
    """
    assert (
        classify_fit_quality(
            r_squared=0.99,
            fwhm=25.0,
            modality="minerals",
            target_type=None,
        )
        == "pass"
    )


# ---------------------------------------------------------------------------
# R² thresholds
# ---------------------------------------------------------------------------


def test_r_squared_unknown_is_review() -> None:
    assert (
        classify_fit_quality(
            r_squared=None,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
        )
        == "review"
    )


@pytest.mark.parametrize("r2", [0.0, 0.1, 0.49])
def test_low_r_squared_is_review(r2: float) -> None:
    assert (
        classify_fit_quality(
            r_squared=r2,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
        )
        == "review"
    )


@pytest.mark.parametrize("r2", [0.5, 0.6, 0.79])
def test_mid_r_squared_is_review(r2: float) -> None:
    """0.5 ≤ R² < 0.8 → review even with everything else in bounds."""
    assert (
        classify_fit_quality(
            r_squared=r2,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
        )
        == "review"
    )


@pytest.mark.parametrize("r2", [0.80, 0.95, 0.999])
def test_high_r_squared_passes(r2: float) -> None:
    assert (
        classify_fit_quality(
            r_squared=r2,
            fwhm=25.0,
            modality="minerals",
            target_type="mars_target",
        )
        == "pass"
    )


# ---------------------------------------------------------------------------
# Modality FWHM bounds
# ---------------------------------------------------------------------------


def test_organics_unphysical_fwhm_is_review() -> None:
    """Reproducer #1: organics fit with FWHM 286.4 cm⁻¹ from
    ``organics_pass_unphysical_fwhm.png`` (broad fluorescence baseline
    mistaken for a peak). Organics bound is 10–60 cm⁻¹; 286 is ~5× too
    wide. Even with a high overall R² the per-peak FWHM gates this to
    ``review``.
    """
    assert (
        classify_fit_quality(
            r_squared=0.92,
            fwhm=286.4,
            modality="organics",
            target_type="mars_target",
        )
        == "review"
    )


def test_organics_fwhm_below_bound_is_review() -> None:
    assert (
        classify_fit_quality(
            r_squared=0.99,
            fwhm=5.0,  # below organics floor of 10
            modality="organics",
            target_type="mars_target",
        )
        == "review"
    )


def test_organics_fwhm_in_bound_passes() -> None:
    assert (
        classify_fit_quality(
            r_squared=0.95,
            fwhm=30.0,
            modality="organics",
            target_type="mars_target",
        )
        == "pass"
    )


def test_minerals_fwhm_in_bound_passes() -> None:
    """Carbonate-like narrow band (~7 cm⁻¹) on a real target."""
    assert (
        classify_fit_quality(
            r_squared=0.92,
            fwhm=7.0,
            modality="minerals",
            target_type="mars_target",
        )
        == "pass"
    )


def test_minerals_fwhm_out_of_bound_is_review() -> None:
    assert (
        classify_fit_quality(
            r_squared=0.99,
            fwhm=150.0,  # above minerals ceiling of 100
            modality="minerals",
            target_type="mars_target",
        )
        == "review"
    )


def test_hydration_broad_fwhm_passes() -> None:
    """Hydration OH stretch envelope is intrinsically broad."""
    assert (
        classify_fit_quality(
            r_squared=0.90,
            fwhm=400.0,
            modality="hydration",
            target_type="mars_target",
        )
        == "pass"
    )


def test_fluorescence_skips_fwhm_gate() -> None:
    """No nm-space FWHM bounds shipped in v4.1.12 — the existing fluorescence
    fitter already enforces ``min_fwhm_nm`` at fit time. Classifier should not
    double-gate.
    """
    assert (
        classify_fit_quality(
            r_squared=0.95,
            fwhm=8.0,  # nm
            modality="fluorescence",
            target_type="mars_target",
        )
        == "pass"
    )


# ---------------------------------------------------------------------------
# Missing-data robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modality", ["minerals", "organics", "hydration"])
def test_fwhm_none_for_bounded_modality_is_review(modality: str) -> None:
    """Seed §32 locks the default to Review; ``pass`` must be earned by
    positive evidence including a plausible FWHM. A bounded Raman modality
    with no FWHM is absence of evidence, not evidence of plausibility — it
    must NOT auto-pass on R² alone.
    """
    assert (
        classify_fit_quality(
            r_squared=0.95,
            fwhm=None,
            modality=modality,
            target_type="mars_target",
        )
        == "review"
    )


def test_unknown_modality_is_review() -> None:
    """Defense in depth: a modality the classifier doesn't recognize is a
    silent path to pass under the old logic. Default it to review.
    """
    assert (
        classify_fit_quality(
            r_squared=0.95,
            fwhm=25.0,
            modality="surprise_modality",
            target_type="mars_target",
        )
        == "review"
    )
