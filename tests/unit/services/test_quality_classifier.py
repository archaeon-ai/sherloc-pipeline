"""Workbench quality labels for synthetic peak measurements."""

import pytest

from sherloc_pipeline.services.quality import classify_fit_quality


@pytest.mark.parametrize(("r2", "expected"), [
    (-1.0, "fail"), (-0.001, "fail"), (None, "review"), (0.0, "review"),
    (0.49, "review"), (0.5, "review"), (0.79, "review"), (0.8, "pass"), (0.99, "pass"),
])
def test_fit_r_squared(r2, expected):
    assert classify_fit_quality(
        r_squared=r2, modality="minerals", fwhm=35, fwhm_bounds=(22, 90),
    ) == expected


@pytest.mark.parametrize(("sharpness", "expected"), [(1.9, "pass"), (2.0, "fail"), (3.5, "fail")])
def test_fit_sharpness(sharpness, expected):
    assert classify_fit_quality(
        r_squared=0.95, modality="minerals", fwhm=35, fwhm_bounds=(22, 90),
        sharpness_ratio=sharpness, sharpness_max=2.0,
    ) == expected


@pytest.mark.parametrize("domain", ["minerals", "organics", "hydration"])
@pytest.mark.parametrize(("width", "expected"), [
    (None, "review"), (39.9, "review"), (40, "pass"), (150, "pass"), (150.1, "review"),
])
def test_raman_width_uses_inclusive_fit_bounds(domain, width, expected):
    assert classify_fit_quality(
        r_squared=0.95, modality=domain, fwhm=width, fwhm_bounds=(40, 150),
    ) == expected


def test_fluorescence_skips_raman_width_check():
    assert classify_fit_quality(
        r_squared=0.95, modality="fluorescence", fwhm=8, fwhm_bounds=None,
    ) == "pass"


def test_unknown_modality_is_review():
    assert classify_fit_quality(
        r_squared=0.95, modality="unknown", fwhm=35, fwhm_bounds=(22, 90),
    ) == "review"
