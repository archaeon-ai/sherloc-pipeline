"""Tests for is_fittable() — fit-eligibility decoupled from target_type.

Covers SCAN_CLASSIFICATION_SPEC §4.2.2 + Key Decision K6:
- mars_target is always fit-eligible
- the cal-target meteorite (SaU 008) is fit-eligible — identified by either
  scan_name (mixed cal sol) or target (dedicated meteorite sol)
- every other cal target (engineered/synthetic standard, diagnostic) is excluded
- engineering scans are excluded even on a meteorite sol
- end-to-end parity: classify_target_type() → is_fittable() on real-shaped data
"""

import pytest

from sherloc_pipeline.models.spectra import (
    classify_target_type,
    is_fittable,
)


class TestIsFittableMarsTarget:
    """All exploration-area rock/regolith is fit-eligible."""

    @pytest.mark.parametrize("scan_name", [
        "detail_1", "survey_1296", "line_1", "HDR_500",
        "detail_2_sum_active_median_dark", None, "",
    ])
    def test_mars_target_always_fittable(self, scan_name):
        assert is_fittable("mars_target", "Amherst Point", scan_name) is True

    def test_mars_target_ignores_target_name(self):
        # type is authoritative for the mars_target branch
        assert is_fittable("mars_target", None, "detail_1") is True


class TestIsFittableMeteorite:
    """The cal-target meteorite (SaU 008) is the one fit-worthy cal sample."""

    # --- mixed cal sol: target == "external calibration", named by material ---

    @pytest.mark.parametrize("scan_name", [
        "meteorite",
        "meteorite_detail_1",
        "meteorite_median_all",
        "meteorite_sum_active_median_dark",
        "meteorite_sum_active_sum_dark",
        "MarsMeteorite_1",          # case-insensitive: lower() contains "meteorite"
    ])
    def test_meteorite_by_scan_name(self, scan_name):
        assert is_fittable("cal_target", "external calibration", scan_name) is True

    def test_meteorite_scan_name_case_insensitive(self):
        assert is_fittable("cal_target", "external calibration", "METEORITE") is True

    # --- dedicated meteorite sol: target == "ext cal meteorite", generic names ---

    @pytest.mark.parametrize("scan_name", [
        "detail_1", "detail_2", "detail_3", "detail_4", "detail_5",
    ])
    def test_meteorite_by_target(self, scan_name):
        assert is_fittable("cal_target", "ext cal meteorite", scan_name) is True

    def test_meteorite_target_case_insensitive(self):
        assert is_fittable("cal_target", "Ext Cal Meteorite", "detail_1") is True


class TestIsFittableExclusions:
    """Engineered standards, diagnostics, and engineering scans are NOT fit."""

    @pytest.mark.parametrize("scan_name", [
        "AlGaN_1", "AlGaN340_pos1", "teflon_1", "maze", "orthofabric",
        "polycarbonate", "polycarbonate_sum_active_median_dark", "passive_diffusil",
    ])
    def test_engineered_cal_standards_excluded(self, scan_name):
        # mixed cal sol, non-meteorite material name → not fit
        assert is_fittable("cal_target", "external calibration", scan_name) is False

    @pytest.mark.parametrize("target", [
        "external calibration", "teflon calibration", "algan340 calibration",
        "maze calibration", "passive diffusil", "calibration",
    ])
    def test_non_meteorite_cal_targets_excluded(self, target):
        assert is_fittable("cal_target", target, "detail_1") is False

    def test_engineering_excluded_even_on_meteorite_sol(self):
        # power_on classifies as engineering even on a sol whose sol-wide
        # target is "ext cal meteorite" — the cal_target guard drops it.
        assert is_fittable("engineering", "ext cal meteorite", "power_on") is False

    @pytest.mark.parametrize("scan_name", ["power_on", "power_off", "detail_1"])
    def test_engineering_type_never_fittable(self, scan_name):
        assert is_fittable("engineering", "conjunction", scan_name) is False

    def test_unknown_type_not_fittable(self):
        assert is_fittable(None, "external calibration", "meteorite") is False
        assert is_fittable("", None, None) is False


class TestIsFittableNullSafety:
    """None/empty inputs must not raise."""

    def test_all_none(self):
        assert is_fittable(None, None, None) is False

    def test_cal_target_none_names(self):
        assert is_fittable("cal_target", None, None) is False


class TestClassifyThenFittableParity:
    """End-to-end: classify_target_type() → is_fittable() on real-shaped pairs.

    This is the actual production flow — target_type is derived, then fed to
    is_fittable. Asserts the predicate selects exactly the meteorite science
    scans and nothing else.
    """

    # (target, scan_name, expected_fittable)
    PARITY_FIXTURES = [
        # mars_target science → fit
        ("Amherst Point", "detail_1", True),
        ("Berry Hollow", "HDR_1", True),
        ("cat arm reservoir", "detail_1", True),
        # mixed cal sol meteorite (by scan_name) → fit
        ("external calibration", "meteorite", True),
        ("external calibration", "meteorite_sum_active_median_dark", True),
        ("external calibration", "MarsMeteorite_1", True),
        # dedicated meteorite sol (by target) → fit
        ("ext cal meteorite", "detail_1", True),
        ("ext cal meteorite", "detail_3", True),
        # engineered cal standards → NOT fit
        ("external calibration", "AlGaN_1", False),
        ("teflon calibration", "detail_1", False),
        ("external calibration", "maze", False),
        ("external calibration", "polycarbonate", False),
        # engineering → NOT fit (power_* drops even on a meteorite sol)
        ("ext cal meteorite", "power_on", False),
        ("Amherst Point", "power_on", False),
        ("conjunction", "detail_1", False),
        (None, "detail_1", False),
    ]

    @pytest.mark.parametrize("target,scan_name,expected", PARITY_FIXTURES)
    def test_classify_then_fittable(self, target, scan_name, expected):
        target_type = classify_target_type(target, scan_name)
        result = is_fittable(target_type, target, scan_name)
        assert result is expected, (
            f"is_fittable(classify_target_type({target!r}, {scan_name!r})="
            f"{target_type!r}, {target!r}, {scan_name!r}) = {result!r}, "
            f"expected {expected!r}"
        )
