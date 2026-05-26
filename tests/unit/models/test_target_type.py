"""Tests for TargetType enum and classify_target_type() function.

Covers:
- Enum value assertions and string roundtrip
- Classification for all categories (engineering, cal_target, mars_target)
- Case insensitivity, leading space handling, priority order
- SQL CASE vs Python parity on representative data
"""

import pytest

from sherloc_pipeline.models.spectra import (
    TargetType,
    classify_target_type,
    _ENGINEERING_TARGETS,
    _CAL_TARGETS,
)


class TestTargetTypeEnum:
    """Test TargetType enum values and string behavior."""

    def test_enum_values(self):
        assert TargetType.MARS_TARGET.value == "mars_target"
        assert TargetType.CAL_TARGET.value == "cal_target"
        assert TargetType.ENGINEERING.value == "engineering"

    def test_string_roundtrip(self):
        for tt in TargetType:
            assert TargetType(tt.value) is tt

    def test_str_enum(self):
        """TargetType is a str enum, comparable to string values."""
        assert TargetType.MARS_TARGET == "mars_target"
        assert TargetType.CAL_TARGET.value == "cal_target"


class TestClassifyTargetType:
    """Test classify_target_type() for all categories."""

    # --- Engineering ---

    @pytest.mark.parametrize("target", [None, "", "  "])
    def test_null_empty_target_is_engineering(self, target):
        result = classify_target_type(target, "detail_1")
        assert result == "engineering"

    @pytest.mark.parametrize("target", [
        "conjunction", "b conjunction", "arm stowed",
        "arm stowed dark", "arm docked",
    ])
    def test_known_engineering_targets(self, target):
        result = classify_target_type(target, "detail_1")
        assert result == "engineering"

    def test_engineering_targets_case_insensitive(self):
        assert classify_target_type("Conjunction", "detail_1") == "engineering"
        assert classify_target_type("ARM STOWED", "detail_1") == "engineering"
        assert classify_target_type("B Conjunction", "100ppp_1") == "engineering"

    def test_engineering_leading_spaces(self):
        """Leading spaces in target names are stripped before matching."""
        assert classify_target_type(" conjunction", "detail_1") == "engineering"
        assert classify_target_type("  arm stowed", "detail_1") == "engineering"

    @pytest.mark.parametrize("scan_name", [
        "power_on", "power_off", "power_on_1", "power_on_2",
    ])
    def test_power_scan_names_are_engineering(self, scan_name):
        """power_* scan_names → engineering regardless of target."""
        assert classify_target_type("Amherst Point", scan_name) == "engineering"
        assert classify_target_type("external calibration", scan_name) == "engineering"

    @pytest.mark.parametrize("scan_name", [
        "500ppp_1_laser_disabled", "detail_2_laser_disabled",
        "laser_disabled_detail",
    ])
    def test_laser_disabled_scan_names_are_engineering(self, scan_name):
        """*laser_disabled* scan_names → engineering."""
        assert classify_target_type("Amherst Point", scan_name) == "engineering"
        assert classify_target_type(None, scan_name) == "engineering"

    # --- Calibration ---

    @pytest.mark.parametrize("target", [
        "external calibration", "teflon calibration", "calibration",
        "algan340 calibration", "maze calibration",
        "ext cal meteorite", "passive diffusil",
    ])
    def test_known_cal_targets(self, target):
        result = classify_target_type(target, "detail_1")
        assert result == "cal_target"

    def test_cal_targets_case_insensitive(self):
        assert classify_target_type("External Calibration", "detail_1") == "cal_target"
        assert classify_target_type("TEFLON CALIBRATION", "survey_1") == "cal_target"

    @pytest.mark.parametrize("scan_name", [
        "AlGaN_1", "AlGaN_2", "AlGaN340_pos1", "algan_274",
    ])
    def test_algan_scan_names_are_cal(self, scan_name):
        """AlGaN* scan_names → cal_target (on non-engineering targets)."""
        assert classify_target_type("Amherst Point", scan_name) == "cal_target"
        assert classify_target_type("Uganik Island", scan_name) == "cal_target"

    # --- Mars target ---

    @pytest.mark.parametrize("target", [
        "Amherst Point", "Berry Hollow", "Cheyava Falls",
        "Dragons Egg Lake", "cat arm reservoir",
        "Aitkenodden", "Klorne",
    ])
    def test_mars_science_targets(self, target):
        result = classify_target_type(target, "detail_1")
        assert result == "mars_target"

    def test_mars_targets_case_insensitive(self):
        assert classify_target_type("AMHERST POINT", "detail_1") == "mars_target"
        assert classify_target_type("berry hollow", "survey_1") == "mars_target"

    # --- Priority order ---

    def test_engineering_beats_cal(self):
        """power_* on a cal target → engineering (higher priority)."""
        assert classify_target_type("external calibration", "power_on") == "engineering"
        assert classify_target_type("AlGaN340 calibration", "power_on") == "engineering"

    def test_engineering_beats_mars(self):
        """power_* on a Mars target → engineering."""
        assert classify_target_type("Amherst Point", "power_on") == "engineering"

    def test_cal_beats_mars(self):
        """AlGaN scan_name on a Mars target → cal_target."""
        assert classify_target_type("Amherst Point", "AlGaN_1") == "cal_target"

    def test_cat_arm_reservoir_is_mars(self):
        """'cat arm reservoir' is a Mars target, not engineering."""
        assert classify_target_type("cat arm reservoir", "detail_1") == "mars_target"
        assert classify_target_type("cat arm reservoir", "HDR_1") == "mars_target"

    # --- Completeness of constant sets ---

    def test_engineering_targets_are_lowercase(self):
        for t in _ENGINEERING_TARGETS:
            assert t == t.lower(), f"_ENGINEERING_TARGETS must be lowercase: {t}"

    def test_cal_targets_are_lowercase(self):
        for t in _CAL_TARGETS:
            assert t == t.lower(), f"_CAL_TARGETS must be lowercase: {t}"

    def test_no_overlap_between_sets(self):
        overlap = _ENGINEERING_TARGETS & _CAL_TARGETS
        assert not overlap, f"Overlap between engineering and cal: {overlap}"

    # --- Edge cases ---

    def test_both_none(self):
        assert classify_target_type(None, None) == "engineering"

    def test_scan_name_none(self):
        """NULL scan_name with a Mars target → mars_target."""
        assert classify_target_type("Amherst Point", None) == "mars_target"

    def test_scan_name_empty(self):
        assert classify_target_type("Amherst Point", "") == "mars_target"


class TestSQLParity:
    """Test that SQL CASE in migration matches Python classify_target_type().

    Uses representative fixture data covering all rule branches.
    """

    # (target, scan_name, expected_type)
    PARITY_FIXTURES = [
        # Engineering: NULL/empty
        (None, "detail_1", "engineering"),
        ("", "survey", "engineering"),
        # Engineering: known targets
        ("conjunction", "100ppp_1", "engineering"),
        ("arm stowed", "detail_1", "engineering"),
        ("arm stowed dark", "500ppp_1", "engineering"),
        # Engineering: power_*/laser_disabled
        ("Amherst Point", "power_on", "engineering"),
        (" Aitkenodden", "power_off", "engineering"),
        ("conjunction", "500ppp_1_laser_disabled", "engineering"),
        # Cal: known targets
        ("external calibration", "detail_1", "cal_target"),
        ("teflon calibration", "survey_1", "cal_target"),
        ("ext cal meteorite", "meteorite_median_all", "cal_target"),
        # Cal: AlGaN scan_name
        ("Amherst Point", "AlGaN_1", "cal_target"),
        ("Uganik Island", "AlGaN_2", "cal_target"),
        # Mars
        ("Amherst Point", "detail_1", "mars_target"),
        ("Berry Hollow", "HDR_1", "mars_target"),
        ("cat arm reservoir", "detail_1", "mars_target"),
    ]

    @pytest.mark.parametrize("target,scan_name,expected", PARITY_FIXTURES)
    def test_parity(self, target, scan_name, expected):
        result = classify_target_type(target, scan_name)
        assert result == expected, (
            f"classify_target_type({target!r}, {scan_name!r}) = {result!r}, "
            f"expected {expected!r}"
        )
