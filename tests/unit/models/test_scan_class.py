"""Tests for classify_scan_class() and derive_parent_name() functions.

Covers all frozen test vectors from the SCAN_CLASS_SPEC-revised.md.
"""

import pytest

from sherloc_pipeline.models.spectra import classify_scan_class, derive_parent_name


# Frozen test vectors from spec Section 10.1
CLASSIFY_VECTORS = [
    # (scan_name, expected_class)
    # Primary scans
    ("detail_1", "primary"),
    ("survey_1296", "primary"),
    ("HDR_100", "primary"),
    ("line", "primary"),
    ("line_1", "primary"),
    ("Orthofabric", "primary"),
    ("orthofabric", "primary"),
    ("power_on", "primary"),
    ("AlGaN_1", "primary"),
    ("100ppp_1", "primary"),
    # Sub-scans (digit before suffix)
    ("detail_1a", "sub_scan"),
    ("detail_1b", "sub_scan"),
    ("detail_1c", "sub_scan"),
    ("HDR_500_1a", "sub_scan"),
    ("detail_900ppp_c", "sub_scan"),
    ("meteorite_detail_1c", "sub_scan"),
    ("detail_2a", "sub_scan"),
    ("detail_3b", "sub_scan"),
    ("detail_4a", "sub_scan"),
    ("MarsMeteorite_1a", "sub_scan"),
    # Sub-scans (underscore before suffix)
    ("HDR_a", "sub_scan"),
    ("HDR_b", "sub_scan"),
    ("HDR_500_b", "sub_scan"),
    ("survey_100_a", "sub_scan"),
    ("survey_100ppp_a", "sub_scan"),
    ("detail_500ppp_a", "sub_scan"),
    ("detail_offset_1a", "sub_scan"),
    ("detailed_center_1a", "sub_scan"),
    ("detailed_corner_2b", "sub_scan"),
    # Composites
    ("detail_all", "composite"),
    ("meteorite_median_all", "composite"),
    ("polycarbonate_sum_active_median_dark", "composite"),
    ("detail_2_median_all", "composite"),
    ("meteorite_sum_active_sum_dark", "composite"),
    ("polycarbonate_median_all", "composite"),
    ("meteorite_sum_active_median_dark", "composite"),
    ("polycarbonate_sum_active_sum_dark", "composite"),
    ("detail_2_sum_active_median_dark", "composite"),
    ("asterisk", "composite"),
    ("cross", "composite"),
]


class TestClassifyScanClass:
    """Test classify_scan_class() with all frozen test vectors."""

    @pytest.mark.parametrize("scan_name,expected", CLASSIFY_VECTORS)
    def test_classification(self, scan_name, expected):
        assert classify_scan_class(scan_name) == expected, (
            f"classify_scan_class('{scan_name}') should be '{expected}'"
        )

    def test_empty_string(self):
        assert classify_scan_class("") == "primary"

    def test_none_input(self):
        assert classify_scan_class(None) == "primary"

    def test_composite_priority_over_sub_scan(self):
        """Composite patterns are checked before sub-scan suffix."""
        # A name with _all that also ends in [a-c] after digit
        # _all takes priority
        assert classify_scan_class("detail_2_median_all") == "composite"


PARENT_VECTORS = [
    # (scan_name, expected_parent)
    # Not sub-scans
    ("detail_1", None),
    ("survey_1296", None),
    ("HDR_100", None),
    ("line", None),
    ("Orthofabric", None),
    ("orthofabric", None),
    # Digit-suffix sub-scans
    ("detail_1a", "detail_1"),
    ("detail_1b", "detail_1"),
    ("detail_1c", "detail_1"),
    ("HDR_500_1a", "HDR_500_1"),
    ("detail_2a", "detail_2"),
    ("detail_3b", "detail_3"),
    ("MarsMeteorite_1a", "MarsMeteorite_1"),
    ("meteorite_detail_1c", "meteorite_detail_1"),
    # Underscore-suffix sub-scans
    ("HDR_a", "HDR"),
    ("HDR_b", "HDR"),
    ("HDR_500_b", "HDR_500"),
    ("survey_100_a", "survey_100"),
    ("detail_500ppp_a", "detail_500ppp"),
    ("detail_900ppp_c", "detail_900ppp"),
]


class TestDeriveParentName:
    """Test derive_parent_name() with all frozen test vectors."""

    @pytest.mark.parametrize("scan_name,expected_parent", PARENT_VECTORS)
    def test_parent_derivation(self, scan_name, expected_parent):
        assert derive_parent_name(scan_name) == expected_parent, (
            f"derive_parent_name('{scan_name}') should be '{expected_parent}'"
        )

    def test_empty_string(self):
        assert derive_parent_name("") is None

    def test_none_input(self):
        assert derive_parent_name(None) is None

    def test_single_char(self):
        assert derive_parent_name("a") is None
