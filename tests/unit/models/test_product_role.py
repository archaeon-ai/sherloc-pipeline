"""Tests for classify_product_role() + multishot helpers.

Value-blind: only scan names are exercised.
"""

import pytest

from sherloc_pipeline.models.spectra import (
    classify_product_role,
    multishot_reduction_role,
    multishot_raw_base,
)


ROLE_VECTORS = [
    # canonical reduction
    ("detail_2_sum_active_median_dark", "canonical"),
    ("polycarbonate_sum_active_median_dark", "canonical"),
    # alternate reductions
    ("detail_2_median_all", "alternate"),
    ("meteorite_sum_active_sum_dark", "alternate"),
    ("polycarbonate_sum_active_sum_dark", "alternate"),
    # NOT a multishot reduction (product_role NULL)
    ("detail_all", None),          # bare spatial union — the _all collision
    ("detail_", None),             # bare trailing-underscore union
    ("detail_2", None),            # the raw itself (role assigned corpus-side)
    ("detail_1", None),            # normal primary
    ("cross", None),               # name union
    ("line_", None),
    ("", None),
    (None, None),
]


class TestClassifyProductRole:
    @pytest.mark.parametrize("name,expected", ROLE_VECTORS)
    def test_role(self, name, expected):
        assert classify_product_role(name) == expected

    def test_role_matches_reduction_role(self):
        for name, _expected in ROLE_VECTORS:
            assert classify_product_role(name) == multishot_reduction_role(name)


class TestAllCollision:
    """`detail_all` (spatial union) vs `*_median_all` (multishot reduction)."""

    def test_bare_all_is_not_a_reduction(self):
        assert multishot_reduction_role("detail_all") is None
        assert multishot_raw_base("detail_all") is None

    def test_median_all_is_a_reduction(self):
        assert multishot_reduction_role("detail_2_median_all") == "alternate"
        assert multishot_raw_base("detail_2_median_all") == "detail_2"


class TestRawBase:
    @pytest.mark.parametrize("name,expected_base", [
        ("detail_2_median_all", "detail_2"),
        ("detail_2_sum_active_median_dark", "detail_2"),
        ("detail_2_sum_active_sum_dark", "detail_2"),
        ("prefix_detail_2_sum_active_median_dark", "prefix_detail_2"),
        ("detail_all", None),
        ("detail_2", None),
        ("", None),
        (None, None),
    ])
    def test_raw_base(self, name, expected_base):
        assert multishot_raw_base(name) == expected_base

    def test_raw_base_preserves_case(self):
        assert multishot_raw_base("Detail_2_Median_All") == "Detail_2"
