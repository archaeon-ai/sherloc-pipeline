"""Unit tests for core/mineral_id.py -- peak wavenumber to mineral assignment.

Regression guards capturing current behavior before structural refactor.
Tests use the actual DEFAULT_RULES from config.yaml.
"""

import pandas as pd
import pytest

from sherloc_pipeline.core.mineral_id import (
    DEFAULT_RULES,
    MineralRule,
    assign_min_id,
    load_mineral_rules,
    map_min_id_series,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rules():
    """Use actual default rules loaded from config.yaml."""
    return list(DEFAULT_RULES)


# ---------------------------------------------------------------------------
# DEFAULT_RULES loading tests
# ---------------------------------------------------------------------------


class TestDefaultRules:

    def test_rules_loaded(self):
        """DEFAULT_RULES should be populated from config.yaml at import time."""
        assert len(DEFAULT_RULES) > 0

    def test_rules_are_mineral_rule_objects(self):
        for rule in DEFAULT_RULES:
            assert isinstance(rule, MineralRule)
            assert isinstance(rule.label, str)
            assert isinstance(rule.lo, float)
            assert isinstance(rule.hi, float)
            assert rule.lo < rule.hi

    def test_expected_minerals_present(self):
        """Key minerals should be in the default rules."""
        labels = {r.label for r in DEFAULT_RULES}
        assert "olivine" in labels
        assert "hi-carb" in labels or "lo-carb" in labels
        # Sulfate rules
        assert any("sulf" in label for label in labels)


# ---------------------------------------------------------------------------
# assign_min_id tests -- known mineral peaks
# ---------------------------------------------------------------------------


class TestAssignMinId:

    def test_olivine(self, rules):
        """~840 cm^-1 should identify as olivine (810-860)."""
        result = assign_min_id(840.0, rules)
        assert result == "olivine"

    def test_sulfate_v1_region1(self, rules):
        """~1010 cm^-1 should identify as sulf1_v1 (1008-1020)."""
        result = assign_min_id(1010.0, rules)
        assert result == "sulf1_v1"

    def test_sulfate_v1_region2(self, rules):
        """~1030 cm^-1 should identify as sulf2_v1 (1020-1045)."""
        result = assign_min_id(1030.0, rules)
        assert result == "sulf2_v1"

    def test_hi_carbonate(self, rules):
        """~1086 cm^-1 should identify as hi-carb (1075-1105)."""
        result = assign_min_id(1086.0, rules)
        assert result == "hi-carb"

    def test_lo_carbonate(self, rules):
        """~1060 cm^-1 should identify as lo-carb (1055-1075)."""
        result = assign_min_id(1060.0, rules)
        assert result == "lo-carb"

    def test_phosphate(self, rules):
        """~965 cm^-1 should identify as phosphate (945-980)."""
        result = assign_min_id(965.0, rules)
        assert result == "phosphate"

    def test_phosphate_whitlockite(self, rules):
        """~946.6 cm^-1 (whitlockite PO4 v1) should identify as phosphate."""
        result = assign_min_id(946.6, rules)
        assert result == "phosphate"

    def test_pyroxene(self, rules):
        """~990 cm^-1 should identify as pyroxene (980-1008)."""
        result = assign_min_id(990.0, rules)
        assert result == "pyroxene"

    def test_sulf_v3(self, rules):
        """~1128 cm^-1 should identify as sulf_v3 (1120-1160)."""
        result = assign_min_id(1128.0, rules)
        assert result == "sulf_v3"

    def test_1050_region(self, rules):
        """~1050 cm^-1 should identify as '1050' (1045-1055)."""
        result = assign_min_id(1050.0, rules)
        assert result == "1050"

    def test_unidentified_outside_all_ranges(self, rules):
        """Wavenumber far outside all ranges → 'unidentified'."""
        assert assign_min_id(500.0, rules) == "unidentified"
        assert assign_min_id(2000.0, rules) == "unidentified"

    def test_boundary_inclusive_lower(self, rules):
        """Lower boundary should be inclusive."""
        result = assign_min_id(810.0, rules)
        assert result == "olivine"

    def test_boundary_inclusive_upper(self, rules):
        """Upper boundary should be inclusive."""
        result = assign_min_id(860.0, rules)
        assert result == "olivine"

    def test_boundary_between_rules(self, rules):
        """At shared boundary (e.g., 1008), deterministic tie-breaking applies."""
        # 1008 is the boundary between pyroxene (980-1008) and sulf1_v1 (1008-1020)
        result = assign_min_id(1008.0, rules)
        # Both match; tie-breaker picks the rule with highest lo
        assert result == "sulf1_v1"

    def test_non_numeric_returns_unidentified(self, rules):
        """Non-numeric input should return 'unidentified' (exception caught)."""
        result = assign_min_id(float("nan"), rules)
        # NaN comparison: NaN <= x is False, so no matches
        assert result == "unidentified"

    def test_empty_rules(self):
        """Empty rules → always 'unidentified'."""
        assert assign_min_id(1010.0, []) == "unidentified"


# ---------------------------------------------------------------------------
# load_mineral_rules tests
# ---------------------------------------------------------------------------


class TestLoadMineralRules:

    def test_default_rules_returned(self):
        """No arguments → return DEFAULT_RULES."""
        rules = load_mineral_rules()
        assert len(rules) == len(DEFAULT_RULES)

    def test_inline_rules(self):
        """Inline rules should override defaults."""
        inline = [{"label": "test_mineral", "lo": 100.0, "hi": 200.0}]
        rules = load_mineral_rules(inline_rules=inline)
        assert len(rules) == 1
        assert rules[0].label == "test_mineral"

    def test_nonexistent_path_falls_back(self, tmp_path):
        """Non-existent path → fallback to defaults."""
        rules = load_mineral_rules(path=tmp_path / "nonexistent.yaml")
        assert len(rules) == len(DEFAULT_RULES)


# ---------------------------------------------------------------------------
# map_min_id_series tests
# ---------------------------------------------------------------------------


class TestMapMinIdSeries:

    def test_series_mapping(self, rules):
        centers = pd.Series([840.0, 1010.0, 1086.0, 500.0])
        result = map_min_id_series(centers, rules)
        assert result.iloc[0] == "olivine"
        assert result.iloc[1] == "sulf1_v1"
        assert result.iloc[2] == "hi-carb"
        assert result.iloc[3] == "unidentified"

    def test_empty_series(self, rules):
        result = map_min_id_series(pd.Series(dtype=float), rules)
        assert len(result) == 0
