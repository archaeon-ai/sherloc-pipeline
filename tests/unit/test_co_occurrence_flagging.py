"""Tests for co-occurrence flagging: aggregate_co_occurrences() and
updated score_cooccurrences() with Group 3 silicate support.
"""

import pytest

from sherloc_pipeline.services.pipeline import aggregate_co_occurrences
from sherloc_pipeline.core.fluor_id import score_cooccurrences


# ---------------------------------------------------------------------------
# aggregate_co_occurrences tests
# ---------------------------------------------------------------------------

class TestAggregatCoOccurrences:
    """Tests for scan-level co-occurrence aggregation."""

    def test_sulfate_group1_co_occurrence(self):
        """sulf1_v1 + group1a at same point -> Ce3+-bearing Ca-sulfate."""
        raman = [{"point_idx": 0, "mineral_assignment": "sulf1_v1"}]
        fluor = [{"point_idx": 0, "fluor_group": "group1a"}]
        result = aggregate_co_occurrences(raman, fluor)
        sulfate_patterns = [r for r in result if r["pattern"] == "Ce3+-bearing Ca-sulfate"]
        assert len(sulfate_patterns) == 1
        assert sulfate_patterns[0]["n_points_confirmed"] == 1
        assert 0 in sulfate_patterns[0]["point_indices"]

    def test_phosphate_group2_co_occurrence(self):
        """phosphate + group2 at same point -> Ce3+-bearing phosphate."""
        raman = [{"point_idx": 5, "mineral_assignment": "phosphate"}]
        fluor = [{"point_idx": 5, "fluor_group": "group2"}]
        result = aggregate_co_occurrences(raman, fluor)
        phos_patterns = [r for r in result if r["pattern"] == "Ce3+-bearing phosphate"]
        assert len(phos_patterns) == 1
        assert phos_patterns[0]["n_points_confirmed"] == 1

    def test_silicate_group3_co_occurrence(self):
        """pyroxene + group3 at same point -> Silicate defect."""
        raman = [{"point_idx": 3, "mineral_assignment": "pyroxene"}]
        fluor = [{"point_idx": 3, "fluor_group": "group3"}]
        result = aggregate_co_occurrences(raman, fluor)
        silicate_patterns = [r for r in result if r["pattern"] == "Silicate defect luminescence"]
        assert len(silicate_patterns) == 1
        assert silicate_patterns[0]["n_points_confirmed"] == 1

    def test_1050_group3_co_occurrence(self):
        """1050 mineral + group3 at same point -> Silicate defect."""
        raman = [{"point_idx": 7, "mineral_assignment": "1050"}]
        fluor = [{"point_idx": 7, "fluor_group": "group3"}]
        result = aggregate_co_occurrences(raman, fluor)
        silicate_patterns = [r for r in result if r["pattern"] == "Silicate defect luminescence"]
        assert len(silicate_patterns) == 1
        assert silicate_patterns[0]["n_points_confirmed"] == 1

    def test_no_co_occurrence_different_points(self):
        """Minerals and fluorescence at different points -> no confirmed."""
        raman = [{"point_idx": 0, "mineral_assignment": "sulf1_v1"}]
        fluor = [{"point_idx": 5, "fluor_group": "group1a"}]
        result = aggregate_co_occurrences(raman, fluor)
        sulfate_patterns = [r for r in result if r["pattern"] == "Ce3+-bearing Ca-sulfate"]
        assert len(sulfate_patterns) == 1
        assert sulfate_patterns[0]["n_points_confirmed"] == 0
        assert sulfate_patterns[0]["n_points_raman_only"] == 1
        assert sulfate_patterns[0]["n_points_fluor_only"] == 1

    def test_mixed_co_occurrence_counts(self):
        """Some points co-occur, some don't -> correct counts."""
        raman = [
            {"point_idx": 0, "mineral_assignment": "sulf1_v1"},
            {"point_idx": 1, "mineral_assignment": "sulf1_v1"},
            {"point_idx": 2, "mineral_assignment": "sulf1_v1"},
        ]
        fluor = [
            {"point_idx": 0, "fluor_group": "group1a"},
            {"point_idx": 1, "fluor_group": "group1b"},
            {"point_idx": 3, "fluor_group": "group1a"},
        ]
        result = aggregate_co_occurrences(raman, fluor)
        sulfate_patterns = [r for r in result if r["pattern"] == "Ce3+-bearing Ca-sulfate"]
        assert len(sulfate_patterns) == 1
        p = sulfate_patterns[0]
        assert p["n_points_confirmed"] == 2  # pts 0, 1
        assert p["n_points_raman_only"] == 1  # pt 2
        assert p["n_points_fluor_only"] == 1  # pt 3

    def test_empty_inputs(self):
        """Empty inputs -> empty results."""
        result = aggregate_co_occurrences([], [])
        assert result == []

    def test_mean_confidence_value(self):
        """Confirmed points get 1.3 confidence each."""
        raman = [
            {"point_idx": 0, "mineral_assignment": "phosphate"},
            {"point_idx": 1, "mineral_assignment": "phosphate"},
        ]
        fluor = [
            {"point_idx": 0, "fluor_group": "group2"},
            {"point_idx": 1, "fluor_group": "group2"},
        ]
        result = aggregate_co_occurrences(raman, fluor)
        phos = [r for r in result if r["pattern"] == "Ce3+-bearing phosphate"][0]
        assert phos["mean_confidence"] == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# score_cooccurrences Group 3 upgrade tests
# ---------------------------------------------------------------------------

class TestScoreCooccurrencesGroup3:
    """Tests for Group 3 silicate support in score_cooccurrences()."""

    def test_group3_with_pyroxene_confirmed(self):
        """Group 3 + pyroxene Raman -> confirmed."""
        scores = score_cooccurrences(["group3"], ["pyroxene"])
        assert len(scores) == 1
        assert scores[0].raman_support == "confirmed"
        assert scores[0].confidence_boost == 1.3

    def test_group3_with_1050_confirmed(self):
        """Group 3 + '1050' Raman -> confirmed."""
        scores = score_cooccurrences(["group3"], ["1050"])
        assert len(scores) == 1
        assert scores[0].raman_support == "confirmed"

    def test_group3_with_silicate_hump_confirmed(self):
        """Group 3 + silicate_hump Raman -> confirmed."""
        scores = score_cooccurrences(["group3"], ["silicate_hump"])
        assert len(scores) == 1
        assert scores[0].raman_support == "confirmed"

    def test_group3_with_non_silicate_raman_unsupported(self):
        """Group 3 + non-silicate Raman -> unsupported."""
        scores = score_cooccurrences(["group3"], ["sulf1_v1"])
        assert len(scores) == 1
        assert scores[0].raman_support == "unsupported"
        assert len(scores[0].notes) > 0

    def test_group3_no_raman(self):
        """Group 3 + no Raman -> no_raman."""
        scores = score_cooccurrences(["group3"], [])
        assert len(scores) == 1
        assert scores[0].raman_support == "no_raman"

    def test_group1_sulfate_still_works(self):
        """Existing Group 1 + sulfate logic unchanged."""
        scores = score_cooccurrences(["group1a"], ["sulf1_v1"])
        assert scores[0].raman_support == "confirmed"
        assert scores[0].phase_interpretation == "Ce3+-bearing anhydrite"

    def test_group2_phosphate_still_works(self):
        """Existing Group 2 + phosphate logic unchanged."""
        scores = score_cooccurrences(["group2"], ["phosphate"])
        assert scores[0].raman_support == "confirmed"
        assert scores[0].phase_interpretation == "Ce3+-bearing phosphate"
