"""Tests for classify_scan_type() — name-authoritative scan_type resolver.

WS-1 scan-classification spec §4.2 (ARC-M2P-308 / -309). Value-blind: every
vector is a name / sequence-code / spectrum-count, never a science value.
"""

import pytest

from sherloc_pipeline.models.spectra import (
    ScanType,
    SCAN_TYPE_QUARANTINE,
    classify_scan_type,
)


def _t(scan_name, sequence_code=None, n_spectra=None):
    result = classify_scan_type(scan_name, sequence_code, n_spectra)
    return result.value if isinstance(result, ScanType) else result


# (scan_name, sequence_code, n_spectra, expected) — the locked taxonomy (§4.1)
# plus the complete #115 mislabel edge-case set (RCA R5).
NAME_TYPE_VECTORS = [
    # survey family — name-authoritative, _NNNppp is an acquisition param
    ("survey_1296", None, 1296, "survey"),
    ("survey", None, 1296, "survey"),
    ("survey_100", None, 100, "survey"),      # #115: small survey mislabeled detail by count
    ("survey_100ppp", None, 100, "survey"),
    ("survey_500ppp", None, 100, "survey"),
    # detail family
    ("detail_1", None, 100, "detail"),
    ("detail_2", None, 300, "detail"),        # #115: multishot raw mislabeled survey by count
    ("detail_1a", None, 50, "detail"),        # sub-scan inherits detail kind
    ("detail_all", None, 400, "detail"),      # spatial union inherits detail
    ("detail_2_median_all", None, 100, "detail"),
    ("detail_2_sum_active_median_dark", None, 100, "detail"),
    # line family (#115: unrepresentable before LINE was added)
    ("line", None, 25, "line"),
    ("line_1", None, 25, "line"),
    ("line_36", None, 36, "line"),
    # HDR family (#115: unrepresentable before HDR was added)
    ("HDR", None, 100, "HDR"),
    ("HDR_500", None, 100, "HDR"),
    ("HDR_a", None, 50, "HDR"),
    # cross / asterisk inherit the constituent `line` kind (K1)
    ("cross", None, 50, "line"),
    ("asterisk", None, 100, "line"),
    # calibration — sequence code first, unshadowable by name
    ("AlGaN_1", "srlc11374", 1, "calibration"),   # AlGaN name (no cal seq) still calibration
    ("AlGaN_340", None, 1, "calibration"),
    ("detail_1", "srlc10000", 100, "calibration"),  # cal seq wins over a detail-like name
    ("survey_1296", "srlc16000", 1296, "calibration"),
]


class TestNameAuthoritative:
    @pytest.mark.parametrize("name,seq,n,expected", NAME_TYPE_VECTORS)
    def test_name_to_type(self, name, seq, n, expected):
        assert _t(name, seq, n) == expected


class TestTokenPrecedenceAndBoundary:
    def test_survey_hdr_precedence(self):
        # 'survey' is checked before 'hdr': the trailing token is a parameter.
        assert _t("survey_HDR", None, 100) == "survey"

    def test_lowercase_hdr(self):
        assert _t("hdr", None, 100) == "HDR"

    def test_mixed_case_normalized(self):
        assert _t("DeTaIl_3", None, 100) == "detail"
        assert _t("SURVEY_1296", None, 1296) == "survey"

    def test_detail_is_token_boundaried(self):
        # 'detailed_center' must NOT match 'detail' (the token is followed by a
        # letter) — it is informative-but-unrecognized -> quarantine.
        assert _t("detailed_center_1a", None, 50) == SCAN_TYPE_QUARANTINE

    def test_line_as_non_token_substring(self):
        # A name where 'line' appears as a non-leading-token substring.
        assert _t("baseline_scan", None, 50) == SCAN_TYPE_QUARANTINE

    def test_hdr_not_matched_inside_word(self):
        assert _t("hydration_x", None, 50) == SCAN_TYPE_QUARANTINE


class TestQuarantine:
    def test_informative_unknown_quarantines(self):
        # A future DO_AREA-class name must quarantine, never fall through to
        # the count rule (ARC-M2P-308 AC5).
        assert _t("DO_AREA_1", None, 1000) == SCAN_TYPE_QUARANTINE
        assert _t("DO_AREA_1", None, 10) == SCAN_TYPE_QUARANTINE

    def test_quarantine_is_a_sentinel_not_a_scantype(self):
        result = classify_scan_type("DO_AREA_1", None, 100)
        assert result == SCAN_TYPE_QUARANTINE
        assert not isinstance(result, ScanType)


class TestUninformativeCountFallback:
    """The count rule is reachable ONLY for explicitly-uninformative names."""

    def test_pds_synthetic_name_uses_count(self):
        # PDS observations carry a synthetic, kind-blind name.
        assert _t("pds_0921_0831289704_001", None, 1296) == "survey"
        assert _t("pds_0921_0831289704_002", None, 100) == "detail"

    def test_empty_and_none_use_count(self):
        assert _t("", None, 1296) == "survey"
        assert _t(None, None, 100) == "detail"
        assert _t("", None, None) == "detail"   # no count -> detail (§4.2)

    def test_count_threshold_boundary(self):
        assert _t("pds_1_1_1", None, 200) == "detail"   # not > 200
        assert _t("pds_1_1_1", None, 201) == "survey"
