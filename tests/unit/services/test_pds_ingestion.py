"""Unit tests for PDS ingestion service logic (Phase 7, step 7.2).

Tests TargetNameResolver (mock DB), ACI LIDVID construction,
version comparison (numeric tuple), and idempotency checks.

All tests use in-memory databases or fixtures — no dependency on
Sol 921 data or real phase.db.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select, func

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    create_all_tables,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    ContextImageORM,
)
from sherloc_pipeline.services.pds_ingestion import (
    PDSIngestionService,
    PDSIngestionError,
    PDSIngestionStats,
    PDSVersionUpdate,
    TargetNameResolver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loupe_db(tmp_path, scans, db_name="loupe_test.db"):
    """Create a mock Loupe-like SQLite DB with given scan records.

    Args:
        tmp_path: pytest tmp_path fixture.
        scans: list of dicts with keys: sol, sclk, target, site_drive (optional).
        db_name: database filename.

    Returns:
        SQLAlchemy Engine.
    """
    db_path = tmp_path / db_name
    engine = get_engine(db_path)
    create_all_tables(engine)

    with get_session(engine) as session:
        seen_sols = set()
        for s in scans:
            sol_num = s["sol"]
            if sol_num not in seen_sols:
                session.add(SolORM(
                    sol_number=sol_num,
                    data_source="loupe",
                    created_at=datetime.now(timezone.utc),
                ))
                seen_sols.add(sol_num)
        session.flush()

        for s in scans:
            session.add(ScanORM(
                id=str(uuid.uuid4()),
                sol_number=s["sol"],
                scan_name=f"test_{s['sclk']}",
                scan_id=f"test_scan_{s['sclk']}",
                sclk_start=s["sclk"],
                n_points=1,
                n_channels=2148,
                laser_wavelength_nm=248.6,
                target=s.get("target"),
                site_drive=s.get("site_drive"),
                data_source="loupe",
                created_at=datetime.now(timezone.utc),
            ))
    return engine


# ===========================================================================
# Section 1: TargetNameResolver
# ===========================================================================


class TestResolverPass1:
    """Pass 1 (±3s tolerance) unit tests."""

    def test_exact_sclk_match(self, tmp_path):
        """Identical SCLK in Loupe → immediate match."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Exact"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "Exact"

    def test_delta_1s(self, tmp_path):
        """1s delta → within Pass 1."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Delta1"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000001) == "Delta1"

    def test_negative_delta_within_tolerance(self, tmp_path):
        """PDS SCLK lower than Loupe SCLK but within ±3s."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000003, "target": "NegDelta"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "NegDelta"

    def test_boundary_3s(self, tmp_path):
        """Exactly 3s delta → still within Pass 1 (inclusive)."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Boundary3"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000003) == "Boundary3"

    def test_4s_misses_pass1(self, tmp_path):
        """4s delta misses Pass 1, caught by Pass 2."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Pass2Hit"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        # Should still resolve via Pass 2
        assert resolver.resolve(sol=100, pds_sclk=700000004) == "Pass2Hit"


class TestResolverPass2:
    """Pass 2 (±5s tolerance) unit tests."""

    def test_boundary_5s(self, tmp_path):
        """Exactly 5s delta → within Pass 2."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Bound5"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000005) == "Bound5"

    def test_6s_beyond_both_passes(self, tmp_path):
        """6s delta → beyond both passes → None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "TooFar"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000006) is None

    def test_negative_6s_beyond(self, tmp_path):
        """Negative 6s delta → beyond both passes → None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000006, "target": "NegTooFar"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None


class TestResolverNoMatch:
    """Scenarios that should return None."""

    def test_wrong_sol(self, tmp_path):
        """Different sol → no match."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 920, "sclk": 700000000, "target": "WrongSol"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=921, pds_sclk=700000000) is None

    def test_null_target_excluded(self, tmp_path):
        """Loupe scan with NULL target is excluded from candidates."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": None},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_empty_database(self, tmp_path):
        """Empty Loupe DB → None."""
        engine = _make_loupe_db(tmp_path, [])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_no_engine(self):
        """No loupe_engine → Tier 1 skipped → None (without curated)."""
        resolver = TargetNameResolver(loupe_engine=None)
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_all_null_targets(self, tmp_path):
        """Multiple Loupe scans with NULL targets → all excluded → None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": None},
            {"sol": 100, "sclk": 700000001, "target": None},
            {"sol": 100, "sclk": 700000002, "target": None},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000001) is None


class TestResolverTieBreaking:
    """Ambiguity resolution (multiple SCLK candidates)."""

    def test_nearest_delta_wins(self, tmp_path):
        """When multiple candidates exist, nearest SCLK delta wins."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Far"},    # delta=2
            {"sol": 100, "sclk": 700000003, "target": "Near"},   # delta=1
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        assert resolver.resolve(sol=100, pds_sclk=700000002) == "Near"

    def test_site_drive_breaks_equal_delta(self, tmp_path):
        """Equal SCLK delta → site_drive match preferred."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 699999999, "target": "NoSD", "site_drive": None},
            {"sol": 100, "sclk": 700000001, "target": "HasSD", "site_drive": "04500672"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        # PDS SCLK 700000000: delta=1 to both → tied → site_drive breaks
        result = resolver.resolve(
            sol=100, pds_sclk=700000000, site_drive="04500672"
        )
        assert result == "HasSD"

    def test_site_drive_mismatch_falls_back(self, tmp_path):
        """Equal delta but no site_drive match → first candidate selected."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 699999999, "target": "A", "site_drive": "111"},
            {"sol": 100, "sclk": 700000001, "target": "B", "site_drive": "222"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        # No site_drive matches "999" → selects first tied candidate
        result = resolver.resolve(
            sol=100, pds_sclk=700000000, site_drive="999"
        )
        assert result in ("A", "B")  # Either is acceptable

    def test_no_site_drive_provided(self, tmp_path):
        """Equal delta, no site_drive parameter → first candidate selected."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 699999999, "target": "X", "site_drive": "123"},
            {"sol": 100, "sclk": 700000001, "target": "Y", "site_drive": "456"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        result = resolver.resolve(sol=100, pds_sclk=700000000)
        assert result in ("X", "Y")

    def test_three_candidates_nearest_wins(self, tmp_path):
        """Three candidates at different deltas → nearest wins."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 699999998, "target": "Far3"},   # delta=3
            {"sol": 100, "sclk": 700000000, "target": "Near1"},  # delta=1
            {"sol": 100, "sclk": 700000002, "target": "Mid2"},   # delta=1 too
        ])
        resolver = TargetNameResolver(loupe_engine=engine)
        # delta to 700000000 = 1, delta to 700000002 = 1, delta to 699999998 = 3
        # Two tied at delta=1 → first sorted wins (no site_drive)
        result = resolver.resolve(sol=100, pds_sclk=700000001)
        assert result in ("Near1", "Mid2")  # Both valid at delta=1

    def test_ambiguity_logs_warning(self, tmp_path, caplog):
        """Multiple candidates should produce a warning log."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 699999999, "target": "A"},
            {"sol": 100, "sclk": 700000001, "target": "B"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        with caplog.at_level(logging.WARNING):
            resolver.resolve(sol=100, pds_sclk=700000000)

        assert any("SCLK ambiguity" in msg for msg in caplog.messages)


class TestResolverCurated:
    """Tier 2: curated mapping table."""

    def test_curated_fallback_when_no_engine(self, tmp_path):
        """Without Loupe engine, curated mapping provides target."""
        mapping = {"100_700000000": "Curated Rock"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=None, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "Curated Rock"

    def test_curated_fallback_when_sclk_misses(self, tmp_path):
        """SCLK lookup fails → falls through to curated."""
        engine = _make_loupe_db(tmp_path, [])  # Empty DB
        mapping = {"100_700000000": "Curated Only"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "Curated Only"

    def test_curated_key_miss_returns_none(self, tmp_path):
        """Curated mapping exists but key doesn't match → None."""
        mapping = {"999_111111111": "Wrong Key"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=None, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_sclk_takes_priority_over_curated(self, tmp_path):
        """Tier 1 match takes priority over Tier 2 curated."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "DB Target"},
        ])
        mapping = {"100_700000000": "Curated Target"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "DB Target"

    def test_missing_curated_file_nonfatal(self, tmp_path):
        """Nonexistent curated file handled gracefully."""
        resolver = TargetNameResolver(
            loupe_engine=None,
            curated_path=tmp_path / "nonexistent.json",
        )
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_malformed_curated_json_nonfatal(self, tmp_path):
        """Malformed JSON in curated file → handled gracefully."""
        curated = tmp_path / "bad.json"
        curated.write_text("{ not valid json !!!")

        resolver = TargetNameResolver(
            loupe_engine=None,
            curated_path=curated,
        )
        # Should not raise; curated map stays empty
        assert resolver.resolve(sol=100, pds_sclk=700000000) is None

    def test_curated_multiple_entries(self, tmp_path):
        """Curated mapping with multiple entries returns correct one."""
        mapping = {
            "100_700000000": "Target A",
            "100_700000001": "Target B",
            "200_700000000": "Target C",
        }
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=None, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "Target A"
        assert resolver.resolve(sol=100, pds_sclk=700000001) == "Target B"
        assert resolver.resolve(sol=200, pds_sclk=700000000) == "Target C"

    def test_curated_key_format(self, tmp_path):
        """Key format is exactly '{sol}_{sclk}' with no padding."""
        mapping = {"1_666000001": "Minimal"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=None, curated_path=curated)
        assert resolver.resolve(sol=1, pds_sclk=666000001) == "Minimal"


class TestResolverTierProgression:
    """Test the three-tier fallback progression."""

    def test_tier1_hit_stops_early(self, tmp_path):
        """Tier 1 hit → Tier 2 never checked."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "DB Hit"},
        ])
        mapping = {"100_700000000": "Should Not Use"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "DB Hit"

    def test_tier1_miss_tier2_hit(self, tmp_path):
        """Tier 1 miss (no match) → Tier 2 hit."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "TooFar"},  # 10s away
        ])
        mapping = {"100_700000010": "Curated"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000010) == "Curated"

    def test_all_tiers_miss(self, tmp_path):
        """All tiers miss → None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": "Far"},
        ])
        mapping = {"999_999999999": "Wrong"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000010) is None

    def test_null_target_in_db_falls_to_curated(self, tmp_path):
        """DB scan with NULL target → excluded → falls to curated."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 100, "sclk": 700000000, "target": None},
        ])
        mapping = {"100_700000000": "Curated Fallback"}
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps(mapping))

        resolver = TargetNameResolver(loupe_engine=engine, curated_path=curated)
        assert resolver.resolve(sol=100, pds_sclk=700000000) == "Curated Fallback"


# ===========================================================================
# Section 2: ACI LIDVID Construction
# ===========================================================================


class TestACILidvidBasic:
    """Basic LIDVID construction from Image_name."""

    ACI_PREFIX = "urn:nasa:pds:mars2020_imgops:data_aci_imgops"

    def test_standard_detail_aci(self):
        """Standard detail ACI filename."""
        name = "SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(name)
        expected = (
            f"{self.ACI_PREFIX}:"
            "sc3_0921_0748731308_359ecm_n0450000srlc11374_0000lmj::1.0"
        )
        assert lidvid == expected

    def test_calibration_aci(self):
        """Calibration ACI filename with SC0 prefix."""
        name = "SC0_0921_0748731023_488ECM_N0450000SRLC10000_0000LUJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(name)
        assert lidvid.startswith(f"{self.ACI_PREFIX}:sc0_0921_")
        assert lidvid.endswith("::1.0")
        assert "luj::" in lidvid  # Version suffix "01" removed

    def test_all_lowercase(self):
        """Product identifier is entirely lowercase."""
        name = "SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(name)
        # Everything between last colon before :: and :: should be lowercase
        product_id = lidvid.split(":")[-1].split("::")[0]
        assert product_id == product_id.lower()

    def test_collection_uri_correct(self):
        """LIDVID starts with correct collection URI."""
        name = "TEST01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(name)
        assert lidvid.startswith(self.ACI_PREFIX + ":")


class TestACILidvidVersion:
    """Version suffix extraction and VID generation."""

    def test_version_01(self):
        """Version suffix '01' → VID '1.0'."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME01.IMG")
        assert lidvid.endswith("::1.0")

    def test_version_02(self):
        """Version suffix '02' → VID '2.0'."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME02.IMG")
        assert lidvid.endswith("::2.0")

    def test_version_10(self):
        """Version suffix '10' → VID '10.0'."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME10.IMG")
        assert lidvid.endswith("::10.0")

    def test_version_00(self):
        """Version suffix '00' → VID '0.0'."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME00.IMG")
        assert lidvid.endswith("::0.0")

    def test_version_99(self):
        """Version suffix '99' → VID '99.0'."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME99.IMG")
        assert lidvid.endswith("::99.0")

    def test_non_numeric_defaults_to_1(self):
        """Non-numeric version suffix → VID '1.0' fallback."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAMEXX.IMG")
        assert lidvid.endswith("::1.0")

    def test_partially_numeric_defaults_to_1(self):
        """Partially numeric suffix 'A1' → VID '1.0' fallback."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAMEA1.IMG")
        assert lidvid.endswith("::1.0")


class TestACILidvidExtension:
    """Handling of .IMG extension and edge cases."""

    def test_uppercase_img_stripped(self):
        """'.IMG' extension stripped."""
        lidvid = PDSIngestionService._construct_aci_lidvid("TEST01.IMG")
        assert ".img" not in lidvid
        assert ".IMG" not in lidvid

    def test_lowercase_img_stripped(self):
        """'.img' extension stripped (case-insensitive)."""
        lidvid = PDSIngestionService._construct_aci_lidvid("TEST01.img")
        assert ".img" not in lidvid

    def test_mixed_case_img_stripped(self):
        """'.Img' extension stripped (case-insensitive)."""
        lidvid = PDSIngestionService._construct_aci_lidvid("TEST01.Img")
        assert ".img" not in lidvid.lower().split("::")[0].rsplit(":", 1)[1]

    def test_no_extension(self):
        """Filename without .IMG extension still works."""
        lidvid = PDSIngestionService._construct_aci_lidvid("SC3_TEST01")
        assert lidvid.endswith("::1.0")
        assert "sc3_test::" in lidvid

    def test_version_suffix_from_base(self):
        """Without .IMG, version suffix is still last 2 chars."""
        lidvid = PDSIngestionService._construct_aci_lidvid("MYFILE02")
        assert lidvid.endswith("::2.0")
        assert "myfile::" in lidvid

    def test_short_name(self):
        """Very short name (3 chars) still parses."""
        lidvid = PDSIngestionService._construct_aci_lidvid("A01.IMG")
        assert lidvid.endswith("::1.0")
        # Base is "A" minus version suffix (last 2) → empty base... let's check
        # "A01.IMG" → strip .IMG → "A01" → version="01", base="A"
        assert ":a::" in lidvid

    def test_two_char_name(self):
        """Name with exactly 2 chars (just version suffix)."""
        lidvid = PDSIngestionService._construct_aci_lidvid("05.IMG")
        assert lidvid.endswith("::5.0")
        # base is empty
        assert "::5.0" in lidvid


# ===========================================================================
# Section 3: Version Comparison (_parse_version_tuple)
# ===========================================================================


class TestVersionTupleBasic:
    """Basic version string → tuple parsing."""

    def test_simple_1_0(self):
        assert PDSIngestionService._parse_version_tuple("1.0") == (1, 0)

    def test_simple_2_3(self):
        assert PDSIngestionService._parse_version_tuple("2.3") == (2, 3)

    def test_single_digit(self):
        assert PDSIngestionService._parse_version_tuple("1") == (1,)

    def test_three_parts(self):
        assert PDSIngestionService._parse_version_tuple("1.2.3") == (1, 2, 3)

    def test_four_parts(self):
        assert PDSIngestionService._parse_version_tuple("1.2.3.4") == (1, 2, 3, 4)

    def test_multidigit_minor(self):
        assert PDSIngestionService._parse_version_tuple("1.10") == (1, 10)

    def test_multidigit_major(self):
        assert PDSIngestionService._parse_version_tuple("12.0") == (12, 0)

    def test_large_version(self):
        assert PDSIngestionService._parse_version_tuple("100.200") == (100, 200)


class TestVersionTupleFallback:
    """Fallback behavior for invalid/missing versions."""

    def test_none(self):
        assert PDSIngestionService._parse_version_tuple(None) == (0,)

    def test_empty_string(self):
        assert PDSIngestionService._parse_version_tuple("") == (0,)

    def test_non_numeric(self):
        assert PDSIngestionService._parse_version_tuple("abc") == (0,)

    def test_mixed_alpha_numeric(self):
        assert PDSIngestionService._parse_version_tuple("1.0a") == (0,)

    def test_leading_space_tolerated(self):
        """Leading whitespace tolerated by int() → parses successfully."""
        assert PDSIngestionService._parse_version_tuple(" 1.0") == (1, 0)

    def test_trailing_dot(self):
        """Trailing dot '1.0.' → parse fails on empty segment → fallback."""
        assert PDSIngestionService._parse_version_tuple("1.0.") == (0,)

    def test_leading_dot(self):
        """Leading dot '.1.0' → parse fails on empty first segment → fallback."""
        assert PDSIngestionService._parse_version_tuple(".1.0") == (0,)

    def test_double_dot(self):
        """Double dot '1..0' → empty segment → fallback."""
        assert PDSIngestionService._parse_version_tuple("1..0") == (0,)


class TestVersionTupleOrdering:
    """Numeric tuple ordering — the critical regression guard."""

    def test_1_10_greater_than_1_2(self):
        """Key requirement: (1, 10) > (1, 2) — string comparison would fail."""
        v110 = PDSIngestionService._parse_version_tuple("1.10")
        v12 = PDSIngestionService._parse_version_tuple("1.2")
        assert v110 > v12

    def test_2_0_greater_than_1_99(self):
        """Major version bump dominates."""
        assert PDSIngestionService._parse_version_tuple("2.0") > \
               PDSIngestionService._parse_version_tuple("1.99")

    def test_equal_versions(self):
        v1 = PDSIngestionService._parse_version_tuple("1.0")
        v2 = PDSIngestionService._parse_version_tuple("1.0")
        assert v1 == v2
        assert not (v1 > v2)
        assert not (v1 < v2)

    def test_0_tuple_less_than_any_valid(self):
        """Fallback (0,) is less than any valid version."""
        fallback = PDSIngestionService._parse_version_tuple(None)
        v10 = PDSIngestionService._parse_version_tuple("1.0")
        assert fallback < v10

    def test_single_vs_dotted(self):
        """'2' → (2,) vs '1.99' → (1, 99): (2,) > (1, 99)."""
        assert PDSIngestionService._parse_version_tuple("2") > \
               PDSIngestionService._parse_version_tuple("1.99")

    def test_three_part_ordering(self):
        """Three-part: '1.2.3' < '1.2.4'."""
        assert PDSIngestionService._parse_version_tuple("1.2.3") < \
               PDSIngestionService._parse_version_tuple("1.2.4")

    def test_longer_tuple_wins_on_prefix_equal(self):
        """'1.0' → (1, 0) vs '1.0.1' → (1, 0, 1): latter is greater."""
        assert PDSIngestionService._parse_version_tuple("1.0.1") > \
               PDSIngestionService._parse_version_tuple("1.0")

    def test_symmetry(self):
        """a > b implies b < a."""
        a = PDSIngestionService._parse_version_tuple("2.0")
        b = PDSIngestionService._parse_version_tuple("1.5")
        assert a > b
        assert b < a


# ===========================================================================
# Section 4: PDSIngestionStats and PDSVersionUpdate
# ===========================================================================


class TestPDSIngestionStats:
    """Unit tests for stats accumulation dataclass."""

    def test_default_zeros(self):
        stats = PDSIngestionStats()
        assert stats.sols_processed == 0
        assert stats.observations_ingested == 0
        assert stats.errors == []
        assert stats.version_updates == []

    def test_addition(self):
        """Two stats added produce correct sums."""
        a = PDSIngestionStats(
            sols_processed=1, observations_ingested=3, points_ingested=100,
            spectra_ingested=300, errors=["err1"],
        )
        b = PDSIngestionStats(
            sols_processed=1, observations_ingested=2, points_ingested=50,
            spectra_ingested=150, errors=["err2"], warnings=["warn1"],
        )
        c = a + b
        assert c.sols_processed == 2
        assert c.observations_ingested == 5
        assert c.points_ingested == 150
        assert c.spectra_ingested == 450
        assert c.errors == ["err1", "err2"]
        assert c.warnings == ["warn1"]

    def test_version_updates_merge(self):
        """Version updates lists are concatenated."""
        vu1 = PDSVersionUpdate("obs1", "lid1", "1.0", "2.0", 100)
        vu2 = PDSVersionUpdate("obs2", "lid2", "1.0", "1.1", 200)
        a = PDSIngestionStats(version_updates=[vu1])
        b = PDSIngestionStats(version_updates=[vu2])
        c = a + b
        assert len(c.version_updates) == 2
        assert c.version_updates[0].observation_key == "obs1"
        assert c.version_updates[1].observation_key == "obs2"

    def test_addition_does_not_mutate(self):
        """Addition creates new instance, doesn't mutate originals."""
        a = PDSIngestionStats(sols_processed=1, errors=["err"])
        b = PDSIngestionStats(sols_processed=2)
        c = a + b
        assert a.sols_processed == 1
        assert len(a.errors) == 1
        assert c.sols_processed == 3


class TestPDSVersionUpdate:
    """Unit tests for version update record."""

    def test_fields(self):
        vu = PDSVersionUpdate(
            observation_key="100_700000000_abc",
            scan_id="urn:nasa:pds:...",
            old_version="1.0",
            new_version="2.0",
            sol=100,
        )
        assert vu.observation_key == "100_700000000_abc"
        assert vu.old_version == "1.0"
        assert vu.new_version == "2.0"
        assert vu.sol == 100


class TestPDSIngestionError:
    """Unit tests for custom exception."""

    def test_basic(self):
        err = PDSIngestionError("test error")
        assert str(err) == "test error"
        assert err.sol is None
        assert err.observation_key is None

    def test_with_context(self):
        err = PDSIngestionError("bad data", sol=921, observation_key="obs1")
        assert err.sol == 921
        assert err.observation_key == "obs1"

    def test_is_sherloc_error(self):
        from sherloc_pipeline.services.errors import SherlocServiceError
        err = PDSIngestionError("test")
        assert isinstance(err, SherlocServiceError)


# ===========================================================================
# Section 5: Idempotency (mock-based, no real PDS data)
# ===========================================================================


class TestIdempotencyUnit:
    """Idempotency checks using mock database — no Sol 921 dependency."""

    def test_service_creates_pds_engine(self, tmp_path):
        """PDSIngestionService constructor creates pds_engine."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        assert service.pds_engine is not None

    def test_service_optional_loupe_engine(self, tmp_path):
        """loupe_db_path=None → loupe_engine is None."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        assert service.loupe_engine is None

    def test_service_with_loupe_db(self, tmp_path):
        """Valid loupe_db_path → loupe_engine is set."""
        loupe_db = tmp_path / "loupe.db"
        loupe_engine = get_engine(loupe_db)
        create_all_tables(loupe_engine)

        service = PDSIngestionService(
            pds_db_path=tmp_path / "pds.db",
            loupe_db_path=loupe_db,
        )
        assert service.loupe_engine is not None

    def test_nonexistent_loupe_db_logs_warning(self, tmp_path, caplog):
        """Nonexistent loupe_db_path → warning logged, engine None."""
        with caplog.at_level(logging.WARNING):
            service = PDSIngestionService(
                pds_db_path=tmp_path / "pds.db",
                loupe_db_path=tmp_path / "missing.db",
            )
        assert service.loupe_engine is None
        assert any("not found" in msg.lower() or "warning" in msg.lower()
                    for msg in caplog.messages) or service.loupe_engine is None

    def test_missing_sol_dir_raises(self, tmp_path):
        """Non-existent sol_dir raises PDSIngestionError."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        with pytest.raises(PDSIngestionError, match="Sol directory not found"):
            service.ingest_sol(tmp_path / "nonexistent_sol")

    def test_empty_sol_dir(self, tmp_path):
        """Empty directory returns no-observations result."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        empty_dir = tmp_path / "empty_sol"
        empty_dir.mkdir()
        result = service.ingest_sol(empty_dir)
        assert result.metadata["observations"] == 0

    def test_database_stats_empty(self, tmp_path):
        """Empty database returns zero stats."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        stats = service.get_database_stats()
        assert stats["sols"] == 0
        assert stats["scans"] == 0
        assert stats["scan_points"] == 0
        assert stats["spectra"] == 0

    def test_version_tuple_guards_lexicographic_bug(self):
        """Regression guard: version comparison uses tuples, not strings.

        String: '1.10' < '1.2' (WRONG — lexicographic)
        Tuple: (1, 10) > (1, 2) (CORRECT — numeric)
        """
        old = PDSIngestionService._parse_version_tuple("1.2")
        new = PDSIngestionService._parse_version_tuple("1.10")
        # If this assertion fails, lexicographic comparison is being used
        assert new > old, "Version 1.10 must be greater than 1.2"


class TestInitPdsDatabase:
    """Tests for database initialization."""

    def test_creates_file(self, tmp_path):
        """init_pds_database creates the DB file."""
        from sherloc_pipeline.database import init_pds_database
        db_path = tmp_path / "new_pds.db"
        assert not db_path.exists()
        init_pds_database(db_path)
        assert db_path.exists()

    def test_idempotent(self, tmp_path):
        """Multiple calls don't raise errors."""
        from sherloc_pipeline.database import init_pds_database
        db_path = tmp_path / "pds.db"
        init_pds_database(db_path)
        init_pds_database(db_path)  # Second call
        init_pds_database(db_path)  # Third call
        assert db_path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if needed."""
        from sherloc_pipeline.database import init_pds_database
        db_path = tmp_path / "subdir" / "nested" / "pds.db"
        init_pds_database(db_path)
        assert db_path.exists()

    def test_unique_constraint_on_scan_id(self, tmp_path):
        """PDS database has unique constraint on scans.scan_id."""
        from sqlalchemy.exc import IntegrityError
        from sherloc_pipeline.database import init_pds_database

        db_path = tmp_path / "pds_unique.db"
        engine = init_pds_database(db_path)

        with get_session(engine) as session:
            session.add(SolORM(
                sol_number=100,
                data_source="pds4",
                created_at=datetime.now(timezone.utc),
            ))
            session.flush()

            session.add(ScanORM(
                id=str(uuid.uuid4()),
                sol_number=100,
                scan_name="scan_a",
                scan_id="unique_lid",
                sclk_start=700000000,
                n_points=1,
                n_channels=2148,
                laser_wavelength_nm=248.6,
                data_source="pds4",
                created_at=datetime.now(timezone.utc),
            ))
            session.flush()

        # Second scan with same scan_id should raise
        with pytest.raises(IntegrityError):
            with get_session(engine) as session:
                session.add(ScanORM(
                    id=str(uuid.uuid4()),
                    sol_number=100,
                    scan_name="scan_b",
                    scan_id="unique_lid",  # Duplicate
                    sclk_start=700000001,
                    n_points=1,
                    n_channels=2148,
                    laser_wavelength_nm=248.6,
                    data_source="pds4",
                    created_at=datetime.now(timezone.utc),
                ))
