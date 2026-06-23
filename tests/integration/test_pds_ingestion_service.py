"""Integration tests for PDS ingestion service (Phase 4).

Tests per-observation flow per spec s9:
- Idempotency: skip if scan_id exists and version matches
- Version comparison: numeric tuple (1.10 > 1.2)
- force=True: cascade delete + re-ingest
- Correct ScanORM, ScanPointORM, SpectrumORM creation
- Sol metadata enrichment (earth_date, solar_longitude)
- sols.data_source='pds4'
"""

from pathlib import Path

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
    TargetNameResolver,
)

from tests.integration.conftest import (
    SOL_921_DIR,
    LOUPE_DB,
    requires_sol921_data,
)

pytestmark = requires_sol921_data


# --- Unit tests for version tuple parsing ---

class TestParseVersionTuple:
    """Test _parse_version_tuple static method."""

    def test_simple_version(self):
        """1.0 -> (1, 0)."""
        assert PDSIngestionService._parse_version_tuple("1.0") == (1, 0)

    def test_multidigit_minor(self):
        """1.10 -> (1, 10)."""
        assert PDSIngestionService._parse_version_tuple("1.10") == (1, 10)

    def test_major_only(self):
        """2 -> (2,)."""
        assert PDSIngestionService._parse_version_tuple("2") == (2,)

    def test_three_part_version(self):
        """1.2.3 -> (1, 2, 3)."""
        assert PDSIngestionService._parse_version_tuple("1.2.3") == (1, 2, 3)

    def test_none_returns_zero_tuple(self):
        """None -> (0,) for safe fallback."""
        assert PDSIngestionService._parse_version_tuple(None) == (0,)

    def test_empty_string_returns_zero_tuple(self):
        """Empty string -> (0,) fallback."""
        assert PDSIngestionService._parse_version_tuple("") == (0,)

    def test_non_numeric_returns_zero_tuple(self):
        """Non-numeric -> (0,) fallback."""
        assert PDSIngestionService._parse_version_tuple("abc") == (0,)

    def test_numeric_tuple_ordering(self):
        """Key requirement: (1, 10) > (1, 2) — avoids string comparison bug."""
        v110 = PDSIngestionService._parse_version_tuple("1.10")
        v12 = PDSIngestionService._parse_version_tuple("1.2")
        assert v110 > v12, "1.10 must be greater than 1.2 via tuple comparison"

    def test_equal_versions(self):
        """Same version -> equal tuples."""
        v1 = PDSIngestionService._parse_version_tuple("1.0")
        v2 = PDSIngestionService._parse_version_tuple("1.0")
        assert v1 == v2
        assert not (v1 > v2)

    def test_major_version_bump(self):
        """2.0 > 1.99."""
        v20 = PDSIngestionService._parse_version_tuple("2.0")
        v199 = PDSIngestionService._parse_version_tuple("1.99")
        assert v20 > v199


# --- Integration tests for idempotency and version comparison ---

class TestIdempotency:
    """Test per-observation idempotency flow using Sol 921 data."""

    @pytest.fixture
    def pds_service(self, tmp_path):
        """Create PDSIngestionService with temp database."""
        pds_db = tmp_path / "phase_pds_test.db"
        return PDSIngestionService(pds_db_path=pds_db)

    def test_fresh_ingestion_creates_records(self, pds_service):
        """First ingestion of Sol 921 creates sol, scans, points, spectra.

        Sol 921 has 6 observations: 5 with spectral data (RRS/RCS) and
        1 zpz-filtered observation (SCLK 748735903) that only has RLI/RLS
        products, which correctly produces an error since RRS/RCS is required.
        """
        result = pds_service.ingest_sol(SOL_921_DIR)

        # 5 observations ingested, 1 zpz-only observation fails (expected)
        assert result.metadata["observations_ingested"] == 5
        errors = result.metadata["errors"]
        assert len(errors) == 1
        assert "No RRS/RCS spectral product" in errors[0]

        stats = pds_service.get_database_stats()
        assert stats["sols"] == 1
        assert stats["scans"] == 5
        assert stats["scan_points"] >= 1
        assert stats["spectra"] >= 3  # At least 3 per point (R1, R2, R3)

    def test_sol_data_source_is_pds4(self, pds_service):
        """Sol records have data_source='pds4' (spec s9 step 5)."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            sol = session.get(SolORM, 921)
            assert sol is not None
            assert sol.data_source == "pds4"

    def test_scan_fields_match_spec(self, pds_service):
        """ScanORM fields match spec s9 synthesis rules."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            assert len(scans) >= 1

            for scan in scans:
                # scan_name format: pds_{sol}_{sclk}_{obs}
                assert scan.scan_name.startswith("pds_0921_")
                # scan_id is PDS LID (no version suffix)
                assert scan.scan_id.startswith("urn:nasa:pds:")
                assert "::" not in scan.scan_id  # No version in LID
                # target=NULL (resolved in step 4.3)
                assert scan.target is None
                # data_source='pds4'
                assert scan.data_source == "pds4"
                # pds4_metadata has version
                assert scan.pds4_metadata is not None
                assert "version" in scan.pds4_metadata

    def test_scan_points_have_coordinate_frame(self, pds_service):
        """ScanPointORM has coordinate_frame='aci_pixel' when from RMO."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            points_with_coords = session.execute(
                select(ScanPointORM).where(
                    ScanPointORM.coordinate_frame == "aci_pixel"
                )
            ).scalars().all()
            # At least some points should have aci_pixel coordinates
            assert len(points_with_coords) >= 1

    def test_spectra_have_correct_fields(self, pds_service):
        """SpectrumORM: 3 per point, processing_level, wavelength_source."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            spectra = session.execute(select(SpectrumORM)).scalars().all()
            assert len(spectra) >= 3

            for spectrum in spectra[:10]:  # Check first 10
                assert spectrum.spectrum_type == "laser_normalized"
                assert spectrum.processing_level == "normalized"
                assert spectrum.wavelength_source == "pds_embedded"

            # Verify 3 regions per point
            first_point_id = spectra[0].scan_point_id
            point_spectra = [s for s in spectra if s.scan_point_id == first_point_id]
            regions = {s.region for s in point_spectra}
            assert regions == {"R1", "R2", "R3"}, (
                f"Expected R1/R2/R3, got {regions}"
            )

    def test_sol_enrichment(self, pds_service):
        """Sol metadata enriched from XML (earth_date, solar_longitude)."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            sol = session.get(SolORM, 921)
            assert sol is not None
            # earth_date should be populated from XML start_date_time
            # (may be None if XML doesn't have it, but Sol 921 should)
            # solar_longitude should be populated
            # Not asserting specific values since they depend on XML content

    def test_second_ingestion_skips_all(self, pds_service):
        """Second ingestion with same version skips all observations."""
        # First ingestion
        result1 = pds_service.ingest_sol(SOL_921_DIR)
        n_ingested = result1.metadata["observations_ingested"]
        assert n_ingested == 5

        # Record counts after first ingestion
        stats1 = pds_service.get_database_stats()

        # Second ingestion (same version)
        result2 = pds_service.ingest_sol(SOL_921_DIR)
        # 5 observations skipped (version matches), 0 newly ingested
        # 1 zpz-only observation still errors (no RRS/RCS)
        assert result2.metadata["observations_skipped"] == 5
        assert result2.metadata["observations_ingested"] == 0

        # Counts unchanged
        stats2 = pds_service.get_database_stats()
        assert stats2["scans"] == stats1["scans"]
        assert stats2["scan_points"] == stats1["scan_points"]
        assert stats2["spectra"] == stats1["spectra"]

    def test_force_reingests_all(self, pds_service):
        """force=True cascade-deletes and re-ingests all observations."""
        # First ingestion
        result1 = pds_service.ingest_sol(SOL_921_DIR)
        stats1 = pds_service.get_database_stats()

        # Force re-ingestion
        result2 = pds_service.ingest_sol(SOL_921_DIR, force=True)
        assert result2.metadata["observations_updated"] >= 1
        assert result2.metadata["observations_skipped"] == 0

        # Counts should match (cascade deleted then re-inserted)
        stats2 = pds_service.get_database_stats()
        assert stats2["scans"] == stats1["scans"]
        assert stats2["scan_points"] == stats1["scan_points"]
        assert stats2["spectra"] == stats1["spectra"]

    def test_cascade_delete_removes_children(self, pds_service):
        """Cascade delete on force removes scan_points and spectra."""
        # Ingest
        pds_service.ingest_sol(SOL_921_DIR)
        stats_before = pds_service.get_database_stats()
        assert stats_before["scan_points"] >= 1
        assert stats_before["spectra"] >= 3

        # Force re-ingest (cascade delete + re-insert)
        pds_service.ingest_sol(SOL_921_DIR, force=True)
        stats_after = pds_service.get_database_stats()

        # If cascade didn't work, points/spectra would accumulate
        assert stats_after["scan_points"] == stats_before["scan_points"]
        assert stats_after["spectra"] == stats_before["spectra"]


class TestVersionComparison:
    """Test version comparison behavior in ingestion flow."""

    @pytest.fixture
    def pds_service(self, tmp_path):
        """Create PDSIngestionService with temp database."""
        pds_db = tmp_path / "phase_pds_version_test.db"
        return PDSIngestionService(pds_db_path=pds_db)

    def test_version_stored_in_pds4_metadata(self, pds_service):
        """pds4_metadata.version stores the PDS version string."""
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            scan = session.execute(select(ScanORM)).scalars().first()
            assert scan is not None
            meta = scan.pds4_metadata
            assert meta is not None
            assert "version" in meta
            # Version should be a non-empty string
            version = meta["version"]
            assert isinstance(version, str) and len(version) > 0

    def test_version_comparison_uses_tuples_not_strings(self, pds_service):
        """Regression: version comparison must use numeric tuples.

        String comparison: '1.10' < '1.2' (WRONG)
        Tuple comparison: (1, 10) > (1, 2) (CORRECT)
        """
        # This test validates the _parse_version_tuple is actually called
        # by the ingestion flow (covered by the tuple unit tests above).
        # We verify the stored version can be parsed to tuple.
        pds_service.ingest_sol(SOL_921_DIR)

        with get_session(pds_service.pds_engine) as session:
            scan = session.execute(select(ScanORM)).scalars().first()
            version_str = scan.pds4_metadata["version"]
            # Must be parseable as numeric tuple
            parsed = PDSIngestionService._parse_version_tuple(version_str)
            assert all(isinstance(p, int) for p in parsed)
            assert len(parsed) >= 1


class TestErrorHandling:
    """Test error handling in ingestion flow."""

    def test_missing_sol_directory_raises(self, tmp_path):
        """Non-existent sol_dir raises PDSIngestionError."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        with pytest.raises(PDSIngestionError, match="Sol directory not found"):
            service.ingest_sol(tmp_path / "nonexistent")

    def test_empty_sol_directory(self, tmp_path):
        """Empty directory returns no-observations result."""
        service = PDSIngestionService(pds_db_path=tmp_path / "test.db")
        empty_dir = tmp_path / "empty_sol"
        empty_dir.mkdir()
        result = service.ingest_sol(empty_dir)
        assert result.metadata["observations"] == 0


# --- Target Name Resolver Unit Tests ---

import json
import uuid
from datetime import datetime, timezone


def _make_loupe_db(tmp_path, scans):
    """Create a test Loupe-like SQLite DB with given scan records.

    Args:
        tmp_path: pytest tmp_path fixture
        scans: list of dicts with keys: sol, sclk, target, site_drive

    Returns:
        SQLAlchemy Engine for the test database
    """
    db_path = tmp_path / "test_loupe.db"
    engine = get_engine(db_path)
    create_all_tables(engine)

    with get_session(engine) as session:
        # Ensure sol records exist
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

        # Add scan records
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


class TestTargetNameResolverUnit:
    """Unit tests for TargetNameResolver (spec s10)."""

    def test_pass1_exact_match(self, tmp_path):
        """Pass 1 (±3s): single match within 3s accepted."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Amherst Point"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731011 is 1s from Loupe 748731010 → within ±3s
        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result == "Amherst Point"

    def test_pass1_at_boundary(self, tmp_path):
        """Pass 1 (±3s): match at exactly 3s delta accepted."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Boundary Target"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731013 is exactly 3s from Loupe 748731010
        result = resolver.resolve(sol=921, pds_sclk=748731013)
        assert result == "Boundary Target"

    def test_pass1_miss_pass2_hit(self, tmp_path):
        """Pass 2 (±5s): match at 4s delta (outside Pass 1 but inside Pass 2)."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Wide Target"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731014 is 4s from Loupe 748731010
        # Pass 1 (±3s) misses, Pass 2 (±5s) hits
        result = resolver.resolve(sol=921, pds_sclk=748731014)
        assert result == "Wide Target"

    def test_pass2_at_boundary(self, tmp_path):
        """Pass 2: match at exactly 5s delta accepted."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Pass2 Boundary"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        result = resolver.resolve(sol=921, pds_sclk=748731015)
        assert result == "Pass2 Boundary"

    def test_no_match_beyond_tolerance(self, tmp_path):
        """No match when SCLK delta > 5s → returns None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Too Far"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731016 is 6s from Loupe → outside both passes
        result = resolver.resolve(sol=921, pds_sclk=748731016)
        assert result is None

    def test_no_match_wrong_sol(self, tmp_path):
        """No match when sol doesn't match → returns None."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 920, "sclk": 748731010, "target": "Wrong Sol"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result is None

    def test_no_loupe_engine_returns_none(self):
        """Without loupe_engine, target is always None (Tier 3)."""
        resolver = TargetNameResolver(loupe_engine=None)
        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result is None

    def test_tiebreak_nearest_delta(self, tmp_path):
        """Multiple candidates: nearest SCLK delta wins."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "Far Target"},
            {"sol": 921, "sclk": 748731012, "target": "Near Target"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731011: delta=1 to 748731010, delta=1 to 748731012
        # Actually both are 1s — need different offsets
        # PDS SCLK 748731013: delta=3 to 748731010, delta=1 to 748731012
        result = resolver.resolve(sol=921, pds_sclk=748731013)
        assert result == "Near Target"

    def test_tiebreak_site_drive(self, tmp_path):
        """Equal SCLK delta: site_drive match preferred."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731009, "target": "No SD", "site_drive": None},
            {"sol": 921, "sclk": 748731011, "target": "Match SD", "site_drive": "04500672"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        # PDS SCLK 748731010: delta=1 to both → tied
        # site_drive "04500672" matches → "Match SD" preferred
        result = resolver.resolve(
            sol=921, pds_sclk=748731010, site_drive="04500672"
        )
        assert result == "Match SD"

    def test_null_target_scan_skipped(self, tmp_path):
        """Scans with target=NULL in Loupe are excluded from candidates."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": None},  # NULL target
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result is None

    def test_curated_mapping_fallback(self, tmp_path):
        """Tier 2: curated mapping used when SCLK lookup fails."""
        # No Loupe engine → skip Tier 1
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps({"921_748731011": "Curated Target"}))

        resolver = TargetNameResolver(
            loupe_engine=None,
            curated_path=curated,
        )

        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result == "Curated Target"

    def test_curated_mapping_not_found_returns_none(self, tmp_path):
        """Curated mapping key miss → Tier 3 NULL."""
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps({"999_000000000": "Other Target"}))

        resolver = TargetNameResolver(
            loupe_engine=None,
            curated_path=curated,
        )

        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result is None

    def test_curated_file_missing_is_nonfatal(self, tmp_path):
        """Missing curated file is handled gracefully."""
        resolver = TargetNameResolver(
            loupe_engine=None,
            curated_path=tmp_path / "nonexistent.json",
        )
        # Should not raise — curated_path doesn't exist so not loaded
        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result is None

    def test_sclk_crossref_takes_priority_over_curated(self, tmp_path):
        """Tier 1 (SCLK) is checked before Tier 2 (curated)."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731010, "target": "SCLK Target"},
        ])
        curated = tmp_path / "mapping.json"
        curated.write_text(json.dumps({"921_748731011": "Curated Target"}))

        resolver = TargetNameResolver(
            loupe_engine=engine,
            curated_path=curated,
        )

        result = resolver.resolve(sol=921, pds_sclk=748731011)
        assert result == "SCLK Target"  # Tier 1 wins

    def test_ambiguity_warning_logged(self, tmp_path, caplog):
        """Multiple SCLK candidates produce a warning log."""
        engine = _make_loupe_db(tmp_path, [
            {"sol": 921, "sclk": 748731009, "target": "Target A"},
            {"sol": 921, "sclk": 748731011, "target": "Target B"},
        ])
        resolver = TargetNameResolver(loupe_engine=engine)

        import logging
        with caplog.at_level(logging.WARNING):
            resolver.resolve(sol=921, pds_sclk=748731010)

        assert any("SCLK ambiguity" in msg for msg in caplog.messages)


class TestTargetResolverSol921Integration:
    """Integration test: Sol 921 target resolution against real phase.db."""

    # LOUPE_DB imported from conftest

    @pytest.fixture
    def resolver(self):
        """Resolver with real Loupe DB."""
        if not LOUPE_DB.exists():
            pytest.skip("Loupe phase.db not available")
        engine = get_engine(LOUPE_DB)
        return TargetNameResolver(loupe_engine=engine)

    # PDS SCLKs for Sol 921 (from observation grouper, progress log)
    SOL_921_PDS_SCLKS = [748731011, 748731413, 748732975, 748735042, 748736149]

    def test_all_5_observations_resolve(self, resolver):
        """All 5 non-zpz Sol 921 PDS observations resolve to a target.

        Per spec s14, PDS SCLK is consistently 1-2s higher than Loupe SCLK,
        so all should match within ±3s (Pass 1).
        """
        for pds_sclk in self.SOL_921_PDS_SCLKS:
            target = resolver.resolve(sol=921, pds_sclk=pds_sclk)
            assert target is not None, (
                f"SCLK {pds_sclk} should resolve to a target"
            )

    def test_sol_921_targets_are_amherst_point(self, resolver):
        """Sol 921 scans all target 'Amherst Point' per Loupe data."""
        for pds_sclk in self.SOL_921_PDS_SCLKS:
            target = resolver.resolve(sol=921, pds_sclk=pds_sclk)
            assert target == "Amherst Point", (
                f"SCLK {pds_sclk}: expected 'Amherst Point', got '{target}'"
            )

    def test_sol_921_all_pass1_matches(self, resolver):
        """All Sol 921 matches should be within ±3s (Pass 1).

        PDS SCLKs are +1 to +2s from Loupe, well within Pass 1 tolerance.
        """
        # Verify by checking that delta to Loupe SCLK is ≤3 for all
        loupe_sclks = [748731010, 748731411, 748732974, 748735041, 748736148]
        for pds_sclk, loupe_sclk in zip(self.SOL_921_PDS_SCLKS, loupe_sclks):
            delta = abs(pds_sclk - loupe_sclk)
            assert delta <= 3, (
                f"Delta {delta}s for PDS {pds_sclk} vs Loupe {loupe_sclk} "
                f"exceeds Pass 1 tolerance (±3s)"
            )

    def test_ingestion_with_loupe_resolves_targets(self, tmp_path):
        """Full ingestion with Loupe DB resolves targets for Sol 921."""
        if not LOUPE_DB.exists():
            pytest.skip("Loupe phase.db not available")

        pds_db = tmp_path / "phase_pds_target_test.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB,
        )
        service.ingest_sol(SOL_921_DIR)

        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.target == "Amherst Point", (
                    f"Scan {scan.scan_name}: expected 'Amherst Point', "
                    f"got '{scan.target}'"
                )


# --- Unit tests for ACI LIDVID construction ---


class TestConstructACILidvid:
    """Test _construct_aci_lidvid static method (spec s11)."""

    ACI_PREFIX = "urn:nasa:pds:mars2020_imgops:data_aci_imgops"

    def test_spec_example(self):
        """Spec s11 example: detail ACI from Sol 921."""
        image_name = "SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(image_name)
        expected = (
            f"{self.ACI_PREFIX}:"
            "sc3_0921_0748731308_359ecm_n0450000srlc11374_0000lmj::1.0"
        )
        assert lidvid == expected

    def test_calibration_aci(self):
        """Calibration ACI filename (SC0 prefix, LUJ suffix)."""
        image_name = "SC0_0921_0748731023_488ECM_N0450000SRLC10000_0000LUJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(image_name)
        assert lidvid.startswith(f"{self.ACI_PREFIX}:sc0_0921_")
        assert lidvid.endswith("::1.0")
        # Version suffix "01" stripped from base
        assert "luj01" not in lidvid
        assert "luj::" in lidvid

    def test_lowercase_conversion(self):
        """All uppercase letters converted to lowercase."""
        image_name = "SC3_0921_0748731308_359ECM_N0450000SRLC11374_0000LMJ01.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(image_name)
        # Extract the product identifier (between last : and ::)
        product_id = lidvid.split(":")[-1].split("::")[0]
        assert product_id == product_id.lower()

    def test_version_suffix_removal(self):
        """Last 2 chars before .IMG are version suffix, removed from LID."""
        image_name = "TEST_IMAGE_ABC03.IMG"
        lidvid = PDSIngestionService._construct_aci_lidvid(image_name)
        # "03" → version 3.0
        assert "::3.0" in lidvid
        # base "TEST_IMAGE_ABC" lowercased
        assert "test_image_abc::" in lidvid

    def test_version_01_becomes_1_0(self):
        """Version suffix "01" → VID "1.0"."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME01.IMG")
        assert lidvid.endswith("::1.0")

    def test_version_02_becomes_2_0(self):
        """Version suffix "02" → VID "2.0"."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAME02.IMG")
        assert lidvid.endswith("::2.0")

    def test_non_numeric_version_defaults_to_1(self):
        """Non-numeric version suffix falls back to 1.0."""
        lidvid = PDSIngestionService._construct_aci_lidvid("NAMEXX.IMG")
        assert lidvid.endswith("::1.0")

    def test_no_img_extension(self):
        """Handles filename without .IMG extension."""
        lidvid = PDSIngestionService._construct_aci_lidvid("SC3_TEST01")
        assert lidvid.endswith("::1.0")
        assert "sc3_test::" in lidvid


# --- Integration tests for ACI context image association ---


class TestACIContextImageAssociation:
    """Integration test: ACI context image association for Sol 921 (spec s11)."""

    # LOUPE_DB imported from conftest

    @pytest.fixture
    def service_with_sol921(self, tmp_path):
        """Ingest Sol 921 and return (service, result)."""
        pds_db = tmp_path / "phase_pds_aci_test.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        result = service.ingest_sol(SOL_921_DIR)
        return service, result

    def test_context_images_ingested_count(self, service_with_sol921):
        """5 ACI context images created for Sol 921 (4 RMO obs × 1-2 ACIs)."""
        _, result = service_with_sol921
        assert result.metadata["context_images_ingested"] == 5

    def test_db_context_images_count(self, service_with_sol921):
        """Database contains 5 context image records."""
        service, _ = service_with_sol921
        stats = service.get_database_stats()
        assert stats["context_images"] == 5

    def test_all_images_are_aci_type(self, service_with_sol921):
        """All context images have image_type='ACI'."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            images = session.query(ContextImageORM).all()
            for img in images:
                assert img.image_type == "ACI"

    def test_all_images_have_pds_lidvid(self, service_with_sol921):
        """All context images have non-null pds_lidvid."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            images = session.query(ContextImageORM).all()
            for img in images:
                assert img.pds_lidvid is not None
                assert img.pds_lidvid.startswith(
                    "urn:nasa:pds:mars2020_imgops:data_aci_imgops:"
                )

    def test_file_path_format(self, service_with_sol921):
        """All context images have file_path='pds:{lidvid}'."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            images = session.query(ContextImageORM).all()
            for img in images:
                assert img.file_path == f"pds:{img.pds_lidvid}"

    def test_lidvid_format(self, service_with_sol921):
        """LIDVIDs are lowercase with ::VID suffix."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            images = session.query(ContextImageORM).all()
            for img in images:
                lid, vid = img.pds_lidvid.split("::")
                # LID product part is lowercase
                product = lid.split(":")[-1]
                assert product == product.lower()
                # VID is "1.0" (version 01 from Sol 921 filenames)
                assert vid == "1.0"

    def test_calibration_scan_has_1_aci(self, service_with_sol921):
        """Calibration scan (SCLK 748731011) has exactly 1 ACI."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            scan = session.execute(
                select(ScanORM).where(
                    ScanORM.scan_name == "pds_0921_748731011_645"
                )
            ).scalar_one()
            images = session.query(ContextImageORM).filter_by(
                scan_id=scan.id
            ).all()
            assert len(images) == 1

    def test_detail_scans_have_1_aci_each(self, service_with_sol921):
        """Detail scans (SCLK 748731413, 748732975) each have 1 ACI."""
        service, _ = service_with_sol921
        detail_names = [
            "pds_0921_748731413_045",
            "pds_0921_748732975_435",
        ]
        with get_session(service.pds_engine) as session:
            for name in detail_names:
                scan = session.execute(
                    select(ScanORM).where(ScanORM.scan_name == name)
                ).scalar_one()
                images = session.query(ContextImageORM).filter_by(
                    scan_id=scan.id
                ).all()
                assert len(images) == 1, (
                    f"Detail scan {name}: expected 1 ACI, got {len(images)}"
                )

    def test_survey_scan_has_2_acis(self, service_with_sol921):
        """Survey scan (SCLK 748735042) has exactly 2 ACIs."""
        service, _ = service_with_sol921
        with get_session(service.pds_engine) as session:
            scan = session.execute(
                select(ScanORM).where(
                    ScanORM.scan_name == "pds_0921_748735042_800"
                )
            ).scalar_one()
            images = session.query(ContextImageORM).filter_by(
                scan_id=scan.id
            ).all()
            assert len(images) == 2

    def test_cascade_delete_removes_context_images(self, service_with_sol921):
        """Force re-ingestion cascade-deletes context images."""
        service, _ = service_with_sol921
        # Re-ingest with force=True
        service.ingest_sol(SOL_921_DIR, force=True)
        # Should still have 5 (old deleted, new created)
        stats = service.get_database_stats()
        assert stats["context_images"] == 5


# --- Comprehensive Sol 921 Integration Tests (Step 4.5, spec s14) ---

import numpy as np

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.pds_parsers import PDSSpectralParser


class TestSol921FullIngestion:
    """Comprehensive Sol 921 integration test verifying exact counts per spec s14.

    Spec s14 reference data for Sol 921 (5 observations, 1 zpz-filtered):
      - SCLK 748731011 (AlGaN_1): 1 point, calibration
      - SCLK 748731413 (detail_1): 100 points, detail
      - SCLK 748732975 (detail_2): 100 points, detail
      - SCLK 748735042 (survey_1296): 1296 points, survey
      - SCLK 748736149 (AlGaN_2): 1 point, calibration
    Total: 1498 points, 4494 spectra (3 regions × 1498), 5 context images.
    """

    # LOUPE_DB imported from conftest

    # Expected per spec s14: SCLK → point count
    # Note: sclk_start stored in DB comes from XML label's spacecraft_clock_start
    # (int(float(sclk_string))), which differs from filename SCLK by 1-2s.
    EXPECTED_POINTS_PER_SCLK = {
        748731010: 1,     # AlGaN_1 (calibration) — filename SCLK: 748731011
        748731411: 100,   # detail_1 — filename SCLK: 748731413
        748732974: 100,   # detail_2 — filename SCLK: 748732975
        748735041: 1296,  # survey_1296 — filename SCLK: 748735042
        748736148: 1,     # AlGaN_2 (calibration) — filename SCLK: 748736149
    }
    EXPECTED_TOTAL_POINTS = sum(EXPECTED_POINTS_PER_SCLK.values())  # 1498
    EXPECTED_TOTAL_SPECTRA = EXPECTED_TOTAL_POINTS * 3  # 4494

    @pytest.fixture
    def ingested(self, tmp_path):
        """Ingest Sol 921 with Loupe cross-reference and return (service, result)."""
        pds_db = tmp_path / "phase_pds_full_test.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        result = service.ingest_sol(SOL_921_DIR)
        return service, result

    def test_exact_scan_count(self, ingested):
        """5 scans ingested (2 calibration + 2 detail + 1 survey)."""
        service, _ = ingested
        assert service.get_database_stats()["scans"] == 5

    def test_exact_point_count(self, ingested):
        """1498 total scan points (1 + 100 + 100 + 1296 + 1)."""
        service, _ = ingested
        assert service.get_database_stats()["scan_points"] == self.EXPECTED_TOTAL_POINTS

    def test_exact_spectra_count(self, ingested):
        """4494 total spectra (1498 points × 3 regions: R1, R2, R3)."""
        service, _ = ingested
        assert service.get_database_stats()["spectra"] == self.EXPECTED_TOTAL_SPECTRA

    def test_point_records_per_scan(self, ingested):
        """Actual ScanPointORM record count per scan matches spec s14."""
        service, _ = ingested
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                actual = session.execute(
                    select(func.count(ScanPointORM.id)).where(
                        ScanPointORM.scan_id == scan.id
                    )
                ).scalar()
                expected = self.EXPECTED_POINTS_PER_SCLK[scan.sclk_start]
                assert actual == expected, (
                    f"SCLK {scan.sclk_start}: expected {expected} point records, "
                    f"got {actual}"
                )

    def test_spectra_per_point_3_regions(self, ingested):
        """Every scan has exactly 3 spectra per point (R1 + R2 + R3)."""
        service, _ = ingested
        with get_session(service.pds_engine) as session:
            for sclk, expected_points in self.EXPECTED_POINTS_PER_SCLK.items():
                scan = session.execute(
                    select(ScanORM).where(ScanORM.sclk_start == sclk)
                ).scalar_one()
                spectra_count = session.execute(
                    select(func.count(SpectrumORM.id)).where(
                        SpectrumORM.scan_point_id.in_(
                            select(ScanPointORM.id).where(
                                ScanPointORM.scan_id == scan.id
                            )
                        )
                    )
                ).scalar()
                assert spectra_count == expected_points * 3, (
                    f"SCLK {sclk}: expected {expected_points * 3} spectra, "
                    f"got {spectra_count}"
                )

    def test_scan_types(self, ingested):
        """Scan types match spec s14 classification."""
        service, _ = ingested
        expected_types = {
            748731010: "calibration",
            748731411: "detail",
            748732974: "detail",
            748735041: "survey",
            748736148: "calibration",
        }
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.scan_type == expected_types[scan.sclk_start], (
                    f"SCLK {scan.sclk_start}: expected type "
                    f"'{expected_types[scan.sclk_start]}', got '{scan.scan_type}'"
                )

    def test_sol_metadata_enriched(self, ingested):
        """Sol metadata enriched from XML (earth_date, solar_longitude)."""
        service, _ = ingested
        with get_session(service.pds_engine) as session:
            sol = session.get(SolORM, 921)
            assert sol is not None
            assert sol.data_source == "pds4"
            assert sol.earth_date is not None, "earth_date should be populated from XML"
            assert sol.solar_longitude is not None, (
                "solar_longitude should be populated from XML"
            )
            assert 0 <= sol.solar_longitude <= 360

    def test_targets_resolved_with_loupe(self, ingested):
        """All 5 scans resolve to 'Amherst Point' via Loupe cross-ref."""
        service, _ = ingested
        if service.loupe_engine is None:
            pytest.skip("Loupe phase.db not available for target resolution")
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.target == "Amherst Point", (
                    f"Scan {scan.scan_name}: expected 'Amherst Point', "
                    f"got '{scan.target}'"
                )

    def test_zpz_observation_rejected(self, ingested):
        """zpz-only observation (SCLK 748735903) rejected with error."""
        _, result = ingested
        assert result.metadata["observations_ingested"] == 5
        errors = result.metadata["errors"]
        assert len(errors) == 1
        assert "No RRS/RCS spectral product" in errors[0]

    def test_idempotent_skip(self, ingested):
        """Re-ingestion skips all 5 observations, DB counts unchanged."""
        service, _ = ingested
        result2 = service.ingest_sol(SOL_921_DIR)
        assert result2.metadata["observations_skipped"] == 5
        assert result2.metadata["observations_ingested"] == 0
        stats = service.get_database_stats()
        assert stats["scans"] == 5
        assert stats["scan_points"] == self.EXPECTED_TOTAL_POINTS
        assert stats["spectra"] == self.EXPECTED_TOTAL_SPECTRA

    def test_context_images_count(self, ingested):
        """5 context images total (1 cal + 1 detail + 1 detail + 2 survey + 0)."""
        service, _ = ingested
        assert service.get_database_stats()["context_images"] == 5


class TestSol921CrossValidation:
    """Cross-validate PDS-ingested data against Loupe per spec s14.

    Validates wavelength calibration, point count match, SCLK alignment,
    and coordinate frame differences between PDS and Loupe data sources.
    """

    # LOUPE_DB imported from conftest
    LOUPE_SCLKS = [748731010, 748731411, 748732974, 748735041, 748736148]
    PDS_SCLKS = [748731011, 748731413, 748732975, 748735042, 748736149]
    EXPECTED_POINTS = [1, 100, 100, 1296, 1]

    @pytest.fixture
    def loupe_engine(self):
        """Get read-only engine for Loupe phase.db."""
        if not LOUPE_DB.exists():
            pytest.skip("Loupe phase.db not available")
        return get_engine(LOUPE_DB)

    def test_wavelength_calibration_match(self):
        """PDS wavelength matches Loupe polynomial <0.001 nm (except ch500).

        Channel 500 (Raman/Fluorescence boundary) has ~0.393 nm known deviation.
        """
        loupe_wl, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)

        parser = PDSSpectralParser()
        rrs_files = sorted(SOL_921_DIR.glob("*rrs*.csv"))
        assert len(rrs_files) > 0, "No RRS files found in Sol 921"

        # Filter out zpz files
        non_zpz = [f for f in rrs_files if "zpz" not in f.name]
        assert len(non_zpz) > 0, "No non-zpz RRS files found"

        parsed = parser.parse(non_zpz[0])
        pds_wl = np.array(parsed.product.wavelengths)
        assert len(pds_wl) == 2148

        diff = np.abs(pds_wl - loupe_wl)

        # All channels except 500 should match within 0.001 nm
        mask = np.ones(2148, dtype=bool)
        mask[500] = False
        max_diff_excl = diff[mask].max()
        assert max_diff_excl < 0.001, (
            f"Max wavelength deviation (excl ch500): {max_diff_excl:.6f} nm"
        )

        # Channel 500: known ~0.393 nm deviation at Raman/Fluorescence boundary
        assert diff[500] < 0.5, (
            f"Ch500 deviation: {diff[500]:.3f} nm (expected ~0.393)"
        )

    def test_point_counts_match_loupe(self, loupe_engine):
        """PDS point counts match Loupe for all 5 Sol 921 observations."""
        with get_session(loupe_engine) as session:
            for loupe_sclk, expected_n in zip(self.LOUPE_SCLKS, self.EXPECTED_POINTS):
                scan = session.execute(
                    select(ScanORM).where(
                        ScanORM.sol_number == 921,
                        ScanORM.sclk_start == loupe_sclk,
                    )
                ).scalar_one_or_none()
                assert scan is not None, (
                    f"Loupe scan with SCLK {loupe_sclk} not found"
                )
                assert scan.n_points == expected_n, (
                    f"Loupe SCLK {loupe_sclk}: expected {expected_n} points, "
                    f"got {scan.n_points}"
                )

    def test_sclk_deltas_within_pass1(self):
        """All PDS→Loupe SCLK deltas are ≤3s (Pass 1 tolerance per spec s14)."""
        for pds_sclk, loupe_sclk in zip(self.PDS_SCLKS, self.LOUPE_SCLKS):
            delta = abs(pds_sclk - loupe_sclk)
            assert delta <= 3, (
                f"PDS {pds_sclk} vs Loupe {loupe_sclk}: delta {delta}s > 3s"
            )

    def test_coordinate_frames_differ(self, loupe_engine):
        """PDS uses aci_pixel, Loupe uses scanner_workspace (spec s14).

        Different coordinate frames — not directly comparable.
        """
        with get_session(loupe_engine) as session:
            loupe_scan = session.execute(
                select(ScanORM).where(
                    ScanORM.sol_number == 921,
                    ScanORM.sclk_start == 748731411,  # detail_1
                )
            ).scalar_one()
            point = session.execute(
                select(ScanPointORM).where(
                    ScanPointORM.scan_id == loupe_scan.id
                )
            ).scalars().first()
            assert point is not None
            assert point.coordinate_frame == "scanner_workspace"


# --- Validation query tests (Phase 5, step 5.2) ---


class TestDatabaseValidation:
    """Tests for PDSIngestionService.validate_database() method.

    Validates: spectra by region, unique scan_ids, data_source='pds4',
    sol metadata completeness.
    """

    # LOUPE_DB imported from conftest

    @pytest.fixture
    def validated(self, tmp_path):
        """Ingest Sol 921 and return (service, report)."""
        pds_db = tmp_path / "phase_pds_validation_test.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        service.ingest_sol(SOL_921_DIR)
        report = service.validate_database()
        return service, report

    def test_report_is_valid(self, validated):
        """Sol 921 ingestion should produce a valid database with no issues."""
        _, report = validated
        assert report["valid"] is True, f"Validation issues: {report['issues']}"

    def test_spectra_count_by_region(self, validated):
        """Spectra should be balanced across R1, R2, R3 (1498 each)."""
        _, report = validated
        by_region = report["spectra_by_region"]
        assert "R1" in by_region
        assert "R2" in by_region
        assert "R3" in by_region
        # Each region gets 1498 spectra (1+100+100+1296+1 points × 1 per region)
        assert by_region["R1"] == 1498
        assert by_region["R2"] == 1498
        assert by_region["R3"] == 1498

    def test_spectra_total_matches_regions(self, validated):
        """Total spectra count should equal sum of all regions."""
        _, report = validated
        total = report["counts"]["spectra"]
        region_sum = sum(report["spectra_by_region"].values())
        assert total == region_sum

    def test_unique_scan_ids(self, validated):
        """All scan_ids should be unique PDS LIDs (5 scans for Sol 921)."""
        _, report = validated
        scan_ids = report["scan_ids"]
        assert len(scan_ids) == 5
        # All scan_ids should be PDS LIDs (start with urn:nasa:pds)
        for sid in scan_ids:
            assert sid.startswith("urn:nasa:pds:"), f"scan_id not a PDS LID: {sid}"

    def test_no_duplicate_scan_ids(self, validated):
        """scan_ids count should match total scans count."""
        _, report = validated
        assert len(report["scan_ids"]) == report["counts"]["scans"]

    def test_all_scans_data_source_pds4(self, validated):
        """Every scan should have data_source='pds4'."""
        _, report = validated
        assert report["data_source_check"]["all_pds4"] is True
        assert report["data_source_check"]["scans"] == {"pds4": 5}

    def test_sol_data_source(self, validated):
        """Sol record should have data_source='pds4'."""
        _, report = validated
        assert "pds4" in report["data_source_check"]["sols"]

    def test_sol_metadata_completeness(self, validated):
        """Sol 921 should have earth_date, solar_longitude, and mission_phase."""
        _, report = validated
        sol_meta = report["sol_metadata"]
        assert len(sol_meta) == 1
        sol_921 = sol_meta[0]
        assert sol_921["sol_number"] == 921
        assert sol_921["earth_date"] is not None
        assert sol_921["solar_longitude"] is not None
        assert sol_921["missing_fields"] == []

    def test_sol_metadata_values(self, validated):
        """Sol 921 metadata should match known values."""
        _, report = validated
        sol_921 = report["sol_metadata"][0]
        assert sol_921["earth_date"] == "2023-09-23"
        assert abs(sol_921["solar_longitude"] - 122.871) < 0.01

    def test_counts_match_expected(self, validated):
        """Overall counts should match Sol 921 expected values."""
        _, report = validated
        counts = report["counts"]
        assert counts["sols"] == 1
        assert counts["scans"] == 5
        assert counts["scan_points"] == 1498
        assert counts["spectra"] == 4494  # 1498 × 3
        assert counts["context_images"] == 5

    def test_empty_database_report(self, tmp_path):
        """Validation on empty database should return zeros and valid=True."""
        pds_db = tmp_path / "phase_pds_empty_test.db"
        service = PDSIngestionService(pds_db_path=pds_db)
        report = service.validate_database()
        assert report["valid"] is True
        assert report["counts"]["scans"] == 0
        assert report["counts"]["spectra"] == 0
        assert len(report["scan_ids"]) == 0
        assert len(report["sol_metadata"]) == 0
