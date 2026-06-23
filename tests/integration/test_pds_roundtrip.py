"""Integration tests: full Sol 921 roundtrip (Step 7.3).

Exercises the complete PDS ingestion pipeline in explicit stages:
  discover → group → parse → ingest → validate

Uses actual Sol 921 data from ./pds/sol_0921/data_processed/.
Verifies counts, targets, metadata, and database integrity per spec s14.
"""

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select, func

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.pds_parsers import (
    PDSLabelParser,
    PDSObservationGrouper,
    PDSSpectralParser,
    PDSRMOParser,
    PDSZpzProductError,
)
from sherloc_pipeline.database import (
    get_engine,
    get_session,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    ContextImageORM,
)
from sherloc_pipeline.models.pds import CORE_PRODUCT_TYPES
from sherloc_pipeline.services.pds_ingestion import PDSIngestionService

from tests.integration.conftest import (
    SOL_921_DIR,
    LOUPE_DB,
    requires_sol921_data,
)

pytestmark = requires_sol921_data

# --- Constants ---

# Spec s14 reference data
SOL_921_SCLK_GROUPS = [748731011, 748731413, 748732975, 748735042, 748735903, 748736149]
SOL_921_NON_ZPZ_SCLKS = [748731011, 748731413, 748732975, 748735042, 748736149]
SOL_921_ZPZ_SCLK = 748735903
# XML-derived SCLKs (int(float(sclk_string))), differ from filename SCLK by 1-2s
SOL_921_DB_SCLKS = [748731010, 748731411, 748732974, 748735041, 748736148]
SOL_921_EXPECTED_POINTS = {
    748731010: 1,     # calibration AlGaN_1
    748731411: 100,   # detail_1
    748732974: 100,   # detail_2
    748735041: 1296,  # survey_1296
    748736148: 1,     # calibration AlGaN_2
}
SOL_921_TOTAL_POINTS = 1498
SOL_921_TOTAL_SPECTRA = 4494  # 1498 × 3 regions
SOL_921_CONTEXT_IMAGES = 5


# ============================================================================
# Stage 1: Discovery
# ============================================================================


class TestStage1Discovery:
    """Stage 1: Discover all CSV products in Sol 921 directory."""

    def test_discovers_all_csv_products(self):
        """52 CSV files discovered as valid PDSProductId instances."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        assert len(products) == 52

    def test_all_products_have_sol_921(self):
        """Every discovered product has sol=921."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        for p in products:
            assert p.sol == 921, f"Product {p.csv_filename} has sol={p.sol}"

    def test_discovers_expected_product_types(self):
        """All 6 core product types present in Sol 921."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        types_found = {p.product_type for p in products}
        # All 6 core types should be present
        for core_type in CORE_PRODUCT_TYPES:
            assert core_type in types_found, (
                f"Core product type '{core_type}' not found. Have: {types_found}"
            )

    def test_discovers_rm_auxiliary_types(self):
        """RM1-RM6 auxiliary types present in Sol 921."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        types_found = {p.product_type for p in products}
        for rm_type in ["rm1", "rm2", "rm3", "rm4", "rm5", "rm6"]:
            assert rm_type in types_found, f"RM type '{rm_type}' not found"

    def test_six_sclk_groups_present(self):
        """6 distinct SCLK values in Sol 921 (5 valid + 1 zpz)."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        sclks = {p.sclk for p in products}
        assert len(sclks) == 6, f"Expected 6 SCLK groups, got {len(sclks)}: {sorted(sclks)}"
        for expected_sclk in SOL_921_SCLK_GROUPS:
            assert expected_sclk in sclks, f"SCLK {expected_sclk} not found"


# ============================================================================
# Stage 2: Grouping & Classification
# ============================================================================


class TestStage2Grouping:
    """Stage 2: Group products by observation and classify."""

    @pytest.fixture
    def grouper(self):
        return PDSObservationGrouper()

    @pytest.fixture
    def groups(self, grouper):
        """Full pipeline grouping with label parser for classification."""
        label_parser = PDSLabelParser()
        return grouper.group_sol_directory(SOL_921_DIR, label_parser=label_parser)

    def test_six_raw_observation_groups(self, grouper):
        """Raw grouping produces 6 observation groups (before zpz filtering)."""
        products = grouper.discover_csv_products(SOL_921_DIR)
        raw_groups = grouper.group_by_observation(products)
        assert len(raw_groups) == 6

    def test_six_groups_after_zpz_filter(self, groups):
        """group_sol_directory returns 6 groups.

        The zpz observation (SCLK 748735903) survives because it has
        2 non-zpz products (rli, rls) even though 8 products are zpz-filtered.
        """
        assert len(groups) == 6

    def test_zpz_observation_has_limited_products(self, groups):
        """zpz observation at SCLK 748735903 retains only rli/rls after zpz filter."""
        zpz_group = [g for g in groups if g.sclk == SOL_921_ZPZ_SCLK]
        assert len(zpz_group) == 1
        g = zpz_group[0]
        assert set(g.products.keys()) == {"rli", "rls"}
        assert len(g.filtered_zpz) == 8  # rrs, rmo, rm1-rm6 all zpz-filtered

    def test_scan_type_classification(self, groups):
        """Scan types correctly classified: 2 calibration, 2 detail, 1 survey."""
        type_counts = {}
        for g in groups:
            t = g.scan_type
            type_counts[t] = type_counts.get(t, 0) + 1
        assert type_counts.get("calibration") == 2, f"Expected 2 calibration, got {type_counts}"
        assert type_counts.get("detail") == 2, f"Expected 2 detail, got {type_counts}"
        assert type_counts.get("survey") == 1, f"Expected 1 survey, got {type_counts}"

    def test_groups_sorted_by_observation_key(self, groups):
        """Groups returned in observation_key sort order."""
        keys = [g.observation_key for g in groups]
        assert keys == sorted(keys)

    def test_five_groups_have_spectral_product(self, groups):
        """5 of 6 groups have RRS or RCS spectral product.

        The zpz observation (SCLK 748735903) has only rli/rls after
        zpz filtering, so it lacks spectral products.
        """
        spectral_groups = [
            g for g in groups
            if "rrs" in g.products or "rcs" in g.products
        ]
        assert len(spectral_groups) == 5

    def test_rrs_rcs_mutually_exclusive(self, groups):
        """No group has both RRS and RCS (spec s8 exclusivity)."""
        for g in groups:
            has_rrs = "rrs" in g.products
            has_rcs = "rcs" in g.products
            assert not (has_rrs and has_rcs), (
                f"Group {g.observation_key} has both RRS and RCS"
            )

    def test_calibration_spectral_products(self, groups):
        """Calibration observations have spectral products.

        srlc10000 uses RCS, srlc16000 uses RRS — product availability
        depends on the specific calibration sequence.
        """
        cal_groups = [g for g in groups if g.scan_type == "calibration"]
        assert len(cal_groups) == 2
        # Each calibration group has either RRS or RCS
        for g in cal_groups:
            has_spectral = "rrs" in g.products or "rcs" in g.products
            assert has_spectral, (
                f"Calibration {g.observation_key} ({g.sequence_code}) "
                f"missing spectral product. Have: {list(g.products.keys())}"
            )

    def test_non_calibration_uses_rrs(self, groups):
        """Detail and survey observations use RRS (not RCS)."""
        for g in groups:
            if g.scan_type in ("detail", "survey"):
                assert "rrs" in g.products, (
                    f"{g.scan_type} {g.observation_key} missing RRS"
                )
                assert "rcs" not in g.products


# ============================================================================
# Stage 3: Parsing
# ============================================================================


class TestStage3Parsing:
    """Stage 3: Parse XML labels, spectral CSVs, and position CSVs."""

    @pytest.fixture
    def groups(self):
        grouper = PDSObservationGrouper()
        label_parser = PDSLabelParser()
        return grouper.group_sol_directory(SOL_921_DIR, label_parser=label_parser)

    @pytest.fixture
    def spectral_groups(self, groups):
        """Groups that have spectral products (RRS/RCS) — excludes zpz-only."""
        return [g for g in groups if "rrs" in g.products or "rcs" in g.products]

    def test_xml_labels_parse_for_all_spectral_groups(self, spectral_groups):
        """Every spectral observation group has a parseable XML label."""
        label_parser = PDSLabelParser()
        for g in spectral_groups:
            # Get spectral product to find XML
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            product = g.products[spectral_type]
            xml_path = SOL_921_DIR / product.xml_filename
            assert xml_path.exists(), f"XML not found: {xml_path}"
            metadata = label_parser.parse_label(xml_path)
            assert metadata.sol_number == 921
            assert metadata.logical_identifier is not None

    def test_xml_metadata_has_sclk(self, spectral_groups):
        """All spectral group XML labels have spacecraft_clock_start."""
        label_parser = PDSLabelParser()
        for g in spectral_groups:
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            product = g.products[spectral_type]
            xml_path = SOL_921_DIR / product.xml_filename
            metadata = label_parser.parse_label(xml_path)
            assert metadata.spacecraft_clock_start is not None
            assert metadata.sclk_start_int > 0

    def test_spectral_csvs_parse_with_2148_channels(self, spectral_groups):
        """All RRS/RCS files parse with 2148 channels per spectrum."""
        spectral_parser = PDSSpectralParser()
        for g in spectral_groups:
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            product = g.products[spectral_type]
            csv_path = SOL_921_DIR / product.csv_filename
            parsed = spectral_parser.parse(csv_path)
            assert len(parsed.product.wavelengths) == 2148
            assert parsed.product.n_channels == 2148
            # At least 1 spectrum per region
            for region_name, data in parsed.spectra.items():
                assert data.shape[1] == 2148, (
                    f"Group {g.observation_key} region {region_name}: "
                    f"expected 2148 channels, got {data.shape[1]}"
                )

    def test_spectral_row_counts_match_classification(self, spectral_groups):
        """Spectral row counts match expected per scan type."""
        spectral_parser = PDSSpectralParser()
        expected_by_type = {"calibration": 1, "detail": 100, "survey": 1296}
        for g in spectral_groups:
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            product = g.products[spectral_type]
            csv_path = SOL_921_DIR / product.csv_filename
            parsed = spectral_parser.parse(csv_path)
            # Total spectra across all regions = N per region × 3 regions
            region_counts = [data.shape[0] for data in parsed.spectra.values()]
            # Each region should have the same count
            assert len(set(region_counts)) == 1, (
                f"Group {g.observation_key}: unequal region counts {region_counts}"
            )
            spectra_per_region = region_counts[0]
            expected = expected_by_type[g.scan_type]
            assert spectra_per_region == expected, (
                f"Group {g.observation_key} ({g.scan_type}): "
                f"expected {expected} spectra/region, got {spectra_per_region}"
            )

    def test_rmo_position_files_parse(self, groups):
        """RMO position files parse for all groups that have them."""
        rmo_parser = PDSRMOParser()
        for g in groups:
            if "rmo" not in g.products:
                continue
            product = g.products["rmo"]
            csv_path = SOL_921_DIR / product.csv_filename
            parsed = rmo_parser.parse(csv_path)
            assert len(parsed.positions) >= 1
            assert len(parsed.wavelength_regions) == 6  # 6 wavelength regions

    def test_rmo_position_counts_match_spectral(self, spectral_groups):
        """RMO position counts match spectral row counts."""
        spectral_parser = PDSSpectralParser()
        rmo_parser = PDSRMOParser()
        for g in spectral_groups:
            if "rmo" not in g.products:
                continue
            # Get spectral count
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            spectral_product = g.products[spectral_type]
            spectral = spectral_parser.parse(SOL_921_DIR / spectral_product.csv_filename)
            spectra_per_region = list(spectral.spectra.values())[0].shape[0]
            # Get position count
            rmo_product = g.products["rmo"]
            rmo = rmo_parser.parse(SOL_921_DIR / rmo_product.csv_filename)
            assert len(rmo.positions) == spectra_per_region, (
                f"Group {g.observation_key}: RMO positions ({len(rmo.positions)}) "
                f"!= spectral rows ({spectra_per_region})"
            )

    def test_wavelength_arrays_identical_across_rrs(self, groups):
        """All non-calibration RRS files share identical wavelength arrays."""
        spectral_parser = PDSSpectralParser()
        wavelengths = []
        for g in groups:
            if "rrs" not in g.products:
                continue
            product = g.products["rrs"]
            parsed = spectral_parser.parse(SOL_921_DIR / product.csv_filename)
            wavelengths.append(np.array(parsed.product.wavelengths))

        assert len(wavelengths) >= 2, "Need at least 2 RRS files to compare"
        for i in range(1, len(wavelengths)):
            np.testing.assert_array_equal(
                wavelengths[0], wavelengths[i],
                err_msg=f"RRS wavelength arrays differ between file 0 and {i}",
            )

    def test_pds_wavelength_matches_loupe_polynomial(self, groups):
        """PDS wavelength matches Loupe polynomial <0.001 nm (except ch500).

        Channel 500 (Raman/Fluorescence boundary) has ~0.393 nm known deviation.
        """
        spectral_parser = PDSSpectralParser()
        rrs_group = [g for g in groups if "rrs" in g.products][0]
        product = rrs_group.products["rrs"]
        parsed = spectral_parser.parse(SOL_921_DIR / product.csv_filename)
        pds_wl = np.array(parsed.product.wavelengths)

        loupe_wl, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)

        diff = np.abs(pds_wl - loupe_wl)
        mask = np.ones(2148, dtype=bool)
        mask[500] = False
        assert diff[mask].max() < 0.001, (
            f"Max wavelength deviation (excl ch500): {diff[mask].max():.6f} nm"
        )
        assert diff[500] < 0.5, f"Ch500 deviation: {diff[500]:.3f} nm"

    def test_zpz_products_rejected_by_spectral_parser(self):
        """zpz-flagged spectral CSV raises PDSZpzProductError."""
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        zpz_rrs = [p for p in products if p.sclk == SOL_921_ZPZ_SCLK and p.product_type == "rrs"]
        if not zpz_rrs:
            pytest.skip("No zpz RRS product found at SCLK 748735903")
        spectral_parser = PDSSpectralParser()
        with pytest.raises(PDSZpzProductError):
            spectral_parser.parse(SOL_921_DIR / zpz_rrs[0].csv_filename)


# ============================================================================
# Stage 4: Ingestion
# ============================================================================


class TestStage4Ingestion:
    """Stage 4: Full ingestion of Sol 921 into phase_pds.db."""

    @pytest.fixture
    def service_result(self, tmp_path):
        """Ingest Sol 921 and return (service, result)."""
        pds_db = tmp_path / "phase_pds_roundtrip.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        result = service.ingest_sol(SOL_921_DIR)
        return service, result

    def test_5_observations_ingested(self, service_result):
        """5 observations ingested (2 cal + 2 detail + 1 survey)."""
        _, result = service_result
        assert result.metadata["observations_ingested"] == 5

    def test_1_observation_rejected(self, service_result):
        """1 zpz-only observation rejected (no RRS/RCS)."""
        _, result = service_result
        errors = result.metadata["errors"]
        assert len(errors) == 1
        assert "No RRS/RCS spectral product" in errors[0]

    def test_exact_scan_count(self, service_result):
        """5 scans in database."""
        service, _ = service_result
        assert service.get_database_stats()["scans"] == 5

    def test_exact_point_count(self, service_result):
        """1498 scan points (1 + 100 + 100 + 1296 + 1)."""
        service, _ = service_result
        assert service.get_database_stats()["scan_points"] == SOL_921_TOTAL_POINTS

    def test_exact_spectra_count(self, service_result):
        """4494 spectra (1498 points × 3 regions)."""
        service, _ = service_result
        assert service.get_database_stats()["spectra"] == SOL_921_TOTAL_SPECTRA

    def test_exact_context_images_count(self, service_result):
        """5 ACI context images."""
        service, _ = service_result
        assert service.get_database_stats()["context_images"] == SOL_921_CONTEXT_IMAGES

    def test_point_counts_per_scan(self, service_result):
        """Per-scan point counts match spec s14."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                actual = session.execute(
                    select(func.count(ScanPointORM.id)).where(
                        ScanPointORM.scan_id == scan.id
                    )
                ).scalar()
                expected = SOL_921_EXPECTED_POINTS[scan.sclk_start]
                assert actual == expected, (
                    f"SCLK {scan.sclk_start}: expected {expected} points, got {actual}"
                )

    def test_three_spectra_per_point(self, service_result):
        """Every scan point has exactly 3 spectra (R1, R2, R3)."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            # Count spectra grouped by scan_point_id
            results = session.execute(
                select(
                    SpectrumORM.scan_point_id,
                    func.count(SpectrumORM.id).label("n"),
                ).group_by(SpectrumORM.scan_point_id)
            ).all()
            for scan_point_id, count in results:
                assert count == 3, (
                    f"Scan point {scan_point_id}: expected 3 spectra, got {count}"
                )

    def test_spectra_regions_balanced(self, service_result):
        """Equal number of spectra across R1, R2, R3."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            region_counts = session.execute(
                select(SpectrumORM.region, func.count(SpectrumORM.id))
                .group_by(SpectrumORM.region)
            ).all()
            counts = {r: c for r, c in region_counts}
            assert counts["R1"] == SOL_921_TOTAL_POINTS
            assert counts["R2"] == SOL_921_TOTAL_POINTS
            assert counts["R3"] == SOL_921_TOTAL_POINTS

    def test_scan_types_match_spec(self, service_result):
        """Scan types: 2 calibration, 2 detail, 1 survey."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            type_counts = {}
            for scan in scans:
                t = scan.scan_type
                type_counts[t] = type_counts.get(t, 0) + 1
            assert type_counts["calibration"] == 2
            assert type_counts["detail"] == 2
            assert type_counts["survey"] == 1

    def test_all_scans_data_source_pds4(self, service_result):
        """All scans have data_source='pds4'."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.data_source == "pds4"

    def test_scan_ids_are_pds_lids(self, service_result):
        """All scan_ids are PDS LIDs (urn:nasa:pds:...)."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.scan_id.startswith("urn:nasa:pds:"), (
                    f"scan_id not a PDS LID: {scan.scan_id}"
                )
                assert "::" not in scan.scan_id  # LID, not LIDVID

    def test_sol_record_created(self, service_result):
        """Sol 921 record created with data_source='pds4'."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            sol = session.get(SolORM, 921)
            assert sol is not None
            assert sol.data_source == "pds4"

    def test_sol_metadata_enriched(self, service_result):
        """Sol 921 has earth_date and solar_longitude from XML."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            sol = session.get(SolORM, 921)
            assert sol.earth_date is not None
            assert str(sol.earth_date) == "2023-09-23"
            assert sol.solar_longitude is not None
            assert abs(sol.solar_longitude - 122.871) < 0.01

    def test_targets_resolved(self, service_result):
        """All 5 scans resolve to 'Amherst Point' via Loupe cross-ref."""
        service, _ = service_result
        if service.loupe_engine is None:
            pytest.skip("Loupe phase.db not available for target resolution")
        with get_session(service.pds_engine) as session:
            scans = session.execute(select(ScanORM)).scalars().all()
            for scan in scans:
                assert scan.target == "Amherst Point", (
                    f"Scan {scan.scan_name}: expected 'Amherst Point', got '{scan.target}'"
                )

    def test_spectra_fields_correct(self, service_result):
        """Spectra have correct processing fields."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            spectra = session.execute(
                select(SpectrumORM).limit(20)
            ).scalars().all()
            for s in spectra:
                assert s.spectrum_type == "laser_normalized"
                assert s.processing_level == "normalized"
                assert s.wavelength_source == "pds_embedded"

    def test_scan_points_have_coordinate_frame(self, service_result):
        """Scan points from RMO have coordinate_frame='aci_pixel'."""
        service, _ = service_result
        with get_session(service.pds_engine) as session:
            aci_points = session.execute(
                select(func.count(ScanPointORM.id)).where(
                    ScanPointORM.coordinate_frame == "aci_pixel"
                )
            ).scalar()
            # All RMO-sourced points should have aci_pixel
            assert aci_points >= 1


# ============================================================================
# Stage 5: Validation
# ============================================================================


class TestStage5Validation:
    """Stage 5: Run validate_database() and verify the report."""

    @pytest.fixture
    def validated(self, tmp_path):
        """Ingest Sol 921 then validate and return (service, report)."""
        pds_db = tmp_path / "phase_pds_roundtrip_val.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        service.ingest_sol(SOL_921_DIR)
        report = service.validate_database()
        return service, report

    def test_database_is_valid(self, validated):
        """Validation report shows valid=True, no issues."""
        _, report = validated
        assert report["valid"] is True, f"Issues: {report['issues']}"
        assert len(report["issues"]) == 0

    def test_validation_counts(self, validated):
        """Validation report counts match spec s14."""
        _, report = validated
        counts = report["counts"]
        assert counts["sols"] == 1
        assert counts["scans"] == 5
        assert counts["scan_points"] == SOL_921_TOTAL_POINTS
        assert counts["spectra"] == SOL_921_TOTAL_SPECTRA
        assert counts["context_images"] == SOL_921_CONTEXT_IMAGES

    def test_spectra_balanced_by_region(self, validated):
        """R1, R2, R3 each have 1498 spectra."""
        _, report = validated
        by_region = report["spectra_by_region"]
        assert by_region["R1"] == SOL_921_TOTAL_POINTS
        assert by_region["R2"] == SOL_921_TOTAL_POINTS
        assert by_region["R3"] == SOL_921_TOTAL_POINTS

    def test_all_scan_ids_are_pds_lids(self, validated):
        """All 5 scan_ids are PDS LIDs."""
        _, report = validated
        assert len(report["scan_ids"]) == 5
        for sid in report["scan_ids"]:
            assert sid.startswith("urn:nasa:pds:")

    def test_all_data_source_pds4(self, validated):
        """All scans and sols have data_source='pds4'."""
        _, report = validated
        assert report["data_source_check"]["all_pds4"] is True
        assert report["data_source_check"]["scans"] == {"pds4": 5}

    def test_sol_metadata_complete(self, validated):
        """Sol 921 metadata has no missing fields."""
        _, report = validated
        sol_meta = report["sol_metadata"]
        assert len(sol_meta) == 1
        assert sol_meta[0]["sol_number"] == 921
        assert sol_meta[0]["missing_fields"] == []


# ============================================================================
# End-to-End: Full Pipeline in One Test
# ============================================================================


class TestEndToEndRoundtrip:
    """Single test that runs all 5 stages sequentially and verifies the complete pipeline."""

    def test_full_roundtrip(self, tmp_path):
        """discover → group → parse → ingest → validate for Sol 921.

        This is the "golden path" test that exercises every pipeline stage
        in sequence with full verification at each step.
        """
        # Stage 1: Discover
        grouper = PDSObservationGrouper()
        products = grouper.discover_csv_products(SOL_921_DIR)
        assert len(products) == 52, f"Discovery: expected 52 products, got {len(products)}"

        # Stage 2: Group & classify
        label_parser = PDSLabelParser()
        groups = grouper.group_sol_directory(SOL_921_DIR, label_parser=label_parser)
        # 6 groups: 5 with spectral + 1 zpz-partial (rli/rls only)
        assert len(groups) == 6, f"Grouping: expected 6 groups, got {len(groups)}"
        spectral_groups = [
            g for g in groups if "rrs" in g.products or "rcs" in g.products
        ]
        assert len(spectral_groups) == 5
        scan_types = sorted(g.scan_type for g in spectral_groups)
        assert scan_types == ["calibration", "calibration", "detail", "detail", "survey"]

        # Stage 3: Parse all spectral and position data
        spectral_parser = PDSSpectralParser()
        rmo_parser = PDSRMOParser()
        total_spectra_parsed = 0
        total_positions_parsed = 0
        for g in spectral_groups:
            spectral_type = "rrs" if "rrs" in g.products else "rcs"
            product = g.products[spectral_type]
            parsed = spectral_parser.parse(SOL_921_DIR / product.csv_filename)
            assert parsed.product.n_channels == 2148
            region_counts = [d.shape[0] for d in parsed.spectra.values()]
            total_spectra_parsed += sum(region_counts)

            if "rmo" in g.products:
                rmo = rmo_parser.parse(SOL_921_DIR / g.products["rmo"].csv_filename)
                total_positions_parsed += len(rmo.positions)

        assert total_spectra_parsed == SOL_921_TOTAL_SPECTRA, (
            f"Parse: expected {SOL_921_TOTAL_SPECTRA} spectra, got {total_spectra_parsed}"
        )
        # 1497 from RMO (1 + 100 + 100 + 1296); 2nd calibration (srlc16000)
        # has no RMO, its 1 point is created by index fallback during ingestion
        assert total_positions_parsed == SOL_921_TOTAL_POINTS - 1, (
            f"Parse: expected {SOL_921_TOTAL_POINTS - 1} RMO positions, "
            f"got {total_positions_parsed}"
        )

        # Stage 4: Ingest
        pds_db = tmp_path / "phase_pds_e2e.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )
        result = service.ingest_sol(SOL_921_DIR)
        assert result.metadata["observations_ingested"] == 5
        stats = service.get_database_stats()
        assert stats["scans"] == 5
        assert stats["scan_points"] == SOL_921_TOTAL_POINTS
        assert stats["spectra"] == SOL_921_TOTAL_SPECTRA
        assert stats["context_images"] == SOL_921_CONTEXT_IMAGES

        # Stage 5: Validate
        report = service.validate_database()
        assert report["valid"] is True, f"Validation failed: {report['issues']}"
        assert report["counts"]["spectra"] == SOL_921_TOTAL_SPECTRA
        assert report["spectra_by_region"]["R1"] == SOL_921_TOTAL_POINTS
        assert report["data_source_check"]["all_pds4"] is True
        assert report["sol_metadata"][0]["missing_fields"] == []


# ============================================================================
# Idempotency Roundtrip
# ============================================================================


class TestIdempotencyRoundtrip:
    """Verify full roundtrip is idempotent: second run skips, third with force re-ingests."""

    def test_idempotent_double_ingestion(self, tmp_path):
        """Second ingestion skips all, counts unchanged."""
        pds_db = tmp_path / "phase_pds_idem.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )

        # First ingestion
        result1 = service.ingest_sol(SOL_921_DIR)
        assert result1.metadata["observations_ingested"] == 5
        stats1 = service.get_database_stats()

        # Second ingestion (idempotent skip)
        result2 = service.ingest_sol(SOL_921_DIR)
        assert result2.metadata["observations_skipped"] == 5
        assert result2.metadata["observations_ingested"] == 0
        stats2 = service.get_database_stats()
        assert stats2 == stats1

    def test_force_reingestion_preserves_counts(self, tmp_path):
        """Force re-ingestion cascade-deletes and re-creates, counts unchanged."""
        pds_db = tmp_path / "phase_pds_force.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )

        # Initial ingestion
        service.ingest_sol(SOL_921_DIR)
        stats_before = service.get_database_stats()

        # Force re-ingestion
        result = service.ingest_sol(SOL_921_DIR, force=True)
        assert result.metadata["observations_updated"] >= 1
        stats_after = service.get_database_stats()

        assert stats_after["scans"] == stats_before["scans"]
        assert stats_after["scan_points"] == stats_before["scan_points"]
        assert stats_after["spectra"] == stats_before["spectra"]
        assert stats_after["context_images"] == stats_before["context_images"]

    def test_validation_passes_after_force(self, tmp_path):
        """Database still valid after force re-ingestion."""
        pds_db = tmp_path / "phase_pds_val_force.db"
        service = PDSIngestionService(
            pds_db_path=pds_db,
            loupe_db_path=LOUPE_DB if LOUPE_DB.exists() else None,
        )

        service.ingest_sol(SOL_921_DIR)
        service.ingest_sol(SOL_921_DIR, force=True)
        report = service.validate_database()
        assert report["valid"] is True, f"Post-force validation failed: {report['issues']}"
