"""Integration tests for PDS parsing pipeline (Sol 921).

Validates the full parsing stack against spec s14:
- Observation grouping (6 groups: 5 non-zpz + 1 zpz-affected)
- SCLK cross-reference (5 Loupe-matched observations)
- Point counts (1, 100, 100, 1296, 1)
- Scan type classification (calibration, detail, survey)
- zpz filtering (observation 665 at SCLK 748735903)
- Wavelength calibration (PDS vs Loupe polynomial < 0.001 nm)
"""

from pathlib import Path

import numpy as np
import pytest

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.pds_parsers import (
    PDSCalibrationParser,
    PDSCrossRefParser,
    PDSLabelParser,
    PDSObservationGroup,
    PDSObservationGrouper,
    PDSPhotodiodeParser,
    PDSRMOParser,
    PDSSpectralParser,
)

from tests.integration.conftest import (
    SOL_921_DIR,
    requires_sol921_data,
)

pytestmark = requires_sol921_data

# Spec s14: SCLK cross-reference (5 Loupe-matched observations)
SPEC_S14_OBSERVATIONS = [
    # (pds_sclk, scan_type, n_spectra)
    (748731011, "calibration", 1),
    (748731413, "detail", 100),
    (748732975, "detail", 100),
    (748735042, "survey", 1296),
    (748736149, "calibration", 1),
]

# zpz observation (spec s14: "Unmatched Observation")
ZPZ_SCLK = 748735903

# Expected SCLK values for the 5 non-zpz observations
EXPECTED_NON_ZPZ_SCLKS = {sclk for sclk, _, _ in SPEC_S14_OBSERVATIONS}

# Wavelength tolerance (spec s14: < 0.001 nm except channel 500)
WAVELENGTH_TOLERANCE_NM = 0.001
CHANNEL_500_MAX_DEVIATION_NM = 0.5  # spec says 0.393 nm at channel 500


@pytest.fixture(scope="module")
def observation_groups() -> list[PDSObservationGroup]:
    """Parse and group all Sol 921 observations."""
    assert SOL_921_DIR.exists(), f"Sol 921 data not found at {SOL_921_DIR}"
    grouper = PDSObservationGrouper()
    label_parser = PDSLabelParser()
    return grouper.group_sol_directory(SOL_921_DIR, label_parser=label_parser)


@pytest.fixture(scope="module")
def non_zpz_groups(observation_groups) -> list[PDSObservationGroup]:
    """Return only the 5 non-zpz-affected observations (have RRS or RCS)."""
    return [g for g in observation_groups if "rrs" in g.products or "rcs" in g.products]


@pytest.fixture(scope="module")
def zpz_group(observation_groups) -> PDSObservationGroup:
    """Return the zpz-affected observation at SCLK 748735903."""
    matches = [g for g in observation_groups if g.sclk == ZPZ_SCLK]
    assert len(matches) == 1, f"Expected 1 zpz group, got {len(matches)}"
    return matches[0]


class TestSol921ObservationGrouping:
    """Verify observation grouping matches spec s14."""

    def test_total_groups(self, observation_groups):
        """Sol 921 should produce 6 observation groups (5 full + 1 zpz-affected)."""
        assert len(observation_groups) == 6

    def test_all_groups_are_sol_921(self, observation_groups):
        """All observation groups should be from Sol 921."""
        for group in observation_groups:
            assert group.sol == 921, f"Group {group.observation_key} has sol={group.sol}"

    def test_non_zpz_count(self, non_zpz_groups):
        """5 observations should have spectral data (RRS or RCS)."""
        assert len(non_zpz_groups) == 5

    def test_sclk_values_match_spec(self, non_zpz_groups):
        """Non-zpz SCLK values should match spec s14 cross-reference table."""
        actual_sclks = {g.sclk for g in non_zpz_groups}
        assert actual_sclks == EXPECTED_NON_ZPZ_SCLKS, (
            f"SCLK mismatch: expected {EXPECTED_NON_ZPZ_SCLKS}, got {actual_sclks}"
        )

    def test_scan_type_classification(self, non_zpz_groups):
        """Each observation should be classified per spec s14."""
        sclk_to_type = {g.sclk: g.scan_type for g in non_zpz_groups}
        for sclk, expected_type, _ in SPEC_S14_OBSERVATIONS:
            actual = sclk_to_type[sclk]
            assert actual == expected_type, (
                f"SCLK {sclk}: expected {expected_type}, got {actual}"
            )

    def test_point_counts_from_labels(self, non_zpz_groups):
        """Point counts should match spec s14: (1, 100, 100, 1296, 1).

        Uses XML label n_spectra (via label_parser in group_sol_directory).
        For each observation, parse the RRS/RCS XML label to get n_spectra.
        """
        label_parser = PDSLabelParser()
        expected_counts = {sclk: count for sclk, _, count in SPEC_S14_OBSERVATIONS}

        for group in non_zpz_groups:
            spectral_type = "rrs" if "rrs" in group.products else "rcs"
            product = group.products[spectral_type]
            xml_path = SOL_921_DIR / product.xml_filename
            metadata = label_parser.parse_label(xml_path)

            expected = expected_counts[group.sclk]
            assert metadata.n_spectra == expected, (
                f"SCLK {group.sclk} ({group.scan_type}): "
                f"expected n_spectra={expected}, got {metadata.n_spectra}"
            )

    def test_observation_keys_sorted(self, observation_groups):
        """Groups should be sorted by observation_key."""
        keys = [g.observation_key for g in observation_groups]
        assert keys == sorted(keys)


class TestSol921ZpzFiltering:
    """Verify zpz filtering per spec s14 unmatched observation."""

    def test_zpz_sclk(self, zpz_group):
        """zpz observation should be at SCLK 748735903."""
        assert zpz_group.sclk == ZPZ_SCLK

    def test_zpz_has_filtered_products(self, zpz_group):
        """zpz observation should have filtered products."""
        assert len(zpz_group.filtered_zpz) > 0, "zpz group should have filtered products"

    def test_zpz_filtered_count(self, zpz_group):
        """zpz observation should have 8 products filtered (rrs,rm1-6,rmo)."""
        assert len(zpz_group.filtered_zpz) == 8, (
            f"Expected 8 zpz-filtered products, got {len(zpz_group.filtered_zpz)}"
        )

    def test_zpz_clean_products(self, zpz_group):
        """zpz observation should retain only rli and rls (non-zpz filenames)."""
        clean_types = set(zpz_group.products.keys())
        assert clean_types == {"rli", "rls"}, (
            f"Expected {{'rli', 'rls'}} clean products, got {clean_types}"
        )

    def test_zpz_no_spectral_data(self, zpz_group):
        """zpz observation should have no RRS or RCS (all zpz-filtered)."""
        assert "rrs" not in zpz_group.products
        assert "rcs" not in zpz_group.products

    def test_zpz_filtered_types(self, zpz_group):
        """Filtered zpz products should include rrs, rm1-6, and rmo."""
        filtered_types = {str(p.product_type) for p in zpz_group.filtered_zpz}
        expected_filtered = {"rrs", "rm1", "rm2", "rm3", "rm4", "rm5", "rm6", "rmo"}
        assert filtered_types == expected_filtered, (
            f"Expected filtered types {expected_filtered}, got {filtered_types}"
        )


class TestSol921SpectralParsing:
    """Verify spectral data parsing for non-zpz observations."""

    def test_parse_all_spectral_products(self, non_zpz_groups):
        """All 5 non-zpz RRS/RCS files should parse without error."""
        parser = PDSSpectralParser()
        for group in non_zpz_groups:
            spectral_type = "rrs" if "rrs" in group.products else "rcs"
            product = group.products[spectral_type]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)
            assert result.product is not None
            assert len(result.spectra) > 0, (
                f"SCLK {group.sclk}: no spectral regions parsed"
            )

    def test_spectra_channel_count(self, non_zpz_groups):
        """All spectral arrays should have 2148 channels."""
        parser = PDSSpectralParser()
        for group in non_zpz_groups:
            spectral_type = "rrs" if "rrs" in group.products else "rcs"
            product = group.products[spectral_type]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)

            for region_name, spectra_array in result.spectra.items():
                assert spectra_array.shape[1] == 2148, (
                    f"SCLK {group.sclk}, region {region_name}: "
                    f"expected 2148 channels, got {spectra_array.shape[1]}"
                )

    def test_spectra_row_counts(self, non_zpz_groups):
        """Spectral row counts should match spec s14 point counts."""
        parser = PDSSpectralParser()
        expected_counts = {sclk: count for sclk, _, count in SPEC_S14_OBSERVATIONS}

        for group in non_zpz_groups:
            spectral_type = "rrs" if "rrs" in group.products else "rcs"
            product = group.products[spectral_type]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)

            # Check first region (R1) — all regions have same row count
            first_region = next(iter(result.spectra.values()))
            expected = expected_counts[group.sclk]
            assert first_region.shape[0] == expected, (
                f"SCLK {group.sclk} ({group.scan_type}): "
                f"expected {expected} spectra rows, got {first_region.shape[0]}"
            )

    def test_wavelength_calibration_vs_loupe(self, non_zpz_groups):
        """PDS wavelength array should match Loupe polynomial within 0.001 nm.

        Spec s14: "PDS wavelength array matches Loupe V5.1.5a polynomial
        within <0.001 nm, except 0.393 nm at channel 500."
        """
        loupe_wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels=2148)
        parser = PDSSpectralParser()

        # Use the first non-calibration RRS file (detail or survey)
        rrs_groups = [g for g in non_zpz_groups if "rrs" in g.products]
        assert len(rrs_groups) > 0, "Need at least one RRS observation"

        group = rrs_groups[0]
        product = group.products["rrs"]
        csv_path = SOL_921_DIR / product.csv_filename
        result = parser.parse(csv_path)

        pds_wavelength = result.product.wavelengths
        assert len(pds_wavelength) == 2148

        # Compare channel-by-channel
        diff = np.abs(pds_wavelength - loupe_wavelength)

        # Exclude channel 500 (Raman/Fluorescence boundary — known discontinuity)
        mask_no_500 = np.ones(2148, dtype=bool)
        mask_no_500[500] = False

        max_diff_excl_500 = diff[mask_no_500].max()
        assert max_diff_excl_500 < WAVELENGTH_TOLERANCE_NM, (
            f"Max wavelength deviation (excl ch500) = {max_diff_excl_500:.6f} nm, "
            f"expected < {WAVELENGTH_TOLERANCE_NM} nm"
        )

        # Channel 500: known 0.393 nm discontinuity
        ch500_diff = diff[500]
        assert ch500_diff < CHANNEL_500_MAX_DEVIATION_NM, (
            f"Channel 500 deviation = {ch500_diff:.6f} nm, "
            f"expected < {CHANNEL_500_MAX_DEVIATION_NM} nm (known: ~0.393 nm)"
        )

    def test_all_rrs_files_same_wavelength(self, non_zpz_groups):
        """All RRS files should use identical wavelength arrays (spec s14)."""
        parser = PDSSpectralParser()
        rrs_groups = [g for g in non_zpz_groups if "rrs" in g.products]

        wavelength_arrays = []
        for group in rrs_groups:
            product = group.products["rrs"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)
            wavelength_arrays.append(result.product.wavelengths)

        for i in range(1, len(wavelength_arrays)):
            assert np.array_equal(wavelength_arrays[0], wavelength_arrays[i]), (
                f"RRS wavelength mismatch between observation 0 and {i}"
            )


class TestSol921PositionParsing:
    """Verify position/motor data parsing matches spec s14 point counts."""

    def test_parse_all_rmo_products(self, non_zpz_groups):
        """All non-zpz observations with RMO should parse without error."""
        parser = PDSRMOParser()
        rmo_groups = [g for g in non_zpz_groups if "rmo" in g.products]
        assert len(rmo_groups) > 0, "Should have RMO products in non-zpz observations"

        for group in rmo_groups:
            product = group.products["rmo"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)
            assert result is not None

    def test_rmo_position_counts(self, non_zpz_groups):
        """RMO unique position counts should match spec s14 point counts."""
        parser = PDSRMOParser()
        expected_counts = {sclk: count for sclk, _, count in SPEC_S14_OBSERVATIONS}

        rmo_groups = [g for g in non_zpz_groups if "rmo" in g.products]

        for group in rmo_groups:
            product = group.products["rmo"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)

            expected = expected_counts[group.sclk]
            actual = len(result.positions)
            assert actual == expected, (
                f"SCLK {group.sclk} ({group.scan_type}): "
                f"expected {expected} positions, got {actual}"
            )

    def test_rmo_wavelength_bands(self, non_zpz_groups):
        """All RMO products should have 6 wavelength bands."""
        parser = PDSRMOParser()
        rmo_groups = [g for g in non_zpz_groups if "rmo" in g.products]

        for group in rmo_groups:
            product = group.products["rmo"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)
            assert len(result.wavelength_regions) == 6, (
                f"SCLK {group.sclk}: expected 6 wavelength bands, "
                f"got {len(result.wavelength_regions)}"
            )


class TestSol921AuxiliaryParsing:
    """Verify auxiliary product parsers (RLI, RCC, RLS) against Sol 921."""

    def test_parse_rli_products(self, non_zpz_groups):
        """All non-zpz RLI products should parse with correct shot counts."""
        parser = PDSPhotodiodeParser()
        expected_counts = {sclk: count for sclk, _, count in SPEC_S14_OBSERVATIONS}

        rli_groups = [g for g in non_zpz_groups if "rli" in g.products]
        assert len(rli_groups) > 0

        for group in rli_groups:
            product = group.products["rli"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)

            expected = expected_counts[group.sclk]
            assert len(result.intensities) == expected, (
                f"SCLK {group.sclk}: expected {expected} photodiode readings, "
                f"got {len(result.intensities)}"
            )

    def test_parse_rcc_calibration(self, non_zpz_groups):
        """Calibration observations should have parseable RCC products."""
        parser = PDSCalibrationParser()
        cal_groups = [
            g for g in non_zpz_groups
            if g.scan_type == "calibration" and "rcc" in g.products
        ]

        for group in cal_groups:
            product = group.products["rcc"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)
            assert len(result.records) > 0, (
                f"SCLK {group.sclk}: calibration records should be non-empty"
            )

            # Sol 921 calibration should include a record for sol 921
            sol_921_record = result.record_for_sol(921)
            assert sol_921_record is not None, "RCC should have a record for sol 921"

    def test_parse_rls_cross_ref(self, non_zpz_groups):
        """All non-zpz RLS products should parse with correct record counts."""
        parser = PDSCrossRefParser()
        expected_counts = {sclk: count for sclk, _, count in SPEC_S14_OBSERVATIONS}

        rls_groups = [g for g in non_zpz_groups if "rls" in g.products]
        assert len(rls_groups) > 0

        for group in rls_groups:
            product = group.products["rls"]
            csv_path = SOL_921_DIR / product.csv_filename
            result = parser.parse(csv_path)

            # RLS record count — survey has 2592 (raw, not de-duped)
            # Other observations match point count
            if group.scan_type == "survey":
                # Survey RLS preserves all rows (2 × 1296 = 2592)
                assert len(result.records) == 2592, (
                    f"Survey RLS: expected 2592 records, got {len(result.records)}"
                )
            else:
                expected = expected_counts[group.sclk]
                assert len(result.records) == expected, (
                    f"SCLK {group.sclk}: expected {expected} RLS records, "
                    f"got {len(result.records)}"
                )
