"""
Unit tests for PDS4 data models (Phase 7, step 7.1).

Tests Pydantic validation rules, field constraints, and computed properties
for all models in models/pds.py: PDSProductType, PDSProductId,
PDSObservationMetadata, PDSSpectralProduct, PDSWavelengthRegion,
PDSPositionRecord, PDSPositionProduct, PDSPhotodiodeProduct,
PDSCalibrationRecord/Product, PDSCrossRefRecord/Product.
"""

import pytest
from pydantic import ValidationError

from sherloc_pipeline.models.pds import (
    CORE_PRODUCT_TYPES,
    PDS_EXPECTED_CHANNELS,
    PDS_MISSION_SCLK_MIN,
    PDSCalibrationProduct,
    PDSCalibrationRecord,
    PDSCrossRefProduct,
    PDSCrossRefRecord,
    PDSObservationMetadata,
    PDSPhotodiodeProduct,
    PDSPositionProduct,
    PDSPositionRecord,
    PDSProductId,
    PDSProductType,
    PDSSpectralProduct,
    PDSWavelengthRegion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A valid PDS filename for reuse across tests.
VALID_FILENAME = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
VALID_XML_FILENAME = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml"


def _make_product_id(**overrides):
    """Build a valid PDSProductId with optional field overrides."""
    defaults = dict(
        filename=VALID_FILENAME,
        sol=921,
        sclk=748731413,
        obs_id="045",
        product_type=PDSProductType.RRS,
        site_drive="0450000",
        sequence_code="srlc11374",
        version=1,
        middle="w104cgnj",
    )
    defaults.update(overrides)
    return PDSProductId(**defaults)


def _make_metadata(**overrides):
    """Build a minimal valid PDSObservationMetadata with optional overrides."""
    defaults = dict(
        logical_identifier="urn:nasa:pds:mars2020_sherloc:data_processed:test",
        version_id="1.0",
        sol_number=921,
        spacecraft_clock_start="748731411.515",
    )
    defaults.update(overrides)
    return PDSObservationMetadata(**defaults)


# ---------------------------------------------------------------------------
# PDSProductType enum
# ---------------------------------------------------------------------------


class TestPDSProductType:
    """Tests for PDSProductType enum."""

    def test_all_12_product_types(self):
        expected = {"rrs", "rcs", "rmo", "rli", "rcc", "rls",
                    "rm1", "rm2", "rm3", "rm4", "rm5", "rm6"}
        assert {pt.value for pt in PDSProductType} == expected

    def test_core_product_types_set(self):
        assert len(CORE_PRODUCT_TYPES) == 6
        core_values = {pt.value for pt in CORE_PRODUCT_TYPES}
        assert core_values == {"rrs", "rcs", "rmo", "rli", "rcc", "rls"}

    def test_str_enum(self):
        assert PDSProductType.RRS == "rrs"
        assert PDSProductType.RCS.value == "rcs"


# ---------------------------------------------------------------------------
# PDSProductId — from_filename
# ---------------------------------------------------------------------------


class TestPDSProductIdFromFilename:
    """Tests for PDSProductId.from_filename()."""

    def test_valid_rrs_filename(self):
        pid = PDSProductId.from_filename(VALID_FILENAME)
        assert pid.sol == 921
        assert pid.sclk == 748731413
        assert pid.obs_id == "045"
        assert pid.product_type == "rrs"
        assert pid.site_drive == "0450000"
        assert pid.sequence_code == "srlc11374"
        assert pid.version == 1
        assert pid.middle == "w104cgnj"

    def test_valid_rcs_filename(self):
        fn = "ss__0921_0748731011_045rcs__0450000srlc10000_104___j01.csv"
        pid = PDSProductId.from_filename(fn)
        assert pid.product_type == "rcs"
        assert pid.sequence_code == "srlc10000"

    def test_valid_rmo_filename(self):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        pid = PDSProductId.from_filename(fn)
        assert pid.product_type == "rmo"

    def test_valid_rm1_to_rm6(self):
        for i in range(1, 7):
            fn = f"ss__0921_0748731413_045rm{i}__0450000srlc11374w0__cgnj01.csv"
            pid = PDSProductId.from_filename(fn)
            assert pid.product_type == f"rm{i}"

    def test_version_02(self):
        fn = "ss__0921_0748735042_045rrs__0450000srlc11420w104cgnj02.csv"
        pid = PDSProductId.from_filename(fn)
        assert pid.version == 2

    def test_xml_extension(self):
        pid = PDSProductId.from_filename(VALID_XML_FILENAME)
        assert pid.filename.endswith(".xml")
        assert pid.sol == 921

    def test_strips_directory_path(self):
        full = "./pds/sol_0921/data_processed/" + VALID_FILENAME
        pid = PDSProductId.from_filename(full)
        assert pid.filename == VALID_FILENAME
        assert pid.sol == 921

    def test_zpz_middle_section(self):
        fn = "ss__0921_0748735903_045rrs__0450000srlc11374b108zpzj01.csv"
        pid = PDSProductId.from_filename(fn)
        assert "zpz" in pid.middle

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="does not match PDS format"):
            PDSProductId.from_filename("not_a_pds_file.csv")

    def test_empty_filename_raises(self):
        with pytest.raises(ValueError, match="does not match PDS format"):
            PDSProductId.from_filename("")

    def test_wrong_prefix_raises(self):
        # Missing "ss__" prefix
        fn = "xx__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
        with pytest.raises(ValueError, match="does not match PDS format"):
            PDSProductId.from_filename(fn)

    def test_wrong_extension_raises(self):
        fn = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.txt"
        with pytest.raises(ValueError, match="does not match PDS format"):
            PDSProductId.from_filename(fn)

    def test_invalid_product_type_raises(self):
        fn = "ss__0921_0748731413_045zzz__0450000srlc11374w104cgnj01.csv"
        with pytest.raises(ValueError):
            PDSProductId.from_filename(fn)


# ---------------------------------------------------------------------------
# PDSProductId — properties
# ---------------------------------------------------------------------------


class TestPDSProductIdProperties:
    """Tests for PDSProductId computed properties."""

    def test_csv_filename_from_csv(self):
        pid = _make_product_id(filename=VALID_FILENAME)
        assert pid.csv_filename == VALID_FILENAME

    def test_csv_filename_from_xml(self):
        pid = _make_product_id(filename=VALID_XML_FILENAME)
        assert pid.csv_filename == VALID_FILENAME

    def test_xml_filename_from_csv(self):
        pid = _make_product_id(filename=VALID_FILENAME)
        assert pid.xml_filename == VALID_XML_FILENAME

    def test_xml_filename_from_xml(self):
        pid = _make_product_id(filename=VALID_XML_FILENAME)
        assert pid.xml_filename == VALID_XML_FILENAME

    def test_observation_key_format(self):
        pid = _make_product_id(sol=921, sclk=748731413, obs_id="045")
        assert pid.observation_key == "0921_0748731413_045"

    def test_observation_key_zero_padded(self):
        pid = _make_product_id(sol=1, sclk=PDS_MISSION_SCLK_MIN, obs_id="001")
        assert pid.observation_key == "0001_0666000000_001"

    def test_is_spectral_rrs(self):
        assert _make_product_id(product_type=PDSProductType.RRS).is_spectral

    def test_is_spectral_rcs(self):
        assert _make_product_id(product_type=PDSProductType.RCS).is_spectral

    def test_is_spectral_rmo_false(self):
        assert not _make_product_id(product_type=PDSProductType.RMO).is_spectral

    def test_is_calibration_rcs(self):
        assert _make_product_id(product_type=PDSProductType.RCS).is_calibration

    def test_is_calibration_rcc(self):
        assert _make_product_id(product_type=PDSProductType.RCC).is_calibration

    def test_is_calibration_rrs_false(self):
        assert not _make_product_id(product_type=PDSProductType.RRS).is_calibration

    def test_is_core_rrs(self):
        assert _make_product_id(product_type=PDSProductType.RRS).is_core

    def test_is_core_rmo(self):
        assert _make_product_id(product_type=PDSProductType.RMO).is_core

    def test_is_core_rm1_false(self):
        assert not _make_product_id(product_type=PDSProductType.RM1).is_core

    def test_is_core_rm6_false(self):
        assert not _make_product_id(product_type=PDSProductType.RM6).is_core


# ---------------------------------------------------------------------------
# PDSProductId — field validation
# ---------------------------------------------------------------------------


class TestPDSProductIdValidation:
    """Tests for PDSProductId field constraint validation."""

    def test_sclk_at_mission_min(self):
        pid = _make_product_id(sclk=PDS_MISSION_SCLK_MIN)
        assert pid.sclk == PDS_MISSION_SCLK_MIN

    def test_sclk_below_mission_min_rejects(self):
        with pytest.raises(ValidationError, match="greater than or equal"):
            _make_product_id(sclk=PDS_MISSION_SCLK_MIN - 1)

    def test_sol_zero_accepted(self):
        pid = _make_product_id(sol=0)
        assert pid.sol == 0

    def test_sol_negative_rejects(self):
        with pytest.raises(ValidationError, match="greater than or equal"):
            _make_product_id(sol=-1)

    def test_version_one_accepted(self):
        pid = _make_product_id(version=1)
        assert pid.version == 1

    def test_version_zero_rejects(self):
        with pytest.raises(ValidationError, match="greater than or equal"):
            _make_product_id(version=0)

    def test_obs_id_3_chars(self):
        pid = _make_product_id(obs_id="045")
        assert pid.obs_id == "045"

    def test_obs_id_too_short_rejects(self):
        with pytest.raises(ValidationError):
            _make_product_id(obs_id="04")

    def test_obs_id_too_long_rejects(self):
        with pytest.raises(ValidationError):
            _make_product_id(obs_id="0456")

    def test_site_drive_7_digits(self):
        pid = _make_product_id(site_drive="0450000")
        assert pid.site_drive == "0450000"

    def test_site_drive_too_short_rejects(self):
        with pytest.raises(ValidationError):
            _make_product_id(site_drive="045000")

    def test_site_drive_too_long_rejects(self):
        with pytest.raises(ValidationError):
            _make_product_id(site_drive="04500001")

    def test_sequence_code_valid(self):
        pid = _make_product_id(sequence_code="srlc11374")
        assert pid.sequence_code == "srlc11374"

    def test_sequence_code_invalid_prefix_rejects(self):
        with pytest.raises(ValidationError, match="pattern"):
            _make_product_id(sequence_code="abcd12345")

    def test_sequence_code_wrong_length_rejects(self):
        with pytest.raises(ValidationError, match="pattern"):
            _make_product_id(sequence_code="srlc1234")


# ---------------------------------------------------------------------------
# PDSObservationMetadata — SCLK validators
# ---------------------------------------------------------------------------


class TestPDSObservationMetadataSCLK:
    """Tests for SCLK field validators on PDSObservationMetadata."""

    def test_valid_sclk_start_accepted(self):
        meta = _make_metadata(spacecraft_clock_start="748731411.515")
        assert meta.spacecraft_clock_start == "748731411.515"

    def test_sclk_start_integer_string_accepted(self):
        meta = _make_metadata(spacecraft_clock_start="748731411")
        assert meta.spacecraft_clock_start == "748731411"

    def test_sclk_start_at_mission_min(self):
        meta = _make_metadata(
            spacecraft_clock_start=str(PDS_MISSION_SCLK_MIN)
        )
        assert meta.spacecraft_clock_start == str(PDS_MISSION_SCLK_MIN)

    def test_sclk_start_below_mission_min_rejects(self):
        with pytest.raises(ValidationError, match="below mission"):
            _make_metadata(
                spacecraft_clock_start=str(PDS_MISSION_SCLK_MIN - 1)
            )

    def test_sclk_start_non_numeric_rejects(self):
        with pytest.raises(ValidationError, match="numeric string"):
            _make_metadata(spacecraft_clock_start="not_a_number")

    def test_sclk_stop_valid(self):
        meta = _make_metadata(spacecraft_clock_stop="748731500.000")
        assert meta.spacecraft_clock_stop == "748731500.000"

    def test_sclk_stop_none_accepted(self):
        meta = _make_metadata(spacecraft_clock_stop=None)
        assert meta.spacecraft_clock_stop is None

    def test_sclk_stop_below_mission_min_rejects(self):
        with pytest.raises(ValidationError, match="below mission"):
            _make_metadata(spacecraft_clock_stop="100")

    def test_sclk_stop_non_numeric_rejects(self):
        with pytest.raises(ValidationError, match="numeric string"):
            _make_metadata(spacecraft_clock_stop="abc")


# ---------------------------------------------------------------------------
# PDSObservationMetadata — properties
# ---------------------------------------------------------------------------


class TestPDSObservationMetadataProperties:
    """Tests for PDSObservationMetadata computed properties."""

    def test_sclk_start_int_truncates(self):
        meta = _make_metadata(spacecraft_clock_start="748731411.515")
        assert meta.sclk_start_int == 748731411

    def test_sclk_start_int_no_fraction(self):
        meta = _make_metadata(spacecraft_clock_start="748731411")
        assert meta.sclk_start_int == 748731411

    def test_sclk_stop_int(self):
        meta = _make_metadata(spacecraft_clock_stop="748731500.999")
        assert meta.sclk_stop_int == 748731500

    def test_sclk_stop_int_none(self):
        meta = _make_metadata(spacecraft_clock_stop=None)
        assert meta.sclk_stop_int is None

    def test_site_drive_str_format(self):
        meta = _make_metadata(site=45, drive=0)
        assert meta.site_drive_str == "0450000"

    def test_site_drive_str_large_values(self):
        meta = _make_metadata(site=123, drive=4567)
        assert meta.site_drive_str == "1234567"

    def test_site_drive_str_none_without_site(self):
        meta = _make_metadata(site=None, drive=0)
        assert meta.site_drive_str is None

    def test_site_drive_str_none_without_drive(self):
        meta = _make_metadata(site=45, drive=None)
        assert meta.site_drive_str is None

    def test_version_tuple_simple(self):
        meta = _make_metadata(version_id="1.0")
        assert meta.version_tuple == (1, 0)

    def test_version_tuple_multi_digit(self):
        meta = _make_metadata(version_id="1.10")
        assert meta.version_tuple == (1, 10)

    def test_version_tuple_comparison_correct(self):
        """Numeric tuple comparison avoids '1.10' < '1.2' string sort bug."""
        v1 = _make_metadata(version_id="1.10").version_tuple
        v2 = _make_metadata(version_id="1.2").version_tuple
        assert v1 > v2  # (1, 10) > (1, 2)

    def test_version_tuple_three_parts(self):
        meta = _make_metadata(version_id="2.5.3")
        assert meta.version_tuple == (2, 5, 3)

    def test_earth_date_from_iso(self):
        meta = _make_metadata(start_date_time="2023-09-23T09:09:06.711Z")
        assert meta.earth_date == "2023-09-23"

    def test_earth_date_none_without_start_time(self):
        meta = _make_metadata(start_date_time=None)
        assert meta.earth_date is None


# ---------------------------------------------------------------------------
# PDSObservationMetadata — to_pds4_metadata_dict
# ---------------------------------------------------------------------------


class TestPDSObservationMetadataDict:
    """Tests for to_pds4_metadata_dict()."""

    def test_minimal_dict(self):
        meta = _make_metadata()
        d = meta.to_pds4_metadata_dict()
        assert d["lidvid"] == meta.logical_identifier
        assert d["version"] == "1.0"

    def test_includes_sequence_id(self):
        meta = _make_metadata(sequence_id="srlc11374")
        d = meta.to_pds4_metadata_dict()
        assert d["sequence_id"] == "srlc11374"

    def test_includes_site_drive(self):
        meta = _make_metadata(site=45, drive=0)
        d = meta.to_pds4_metadata_dict()
        assert d["site_drive"] == "0450000"
        assert d["rover_motion_counter"] == {"SITE": 45, "DRIVE": 0}

    def test_includes_utc_times(self):
        meta = _make_metadata(
            start_date_time="2023-09-23T09:09:06Z",
            stop_date_time="2023-09-23T09:15:00Z",
        )
        d = meta.to_pds4_metadata_dict()
        assert d["start_utc"] == "2023-09-23T09:09:06Z"
        assert d["stop_utc"] == "2023-09-23T09:15:00Z"

    def test_includes_rsm_geometry(self):
        meta = _make_metadata(rsm_azimuth_rad=1.234, rsm_elevation_rad=-0.5)
        d = meta.to_pds4_metadata_dict()
        assert d["rsm_azimuth_rad"] == 1.234
        assert d["rsm_elevation_rad"] == -0.5

    def test_omits_none_optional_fields(self):
        meta = _make_metadata()
        d = meta.to_pds4_metadata_dict()
        assert "sequence_id" not in d
        assert "site_drive" not in d
        assert "start_utc" not in d
        assert "stop_utc" not in d
        assert "rsm_azimuth_rad" not in d

    def test_includes_processing_suffix(self):
        pid = _make_product_id()
        meta = _make_metadata(product_id=pid)
        d = meta.to_pds4_metadata_dict()
        assert d["processing_suffix"] == "w104cgnj"


# ---------------------------------------------------------------------------
# PDSObservationMetadata — field constraints
# ---------------------------------------------------------------------------


class TestPDSObservationMetadataConstraints:
    """Tests for PDSObservationMetadata field constraints."""

    def test_solar_longitude_0_accepted(self):
        meta = _make_metadata(solar_longitude=0.0)
        assert meta.solar_longitude == 0.0

    def test_solar_longitude_360_accepted(self):
        meta = _make_metadata(solar_longitude=360.0)
        assert meta.solar_longitude == 360.0

    def test_solar_longitude_over_360_rejects(self):
        with pytest.raises(ValidationError):
            _make_metadata(solar_longitude=361.0)

    def test_solar_longitude_negative_rejects(self):
        with pytest.raises(ValidationError):
            _make_metadata(solar_longitude=-0.1)

    def test_site_negative_rejects(self):
        with pytest.raises(ValidationError):
            _make_metadata(site=-1)

    def test_drive_negative_rejects(self):
        with pytest.raises(ValidationError):
            _make_metadata(drive=-1)

    def test_extra_fields_forbid(self):
        with pytest.raises(ValidationError, match="extra"):
            _make_metadata(unknown_field="oops")

    def test_sol_negative_rejects(self):
        with pytest.raises(ValidationError):
            _make_metadata(sol_number=-1)


# ---------------------------------------------------------------------------
# PDSSpectralProduct
# ---------------------------------------------------------------------------


class TestPDSSpectralProduct:
    """Tests for PDSSpectralProduct validation rules."""

    def _make_spectral(self, **overrides):
        defaults = dict(
            product_id=_make_product_id(),
            n_spectra=100,
            n_channels=PDS_EXPECTED_CHANNELS,
            wavelengths=[250.0 + i * 0.05 for i in range(PDS_EXPECTED_CHANNELS)],
        )
        defaults.update(overrides)
        return PDSSpectralProduct(**defaults)

    def test_valid_construction(self):
        sp = self._make_spectral()
        assert sp.n_channels == 2148
        assert len(sp.wavelengths) == 2148
        assert sp.regions_present == ["R1", "R2", "R3"]

    def test_wrong_channel_count_rejects(self):
        with pytest.raises(ValidationError, match="2148"):
            self._make_spectral(n_channels=1024)

    def test_channel_count_zero_rejects(self):
        with pytest.raises(ValidationError):
            self._make_spectral(n_channels=0)

    def test_invalid_region_name_rejects(self):
        with pytest.raises(ValidationError, match="Invalid region"):
            self._make_spectral(regions_present=["R1", "R4"])

    def test_empty_region_list_accepted(self):
        sp = self._make_spectral(regions_present=[])
        assert sp.regions_present == []

    def test_single_region_accepted(self):
        sp = self._make_spectral(regions_present=["R1"])
        assert sp.regions_present == ["R1"]

    def test_empty_wavelengths_rejects(self):
        with pytest.raises(ValidationError):
            self._make_spectral(wavelengths=[])

    def test_n_spectra_zero_rejects(self):
        with pytest.raises(ValidationError):
            self._make_spectral(n_spectra=0)


# ---------------------------------------------------------------------------
# PDSWavelengthRegion
# ---------------------------------------------------------------------------


class TestPDSWavelengthRegion:
    """Tests for PDSWavelengthRegion model validator."""

    def test_valid_region(self):
        r = PDSWavelengthRegion(
            column_index=0, wavelength_start=250.9, wavelength_stop=260.0
        )
        assert r.wavelength_start == 250.9
        assert r.wavelength_stop == 260.0

    def test_stop_equals_start_rejects(self):
        with pytest.raises(ValidationError, match="must be >"):
            PDSWavelengthRegion(
                column_index=0, wavelength_start=250.0, wavelength_stop=250.0
            )

    def test_stop_less_than_start_rejects(self):
        with pytest.raises(ValidationError, match="must be >"):
            PDSWavelengthRegion(
                column_index=0, wavelength_start=260.0, wavelength_stop=250.0
            )

    def test_small_valid_difference(self):
        r = PDSWavelengthRegion(
            column_index=0, wavelength_start=250.0, wavelength_stop=250.001
        )
        assert r.wavelength_stop > r.wavelength_start

    def test_column_index_0_accepted(self):
        r = PDSWavelengthRegion(
            column_index=0, wavelength_start=250.0, wavelength_stop=260.0
        )
        assert r.column_index == 0

    def test_column_index_5_accepted(self):
        r = PDSWavelengthRegion(
            column_index=5, wavelength_start=250.0, wavelength_stop=260.0
        )
        assert r.column_index == 5

    def test_column_index_6_rejects(self):
        with pytest.raises(ValidationError):
            PDSWavelengthRegion(
                column_index=6, wavelength_start=250.0, wavelength_stop=260.0
            )

    def test_negative_wavelength_rejects(self):
        with pytest.raises(ValidationError):
            PDSWavelengthRegion(
                column_index=0, wavelength_start=-1.0, wavelength_stop=260.0
            )

    def test_zero_wavelength_rejects(self):
        with pytest.raises(ValidationError):
            PDSWavelengthRegion(
                column_index=0, wavelength_start=0.0, wavelength_stop=260.0
            )


# ---------------------------------------------------------------------------
# PDSPositionRecord
# ---------------------------------------------------------------------------


class TestPDSPositionRecord:
    """Tests for PDSPositionRecord field constraints."""

    def test_valid_record(self):
        r = PDSPositionRecord(
            image_name="ACI_0921.IMG", position_index=0, x=512.5, y=768.3
        )
        assert r.position_index == 0
        assert r.x == 512.5

    def test_negative_position_index_rejects(self):
        with pytest.raises(ValidationError):
            PDSPositionRecord(
                image_name="ACI.IMG", position_index=-1, x=0.0, y=0.0
            )


# ---------------------------------------------------------------------------
# PDSPositionProduct
# ---------------------------------------------------------------------------


class TestPDSPositionProduct:
    """Tests for PDSPositionProduct."""

    def test_n_positions_property(self):
        pid = _make_product_id(product_type=PDSProductType.RMO)
        positions = [
            PDSPositionRecord(image_name="A", position_index=i, x=float(i), y=0.0)
            for i in range(5)
        ]
        pp = PDSPositionProduct(
            product_id=pid,
            positions=positions,
            wavelength_regions=[],
            band_intensities=[],
            image_names=["A"],
        )
        assert pp.n_positions == 5

    def test_empty_positions(self):
        pid = _make_product_id(product_type=PDSProductType.RMO)
        pp = PDSPositionProduct(product_id=pid)
        assert pp.n_positions == 0


# ---------------------------------------------------------------------------
# PDSPhotodiodeProduct
# ---------------------------------------------------------------------------


class TestPDSPhotodiodeProduct:
    """Tests for PDSPhotodiodeProduct."""

    def test_n_shots_property(self):
        pid = _make_product_id(product_type=PDSProductType.RLI)
        pp = PDSPhotodiodeProduct(
            product_id=pid,
            intensities=[1.0, 2.0, 3.0],
        )
        assert pp.n_shots == 3

    def test_empty_intensities_rejects(self):
        pid = _make_product_id(product_type=PDSProductType.RLI)
        with pytest.raises(ValidationError):
            PDSPhotodiodeProduct(product_id=pid, intensities=[])

    def test_sentinel_minus_one_accepted(self):
        pid = _make_product_id(product_type=PDSProductType.RLI)
        pp = PDSPhotodiodeProduct(product_id=pid, intensities=[-1.0])
        assert pp.intensities == [-1.0]


# ---------------------------------------------------------------------------
# PDSCalibrationRecord
# ---------------------------------------------------------------------------


class TestPDSCalibrationRecord:
    """Tests for PDSCalibrationRecord field constraints."""

    def test_valid_record(self):
        rec = PDSCalibrationRecord(
            sol=921, sclk="748731011_555",
            laser_peak_nm=258.3, laser_fwhm_nm=0.5,
            algan_peak_nm=276.548, algan_fwhm_nm=0.4,
        )
        assert rec.sol == 921
        assert rec.algan_peak_nm == 276.548

    def test_laser_peak_zero_accepted(self):
        """0.0 means laser peak was not fit."""
        rec = PDSCalibrationRecord(
            sol=100, sclk="700000000_000",
            laser_peak_nm=0.0, laser_fwhm_nm=0.0,
            algan_peak_nm=276.0, algan_fwhm_nm=0.4,
        )
        assert rec.laser_peak_nm == 0.0

    def test_negative_peak_rejects(self):
        with pytest.raises(ValidationError):
            PDSCalibrationRecord(
                sol=100, sclk="700000000_000",
                laser_peak_nm=-1.0, laser_fwhm_nm=0.0,
                algan_peak_nm=276.0, algan_fwhm_nm=0.4,
            )


# ---------------------------------------------------------------------------
# PDSCalibrationProduct — record_for_sol
# ---------------------------------------------------------------------------


class TestPDSCalibrationProductRecordForSol:
    """Tests for PDSCalibrationProduct.record_for_sol() lookup."""

    def _make_product(self, sols):
        pid = _make_product_id(product_type=PDSProductType.RCC)
        records = [
            PDSCalibrationRecord(
                sol=s, sclk=f"{700000000 + s}_000",
                laser_peak_nm=258.0, laser_fwhm_nm=0.5,
                algan_peak_nm=276.0, algan_fwhm_nm=0.4,
            )
            for s in sols
        ]
        return PDSCalibrationProduct(product_id=pid, records=records)

    def test_exact_match(self):
        prod = self._make_product([100, 500, 921])
        rec = prod.record_for_sol(500)
        assert rec is not None
        assert rec.sol == 500

    def test_closest_lower(self):
        prod = self._make_product([100, 500, 921])
        rec = prod.record_for_sol(600)
        assert rec is not None
        assert rec.sol == 500

    def test_highest_below(self):
        prod = self._make_product([100, 500, 921])
        rec = prod.record_for_sol(920)
        assert rec is not None
        assert rec.sol == 500

    def test_all_records_above_target(self):
        prod = self._make_product([100, 500, 921])
        rec = prod.record_for_sol(50)
        assert rec is None

    def test_exact_match_at_last(self):
        prod = self._make_product([100, 500, 921])
        rec = prod.record_for_sol(921)
        assert rec is not None
        assert rec.sol == 921

    def test_n_records_property(self):
        prod = self._make_product([100, 500, 921])
        assert prod.n_records == 3


# ---------------------------------------------------------------------------
# PDSCrossRefRecord
# ---------------------------------------------------------------------------


class TestPDSCrossRefRecord:
    """Tests for PDSCrossRefRecord field constraints."""

    def test_valid_record(self):
        rec = PDSCrossRefRecord(
            number=0, spec_name="rrs_file.csv",
            image_name="ACI.IMG", samp=512.5, line=768.3,
        )
        assert rec.number == 0

    def test_negative_number_rejects(self):
        with pytest.raises(ValidationError):
            PDSCrossRefRecord(
                number=-1, spec_name="rrs.csv",
                image_name="ACI.IMG", samp=0.0, line=0.0,
            )

    def test_empty_spec_name_rejects(self):
        with pytest.raises(ValidationError):
            PDSCrossRefRecord(
                number=0, spec_name="",
                image_name="ACI.IMG", samp=0.0, line=0.0,
            )


# ---------------------------------------------------------------------------
# PDSCrossRefProduct — dedup properties
# ---------------------------------------------------------------------------


class TestPDSCrossRefProductDedup:
    """Tests for PDSCrossRefProduct.image_names and spec_names dedup."""

    def _make_crossref(self, records_data):
        pid = _make_product_id(product_type=PDSProductType.RLS)
        records = [
            PDSCrossRefRecord(
                number=i,
                spec_name=r[0],
                image_name=r[1],
                samp=0.0,
                line=0.0,
            )
            for i, r in enumerate(records_data)
        ]
        return PDSCrossRefProduct(product_id=pid, records=records)

    def test_image_names_dedup(self):
        prod = self._make_crossref([
            ("rrs.csv", "ACI1.IMG"),
            ("rrs.csv", "ACI2.IMG"),
            ("rrs.csv", "ACI1.IMG"),  # duplicate
        ])
        assert prod.image_names == ["ACI1.IMG", "ACI2.IMG"]

    def test_image_names_preserves_order(self):
        prod = self._make_crossref([
            ("rrs.csv", "B.IMG"),
            ("rrs.csv", "A.IMG"),
            ("rrs.csv", "B.IMG"),
        ])
        assert prod.image_names == ["B.IMG", "A.IMG"]

    def test_spec_names_dedup(self):
        prod = self._make_crossref([
            ("rrs_v1.csv", "ACI.IMG"),
            ("rrs_v2.csv", "ACI.IMG"),
            ("rrs_v1.csv", "ACI.IMG"),  # duplicate
        ])
        assert prod.spec_names == ["rrs_v1.csv", "rrs_v2.csv"]

    def test_spec_names_preserves_order(self):
        prod = self._make_crossref([
            ("b.csv", "ACI.IMG"),
            ("a.csv", "ACI.IMG"),
        ])
        assert prod.spec_names == ["b.csv", "a.csv"]

    def test_n_records_property(self):
        prod = self._make_crossref([
            ("rrs.csv", "ACI.IMG"),
            ("rrs.csv", "ACI.IMG"),
        ])
        assert prod.n_records == 2

    def test_empty_records_rejects(self):
        pid = _make_product_id(product_type=PDSProductType.RLS)
        with pytest.raises(ValidationError):
            PDSCrossRefProduct(product_id=pid, records=[])
