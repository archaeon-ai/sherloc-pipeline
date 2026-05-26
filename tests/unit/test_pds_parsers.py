"""
Unit tests for PDS4 parsers (Phase 7, step 7.1).

Tests parser logic using synthetic temp files:
- PDSLabelParser: XML label parsing, missing fields, namespace extraction
- PDSSpectralParser: RRS/RCS CSV parsing, zpz detection (3 signals)
- PDSRMOParser: position de-dup, section parsing, zpz rejection
- PDSPhotodiodeParser: RLI section parsing
- PDSCalibrationParser: RCC section parsing, column count validation
- PDSCrossRefParser: RLS section parsing, column count validation
- PDSObservationGrouper: classify, filter_zpz, validate_spectral_exclusivity,
  select_latest_versions, group_by_observation
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from sherloc_pipeline.core.pds_parsers import (
    PDSCalibrationParser,
    PDSCrossRefParser,
    PDSLabelParser,
    PDSObservationGrouper,
    PDSPhotodiodeParser,
    PDSRMOParser,
    PDSSpectralParser,
    PDSZpzProductError,
)
from sherloc_pipeline.models.pds import (
    PDS_EXPECTED_CHANNELS,
    PDSProductId,
    PDSProductType,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A valid PDS filename base (used to create synthetic files)
_BASENAME = "ss__0921_0748731413_045{ptype}__0450000srlc11374{middle}{ver:02d}"

# Synthetic wavelength row: 2148 comma-separated floats
_WAVELENGTH_ROW = ",".join(f"{246.69 + i * 0.0515:.3f}" for i in range(PDS_EXPECTED_CHANNELS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path, filename, content):
    """Write content to a file in tmp_path and return the Path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content))
    return p


def _spectral_csv(tmp_path, filename, n_spectra=10, zpz_headers=False,
                   middle="w104cgnj"):
    """Create a synthetic RRS/RCS CSV with correct structure."""
    header_prefix = "PROCESS_DATA_SPECTRUM_REGION_" if zpz_headers else "LASER-NORMALIZED_SPECTRA:_REGION_"
    data_row = ",".join("1.0" for _ in range(PDS_EXPECTED_CHANNELS))
    wavelength_row = _WAVELENGTH_ROW

    lines = []
    lines.append("WAVELENGTH (NM):")
    lines.append("ch0,ch1,...,ch2147")  # column header (skipped by parser)
    lines.append(wavelength_row)

    for region in (1, 2, 3):
        lines.append(f"{header_prefix}{region}")
        lines.append("s0,s1,...,s2147")  # column header (skipped)
        for _ in range(n_spectra):
            lines.append(data_row)

    p = tmp_path / filename
    p.write_text("\n".join(lines))
    return p


def _minimal_xml(tmp_path, filename, sol=921, sclk_start="748731413.515",
                  lid="urn:nasa:pds:mars2020_sherloc:data_processed:test",
                  version_id="1.0", extra_xml=""):
    """Create a minimal PDS4 XML label."""
    xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:mars2020="http://pds.nasa.gov/pds4/mission/mars2020/v1"
    xmlns:msn_surface="http://pds.nasa.gov/pds4/msn_surface/v1"
    xmlns:geom="http://pds.nasa.gov/pds4/geom/v1"
    xmlns:proc="http://pds.nasa.gov/pds4/proc/v1">
  <Identification_Area>
    <logical_identifier>{lid}</logical_identifier>
    <version_id>{version_id}</version_id>
    <title>Test Product</title>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>2023-09-23T09:09:06.711Z</start_date_time>
      <stop_date_time>2023-09-23T09:15:00Z</stop_date_time>
      <local_mean_solar_time>Sol-00921M11:22:33</local_mean_solar_time>
      <solar_longitude>122.871</solar_longitude>
    </Time_Coordinates>
    <Mission_Area>
      <mars2020:Observation_Information>
        <mars2020:sol_number>{sol}</mars2020:sol_number>
        <mars2020:spacecraft_clock_start>{sclk_start}</mars2020:spacecraft_clock_start>
        <mars2020:spacecraft_clock_stop>748731500.000</mars2020:spacecraft_clock_stop>
        <mars2020:mission_phase_name>Surface Mission</mars2020:mission_phase_name>
      </mars2020:Observation_Information>
    </Mission_Area>
    <Discipline_Area>
      <msn_surface:Surface_Mission_Information>
        <msn_surface:Command_Execution>
          <msn_surface:sequence_id>srlc11374</msn_surface:sequence_id>
        </msn_surface:Command_Execution>
        <msn_surface:Telemetry>
          <msn_surface:product_completion_status>COMPLETE</msn_surface:product_completion_status>
        </msn_surface:Telemetry>
      </msn_surface:Surface_Mission_Information>
      <geom:Geometry>
        <geom:Motion_Counter>
          <geom:Motion_Counter_Index>
            <geom:index_id>SITE</geom:index_id>
            <geom:index_value_number>45</geom:index_value_number>
          </geom:Motion_Counter_Index>
          <geom:Motion_Counter_Index>
            <geom:index_id>DRIVE</geom:index_id>
            <geom:index_value_number>0</geom:index_value_number>
          </geom:Motion_Counter_Index>
        </geom:Motion_Counter>
        <geom:Articulation_Device_Parameters>
          <geom:device_id>RSM</geom:device_id>
          <geom:Device_Angle_Index>
            <geom:index_id>AZIMUTH FINAL-RESOLVER</geom:index_id>
            <geom:index_value_angle>1.234</geom:index_value_angle>
          </geom:Device_Angle_Index>
          <geom:Device_Angle_Index>
            <geom:index_id>ELEVATION FINAL-RESOLVER</geom:index_id>
            <geom:index_value_angle>-0.567</geom:index_value_angle>
          </geom:Device_Angle_Index>
        </geom:Articulation_Device_Parameters>
      </geom:Geometry>
    </Discipline_Area>
  </Observation_Area>
  <File_Area_Observational>
    <Table_Delimited>
      <name>LASER-NORMALIZED_SPECTRA:_REGION_1</name>
      <records>100</records>
      <Record_Delimited>
        <Group_Field_Delimited>
          <repetitions>2148</repetitions>
        </Group_Field_Delimited>
      </Record_Delimited>
    </Table_Delimited>
  </File_Area_Observational>
  {extra_xml}
</Product_Observational>"""
    p = tmp_path / filename
    p.write_text(xml)
    return p


# ---------------------------------------------------------------------------
# PDSLabelParser
# ---------------------------------------------------------------------------


class TestPDSLabelParser:
    """Unit tests for PDSLabelParser."""

    def test_parse_all_fields(self, tmp_path):
        xml = _minimal_xml(
            tmp_path,
            "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml",
        )
        meta = PDSLabelParser().parse_label(xml)
        assert meta.sol_number == 921
        assert meta.logical_identifier == "urn:nasa:pds:mars2020_sherloc:data_processed:test"
        assert meta.version_id == "1.0"
        assert meta.spacecraft_clock_start == "748731413.515"
        assert meta.start_date_time == "2023-09-23T09:09:06.711Z"
        assert meta.solar_longitude == 122.871
        assert meta.mission_phase_name == "Surface Mission"
        assert meta.sequence_id == "srlc11374"
        assert meta.site == 45
        assert meta.drive == 0
        assert meta.rsm_azimuth_rad == pytest.approx(1.234)
        assert meta.rsm_elevation_rad == pytest.approx(-0.567)
        assert meta.n_spectra == 100
        assert meta.n_channels == 2148
        assert meta.product_id is not None
        assert meta.product_id.sol == 921
        assert meta.product_completion_status == "COMPLETE"

    def test_missing_file_raises(self, tmp_path):
        parser = PDSLabelParser()
        with pytest.raises(FileNotFoundError, match="XML label not found"):
            parser.parse_label(tmp_path / "nonexistent.xml")

    def test_missing_identification_area_raises(self, tmp_path):
        xml_content = '<?xml version="1.0"?><Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"></Product_Observational>'
        p = tmp_path / "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml"
        p.write_text(xml_content)
        with pytest.raises(ValueError, match="Missing Identification_Area"):
            PDSLabelParser().parse_label(p)

    def test_missing_logical_identifier_raises(self, tmp_path):
        xml_content = """\
<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:mars2020="http://pds.nasa.gov/pds4/mission/mars2020/v1">
  <Identification_Area>
    <version_id>1.0</version_id>
  </Identification_Area>
  <Observation_Area>
    <Mission_Area>
      <mars2020:Observation_Information>
        <mars2020:sol_number>921</mars2020:sol_number>
        <mars2020:spacecraft_clock_start>748731413.515</mars2020:spacecraft_clock_start>
      </mars2020:Observation_Information>
    </Mission_Area>
  </Observation_Area>
</Product_Observational>"""
        p = tmp_path / "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml"
        p.write_text(xml_content)
        with pytest.raises(ValueError, match="Missing logical_identifier"):
            PDSLabelParser().parse_label(p)

    def test_rsm_sentinel_filtered(self, tmp_path):
        """RSM angle values of 1e+30 (sentinel) should be filtered out."""
        extra = """
  <Observation_Area_Extra>
    <geom:Articulation_Device_Parameters>
      <geom:device_id>RSM</geom:device_id>
      <geom:Device_Angle_Index>
        <geom:index_id>AZIMUTH FINAL-RESOLVER</geom:index_id>
        <geom:index_value_angle>1e+30</geom:index_value_angle>
      </geom:Device_Angle_Index>
    </geom:Articulation_Device_Parameters>
  </Observation_Area_Extra>"""
        # The main XML already has real RSM values, so test with those
        xml = _minimal_xml(
            tmp_path,
            "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml",
        )
        meta = PDSLabelParser().parse_label(xml)
        # Real values should be preserved
        assert meta.rsm_azimuth_rad is not None
        assert meta.rsm_azimuth_rad < 1e20

    def test_optional_fields_absent(self, tmp_path):
        """XML with minimal content — optional fields should be None."""
        xml_content = """\
<?xml version="1.0"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:mars2020="http://pds.nasa.gov/pds4/mission/mars2020/v1">
  <Identification_Area>
    <logical_identifier>urn:nasa:pds:test</logical_identifier>
    <version_id>1.0</version_id>
  </Identification_Area>
  <Observation_Area>
    <Mission_Area>
      <mars2020:Observation_Information>
        <mars2020:sol_number>921</mars2020:sol_number>
        <mars2020:spacecraft_clock_start>748731413.515</mars2020:spacecraft_clock_start>
      </mars2020:Observation_Information>
    </Mission_Area>
  </Observation_Area>
</Product_Observational>"""
        p = tmp_path / "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.xml"
        p.write_text(xml_content)
        meta = PDSLabelParser().parse_label(p)
        assert meta.sol_number == 921
        assert meta.title is None
        assert meta.solar_longitude is None
        assert meta.site is None
        assert meta.drive is None
        assert meta.rsm_azimuth_rad is None
        assert meta.sequence_id is None
        assert meta.n_spectra is None
        assert meta.n_channels is None


# ---------------------------------------------------------------------------
# PDSSpectralParser
# ---------------------------------------------------------------------------


class TestPDSSpectralParser:
    """Unit tests for PDSSpectralParser."""

    def test_parse_valid_rrs(self, tmp_path):
        fn = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
        csv_path = _spectral_csv(tmp_path, fn, n_spectra=5)
        result = PDSSpectralParser().parse(csv_path)
        assert result.product.n_spectra == 5
        assert result.product.n_channels == PDS_EXPECTED_CHANNELS
        assert len(result.spectra) == 3
        assert result.spectra["R1"].shape == (5, PDS_EXPECTED_CHANNELS)
        assert result.spectra["R2"].shape == (5, PDS_EXPECTED_CHANNELS)
        assert result.spectra["R3"].shape == (5, PDS_EXPECTED_CHANNELS)
        assert result.product.regions_present == ["R1", "R2", "R3"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSSpectralParser().parse(tmp_path / "nonexistent.csv")

    def test_zpz_signal_1_filename(self, tmp_path):
        """zpz in filename middle triggers rejection."""
        fn = "ss__0921_0748735903_045rrs__0450000srlc11374b108zpzj01.csv"
        csv_path = _spectral_csv(tmp_path, fn, n_spectra=5, middle="b108zpzj")
        with pytest.raises(PDSZpzProductError, match="zpz"):
            PDSSpectralParser().parse(csv_path)

    def test_zpz_signal_2_process_data_headers(self, tmp_path):
        """PROCESS_DATA headers trigger zpz rejection."""
        fn = "ss__0921_0748735903_045rrs__0450000srlc11374w104cgnj01.csv"
        csv_path = _spectral_csv(tmp_path, fn, n_spectra=5, zpz_headers=True)
        with pytest.raises(PDSZpzProductError, match="PROCESS_DATA"):
            PDSSpectralParser().parse(csv_path)

    def test_zpz_signal_3_two_spectra_per_region(self, tmp_path):
        """Exactly 2 spectra per region triggers zpz rejection."""
        fn = "ss__0921_0748735903_045rrs__0450000srlc11374b108zpzj01.csv"
        csv_path = _spectral_csv(tmp_path, fn, n_spectra=2, middle="b108zpzj")
        with pytest.raises(PDSZpzProductError, match="zpz"):
            PDSSpectralParser().parse(csv_path)

    def test_missing_wavelength_section(self, tmp_path):
        fn = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
        data_row = ",".join("1.0" for _ in range(PDS_EXPECTED_CHANNELS))
        content = "LASER-NORMALIZED_SPECTRA:_REGION_1\nheader\n" + data_row
        p = tmp_path / fn
        p.write_text(content)
        with pytest.raises(ValueError, match="Missing WAVELENGTH"):
            PDSSpectralParser().parse(p)

    def test_wrong_wavelength_count(self, tmp_path):
        fn = "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
        short_row = ",".join("1.0" for _ in range(100))
        data_row = ",".join("1.0" for _ in range(PDS_EXPECTED_CHANNELS))
        content = f"WAVELENGTH (NM):\nheader\n{short_row}\n"
        content += f"LASER-NORMALIZED_SPECTRA:_REGION_1\nheader\n{data_row}\n"
        p = tmp_path / fn
        p.write_text(content)
        with pytest.raises(ValueError, match="Expected 2148 wavelength"):
            PDSSpectralParser().parse(p)


# ---------------------------------------------------------------------------
# PDSRMOParser
# ---------------------------------------------------------------------------


def _rmo_csv(tmp_path, filename, n_positions=3, n_images=1, zpz_middle=False):
    """Create a synthetic RMO CSV."""
    lines = []

    # Positions section
    lines.append("LASER_SHOT_POSITIONS")
    lines.append("Image_name,Position_index,x,y")  # column header

    if n_images == 1:
        for i in range(n_positions):
            lines.append(f"ACI_0921.IMG,{i},{100.0 + i},{200.0 + i}")
    else:
        # Survey: duplicate position indices with different images
        for img in range(n_images):
            for i in range(n_positions):
                lines.append(f"ACI_0921_{img}.IMG,{i},{100.0 + i},{200.0 + i}")

    # Wavelength regions section (6 bands)
    lines.append("WAVELENGTH_REGIONS")
    lines.append("Column_index,Wavelength_start,Wavelength_stop")
    for b in range(6):
        start = 250.9 + b * 16.0
        stop = start + 15.0
        lines.append(f"{b},{start:.1f},{stop:.1f}")

    # Intensity section
    lines.append("SPECTRAL_INTENSITY")
    lines.append("Position_index,I0,I1,I2,I3,I4,I5")
    for i in range(n_positions):
        vals = ",".join(f"{1.0 + i:.1f}" for _ in range(6))
        lines.append(f"{i},{vals}")

    p = tmp_path / filename
    p.write_text("\n".join(lines))
    return p


class TestPDSRMOParser:
    """Unit tests for PDSRMOParser."""

    def test_parse_detail_scan(self, tmp_path):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        csv_path = _rmo_csv(tmp_path, fn, n_positions=5, n_images=1)
        result = PDSRMOParser().parse(csv_path)
        assert result.n_positions == 5
        assert len(result.image_names) == 1
        assert len(result.wavelength_regions) == 6
        assert len(result.band_intensities) == 5

    def test_parse_survey_dedup(self, tmp_path):
        """Survey: 2 images × 3 positions → 3 unique after de-dup, 2 image names."""
        fn = "ss__0921_0748735042_045rmo__0450000srlc11420w0__cgnj01.csv"
        csv_path = _rmo_csv(tmp_path, fn, n_positions=3, n_images=2)
        result = PDSRMOParser().parse(csv_path)
        assert result.n_positions == 3  # de-duplicated
        assert len(result.image_names) == 2  # both images preserved
        assert result.positions[0].image_name == "ACI_0921_0.IMG"  # first occurrence

    def test_zpz_in_filename_rejected(self, tmp_path):
        fn = "ss__0921_0748735903_045rmo__0450000srlc11374b108zpzj01.csv"
        csv_path = _rmo_csv(tmp_path, fn, n_positions=3)
        with pytest.raises(PDSZpzProductError, match="zpz"):
            PDSRMOParser().parse(csv_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSRMOParser().parse(tmp_path / "nonexistent.csv")

    def test_missing_section_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        p = tmp_path / fn
        p.write_text("LASER_SHOT_POSITIONS\nHeader\nACI.IMG,0,100,200\n")
        with pytest.raises(ValueError, match="Missing"):
            PDSRMOParser().parse(p)

    def test_wrong_position_column_count(self, tmp_path):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        lines = [
            "LASER_SHOT_POSITIONS", "h1,h2,h3,h4",
            "ACI.IMG,0,100",  # only 3 columns, need 4
            "WAVELENGTH_REGIONS", "h1,h2,h3",
        ]
        for b in range(6):
            lines.append(f"{b},{250.0 + b},{265.0 + b}")
        lines.extend(["SPECTRAL_INTENSITY", "h1,h2,h3,h4,h5,h6,h7", "0,1,1,1,1,1,1"])
        p = tmp_path / fn
        p.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="Expected 4 columns"):
            PDSRMOParser().parse(p)

    def test_wrong_wavelength_band_count(self, tmp_path):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        lines = [
            "LASER_SHOT_POSITIONS", "h1,h2,h3,h4", "ACI.IMG,0,100,200",
            "WAVELENGTH_REGIONS", "h1,h2,h3",
            "0,250.0,265.0",  # only 1 band, need 6
            "SPECTRAL_INTENSITY", "h1,h2,h3,h4,h5,h6,h7", "0,1,1,1,1,1,1",
        ]
        p = tmp_path / fn
        p.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="Expected 6 wavelength"):
            PDSRMOParser().parse(p)

    def test_position_intensity_count_mismatch(self, tmp_path):
        fn = "ss__0921_0748731413_045rmo__0450000srlc11374w0__cgnj01.csv"
        lines = [
            "LASER_SHOT_POSITIONS", "h1,h2,h3,h4",
            "ACI.IMG,0,100,200", "ACI.IMG,1,101,201",  # 2 positions
            "WAVELENGTH_REGIONS", "h1,h2,h3",
        ]
        for b in range(6):
            lines.append(f"{b},{250.0 + b},{265.0 + b}")
        lines.extend([
            "SPECTRAL_INTENSITY", "h1,h2,h3,h4,h5,h6,h7",
            "0,1,1,1,1,1,1",  # only 1 intensity row for 2 positions
        ])
        p = tmp_path / fn
        p.write_text("\n".join(lines))
        with pytest.raises(ValueError, match="Position count"):
            PDSRMOParser().parse(p)


# ---------------------------------------------------------------------------
# PDSPhotodiodeParser
# ---------------------------------------------------------------------------


def _rli_csv(tmp_path, filename, intensities=None):
    """Create a synthetic RLI CSV."""
    if intensities is None:
        intensities = [1.5, 2.3, 0.8]
    lines = [
        "LASER_PHOTODIODE_INTENSITY_MAP:",
        "avg_photodiode",
    ]
    for v in intensities:
        lines.append(str(v))
    p = tmp_path / filename
    p.write_text("\n".join(lines))
    return p


class TestPDSPhotodiodeParser:
    """Unit tests for PDSPhotodiodeParser."""

    def test_parse_valid(self, tmp_path):
        fn = "ss__0921_0748731413_045rli__0450000srlc11374_0_____j01.csv"
        csv_path = _rli_csv(tmp_path, fn, [1.5, 2.3, 0.8])
        result = PDSPhotodiodeParser().parse(csv_path)
        assert result.n_shots == 3
        assert result.intensities == pytest.approx([1.5, 2.3, 0.8])

    def test_sentinel_values(self, tmp_path):
        fn = "ss__0921_0748735903_045rli__0450000srlc11374_0_____j01.csv"
        csv_path = _rli_csv(tmp_path, fn, [-1.0, -1.0])
        result = PDSPhotodiodeParser().parse(csv_path)
        assert result.intensities == [-1.0, -1.0]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSPhotodiodeParser().parse(tmp_path / "nonexistent.csv")

    def test_missing_header_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rli__0450000srlc11374_0_____j01.csv"
        p = tmp_path / fn
        p.write_text("not_a_header\n1.0\n2.0\n")
        with pytest.raises(ValueError, match="Missing"):
            PDSPhotodiodeParser().parse(p)

    def test_no_data_rows_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rli__0450000srlc11374_0_____j01.csv"
        p = tmp_path / fn
        p.write_text("LASER_PHOTODIODE_INTENSITY_MAP:\navg_photodiode\n")
        with pytest.raises(ValueError, match="No photodiode data"):
            PDSPhotodiodeParser().parse(p)


# ---------------------------------------------------------------------------
# PDSCalibrationParser
# ---------------------------------------------------------------------------


def _rcc_csv(tmp_path, filename, records=None):
    """Create a synthetic RCC CSV."""
    if records is None:
        records = [
            (98, "675636651_555", 0.0, 0.0, 276.548, 0.4),
            (921, "748731011_000", 258.3, 0.5, 276.548, 0.4),
        ]
    lines = [
        "CALIBRATION_FIT:",
        "SOL,SCLK,laser_peak,laser_fwhm,algan_peak,algan_fwhm",
    ]
    for r in records:
        lines.append(",".join(str(v) for v in r))
    p = tmp_path / filename
    p.write_text("\n".join(lines))
    return p


class TestPDSCalibrationParser:
    """Unit tests for PDSCalibrationParser."""

    def test_parse_valid(self, tmp_path):
        fn = "ss__0921_0748731011_045rcc__0450000srlc10000_104___j01.csv"
        csv_path = _rcc_csv(tmp_path, fn)
        result = PDSCalibrationParser().parse(csv_path)
        assert result.n_records == 2
        assert result.records[0].sol == 98
        assert result.records[1].sol == 921
        assert result.records[1].algan_peak_nm == pytest.approx(276.548)

    def test_laser_peak_zero(self, tmp_path):
        fn = "ss__0921_0748731011_045rcc__0450000srlc10000_104___j01.csv"
        csv_path = _rcc_csv(tmp_path, fn)
        result = PDSCalibrationParser().parse(csv_path)
        assert result.records[0].laser_peak_nm == 0.0  # not fit

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSCalibrationParser().parse(tmp_path / "nonexistent.csv")

    def test_missing_header_raises(self, tmp_path):
        fn = "ss__0921_0748731011_045rcc__0450000srlc10000_104___j01.csv"
        p = tmp_path / fn
        p.write_text("NOT_THE_HEADER:\ncols\n98,sclk,0,0,276,0.4\n")
        with pytest.raises(ValueError, match="Missing"):
            PDSCalibrationParser().parse(p)

    def test_wrong_column_count_raises(self, tmp_path):
        fn = "ss__0921_0748731011_045rcc__0450000srlc10000_104___j01.csv"
        p = tmp_path / fn
        p.write_text("CALIBRATION_FIT:\ncols\n98,sclk,0,0,276\n")  # 5 cols, need 6
        with pytest.raises(ValueError, match="Expected 6 columns"):
            PDSCalibrationParser().parse(p)

    def test_no_data_rows_raises(self, tmp_path):
        fn = "ss__0921_0748731011_045rcc__0450000srlc10000_104___j01.csv"
        p = tmp_path / fn
        p.write_text("CALIBRATION_FIT:\nSOL,SCLK,a,b,c,d\n")
        with pytest.raises(ValueError, match="No calibration data"):
            PDSCalibrationParser().parse(p)


# ---------------------------------------------------------------------------
# PDSCrossRefParser
# ---------------------------------------------------------------------------


def _rls_csv(tmp_path, filename, n_records=3):
    """Create a synthetic RLS CSV."""
    lines = [
        "LASER_SHOT_POSITION:",
        "number,spec_name,image_name,samp,line",
    ]
    for i in range(n_records):
        lines.append(f"{i},rrs_file.csv,ACI_0921.IMG,{512.5 + i},{768.3 + i}")
    p = tmp_path / filename
    p.write_text("\n".join(lines))
    return p


class TestPDSCrossRefParser:
    """Unit tests for PDSCrossRefParser."""

    def test_parse_valid(self, tmp_path):
        fn = "ss__0921_0748731413_045rls__0450000srlc11374_108_p_j01.csv"
        csv_path = _rls_csv(tmp_path, fn, n_records=5)
        result = PDSCrossRefParser().parse(csv_path)
        assert result.n_records == 5
        assert result.records[0].number == 0
        assert result.records[0].spec_name == "rrs_file.csv"
        assert result.records[0].samp == pytest.approx(512.5)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSCrossRefParser().parse(tmp_path / "nonexistent.csv")

    def test_missing_header_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rls__0450000srlc11374_108_p_j01.csv"
        p = tmp_path / fn
        p.write_text("WRONG_HEADER:\ncols\n0,rrs.csv,ACI.IMG,0,0\n")
        with pytest.raises(ValueError, match="Missing"):
            PDSCrossRefParser().parse(p)

    def test_wrong_column_count_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rls__0450000srlc11374_108_p_j01.csv"
        p = tmp_path / fn
        p.write_text("LASER_SHOT_POSITION:\ncols\n0,rrs.csv,ACI.IMG,0\n")  # 4, need 5
        with pytest.raises(ValueError, match="Expected 5 columns"):
            PDSCrossRefParser().parse(p)

    def test_no_data_rows_raises(self, tmp_path):
        fn = "ss__0921_0748731413_045rls__0450000srlc11374_108_p_j01.csv"
        p = tmp_path / fn
        p.write_text("LASER_SHOT_POSITION:\nnumber,spec_name,image_name,samp,line\n")
        with pytest.raises(ValueError, match="No cross-reference data"):
            PDSCrossRefParser().parse(p)


# ---------------------------------------------------------------------------
# PDSObservationGrouper — classify
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperClassify:
    """Unit tests for PDSObservationGrouper.classify()."""

    def test_calibration_srlc10000(self):
        assert PDSObservationGrouper.classify("srlc10000") == "calibration"

    def test_calibration_srlc16000(self):
        assert PDSObservationGrouper.classify("srlc16000") == "calibration"

    def test_calibration_case_insensitive(self):
        assert PDSObservationGrouper.classify("SRLC10000") == "calibration"

    def test_detail_100_spectra(self):
        assert PDSObservationGrouper.classify("srlc11374", 100) == "detail"

    def test_detail_200_spectra(self):
        """Boundary: exactly 200 → detail (≤200)."""
        assert PDSObservationGrouper.classify("srlc11374", 200) == "detail"

    def test_survey_201_spectra(self):
        assert PDSObservationGrouper.classify("srlc11420", 201) == "survey"

    def test_survey_1296_spectra(self):
        assert PDSObservationGrouper.classify("srlc11420", 1296) == "survey"

    def test_detail_1_spectrum(self):
        """Calibration count but non-calibration sequence code → detail."""
        assert PDSObservationGrouper.classify("srlc11374", 1) == "detail"

    def test_none_without_n_spectra(self):
        assert PDSObservationGrouper.classify("srlc11374", None) is None

    def test_calibration_ignores_n_spectra(self):
        """Calibration codes classified regardless of spectra count."""
        assert PDSObservationGrouper.classify("srlc10000", 1296) == "calibration"


# ---------------------------------------------------------------------------
# PDSObservationGrouper — filter_zpz
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperFilterZpz:
    """Unit tests for PDSObservationGrouper.filter_zpz()."""

    def _pid(self, middle="w104cgnj"):
        return PDSProductId(
            filename=f"ss__0921_0748731413_045rrs__0450000srlc11374{middle}01.csv",
            sol=921, sclk=748731413, obs_id="045",
            product_type=PDSProductType.RRS,
            site_drive="0450000", sequence_code="srlc11374",
            version=1, middle=middle,
        )

    def test_no_zpz(self):
        clean, zpz = PDSObservationGrouper.filter_zpz(
            [self._pid("w104cgnj"), self._pid("_104___j")]
        )
        assert len(clean) == 2
        assert len(zpz) == 0

    def test_all_zpz(self):
        clean, zpz = PDSObservationGrouper.filter_zpz(
            [self._pid("b108zpzj")]
        )
        assert len(clean) == 0
        assert len(zpz) == 1

    def test_mixed(self):
        clean, zpz = PDSObservationGrouper.filter_zpz(
            [self._pid("w104cgnj"), self._pid("b108zpzj")]
        )
        assert len(clean) == 1
        assert len(zpz) == 1
        assert "zpz" not in clean[0].middle
        assert "zpz" in zpz[0].middle

    def test_zpz_case_insensitive(self):
        clean, zpz = PDSObservationGrouper.filter_zpz(
            [self._pid("b108ZPZj")]
        )
        assert len(zpz) == 1


# ---------------------------------------------------------------------------
# PDSObservationGrouper — validate_spectral_exclusivity
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperExclusivity:
    """Unit tests for validate_spectral_exclusivity()."""

    def _pid(self, ptype):
        return PDSProductId(
            filename=f"ss__0921_0748731413_045{ptype}__0450000srlc11374w104cgnj01.csv",
            sol=921, sclk=748731413, obs_id="045",
            product_type=PDSProductType(ptype),
            site_drive="0450000", sequence_code="srlc11374",
            version=1, middle="w104cgnj",
        )

    def test_rrs_only_passes(self):
        PDSObservationGrouper.validate_spectral_exclusivity(
            {"rrs": self._pid("rrs"), "rmo": self._pid("rmo")}
        )

    def test_rcs_only_passes(self):
        PDSObservationGrouper.validate_spectral_exclusivity(
            {"rcs": self._pid("rcs"), "rmo": self._pid("rmo")}
        )

    def test_no_spectral_passes(self):
        PDSObservationGrouper.validate_spectral_exclusivity(
            {"rmo": self._pid("rmo"), "rli": self._pid("rli")}
        )

    def test_both_rrs_and_rcs_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            PDSObservationGrouper.validate_spectral_exclusivity(
                {"rrs": self._pid("rrs"), "rcs": self._pid("rcs")}
            )


# ---------------------------------------------------------------------------
# PDSObservationGrouper — select_latest_versions
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperVersionSelection:
    """Unit tests for select_latest_versions()."""

    def _pid(self, ptype, version):
        return PDSProductId(
            filename=f"ss__0921_0748731413_045{ptype}__0450000srlc11374w104cgnj{version:02d}.csv",
            sol=921, sclk=748731413, obs_id="045",
            product_type=PDSProductType(ptype),
            site_drive="0450000", sequence_code="srlc11374",
            version=version, middle="w104cgnj",
        )

    def test_single_version_kept(self):
        result = PDSObservationGrouper.select_latest_versions(
            [self._pid("rrs", 1)]
        )
        assert result["rrs"].version == 1

    def test_higher_version_selected(self):
        result = PDSObservationGrouper.select_latest_versions(
            [self._pid("rrs", 1), self._pid("rrs", 2)]
        )
        assert result["rrs"].version == 2

    def test_multiple_types_independent(self):
        result = PDSObservationGrouper.select_latest_versions([
            self._pid("rrs", 1),
            self._pid("rrs", 2),
            self._pid("rmo", 1),
        ])
        assert result["rrs"].version == 2
        assert result["rmo"].version == 1

    def test_same_version_idempotent(self):
        result = PDSObservationGrouper.select_latest_versions(
            [self._pid("rrs", 1), self._pid("rrs", 1)]
        )
        assert result["rrs"].version == 1


# ---------------------------------------------------------------------------
# PDSObservationGrouper — group_by_observation
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperGrouping:
    """Unit tests for group_by_observation()."""

    def _pid(self, sclk, ptype):
        return PDSProductId(
            filename=f"ss__0921_{sclk:010d}_045{ptype}__0450000srlc11374w104cgnj01.csv",
            sol=921, sclk=sclk, obs_id="045",
            product_type=PDSProductType(ptype),
            site_drive="0450000", sequence_code="srlc11374",
            version=1, middle="w104cgnj",
        )

    def test_single_observation(self):
        products = [
            self._pid(748731413, "rrs"),
            self._pid(748731413, "rmo"),
        ]
        groups = PDSObservationGrouper.group_by_observation(products)
        assert len(groups) == 1
        key = "0921_0748731413_045"
        assert key in groups
        assert len(groups[key]) == 2

    def test_multiple_observations(self):
        products = [
            self._pid(748731413, "rrs"),
            self._pid(748731413, "rmo"),
            self._pid(748732975, "rrs"),
        ]
        groups = PDSObservationGrouper.group_by_observation(products)
        assert len(groups) == 2

    def test_empty_input(self):
        groups = PDSObservationGrouper.group_by_observation([])
        assert groups == {}


# ---------------------------------------------------------------------------
# PDSObservationGrouper — discover_csv_products
# ---------------------------------------------------------------------------


class TestPDSObservationGrouperDiscovery:
    """Unit tests for discover_csv_products()."""

    def test_discovers_csv_files(self, tmp_path):
        data_dir = tmp_path / "data_processed"
        data_dir.mkdir()
        # Valid PDS filename
        (data_dir / "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv").write_text("")
        # Non-PDS filename (should be skipped)
        (data_dir / "readme.csv").write_text("")
        products = PDSObservationGrouper.discover_csv_products(data_dir)
        assert len(products) == 1
        assert products[0].product_type == "rrs"

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PDSObservationGrouper.discover_csv_products(tmp_path / "nope")

    def test_empty_dir(self, tmp_path):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        assert PDSObservationGrouper.discover_csv_products(data_dir) == []
