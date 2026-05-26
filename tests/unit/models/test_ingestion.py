"""
Unit tests for ingestion models (bd-15c: WS2-B).

Tests the data ingestion models and parsers defined in models/ingestion.py:
- RawLoupeMetadata: loupe.csv parsing
- RawSpatialData: spatial.csv parsing
- RawPhotodiodeData: photodiodeRaw.csv parsing
- RawROIData: roi.csv parsing
- RawSpectraFile: spectra CSV parsing
- LoupeSessionFile: .lpe session file parsing
- LoupeWorkspaceParser: Full workspace parsing
"""

import csv
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from sherloc_pipeline.models import (
    SpectralRegion,
    SpectrumType,
    ProcessingLevel,
    Scan,
    ScanPoint,
    Spectrum,
    InstrumentState,
    CCDConfiguration,
    ScannerCalibration,
    RegionOfInterest,
    ContextImage,
    ImageType,
)
from sherloc_pipeline.models.ingestion import (
    RawLoupeMetadata,
    RawSpatialData,
    RawSpatialPoint,
    RawPhotodiodeData,
    RawPhotodiodeStats,
    RawROI,
    RawROIData,
    RawSpectraFile,
    LoupeSessionEntry,
    LoupeSessionFile,
    LoupeWorkspaceParser,
    LoupeWorkspaceResult,
    extract_sol_from_path,
    extract_target_from_lpe,
    discover_workspaces,
    _safe_int,
    _safe_float,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_safe_int_valid(self):
        """_safe_int converts valid strings."""
        assert _safe_int("123") == 123
        assert _safe_int("45.6") == 45
        assert _safe_int("-100") == -100

    def test_safe_int_invalid(self):
        """_safe_int returns None for invalid values."""
        assert _safe_int("") is None
        assert _safe_int("N/A") is None
        assert _safe_int("None") is None
        assert _safe_int("abc") is None

    def test_safe_float_valid(self):
        """_safe_float converts valid strings."""
        assert _safe_float("123.45") == 123.45
        assert _safe_float("-0.5") == -0.5
        assert _safe_float("100") == 100.0

    def test_safe_float_invalid(self):
        """_safe_float returns None for invalid values."""
        assert _safe_float("") is None
        assert _safe_float("N/A") is None
        assert _safe_float("abc") is None

    def test_extract_sol_from_path(self):
        """extract_sol_from_path finds sol number in paths."""
        assert extract_sol_from_path(Path("/data/loupe/sol_0921/detail")) == 921
        assert extract_sol_from_path(Path("/sol_0059/workspace")) == 59
        assert extract_sol_from_path(Path("/data/sol_1234_a/ws")) == 1234
        assert extract_sol_from_path(Path("/data/other/")) is None

    def test_extract_target_from_lpe_underscore(self, tmp_path):
        """extract_target_from_lpe parses underscore-separated target."""
        sol_dir = tmp_path / "sol_0921"
        sol_dir.mkdir()
        (sol_dir / "Sol_0921_Amherst_Point.lpe").write_text("")
        assert extract_target_from_lpe(sol_dir) == "Amherst Point"

    def test_extract_target_from_lpe_with_leading_space(self, tmp_path):
        """extract_target_from_lpe handles leading space after sol number."""
        sol_dir = tmp_path / "sol_1771"
        sol_dir.mkdir()
        (sol_dir / "Sol_1771_ Djuma.lpe").write_text("")
        assert extract_target_from_lpe(sol_dir) == "Djuma"

    def test_extract_target_from_lpe_single_word(self, tmp_path):
        """extract_target_from_lpe handles single-word targets."""
        sol_dir = tmp_path / "sol_0293"
        sol_dir.mkdir()
        (sol_dir / "Sol_0293_Quartier.lpe").write_text("")
        assert extract_target_from_lpe(sol_dir) == "Quartier"

    def test_extract_target_from_lpe_engineering(self, tmp_path):
        """extract_target_from_lpe extracts engineering target names too."""
        sol_dir = tmp_path / "sol_1677"
        sol_dir.mkdir()
        (sol_dir / "Sol_1677_arm_stowed_dark.lpe").write_text("")
        assert extract_target_from_lpe(sol_dir) == "arm stowed dark"

    def test_extract_target_from_lpe_no_lpe(self, tmp_path):
        """extract_target_from_lpe returns None when no .lpe file exists."""
        sol_dir = tmp_path / "sol_0100"
        sol_dir.mkdir()
        assert extract_target_from_lpe(sol_dir) is None

    def test_extract_target_from_lpe_unparseable(self, tmp_path):
        """extract_target_from_lpe returns None for non-standard names."""
        sol_dir = tmp_path / "sol_0100"
        sol_dir.mkdir()
        (sol_dir / "random_file.lpe").write_text("")
        assert extract_target_from_lpe(sol_dir) is None


class TestRawLoupeMetadata:
    """Tests for RawLoupeMetadata parsing."""

    @pytest.fixture
    def sample_loupe_csv(self, tmp_path):
        """Create a sample loupe.csv file."""
        csv_path = tmp_path / "loupe.csv"
        content = """original_data_file,SrlcSpecSpecSohRaw_0748731411-51550-1
human_readable_workspace,detail_1
n_spectra,100
n_channels,2148
laser_wavelength,248.5794
shots_per_spec,500
az_scale,0.628154699
el_scale,0.422441487
laser_x,809
laser_y,664
rotation,20.6793583
specProcessingApplied,None
CNDH_PCB_TEMP_STAT_REG,26.698 C
SE_CCD_TEMP_STAT_REG,-29.059 C
laser_shot_counter,7091325
LASER_INT_TIME,15 us
LASER_REP_RATE,80 Hz
"""
        csv_path.write_text(content)
        return csv_path

    def test_from_csv_path(self, sample_loupe_csv):
        """Parse loupe.csv file."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)

        assert metadata.original_data_file == "SrlcSpecSpecSohRaw_0748731411-51550-1"
        assert metadata.human_readable_workspace == "detail_1"
        assert metadata.n_spectra == "100"
        assert metadata.n_channels == "2148"
        assert metadata.laser_wavelength == "248.5794"
        assert metadata.shots_per_spec == "500"
        assert metadata.specProcessingApplied == "None"

    def test_extract_sclk(self, sample_loupe_csv):
        """Extract SCLK from original_data_file."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)
        sclk = metadata.extract_sclk()
        assert sclk == 748731411

    def test_to_scan(self, sample_loupe_csv):
        """Convert to Scan domain model."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)
        scan = metadata.to_scan(sol_number=921, source_path="/data/workspace")

        assert isinstance(scan, Scan)
        assert scan.sol_number == 921
        assert scan.scan_name == "detail_1"
        assert scan.scan_id == "SrlcSpecSpecSohRaw_0748731411-51550-1"
        assert scan.sclk_start == 748731411
        assert scan.n_points == 100
        assert scan.n_channels == 2148
        assert scan.shots_per_point == 500
        assert scan.source_path == "/data/workspace"
        assert scan.loupe_metadata is not None
        assert scan.processing_applied is None  # "None" converted to None

    def test_to_instrument_state(self, sample_loupe_csv):
        """Convert to InstrumentState domain model."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)
        scan = metadata.to_scan(sol_number=921)
        state = metadata.to_instrument_state(scan.id)

        assert isinstance(state, InstrumentState)
        assert state.scan_id == scan.id
        assert state.pcb_temp_c == pytest.approx(26.698, rel=0.01)
        assert state.ccd_temp_c == pytest.approx(-29.059, rel=0.01)
        assert state.laser_shot_counter == 7091325
        assert state.laser_int_time_us == 15
        assert state.laser_rep_rate_hz == 80

    def test_to_ccd_configuration(self, sample_loupe_csv):
        """Convert to CCDConfiguration domain model."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)
        scan = metadata.to_scan(sol_number=921)
        config = metadata.to_ccd_configuration(scan.id)

        assert isinstance(config, CCDConfiguration)
        assert config.scan_id == scan.id

    def test_to_scanner_calibration(self, sample_loupe_csv):
        """Convert to ScannerCalibration domain model."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)
        scan = metadata.to_scan(sol_number=921)
        cal = metadata.to_scanner_calibration(scan.id)

        assert isinstance(cal, ScannerCalibration)
        assert cal.az_scale == pytest.approx(0.628154699, rel=0.01)
        assert cal.el_scale == pytest.approx(0.422441487, rel=0.01)
        assert cal.laser_x == 809
        assert cal.laser_y == 664
        assert cal.rotation_deg == pytest.approx(20.6793583, rel=0.01)

    def test_raw_fields_preserved(self, sample_loupe_csv):
        """All raw fields are preserved in raw_fields dict."""
        metadata = RawLoupeMetadata.from_csv_path(sample_loupe_csv)

        assert "CNDH_PCB_TEMP_STAT_REG" in metadata.raw_fields
        assert metadata.raw_fields["CNDH_PCB_TEMP_STAT_REG"] == "26.698 C"


class TestRawSpatialData:
    """Tests for RawSpatialData parsing."""

    @pytest.fixture
    def sample_spatial_csv(self, tmp_path):
        """Create a sample spatial.csv file."""
        csv_path = tmp_path / "spatial.csv"
        content = """az,el
1041,726
994,503
934,293
x,y
0.518,0.503
0.419,0.509
0.323,0.505
az_err,el_err
38,-3
42,28
0,35
sum_current,diff_current
674,910
558,942
553,857
"""
        csv_path.write_text(content)
        return csv_path

    def test_from_csv_path(self, sample_spatial_csv):
        """Parse spatial.csv file."""
        spatial = RawSpatialData.from_csv_path(sample_spatial_csv)

        assert len(spatial.points) == 3

        # Check first point
        p0 = spatial.points[0]
        assert p0.point_index == 0
        assert p0.az == 1041
        assert p0.el == 726
        assert p0.x == pytest.approx(0.518, rel=0.01)
        assert p0.y == pytest.approx(0.503, rel=0.01)
        assert p0.az_err == 38
        assert p0.el_err == -3
        assert p0.sum_current == 674
        assert p0.diff_current == 910

    def test_to_scan_points(self, sample_spatial_csv):
        """Convert to ScanPoint domain models."""
        spatial = RawSpatialData.from_csv_path(sample_spatial_csv)
        scan_id = uuid.uuid4()
        scan_points = spatial.to_scan_points(scan_id)

        assert len(scan_points) == 3
        assert all(isinstance(p, ScanPoint) for p in scan_points)

        p0 = scan_points[0]
        assert p0.scan_id == scan_id
        assert p0.point_index == 0
        assert p0.azimuth_dn == 1041
        assert p0.elevation_dn == 726
        assert p0.x_pixel == pytest.approx(0.518, rel=0.01)


class TestRawSpatialPoint:
    """Tests for RawSpatialPoint model."""

    def test_to_scan_point(self):
        """Convert single point to ScanPoint."""
        scan_id = uuid.uuid4()
        raw_point = RawSpatialPoint(
            point_index=5,
            az=1000,
            el=500,
            x=0.5,
            y=0.3,
            az_err=10,
            el_err=20,
        )

        scan_point = raw_point.to_scan_point(scan_id)

        assert scan_point.scan_id == scan_id
        assert scan_point.point_index == 5
        assert scan_point.azimuth_dn == 1000
        assert scan_point.elevation_dn == 500
        assert scan_point.x_pixel == 0.5
        assert scan_point.y_pixel == 0.3
        assert scan_point.azimuth_error == 10.0
        assert scan_point.elevation_error == 20.0


class TestRawPhotodiodeData:
    """Tests for RawPhotodiodeData parsing."""

    @pytest.fixture
    def sample_photodiode_csv(self, tmp_path):
        """Create a sample photodiodeRaw.csv file."""
        csv_path = tmp_path / "photodiodeRaw.csv"
        # Header + 3 rows of data
        header = ",".join([f"shot_number_{i}" for i in range(10)])
        row1 = ",".join(["90", "91", "92", "93", "94", "95", "96", "97", "98", "99"])
        row2 = ",".join(["80", "81", "82", "83", "84", "85", "86", "87", "88", "89"])
        row3 = ",".join(["70", "71", "72", "73", "74", "75", "76", "77", "78", "79"])
        content = f"{header}\n{row1}\n{row2}\n{row3}"
        csv_path.write_text(content)
        return csv_path

    def test_from_csv_path(self, sample_photodiode_csv):
        """Parse photodiodeRaw.csv and compute statistics."""
        pd_data = RawPhotodiodeData.from_csv_path(sample_photodiode_csv)

        assert len(pd_data.stats) == 3

        # Point 0: values 90-99
        s0 = pd_data.stats[0]
        assert s0.point_index == 0
        assert s0.mean == pytest.approx(94.5, rel=0.01)
        assert s0.min_value == 90
        assert s0.max_value == 99

    def test_update_scan_points(self, sample_photodiode_csv):
        """Update ScanPoints with photodiode statistics."""
        pd_data = RawPhotodiodeData.from_csv_path(sample_photodiode_csv)

        # Create scan points
        scan_id = uuid.uuid4()
        scan_points = [
            ScanPoint(scan_id=scan_id, point_index=i) for i in range(3)
        ]

        # Update with stats
        pd_data.update_scan_points(scan_points)

        assert scan_points[0].photodiode_mean == pytest.approx(94.5, rel=0.01)
        assert scan_points[0].photodiode_std is not None


class TestRawROIData:
    """Tests for RawROIData parsing."""

    @pytest.fixture
    def sample_roi_csv(self, tmp_path):
        """Create a sample roi.csv file."""
        csv_path = tmp_path / "roi.csv"
        content = """Full Map
#ffffff
0
1
2
3
ENDROI
Carbonate Vein
#00ff00
5
6
7
ENDROI
"""
        csv_path.write_text(content)
        return csv_path

    def test_from_csv_path(self, sample_roi_csv):
        """Parse roi.csv file."""
        roi_data = RawROIData.from_csv_path(sample_roi_csv)

        assert len(roi_data.rois) == 2

        # First ROI
        r0 = roi_data.rois[0]
        assert r0.name == "Full Map"
        assert r0.color == "#ffffff"
        assert r0.point_indices == [0, 1, 2, 3]

        # Second ROI
        r1 = roi_data.rois[1]
        assert r1.name == "Carbonate Vein"
        assert r1.color == "#00ff00"
        assert r1.point_indices == [5, 6, 7]

    def test_to_regions_of_interest(self, sample_roi_csv):
        """Convert to RegionOfInterest domain models."""
        roi_data = RawROIData.from_csv_path(sample_roi_csv)
        scan_id = uuid.uuid4()
        rois = roi_data.to_regions_of_interest(scan_id)

        assert len(rois) == 2
        assert all(isinstance(r, RegionOfInterest) for r in rois)

        r0 = rois[0]
        assert r0.scan_id == scan_id
        assert r0.name == "Full Map"


class TestRawROI:
    """Tests for RawROI model."""

    def test_to_region_of_interest(self):
        """Convert single ROI to domain model."""
        scan_id = uuid.uuid4()
        raw_roi = RawROI(
            name="Test ROI",
            color="#ff0000",
            point_indices=[1, 2, 3],
        )

        roi = raw_roi.to_region_of_interest(scan_id)

        assert roi.scan_id == scan_id
        assert roi.name == "Test ROI"
        assert roi.color_hex == "#FF0000"  # Uppercased
        assert roi.point_indices == [1, 2, 3]


class TestRawSpectraFile:
    """Tests for RawSpectraFile parsing."""

    @pytest.fixture
    def sample_spectra_csv(self, tmp_path):
        """Create a sample spectra CSV file."""
        csv_path = tmp_path / "activeSpectra.csv"
        # 5 channels, 3 points
        header = "R1_Channel0,R1_Channel1,R1_Channel2,R1_Channel3,R1_Channel4"
        row1 = "100.0,110.0,120.0,130.0,140.0"
        row2 = "200.0,210.0,220.0,230.0,240.0"
        row3 = "300.0,310.0,320.0,330.0,340.0"
        content = f"{header}\n{row1}\n{row2}\n{row3}"
        csv_path.write_text(content)
        return csv_path

    def test_parse_header(self, sample_spectra_csv):
        """Parse just the header."""
        raw_file = RawSpectraFile.parse_header(
            sample_spectra_csv, SpectrumType.ACTIVE
        )

        assert raw_file.file_type == SpectrumType.ACTIVE
        assert raw_file.n_channels == 5
        assert len(raw_file.channel_names) == 5
        assert raw_file.channel_names[0] == "R1_Channel0"

    def test_from_csv_path(self, sample_spectra_csv):
        """Parse full spectra file."""
        raw_file, data = RawSpectraFile.from_csv_path(
            sample_spectra_csv, SpectrumType.ACTIVE
        )

        assert raw_file.n_points == 3
        assert raw_file.n_channels == 5
        assert len(data) == 3
        assert data[0] == [100.0, 110.0, 120.0, 130.0, 140.0]

    def test_iter_spectra(self, sample_spectra_csv):
        """Iterate and create Spectrum models."""
        raw_file, data = RawSpectraFile.from_csv_path(
            sample_spectra_csv, SpectrumType.ACTIVE
        )

        # Create scan point IDs
        scan_point_ids = [uuid.uuid4() for _ in range(3)]

        spectra = list(
            raw_file.iter_spectra(data, scan_point_ids, ProcessingLevel.RAW)
        )

        assert len(spectra) == 3
        assert all(isinstance(s, Spectrum) for s in spectra)

        s0 = spectra[0]
        assert s0.scan_point_id == scan_point_ids[0]
        # Sprint 4 fix: default section is R1, so region should be R1 (not R123)
        assert s0.region == SpectralRegion.R1
        assert s0.spectrum_type == SpectrumType.ACTIVE
        assert s0.processing_level == ProcessingLevel.RAW

        # Check intensities
        values = s0.intensity_values
        assert len(values) == 5
        assert values[0] == pytest.approx(100.0, rel=0.01)


class TestRawSpectraFileMultiSection:
    """Tests for RawSpectraFile R2/R3 section parsing (bd-13h).

    Loupe CSVs have 3 sections (R1, R2, R3) each with a header row and N data rows.
    The full CSV has 3N+3 rows total (3 headers + 3*N data rows).
    """

    @pytest.fixture
    def multi_section_spectra_csv(self, tmp_path):
        """Create a multi-section spectra CSV (R1 + R2 + R3).

        Simulates a darkSubSpectra.csv with 3 scan points and 5 channels.
        R1 values: 100s, R2 values: 200s, R3 values: 300s
        """
        csv_path = tmp_path / "darkSubSpectra.csv"
        lines = [
            # R1 section: header + 3 data rows
            "R1_Channel0,R1_Channel1,R1_Channel2,R1_Channel3,R1_Channel4",
            "100.0,110.0,120.0,130.0,140.0",
            "101.0,111.0,121.0,131.0,141.0",
            "102.0,112.0,122.0,132.0,142.0",
            # R2 section: header + 3 data rows
            "R2_Channel0,R2_Channel1,R2_Channel2,R2_Channel3,R2_Channel4",
            "200.0,210.0,220.0,230.0,240.0",
            "201.0,211.0,221.0,231.0,241.0",
            "202.0,212.0,222.0,232.0,242.0",
            # R3 section: header + 3 data rows
            "R3_Channel0,R3_Channel1,R3_Channel2,R3_Channel3,R3_Channel4",
            "300.0,310.0,320.0,330.0,340.0",
            "301.0,311.0,321.0,331.0,341.0",
            "302.0,312.0,322.0,332.0,342.0",
        ]
        csv_path.write_text("\n".join(lines))
        return csv_path

    def test_read_r1_section(self, multi_section_spectra_csv):
        """R1 section reads rows 0:N (first section)."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R1"
        )

        assert raw_file.section == "R1"
        assert raw_file.n_points == 3
        assert raw_file.n_channels == 5
        assert len(data) == 3
        # R1 values start at 100
        assert data[0][0] == 100.0
        assert data[1][0] == 101.0
        assert data[2][0] == 102.0

    def test_read_r2_section(self, multi_section_spectra_csv):
        """R2 section reads rows N:2N (second section)."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R2"
        )

        assert raw_file.section == "R2"
        assert raw_file.n_points == 3
        assert raw_file.n_channels == 5
        assert len(data) == 3
        # R2 values start at 200
        assert data[0][0] == 200.0
        assert data[1][0] == 201.0
        assert data[2][0] == 202.0

    def test_read_r3_section(self, multi_section_spectra_csv):
        """R3 section reads rows 2N:3N (third section)."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R3"
        )

        assert raw_file.section == "R3"
        assert raw_file.n_points == 3
        assert raw_file.n_channels == 5
        assert len(data) == 3
        # R3 values start at 300
        assert data[0][0] == 300.0
        assert data[1][0] == 301.0
        assert data[2][0] == 302.0

    def test_section_case_insensitive(self, multi_section_spectra_csv):
        """Section parameter is case-insensitive (e.g., 'r2' -> 'R2')."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="r2"
        )

        assert raw_file.section == "R2"
        assert data[0][0] == 200.0

    def test_sections_are_independent(self, multi_section_spectra_csv):
        """Each section returns distinct data (no cross-contamination)."""
        _, r1_data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R1"
        )
        _, r2_data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R2"
        )
        _, r3_data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R3"
        )

        # R1, R2, R3 have distinct value ranges
        assert r1_data[0][0] == 100.0
        assert r2_data[0][0] == 200.0
        assert r3_data[0][0] == 300.0

        # Same number of points across all sections
        assert len(r1_data) == len(r2_data) == len(r3_data) == 3

    def test_section_count_consistency(self, multi_section_spectra_csv):
        """_count_section_rows returns correct N for all sections."""
        n_spectra = RawSpectraFile._count_section_rows(multi_section_spectra_csv)
        assert n_spectra == 3

    def test_r2_region_label(self, multi_section_spectra_csv):
        """R2 section produces Spectrum with SpectralRegion.R2."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R2"
        )
        scan_point_ids = [uuid.uuid4() for _ in range(3)]

        spectra = list(
            raw_file.iter_spectra(data, scan_point_ids, ProcessingLevel.RAW)
        )

        assert len(spectra) == 3
        for s in spectra:
            assert s.region == SpectralRegion.R2

    def test_r3_region_label(self, multi_section_spectra_csv):
        """R3 section produces Spectrum with SpectralRegion.R3."""
        raw_file, data = RawSpectraFile.from_csv_path(
            multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section="R3"
        )
        scan_point_ids = [uuid.uuid4() for _ in range(3)]

        spectra = list(
            raw_file.iter_spectra(data, scan_point_ids, ProcessingLevel.RAW)
        )

        assert len(spectra) == 3
        for s in spectra:
            assert s.region == SpectralRegion.R3

    def test_all_sections_full_channel_count(self, multi_section_spectra_csv):
        """Each section returns full channel count (all 2148 channels in production)."""
        for section in ("R1", "R2", "R3"):
            _, data = RawSpectraFile.from_csv_path(
                multi_section_spectra_csv, SpectrumType.DARK_SUBTRACTED, section=section
            )
            # In our fixture it's 5 channels; in production it's 2148
            assert len(data[0]) == 5

    @pytest.fixture
    def multi_section_with_summary_rows(self, tmp_path):
        """Create CSV with 3N+3 rows (includes 3 summary rows at end)."""
        csv_path = tmp_path / "darkSubSpectra_with_summary.csv"
        lines = [
            # R1 section (2 points)
            "R1_C0,R1_C1,R1_C2",
            "10.0,11.0,12.0",
            "13.0,14.0,15.0",
            # R2 section (2 points)
            "R2_C0,R2_C1,R2_C2",
            "20.0,21.0,22.0",
            "23.0,24.0,25.0",
            # R3 section (2 points)
            "R3_C0,R3_C1,R3_C2",
            "30.0,31.0,32.0",
            "33.0,34.0,35.0",
            # Summary rows (would be present in real data as 3 extra)
            "Summary_C0,Summary_C1,Summary_C2",
            "999.0,999.0,999.0",
            "998.0,998.0,998.0",
        ]
        csv_path.write_text("\n".join(lines))
        return csv_path

    def test_summary_rows_not_included_in_sections(self, multi_section_with_summary_rows):
        """Summary rows at end of CSV do not leak into R3 data."""
        _, r3_data = RawSpectraFile.from_csv_path(
            multi_section_with_summary_rows, SpectrumType.DARK_SUBTRACTED, section="R3"
        )

        assert len(r3_data) == 2
        # Should be R3 data, not summary
        assert r3_data[0][0] == 30.0
        assert r3_data[1][0] == 33.0


class TestLoupeWorkspaceParserMultiSection:
    """Tests for LoupeWorkspaceParser.parse_spectra with R2/R3 sections (bd-13h)."""

    @pytest.fixture
    def multi_region_workspace(self, tmp_path):
        """Create a workspace with multi-section spectra CSV."""
        workspace = tmp_path / "sol_0100" / "detail_1" / "SrlcSpec_Loupe_working"
        workspace.mkdir(parents=True)

        # loupe.csv
        (workspace / "loupe.csv").write_text(
            "original_data_file,SrlcSpecSpecSohRaw_0700000000-10000-1\n"
            "human_readable_workspace,detail_1\n"
            "n_spectra,2\n"
            "n_channels,5\n"
            "laser_wavelength,248.5794\n"
            "shots_per_spec,500\n"
            "az_scale,0.628\n"
            "el_scale,0.422\n"
            "laser_x,809\n"
            "laser_y,664\n"
            "rotation,20.0\n"
            "specProcessingApplied,None\n"
            "CNDH_PCB_TEMP_STAT_REG,26.0 C\n"
            "SE_CCD_TEMP_STAT_REG,-29.0 C\n"
            "laser_shot_counter,1000000\n"
        )

        # spatial.csv
        (workspace / "spatial.csv").write_text(
            "az,el\n1041,726\n994,503\n"
            "x,y\n0.518,0.503\n0.419,0.509\n"
        )

        # darkSubSpectra.csv with R1+R2+R3
        spectra_lines = [
            "R1_C0,R1_C1,R1_C2,R1_C3,R1_C4",
            "100,110,120,130,140",
            "101,111,121,131,141",
            "R2_C0,R2_C1,R2_C2,R2_C3,R2_C4",
            "200,210,220,230,240",
            "201,211,221,231,241",
            "R3_C0,R3_C1,R3_C2,R3_C3,R3_C4",
            "300,310,320,330,340",
            "301,311,321,331,341",
        ]
        (workspace / "darkSubSpectra.csv").write_text("\n".join(spectra_lines))

        return workspace

    def test_parse_spectra_r2(self, multi_region_workspace):
        """parse_spectra with section='R2' returns R2 data."""
        parser = LoupeWorkspaceParser(multi_region_workspace, sol_number=100)
        result = parser.parse()
        scan_point_ids = [p.id for p in result.scan_points]

        spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED,
            scan_point_ids,
            ProcessingLevel.RAW,
            section="R2",
        )

        assert len(spectra) == 2
        assert all(s.region == SpectralRegion.R2 for s in spectra)
        assert spectra[0].intensity_values[0] == pytest.approx(200.0, rel=0.01)
        assert spectra[1].intensity_values[0] == pytest.approx(201.0, rel=0.01)

    def test_parse_spectra_r3(self, multi_region_workspace):
        """parse_spectra with section='R3' returns R3 data."""
        parser = LoupeWorkspaceParser(multi_region_workspace, sol_number=100)
        result = parser.parse()
        scan_point_ids = [p.id for p in result.scan_points]

        spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED,
            scan_point_ids,
            ProcessingLevel.RAW,
            section="R3",
        )

        assert len(spectra) == 2
        assert all(s.region == SpectralRegion.R3 for s in spectra)
        assert spectra[0].intensity_values[0] == pytest.approx(300.0, rel=0.01)
        assert spectra[1].intensity_values[0] == pytest.approx(301.0, rel=0.01)

    def test_parse_all_three_sections(self, multi_region_workspace):
        """All 3 sections can be parsed independently from same workspace."""
        parser = LoupeWorkspaceParser(multi_region_workspace, sol_number=100)
        result = parser.parse()
        scan_point_ids = [p.id for p in result.scan_points]

        r1_spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED, scan_point_ids, section="R1"
        )
        r2_spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED, scan_point_ids, section="R2"
        )
        r3_spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED, scan_point_ids, section="R3"
        )

        # Each section has 2 spectra (one per scan point)
        assert len(r1_spectra) == len(r2_spectra) == len(r3_spectra) == 2

        # Regions are correct
        assert all(s.region == SpectralRegion.R1 for s in r1_spectra)
        assert all(s.region == SpectralRegion.R2 for s in r2_spectra)
        assert all(s.region == SpectralRegion.R3 for s in r3_spectra)

        # Data values are distinct
        assert r1_spectra[0].intensity_values[0] == pytest.approx(100.0, rel=0.01)
        assert r2_spectra[0].intensity_values[0] == pytest.approx(200.0, rel=0.01)
        assert r3_spectra[0].intensity_values[0] == pytest.approx(300.0, rel=0.01)


class TestLoupeSessionFile:
    """Tests for LoupeSessionFile parsing."""

    @pytest.fixture
    def sample_session_file(self, tmp_path):
        """Create a sample .lpe session file."""
        sol_dir = tmp_path / "sol_0921"
        sol_dir.mkdir()
        lpe_path = sol_dir / "Sol_0921_Amherst_Point.lpe"
        content = """workspaceDictName,workspaceHumanReadableName,soffPath
SrlcSpecSpecSohRaw_0748731010-39353-1,AlGaN_1,AlGaN_1/SrlcSpecSpecSohRaw_0748731010-39353-1_Loupe_working/soff.xml
SrlcSpecSpecSohRaw_0748731411-51550-1,detail_1,detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working/soff.xml
"""
        lpe_path.write_text(content)
        return lpe_path

    def test_from_path(self, sample_session_file):
        """Parse .lpe session file."""
        session = LoupeSessionFile.from_path(sample_session_file)

        assert session.sol_number == 921
        assert len(session.entries) == 2

        e0 = session.entries[0]
        assert e0.workspace_dict_name == "SrlcSpecSpecSohRaw_0748731010-39353-1"
        assert e0.workspace_human_readable_name == "AlGaN_1"

    def test_workspace_paths(self, sample_session_file, tmp_path):
        """Get workspace directory paths."""
        # Create workspace directories
        sol_dir = sample_session_file.parent
        ws1 = sol_dir / "AlGaN_1" / "SrlcSpecSpecSohRaw_0748731010-39353-1_Loupe_working"
        ws1.mkdir(parents=True)
        (ws1 / "soff.xml").touch()

        session = LoupeSessionFile.from_path(sample_session_file)
        paths = session.workspace_paths(sol_dir)

        assert len(paths) >= 1
        assert ws1 in paths


class TestLoupeWorkspaceParser:
    """Tests for LoupeWorkspaceParser."""

    @pytest.fixture
    def sample_workspace(self, tmp_path):
        """Create a complete sample workspace."""
        workspace = tmp_path / "sol_0921" / "detail_1" / "SrlcSpec_Loupe_working"
        workspace.mkdir(parents=True)

        # loupe.csv
        (workspace / "loupe.csv").write_text("""original_data_file,SrlcSpecSpecSohRaw_0748731411-51550-1
human_readable_workspace,detail_1
n_spectra,3
n_channels,5
laser_wavelength,248.5794
shots_per_spec,500
az_scale,0.628154699
el_scale,0.422441487
laser_x,809
laser_y,664
rotation,20.6793583
specProcessingApplied,None
CNDH_PCB_TEMP_STAT_REG,26.698 C
SE_CCD_TEMP_STAT_REG,-29.059 C
laser_shot_counter,7091325
""")

        # spatial.csv
        (workspace / "spatial.csv").write_text("""az,el
1041,726
994,503
934,293
x,y
0.518,0.503
0.419,0.509
0.323,0.505
""")

        # roi.csv
        (workspace / "roi.csv").write_text("""Full Map
#ffffff
0
1
2
ENDROI
""")

        # activeSpectra.csv (minimal)
        (workspace / "activeSpectra.csv").write_text(
            "R1_C0,R1_C1,R1_C2,R1_C3,R1_C4\n"
            "100,110,120,130,140\n"
            "200,210,220,230,240\n"
            "300,310,320,330,340\n"
        )

        # img directory
        img_dir = workspace / "img"
        img_dir.mkdir()
        (img_dir / "SC3_0921_test.png").touch()

        return workspace

    def test_parse(self, sample_workspace):
        """Parse complete workspace."""
        parser = LoupeWorkspaceParser(sample_workspace, sol_number=921)
        result = parser.parse()

        assert isinstance(result, LoupeWorkspaceResult)

        # Check scan
        assert result.scan.sol_number == 921
        assert result.scan.scan_name == "detail_1"
        assert result.scan.n_points == 3

        # Check instrument state
        assert result.instrument_state.scan_id == result.scan.id
        assert result.instrument_state.ccd_temp_c == pytest.approx(-29.059, rel=0.01)

        # Check scan points
        assert len(result.scan_points) == 3
        assert result.scan_points[0].azimuth_dn == 1041

        # Check ROIs
        assert len(result.regions_of_interest) == 1
        assert result.regions_of_interest[0].name == "Full Map"

        # Check context images
        assert len(result.context_images) == 1
        assert result.context_images[0].image_type == ImageType.ACI

        # Check spectra files detected
        assert SpectrumType.ACTIVE in result.spectra_files

    def test_parse_spectra(self, sample_workspace):
        """Parse spectra from workspace."""
        parser = LoupeWorkspaceParser(sample_workspace, sol_number=921)
        result = parser.parse()

        # Get scan point IDs
        scan_point_ids = [p.id for p in result.scan_points]

        # Parse active spectra
        spectra = parser.parse_spectra(
            SpectrumType.ACTIVE,
            scan_point_ids,
            ProcessingLevel.RAW,
        )

        assert len(spectra) == 3
        assert spectra[0].spectrum_type == SpectrumType.ACTIVE
        values = spectra[0].intensity_values
        assert values[0] == pytest.approx(100.0, rel=0.01)

    def test_extract_sol_number_from_path(self, sample_workspace):
        """Sol number extracted from workspace path."""
        parser = LoupeWorkspaceParser(sample_workspace)  # No sol_number provided
        assert parser.sol_number == 921

    def test_missing_loupe_csv_raises(self, tmp_path):
        """Raise FileNotFoundError if loupe.csv missing."""
        empty_workspace = tmp_path / "empty"
        empty_workspace.mkdir()

        parser = LoupeWorkspaceParser(empty_workspace)
        with pytest.raises(FileNotFoundError):
            parser.parse()


class TestLoupeWorkspaceResult:
    """Tests for LoupeWorkspaceResult model."""

    def test_has_all_fields(self, tmp_path):
        """Result contains all expected fields."""
        scan = Scan(
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=12345,
            n_points=10,
            shots_per_point=100,
        )
        state = InstrumentState(scan_id=scan.id)
        config = CCDConfiguration(scan_id=scan.id)

        result = LoupeWorkspaceResult(
            scan=scan,
            instrument_state=state,
            ccd_configuration=config,
            workspace_path=tmp_path,
        )

        assert result.scan == scan
        assert result.instrument_state == state
        assert result.ccd_configuration == config
        assert result.scanner_calibration is None
        assert result.scan_points == []
        assert result.regions_of_interest == []
        assert result.context_images == []
        assert result.spectra_files == {}


class TestDiscoverWorkspaces:
    """Tests for discover_workspaces function."""

    def test_discover_with_lpe(self, tmp_path):
        """Discover workspaces using .lpe file."""
        sol_dir = tmp_path / "sol_0921"
        sol_dir.mkdir()

        # Create .lpe
        lpe = sol_dir / "session.lpe"
        lpe.write_text("""workspaceDictName,workspaceHumanReadableName,soffPath
SrlcSpec_0001,ws1,ws1/SrlcSpec_0001_Loupe_working/soff.xml
""")

        # Create workspace
        ws = sol_dir / "ws1" / "SrlcSpec_0001_Loupe_working"
        ws.mkdir(parents=True)
        (ws / "soff.xml").touch()

        workspaces = discover_workspaces(sol_dir)
        assert len(workspaces) >= 1

    def test_discover_without_lpe(self, tmp_path):
        """Discover workspaces by searching for Loupe_working directories."""
        sol_dir = tmp_path / "sol_0921"
        sol_dir.mkdir()

        # Create workspace without .lpe
        ws = sol_dir / "detail_1" / "SrlcSpec_Loupe_working"
        ws.mkdir(parents=True)
        (ws / "loupe.csv").touch()

        workspaces = discover_workspaces(sol_dir)
        assert len(workspaces) == 1
        assert ws in workspaces


class TestIntegrationWithRealData:
    """Integration tests with real Loupe data (if available)."""

    @pytest.fixture
    def real_workspace(self):
        """Path to real Loupe workspace (skip if not available)."""
        # This is the workspace from your sample data
        path = Path(
            "./data/loupe/sol_0921/detail_1/"
            "SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"
        )
        if not path.exists():
            pytest.skip("Real Loupe data not available")
        return path

    def test_parse_real_workspace(self, real_workspace):
        """Parse real Loupe workspace end-to-end."""
        parser = LoupeWorkspaceParser(real_workspace, sol_number=921)
        result = parser.parse()

        # Basic checks
        assert result.scan.sol_number == 921
        assert result.scan.scan_name == "detail_1"
        assert result.scan.n_points == 100

        # Check we got points
        assert len(result.scan_points) == 100

        # Check instrument state has reasonable values
        assert result.instrument_state.ccd_temp_c is not None
        assert result.instrument_state.ccd_temp_c < 0  # CCD is cooled

        # Check scanner calibration
        assert result.scanner_calibration is not None
        assert result.scanner_calibration.laser_x > 0

    def test_parse_real_spectra(self, real_workspace):
        """Parse spectra from real workspace."""
        parser = LoupeWorkspaceParser(real_workspace, sol_number=921)
        result = parser.parse()

        # Parse dark-subtracted spectra
        scan_point_ids = [p.id for p in result.scan_points]
        spectra = parser.parse_spectra(
            SpectrumType.DARK_SUBTRACTED,
            scan_point_ids,
            ProcessingLevel.NORMALIZED,
        )

        assert len(spectra) == 100

        # Check spectrum has 2148 channels
        s0 = spectra[0]
        values = s0.intensity_values
        assert len(values) == 2148
