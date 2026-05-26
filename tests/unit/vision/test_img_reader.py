"""
Unit tests for SHERLOC ACI image reader.

Tests cover:
- VICAR header parsing
- Image loading and metadata extraction
- Edge cases and error handling
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import struct

from sherloc_pipeline.vision.img_reader import (
    read_aci_image,
    scan_img_files,
    get_raw_vicar_label,
    parse_vicar_label,
    ACIImageMetadata,
    ACI_WIDTH,
    ACI_HEIGHT,
    ACI_RESOLUTION_UM,
    _get_dtype_for_format,
)


# Sample VICAR header template for testing
SAMPLE_VICAR_HEADER = """LBLSIZE={lblsize}           FORMAT='BYTE'  TYPE='IMAGE'  BUFSIZ=1648  DIM=3  EOL=0  RECSIZE=1648  ORG='BSQ'  NL={nl}  NS={ns}  NB=1  N1={ns}  N2={nl}  N3=1  N4=0  NBB=0  NLB=0  HOST='JAVA'  INTFMT='HIGH'  REALFMT='RIEEE'  PROPERTY='IDENTIFICATION'  INSTRUMENT_ID='SHERLOC_ACI'  INSTRUMENT_NAME='SHERLOC AUTOFOCUS AND CONTEXT IMAGER'  IMAGE_TIME='2022-02-12T15:26:14.625'  PLANET_DAY_NUMBER=349  SOLAR_LONGITUDE=173.342  PRODUCT_ID='TEST_PRODUCT_001'  SEQUENCE_ID='srlc11360'  FRAME_TYPE='MONO'  DATA_PRODUCT_COMPRESSION_TYPE='Uncompressed'  SPACECRAFT_CLOCK_START_COUNT='697951240.076'  LOCAL_MEAN_SOLAR_TIME='Sol-00349M20:12:38.442'  """


def create_test_vicar_file(tmp_path: Path, width: int = 64, height: int = 48) -> Path:
    """Create a minimal VICAR format test file."""
    # Create header
    header_content = SAMPLE_VICAR_HEADER.format(
        lblsize=1024,  # Fixed header size
        nl=height,
        ns=width,
    )

    # Pad header to exact size
    header_bytes = header_content.encode('ascii')
    header_bytes = header_bytes.ljust(1024, b'\x00')

    # Create test image data (gradient pattern)
    image_data = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image_data[y, x] = (x + y) % 256

    # Write file
    test_file = tmp_path / "test_image.IMG"
    with open(test_file, 'wb') as f:
        f.write(header_bytes)
        f.write(image_data.tobytes())

    return test_file


class TestVICARHeaderParsing:
    """Test VICAR header parsing functionality."""

    def test_parse_basic_header(self):
        """Test parsing of basic VICAR header fields."""
        header = b"LBLSIZE=1024  FORMAT='BYTE'  NL=100  NS=200  NB=1  "
        result = parse_vicar_label(header)

        assert result['LBLSIZE'] == 1024
        assert result['FORMAT'] == 'BYTE'
        assert result['NL'] == 100
        assert result['NS'] == 200
        assert result['NB'] == 1

    def test_parse_string_fields(self):
        """Test parsing of quoted string fields."""
        header = b"INSTRUMENT_ID='SHERLOC_ACI'  PRODUCT_ID='TEST_123'  "
        result = parse_vicar_label(header)

        assert result['INSTRUMENT_ID'] == 'SHERLOC_ACI'
        assert result['PRODUCT_ID'] == 'TEST_123'

    def test_parse_timestamp(self):
        """Test parsing of timestamp fields."""
        header = b"IMAGE_TIME='2022-02-12T15:26:14.625'  START_TIME='2022-02-12T15:26:14.625'  "
        result = parse_vicar_label(header)

        assert result['IMAGE_TIME'] == '2022-02-12T15:26:14.625'
        assert result['START_TIME'] == '2022-02-12T15:26:14.625'

    def test_parse_float_fields(self):
        """Test parsing of floating point fields."""
        header = b"SOLAR_LONGITUDE=173.342  "
        result = parse_vicar_label(header)

        assert result['SOLAR_LONGITUDE'] == pytest.approx(173.342)

    def test_missing_fields_return_empty(self):
        """Test that missing fields are not in result."""
        header = b"LBLSIZE=1024  "
        result = parse_vicar_label(header)

        assert 'INSTRUMENT_ID' not in result
        assert 'PRODUCT_ID' not in result


class TestDtypeMapping:
    """Test VICAR format to numpy dtype mapping."""

    def test_byte_format(self):
        """Test BYTE format maps to uint8."""
        dtype = _get_dtype_for_format('BYTE')
        assert dtype == np.dtype(np.uint8)

    def test_half_format(self):
        """Test HALF format maps to int16."""
        dtype = _get_dtype_for_format('HALF')
        assert dtype == np.dtype(np.int16)

    def test_full_format(self):
        """Test FULL format maps to int32."""
        dtype = _get_dtype_for_format('FULL')
        assert dtype == np.dtype(np.int32)

    def test_real_format(self):
        """Test REAL format maps to float32."""
        dtype = _get_dtype_for_format('REAL')
        assert dtype == np.dtype(np.float32)

    def test_unknown_format_raises(self):
        """Test that unknown format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown VICAR format"):
            _get_dtype_for_format('UNKNOWN')


class TestACIImageMetadata:
    """Test ACIImageMetadata Pydantic model."""

    def test_create_basic_metadata(self):
        """Test creating metadata with minimal fields."""
        metadata = ACIImageMetadata(
            product_id="TEST_001",
            sol=349,
            label_size=16480,
        )

        assert metadata.product_id == "TEST_001"
        assert metadata.sol == 349
        assert metadata.label_size == 16480
        assert metadata.width == ACI_WIDTH
        assert metadata.height == ACI_HEIGHT

    def test_resolution_property(self):
        """Test resolution property returns correct value."""
        metadata = ACIImageMetadata(
            product_id="TEST",
            sol=0,
            label_size=1024,
        )

        assert metadata.resolution_um == ACI_RESOLUTION_UM

    def test_field_of_view_property(self):
        """Test field of view calculation."""
        metadata = ACIImageMetadata(
            product_id="TEST",
            sol=0,
            label_size=1024,
            width=1648,
            height=1200,
        )

        fov = metadata.field_of_view_mm
        expected_width = 1648 * ACI_RESOLUTION_UM / 1000  # ~16.64 mm
        expected_height = 1200 * ACI_RESOLUTION_UM / 1000  # ~12.12 mm

        assert fov[0] == pytest.approx(expected_width, rel=0.01)
        assert fov[1] == pytest.approx(expected_height, rel=0.01)

    def test_sol_validation(self):
        """Test that negative sol values are rejected."""
        with pytest.raises(ValueError):
            ACIImageMetadata(
                product_id="TEST",
                sol=-1,
                label_size=1024,
            )

    def test_solar_longitude_validation(self):
        """Test solar longitude bounds validation."""
        # Valid range
        metadata = ACIImageMetadata(
            product_id="TEST",
            sol=0,
            label_size=1024,
            solar_longitude=180.0,
        )
        assert metadata.solar_longitude == 180.0

        # Invalid: > 360
        with pytest.raises(ValueError):
            ACIImageMetadata(
                product_id="TEST",
                sol=0,
                label_size=1024,
                solar_longitude=400.0,
            )

    def test_serialization(self):
        """Test JSON serialization of metadata."""
        metadata = ACIImageMetadata(
            product_id="TEST_001",
            sol=349,
            label_size=16480,
            image_time=datetime(2022, 2, 12, 15, 26, 14),
        )

        data = metadata.model_dump()
        assert data['product_id'] == "TEST_001"
        assert data['sol'] == 349

        # JSON serialization
        json_str = metadata.model_dump_json()
        assert "TEST_001" in json_str


class TestReadACIImage:
    """Test the main read_aci_image function."""

    def test_read_synthetic_image(self, tmp_path):
        """Test reading a synthetic VICAR file."""
        test_file = create_test_vicar_file(tmp_path, width=64, height=48)

        image, metadata = read_aci_image(test_file, validate_dimensions=False)

        # Check image properties
        assert image.shape == (48, 64)
        assert image.dtype == np.uint8

        # Check metadata
        assert metadata.product_id == "TEST_PRODUCT_001"
        assert metadata.sol == 349
        assert metadata.sequence_id == "srlc11360"
        assert metadata.instrument_id == "SHERLOC_ACI"
        assert metadata.label_size == 1024

    def test_image_data_integrity(self, tmp_path):
        """Test that image pixel data is read correctly."""
        test_file = create_test_vicar_file(tmp_path, width=32, height=24)

        image, _ = read_aci_image(test_file, validate_dimensions=False)

        # Verify gradient pattern
        for y in range(24):
            for x in range(32):
                expected = (x + y) % 256
                assert image[y, x] == expected, f"Mismatch at ({x}, {y})"

    def test_file_not_found_raises(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_aci_image(tmp_path / "nonexistent.IMG")

    def test_invalid_vicar_raises(self, tmp_path):
        """Test that invalid VICAR file raises ValueError."""
        bad_file = tmp_path / "bad.IMG"
        bad_file.write_bytes(b"This is not a VICAR file")

        with pytest.raises(ValueError, match="Could not determine format"):
            read_aci_image(bad_file)

    def test_metadata_timestamp_parsing(self, tmp_path):
        """Test that timestamps are parsed correctly."""
        test_file = create_test_vicar_file(tmp_path)

        _, metadata = read_aci_image(test_file, validate_dimensions=False)

        assert metadata.image_time is not None
        assert metadata.image_time.year == 2022
        assert metadata.image_time.month == 2
        assert metadata.image_time.day == 12


class TestScanImgFiles:
    """Test the scan_img_files function."""

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning an empty directory."""
        result = scan_img_files(tmp_path)
        assert result == []

    def test_scan_with_img_files(self, tmp_path):
        """Test scanning directory with IMG files."""
        # Create test files
        (tmp_path / "test1.IMG").touch()
        (tmp_path / "test2.IMG").touch()
        (tmp_path / "other.txt").touch()

        result = scan_img_files(tmp_path)

        assert len(result) == 2
        assert all(p.suffix == ".IMG" for p in result)

    def test_scan_recursive(self, tmp_path):
        """Test recursive scanning."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "top.IMG").touch()
        (subdir / "nested.IMG").touch()

        # Recursive (default)
        result = scan_img_files(tmp_path, recursive=True)
        assert len(result) == 2

        # Non-recursive
        result = scan_img_files(tmp_path, recursive=False)
        assert len(result) == 1


class TestGetRawVICARLabel:
    """Test the get_raw_vicar_label function."""

    def test_get_raw_label(self, tmp_path):
        """Test extracting raw VICAR label."""
        test_file = create_test_vicar_file(tmp_path)

        label = get_raw_vicar_label(test_file)

        assert 'LBLSIZE' in label
        assert label['LBLSIZE'] == 1024
        assert label['INSTRUMENT_ID'] == 'SHERLOC_ACI'


# Skip integration tests if real data not available
@pytest.mark.skipif(
    not Path("./data/loupe").exists(),
    reason="Real ACI data not available"
)
class TestRealACIData:
    """Integration tests with real SHERLOC ACI data."""

    def test_read_real_image(self):
        """Test reading a real ACI image."""
        # Find first available image
        files = scan_img_files("./data/loupe")
        assert len(files) > 0, "No IMG files found"

        image, metadata = read_aci_image(files[0])

        # Standard ACI dimensions
        assert image.shape == (ACI_HEIGHT, ACI_WIDTH)
        assert image.dtype == np.uint8
        assert metadata.instrument_id == "SHERLOC_ACI"

    def test_all_images_have_consistent_dimensions(self):
        """Verify all ACI images have consistent dimensions."""
        files = scan_img_files("./data/loupe")

        # Sample subset for speed
        sample_files = files[:10] if len(files) > 10 else files

        for f in sample_files:
            image, metadata = read_aci_image(f)
            assert image.shape == (ACI_HEIGHT, ACI_WIDTH), f"Inconsistent dimensions in {f}"
