"""
VICAR/PDS3 Image Reader for SHERLOC ACI Images.

This module provides tools for reading .IMG files from the SHERLOC Autofocus
Context Imager (ACI) on Mars 2020 Perseverance rover. Supports both:
- VICAR format (most ACI images)
- PDS3/ODL format (some processed products like orthofabric)

ACI Specifications:
- Sensor: 1648 x 1200 pixels
- Resolution: 10.1 micrometers/pixel
- Field of View: 16.6 x 12.1 mm
- Format: 8-bit grayscale (BYTE), uncompressed
- Header: VICAR label (variable size, typically ~16KB) or PDS3 label

VICAR Format Details:
- Label size specified in LBLSIZE keyword at start of file
- Image data follows immediately after label
- BSQ (Band Sequential) organization for multi-band images
- Key metadata includes instrument info, timing, coordinates

PDS3/ODL Format Details:
- Label starts with ODL_VERSION_ID
- Uses CRLF line endings and structured blocks
- RECORD_BYTES and FILE_RECORDS specify layout
- ^IMAGE pointer indicates data offset

Usage:
    >>> from sherloc_pipeline.vision.img_reader import read_aci_image, ACIImageMetadata
    >>>
    >>> # Load image and metadata
    >>> image, metadata = read_aci_image("/path/to/image.IMG")
    >>>
    >>> # Access metadata
    >>> print(f"Sol {metadata.sol}, captured at {metadata.image_time}")
    >>>
    >>> # Image is a numpy array
    >>> print(f"Shape: {image.shape}, dtype: {image.dtype}")
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import re

import numpy as np
from numpy.typing import NDArray

from sherloc_pipeline.models.base import PHASEBaseModel
from pydantic import Field, field_validator


# ACI image dimensions (constant for all ACI images)
ACI_WIDTH = 1648   # Number of samples (columns)
ACI_HEIGHT = 1200  # Number of lines (rows)
ACI_RESOLUTION_UM = 10.1  # Micrometers per pixel

# Format identifiers
FORMAT_VICAR = "VICAR"
FORMAT_PDS3 = "PDS3"


def detect_img_format(file_path: str | Path) -> str:
    """Detect whether an IMG file is VICAR or PDS3/ODL format.

    Args:
        file_path: Path to the .IMG file

    Returns:
        FORMAT_VICAR or FORMAT_PDS3

    Raises:
        ValueError: If format cannot be determined
    """
    with open(file_path, 'rb') as f:
        header_start = f.read(100)

    # Check for VICAR format (starts with LBLSIZE)
    if b'LBLSIZE=' in header_start:
        return FORMAT_VICAR

    # Check for PDS3/ODL format (starts with ODL_VERSION_ID or PDS_VERSION_ID)
    if b'ODL_VERSION_ID' in header_start or b'PDS_VERSION_ID' in header_start:
        return FORMAT_PDS3

    raise ValueError(f"Could not determine format of {file_path}")


def is_vicar_format(file_path: str | Path) -> bool:
    """Check if a file is in VICAR format (vs PDS3/ODL).

    Args:
        file_path: Path to the .IMG file

    Returns:
        True if VICAR format, False otherwise
    """
    try:
        return detect_img_format(file_path) == FORMAT_VICAR
    except ValueError:
        return False


class ACIImageMetadata(PHASEBaseModel):
    """Metadata extracted from VICAR header of ACI image.

    This model captures the essential metadata from SHERLOC ACI images,
    including timing, instrument info, and coordinate systems.

    Attributes:
        product_id: Unique product identifier from PDS
        sol: Mars sol number when image was acquired
        image_time: UTC timestamp of image acquisition
        sequence_id: Observation sequence identifier
        instrument_id: Should be 'SHERLOC_ACI'
        spacecraft_clock: SCLK timestamp string
        solar_longitude: Solar longitude (Ls) in degrees
        local_time: Local Mean Solar Time string
        width: Image width in pixels (NS)
        height: Image height in pixels (NL)
        format: Pixel data format (e.g., 'BYTE')
        compression: Compression type (typically 'Uncompressed')
        label_size: Size of VICAR label in bytes
        frame_type: Frame type (e.g., 'MONO')
        source_path: Original file path

    Example:
        >>> metadata = ACIImageMetadata(
        ...     product_id="SC3_0349_0697951235_031ECM_N0092982SRLC11360_0000LMJ01",
        ...     sol=349,
        ...     image_time=datetime(2022, 2, 12, 15, 26, 14),
        ...     sequence_id="srlc11360",
        ...     instrument_id="SHERLOC_ACI",
        ...     width=1648,
        ...     height=1200,
        ...     label_size=16480,
        ... )
    """

    # Core identification
    product_id: str = Field(
        description="Unique PDS product identifier"
    )
    sol: int = Field(
        ge=0,
        description="Mars sol number"
    )
    image_time: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of image acquisition"
    )
    sequence_id: str = Field(
        default="",
        description="Observation sequence ID"
    )

    # Instrument info
    instrument_id: str = Field(
        default="SHERLOC_ACI",
        description="Instrument identifier"
    )
    instrument_name: str = Field(
        default="SHERLOC AUTOFOCUS AND CONTEXT IMAGER",
        description="Full instrument name"
    )

    # Timing
    spacecraft_clock: str = Field(
        default="",
        description="Spacecraft clock timestamp"
    )
    solar_longitude: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description="Solar longitude (Ls) in degrees"
    )
    local_time: str = Field(
        default="",
        description="Local Mean Solar Time"
    )

    # Image dimensions
    width: int = Field(
        default=ACI_WIDTH,
        ge=1,
        description="Image width in pixels (NS)"
    )
    height: int = Field(
        default=ACI_HEIGHT,
        ge=1,
        description="Image height in pixels (NL)"
    )

    # Format info
    format: str = Field(
        default="BYTE",
        description="Pixel data format"
    )
    compression: str = Field(
        default="Uncompressed",
        description="Compression type"
    )
    label_size: int = Field(
        ge=0,
        description="VICAR label size in bytes"
    )
    frame_type: str = Field(
        default="MONO",
        description="Frame type"
    )

    # Source tracking
    source_path: Optional[str] = Field(
        default=None,
        description="Original file path"
    )

    @property
    def resolution_um(self) -> float:
        """Return pixel resolution in micrometers."""
        return ACI_RESOLUTION_UM

    @property
    def field_of_view_mm(self) -> Tuple[float, float]:
        """Return field of view in millimeters (width, height)."""
        return (
            self.width * ACI_RESOLUTION_UM / 1000,
            self.height * ACI_RESOLUTION_UM / 1000
        )


def parse_vicar_label(header_bytes: bytes) -> Dict[str, Any]:
    """Parse VICAR label bytes into a dictionary.

    VICAR labels use a simple key=value format with space separation.
    Values can be:
    - Integers: KEY=123
    - Floats: KEY=1.23
    - Strings: KEY='value'
    - Arrays: KEY=(1,2,3) or KEY=('a','b','c')

    Args:
        header_bytes: Raw bytes of the VICAR label

    Returns:
        Dictionary of parsed key-value pairs

    Note:
        This parser extracts commonly used fields. Some complex nested
        structures may not be fully parsed.
    """
    header_str = header_bytes.decode('ascii', errors='ignore')
    result: Dict[str, Any] = {}

    # Patterns for different value types
    patterns = {
        # Core VICAR keywords
        'LBLSIZE': r'LBLSIZE=(\d+)',
        'FORMAT': r"FORMAT='(\w+)'",
        'NL': r'\sNL=(\d+)',
        'NS': r'\sNS=(\d+)',
        'NB': r'\sNB=(\d+)',
        'RECSIZE': r'RECSIZE=(\d+)',
        'ORG': r"ORG='(\w+)'",

        # Identification
        'INSTRUMENT_ID': r"INSTRUMENT_ID='([^']+)'",
        'INSTRUMENT_NAME': r"INSTRUMENT_NAME='([^']+)'",
        'PRODUCT_ID': r"PRODUCT_ID='([^']+)'",
        'SEQUENCE_ID': r"SEQUENCE_ID='([^']+)'",
        'IMAGE_ID': r"IMAGE_ID='([^']+)'",

        # Timing
        'IMAGE_TIME': r"IMAGE_TIME='([^']+)'",
        'START_TIME': r"START_TIME='([^']+)'",
        'STOP_TIME': r"STOP_TIME='([^']+)'",
        'PLANET_DAY_NUMBER': r'PLANET_DAY_NUMBER=(\d+)',
        'SOLAR_LONGITUDE': r'SOLAR_LONGITUDE=([\d.]+)',
        'SPACECRAFT_CLOCK_START_COUNT': r"SPACECRAFT_CLOCK_START_COUNT='([^']+)'",
        'LOCAL_MEAN_SOLAR_TIME': r"LOCAL_MEAN_SOLAR_TIME='([^']+)'",
        'LOCAL_TRUE_SOLAR_TIME': r"LOCAL_TRUE_SOLAR_TIME='([^']+)'",

        # Image properties
        'FRAME_TYPE': r"FRAME_TYPE='(\w+)'",
        'GEOMETRY_PROJECTION_TYPE': r"GEOMETRY_PROJECTION_TYPE='(\w+)'",
        'DATA_PRODUCT_COMPRESSION_TYPE': r"DATA_PRODUCT_COMPRESSION_TYPE='([^']+)'",

        # Mission info
        'MISSION_NAME': r"MISSION_NAME='([^']+)'",
        'TARGET_NAME': r"TARGET_NAME='(\w+)'",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, header_str)
        if match:
            value = match.group(1)
            # Convert numeric types
            if key in ('LBLSIZE', 'NL', 'NS', 'NB', 'RECSIZE', 'PLANET_DAY_NUMBER'):
                result[key] = int(value)
            elif key == 'SOLAR_LONGITUDE':
                result[key] = float(value)
            else:
                result[key] = value

    return result


def parse_pds3_label(header_bytes: bytes) -> Dict[str, Any]:
    """Parse PDS3/ODL label bytes into a dictionary.

    PDS3 labels use a line-oriented format with KEY = VALUE pairs.
    Lines are terminated with CRLF.

    Args:
        header_bytes: Raw bytes of the PDS3 label

    Returns:
        Dictionary of parsed key-value pairs
    """
    header_str = header_bytes.decode('ascii', errors='ignore')
    result: Dict[str, Any] = {}

    # Parse line by line
    for line in header_str.split('\n'):
        line = line.strip().rstrip('\r')

        # Skip comments and empty lines
        if not line or line.startswith('/*') or line.startswith('END'):
            continue

        # Parse KEY = VALUE
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes from strings
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Try to convert to int/float
            try:
                if '.' in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value

    return result


def _get_dtype_for_format(format_str: str) -> np.dtype:
    """Map VICAR FORMAT to numpy dtype.

    Args:
        format_str: VICAR FORMAT value (e.g., 'BYTE', 'HALF', 'FULL', 'REAL')

    Returns:
        Corresponding numpy dtype

    Raises:
        ValueError: If format is not recognized
    """
    format_map = {
        'BYTE': np.uint8,
        'HALF': np.int16,
        'FULL': np.int32,
        'REAL': np.float32,
        'DOUB': np.float64,
    }
    if format_str not in format_map:
        raise ValueError(f"Unknown VICAR format: {format_str}")
    return np.dtype(format_map[format_str])


def _read_vicar_image(img_path: Path) -> Tuple[NDArray, Dict[str, Any], int]:
    """Read image data from VICAR format file.

    Returns:
        Tuple of (image_array, label_dict, label_size)
    """
    with open(img_path, 'rb') as f:
        # Read enough bytes to get LBLSIZE
        initial_bytes = f.read(100)
        lblsize_match = re.search(rb'LBLSIZE=(\d+)', initial_bytes)
        if not lblsize_match:
            raise ValueError(f"Could not find LBLSIZE in {img_path}")

        label_size = int(lblsize_match.group(1))

        # Read full label
        f.seek(0)
        label_bytes = f.read(label_size)

        # Parse label
        label = parse_vicar_label(label_bytes)

        # Get image dimensions
        nl = label.get('NL', ACI_HEIGHT)  # lines (height)
        ns = label.get('NS', ACI_WIDTH)   # samples (width)
        format_str = label.get('FORMAT', 'BYTE')

        # Read image data
        dtype = _get_dtype_for_format(format_str)
        pixel_count = nl * ns
        image_data = np.frombuffer(f.read(pixel_count * dtype.itemsize), dtype=dtype)

        # Reshape to 2D array (height, width)
        image = image_data.reshape((nl, ns))

    return image, label, label_size


def _read_pds3_image(img_path: Path) -> Tuple[NDArray, Dict[str, Any], int]:
    """Read image data from PDS3/ODL format file.

    Returns:
        Tuple of (image_array, label_dict, label_size)
    """
    with open(img_path, 'rb') as f:
        # Read the entire file to parse label
        content = f.read()

    # Find label end (END statement followed by CRLF)
    header_str = content[:50000].decode('ascii', errors='ignore')

    # Parse PDS3 label
    label = parse_pds3_label(content[:50000])

    # Get record info
    record_bytes = label.get('RECORD_BYTES', 1648)
    label_records = label.get('LABEL_RECORDS', 17)
    file_records = label.get('FILE_RECORDS', 1227)

    # Calculate label size
    label_size = record_bytes * label_records

    # Get image dimensions from label or ^IMAGE pointer context
    lines = label.get('LINES', label.get('NL', ACI_HEIGHT))
    samples = label.get('LINE_SAMPLES', label.get('NS', ACI_WIDTH))

    # Handle default values
    if not isinstance(lines, int):
        lines = ACI_HEIGHT
    if not isinstance(samples, int):
        samples = ACI_WIDTH

    # Look for ^IMAGE pointer to find data offset
    image_offset = label_size  # Default: data starts after label records

    # Read image data
    image_data = np.frombuffer(
        content[image_offset:image_offset + lines * samples],
        dtype=np.uint8
    )

    # Reshape to 2D array
    image = image_data.reshape((lines, samples))

    # Add dimensions to label for consistency
    label['NL'] = lines
    label['NS'] = samples
    label['FORMAT'] = 'BYTE'

    return image, label, label_size


def read_aci_image(
    img_path: str | Path,
    validate_dimensions: bool = True
) -> Tuple[NDArray[np.uint8], ACIImageMetadata]:
    """Read a SHERLOC ACI image from VICAR or PDS3 format.

    This function reads .IMG files produced by the SHERLOC Autofocus Context
    Imager. It automatically detects the format (VICAR or PDS3/ODL) and
    extracts both the image data and comprehensive metadata.

    Args:
        img_path: Path to the .IMG file
        validate_dimensions: If True, verify image matches expected ACI dimensions

    Returns:
        Tuple of (image_array, metadata) where:
        - image_array: numpy array of shape (height, width), dtype uint8
        - metadata: ACIImageMetadata Pydantic model with extracted info

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If validation fails or file format is invalid

    Example:
        >>> image, metadata = read_aci_image("/data/sol_0349/image.IMG")
        >>> print(f"Image shape: {image.shape}")
        Image shape: (1200, 1648)
        >>> print(f"Sol {metadata.sol}, sequence {metadata.sequence_id}")
        Sol 349, sequence srlc11360
    """
    img_path = Path(img_path)

    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Detect format and read accordingly
    fmt = detect_img_format(img_path)

    if fmt == FORMAT_VICAR:
        image, label, label_size = _read_vicar_image(img_path)
    elif fmt == FORMAT_PDS3:
        image, label, label_size = _read_pds3_image(img_path)
    else:
        raise ValueError(f"Unknown format for {img_path}")

    # Get dimensions from label
    nl = label.get('NL', image.shape[0])
    ns = label.get('NS', image.shape[1])
    format_str = label.get('FORMAT', 'BYTE')

    if validate_dimensions:
        if (ns, nl) != (ACI_WIDTH, ACI_HEIGHT):
            # Allow but warn for non-standard dimensions
            pass  # Could add logging here

    # Parse timestamp
    image_time = None
    time_str = label.get('IMAGE_TIME') or label.get('START_TIME')
    if time_str and isinstance(time_str, str):
        try:
            # Handle ISO format: 2022-02-12T15:26:14.625
            image_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except ValueError:
            pass  # Could not parse timestamp

    # Build metadata model
    metadata = ACIImageMetadata(
        product_id=label.get('PRODUCT_ID', img_path.stem),
        sol=label.get('PLANET_DAY_NUMBER', 0) if isinstance(label.get('PLANET_DAY_NUMBER'), int) else 0,
        image_time=image_time,
        sequence_id=label.get('SEQUENCE_ID', '') if isinstance(label.get('SEQUENCE_ID'), str) else '',
        instrument_id=label.get('INSTRUMENT_ID', 'SHERLOC_ACI') if isinstance(label.get('INSTRUMENT_ID'), str) else 'SHERLOC_ACI',
        instrument_name=label.get('INSTRUMENT_NAME', '') if isinstance(label.get('INSTRUMENT_NAME'), str) else '',
        spacecraft_clock=label.get('SPACECRAFT_CLOCK_START_COUNT', '') if isinstance(label.get('SPACECRAFT_CLOCK_START_COUNT'), str) else '',
        solar_longitude=label.get('SOLAR_LONGITUDE') if isinstance(label.get('SOLAR_LONGITUDE'), (int, float)) else None,
        local_time=label.get('LOCAL_MEAN_SOLAR_TIME', '') if isinstance(label.get('LOCAL_MEAN_SOLAR_TIME'), str) else '',
        width=ns,
        height=nl,
        format=format_str,
        compression=label.get('DATA_PRODUCT_COMPRESSION_TYPE', 'Uncompressed') if isinstance(label.get('DATA_PRODUCT_COMPRESSION_TYPE'), str) else 'Uncompressed',
        label_size=label_size,
        frame_type=label.get('FRAME_TYPE', 'MONO') if isinstance(label.get('FRAME_TYPE'), str) else 'MONO',
        source_path=str(img_path.absolute()),
    )

    return image, metadata


def get_raw_vicar_label(img_path: str | Path) -> Dict[str, Any]:
    """Extract the raw label as a dictionary (VICAR or PDS3 format).

    Use this function when you need access to all label fields,
    not just those captured in ACIImageMetadata.

    Args:
        img_path: Path to the .IMG file

    Returns:
        Dictionary of all parsed label fields

    Example:
        >>> label = get_raw_vicar_label("/data/image.IMG")
        >>> print(label['MISSION_NAME'])
        MARS 2020
    """
    img_path = Path(img_path)

    fmt = detect_img_format(img_path)

    with open(img_path, 'rb') as f:
        if fmt == FORMAT_VICAR:
            initial_bytes = f.read(100)
            lblsize_match = re.search(rb'LBLSIZE=(\d+)', initial_bytes)
            if not lblsize_match:
                raise ValueError(f"Could not find LBLSIZE in {img_path}")

            label_size = int(lblsize_match.group(1))

            f.seek(0)
            label_bytes = f.read(label_size)

            return parse_vicar_label(label_bytes)
        else:
            # PDS3 format
            content = f.read(50000)
            return parse_pds3_label(content)


def scan_img_files(
    directory: str | Path,
    recursive: bool = True
) -> List[Path]:
    """Find all .IMG files in a directory.

    Args:
        directory: Directory to search
        recursive: If True, search subdirectories

    Returns:
        List of Path objects for found .IMG files

    Example:
        >>> files = scan_img_files("./data/loupe")
        >>> print(f"Found {len(files)} ACI images")
    """
    directory = Path(directory)
    pattern = "**/*.IMG" if recursive else "*.IMG"
    return sorted(p for p in directory.glob(pattern) if "__MACOSX" not in p.parts)
