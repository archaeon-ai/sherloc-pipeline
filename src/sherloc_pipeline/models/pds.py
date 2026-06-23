"""
PDS4 data models for SHERLOC ingestion.

This module defines Pydantic models for PDS4-specific data structures used
during ingestion of SHERLOC processed products from the PDS Geosciences Node.

Models:
    PDSProductType: Enum of PDS product type codes (rrs, rcs, rmo, etc.)
    PDSProductId: Parsed PDS4 filename components (sol, sclk, obs_id, etc.)
    PDSObservationMetadata: Metadata extracted from PDS4 XML labels
    PDSSpectralProduct: Wavelength array + R1/R2/R3 spectral data from RRS/RCS
    PDSPositionProduct: Laser shot positions + band intensities from RMO
    PDSPhotodiodeProduct: Average photodiode intensity from RLI
    PDSCalibrationRecord/Product: AlGaN calibration drift history from RCC
    PDSCrossRefRecord/Product: Shot-spectrum-image cross-reference from RLS

These models represent raw PDS4 data before transformation into PHASE domain
models (Scan, ScanPoint, Spectrum). They are used by the PDS parser (Phase 3)
and ingestion service (Phase 4).

Example:
    >>> from sherloc_pipeline.models.pds import PDSProductId, PDSProductType
    >>> pid = PDSProductId.from_filename(
    ...     "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
    ... )
    >>> pid.sol
    921
    >>> pid.product_type
    <PDSProductType.RRS: 'rrs'>
"""

import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

from pydantic import Field, field_validator, model_validator

from sherloc_pipeline.models.base import PHASEBaseModel

# --- Mission constants for validation ---

# Approximate SCLK at Perseverance landing (Feb 18, 2021).
# All SHERLOC data must have SCLK >= this value.
PDS_MISSION_SCLK_MIN: int = 666_000_000

# Expected number of CCD channels for full-frame spectral products.
PDS_EXPECTED_CHANNELS: int = 2148


class PDSProductType(str, Enum):
    """PDS4 processed product type codes.

    Each code identifies a specific data product in the SHERLOC PDS archive.
    Products come as CSV + XML label pairs.

    Core products (used in ingestion):
        RRS: Reduced Raw Spectra - laser-normalized Mars surface spectra
        RCS: Reduced Calibrated Spectra - laser-normalized calibration spectra
        RMO: Reduced Measurement Overview - positions + band intensities
        RLI: Reduced Laser Intensity - average photodiode per shot
        RCC: Reduced Calibrated Compact - AlGaN calibration fits
        RLS: Reduced Laser Shot - shot/spectrum/image cross-reference

    Band intensity products (supplementary):
        RM1-RM6: Per-band integrated intensity images
    """
    RRS = "rrs"
    RCS = "rcs"
    RMO = "rmo"
    RLI = "rli"
    RCC = "rcc"
    RLS = "rls"
    RM1 = "rm1"
    RM2 = "rm2"
    RM3 = "rm3"
    RM4 = "rm4"
    RM5 = "rm5"
    RM6 = "rm6"


# The 6 core product types used in PDS ingestion (excludes RM1-RM6 band images).
CORE_PRODUCT_TYPES: Set[PDSProductType] = {
    PDSProductType.RRS,
    PDSProductType.RCS,
    PDSProductType.RMO,
    PDSProductType.RLI,
    PDSProductType.RCC,
    PDSProductType.RLS,
}


# Regex for PDS4 SHERLOC processed product filenames.
# Format: ss__SSSS_CCCCCCCCCC_NNNxxx__DDDDDDDsrlcQQQQQ<middle>VV.ext
# The middle section (grid position + processing suffix) varies by product type.
_PDS_FILENAME_RE = re.compile(
    r"^ss__"
    r"(?P<sol>\d{4})_"
    r"(?P<sclk>\d{10})_"
    r"(?P<obs_id>\d{3})"
    r"(?P<product_type>[a-z0-9]{3})__"
    r"(?P<site_drive>\d{7})"
    r"(?P<sequence_code>srlc\d{5})"
    r"(?P<middle>.+?)"
    r"(?P<version>\d{2})"
    r"\.(?P<ext>csv|xml)$"
)


class PDSProductId(PHASEBaseModel):
    """Parsed PDS4 product filename components.

    Parses the structured SHERLOC PDS filename convention (spec s3) into
    individual fields for database storage and observation grouping.

    Filename format:
        ss__SSSS_CCCCCCCCCC_NNNxxx__DDDDDDDsrlcQQQQQ<grid><suffix>VV.ext

    Attributes:
        filename: Original filename (without directory path)
        sol: 4-digit sol number (0-padded in filename, stored as int)
        sclk: 10-digit spacecraft clock value
        obs_id: 3-digit observation ID within the sol
        product_type: 3-character product type code (rrs, rcs, rmo, etc.)
        site_drive: 7-digit Rover Motion Counter (site + drive code)
        sequence_code: SRLC sequence code (e.g., 'srlc11374')
        version: 2-digit product version number
        middle: Variable middle section (grid position + processing suffix)

    Example:
        >>> pid = PDSProductId.from_filename(
        ...     "ss__0921_0748731413_045rrs__0450000srlc11374w104cgnj01.csv"
        ... )
        >>> pid.sol
        921
        >>> pid.sclk
        748731413
        >>> pid.product_type
        <PDSProductType.RRS: 'rrs'>
        >>> pid.site_drive
        '0450000'
        >>> pid.sequence_code
        'srlc11374'
        >>> pid.version
        1
    """

    filename: str = Field(description="Original filename")
    sol: int = Field(ge=0, description="Sol number")
    sclk: int = Field(
        ge=PDS_MISSION_SCLK_MIN,
        description="Spacecraft clock value (must be within mission range)"
    )
    obs_id: str = Field(
        min_length=3, max_length=3,
        description="3-digit observation ID"
    )
    product_type: PDSProductType = Field(description="Product type code")
    site_drive: str = Field(
        min_length=7, max_length=7,
        description="7-digit Rover Motion Counter"
    )
    sequence_code: str = Field(
        pattern=r"^srlc\d{5}$",
        description="SRLC sequence code (e.g., 'srlc11374')"
    )
    version: int = Field(ge=1, description="Product version number (starts at 1)")
    middle: str = Field(
        default="",
        description="Variable middle section (grid + processing suffix)"
    )

    @classmethod
    def from_filename(cls, filename: str) -> "PDSProductId":
        """Parse a PDS4 filename into structured components.

        Args:
            filename: PDS filename (with or without directory path)

        Returns:
            Parsed PDSProductId instance

        Raises:
            ValueError: If filename does not match expected PDS format
        """
        # Strip directory path if present
        basename = Path(filename).name

        match = _PDS_FILENAME_RE.match(basename)
        if not match:
            raise ValueError(
                f"Filename does not match PDS format: {basename}"
            )

        return cls(
            filename=basename,
            sol=int(match.group("sol")),
            sclk=int(match.group("sclk")),
            obs_id=match.group("obs_id"),
            product_type=PDSProductType(match.group("product_type")),
            site_drive=match.group("site_drive"),
            sequence_code=match.group("sequence_code"),
            version=int(match.group("version")),
            middle=match.group("middle"),
        )

    @property
    def csv_filename(self) -> str:
        """Return the CSV filename (replace .xml with .csv if needed)."""
        if self.filename.endswith(".xml"):
            return self.filename[:-4] + ".csv"
        return self.filename

    @property
    def xml_filename(self) -> str:
        """Return the XML label filename (replace .csv with .xml if needed)."""
        if self.filename.endswith(".csv"):
            return self.filename[:-4] + ".xml"
        return self.filename

    @property
    def observation_key(self) -> str:
        """Unique key grouping products from the same observation.

        Products sharing sol + sclk + obs_id belong to the same observation.
        """
        return f"{self.sol:04d}_{self.sclk:010d}_{self.obs_id}"

    @property
    def is_spectral(self) -> bool:
        """True if this is a spectral product (RRS or RCS)."""
        return self.product_type in ("rrs", "rcs")

    @property
    def is_calibration(self) -> bool:
        """True if this is a calibration product (RCS or RCC)."""
        return self.product_type in ("rcs", "rcc")

    @property
    def is_core(self) -> bool:
        """True if this is a core ingestion product (not RM1-RM6)."""
        return PDSProductType(self.product_type) in CORE_PRODUCT_TYPES


class PDSObservationMetadata(PHASEBaseModel):
    """Metadata extracted from PDS4 XML labels.

    Captures the key fields from PDS4 XML labels per spec s7. These fields
    map to various database columns across sols, scans, and context images.

    Attributes:
        logical_identifier: Full PDS LIDVID (e.g., 'urn:nasa:pds:...')
        version_id: Product version from XML (e.g., '1.0')
        title: Product title from XML label
        sol_number: Mars sol number
        spacecraft_clock_start: SCLK start (with fractional seconds)
        spacecraft_clock_stop: SCLK stop (with fractional seconds)
        start_date_time: UTC start time
        stop_date_time: UTC stop time
        solar_longitude: Ls in degrees (Mars orbital position)
        mission_phase_name: Mission phase (e.g., 'Surface Mission')
        local_mean_solar_time: Local mean solar time string
        sequence_id: SRLC sequence code from Command_Execution
        site: Site index from Rover Motion Counter
        drive: Drive index from Rover Motion Counter
        rsm_azimuth_rad: RSM azimuth (FINAL-RESOLVER) in radians
        rsm_elevation_rad: RSM elevation (FINAL-RESOLVER) in radians
        software_version: iSDS PGE software version
        product_completion_status: Telemetry completion status
        n_spectra: Number of spectra rows (from Table_Delimited records)
        n_channels: Number of CCD channels (from Group repetitions)
        product_id: Parsed product ID (from filename)
        raw_label: Full XML label content as dict (for pds4_metadata JSON blob)

    Example:
        >>> meta = PDSObservationMetadata(
        ...     logical_identifier="urn:nasa:pds:mars2020_sherloc:data_processed:...",
        ...     version_id="1.0",
        ...     sol_number=921,
        ...     spacecraft_clock_start="748731411.515",
        ...     start_date_time="2023-09-23T09:09:06.711Z",
        ...     sequence_id="srlc11374",
        ...     site=45, drive=0,
        ... )
    """

    model_config = {"extra": "forbid"}

    logical_identifier: str = Field(
        min_length=1,
        description="Full PDS4 logical identifier (LIDVID)"
    )
    version_id: str = Field(
        min_length=1,
        description="Product version from XML (e.g., '1.0')"
    )
    title: Optional[str] = Field(
        default=None,
        description="Product title from XML label"
    )

    # Time coordinates
    sol_number: int = Field(ge=0, description="Mars sol number")
    spacecraft_clock_start: str = Field(
        min_length=1,
        description="SCLK start with fractional seconds (e.g., '748731411.515')"
    )
    spacecraft_clock_stop: Optional[str] = Field(
        default=None,
        description="SCLK stop with fractional seconds"
    )
    start_date_time: Optional[str] = Field(
        default=None,
        description="UTC start time (ISO 8601)"
    )
    stop_date_time: Optional[str] = Field(
        default=None,
        description="UTC stop time (ISO 8601)"
    )
    solar_longitude: Optional[float] = Field(
        default=None, ge=0, le=360,
        description="Solar longitude Ls in degrees"
    )
    mission_phase_name: Optional[str] = Field(
        default=None,
        description="Mission phase (e.g., 'Surface Mission')"
    )
    local_mean_solar_time: Optional[str] = Field(
        default=None,
        description="Local mean solar time string"
    )

    # Surface mission / command execution
    sequence_id: Optional[str] = Field(
        default=None,
        description="SRLC sequence code from Command_Execution"
    )

    # Rover Motion Counter (from geom:Motion_Counter)
    site: Optional[int] = Field(
        default=None, ge=0,
        description="Site index from RMC"
    )
    drive: Optional[int] = Field(
        default=None, ge=0,
        description="Drive index from RMC"
    )

    # RSM geometry (azimuth/elevation from FINAL-RESOLVER)
    rsm_azimuth_rad: Optional[float] = Field(
        default=None,
        description="RSM azimuth FINAL-RESOLVER in radians"
    )
    rsm_elevation_rad: Optional[float] = Field(
        default=None,
        description="RSM elevation FINAL-RESOLVER in radians"
    )

    # Processing info
    software_version: Optional[str] = Field(
        default=None,
        description="iSDS PGE software version (e.g., 'v1.6.8')"
    )
    product_completion_status: Optional[str] = Field(
        default=None,
        description="Telemetry completion status"
    )

    # Table structure (from File_Area_Observational)
    n_spectra: Optional[int] = Field(
        default=None, ge=0,
        description="Number of spectra rows (from XML Table_Delimited records)"
    )
    n_channels: Optional[int] = Field(
        default=None, ge=0,
        description="Number of CCD channels (from XML Group repetitions)"
    )

    # Parsed product ID
    product_id: Optional[PDSProductId] = Field(
        default=None,
        description="Parsed filename components"
    )

    # Raw label for full pds4_metadata JSON storage
    raw_label: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Selected XML label fields as dict for pds4_metadata blob"
    )

    @field_validator("spacecraft_clock_start")
    @classmethod
    def validate_sclk_start(cls, v: str) -> str:
        """Validate SCLK start is a parseable numeric string in mission range."""
        try:
            sclk_int = int(float(v))
        except (ValueError, OverflowError):
            raise ValueError(
                f"spacecraft_clock_start must be a numeric string, got '{v}'"
            )
        if sclk_int < PDS_MISSION_SCLK_MIN:
            raise ValueError(
                f"spacecraft_clock_start SCLK {sclk_int} is below mission "
                f"range minimum {PDS_MISSION_SCLK_MIN}"
            )
        return v

    @field_validator("spacecraft_clock_stop")
    @classmethod
    def validate_sclk_stop(cls, v: Optional[str]) -> Optional[str]:
        """Validate SCLK stop is a parseable numeric string in mission range."""
        if v is None:
            return v
        try:
            sclk_int = int(float(v))
        except (ValueError, OverflowError):
            raise ValueError(
                f"spacecraft_clock_stop must be a numeric string, got '{v}'"
            )
        if sclk_int < PDS_MISSION_SCLK_MIN:
            raise ValueError(
                f"spacecraft_clock_stop SCLK {sclk_int} is below mission "
                f"range minimum {PDS_MISSION_SCLK_MIN}"
            )
        return v

    @property
    def sclk_start_int(self) -> int:
        """Truncated SCLK start as integer (for Scan.sclk_start)."""
        return int(float(self.spacecraft_clock_start))

    @property
    def sclk_stop_int(self) -> Optional[int]:
        """Truncated SCLK stop as integer (for Scan.sclk_stop)."""
        if self.spacecraft_clock_stop is None:
            return None
        return int(float(self.spacecraft_clock_stop))

    @property
    def site_drive_str(self) -> Optional[str]:
        """7-digit site_drive string (for Scan.site_drive).

        Format: SSSDDDD (3-digit site + 4-digit drive).
        Example: site=45, drive=0 -> '0450000'
        """
        if self.site is None or self.drive is None:
            return None
        return f"{self.site:03d}{self.drive:04d}"

    @property
    def version_tuple(self) -> tuple:
        """Parse version_id into numeric tuple for comparison.

        Parses '1.0' -> (1, 0), '1.10' -> (1, 10).
        Numeric tuple comparison avoids lexicographic issues
        where '1.10' < '1.2' would be incorrect.
        """
        parts = self.version_id.split(".")
        return tuple(int(p) for p in parts)

    @property
    def earth_date(self) -> Optional[str]:
        """Extract date part from start_date_time (for Sol.earth_date)."""
        if self.start_date_time is None:
            return None
        # Parse ISO 8601 and return date portion
        return self.start_date_time[:10]

    def to_pds4_metadata_dict(self) -> Dict[str, Any]:
        """Build the pds4_metadata JSON blob for Scan storage.

        Returns dict matching the schema in spec s13.
        """
        result: Dict[str, Any] = {
            "lidvid": self.logical_identifier,
            "version": self.version_id,
        }
        if self.sequence_id is not None:
            result["sequence_id"] = self.sequence_id
        if self.site is not None and self.drive is not None:
            result["site_drive"] = self.site_drive_str
        if self.start_date_time is not None:
            result["start_utc"] = self.start_date_time
        if self.stop_date_time is not None:
            result["stop_utc"] = self.stop_date_time
        if self.local_mean_solar_time is not None:
            result["local_mean_solar_time"] = self.local_mean_solar_time
        if self.rsm_azimuth_rad is not None:
            result["rsm_azimuth_rad"] = self.rsm_azimuth_rad
        if self.rsm_elevation_rad is not None:
            result["rsm_elevation_rad"] = self.rsm_elevation_rad
        if self.site is not None:
            rmc: Dict[str, int] = {"SITE": self.site}
            if self.drive is not None:
                rmc["DRIVE"] = self.drive
            result["rover_motion_counter"] = rmc
        if self.software_version is not None:
            result["software_version"] = self.software_version
        if self.product_id is not None:
            result["processing_suffix"] = self.product_id.middle
        if self.product_completion_status is not None:
            result["product_completion"] = self.product_completion_status
        return result


class PDSSpectralProduct(PHASEBaseModel):
    """Spectral data from an RRS or RCS product.

    Represents the parsed CSV content of a laser-normalized spectral product.
    Contains a wavelength calibration array and up to three spectral regions
    (R1 Raman, R2 Fluorescence 1, R3 Fluorescence 2).

    Each region has 2148 channels per spectrum, with n_spectra rows.

    Note: Actual spectral data arrays are NOT stored in this model for memory
    efficiency. This model carries metadata; the parser yields spectra
    individually as Spectrum domain model instances.

    Attributes:
        product_id: Parsed PDS product identifier
        n_spectra: Number of spectra (rows per region)
        n_channels: Number of CCD channels (columns, typically 2148)
        wavelengths: Wavelength calibration array (2148 values in nm)
        regions_present: Which spectral regions are present (R1, R2, R3)
        source_path: Path to the source CSV file

    Example:
        >>> product = PDSSpectralProduct(
        ...     product_id=pid,
        ...     n_spectra=100,
        ...     n_channels=2148,
        ...     wavelengths=[250.123, 250.188, ...],
        ...     regions_present=["R1", "R2", "R3"],
        ... )
    """

    product_id: PDSProductId = Field(description="Parsed product identifier")
    n_spectra: int = Field(gt=0, description="Number of spectra per region")
    n_channels: int = Field(gt=0, description="Number of CCD channels")
    wavelengths: List[float] = Field(
        min_length=1,
        description="Wavelength calibration array (nm)"
    )
    regions_present: List[str] = Field(
        default_factory=lambda: ["R1", "R2", "R3"],
        description="Spectral regions present in the product"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to source CSV file"
    )

    @field_validator("n_channels")
    @classmethod
    def validate_channel_count(cls, v: int) -> int:
        """Validate channel count matches expected full CCD size (2148)."""
        if v != PDS_EXPECTED_CHANNELS:
            raise ValueError(
                f"n_channels must be {PDS_EXPECTED_CHANNELS} for full CCD "
                f"spectral products, got {v}"
            )
        return v

    @field_validator("regions_present")
    @classmethod
    def validate_regions(cls, v: List[str]) -> List[str]:
        """Validate that region names are valid."""
        valid = {"R1", "R2", "R3"}
        for region in v:
            if region not in valid:
                raise ValueError(
                    f"Invalid region '{region}', must be one of {valid}"
                )
        return v


class PDSPositionRecord(PHASEBaseModel):
    """Single laser shot position from RMO LASER_SHOT_POSITIONS table.

    Attributes:
        image_name: ACI image filename for this position
        position_index: 0-based position index
        x: X coordinate in ACI pixel space
        y: Y coordinate in ACI pixel space
    """

    image_name: str = Field(description="ACI image filename")
    position_index: int = Field(ge=0, description="0-based position index")
    x: float = Field(description="X coordinate (ACI pixels)")
    y: float = Field(description="Y coordinate (ACI pixels)")


class PDSWavelengthRegion(PHASEBaseModel):
    """Wavelength region definition from RMO WAVELENGTH_REGIONS table.

    Defines the 6 spectral bands used for integrated intensity calculations.

    Attributes:
        column_index: 0-based band index (0-5)
        wavelength_start: Band start wavelength (nm)
        wavelength_stop: Band stop wavelength (nm)
    """

    column_index: int = Field(ge=0, le=5, description="0-based band index")
    wavelength_start: float = Field(gt=0, description="Band start (nm)")
    wavelength_stop: float = Field(gt=0, description="Band stop (nm)")

    @model_validator(mode="after")
    def validate_range(self) -> "PDSWavelengthRegion":
        """Validate that stop > start."""
        if self.wavelength_stop <= self.wavelength_start:
            raise ValueError(
                f"wavelength_stop ({self.wavelength_stop}) must be > "
                f"wavelength_start ({self.wavelength_start})"
            )
        return self


class PDSPositionProduct(PHASEBaseModel):
    """Position and band intensity data from an RMO product.

    Represents the parsed CSV content of a Reduced Measurement Overview,
    which contains three tables:
    1. LASER_SHOT_POSITIONS - ACI pixel coordinates per shot
    2. WAVELENGTH_REGIONS - 6 band definitions
    3. SPECTRAL_INTENSITY - per-band intensity per position

    Survey scans may reference multiple ACI images and have duplicate
    position rows (same position_index with different Image_name).
    De-duplication by position_index keeps the first occurrence;
    all unique Image_name values are preserved separately.

    Attributes:
        product_id: Parsed PDS product identifier
        positions: De-duplicated position records
        wavelength_regions: 6 band definitions
        band_intensities: Per-position intensity values (n_positions x 6)
        image_names: All unique ACI image names (collected before de-dup)
        source_path: Path to the source CSV file

    Example:
        >>> rmo = PDSPositionProduct(
        ...     product_id=pid,
        ...     positions=[...],
        ...     wavelength_regions=[...],
        ...     band_intensities=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ...],
        ...     image_names=["SI1_0921_..._ACI.png"],
        ... )
    """

    product_id: PDSProductId = Field(description="Parsed product identifier")
    positions: List[PDSPositionRecord] = Field(
        default_factory=list,
        description="De-duplicated laser shot positions"
    )
    wavelength_regions: List[PDSWavelengthRegion] = Field(
        default_factory=list,
        description="6 wavelength band definitions"
    )
    band_intensities: List[List[float]] = Field(
        default_factory=list,
        description="Per-position band intensities (n_positions x 6)"
    )
    image_names: List[str] = Field(
        default_factory=list,
        description="All unique ACI image names (before position de-dup)"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to source CSV file"
    )

    @property
    def n_positions(self) -> int:
        """Number of unique laser shot positions."""
        return len(self.positions)


class PDSPhotodiodeProduct(PHASEBaseModel):
    """Average photodiode intensity data from an RLI product.

    RLI (Reduced Laser Intensity) CSVs contain one avg_photodiode value
    per laser shot position. Maps to ScanPoint.photodiode_mean during
    ingestion. Values of -1.0 are sentinel "no data" markers from zpz
    summary observations.

    Attributes:
        product_id: Parsed PDS product identifier
        intensities: Per-shot average photodiode ADC counts
        source_path: Path to the source CSV file
    """

    product_id: PDSProductId = Field(description="Parsed product identifier")
    intensities: List[float] = Field(
        min_length=1,
        description="Average photodiode intensity per laser shot"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to source CSV file"
    )

    @property
    def n_shots(self) -> int:
        """Number of laser shots with photodiode readings."""
        return len(self.intensities)


class PDSCalibrationRecord(PHASEBaseModel):
    """Single calibration fit record from an RCC product.

    Each row contains laser reflection and AlGaN 275 nm peak fit
    parameters for one calibration observation. Values of 0.0 for
    laser_peak_nm indicate the laser peak was not fit.

    Attributes:
        sol: Sol number for this calibration measurement
        sclk: Spacecraft clock with fractional seconds (e.g., '0675636651_555')
        laser_peak_nm: Laser reflection peak wavelength in nm (0.0 = not fit)
        laser_fwhm_nm: Laser reflection FWHM in nm
        algan_peak_nm: AlGaN 275 nm calibration peak wavelength in nm
        algan_fwhm_nm: AlGaN 275 nm calibration peak FWHM in nm
    """

    sol: int = Field(ge=0, description="Sol number")
    sclk: str = Field(min_length=1, description="SCLK with fractional seconds")
    laser_peak_nm: float = Field(ge=0, description="Laser peak wavelength (nm)")
    laser_fwhm_nm: float = Field(ge=0, description="Laser peak FWHM (nm)")
    algan_peak_nm: float = Field(ge=0, description="AlGaN peak wavelength (nm)")
    algan_fwhm_nm: float = Field(ge=0, description="AlGaN peak FWHM (nm)")


class PDSCalibrationProduct(PHASEBaseModel):
    """Multi-sol calibration drift history from an RCC product.

    RCC (Reduced Calibrated Compact) CSVs contain calibration fit
    parameters across multiple sols, tracking wavelength calibration
    drift via the AlGaN 275 nm reference target. Typically ~47 records
    spanning the mission.

    Attributes:
        product_id: Parsed PDS product identifier
        records: Calibration records ordered by sol
        source_path: Path to the source CSV file
    """

    product_id: PDSProductId = Field(description="Parsed product identifier")
    records: List[PDSCalibrationRecord] = Field(
        min_length=1,
        description="Calibration fit records ordered by sol"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to source CSV file"
    )

    @property
    def n_records(self) -> int:
        """Number of calibration records."""
        return len(self.records)

    def record_for_sol(self, sol: int) -> Optional[PDSCalibrationRecord]:
        """Find the calibration record closest to (and not exceeding) a sol.

        Returns the most recent calibration record where record.sol <= sol.
        Returns None if no records exist at or before the given sol.
        """
        best: Optional[PDSCalibrationRecord] = None
        for rec in self.records:
            if rec.sol <= sol:
                if best is None or rec.sol > best.sol:
                    best = rec
        return best


class PDSCrossRefRecord(PHASEBaseModel):
    """Single laser shot cross-reference from an RLS product.

    Maps a shot number to its associated spectral product filename,
    ACI image filename, and ACI pixel coordinates. Used for
    cross-validation of RMO position associations.

    Attributes:
        number: Shot sequence number (0-based)
        spec_name: Associated spectral product filename (e.g., RRS CSV)
        image_name: Associated ACI image filename
        samp: Sample (x) coordinate on ACI image (pixels)
        line: Line (y) coordinate on ACI image (pixels)
    """

    number: int = Field(ge=0, description="Shot sequence number (0-based)")
    spec_name: str = Field(min_length=1, description="Spectral product filename")
    image_name: str = Field(min_length=1, description="ACI image filename")
    samp: float = Field(description="Sample (x) ACI pixel coordinate")
    line: float = Field(description="Line (y) ACI pixel coordinate")


class PDSCrossRefProduct(PHASEBaseModel):
    """Shot-to-spectrum-to-image cross-reference from an RLS product.

    RLS (Reduced Laser Shot) CSVs map each laser shot to its associated
    spectral product and ACI image with pixel coordinates. This provides
    independent validation of RMO position/image associations.

    Survey scans have 2× rows (one per ACI image), unlike RMO which
    de-duplicates. RLS preserves all rows for full cross-validation.

    Attributes:
        product_id: Parsed PDS product identifier
        records: Cross-reference records ordered by shot number
        source_path: Path to the source CSV file
    """

    product_id: PDSProductId = Field(description="Parsed product identifier")
    records: List[PDSCrossRefRecord] = Field(
        min_length=1,
        description="Cross-reference records ordered by shot number"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Path to source CSV file"
    )

    @property
    def n_records(self) -> int:
        """Number of cross-reference records."""
        return len(self.records)

    @property
    def image_names(self) -> List[str]:
        """All unique ACI image names referenced, in order of first appearance."""
        seen: Dict[str, None] = {}
        for rec in self.records:
            if rec.image_name not in seen:
                seen[rec.image_name] = None
        return list(seen.keys())

    @property
    def spec_names(self) -> List[str]:
        """All unique spectral product names referenced, in order of first appearance."""
        seen: Dict[str, None] = {}
        for rec in self.records:
            if rec.spec_name not in seen:
                seen[rec.spec_name] = None
        return list(seen.keys())
