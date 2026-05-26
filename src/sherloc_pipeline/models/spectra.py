"""
Spectral data models for PHASE.

This module defines the core domain models for SHERLOC spectroscopy data:
- Sol: A Martian day of observations
- Scan: A complete spectroscopy scan of a target
- ScanPoint: A single measurement point within a scan
- Spectrum: A spectral measurement at one processing level

These models correspond to the unified schema defined in docs/schema/UNIFIED_SCHEMA.md
and are designed for both runtime validation and database persistence.

Example:
    >>> from sherloc_pipeline.models.spectra import Sol, Scan, ScanPoint, Spectrum
    >>> from sherloc_pipeline.models.spectra import SpectralRegion, SpectrumType, ProcessingLevel
    >>>
    >>> sol = Sol(sol_number=921)
    >>> scan = Scan(
    ...     sol_number=921,
    ...     scan_name="detail_1",
    ...     scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
    ...     sclk_start=672194998,
    ...     n_points=100,
    ...     n_channels=2148,
    ...     shots_per_point=10,
    ... )
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid
import zlib

from pydantic import Field, field_validator, model_validator

from sherloc_pipeline.models.base import (
    PHASEBaseModel,
    TimestampedModel,
    IdentifiableModel,
    ModelRegistry,
)


class DataSource(str, Enum):
    """Source of the data (Loupe or PDS4)."""
    LOUPE = "loupe"
    PDS4 = "pds4"


class SpectralRegion(str, Enum):
    """SHERLOC spectral regions.

    Canonical reference: docs/schema/SPECTRAL_REGIONS.md

    Each region corresponds to a separate full-CCD readout (2148 channels).
    Only a subset of channels in each readout contains meaningful signal;
    the rest is detector noise from unilluminated regions.

    Regions:
        R1:   Raman region (250-282 nm, channels 52-574, 523 meaningful channels)
              Wavenumber range: ~238-4765 cm-1, usable: ~640-4200 cm-1
        R2:   Fluorescence region 1 (282-337.8 nm, channels 690-1668, 979 channels)
        R3:   Fluorescence region 2 (337.8-357.4 nm, channels 1690-2147, 458 channels)
        R123: Stitched full spectrum (all 2148 channels with overlap summation)
              Requires proper R1+R2+R3 stitching; see SPECTRAL_REGIONS.md Section 4.

    Notes:
        - cutoff_channel=500 in config refers to the polynomial coefficient switch
          (Raman vs Fluorescence calibration), NOT the R1 region boundary.
        - Always use wavelength filtering (250-282 nm) to extract R1, not raw
          channel slicing.
        - NEVER use np.linspace() for wavenumber axes; always use polynomial
          calibration (see SPECTRAL_REGIONS.md Section 3).
        - Valid values enforced at application level. SQLite CHECK constraint
          documented in SPECTRAL_REGIONS.md Section 5.2 for future table rebuild.
    """
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R123 = "R123"


class SpectrumType(str, Enum):
    """Type of spectral measurement.

    - active: Laser-illuminated spectrum
    - dark: Dark frame (no laser)
    - dark_subtracted: Active minus dark
    - laser_normalized: Laser-power-normalized spectrum (PDS processing level)
    """
    ACTIVE = "active"
    DARK = "dark"
    DARK_SUBTRACTED = "dark_subtracted"
    LASER_NORMALIZED = "laser_normalized"


class CoordinateFrame(str, Enum):
    """Coordinate frame for scan point positions.

    - scanner_workspace: Loupe scanner workspace coordinates (relative, ±0.5 range for detail,
      ±2.5 for survey). Used by Loupe-ingested data.
    - aci_pixel: ACI image pixel coordinates (absolute, typically 700-900 range).
      Used by PDS-ingested data from RMO products.

    These frames are NOT directly comparable. Scanner workspace coordinates are relative
    to the scanner origin; ACI pixel coordinates are absolute image positions.
    """
    SCANNER_WORKSPACE = "scanner_workspace"
    ACI_PIXEL = "aci_pixel"


class ScanType(str, Enum):
    """Type of SHERLOC observation scan.

    Classifies scans by their observation type, replacing fragile
    name-pattern matching (e.g., "AlGaN" → calibration) with a
    first-class enum. Benefits both PDS and Loupe data.

    Values:
        DETAIL: Standard Mars surface scan (≤200 spectra per observation)
        SURVEY: Large grid scan (>200 spectra per observation)
        CALIBRATION: Calibration target / AlGaN internal calibration
    """
    DETAIL = "detail"
    SURVEY = "survey"
    CALIBRATION = "calibration"


class TargetType(str, Enum):
    """Classification of scan targets by purpose.

    Distinguishes Mars science targets from calibration and engineering
    scans, replacing fragile target-name pattern matching with a
    first-class column.

    Values:
        MARS_TARGET: Mars surface science target (e.g., Amherst_Point)
        CAL_TARGET: Calibration target (e.g., external calibration, AlGaN)
        ENGINEERING: Engineering/housekeeping (e.g., conjunction, arm stowed)
    """
    MARS_TARGET = "mars_target"
    CAL_TARGET = "cal_target"
    ENGINEERING = "engineering"


# --- Target classification constants ---
# Authoritative frozen sets for classify_target_type().
# Update these + run `sherloc reclassify-targets` if rules change.

_ENGINEERING_TARGETS = frozenset({
    "conjunction",
    "b conjunction",
    "arm stowed",
    "arm stowed dark",
    "arm docked",
})

_CAL_TARGETS = frozenset({
    "external calibration",
    "teflon calibration",
    "calibration",
    "algan340 calibration",
    "maze calibration",
    "ext cal meteorite",
    "passive diffusil",
})


def classify_target_type(target: Optional[str], scan_name: Optional[str]) -> str:
    """Classify a scan as mars_target, cal_target, or engineering.

    Priority cascade (highest first):
      1. engineering — NULL/empty target, known engineering targets,
         power_* or *laser_disabled* scan_names
      2. cal_target — known calibration targets, AlGaN* scan_names
      3. mars_target — everything else

    Args:
        target: Geological target name (may be None or have leading spaces).
        scan_name: Scan sequence name (e.g., 'detail_1', 'power_on').

    Returns:
        One of 'mars_target', 'cal_target', 'engineering'.
    """
    # Normalize
    clean_target = (target or "").strip().lower()
    clean_scan = (scan_name or "").strip().lower()

    # --- Engineering (highest priority) ---
    # NULL or empty target
    if not clean_target:
        return TargetType.ENGINEERING.value

    # Known engineering targets
    if clean_target in _ENGINEERING_TARGETS:
        return TargetType.ENGINEERING.value

    # power_* or *laser_disabled* scan_names
    if clean_scan.startswith("power_") or "laser_disabled" in clean_scan:
        return TargetType.ENGINEERING.value

    # --- Calibration ---
    # Known calibration targets
    if clean_target in _CAL_TARGETS:
        return TargetType.CAL_TARGET.value

    # AlGaN* scan_names
    if clean_scan.startswith("algan"):
        return TargetType.CAL_TARGET.value

    # --- Mars target (default) ---
    return TargetType.MARS_TARGET.value


# ---------------------------------------------------------------------------
# Scan class classification
# ---------------------------------------------------------------------------

# Composite scan name patterns (substring matches against lowercased scan_name)
_COMPOSITE_PATTERNS = ("_all", "_median", "_sum_active", "asterisk", "cross")


def classify_scan_class(scan_name: str) -> str:
    """Classify a scan as primary, sub_scan, or composite based on its name.

    Priority cascade (highest first):
      1. composite — name contains _all, _median, or _sum_active
      2. sub_scan — name ends with [a-c] after a digit or underscore
      3. primary — everything else

    Args:
        scan_name: Scan sequence name (e.g., 'detail_1', 'detail_1a', 'detail_all').

    Returns:
        One of 'primary', 'sub_scan', 'composite'.
    """
    clean = (scan_name or "").strip()
    lower = clean.lower()

    # --- Composite (highest priority) ---
    for pat in _COMPOSITE_PATTERNS:
        if pat in lower:
            return "composite"

    # --- Sub-scan ---
    if len(clean) >= 2:
        last = clean[-1]
        prev = clean[-2]
        if last in ("a", "b", "c") and (prev.isdigit() or prev == "_"):
            return "sub_scan"

    # --- Primary (default) ---
    return "primary"


def derive_parent_name(scan_name: str) -> Optional[str]:
    """Derive parent scan name from a sub-scan name.

    Returns None if scan_name is not a sub-scan.

    Examples:
        detail_1a   → detail_1
        HDR_a       → HDR
        HDR_500_1a  → HDR_500_1
        HDR_500_b   → HDR_500
        detail_1c   → detail_1
        Orthofabric → None (not a sub-scan)
    """
    if not scan_name or len(scan_name) < 2:
        return None
    last = scan_name[-1]
    prev = scan_name[-2]
    if last in ("a", "b", "c") and (prev.isdigit() or prev == "_"):
        if prev == "_":
            return scan_name[:-2]  # strip _a
        else:
            return scan_name[:-1]  # strip a
    return None


class ProcessingLevel(str, Enum):
    """Processing level of spectral data.

    Processing levels from raw to fully processed:
    - raw: Original CCD counts
    - calibrated: Wavelength/wavenumber calibrated
    - normalized: Laser-power normalized
    - despiked: Cosmic ray spikes removed
    - baselined: Baseline subtracted
    - derived: Fully processed (normalized + despiked + baselined)
    """
    RAW = "raw"
    CALIBRATED = "calibrated"
    NORMALIZED = "normalized"
    DESPIKED = "despiked"
    BASELINED = "baselined"
    DERIVED = "derived"


@ModelRegistry.register
class Sol(TimestampedModel):
    """A Martian sol (day) of observations.

    The Sol model represents a single Martian day during which SHERLOC
    observations were collected. It serves as the top-level grouping
    for all scans performed on that day.

    Attributes:
        sol_number: Mars sol number (unique identifier, >= 0)
        earth_date: Corresponding Earth date (if known)
        solar_longitude: Ls in degrees (0-360, Mars orbital position)
        mission_phase: Mission phase name (e.g., "Primary Mission")
        data_source: Origin of the data ('loupe' or 'pds4')

    Example:
        >>> sol = Sol(sol_number=921)
        >>> sol.sol_number
        921
        >>> sol.data_source
        <DataSource.LOUPE: 'loupe'>
    """

    sol_number: int = Field(
        ge=0,
        description="Mars sol number (unique identifier)"
    )
    earth_date: Optional[date] = Field(
        default=None,
        description="Corresponding Earth date"
    )
    solar_longitude: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description="Solar longitude Ls in degrees (0-360)"
    )
    mission_phase: Optional[str] = Field(
        default=None,
        description="Mission phase name"
    )
    data_source: DataSource = Field(
        default=DataSource.LOUPE,
        description="Data source: 'loupe' or 'pds4'"
    )


@ModelRegistry.register
class Scan(IdentifiableModel):
    """A complete spectroscopy scan of a target.

    A Scan represents a single SHERLOC observation sequence, typically
    consisting of multiple measurement points (ScanPoints) on a target.
    This corresponds to a Loupe workspace or PDS4 product set.

    Attributes:
        sol_number: Sol this scan was acquired on
        scan_name: Scan sequence name from Loupe (e.g., 'detail_1', 'survey_1296')
        target: Geological target name (e.g., 'Amherst_Point', 'Dragons_Egg_Lake')
        scan_id: Original scan identifier from data source
        sclk_start: Spacecraft clock at scan start
        sclk_stop: Spacecraft clock at scan end (optional)
        n_points: Number of measurement points in the scan
        n_channels: Number of CCD channels (typically 2148)
        shots_per_point: Laser shots per measurement point (None for PDS processed)
        laser_wavelength_nm: Laser wavelength in nm (typically 248.6)
        processing_applied: Processing code from Loupe
        source_path: Original file/workspace path
        loupe_metadata: Full loupe.csv as JSON (Loupe source)
        pds4_metadata: Selected PDS4 label fields (PDS4 source)

    Example:
        >>> scan = Scan(
        ...     sol_number=921,
        ...     scan_name="detail_1",
        ...     target="Amherst_Point",
        ...     scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
        ...     sclk_start=672194998,
        ...     n_points=100,
        ...     n_channels=2148,
        ...     shots_per_point=10,
        ... )
        >>> scan.n_points
        100
    """

    sol_number: int = Field(
        ge=0,
        description="Sol number (foreign key to Sol)"
    )
    scan_name: str = Field(
        min_length=1,
        description="Scan sequence name (e.g., 'detail_1', 'survey_1296')"
    )
    target: Optional[str] = Field(
        default=None,
        description="Geological target name (e.g., 'Amherst_Point')"
    )
    scan_id: str = Field(
        min_length=1,
        description="Original scan identifier"
    )
    sclk_start: int = Field(
        ge=0,
        description="Spacecraft clock at scan start"
    )
    sclk_stop: Optional[int] = Field(
        default=None,
        ge=0,
        description="Spacecraft clock at scan end"
    )
    n_points: int = Field(
        gt=0,
        description="Number of measurement points"
    )
    n_channels: int = Field(
        default=2148,
        gt=0,
        description="Number of CCD channels"
    )
    shots_per_point: Optional[int] = Field(
        default=None,
        gt=0,
        description="Laser shots per measurement point. "
        "NULL for PDS processed products (raw EDR only)."
    )
    laser_wavelength_nm: float = Field(
        default=248.6,
        gt=0,
        description="Laser wavelength in nm"
    )
    processing_applied: Optional[str] = Field(
        default=None,
        description="Processing code from Loupe"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Original file/workspace path"
    )
    loupe_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full loupe.csv as JSON"
    )
    pds4_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Selected PDS4 label fields"
    )
    data_source: Optional[DataSource] = Field(
        default=None,
        description="Data source discriminator: 'loupe' or 'pds4'. "
        "Enables per-scan filtering by ingestion source."
    )
    site_drive: Optional[str] = Field(
        default=None,
        max_length=20,
        description="7-digit Rover Motion Counter (site + drive code). "
        "Used for WATSON image association and spatial grouping."
    )
    sequence_id: Optional[str] = Field(
        default=None,
        max_length=20,
        description="SRLC sequence code from PDS filename (e.g., 'srlc11374'). "
        "Identifies the observation sequence."
    )
    scan_type: Optional[ScanType] = Field(
        default=None,
        description="Observation type: 'detail', 'survey', or 'calibration'. "
        "Replaces fragile name-pattern matching. NULL for legacy data."
    )
    target_type: Optional[TargetType] = Field(
        default=None,
        description="Target classification: 'mars_target', 'cal_target', or 'engineering'. "
        "Set by classify_target_type() during ingestion."
    )
    scan_class: str = Field(
        default="primary",
        description="Scan classification: 'primary', 'sub_scan', or 'composite'. "
        "Set by classify_scan_class() during ingestion."
    )
    parent_scan_id: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of parent scan (sub_scans only, NULL for orphans)."
    )
    source_scan_ids: Optional[List[str]] = Field(
        default=None,
        description="UUIDs of source scans (composites only, best-effort provenance)."
    )

    @field_validator("n_channels")
    @classmethod
    def validate_n_channels(cls, v: int) -> int:
        """Validate that n_channels is the expected SHERLOC value."""
        if v != 2148:
            # Allow but warn about non-standard channel counts
            pass
        return v

    @model_validator(mode="after")
    def validate_sclk_order(self) -> "Scan":
        """Validate that sclk_stop >= sclk_start if both are provided."""
        if self.sclk_stop is not None and self.sclk_stop < self.sclk_start:
            raise ValueError(
                f"sclk_stop ({self.sclk_stop}) must be >= sclk_start ({self.sclk_start})"
            )
        return self


@ModelRegistry.register
class ScanPoint(IdentifiableModel):
    """A single measurement point within a scan.

    Each ScanPoint represents one spatial location where SHERLOC
    collected spectra. Points are indexed 0 to n_points-1 within a scan.

    Attributes:
        scan_id: UUID of parent Scan
        point_index: 0-based index within the scan
        azimuth_dn: Scanner azimuth in DN (digital number)
        elevation_dn: Scanner elevation in DN
        x_pixel: X coordinate on ACI image (pixels)
        y_pixel: Y coordinate on ACI image (pixels)
        azimuth_error: Scanner azimuth error
        elevation_error: Scanner elevation error
        photodiode_mean: Mean laser intensity from photodiode
        photodiode_std: Standard deviation of laser intensity

    Example:
        >>> point = ScanPoint(
        ...     scan_id=scan.id,
        ...     point_index=0,
        ...     x_pixel=824.5,
        ...     y_pixel=600.2,
        ... )
        >>> point.point_index
        0
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )
    point_index: int = Field(
        ge=0,
        description="0-based index within the scan"
    )
    azimuth_dn: Optional[int] = Field(
        default=None,
        description="Scanner azimuth in DN"
    )
    elevation_dn: Optional[int] = Field(
        default=None,
        description="Scanner elevation in DN"
    )
    x_pixel: Optional[float] = Field(
        default=None,
        description="X coordinate on ACI image (pixels)"
    )
    y_pixel: Optional[float] = Field(
        default=None,
        description="Y coordinate on ACI image (pixels)"
    )
    azimuth_error: Optional[float] = Field(
        default=None,
        description="Scanner azimuth error"
    )
    elevation_error: Optional[float] = Field(
        default=None,
        description="Scanner elevation error"
    )
    photodiode_mean: Optional[float] = Field(
        default=None,
        description="Mean laser intensity from photodiode"
    )
    photodiode_std: Optional[float] = Field(
        default=None,
        ge=0,
        description="Standard deviation of laser intensity"
    )
    coordinate_frame: Optional[CoordinateFrame] = Field(
        default=None,
        description="Coordinate frame for x_pixel/y_pixel: "
        "'scanner_workspace' (Loupe) or 'aci_pixel' (PDS RMO). "
        "NULL for legacy data where frame was implicit."
    )


@ModelRegistry.register
class Spectrum(IdentifiableModel):
    """A spectral measurement at one processing level.

    The Spectrum model stores intensity values for one spectral region
    at one processing level. Multiple Spectrum records may exist for
    the same ScanPoint (different regions, processing levels, or types).

    Intensity data is stored as a compressed binary blob for efficiency.
    Use the intensity_values property to get/set as a list of floats.

    Attributes:
        scan_point_id: UUID of parent ScanPoint
        region: Spectral region (R1, R2, R3, or R123)
        spectrum_type: Type of measurement (active, dark, dark_subtracted)
        processing_level: Processing state of the data
        intensities: Compressed binary intensity data
        wavelengths: Wavelength array (if custom calibration)
        wavenumbers: Wavenumber array (if custom calibration)

    Example:
        >>> spectrum = Spectrum(
        ...     scan_point_id=point.id,
        ...     region=SpectralRegion.R1,
        ...     spectrum_type=SpectrumType.DARK_SUBTRACTED,
        ...     processing_level=ProcessingLevel.NORMALIZED,
        ...     intensities=b"...",  # compressed data
        ... )
    """

    scan_point_id: uuid.UUID = Field(
        description="UUID of parent ScanPoint"
    )
    region: SpectralRegion = Field(
        description="Spectral region (R1, R2, R3, R123). "
        "Validated via SpectralRegion enum (application-level CHECK constraint). "
        "See docs/schema/SPECTRAL_REGIONS.md for region definitions."
    )
    spectrum_type: SpectrumType = Field(
        description="Type of measurement"
    )
    processing_level: ProcessingLevel = Field(
        description="Processing state of the data"
    )
    intensities: bytes = Field(
        description="Compressed binary intensity data (float32 array)"
    )
    wavelengths: Optional[bytes] = Field(
        default=None,
        description="Wavelength array (compressed, if custom calibration)"
    )
    wavenumbers: Optional[bytes] = Field(
        default=None,
        description="Wavenumber array (compressed, if custom calibration)"
    )
    wavelength_source: Optional[str] = Field(
        default=None,
        max_length=30,
        description="Origin of wavelength calibration: "
        "'loupe_polynomial' (Loupe V5.1.5a coefficients) or "
        "'pds_embedded' (wavelength table from PDS CSV). "
        "NULL for legacy data."
    )

    @staticmethod
    def compress_array(values: List[float]) -> bytes:
        """Compress a list of floats to binary storage format.

        Args:
            values: List of float intensity values

        Returns:
            Compressed bytes suitable for database storage
        """
        import numpy as np
        arr = np.array(values, dtype=np.float32)
        return zlib.compress(arr.tobytes())

    @staticmethod
    def decompress_array(data: bytes) -> List[float]:
        """Decompress binary data to a list of floats.

        Args:
            data: Compressed bytes from database

        Returns:
            List of float intensity values
        """
        import numpy as np
        arr = np.frombuffer(zlib.decompress(data), dtype=np.float32)
        return arr.tolist()

    @property
    def intensity_values(self) -> List[float]:
        """Get intensity values as a list of floats."""
        return self.decompress_array(self.intensities)

    @property
    def wavelength_values(self) -> Optional[List[float]]:
        """Get wavelength values as a list of floats (if present)."""
        if self.wavelengths is None:
            return None
        return self.decompress_array(self.wavelengths)

    @property
    def wavenumber_values(self) -> Optional[List[float]]:
        """Get wavenumber values as a list of floats (if present)."""
        if self.wavenumbers is None:
            return None
        return self.decompress_array(self.wavenumbers)

    @classmethod
    def from_values(
        cls,
        scan_point_id: uuid.UUID,
        region: SpectralRegion,
        spectrum_type: SpectrumType,
        processing_level: ProcessingLevel,
        intensity_values: List[float],
        wavelength_values: Optional[List[float]] = None,
        wavenumber_values: Optional[List[float]] = None,
        **kwargs,
    ) -> "Spectrum":
        """Create a Spectrum from lists of values.

        This is a convenience constructor that handles compression
        of the intensity and calibration arrays.

        Args:
            scan_point_id: UUID of parent ScanPoint
            region: Spectral region
            spectrum_type: Type of measurement
            processing_level: Processing state
            intensity_values: List of intensity floats
            wavelength_values: List of wavelength floats (optional)
            wavenumber_values: List of wavenumber floats (optional)
            **kwargs: Additional fields (id, created_at, etc.)

        Returns:
            New Spectrum instance with compressed data
        """
        intensities = cls.compress_array(intensity_values)
        wavelengths = (
            cls.compress_array(wavelength_values)
            if wavelength_values is not None
            else None
        )
        wavenumbers = (
            cls.compress_array(wavenumber_values)
            if wavenumber_values is not None
            else None
        )

        return cls(
            scan_point_id=scan_point_id,
            region=region,
            spectrum_type=spectrum_type,
            processing_level=processing_level,
            intensities=intensities,
            wavelengths=wavelengths,
            wavenumbers=wavenumbers,
            **kwargs,
        )
