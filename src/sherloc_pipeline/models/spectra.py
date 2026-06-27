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
import re
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
    """Type of SHERLOC observation scan (the *kind* / geometry axis).

    Classifies scans by their observation type, replacing fragile
    name-pattern matching (e.g., "AlGaN" → calibration) with a
    first-class enum. Benefits both PDS and Loupe data.

    Per WS-1 (scan-classification), ``scan_type`` is **name-authoritative**:
    derived from the scan name via :func:`classify_scan_type`, with the
    spectrum-count rule used only as a fallback for explicitly-uninformative
    names. ``LINE`` and ``HDR`` were previously unrepresentable, which forced
    every ``line``/``HDR`` scan to be mislabeled ``detail``/``survey``.

    Values:
        DETAIL: Standard Mars surface scan
        SURVEY: Large grid / areal-coverage scan
        LINE: 1-D traverse scan (constituent of cross/asterisk composites)
        HDR: High-dynamic-range scan
        CALIBRATION: Calibration target / AlGaN internal calibration
    """
    DETAIL = "detail"
    SURVEY = "survey"
    LINE = "line"
    HDR = "HDR"
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
      1. composite — name contains _all, _median, _sum_active, asterisk, or
         cross, OR is a bare trailing-underscore union (e.g. 'detail_',
         'line_')
      2. sub_scan — name ends with [a-c] after a digit or underscore
      3. primary — everything else

    The bare trailing-underscore union (WS-1 spec §4.3, defect D2) is a
    name-union composite that the substring patterns above miss — e.g.
    ``line_`` (Aitkenodden) and ``detail_`` were stored ``primary`` at the
    source, which is why PHASE needed the ``COMPOSITE_BY_NAME_PRIMARIES``
    consumer band-aid.

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
    # Bare trailing-underscore union (e.g. 'detail_', 'line_').
    if clean.endswith("_"):
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


# ---------------------------------------------------------------------------
# Scan type classification (name-authoritative; WS-1 spec §4.2)
# ---------------------------------------------------------------------------

# Sentinel returned by classify_scan_type() for an informative-but-unrecognized
# name (e.g. a future DO_AREA_* pattern). The caller MUST NOT write a guessed
# scan_type — the scan is quarantined (left NULL) for manual / role-expansion
# review, guarding against silently mis-typing a new acquisition kind.
SCAN_TYPE_QUARANTINE = "quarantine"

# Calibration sequence codes (SRLC). Keyed FIRST and unshadowable by name.
# Mirrors core.pds_parsers._CALIBRATION_SEQUENCE_CODES.
_CALIBRATION_SEQUENCE_CODES = frozenset({"srlc10000", "srlc16000"})

# Spectrum-count threshold separating survey from detail. Used ONLY as a
# fallback for explicitly-uninformative names (the count rule is never the
# authority when the name carries a kind signal).
_SURVEY_SPECTRA_THRESHOLD = 200

# Token-boundaried PREFIX rules (the `survey*` / `detail*` / `line*` map
# entries), highest precedence first. Matching is case-normalized: the token
# must be followed by a non-alphabetic character (or end of name), so
# "detailed_center" does NOT match "detail".
_SCAN_TYPE_PREFIX_RULES = (
    ("survey", ScanType.SURVEY),
    ("detail", ScanType.DETAIL),
    ("line", ScanType.LINE),
)

# HDR is the `*HDR*` map entry (ARC-M2P-308): a token-boundaried match
# ANYWHERE in the name (not just a prefix), at the LOWEST precedence so
# "survey_HDR" still resolves to survey. The boundary guard means "hydration"
# does not match (no standalone `hdr` token).
_HDR_TOKEN_RE = re.compile(r"(?:^|[^a-z])hdr(?:[^a-z]|$)")

# Named composite groupings whose scan_type INHERITS the constituent kind
# (Key Decision K1). cross / asterisk are unions of `line` primaries.
_INHERITED_LINE_NAMES = frozenset({"cross", "asterisk"})

# Calibration scan-name prefixes — Loupe targets carry no SRLC sequence code,
# so AlGaN internal-calibration scans are name-identified.
_CALIBRATION_NAME_PREFIXES = ("algan",)


def _scan_type_from_name(scan_name: Optional[str]) -> Optional[ScanType]:
    """Return the name-implied ScanType, or None if the name carries no
    recognized kind token. Case-normalized, token-boundaried, ordered.

    Precedence: survey > detail > line (token-boundaried prefixes) >
    cross/asterisk (inherit line) > HDR (`*HDR*`, token-boundaried anywhere).
    """
    low = (scan_name or "").strip().lower()
    if not low:
        return None
    for token, scan_type in _SCAN_TYPE_PREFIX_RULES:
        if low.startswith(token):
            rest = low[len(token):]
            if not rest or not rest[0].isalpha():
                return scan_type
    if low in _INHERITED_LINE_NAMES:
        return ScanType.LINE
    if _HDR_TOKEN_RE.search(low):
        return ScanType.HDR
    return None


def _is_calibration_name(scan_name: Optional[str]) -> bool:
    """True if the scan name identifies an internal-calibration scan (AlGaN)."""
    low = (scan_name or "").strip().lower()
    return any(low.startswith(p) for p in _CALIBRATION_NAME_PREFIXES)


def _is_uninformative_name(scan_name: Optional[str]) -> bool:
    """True for explicitly-enumerated empty / opaque name forms that carry no
    kind signal — the ONLY class for which the spectrum-count fallback is
    permitted.

    Qualifying forms: empty / None, and the synthetic PDS observation name
    ``pds_<sol>_<sclk>_<obs>`` (PDS products carry no human-readable scan
    name). Every other non-empty name is treated as informative.
    """
    if scan_name is None:
        return True
    low = scan_name.strip().lower()
    if not low:
        return True
    if low.startswith("pds_"):
        return True
    return False


def classify_scan_type(
    scan_name: Optional[str],
    sequence_code: Optional[str] = None,
    n_spectra: Optional[int] = None,
) -> "ScanType | str":
    """Name-authoritative scan-type resolver (WS-1 spec §4.2).

    Resolution order:
      1. **Calibration** — sequence code in {SRLC10000, SRLC16000} (first,
         unshadowable by name); or a calibration scan-name prefix (AlGaN).
      2. **RECOGNIZED name** — token-boundaried ordered map
         (survey > detail > line > HDR; cross/asterisk inherit line per K1).
      3. **UNINFORMATIVE name** (empty / synthetic ``pds_*``) — spectrum-count
         fallback (> threshold ⇒ survey, else detail).
      4. **UNKNOWN name** (informative but unrecognized) — the
         :data:`SCAN_TYPE_QUARANTINE` sentinel; the caller MUST NOT write a
         guessed scan_type.

    Value-blind: reads only the scan name, sequence code, and spectrum count.

    Returns:
        A :class:`ScanType`, or :data:`SCAN_TYPE_QUARANTINE` for an
        informative-unknown name.
    """
    seq = (sequence_code or "").strip().lower()
    if seq in _CALIBRATION_SEQUENCE_CODES:
        return ScanType.CALIBRATION
    if _is_calibration_name(scan_name):
        return ScanType.CALIBRATION

    recognized = _scan_type_from_name(scan_name)
    if recognized is not None:
        return recognized

    if _is_uninformative_name(scan_name):
        if n_spectra is not None and n_spectra > _SURVEY_SPECTRA_THRESHOLD:
            return ScanType.SURVEY
        return ScanType.DETAIL

    return SCAN_TYPE_QUARANTINE


# ---------------------------------------------------------------------------
# Analytical product role classification (multishot; WS-1 spec §4.4)
# ---------------------------------------------------------------------------

# Multishot SNR-reduction product suffixes (WS-1 spec §4.4). Ordered
# most-specific first so ``_median_all`` is matched before the bare ``_all``
# SPATIAL union and ``_sum_active_median_dark`` before any ``_dark`` shorthand.
# The analytical CANONICAL product is ``*_sum_active_median_dark`` (sums active
# → full ~900 ppp signal; 3× median dark → cleaner dark); other recognized
# reductions are ALTERNATE. The bare ``*_all`` spatial union is NOT a multishot
# product (product_role NULL) — this is the ``_all`` naming collision.
_MULTISHOT_REDUCTION_SUFFIXES = (
    ("_sum_active_median_dark", "canonical"),
    ("_sum_active_sum_dark", "alternate"),
    ("_median_all", "alternate"),
)


def multishot_reduction_role(scan_name: Optional[str]) -> Optional[str]:
    """Return the analytical role of a multishot SNR-reduction product name.

    'canonical' for ``*_sum_active_median_dark``, 'alternate' for other
    recognized reductions (``*_median_all``, ``*_sum_active_sum_dark``), or
    None if the name is not a recognized reduction.

    Distinguishes the multishot reductions from the bare ``*_all`` SPATIAL
    union (a composite with product_role NULL) — the ``_all`` naming collision
    (spec §4.3/§4.4).
    """
    low = (scan_name or "").strip().lower()
    for suffix, role in _MULTISHOT_REDUCTION_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix):
            return role
    return None


def multishot_raw_base(scan_name: Optional[str]) -> Optional[str]:
    """For a recognized multishot reduction name, return the base (raw) scan
    name it reduces (e.g. ``detail_2_median_all`` -> ``detail_2``); else None.

    Case-preserving on the base portion so the result can be matched against
    the sibling raw scan's stored name.
    """
    clean = (scan_name or "").strip()
    low = clean.lower()
    for suffix, _role in _MULTISHOT_REDUCTION_SUFFIXES:
        if low.endswith(suffix) and len(low) > len(suffix):
            return clean[: -len(suffix)]
    return None


def classify_product_role(scan_name: Optional[str]) -> Optional[str]:
    """Name-derivable analytical ``product_role`` for a multishot reduction.

    Returns 'canonical' for ``*_sum_active_median_dark``, 'alternate' for
    other recognized reductions, or None for everything else (normal scans,
    spatial unions, name-union composites).

    Note: the 'raw' role and the ``source_scan_ids = [raw_id]`` lineage are
    **corpus-level** (assigned by ``reclassify-product-roles``), since they
    require resolving the sibling raw scan in the same sol/target group — they
    cannot be derived from the name alone.
    """
    return multishot_reduction_role(scan_name)


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
    product_role: Optional[str] = Field(
        default=None,
        description="Analytical role of a multishot product: 'raw', 'canonical', "
        "or 'alternate'. NULL for every non-multishot scan (≈the entire corpus). "
        "The '*_sum_active_median_dark' reduction is the counted 'canonical' "
        "product; the multishot raw is 'raw' (not counted); '*_median_all' is "
        "'alternate'. Set by reclassify-product-roles (corpus-level)."
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


# Number of channels in a SHERLOC full-CCD region plane; mask channel
# indices are absolute offsets 0..N_PLANE_CHANNELS-1 on this plane.
N_PLANE_CHANNELS = 2148


@ModelRegistry.register
class CosmicRayMask(IdentifiableModel):
    """A persisted cosmic-ray detection mask for one stored spectrum.

    One ``CosmicRayMask`` exists per (DARK_SUBTRACTED spectrum, method)
    pair: it records the absolute channel indices flagged as cosmic-ray
    contaminated by a despike method, together with the provenance trio
    that makes the result auditable (per MLD-SYS-008): the method
    identity (e.g. ``ml_v1.3_tau_matched``), the model artifact's sha256
    digest, and the threshold ``tau`` applied to this region.

    Channel indices are **absolute** offsets on the region's
    2148-channel plane (0..2147); they are sorted strictly increasing
    with no duplicates. ``n_flagged`` equals ``len(channel_indices)`` and
    is stored as a query convenience. Masks are representation-invariant
    channel sets, so the same mask can be applied to any downstream
    representation that carries the region's channels.

    Masks are immutable: a re-run deletes the existing row for the
    (spectrum, method) pair and inserts a fresh one, so there is no
    ``updated_at`` column on the ORM table. ``IdentifiableModel`` still
    brings ``id``/``created_at``/``updated_at`` to this domain model; the
    ORM's ``to_pydantic`` leaves ``updated_at`` ``None`` because the table
    intentionally has no such column.

    Attributes:
        spectrum_id: UUID of the parent DARK_SUBTRACTED Spectrum.
        method: Despike method identity (e.g. ``ml_v1.3_tau_matched``).
        model_sha256: 64-hex sha256 digest of the model artifact that
            produced the mask.
        tau: Decision threshold applied for this region.
        channel_indices: Sorted, strictly increasing absolute channel
            indices (each in [0, 2148)) flagged as cosmic-ray hits.
        n_flagged: Number of flagged channels (== len(channel_indices)).

    Example:
        >>> mask = CosmicRayMask(
        ...     spectrum_id=spectrum.id,
        ...     method="ml_v1.3_tau_matched",
        ...     model_sha256="96" * 32,
        ...     tau=0.29882812500038747,
        ...     channel_indices=[120, 405, 511],
        ...     n_flagged=3,
        ... )
    """

    spectrum_id: uuid.UUID = Field(
        description="UUID of the parent DARK_SUBTRACTED Spectrum"
    )
    method: str = Field(
        max_length=40,
        description="Despike method identity (e.g. 'ml_v1.3_tau_matched')"
    )
    model_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="64-hex sha256 digest of the model artifact"
    )
    tau: float = Field(
        description="Decision threshold applied to this region"
    )
    channel_indices: List[int] = Field(
        description="Sorted, strictly increasing absolute channel indices "
        "(each in [0, 2148)) flagged as cosmic-ray hits"
    )
    n_flagged: int = Field(
        description="Number of flagged channels (== len(channel_indices))"
    )

    @field_validator("channel_indices")
    @classmethod
    def validate_channel_indices(cls, v: List[int]) -> List[int]:
        """Validate channel indices are in range and strictly increasing.

        Each index must lie in [0, 2148); the list must be sorted with
        no duplicates (strictly increasing).
        """
        for idx in v:
            if idx < 0 or idx >= N_PLANE_CHANNELS:
                raise ValueError(
                    f"channel index {idx} out of range "
                    f"[0, {N_PLANE_CHANNELS})"
                )
        for prev, cur in zip(v, v[1:]):
            if cur <= prev:
                raise ValueError(
                    "channel_indices must be strictly increasing "
                    f"(sorted, no duplicates); found {prev} >= {cur}"
                )
        return v

    @model_validator(mode="after")
    def validate_n_flagged(self) -> "CosmicRayMask":
        """Validate that n_flagged equals len(channel_indices)."""
        if self.n_flagged != len(self.channel_indices):
            raise ValueError(
                f"n_flagged ({self.n_flagged}) must equal "
                f"len(channel_indices) ({len(self.channel_indices)})"
            )
        return self
