"""
Pydantic models for PHASE (Planetary Hyperspectral Analysis and Synthesis Environment).

This package provides all Pydantic data models used throughout PHASE, implementing
the Pydantic-first design philosophy for validation, serialization, and documentation.

Module Structure:
    base: Base model configuration and common mixins
    spectra: Spectral data models (Sol, Scan, ScanPoint, Spectrum)
    instrument: Instrument state and configuration models
    fitting: Peak fitting models
    context: Context image and region of interest models
    ingestion: Raw data models and parsers for Loupe working directories
    pds: PDS4 data models for SHERLOC ingestion from PDS Geosciences Node
    spectrogram: Spectrogram visualization models (2D spectral heatmaps)

Usage:
    >>> from sherloc_pipeline.models import Sol, Scan, ScanPoint, Spectrum
    >>> from sherloc_pipeline.models import SpectralRegion, ProcessingLevel
    >>>
    >>> sol = Sol(sol_number=921)
    >>> scan = Scan(
    ...     sol_number=921,
    ...     scan_name="detail_1",
    ...     scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
    ...     sclk_start=672194998,
    ...     n_points=100,
    ...     shots_per_point=10,
    ... )

See Also:
    docs/schema/UNIFIED_SCHEMA.md for the full schema specification.
    docs/PHASE_SPEC.md for the overall system design.
"""

# Base models and utilities
from sherloc_pipeline.models.base import (
    PHASEBaseModel,
    TimestampedModel,
    IdentifiableModel,
    ValidatedMixin,
    ModelRegistry,
    utc_now,
)

# Spectral data models
from sherloc_pipeline.models.spectra import (
    DataSource,
    SpectralRegion,
    SpectrumType,
    ScanType,
    TargetType,
    ProcessingLevel,
    CoordinateFrame,
    Sol,
    Scan,
    ScanPoint,
    Spectrum,
    classify_target_type,
    classify_scan_class,
    derive_parent_name,
)

# Instrument state and configuration
from sherloc_pipeline.models.instrument import (
    InstrumentState,
    CCDConfiguration,
    ScannerCalibration,
)

# Peak fitting models
from sherloc_pipeline.models.fitting import (
    PeakType,
    FittedPeak,
    FittingResult,
)

# Context images and ROIs
from sherloc_pipeline.models.context import (
    ImageType,
    ContextImage,
    RegionOfInterest,
)

# Ingestion models and parsers
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
    discover_workspaces,
)

# PDS4 ingestion models
from sherloc_pipeline.models.pds import (
    PDSProductType,
    PDSProductId,
    PDSObservationMetadata,
    PDSSpectralProduct,
    PDSPositionProduct,
    PDSPhotodiodeProduct,
    PDSCalibrationRecord,
    PDSCalibrationProduct,
    PDSCrossRefRecord,
    PDSCrossRefProduct,
    CORE_PRODUCT_TYPES,
    PDS_MISSION_SCLK_MIN,
    PDS_EXPECTED_CHANNELS,
)

# Spectrogram visualization
from sherloc_pipeline.models.spectrogram import (
    ColorMapType,
    AxisScale,
    NormalizationType,
    InterpolationMethod,
    SpectrogramConfig,
    SpectrogramData,
    Spectrogram,
    DifferenceSpectrogram,
)

# Integration utilities for core.fitting <-> Pydantic conversion
from sherloc_pipeline.models.integration import (
    peak_fit_to_pydantic,
    peak_fits_to_pydantic,
    fit_result_to_pydantic,
    pydantic_to_peak_fit,
    pydantic_to_fit_result,
)

# PIXL Pixlise models
from sherloc_pipeline.models.pixl import (
    OXIDE_NAMES,
    N_OXIDES,
    PixliseImageType,
    PixliseTarget,
    PixliseQuantPoint,
    PixliseBeamLocation,
    PixliseImage,
    PixliseExportResult,
    PixliseExportParser,
)

__all__ = [
    # Base models
    "PHASEBaseModel",
    "TimestampedModel",
    "IdentifiableModel",
    # Mixins and utilities
    "ValidatedMixin",
    "ModelRegistry",
    "utc_now",
    # Enums - spectra
    "DataSource",
    "SpectralRegion",
    "SpectrumType",
    "ScanType",
    "TargetType",
    "ProcessingLevel",
    "CoordinateFrame",
    # Classification functions
    "classify_target_type",
    "classify_scan_class",
    "derive_parent_name",
    # Core domain models
    "Sol",
    "Scan",
    "ScanPoint",
    "Spectrum",
    # Instrument models
    "InstrumentState",
    "CCDConfiguration",
    "ScannerCalibration",
    # Fitting models
    "PeakType",
    "FittedPeak",
    "FittingResult",
    # Context models
    "ImageType",
    "ContextImage",
    "RegionOfInterest",
    # Ingestion models
    "RawLoupeMetadata",
    "RawSpatialData",
    "RawSpatialPoint",
    "RawPhotodiodeData",
    "RawPhotodiodeStats",
    "RawROI",
    "RawROIData",
    "RawSpectraFile",
    "LoupeSessionEntry",
    "LoupeSessionFile",
    "LoupeWorkspaceParser",
    "LoupeWorkspaceResult",
    "extract_sol_from_path",
    "discover_workspaces",
    # PDS4 models
    "PDSProductType",
    "PDSProductId",
    "PDSObservationMetadata",
    "PDSSpectralProduct",
    "PDSPositionProduct",
    "PDSPhotodiodeProduct",
    "PDSCalibrationRecord",
    "PDSCalibrationProduct",
    "PDSCrossRefRecord",
    "PDSCrossRefProduct",
    "CORE_PRODUCT_TYPES",
    "PDS_MISSION_SCLK_MIN",
    "PDS_EXPECTED_CHANNELS",
    # Spectrogram models
    "ColorMapType",
    "AxisScale",
    "NormalizationType",
    "InterpolationMethod",
    "SpectrogramConfig",
    "SpectrogramData",
    "Spectrogram",
    "DifferenceSpectrogram",
    # Integration utilities
    "peak_fit_to_pydantic",
    "peak_fits_to_pydantic",
    "fit_result_to_pydantic",
    "pydantic_to_peak_fit",
    "pydantic_to_fit_result",
    # PIXL Pixlise models
    "OXIDE_NAMES",
    "N_OXIDES",
    "PixliseImageType",
    "PixliseTarget",
    "PixliseQuantPoint",
    "PixliseBeamLocation",
    "PixliseImage",
    "PixliseExportResult",
    "PixliseExportParser",
]
