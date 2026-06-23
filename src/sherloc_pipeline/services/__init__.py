"""
Services layer for SHERLOC pipeline orchestration.

This package provides orchestration services that encapsulate business logic
and coordinate between CLI commands and core modules. Services return
lightweight structured results suitable for both CLI consumption and
programmatic use.

Key Modules:
- base: ServiceResult dataclass for standardized service outputs
- config: Cached runtime configuration access
- paths: Path resolution and scan context utilities
- errors: Typed error hierarchy for service-layer exceptions

Usage:
    from sherloc_pipeline.services.config import get_runtime_config
    from sherloc_pipeline.services.paths import resolve_scan_context
    from sherloc_pipeline.services.errors import PreprocessingError, enrich
    
    config = get_runtime_config()
    context = resolve_scan_context("0921", "Amherst_Point", "detail_1")
    
    # Error handling
    try:
        # Service operation
        pass
    except PreprocessingError as e:
        enriched = enrich(e, sol="0921", target="Amherst_Point", scan="detail_1")
        raise enriched
"""

__all__ = [
    "ServiceResult",
    "get_runtime_config",
    "compute_config_hash",
    "resolve_scan_context",
    "ScanContext",
    "SherlocServiceError",
    "PipelineRunError",
    "PreprocessingError",
    "FittingError",
    "SpatialError",
    "ReviewError",
    "SpectralPlotError",
    "IngestionError",
    "ImageIngestionError",
    "ImageQueryError",
    "enrich",
    "PreprocessingService",
    "FittingService",
    "ReviewService",
    "SpatialService",
    "PipelineService",
    "SpectralService",
    "SpectralPlotRequest",
    "SpectrogramService",
    "SpectrogramRequest",
    "SpectrogramResult",
    "IngestionService",
    "ImageIngestionService",
    "ImageQueryService",
    "RunMetadata",
    "StageMetadata",
]

from .base import ServiceResult
from .config import compute_config_hash, get_runtime_config
from .paths import resolve_scan_context, ScanContext
from .errors import (
    SherlocServiceError,
    PipelineRunError,
    PreprocessingError,
    FittingError,
    SpatialError,
    ReviewError,
    enrich,
)
from .preprocessing import PreprocessingService
from .fitting import FittingService
from .review import ReviewService
from .spatial import SpatialService
from .pipeline import PipelineService
from .spectral import SpectralService, SpectralPlotRequest, SpectralPlotError
from .spectrogram import SpectrogramService, SpectrogramRequest, SpectrogramResult
from .ingestion import IngestionService, IngestionError
from .image_ingestion import ImageIngestionService, ImageIngestionError
from .image_query import ImageQueryService, ImageQueryError
from .metadata import RunMetadata, StageMetadata

