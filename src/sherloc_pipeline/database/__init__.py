"""
PHASE Database Package - SQLAlchemy ORM models and Alembic migrations.

This package provides the database layer for PHASE (Planetary Hyperspectral
Analysis and Synthesis Environment). It mirrors the Pydantic models in
sherloc_pipeline.models with SQLAlchemy ORM equivalents.

The database package supports:
- SQLite database at ./phase.db
- Full schema migrations via Alembic
- Indexes on sol_number, scan_name, target, and scan_id for efficient queries
- Bidirectional conversion between Pydantic and SQLAlchemy models

Usage:
    >>> from sherloc_pipeline.database import get_engine, get_session
    >>> from sherloc_pipeline.database.models import SolORM, ScanORM
    >>>
    >>> engine = get_engine("./phase.db")
    >>> with get_session(engine) as session:
    ...     sols = session.query(SolORM).all()

See Also:
    sherloc_pipeline.models: Pydantic models (validation layer)
    docs/schema/UNIFIED_SCHEMA.md: Full schema specification
"""

from sherloc_pipeline.database.connection import (
    get_engine,
    get_session,
    create_all_tables,
    init_database,
    init_pds_database,
    DATABASE_URL,
    PDS_DATABASE_PATH,
)

from sherloc_pipeline.database.models import (
    Base,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    InstrumentStateORM,
    CCDConfigurationORM,
    ScannerCalibrationORM,
    ContextImageORM,
    RegionOfInterestORM,
    FittedPeakORM,
    SpectrogramORM,
)

from sherloc_pipeline.database.pixl_models import (
    PixliseBase,
    PixliseTargetORM,
    PixliseQuantPointORM,
    PixliseImageORM,
    PixliseBeamLocationORM,
)

__all__ = [
    # Connection utilities
    "get_engine",
    "get_session",
    "create_all_tables",
    "init_database",
    "init_pds_database",
    "DATABASE_URL",
    "PDS_DATABASE_PATH",
    # Base class
    "Base",
    # ORM models
    "SolORM",
    "ScanORM",
    "ScanPointORM",
    "SpectrumORM",
    "InstrumentStateORM",
    "CCDConfigurationORM",
    "ScannerCalibrationORM",
    "ContextImageORM",
    "RegionOfInterestORM",
    "FittedPeakORM",
    "SpectrogramORM",
    # Pixlise ORM models
    "PixliseBase",
    "PixliseTargetORM",
    "PixliseQuantPointORM",
    "PixliseImageORM",
    "PixliseBeamLocationORM",
]
