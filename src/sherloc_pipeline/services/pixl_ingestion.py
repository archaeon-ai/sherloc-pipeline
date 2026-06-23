"""
Ingestion service for loading Pixlise PIXL data into SQLite database.

This service orchestrates parsing Pixlise zip exports and persisting
the data to the Pixlise SQLite database. It provides idempotent ingestion
with progress logging.

Database location: /data/pixl/pixlise.db

Example:
    >>> from sherloc_pipeline.services.pixl_ingestion import PixliseIngestionService
    >>>
    >>> service = PixliseIngestionService()
    >>> result = service.ingest_directory("/nas/000_pixl")
    >>> print(result.summary)
    "Ingested 52 targets (99,503 points)"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from sherloc_pipeline.database.connection import get_engine, get_session
from sherloc_pipeline.database.pixl_models import (
    PixliseBase,
    PixliseTargetORM,
    PixliseQuantPointORM,
    PixliseImageORM,
    PixliseBeamLocationORM,
)
from sherloc_pipeline.models.pixl import (
    PixliseExportParser,
    PixliseExportResult,
)
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError


logger = logging.getLogger(__name__)


# Default Pixlise database path
PIXLISE_DATABASE_PATH = Path("/data/pixl/pixlise.db")


class PixliseIngestionError(SherlocServiceError):
    """Error during Pixlise data ingestion."""

    def __init__(
        self,
        message: str,
        target: Optional[str] = None,
        zip_file: Optional[str] = None,
    ):
        super().__init__(message)
        self.target = target
        self.zip_file = zip_file


@dataclass
class PixliseIngestionStats:
    """Statistics from a Pixlise ingestion operation."""

    targets_ingested: int = 0
    targets_skipped: int = 0
    points_ingested: int = 0
    images_ingested: int = 0
    beam_locations_ingested: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __add__(self, other: "PixliseIngestionStats") -> "PixliseIngestionStats":
        """Combine two stats objects."""
        return PixliseIngestionStats(
            targets_ingested=self.targets_ingested + other.targets_ingested,
            targets_skipped=self.targets_skipped + other.targets_skipped,
            points_ingested=self.points_ingested + other.points_ingested,
            images_ingested=self.images_ingested + other.images_ingested,
            beam_locations_ingested=self.beam_locations_ingested
            + other.beam_locations_ingested,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


class PixliseIngestionService:
    """Service for ingesting Pixlise PIXL data into SQLite database.

    This service provides methods for:
    - Ingesting individual zip exports
    - Ingesting directories of zip files
    - Querying ingested data

    All ingestion is idempotent: re-ingesting the same data is a no-op
    (based on RTT uniqueness).

    Attributes:
        console: Rich console for output (optional)
        engine: SQLAlchemy engine for database access
        database_path: Path to the Pixlise database

    Example:
        >>> service = PixliseIngestionService()
        >>> result = service.ingest_directory(Path("/nas/000_pixl"))
        >>> print(result.summary)
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        database_path: Optional[Path] = None,
    ):
        """Initialize the ingestion service.

        Args:
            console: Rich console for progress output
            database_path: Path to SQLite database (defaults to /data/pixl/pixlise.db)
        """
        self.console = console or Console()
        self.parser = PixliseExportParser()

        # Initialize database
        if database_path is None:
            database_path = PIXLISE_DATABASE_PATH

        self.database_path = Path(database_path)

        # Ensure parent directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = get_engine(self.database_path)

        # Create Pixlise tables
        PixliseBase.metadata.create_all(self.engine)

    def ingest_directory(
        self,
        source_dir: Path,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> ServiceResult:
        """Ingest all Pixlise zip files from a directory.

        Args:
            source_dir: Directory containing Pixlise export zip files
            force: If True, re-ingest even if target already exists
            limit: Maximum number of zips to process (for testing)

        Returns:
            ServiceResult with ingestion summary and statistics

        Raises:
            PixliseIngestionError: If the directory is invalid or ingestion fails
        """
        source_dir = Path(source_dir)

        if not source_dir.exists():
            raise PixliseIngestionError(f"Source directory not found: {source_dir}")

        # Find all Pixlise export zip files
        zip_files = sorted(source_dir.glob("Pixlise Data Export*.zip"))
        if not zip_files:
            raise PixliseIngestionError(
                f"No Pixlise export zip files found in: {source_dir}"
            )

        if limit:
            zip_files = zip_files[:limit]

        self.console.print(
            f"[bold]Ingesting {len(zip_files)} Pixlise exports from {source_dir}[/bold]"
        )

        total_stats = PixliseIngestionStats()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing exports...", total=len(zip_files))

            for zip_path in zip_files:
                progress.update(task, description=f"{zip_path.name[:40]}...")

                try:
                    stats = self._ingest_zip_internal(zip_path, force=force)
                    total_stats = total_stats + stats
                except Exception as e:
                    total_stats.errors.append(f"{zip_path.name}: {e}")
                    logger.exception(f"Error ingesting {zip_path}")

                progress.advance(task)

        # Build summary
        summary = (
            f"Ingested {total_stats.targets_ingested} targets "
            f"({total_stats.points_ingested:,} points)"
        )
        if total_stats.targets_skipped:
            summary += f", {total_stats.targets_skipped} skipped"
        if total_stats.images_ingested:
            summary += f", {total_stats.images_ingested} images"

        return ServiceResult(
            summary=summary,
            warnings=total_stats.warnings,
            metadata={
                "success": len(total_stats.errors) == 0,
                "targets_ingested": total_stats.targets_ingested,
                "targets_skipped": total_stats.targets_skipped,
                "points_ingested": total_stats.points_ingested,
                "images_ingested": total_stats.images_ingested,
                "beam_locations_ingested": total_stats.beam_locations_ingested,
                "errors": total_stats.errors,
            },
        )

    def ingest_zip(
        self,
        zip_path: Path,
        force: bool = False,
    ) -> ServiceResult:
        """Ingest a single Pixlise export zip file.

        Args:
            zip_path: Path to Pixlise export zip file
            force: If True, re-ingest even if target already exists

        Returns:
            ServiceResult with ingestion summary

        Raises:
            PixliseIngestionError: If ingestion fails
        """
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise PixliseIngestionError(f"Zip file not found: {zip_path}")

        self.console.print(f"[bold]Ingesting: {zip_path.name}[/bold]")

        try:
            stats = self._ingest_zip_internal(zip_path, force=force)
        except Exception as e:
            raise PixliseIngestionError(
                f"Failed to ingest {zip_path.name}: {e}",
                zip_file=str(zip_path),
            )

        if stats.targets_skipped:
            summary = f"Target already exists (skipped)"
        else:
            summary = (
                f"Ingested target: {stats.points_ingested:,} points, "
                f"{stats.images_ingested} images"
            )

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": True,
                "targets_ingested": stats.targets_ingested,
                "points_ingested": stats.points_ingested,
                "images_ingested": stats.images_ingested,
                "beam_locations_ingested": stats.beam_locations_ingested,
            },
        )

    def _ingest_zip_internal(
        self,
        zip_path: Path,
        force: bool = False,
    ) -> PixliseIngestionStats:
        """Internal method to ingest a single zip file.

        Args:
            zip_path: Path to zip file
            force: If True, re-ingest existing data

        Returns:
            PixliseIngestionStats with counts
        """
        stats = PixliseIngestionStats()

        # Parse the zip file
        result = self.parser.parse_zip(zip_path)
        stats.warnings.extend(result.warnings)

        with get_session(self.engine) as session:
            # Check if target exists (by RTT)
            existing_target = session.execute(
                select(PixliseTargetORM).where(
                    PixliseTargetORM.rtt == result.target.rtt
                )
            ).scalar_one_or_none()

            if existing_target and not force:
                stats.targets_skipped = 1
                return stats

            # Delete existing target if force
            if existing_target and force:
                session.delete(existing_target)
                session.flush()

            # Insert target
            target_orm = PixliseTargetORM.from_pydantic(result.target)
            session.add(target_orm)
            session.flush()

            # Insert quant points in batches
            batch_size = 1000
            for i in range(0, len(result.quant_points), batch_size):
                batch = result.quant_points[i : i + batch_size]
                for point in batch:
                    point_orm = PixliseQuantPointORM.from_pydantic(point)
                    session.add(point_orm)
                session.flush()
                stats.points_ingested += len(batch)

            # Insert images
            for image in result.images:
                image_orm = PixliseImageORM.from_pydantic(image)
                session.add(image_orm)
                stats.images_ingested += 1

            # Insert beam locations in batches
            for i in range(0, len(result.beam_locations), batch_size):
                batch = result.beam_locations[i : i + batch_size]
                for loc in batch:
                    loc_orm = PixliseBeamLocationORM.from_pydantic(loc)
                    session.add(loc_orm)
                session.flush()
                stats.beam_locations_ingested += len(batch)

            stats.targets_ingested = 1

        return stats

    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics.

        Returns:
            Dictionary with table counts
        """
        with get_session(self.engine) as session:
            return {
                "targets": session.query(PixliseTargetORM).count(),
                "quant_points": session.query(PixliseQuantPointORM).count(),
                "images": session.query(PixliseImageORM).count(),
                "beam_locations": session.query(PixliseBeamLocationORM).count(),
            }

    def list_targets(self) -> List[Dict[str, Any]]:
        """List all ingested targets.

        Returns:
            List of target info dictionaries
        """
        with get_session(self.engine) as session:
            targets = (
                session.query(PixliseTargetORM)
                .order_by(PixliseTargetORM.name_normalized)
                .all()
            )

            return [
                {
                    "name": t.name,
                    "name_normalized": t.name_normalized,
                    "rtt": t.rtt,
                    "sol": t.sol,
                    "n_points": t.n_points,
                    "piquant_version": t.piquant_version,
                }
                for t in targets
            ]

    def get_target_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get target info by normalized name.

        Args:
            name: Target name (will be normalized)

        Returns:
            Target info dict or None if not found
        """
        normalized = name.strip().lower()

        with get_session(self.engine) as session:
            target = session.execute(
                select(PixliseTargetORM).where(
                    PixliseTargetORM.name_normalized == normalized
                )
            ).scalar_one_or_none()

            if not target:
                return None

            return {
                "id": target.id,
                "name": target.name,
                "name_normalized": target.name_normalized,
                "rtt": target.rtt,
                "sol": target.sol,
                "n_points": target.n_points,
                "piquant_version": target.piquant_version,
                "detector_config": target.detector_config,
                "export_date": target.export_date,
                "source_zip": target.source_zip,
            }
