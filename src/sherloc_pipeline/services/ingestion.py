"""
Ingestion service for loading Loupe data into SQLite database.

This service orchestrates parsing Loupe working directories and persisting
the data to the PHASE SQLite database. It provides idempotent ingestion
with progress logging.

Example:
    >>> from sherloc_pipeline.services.ingestion import IngestionService
    >>>
    >>> service = IngestionService()
    >>> result = service.ingest_sol(Path("./data/loupe/sol_0921"))
    >>> print(result.summary)
    "Ingested sol 921: 5 scans, 450 points"
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
import uuid

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    create_all_tables,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    InstrumentStateORM,
    CCDConfigurationORM,
    ScannerCalibrationORM,
    ContextImageORM,
    RegionOfInterestORM,
)
from sherloc_pipeline.models.ingestion import (
    LoupeWorkspaceParser,
    LoupeWorkspaceResult,
    LoupeSessionFile,
    discover_workspaces,
    extract_sol_from_path,
    extract_target_from_lpe,
    RawSpectraFile,
    SpectrumType,
)
from sherloc_pipeline.models.spectra import (
    ProcessingLevel, Sol, DataSource, TargetType, classify_target_type,
)
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError


logger = logging.getLogger(__name__)


class IngestionError(SherlocServiceError):
    """Error during data ingestion."""

    def __init__(self, message: str, sol: Optional[int] = None, scan: Optional[str] = None):
        super().__init__(message)
        self.sol = sol
        self.scan = scan


@dataclass
class IngestionStats:
    """Statistics from an ingestion operation."""

    sols_processed: int = 0
    sols_skipped: int = 0
    scans_ingested: int = 0
    scans_skipped: int = 0
    points_ingested: int = 0
    spectra_ingested: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __add__(self, other: "IngestionStats") -> "IngestionStats":
        """Combine two stats objects."""
        return IngestionStats(
            sols_processed=self.sols_processed + other.sols_processed,
            sols_skipped=self.sols_skipped + other.sols_skipped,
            scans_ingested=self.scans_ingested + other.scans_ingested,
            scans_skipped=self.scans_skipped + other.scans_skipped,
            points_ingested=self.points_ingested + other.points_ingested,
            spectra_ingested=self.spectra_ingested + other.spectra_ingested,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


class IngestionService:
    """Service for ingesting Loupe data into the PHASE database.

    This service provides methods for:
    - Ingesting individual workspaces
    - Ingesting complete sols
    - Ingesting entire Loupe data directories

    All ingestion is idempotent: re-ingesting the same data is a no-op.

    Attributes:
        console: Rich console for output (optional)
        engine: SQLAlchemy engine for database access
        include_spectra: Whether to include spectra data (can be slow)
        ingestion_mode: Which spectral regions to ingest.
            'all_regions' (default) - Ingest R1, R2, R3 as separate spectra rows per scan point.
            'R1_only' - Ingest only R1 (Raman) section.

    Example:
        >>> service = IngestionService(database_path="./phase.db")
        >>> result = service.ingest_directory(Path("./data/loupe"))
        >>>
        >>> # Ingest only R1 for Raman-only analysis
        >>> service = IngestionService(
        ...     database_path="./phase.db",
        ...     ingestion_mode='R1_only',
        ... )
    """

    VALID_INGESTION_MODES = ("R1_only", "all_regions")

    def __init__(
        self,
        console: Optional[Console] = None,
        database_path: Optional[Path] = None,
        include_spectra: bool = True,
        ingestion_mode: str = "all_regions",
    ):
        """Initialize the ingestion service.

        Args:
            console: Rich console for progress output
            database_path: Path to SQLite database (defaults to ./phase.db)
            include_spectra: Whether to load and store spectra data
            ingestion_mode: Which spectral regions to ingest.
                'all_regions' (default) - R1, R2, R3 as separate spectra rows per scan point.
                'R1_only' - Ingest only R1 (Raman) section.
        """
        if ingestion_mode not in self.VALID_INGESTION_MODES:
            raise ValueError(
                f"Invalid ingestion_mode '{ingestion_mode}'. "
                f"Must be one of: {self.VALID_INGESTION_MODES}"
            )

        self.console = console or Console()
        self.include_spectra = include_spectra
        self.ingestion_mode = ingestion_mode

        # Initialize database
        if database_path is None:
            database_path = Path("./phase.db")

        self.database_path = database_path
        self.engine = get_engine(database_path)
        create_all_tables(self.engine)

    def ingest_directory(
        self,
        loupe_dir: Path,
        force: bool = False,
        limit: Optional[int] = None,
    ) -> ServiceResult:
        """Ingest all sols from a Loupe data directory.

        Args:
            loupe_dir: Path to Loupe data directory containing sol_XXXX dirs
            force: If True, re-ingest even if sol already exists
            limit: Maximum number of sols to process (for testing)

        Returns:
            ServiceResult with ingestion summary and statistics

        Raises:
            IngestionError: If the directory is invalid or ingestion fails
        """
        loupe_dir = Path(loupe_dir)

        if not loupe_dir.exists():
            raise IngestionError(f"Loupe directory not found: {loupe_dir}")

        # Find all sol directories
        sol_dirs = sorted(loupe_dir.glob("sol_*"))
        if not sol_dirs:
            raise IngestionError(f"No sol directories found in: {loupe_dir}")

        if limit:
            sol_dirs = sol_dirs[:limit]

        self.console.print(f"[bold]Ingesting {len(sol_dirs)} sols from {loupe_dir}[/bold]")

        total_stats = IngestionStats()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing sols...", total=len(sol_dirs))

            for sol_dir in sol_dirs:
                sol_number = extract_sol_from_path(sol_dir)
                progress.update(task, description=f"Sol {sol_number}")

                try:
                    stats = self._ingest_sol_internal(sol_dir, force=force)
                    total_stats = total_stats + stats
                except Exception as e:
                    total_stats.errors.append(f"Sol {sol_number}: {e}")
                    logger.exception(f"Error ingesting sol {sol_number}")

                progress.advance(task)

        # Build summary
        summary = (
            f"Ingested {total_stats.scans_ingested} scans "
            f"({total_stats.points_ingested} points) "
            f"from {total_stats.sols_processed} sols"
        )
        if total_stats.sols_skipped:
            summary += f", {total_stats.sols_skipped} sols skipped"
        if total_stats.scans_skipped:
            summary += f", {total_stats.scans_skipped} scans skipped"

        return ServiceResult(
            summary=summary,
            warnings=total_stats.warnings,
            metadata={
                "success": len(total_stats.errors) == 0,
                "sols_processed": total_stats.sols_processed,
                "sols_skipped": total_stats.sols_skipped,
                "scans_ingested": total_stats.scans_ingested,
                "scans_skipped": total_stats.scans_skipped,
                "points_ingested": total_stats.points_ingested,
                "spectra_ingested": total_stats.spectra_ingested,
                "errors": total_stats.errors,
            },
        )

    def ingest_sol(
        self,
        sol_dir: Path,
        force: bool = False,
    ) -> ServiceResult:
        """Ingest a single sol directory.

        Args:
            sol_dir: Path to sol_XXXX directory
            force: If True, re-ingest even if sol already exists

        Returns:
            ServiceResult with ingestion summary

        Raises:
            IngestionError: If ingestion fails
        """
        sol_dir = Path(sol_dir)

        if not sol_dir.exists():
            raise IngestionError(f"Sol directory not found: {sol_dir}")

        sol_number = extract_sol_from_path(sol_dir)
        if sol_number is None:
            raise IngestionError(f"Cannot extract sol number from path: {sol_dir}")

        self.console.print(f"[bold]Ingesting sol {sol_number}...[/bold]")

        try:
            stats = self._ingest_sol_internal(sol_dir, force=force)
        except Exception as e:
            raise IngestionError(f"Failed to ingest sol {sol_number}: {e}", sol=sol_number)

        if stats.sols_skipped:
            summary = f"Sol {sol_number} already exists (skipped)"
        else:
            summary = (
                f"Ingested sol {sol_number}: {stats.scans_ingested} scans, "
                f"{stats.points_ingested} points"
            )
            if self.include_spectra:
                summary += f", {stats.spectra_ingested} spectra"

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": True,
                "sol_number": sol_number,
                "scans_ingested": stats.scans_ingested,
                "scans_skipped": stats.scans_skipped,
                "points_ingested": stats.points_ingested,
                "spectra_ingested": stats.spectra_ingested,
            },
        )

    def ingest_workspace(
        self,
        workspace_path: Path,
        force: bool = False,
    ) -> ServiceResult:
        """Ingest a single Loupe workspace directory.

        Args:
            workspace_path: Path to *_Loupe_working directory
            force: If True, re-ingest even if scan already exists

        Returns:
            ServiceResult with ingestion summary

        Raises:
            IngestionError: If ingestion fails
        """
        workspace_path = Path(workspace_path)

        if not workspace_path.exists():
            raise IngestionError(f"Workspace not found: {workspace_path}")

        if not (workspace_path / "loupe.csv").exists():
            raise IngestionError(f"Not a valid Loupe workspace (missing loupe.csv): {workspace_path}")

        sol_number = extract_sol_from_path(workspace_path)
        if sol_number is None:
            sol_number = 0  # Default if not extractable

        self.console.print(f"[bold]Ingesting workspace: {workspace_path.name}[/bold]")

        try:
            with get_session(self.engine) as session:
                stats = self._ingest_workspace_internal(
                    session, workspace_path, sol_number, force=force
                )
        except Exception as e:
            raise IngestionError(f"Failed to ingest workspace: {e}")

        if stats.scans_skipped:
            summary = f"Workspace already exists (skipped)"
        else:
            summary = (
                f"Ingested workspace: {stats.scans_ingested} scan, "
                f"{stats.points_ingested} points"
            )

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": True,
                "scans_ingested": stats.scans_ingested,
                "points_ingested": stats.points_ingested,
                "spectra_ingested": stats.spectra_ingested,
            },
        )

    def _ingest_sol_internal(
        self,
        sol_dir: Path,
        force: bool = False,
    ) -> IngestionStats:
        """Internal method to ingest a sol directory.

        Args:
            sol_dir: Path to sol directory
            force: If True, re-ingest existing data

        Returns:
            IngestionStats with counts
        """
        stats = IngestionStats()

        sol_number = extract_sol_from_path(sol_dir)
        if sol_number is None:
            stats.errors.append(f"Cannot extract sol number from: {sol_dir}")
            return stats

        with get_session(self.engine) as session:
            # Check if sol exists
            existing_sol = session.get(SolORM, sol_number)

            if existing_sol and not force:
                stats.sols_skipped = 1
                return stats

            # Create or update sol
            if not existing_sol:
                sol_orm = SolORM(
                    sol_number=sol_number,
                    data_source="loupe",
                    created_at=datetime.now(timezone.utc),
                )
                session.add(sol_orm)
                session.flush()  # Ensure sol exists before workspace ingestion
            else:
                # Update timestamp
                existing_sol.updated_at = datetime.now(timezone.utc)

            stats.sols_processed = 1

            # Discover workspaces
            workspaces = discover_workspaces(sol_dir)

            # Extract target name from .lpe filename
            lpe_target = extract_target_from_lpe(sol_dir)

            for workspace_path in workspaces:
                try:
                    ws_stats = self._ingest_workspace_internal(
                        session, workspace_path, sol_number,
                        force=force, target=lpe_target,
                    )
                    stats = stats + ws_stats
                except Exception as e:
                    stats.errors.append(f"Workspace {workspace_path.name}: {e}")
                    logger.exception(f"Error ingesting workspace {workspace_path}")

        return stats

    def _ingest_workspace_internal(
        self,
        session: Session,
        workspace_path: Path,
        sol_number: int,
        force: bool = False,
        target: Optional[str] = None,
    ) -> IngestionStats:
        """Internal method to ingest a workspace within a session.

        Args:
            session: SQLAlchemy session
            workspace_path: Path to Loupe_working directory
            sol_number: Sol number for this workspace
            force: If True, re-ingest existing data
            target: Target name extracted from .lpe filename (optional).
                If provided, sets scan.target before classify_target_type()
                so that engineering scans (power_on etc.) are correctly
                classified even when the target is known.

        Returns:
            IngestionStats with counts
        """
        stats = IngestionStats()

        # Parse workspace
        parser = LoupeWorkspaceParser(workspace_path, sol_number=sol_number)
        result = parser.parse()

        # Set target from .lpe filename if not already set
        if target and not result.scan.target:
            result.scan.target = target

        # Check if scan exists (by scan_id)
        existing_scan = session.execute(
            select(ScanORM).where(ScanORM.scan_id == result.scan.scan_id)
        ).scalar_one_or_none()

        if existing_scan and not force:
            stats.scans_skipped = 1
            return stats

        # Delete existing scan if force
        if existing_scan and force:
            session.delete(existing_scan)
            session.flush()

        # Ensure sol exists
        existing_sol = session.get(SolORM, sol_number)
        if not existing_sol:
            sol_orm = SolORM(
                sol_number=sol_number,
                data_source="loupe",
                created_at=datetime.now(timezone.utc),
            )
            session.add(sol_orm)

        # Classify target type before ORM conversion
        if result.scan.target_type is None:
            result.scan.target_type = TargetType(
                classify_target_type(result.scan.target, result.scan.scan_name)
            )

        # Create scan ORM
        scan_orm = ScanORM.from_pydantic(result.scan)
        session.add(scan_orm)

        # Add instrument state
        if result.instrument_state:
            state_orm = InstrumentStateORM.from_pydantic(result.instrument_state)
            session.add(state_orm)

        # Add CCD configuration
        if result.ccd_configuration:
            ccd_orm = CCDConfigurationORM.from_pydantic(result.ccd_configuration)
            session.add(ccd_orm)

        # Add scanner calibration
        if result.scanner_calibration:
            cal_orm = ScannerCalibrationORM.from_pydantic(result.scanner_calibration)
            session.add(cal_orm)

        # Add scan points
        point_id_map: Dict[int, uuid.UUID] = {}  # point_index -> point_id
        for point in result.scan_points:
            point_orm = ScanPointORM.from_pydantic(point)
            session.add(point_orm)
            point_id_map[point.point_index] = point.id
            stats.points_ingested += 1

        # Add regions of interest
        for roi in result.regions_of_interest:
            roi_orm = RegionOfInterestORM.from_pydantic(roi)
            session.add(roi_orm)

        # Add context images
        for img in result.context_images:
            img_orm = ContextImageORM.from_pydantic(img)
            session.add(img_orm)

        # Add spectra (if enabled and available)
        # Determine which sections to ingest based on ingestion_mode.
        # R1_only (default): Only R1 Raman section (backward compatible).
        # all_regions: R1, R2, R3 as separate spectra rows per scan point.
        if self.include_spectra:
            if self.ingestion_mode == "all_regions":
                sections_to_ingest = ["R1", "R2", "R3"]
            else:
                sections_to_ingest = ["R1"]

            for spectrum_type, fpath in result.spectra_files.items():
                for section in sections_to_ingest:
                    try:
                        spectra = parser.parse_spectra(
                            spectrum_type,
                            list(point_id_map.values()),
                            ProcessingLevel.RAW,
                            section=section,
                        )
                        for spectrum in spectra:
                            spectrum_orm = SpectrumORM.from_pydantic(spectrum)
                            session.add(spectrum_orm)
                            stats.spectra_ingested += 1
                    except Exception as e:
                        stats.warnings.append(
                            f"Failed to parse {spectrum_type.value} "
                            f"spectra (section {section}): {e}"
                        )

        stats.scans_ingested = 1
        return stats

    def get_database_stats(self) -> Dict[str, int]:
        """Get current database statistics.

        Returns:
            Dictionary with table counts
        """
        with get_session(self.engine) as session:
            return {
                "sols": session.query(SolORM).count(),
                "scans": session.query(ScanORM).count(),
                "scan_points": session.query(ScanPointORM).count(),
                "spectra": session.query(SpectrumORM).count(),
                "instrument_states": session.query(InstrumentStateORM).count(),
                "ccd_configurations": session.query(CCDConfigurationORM).count(),
                "scanner_calibrations": session.query(ScannerCalibrationORM).count(),
                "context_images": session.query(ContextImageORM).count(),
                "regions_of_interest": session.query(RegionOfInterestORM).count(),
            }

    def list_sols(self) -> List[int]:
        """List all ingested sol numbers.

        Returns:
            Sorted list of sol numbers
        """
        with get_session(self.engine) as session:
            sols = session.query(SolORM.sol_number).order_by(SolORM.sol_number).all()
            return [s[0] for s in sols]

    def get_sol_scans(self, sol_number: int) -> List[Dict[str, Any]]:
        """Get scan information for a sol.

        Args:
            sol_number: Sol number to query

        Returns:
            List of scan info dictionaries
        """
        with get_session(self.engine) as session:
            scans = session.query(ScanORM).filter(
                ScanORM.sol_number == sol_number
            ).all()

            return [
                {
                    "scan_id": s.scan_id,
                    "scan_name": s.scan_name,
                    "target": s.target,
                    "n_points": s.n_points,
                    "sclk_start": s.sclk_start,
                }
                for s in scans
            ]
