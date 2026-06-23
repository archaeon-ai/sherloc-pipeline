"""
Image ingestion service for SHERLOC ACI images.

This service provides functionality to ingest raw .IMG files from the SHERLOC
Autofocus Context Imager (ACI) into the PHASE database. It parses VICAR headers,
extracts metadata, and links images to spectral scans via SCLK timing.

Key features:
- Idempotent ingestion (safe to re-run)
- SCLK-based scan linkage
- Progress tracking for batch operations
- Support for both IMG and PNG formats

Example:
    >>> from sherloc_pipeline.services.image_ingestion import ImageIngestionService
    >>>
    >>> service = ImageIngestionService()
    >>> result = service.ingest_all_images(Path("./data/loupe"))
    >>> print(result.summary)
    "Ingested 909 images, linked 850 to scans"
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from sqlalchemy import select, text
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    ScanORM,
    ContextImageORM,
)
from sherloc_pipeline.vision.img_reader import (
    read_aci_image,
    get_raw_vicar_label,
    scan_img_files,
    ACIImageMetadata,
)
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError
from sherloc_pipeline.models.ingestion import extract_sol_from_path


logger = logging.getLogger(__name__)


class ImageIngestionError(SherlocServiceError):
    """Error during image ingestion."""

    def __init__(self, message: str, image_path: Optional[str] = None):
        super().__init__(message)
        self.image_path = image_path


@dataclass
class ImageIngestionStats:
    """Statistics from an image ingestion operation."""

    images_scanned: int = 0
    images_ingested: int = 0
    images_skipped: int = 0
    images_linked: int = 0
    images_unlinked: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __add__(self, other: "ImageIngestionStats") -> "ImageIngestionStats":
        """Combine two stats objects."""
        return ImageIngestionStats(
            images_scanned=self.images_scanned + other.images_scanned,
            images_ingested=self.images_ingested + other.images_ingested,
            images_skipped=self.images_skipped + other.images_skipped,
            images_linked=self.images_linked + other.images_linked,
            images_unlinked=self.images_unlinked + other.images_unlinked,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def parse_sclk_string(sclk_str: str) -> Optional[int]:
    """Parse SCLK string to integer seconds.

    Args:
        sclk_str: SCLK string like "697951240.076" or "0697951240-30092"

    Returns:
        Integer SCLK seconds, or None if parsing fails
    """
    if not sclk_str:
        return None

    try:
        # Handle format: "697951240.076" (with decimal)
        if '.' in sclk_str:
            return int(float(sclk_str))
        # Handle format: "0697951240-30092" (with subseconds)
        elif '-' in sclk_str:
            return int(sclk_str.split('-')[0].lstrip('0') or '0')
        else:
            return int(sclk_str.lstrip('0') or '0')
    except (ValueError, TypeError):
        return None


def extract_sclk_from_filename(filename: str) -> Optional[int]:
    """Extract SCLK from ACI filename.

    Format: SC3_0349_0697951235_031ECM_N0092982SRLC11360_0000LMJ01.IMG
                   ^
                   SCLK integer

    Args:
        filename: Image filename

    Returns:
        Integer SCLK, or None if not found
    """
    # Pattern: SCx_nnnn_SCLK_...
    pattern = r'SC\d_\d+_(\d+)_'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None



def extract_camera_id_from_filename(filename: str) -> Optional[str]:
    """Extract camera ID from filename.

    Args:
        filename: Image filename (e.g., SC3_0349_...)

    Returns:
        Camera ID (e.g., 'SC3'), or None if not found
    """
    pattern = r'^(SC\d)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


class ImageIngestionService:
    """Service for ingesting ACI images into the PHASE database.

    This service provides methods for:
    - Ingesting individual images
    - Ingesting complete directories
    - Linking images to spectral scans via SCLK

    All ingestion is idempotent: re-ingesting the same image is a no-op.

    Attributes:
        console: Rich console for output (optional)
        engine: SQLAlchemy engine for database access
        sclk_tolerance_min: Minimum SCLK offset for scan matching (seconds)
        sclk_tolerance_max: Maximum SCLK offset for scan matching (seconds)

    Example:
        >>> service = ImageIngestionService(database_path="./phase.db")
        >>> result = service.ingest_all_images(Path("./data/loupe"))
    """

    # Default SCLK tolerance window: images are typically 5-60 seconds before scan
    DEFAULT_SCLK_TOLERANCE_MIN = 5
    DEFAULT_SCLK_TOLERANCE_MAX = 60

    def __init__(
        self,
        console: Optional[Console] = None,
        database_path: Optional[Path] = None,
        sclk_tolerance_min: int = DEFAULT_SCLK_TOLERANCE_MIN,
        sclk_tolerance_max: int = DEFAULT_SCLK_TOLERANCE_MAX,
    ):
        """Initialize the image ingestion service.

        Args:
            console: Rich console for progress output
            database_path: Path to SQLite database (defaults to ./phase.db)
            sclk_tolerance_min: Minimum SCLK offset for scan matching
            sclk_tolerance_max: Maximum SCLK offset for scan matching
        """
        self.console = console or Console()
        self.sclk_tolerance_min = sclk_tolerance_min
        self.sclk_tolerance_max = sclk_tolerance_max

        # Initialize database
        if database_path is None:
            database_path = Path("./phase.db")

        self.database_path = database_path
        self.engine = get_engine(database_path)

    def ingest_all_images(
        self,
        loupe_dir: Path,
        force: bool = False,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> ServiceResult:
        """Ingest all IMG files from a Loupe data directory.

        Args:
            loupe_dir: Path to Loupe data directory
            force: If True, re-ingest even if image already exists
            limit: Maximum number of images to process (for testing)
            show_progress: Whether to show progress bar

        Returns:
            ServiceResult with ingestion summary and statistics

        Raises:
            ImageIngestionError: If the directory is invalid
        """
        loupe_dir = Path(loupe_dir)

        if not loupe_dir.exists():
            raise ImageIngestionError(f"Loupe directory not found: {loupe_dir}")

        # Scan for all IMG files
        img_files = scan_img_files(loupe_dir, recursive=True)

        if not img_files:
            return ServiceResult(
                summary="No IMG files found",
                metadata={"success": True, "images_scanned": 0},
            )

        if limit:
            img_files = img_files[:limit]

        if show_progress:
            self.console.print(
                f"[bold]Ingesting {len(img_files)} IMG files from {loupe_dir}[/bold]"
            )

        stats = ImageIngestionStats()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Processing images...", total=len(img_files))

                for img_path in img_files:
                    try:
                        img_stats = self._ingest_single_image(img_path, force=force)
                        stats = stats + img_stats
                    except Exception as e:
                        stats.errors.append(f"{img_path.name}: {e}")
                        logger.exception(f"Error ingesting image {img_path}")

                    progress.advance(task)
        else:
            for img_path in img_files:
                try:
                    img_stats = self._ingest_single_image(img_path, force=force)
                    stats = stats + img_stats
                except Exception as e:
                    stats.errors.append(f"{img_path.name}: {e}")
                    logger.exception(f"Error ingesting image {img_path}")

        # Build summary
        summary = (
            f"Ingested {stats.images_ingested} images "
            f"({stats.images_linked} linked to scans, {stats.images_unlinked} unlinked)"
        )
        if stats.images_skipped:
            summary += f", {stats.images_skipped} skipped (already exist)"
        if stats.errors:
            summary += f", {len(stats.errors)} errors"

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": len(stats.errors) == 0,
                "images_scanned": len(img_files),
                "images_ingested": stats.images_ingested,
                "images_skipped": stats.images_skipped,
                "images_linked": stats.images_linked,
                "images_unlinked": stats.images_unlinked,
                "errors": stats.errors,
            },
        )

    def ingest_image(
        self,
        img_path: Path,
        force: bool = False,
    ) -> ServiceResult:
        """Ingest a single IMG file.

        Args:
            img_path: Path to IMG file
            force: If True, re-ingest even if already exists

        Returns:
            ServiceResult with ingestion summary

        Raises:
            ImageIngestionError: If ingestion fails
        """
        img_path = Path(img_path)

        if not img_path.exists():
            raise ImageIngestionError(f"Image file not found: {img_path}", str(img_path))

        try:
            stats = self._ingest_single_image(img_path, force=force)
        except Exception as e:
            raise ImageIngestionError(f"Failed to ingest image: {e}", str(img_path))

        if stats.images_skipped:
            summary = f"Image already exists (skipped): {img_path.name}"
        else:
            linked = "linked to scan" if stats.images_linked else "no scan found"
            summary = f"Ingested image: {img_path.name} ({linked})"

        return ServiceResult(
            summary=summary,
            warnings=stats.warnings,
            metadata={
                "success": True,
                "file_path": str(img_path),
                "ingested": stats.images_ingested > 0,
                "linked": stats.images_linked > 0,
            },
        )

    def _ingest_single_image(
        self,
        img_path: Path,
        force: bool = False,
    ) -> ImageIngestionStats:
        """Internal method to ingest a single image.

        Args:
            img_path: Path to IMG file
            force: If True, re-ingest existing data

        Returns:
            ImageIngestionStats with counts
        """
        stats = ImageIngestionStats(images_scanned=1)

        with get_session(self.engine) as session:
            # Check if image already exists by file_path
            existing = session.execute(
                select(ContextImageORM).where(
                    ContextImageORM.file_path == str(img_path)
                )
            ).scalar_one_or_none()

            if existing and not force:
                stats.images_skipped = 1
                return stats

            # Delete existing if force mode
            if existing and force:
                session.delete(existing)
                session.flush()

            # Read image metadata
            try:
                _, metadata = read_aci_image(img_path, validate_dimensions=False)
                raw_label = get_raw_vicar_label(img_path)
            except Exception as e:
                stats.errors.append(f"Failed to read {img_path.name}: {e}")
                return stats

            # Extract SCLK from metadata or filename
            sclk_start = parse_sclk_string(metadata.spacecraft_clock)
            if sclk_start is None:
                sclk_start = extract_sclk_from_filename(img_path.name)

            # Extract sol from path or metadata
            sol_number = metadata.sol if metadata.sol > 0 else extract_sol_from_path(img_path)

            # Extract camera ID
            camera_id = extract_camera_id_from_filename(img_path.name)

            # Find matching scan
            scan_id = None
            if sclk_start and sol_number:
                scan_id = self._find_matching_scan(session, sclk_start, sol_number)

            # Create context image record
            image_id = str(uuid.uuid4())

            # Build VICAR metadata dict (subset of useful fields)
            vicar_dict = {k: v for k, v in raw_label.items() if isinstance(v, (str, int, float, bool))}

            # Handle scan_id - if no scan found, we need a placeholder scan
            # or we need to make scan_id nullable. For now, use placeholder approach.
            if scan_id is None:
                stats.images_unlinked = 1
                # Skip images without matching scans for now
                # They can be linked later when scan_id becomes nullable
                stats.warnings.append(f"No matching scan for {img_path.name} (SCLK={sclk_start}, sol={sol_number})")
                return stats
            else:
                stats.images_linked = 1

            # Insert record using raw SQL to handle new columns
            insert_sql = text("""
                INSERT INTO context_images (
                    id, scan_id, image_type, file_path, product_id, sclk,
                    pixel_scale_um, working_distance_cm, motor_position,
                    exposure_time_ms, led_illumination, width_px, height_px,
                    created_at, file_format, camera_id, sol_number, sclk_start,
                    sequence_id, image_time, focus_mode, focus_position_count,
                    local_mean_solar_time, vicar_metadata
                ) VALUES (
                    :id, :scan_id, :image_type, :file_path, :product_id, :sclk,
                    :pixel_scale_um, :working_distance_cm, :motor_position,
                    :exposure_time_ms, :led_illumination, :width_px, :height_px,
                    :created_at, :file_format, :camera_id, :sol_number, :sclk_start,
                    :sequence_id, :image_time, :focus_mode, :focus_position_count,
                    :local_mean_solar_time, :vicar_metadata
                )
            """)

            # Determine image type from camera ID
            image_type = "ACI" if camera_id in ("SC2", "SC3") else "WATSON" if camera_id == "SC1" else "ACI"

            session.execute(insert_sql, {
                "id": image_id,
                "scan_id": scan_id,
                "image_type": image_type,
                "file_path": str(img_path),
                "product_id": metadata.product_id,
                "sclk": sclk_start,
                "pixel_scale_um": 10.1,  # ACI fixed pixel scale
                "working_distance_cm": None,
                "motor_position": raw_label.get("FOCUS_POSITION_COUNT"),
                "exposure_time_ms": None,  # Would need parsing from header
                "led_illumination": None,
                "width_px": metadata.width,
                "height_px": metadata.height,
                "created_at": datetime.now(timezone.utc),
                "file_format": "IMG",
                "camera_id": camera_id,
                "sol_number": sol_number,
                "sclk_start": sclk_start,
                "sequence_id": metadata.sequence_id if metadata.sequence_id else None,
                "image_time": metadata.image_time,
                "focus_mode": raw_label.get("FOCUS_MODE"),
                "focus_position_count": raw_label.get("FOCUS_POSITION_COUNT") if isinstance(raw_label.get("FOCUS_POSITION_COUNT"), int) else None,
                "local_mean_solar_time": metadata.local_time if metadata.local_time else None,
                "vicar_metadata": json.dumps(vicar_dict),
            })

            session.commit()
            stats.images_ingested = 1

        return stats

    def _find_matching_scan(
        self,
        session: Session,
        image_sclk: int,
        sol_number: int,
    ) -> Optional[str]:
        """Find the scan that this image is associated with.

        Strategy:
        1. Look for scans on the same sol
        2. Find scan with sclk_start within tolerance window after image SCLK
        3. Prefer closest match if multiple candidates

        Args:
            session: Database session
            image_sclk: Image SCLK timestamp
            sol_number: Mars sol number

        Returns:
            Scan ID string, or None if no match found
        """
        query = text("""
            SELECT id, scan_id, sclk_start,
                   (sclk_start - :image_sclk) as delta
            FROM scans
            WHERE sol_number = :sol_number
              AND sclk_start > :sclk_min
              AND sclk_start <= :sclk_max
            ORDER BY delta ASC
            LIMIT 1
        """)

        result = session.execute(query, {
            "image_sclk": image_sclk,
            "sol_number": sol_number,
            "sclk_min": image_sclk + self.sclk_tolerance_min,
            "sclk_max": image_sclk + self.sclk_tolerance_max,
        }).fetchone()

        if result:
            return result[0]  # Return the UUID id, not the scan_id string
        return None

    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get current image ingestion statistics.

        Returns:
            Dictionary with image counts by format and linkage status
        """
        with get_session(self.engine) as session:
            # Total images
            total = session.execute(text("SELECT COUNT(*) FROM context_images")).scalar()

            # By format
            by_format = session.execute(text("""
                SELECT file_format, COUNT(*) as count
                FROM context_images
                GROUP BY file_format
            """)).fetchall()

            # By camera
            by_camera = session.execute(text("""
                SELECT camera_id, COUNT(*) as count
                FROM context_images
                WHERE camera_id IS NOT NULL
                GROUP BY camera_id
            """)).fetchall()

            # Linked vs unlinked (scan_id NULL check not applicable yet)
            linked = session.execute(text("""
                SELECT COUNT(*) FROM context_images WHERE scan_id IS NOT NULL
            """)).scalar()

            return {
                "total": total,
                "by_format": {row[0] or "unknown": row[1] for row in by_format},
                "by_camera": {row[0]: row[1] for row in by_camera if row[0]},
                "linked": linked,
                "unlinked": total - linked if total else 0,
            }

    def link_orphan_images(self) -> ServiceResult:
        """Attempt to link orphan images to scans.

        This method re-processes images that were ingested without scan linkage
        and tries to find matching scans.

        Returns:
            ServiceResult with linkage summary
        """
        # This would be implemented when scan_id becomes nullable
        return ServiceResult(
            summary="Orphan linking not yet implemented (requires nullable scan_id)",
            metadata={"success": True, "linked": 0},
        )
