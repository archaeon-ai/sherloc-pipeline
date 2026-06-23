"""
Segmentation Service for batch processing and database integration.

This service provides:
- Batch processing of ACI images with progress tracking
- Checkpoint/resume capability for long-running jobs
- Database storage of segmentation results
- GPU memory management for batch operations

Usage:
    from sherloc_pipeline.services.segmentation import SegmentationService

    service = SegmentationService()
    result = service.process_all_images(show_progress=True)
    print(result.summary)
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from sqlalchemy import text
from sqlalchemy.engine import Engine

from sherloc_pipeline.database import get_engine, get_session
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError
from sherloc_pipeline.vision.img_reader import read_aci_image
from sherloc_pipeline.vision.segmentation import (
    GrainSegmenter,
    GrainMask,
    SegmentationConfig,
    SegmentationResult,
    SegmentationModel,
    _normalize_model,
)


logger = logging.getLogger(__name__)


class SegmentationError(SherlocServiceError):
    """Error during segmentation processing."""

    def __init__(self, message: str, image_path: Optional[str] = None):
        super().__init__(message)
        self.image_path = image_path


@dataclass
class BatchStats:
    """Statistics from batch segmentation."""

    images_processed: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    total_grains: int = 0
    total_time_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def __add__(self, other: "BatchStats") -> "BatchStats":
        return BatchStats(
            images_processed=self.images_processed + other.images_processed,
            images_skipped=self.images_skipped + other.images_skipped,
            images_failed=self.images_failed + other.images_failed,
            total_grains=self.total_grains + other.total_grains,
            total_time_s=self.total_time_s + other.total_time_s,
            errors=self.errors + other.errors,
        )


# SQL for grain_segments table creation
CREATE_GRAIN_SEGMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS grain_segments (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    image_id VARCHAR(36) NOT NULL,

    -- Segment Identity
    segment_index INTEGER NOT NULL,
    segment_label VARCHAR(50),

    -- Bounding Box
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,

    -- Mask (for precise boundaries)
    mask_rle TEXT,

    -- Morphometry
    area_px INTEGER,
    perimeter_px FLOAT,
    aspect_ratio FLOAT,
    circularity FLOAT,
    centroid_x FLOAT,
    centroid_y FLOAT,

    -- Spectral Linkage
    linked_point_indices TEXT,

    -- Model Metadata
    model_name VARCHAR(100),
    confidence FLOAT,
    stability_score FLOAT,

    created_at DATETIME NOT NULL,

    FOREIGN KEY(image_id) REFERENCES context_images(id) ON DELETE CASCADE
);
"""

CREATE_GRAIN_SEGMENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_grain_segments_image ON grain_segments(image_id)",
    "CREATE INDEX IF NOT EXISTS idx_grain_segments_label ON grain_segments(segment_label)",
    "CREATE INDEX IF NOT EXISTS idx_grain_segments_area ON grain_segments(area_px)",
]

# SQL for segmentation_jobs table (checkpointing)
CREATE_SEGMENTATION_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS segmentation_jobs (
    id VARCHAR(36) NOT NULL PRIMARY KEY,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status VARCHAR(20) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    config_json TEXT NOT NULL,
    images_total INTEGER NOT NULL,
    images_processed INTEGER NOT NULL DEFAULT 0,
    images_failed INTEGER NOT NULL DEFAULT 0,
    total_grains INTEGER NOT NULL DEFAULT 0,
    last_image_id VARCHAR(36),
    error_log TEXT
);
"""


class SegmentationService:
    """Service for batch grain segmentation with database integration.

    Features:
    - Batch processing of all ingested ACI images
    - Progress tracking with ETA
    - Checkpoint/resume for long jobs
    - Automatic GPU memory management
    - Database storage of segmentation results

    Example:
        >>> service = SegmentationService()
        >>> result = service.process_all_images()
        >>> print(f"Processed {result.metadata['images_processed']} images")
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        database_path: Optional[Path] = None,
        segmentation_config: Optional[SegmentationConfig] = None,
    ):
        """Initialize the segmentation service.

        Args:
            console: Rich console for progress output
            database_path: Path to SQLite database
            segmentation_config: Configuration for the segmenter
        """
        self.console = console or Console()
        self.seg_config = segmentation_config or SegmentationConfig()

        if database_path is None:
            database_path = Path("./phase.db")

        self.database_path = database_path
        self.engine = get_engine(database_path)

        # Initialize schema
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure grain_segments and segmentation_jobs tables exist."""
        with get_session(self.engine) as session:
            session.execute(text(CREATE_GRAIN_SEGMENTS_TABLE))
            for index_sql in CREATE_GRAIN_SEGMENTS_INDEXES:
                session.execute(text(index_sql))
            session.execute(text(CREATE_SEGMENTATION_JOBS_TABLE))
            session.commit()
            logger.info("Segmentation schema ensured")

    def _get_images_to_process(
        self,
        session,
        resume_from: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of images to process.

        Args:
            session: Database session
            resume_from: Image ID to resume from (exclusive)
            limit: Maximum number of images

        Returns:
            List of image records with id, file_path
        """
        # Get images that haven't been segmented yet
        query = """
            SELECT ci.id, ci.file_path, ci.scan_id, ci.sol_number
            FROM context_images ci
            WHERE ci.file_format = 'IMG'
            AND NOT EXISTS (
                SELECT 1 FROM grain_segments gs WHERE gs.image_id = ci.id
            )
        """

        if resume_from:
            query += f" AND ci.id > '{resume_from}'"

        query += " ORDER BY ci.id"

        if limit:
            query += f" LIMIT {limit}"

        result = session.execute(text(query)).fetchall()

        return [
            {
                "id": row[0],
                "file_path": row[1],
                "scan_id": row[2],
                "sol_number": row[3],
            }
            for row in result
        ]

    def _store_grain(
        self,
        session,
        image_id: str,
        grain: GrainMask,
        model_name: str,
    ) -> str:
        """Store a single grain segment in the database.

        Returns:
            The grain segment ID
        """
        grain_id = str(uuid.uuid4())

        # Encode mask as RLE
        try:
            mask_rle = grain.to_rle()
        except ImportError:
            # pycocotools not available, skip RLE
            mask_rle = None

        insert_sql = text("""
            INSERT INTO grain_segments (
                id, image_id, segment_index, bbox_x, bbox_y, bbox_width, bbox_height,
                mask_rle, area_px, perimeter_px, aspect_ratio, circularity,
                centroid_x, centroid_y, model_name, confidence, stability_score, created_at
            ) VALUES (
                :id, :image_id, :segment_index, :bbox_x, :bbox_y, :bbox_width, :bbox_height,
                :mask_rle, :area_px, :perimeter_px, :aspect_ratio, :circularity,
                :centroid_x, :centroid_y, :model_name, :confidence, :stability_score, :created_at
            )
        """)

        session.execute(insert_sql, {
            "id": grain_id,
            "image_id": image_id,
            "segment_index": grain.segment_index,
            "bbox_x": grain.bbox[0],
            "bbox_y": grain.bbox[1],
            "bbox_width": grain.bbox[2],
            "bbox_height": grain.bbox[3],
            "mask_rle": mask_rle,
            "area_px": grain.area,
            "perimeter_px": grain.perimeter,
            "aspect_ratio": grain.aspect_ratio,
            "circularity": grain.circularity,
            "centroid_x": grain.centroid[0],
            "centroid_y": grain.centroid[1],
            "model_name": model_name,
            "confidence": grain.predicted_iou,
            "stability_score": grain.stability_score,
            "created_at": datetime.now(timezone.utc),
        })

        return grain_id

    def _create_job(self, session, total_images: int) -> str:
        """Create a new segmentation job for checkpointing."""
        job_id = str(uuid.uuid4())

        insert_sql = text("""
            INSERT INTO segmentation_jobs (
                id, started_at, status, model_name, config_json,
                images_total, images_processed, images_failed, total_grains
            ) VALUES (
                :id, :started_at, 'running', :model_name, :config_json,
                :images_total, 0, 0, 0
            )
        """)

        session.execute(insert_sql, {
            "id": job_id,
            "started_at": datetime.now(timezone.utc),
            "model_name": _normalize_model(self.seg_config.model),
            "config_json": self.seg_config.model_dump_json(),
            "images_total": total_images,
        })

        return job_id

    def _update_job(
        self,
        session,
        job_id: str,
        images_processed: int,
        images_failed: int,
        total_grains: int,
        last_image_id: Optional[str] = None,
        status: str = "running",
        error_log: Optional[str] = None,
    ) -> None:
        """Update job progress."""
        update_sql = text("""
            UPDATE segmentation_jobs SET
                images_processed = :images_processed,
                images_failed = :images_failed,
                total_grains = :total_grains,
                last_image_id = :last_image_id,
                status = :status,
                error_log = :error_log,
                completed_at = CASE WHEN :status IN ('completed', 'failed') THEN :now ELSE NULL END
            WHERE id = :job_id
        """)

        session.execute(update_sql, {
            "job_id": job_id,
            "images_processed": images_processed,
            "images_failed": images_failed,
            "total_grains": total_grains,
            "last_image_id": last_image_id,
            "status": status,
            "error_log": error_log,
            "now": datetime.now(timezone.utc),
        })

    def process_all_images(
        self,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 10,
        resume_job_id: Optional[str] = None,
    ) -> ServiceResult:
        """Process all ingested ACI images.

        Args:
            limit: Maximum number of images to process
            show_progress: Whether to show progress bar
            batch_size: Number of images to process before committing
            resume_job_id: Job ID to resume from (if any)

        Returns:
            ServiceResult with processing summary
        """
        # Initialize segmenter
        segmenter = GrainSegmenter(self.seg_config)

        stats = BatchStats()
        job_id = None

        try:
            with get_session(self.engine) as session:
                # Get images to process
                resume_from = None
                if resume_job_id:
                    # Get last processed image from job
                    result = session.execute(
                        text("SELECT last_image_id FROM segmentation_jobs WHERE id = :id"),
                        {"id": resume_job_id}
                    ).fetchone()
                    if result:
                        resume_from = result[0]

                images = self._get_images_to_process(session, resume_from, limit)

                if not images:
                    return ServiceResult(
                        summary="No images to process",
                        metadata={"success": True, "images_processed": 0},
                    )

                # Create job for tracking
                job_id = self._create_job(session, len(images))
                session.commit()

                if show_progress:
                    self.console.print(
                        f"[bold]Segmenting {len(images)} images with {_normalize_model(self.seg_config.model)}[/bold]"
                    )

                # Process images
                if show_progress:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeRemainingColumn(),
                        console=self.console,
                    ) as progress:
                        task = progress.add_task("Processing...", total=len(images))

                        for i, img_record in enumerate(images):
                            try:
                                img_stats = self._process_single_image(
                                    session, segmenter, img_record
                                )
                                stats = stats + img_stats

                                # Commit in batches
                                if (i + 1) % batch_size == 0:
                                    self._update_job(
                                        session, job_id,
                                        stats.images_processed,
                                        stats.images_failed,
                                        stats.total_grains,
                                        img_record["id"],
                                    )
                                    session.commit()

                            except Exception as e:
                                stats.images_failed += 1
                                stats.errors.append(f"{img_record['file_path']}: {e}")
                                logger.exception(f"Error processing {img_record['file_path']}")

                            progress.advance(task)
                else:
                    for i, img_record in enumerate(images):
                        try:
                            img_stats = self._process_single_image(
                                session, segmenter, img_record
                            )
                            stats = stats + img_stats

                            if (i + 1) % batch_size == 0:
                                self._update_job(
                                    session, job_id,
                                    stats.images_processed,
                                    stats.images_failed,
                                    stats.total_grains,
                                    img_record["id"],
                                )
                                session.commit()

                        except Exception as e:
                            stats.images_failed += 1
                            stats.errors.append(f"{img_record['file_path']}: {e}")
                            logger.exception(f"Error processing {img_record['file_path']}")

                # Final commit
                self._update_job(
                    session, job_id,
                    stats.images_processed,
                    stats.images_failed,
                    stats.total_grains,
                    status="completed" if not stats.errors else "completed_with_errors",
                    error_log=json.dumps(stats.errors[:100]) if stats.errors else None,
                )
                session.commit()

        finally:
            # Unload model to free GPU memory
            segmenter.unload_model()

        # Build summary
        summary = (
            f"Segmented {stats.images_processed} images, "
            f"found {stats.total_grains} grains"
        )
        if stats.images_failed:
            summary += f" ({stats.images_failed} failed)"

        return ServiceResult(
            summary=summary,
            warnings=[f"Errors: {len(stats.errors)}"] if stats.errors else [],
            metadata={
                "success": stats.images_failed == 0,
                "job_id": job_id,
                "images_processed": stats.images_processed,
                "images_failed": stats.images_failed,
                "total_grains": stats.total_grains,
                "total_time_s": stats.total_time_s,
                "model": _normalize_model(self.seg_config.model),
                "errors": stats.errors[:20],  # First 20 errors
            },
        )

    def _process_single_image(
        self,
        session,
        segmenter: GrainSegmenter,
        img_record: Dict[str, Any],
    ) -> BatchStats:
        """Process a single image and store results.

        Returns:
            BatchStats for this image
        """
        stats = BatchStats()
        image_id = img_record["id"]
        file_path = img_record["file_path"]

        # Read image
        image, metadata = read_aci_image(file_path, validate_dimensions=False)

        # Segment
        result = segmenter.segment(
            image,
            image_id=image_id,
            image_path=file_path,
            compute_morphometry=True,
        )

        # Store grains
        for grain in result.grains:
            self._store_grain(session, image_id, grain, result.model_name)

        stats.images_processed = 1
        stats.total_grains = result.n_grains
        stats.total_time_s = result.inference_time_s

        return stats

    def process_image(
        self,
        image_id: str,
        force: bool = False,
    ) -> ServiceResult:
        """Process a single image by ID.

        Args:
            image_id: Database ID of the image
            force: If True, re-segment even if already processed

        Returns:
            ServiceResult with segmentation summary
        """
        segmenter = GrainSegmenter(self.seg_config)

        try:
            with get_session(self.engine) as session:
                # Get image record
                result = session.execute(
                    text("SELECT file_path FROM context_images WHERE id = :id"),
                    {"id": image_id}
                ).fetchone()

                if not result:
                    raise SegmentationError(f"Image not found: {image_id}")

                file_path = result[0]

                # Check if already processed
                if not force:
                    existing = session.execute(
                        text("SELECT COUNT(*) FROM grain_segments WHERE image_id = :id"),
                        {"id": image_id}
                    ).scalar()

                    if existing:
                        return ServiceResult(
                            summary=f"Image already segmented ({existing} grains)",
                            metadata={"success": True, "skipped": True, "grains": existing},
                        )

                # Delete existing if force mode
                if force:
                    session.execute(
                        text("DELETE FROM grain_segments WHERE image_id = :id"),
                        {"id": image_id}
                    )

                # Process
                image, metadata = read_aci_image(file_path, validate_dimensions=False)
                seg_result = segmenter.segment(
                    image,
                    image_id=image_id,
                    image_path=file_path,
                    compute_morphometry=True,
                )

                # Store
                for grain in seg_result.grains:
                    self._store_grain(session, image_id, grain, seg_result.model_name)

                session.commit()

                return ServiceResult(
                    summary=f"Segmented {seg_result.n_grains} grains in {seg_result.inference_time_s:.2f}s",
                    metadata={
                        "success": True,
                        "grains": seg_result.n_grains,
                        "time_s": seg_result.inference_time_s,
                        "model": seg_result.model_name,
                    },
                )

        finally:
            segmenter.unload_model()

    def get_stats(self) -> Dict[str, Any]:
        """Get segmentation statistics.

        Returns:
            Dictionary with grain counts and coverage
        """
        with get_session(self.engine) as session:
            # Total grains
            total_grains = session.execute(
                text("SELECT COUNT(*) FROM grain_segments")
            ).scalar()

            # Images with segments
            images_with_segments = session.execute(
                text("SELECT COUNT(DISTINCT image_id) FROM grain_segments")
            ).scalar()

            # Total images
            total_images = session.execute(
                text("SELECT COUNT(*) FROM context_images WHERE file_format = 'IMG'")
            ).scalar()

            # Area stats
            area_stats = session.execute(text("""
                SELECT
                    MIN(area_px) as min_area,
                    MAX(area_px) as max_area,
                    AVG(area_px) as avg_area,
                    SUM(area_px) as total_area
                FROM grain_segments
            """)).fetchone()

            # By model
            by_model = session.execute(text("""
                SELECT model_name, COUNT(*) as count
                FROM grain_segments
                GROUP BY model_name
            """)).fetchall()

            return {
                "total_grains": total_grains,
                "images_with_segments": images_with_segments,
                "total_images": total_images,
                "coverage_pct": (images_with_segments / total_images * 100) if total_images else 0,
                "area_stats": {
                    "min": area_stats[0],
                    "max": area_stats[1],
                    "avg": area_stats[2],
                    "total": area_stats[3],
                } if area_stats[0] else {},
                "by_model": {row[0]: row[1] for row in by_model},
            }

    def get_grains_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """Get all grains for an image.

        Args:
            image_id: Database ID of the image

        Returns:
            List of grain records
        """
        with get_session(self.engine) as session:
            result = session.execute(text("""
                SELECT
                    id, segment_index, bbox_x, bbox_y, bbox_width, bbox_height,
                    area_px, perimeter_px, aspect_ratio, circularity,
                    centroid_x, centroid_y, confidence, stability_score
                FROM grain_segments
                WHERE image_id = :image_id
                ORDER BY segment_index
            """), {"image_id": image_id}).fetchall()

            return [
                {
                    "id": row[0],
                    "segment_index": row[1],
                    "bbox": [row[2], row[3], row[4], row[5]],
                    "area_px": row[6],
                    "perimeter_px": row[7],
                    "aspect_ratio": row[8],
                    "circularity": row[9],
                    "centroid": [row[10], row[11]],
                    "confidence": row[12],
                    "stability_score": row[13],
                }
                for row in result
            ]
