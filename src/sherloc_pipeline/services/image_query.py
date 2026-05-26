"""
Image query service for SHERLOC ACI images.

This service provides functionality to query, retrieve, and export context images
from the PHASE database. It builds on top of the image ingestion service to provide
a comprehensive image access layer.

Key features:
- Query by sol number, scan ID, or SCLK range
- Query by camera type and file format
- Export images in PNG, TIFF formats
- Batch export functionality
- Thumbnail generation
- Image metadata retrieval

Example:
    >>> from sherloc_pipeline.services.image_query import ImageQueryService
    >>>
    >>> service = ImageQueryService()
    >>> images = service.query_by_sol(921)
    >>> print(f"Found {len(images)} images for sol 921")
    >>>
    >>> # Export images
    >>> service.export_images(images[:5], output_dir, format="png")
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Literal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import numpy as np
from numpy.typing import NDArray

from sqlalchemy import select, text, and_, or_
from sqlalchemy.orm import Session

from sherloc_pipeline.database import (
    get_engine,
    get_session,
    ScanORM,
    ScanPointORM,
    ContextImageORM,
)
from sherloc_pipeline.vision.img_reader import (
    read_aci_image,
    ACIImageMetadata,
)
from sherloc_pipeline.services.base import ServiceResult
from sherloc_pipeline.services.errors import SherlocServiceError


logger = logging.getLogger(__name__)


class ImageQueryError(SherlocServiceError):
    """Error during image query or export operations."""

    def __init__(self, message: str, query_details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.query_details = query_details or {}


@dataclass
class ImageInfo:
    """Information about a context image.

    This dataclass provides a comprehensive view of image metadata
    from the database, useful for query results and filtering.

    Attributes:
        id: Image UUID
        scan_id: Associated scan UUID
        sol_number: Mars sol number
        sclk_start: Spacecraft clock timestamp
        file_path: Path to image file
        file_format: File format (IMG or PNG)
        camera_id: Camera identifier (SC0, SC1, SC2, SC3)
        image_type: Image type (ACI or WATSON)
        product_id: PDS product identifier
        width_px: Image width in pixels
        height_px: Image height in pixels
        image_time: UTC timestamp of image capture
        focus_mode: Focus mode (MANUAL or AUTOFOCUS)
        local_time: Local Mean Solar Time string
        scan_target: Target name from associated scan
        scan_n_points: Number of points in associated scan
    """

    id: str
    scan_id: str
    sol_number: Optional[int]
    sclk_start: Optional[int]
    file_path: str
    file_format: Optional[str]
    camera_id: Optional[str]
    image_type: str
    product_id: Optional[str]
    width_px: Optional[int]
    height_px: Optional[int]
    image_time: Optional[datetime]
    focus_mode: Optional[str]
    local_time: Optional[str]
    scan_target: Optional[str] = None
    scan_n_points: Optional[int] = None

    @classmethod
    def from_orm(cls, image: ContextImageORM, scan: Optional[ScanORM] = None) -> "ImageInfo":
        """Create ImageInfo from ORM model."""
        return cls(
            id=image.id,
            scan_id=image.scan_id,
            sol_number=image.sol_number,
            sclk_start=image.sclk_start,
            file_path=image.file_path,
            file_format=image.file_format,
            camera_id=image.camera_id,
            image_type=image.image_type,
            product_id=image.product_id,
            width_px=image.width_px,
            height_px=image.height_px,
            image_time=image.image_time,
            focus_mode=image.focus_mode,
            local_time=image.local_mean_solar_time,
            scan_target=scan.scan_name if scan else None,
            scan_n_points=scan.n_points if scan else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "sol_number": self.sol_number,
            "sclk_start": self.sclk_start,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "camera_id": self.camera_id,
            "image_type": self.image_type,
            "product_id": self.product_id,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "image_time": self.image_time.isoformat() if self.image_time else None,
            "focus_mode": self.focus_mode,
            "local_time": self.local_time,
            "scan_target": self.scan_target,
            "scan_n_points": self.scan_n_points,
        }


@dataclass
class ExportStats:
    """Statistics from an export operation."""

    images_requested: int = 0
    images_exported: int = 0
    images_skipped: int = 0
    images_failed: int = 0
    total_bytes: int = 0
    errors: List[str] = field(default_factory=list)


class ImageQueryService:
    """Service for querying and exporting SHERLOC context images.

    This service provides methods for:
    - Querying images by sol, scan, SCLK, camera, or format
    - Exporting images to PNG or TIFF format
    - Generating thumbnails
    - Loading image data with metadata

    Attributes:
        console: Rich console for output (optional)
        engine: SQLAlchemy engine for database access

    Example:
        >>> service = ImageQueryService(database_path="./phase.db")
        >>>
        >>> # Query by sol
        >>> images = service.query_by_sol(921)
        >>> print(f"Found {len(images)} images")
        >>>
        >>> # Query by scan
        >>> images = service.query_by_scan("scan-uuid-here")
        >>>
        >>> # Export to directory
        >>> result = service.export_images(images, Path("/output"), format="png")
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        database_path: Optional[Path] = None,
    ):
        """Initialize the image query service.

        Args:
            console: Rich console for progress output
            database_path: Path to SQLite database (defaults to ./phase.db)
        """
        self.console = console or Console()

        if database_path is None:
            database_path = Path("./phase.db")

        self.database_path = database_path
        self.engine = get_engine(database_path)

    # ===========================================================================
    # Query Methods
    # ===========================================================================

    def query_by_sol(
        self,
        sol_number: int,
        file_format: Optional[str] = None,
        camera_id: Optional[str] = None,
        image_type: Optional[str] = None,
        include_scan_info: bool = True,
    ) -> List[ImageInfo]:
        """Query images by sol number.

        Args:
            sol_number: Mars sol number
            file_format: Filter by format (IMG or PNG)
            camera_id: Filter by camera (SC0, SC1, SC2, SC3)
            image_type: Filter by type (ACI or WATSON)
            include_scan_info: Include scan target/point info

        Returns:
            List of ImageInfo objects matching the query

        Example:
            >>> images = service.query_by_sol(921, file_format="IMG", camera_id="SC3")
        """
        with get_session(self.engine) as session:
            # Build query
            query = select(ContextImageORM).where(ContextImageORM.sol_number == sol_number)

            if file_format:
                query = query.where(ContextImageORM.file_format == file_format.upper())
            if camera_id:
                query = query.where(ContextImageORM.camera_id == camera_id.upper())
            if image_type:
                query = query.where(ContextImageORM.image_type == image_type.upper())

            query = query.order_by(ContextImageORM.sclk_start)

            images = session.execute(query).scalars().all()

            results = []
            for image in images:
                scan = None
                if include_scan_info and image.scan_id:
                    scan = session.get(ScanORM, image.scan_id)
                results.append(ImageInfo.from_orm(image, scan))

            return results

    def query_by_scan(
        self,
        scan_id: str,
        file_format: Optional[str] = None,
    ) -> List[ImageInfo]:
        """Query images associated with a specific scan.

        Args:
            scan_id: Scan UUID
            file_format: Filter by format (IMG or PNG)

        Returns:
            List of ImageInfo objects for the scan

        Example:
            >>> images = service.query_by_scan("abc123-...")
        """
        with get_session(self.engine) as session:
            query = select(ContextImageORM).where(ContextImageORM.scan_id == scan_id)

            if file_format:
                query = query.where(ContextImageORM.file_format == file_format.upper())

            query = query.order_by(ContextImageORM.sclk_start)

            images = session.execute(query).scalars().all()

            # Get scan info once
            scan = session.get(ScanORM, scan_id)

            return [ImageInfo.from_orm(image, scan) for image in images]

    def query_by_sclk_range(
        self,
        sclk_start: int,
        sclk_end: int,
        sol_number: Optional[int] = None,
        file_format: Optional[str] = None,
    ) -> List[ImageInfo]:
        """Query images within a SCLK timestamp range.

        Args:
            sclk_start: Start of SCLK range (inclusive)
            sclk_end: End of SCLK range (inclusive)
            sol_number: Optional sol filter for efficiency
            file_format: Filter by format (IMG or PNG)

        Returns:
            List of ImageInfo objects within the SCLK range

        Example:
            >>> images = service.query_by_sclk_range(748731000, 748732000)
        """
        with get_session(self.engine) as session:
            query = select(ContextImageORM).where(
                and_(
                    ContextImageORM.sclk_start >= sclk_start,
                    ContextImageORM.sclk_start <= sclk_end
                )
            )

            if sol_number:
                query = query.where(ContextImageORM.sol_number == sol_number)
            if file_format:
                query = query.where(ContextImageORM.file_format == file_format.upper())

            query = query.order_by(ContextImageORM.sclk_start)

            images = session.execute(query).scalars().all()

            results = []
            for image in images:
                scan = session.get(ScanORM, image.scan_id) if image.scan_id else None
                results.append(ImageInfo.from_orm(image, scan))

            return results

    def query_by_target(
        self,
        scan_name: str,
        file_format: Optional[str] = None,
        partial_match: bool = False,
    ) -> List[ImageInfo]:
        """Query images by scan name (from associated scan).

        Args:
            scan_name: Scan name to search for (e.g., 'detail_1')
            file_format: Filter by format (IMG or PNG)
            partial_match: If True, use LIKE matching

        Returns:
            List of ImageInfo objects for matching scan names

        Example:
            >>> images = service.query_by_target("detail", partial_match=True)
        """
        with get_session(self.engine) as session:
            # Join with scans to filter by scan_name
            if partial_match:
                scan_query = select(ScanORM.id).where(
                    ScanORM.scan_name.like(f"%{scan_name}%")
                )
            else:
                scan_query = select(ScanORM.id).where(ScanORM.scan_name == scan_name)

            scan_ids = session.execute(scan_query).scalars().all()

            if not scan_ids:
                return []

            query = select(ContextImageORM).where(
                ContextImageORM.scan_id.in_(scan_ids)
            )

            if file_format:
                query = query.where(ContextImageORM.file_format == file_format.upper())

            query = query.order_by(ContextImageORM.sclk_start)

            images = session.execute(query).scalars().all()

            # Build result with scan info
            results = []
            scan_cache = {}
            for image in images:
                if image.scan_id not in scan_cache:
                    scan_cache[image.scan_id] = session.get(ScanORM, image.scan_id)
                results.append(ImageInfo.from_orm(image, scan_cache[image.scan_id]))

            return results

    def query_all(
        self,
        file_format: Optional[str] = None,
        camera_id: Optional[str] = None,
        image_type: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ImageInfo]:
        """Query all images with optional filters.

        Args:
            file_format: Filter by format (IMG or PNG)
            camera_id: Filter by camera (SC0, SC1, SC2, SC3)
            image_type: Filter by type (ACI or WATSON)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of ImageInfo objects

        Example:
            >>> images = service.query_all(file_format="IMG", limit=100)
        """
        with get_session(self.engine) as session:
            query = select(ContextImageORM)

            if file_format:
                query = query.where(ContextImageORM.file_format == file_format.upper())
            if camera_id:
                query = query.where(ContextImageORM.camera_id == camera_id.upper())
            if image_type:
                query = query.where(ContextImageORM.image_type == image_type.upper())

            query = query.order_by(ContextImageORM.sol_number, ContextImageORM.sclk_start)

            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            images = session.execute(query).scalars().all()

            return [ImageInfo.from_orm(image) for image in images]

    def get_image_by_id(self, image_id: str) -> Optional[ImageInfo]:
        """Get a single image by ID.

        Args:
            image_id: Image UUID

        Returns:
            ImageInfo or None if not found
        """
        with get_session(self.engine) as session:
            image = session.get(ContextImageORM, image_id)
            if image:
                scan = session.get(ScanORM, image.scan_id) if image.scan_id else None
                return ImageInfo.from_orm(image, scan)
            return None

    def count_images(
        self,
        file_format: Optional[str] = None,
        camera_id: Optional[str] = None,
        sol_number: Optional[int] = None,
    ) -> int:
        """Count images matching criteria.

        Args:
            file_format: Filter by format
            camera_id: Filter by camera
            sol_number: Filter by sol

        Returns:
            Count of matching images
        """
        with get_session(self.engine) as session:
            sql = "SELECT COUNT(*) FROM context_images WHERE 1=1"
            params = {}

            if file_format:
                sql += " AND file_format = :format"
                params["format"] = file_format.upper()
            if camera_id:
                sql += " AND camera_id = :camera"
                params["camera"] = camera_id.upper()
            if sol_number:
                sql += " AND sol_number = :sol"
                params["sol"] = sol_number

            result = session.execute(text(sql), params).scalar()
            return result or 0

    def get_available_sols(self) -> List[int]:
        """Get list of sols that have images.

        Returns:
            Sorted list of sol numbers with images
        """
        with get_session(self.engine) as session:
            result = session.execute(text("""
                SELECT DISTINCT sol_number
                FROM context_images
                WHERE sol_number IS NOT NULL
                ORDER BY sol_number
            """)).fetchall()
            return [row[0] for row in result]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive image statistics.

        Returns:
            Dictionary with image counts, format distribution, etc.
        """
        with get_session(self.engine) as session:
            total = session.execute(
                text("SELECT COUNT(*) FROM context_images")
            ).scalar()

            by_format = session.execute(text("""
                SELECT file_format, COUNT(*) as count
                FROM context_images
                GROUP BY file_format
            """)).fetchall()

            by_camera = session.execute(text("""
                SELECT camera_id, COUNT(*) as count
                FROM context_images
                WHERE camera_id IS NOT NULL
                GROUP BY camera_id
            """)).fetchall()

            by_type = session.execute(text("""
                SELECT image_type, COUNT(*) as count
                FROM context_images
                GROUP BY image_type
            """)).fetchall()

            sol_count = session.execute(text("""
                SELECT COUNT(DISTINCT sol_number)
                FROM context_images
                WHERE sol_number IS NOT NULL
            """)).scalar()

            linked = session.execute(text("""
                SELECT COUNT(*) FROM context_images WHERE scan_id IS NOT NULL
            """)).scalar()

            return {
                "total": total,
                "by_format": {row[0] or "unknown": row[1] for row in by_format},
                "by_camera": {row[0]: row[1] for row in by_camera if row[0]},
                "by_type": {row[0]: row[1] for row in by_type},
                "sols_with_images": sol_count,
                "linked_to_scans": linked,
            }

    # ===========================================================================
    # Export Methods
    # ===========================================================================

    def export_images(
        self,
        images: List[ImageInfo],
        output_dir: Path,
        format: Literal["png", "tiff"] = "png",
        create_thumbnails: bool = False,
        thumbnail_size: Tuple[int, int] = (256, 256),
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> ServiceResult:
        """Export images to standard formats.

        Args:
            images: List of ImageInfo objects to export
            output_dir: Directory to write exported images
            format: Output format (png or tiff)
            create_thumbnails: Also create thumbnail images
            thumbnail_size: Thumbnail dimensions (width, height)
            overwrite: Overwrite existing files
            show_progress: Show progress bar

        Returns:
            ServiceResult with export summary

        Example:
            >>> images = service.query_by_sol(921)
            >>> result = service.export_images(images, Path("/output"), format="png")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if create_thumbnails:
            thumb_dir = output_dir / "thumbnails"
            thumb_dir.mkdir(exist_ok=True)

        stats = ExportStats(images_requested=len(images))
        exported_files = []

        def process_image(image_info: ImageInfo) -> Optional[Path]:
            """Process a single image for export."""
            src_path = Path(image_info.file_path)

            if not src_path.exists():
                stats.images_failed += 1
                stats.errors.append(f"Source not found: {src_path}")
                return None

            # Determine output filename
            base_name = src_path.stem
            if format == "png":
                out_name = f"{base_name}.png"
            else:
                out_name = f"{base_name}.tiff"

            out_path = output_dir / out_name

            if out_path.exists() and not overwrite:
                stats.images_skipped += 1
                return None

            try:
                # Handle different source formats
                if src_path.suffix.upper() == ".IMG":
                    # Read IMG file and convert
                    image_data, metadata = read_aci_image(src_path)
                    self._save_image(image_data, out_path, format)

                    if create_thumbnails:
                        thumb_path = thumb_dir / f"{base_name}_thumb.png"
                        self._save_thumbnail(image_data, thumb_path, thumbnail_size)

                elif src_path.suffix.upper() == ".PNG":
                    if format == "png":
                        # Just copy PNG
                        shutil.copy2(src_path, out_path)
                    else:
                        # Convert PNG to TIFF
                        from PIL import Image
                        img = Image.open(src_path)
                        img.save(out_path, format="TIFF")

                    if create_thumbnails:
                        from PIL import Image
                        thumb_path = thumb_dir / f"{base_name}_thumb.png"
                        img = Image.open(src_path)
                        img.thumbnail(thumbnail_size)
                        img.save(thumb_path, format="PNG")
                else:
                    stats.images_failed += 1
                    stats.errors.append(f"Unknown format: {src_path.suffix}")
                    return None

                stats.images_exported += 1
                stats.total_bytes += out_path.stat().st_size
                return out_path

            except Exception as e:
                stats.images_failed += 1
                stats.errors.append(f"{src_path.name}: {e}")
                logger.exception(f"Error exporting {src_path}")
                return None

        if show_progress and images:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Exporting to {format.upper()}...", total=len(images))

                for image_info in images:
                    result_path = process_image(image_info)
                    if result_path:
                        exported_files.append(result_path)
                    progress.advance(task)
        else:
            for image_info in images:
                result_path = process_image(image_info)
                if result_path:
                    exported_files.append(result_path)

        # Build summary
        summary = f"Exported {stats.images_exported} images to {format.upper()}"
        if stats.images_skipped:
            summary += f", skipped {stats.images_skipped} (already exist)"
        if stats.images_failed:
            summary += f", {stats.images_failed} failed"

        return ServiceResult(
            summary=summary,
            artifacts=exported_files,
            warnings=stats.errors[:10] if len(stats.errors) > 10 else stats.errors,
            metadata={
                "success": stats.images_failed == 0,
                "images_requested": stats.images_requested,
                "images_exported": stats.images_exported,
                "images_skipped": stats.images_skipped,
                "images_failed": stats.images_failed,
                "total_bytes": stats.total_bytes,
                "output_dir": str(output_dir),
                "format": format,
            },
        )

    def export_by_sol(
        self,
        sol_number: int,
        output_dir: Path,
        format: Literal["png", "tiff"] = "png",
        file_format: Optional[str] = None,
        **kwargs,
    ) -> ServiceResult:
        """Export all images for a sol.

        Args:
            sol_number: Sol number to export
            output_dir: Output directory
            format: Output format (png or tiff)
            file_format: Filter source format (IMG or PNG)
            **kwargs: Additional arguments for export_images

        Returns:
            ServiceResult with export summary
        """
        images = self.query_by_sol(sol_number, file_format=file_format)

        if not images:
            return ServiceResult(
                summary=f"No images found for sol {sol_number}",
                metadata={"success": True, "images_found": 0},
            )

        # Create sol-specific subdirectory
        sol_dir = output_dir / f"sol_{sol_number:04d}"

        return self.export_images(images, sol_dir, format=format, **kwargs)

    def _save_image(
        self,
        image_data: NDArray,
        output_path: Path,
        format: str,
    ) -> None:
        """Save numpy array as image file."""
        from PIL import Image

        # Ensure proper dtype for PIL
        if image_data.dtype != np.uint8:
            # Normalize to 0-255 if needed
            if image_data.max() > 255:
                image_data = ((image_data / image_data.max()) * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)

        img = Image.fromarray(image_data)

        if format == "png":
            img.save(output_path, format="PNG")
        elif format == "tiff":
            img.save(output_path, format="TIFF")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_thumbnail(
        self,
        image_data: NDArray,
        output_path: Path,
        size: Tuple[int, int],
    ) -> None:
        """Save a thumbnail version of the image."""
        from PIL import Image

        if image_data.dtype != np.uint8:
            if image_data.max() > 255:
                image_data = ((image_data / image_data.max()) * 255).astype(np.uint8)
            else:
                image_data = image_data.astype(np.uint8)

        img = Image.fromarray(image_data)
        img.thumbnail(size)
        img.save(output_path, format="PNG")

    # ===========================================================================
    # Image Loading Methods
    # ===========================================================================

    def load_image(
        self,
        image_info: ImageInfo,
    ) -> Tuple[NDArray, ACIImageMetadata]:
        """Load image data and metadata.

        Args:
            image_info: ImageInfo from a query result

        Returns:
            Tuple of (numpy array, ACIImageMetadata)

        Raises:
            ImageQueryError: If image cannot be loaded

        Example:
            >>> images = service.query_by_sol(921)
            >>> data, metadata = service.load_image(images[0])
        """
        path = Path(image_info.file_path)

        if not path.exists():
            raise ImageQueryError(
                f"Image file not found: {path}",
                {"image_id": image_info.id, "file_path": str(path)},
            )

        if path.suffix.upper() == ".IMG":
            return read_aci_image(path)
        elif path.suffix.upper() == ".PNG":
            from PIL import Image
            img = Image.open(path)
            data = np.array(img)

            # Create minimal metadata for PNG
            metadata = ACIImageMetadata(
                product_id=path.stem,
                sol=image_info.sol_number or 0,
                image_time=image_info.image_time,
                width=img.width,
                height=img.height,
                label_size=0,
                source_path=str(path),
            )
            return data, metadata
        else:
            raise ImageQueryError(
                f"Unsupported image format: {path.suffix}",
                {"image_id": image_info.id, "file_path": str(path)},
            )

    def load_image_by_id(
        self,
        image_id: str,
    ) -> Tuple[NDArray, ACIImageMetadata]:
        """Load image data by ID.

        Args:
            image_id: Image UUID

        Returns:
            Tuple of (numpy array, ACIImageMetadata)

        Raises:
            ImageQueryError: If image not found or cannot be loaded
        """
        image_info = self.get_image_by_id(image_id)

        if not image_info:
            raise ImageQueryError(
                f"Image not found: {image_id}",
                {"image_id": image_id},
            )

        return self.load_image(image_info)

    # ===========================================================================
    # Scan Point Overlay Methods
    # ===========================================================================

    def get_scan_points_for_image(
        self,
        image_info: ImageInfo,
    ) -> List[Dict[str, Any]]:
        """Get scan point coordinates for an image.

        Returns the pixel coordinates of scan points that can be
        overlaid on the image for visualization.

        Args:
            image_info: ImageInfo from a query result

        Returns:
            List of point dictionaries with x_pixel, y_pixel, point_index

        Example:
            >>> images = service.query_by_scan(scan_id)
            >>> points = service.get_scan_points_for_image(images[0])
            >>> for p in points:
            ...     print(f"Point {p['point_index']}: ({p['x_pixel']}, {p['y_pixel']})")
        """
        if not image_info.scan_id:
            return []

        with get_session(self.engine) as session:
            points = session.execute(
                select(ScanPointORM)
                .where(ScanPointORM.scan_id == image_info.scan_id)
                .order_by(ScanPointORM.point_index)
            ).scalars().all()

            result = []
            for p in points:
                if p.x_pixel is not None and p.y_pixel is not None:
                    result.append({
                        "point_index": p.point_index,
                        "x_pixel": p.x_pixel,
                        "y_pixel": p.y_pixel,
                        "azimuth_dn": p.azimuth_dn,
                        "elevation_dn": p.elevation_dn,
                    })

            return result

    def get_vicar_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get full VICAR metadata for an image.

        Args:
            image_id: Image UUID

        Returns:
            Dictionary of VICAR header fields, or None if not available
        """
        with get_session(self.engine) as session:
            image = session.get(ContextImageORM, image_id)

            if not image:
                return None

            if image.vicar_metadata:
                return image.vicar_metadata

            # Try to read from file if not in database
            if image.file_path and Path(image.file_path).suffix.upper() == ".IMG":
                try:
                    from sherloc_pipeline.vision.img_reader import get_raw_vicar_label
                    return get_raw_vicar_label(image.file_path)
                except Exception:
                    pass

            return None
