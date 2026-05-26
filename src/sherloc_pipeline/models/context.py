"""
Context image and region of interest models for PHASE.

This module defines models for associated imagery and spatial annotations:
- ContextImage: ACI or WATSON context images associated with a scan
- RegionOfInterest: User-defined spatial groupings of scan points

Context images provide the visual reference for interpreting scan data,
while ROIs allow grouping of points with similar characteristics.

Example:
    >>> from sherloc_pipeline.models.context import (
    ...     ContextImage, RegionOfInterest, ImageType
    ... )
    >>>
    >>> image = ContextImage(
    ...     scan_id=scan.id,
    ...     image_type=ImageType.ACI,
    ...     file_path="/data/img/aci_921.png",
    ...     pixel_scale_um=10.1,
    ... )
"""

from enum import Enum
from typing import Optional, List
import uuid

from pydantic import Field, field_validator

from sherloc_pipeline.models.base import (
    IdentifiableModel,
    ModelRegistry,
)


class ImageType(str, Enum):
    """Type of context image.

    - ACI: Autofocus Context Imager (primary context for spectroscopy)
    - WATSON: Wide Angle Topographic Sensor for Operations and eNgineering
    """
    ACI = "ACI"
    WATSON = "WATSON"


@ModelRegistry.register
class ContextImage(IdentifiableModel):
    """ACI or WATSON context image associated with a scan.

    ContextImage stores metadata about context images that provide
    visual reference for spectroscopy scan points. ACI images are
    the primary context, showing exactly where laser spots are located.

    Attributes:
        scan_id: UUID of parent Scan
        image_type: Type of image (ACI or WATSON)
        file_path: Path to image file
        product_id: PDS product ID (if from PDS)
        sclk: Acquisition spacecraft clock
        pixel_scale_um: Micrometers per pixel
        working_distance_cm: Distance to target in cm
        motor_position: Focus motor position
        exposure_time_ms: Exposure time in milliseconds
        led_illumination: Whether LEDs were on
        width_px: Image width in pixels
        height_px: Image height in pixels

    Example:
        >>> image = ContextImage(
        ...     scan_id=scan.id,
        ...     image_type=ImageType.ACI,
        ...     file_path="/data/img/aci_921.png",
        ...     pixel_scale_um=10.1,
        ...     width_px=1648,
        ...     height_px=1200,
        ... )
        >>> image.pixel_scale_um
        10.1
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )
    image_type: ImageType = Field(
        description="Type of image (ACI or WATSON)"
    )
    file_path: str = Field(
        min_length=1,
        description="Path to image file"
    )
    product_id: Optional[str] = Field(
        default=None,
        description="PDS product ID"
    )
    pds_lidvid: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Full PDS4 LIDVID for the image product "
        "(e.g., 'urn:nasa:pds:mars2020_imgops:data_aci_imgops:...::1.0'). "
        "Constructed from RMO Image_name field during PDS ingestion."
    )
    sclk: Optional[int] = Field(
        default=None,
        ge=0,
        description="Acquisition spacecraft clock"
    )
    pixel_scale_um: Optional[float] = Field(
        default=None,
        gt=0,
        description="Micrometers per pixel"
    )
    working_distance_cm: Optional[float] = Field(
        default=None,
        gt=0,
        description="Distance to target in cm"
    )
    motor_position: Optional[int] = Field(
        default=None,
        description="Focus motor position"
    )
    exposure_time_ms: Optional[float] = Field(
        default=None,
        gt=0,
        description="Exposure time in milliseconds"
    )
    led_illumination: Optional[bool] = Field(
        default=None,
        description="Whether LEDs were on"
    )
    width_px: Optional[int] = Field(
        default=None,
        gt=0,
        description="Image width in pixels"
    )
    height_px: Optional[int] = Field(
        default=None,
        gt=0,
        description="Image height in pixels"
    )

    @property
    def aspect_ratio(self) -> Optional[float]:
        """Calculate image aspect ratio (width/height)."""
        if self.width_px and self.height_px:
            return self.width_px / self.height_px
        return None

    @property
    def total_pixels(self) -> Optional[int]:
        """Calculate total pixel count."""
        if self.width_px and self.height_px:
            return self.width_px * self.height_px
        return None

    @classmethod
    def from_loupe_metadata(
        cls,
        scan_id: uuid.UUID,
        file_path: str,
        image_type: ImageType,
        metadata: dict,
        **kwargs,
    ) -> "ContextImage":
        """Create ContextImage from Loupe metadata dictionary.

        Args:
            scan_id: UUID of parent Scan
            file_path: Path to image file
            image_type: Type of image
            metadata: Dictionary with image metadata
            **kwargs: Additional fields

        Returns:
            New ContextImage instance
        """
        return cls(
            scan_id=scan_id,
            image_type=image_type,
            file_path=file_path,
            product_id=metadata.get("product_ID"),
            sclk=cls._safe_int(metadata.get("sclk")),
            pixel_scale_um=cls._safe_float(metadata.get("pixel_scale")),
            working_distance_cm=cls._safe_float(metadata.get("range")),
            motor_position=cls._safe_int(metadata.get("motor_pos")),
            exposure_time_ms=cls._safe_float(metadata.get("exp_time")),
            led_illumination=cls._parse_bool(metadata.get("led_flag")),
            **kwargs,
        )

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert value to int."""
        if value in (None, "N/A", "None", ""):
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert value to float."""
        if value in (None, "N/A", "None", ""):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_bool(value) -> Optional[bool]:
        """Parse boolean value from various formats."""
        if value in (None, "N/A", "None", ""):
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return None


@ModelRegistry.register
class RegionOfInterest(IdentifiableModel):
    """User-defined region of interest grouping scan points.

    RegionOfInterest allows grouping of scan points that share
    common characteristics (e.g., same mineral phase, morphology).
    This is a Loupe-specific feature for interactive analysis.

    Attributes:
        scan_id: UUID of parent Scan
        name: Display name for the ROI
        color_hex: Color code for visualization (e.g., "#FF0000")
        point_indices: List of point indices in this ROI
        description: Optional description of the ROI

    Example:
        >>> roi = RegionOfInterest(
        ...     scan_id=scan.id,
        ...     name="Carbonate Vein",
        ...     color_hex="#00FF00",
        ...     point_indices=[5, 6, 7, 12, 13, 14],
        ... )
        >>> roi.n_points
        6
    """

    scan_id: uuid.UUID = Field(
        description="UUID of parent Scan"
    )
    name: str = Field(
        min_length=1,
        description="Display name for the ROI"
    )
    color_hex: str = Field(
        pattern=r"^#[0-9A-Fa-f]{6}$",
        description="Color code for visualization (e.g., '#FF0000')"
    )
    point_indices: List[int] = Field(
        min_length=1,
        description="List of point indices in this ROI"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the ROI"
    )

    @field_validator("point_indices")
    @classmethod
    def validate_point_indices(cls, v: List[int]) -> List[int]:
        """Validate that point indices are non-negative and unique."""
        if any(i < 0 for i in v):
            raise ValueError("Point indices must be non-negative")
        if len(v) != len(set(v)):
            raise ValueError("Point indices must be unique")
        return sorted(v)

    @property
    def n_points(self) -> int:
        """Number of points in this ROI."""
        return len(self.point_indices)

    @property
    def point_range(self) -> tuple:
        """Return (min, max) point indices."""
        if not self.point_indices:
            return (0, 0)
        return (min(self.point_indices), max(self.point_indices))

    def contains_point(self, point_index: int) -> bool:
        """Check if a point index is in this ROI.

        Args:
            point_index: Point index to check

        Returns:
            True if point is in ROI
        """
        return point_index in self.point_indices

    def overlaps_with(self, other: "RegionOfInterest") -> bool:
        """Check if this ROI overlaps with another.

        Args:
            other: Another RegionOfInterest

        Returns:
            True if any points are shared
        """
        return bool(set(self.point_indices) & set(other.point_indices))

    @classmethod
    def from_loupe_roi(
        cls,
        scan_id: uuid.UUID,
        name: str,
        color: str,
        points: List[int],
        **kwargs,
    ) -> "RegionOfInterest":
        """Create RegionOfInterest from Loupe roi.csv data.

        Args:
            scan_id: UUID of parent Scan
            name: ROI name from Loupe
            color: Color string (may need conversion to hex)
            points: List of point indices
            **kwargs: Additional fields

        Returns:
            New RegionOfInterest instance
        """
        # Ensure color is in hex format
        if not color.startswith("#"):
            # Try to parse as hex without #
            try:
                int(color, 16)
                color = f"#{color}"
            except ValueError:
                # Default to a safe color
                color = "#888888"

        return cls(
            scan_id=scan_id,
            name=name,
            color_hex=color.upper(),
            point_indices=points,
            **kwargs,
        )
