"""Service-to-DTO translation layer.

Converts between domain objects (numpy arrays, ORM models, dataclasses)
and Pydantic response schemas.  No HTTP/JSON concerns leak into services.
"""

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from sherloc_pipeline.database.models import ScanORM, ScanPointORM
from sherloc_pipeline.web.schemas import (
    PointItem,
    ScanDetail,
    ScanListItem,
)


def _format_dt(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime to ISO 8601 UTC with 'Z' suffix."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Treat naive as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def scan_orm_to_list_item(scan: ScanORM) -> ScanListItem:
    """Convert a ScanORM to a scan list DTO."""
    return ScanListItem(
        id=scan.id,
        sol_number=scan.sol_number,
        target=scan.target,
        scan_name=scan.scan_name,
        scan_id=scan.scan_id,
        n_points=scan.n_points,
        scan_class=getattr(scan, "scan_class", None),
        scan_type=getattr(scan, "scan_type", None),
        target_type=getattr(scan, "target_type", None),
        processing_status=getattr(scan, "processing_status", None),
        processed_at=_format_dt(getattr(scan, "processed_at", None)),
        processing_pipeline_version=getattr(scan, "processing_pipeline_version", None),
    )


def scan_orm_to_detail(scan: ScanORM) -> ScanDetail:
    """Convert a ScanORM to a full scan detail DTO."""
    return ScanDetail(
        id=scan.id,
        sol_number=scan.sol_number,
        target=scan.target,
        scan_name=scan.scan_name,
        scan_id=scan.scan_id,
        n_points=scan.n_points,
        n_channels=getattr(scan, "n_channels", None),
        shots_per_point=getattr(scan, "shots_per_point", None),
        laser_wavelength_nm=getattr(scan, "laser_wavelength_nm", None),
        scan_class=getattr(scan, "scan_class", None),
        scan_type=getattr(scan, "scan_type", None),
        target_type=getattr(scan, "target_type", None),
        data_source=getattr(scan, "data_source", None),
        site_drive=getattr(scan, "site_drive", None),
        sequence_id=getattr(scan, "sequence_id", None),
        parent_scan_id=getattr(scan, "parent_scan_id", None),
        source_scan_ids=getattr(scan, "source_scan_ids", None),
        processing_status=getattr(scan, "processing_status", None),
        processed_at=_format_dt(getattr(scan, "processed_at", None)),
        processing_pipeline_version=getattr(scan, "processing_pipeline_version", None),
        processing_config_hash=getattr(scan, "processing_config_hash", None),
        processing_error=getattr(scan, "processing_error", None),
        sclk_start=getattr(scan, "sclk_start", None),
        sclk_stop=getattr(scan, "sclk_stop", None),
        created_at=_format_dt(getattr(scan, "created_at", None)),
        updated_at=_format_dt(getattr(scan, "updated_at", None)),
    )


def point_orm_to_dto(point: ScanPointORM) -> PointItem:
    """Convert a ScanPointORM to a point DTO."""
    return PointItem(
        id=point.id,
        point_index=point.point_index,
        x_pixel=point.x_pixel,
        y_pixel=point.y_pixel,
        azimuth_dn=point.azimuth_dn,
        elevation_dn=point.elevation_dn,
        azimuth_error=point.azimuth_error,
        elevation_error=point.elevation_error,
        photodiode_mean=point.photodiode_mean,
        photodiode_std=point.photodiode_std,
        coordinate_frame=point.coordinate_frame,
    )


def numpy_to_list(arr: np.ndarray) -> List[float]:
    """Convert a numpy array to a plain Python list of floats."""
    return [float(v) for v in arr]
