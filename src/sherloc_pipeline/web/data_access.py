"""Data access gating service for public vs internal mode.

In ``internal`` mode (default), all data is accessible. In ``public`` mode,
Loupe-sourced records are filtered out so that only PDS4-sourced data is
returned. This ensures that proprietary Loupe workspace data is not exposed
through the public-facing API.
"""

import logging
from typing import Literal

from sqlalchemy.orm import Query

from sherloc_pipeline.database.models import ScanORM

logger = logging.getLogger(__name__)


class DataAccessService:
    """Gates data queries based on the configured access mode.

    Args:
        access_mode: ``"internal"`` (all data) or ``"public"``
            (PDS-only, no Loupe data).
    """

    def __init__(self, access_mode: Literal["internal", "public"] = "internal"):
        self.access_mode = access_mode

    @property
    def is_public(self) -> bool:
        """True when operating in public (PDS-only) mode."""
        return self.access_mode == "public"

    def validate_scan_access(self, scan) -> None:
        """Raise HTTPException(403) if scan is Loupe-sourced and mode is public."""
        if self.is_public and getattr(scan, "data_source", None) == "loupe":
            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail="This resource is not available in public mode",
            )

    def filter_scans_query(self, query: Query) -> Query:
        """Apply access-mode filtering to a scans query.

        In public mode, excludes scans whose ``data_source`` is ``"loupe"``.

        Args:
            query: SQLAlchemy query over :class:`ScanORM`.

        Returns:
            The (possibly filtered) query.
        """
        if self.is_public:
            query = query.filter(ScanORM.data_source != "loupe")
        return query
