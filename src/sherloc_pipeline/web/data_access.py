"""Data access gating service for public vs internal mode.

In ``internal`` mode (default), all data is accessible. In ``public`` mode,
only explicitly allowlisted PDS4-sourced records are accessible. Missing,
null, and unrecognized sources are denied by the public-facing API.
An optional pinned release index also requires trusted release evidence.
"""

import logging
from typing import Literal

from sqlalchemy.orm import Query

from sherloc_pipeline.database.models import ScanORM
from sherloc_pipeline.services.public_release import (
    PublicReleaseIndex, has_public_release, public_release_expression,
)

logger = logging.getLogger(__name__)

PUBLIC_DATA_SOURCES = ("pds4",)


class DataAccessService:
    """Gates data queries based on the configured access mode.

    Args:
        access_mode: ``"internal"`` (all data) or ``"public"``
            (allowlisted PDS4 data only).
    """

    def __init__(
        self, access_mode: Literal["internal", "public"] = "internal",
        public_release_index: PublicReleaseIndex | None = None,
    ):
        if access_mode not in {"internal", "public"}:
            raise ValueError("unknown access mode")
        self.access_mode = access_mode
        self.public_release_index = public_release_index

    @property
    def is_public(self) -> bool:
        """True when operating in public (PDS-only) mode."""
        return self.access_mode == "public"

    def validate_scan_access(self, scan) -> None:
        """In public mode, raise 403 unless the scan's source is allowlisted.

        Missing, null, and unrecognized sources are denied. Internal mode
        permits every source.
        """
        if self.is_public and (
            getattr(scan, "data_source", None) not in PUBLIC_DATA_SOURCES
            or (
                self.public_release_index is not None
                and not has_public_release(scan, self.public_release_index)
            )
        ):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=403,
                detail="This resource is not available in public mode",
            )

    def filter_scans_query(self, query: Query) -> Query:
        """Apply access-mode filtering to a scans query.

        Public mode keeps only allowlisted sources, excluding null and
        unrecognized values. Internal mode returns the query unchanged.

        Args:
            query: SQLAlchemy query over :class:`ScanORM`.

        Returns:
            The (possibly filtered) query.
        """
        if self.is_public:
            query = query.filter(ScanORM.data_source.in_(PUBLIC_DATA_SOURCES))
            if self.public_release_index is not None:
                query = query.filter(public_release_expression(self.public_release_index))
        return query
