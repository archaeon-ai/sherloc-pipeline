"""
SQLAlchemy ORM models for PIXL Pixlise database.

This module defines SQLAlchemy ORM models for PIXL XRF data from Pixlise exports.
These models mirror the Pydantic models in sherloc_pipeline.models.pixl.

The Pixlise database stores:
- Targets: PIXL observation metadata
- Quant points: AutoQuant-PDS quantification (16 oxides with errors and intensities)
- Beam locations: Pixel coordinates linking points to images
- Images: Context images from exports

Database location: /data/pixl/pixlise.db

Example:
    >>> from sherloc_pipeline.database.pixl_models import (
    ...     PixliseBase, PixliseTargetORM, PixliseQuantPointORM
    ... )
    >>> from sqlalchemy import create_engine
    >>>
    >>> engine = create_engine("sqlite:///data/pixl/pixlise.db")
    >>> PixliseBase.metadata.create_all(engine)

See Also:
    sherloc_pipeline.models.pixl: Pydantic models (validation layer)
    docs/seeds/pixlise-database-seed.md: Schema specification
"""

from datetime import datetime, date, timezone
from typing import Optional, List
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    LargeBinary,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from sherloc_pipeline.models.pixl import (
    PixliseTarget,
    PixliseQuantPoint,
    PixliseBeamLocation,
    PixliseImage,
    PixliseImageType,
)


class PixliseBase(DeclarativeBase):
    """Base class for all Pixlise ORM models."""
    pass


def _uuid_to_str(uid: uuid.UUID) -> str:
    """Convert UUID to string for storage."""
    return str(uid)


def _str_to_uuid(s: str) -> uuid.UUID:
    """Convert string to UUID."""
    return uuid.UUID(s)


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class PixliseTargetORM(PixliseBase):
    """SQLAlchemy model for Pixlise target.

    Represents a PIXL observation target from a Pixlise export.
    """

    __tablename__ = "pixlise_targets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_normalized: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    rtt: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    sol: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    export_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    piquant_version: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    detector_config: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    source_zip: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    quant_points: Mapped[List["PixliseQuantPointORM"]] = relationship(
        "PixliseQuantPointORM",
        back_populates="target",
        cascade="all, delete-orphan",
    )
    images: Mapped[List["PixliseImageORM"]] = relationship(
        "PixliseImageORM",
        back_populates="target",
        cascade="all, delete-orphan",
    )
    beam_locations: Mapped[List["PixliseBeamLocationORM"]] = relationship(
        "PixliseBeamLocationORM",
        back_populates="target",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_pixlise_targets_name_rtt", "name_normalized", "rtt"),
    )

    def to_pydantic(self) -> PixliseTarget:
        """Convert to Pydantic PixliseTarget model."""
        return PixliseTarget(
            id=_str_to_uuid(self.id),
            name=self.name,
            name_normalized=self.name_normalized,
            rtt=self.rtt,
            sol=self.sol,
            n_points=self.n_points,
            export_date=self.export_date,
            piquant_version=self.piquant_version,
            detector_config=self.detector_config,
            source_zip=self.source_zip,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, target: PixliseTarget) -> "PixliseTargetORM":
        """Create from Pydantic PixliseTarget model."""
        return cls(
            id=_uuid_to_str(target.id),
            name=target.name,
            name_normalized=target.name_normalized,
            rtt=target.rtt,
            sol=target.sol,
            n_points=target.n_points,
            export_date=target.export_date,
            piquant_version=target.piquant_version,
            detector_config=target.detector_config,
            source_zip=target.source_zip,
            created_at=target.created_at,
            updated_at=target.updated_at,
        )


class PixliseQuantPointORM(PixliseBase):
    """SQLAlchemy model for Pixlise quantified point.

    Stores XRF quantification results. Oxide data is stored as
    binary blobs (16 x float32 each) for efficiency.
    """

    __tablename__ = "pixlise_quant_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("pixlise_targets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pmc: Mapped[int] = mapped_column(Integer, nullable=False)

    # Oxide data as binary blobs (16 x float32 = 64 bytes each)
    oxide_wt_pct: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    oxide_err: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    oxide_intensity: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Instrument metadata
    total_counts: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    livetime: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    chisq: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_start: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_per_ch: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    res: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fit_iter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    events: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    triggers: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sclk: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_filename: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    target: Mapped["PixliseTargetORM"] = relationship(
        "PixliseTargetORM",
        back_populates="quant_points"
    )

    __table_args__ = (
        Index("ix_pixlise_quant_points_target_pmc", "target_id", "pmc"),
    )

    def to_pydantic(self) -> PixliseQuantPoint:
        """Convert to Pydantic PixliseQuantPoint model."""
        return PixliseQuantPoint(
            id=_str_to_uuid(self.id),
            target_id=self.target_id,
            pmc=self.pmc,
            oxide_wt_pct=self.oxide_wt_pct,
            oxide_err=self.oxide_err,
            oxide_intensity=self.oxide_intensity,
            total_counts=self.total_counts,
            livetime=self.livetime,
            chisq=self.chisq,
            ev_start=self.ev_start,
            ev_per_ch=self.ev_per_ch,
            res=self.res,
            fit_iter=self.fit_iter,
            events=self.events,
            triggers=self.triggers,
            sclk=self.sclk,
            source_filename=self.source_filename,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, point: PixliseQuantPoint) -> "PixliseQuantPointORM":
        """Create from Pydantic PixliseQuantPoint model."""
        return cls(
            id=_uuid_to_str(point.id),
            target_id=point.target_id,
            pmc=point.pmc,
            oxide_wt_pct=point.oxide_wt_pct,
            oxide_err=point.oxide_err,
            oxide_intensity=point.oxide_intensity,
            total_counts=point.total_counts,
            livetime=point.livetime,
            chisq=point.chisq,
            ev_start=point.ev_start,
            ev_per_ch=point.ev_per_ch,
            res=point.res,
            fit_iter=point.fit_iter,
            events=point.events,
            triggers=point.triggers,
            sclk=point.sclk,
            source_filename=point.source_filename,
            created_at=point.created_at,
            updated_at=point.updated_at,
        )


class PixliseImageORM(PixliseBase):
    """SQLAlchemy model for Pixlise context image."""

    __tablename__ = "pixlise_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("pixlise_targets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(200), nullable=False)
    image_type: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    file_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    target: Mapped["PixliseTargetORM"] = relationship(
        "PixliseTargetORM",
        back_populates="images"
    )
    beam_locations: Mapped[List["PixliseBeamLocationORM"]] = relationship(
        "PixliseBeamLocationORM",
        back_populates="image",
        cascade="all, delete-orphan",
    )

    def to_pydantic(self) -> PixliseImage:
        """Convert to Pydantic PixliseImage model."""
        return PixliseImage(
            id=_str_to_uuid(self.id),
            target_id=self.target_id,
            filename=self.filename,
            image_type=PixliseImageType(self.image_type),
            file_path=self.file_path,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, image: PixliseImage) -> "PixliseImageORM":
        """Create from Pydantic PixliseImage model."""
        image_type = (
            image.image_type.value
            if isinstance(image.image_type, PixliseImageType)
            else image.image_type
        )
        return cls(
            id=_uuid_to_str(image.id),
            target_id=image.target_id,
            filename=image.filename,
            image_type=image_type,
            file_path=image.file_path,
            created_at=image.created_at,
            updated_at=image.updated_at,
        )


class PixliseBeamLocationORM(PixliseBase):
    """SQLAlchemy model for Pixlise beam location."""

    __tablename__ = "pixlise_beam_locations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    target_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("pixlise_targets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("pixlise_images.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    pmc: Mapped[int] = mapped_column(Integer, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)
    pixel_i: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pixel_j: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    target: Mapped["PixliseTargetORM"] = relationship(
        "PixliseTargetORM",
        back_populates="beam_locations"
    )
    image: Mapped[Optional["PixliseImageORM"]] = relationship(
        "PixliseImageORM",
        back_populates="beam_locations"
    )

    __table_args__ = (
        Index("ix_pixlise_beam_locations_target_pmc", "target_id", "pmc"),
    )

    def to_pydantic(self) -> PixliseBeamLocation:
        """Convert to Pydantic PixliseBeamLocation model."""
        return PixliseBeamLocation(
            id=_str_to_uuid(self.id),
            target_id=self.target_id,
            image_id=self.image_id,
            pmc=self.pmc,
            x=self.x,
            y=self.y,
            z=self.z,
            pixel_i=self.pixel_i,
            pixel_j=self.pixel_j,
            image_name=self.image_name,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, loc: PixliseBeamLocation) -> "PixliseBeamLocationORM":
        """Create from Pydantic PixliseBeamLocation model."""
        return cls(
            id=_uuid_to_str(loc.id),
            target_id=loc.target_id,
            image_id=loc.image_id,
            pmc=loc.pmc,
            x=loc.x,
            y=loc.y,
            z=loc.z,
            pixel_i=loc.pixel_i,
            pixel_j=loc.pixel_j,
            image_name=loc.image_name,
            created_at=loc.created_at,
            updated_at=loc.updated_at,
        )
