"""
SQLAlchemy ORM models for PHASE database.

This module defines SQLAlchemy ORM models that mirror the Pydantic models
in sherloc_pipeline.models. These models are used for database persistence
and querying.

Each ORM model includes:
- All fields from the corresponding Pydantic model
- Appropriate indexes for common query patterns
- Foreign key relationships with cascading deletes
- Methods for converting to/from Pydantic models

The models follow the unified schema defined in docs/schema/UNIFIED_SCHEMA.md.

Example:
    >>> from sherloc_pipeline.database.models import SolORM, ScanORM
    >>> from sherloc_pipeline.models import Sol, Scan
    >>>
    >>> # Create from Pydantic model
    >>> pydantic_sol = Sol(sol_number=921)
    >>> orm_sol = SolORM.from_pydantic(pydantic_sol)
    >>>
    >>> # Convert back to Pydantic
    >>> pydantic_again = orm_sol.to_pydantic()
"""

from datetime import datetime, date, timezone
from typing import Optional, List, Any, Dict
import uuid
import json

from sqlalchemy import (
    BigInteger,
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
    UniqueConstraint,
    Enum as SQLEnum,
    text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

from sherloc_pipeline.models import (
    Sol,
    Scan,
    ScanPoint,
    Spectrum,
    InstrumentState,
    CCDConfiguration,
    ScannerCalibration,
    ContextImage,
    RegionOfInterest,
    FittedPeak,
    DataSource,
    SpectralRegion,
    SpectrumType,
    ScanType,
    TargetType,
    classify_target_type,
    classify_scan_class,
    CoordinateFrame,
    ProcessingLevel,
    ImageType,
    PeakType,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
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


class SolORM(Base):
    """SQLAlchemy model for Sol (Martian day).

    Represents a single Martian sol during which SHERLOC observations
    were collected.
    """

    __tablename__ = "sols"

    sol_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    earth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    solar_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mission_phase: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    data_source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="loupe"
    )
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
    scans: Mapped[List["ScanORM"]] = relationship(
        "ScanORM",
        back_populates="sol",
        cascade="all, delete-orphan",
    )

    def to_pydantic(self) -> Sol:
        """Convert to Pydantic Sol model."""
        return Sol(
            sol_number=self.sol_number,
            earth_date=self.earth_date,
            solar_longitude=self.solar_longitude,
            mission_phase=self.mission_phase,
            data_source=DataSource(self.data_source),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, sol: Sol) -> "SolORM":
        """Create from Pydantic Sol model."""
        return cls(
            sol_number=sol.sol_number,
            earth_date=sol.earth_date,
            solar_longitude=sol.solar_longitude,
            mission_phase=sol.mission_phase,
            data_source=sol.data_source.value if isinstance(sol.data_source, DataSource) else sol.data_source,
            created_at=sol.created_at,
            updated_at=sol.updated_at,
        )


class ScanORM(Base):
    """SQLAlchemy model for Scan.

    Represents a complete spectroscopy scan of a target.
    """

    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    sol_number: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("sols.sol_number", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scan_name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    target: Mapped[Optional[str]] = mapped_column(String(200), nullable=True, index=True)
    scan_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    sclk_start: Mapped[int] = mapped_column(Integer, nullable=False)
    sclk_stop: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    n_channels: Mapped[int] = mapped_column(Integer, nullable=False, default=2148)
    shots_per_point: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    laser_wavelength_nm: Mapped[float] = mapped_column(Float, nullable=False, default=248.6)
    processing_applied: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    source_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    loupe_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    pds4_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    data_source: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    site_drive: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    sequence_id: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    scan_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    scan_class: Mapped[str] = mapped_column(
        String(20), nullable=False, default="primary"
    )
    parent_scan_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("scans.id", ondelete="SET NULL"), nullable=True
    )
    source_scan_ids: Mapped[Optional[dict]] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    processing_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    processing_config_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    processing_pipeline_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
    sol: Mapped["SolORM"] = relationship("SolORM", back_populates="scans")
    scan_points: Mapped[List["ScanPointORM"]] = relationship(
        "ScanPointORM",
        back_populates="scan",
        cascade="all, delete-orphan",
    )
    instrument_state: Mapped[Optional["InstrumentStateORM"]] = relationship(
        "InstrumentStateORM",
        back_populates="scan",
        uselist=False,
        cascade="all, delete-orphan",
    )
    ccd_configuration: Mapped[Optional["CCDConfigurationORM"]] = relationship(
        "CCDConfigurationORM",
        back_populates="scan",
        uselist=False,
        cascade="all, delete-orphan",
    )
    scanner_calibration: Mapped[Optional["ScannerCalibrationORM"]] = relationship(
        "ScannerCalibrationORM",
        back_populates="scan",
        uselist=False,
        cascade="all, delete-orphan",
    )
    context_images: Mapped[List["ContextImageORM"]] = relationship(
        "ContextImageORM",
        back_populates="scan",
        cascade="all, delete-orphan",
    )
    regions_of_interest: Mapped[List["RegionOfInterestORM"]] = relationship(
        "RegionOfInterestORM",
        back_populates="scan",
        cascade="all, delete-orphan",
    )
    spectrograms: Mapped[List["SpectrogramORM"]] = relationship(
        "SpectrogramORM",
        back_populates="scan",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_scans_sol_scan_name", "sol_number", "scan_name"),
        Index("ix_scans_sclk", "sclk_start"),
        Index("ix_scans_target_type", "target_type"),
        Index("ix_scans_scan_class", "scan_class"),
        # Note: target column already has index=True which creates ix_scans_target
    )

    def __init__(self, **kwargs):
        if "target_type" not in kwargs or kwargs.get("target_type") is None:
            kwargs["target_type"] = classify_target_type(
                kwargs.get("target"), kwargs.get("scan_name")
            )
        if "scan_class" not in kwargs or kwargs.get("scan_class") is None:
            kwargs["scan_class"] = classify_scan_class(kwargs.get("scan_name"))
        super().__init__(**kwargs)

    def to_pydantic(self) -> Scan:
        """Convert to Pydantic Scan model."""
        return Scan(
            id=_str_to_uuid(self.id),
            sol_number=self.sol_number,
            scan_name=self.scan_name,
            target=self.target,
            scan_id=self.scan_id,
            sclk_start=self.sclk_start,
            sclk_stop=self.sclk_stop,
            n_points=self.n_points,
            n_channels=self.n_channels,
            shots_per_point=self.shots_per_point,
            laser_wavelength_nm=self.laser_wavelength_nm,
            processing_applied=self.processing_applied,
            source_path=self.source_path,
            loupe_metadata=self.loupe_metadata,
            pds4_metadata=self.pds4_metadata,
            data_source=DataSource(self.data_source) if self.data_source else None,
            site_drive=self.site_drive,
            sequence_id=self.sequence_id,
            scan_type=ScanType(self.scan_type) if self.scan_type else None,
            target_type=TargetType(self.target_type) if self.target_type else None,
            scan_class=self.scan_class,
            parent_scan_id=_str_to_uuid(self.parent_scan_id) if self.parent_scan_id else None,
            source_scan_ids=self.source_scan_ids,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, scan: Scan) -> "ScanORM":
        """Create from Pydantic Scan model."""
        data_source = scan.data_source.value if isinstance(scan.data_source, DataSource) else scan.data_source
        scan_type = scan.scan_type.value if isinstance(scan.scan_type, ScanType) else scan.scan_type
        target_type = scan.target_type.value if isinstance(scan.target_type, TargetType) else scan.target_type

        return cls(
            id=_uuid_to_str(scan.id),
            sol_number=scan.sol_number,
            scan_name=scan.scan_name,
            target=scan.target,
            scan_id=scan.scan_id,
            sclk_start=scan.sclk_start,
            sclk_stop=scan.sclk_stop,
            n_points=scan.n_points,
            n_channels=scan.n_channels,
            shots_per_point=scan.shots_per_point,
            laser_wavelength_nm=scan.laser_wavelength_nm,
            processing_applied=scan.processing_applied,
            source_path=scan.source_path,
            loupe_metadata=scan.loupe_metadata,
            pds4_metadata=scan.pds4_metadata,
            data_source=data_source,
            site_drive=scan.site_drive,
            sequence_id=scan.sequence_id,
            scan_type=scan_type,
            target_type=target_type,
            # scan_class is omitted here so __init__ auto-classifies via
            # classify_scan_class().  Only pass it when explicitly non-default
            # (e.g. a manually-classified composite).
            **({"scan_class": scan.scan_class} if scan.scan_class != "primary" else {}),
            parent_scan_id=_uuid_to_str(scan.parent_scan_id) if scan.parent_scan_id else None,
            source_scan_ids=scan.source_scan_ids,
            created_at=scan.created_at,
            updated_at=scan.updated_at,
        )


class ScanPointORM(Base):
    """SQLAlchemy model for ScanPoint.

    Represents a single measurement point within a scan.
    """

    __tablename__ = "scan_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    point_index: Mapped[int] = mapped_column(Integer, nullable=False)
    azimuth_dn: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    elevation_dn: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    x_pixel: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y_pixel: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    azimuth_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    photodiode_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    photodiode_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    coordinate_frame: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="scan_points")
    spectra: Mapped[List["SpectrumORM"]] = relationship(
        "SpectrumORM",
        back_populates="scan_point",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_scan_points_scan_index", "scan_id", "point_index"),
    )

    def to_pydantic(self) -> ScanPoint:
        """Convert to Pydantic ScanPoint model."""
        return ScanPoint(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            point_index=self.point_index,
            azimuth_dn=self.azimuth_dn,
            elevation_dn=self.elevation_dn,
            x_pixel=self.x_pixel,
            y_pixel=self.y_pixel,
            azimuth_error=self.azimuth_error,
            elevation_error=self.elevation_error,
            photodiode_mean=self.photodiode_mean,
            photodiode_std=self.photodiode_std,
            coordinate_frame=CoordinateFrame(self.coordinate_frame) if self.coordinate_frame else None,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, point: ScanPoint) -> "ScanPointORM":
        """Create from Pydantic ScanPoint model."""
        coordinate_frame = point.coordinate_frame.value if isinstance(point.coordinate_frame, CoordinateFrame) else point.coordinate_frame

        return cls(
            id=_uuid_to_str(point.id),
            scan_id=_uuid_to_str(point.scan_id),
            point_index=point.point_index,
            azimuth_dn=point.azimuth_dn,
            elevation_dn=point.elevation_dn,
            x_pixel=point.x_pixel,
            y_pixel=point.y_pixel,
            azimuth_error=point.azimuth_error,
            elevation_error=point.elevation_error,
            photodiode_mean=point.photodiode_mean,
            photodiode_std=point.photodiode_std,
            coordinate_frame=coordinate_frame,
            created_at=point.created_at,
            updated_at=point.updated_at,
        )


class SpectrumORM(Base):
    """SQLAlchemy model for Spectrum.

    Represents a spectral measurement at one processing level.
    """

    __tablename__ = "spectra"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_point_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scan_points.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    region: Mapped[str] = mapped_column(String(10), nullable=False)
    spectrum_type: Mapped[str] = mapped_column(String(20), nullable=False)
    processing_level: Mapped[str] = mapped_column(String(20), nullable=False)
    intensities: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    wavelengths: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    wavenumbers: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    wavelength_source: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
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
    scan_point: Mapped["ScanPointORM"] = relationship(
        "ScanPointORM",
        back_populates="spectra"
    )
    fitted_peaks: Mapped[List["FittedPeakORM"]] = relationship(
        "FittedPeakORM",
        back_populates="spectrum",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_spectra_point_region_type", "scan_point_id", "region", "spectrum_type"),
    )

    def to_pydantic(self) -> Spectrum:
        """Convert to Pydantic Spectrum model."""
        return Spectrum(
            id=_str_to_uuid(self.id),
            scan_point_id=_str_to_uuid(self.scan_point_id),
            region=SpectralRegion(self.region),
            spectrum_type=SpectrumType(self.spectrum_type),
            processing_level=ProcessingLevel(self.processing_level),
            intensities=self.intensities,
            wavelengths=self.wavelengths,
            wavenumbers=self.wavenumbers,
            wavelength_source=self.wavelength_source,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, spectrum: Spectrum) -> "SpectrumORM":
        """Create from Pydantic Spectrum model."""
        region = spectrum.region.value if isinstance(spectrum.region, SpectralRegion) else spectrum.region
        spectrum_type = spectrum.spectrum_type.value if isinstance(spectrum.spectrum_type, SpectrumType) else spectrum.spectrum_type
        processing_level = spectrum.processing_level.value if isinstance(spectrum.processing_level, ProcessingLevel) else spectrum.processing_level

        return cls(
            id=_uuid_to_str(spectrum.id),
            scan_point_id=_uuid_to_str(spectrum.scan_point_id),
            region=region,
            spectrum_type=spectrum_type,
            processing_level=processing_level,
            intensities=spectrum.intensities,
            wavelengths=spectrum.wavelengths,
            wavenumbers=spectrum.wavenumbers,
            wavelength_source=spectrum.wavelength_source,
            created_at=spectrum.created_at,
            updated_at=spectrum.updated_at,
        )


class InstrumentStateORM(Base):
    """SQLAlchemy model for InstrumentState.

    Stores state-of-health telemetry for a scan.
    """

    __tablename__ = "instrument_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    ccd_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pcb_temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    laser_prt1_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    laser_prt2_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    laser_shot_counter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    laser_misfire_counter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    arc_event_counter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    voltage_1_2v: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voltage_3_3v: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voltage_5v_dac: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voltage_5v_adc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voltage_15v: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voltage_neg_15v: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    laser_int_time_us: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    laser_rep_rate_hz: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    laser_current_a: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    full_telemetry: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="instrument_state")

    def to_pydantic(self) -> InstrumentState:
        """Convert to Pydantic InstrumentState model."""
        return InstrumentState(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            ccd_temp_c=self.ccd_temp_c,
            pcb_temp_c=self.pcb_temp_c,
            laser_prt1_c=self.laser_prt1_c,
            laser_prt2_c=self.laser_prt2_c,
            laser_shot_counter=self.laser_shot_counter,
            laser_misfire_counter=self.laser_misfire_counter,
            arc_event_counter=self.arc_event_counter,
            voltage_1_2v=self.voltage_1_2v,
            voltage_3_3v=self.voltage_3_3v,
            voltage_5v_dac=self.voltage_5v_dac,
            voltage_5v_adc=self.voltage_5v_adc,
            voltage_15v=self.voltage_15v,
            voltage_neg_15v=self.voltage_neg_15v,
            laser_int_time_us=self.laser_int_time_us,
            laser_rep_rate_hz=self.laser_rep_rate_hz,
            laser_current_a=self.laser_current_a,
            full_telemetry=self.full_telemetry,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, state: InstrumentState) -> "InstrumentStateORM":
        """Create from Pydantic InstrumentState model."""
        return cls(
            id=_uuid_to_str(state.id),
            scan_id=_uuid_to_str(state.scan_id),
            ccd_temp_c=state.ccd_temp_c,
            pcb_temp_c=state.pcb_temp_c,
            laser_prt1_c=state.laser_prt1_c,
            laser_prt2_c=state.laser_prt2_c,
            laser_shot_counter=state.laser_shot_counter,
            laser_misfire_counter=state.laser_misfire_counter,
            arc_event_counter=state.arc_event_counter,
            voltage_1_2v=state.voltage_1_2v,
            voltage_3_3v=state.voltage_3_3v,
            voltage_5v_dac=state.voltage_5v_dac,
            voltage_5v_adc=state.voltage_5v_adc,
            voltage_15v=state.voltage_15v,
            voltage_neg_15v=state.voltage_neg_15v,
            laser_int_time_us=state.laser_int_time_us,
            laser_rep_rate_hz=state.laser_rep_rate_hz,
            laser_current_a=state.laser_current_a,
            full_telemetry=state.full_telemetry,
            created_at=state.created_at,
            updated_at=state.updated_at,
        )


class CCDConfigurationORM(Base):
    """SQLAlchemy model for CCDConfiguration.

    Stores CCD timing and region configuration.
    """

    __tablename__ = "ccd_configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    region_enable: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gain_2d: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mode_2d: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col1_low: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col1_high: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col2_low: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col2_high: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col3_low: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    vert_col3_high: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    horz_clock_lim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="ccd_configuration")

    def to_pydantic(self) -> CCDConfiguration:
        """Convert to Pydantic CCDConfiguration model."""
        return CCDConfiguration(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            region_enable=self.region_enable,
            gain_2d=self.gain_2d,
            mode_2d=self.mode_2d,
            vert_col1_low=self.vert_col1_low,
            vert_col1_high=self.vert_col1_high,
            vert_col2_low=self.vert_col2_low,
            vert_col2_high=self.vert_col2_high,
            vert_col3_low=self.vert_col3_low,
            vert_col3_high=self.vert_col3_high,
            horz_clock_lim=self.horz_clock_lim,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, config: CCDConfiguration) -> "CCDConfigurationORM":
        """Create from Pydantic CCDConfiguration model."""
        return cls(
            id=_uuid_to_str(config.id),
            scan_id=_uuid_to_str(config.scan_id),
            region_enable=config.region_enable,
            gain_2d=config.gain_2d,
            mode_2d=config.mode_2d,
            vert_col1_low=config.vert_col1_low,
            vert_col1_high=config.vert_col1_high,
            vert_col2_low=config.vert_col2_low,
            vert_col2_high=config.vert_col2_high,
            vert_col3_low=config.vert_col3_low,
            vert_col3_high=config.vert_col3_high,
            horz_clock_lim=config.horz_clock_lim,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )


class ScannerCalibrationORM(Base):
    """SQLAlchemy model for ScannerCalibration.

    Stores scanner coordinate calibration parameters.
    """

    __tablename__ = "scanner_calibrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    az_scale: Mapped[float] = mapped_column(Float, nullable=False)
    el_scale: Mapped[float] = mapped_column(Float, nullable=False)
    laser_x: Mapped[int] = mapped_column(Integer, nullable=False)
    laser_y: Mapped[int] = mapped_column(Integer, nullable=False)
    rotation_deg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="scanner_calibration")

    def to_pydantic(self) -> ScannerCalibration:
        """Convert to Pydantic ScannerCalibration model."""
        return ScannerCalibration(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            az_scale=self.az_scale,
            el_scale=self.el_scale,
            laser_x=self.laser_x,
            laser_y=self.laser_y,
            rotation_deg=self.rotation_deg,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, cal: ScannerCalibration) -> "ScannerCalibrationORM":
        """Create from Pydantic ScannerCalibration model."""
        return cls(
            id=_uuid_to_str(cal.id),
            scan_id=_uuid_to_str(cal.scan_id),
            az_scale=cal.az_scale,
            el_scale=cal.el_scale,
            laser_x=cal.laser_x,
            laser_y=cal.laser_y,
            rotation_deg=cal.rotation_deg,
            created_at=cal.created_at,
            updated_at=cal.updated_at,
        )


class ContextImageORM(Base):
    """SQLAlchemy model for ContextImage.

    Stores ACI or WATSON context images associated with a scan.
    Extended with VICAR metadata columns for raw IMG ingestion.
    """

    __tablename__ = "context_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_type: Mapped[str] = mapped_column(String(10), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    product_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sclk: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pixel_scale_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    working_distance_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    motor_position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exposure_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    led_illumination: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    width_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height_px: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pds_lidvid: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # New VICAR metadata columns (added in migration 0385ab87eb83)
    file_format: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    camera_id: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    sol_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    sclk_start: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    sclk_stop: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    sequence_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    image_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    focus_mode: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    focus_position_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    local_mean_solar_time: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    rover_motion_counter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vicar_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    source_img_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="context_images")

    __table_args__ = (
        Index("ix_context_images_sol", "sol_number"),
        Index("ix_context_images_sclk", "sclk_start"),
        Index("ix_context_images_format", "file_format"),
        Index("ix_context_images_camera", "camera_id"),
    )

    def to_pydantic(self) -> ContextImage:
        """Convert to Pydantic ContextImage model."""
        return ContextImage(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            image_type=ImageType(self.image_type),
            file_path=self.file_path,
            product_id=self.product_id,
            pds_lidvid=self.pds_lidvid,
            sclk=self.sclk,
            pixel_scale_um=self.pixel_scale_um,
            working_distance_cm=self.working_distance_cm,
            motor_position=self.motor_position,
            exposure_time_ms=self.exposure_time_ms,
            led_illumination=self.led_illumination,
            width_px=self.width_px,
            height_px=self.height_px,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, image: ContextImage) -> "ContextImageORM":
        """Create from Pydantic ContextImage model."""
        image_type = image.image_type.value if isinstance(image.image_type, ImageType) else image.image_type

        return cls(
            id=_uuid_to_str(image.id),
            scan_id=_uuid_to_str(image.scan_id),
            image_type=image_type,
            file_path=image.file_path,
            product_id=image.product_id,
            pds_lidvid=image.pds_lidvid,
            sclk=image.sclk,
            pixel_scale_um=image.pixel_scale_um,
            working_distance_cm=image.working_distance_cm,
            motor_position=image.motor_position,
            exposure_time_ms=image.exposure_time_ms,
            led_illumination=image.led_illumination,
            width_px=image.width_px,
            height_px=image.height_px,
            created_at=image.created_at,
            updated_at=image.updated_at,
        )


class RegionOfInterestORM(Base):
    """SQLAlchemy model for RegionOfInterest.

    Stores user-defined regions of interest grouping scan points.
    """

    __tablename__ = "regions_of_interest"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    color_hex: Mapped[str] = mapped_column(String(7), nullable=False)
    point_indices: Mapped[list] = mapped_column(JSON, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="regions_of_interest")

    def to_pydantic(self) -> RegionOfInterest:
        """Convert to Pydantic RegionOfInterest model."""
        return RegionOfInterest(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            name=self.name,
            color_hex=self.color_hex,
            point_indices=self.point_indices,
            description=self.description,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, roi: RegionOfInterest) -> "RegionOfInterestORM":
        """Create from Pydantic RegionOfInterest model."""
        return cls(
            id=_uuid_to_str(roi.id),
            scan_id=_uuid_to_str(roi.scan_id),
            name=roi.name,
            color_hex=roi.color_hex,
            point_indices=roi.point_indices,
            description=roi.description,
            created_at=roi.created_at,
            updated_at=roi.updated_at,
        )


class FittedPeakORM(Base):
    """SQLAlchemy model for FittedPeak.

    Stores peak fitting results from spectral analysis.
    """

    __tablename__ = "fitted_peaks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    spectrum_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("spectra.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    peak_type: Mapped[str] = mapped_column(String(20), nullable=False, default="gaussian")
    center_cm1: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    center_uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    amplitude: Mapped[float] = mapped_column(Float, nullable=False)
    amplitude_uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fwhm_cm1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fwhm_uncertainty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    area: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fit_quality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mineral_assignment: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    assignment_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fit_modality: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    center_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    fwhm_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_saturated: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
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
    spectrum: Mapped["SpectrumORM"] = relationship("SpectrumORM", back_populates="fitted_peaks")

    __table_args__ = (
        Index("ix_fitted_peaks_modality_assignment", "fit_modality", "mineral_assignment"),
    )

    def to_pydantic(self) -> FittedPeak:
        """Convert to Pydantic FittedPeak model."""
        return FittedPeak(
            id=_str_to_uuid(self.id),
            spectrum_id=_str_to_uuid(self.spectrum_id),
            peak_type=PeakType(self.peak_type),
            fit_modality=self.fit_modality,
            center_cm1=self.center_cm1,
            center_uncertainty=self.center_uncertainty,
            center_nm=self.center_nm,
            amplitude=self.amplitude,
            amplitude_uncertainty=self.amplitude_uncertainty,
            fwhm_cm1=self.fwhm_cm1,
            fwhm_uncertainty=self.fwhm_uncertainty,
            fwhm_nm=self.fwhm_nm,
            is_saturated=self.is_saturated,
            area=self.area,
            snr=self.snr,
            fit_quality=self.fit_quality,
            mineral_assignment=self.mineral_assignment,
            assignment_confidence=self.assignment_confidence,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, peak: FittedPeak) -> "FittedPeakORM":
        """Create from Pydantic FittedPeak model."""
        peak_type = peak.peak_type.value if isinstance(peak.peak_type, PeakType) else peak.peak_type

        return cls(
            id=_uuid_to_str(peak.id),
            spectrum_id=_uuid_to_str(peak.spectrum_id),
            peak_type=peak_type,
            fit_modality=peak.fit_modality,
            center_cm1=peak.center_cm1,
            center_uncertainty=peak.center_uncertainty,
            center_nm=peak.center_nm,
            amplitude=peak.amplitude,
            amplitude_uncertainty=peak.amplitude_uncertainty,
            fwhm_cm1=peak.fwhm_cm1,
            fwhm_uncertainty=peak.fwhm_uncertainty,
            fwhm_nm=peak.fwhm_nm,
            is_saturated=peak.is_saturated,
            area=peak.area,
            snr=peak.snr,
            fit_quality=peak.fit_quality,
            mineral_assignment=peak.mineral_assignment,
            assignment_confidence=peak.assignment_confidence,
            created_at=peak.created_at,
            updated_at=peak.updated_at,
        )


class SpectrogramORM(Base):
    """SQLAlchemy model for Spectrogram.

    Stores spectrogram visualization data: a 2D heatmap of spectral
    intensity across multiple measurement points in a scan.
    """

    __tablename__ = "spectrograms"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scan_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    region: Mapped[str] = mapped_column(String(10), nullable=False)
    processing_level: Mapped[str] = mapped_column(String(20), nullable=False)

    # Configuration stored as JSON
    config: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Intensity matrix as compressed binary
    intensity_matrix: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    n_channels: Mapped[int] = mapped_column(Integer, nullable=False)
    wavenumber_min: Mapped[float] = mapped_column(Float, nullable=False)
    wavenumber_max: Mapped[float] = mapped_column(Float, nullable=False)

    # Optional wavenumber array (compressed)
    wavenumbers: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # Optional metadata
    point_labels: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    point_indices: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    intensity_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

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
    scan: Mapped["ScanORM"] = relationship("ScanORM", back_populates="spectrograms")

    __table_args__ = (
        Index("ix_spectrograms_scan_region", "scan_id", "region"),
    )

    def to_pydantic(self) -> "Spectrogram":
        """Convert to Pydantic Spectrogram model."""
        from sherloc_pipeline.models.spectrogram import (
            Spectrogram,
            SpectrogramConfig,
            SpectrogramData,
        )

        # Reconstruct SpectrogramData
        data = SpectrogramData(
            intensity_matrix=self.intensity_matrix,
            n_points=self.n_points,
            n_channels=self.n_channels,
            wavenumber_min=self.wavenumber_min,
            wavenumber_max=self.wavenumber_max,
            wavenumbers=self.wavenumbers,
            point_labels=self.point_labels,
            intensity_min=self.intensity_min,
            intensity_max=self.intensity_max,
        )

        # Reconstruct config from JSON
        config = SpectrogramConfig(**self.config)

        return Spectrogram(
            id=_str_to_uuid(self.id),
            scan_id=_str_to_uuid(self.scan_id),
            region=SpectralRegion(self.region),
            processing_level=ProcessingLevel(self.processing_level),
            config=config,
            data=data,
            point_indices=self.point_indices,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_pydantic(cls, spectrogram: "Spectrogram") -> "SpectrogramORM":
        """Create from Pydantic Spectrogram model."""
        region = (
            spectrogram.region.value
            if isinstance(spectrogram.region, SpectralRegion)
            else spectrogram.region
        )
        processing_level = (
            spectrogram.processing_level.value
            if isinstance(spectrogram.processing_level, ProcessingLevel)
            else spectrogram.processing_level
        )

        return cls(
            id=_uuid_to_str(spectrogram.id),
            scan_id=_uuid_to_str(spectrogram.scan_id),
            region=region,
            processing_level=processing_level,
            config=spectrogram.config.model_dump(),
            intensity_matrix=spectrogram.data.intensity_matrix,
            n_points=spectrogram.data.n_points,
            n_channels=spectrogram.data.n_channels,
            wavenumber_min=spectrogram.data.wavenumber_min,
            wavenumber_max=spectrogram.data.wavenumber_max,
            wavenumbers=spectrogram.data.wavenumbers,
            point_labels=spectrogram.data.point_labels,
            point_indices=spectrogram.point_indices,
            intensity_min=spectrogram.data.intensity_min,
            intensity_max=spectrogram.data.intensity_max,
            title=spectrogram.title,
            created_at=spectrogram.created_at,
            updated_at=spectrogram.updated_at,
        )


class MapDisplayCoordinateORM(Base):
    """Cache table for resolved ACI pixel coordinates used in Map Mode display.

    Keyed by scan_point_id (one row per scan point). Populated on first access
    by resolve_display_coordinates() in core.coordinates, then reused on
    subsequent calls without touching the workspace files.

    transform_method distinguishes how the coordinate was produced:
    - 'identity': scan point already in aci_pixel frame; x_pixel/y_pixel copied directly
    - 'scanner_calibration': scanner_workspace coordinates transformed via
      load_spatial_table() + Loupe polynomial calibration
    """

    __tablename__ = "map_display_coordinates"

    scan_point_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scan_points.id", ondelete="CASCADE"),
        primary_key=True,
    )
    aci_x: Mapped[float] = mapped_column(Float, nullable=False)
    aci_y: Mapped[float] = mapped_column(Float, nullable=False)
    transform_method: Mapped[str] = mapped_column(String(30), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
    )


class UserORM(Base):
    """User identity for preference persistence.

    ``sub`` is the stable identity key from JWT (per spec §13.1 +
    B.12 F4). Auth0 access tokens always carry ``sub`` (e.g.,
    ``auth0|abc123``); CF Access tokens carry ``sub`` equal to the
    user email; Dev mode uses ``DevValidator.DEV_SUB``. ``email`` is
    optional metadata — populated whenever the validator surfaces it
    (CF Access always does; Auth0 only when the namespaced email
    claim is added to the §13.0.6 Action).
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sub: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(Text, unique=True, nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
    )
    last_seen_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
    )

    preferences: Mapped[List["UserPreferenceORM"]] = relationship(
        "UserPreferenceORM", back_populates="user", cascade="all, delete-orphan"
    )
    profiles: Mapped[List["ClassificationProfileORM"]] = relationship(
        "ClassificationProfileORM", back_populates="user", cascade="all, delete-orphan"
    )


class UserPreferenceORM(Base):
    """Key-value user preferences."""

    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)  # JSON-encoded
    updated_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"),
    )

    user: Mapped["UserORM"] = relationship("UserORM", back_populates="preferences")

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_user_preferences_user_key"),
    )


class ClassificationProfileORM(Base):
    """Custom peak classification profiles."""

    __tablename__ = "classification_profiles"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # UUID
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    profile_json: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Full ClassificationProfile as JSON
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)

    user: Mapped["UserORM"] = relationship("UserORM", back_populates="profiles")


class MapFitCacheORM(Base):
    """Ephemeral map fit results cache."""

    __tablename__ = "map_fit_cache"

    id: Mapped[str] = mapped_column(Text, primary_key=True)  # UUID
    scan_id: Mapped[str] = mapped_column(Text, nullable=False)
    domains: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array, sorted
    point_subset: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # JSON array or null (= all points)
    profile_hash: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # SHA-256 of classification profile
    profile_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    results_json: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Full per-point results
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    n_detections_json: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Per-domain detection counts
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # null = permanent; auto-save expires after 7 days
