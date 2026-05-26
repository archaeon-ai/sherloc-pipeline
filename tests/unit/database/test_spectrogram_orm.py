"""
Unit tests for SpectrogramORM database model (bd-3mx: WS4-B).

Tests the SQLAlchemy ORM model for spectrograms:
- Create from Pydantic model
- Convert back to Pydantic model
- Database persistence and retrieval
"""

import uuid

import numpy as np
import pytest

from sherloc_pipeline.database import (
    get_engine,
    create_all_tables,
    get_session,
    SolORM,
    ScanORM,
    SpectrogramORM,
)
from sherloc_pipeline.models import (
    SpectralRegion,
    ProcessingLevel,
)
from sherloc_pipeline.models.spectrogram import (
    ColorMapType,
    NormalizationType,
    SpectrogramConfig,
    SpectrogramData,
    Spectrogram,
)


@pytest.fixture
def in_memory_engine():
    """Create an in-memory SQLite database."""
    engine = get_engine(":memory:")
    create_all_tables(engine)
    return engine


@pytest.fixture
def sample_spectrogram():
    """Create a sample Spectrogram model."""
    matrix = np.random.rand(10, 50).astype(np.float32)

    data = SpectrogramData(
        intensity_matrix=SpectrogramData.compress_matrix(matrix),
        n_points=10,
        n_channels=50,
        wavenumber_min=200.0,
        wavenumber_max=4000.0,
        point_labels=[f"p{i}" for i in range(10)],
        intensity_min=float(matrix.min()),
        intensity_max=float(matrix.max()),
    )

    config = SpectrogramConfig(
        colormap=ColorMapType.PLASMA,
        normalization=NormalizationType.GLOBAL,
    )

    return Spectrogram(
        scan_id=uuid.uuid4(),
        region=SpectralRegion.R1,
        processing_level=ProcessingLevel.NORMALIZED,
        config=config,
        data=data,
        point_indices=list(range(10)),
        title="Test Spectrogram",
    )


class TestSpectrogramORMConversion:
    """Tests for SpectrogramORM conversion methods."""

    def test_from_pydantic(self, sample_spectrogram):
        """Create ORM model from Pydantic model."""
        orm = SpectrogramORM.from_pydantic(sample_spectrogram)

        assert orm.id == str(sample_spectrogram.id)
        assert orm.scan_id == str(sample_spectrogram.scan_id)
        assert orm.region == "R1"
        assert orm.processing_level == "normalized"
        assert orm.n_points == 10
        assert orm.n_channels == 50
        assert orm.wavenumber_min == 200.0
        assert orm.wavenumber_max == 4000.0
        assert orm.title == "Test Spectrogram"
        assert orm.point_indices == list(range(10))
        assert orm.point_labels == [f"p{i}" for i in range(10)]

    def test_to_pydantic(self, sample_spectrogram):
        """Convert ORM model back to Pydantic."""
        orm = SpectrogramORM.from_pydantic(sample_spectrogram)
        pydantic = orm.to_pydantic()

        assert pydantic.id == sample_spectrogram.id
        assert pydantic.scan_id == sample_spectrogram.scan_id
        # Region and processing_level may be strings due to use_enum_values
        assert str(pydantic.region) == "R1" or pydantic.region == SpectralRegion.R1
        assert pydantic.data.n_points == 10
        assert pydantic.data.n_channels == 50
        assert pydantic.title == "Test Spectrogram"

    def test_config_roundtrip(self, sample_spectrogram):
        """Config survives ORM conversion."""
        orm = SpectrogramORM.from_pydantic(sample_spectrogram)
        pydantic = orm.to_pydantic()

        # Config should be preserved
        assert pydantic.config.colormap == ColorMapType.PLASMA
        assert pydantic.config.normalization == NormalizationType.GLOBAL

    def test_intensity_matrix_roundtrip(self, sample_spectrogram):
        """Intensity matrix survives ORM conversion."""
        original_matrix = sample_spectrogram.data.get_intensity_matrix()

        orm = SpectrogramORM.from_pydantic(sample_spectrogram)
        pydantic = orm.to_pydantic()
        restored_matrix = pydantic.data.get_intensity_matrix()

        np.testing.assert_array_almost_equal(original_matrix, restored_matrix)


class TestSpectrogramORMPersistence:
    """Tests for SpectrogramORM database persistence."""

    def test_save_and_retrieve(self, in_memory_engine, sample_spectrogram):
        """Save and retrieve spectrogram from database."""
        with get_session(in_memory_engine) as session:
            # First create sol and scan (foreign key dependencies)
            sol = SolORM(sol_number=921)
            session.add(sol)
            session.flush()

            scan = ScanORM(
                id=str(sample_spectrogram.scan_id),
                sol_number=921,
                scan_name="Test_Target",
                scan_id="test_scan_001",
                sclk_start=672000000,
                n_points=100,
                shots_per_point=10,
            )
            session.add(scan)
            session.flush()

            # Now save spectrogram
            orm = SpectrogramORM.from_pydantic(sample_spectrogram)
            session.add(orm)
            session.flush()

            # Retrieve
            retrieved = session.query(SpectrogramORM).filter_by(
                id=str(sample_spectrogram.id)
            ).first()

            assert retrieved is not None
            assert retrieved.id == str(sample_spectrogram.id)
            assert retrieved.n_points == 10
            assert retrieved.n_channels == 50

    def test_scan_relationship(self, in_memory_engine, sample_spectrogram):
        """Spectrogram has relationship to scan."""
        with get_session(in_memory_engine) as session:
            # Create sol and scan
            sol = SolORM(sol_number=921)
            session.add(sol)
            session.flush()

            scan = ScanORM(
                id=str(sample_spectrogram.scan_id),
                sol_number=921,
                scan_name="Test_Target",
                scan_id="test_scan_001",
                sclk_start=672000000,
                n_points=100,
                shots_per_point=10,
            )
            session.add(scan)
            session.flush()

            # Save spectrogram
            orm = SpectrogramORM.from_pydantic(sample_spectrogram)
            session.add(orm)
            session.flush()

            # Check relationship
            retrieved = session.query(SpectrogramORM).first()
            assert retrieved.scan is not None
            assert retrieved.scan.scan_name == "Test_Target"

    def test_cascade_delete(self, in_memory_engine, sample_spectrogram):
        """Spectrogram is deleted when scan is deleted."""
        with get_session(in_memory_engine) as session:
            # Create sol and scan
            sol = SolORM(sol_number=921)
            session.add(sol)
            session.flush()

            scan = ScanORM(
                id=str(sample_spectrogram.scan_id),
                sol_number=921,
                scan_name="Test_Target",
                scan_id="test_scan_001",
                sclk_start=672000000,
                n_points=100,
                shots_per_point=10,
            )
            session.add(scan)
            session.flush()

            # Save spectrogram
            orm = SpectrogramORM.from_pydantic(sample_spectrogram)
            session.add(orm)
            session.flush()

            # Verify spectrogram exists
            assert session.query(SpectrogramORM).count() == 1

            # Delete scan
            session.delete(scan)
            session.flush()

            # Spectrogram should be cascade deleted
            assert session.query(SpectrogramORM).count() == 0

    def test_multiple_spectrograms_per_scan(self, in_memory_engine, sample_spectrogram):
        """Multiple spectrograms can be associated with one scan."""
        with get_session(in_memory_engine) as session:
            # Create sol and scan
            sol = SolORM(sol_number=921)
            session.add(sol)
            session.flush()

            scan = ScanORM(
                id=str(sample_spectrogram.scan_id),
                sol_number=921,
                scan_name="Test_Target",
                scan_id="test_scan_001",
                sclk_start=672000000,
                n_points=100,
                shots_per_point=10,
            )
            session.add(scan)
            session.flush()

            # Create multiple spectrograms
            matrix = np.random.rand(10, 50).astype(np.float32)
            data = SpectrogramData(
                intensity_matrix=SpectrogramData.compress_matrix(matrix),
                n_points=10,
                n_channels=50,
                wavenumber_min=200.0,
                wavenumber_max=4000.0,
            )

            for region in [SpectralRegion.R1, SpectralRegion.R2, SpectralRegion.R3]:
                spec = Spectrogram(
                    scan_id=sample_spectrogram.scan_id,
                    region=region,
                    processing_level=ProcessingLevel.NORMALIZED,
                    data=data,
                )
                orm = SpectrogramORM.from_pydantic(spec)
                session.add(orm)

            session.flush()

            # Verify all three spectrograms exist
            assert session.query(SpectrogramORM).count() == 3

            # Check scan relationship
            scan_orm = session.query(ScanORM).first()
            assert len(scan_orm.spectrograms) == 3


class TestSpectrogramORMImport:
    """Tests for SpectrogramORM import."""

    def test_import_from_database_package(self):
        """SpectrogramORM can be imported from database package."""
        from sherloc_pipeline.database import SpectrogramORM
        assert SpectrogramORM is not None

    def test_orm_has_required_attributes(self):
        """SpectrogramORM has required attributes."""
        assert hasattr(SpectrogramORM, "id")
        assert hasattr(SpectrogramORM, "scan_id")
        assert hasattr(SpectrogramORM, "region")
        assert hasattr(SpectrogramORM, "processing_level")
        assert hasattr(SpectrogramORM, "config")
        assert hasattr(SpectrogramORM, "intensity_matrix")
        assert hasattr(SpectrogramORM, "n_points")
        assert hasattr(SpectrogramORM, "n_channels")
        assert hasattr(SpectrogramORM, "wavenumber_min")
        assert hasattr(SpectrogramORM, "wavenumber_max")
        assert hasattr(SpectrogramORM, "to_pydantic")
        assert hasattr(SpectrogramORM, "from_pydantic")
