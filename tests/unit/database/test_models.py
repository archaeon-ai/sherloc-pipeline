"""
Tests for SQLAlchemy ORM models.

Tests cover:
- Table creation and schema
- Pydantic to ORM conversion
- ORM to Pydantic conversion
- Foreign key relationships
- Cascading deletes
- Index existence
"""

import uuid
from datetime import date, datetime, timezone

import pytest

from sherloc_pipeline.database.connection import get_engine, get_session, create_all_tables
from sherloc_pipeline.database.models import (
    Base,
    SolORM,
    ScanORM,
    ScanPointORM,
    SpectrumORM,
    InstrumentStateORM,
    CCDConfigurationORM,
    ScannerCalibrationORM,
    ContextImageORM,
    RegionOfInterestORM,
    FittedPeakORM,
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
    ProcessingLevel,
    ImageType,
    PeakType,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    eng = get_engine(":memory:")
    create_all_tables(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a database session for testing."""
    with get_session(engine) as sess:
        yield sess


class TestSolORM:
    """Tests for SolORM model."""

    def test_create_sol(self, session):
        """Test creating a Sol record."""
        sol = SolORM(
            sol_number=921,
            earth_date=date(2023, 9, 21),
            solar_longitude=180.5,
            mission_phase="Extended Mission",
            data_source="loupe",
        )
        session.add(sol)
        session.flush()

        retrieved = session.query(SolORM).filter_by(sol_number=921).first()
        assert retrieved is not None
        assert retrieved.sol_number == 921
        assert retrieved.earth_date == date(2023, 9, 21)
        assert retrieved.solar_longitude == 180.5
        assert retrieved.mission_phase == "Extended Mission"
        assert retrieved.data_source == "loupe"

    def test_sol_to_pydantic(self, session):
        """Test converting SolORM to Pydantic Sol."""
        sol_orm = SolORM(
            sol_number=100,
            earth_date=date(2021, 1, 1),
            data_source="loupe",
        )
        session.add(sol_orm)
        session.flush()

        pydantic_sol = sol_orm.to_pydantic()
        assert isinstance(pydantic_sol, Sol)
        assert pydantic_sol.sol_number == 100
        assert pydantic_sol.earth_date == date(2021, 1, 1)
        assert pydantic_sol.data_source == DataSource.LOUPE

    def test_sol_from_pydantic(self):
        """Test creating SolORM from Pydantic Sol."""
        pydantic_sol = Sol(
            sol_number=200,
            earth_date=date(2022, 6, 15),
            data_source=DataSource.PDS4,
        )
        sol_orm = SolORM.from_pydantic(pydantic_sol)

        assert sol_orm.sol_number == 200
        assert sol_orm.earth_date == date(2022, 6, 15)
        assert sol_orm.data_source == "pds4"


class TestScanORM:
    """Tests for ScanORM model."""

    def test_create_scan_with_sol(self, session):
        """Test creating a Scan with Sol foreign key."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan = ScanORM(
            id=str(uuid.uuid4()),
            sol_number=921,
            scan_name="Amherst_Point",
            scan_id="SrlcSpecSpecSohRaw_0672194998-62417-1",
            sclk_start=672194998,
            n_points=100,
            n_channels=2148,
            shots_per_point=10,
            laser_wavelength_nm=248.6,
        )
        session.add(scan)
        session.flush()

        retrieved = session.query(ScanORM).filter_by(scan_name="Amherst_Point").first()
        assert retrieved is not None
        assert retrieved.sol_number == 921
        assert retrieved.n_points == 100
        assert retrieved.sol.sol_number == 921

    def test_scan_to_pydantic(self, session):
        """Test converting ScanORM to Pydantic Scan."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = uuid.uuid4()
        scan_orm = ScanORM(
            id=str(scan_id),
            sol_number=921,
            scan_name="Test_Target",
            scan_id="test_scan",
            sclk_start=100000,
            n_points=50,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan_orm)
        session.flush()

        pydantic_scan = scan_orm.to_pydantic()
        assert isinstance(pydantic_scan, Scan)
        assert pydantic_scan.id == scan_id
        assert pydantic_scan.scan_name == "Test_Target"

    def test_scan_from_pydantic(self, session):
        """Test creating ScanORM from Pydantic Scan."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        pydantic_scan = Scan(
            sol_number=921,
            scan_name="Pydantic_Target",
            scan_id="pydantic_scan",
            sclk_start=200000,
            n_points=75,
            shots_per_point=100,
        )
        scan_orm = ScanORM.from_pydantic(pydantic_scan)
        session.add(scan_orm)
        session.flush()

        retrieved = session.query(ScanORM).filter_by(scan_name="Pydantic_Target").first()
        assert retrieved is not None
        assert retrieved.n_points == 75

    def test_scan_cascade_delete(self, session):
        """Test that deleting a Sol cascades to Scans."""
        sol = SolORM(sol_number=999, data_source="loupe")
        session.add(sol)
        session.flush()

        scan = ScanORM(
            id=str(uuid.uuid4()),
            sol_number=999,
            scan_name="Cascade_Test",
            scan_id="cascade_scan",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        # Verify scan exists
        assert session.query(ScanORM).filter_by(sol_number=999).first() is not None

        # Delete sol
        session.delete(sol)
        session.flush()

        # Scan should be deleted
        assert session.query(ScanORM).filter_by(sol_number=999).first() is None


class TestScanPointORM:
    """Tests for ScanPointORM model."""

    def test_create_scan_point(self, session):
        """Test creating a ScanPoint."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        point = ScanPointORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            point_index=0,
            x_pixel=824.5,
            y_pixel=600.2,
            azimuth_dn=1000,
            elevation_dn=2000,
        )
        session.add(point)
        session.flush()

        retrieved = session.query(ScanPointORM).filter_by(point_index=0).first()
        assert retrieved is not None
        assert retrieved.x_pixel == 824.5
        assert retrieved.y_pixel == 600.2


class TestSpectrumORM:
    """Tests for SpectrumORM model."""

    def test_create_spectrum(self, session):
        """Test creating a Spectrum with compressed data."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        point_id = str(uuid.uuid4())
        point = ScanPointORM(
            id=point_id,
            scan_id=scan_id,
            point_index=0,
        )
        session.add(point)
        session.flush()

        # Create spectrum with compressed intensities
        intensity_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        intensities = Spectrum.compress_array(intensity_values)

        spectrum = SpectrumORM(
            id=str(uuid.uuid4()),
            scan_point_id=point_id,
            region="R1",
            spectrum_type="active",
            processing_level="normalized",
            intensities=intensities,
        )
        session.add(spectrum)
        session.flush()

        retrieved = session.query(SpectrumORM).first()
        assert retrieved is not None
        assert retrieved.region == "R1"

        # Verify decompression works
        pydantic_spectrum = retrieved.to_pydantic()
        decompressed = pydantic_spectrum.intensity_values
        assert len(decompressed) == 5
        assert decompressed[0] == pytest.approx(1.0)


class TestInstrumentStateORM:
    """Tests for InstrumentStateORM model."""

    def test_create_instrument_state(self, session):
        """Test creating an InstrumentState."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        state = InstrumentStateORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            ccd_temp_c=-30.5,
            pcb_temp_c=25.0,
            laser_shot_counter=1500000,
        )
        session.add(state)
        session.flush()

        retrieved = session.query(InstrumentStateORM).first()
        assert retrieved is not None
        assert retrieved.ccd_temp_c == -30.5
        assert retrieved.laser_shot_counter == 1500000


class TestCCDConfigurationORM:
    """Tests for CCDConfigurationORM model."""

    def test_create_ccd_configuration(self, session):
        """Test creating a CCDConfiguration."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        config = CCDConfigurationORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            region_enable=7,
            gain_2d=1,
        )
        session.add(config)
        session.flush()

        retrieved = session.query(CCDConfigurationORM).first()
        assert retrieved is not None
        assert retrieved.region_enable == 7


class TestScannerCalibrationORM:
    """Tests for ScannerCalibrationORM model."""

    def test_create_scanner_calibration(self, session):
        """Test creating a ScannerCalibration."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        cal = ScannerCalibrationORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            az_scale=0.0285,
            el_scale=0.0285,
            laser_x=824,
            laser_y=600,
            rotation_deg=0.0,
        )
        session.add(cal)
        session.flush()

        retrieved = session.query(ScannerCalibrationORM).first()
        assert retrieved is not None
        assert retrieved.az_scale == 0.0285
        assert retrieved.laser_x == 824


class TestContextImageORM:
    """Tests for ContextImageORM model."""

    def test_create_context_image(self, session):
        """Test creating a ContextImage."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        image = ContextImageORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            image_type="ACI",
            file_path="/data/img/test.png",
            pixel_scale_um=10.1,
            width_px=1648,
            height_px=1200,
        )
        session.add(image)
        session.flush()

        retrieved = session.query(ContextImageORM).first()
        assert retrieved is not None
        assert retrieved.image_type == "ACI"
        assert retrieved.pixel_scale_um == 10.1


class TestRegionOfInterestORM:
    """Tests for RegionOfInterestORM model."""

    def test_create_region_of_interest(self, session):
        """Test creating a RegionOfInterest."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        roi = RegionOfInterestORM(
            id=str(uuid.uuid4()),
            scan_id=scan_id,
            name="Carbonate Vein",
            color_hex="#00FF00",
            point_indices=[1, 2, 3, 5, 8],
        )
        session.add(roi)
        session.flush()

        retrieved = session.query(RegionOfInterestORM).first()
        assert retrieved is not None
        assert retrieved.name == "Carbonate Vein"
        assert retrieved.point_indices == [1, 2, 3, 5, 8]


class TestFittedPeakORM:
    """Tests for FittedPeakORM model."""

    def test_create_fitted_peak(self, session):
        """Test creating a FittedPeak."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        point_id = str(uuid.uuid4())
        point = ScanPointORM(
            id=point_id,
            scan_id=scan_id,
            point_index=0,
        )
        session.add(point)
        session.flush()

        spectrum_id = str(uuid.uuid4())
        spectrum = SpectrumORM(
            id=spectrum_id,
            scan_point_id=point_id,
            region="R1",
            spectrum_type="active",
            processing_level="normalized",
            intensities=b"\x00",
        )
        session.add(spectrum)
        session.flush()

        peak = FittedPeakORM(
            id=str(uuid.uuid4()),
            spectrum_id=spectrum_id,
            peak_type="gaussian",
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
            snr=15.2,
            mineral_assignment="calcite",
            assignment_confidence=0.92,
        )
        session.add(peak)
        session.flush()

        retrieved = session.query(FittedPeakORM).first()
        assert retrieved is not None
        assert retrieved.center_cm1 == 1085.5
        assert retrieved.mineral_assignment == "calcite"


class TestIndexes:
    """Tests for database indexes."""

    def test_sol_scan_name_index_used(self, engine):
        """Test that the composite sol_scan_name index exists."""
        # Check index exists in metadata
        from sqlalchemy import inspect
        inspector = inspect(engine)
        indexes = inspector.get_indexes("scans")
        index_names = [idx["name"] for idx in indexes]
        assert "ix_scans_sol_scan_name" in index_names

    def test_scan_id_index_exists(self, engine):
        """Test that scan_id index exists."""
        from sqlalchemy import inspect
        inspector = inspect(engine)
        indexes = inspector.get_indexes("scans")
        index_names = [idx["name"] for idx in indexes]
        assert "ix_scans_scan_id" in index_names

    def test_scan_name_index_exists(self, engine):
        """Test that scan_name index exists."""
        from sqlalchemy import inspect
        inspector = inspect(engine)
        indexes = inspector.get_indexes("scans")
        index_names = [idx["name"] for idx in indexes]
        assert "ix_scans_scan_name" in index_names


class TestRelationships:
    """Tests for ORM relationships."""

    def test_scan_to_points_relationship(self, session):
        """Test navigating from Scan to ScanPoints."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        # Add multiple points
        for i in range(3):
            point = ScanPointORM(
                id=str(uuid.uuid4()),
                scan_id=scan_id,
                point_index=i,
            )
            session.add(point)
        session.flush()

        # Navigate relationship
        retrieved = session.query(ScanORM).first()
        assert len(retrieved.scan_points) == 3
        assert all(isinstance(p, ScanPointORM) for p in retrieved.scan_points)

    def test_full_hierarchy(self, session):
        """Test full hierarchy: Sol -> Scan -> ScanPoint -> Spectrum -> FittedPeak."""
        sol = SolORM(sol_number=921, data_source="loupe")
        session.add(sol)
        session.flush()

        scan_id = str(uuid.uuid4())
        scan = ScanORM(
            id=scan_id,
            sol_number=921,
            scan_name="Test",
            scan_id="test",
            sclk_start=100000,
            n_points=10,
            n_channels=2148,
            shots_per_point=10,
        )
        session.add(scan)
        session.flush()

        point_id = str(uuid.uuid4())
        point = ScanPointORM(
            id=point_id,
            scan_id=scan_id,
            point_index=0,
        )
        session.add(point)
        session.flush()

        spectrum_id = str(uuid.uuid4())
        spectrum = SpectrumORM(
            id=spectrum_id,
            scan_point_id=point_id,
            region="R1",
            spectrum_type="active",
            processing_level="normalized",
            intensities=b"\x00",
        )
        session.add(spectrum)
        session.flush()

        peak = FittedPeakORM(
            id=str(uuid.uuid4()),
            spectrum_id=spectrum_id,
            peak_type="gaussian",
            fit_modality="minerals",
            center_cm1=1085.5,
            amplitude=1500.0,
            fwhm_cm1=25.0,
        )
        session.add(peak)
        session.flush()

        # Navigate full hierarchy
        retrieved_sol = session.query(SolORM).first()
        assert len(retrieved_sol.scans) == 1
        assert len(retrieved_sol.scans[0].scan_points) == 1
        assert len(retrieved_sol.scans[0].scan_points[0].spectra) == 1
        assert len(retrieved_sol.scans[0].scan_points[0].spectra[0].fitted_peaks) == 1
